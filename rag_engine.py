"""
Retrieval-Augmented Generation engine using SQLite FTS5.

Stores article chunks in a full-text search table so the LLM prompt
can be enriched with the most relevant passages via BM25 ranking.
Replaces the previous ChromaDB + sentence-transformers implementation
for a much lighter footprint suitable for constrained deployments.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3

log = logging.getLogger(__name__)

_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "store.db")
_CHUNK_SIZE = 400
_CHUNK_OVERLAP = 80
_initialized = False


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_fts(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            doc_id,
            ticker,
            title,
            url,
            source,
            chunk_text,
            tokenize='porter'
        );
    """)


def _get_conn() -> sqlite3.Connection:
    global _initialized
    conn = _connect()
    if not _initialized:
        _init_fts(conn)
        _initialized = True
    return conn


def _chunk_text(text: str) -> list[str]:
    if not text:
        return []
    if len(text) <= _CHUNK_SIZE:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + _CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += _CHUNK_SIZE - _CHUNK_OVERLAP
    return chunks


def _doc_id(url: str, chunk_idx: int) -> str:
    h = hashlib.md5(url.encode()).hexdigest()[:12]
    return f"{h}_{chunk_idx}"


def store_articles(articles: list[dict], ticker: str) -> int:
    """Chunk and store *articles* in the FTS5 table.  Returns new chunk count."""
    conn = _get_conn()
    stored = 0

    for art in articles:
        url = art.get("url", "")
        content = art.get("content", "")
        title = art.get("title", "")
        source = art.get("source", "")

        text = content if content else title
        if not text:
            continue

        chunks = _chunk_text(text)
        for i, chunk in enumerate(chunks):
            doc_id = _doc_id(url or title, i)

            existing = conn.execute(
                "SELECT doc_id FROM chunks_fts WHERE doc_id = ? LIMIT 1",
                (doc_id,),
            ).fetchone()
            if existing:
                continue

            conn.execute(
                "INSERT INTO chunks_fts (doc_id, ticker, title, url, source, chunk_text) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (doc_id, ticker.upper(), title[:200], url, source, chunk),
            )
            stored += 1

    conn.commit()
    log.info("Stored %d new chunks for %s", stored, ticker)
    return stored


def retrieve_relevant(
    query: str,
    ticker: str,
    top_k: int = 5,
) -> list[dict]:
    """Return the *top_k* most relevant chunks for *query* via BM25."""
    conn = _get_conn()

    words = [w for w in query.split() if w.isalnum() and len(w) > 1]
    if not words:
        return []
    clean_query = " OR ".join(words)

    try:
        rows = conn.execute(
            """SELECT title, url, source, chunk_text,
                      bm25(chunks_fts) AS rank
               FROM chunks_fts
               WHERE ticker = ? AND chunks_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (ticker.upper(), clean_query, top_k),
        ).fetchall()
    except Exception as exc:
        log.warning("FTS5 query failed: %s", exc)
        return []

    return [
        {
            "text": row["chunk_text"],
            "title": row["title"],
            "url": row["url"],
            "source": row["source"],
            "distance": row["rank"],
        }
        for row in rows
    ]
