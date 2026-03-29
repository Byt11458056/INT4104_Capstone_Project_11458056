"""
SQLite persistence layer.

Stores analysis runs, scraped articles, chat conversations, chat messages,
tool results (with model metadata), and portfolio holdings.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime

_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "store.db")


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("ALTER TABLE analyses ADD COLUMN details_json TEXT")
    except sqlite3.OperationalError:
        pass

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS analyses (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT NOT NULL,
            trader_type TEXT,
            lookback    TEXT,
            signal      TEXT,
            score       REAL,
            sentiment   REAL,
            summary     TEXT,
            details_json TEXT,
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS articles (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker          TEXT NOT NULL,
            title           TEXT,
            url             TEXT,
            source          TEXT,
            content         TEXT,
            content_type    TEXT,
            sentiment_score REAL,
            created_at      TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS conversations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT NOT NULL,
            ticker      TEXT,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chat_messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role            TEXT NOT NULL,
            content         TEXT NOT NULL,
            tool_log_json   TEXT,
            created_at      TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        );

        CREATE TABLE IF NOT EXISTS tool_results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            tool_type   TEXT NOT NULL,
            ticker      TEXT,
            model_id    TEXT,
            provider    TEXT,
            input_json  TEXT,
            result_json TEXT,
            summary     TEXT,
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS portfolio_holdings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_name TEXT NOT NULL DEFAULT 'default',
            ticker      TEXT NOT NULL,
            quantity    REAL NOT NULL,
            avg_cost    REAL NOT NULL,
            direction   TEXT NOT NULL DEFAULT 'long',
            added_at    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_name TEXT NOT NULL DEFAULT 'default',
            total_value   REAL,
            total_pnl     REAL,
            snapshot_json TEXT,
            created_at    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS transcripts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT NOT NULL,
            fiscal_year INTEGER,
            fiscal_quarter INTEGER,
            report_date TEXT,
            full_text   TEXT NOT NULL,
            created_at  TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_analyses_ticker ON analyses(ticker);
        CREATE INDEX IF NOT EXISTS idx_articles_ticker ON articles(ticker);
        CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url);
        CREATE INDEX IF NOT EXISTS idx_messages_conv ON chat_messages(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at);
        CREATE INDEX IF NOT EXISTS idx_tool_results_type ON tool_results(tool_type);
        CREATE INDEX IF NOT EXISTS idx_tool_results_ticker ON tool_results(ticker);
        CREATE INDEX IF NOT EXISTS idx_portfolio_name ON portfolio_holdings(portfolio_name);
        CREATE INDEX IF NOT EXISTS idx_transcripts_ticker ON transcripts(ticker);
    """)

    # FTS5 virtual table for full-text search on transcripts
    try:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS transcripts_fts
            USING fts5(ticker, fiscal_year, fiscal_quarter, full_text,
                       content='transcripts', content_rowid='id')
        """)
    except Exception:
        pass


_initialized = False


def _get_conn() -> sqlite3.Connection:
    global _initialized
    conn = _connect()
    if not _initialized:
        _init_db(conn)
        _initialized = True
    return conn


# ---------------------------------------------------------------------------
# Analyses (legacy — kept for backward compat)
# ---------------------------------------------------------------------------

def save_analysis(
    ticker: str,
    trader_type: str,
    lookback: str,
    signal: str,
    score: float,
    sentiment: float,
    summary: str,
    details: dict | None = None,
) -> int:
    conn = _get_conn()
    details_json = json.dumps(details) if details else None
    cur = conn.execute(
        """INSERT INTO analyses
           (ticker, trader_type, lookback, signal, score, sentiment, summary, details_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (ticker.upper(), trader_type, lookback, signal, score, sentiment, summary,
         details_json, datetime.utcnow().isoformat()),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def get_analysis_history(ticker: str | None = None, limit: int = 20) -> list[dict]:
    conn = _get_conn()
    if ticker:
        rows = conn.execute(
            "SELECT * FROM analyses WHERE ticker = ? ORDER BY created_at DESC LIMIT ?",
            (ticker.upper(), limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM analyses ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        if d.get("details_json"):
            try:
                d["details"] = json.loads(d["details_json"])
            except (json.JSONDecodeError, TypeError):
                d["details"] = None
        else:
            d["details"] = None
        result.append(d)
    return result


def get_analysis_by_id(analysis_id: int) -> dict | None:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    if d.get("details_json"):
        try:
            d["details"] = json.loads(d["details_json"])
        except (json.JSONDecodeError, TypeError):
            d["details"] = None
    else:
        d["details"] = None
    return d


# ---------------------------------------------------------------------------
# Articles
# ---------------------------------------------------------------------------

def save_articles(articles: list[dict], ticker: str) -> int:
    conn = _get_conn()
    now = datetime.utcnow().isoformat()
    inserted = 0
    for a in articles:
        url = a.get("url", "")
        if url:
            exists = conn.execute(
                "SELECT 1 FROM articles WHERE ticker = ? AND url = ?",
                (ticker.upper(), url),
            ).fetchone()
            if exists:
                continue
        conn.execute(
            """INSERT INTO articles
               (ticker, title, url, source, content, content_type, sentiment_score, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (ticker.upper(), a.get("title", ""), url, a.get("source", ""),
             a.get("content", ""), a.get("content_type", ""),
             a.get("sentiment_score"), now),
        )
        inserted += 1
    conn.commit()
    return inserted


def get_stored_articles(ticker: str, limit: int = 50) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM articles WHERE ticker = ? ORDER BY created_at DESC LIMIT ?",
        (ticker.upper(), limit),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

def create_conversation(title: str, ticker: str | None = None) -> int:
    conn = _get_conn()
    now = datetime.utcnow().isoformat()
    cur = conn.execute(
        "INSERT INTO conversations (title, ticker, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (title, ticker, now, now),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def update_conversation_title(conversation_id: int, title: str) -> None:
    conn = _get_conn()
    conn.execute(
        "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
        (title, datetime.utcnow().isoformat(), conversation_id),
    )
    conn.commit()


def get_conversations(limit: int = 30) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        """SELECT c.*, COUNT(m.id) as message_count
           FROM conversations c
           LEFT JOIN chat_messages m ON m.conversation_id = c.id
           GROUP BY c.id
           ORDER BY c.updated_at DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_conversation_by_id(conversation_id: int) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM conversations WHERE id = ?", (conversation_id,),
    ).fetchone()
    return dict(row) if row else None


def delete_conversation(conversation_id: int) -> None:
    conn = _get_conn()
    conn.execute("DELETE FROM chat_messages WHERE conversation_id = ?", (conversation_id,))
    conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()


# ---------------------------------------------------------------------------
# Chat Messages
# ---------------------------------------------------------------------------

def save_message(
    conversation_id: int, role: str, content: str, tool_log: list[dict] | None = None,
) -> int:
    conn = _get_conn()
    now = datetime.utcnow().isoformat()
    tool_log_json = json.dumps(tool_log) if tool_log else None
    cur = conn.execute(
        """INSERT INTO chat_messages
           (conversation_id, role, content, tool_log_json, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (conversation_id, role, content, tool_log_json, now),
    )
    conn.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        (now, conversation_id),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def get_messages(conversation_id: int) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM chat_messages WHERE conversation_id = ? ORDER BY created_at ASC",
        (conversation_id,),
    ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        if d.get("tool_log_json"):
            try:
                d["tool_log"] = json.loads(d["tool_log_json"])
            except (json.JSONDecodeError, TypeError):
                d["tool_log"] = None
        else:
            d["tool_log"] = None
        result.append(d)
    return result


# ---------------------------------------------------------------------------
# Tool Results (new — supports model metadata for evaluation)
# ---------------------------------------------------------------------------

def save_tool_result(
    tool_type: str,
    *,
    ticker: str | None = None,
    model_id: str | None = None,
    provider: str | None = None,
    inputs: dict | None = None,
    result: dict | None = None,
    summary: str | None = None,
) -> int:
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO tool_results
           (tool_type, ticker, model_id, provider, input_json, result_json, summary, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            tool_type,
            ticker.upper() if ticker else None,
            model_id,
            provider,
            json.dumps(inputs) if inputs else None,
            json.dumps(result) if result else None,
            summary,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def get_tool_results(
    tool_type: str | None = None,
    ticker: str | None = None,
    limit: int = 20,
) -> list[dict]:
    conn = _get_conn()
    clauses: list[str] = []
    params: list = []
    if tool_type:
        clauses.append("tool_type = ?")
        params.append(tool_type)
    if ticker:
        clauses.append("ticker = ?")
        params.append(ticker.upper())
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    params.append(limit)
    rows = conn.execute(
        f"SELECT * FROM tool_results {where} ORDER BY created_at DESC LIMIT ?",
        params,
    ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        for key in ("input_json", "result_json"):
            if d.get(key):
                try:
                    d[key.replace("_json", "")] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    d[key.replace("_json", "")] = None
            else:
                d[key.replace("_json", "")] = None
        result.append(d)
    return result


def get_tool_result_by_id(result_id: int) -> dict | None:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM tool_results WHERE id = ?", (result_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    for key in ("input_json", "result_json"):
        if d.get(key):
            try:
                d[key.replace("_json", "")] = json.loads(d[key])
            except (json.JSONDecodeError, TypeError):
                d[key.replace("_json", "")] = None
        else:
            d[key.replace("_json", "")] = None
    return d


def get_all_tool_results(limit: int = 50) -> list[dict]:
    return get_tool_results(limit=limit)


# ---------------------------------------------------------------------------
# Transcripts (FTS5 searchable)
# ---------------------------------------------------------------------------

def save_transcript(
    ticker: str, fiscal_year: int, fiscal_quarter: int,
    report_date: str, full_text: str,
) -> int:
    """Store a transcript and index it in FTS5 for full-text search."""
    conn = _get_conn()
    existing = conn.execute(
        "SELECT id FROM transcripts WHERE ticker = ? AND fiscal_year = ? AND fiscal_quarter = ?",
        (ticker.upper(), fiscal_year, fiscal_quarter),
    ).fetchone()
    if existing:
        return existing["id"]
    cur = conn.execute(
        """INSERT INTO transcripts
           (ticker, fiscal_year, fiscal_quarter, report_date, full_text, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (ticker.upper(), fiscal_year, fiscal_quarter, report_date,
         full_text, datetime.utcnow().isoformat()),
    )
    row_id = cur.lastrowid
    try:
        conn.execute(
            """INSERT INTO transcripts_fts (rowid, ticker, fiscal_year, fiscal_quarter, full_text)
               VALUES (?, ?, ?, ?, ?)""",
            (row_id, ticker.upper(), str(fiscal_year), str(fiscal_quarter), full_text),
        )
    except Exception:
        pass
    conn.commit()
    return row_id


def search_transcripts(query: str, ticker: str | None = None, limit: int = 5) -> list[dict]:
    """Full-text search across stored transcripts."""
    conn = _get_conn()
    try:
        if ticker:
            rows = conn.execute(
                """SELECT t.*, rank FROM transcripts_fts f
                   JOIN transcripts t ON t.id = f.rowid
                   WHERE transcripts_fts MATCH ? AND t.ticker = ?
                   ORDER BY rank LIMIT ?""",
                (query, ticker.upper(), limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT t.*, rank FROM transcripts_fts f
                   JOIN transcripts t ON t.id = f.rowid
                   WHERE transcripts_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
                (query, limit),
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def get_transcript_text(ticker: str, fiscal_year: int, fiscal_quarter: int) -> str | None:
    """Retrieve the full text of a specific transcript."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT full_text FROM transcripts WHERE ticker = ? AND fiscal_year = ? AND fiscal_quarter = ?",
        (ticker.upper(), fiscal_year, fiscal_quarter),
    ).fetchone()
    return row["full_text"] if row else None


# ---------------------------------------------------------------------------
# Portfolio Holdings
# ---------------------------------------------------------------------------

def add_holding(
    ticker: str,
    quantity: float,
    avg_cost: float,
    direction: str = "long",
    portfolio_name: str = "default",
) -> int:
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO portfolio_holdings
           (portfolio_name, ticker, quantity, avg_cost, direction, added_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (portfolio_name, ticker.upper(), quantity, avg_cost, direction,
         datetime.utcnow().isoformat()),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def get_holdings(portfolio_name: str = "default") -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM portfolio_holdings WHERE portfolio_name = ? ORDER BY added_at DESC",
        (portfolio_name,),
    ).fetchall()
    return [dict(r) for r in rows]


def remove_holding(holding_id: int) -> None:
    conn = _get_conn()
    conn.execute("DELETE FROM portfolio_holdings WHERE id = ?", (holding_id,))
    conn.commit()


def clear_holdings(portfolio_name: str = "default") -> None:
    conn = _get_conn()
    conn.execute("DELETE FROM portfolio_holdings WHERE portfolio_name = ?", (portfolio_name,))
    conn.commit()


def get_holding_by_ticker(ticker: str, portfolio_name: str = "default") -> dict | None:
    """Get a specific holding by ticker."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM portfolio_holdings WHERE portfolio_name = ? AND ticker = ? LIMIT 1",
        (portfolio_name, ticker.upper()),
    ).fetchone()
    return dict(row) if row else None


def update_holding(ticker: str, quantity: float, avg_cost: float, portfolio_name: str = "default") -> None:
    """Update an existing holding."""
    conn = _get_conn()
    conn.execute(
        "UPDATE portfolio_holdings SET quantity = ?, avg_cost = ? WHERE portfolio_name = ? AND ticker = ?",
        (quantity, avg_cost, portfolio_name, ticker.upper()),
    )
    conn.commit()


def save_portfolio_snapshot(
    total_value: float,
    total_pnl: float,
    snapshot: dict,
    portfolio_name: str = "default",
) -> int:
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO portfolio_snapshots
           (portfolio_name, total_value, total_pnl, snapshot_json, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (portfolio_name, total_value, total_pnl,
         json.dumps(snapshot), datetime.utcnow().isoformat()),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def get_portfolio_snapshots(portfolio_name: str = "default", limit: int = 90) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        """SELECT * FROM portfolio_snapshots
           WHERE portfolio_name = ?
           ORDER BY created_at ASC
           LIMIT ?""",
        (portfolio_name, limit),
    ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        if d.get("snapshot_json"):
            try:
                d["snapshot"] = json.loads(d["snapshot_json"])
            except (json.JSONDecodeError, TypeError):
                d["snapshot"] = None
        else:
            d["snapshot"] = None
        result.append(d)
    return result
