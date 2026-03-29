"""
Unified news fetcher with resilient scraping and multiple sources.

Sources (tried in priority order):
  1. DefeatBeta  — full paragraph-level content (stocks only)
  2. GNews API   — title + URL, then httpx + readability scrapes the body
  3. DuckDuckGo News — recent articles with snippets
  4. yfinance     — title + URL (fallback)
  5. Trump policy/tariff/action signals via web search
  6. Polymarket   — prediction market data for sentiment
"""

from __future__ import annotations

import logging
from typing import Any

import requests

import defeatbeta_client as db
from config import get_secret
from web_research import scrape_url, search_news

log = logging.getLogger(__name__)

_GNEWS_API_KEY: str | None = get_secret("GNEWS_API_KEY")


# ---------------------------------------------------------------------------
# Source-specific fetchers
# ---------------------------------------------------------------------------

def _fetch_defeatbeta(ticker: str) -> list[dict]:
    return db.get_news_articles(ticker, max_articles=15)


def _fetch_gnews(ticker: str, max_results: int = 10) -> list[dict]:
    if not _GNEWS_API_KEY:
        return []
    url = "https://gnews.io/api/v4/search"
    params = {"q": f"{ticker} stock", "lang": "en", "max": max_results, "token": _GNEWS_API_KEY}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        log.warning("GNews API failed: %s", exc)
        return []
    raw = resp.json().get("articles", [])
    articles: list[dict] = []
    for a in raw:
        title = a.get("title", "")
        link = a.get("url", "")
        source = (a.get("source", {}) or {}).get("name", "GNews")
        published = a.get("publishedAt", "")
        description = a.get("description", "")
        content = scrape_url(link) if link else ""
        if not content:
            content = a.get("content", description) or description
        articles.append({
            "title": title, "url": link, "source": source,
            "published": published[:10] if published else "",
            "content": content,
            "content_type": "scraped" if content and content != description else "headline_only",
        })
    return articles


def _fetch_ddg_news(ticker: str, max_results: int = 8) -> list[dict]:
    return search_news(f"{ticker} stock", max_results=max_results, recency="month", scrape=True)


def _fetch_yfinance(ticker: str) -> list[dict]:
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        news = tk.news or []
    except Exception:
        return []
    articles: list[dict] = []
    for item in news:
        try:
            title = item.get("title", "")
            if not title:
                content_obj = item.get("content") or {}
                title = content_obj.get("title", "") if isinstance(content_obj, dict) else ""
            link = item.get("link", "")
            if not link:
                content_obj = item.get("content") or {}
                if isinstance(content_obj, dict):
                    click_url = content_obj.get("clickThroughUrl") or {}
                    link = click_url.get("url", "") if isinstance(click_url, dict) else ""
            if not link:
                link = item.get("url", "")
            publisher = item.get("publisher", "Yahoo Finance")
            content = scrape_url(link) if link else ""
            articles.append({
                "title": title or "(untitled)", "url": link, "source": publisher,
                "published": "", "content": content,
                "content_type": "scraped" if content else "headline_only",
            })
        except Exception:
            continue
    return articles


# ---------------------------------------------------------------------------
# Trump policy / tariff / executive action signals
# ---------------------------------------------------------------------------

_TRUMP_POLICY_QUERIES = [
    "Trump tariff trade policy executive order {topic}",
    "Trump administration economic policy sanctions {topic}",
    "Trump Truth Social market impact {topic}",
]


def get_trump_posts(topic: str = "", max_results: int = 8) -> list[dict]:
    """Fetch news about Trump's policy actions, tariffs, executive orders,
    and Truth Social posts that impact markets."""
    posts: list[dict] = []
    seen: set[str] = set()

    queries = [q.format(topic=topic) for q in _TRUMP_POLICY_QUERIES]
    if topic:
        queries.append(f"Trump {topic} policy impact market")

    try:
        for query in queries:
            if len(posts) >= max_results:
                break
            for article in search_news(query, max_results=max(2, max_results // 2), recency="week", scrape=True):
                url = article.get("url", "")
                if url and url in seen:
                    continue
                seen.add(url)
                article["content_type"] = "political"
                posts.append(article)
                if len(posts) >= max_results:
                    break
    except Exception as exc:
        log.warning("Trump policy search failed: %s", exc)

    return posts[:max_results]


# ---------------------------------------------------------------------------
# Polymarket — prediction market sentiment
# ---------------------------------------------------------------------------

_POLYMARKET_API = "https://gamma-api.polymarket.com"


def get_polymarket_data(query: str = "", max_results: int = 5) -> list[dict]:
    """Fetch active prediction markets from Polymarket matching the query."""
    try:
        params: dict[str, Any] = {
            "active": "true",
            "closed": "false",
            "limit": max_results,
            "order": "volume",
            "ascending": "false",
        }
        if query:
            params["tag"] = query

        resp = requests.get(
            f"{_POLYMARKET_API}/markets",
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        markets = resp.json()

        results: list[dict] = []
        for m in markets:
            question = m.get("question", "")
            if not question:
                continue

            # If we have a search query, do basic relevance filtering
            if query:
                q_lower = query.lower()
                q_text = f"{question} {m.get('description', '')}".lower()
                if not any(word in q_text for word in q_lower.split()):
                    continue

            outcomes = m.get("outcomes", "")
            outcome_prices = m.get("outcomePrices", "")
            try:
                import json
                if isinstance(outcomes, str):
                    outcomes = json.loads(outcomes)
                if isinstance(outcome_prices, str):
                    outcome_prices = json.loads(outcome_prices)
            except Exception:
                outcomes = []
                outcome_prices = []

            price_str = ""
            if outcomes and outcome_prices:
                parts = []
                for o, p in zip(outcomes, outcome_prices):
                    try:
                        parts.append(f"{o}: {float(p) * 100:.0f}%")
                    except (ValueError, TypeError):
                        parts.append(f"{o}: {p}")
                price_str = " | ".join(parts)

            results.append({
                "question": question,
                "odds": price_str,
                "volume": m.get("volume", 0),
                "liquidity": m.get("liquidity", 0),
                "url": f"https://polymarket.com/event/{m.get('slug', '')}",
                "end_date": m.get("endDate", ""),
                "description": m.get("description", "")[:200],
            })
        return results[:max_results]
    except Exception as exc:
        log.warning("Polymarket fetch failed: %s", exc)
        return []


def search_polymarket(query: str, max_results: int = 5) -> list[dict]:
    """Search Polymarket for prediction markets relevant to a topic."""
    try:
        import json as _json
        resp = requests.get(
            f"{_POLYMARKET_API}/markets",
            params={
                "active": "true",
                "closed": "false",
                "limit": 100,
                "order": "volume",
                "ascending": "false",
            },
            timeout=10,
        )
        resp.raise_for_status()
        markets = resp.json()

        q_words = set(query.lower().split())
        scored = []
        for m in markets:
            text = f"{m.get('question', '')} {m.get('description', '')}".lower()

            # Keyword matching score (0-1)
            matches = sum(1 for w in q_words if w in text)
            keyword_score = min(matches / len(q_words), 1.0) if q_words else 0

            # Volume score (0-1, normalized)
            volume = float(m.get("volume", 0) or 0)
            volume_score = min(volume / 1000000, 1.0)  # Normalize to 1M

            # Liquidity score (0-1, normalized)
            liquidity = float(m.get("liquidity", 0) or 0)
            liquidity_score = min(liquidity / 100000, 1.0)  # Normalize to 100K

            # Weighted score: keywords 40%, volume 30%, liquidity 30%
            total_score = (keyword_score * 0.4) + (volume_score * 0.3) + (liquidity_score * 0.3)

            if keyword_score > 0:  # Only include if keywords match
                scored.append((total_score, m))

        scored.sort(key=lambda x: -x[0])

        results: list[dict] = []
        for _, m in scored[:max_results]:
            question = m.get("question", "")
            outcomes = m.get("outcomes", "")
            outcome_prices = m.get("outcomePrices", "")
            try:
                if isinstance(outcomes, str):
                    outcomes = _json.loads(outcomes)
                if isinstance(outcome_prices, str):
                    outcome_prices = _json.loads(outcome_prices)
            except Exception:
                outcomes, outcome_prices = [], []

            price_str = ""
            if outcomes and outcome_prices:
                parts = []
                for o, p in zip(outcomes, outcome_prices):
                    try:
                        parts.append(f"{o}: {float(p) * 100:.0f}%")
                    except (ValueError, TypeError):
                        parts.append(f"{o}: {p}")
                price_str = " | ".join(parts)

            results.append({
                "question": question,
                "odds": price_str,
                "volume": m.get("volume", 0),
                "liquidity": m.get("liquidity", 0),
                "url": f"https://polymarket.com/event/{m.get('slug', '')}",
                "end_date": m.get("endDate", ""),
                "description": m.get("description", "")[:200],
            })
        return results
    except Exception as exc:
        log.warning("Polymarket search failed: %s", exc)
        return []


def get_market_sentiment_from_polymarket(ticker: str, sector: str = "") -> dict:
    """Get prediction markets specifically related to a stock/sector for sentiment analysis."""
    queries = [f"{ticker} stock", ticker]
    if sector:
        queries.extend([f"{sector} sector", f"{sector} stocks"])

    all_markets = []
    seen_urls = set()

    for query in queries:
        markets = search_polymarket(query, max_results=3)
        for m in markets:
            url = m.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_markets.append(m)

    if not all_markets:
        return {"sentiment_score": 0, "markets": [], "summary": "No prediction markets found"}

    # Calculate aggregate sentiment from odds
    sentiment_scores = []
    for m in all_markets:
        odds_str = m.get("odds", "")
        if "Yes:" in odds_str:
            try:
                yes_pct = float(odds_str.split("Yes:")[1].split("%")[0].strip())
                # Convert to -1 to +1 scale (50% = neutral)
                sentiment_scores.append((yes_pct - 50) / 50)
            except (ValueError, IndexError):
                pass

    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

    return {
        "sentiment_score": round(avg_sentiment, 2),
        "markets": all_markets[:5],
        "summary": f"{'Bullish' if avg_sentiment > 0.1 else 'Bearish' if avg_sentiment < -0.1 else 'Neutral'} sentiment from {len(all_markets)} prediction markets"
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_news(ticker: str, include_political: bool = False) -> list[dict]:
    """Return a deduplicated list of article dicts for *ticker*."""
    articles: list[dict] = []
    seen_urls: set[str] = set()

    # Live sources first, then DefeatBeta (which has a HuggingFace dataset cutoff)
    for fetcher in [_fetch_ddg_news, _fetch_gnews, _fetch_yfinance, _fetch_defeatbeta]:
        try:
            batch = fetcher(ticker)
        except Exception as exc:
            log.warning("News fetcher %s failed: %s", fetcher.__name__, exc)
            continue
        for a in batch:
            url = a.get("url", "")
            if url and url in seen_urls:
                continue
            seen_urls.add(url)
            articles.append(a)

    if include_political:
        try:
            political = get_trump_posts(ticker)
            for a in political:
                url = a.get("url", "")
                if url and url in seen_urls:
                    continue
                seen_urls.add(url)
                articles.append(a)
        except Exception as exc:
            log.warning("Political news failed: %s", exc)

    if not articles:
        log.warning("No news found for %s from any source", ticker)
    return articles


def get_topic_news(topic: str, max_results: int = 10) -> list[dict]:
    """Fetch news for a topic (not ticker-specific)."""
    return search_news(topic, max_results=max_results, recency="month", scrape=True)
