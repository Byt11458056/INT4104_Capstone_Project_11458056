"""
Shared web search and article extraction helpers.

This centralizes search/scraping behavior so the app can keep a single code
path for news discovery today, while leaving a clean seam for future MCP-backed
providers such as g-search-mcp and fetcher-mcp.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlparse

import httpx

from config import get_secret

log = logging.getLogger(__name__)

_FAILED_DOMAINS: dict[str, int] = {}
_MAX_DOMAIN_FAILURES = 3
_RECENCY_TO_DDG = {
    "day": "d",
    "week": "w",
    "month": "m",
    "year": "y",
}


def _is_mcp_available() -> bool:
    """Check if MCP tools are available (placeholder for MCP detection)."""
    # TODO: Implement actual MCP server detection
    return False


def preferred_search_provider() -> str:
    """Return the configured web search provider."""
    provider = (get_secret("WEB_SEARCH_PROVIDER", "ddgs") or "ddgs").strip().lower()
    return provider or "ddgs"


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc
    except Exception:
        return ""


def _should_skip_domain(url: str) -> bool:
    return _FAILED_DOMAINS.get(_domain(url), 0) >= _MAX_DOMAIN_FAILURES


def _record_failure(url: str) -> None:
    domain = _domain(url)
    _FAILED_DOMAINS[domain] = _FAILED_DOMAINS.get(domain, 0) + 1


def scrape_url(url: str, max_length: int = 5000) -> str:
    """Fetch a URL and return readability-extracted body text when possible."""
    if not url or _should_skip_domain(url):
        return ""

    # Try primary method first
    content = _scrape_url_primary(url, max_length)

    # Fallback to MCP if primary fails and MCP is available
    if not content and _is_mcp_available():
        content = _scrape_url_with_mcp(url, max_length)

    return content


def _scrape_url_primary(url: str, max_length: int = 5000) -> str:
    """Primary scraping method using httpx + readability."""
    if not url or _should_skip_domain(url):
        return ""

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    try:
        with httpx.Client(follow_redirects=True, timeout=8.0, headers=headers) as client:
            resp = client.get(url)
            if resp.status_code != 200:
                _record_failure(url)
                return ""
            html = resp.text
    except Exception as exc:
        log.debug("Web fetch failed for %s: %s", url, exc)
        _record_failure(url)
        return ""

    try:
        from readability import Document

        doc = Document(html)
        text = re.sub(r"<[^>]+>", " ", doc.summary())
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_length] if text else ""
    except Exception as exc:
        log.debug("Readability extraction failed for %s: %s", url, exc)
        return ""


def _scrape_url_with_mcp(url: str, max_length: int = 5000) -> str:
    """Fallback scraping using fetcher-mcp (if available)."""
    try:
        # TODO: Implement actual MCP fetcher-mcp call
        # This would use the MCP protocol to call fetcher-mcp server
        log.debug("MCP scraping not yet implemented for %s", url)
        return ""
    except Exception as exc:
        log.debug("MCP scraping failed for %s: %s", url, exc)
        return ""


def _ddgs_text_search(query: str, max_results: int, recency: str) -> list[dict[str, Any]]:
    from ddgs import DDGS

    timelimit = _RECENCY_TO_DDG.get(recency, "m")
    results = list(DDGS().text(query, max_results=max_results, timelimit=timelimit))
    return [
        {
            "title": item.get("title", ""),
            "body": item.get("body", ""),
            "url": item.get("href", item.get("url", "")),
            "source": "DuckDuckGo",
        }
        for item in results
    ]


def _google_text_search(query: str, max_results: int) -> list[dict[str, Any]]:
    from googlesearch import search as gsearch

    results: list[dict[str, Any]] = []
    for url in gsearch(query, num_results=max_results, lang="en"):
        results.append({
            "title": "",
            "body": "",
            "url": url,
            "source": "Google",
        })
    return results


def search_text(
    query: str,
    *,
    max_results: int = 5,
    recency: str = "month",
    prefer_provider: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search the web and return normalized result dictionaries.

    Supported providers today:
      - ``ddgs``  : DuckDuckGo text search
      - ``google``: googlesearch-python fallback
      - ``auto``  : DDGS first, then Google supplement
    """
    provider = (prefer_provider or preferred_search_provider()).strip().lower()
    results: list[dict[str, Any]] = []

    if provider in {"ddgs", "auto"}:
        try:
            results.extend(_ddgs_text_search(query, max_results=max_results, recency=recency))
        except Exception as exc:
            log.warning("DDGS text search failed: %s", exc)

    if provider in {"google", "auto"} and len(results) < max_results:
        try:
            seen_urls = {item.get("url", "") for item in results}
            for item in _google_text_search(query, max_results=max_results - len(results)):
                if item["url"] and item["url"] not in seen_urls:
                    results.append(item)
                    seen_urls.add(item["url"])
        except Exception as exc:
            log.warning("Google text search failed: %s", exc)

    if provider not in {"ddgs", "google", "auto"}:
        log.warning("Unknown WEB_SEARCH_PROVIDER '%s'; falling back to DDGS", provider)
        return search_text(query, max_results=max_results, recency=recency, prefer_provider="ddgs")

    return results[:max_results]


def search_news(
    query: str,
    *,
    max_results: int = 8,
    recency: str = "month",
    scrape: bool = True,
) -> list[dict[str, Any]]:
    """Search news results and optionally enrich them with scraped article text."""
    try:
        from ddgs import DDGS

        timelimit = _RECENCY_TO_DDG.get(recency, "m")
        raw_results = list(DDGS().news(query, max_results=max_results, timelimit=timelimit))
    except Exception as exc:
        log.warning("DDGS news search failed: %s", exc)
        return []

    articles: list[dict[str, Any]] = []
    for item in raw_results:
        url = item.get("url", item.get("link", ""))
        body = item.get("body", "")
        content = scrape_url(url) if scrape and url else ""
        if not content:
            content = body
        articles.append({
            "title": item.get("title", ""),
            "url": url,
            "source": item.get("source", "DuckDuckGo News"),
            "published": str(item.get("date", ""))[:10],
            "content": content,
            "content_type": "scraped" if content and content != body else "snippet",
        })
    return articles
