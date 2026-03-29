"""
Multi-agent orchestration system with tool-calling.

Exposes DefeatBeta API data, technical analysis, and web search as
OpenAI-compatible function-calling tools.  The LLM autonomously decides
which tools to invoke while building a comprehensive analysis.

Agent categories:
  - **Data Research** — stock data, financial statements, valuations
  - **Fundamental Analysis** — margins, growth, earnings, SEC filings
  - **Web Research** — internet search for live information
  - **Technical Analysis** — indicators, signals, backtesting
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

import defeatbeta_client as db
from data_pipeline import get_stock_data, create_features
from signal_engine import generate_signal
from price_model import train_price_model, predict_next_day
from utils import dates_for_lookback, format_pct, TRADER_PROFILES
from web_research import search_text

log = logging.getLogger(__name__)

# ── Tool definitions (OpenAI function-calling schema) ────────────────────────

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": (
                "Get recent stock price data including latest close, change, "
                "and a short price history summary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. AAPL, MSFT, TSLA)",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_profile",
            "description": "Get company profile information including sector, industry, and description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fundamentals",
            "description": (
                "Get key fundamental metrics: PE ratio, PB, PS, ROE, ROA, "
                "ROIC, market cap, WACC, PEG ratio."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_income_statement",
            "description": "Get income statement data (revenue, operating income, net income, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "quarterly": {
                        "type": "boolean",
                        "description": "True for quarterly, False for annual. Default: true.",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_balance_sheet",
            "description": "Get balance sheet data (assets, liabilities, equity).",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "quarterly": {
                        "type": "boolean",
                        "description": "True for quarterly, False for annual. Default: true.",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cash_flow",
            "description": "Get cash flow statement data (operating, investing, financing cash flows).",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "quarterly": {
                        "type": "boolean",
                        "description": "True for quarterly, False for annual. Default: true.",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_dcf_analysis",
            "description": "Get DCF (Discounted Cash Flow) valuation analysis including fair price and recommendation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_margins",
            "description": "Get margin analysis: gross, operating, net, EBITDA, and FCF margins.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_growth_metrics",
            "description": "Get YoY growth rates: revenue, operating income, EBITDA, net income, FCF, EPS.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_earnings_transcripts",
            "description": "Get recent earnings call transcript summaries with key quotes from management.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of transcripts to retrieve (default: 2)",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sec_filings",
            "description": "Get recent SEC filings (10-K, 10-Q, 8-K, etc.) for a company.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_news",
            "description": "Get recent news articles for a stock with full content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_industry_comparison",
            "description": "Compare a stock's valuation metrics against industry averages (PE, PB, PS, ROE, ROA).",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_technical_analysis",
            "description": (
                "Run technical analysis on a stock: RSI, moving averages, "
                "volatility, ML prediction probability, and composite signal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "lookback_days": {
                        "type": "integer",
                        "description": "Number of days to look back (default: 180)",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the internet for the LATEST real-time information. "
                "Results are filtered to recent content by default. "
                "Use for current events, breaking news, recent earnings, "
                "analyst opinions, or any up-to-date information. "
                "Include the current year in your query for best results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The search query. Include the current year (e.g. '2026') "
                            "for time-sensitive topics to get the most recent results."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                    },
                    "recency": {
                        "type": "string",
                        "enum": ["day", "week", "month", "year"],
                        "description": "Time filter: 'day', 'week', 'month' (default), or 'year'.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_enterprise_metrics",
            "description": "Get enterprise value, EV/Revenue, EV/EBITDA, debt-to-equity, and related metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_earnings_transcripts",
            "description": (
                "Search stored earnings call transcripts using full-text search. "
                "Use this to find specific topics, guidance, or statements across "
                "all previously fetched transcripts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g. 'revenue guidance', 'AI spending', 'margin pressure')",
                    },
                    "symbol": {
                        "type": "string",
                        "description": "Optional: limit search to a specific ticker symbol",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# ── Tool execution functions ─────────────────────────────────────────────────

def _format_dict(data: dict | None, label: str = "Data") -> str:
    """Format a dict into a readable string for LLM consumption."""
    if not data:
        return f"No {label.lower()} available."
    lines = [f"{label}:"]
    for k, v in data.items():
        if isinstance(v, float):
            lines.append(f"  {k}: {v:.4f}")
        elif isinstance(v, dict):
            lines.append(f"  {k}:")
            for k2, v2 in v.items():
                lines.append(f"    {k2}: {v2}")
        else:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def _format_records(records: list[dict], label: str = "Data") -> str:
    """Format a list of record dicts into readable text."""
    if not records:
        return f"No {label.lower()} available."
    lines = [f"{label} ({len(records)} record(s)):"]
    for i, rec in enumerate(records[:6]):
        parts = []
        for k, v in rec.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.2f}")
            elif v is not None:
                parts.append(f"{k}={v}")
        lines.append(f"  [{i+1}] {', '.join(parts[:10])}")
    return "\n".join(lines)


def _exec_get_stock_price(symbol: str) -> str:
    try:
        df = get_stock_data(symbol)
        if df is None or df.empty:
            return f"No price data available for {symbol}."
        recent = df.tail(10)
        latest = recent.iloc[-1]
        prev = recent.iloc[-2] if len(recent) > 1 else latest
        change = (latest["Close"] - prev["Close"]) / prev["Close"] * 100

        lines = [
            f"Stock Price Data for {symbol}:",
            f"  Latest Close: ${latest['Close']:.2f}",
            f"  Daily Change: {change:+.2f}%",
            f"  High: ${latest['High']:.2f}",
            f"  Low: ${latest['Low']:.2f}",
            f"  Volume: {latest['Volume']:,.0f}",
            f"",
            f"  Recent 5-day prices:",
        ]
        for date, row in recent.tail(5).iterrows():
            d = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
            lines.append(f"    {d}: Close=${row['Close']:.2f}, Vol={row['Volume']:,.0f}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Error fetching price for {symbol}: {exc}"


def _exec_get_stock_profile(symbol: str) -> str:
    result = db.get_stock_profile(symbol)
    return _format_dict(result, f"{symbol} Company Profile")


def _exec_get_fundamentals(symbol: str) -> str:
    result = db.get_fundamentals(symbol)
    return _format_dict(result, f"{symbol} Fundamentals")


def _exec_get_income_statement(symbol: str, quarterly: bool = True) -> str:
    period = "Quarterly" if quarterly else "Annual"
    records = db.get_income_statement(symbol, quarterly=quarterly, limit=4)
    if records:
        return _format_records(records, f"{symbol} {period} Income Statement")
    web = _exec_web_search(f"{symbol} income statement {period.lower()} 2026", max_results=3, recency="month")
    return f"No direct data. Web search results:\n{web}"


def _exec_get_balance_sheet(symbol: str, quarterly: bool = True) -> str:
    period = "Quarterly" if quarterly else "Annual"
    records = db.get_balance_sheet(symbol, quarterly=quarterly, limit=4)
    if records:
        return _format_records(records, f"{symbol} {period} Balance Sheet")
    web = _exec_web_search(f"{symbol} balance sheet {period.lower()} 2026", max_results=3, recency="month")
    return f"No direct data. Web search results:\n{web}"


def _exec_get_cash_flow(symbol: str, quarterly: bool = True) -> str:
    period = "Quarterly" if quarterly else "Annual"
    records = db.get_cash_flow(symbol, quarterly=quarterly, limit=4)
    if records:
        return _format_records(records, f"{symbol} {period} Cash Flow")
    web = _exec_web_search(f"{symbol} cash flow statement {period.lower()} 2026", max_results=3, recency="month")
    return f"No direct data. Web search results:\n{web}"


def _exec_get_dcf_analysis(symbol: str) -> str:
    result = db.get_dcf_valuation(symbol)
    return _format_dict(result, f"{symbol} DCF Valuation Analysis")


def _exec_get_margins(symbol: str) -> str:
    result = db.get_margins(symbol)
    return _format_dict(result, f"{symbol} Margin Analysis")


def _exec_get_growth_metrics(symbol: str) -> str:
    result = db.get_growth_metrics(symbol)
    return _format_dict(result, f"{symbol} Growth Metrics (YoY)")


def _exec_get_earnings_transcripts(symbol: str, limit: int = 2) -> str:
    transcripts = db.get_earnings_transcripts(symbol, limit=limit)
    if not transcripts:
        web = _exec_web_search(f"{symbol} earnings call transcript 2026", max_results=3, recency="month")
        return f"No direct transcripts for {symbol}. Web search results:\n{web}"
    lines = [f"{symbol} Earnings Call Transcripts ({len(transcripts)} found):"]
    for t in transcripts:
        fy = t.get("fiscal_year", "?")
        fq = t.get("fiscal_quarter", "?")
        date = t.get("report_date", "")
        text = t.get("text", "")
        preview = text[:800] + "..." if len(text) > 800 else text
        lines.append(f"\n  --- FY{fy} Q{fq} ({date}) ---")
        lines.append(f"  {preview}")
    return "\n".join(lines)


def _exec_get_sec_filings(symbol: str) -> str:
    filings = db.get_sec_filings(symbol)
    if filings:
        return _format_records(filings, f"{symbol} SEC Filings")
    web = _exec_web_search(f"{symbol} SEC filing 10-K 10-Q 2026", max_results=3, recency="month")
    return f"No direct SEC filings for {symbol}. Web search results:\n{web}"


def _exec_get_stock_news(symbol: str) -> str:
    articles = db.get_news_articles(symbol, max_articles=8)
    if not articles:
        return f"No news articles found for {symbol}."
    lines = [f"{symbol} Recent News ({len(articles)} articles):"]
    for i, a in enumerate(articles, 1):
        title = a.get("title", "Untitled")
        source = a.get("source", "")
        published = a.get("published", "")
        content = a.get("content", "")
        preview = content[:300] + "..." if len(content) > 300 else content
        lines.append(f"\n  [{i}] {title}")
        if source:
            lines.append(f"      Source: {source} | Published: {published}")
        if preview:
            lines.append(f"      {preview}")
    return "\n".join(lines)


def _exec_get_industry_comparison(symbol: str) -> str:
    result = db.get_industry_comparison(symbol)
    return _format_dict(result, f"{symbol} vs Industry Benchmarks")


def _exec_get_technical_analysis(symbol: str, lookback_days: int = 180) -> str:
    try:
        from datetime import datetime, timedelta
        end = datetime.today()
        start = end - timedelta(days=lookback_days)
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        raw_df = get_stock_data(symbol, start_str, end_str)
        if raw_df is None or raw_df.empty:
            return f"No data available for {symbol}."

        feat_df = create_features(raw_df)
        if feat_df.empty or len(feat_df) < 5:
            return f"Insufficient data for technical analysis on {symbol}."

        latest = feat_df.iloc[-1]
        lines = [
            f"Technical Analysis for {symbol} ({lookback_days}-day lookback):",
            f"  Latest Close: ${latest['Close']:.2f}",
            f"  RSI (14): {latest['rsi']:.2f}",
            f"  MA-Short (5): ${latest['ma_5']:.2f}",
            f"  MA-Long (20): ${latest['ma_20']:.2f}",
            f"  Volatility: {latest['volatility']:.4f}",
            f"  Daily Return: {format_pct(latest['returns'])}",
        ]

        rsi_val = latest["rsi"]
        if rsi_val > 70:
            lines.append("  RSI Status: OVERBOUGHT (>70)")
        elif rsi_val < 30:
            lines.append("  RSI Status: OVERSOLD (<30)")
        else:
            lines.append("  RSI Status: Neutral range")

        if latest["ma_5"] > latest["ma_20"]:
            lines.append("  MA Crossover: Short MA above Long MA (bullish)")
        else:
            lines.append("  MA Crossover: Short MA below Long MA (bearish)")

        try:
            model, train_acc, test_acc = train_price_model(feat_df)
            direction, up_prob = predict_next_day(model, feat_df)
            sig = generate_signal(up_prob, 0.0)
            lines.extend([
                f"",
                f"  ML Model Prediction:",
                f"    Up Probability: {up_prob:.2%}",
                f"    Direction: {'UP' if direction == 1 else 'DOWN'}",
                f"    Train Accuracy: {train_acc:.1%}",
                f"    Test Accuracy: {test_acc:.1%}",
                f"    Signal: {sig['signal']} (score: {sig['score']:.4f})",
            ])
        except Exception:
            lines.append("  ML model could not be trained (insufficient data).")

        return "\n".join(lines)
    except Exception as exc:
        return f"Error in technical analysis for {symbol}: {exc}"


def _exec_web_search(query: str, max_results: int = 5, recency: str = "month") -> str:
    from datetime import datetime
    today = datetime.today().strftime("%Y-%m-%d")

    results = search_text(query, max_results=max_results, recency=recency, prefer_provider="auto")

    if not results:
        return f"No web search results found for: {query}"

    lines = [f"Web Search Results for '{query}' (as of {today}):"]
    for i, r in enumerate(results[:max_results], 1):
        title = r.get("title") or "(title not available)"
        body = r.get("body", "")
        url = r.get("url", "")
        source = r.get("source", "")
        lines.append(f"\n  [{i}] {title}")
        if body:
            lines.append(f"      {body}")
        lines.append(f"      URL: {url}  ({source})")
    return "\n".join(lines)


def _exec_get_enterprise_metrics(symbol: str) -> str:
    result = db.get_enterprise_metrics(symbol)
    return _format_dict(result, f"{symbol} Enterprise Value Metrics")


def _exec_search_earnings_transcripts(query: str, symbol: str | None = None) -> str:
    import data_store
    results = data_store.search_transcripts(query, ticker=symbol, limit=3)
    if not results:
        return f"No transcript matches for '{query}'" + (f" ({symbol})" if symbol else "") + "."
    lines = [f"Transcript search results for '{query}':"]
    for r in results:
        tk = r.get("ticker", "?")
        fy = r.get("fiscal_year", "?")
        fq = r.get("fiscal_quarter", "?")
        text = r.get("full_text", "")
        # Find the most relevant snippet around the query terms
        snippet = _find_snippet(text, query, context_chars=500)
        lines.append(f"\n  --- {tk} FY{fy} Q{fq} ---")
        lines.append(f"  {snippet}")
    return "\n".join(lines)


def _find_snippet(text: str, query: str, context_chars: int = 500) -> str:
    """Find the most relevant snippet of text around query terms."""
    text_lower = text.lower()
    terms = query.lower().split()
    best_pos = 0
    for term in terms:
        pos = text_lower.find(term)
        if pos >= 0:
            best_pos = pos
            break
    start = max(0, best_pos - context_chars // 2)
    end = min(len(text), best_pos + context_chars // 2)
    snippet = text[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


# Tool dispatch table
_TOOL_DISPATCH: dict[str, Any] = {
    "get_stock_price": lambda args: _exec_get_stock_price(args["symbol"]),
    "get_stock_profile": lambda args: _exec_get_stock_profile(args["symbol"]),
    "get_fundamentals": lambda args: _exec_get_fundamentals(args["symbol"]),
    "get_income_statement": lambda args: _exec_get_income_statement(
        args["symbol"], args.get("quarterly", True)
    ),
    "get_balance_sheet": lambda args: _exec_get_balance_sheet(
        args["symbol"], args.get("quarterly", True)
    ),
    "get_cash_flow": lambda args: _exec_get_cash_flow(
        args["symbol"], args.get("quarterly", True)
    ),
    "get_dcf_analysis": lambda args: _exec_get_dcf_analysis(args["symbol"]),
    "get_margins": lambda args: _exec_get_margins(args["symbol"]),
    "get_growth_metrics": lambda args: _exec_get_growth_metrics(args["symbol"]),
    "get_earnings_transcripts": lambda args: _exec_get_earnings_transcripts(
        args["symbol"], args.get("limit", 2)
    ),
    "get_sec_filings": lambda args: _exec_get_sec_filings(args["symbol"]),
    "get_stock_news": lambda args: _exec_get_stock_news(args["symbol"]),
    "get_industry_comparison": lambda args: _exec_get_industry_comparison(args["symbol"]),
    "get_technical_analysis": lambda args: _exec_get_technical_analysis(
        args["symbol"], args.get("lookback_days", 180)
    ),
    "web_search": lambda args: _exec_web_search(
        args["query"], args.get("max_results", 5), args.get("recency", "month")
    ),
    "get_enterprise_metrics": lambda args: _exec_get_enterprise_metrics(args["symbol"]),
    "search_earnings_transcripts": lambda args: _exec_search_earnings_transcripts(
        args["query"], args.get("symbol")
    ),
}


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a named tool with the given arguments and return the result."""
    handler = _TOOL_DISPATCH.get(name)
    if handler is None:
        return f"Unknown tool: {name}"
    try:
        return handler(arguments)
    except Exception as exc:
        return f"Tool '{name}' failed: {exc}"


# ── Multi-agent system prompt ────────────────────────────────────────────────

def _build_agent_system_prompt() -> str:
    """Build the system prompt with the current date injected."""
    from datetime import datetime
    today = datetime.today().strftime("%B %d, %Y")
    return f"""\
You are a senior financial analyst AI assistant with access to real-time \
market data tools. You operate as a multi-agent system with specialized \
capabilities.

TODAY'S DATE: {today}

**Data Research Agent**: Fetches stock prices, company profiles, and news.
**Fundamental Analysis Agent**: Retrieves financial statements, valuations, \
margins, growth metrics, DCF analysis, and industry comparisons.
**Earnings & Regulatory Agent**: Accesses earnings call transcripts and SEC filings.
**Technical Analysis Agent**: Computes RSI, moving averages, ML predictions, \
and trading signals.
**Web Research Agent**: Searches the internet for real-time information using \
both DuckDuckGo and Google.

GUIDELINES:
- Use the tools to gather data BEFORE answering. Do not guess financial data.
- Call multiple tools when needed for comprehensive analysis.
- When a user asks about a stock, typically gather at minimum the stock price \
and fundamentals. Add technical analysis, news, or other data as relevant.
- For deep-dive analysis, include earnings transcripts, SEC filings, margins, \
and growth metrics.
- Use web_search for current events, market sentiment, or information not \
available through financial data tools.
- IMPORTANT: When using web_search, ALWAYS include the current year ({today[:4]}) \
in your search queries to ensure you get the most recent information. \
For example, search "AAPL earnings {today[:4]}" instead of just "AAPL earnings". \
The web_search tool filters results by recency by default.
- Cite specific data points from tool results in your answers.
- Present balanced analysis with both bullish and bearish perspectives.
- Never give direct buy/sell orders. Present arguments for both sides.
- When analysis context from a previous overview is provided, reference it.
- Keep responses thorough but structured with clear sections.
"""


AGENT_SYSTEM_PROMPT = _build_agent_system_prompt()


# ── Agent loop ───────────────────────────────────────────────────────────────

def run_agent_loop(
    client,
    model: str,
    messages: list[dict],
    *,
    max_iterations: int = 8,
    on_tool_call: Any | None = None,
) -> tuple[str, list[dict]]:
    """Execute the tool-calling loop until the LLM produces a final response.

    Parameters
    ----------
    client : OpenAI
        Configured OpenAI client.
    model : str
        Model identifier.
    messages : list[dict]
        Full conversation messages (system + history + user).
    max_iterations : int
        Maximum number of tool-calling rounds.
    on_tool_call : callable | None
        Optional callback ``fn(tool_name, args, result)`` for progress updates.

    Returns
    -------
    (final_text, tool_log) : tuple
        The LLM's final response text and a log of tool calls made.
    """
    tool_log: list[dict] = []

    for i in range(max_iterations):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                temperature=0.3,
                max_tokens=2500,
            )
        except Exception as exc:
            err_str = str(exc).lower()
            if "tool" in err_str or "function" in err_str:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2500,
                )
            else:
                raise

        choice = response.choices[0]
        msg = choice.message

        if not msg.tool_calls:
            return msg.content or "", tool_log

        messages.append(msg)

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            result = execute_tool(tc.function.name, args)

            tool_entry = {
                "tool": tc.function.name,
                "args": args,
                "result_preview": result[:300],
            }
            tool_log.append(tool_entry)

            if on_tool_call:
                try:
                    on_tool_call(tc.function.name, args, result)
                except Exception:
                    pass

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    final = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=2500,
    )
    return final.choices[0].message.content or "", tool_log


def run_agent_streaming(
    client,
    model: str,
    messages: list[dict],
    *,
    max_iterations: int = 8,
    on_tool_call: Any | None = None,
) -> tuple[Any, list[dict]]:
    """Like run_agent_loop but streams the final response.

    Returns (stream_generator, tool_log).
    """
    tool_log: list[dict] = []

    for i in range(max_iterations):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                temperature=0.3,
                max_tokens=2500,
            )
        except Exception as exc:
            err_str = str(exc).lower()
            if "tool" in err_str or "function" in err_str:
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2500,
                    stream=True,
                )
                return stream, tool_log
            raise

        choice = response.choices[0]
        msg = choice.message

        if not msg.tool_calls:
            if msg.content and tool_log:
                def _yield_existing(text=msg.content):
                    yield text
                return _yield_existing(), tool_log
            break

        messages.append(msg)

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            result = execute_tool(tc.function.name, args)

            tool_entry = {
                "tool": tc.function.name,
                "args": args,
                "result_preview": result[:300],
            }
            tool_log.append(tool_entry)

            if on_tool_call:
                try:
                    on_tool_call(tc.function.name, args, result)
                except Exception:
                    pass

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
        continue

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=2500,
        stream=True,
    )
    return stream, tool_log
