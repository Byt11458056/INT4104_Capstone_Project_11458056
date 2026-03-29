"""
LLM-powered market summary generation with RAG-augmented context.

Supports OpenRouter, Alibaba Qwen (DashScope), or any OpenAI-compatible
endpoint.  Generates investor-type-aware advice when a profile is provided.
"""

from __future__ import annotations

import textwrap
from datetime import datetime

from openai import OpenAI

from config import get_secret

PROVIDERS = {
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "default_model": "openai/gpt-4o-mini",
        "models": [
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "openai/gpt-4.1-nano",
            "openai/gpt-4.1-mini",
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3.5-haiku",
            "google/gemini-2.0-flash-001",
            "google/gemini-2.5-pro-preview-03-25",
            "deepseek/deepseek-chat-v3-0324",
            "deepseek/deepseek-r1",
            "meta-llama/llama-4-maverick",
            "meta-llama/llama-4-scout",
        ],
    },
    "Alibaba Qwen (DashScope)": {
        "base_url": "https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
        "default_model": "qwen-plus",
        "models": ["qwen-plus", "qwen-turbo", "qwen-max", "qwen-long"],
    },
    "Custom Provider": {
        "base_url": "",
        "env_key": "",
        "default_model": "",
        "models": [],
    },
}

_OPENROUTER_EXTRA_HEADERS = {
    "HTTP-Referer": "http://localhost:8501",
    "X-Title": "AI Financial Analyst",
}

SYSTEM_PROMPT = (
    "You are a senior financial analyst. Given market data, technical indicators, "
    "news headlines, article excerpts, and fundamental data for an investment product, "
    "produce a concise briefing that covers:\n"
    "1. Recent performance summary (price trend, momentum)\n"
    "2. Technical analysis highlights\n"
    "3. Fundamental assessment (if data available)\n"
    "4. News sentiment overview\n"
    "5. Key risk factors\n"
    "6. Bullish and bearish arguments\n"
    "7. Tailored advice based on the investor profile\n\n"
    "IMPORTANT: Focus on forward-looking analysis from the current date/time. "
    "Analyze recent price action (last 24h, last week) and upcoming catalysts. "
    "When article excerpts are provided, cite them by number (e.g. [1], [2]).\n"
    "Keep the tone professional and balanced. Present arguments for both sides."
)


def _build_prompt(
    data: dict,
    rag_context: list[dict] | None = None,
    investor_hint: str | None = None,
) -> str:
    ticker = data.get("ticker", "N/A")
    close = data.get("close", "N/A")
    change_pct = data.get("change_pct", "N/A")
    rsi = data.get("rsi", "N/A")
    ma_5 = data.get("ma_5", "N/A")
    ma_20 = data.get("ma_20", "N/A")
    volatility = data.get("volatility", "N/A")
    sentiment = data.get("sentiment_score", "N/A")
    asset_type = data.get("asset_type", "stock")

    headlines = data.get("headlines", [])
    headline_block = ""
    for h in headlines[:8]:
        if isinstance(h, dict):
            title = h.get("text", h.get("title", ""))
            url = h.get("url", "")
            headline_block += f"  - [{title}]({url})\n" if url else f"  - {title}\n"
        else:
            headline_block += f"  - {h}\n"
    if not headline_block:
        headline_block = "  (none available)\n"

    rag_block = ""
    if rag_context:
        rag_block = "\nRelevant article excerpts (cite by number):\n"
        for i, ctx in enumerate(rag_context, 1):
            source = ctx.get("source", "Unknown")
            url = ctx.get("url", "")
            excerpt = ctx.get("text", "")[:500]
            ref = f"({url})" if url else ""
            rag_block += f'  [{i}] "{excerpt}" — {source} {ref}\n'

    fundamentals_block = ""
    if data.get("fundamentals"):
        fundamentals_block = "\nFundamental data:\n"
        for k, v in data["fundamentals"].items():
            fundamentals_block += f"  {k}: {v}\n"

    dcf_block = ""
    if data.get("dcf"):
        dcf_block = "\nDCF valuation:\n"
        for k, v in data["dcf"].items():
            dcf_block += f"  {k}: {v}\n"

    margins_block = ""
    if data.get("margins"):
        margins_block = "\nMargins:\n"
        for k, v in data["margins"].items():
            margins_block += f"  {k}: {v}\n"

    growth_block = ""
    if data.get("growth"):
        growth_block = "\nGrowth metrics (YoY):\n"
        for k, v in data["growth"].items():
            growth_block += f"  {k}: {v}\n"

    investor_block = ""
    if investor_hint:
        investor_block = f"\nINVESTOR PROFILE: {investor_hint}\n"

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    return textwrap.dedent(f"""\
        Current date and time: {current_time}

        Asset type: {asset_type}
        Ticker: {ticker}
        Latest close: {close}
        Daily change: {change_pct}
        RSI (14): {rsi}
        MA-short: {ma_5}
        MA-long: {ma_20}
        Volatility: {volatility}
        Aggregate news sentiment (-1 bearish ... +1 bullish): {sentiment}

        Recent headlines:
        {headline_block}
        {rag_block}
        {fundamentals_block}
        {dcf_block}
        {margins_block}
        {growth_block}
        {investor_block}
    """)


def _resolve_base_url(provider: str, preset: dict, *, base_url_override: str | None = None) -> str:
    if base_url_override:
        return base_url_override.rstrip("/")
    if provider == "Alibaba Qwen (DashScope)":
        return (get_secret("DASHSCOPE_BASE_URL") or preset["base_url"]).rstrip("/")
    return preset["base_url"].rstrip("/")


def generate_llm_summary(
    data: dict,
    *,
    provider: str = "OpenRouter",
    model: str | None = None,
    api_key: str | None = None,
    rag_context: list[dict] | None = None,
    base_url: str | None = None,
    extra_headers: dict | None = None,
    investor_hint: str | None = None,
) -> str:
    preset = PROVIDERS.get(provider, PROVIDERS["OpenRouter"])
    api_key = api_key or (get_secret(preset["env_key"]) if preset.get("env_key") else None)
    if not api_key:
        return _fallback_summary(data)

    if not base_url:
        base_url = _resolve_base_url(provider, preset)
    model = model or preset["default_model"]
    if extra_headers is None:
        extra_headers = _OPENROUTER_EXTRA_HEADERS if provider == "OpenRouter" else {}

    client = OpenAI(api_key=api_key, base_url=base_url, default_headers=extra_headers)
    user_msg = _build_prompt(data, rag_context=rag_context, investor_hint=investor_hint)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.4,
            max_tokens=1200,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        err = str(exc).lower()
        hint = ""
        if provider == "Alibaba Qwen (DashScope)" and ("403" in err or "access" in err):
            hint = (
                "\n\n**Hint:** Check that your DashScope region matches "
                "where your API key was created."
            )
        return f"**LLM request failed:** `{exc}`{hint}"


def _fallback_summary(data: dict) -> str:
    ticker = data.get("ticker", "N/A")
    sentiment = data.get("sentiment_score", 0)
    rsi = data.get("rsi", 50)
    bias = "bullish" if sentiment > 0.1 else ("bearish" if sentiment < -0.1 else "neutral")
    rsi_note = "overbought" if rsi > 70 else ("oversold" if rsi < 30 else "neutral range")
    return textwrap.dedent(f"""\
        **{ticker} — Automated Summary** *(LLM unavailable — template mode)*

        **Outlook:** News sentiment leans **{bias}** (score {sentiment:+.2f})
        and RSI sits in the **{rsi_note}** ({rsi:.1f}).

        **Risk factors:**
        - No LLM enrichment — this summary is purely rule-based.
        - Sentiment is derived from a small headline sample.

        **Note:** Configure an LLM provider in the sidebar for full AI analysis.
    """)
