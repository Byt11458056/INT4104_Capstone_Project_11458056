"""
Investment Screener — AI-powered discovery of investment opportunities
by analyzing recent market trends, news, and emerging themes.

No hardcoded ticker lists. The LLM analyzes current market conditions
and dynamically suggests tickers to screen.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from sidebar_config import render_llm_sidebar, LLMConfig
from data_pipeline import get_stock_data
from news_scraper import get_topic_news
import data_store

st.markdown("<style>@media(max-width:768px){[data-testid='column']{width:100%!important;flex:100%!important}}</style>", unsafe_allow_html=True)

_QUICK_PROMPTS = [
    "High momentum tech stocks showing breakout signals",
    "Undervalued dividend stocks in a defensive sector",
    "Emerging AI and semiconductor plays beyond the mega-caps",
    "Clean energy stocks benefiting from recent policy changes",
    "Beaten-down growth stocks with recovery potential",
    "Crypto and blockchain-related assets gaining momentum",
    "Small-cap biotech with upcoming catalysts",
    "Commodities and resources benefiting from inflation",
]

with st.sidebar:
    st.header("Screener Settings")
    st.markdown("Pick a quick prompt **or** describe what you're looking for.")

    quick = st.selectbox(
        "Quick prompts",
        ["(custom — write below)"] + _QUICK_PROMPTS,
        key="scr_quick",
    )

    _use_custom = quick == "(custom — write below)"
    custom_query = ""
    if _use_custom:
        custom_query = st.text_area(
            "Describe what you're looking for",
            value="",
            placeholder="e.g. undervalued small-cap growth stocks with strong earnings momentum",
            height=100,
            key="scr_query",
        )

    user_query = custom_query if _use_custom else quick

    lookback = st.selectbox("Performance period", ["1 week", "1 month", "3 months"], index=1, key="scr_lb")
    lb_days = {"1 week": 7, "1 month": 30, "3 months": 90}[lookback]
    st.divider()
    llm_cfg: LLMConfig = render_llm_sidebar("scr")
    st.divider()
    run_btn = st.button("Discover", type="primary", width="stretch", key="scr_run")

st.title("Investment Screener")
st.markdown(
    "AI analyzes current market news and trends to discover investment "
    "opportunities matching your criteria. No predefined lists — every "
    "search is a fresh market analysis."
)


def _scan_tickers(tickers: list[str], start: str, end: str) -> list[dict]:
    results = []
    for t in tickers:
        try:
            df = get_stock_data(t, start, end)
            if df is None or df.empty or len(df) < 2:
                continue
            latest = df["Close"].iloc[-1]
            first = df["Close"].iloc[0]
            ret = (latest - first) / first
            high = df["High"].max()
            low = df["Low"].min()
            vol = df["Close"].pct_change().std()
            from_high = (latest - high) / high
            results.append({
                "Ticker": t,
                "Price": latest,
                "Return": ret,
                "High": high,
                "Low": low,
                "Volatility": vol,
                "From High": from_high,
            })
        except Exception:
            continue
    return results


def _ai_discover_tickers(query: str, news_text: str, llm_cfg: LLMConfig) -> tuple[list[str], str]:
    """Ask the LLM to analyze market conditions and suggest tickers."""
    from openai import OpenAI
    client = OpenAI(
        api_key=llm_cfg.api_key, base_url=llm_cfg.base_url,
        default_headers=llm_cfg.extra_headers,
    )
    resp = client.chat.completions.create(
        model=llm_cfg.model,
        messages=[
            {"role": "system", "content": (
                "You are a financial market analyst. The user wants to discover "
                "investment opportunities. Based on their query AND the recent "
                "market news provided, suggest 10-15 specific tickers (stocks, "
                "ETFs, or crypto with -USD suffix) that best match their criteria.\n\n"
                "IMPORTANT: Analyze the current market conditions from the news to "
                "make informed suggestions. Don't just list well-known names — "
                "find opportunities that match the query AND current market dynamics.\n\n"
                "Return a JSON object with two fields:\n"
                "- \"tickers\": array of ticker symbols\n"
                "- \"reasoning\": brief explanation of why these tickers were chosen "
                "based on current market analysis\n\n"
                "Example: {\"tickers\": [\"NVDA\", \"AMD\", \"SMCI\"], \"reasoning\": "
                "\"AI infrastructure spending is accelerating based on recent earnings...\"}"
            )},
            {"role": "user", "content": (
                f"Query: {query}\n\n"
                f"Recent market news:\n{news_text}"
            )},
        ],
        temperature=0.4, max_tokens=500,
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()
    parsed = json.loads(raw)
    tickers = parsed.get("tickers", [])
    reasoning = parsed.get("reasoning", "")
    return tickers, reasoning


if run_btn:
    if not user_query.strip():
        st.warning("Please describe what you're looking for.")
        st.stop()

    if not llm_cfg.api_key or not llm_cfg.base_url:
        st.error("An LLM provider is required for AI-powered discovery. Configure one in the sidebar.")
        st.stop()

    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=lb_days)).strftime("%Y-%m-%d")

    # Step 1: Gather market intelligence
    with st.status("Gathering market intelligence...", expanded=True) as status:
        news = get_topic_news(user_query, max_results=12)
        general_news = get_topic_news("stock market today", max_results=5)
        all_news = news + general_news
        news_text = "\n".join(f"- {n.get('title', '')}" for n in all_news[:15])
        status.update(label=f"Collected {len(all_news)} news articles", state="complete")

    # Step 2: AI analyzes market and suggests tickers
    with st.status("AI analyzing market conditions and discovering tickers...") as status:
        try:
            tickers, ai_reasoning = _ai_discover_tickers(user_query, news_text, llm_cfg)
            status.update(label=f"Discovered {len(tickers)} opportunities", state="complete")
        except Exception as exc:
            st.error(f"AI discovery failed: {exc}")
            st.caption("Try rephrasing your query or check your LLM configuration.")
            st.stop()

    if not tickers:
        st.warning("AI couldn't find matching tickers. Try a different query.")
        st.stop()

    # Show AI reasoning
    st.subheader("AI Market Analysis")
    st.info(ai_reasoning)

    # Step 3: Scan discovered tickers
    with st.status(f"Scanning {len(tickers)} discovered assets...") as status:
        results = _scan_tickers(tickers, start, end)
        status.update(label=f"Scanned {len(results)} of {len(tickers)} assets", state="complete")

    if not results:
        st.warning("No price data available for the discovered tickers.")
        st.stop()

    df_results = pd.DataFrame(results).sort_values("Return", ascending=False)

    # Top movers
    st.subheader(f"Discovered Opportunities ({lookback})")
    top_n = min(3, len(df_results))
    bottom_n = min(3, len(df_results))
    top_3 = df_results.head(top_n)
    bottom_3 = df_results.tail(bottom_n)

    col_top, col_bot = st.columns(2)
    with col_top:
        st.markdown("**Top Performers**")
        for _, r in top_3.iterrows():
            st.markdown(f"**{r['Ticker']}** — ${r['Price']:,.2f} ({r['Return']:+.2%})")
    with col_bot:
        st.markdown("**Laggards / Potential Value**")
        for _, r in bottom_3.iterrows():
            st.markdown(f"**{r['Ticker']}** — ${r['Price']:,.2f} ({r['Return']:+.2%})")

    # Full table
    df_display = df_results.copy()
    df_display["Price"] = df_display["Price"].apply(lambda x: f"${x:,.2f}")
    df_display["Return"] = df_display["Return"].apply(lambda x: f"{x:+.2%}")
    df_display["High"] = df_display["High"].apply(lambda x: f"${x:,.2f}")
    df_display["Low"] = df_display["Low"].apply(lambda x: f"${x:,.2f}")
    df_display["Volatility"] = df_display["Volatility"].apply(lambda x: f"{x:.4f}")
    df_display["From High"] = df_display["From High"].apply(lambda x: f"{x:+.2%}")

    with st.expander("Full Screening Results", expanded=True):
        st.dataframe(df_display, width="stretch", hide_index=True)

    # Related news
    if news:
        with st.expander(f"Related News ({len(news)} articles)", expanded=False):
            for n in news:
                title = n.get("title", "")
                url = n.get("url", "")
                source = n.get("source", "")
                if url:
                    st.markdown(f"- [{title}]({url}) — *{source}*")
                else:
                    st.markdown(f"- {title} — *{source}*")

    # Step 4: AI investment insights
    with st.status("Generating AI investment insights...") as status:
        from openai import OpenAI
        table_text = df_results.to_string(index=False)
        try:
            client = OpenAI(api_key=llm_cfg.api_key, base_url=llm_cfg.base_url, default_headers=llm_cfg.extra_headers)
            resp = client.chat.completions.create(
                model=llm_cfg.model,
                messages=[
                    {"role": "system", "content": (
                        "You are a financial analyst helping investors discover opportunities. "
                        "Based on the screening results, recent news, and your market analysis, provide:\n"
                        "1. Key theme/trend analysis based on current market conditions\n"
                        "2. Top 3 picks with specific reasoning (fundamentals, momentum, catalysts)\n"
                        "3. Contrarian/value picks (beaten-down names with recovery potential)\n"
                        "4. Key risk factors and what could go wrong\n"
                        "5. Suggested approach (entry timing, position sizing considerations)\n"
                        "Be specific, actionable, and grounded in the data provided."
                    )},
                    {"role": "user", "content": (
                        f"User's investment thesis: {user_query}\n\n"
                        f"Performance data ({lookback}):\n{table_text}\n\n"
                        f"Recent news:\n{news_text[:2000]}"
                    )},
                ],
                temperature=0.4, max_tokens=1000,
            )
            status.update(label="Insights ready", state="complete")
            st.subheader("AI Investment Insights")
            st.markdown(resp.choices[0].message.content.strip())
        except Exception as exc:
            st.warning(f"AI insights failed: {exc}")

    # Save results for navigation
    st.session_state["scr_last"] = {
        "query": user_query,
        "lookback": lookback,
        "tickers": [r["Ticker"] for r in results],
        "reasoning": ai_reasoning,
        "results": results,
    }

    try:
        data_store.save_tool_result(
            "screener", model_id=llm_cfg.model, provider=llm_cfg.provider,
            inputs={"query": user_query, "lookback": lookback, "tickers_discovered": tickers},
            result={"top": results[0]["Ticker"] if results else None, "count": len(results),
                    "reasoning": ai_reasoning},
            summary=f"AI discovered {len(results)} opportunities for: {user_query[:60]}",
        )
    except Exception:
        pass

# Navigation — outside run_btn so buttons survive reruns
_scr = st.session_state.get("scr_last")
if _scr:
    st.divider()
    _tickers = _scr.get("tickers", [])
    sel_ticker = st.selectbox("Dive deeper into", _tickers, key="scr_deep") if _tickers else None
    col_ov, col_chat = st.columns(2)
    with col_ov:
        if sel_ticker and st.button("Open Overview Analysis", key="scr_to_ov"):
            st.session_state["ov_ticker"] = sel_ticker
            st.switch_page("pages/2_Overview_Analysis.py")
    with col_chat:
        if st.button("Discuss in AI Chat →", type="primary", key="scr_to_chat", width="stretch"):
            st.session_state["chat_with_analysis"] = {
                "ticker": ", ".join(_tickers[:5]),
                "tool_type": "screener",
                "profile": "screener",
                "lookback": _scr.get("lookback", ""),
                "summary": _scr.get("reasoning", ""),
                "result": {"tickers": _tickers, "reasoning": _scr.get("reasoning", "")},
                "llm_data": {"tickers": _tickers, "reasoning": _scr.get("reasoning", "")},
                "inputs": {"query": _scr.get("query", "")},
            }
            st.session_state["chat_messages"] = []
            st.switch_page("pages/8_AI_Chat.py")
