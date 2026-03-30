"""
Overview Analysis — AI-powered analysis of investment products.

Covers recent performance, technical indicators, fundamentals, news sentiment,
and generates tailored advice based on investor type.  Supports stocks, ETFs,
crypto, and forex.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

from sidebar_config import render_llm_sidebar, LLMConfig
from data_pipeline import get_stock_data, create_features, detect_asset_type, get_data_cutoff_info
from news_scraper import get_news
from sentiment_analysis import analyze_sentiment
from llm_summary import generate_llm_summary
from utils import format_pct, INVESTOR_TYPES, InvestorType
import rag_engine
import data_store
import defeatbeta_client as db

_CSS = """
<style>
    @media (max-width: 768px) {
        [data-testid="column"] {
            width: 100% !important; flex: 100% !important; min-width: 100% !important;
        }
    }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Analysis Settings")
    ticker = st.text_input("Ticker symbol", value="AAPL", key="ov_ticker").upper().strip()
    st.caption("Stocks: AAPL, TSLA  |  ETFs: SPY, QQQ  |  Crypto: BTC-USD  |  Forex: EURUSD=X")

    st.subheader("Investor Profile")
    inv_key = st.selectbox(
        "Investor type",
        list(INVESTOR_TYPES.keys()),
        format_func=lambda k: INVESTOR_TYPES[k].label,
        index=1,
        key="ov_investor",
    )
    inv_type: InvestorType = INVESTOR_TYPES[inv_key]
    st.caption(inv_type.description)

    lookback = st.selectbox(
        "Lookback period",
        ["1 month", "3 months", "6 months", "1 year", "2 years"],
        index=2,
        key="ov_lookback",
    )
    lookback_days = {"1 month": 30, "3 months": 90, "6 months": 180, "1 year": 365, "2 years": 730}[lookback]

    include_political = st.checkbox("Include political signals", value=False, key="ov_political")

    st.divider()
    llm_cfg: LLMConfig = render_llm_sidebar("ov")
    st.divider()
    run_btn = st.button("Analyse", type="primary", width="stretch", key="ov_run")

# ── Page header ──────────────────────────────────────────────────────────────

st.title("Overview Analysis")
st.markdown(
    "Enter a ticker and investor profile to get a comprehensive AI briefing "
    "on performance, fundamentals, sentiment, and tailored advice."
)

# ── Main pipeline ────────────────────────────────────────────────────────────

if run_btn:
    from datetime import datetime, timedelta
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    asset_type = detect_asset_type(ticker)

    with st.status("Fetching market data...", expanded=True) as status:
        try:
            raw_df = get_stock_data(ticker, start_date, end_date)
        except Exception as exc:
            st.error(f"Failed to fetch data for **{ticker}**: {exc}")
            st.stop()
        feat_df = create_features(raw_df)
        if feat_df.empty or len(feat_df) < 2:
            st.error(f"Insufficient data for **{ticker}** with {lookback} lookback.")
            st.stop()

        realtime = db.get_realtime_quote(ticker)
        status.update(label="Market data ready", state="complete")

    with st.status("Fetching & analysing news...") as status:
        articles = get_news(ticker, include_political=include_political)
        sentiment = analyze_sentiment(
            articles,
            api_key=llm_cfg.api_key or None,
            base_url=llm_cfg.base_url or None,
            model=llm_cfg.model or None,
            extra_headers=llm_cfg.extra_headers or None,
        )
        for h in sentiment["headlines"]:
            for a in articles:
                if isinstance(a, dict) and a.get("title") == h.get("text"):
                    a["sentiment_score"] = h.get("score", 0.0)
                    break
        status.update(label="Sentiment ready", state="complete")

    with st.status("Building knowledge base...") as status:
        try:
            rag_engine.store_articles(articles, ticker)
            rag_context = rag_engine.retrieve_relevant(
                f"{ticker} market outlook risks opportunities analysis", ticker, top_k=5,
            )
        except Exception:
            rag_context = []
        status.update(label="RAG ready", state="complete")

    fundamentals = None
    dcf = None
    margins = None
    growth = None

    if asset_type == "stock":
        with st.status("Fetching fundamentals...") as status:
            fundamentals = db.get_fundamentals(ticker)
            dcf = db.get_dcf_valuation(ticker)
            margins = db.get_margins(ticker)
            growth = db.get_growth_metrics(ticker)
            status.update(label="Fundamentals ready", state="complete")

    # Determine current price: prefer real-time, fallback to last historical
    hist_close = float(feat_df["Close"].iloc[-1])
    current_price = hist_close
    data_as_of = str(feat_df.index[-1])[:10]
    if realtime and realtime.get("last_price", 0) > 0:
        current_price = realtime["last_price"]
        data_as_of = "real-time"

    with st.status("Generating AI summary...") as status:
        headline_dicts = [
            {"text": h["text"], "url": h.get("url", ""), "source": h.get("source", "")}
            for h in sentiment["headlines"]
        ]
        llm_data = {
            "ticker": ticker,
            "asset_type": asset_type,
            "close": round(current_price, 2),
            "hist_close": round(hist_close, 2),
            "change_pct": format_pct(float(feat_df["returns"].iloc[-1])),
            "rsi": round(float(feat_df["rsi"].iloc[-1]), 2),
            "ma_5": round(float(feat_df["ma_5"].iloc[-1]), 2),
            "ma_20": round(float(feat_df["ma_20"].iloc[-1]), 2),
            "volatility": round(float(feat_df["volatility"].iloc[-1]), 4),
            "sentiment_score": round(sentiment["avg_score"], 3),
            "headlines": headline_dicts,
            "fundamentals": fundamentals,
            "dcf": dcf,
            "margins": margins,
            "growth": growth,
        }
        summary_text = generate_llm_summary(
            llm_data,
            provider=llm_cfg.provider,
            model=llm_cfg.model,
            api_key=llm_cfg.api_key or None,
            rag_context=rag_context,
            base_url=llm_cfg.base_url,
            extra_headers=llm_cfg.extra_headers,
            investor_hint=inv_type.prompt_hint,
        )
        status.update(label="Summary ready", state="complete")

    try:
        data_store.save_tool_result(
            "overview",
            ticker=ticker,
            model_id=llm_cfg.model,
            provider=llm_cfg.provider,
            inputs={"investor_type": inv_key, "lookback": lookback, "asset_type": asset_type},
            result=llm_data,
            summary=summary_text,
        )
        data_store.save_articles(articles, ticker)
    except Exception:
        pass

    st.session_state["last_analysis"] = {
        "ticker": ticker,
        "asset_type": asset_type,
        "investor_type": inv_type.label,
        "lookback": lookback,
        "sentiment": sentiment,
        "summary": summary_text,
        "articles": articles,
        "rag_context": rag_context,
        "llm_data": llm_data,
        "fundamentals": fundamentals,
        "dcf": dcf,
        "margins": margins,
        "growth": growth,
        "model_id": llm_cfg.model,
        "provider": llm_cfg.provider,
    }

    # ── Display ──────────────────────────────────────────────────────────

    first_close = feat_df["Close"].iloc[0]
    period_return = (hist_close - first_close) / first_close

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${current_price:.2f}",
                delta=f"{((current_price - hist_close) / hist_close * 100):+.2f}% vs hist" if data_as_of == "real-time" else None)
    col2.metric("Period Return", format_pct(period_return))
    col3.metric("RSI", f"{feat_df['rsi'].iloc[-1]:.1f}")
    col4.metric("Sentiment", f"{sentiment['avg_score']:+.3f}")

    cutoff_info = get_data_cutoff_info(ticker)
    db_cutoff = cutoff_info.get("defeatbeta_cutoff")
    source_parts = []
    if db_cutoff:
        source_parts.append(f"DefeatBeta data through {db_cutoff}")
    hist_end = str(feat_df.index[-1])[:10]
    if hist_end != db_cutoff:
        source_parts.append(f"yfinance gap-fill through {hist_end}")
    if data_as_of == "real-time":
        source_parts.append(f"Real-time price: ${current_price:.2f}")
    else:
        source_parts.append("Real-time quote unavailable")
    st.caption(" | ".join(source_parts))

    # TradingView interactive chart
    st.subheader("Price & Technical Indicators")

    def _tv_symbol(tkr: str, atype: str) -> str:
        """Map our ticker format to TradingView symbol format."""
        if atype == "crypto":
            base = tkr.replace("-USD", "").replace("-EUR", "").replace("-GBP", "")
            return f"BINANCE:{base}USDT"
        if atype == "forex":
            return f"FX:{tkr.replace('=X', '')}"
        return tkr

    tv_sym = _tv_symbol(ticker, asset_type)
    tv_interval = "D"
    if lookback_days <= 30:
        tv_interval = "60"
    elif lookback_days <= 90:
        tv_interval = "D"

    tv_html = f"""
    <div id="tv_chart_container" style="height:560px;">
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "autosize": true,
        "symbol": "{tv_sym}",
        "interval": "{tv_interval}",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "studies": ["RSI@tv-basicstudies", "MASimple@tv-basicstudies"],
        "container_id": "tv_chart_container",
        "hide_side_toolbar": false,
        "withdateranges": true,
        "details": true,
        "calendar": false
      }});
      </script>
    </div>
    """
    components.html(tv_html, height=580)

    # Fundamentals (now scalar values, not nested dicts)
    if fundamentals:
        st.divider()
        st.subheader("Fundamental Data")

        _FMT = {
            "pe_ttm": ("P/E (TTM)", "{:.2f}"),
            "pb": ("P/B", "{:.2f}"),
            "ps": ("P/S", "{:.2f}"),
            "roe": ("ROE", "{:.2%}"),
            "roa": ("ROA", "{:.2%}"),
            "roic": ("ROIC", "{:.2%}"),
            "wacc": ("WACC", "{:.2%}"),
            "peg": ("PEG", "{:.2f}"),
            "market_cap": ("Market Cap", "${:,.0f}"),
        }
        display_items = []
        for key, val in fundamentals.items():
            label, fmt = _FMT.get(key, (key.upper().replace("_", " "), "{}"))
            if isinstance(val, dict):
                display_items.append((label, str(val)))
            elif isinstance(val, (int, float)):
                display_items.append((label, fmt.format(val)))
            else:
                display_items.append((label, str(val)))

        n_cols = min(len(display_items), 5)
        if n_cols > 0:
            fund_cols = st.columns(n_cols)
            for i, (label, display_val) in enumerate(display_items):
                with fund_cols[i % n_cols]:
                    st.metric(label, display_val)

        if dcf:
            st.markdown("**DCF Valuation**")
            dcf_items = []
            for k, v in dcf.items():
                label = k.replace("_", " ").title()
                if isinstance(v, float):
                    dcf_items.append((label, f"{v:.2f}"))
                else:
                    dcf_items.append((label, str(v)))
            if dcf_items:
                dcf_cols = st.columns(min(len(dcf_items), 4))
                for i, (label, dval) in enumerate(dcf_items):
                    with dcf_cols[i % len(dcf_cols)]:
                        st.metric(label, dval)

    if margins or growth:
        st.divider()
        mc, gc = st.columns(2)
        if margins:
            with mc:
                st.subheader("Margins")
                for k, v in margins.items():
                    label = k.replace("_", " ").title()
                    if isinstance(v, (int, float)):
                        st.write(f"- **{label}**: {v:.2%}")
                    else:
                        st.write(f"- **{label}**: {v}")
        if growth:
            with gc:
                st.subheader("Growth (YoY)")
                for k, v in growth.items():
                    label = k.replace("_", " ").title()
                    if isinstance(v, (int, float)):
                        st.write(f"- **{label}**: {v:.2%}")
                    else:
                        st.write(f"- **{label}**: {v}")

    st.divider()

    tab_summary, tab_news = st.tabs(["AI Analysis", "News Sources"])

    with tab_summary:
        st.markdown(summary_text)
        if rag_context:
            with st.expander("Sources referenced by the AI"):
                for i, ctx in enumerate(rag_context, 1):
                    title = ctx.get("title", "Untitled")
                    url = ctx.get("url", "")
                    source = ctx.get("source", "")
                    if url:
                        st.markdown(f"**[{i}]** [{title}]({url}) — *{source}*")
                    else:
                        st.markdown(f"**[{i}]** {title} — *{source}*")

    with tab_news:
        if sentiment["headlines"]:
            for h in sentiment["headlines"]:
                score = h["score"]
                emoji = "🟢" if score > 0.1 else ("🔴" if score < -0.1 else "⚪")
                title = h.get("text", "")
                url = h.get("url", "")
                source = h.get("source", "")
                link_text = f"[{title}]({url})" if url else title
                source_tag = f" — *{source}*" if source else ""
                st.markdown(f"{emoji} {link_text}{source_tag}  \n`sentiment {score:+.2f}`")
        else:
            st.info("No articles found.")

    st.divider()
    st.markdown(
        f"**Analysis complete for {ticker}** ({inv_type.label}, {lookback}) "
        f"using *{llm_cfg.model}* via {llm_cfg.provider}"
    )

# ── Persist navigation buttons outside run_btn so they survive reruns ─────
_last = st.session_state.get("last_analysis")
if _last:
    col_chat, col_backtest = st.columns(2)
    with col_chat:
        if st.button("Discuss in AI Chat →", type="primary", key="ov_to_chat", width="stretch"):
            st.session_state["chat_with_analysis"] = {
                "ticker": _last.get("ticker", ""),
                "tool_type": "overview",
                "profile": _last.get("investor_type", ""),
                "lookback": _last.get("lookback", ""),
                "summary": _last.get("summary", ""),
                "llm_data": _last.get("llm_data", {}),
                "result": _last.get("llm_data", {}),
                "inputs": {"investor_type": _last.get("investor_type", ""), "lookback": _last.get("lookback", "")},
                "sentiment": _last.get("sentiment", {}),
                "fundamentals": _last.get("fundamentals"),
            }
            st.session_state["chat_messages"] = []
            st.switch_page("pages/8_AI_Chat.py")
    with col_backtest:
        if st.button("Run Backtest →", key="ov_to_bt", width="stretch"):
            st.session_state["bt_ticker"] = _last.get("ticker", "AAPL")
            st.switch_page("pages/7_Backtest_Lab.py")
