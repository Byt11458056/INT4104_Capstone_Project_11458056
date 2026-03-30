"""
Market Overview — real-time TradingView widgets, Polymarket sentiment, and AI summary.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import streamlit.components.v1 as components

from sidebar_config import render_llm_sidebar, LLMConfig
from news_scraper import get_topic_news, get_trump_posts
import data_store

st.markdown(
    "<style>@media(max-width:768px){[data-testid='column']{width:100%!important;flex:100%!important}}</style>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Market Overview")
    llm_cfg: LLMConfig = render_llm_sidebar("mkt")

st.title("Market Overview")
st.markdown("Track your watchlist with real-time charts and AI-powered analysis.")

# ── Interactive Charts — user picks tickers ───────────────────────────────
st.subheader("Watchlist")
st.caption("Add tickers below to monitor. Each chart has full TradingView controls — draw, add indicators, and change timeframes.")

import json as _json

_DEFAULT_WATCHLIST = ["AAPL", "NVDA", "SPY", "BTC-USD"]
if "mkt_watchlist" not in st.session_state:
    st.session_state["mkt_watchlist"] = list(_DEFAULT_WATCHLIST)

col_add, col_reset = st.columns([4, 1])
with col_add:
    new_ticker = st.text_input(
        "Add ticker", value="", key="mkt_add_ticker",
        placeholder="e.g. TSLA, QQQ, ETH-USD, EURUSD=X",
    )
with col_reset:
    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
    if st.button("Reset", key="mkt_reset_wl"):
        st.session_state["mkt_watchlist"] = list(_DEFAULT_WATCHLIST)
        st.rerun()

if new_ticker:
    clean = new_ticker.upper().strip()
    if clean and clean not in st.session_state["mkt_watchlist"]:
        st.session_state["mkt_watchlist"].append(clean)
        st.rerun()

watchlist = st.session_state["mkt_watchlist"]

if watchlist:
    # Show positions with prices and P&L
    try:
        import yfinance as yf
        for ticker in watchlist:
            holding = data_store.get_holding_by_ticker(ticker, "watchlist")

            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.markdown(f"**{ticker}**")

            # Fetch current price
            try:
                tk = yf.Ticker(ticker)
                info = tk.info
                current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
                prev_close = info.get("previousClose", current_price)
                change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0

                with col2:
                    color = "green" if change_pct >= 0 else "red"
                    st.markdown(f"${current_price:.2f} <span style='color:{color}'>({change_pct:+.2f}%)</span>", unsafe_allow_html=True)
            except Exception:
                current_price = 0
                with col2:
                    st.markdown("Price unavailable")

            with col3:
                if holding:
                    shares = holding.get("quantity", 0)
                    cost = holding.get("avg_cost", 0)
                    pnl = (current_price - cost) * shares if current_price else 0
                    pnl_pct = ((current_price - cost) / cost * 100) if cost else 0
                    pnl_color = "green" if pnl >= 0 else "red"
                    st.markdown(f"<span style='color:{pnl_color}'>{shares} @ ${cost:.2f} | P&L: ${pnl:,.0f} ({pnl_pct:+.1f}%)</span>", unsafe_allow_html=True)
                    if st.button("Edit", key=f"edit_{ticker}"):
                        st.session_state[f"editing_{ticker}"] = True
                        st.rerun()
                else:
                    if st.button("+ Add Position", key=f"add_{ticker}"):
                        st.session_state[f"editing_{ticker}"] = True
                        st.rerun()

            # Edit position form
            if st.session_state.get(f"editing_{ticker}"):
                with st.form(key=f"form_{ticker}"):
                    st.markdown(f"**Position for {ticker}**")
                    shares = st.number_input("Shares", min_value=0.0, value=holding.get("quantity", 0.0) if holding else 0.0, step=1.0)
                    avg_cost = st.number_input("Avg Cost ($)", min_value=0.0, value=holding.get("avg_cost", 0.0) if holding else 0.0, step=0.01)

                    col_save, col_remove, col_cancel = st.columns(3)
                    with col_save:
                        if st.form_submit_button("Save"):
                            if shares > 0 and avg_cost > 0:
                                if holding:
                                    data_store.update_holding(ticker, shares, avg_cost, "watchlist")
                                else:
                                    data_store.add_holding(ticker, shares, avg_cost, "long", "watchlist")
                            st.session_state[f"editing_{ticker}"] = False
                            st.rerun()
                    with col_remove:
                        if holding and st.form_submit_button("Remove"):
                            data_store.remove_holding(holding["id"])
                            st.session_state[f"editing_{ticker}"] = False
                            st.rerun()
                    with col_cancel:
                        if st.form_submit_button("Cancel"):
                            st.session_state[f"editing_{ticker}"] = False
                            st.rerun()

            st.divider()
    except Exception as e:
        st.error(f"Error loading positions: {e}")

    st.markdown("---")
    st.subheader("Charts")
    remove_ticker = st.pills(
        "Remove from watchlist", watchlist, key="mkt_remove_pill",
        selection_mode="single",
    )
    if remove_ticker and remove_ticker in watchlist:
        watchlist.remove(remove_ticker)
        st.session_state["mkt_watchlist"] = watchlist
        st.rerun()


def _tv_symbol(tkr: str) -> str:
    """Map ticker format to TradingView symbol."""
    if tkr.endswith("-USD"):
        base = tkr.replace("-USD", "")
        return f"BINANCE:{base}USDT"
    if "=X" in tkr:
        return f"FX:{tkr.replace('=X', '')}"
    return tkr


_chart_cols = 2 if len(watchlist) > 1 else 1
for row_start in range(0, len(watchlist), _chart_cols):
    cols = st.columns(_chart_cols)
    for col_idx, col in enumerate(cols):
        idx = row_start + col_idx
        if idx >= len(watchlist):
            break
        tkr = watchlist[idx]
        tv_sym = _tv_symbol(tkr)
        with col:
            chart_html = f"""
            <div id="tvc_{idx}" style="height:420px;">
              <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
              <script type="text/javascript">
              new TradingView.widget({{
                "autosize": true,
                "symbol": "{tv_sym}",
                "interval": "D",
                "timezone": "Etc/UTC",
                "theme": "dark",
                "style": "1",
                "locale": "en",
                "enable_publishing": false,
                "allow_symbol_change": true,
                "studies": ["RSI@tv-basicstudies","MASimple@tv-basicstudies"],
                "container_id": "tvc_{idx}",
                "hide_side_toolbar": false,
                "withdateranges": true,
                "details": true
              }});
              </script>
            </div>
            """
            components.html(chart_html, height=440)

# ── AI Market Summary ─────────────────────────────────────────────────────
st.divider()
st.subheader("AI Market Summary")
if st.button("Generate AI Summary", key="mkt_ai_btn", type="primary"):
    if not llm_cfg.api_key or not llm_cfg.base_url:
        st.warning("Configure an LLM provider in the sidebar for AI summaries.")
    else:
        with st.status("Gathering headlines & generating summary...", expanded=True) as status:
            market_news = get_topic_news("stock market today", max_results=10)
            trump_news = get_trump_posts("economy tariffs trade")

            from news_scraper import search_polymarket
            # Fetch multiple themed prediction markets
            pm_politics = search_polymarket("Trump election politics", max_results=3)
            pm_crypto = search_polymarket("Bitcoin Ethereum crypto", max_results=3)
            pm_economy = search_polymarket("economy recession inflation", max_results=3)
            pm_markets = pm_politics + pm_crypto + pm_economy

            # Get portfolio holdings
            holdings = data_store.get_holdings("watchlist")
            portfolio_text = ""
            if holdings:
                import yfinance as yf
                portfolio_text = "\n\nUser's portfolio:\n"
                for h in holdings[:10]:
                    ticker = h.get("ticker", "")
                    shares = h.get("quantity", 0)
                    cost = h.get("avg_cost", 0)
                    try:
                        current = yf.Ticker(ticker).info.get("currentPrice", cost)
                        pnl_pct = ((current - cost) / cost * 100) if cost else 0
                        portfolio_text += f"- {ticker}: {shares} shares @ ${cost:.2f}, P&L: {pnl_pct:+.1f}%\n"
                    except:
                        portfolio_text += f"- {ticker}: {shares} shares @ ${cost:.2f}\n"

            status.update(label="Summarizing...", state="running")

            headlines = "\n".join(f"- {a['title']}" for a in (market_news + trump_news)[:15])

            pm_text = ""
            if pm_markets:
                pm_text = "\n\nPrediction market signals:\n" + "\n".join(
                    f"- {p.get('question', '')}: {p.get('odds', '')}" for p in pm_markets
                )

            from datetime import datetime
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            from openai import OpenAI
            try:
                client = OpenAI(
                    api_key=llm_cfg.api_key, base_url=llm_cfg.base_url,
                    default_headers=llm_cfg.extra_headers,
                )
                resp = client.chat.completions.create(
                    model=llm_cfg.model,
                    messages=[
                        {"role": "system", "content": (
                            f"Current date and time: {current_time}\n\n"
                            "Summarize these financial headlines and prediction market signals "
                            "into a concise market briefing: 3-5 key themes, overall market "
                            "sentiment, and notable risks/opportunities. Focus on recent developments "
                            "(last 24h, last week) and forward-looking outlook. "
                            "If portfolio data is provided, mention relevant impacts to those holdings. Be actionable."
                        )},
                        {"role": "user", "content": headlines + pm_text + portfolio_text},
                    ],
                    temperature=0.3, max_tokens=600,
                )
                st.session_state["mkt_ai_summary"] = resp.choices[0].message.content.strip()
                status.update(label="Summary ready", state="complete")
            except Exception as exc:
                st.warning(f"LLM summary failed: {exc}")
                status.update(label="Failed", state="error")

        try:
            from datetime import datetime
            data_store.save_tool_result(
                "market_overview", model_id=llm_cfg.model, provider=llm_cfg.provider,
                inputs={"date": datetime.today().strftime("%Y-%m-%d")},
                summary=st.session_state.get("mkt_ai_summary", ""),
            )
        except Exception:
            pass

if "mkt_ai_summary" in st.session_state:
    st.markdown(st.session_state["mkt_ai_summary"])

    if st.button("Discuss in AI Chat →", type="primary", key="mkt_to_chat", width="stretch"):
        st.session_state["chat_with_analysis"] = {
            "ticker": "Market Overview",
            "tool_type": "market_overview",
            "profile": "market_overview",
            "lookback": "",
            "summary": st.session_state["mkt_ai_summary"],
            "result": {},
            "llm_data": {},
            "inputs": {},
        }
        st.session_state["chat_messages"] = []
        st.switch_page("pages/8_AI_Chat.py")
