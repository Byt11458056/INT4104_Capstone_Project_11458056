"""
History — view past tool results and chat conversations with model metadata.

Every result is tagged with tool type, model used, provider, and timestamp
for evaluating AI model performance over time.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd

import data_store
from utils import format_pct

st.markdown("<style>@media(max-width:768px){[data-testid='column']{width:100%!important;flex:100%!important}}</style>", unsafe_allow_html=True)

st.title("History")
st.markdown("Review past tool results and conversations. Compare AI model performance.")

TOOL_ICONS = {
    "overview": "📊", "market_overview": "🌍", "news_summary": "📰",
    "screener": "🔍", "financial_reports": "📑", "backtest": "⚡",
}

tab_tools, tab_chats, tab_legacy = st.tabs(["Tool Results", "Chat History", "Legacy Analyses"])

# ── Tool Results ─────────────────────────────────────────────────────────────

with tab_tools:
    col_type, col_ticker, col_limit = st.columns([2, 2, 1])
    with col_type:
        filter_type = st.selectbox(
            "Filter by tool", ["All"] + list(TOOL_ICONS.keys()), key="hist_type",
        )
    with col_ticker:
        filter_ticker = st.text_input("Filter by ticker", value="", key="hist_ticker").upper().strip()
    with col_limit:
        limit = st.selectbox("Show", [10, 25, 50], index=1, key="hist_limit")

    results = data_store.get_tool_results(
        tool_type=filter_type if filter_type != "All" else None,
        ticker=filter_ticker or None,
        limit=limit,
    )

    if not results:
        st.info("No tool results yet. Run some tools to see history here.")
    else:
        st.caption(f"Showing {len(results)} result(s)")
        for r in results:
            icon = TOOL_ICONS.get(r["tool_type"], "🔧")
            ticker_tag = f"**{r['ticker']}** " if r.get("ticker") else ""
            model_tag = f"`{r.get('model_id', 'unknown')}`" if r.get("model_id") else ""
            provider_tag = f"via {r.get('provider', '')}" if r.get("provider") else ""
            ts = r["created_at"][:16]

            header = f"{icon} {ticker_tag}{r['tool_type']} — {model_tag} {provider_tag} — {ts}"

            with st.expander(header, expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tool", r["tool_type"])
                with col2:
                    st.metric("Model", r.get("model_id", "N/A"))
                with col3:
                    st.metric("Time", r["created_at"][:19])

                if r.get("ticker"):
                    st.write(f"**Ticker:** {r['ticker']}")
                if r.get("provider"):
                    st.write(f"**Provider:** {r['provider']}")

                if r.get("summary"):
                    st.divider()
                    st.markdown("**Summary**")
                    st.markdown(r["summary"])

                inputs = r.get("input")
                if inputs:
                    with st.expander("Inputs"):
                        st.json(inputs)

                result_data = r.get("result")
                if result_data:
                    with st.expander("Result Data"):
                        st.json(result_data)

                col_load, col_spacer = st.columns([1, 3])
                with col_load:
                    if st.button("Load in AI Chat", key=f"load_tool_{r['id']}"):
                        st.session_state["chat_with_analysis"] = {
                            "ticker": r.get("ticker", ""),
                            "profile": r.get("tool_type", ""),
                            "lookback": "",
                            "signal": {},
                            "sentiment": {},
                            "summary": r.get("summary", ""),
                            "llm_data": result_data or {},
                        }
                        st.session_state["chat_messages"] = []
                        st.switch_page("pages/8_AI_Chat.py")


# ── Chat History ─────────────────────────────────────────────────────────────

with tab_chats:
    conversations = data_store.get_conversations(limit=50)

    if not conversations:
        st.info("No conversations yet.")
    else:
        st.caption(f"Showing {len(conversations)} conversation(s)")
        for conv in conversations:
            title = conv["title"]
            if conv.get("ticker"):
                title = f"[{conv['ticker']}] {title}"
            msg_count = conv.get("message_count", 0)
            updated = conv["updated_at"][:16]

            with st.expander(f"{title} — {msg_count} msgs — {updated}", expanded=False):
                messages = data_store.get_messages(conv["id"])
                if messages:
                    for msg in messages:
                        icon = "👤" if msg["role"] == "user" else "🤖"
                        st.markdown(f"{icon} **{msg['role'].title()}** ({msg['created_at'][:16]})")
                        st.markdown(msg["content"])
                        tool_log = msg.get("tool_log")
                        if tool_log:
                            with st.expander("Tool calls"):
                                for entry in tool_log:
                                    st.markdown(f"- **{entry.get('tool', '?')}**({entry.get('args', {})})")
                                    st.caption(entry.get("result_preview", "")[:200])
                        st.divider()

                col_load, col_delete = st.columns(2)
                with col_load:
                    if st.button("Load in Chat", key=f"load_conv_{conv['id']}", width="stretch"):
                        msgs = data_store.get_messages(conv["id"])
                        st.session_state["chat_messages"] = [
                            {"role": m["role"], "content": m["content"]}
                            for m in msgs if m["role"] in ("user", "assistant")
                        ]
                        st.session_state["current_conversation_id"] = conv["id"]
                        st.switch_page("pages/8_AI_Chat.py")
                with col_delete:
                    if st.button("Delete", key=f"del_conv_{conv['id']}", width="stretch"):
                        data_store.delete_conversation(conv["id"])
                        st.rerun()


# ── Legacy Analyses ──────────────────────────────────────────────────────────

with tab_legacy:
    st.caption("Analyses from the previous version of the app.")
    analyses = data_store.get_analysis_history(limit=25)

    if not analyses:
        st.info("No legacy analyses found.")
    else:
        for rec in analyses:
            sig_icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(rec["signal"], "⚪")
            header = (
                f"{sig_icon} **{rec['ticker']}** — {rec['signal']} "
                f"(score {rec['score']:.2f}) — "
                f"{rec['trader_type']} · {rec['lookback']} — "
                f"{rec['created_at'][:16]}"
            )
            with st.expander(header, expanded=False):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Signal", rec["signal"])
                    st.metric("Score", f"{rec['score']:.4f}")
                with c2:
                    st.metric("Sentiment", f"{rec['sentiment']:+.3f}")
                    st.metric("Trader Type", rec["trader_type"])
                with c3:
                    st.metric("Lookback", rec["lookback"])
                    st.metric("Date", rec["created_at"][:10])

                if rec.get("summary"):
                    st.divider()
                    st.markdown("**AI Summary**")
                    st.markdown(rec["summary"])

                details = rec.get("details")
                if details:
                    bt_metrics = details.get("backtest_metrics")
                    if bt_metrics:
                        st.divider()
                        st.markdown("**Backtest Results**")
                        strat = bt_metrics.get("strategy", {})
                        bh = bt_metrics.get("buy_and_hold", {})
                        bc1, bc2 = st.columns(2)
                        with bc1:
                            st.write(f"- Strategy Return: {format_pct(strat.get('cumulative_return', 0))}")
                            st.write(f"- Sharpe: {strat.get('sharpe_ratio', 0):.2f}")
                        with bc2:
                            st.write(f"- B&H Return: {format_pct(bh.get('cumulative_return', 0))}")
                            st.write(f"- B&H Sharpe: {bh.get('sharpe_ratio', 0):.2f}")
