"""
AI Chat — multi-agent conversational assistant with tool-calling.

Uses DefeatBeta API tools, web search, technical analysis, and can
reference any tool result from the history.  Conversations are persisted.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from openai import OpenAI

from sidebar_config import render_llm_sidebar, LLMConfig
from agents import _build_agent_system_prompt, run_agent_streaming, TOOLS
import data_store

st.markdown("<style>@media(max-width:768px){[data-testid='column']{width:100%!important;flex:100%!important}}</style>", unsafe_allow_html=True)


def _build_context_message(analysis: dict) -> str:
    """Build a rich context string from any tool result for the LLM."""
    import json as _json

    ticker = analysis.get("ticker") or "N/A"
    tool_type = analysis.get("tool_type") or analysis.get("profile") or "analysis"
    summary = analysis.get("summary") or ""

    parts = [
        f"=== Loaded context: {tool_type} for {ticker} ===",
        "The user loaded the following tool result. Use it to answer follow-up questions.",
    ]

    lookback = analysis.get("lookback")
    if lookback:
        parts.append(f"Lookback: {lookback}")

    sig = analysis.get("signal")
    if isinstance(sig, dict) and sig:
        parts.append(f"Signal: {sig.get('signal', 'N/A')} (score {sig.get('score', 0):.2f})")

    sentiment = analysis.get("sentiment")
    if isinstance(sentiment, dict) and "avg_score" in sentiment:
        parts.append(f"Average news sentiment: {sentiment.get('avg_score', 0):+.3f}")

    result_data = analysis.get("llm_data") or analysis.get("result") or {}
    if isinstance(result_data, dict) and result_data:
        parts.append("\nDetailed data:")
        for k, v in result_data.items():
            if isinstance(v, dict):
                parts.append(f"  {k}:")
                for k2, v2 in v.items():
                    parts.append(f"    {k2}: {v2}")
            elif isinstance(v, list) and v:
                parts.append(f"  {k}: {_json.dumps(v[:10], default=str)}")
            elif v is not None:
                parts.append(f"  {k}: {v}")

    inputs = analysis.get("inputs")
    if isinstance(inputs, dict) and inputs:
        parts.append(f"\nInputs/settings: {_json.dumps(inputs, default=str)}")

    fundamentals = analysis.get("fundamentals")
    if isinstance(fundamentals, dict) and fundamentals:
        parts.append("\nFundamentals:")
        for k, v in fundamentals.items():
            parts.append(f"  {k}: {v}")

    if summary:
        parts.append(f"\nSummary:\n{summary[:3000]}")

    return "\n".join(parts)


def _auto_title(content: str) -> str:
    title = content.strip()[:60]
    if len(content) > 60:
        title += "..."
    return title


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Chat Settings")
    llm_cfg: LLMConfig = render_llm_sidebar("chat")
    st.divider()

    if st.button("New conversation", width="stretch", key="chat_clear"):
        st.session_state.pop("chat_messages", None)
        st.session_state.pop("chat_with_analysis", None)
        st.session_state.pop("current_conversation_id", None)
        st.session_state.pop("tool_log_display", None)
        st.rerun()

    # Load tool result as context
    st.subheader("Load Tool Result")
    try:
        recent_results = data_store.get_all_tool_results(limit=10)
        if recent_results:
            result_options = {
                f"{r['tool_type']} — {r.get('ticker', '')} — {r['created_at'][:16]} ({r.get('model_id', '')})": r["id"]
                for r in recent_results
            }
            selected_result = st.selectbox("Select result", ["(none)"] + list(result_options.keys()), key="chat_load_result")
            if selected_result != "(none)" and st.button("Load as Context", key="chat_load_btn"):
                result_id = result_options[selected_result]
                result_data = data_store.get_tool_result_by_id(result_id)
                if result_data:
                    _r = result_data.get("result") or {}
                    _i = result_data.get("input") or {}
                    st.session_state["chat_with_analysis"] = {
                        "ticker": result_data.get("ticker") or "",
                        "tool_type": result_data.get("tool_type") or "",
                        "profile": result_data.get("tool_type") or "",
                        "lookback": "",
                        "summary": result_data.get("summary") or "",
                        "result": _r,
                        "llm_data": _r,
                        "inputs": _i,
                    }
                    st.success(f"Loaded {result_data.get('tool_type', 'tool')} result")
    except Exception:
        pass

    if "chat_with_analysis" in st.session_state:
        a = st.session_state["chat_with_analysis"]
        st.success(f"Context: **{a.get('ticker', 'N/A')}** ({a.get('profile', a.get('investor_type', ''))})")

    st.divider()
    st.subheader("Conversation History")
    conversations = data_store.get_conversations(limit=15)
    if conversations:
        for conv in conversations:
            col_title, col_del = st.columns([4, 1])
            with col_title:
                label = conv["title"]
                if conv.get("ticker"):
                    label = f"[{conv['ticker']}] {label}"
                if st.button(label, key=f"conv_{conv['id']}", width="stretch"):
                    messages = data_store.get_messages(conv["id"])
                    st.session_state["chat_messages"] = [
                        {"role": m["role"], "content": m["content"]}
                        for m in messages if m["role"] in ("user", "assistant")
                    ]
                    st.session_state["current_conversation_id"] = conv["id"]
                    st.session_state.pop("chat_with_analysis", None)
                    st.session_state.pop("tool_log_display", None)
                    st.rerun()
            with col_del:
                if st.button("X", key=f"del_{conv['id']}"):
                    data_store.delete_conversation(conv["id"])
                    if st.session_state.get("current_conversation_id") == conv["id"]:
                        st.session_state.pop("chat_messages", None)
                        st.session_state.pop("current_conversation_id", None)
                    st.rerun()
    else:
        st.caption("No conversations yet.")


# ── Page ─────────────────────────────────────────────────────────────────────

st.title("AI Chat")

analysis_ref = st.session_state.get("chat_with_analysis")
if analysis_ref:
    st.markdown(
        f"Context loaded: **{analysis_ref.get('ticker', 'N/A')}** "
        f"({analysis_ref.get('profile', analysis_ref.get('investor_type', ''))}). "
        "Ask follow-up questions or request deeper analysis."
    )
else:
    st.markdown(
        "Ask any financial question. The AI has access to market data, "
        "financial statements, earnings, SEC filings, and web search."
    )

if st.session_state.get("tool_log_display"):
    with st.expander("Agent tool calls (last response)", expanded=False):
        for entry in st.session_state["tool_log_display"]:
            st.markdown(f"**{entry.get('tool', '?')}**({entry.get('args', {})})")
            st.caption(entry.get("result_preview", "")[:200])

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

for msg in st.session_state["chat_messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask anything about markets, stocks, or strategy..."):
    st.session_state["chat_messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if not llm_cfg.api_key:
        with st.chat_message("assistant"):
            st.warning("No API key configured.")
        st.session_state["chat_messages"].append({"role": "assistant", "content": "No API key configured."})
        st.stop()

    conv_id = st.session_state.get("current_conversation_id")
    if conv_id is None:
        ticker_hint = analysis_ref.get("ticker") if analysis_ref else None
        conv_id = data_store.create_conversation(_auto_title(user_input), ticker=ticker_hint)
        st.session_state["current_conversation_id"] = conv_id

    data_store.save_message(conv_id, "user", user_input)

    system_parts = [_build_agent_system_prompt()]
    if analysis_ref:
        system_parts.append("\n\n" + _build_context_message(analysis_ref))

    messages_for_api = [{"role": "system", "content": "".join(system_parts)}]
    for msg in st.session_state["chat_messages"]:
        messages_for_api.append({"role": msg["role"], "content": msg["content"]})

    client = OpenAI(
        api_key=llm_cfg.api_key, base_url=llm_cfg.base_url,
        default_headers=llm_cfg.extra_headers,
    )

    with st.chat_message("assistant"):
        tool_log: list[dict] = []
        streamed_text = ""
        try:
            tool_status = st.status("Analyzing...", expanded=True)

            def on_tool_call(name: str, args: dict, result: str):
                symbol = args.get("symbol", args.get("query", ""))
                tool_status.write(f"**{name}**({symbol})")

            stream, tool_log = run_agent_streaming(
                client, llm_cfg.model, messages_for_api,
                max_iterations=8, on_tool_call=on_tool_call,
            )
            if tool_log:
                tool_status.update(label=f"Gathered data ({len(tool_log)} tool calls)", state="complete")
            else:
                tool_status.update(label="Responding...", state="complete")

            streamed_text = st.write_stream(stream)

        except Exception as exc:
            err = str(exc).lower()
            hint = ""
            if llm_cfg.provider == "Alibaba Qwen (DashScope)" and ("403" in err or "access" in err):
                hint = "\n\n**Hint:** Check DashScope region matches your API key."
            streamed_text = f"**Error:** `{exc}`{hint}"
            st.error(streamed_text)

    st.session_state["chat_messages"].append({"role": "assistant", "content": streamed_text})
    data_store.save_message(conv_id, "assistant", streamed_text, tool_log=tool_log or None)

    if tool_log:
        st.session_state["tool_log_display"] = tool_log

    st.rerun()
