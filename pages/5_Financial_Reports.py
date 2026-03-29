"""
Financial Reports — analyze earnings, SEC filings, and financial statements.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import streamlit as st
import pandas as pd

from sidebar_config import render_llm_sidebar, LLMConfig
import defeatbeta_client as db
import data_store
from web_research import search_text

log = logging.getLogger(__name__)

st.markdown("<style>@media(max-width:768px){[data-testid='column']{width:100%!important;flex:100%!important}}</style>", unsafe_allow_html=True)


def _fmt_number(val):
    """Format large numbers with K/M/B suffixes."""
    if val is None:
        return ""
    if not isinstance(val, (int, float)):
        return str(val)
    abs_val = abs(val)
    if abs_val >= 1e9:
        return f"{val / 1e9:,.2f}B"
    if abs_val >= 1e6:
        return f"{val / 1e6:,.2f}M"
    if abs_val >= 1e3:
        return f"{val / 1e3:,.1f}K"
    return f"{val:,.2f}"


def _render_statement(records: list[dict], label: str, ticker: str, quarterly: bool):
    """Display a financial statement from records, formatting numbers."""
    if not records:
        st.info(f"No {'quarterly' if quarterly else 'annual'} {label} data for {ticker}.")
        return
    df = pd.DataFrame(records)
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        df[col] = df[col].apply(_fmt_number)
    st.dataframe(df, width="stretch", hide_index=True)


def _chunk_text(text: str, max_chunk: int = 3500) -> list[str]:
    """Split text into chunks that fit within LLM context."""
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > max_chunk:
            if current:
                chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def _summarize_transcript(full_text: str, llm_cfg: LLMConfig, ticker: str, quarter_label: str) -> str:
    """Summarize a full transcript, chunking if necessary."""
    from openai import OpenAI
    client = OpenAI(
        api_key=llm_cfg.api_key, base_url=llm_cfg.base_url,
        default_headers=llm_cfg.extra_headers,
    )

    if len(full_text) <= 6000:
        resp = client.chat.completions.create(
            model=llm_cfg.model,
            messages=[
                {"role": "system", "content": (
                    f"Summarize this {ticker} {quarter_label} earnings call transcript. "
                    "Provide: 1) Key financial highlights, 2) Forward guidance, "
                    "3) Management tone and confidence, 4) Key risks mentioned, "
                    "5) Notable Q&A points. Be thorough but concise."
                )},
                {"role": "user", "content": full_text},
            ],
            temperature=0.3, max_tokens=800,
        )
        return resp.choices[0].message.content.strip()

    # For long transcripts: summarize each chunk, then combine
    chunks = _chunk_text(full_text, max_chunk=3500)
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        resp = client.chat.completions.create(
            model=llm_cfg.model,
            messages=[
                {"role": "system", "content": (
                    f"Summarize this section ({i+1}/{len(chunks)}) of {ticker} "
                    f"{quarter_label} earnings call. Extract key points, numbers, "
                    "guidance, and notable statements. Be concise."
                )},
                {"role": "user", "content": chunk},
            ],
            temperature=0.2, max_tokens=400,
        )
        chunk_summaries.append(resp.choices[0].message.content.strip())

    combined = "\n\n---\n\n".join(chunk_summaries)
    final_resp = client.chat.completions.create(
        model=llm_cfg.model,
        messages=[
            {"role": "system", "content": (
                f"Below are summaries of all sections of {ticker} {quarter_label} "
                "earnings call transcript. Synthesize into ONE comprehensive summary with: "
                "1) Key financial highlights, 2) Forward guidance, "
                "3) Management tone, 4) Key risks, 5) Notable Q&A points."
            )},
            {"role": "user", "content": combined},
        ],
        temperature=0.3, max_tokens=800,
    )
    return final_resp.choices[0].message.content.strip()


def _web_search_earnings(ticker: str) -> list[dict]:
    """Search the web for recent earnings info when DefeatBeta has none."""
    return search_text(
        f"{ticker} earnings call transcript 2026",
        max_results=5,
        recency="month",
        prefer_provider="auto",
    )


with st.sidebar:
    st.header("Report Settings")
    ticker = st.text_input("Ticker", value="AAPL", key="fr_ticker").upper().strip()
    quarterly = st.toggle("Quarterly (vs Annual)", value=True, key="fr_quarterly")
    st.divider()
    llm_cfg: LLMConfig = render_llm_sidebar("fr")
    st.divider()
    run_btn = st.button("Fetch Reports", type="primary", width="stretch", key="fr_run")

st.title("Financial Reports")
st.markdown("Explore financial statements, earnings calls, and SEC filings with AI analysis.")

# Persist fetched data in session_state so it survives reruns (AI Summary clicks)
if run_btn:
    with st.spinner("Loading financial data..."):
        st.session_state["fr_data"] = {
            "ticker": ticker,
            "quarterly": quarterly,
            "income": db.get_income_statement(ticker, quarterly=quarterly),
            "balance": db.get_balance_sheet(ticker, quarterly=quarterly),
            "cashflow": db.get_cash_flow(ticker, quarterly=quarterly),
            "transcripts": db.get_earnings_transcripts(ticker, limit=4),
            "filings": db.get_sec_filings(ticker, limit=15),
        }
        # Store transcripts in SQLite FTS5 for searchability
        for t_data in st.session_state["fr_data"]["transcripts"]:
            text = t_data.get("text", "")
            if text:
                try:
                    data_store.save_transcript(
                        ticker,
                        int(t_data.get("fiscal_year", 0)),
                        int(t_data.get("fiscal_quarter", 0)),
                        t_data.get("report_date", ""),
                        text,
                    )
                except Exception:
                    pass

    try:
        data_store.save_tool_result(
            "financial_reports", ticker=ticker,
            model_id=llm_cfg.model, provider=llm_cfg.provider,
            inputs={"quarterly": quarterly},
            summary=f"Financial reports fetched for {ticker}",
        )
    except Exception:
        pass

# Display data from session_state (persists across button clicks)
fr = st.session_state.get("fr_data")
if fr:
    tab_is, tab_bs, tab_cf, tab_earn, tab_sec = st.tabs(
        ["Income Statement", "Balance Sheet", "Cash Flow", "Earnings Calls", "SEC Filings"]
    )

    with tab_is:
        _render_statement(fr["income"], "income statement", fr["ticker"], fr["quarterly"])

    with tab_bs:
        _render_statement(fr["balance"], "balance sheet", fr["ticker"], fr["quarterly"])

    with tab_cf:
        _render_statement(fr["cashflow"], "cash flow", fr["ticker"], fr["quarterly"])

    with tab_earn:
        transcripts = fr["transcripts"]
        if transcripts:
            for t_data in transcripts:
                fy = t_data.get("fiscal_year", "?")
                fq = t_data.get("fiscal_quarter", "?")
                date = t_data.get("report_date", "")
                text = t_data.get("text", "No transcript available.")
                quarter_label = f"FY{fy} Q{fq}"

                with st.expander(f"{quarter_label} — {date}"):
                    # Preview (truncated)
                    preview_len = 2000
                    st.markdown(text[:preview_len])
                    if len(text) > preview_len:
                        st.caption(f"Showing first {preview_len} of {len(text):,} characters.")

                    # Download full transcript
                    col_dl, col_ai = st.columns(2)
                    with col_dl:
                        st.download_button(
                            f"Download Full Transcript",
                            data=text,
                            file_name=f"{fr['ticker']}_{quarter_label.replace(' ', '_')}_transcript.txt",
                            mime="text/plain",
                            key=f"dl_{fy}_{fq}",
                        )

                    # AI Summary
                    with col_ai:
                        if llm_cfg.api_key and llm_cfg.base_url:
                            ai_btn = st.button(
                                f"AI Summary",
                                key=f"earn_ai_{fy}_{fq}",
                            )
                        else:
                            ai_btn = False
                            st.caption("Configure LLM for AI summary")

                    # Check if summary was requested
                    summary_key = f"summary_{fr['ticker']}_{fy}_{fq}"
                    if ai_btn:
                        with st.spinner(f"Summarizing {quarter_label} ({len(text):,} chars)..."):
                            try:
                                summary = _summarize_transcript(
                                    text, llm_cfg, fr["ticker"], quarter_label,
                                )
                                st.session_state[summary_key] = summary
                            except Exception as exc:
                                st.error(f"AI summary failed: {exc}")

                    # Display stored summary
                    if summary_key in st.session_state:
                        st.markdown("---")
                        st.markdown("**AI Summary:**")
                        st.markdown(st.session_state[summary_key])
        else:
            st.info(f"No earnings transcripts from DefeatBeta for {fr['ticker']}. Searching the web...")
            web_results = _web_search_earnings(fr["ticker"])
            if web_results:
                for wr in web_results:
                    url = wr.get("url", "")
                    title = wr.get("title", "")
                    body = wr.get("body", "")
                    if url:
                        st.markdown(f"- [{title}]({url})")
                    else:
                        st.markdown(f"- {title}")
                    if body:
                        st.caption(body[:200])
            else:
                st.info("No earnings data available from any source.")

    with tab_sec:
        filings = fr["filings"]
        if filings:
            for f in filings:
                f_type = f.get("type", f.get("form_type", f.get("filing_type", "Unknown")))
                date = f.get("report_date", f.get("filed_date", f.get("filing_date", "")))
                desc = f.get("description", f.get("title", ""))
                url = f.get("url", f.get("link", f.get("filing_url", "")))
                if url:
                    st.markdown(f"- **{f_type}** ({date}) — [{desc or 'View Filing'}]({url})")
                else:
                    st.markdown(f"- **{f_type}** ({date}) — {desc}")
        else:
            st.info("No SEC filings available.")

    st.divider()
    if st.button("Discuss in AI Chat →", type="primary", key="fr_to_chat", width="stretch"):
        st.session_state["chat_with_analysis"] = {
            "ticker": fr["ticker"],
            "tool_type": "financial_reports",
            "profile": "financial_reports",
            "lookback": "",
            "summary": f"Financial reports for {fr['ticker']}",
            "result": {
                "has_income": bool(fr.get("income")),
                "has_balance": bool(fr.get("balance")),
                "has_cashflow": bool(fr.get("cashflow")),
                "transcript_count": len(fr.get("transcripts", [])),
                "filing_count": len(fr.get("filings", [])),
            },
            "llm_data": {
                "has_income": bool(fr.get("income")),
                "has_balance": bool(fr.get("balance")),
                "has_cashflow": bool(fr.get("cashflow")),
                "transcript_count": len(fr.get("transcripts", [])),
                "filing_count": len(fr.get("filings", [])),
            },
            "inputs": {"ticker": fr["ticker"], "quarterly": fr.get("quarterly", True)},
        }
        st.session_state["chat_messages"] = []
        st.switch_page("pages/8_AI_Chat.py")
