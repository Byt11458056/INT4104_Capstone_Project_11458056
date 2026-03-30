"""
Tools Hub — main landing page for the AI Financial Analysis Assistant.

Displays a grid of predefined financial tools that users can launch
without writing prompts.  Each tool is a zero-prompt workflow.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import data_store
from llm_summary import PROVIDERS
from sidebar_config import inject_sidebar_main_entry_hide_css

st.set_page_config(page_title="AI Financial Analyst", page_icon="📊", layout="wide")

inject_sidebar_main_entry_hide_css()

# ── Global LLM config defaults (persist across all pages in a session) ────
_GLOBAL_KEY_PREFIX = "global_llm"

def _gk(name: str) -> str:
    return f"{_GLOBAL_KEY_PREFIX}_{name}"

if _gk("provider") not in st.session_state:
    st.session_state[_gk("provider")] = "OpenRouter"
if _gk("model") not in st.session_state:
    st.session_state[_gk("model")] = PROVIDERS["OpenRouter"]["default_model"]

_CSS = """
<style>
    /* Mobile dock-style navigation */
    @media(max-width:768px) {
        /* Hide default sidebar */
        [data-testid="stSidebar"] {
            display: none;
        }

        /* Create bottom dock */
        [data-testid="stSidebarNav"] {
            display: flex !important;
            position: fixed !important;
            bottom: 0 !important;
            left: 0 !important;
            right: 0 !important;
            background: rgba(14, 17, 23, 0.95) !important;
            backdrop-filter: blur(10px) !important;
            border-top: 1px solid rgba(255,255,255,0.1) !important;
            padding: 0.5rem !important;
            z-index: 9999 !important;
            overflow-x: auto !important;
            height: auto !important;
        }
        [data-testid="stSidebarNav"] ul {
            display: flex !important;
            flex-direction: row !important;
            gap: 0.5rem !important;
            margin: 0 !important;
            padding: 0 !important;
            list-style: none !important;
        }
        [data-testid="stSidebarNav"] li {
            flex-shrink: 0 !important;
        }
        [data-testid="stSidebarNav"] a {
            padding: 0.5rem 1rem !important;
            border-radius: 8px !important;
            font-size: 0.85rem !important;
            white-space: nowrap !important;
        }
        /* Add bottom padding to main content */
        [data-testid="stAppViewContainer"] {
            padding-bottom: 70px !important;
        }
    }

    .tool-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
        padding: 1.2rem;
        height: 100%;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .tool-card:hover {
        border-color: rgba(80,160,255,0.5);
        box-shadow: 0 2px 12px rgba(80,160,255,0.15);
    }
    .tool-icon { font-size: 2rem; margin-bottom: 0.5rem; }
    .tool-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.3rem; }
    .tool-desc { font-size: 0.85rem; color: #888; margin-bottom: 0.8rem; }
    .disclaimer {
        background: rgba(255, 170, 0, 0.1);
        border: 1px solid rgba(255, 170, 0, 0.3);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 0.8rem;
        color: #aa7700;
        margin-bottom: 1rem;
    }
    @media (max-width: 768px) {
        [data-testid="column"] {
            width: 100% !important; flex: 100% !important; min-width: 100% !important;
        }
    }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# Auto-redirect to Market Overview
if "redirected" not in st.session_state:
    st.session_state["redirected"] = True
    st.switch_page("pages/1_Market_Overview.py")

st.markdown(
    '<div class="disclaimer">'
    '<strong>Disclaimer:</strong> This application does not provide financial advice. '
    'All analysis, predictions, and recommendations are for educational and reference purposes only. '
    'Users must conduct their own due diligence and bear their own investment risks.'
    '</div>',
    unsafe_allow_html=True,
)

st.title("AI Financial Analyst")
st.markdown(
    "Your AI-powered toolkit for financial research.  "
    "Pick a tool below to get started — no prompts needed."
)
st.divider()

TOOLS = [
    {
        "icon": "🌍", "title": "Market Overview",
        "desc": "Latest market conditions, sector trends, crypto, forex, and hot topics.",
        "page": "pages/1_Market_Overview.py",
    },
    {
        "icon": "📊", "title": "Overview Analysis",
        "desc": "Analyze performance, fundamentals, and get investor-specific advice for any asset.",
        "page": "pages/2_Overview_Analysis.py",
    },
    {
        "icon": "📰", "title": "News Summarizer",
        "desc": "AI-summarized financial news with sentiment scoring and political signals.",
        "page": "pages/3_News_Summarizer.py",
    },
    {
        "icon": "🔍", "title": "Investment Screener",
        "desc": "Discover investment opportunities by analyzing market trends and themes with AI.",
        "page": "pages/4_Investment_Screener.py",
    },
    {
        "icon": "📑", "title": "Financial Reports",
        "desc": "Analyze earnings, SEC filings, and financial statements with AI insights.",
        "page": "pages/5_Financial_Reports.py",
    },
    {
        "icon": "⚡", "title": "Backtest Lab",
        "desc": "Test trading strategies with historical data and AI-assisted optimization.",
        "page": "pages/7_Backtest_Lab.py",
    },
    {
        "icon": "💬", "title": "AI Chat",
        "desc": "Ask any financial question with full access to all data tools.",
        "page": "pages/8_AI_Chat.py",
    },
]

cols = st.columns(min(4, len(TOOLS)))
for i, tool in enumerate(TOOLS):
    with cols[i % len(cols)]:
        st.markdown(
            f"""<div class="tool-card">
                <div class="tool-icon">{tool['icon']}</div>
                <div class="tool-title">{tool['title']}</div>
                <div class="tool-desc">{tool['desc']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        if st.button("Launch", key=f"launch_{i}", width="stretch"):
            st.switch_page(tool["page"])
        st.markdown("")

st.divider()

# Recent activity
st.subheader("Recent Activity")
col_tools, col_chats = st.columns(2)

with col_tools:
    st.markdown("**Recent Tool Results**")
    try:
        results = data_store.get_all_tool_results(limit=5)
        if results:
            for r in results:
                icon = {"overview": "📊", "market": "🌍", "news": "📰",
                        "screener": "🔍", "reports": "📑", "portfolio": "💼",
                        "backtest": "⚡"}.get(r["tool_type"], "🔧")
                ticker_tag = f"**{r['ticker']}** " if r.get("ticker") else ""
                model_tag = f"*{r.get('model_id', '')}*" if r.get("model_id") else ""
                ts = r["created_at"][:16]
                st.markdown(f"{icon} {ticker_tag}{r['tool_type']} — {model_tag} — {ts}")
        else:
            st.caption("No tool results yet. Launch a tool above to get started!")
    except Exception:
        st.caption("No tool results yet.")

with col_chats:
    st.markdown("**Recent Conversations**")
    try:
        conversations = data_store.get_conversations(limit=5)
        if conversations:
            for conv in conversations:
                title = conv["title"]
                if conv.get("ticker"):
                    title = f"[{conv['ticker']}] {title}"
                ts = conv["updated_at"][:16]
                st.markdown(f"💬 {title} — {ts}")
        else:
            st.caption("No conversations yet. Try the AI Chat!")
    except Exception:
        st.caption("No conversations yet.")
