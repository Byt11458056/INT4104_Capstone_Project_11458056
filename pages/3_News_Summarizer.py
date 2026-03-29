"""
News Summarizer — AI-summarized financial news with sentiment and political signals.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

from sidebar_config import render_llm_sidebar, LLMConfig
from news_scraper import get_news, get_topic_news, get_trump_posts, search_polymarket
from sentiment_analysis import analyze_sentiment
import data_store

st.markdown("<style>@media(max-width:768px){[data-testid='column']{width:100%!important;flex:100%!important}}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.header("News Settings")
    search_mode = st.radio("Search by", ["Ticker", "Topic"], key="news_mode")
    if search_mode == "Ticker":
        query = st.text_input("Ticker", value="AAPL", key="news_ticker").upper().strip()
    else:
        query = st.text_input("Topic", value="AI stocks", key="news_topic").strip()
    include_trump = st.checkbox("Include Trump/political signals", value=True, key="news_trump")
    st.divider()
    llm_cfg: LLMConfig = render_llm_sidebar("news")
    st.divider()
    run_btn = st.button("Summarize", type="primary", width="stretch", key="news_run")

st.title("News Summarizer")
st.markdown("Get AI-summarized news with sentiment scoring.")

if run_btn:
    with st.status("Fetching news...", expanded=True) as status:
        if search_mode == "Ticker":
            articles = get_news(query, include_political=include_trump)
        else:
            articles = get_topic_news(query, max_results=15)
            if include_trump:
                articles += get_trump_posts(query)

        # Fetch themed prediction markets and filter by relevance
        pm_themes = {
            "politics": ["Trump", "election", "president", "congress", "policy", "government"],
            "crypto": ["Bitcoin", "Ethereum", "crypto", "blockchain", "BTC", "ETH"],
            "economy": ["economy", "recession", "inflation", "GDP", "unemployment", "Fed"],
            "tech": ["AI", "tech", "technology", "software", "Apple", "Google", "Microsoft"],
            "market": ["stock", "market", "S&P", "Nasdaq", "Dow", "trading"]
        }

        # Determine which themes match the query
        query_lower = query.lower()
        relevant_themes = []
        for theme, keywords in pm_themes.items():
            if any(kw.lower() in query_lower for kw in keywords):
                relevant_themes.append(theme)

        # If no specific theme, use general market queries
        if not relevant_themes:
            relevant_themes = ["market", "economy"]

        # Fetch prediction markets for relevant themes
        pm_markets = []
        for theme in relevant_themes[:3]:  # Max 3 themes
            theme_query = " ".join(pm_themes[theme][:3])
            pm_markets.extend(search_polymarket(theme_query, max_results=2))

        # Also search directly with user query
        pm_markets.extend(search_polymarket(query, max_results=3))

        # Deduplicate by URL
        seen_urls = set()
        unique_pm = []
        for pm in pm_markets:
            url = pm.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_pm.append(pm)

        pm_markets = unique_pm[:5]  # Keep top 5

        status.update(label=f"Found {len(articles)} articles, {len(pm_markets)} prediction markets", state="complete")

    if not articles:
        st.warning("No articles found. Try a different query.")
        st.stop()

    with st.status("Analyzing sentiment...") as status:
        sentiment = analyze_sentiment(
            articles,
            api_key=llm_cfg.api_key or None,
            base_url=llm_cfg.base_url or None,
            model=llm_cfg.model or None,
            extra_headers=llm_cfg.extra_headers or None,
        )
        status.update(label="Sentiment ready", state="complete")

    # LLM digest
    with st.status("Generating digest...") as status:
        digest_text = ""
        if llm_cfg.api_key and llm_cfg.base_url:
            from openai import OpenAI
            article_texts = "\n\n".join(
                f"[{i+1}] {a.get('title', '')} ({a.get('source', '')})\n{a.get('content', '')[:300]}"
                for i, a in enumerate(articles[:12])
            )

            pm_text = ""
            if pm_markets:
                pm_text = "\n\nPrediction markets:\n" + "\n".join(
                    f"- {p.get('question', '')}: {p.get('odds', '')}" for p in pm_markets
                )
            try:
                client = OpenAI(api_key=llm_cfg.api_key, base_url=llm_cfg.base_url, default_headers=llm_cfg.extra_headers)
                resp = client.chat.completions.create(
                    model=llm_cfg.model,
                    messages=[
                        {"role": "system", "content": (
                            "You are a financial news analyst. Summarize the articles and prediction markets into a structured digest:\n"
                            "1. Key themes (3-5 bullet points)\n"
                            "2. Bullish signals\n3. Bearish signals\n4. Notable quotes/developments\n"
                            "5. Market sentiment from prediction markets (if provided)\n"
                            "Cite sources by number [1], [2], etc."
                        )},
                        {"role": "user", "content": f"Summarize these articles about {query}:\n\n{article_texts}{pm_text}"},
                    ],
                    temperature=0.3, max_tokens=800,
                )
                digest_text = resp.choices[0].message.content.strip()
            except Exception as exc:
                digest_text = f"*LLM digest failed: {exc}*"
        status.update(label="Digest ready", state="complete")

    # Display
    col_score, col_count = st.columns(2)
    col_score.metric("Average Sentiment", f"{sentiment['avg_score']:+.3f}")
    col_count.metric("Articles Found", len(articles))

    tab_digest, tab_articles = st.tabs(["AI Digest", "All Articles"])

    with tab_digest:
        if digest_text:
            st.markdown(digest_text)
        else:
            st.info("Configure an LLM provider for AI digest generation.")

    with tab_articles:
        for h in sentiment["headlines"]:
            score = h["score"]
            emoji = "🟢" if score > 0.1 else ("🔴" if score < -0.1 else "⚪")
            title = h.get("text", "")
            url = h.get("url", "")
            source = h.get("source", "")
            link = f"[{title}]({url})" if url else title
            src = f" — *{source}*" if source else ""
            st.markdown(f"{emoji} {link}{src}  \n`sentiment {score:+.2f}`")

    # Polymarket prediction markets
    with st.expander("Related Prediction Markets (Polymarket)", expanded=False):
        pm_results = search_polymarket(query, max_results=5)
        if pm_results:
            for pm in pm_results:
                q = pm.get("question", "")
                odds = pm.get("odds", "")
                url = pm.get("url", "")
                link = f"[{q}]({url})" if url else q
                st.markdown(f"- {link}  \n  `{odds}`")
        else:
            st.caption("No relevant prediction markets found.")

    st.session_state["news_last"] = {
        "query": query,
        "mode": search_mode,
        "avg_sentiment": sentiment["avg_score"],
        "article_count": len(articles),
        "digest": digest_text,
    }

    try:
        data_store.save_tool_result(
            "news_summary",
            ticker=query if search_mode == "Ticker" else None,
            model_id=llm_cfg.model, provider=llm_cfg.provider,
            inputs={"query": query, "mode": search_mode},
            result={"avg_sentiment": sentiment["avg_score"], "article_count": len(articles)},
            summary=digest_text or f"Sentiment: {sentiment['avg_score']:+.3f} from {len(articles)} articles",
        )
    except Exception:
        pass

# Navigation — outside run_btn so buttons survive reruns
_news = st.session_state.get("news_last")
if _news:
    st.divider()
    if st.button("Discuss in AI Chat →", type="primary", key="news_to_chat", width="stretch"):
        st.session_state["chat_with_analysis"] = {
            "ticker": _news.get("query", ""),
            "tool_type": "news_summary",
            "profile": "news_summary",
            "lookback": "",
            "summary": _news.get("digest", ""),
            "result": {"avg_sentiment": _news.get("avg_sentiment", 0), "article_count": _news.get("article_count", 0)},
            "llm_data": {"avg_sentiment": _news.get("avg_sentiment", 0), "article_count": _news.get("article_count", 0)},
            "inputs": {"query": _news.get("query", ""), "mode": _news.get("mode", "")},
        }
        st.session_state["chat_messages"] = []
        st.switch_page("pages/8_AI_Chat.py")
