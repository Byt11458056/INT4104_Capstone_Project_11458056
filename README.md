# AI Financial Analyst

A **Streamlit** application that combines market data (DefeatBeta API and yfinance), news and web research, lightweight ML (Random Forest price direction), retrieval-augmented prompts (SQLite FTS5), and **OpenAI-compatible LLMs** (OpenRouter, Alibaba Qwen, or a custom base URL) for summaries, screening, reports, backtests, and a tool-calling chat assistant.

**Disclaimer:** This project is for education and research. It does not provide financial advice.

## Setup (Docker only)

**Prerequisites:** [Docker](https://docs.docker.com/get-docker/) and Docker Compose.

1. Clone the repository and open a terminal in the project root.

2. Create a `.env` file in the project root. Compose passes these into the container (see `docker-compose.yml`). Include at least one LLM key, for example:

   ```env
   OPENROUTER_API_KEY=your-key
   # DASHSCOPE_API_KEY=your-key
   # GNEWS_API_KEY=your-key
   ```

   Optional keys: `DASHSCOPE_API_KEY`, `GNEWS_API_KEY`. You can also use **Custom Provider** in the app sidebar with your own endpoint. DefeatBeta-backed features rely on the `defeatbeta-api` package; check their documentation for any account or key requirements.

3. Build and run:

   ```bash
   docker compose up --build
   ```

4. Open **http://localhost:8501**. The app redirects to **Market Overview** on first load; use the sidebar for other tools.

A named Docker volume persists `data/` inside the container (SQLite database).

## Project layout (overview)

| Path | Role |
|------|------|
| `app.py` | Home / tools hub; session LLM defaults; redirects to Market Overview |
| `pages/*.py` | Multi-page Streamlit UI (overview, market, news, screener, reports, backtest, chat, history) |
| `data_pipeline.py` | OHLCV, features (RSI, MAs, volatility); DefeatBeta + yfinance |
| `defeatbeta_client.py` | Wrapper around `defeatbeta-api` (prices, fundamentals, news, transcripts, etc.) |
| `data_store.py` | SQLite persistence (analyses, articles, chats, tool results, portfolios, transcripts) |
| `rag_engine.py` | FTS5 chunk storage and search for RAG |
| `llm_summary.py` | Provider presets and LLM prompts for summaries |
| `agents.py` | OpenAI-style tool definitions and agent loop for chat |
| `backtest.py` | Historical checkpoint backtesting vs benchmarks |
| `signal_engine.py` | Composite BUY/HOLD/SELL from model + sentiment |
| `news_scraper.py` | Multi-source news aggregation |
| `config.py` | `get_secret()` — `st.secrets` then environment |

Data files live under `data/` (created automatically; `store.db`).
