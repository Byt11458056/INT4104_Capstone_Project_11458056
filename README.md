# AI Financial Analyst

**AI Financial Analyst** is a multipage **Streamlit** app for research-style workflows: live-style market views, ticker deep-dives, news with sentiment, thematic screening, filings and earnings context, strategy backtests, and an **LLM agent** that can call data and web tools. Market data flows through **DefeatBeta** (`defeatbeta-api`) where available, with **yfinance** as fallback or gap-fill. News uses several sources (including optional **GNews**). A small **Random Forest** model and **composite signals** illustrate a quant-plus-narrative pipeline; **SQLite** (with **FTS5** for RAG chunks) stores runs, chats, and tool results.

**LLMs** are **OpenAI-compatible**: built-in presets for **OpenRouter** and **Alibaba Qwen (DashScope)**, or a **Custom Provider** in the sidebar. API keys come from the environment (Docker `.env`) or Streamlit secrets.

On first open, the app switches to **Market Overview**. The **Tools Hub** (tool grid and recent activity) lives on the home route **`http://localhost:8501/`** — open that URL when you want it; the main script is omitted from the sidebar list. Page filenames set sidebar order (e.g. `1_Market_Overview.py` before `2_Overview_Analysis.py`).

**Disclaimer:** For education and research only. This is not financial advice.

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

4. Open **http://localhost:8501**. The first run switches to **Market Overview**. For the **Tools Hub** (launchers and recent activity), go to **http://localhost:8501/** again or bookmark the home URL.

A named Docker volume persists `data/` inside the container (SQLite at `data/store.db`).

## App pages

Streamlit lists these under **pages** in the sidebar (filename order). The **Tools Hub** is only on the home route, not in that list.

| Page | File | Purpose |
|------|------|---------|
| Market Overview | `pages/1_Market_Overview.py` | Watchlists, charts, macro-style views |
| Overview Analysis | `pages/2_Overview_Analysis.py` | Single-ticker performance, fundamentals, AI brief |
| News Summarizer | `pages/3_News_Summarizer.py` | Headlines, sentiment, political / market signals |
| Investment Screener | `pages/4_Investment_Screener.py` | Theme- and trend-driven ideas |
| Financial Reports | `pages/5_Financial_Reports.py` | Filings, earnings, statement-style Q&A |
| Backtest Lab | `pages/7_Backtest_Lab.py` | Historical checkpoints vs benchmarks |
| AI Chat | `pages/8_AI_Chat.py` | Tool-calling assistant, saved conversations |
| History | `pages/9_History.py` | Past tool outputs and chats |

## Project layout (overview)

| Path | Role |
|------|------|
| `Dockerfile`, `docker-compose.yml` | Python 3.11 image; port 8501; `app_data` volume for `data/` |
| `sidebar_config.py` | Shared LLM sidebar; hides main-script nav label |
| `pages/*.py` | Multipage UI (see table above) |
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
