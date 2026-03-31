"""
Backtest Lab — multi-checkpoint AI trading vs Buy-and-Hold vs S&P 500.

The AI acts as a trader, making LONG / SHORT / CASH decisions at regular
intervals across the timeframe.  Results include a trade log, equity curve
with entry/exit markers, and performance metrics.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components
from datetime import datetime, timedelta

from sidebar_config import render_llm_sidebar, LLMConfig
from backtest import (
    run_multi_checkpoint_backtest,
    compute_metrics,
    BacktestResult,
    _CHECKPOINT_INTERVALS,
)
import data_store

st.markdown("<style>@media(max-width:768px){[data-testid='column']{width:100%!important;flex:100%!important}}</style>", unsafe_allow_html=True)

_TIMEFRAMES: dict[str, int] = {
    "1 week": 7,
    "2 weeks": 14,
    "1 month": 30,
    "3 months": 90,
    "6 months": 180,
    "1 year": 365,
    "2 years": 730,
}

with st.sidebar:
    st.header("Backtest Settings")
    ticker = st.text_input("Ticker", value="AAPL", key="bt_ticker").upper().strip()
    lookback = st.selectbox("Timeframe", list(_TIMEFRAMES.keys()), index=3, key="bt_lookback")

    interval_days = _CHECKPOINT_INTERVALS.get(lookback, 14)
    lb_days = _TIMEFRAMES[lookback]
    est_checkpoints = max(1, lb_days // max(1, interval_days))
    st.caption(f"~{est_checkpoints} AI checkpoints (every ~{interval_days} trading days)")

    initial_capital = st.number_input("Starting capital ($)", min_value=1000, value=100_000, step=10000, key="bt_cap")

    st.divider()
    st.subheader("Strategy")
    strategy_mode = st.radio("Mode", ["AI Discretionary", "Template", "Custom"], key="bt_strategy_mode")

    strategy_desc = ""
    if strategy_mode == "Template":
        template = st.selectbox("Select template", [
            "Momentum (RSI-based)",
            "Mean Reversion",
            "Trend Following (MA crossover)",
            "Breakout Trading",
        ], key="bt_template")

        if template == "Momentum (RSI-based)":
            strategy_desc = "Buy when RSI < 30 (oversold), sell when RSI > 70 (overbought). Use strong signals for extreme values."
        elif template == "Mean Reversion":
            strategy_desc = "Buy when price drops significantly below 20-day MA, sell when it rises significantly above. Target quick reversals."
        elif template == "Trend Following (MA crossover)":
            strategy_desc = "Buy when 5-day MA crosses above 20-day MA, sell when it crosses below. Follow the trend."
        elif template == "Breakout Trading":
            strategy_desc = "Buy on strong upward momentum with high volume, sell on weakness. Ride breakouts."

        st.caption(strategy_desc)
    elif strategy_mode == "Custom":
        strategy_desc = st.text_area(
            "Describe your strategy",
            placeholder="e.g., Buy when RSI < 30 and price > 50-day MA, sell when RSI > 70 or price drops 5%",
            key="bt_custom_strategy",
            height=100
        )

    st.session_state["bt_strategy"] = strategy_desc if strategy_mode != "AI Discretionary" else ""

    st.divider()
    llm_cfg: LLMConfig = render_llm_sidebar("bt")
    st.divider()
    run_btn = st.button("Run Backtest", type="primary", use_container_width=True, key="bt_run")

st.title("Backtest Lab")
st.markdown(
    "The AI trades actively across the timeframe, deciding at each checkpoint "
    "whether to go **LONG**, **SHORT**, or sit in **CASH** with a position size. "
    "Compare its trading performance against **buy-and-hold** and the **S&P 500**."
)
st.warning(
    "**Token usage warning:** Each checkpoint costs one LLM call. "
    f"This backtest will make **~{est_checkpoints} API calls** to your selected model. "
    "Use a fast/cheap model (e.g. gpt-4o-mini, qwen-turbo) to keep costs low.",
    icon="⚠️",
)

if run_btn:
    if not llm_cfg.api_key or not llm_cfg.base_url:
        st.error("An LLM provider is required. Configure one in the sidebar.")
        st.stop()

    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=lb_days)).strftime("%Y-%m-%d")

    from openai import OpenAI
    client = OpenAI(
        api_key=llm_cfg.api_key, base_url=llm_cfg.base_url,
        default_headers=llm_cfg.extra_headers,
    )

    progress_bar = st.progress(0, text="Fetching fundamentals, news & sentiment...")

    strategy_ctx = None
    if st.session_state.get("bt_strategy"):
        from backtest import parse_strategy_from_nl
        with st.spinner("Parsing strategy..."):
            strategy_ctx = parse_strategy_from_nl(
                st.session_state["bt_strategy"],
                client,
                llm_cfg.model
            )

    def _on_checkpoint(done: int, total: int, decision: dict):
        pct = done / total if total else 1.0
        sig = decision.get("signal", "?")
        pos = decision.get("position", "?")
        alloc = decision.get("allocation", 0)
        progress_bar.progress(pct, text=f"Checkpoint {done}/{total} — {sig} → {pos} {alloc:.0%}")

    try:
        result: BacktestResult = run_multi_checkpoint_backtest(
            ticker, start_date, end_date,
            timeframe=lookback,
            llm_client=client,
            model=llm_cfg.model,
            initial_capital=initial_capital,
            on_checkpoint=_on_checkpoint,
            strategy_context=strategy_ctx,
        )
    except Exception as exc:
        st.error(f"Backtest failed: {exc}")
        st.stop()

    progress_bar.progress(1.0, text="Backtest complete!")
    metrics = compute_metrics(result)

    # Store for persistence across reruns
    st.session_state["bt_result"] = result
    st.session_state["bt_metrics"] = metrics
    st.session_state["bt_last"] = {
        "ticker": ticker,
        "lookback": lookback,
        "ai_return": result.ai_return,
        "bh_return": result.bh_return,
        "sp500_return": result.sp_return,
        "alpha_vs_bh": metrics.get("alpha_vs_bh", 0),
        "alpha_vs_sp": metrics.get("alpha_vs_sp", 0),
        "num_trades": metrics.get("num_trades", 0),
        "sharpe_ratio": metrics.get("sharpe_ratio", 0),
        "signal": f"{metrics.get('num_trades', 0)} trades",
        "position": "multi",
        "confidence": 0,
        "reasoning": f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}, Alpha vs B&H: {metrics.get('alpha_vs_bh', 0):+.2%}",
        "allocation": 0,
    }

# ── Display results (from session state, survives reruns) ────────────────
_result: BacktestResult | None = st.session_state.get("bt_result")
_metrics: dict | None = st.session_state.get("bt_metrics")

if _result and _result.dates:
    result = _result
    metrics = _metrics or {}

    # Metrics cards
    st.subheader("Performance Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AI Return", f"{result.ai_return:+.2%}")
    c2.metric("Buy & Hold", f"{result.bh_return:+.2%}")
    c3.metric("S&P 500", f"{result.sp_return:+.2%}")
    c4.metric("Sharpe (ann.)", f"{metrics.get('sharpe_ratio', 0):.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Alpha vs B&H", f"{metrics.get('alpha_vs_bh', 0):+.2%}")
    c6.metric("Alpha vs S&P", f"{metrics.get('alpha_vs_sp', 0):+.2%}")
    c7.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
    c8.metric("Total Trades", metrics.get("num_trades", 0))

    # Interactive TradingView chart
    with st.expander(f"Interactive Chart: {result.ticker}", expanded=False):
        tv_html = f"""
        <div id="tv_bt" style="height:400px;">
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget({{
            "autosize": true, "symbol": "{result.ticker}", "interval": "D",
            "timezone": "Etc/UTC", "theme": "dark", "style": "1",
            "locale": "en", "enable_publishing": false,
            "allow_symbol_change": false,
            "studies": ["RSI@tv-basicstudies"],
            "container_id": "tv_bt",
            "hide_side_toolbar": true, "withdateranges": true
          }});
          </script>
        </div>
        """
        components.html(tv_html, height=420)

    # Equity curve with trade markers
    st.subheader("Equity Curve")
    dates = pd.to_datetime(result.dates)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=result.bh_equity,
        name=f"Buy & Hold {result.ticker}",
        line=dict(color="orange", width=2),
    ))
    if result.sp_equity:
        fig.add_trace(go.Scatter(
            x=dates, y=result.sp_equity,
            name="S&P 500",
            line=dict(color="grey", width=2, dash="dash"),
        ))
    fig.add_trace(go.Scatter(
        x=dates, y=result.ai_equity,
        name="AI Trader",
        line=dict(color="royalblue", width=3),
    ))

    # Trade markers on AI equity curve
    trade_dates = []
    trade_equities = []
    trade_colors = []
    trade_symbols_list = []
    trade_texts = []
    date_to_idx = {d: i for i, d in enumerate(result.dates)}
    for t in result.trades:
        idx = date_to_idx.get(t.date)
        if idx is not None and idx < len(result.ai_equity):
            trade_dates.append(pd.to_datetime(t.date))
            trade_equities.append(result.ai_equity[idx])
            if t.position == "LONG":
                trade_colors.append("lime")
                trade_symbols_list.append("triangle-up")
            elif t.position == "SHORT":
                trade_colors.append("red")
                trade_symbols_list.append("triangle-down")
            else:
                trade_colors.append("white")
                trade_symbols_list.append("circle")
            trade_texts.append(f"{t.signal} {t.allocation:.0%}<br>${t.price}")

    if trade_dates:
        fig.add_trace(go.Scatter(
            x=trade_dates,
            y=trade_equities,
            mode="markers",
            name="Trades",
            marker=dict(
                size=12,
                color=trade_colors,
                symbol=trade_symbols_list,
                line=dict(width=1, color="white"),
            ),
            text=trade_texts,
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        height=450,
        yaxis_title="Portfolio Value ($)",
        margin=dict(l=40, r=20, t=20, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Trade log
    st.subheader("Trade Log")
    if result.trades:
        trade_rows = []
        for t in result.trades:
            trade_rows.append({
                "Date": t.date,
                "Signal": t.signal,
                "Position": t.position,
                "Allocation": f"{t.allocation:.0%}",
                "Confidence": f"{t.confidence}%",
                "Price": f"${t.price:,.2f}",
                "Reasoning": t.reasoning,
            })
        st.dataframe(pd.DataFrame(trade_rows), width="stretch", hide_index=True)
    else:
        st.info("No position changes during this period (AI held initial position throughout).")

    # Detailed metrics
    with st.expander("Detailed Metrics", expanded=False):
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Long Trades", metrics.get("long_trades", 0))
        mc2.metric("Short Trades", metrics.get("short_trades", 0))
        mc3.metric("Cash Moves", metrics.get("cash_moves", 0))
        st.caption(
            "Sharpe uses daily returns on the AI equity curve, annualized (~252 trading days), "
            "risk-free rate 0."
        )

    # Save tool result
    try:
        data_store.save_tool_result(
            "backtest", ticker=result.ticker,
            model_id=llm_cfg.model, provider=llm_cfg.provider,
            inputs={"lookback": result.timeframe, "capital": result.initial_capital},
            result={
                "ai_return": result.ai_return,
                "bh_return": result.bh_return,
                "sp_return": result.sp_return,
                "num_trades": metrics.get("num_trades", 0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                "max_drawdown": metrics.get("max_drawdown", 0),
                "alpha_vs_bh": metrics.get("alpha_vs_bh", 0),
                "alpha_vs_sp": metrics.get("alpha_vs_sp", 0),
            },
            summary=(
                f"AI: {result.ai_return:+.2%} ({metrics.get('num_trades', 0)} trades, "
                f"Sharpe {metrics.get('sharpe_ratio', 0):.2f}), "
                f"B&H: {result.bh_return:+.2%}, S&P: {result.sp_return:+.2%}"
            ),
        )
    except Exception:
        pass

# ── Navigation — outside run_btn ─────────────────────────────────────────
_bt = st.session_state.get("bt_last")
if _bt:
    st.divider()
    if st.button("Discuss in AI Chat →", type="primary", key="bt_to_chat", width="stretch"):
        st.session_state["chat_with_analysis"] = {
            "ticker": _bt.get("ticker", ""),
            "tool_type": "backtest",
            "profile": "",
            "lookback": _bt.get("lookback", ""),
            "summary": (
                f"AI traded {_bt.get('ticker', '')} over {_bt.get('lookback', '')}: "
                f"Return {_bt.get('ai_return', 0):+.2%}, "
                f"{_bt.get('num_trades', 0)} trades, "
                f"Sharpe {_bt.get('sharpe_ratio', 0):.2f}, "
                f"B&H: {_bt.get('bh_return', 0):+.2%}"
            ),
            "result": _bt,
            "llm_data": _bt,
            "inputs": {"lookback": _bt.get("lookback", "")},
        }
        st.session_state["chat_messages"] = []
        st.switch_page("pages/8_AI_Chat.py")

    st.divider()
    if st.button("Get AI Commentary on Results", key="bt_ai"):
        if not llm_cfg.api_key or not llm_cfg.base_url:
            st.warning("Configure an LLM provider in the sidebar.")
        else:
            summary_text = (
                f"Multi-checkpoint backtest for {_bt.get('ticker', '')} "
                f"({_bt.get('lookback', '')}):\n\n"
                f"AI trader made {_bt.get('num_trades', 0)} position changes\n"
                f"Sharpe (annualized): {_bt.get('sharpe_ratio', 0):.2f}\n\n"
                f"Results:\n"
                f"  AI return: {_bt.get('ai_return', 0):+.2%}\n"
                f"  Buy & Hold: {_bt.get('bh_return', 0):+.2%}\n"
                f"  S&P 500: {_bt.get('sp500_return', 0):+.2%}\n"
                f"  Alpha vs B&H: {_bt.get('alpha_vs_bh', 0):+.2%}\n"
                f"  Alpha vs S&P: {_bt.get('alpha_vs_sp', 0):+.2%}\n"
            )
            from openai import OpenAI
            ai_client = OpenAI(
                api_key=llm_cfg.api_key, base_url=llm_cfg.base_url,
                default_headers=llm_cfg.extra_headers,
            )
            with st.spinner("Generating commentary..."):
                try:
                    resp = ai_client.chat.completions.create(
                        model=llm_cfg.model,
                        messages=[
                            {"role": "system", "content": (
                                "Analyze these multi-checkpoint backtest results. "
                                "The AI made multiple trading decisions across the period. "
                                "Was its timing good? Did it correctly identify trends? "
                                "Discuss the Sharpe ratio, alpha, drawdown, and what could improve."
                            )},
                            {"role": "user", "content": summary_text},
                        ],
                        temperature=0.3, max_tokens=600,
                    )
                    st.session_state["bt_commentary"] = resp.choices[0].message.content.strip()
                except Exception as exc:
                    st.warning(f"AI commentary failed: {exc}")

    if "bt_commentary" in st.session_state:
        st.markdown("**AI Commentary:**")
        st.markdown(st.session_state["bt_commentary"])
