"""
Backtest — multi-checkpoint AI trading vs buy-and-hold vs S&P 500.

The AI acts as a trader, making decisions at regular intervals across the
timeframe.  At each checkpoint it sees only data available up to that point
and decides LONG / SHORT / CASH with an allocation size.

Signal → Position mapping:
    STRONG_BUY  → LONG  100%
    BUY         → LONG  scaled by confidence
    HOLD        → LONG  100% (buy-and-hold is fine)
    CASH        → CASH  0%
    SELL        → SHORT scaled by confidence
    STRONG_SELL → SHORT 100%
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from data_pipeline import get_stock_data, create_features
from price_model import train_price_model, predict_next_day

log = logging.getLogger(__name__)


def parse_strategy_from_nl(description: str, llm_client, model: str) -> dict:
    """Convert natural language strategy description to structured parameters."""
    if not description:
        return {}

    try:
        resp = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "Extract trading strategy parameters from the description. Return JSON with:\n"
                    "- entry_rules: when to enter positions\n"
                    "- exit_rules: when to exit positions\n"
                    "- indicators: which indicators to focus on\n"
                    "- risk_tolerance: conservative/moderate/aggressive\n"
                    "- position_sizing: how to size positions"
                )},
                {"role": "user", "content": f"Strategy: {description}"}
            ],
            temperature=0.2,
            max_tokens=300,
        )
        content = resp.choices[0].message.content.strip()
        # Try to parse JSON from response
        import json
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return json.loads(content)
    except Exception as exc:
        log.warning("Strategy parsing failed: %s", exc)
        return {"description": description}


_SIGNAL_MAP: dict[str, str] = {
    "STRONG_BUY": "LONG",
    "BUY": "LONG",
    "HOLD": "LONG",
    "CASH": "CASH",
    "SELL": "SHORT",
    "STRONG_SELL": "SHORT",
}

_CHECKPOINT_INTERVALS: dict[str, int] = {
    "1 week": 1,
    "2 weeks": 2,
    "1 month": 7,
    "3 months": 14,
    "6 months": 14,
    "1 year": 30,
    "2 years": 30,
}


def derive_position(signal: str, confidence: int) -> tuple[str, float]:
    """Derive position direction and allocation from signal + confidence."""
    position = _SIGNAL_MAP.get(signal, "CASH")
    conf = max(0, min(100, confidence)) / 100.0

    if signal in ("STRONG_BUY", "STRONG_SELL"):
        allocation = 1.0
    elif signal in ("BUY", "SELL"):
        allocation = conf
    elif signal == "HOLD":
        allocation = 1.0
    else:
        allocation = 0.0
    return position, round(allocation, 2)


@dataclass
class Trade:
    date: str
    signal: str
    position: str
    allocation: float
    confidence: int
    reasoning: str
    price: float


@dataclass
class BacktestResult:
    ticker: str
    timeframe: str
    trades: list[Trade] = field(default_factory=list)
    ai_equity: list[float] = field(default_factory=list)
    bh_equity: list[float] = field(default_factory=list)
    sp_equity: list[float] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    ai_return: float = 0.0
    bh_return: float = 0.0
    sp_return: float = 0.0
    initial_capital: float = 100_000
    num_checkpoints: int = 0


def _fetch_static_context(ticker: str) -> dict:
    """Fetch fundamentals and news once — reused across all checkpoints."""
    import defeatbeta_client as db

    ctx: dict = {}

    fundamentals = db.get_fundamentals(ticker)
    if fundamentals:
        ctx["fundamentals"] = fundamentals

    margins = db.get_margins(ticker)
    if margins:
        ctx["margins"] = margins

    growth = db.get_growth_metrics(ticker)
    if growth:
        ctx["growth"] = growth

    try:
        from news_scraper import get_news
        articles = get_news(ticker)
        if articles:
            ctx["recent_news"] = [a.get("title", "") for a in articles[:8]]
    except Exception:
        pass

    try:
        from sentiment_analysis import analyze_sentiment
        if articles:
            sent = analyze_sentiment(articles)
            ctx["news_sentiment"] = {
                "avg_score": round(sent.get("avg_score", 0), 3),
                "num_articles": len(articles),
            }
    except Exception:
        pass

    return ctx


def _build_checkpoint_snapshot(
    ticker: str, df: pd.DataFrame, idx: int,
    static_context: dict | None = None,
) -> dict:
    """Build a data snapshot using only data available up to index ``idx``."""
    window = df.iloc[: idx + 1]
    latest = window["Close"].iloc[-1]
    first = window["Close"].iloc[0]
    period_return = (latest - first) / first
    high = window["High"].max()
    low = window["Low"].min()

    snapshot: dict = {
        "ticker": ticker,
        "date": str(window.index[-1])[:10],
        "price": {
            "latest_close": round(latest, 2),
            "period_start": round(first, 2),
            "period_return": f"{period_return:+.2%}",
            "period_high": round(high, 2),
            "period_low": round(low, 2),
        },
    }

    try:
        feat = create_features(window)
        if not feat.empty:
            last = feat.iloc[-1]
            snapshot["technicals"] = {
                "rsi": round(float(last.get("rsi", 50)), 2),
                "ma_short": round(float(last.get("ma_5", 0)), 2),
                "ma_long": round(float(last.get("ma_20", 0)), 2),
                "volatility": round(float(last.get("volatility", 0)), 4),
            }
            try:
                model, _, acc = train_price_model(feat)
                direction, up_prob = predict_next_day(model, feat)
                snapshot["ml_prediction"] = {
                    "direction": "UP" if direction == 1 else "DOWN",
                    "confidence": f"{up_prob:.0%}",
                    "model_accuracy": f"{acc:.0%}",
                }
            except Exception:
                snapshot["ml_prediction"] = "NOT AVAILABLE — insufficient data for ML model"
    except Exception:
        pass

    # Recent price action (last 5 bars)
    recent = window.tail(5)
    snapshot["recent_prices"] = [
        {"date": str(d)[:10], "close": round(float(c), 2)}
        for d, c in zip(recent.index, recent["Close"])
    ]

    # Inject fundamentals, news, sentiment (fetched once, shared across checkpoints)
    if static_context:
        for key in ("fundamentals", "margins", "growth", "recent_news", "news_sentiment"):
            if key in static_context:
                snapshot[key] = static_context[key]

    return snapshot


def _ai_checkpoint_decision(
    snapshot: dict,
    current_position: str,
    current_allocation: float,
    llm_client,
    model: str,
    strategy_context: dict | None = None,
) -> dict:
    """Ask the LLM for a trading decision at one checkpoint."""
    data_text = json.dumps(snapshot, indent=2, default=str)

    pos_desc = f"{current_position} at {current_allocation:.0%}" if current_position != "CASH" else "CASH (no position)"

    strategy_instruction = ""
    if strategy_context:
        if "description" in strategy_context:
            strategy_instruction = f"\n\nSTRATEGY: Follow this strategy: {strategy_context['description']}\n"
        else:
            strategy_instruction = f"\n\nSTRATEGY: {json.dumps(strategy_context, indent=2)}\n"

    resp = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                "You are an active trader managing a portfolio. You are at a "
                "decision checkpoint and must decide your next position.\n\n"
                f"Your CURRENT position: {pos_desc}\n"
                f"{strategy_instruction}\n"
                "Based on the market data snapshot provided, decide what to do:\n\n"
                "Return a JSON object with:\n"
                "- \"signal\": one of:\n"
                "    \"STRONG_BUY\" — very bullish, go all-in long\n"
                "    \"BUY\" — moderately bullish, partial long\n"
                "    \"HOLD\" — stay long / buy-and-hold is fine\n"
                "    \"CASH\" — close positions, stay in cash\n"
                "    \"SELL\" — moderately bearish, partial short\n"
                "    \"STRONG_SELL\" — very bearish, go all-in short\n"
                "- \"confidence\": 0-100\n"
                "- \"reasoning\": 1-2 sentences\n\n"
                "Return ONLY the JSON object."
            )},
            {"role": "user", "content": f"Checkpoint data:\n{data_text[:3000]}"},
        ],
        temperature=0.3,
        max_tokens=200,
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()

    parsed = json.loads(raw)
    signal = parsed.get("signal", "CASH").upper()
    confidence = int(parsed.get("confidence", 50))
    reasoning = parsed.get("reasoning", "")

    position, allocation = derive_position(signal, confidence)

    return {
        "signal": signal,
        "confidence": confidence,
        "reasoning": reasoning,
        "position": position,
        "allocation": allocation,
    }


def _compute_checkpoint_indices(df: pd.DataFrame, interval_days: int) -> list[int]:
    """Return DataFrame integer indices for each checkpoint."""
    if len(df) < 2:
        return [0]
    indices = list(range(0, len(df), max(1, interval_days)))
    if indices[-1] != len(df) - 1:
        indices.append(len(df) - 1)
    return indices


def run_multi_checkpoint_backtest(
    ticker: str,
    start: str,
    end: str,
    timeframe: str,
    llm_client,
    model: str,
    initial_capital: float = 100_000,
    on_checkpoint=None,
    strategy_context: dict | None = None,
) -> BacktestResult:
    """Run a multi-checkpoint backtest where AI trades across the period."""
    result = BacktestResult(
        ticker=ticker,
        timeframe=timeframe,
        initial_capital=initial_capital,
    )

    df = get_stock_data(ticker, start, end)
    if df is None or df.empty or len(df) < 2:
        return result

    # Fetch fundamentals, news, sentiment once — reused at every checkpoint
    static_ctx = _fetch_static_context(ticker)

    interval = _CHECKPOINT_INTERVALS.get(timeframe, 14)
    cp_indices = _compute_checkpoint_indices(df, interval)
    result.num_checkpoints = len(cp_indices)

    daily_returns = df["Close"].pct_change().fillna(0).values
    dates_str = [
        d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
        for d in df.index
    ]
    prices = df["Close"].values

    # Track AI position across the full daily series
    ai_direction = np.zeros(len(df))
    ai_alloc = np.zeros(len(df))

    current_pos = "CASH"
    current_alloc = 0.0
    cp_set = set(cp_indices)

    for i in range(len(df)):
        if i in cp_set:
            try:
                snapshot = _build_checkpoint_snapshot(ticker, df, i, static_ctx)
                decision = _ai_checkpoint_decision(
                    snapshot, current_pos, current_alloc,
                    llm_client, model, strategy_context,
                )
                new_pos = decision["position"]
                new_alloc = decision["allocation"]

                if new_pos != current_pos or new_alloc != current_alloc:
                    result.trades.append(Trade(
                        date=dates_str[i],
                        signal=decision["signal"],
                        position=new_pos,
                        allocation=new_alloc,
                        confidence=decision["confidence"],
                        reasoning=decision["reasoning"],
                        price=round(float(prices[i]), 2),
                    ))

                current_pos = new_pos
                current_alloc = new_alloc

                if on_checkpoint:
                    on_checkpoint(
                        len([j for j in cp_indices if j <= i]),
                        len(cp_indices),
                        decision,
                    )
            except Exception as exc:
                log.warning("Checkpoint %d failed: %s", i, exc)

        if current_pos == "LONG":
            ai_direction[i] = 1.0
        elif current_pos == "SHORT":
            ai_direction[i] = -1.0
        else:
            ai_direction[i] = 0.0
        ai_alloc[i] = current_alloc

    # Calculate equity curves
    ai_daily_ret = ai_alloc * ai_direction * daily_returns
    ai_equity = initial_capital * np.cumprod(1 + ai_daily_ret)
    bh_equity = initial_capital * np.cumprod(1 + daily_returns)

    result.dates = dates_str
    result.ai_equity = ai_equity.tolist()
    result.bh_equity = bh_equity.tolist()
    result.ai_return = round((ai_equity[-1] - initial_capital) / initial_capital, 4)
    result.bh_return = round((bh_equity[-1] - initial_capital) / initial_capital, 4)

    # S&P 500 benchmark
    spy_df = get_stock_data("SPY", start, end)
    if spy_df is not None and not spy_df.empty:
        spy_daily = spy_df["Close"].pct_change().fillna(0).values
        sp_equity = initial_capital * np.cumprod(1 + spy_daily)
        result.sp_equity = sp_equity.tolist()
        result.sp_return = round((sp_equity[-1] - initial_capital) / initial_capital, 4)

    return result


def _compute_sharpe_ratio(
    equity: np.ndarray,
    *,
    trading_days_per_year: int = 252,
    risk_free_daily: float = 0.0,
) -> float:
    """Annualized Sharpe from daily simple returns on the equity curve.

    Uses sample standard deviation (ddof=1). Assumes ~252 trading days per year.
    Risk-free rate defaults to zero (common for strategy comparison).
    """
    if equity is None or len(equity) < 3:
        return 0.0
    eq = np.asarray(equity, dtype=float)
    if np.any(~np.isfinite(eq)) or np.any(eq <= 0):
        return 0.0
    daily_ret = np.diff(eq) / eq[:-1]
    daily_ret = daily_ret[np.isfinite(daily_ret)]
    if len(daily_ret) < 2:
        return 0.0
    excess = daily_ret - risk_free_daily
    std = float(np.std(excess, ddof=1))
    if std < 1e-12:
        return 0.0
    mean_excess = float(np.mean(excess))
    return float(np.sqrt(trading_days_per_year) * mean_excess / std)


def compute_metrics(result: BacktestResult) -> dict:
    """Compute trading performance metrics from a backtest result."""
    eq = np.array(result.ai_equity, dtype=float)
    sharpe_ratio = _compute_sharpe_ratio(eq)

    if len(eq) > 0:
        peak = np.maximum.accumulate(eq)
        drawdown = (eq - peak) / peak
        max_dd = float(drawdown.min())
    else:
        max_dd = 0.0

    trades = result.trades
    if not trades:
        return {
            "num_trades": 0,
            "long_trades": 0,
            "short_trades": 0,
            "cash_moves": 0,
            "sharpe_ratio": round(sharpe_ratio, 4),
            "max_drawdown": round(max_dd, 4),
            "alpha_vs_bh": round(result.ai_return - result.bh_return, 4),
            "alpha_vs_sp": round(result.ai_return - result.sp_return, 4),
        }

    cash_moves = [t for t in trades if t.position == "CASH"]

    return {
        "num_trades": len(trades),
        "long_trades": sum(1 for t in trades if t.position == "LONG"),
        "short_trades": sum(1 for t in trades if t.position == "SHORT"),
        "cash_moves": len(cash_moves),
        "sharpe_ratio": round(sharpe_ratio, 4),
        "max_drawdown": round(max_dd, 4),
        "alpha_vs_bh": round(result.ai_return - result.bh_return, 4),
        "alpha_vs_sp": round(result.ai_return - result.sp_return, 4),
    }
