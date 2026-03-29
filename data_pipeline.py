"""
Market data retrieval and feature engineering.

Supports stocks, ETFs, crypto, and forex.  Pulls OHLCV data via DefeatBeta
(primary, stocks only) or yfinance (fallback / all asset types).  Detects
DefeatBeta data cutoff and fills the gap to present with yfinance.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import numpy as np

from utils import default_date_range
import defeatbeta_client as db

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Asset type detection
# ---------------------------------------------------------------------------

_CRYPTO_SUFFIXES = ("-USD", "-EUR", "-GBP", "-BTC", "-ETH")
_FOREX_SUFFIXES = ("=X",)
_KNOWN_ETFS = {
    "SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "VEA", "VWO",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE",
    "GLD", "SLV", "TLT", "HYG", "LQD", "BND", "AGG",
    "ARKK", "ARKW", "ARKG", "ARKF",
    "TQQQ", "SQQQ", "UVXY", "VXX",
}


def detect_asset_type(ticker: str) -> str:
    """Classify *ticker* as 'stock', 'etf', 'crypto', or 'forex'."""
    t = ticker.upper()
    if any(t.endswith(s) for s in _CRYPTO_SUFFIXES):
        return "crypto"
    if any(t.endswith(s) for s in _FOREX_SUFFIXES):
        return "forex"
    if t in _KNOWN_ETFS:
        return "etf"
    return "stock"


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

def _fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    df = tk.history(start=start, end=end, auto_adjust=False)
    if df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. "
            "Check the symbol is valid and not delisted."
        )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    if len(keep) < 4:
        raise ValueError(f"Incomplete data for '{ticker}' — got columns {list(df.columns)}")
    df = df[keep].copy()
    df.dropna(inplace=True)
    return df


def _detect_cutoff(df: pd.DataFrame) -> str | None:
    """Return the last date in a DefeatBeta DataFrame as YYYY-MM-DD, or None."""
    if df is None or df.empty:
        return None
    last_date = df.index[-1]
    if hasattr(last_date, "strftime"):
        return last_date.strftime("%Y-%m-%d")
    return str(last_date)[:10]


def _fill_gap(ticker: str, db_df: pd.DataFrame, end: str) -> pd.DataFrame:
    """If DefeatBeta data ends before *end*, append yfinance data for the gap."""
    cutoff = _detect_cutoff(db_df)
    if cutoff is None:
        return db_df

    cutoff_dt = datetime.strptime(cutoff, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    today = datetime.today()
    target_end = min(end_dt, today)

    gap_days = (target_end - cutoff_dt).days
    if gap_days <= 1:
        return db_df

    gap_start = (cutoff_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    gap_end = target_end.strftime("%Y-%m-%d")

    log.info(
        "DefeatBeta data for %s ends %s — filling %d-day gap to %s with yfinance",
        ticker, cutoff, gap_days, gap_end,
    )

    try:
        yf_df = _fetch_yfinance(ticker, gap_start, gap_end)
        if not yf_df.empty:
            yf_df.index = pd.to_datetime(yf_df.index)
            if yf_df.index.tz is not None:
                yf_df.index = yf_df.index.tz_localize(None)
            combined = pd.concat([db_df, yf_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
            return combined
    except Exception as exc:
        log.warning("yfinance gap-fill failed for %s: %s", ticker, exc)

    return db_df


def get_stock_data(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Download daily OHLCV data for *ticker* between *start* and *end*.

    Tries DefeatBeta first for stocks (faster, no rate-limits).  Detects if
    the data has a cutoff before *end* and fills the gap with yfinance.
    Falls back entirely to yfinance for non-stock assets or on failure.
    """
    if start is None or end is None:
        start, end = default_date_range()

    asset_type = detect_asset_type(ticker)

    if asset_type == "stock":
        df = db.get_price_data(ticker, start, end)
        if df is not None and not df.empty:
            log.info("Using DefeatBeta price data for %s (cutoff: %s)", ticker, _detect_cutoff(df))
            df = _fill_gap(ticker, df, end)
            return df

    log.info("Using yfinance for %s (%s)", ticker, asset_type)
    return _fetch_yfinance(ticker, start, end)


def get_data_cutoff_info(ticker: str) -> dict:
    """Return info about data sources and their cutoff dates."""
    result = {"defeatbeta_cutoff": None, "has_realtime": False}

    if detect_asset_type(ticker) == "stock":
        df = db.get_price_data(ticker)
        if df is not None and not df.empty:
            result["defeatbeta_cutoff"] = _detect_cutoff(df)

    quote = db.get_realtime_quote(ticker)
    if quote and quote.get("last_price", 0) > 0:
        result["has_realtime"] = True
        result["realtime_price"] = quote["last_price"]

    return result


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=1).mean()
    avg_loss = loss.rolling(window, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - 100 / (1 + rs)


def create_features(
    df: pd.DataFrame,
    rsi_window: int = 14,
    ma_short: int = 5,
    ma_long: int = 20,
    vol_window: int = 20,
) -> pd.DataFrame:
    """Augment *df* with technical features.

    Rolling windows use ``min_periods=1`` so short lookbacks still produce
    usable rows.
    """
    df = df.copy()

    df["returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df["ma_5"] = df["Close"].rolling(ma_short, min_periods=1).mean()
    df["ma_20"] = df["Close"].rolling(ma_long, min_periods=1).mean()
    df["rsi"] = _compute_rsi(df["Close"], window=min(rsi_window, max(2, len(df) - 1)))
    df["volatility"] = df["returns"].rolling(vol_window, min_periods=2).std()

    if "Volume" in df.columns and df["Volume"].sum() > 0:
        df["volume_sma"] = df["Volume"].rolling(20, min_periods=1).mean()
        df["volume_ratio"] = df["Volume"] / (df["volume_sma"] + 1)
    else:
        df["volume_sma"] = 0.0
        df["volume_ratio"] = 1.0

    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)
    return df
