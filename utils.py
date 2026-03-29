"""
Shared utility helpers, trader-profile configuration, and investor types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def default_date_range(lookback_years: int = 2):
    end = datetime.today()
    start = end - timedelta(days=365 * lookback_years)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def normalize(value: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    return float(np.clip((value - min_val) / (max_val - min_val + 1e-9), 0, 1))


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator


# ---------------------------------------------------------------------------
# Investor types (for Overview Analysis advice)
# ---------------------------------------------------------------------------

@dataclass
class InvestorType:
    key: str
    label: str
    description: str
    prompt_hint: str


INVESTOR_TYPES: dict[str, InvestorType] = {
    "conservative": InvestorType(
        key="conservative",
        label="Conservative Investor",
        description="Capital preservation, dividends, low volatility",
        prompt_hint=(
            "Focus on downside risks, dividend sustainability, debt levels, "
            "defensive positioning, and capital preservation strategies."
        ),
    ),
    "moderate": InvestorType(
        key="moderate",
        label="Moderate Investor",
        description="Balanced growth and stability",
        prompt_hint=(
            "Balance growth opportunities with risk management. Consider "
            "valuations, quality metrics, and reasonable entry points."
        ),
    ),
    "aggressive": InvestorType(
        key="aggressive",
        label="Aggressive Investor",
        description="High growth, accepts higher risk",
        prompt_hint=(
            "Focus on growth catalysts, momentum, market positioning, upside "
            "potential, and emerging trends. Acknowledge but weigh risks less."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Trader profiles (for Backtest Lab)
# ---------------------------------------------------------------------------

@dataclass
class TraderProfile:
    key: str
    label: str
    description: str
    lookback_options: dict[str, int]
    default_lookback: str
    rsi_window: int = 14
    ma_short: int = 5
    ma_long: int = 20
    vol_window: int = 20
    buy_threshold: float = 0.60
    sell_threshold: float = 0.40
    model_weight: float = 0.60
    sentiment_weight: float = 0.40
    show_fundamentals: bool = False
    show_dcf: bool = False
    show_transcripts: bool = False


TRADER_PROFILES: dict[str, TraderProfile] = {
    "day": TraderProfile(
        key="day", label="Day Trader",
        description="Short-term momentum and intraday setups",
        lookback_options={"1 week": 7, "2 weeks": 14, "1 month": 30, "3 months": 90},
        default_lookback="1 month",
        rsi_window=7, ma_short=9, ma_long=21, vol_window=10,
        buy_threshold=0.55, sell_threshold=0.45,
        model_weight=0.70, sentiment_weight=0.30,
    ),
    "swing": TraderProfile(
        key="swing", label="Swing Trader",
        description="Multi-day to multi-week trend captures",
        lookback_options={"1 month": 30, "3 months": 90, "6 months": 180},
        default_lookback="3 months",
        rsi_window=14, ma_short=10, ma_long=50, vol_window=14,
        buy_threshold=0.58, sell_threshold=0.42,
        model_weight=0.60, sentiment_weight=0.40,
    ),
    "position": TraderProfile(
        key="position", label="Position Trader",
        description="Weeks-to-months trend following",
        lookback_options={"3 months": 90, "6 months": 180, "1 year": 365, "2 years": 730},
        default_lookback="1 year",
        rsi_window=14, ma_short=50, ma_long=200, vol_window=20,
        buy_threshold=0.60, sell_threshold=0.40,
        model_weight=0.55, sentiment_weight=0.45,
    ),
    "longterm": TraderProfile(
        key="longterm", label="Long-Term Investor",
        description="Fundamental-driven, multi-year horizon",
        lookback_options={"1 year": 365, "2 years": 730, "5 years": 1825, "10 years": 3650},
        default_lookback="2 years",
        rsi_window=21, ma_short=50, ma_long=200, vol_window=30,
        buy_threshold=0.65, sell_threshold=0.35,
        model_weight=0.40, sentiment_weight=0.60,
        show_fundamentals=True, show_dcf=True, show_transcripts=True,
    ),
}


def dates_for_lookback(lookback_label: str, profile: TraderProfile) -> tuple[str, str]:
    days = profile.lookback_options.get(lookback_label, 365)
    end = datetime.today()
    start = end - timedelta(days=days)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
