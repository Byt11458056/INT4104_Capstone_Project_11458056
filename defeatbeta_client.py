"""
Wrapper around the defeatbeta-api package.

Provides high-level helpers for price data, fundamentals, news articles,
earnings-call transcripts, DCF valuation, SEC filings, financial statements,
margins, growth metrics, and company profile.  Every public function catches
import / runtime errors so the rest of the app can gracefully fall back to
alternative data sources when the package is unavailable.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

_AVAILABLE: bool | None = None


def _is_available() -> bool:
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            from defeatbeta_api.data.ticker import Ticker  # noqa: F401
            _AVAILABLE = True
        except Exception:
            _AVAILABLE = False
    return _AVAILABLE


def _ticker(symbol: str):
    from defeatbeta_api.data.ticker import Ticker
    return Ticker(symbol)


def _df_to_records(df: pd.DataFrame | None, limit: int = 8) -> list[dict]:
    """Convert a DataFrame to a list of row dicts (latest first, capped)."""
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return []
    try:
        if "report_date" in df.columns:
            # Validate and clean report_date to prevent timestamp overflow errors
            df = _validate_report_dates(df)
            df = df.sort_values("report_date", ascending=False)
        return df.head(limit).to_dict("records")
    except Exception:
        return []


def _validate_report_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean report_date column to prevent timestamp overflow errors.
    
    Removes or fixes dates that are outside reasonable bounds to avoid
    DuckDB timestamp overflow errors.
    """
    if "report_date" not in df.columns:
        return df
    
    df = df.copy()
    try:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df["report_date"]):
            df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
        
        # Define reasonable date bounds (1900-2100)
        min_date = pd.Timestamp("1900-01-01")
        max_date = pd.Timestamp("2100-12-31")
        
        # Filter out dates outside reasonable bounds
        valid_mask = (
            (df["report_date"] >= min_date) & 
            (df["report_date"] <= max_date) & 
            df["report_date"].notna()
        )
        
        if not valid_mask.all():
            # Keep only valid dates
            df = df[valid_mask].copy()
            
        if df.empty:
            # If all dates were invalid, remove the report_date column
            df = df.drop(columns=["report_date"])
            
    except Exception:
        # If date validation fails completely, drop the report_date column
        df = df.drop(columns=["report_date"], errors="ignore")
    
    return df


def _extract_scalar(result: Any, value_key: str | None = None) -> Any:
    """Extract the latest scalar value from a DataFrame result.

    DefeatBeta methods like ttm_pe(), roe(), etc. return DataFrames.
    The last row contains the latest data. If *value_key* is given,
    extract that specific column; otherwise return the full last-row dict.
    """
    if result is None:
        return None
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return None
        last = result.iloc[-1]
        if value_key and value_key in last.index:
            v = last[value_key]
            return float(v) if isinstance(v, (Decimal, float, int)) else v
        d = last.to_dict()
        for k in ("symbol", "report_date", "eps_report_date"):
            d.pop(k, None)
        if len(d) == 1:
            return float(list(d.values())[0]) if isinstance(list(d.values())[0], (Decimal, float, int)) else list(d.values())[0]
        return d
    if isinstance(result, (int, float, Decimal)):
        return float(result)
    return result


def _statement_to_records(statement_obj) -> list[dict]:
    """Convert a DefeatBeta Statement object to a list of row dicts.

    Statement.df() returns the underlying DataFrame directly.
    Falls back to yfinance if DefeatBeta returns nothing.
    """
    try:
        df = statement_obj.df()
        if df is None or df.empty:
            return []
        return df.to_dict("records")
    except Exception as exc:
        log.debug("Statement conversion failed: %s", exc)
        return []


def _yf_financial_statement(symbol: str, stmt_type: str, quarterly: bool) -> list[dict]:
    """Fallback: fetch financial statements from yfinance."""
    try:
        import yfinance as yf
        tk = yf.Ticker(symbol)
        if stmt_type == "income":
            df = tk.quarterly_financials if quarterly else tk.financials
        elif stmt_type == "balance":
            df = tk.quarterly_balance_sheet if quarterly else tk.balance_sheet
        elif stmt_type == "cashflow":
            df = tk.quarterly_cashflow if quarterly else tk.cashflow
        else:
            return []
        if df is None or df.empty:
            return []
        df = df.T.reset_index()
        df.rename(columns={"index": "Date"}, inplace=True)
        return df.head(8).to_dict("records")
    except Exception as exc:
        log.debug("yfinance %s fallback failed for %s: %s", stmt_type, symbol, exc)
        return []


# ── Price data ───────────────────────────────────────────────────────────────

def get_price_data(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame | None:
    """Return daily OHLCV from DefeatBeta, or *None* on failure."""
    if not _is_available():
        return None
    try:
        t = _ticker(symbol)
        df = t.price()
        if df is None or df.empty:
            return None

        df["report_date"] = pd.to_datetime(df["report_date"])
        df = df.set_index("report_date").sort_index()

        rename = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        df = df.rename(columns=rename)

        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep]

        if start:
            df = df.loc[start:]
        if end:
            df = df.loc[:end]

        return df if not df.empty else None
    except Exception as exc:
        log.warning("DefeatBeta price fetch failed for %s: %s", symbol, exc)
        return None


# ── Real-time quote (yfinance supplement) ────────────────────────────────────

def get_realtime_quote(symbol: str) -> dict[str, Any] | None:
    """Get current price and day change from yfinance for real-time data."""
    try:
        import yfinance as yf
        tk = yf.Ticker(symbol)
        info = tk.fast_info
        return {
            "last_price": float(getattr(info, "last_price", 0) or 0),
            "previous_close": float(getattr(info, "previous_close", 0) or 0),
            "open": float(getattr(info, "open", 0) or 0),
            "day_high": float(getattr(info, "day_high", 0) or 0),
            "day_low": float(getattr(info, "day_low", 0) or 0),
            "market_cap": float(getattr(info, "market_cap", 0) or 0),
        }
    except Exception as exc:
        log.debug("yfinance real-time quote failed for %s: %s", symbol, exc)
        return None


# ── Company profile ──────────────────────────────────────────────────────────

def get_stock_profile(symbol: str) -> dict[str, Any] | None:
    if not _is_available():
        return None
    try:
        t = _ticker(symbol)
        fn = getattr(t, "profile", None)
        if fn is None:
            return None
        result = fn()
        if result is None:
            return None
        if isinstance(result, pd.DataFrame):
            if result.empty:
                return None
            return result.iloc[0].to_dict()
        if isinstance(result, dict):
            return result
        return None
    except Exception as exc:
        log.warning("DefeatBeta profile failed for %s: %s", symbol, exc)
        return None


# ── Fundamental ratios ───────────────────────────────────────────────────────

_FUNDAMENTAL_METHODS = [
    ("pe_ttm",     "ttm_pe",     "ttm_pe"),
    ("pb",         "pb",         "pb"),
    ("ps",         "ps",         "ps"),
    ("roe",        "roe",        "roe"),
    ("roa",        "roa",        "roa"),
    ("roic",       "roic",       "roic"),
    ("wacc",       "wacc",       "wacc"),
    ("peg",        "peg",        "peg"),
    ("market_cap", "market_cap", "market_capitalization"),
]


def get_fundamentals(symbol: str) -> dict[str, Any] | None:
    """Return a dict of key fundamental metrics as scalar values."""
    if not _is_available():
        return None
    try:
        t = _ticker(symbol)
        result: dict[str, Any] = {}

        for display_name, method_name, value_key in _FUNDAMENTAL_METHODS:
            try:
                fn = getattr(t, method_name, None)
                if fn is None:
                    continue
                val = fn()
                extracted = _extract_scalar(val, value_key)
                if extracted is not None:
                    result[display_name] = extracted
            except Exception:
                continue

        return result if result else None
    except Exception as exc:
        log.warning("DefeatBeta fundamentals failed for %s: %s", symbol, exc)
        return None


# ── News articles ────────────────────────────────────────────────────────────

def get_news_articles(symbol: str, max_articles: int = 20) -> list[dict]:
    if not _is_available():
        return []
    try:
        t = _ticker(symbol)
        news_obj = t.news()
        news_df = news_obj.get_news_list()

        if news_df is None or news_df.empty:
            return []

        articles: list[dict] = []
        for _, row in news_df.head(max_articles).iterrows():
            uuid = row.get("uuid", "")
            link = row.get("link", "")
            title = row.get("title", "")
            source = row.get("source", "")
            published = str(row.get("publish_date", row.get("report_date", "")))

            paragraphs = row.get("news", [])
            content_parts: list[str] = []
            if isinstance(paragraphs, list):
                for p in paragraphs:
                    text = p.get("paragraph", "") if isinstance(p, dict) else str(p)
                    if text:
                        content_parts.append(text)

            articles.append({
                "title": title or "(untitled)",
                "url": link,
                "source": source or "Yahoo Finance",
                "published": published,
                "content": "\n\n".join(content_parts),
                "uuid": uuid,
                "content_type": "defeatbeta",
            })

        return articles
    except Exception as exc:
        log.warning("DefeatBeta news failed for %s: %s", symbol, exc)
        return []


# ── Earnings-call transcripts ───────────────────────────────────────────────

def get_earnings_transcripts(symbol: str, limit: int = 4) -> list[dict]:
    if not _is_available():
        return []
    try:
        t = _ticker(symbol)
        tc = t.earning_call_transcripts()
        listing = tc.get_transcripts_list()
        if listing is None or listing.empty:
            return []

        # Validate report dates to prevent timestamp overflow errors
        listing = _validate_report_dates(listing)
        listing = listing.sort_values("report_date", ascending=False).head(limit)
        results: list[dict] = []
        for _, row in listing.iterrows():
            fy = row.get("fiscal_year")
            fq = row.get("fiscal_quarter")
            text_parts: list[str] = []

            try:
                detail_df = tc.get_transcript(int(fy), int(fq))
                if detail_df is not None and not detail_df.empty:
                    for _, p in detail_df.iterrows():
                        speaker = p.get("speaker", "")
                        content = p.get("content", p.get("paragraph", ""))
                        if content:
                            prefix = f"**{speaker}:** " if speaker else ""
                            text_parts.append(f"{prefix}{content}")
            except Exception:
                paragraphs = row.get("transcripts", [])
                if isinstance(paragraphs, list):
                    for p in paragraphs:
                        speaker = p.get("speaker", "") if isinstance(p, dict) else ""
                        para = p.get("paragraph", "") if isinstance(p, dict) else str(p)
                        if para:
                            prefix = f"**{speaker}:** " if speaker else ""
                            text_parts.append(f"{prefix}{para}")

            results.append({
                "fiscal_year": fy,
                "fiscal_quarter": fq,
                "report_date": str(row.get("report_date", "")),
                "text": "\n\n".join(text_parts),
            })
        return results
    except Exception as exc:
        log.warning("DefeatBeta transcripts failed for %s: %s", symbol, exc)
        return []


# ── DCF valuation ────────────────────────────────────────────────────────────

def get_dcf_valuation(symbol: str) -> dict[str, Any] | None:
    if not _is_available():
        return None
    try:
        t = _ticker(symbol)
        dcf = t.dcf()
        if dcf is None:
            return None

        summary = {}
        for attr in ["fair_price", "enterprise_value", "wacc", "recommendation"]:
            val = getattr(dcf, attr, None)
            if val is not None:
                summary[attr] = float(val) if isinstance(val, (Decimal, int, float)) else val

        if hasattr(dcf, "summary"):
            try:
                s = dcf.summary()
                if isinstance(s, dict):
                    summary.update(s)
            except Exception:
                pass

        return summary if summary else None
    except Exception as exc:
        log.warning("DefeatBeta DCF failed for %s: %s", symbol, exc)
        return None


# ── SEC filings ──────────────────────────────────────────────────────────────

def get_sec_filings(symbol: str, limit: int = 10) -> list[dict]:
    if not _is_available():
        return []
    try:
        t = _ticker(symbol)
        fn = getattr(t, "sec_filing", None)
        if fn is None:
            return []
        result = fn()
        if result is None:
            return []
        if isinstance(result, pd.DataFrame):
            if result.empty:
                return []
            if "report_date" in result.columns:
                # Validate report dates to prevent timestamp overflow errors
                result = _validate_report_dates(result)
                result = result.sort_values("report_date", ascending=False)
            return result.head(limit).to_dict("records")
        if isinstance(result, list):
            return result[:limit]
        return []
    except Exception as exc:
        log.warning("DefeatBeta SEC filings failed for %s: %s", symbol, exc)
        return []


# ── Financial statements ─────────────────────────────────────────────────────

def get_income_statement(symbol: str, quarterly: bool = True, limit: int = 8) -> list[dict]:
    """Return income statement rows. Tries DefeatBeta first, then yfinance."""
    if _is_available():
        try:
            t = _ticker(symbol)
            method_name = "quarterly_income_statement" if quarterly else "annual_income_statement"
            fn = getattr(t, method_name, None)
            if fn is not None:
                stmt = fn()
                records = _statement_to_records(stmt)
                if records:
                    return records
        except Exception as exc:
            log.warning("DefeatBeta income statement failed for %s: %s", symbol, exc)
    return _yf_financial_statement(symbol, "income", quarterly)


def get_balance_sheet(symbol: str, quarterly: bool = True, limit: int = 8) -> list[dict]:
    """Return balance sheet rows. Tries DefeatBeta first, then yfinance."""
    if _is_available():
        try:
            t = _ticker(symbol)
            method_name = "quarterly_balance_sheet" if quarterly else "annual_balance_sheet"
            fn = getattr(t, method_name, None)
            if fn is not None:
                stmt = fn()
                records = _statement_to_records(stmt)
                if records:
                    return records
        except Exception as exc:
            log.warning("DefeatBeta balance sheet failed for %s: %s", symbol, exc)
    return _yf_financial_statement(symbol, "balance", quarterly)


def get_cash_flow(symbol: str, quarterly: bool = True, limit: int = 8) -> list[dict]:
    """Return cash flow rows. Tries DefeatBeta first, then yfinance."""
    if _is_available():
        try:
            t = _ticker(symbol)
            method_name = "quarterly_cash_flow" if quarterly else "annual_cash_flow"
            fn = getattr(t, method_name, None)
            if fn is not None:
                stmt = fn()
                records = _statement_to_records(stmt)
                if records:
                    return records
        except Exception as exc:
            log.warning("DefeatBeta cash flow failed for %s: %s", symbol, exc)
    return _yf_financial_statement(symbol, "cashflow", quarterly)


# ── Margins ──────────────────────────────────────────────────────────────────

_MARGIN_METHODS = [
    ("gross_margin",     "quarterly_gross_margin",     "gross_margin"),
    ("operating_margin", "quarterly_operating_margin", "operating_margin"),
    ("net_margin",       "quarterly_net_margin",       "net_margin"),
    ("ebitda_margin",    "quarterly_ebitda_margin",    "ebitda_margin"),
    ("fcf_margin",       "quarterly_fcf_margin",       "fcf_margin"),
]


def get_margins(symbol: str, quarterly: bool = True) -> dict[str, Any]:
    """Return margin metrics as scalar floats."""
    if not _is_available():
        return {}
    try:
        t = _ticker(symbol)
        prefix = "quarterly" if quarterly else "annual"
        result: dict[str, Any] = {}

        for display_name, method_name_q, value_key in _MARGIN_METHODS:
            mname = method_name_q if quarterly else method_name_q.replace("quarterly_", "annual_")
            try:
                fn = getattr(t, mname, None)
                if fn is None:
                    continue
                val = fn()
                extracted = _extract_scalar(val, value_key)
                if extracted is not None and isinstance(extracted, (int, float)):
                    result[display_name] = extracted
            except Exception:
                continue

        return result
    except Exception as exc:
        log.warning("DefeatBeta margins failed for %s: %s", symbol, exc)
        return {}


# ── Growth metrics ───────────────────────────────────────────────────────────

_GROWTH_METHODS = [
    ("revenue_yoy",          "quarterly_revenue_yoy_growth",          "yoy_growth"),
    ("operating_income_yoy", "quarterly_operating_income_yoy_growth", "yoy_growth"),
    ("ebitda_yoy",           "quarterly_ebitda_yoy_growth",           "yoy_growth"),
    ("net_income_yoy",       "quarterly_net_income_yoy_growth",       "yoy_growth"),
    ("fcf_yoy",              "quarterly_fcf_yoy_growth",              "yoy_growth"),
]


def get_growth_metrics(symbol: str, quarterly: bool = True) -> dict[str, Any]:
    """Return YoY growth rates as scalar floats."""
    if not _is_available():
        return {}
    try:
        t = _ticker(symbol)
        result: dict[str, Any] = {}

        for display_name, method_name_q, value_key in _GROWTH_METHODS:
            mname = method_name_q if quarterly else method_name_q.replace("quarterly_", "annual_")
            try:
                fn = getattr(t, mname, None)
                if fn is None:
                    continue
                val = fn()
                extracted = _extract_scalar(val, value_key)
                if extracted is not None and isinstance(extracted, (int, float)):
                    result[display_name] = extracted
            except Exception:
                continue

        if quarterly:
            for display_name, method_name, value_key in [
                ("diluted_eps_yoy",     "quarterly_diluted_eps_yoy_growth",     "yoy_growth"),
                ("ttm_diluted_eps_yoy", "quarterly_ttm_diluted_eps_yoy_growth", "yoy_growth"),
            ]:
                try:
                    fn = getattr(t, method_name, None)
                    if fn is None:
                        continue
                    val = fn()
                    extracted = _extract_scalar(val, value_key)
                    if extracted is not None and isinstance(extracted, (int, float)):
                        result[display_name] = extracted
                except Exception:
                    continue

        return result
    except Exception as exc:
        log.warning("DefeatBeta growth metrics failed for %s: %s", symbol, exc)
        return {}


# ── EPS data ─────────────────────────────────────────────────────────────────

def get_eps_data(symbol: str) -> dict[str, Any] | None:
    if not _is_available():
        return None
    try:
        t = _ticker(symbol)
        fn = getattr(t, "eps_and_ttm_eps", getattr(t, "eps", None))
        if fn is None:
            return None
        val = fn()
        if val is None:
            return None
        if isinstance(val, pd.DataFrame):
            if val.empty:
                return None
            return val.iloc[-1].to_dict()
        return val if isinstance(val, dict) else None
    except Exception as exc:
        log.warning("DefeatBeta EPS failed for %s: %s", symbol, exc)
        return None


# ── Revenue breakdown ────────────────────────────────────────────────────────

def get_revenue_breakdown(symbol: str) -> dict[str, Any]:
    if not _is_available():
        return {}
    try:
        t = _ticker(symbol)
        result: dict[str, Any] = {}
        for name, method_name in [
            ("by_segment", "revenue_by_segment"),
            ("by_geography", "revenue_by_geography"),
        ]:
            try:
                fn = getattr(t, method_name, None)
                if fn is None:
                    continue
                val = fn()
                if isinstance(val, pd.DataFrame) and not val.empty:
                    result[name] = val.tail(4).to_dict("records")
            except Exception:
                continue
        return result
    except Exception as exc:
        log.warning("DefeatBeta revenue breakdown failed for %s: %s", symbol, exc)
        return {}


# ── Industry comparisons ─────────────────────────────────────────────────────

def get_industry_comparison(symbol: str) -> dict[str, Any]:
    if not _is_available():
        return {}
    try:
        t = _ticker(symbol)
        result: dict[str, Any] = {}
        for name, method_name, value_key in [
            ("industry_pe",  "industry_ttm_pe",       "ttm_pe"),
            ("industry_pb",  "industry_pb",            "pb"),
            ("industry_ps",  "industry_ps",            "ps"),
            ("industry_roe", "industry_quarterly_roe", "roe"),
            ("industry_roa", "industry_quarterly_roa", "roa"),
        ]:
            try:
                fn = getattr(t, method_name, None)
                if fn is None:
                    continue
                val = fn()
                extracted = _extract_scalar(val, value_key)
                if extracted is not None:
                    result[name] = extracted
            except Exception:
                continue
        return result
    except Exception as exc:
        log.warning("DefeatBeta industry comparison failed for %s: %s", symbol, exc)
        return {}


# ── Enterprise value metrics ─────────────────────────────────────────────────

def get_enterprise_metrics(symbol: str) -> dict[str, Any]:
    if not _is_available():
        return {}
    try:
        t = _ticker(symbol)
        result: dict[str, Any] = {}
        for name, method_name in [
            ("enterprise_value", "enterprise_value"),
            ("ev_to_revenue",    "enterprise_to_revenue"),
            ("ev_to_ebitda",     "enterprise_to_ebitda"),
            ("debt_to_equity",   "quarterly_debt_to_equity"),
            ("equity_multiplier","quarterly_equity_multiplier"),
            ("asset_turnover",   "quarterly_asset_turnover"),
        ]:
            try:
                fn = getattr(t, method_name, None)
                if fn is None:
                    continue
                val = fn()
                extracted = _extract_scalar(val)
                if extracted is not None:
                    result[name] = extracted
            except Exception:
                continue
        return result
    except Exception as exc:
        log.warning("DefeatBeta enterprise metrics failed for %s: %s", symbol, exc)
        return {}


# ── Market / Economy ─────────────────────────────────────────────────────────

def get_sp500_returns() -> dict[str, Any]:
    if not _is_available():
        return {}
    try:
        from defeatbeta_api.data.ticker import Ticker
        result: dict[str, Any] = {}
        for name, method_name in [
            ("annual_returns", "sp500_historical_annual_returns"),
            ("cagr", "sp500_cagr_returns"),
        ]:
            try:
                fn = getattr(Ticker, method_name, None)
                if fn is None:
                    continue
                val = fn()
                if isinstance(val, pd.DataFrame) and not val.empty:
                    result[name] = val.tail(5).to_dict("records")
                elif val is not None:
                    result[name] = val
            except Exception:
                continue
        return result
    except Exception:
        return {}


def get_treasury_yield() -> dict[str, Any] | None:
    if not _is_available():
        return None
    try:
        from defeatbeta_api.data.ticker import Ticker
        fn = getattr(Ticker, "daily_treasury_yield", None)
        if fn is None:
            return None
        val = fn()
        if isinstance(val, pd.DataFrame) and not val.empty:
            return val.iloc[-1].to_dict()
        return val if isinstance(val, dict) else None
    except Exception:
        return None
