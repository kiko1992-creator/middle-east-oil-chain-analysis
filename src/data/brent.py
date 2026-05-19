"""
Brent crude oil price data module.

Single source of truth for all Brent price fetching and derived calculations.
Uses Yahoo Finance (BZ=F) via yfinance.  Replaces the former World Bank
CRUDE_BRENT API path.

Key functions
-------------
fetch_live_brent            Latest Brent Futures close (float, NaN on failure)
fetch_brent_ytd             YTD daily closes as a DataFrame (empty on failure)
fetch_brent_history         Annual average prices since *start_year* (DataFrame)
calculate_returns           Year-on-year % returns from a price Series
calculate_rolling_volatility Rolling std of returns over N periods

Graceful failure contract
--------------------------
All fetch functions return a safe empty/NaN value rather than raising.
Callers can detect failure via:
  - ``math.isnan(fetch_live_brent())``
  - ``fetch_brent_ytd().empty``
  - ``live_ok == False`` from ``fetch_brent_history``

The hard-coded annual fallback in ``_BRENT_HISTORY_FALLBACK`` is used by
``fetch_brent_history`` when yfinance is unavailable.  Source: EIA / World
Bank Commodity Price Data, 2000–2024 (annual nominal USD/bbl averages).
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_BRENT_TICKER = "BZ=F"   # Brent Crude Futures on Yahoo Finance

# Annual fallback prices (EIA / World Bank, nominal USD/bbl).
# Used by fetch_brent_history when yfinance is unavailable.
_BRENT_HISTORY_FALLBACK: dict[int, float] = {
    2000: 28.51, 2001: 24.44, 2002: 25.02, 2003: 28.85, 2004: 38.27,
    2005: 54.52, 2006: 65.14, 2007: 72.44, 2008: 96.94, 2009: 61.51,
    2010: 79.50, 2011: 111.26, 2012: 111.63, 2013: 108.66, 2014: 98.97,
    2015: 52.39, 2016: 43.55, 2017: 54.25, 2018: 71.69, 2019: 64.37,
    2020: 41.96, 2021: 70.68, 2022: 100.93, 2023: 82.17, 2024: 80.30,
}


# ── Live price ─────────────────────────────────────────────────────────────────

def fetch_live_brent() -> float:
    """Fetch the most recent Brent Crude Futures (BZ=F) close price.

    Uses the last 5 trading days of history so the result is robust to
    weekends and public holidays when markets are closed.

    Returns:
        Latest available close price in USD/bbl, or ``float('nan')``
        on any network or parse error.
    """
    try:
        hist = yf.Ticker(_BRENT_TICKER).history(period="5d")
        if hist.empty:
            log.warning("yfinance returned empty history for %s", _BRENT_TICKER)
            return float("nan")
        price = float(hist["Close"].iloc[-1])
        log.info("Live Brent (%s): $%.2f/bbl", _BRENT_TICKER, price)
        return price
    except Exception as exc:
        log.warning("fetch_live_brent failed: %s", exc)
        return float("nan")


# ── YTD daily history ──────────────────────────────────────────────────────────

def fetch_brent_ytd() -> pd.DataFrame:
    """Fetch year-to-date daily Brent Crude Futures close prices.

    Fetches from 1 January of the current calendar year through today.

    Returns:
        DataFrame with a DatetimeIndex and a single ``Close`` column
        (USD/bbl).  Returns an empty DataFrame with a ``Close`` column
        on any error.
    """
    year_start = date(date.today().year, 1, 1).isoformat()
    try:
        hist = yf.Ticker(_BRENT_TICKER).history(start=year_start)
        if hist.empty:
            log.warning("No YTD data returned for %s (start=%s)", _BRENT_TICKER, year_start)
            return pd.DataFrame(columns=["Close"])
        log.info(
            "YTD Brent: %d trading days since %s  ($%.2f–$%.2f)",
            len(hist), year_start,
            float(hist["Close"].min()), float(hist["Close"].max()),
        )
        return hist[["Close"]].copy()
    except Exception as exc:
        log.warning("fetch_brent_ytd failed: %s", exc)
        return pd.DataFrame(columns=["Close"])


# ── Annual history ─────────────────────────────────────────────────────────────

def fetch_brent_history(
    start_year: int = 2000,
    end_year: int | None = None,
) -> tuple[pd.DataFrame, bool]:
    """Fetch annual average Brent crude prices.

    Attempts to download daily Brent Futures prices from Yahoo Finance
    and aggregates to calendar-year means.  Falls back to the hard-coded
    ``_BRENT_HISTORY_FALLBACK`` table (EIA/WB 2000–2024) if yfinance
    is unavailable.

    Args:
        start_year: First year to include (default 2000).
        end_year:   Last year to include (default: current year).

    Returns:
        Tuple ``(df, live_ok)`` where:
          - *df* has columns ``year`` (int) and ``price_usd`` (float),
            sorted ascending by year.
          - *live_ok* is ``True`` when live data was retrieved.
    """
    if end_year is None:
        end_year = date.today().year

    try:
        hist = yf.Ticker(_BRENT_TICKER).history(
            start=f"{start_year}-01-01",
            end=f"{end_year + 1}-01-01",
        )
        if hist.empty:
            raise ValueError("Empty yfinance response")

        annual = (
            hist["Close"]
            .rename("price_usd")
            .to_frame()
            .assign(year=lambda df: df.index.year)
            .groupby("year")["price_usd"]
            .mean()
            .reset_index()
            .sort_values("year")
            .reset_index(drop=True)
        )
        log.info(
            "fetch_brent_history: %d annual observations (%d–%d) from yfinance",
            len(annual), int(annual["year"].min()), int(annual["year"].max()),
        )
        return annual, True

    except Exception as exc:
        log.warning("fetch_brent_history failed (%s) — using fallback data", exc)
        df = pd.DataFrame(
            [{"year": y, "price_usd": p}
             for y, p in sorted(_BRENT_HISTORY_FALLBACK.items())]
        )
        df = df[df["year"].between(start_year, end_year)].reset_index(drop=True)
        return df, False


# ── Derived calculations ───────────────────────────────────────────────────────

def calculate_returns(prices: pd.Series) -> pd.Series:
    """Compute year-on-year percentage returns.

    Args:
        prices: Ordered price series (annual or higher frequency).

    Returns:
        Series of the same length; first element is NaN.
    """
    return prices.pct_change().mul(100)


def calculate_rolling_volatility(returns: pd.Series, window: int = 3) -> pd.Series:
    """Compute rolling standard deviation of *returns* over *window* periods.

    Args:
        returns: Return series (e.g. from :func:`calculate_returns`).
        window:  Look-back window in periods (default 3).

    Returns:
        Series of rolling std; first ``window - 1`` elements are NaN.
    """
    return returns.rolling(window).std()
