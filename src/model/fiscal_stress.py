"""
IMF-style fiscal breakeven stress model for MENA governments.

Answers the question: "At current Brent price, which MENA governments
are above or below their fiscal breakeven?"

The fiscal breakeven is the Brent crude price (USD/bbl) at which a
government's budget is in balance given its current spending commitments.
Exporters trading below their breakeven must draw down reserves, cut
spending, or borrow — fiscal stress.

Key functions
-------------
load_breakeven      Load and validate data/reference/fiscal_breakeven.csv
fetch_brent_live    Fetch the latest Brent Crude Futures close (BZ=F)
fetch_brent_ytd     Fetch year-to-date daily Brent close history (BZ=F)
classify_stress     Assign Red / Amber / Green / Gray stress status
compute_stress_days Count YTD trading days where Brent < breakeven
build_stress_table  Assemble per-country stress metrics into a DataFrame
run_fiscal_stress   End-to-end orchestrator — returns live price + table

Stress classification thresholds
---------------------------------
  Red   : Brent < breakeven               (government in fiscal deficit)
  Amber : breakeven <= Brent < breakeven + 15  (thin buffer)
  Green : Brent >= breakeven + 15         (comfortable fiscal headroom)
  Gray  : net_importer or breakeven == 0  (concept not applicable)

Data sources
------------
- Breakeven estimates: data/reference/fiscal_breakeven.csv
  (IMF Article IV consultations / Regional Economic Outlooks, 2023;
   all values are preliminary estimates — see confidence field)
- Live Brent: Yahoo Finance ticker BZ=F via yfinance
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data.brent import fetch_live_brent, fetch_brent_ytd

log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

BREAKEVEN_PATH = Path("data/reference/fiscal_breakeven.csv")
WEO_PANEL_PATH = Path("data/processed/imf_weo_panel.csv")

# USD above breakeven before a country is classified Green (not merely Amber).
_AMBER_BUFFER = 15.0

# Columns that must be present in the breakeven CSV.
_REQUIRED_COLS: list[str] = [
    "country_code",
    "country_code_a3",
    "country_name",
    "country_label",
    "country_type",
    "fiscal_breakeven_usd",
]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_breakeven(path: Path = BREAKEVEN_PATH) -> pd.DataFrame:
    """Load and validate the fiscal breakeven reference CSV.

    Args:
        path: Path to fiscal_breakeven.csv.

    Returns:
        DataFrame with one row per country and all reference fields.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If any required column is absent.
    """
    if not path.exists():
        raise FileNotFoundError(f"Fiscal breakeven CSV not found: {path}")

    df = pd.read_csv(path)

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {missing}")

    # Coerce breakeven to numeric; bad values become NaN (treated as N/A).
    df["fiscal_breakeven_usd"] = pd.to_numeric(
        df["fiscal_breakeven_usd"], errors="coerce"
    ).fillna(0.0)

    log.info(
        "Breakeven data loaded: %d countries  (%d exporters, %d importers)",
        len(df),
        int((df["country_type"] == "exporter").sum()),
        int((df["country_type"] == "net_importer").sum()),
    )
    return df


# ── Live price fetching ────────────────────────────────────────────────────────
# Implementations live in src.data.brent; re-exported here for backward
# compatibility with existing imports in pages and other model modules.

def fetch_brent_live() -> float:
    """Fetch the most recent Brent Crude Futures (BZ=F) close price.

    Delegates to :func:`src.data.brent.fetch_live_brent`.

    Returns:
        Latest available close price in USD/bbl, or ``float('nan')`` on error.
    """
    return fetch_live_brent()


# fetch_brent_ytd is imported directly from src.data.brent above and
# re-exported as-is — no wrapper needed.


# ── Stress classification ──────────────────────────────────────────────────────

def classify_stress(brent: float, breakeven: float, country_type: str) -> str:
    """Classify a country's fiscal stress status given the current Brent price.

    Classification rules (applied in order):
        1. Gray  — brent is NaN (price unavailable)
        2. Gray  — net_importer OR breakeven == 0  (concept not applicable)
        3. Red   — brent < breakeven               (below breakeven)
        4. Amber — breakeven <= brent < breakeven + _AMBER_BUFFER  (thin buffer)
        5. Green — brent >= breakeven + _AMBER_BUFFER              (comfortable)

    Args:
        brent:        Current Brent Crude price (USD/bbl).
        breakeven:    Fiscal breakeven price (USD/bbl); 0 signals N/A.
        country_type: One of ``'exporter'``, ``'mixed_importer'``,
                      ``'net_importer'``.

    Returns:
        One of ``'Red'``, ``'Amber'``, ``'Green'``, ``'Gray'``.
        ``'Gray'`` is returned both for N/A countries and when the live
        price is unavailable (NaN), so the caller can treat both as
        "cannot classify" without distinguishing the cause.
    """
    if pd.isna(brent):
        return "Gray"   # live price unavailable — NaN comparisons silently return False
    if country_type == "net_importer" or breakeven == 0.0:
        return "Gray"
    if brent < breakeven:
        return "Red"
    if brent < breakeven + _AMBER_BUFFER:
        return "Amber"
    return "Green"


# ── YTD stress counting ────────────────────────────────────────────────────────

def compute_stress_days(
    ytd_prices: pd.DataFrame,
    breakeven: float,
    country_type: str,
) -> tuple[int, float, int]:
    """Count YTD trading days where Brent closed below the fiscal breakeven.

    For net importers and zero-breakeven countries the concept is not
    applicable; this function returns zeros for those cases.

    Args:
        ytd_prices:   DataFrame with a ``Close`` column (daily Brent history).
        breakeven:    Fiscal breakeven price (USD/bbl); 0 signals N/A.
        country_type: ``'exporter'``, ``'mixed_importer'``, or
                      ``'net_importer'``.

    Returns:
        Tuple of ``(stress_days, stress_share, total_trading_days)`` where
        ``stress_share`` is ``stress_days / total_trading_days`` ∈ [0, 1].
        All three are zero for N/A countries or when history is empty.
    """
    total = len(ytd_prices)
    if country_type == "net_importer" or breakeven == 0.0 or total == 0:
        return (0, 0.0, total)

    stress_days  = int((ytd_prices["Close"] < breakeven).sum())
    stress_share = stress_days / total
    return (stress_days, stress_share, total)


# ── Stress table assembly ──────────────────────────────────────────────────────

def build_stress_table(
    breakeven_df: pd.DataFrame,
    brent_live: float,
    ytd_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the per-country fiscal stress metrics table.

    For each country the following columns are added to the breakeven
    reference data:

        brent_live_usd      Current Brent close (USD/bbl)
        price_gap_usd       brent_live − fiscal_breakeven_usd
                            (positive = above breakeven; negative = below)
        stress_status       'Red' / 'Amber' / 'Green' / 'Gray'
        stress_days_ytd     Trading days this year where Close < breakeven
        stress_share_ytd    stress_days_ytd / total_trading_days_ytd
        total_trading_days  Total YTD trading days in the Brent history

    Net-importer countries (Gray) receive NaN for price_gap_usd,
    stress_days_ytd, and stress_share_ytd to signal N/A rather than zero.

    Args:
        breakeven_df: Output of :func:`load_breakeven`.
        brent_live:   Current Brent price from :func:`fetch_brent_live`.
        ytd_prices:   YTD daily Brent from :func:`fetch_brent_ytd`.

    Returns:
        DataFrame sorted by country_type then price_gap_usd (ascending),
        so the most stressed exporters appear first.
    """
    df = breakeven_df.copy()
    df["brent_live_usd"] = brent_live
    df["price_gap_usd"]  = brent_live - df["fiscal_breakeven_usd"]

    # Classify stress status for every country
    df["stress_status"] = df.apply(
        lambda r: classify_stress(
            brent_live,
            float(r["fiscal_breakeven_usd"]),
            str(r["country_type"]),
        ),
        axis=1,
    )

    # Compute YTD stress days per country (same Brent series; breakeven differs)
    raw_results = df.apply(
        lambda r: compute_stress_days(
            ytd_prices,
            float(r["fiscal_breakeven_usd"]),
            str(r["country_type"]),
        ),
        axis=1,
        result_type="expand",
    )
    df["stress_days_ytd"]    = raw_results[0].astype(int)
    df["stress_share_ytd"]   = raw_results[1].astype(float)
    df["total_trading_days"] = raw_results[2].astype(int)

    # Nullify gap and stress metrics for Gray (N/A) countries — 0 would be misleading
    gray_mask = df["stress_status"] == "Gray"
    df.loc[gray_mask, "price_gap_usd"]    = float("nan")
    df.loc[gray_mask, "stress_days_ytd"]  = pd.NA
    df.loc[gray_mask, "stress_share_ytd"] = float("nan")

    # Optional IMF WEO 2025 fiscal balance context (informational only).
    df["weo_fiscal_balance_2025"] = float("nan")
    if WEO_PANEL_PATH.exists():
        try:
            weo = pd.read_csv(WEO_PANEL_PATH, usecols=["country_code_a3", "year", "indicator", "value"])
            weo = weo[(weo["indicator"] == "GGXCNL_NGDP") & (weo["year"] == 2025)][["country_code_a3", "value"]]
            weo = weo.rename(columns={"value": "weo_fiscal_balance_2025"})
            df = df.merge(weo, on="country_code_a3", how="left", suffixes=("", "_weo"))
            if "weo_fiscal_balance_2025_weo" in df.columns:
                df["weo_fiscal_balance_2025"] = df["weo_fiscal_balance_2025_weo"]
                df = df.drop(columns=["weo_fiscal_balance_2025_weo"])
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not attach WEO fiscal balance context: %s", exc)

    # Nullify YTD stress metrics when the Brent history fetch returned no data.
    # Leaving stress_days_ytd=0 would imply "no stress days" rather than "no data".
    no_ytd_mask = (df["total_trading_days"] == 0) & (df["stress_status"] != "Gray")
    df.loc[no_ytd_mask, "stress_days_ytd"]  = pd.NA
    df.loc[no_ytd_mask, "stress_share_ytd"] = float("nan")

    # Sort: exporters first, within each group worst gap first
    _type_order = {"exporter": 0, "mixed_importer": 1, "net_importer": 2}
    df["_sort_type"] = df["country_type"].map(_type_order).fillna(9)
    df = (
        df.sort_values(["_sort_type", "price_gap_usd"], ascending=[True, True])
        .drop(columns=["_sort_type"])
        .reset_index(drop=True)
    )

    n_red   = int((df["stress_status"] == "Red").sum())
    n_amber = int((df["stress_status"] == "Amber").sum())
    n_green = int((df["stress_status"] == "Green").sum())
    log.info(
        "Stress table built: Red=%d  Amber=%d  Green=%d  Gray=%d  (Brent=$%.2f)",
        n_red, n_amber, n_green,
        int((df["stress_status"] == "Gray").sum()),
        brent_live,
    )
    return df


# ── End-to-end orchestrator ────────────────────────────────────────────────────

def run_fiscal_stress(
    breakeven_path: Path = BREAKEVEN_PATH,
) -> dict:
    """End-to-end fiscal breakeven stress pipeline.

    Loads the static reference CSV, fetches live and YTD Brent prices from
    Yahoo Finance, and assembles the per-country stress table.

    Args:
        breakeven_path: Path to ``fiscal_breakeven.csv``.

    Returns:
        dict with keys:

            ``'brent_live'``    : float — latest Brent close (USD/bbl)
            ``'ytd_prices'``    : pd.DataFrame — daily YTD Brent closes
            ``'stress_table'``  : pd.DataFrame — per-country stress metrics

    Raises:
        FileNotFoundError: If *breakeven_path* does not exist.
        ValueError: If required columns are absent from the CSV.
    """
    breakeven_df = load_breakeven(breakeven_path)
    brent_live   = fetch_brent_live()
    ytd_prices   = fetch_brent_ytd()
    stress_table = build_stress_table(breakeven_df, brent_live, ytd_prices)

    return {
        "brent_live":   brent_live,
        "ytd_prices":   ytd_prices,
        "stress_table": stress_table,
    }
