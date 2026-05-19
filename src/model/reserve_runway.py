"""
Sovereign wealth fund and FX reserve runway model.

Answers: "For governments currently under fiscal stress, how many months
can they sustain their spending rate from liquid reserves?"

The reserve runway is the ratio of a government's liquid buffer (accessible
FX reserves + deployable SWF assets) to its estimated monthly fiscal burn
rate — the rate at which reserves are drawn down when oil is below the
fiscal breakeven price.

Runway is only classified (Critical / Red / Amber / Green) when a country
is already under fiscal stress (fiscal_stress status Red or Amber from
Addition 1).  Countries currently above their breakeven (Green fiscal) or
net importers (Gray fiscal) are shown as Gray — their runway is not the
pressing question at that price.

Key functions
-------------
load_reserves          Load and validate data/reference/swf_reserves.csv
compute_runway         Compute reserve_runway_months = buffer / monthly_burn
classify_runway        Assign Critical / Red / Amber / Green / Gray status
build_runway_table     Merge reserves with fiscal stress table; add runway
run_reserve_runway     End-to-end orchestrator

Runway status thresholds
------------------------
  Critical : months <  6   (imminent depletion risk)
  Red      : months <  12  (very short runway)
  Amber    : months <  36  (limited buffer — 1–3 years)
  Green    : months >= 36  (comfortable buffer at current burn rate)
  Gray     : fiscal stress = Green or Gray  (not applicable at this price)

liquid_buffer_usd_bn definition
--------------------------------
  = FX reserves (central bank) + accessible / liquid SWF portion

  Notable exclusions:
    - Kuwait RFFG (~85% of KIA): constitutionally ring-fenced
    - ADIA / Mubadala / ADQ: long-term illiquid equity
    - QIA: majority illiquid global equity
    - PIF: long-term equity / project fund
    - GCC Development Fund support: off-balance-sheet, not guaranteed
    - Iran NDFI + CBI: largely frozen/inaccessible under sanctions

Data sources
------------
- Reserve estimates: data/reference/swf_reserves.csv
  (SWF annual reports, IMF Article IV Consultations, central bank
   statistical bulletins, 2023; all values are preliminary estimates —
   see confidence and notes fields)
- Fiscal stress table: output of src.model.fiscal_stress.build_stress_table
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd

from src.model.fiscal_stress import (
    build_stress_table,
    fetch_brent_live,
    fetch_brent_ytd,
    load_breakeven,
)

log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

RESERVES_PATH  = Path("data/reference/swf_reserves.csv")
BREAKEVEN_PATH = Path("data/reference/fiscal_breakeven.csv")

# Runway status thresholds (months).
_CRITICAL_THRESHOLD = 6.0
_RED_THRESHOLD      = 12.0
_AMBER_THRESHOLD    = 36.0

# Fiscal stress statuses that suppress runway classification.
_NOT_STRESSED = frozenset({"Green", "Gray"})

# Columns required in the reserves CSV.
_REQUIRED_COLS: list[str] = [
    "country_code",
    "country_code_a3",
    "country_name",
    "country_label",
    "liquid_buffer_usd_bn",
    "estimated_monthly_burn_usd_bn",
]

# Fiscal stress columns expected in the stress table (from Addition 1).
_STRESS_COLS: list[str] = [
    "country_code_a3",
    "stress_status",
    "price_gap_usd",
    "brent_live_usd",
]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_reserves(path: Path = RESERVES_PATH) -> pd.DataFrame:
    """Load and validate the SWF / FX reserves reference CSV.

    Args:
        path: Path to ``swf_reserves.csv``.

    Returns:
        DataFrame with one row per country and all reference fields.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If any required column is absent.
    """
    if not path.exists():
        raise FileNotFoundError(f"SWF/reserves CSV not found: {path}")

    df = pd.read_csv(path)

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {missing}")

    # Coerce numeric fields; bad values become NaN.
    for col in ("liquid_buffer_usd_bn", "estimated_monthly_burn_usd_bn",
                "swf_assets_usd_bn", "fx_reserves_usd_bn"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info(
        "Reserves data loaded: %d countries  "
        "(total liquid buffer $%.0f bn, %d with non-zero burn)",
        len(df),
        df["liquid_buffer_usd_bn"].sum(skipna=True),
        int((df["estimated_monthly_burn_usd_bn"].fillna(0) > 0).sum()),
    )
    return df


# ── Core calculations ──────────────────────────────────────────────────────────

def compute_runway(liquid_buffer: float, monthly_burn: float) -> float:
    """Compute reserve runway in months.

    Formula:
        reserve_runway_months = liquid_buffer_usd_bn / estimated_monthly_burn_usd_bn

    Returns ``float('nan')`` — not zero — when the burn rate is zero or
    missing, because a zero burn rate signals "not applicable" rather than
    "infinite runway."  The downstream classifier treats NaN as Gray.

    Args:
        liquid_buffer: Accessible liquid reserves (USD bn).
        monthly_burn:  Estimated monthly reserve drawdown (USD bn / month).

    Returns:
        Months of runway, or ``float('nan')`` when undefined.
    """
    if pd.isna(liquid_buffer) or pd.isna(monthly_burn):
        return float("nan")
    if monthly_burn <= 0.0:
        return float("nan")   # zero burn = N/A, not infinite
    return liquid_buffer / monthly_burn


def classify_runway(months: float, fiscal_stress: str) -> str:
    """Classify a country's reserve runway status.

    Runway is only meaningful when a country is currently under fiscal
    stress (drawing down reserves to cover a budget deficit).  Countries
    with Green or Gray fiscal status have no urgent depletion risk at the
    current price, so their runway is shown as Gray.

    Classification rules (applied in order):
        1. Gray     — fiscal_stress is Green or Gray (not under stress)
        2. Gray     — months is NaN (burn is zero or data unavailable)
        3. Critical — months < 6
        4. Red      — 6 <= months < 12
        5. Amber    — 12 <= months < 36
        6. Green    — months >= 36

    Args:
        months:        reserve_runway_months from :func:`compute_runway`.
        fiscal_stress: Fiscal stress status from Addition 1
                       (one of ``'Red'``, ``'Amber'``, ``'Green'``, ``'Gray'``).

    Returns:
        One of ``'Critical'``, ``'Red'``, ``'Amber'``, ``'Green'``, ``'Gray'``.
    """
    if fiscal_stress in _NOT_STRESSED:
        return "Gray"
    if pd.isna(months) or math.isnan(months):
        return "Gray"
    if months < _CRITICAL_THRESHOLD:
        return "Critical"
    if months < _RED_THRESHOLD:
        return "Red"
    if months < _AMBER_THRESHOLD:
        return "Amber"
    return "Green"


# ── Table assembly ─────────────────────────────────────────────────────────────

def build_runway_table(
    reserves_df: pd.DataFrame,
    stress_table: pd.DataFrame,
) -> pd.DataFrame:
    """Merge reserve data with the fiscal stress table and compute runway metrics.

    Joins on ``country_code_a3`` (left join on reserves, so all 14 countries
    are retained even if fiscal stress data is absent for any row).

    Columns added:

        stress_status         Fiscal stress status from Addition 1
        price_gap_usd         Live Brent minus fiscal breakeven (USD/bbl)
        brent_live_usd        Current Brent close (USD/bbl)
        reserve_runway_months liquid_buffer / monthly_burn
        runway_status         Critical / Red / Amber / Green / Gray

    Gray rows (runway_status == "Gray") receive NaN for
    ``reserve_runway_months`` so the UI can display "N/A" rather than
    a potentially misleading numeric value.

    Args:
        reserves_df:  Output of :func:`load_reserves`.
        stress_table: Output of ``src.model.fiscal_stress.build_stress_table``.

    Returns:
        DataFrame with one row per country, sorted so the most urgent
        (shortest non-Gray runway) rows appear first.
    """
    # Pull only the columns we need from the stress table to avoid name clashes.
    stress_cols_present = [c for c in _STRESS_COLS if c in stress_table.columns]
    stress_slim = stress_table[stress_cols_present].copy()

    df = reserves_df.merge(stress_slim, on="country_code_a3", how="left")

    # Fallback: if a country has no stress data, treat as Gray.
    df["stress_status"] = df["stress_status"].fillna("Gray")
    df["brent_live_usd"] = df.get("brent_live_usd", pd.Series(float("nan"), index=df.index))
    df["price_gap_usd"]  = df.get("price_gap_usd",  pd.Series(float("nan"), index=df.index))

    # Runway calculation (vectorised via apply to handle NaN cleanly).
    df["reserve_runway_months"] = df.apply(
        lambda r: compute_runway(
            float(r["liquid_buffer_usd_bn"]),
            float(r["estimated_monthly_burn_usd_bn"]),
        ),
        axis=1,
    )

    # Runway status classification.
    df["runway_status"] = df.apply(
        lambda r: classify_runway(
            float(r["reserve_runway_months"]) if not pd.isna(r["reserve_runway_months"]) else float("nan"),
            str(r["stress_status"]),
        ),
        axis=1,
    )

    # Nullify runway months for Gray rows — a numeric value would be misleading.
    gray_mask = df["runway_status"] == "Gray"
    df.loc[gray_mask, "reserve_runway_months"] = float("nan")

    # Sort: most urgent (shortest runway) first, Gray rows last.
    _status_order = {"Critical": 0, "Red": 1, "Amber": 2, "Green": 3, "Gray": 4}
    df["_sort_status"] = df["runway_status"].map(_status_order).fillna(9)
    df = (
        df.sort_values(
            ["_sort_status", "reserve_runway_months"],
            ascending=[True, True],
            na_position="last",
        )
        .drop(columns=["_sort_status"])
        .reset_index(drop=True)
    )

    n_critical = int((df["runway_status"] == "Critical").sum())
    n_red      = int((df["runway_status"] == "Red").sum())
    n_amber    = int((df["runway_status"] == "Amber").sum())
    n_green    = int((df["runway_status"] == "Green").sum())
    n_gray     = int((df["runway_status"] == "Gray").sum())
    log.info(
        "Runway table built: Critical=%d  Red=%d  Amber=%d  Green=%d  Gray=%d",
        n_critical, n_red, n_amber, n_green, n_gray,
    )
    return df


# ── End-to-end orchestrator ────────────────────────────────────────────────────

def run_reserve_runway(
    reserves_path: Path   = RESERVES_PATH,
    breakeven_path: Path  = BREAKEVEN_PATH,
) -> dict:
    """End-to-end reserve runway pipeline.

    Fetches live Brent, builds the Addition 1 fiscal stress table, loads
    reserve reference data, and assembles the runway table.

    Args:
        reserves_path:  Path to ``swf_reserves.csv``.
        breakeven_path: Path to ``fiscal_breakeven.csv``.

    Returns:
        dict with keys:

            ``'brent_live'``    : float — latest Brent close (USD/bbl)
            ``'stress_table'``  : pd.DataFrame — fiscal stress table (Addition 1)
            ``'runway_table'``  : pd.DataFrame — per-country runway metrics

    Raises:
        FileNotFoundError: If either CSV does not exist.
        ValueError: If required columns are absent.
    """
    breakeven_df = load_breakeven(breakeven_path)
    brent_live   = fetch_brent_live()
    ytd_prices   = fetch_brent_ytd()
    stress_table = build_stress_table(breakeven_df, brent_live, ytd_prices)
    reserves_df  = load_reserves(reserves_path)
    runway_table = build_runway_table(reserves_df, stress_table)

    return {
        "brent_live":   brent_live,
        "stress_table": stress_table,
        "runway_table": runway_table,
    }
