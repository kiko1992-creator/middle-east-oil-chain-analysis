"""
Historical backtesting scaffolding for the Right Now Risk model.

This module performs CONDITIONAL backtests: it applies historical Brent
crude oil prices and chain transmission data to the CURRENT reference
datasets (fiscal breakeven, reserves, food security) and computes what
the Right Now Risk score would have been.

This is intentionally NOT a true simulation of past state — the fiscal
breakeven estimates and reserve data are 2023 reference figures.  The
value is in understanding how the composite score responds across different
oil price environments and chain severity levels.

See docs/backtesting_plan.md for target periods and success criteria.

Key functions
-------------
build_historical_brent_proxy    Fetch annual average Brent prices (2000+)
run_backtest_snapshot           Right Now Risk score for one specific year
run_backtest_range              Run snapshots for a range of years
summarize_rank_stability        Per-country rank statistics across years
export_backtest_outputs         Write per-year CSVs and summary to outputs/

Scenarios
---------
  "base"       : use breakeven_base_usd and liquid_buffer_base_usd_bn
  "stress"     : use breakeven_high_usd + liquid_buffer_low_usd_bn
                 (most pessimistic: high breakeven, low accessible buffer)
  "optimistic" : use breakeven_low_usd + liquid_buffer_high_usd_bn
                 (most favorable: low breakeven, high accessible buffer)

Metadata columns added to every backtest output row
----------------------------------------------------
    snapshot_year, scenario, method_version,
    missing_components_count, historical_brent_usd
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data.brent import fetch_brent_history
from src.model.fiscal_stress import build_stress_table, load_breakeven
from src.model.reserve_runway import build_runway_table, load_reserves
from src.model.right_now_risk import (
    compute_chain_recent,
    compute_right_now_risk,
    compute_reserve_runway_risk,
    _W_FISCAL, _W_RUNWAY, _W_SOCIAL, _W_CHAIN,
)
from src.model.social_stability import (
    build_stability_table,
    derive_inflation_vol,
    load_food_security,
)

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_METHOD_VERSION = "1.0"

BREAKEVEN_PATH = Path("data/reference/fiscal_breakeven.csv")
RESERVES_PATH  = Path("data/reference/swf_reserves.csv")
FOOD_PATH      = Path("data/reference/food_security.csv")
CHAIN_PATH     = Path("outputs/tables/chain_transmission.csv")
PANEL_PATH     = Path("data/processed/world_bank_panel.csv")
BACKTEST_DIR   = Path("outputs/tables/backtest")

_VALID_SCENARIOS = frozenset({"base", "stress", "optimistic"})


# ── Historical Brent proxy ─────────────────────────────────────────────────────

def build_historical_brent_proxy(
    start_year: int = 2000,
    end_year:   int | None = None,
) -> pd.DataFrame:
    """Return annual average Brent crude prices from the project's data source.

    Delegates to :func:`src.data.brent.fetch_brent_history`, which tries
    yfinance first and falls back to the hard-coded EIA/World Bank reference
    table (2000–2024).

    Args:
        start_year: First year to include (default 2000).
        end_year:   Last year to include (default: current year).

    Returns:
        DataFrame with columns ``year`` (int) and ``price_usd`` (float),
        sorted ascending by year.  Includes a ``source`` column ('yfinance'
        or 'fallback') for transparency.
    """
    df, live_ok = fetch_brent_history(start_year=start_year, end_year=end_year)
    df["source"] = "yfinance" if live_ok else "fallback_EIA_WB"
    log.info(
        "Historical Brent proxy: %d years (%d–%d) source=%s",
        len(df),
        int(df["year"].min()), int(df["year"].max()),
        df["source"].iloc[0],
    )
    return df


# ── Scenario helpers ───────────────────────────────────────────────────────────

def _apply_scenario_to_breakeven(
    breakeven_df: pd.DataFrame,
    scenario: str,
) -> pd.DataFrame:
    """Swap fiscal_breakeven_usd to the scenario-appropriate uncertainty band.

    base      : use existing fiscal_breakeven_usd (no change)
    stress    : use breakeven_high_usd where available (higher breakeven = more stress)
    optimistic: use breakeven_low_usd  where available (lower breakeven = less stress)

    Rows where the target column is null retain their base value.
    """
    df = breakeven_df.copy()
    if scenario == "base":
        return df
    col = "breakeven_high_usd" if scenario == "stress" else "breakeven_low_usd"
    if col not in df.columns:
        log.warning("%s column absent — falling back to base scenario", col)
        return df
    mask = df[col].notna() & (pd.to_numeric(df[col], errors="coerce") > 0)
    df.loc[mask, "fiscal_breakeven_usd"] = pd.to_numeric(
        df.loc[mask, col], errors="coerce"
    )
    return df


def _apply_scenario_to_reserves(
    reserves_df: pd.DataFrame,
    scenario: str,
) -> pd.DataFrame:
    """Swap reserves/burn values to the scenario-appropriate uncertainty band.

    base      : use existing liquid_buffer_usd_bn / estimated_monthly_burn_usd_bn
    stress    : smaller accessible buffer + higher burn (most pessimistic runway)
    optimistic: larger accessible buffer + lower burn  (most favorable runway)
    """
    df = reserves_df.copy()
    if scenario == "base":
        return df

    if scenario == "stress":
        buf_col  = "liquid_buffer_low_usd_bn"
        burn_col = "monthly_burn_high_usd_bn"
    else:  # optimistic
        buf_col  = "liquid_buffer_high_usd_bn"
        burn_col = "monthly_burn_low_usd_bn"

    for src_col, tgt_col in [
        (buf_col,  "liquid_buffer_usd_bn"),
        (burn_col, "estimated_monthly_burn_usd_bn"),
    ]:
        if src_col in df.columns:
            mask = df[src_col].notna()
            df.loc[mask, tgt_col] = pd.to_numeric(df.loc[mask, src_col], errors="coerce")

    return df


# ── Single-year snapshot ───────────────────────────────────────────────────────

def run_backtest_snapshot(
    year:           int,
    scenario:       str  = "base",
    breakeven_path: Path = BREAKEVEN_PATH,
    reserves_path:  Path = RESERVES_PATH,
    food_path:      Path = FOOD_PATH,
    chain_path:     Path = CHAIN_PATH,
    panel_path:     Path = PANEL_PATH,
    brent_override: float | None = None,
) -> pd.DataFrame:
    """Compute Right Now Risk scores for a single historical year.

    Uses the historical annual-average Brent price for *year* as the
    ``brent_live`` input to the fiscal stress model.  Chain transmission
    severity uses only that specific year's data from chain_transmission.csv.
    All other reference data (breakeven estimates, reserves, food security)
    are the current 2023 reference figures, optionally modified by *scenario*.

    Args:
        year:           Calendar year to evaluate (must be in Brent history).
        scenario:       "base", "stress", or "optimistic".
        breakeven_path: Path to fiscal_breakeven.csv.
        reserves_path:  Path to swf_reserves.csv.
        food_path:      Path to food_security.csv.
        chain_path:     Path to chain_transmission.csv.
        panel_path:     Path to world_bank_panel.csv.
        brent_override: Override the historical Brent price (for unit tests).

    Returns:
        DataFrame of Right Now Risk scores with metadata columns:
            snapshot_year, scenario, method_version,
            missing_components_count, historical_brent_usd.

    Raises:
        ValueError: If *scenario* is not in _VALID_SCENARIOS, or if no
                    Brent price is available for *year*.
    """
    if scenario not in _VALID_SCENARIOS:
        raise ValueError(
            f"scenario must be one of {_VALID_SCENARIOS}, got {scenario!r}"
        )

    # ── Historical Brent price for this year ──────────────────────────────────
    if brent_override is not None:
        brent_price = float(brent_override)
    else:
        brent_hist = build_historical_brent_proxy()
        year_row   = brent_hist[brent_hist["year"] == year]
        if year_row.empty:
            raise ValueError(
                f"No historical Brent price available for year {year}. "
                f"Available range: {int(brent_hist['year'].min())}–"
                f"{int(brent_hist['year'].max())}"
            )
        brent_price = float(year_row["price_usd"].iloc[0])

    log.info("Backtest snapshot: year=%d  scenario=%s  Brent=$%.2f",
             year, scenario, brent_price)

    # ── Load and apply scenario modifications ─────────────────────────────────
    ytd_prices   = pd.DataFrame(columns=["Close"])   # no YTD data in backtest
    breakeven_df = _apply_scenario_to_breakeven(load_breakeven(breakeven_path), scenario)
    stress_table = build_stress_table(breakeven_df, brent_price, ytd_prices)

    reserves_df  = _apply_scenario_to_reserves(load_reserves(reserves_path), scenario)
    runway_table = build_runway_table(reserves_df, stress_table)

    food_df         = load_food_security(food_path)
    inflation_df    = derive_inflation_vol(panel_path)
    stability_table = build_stability_table(food_df, stress_table, inflation_df)

    # Chain: use only this specific year's transmission data
    chain_df = pd.DataFrame(columns=["country_code_a3", "year", "transmission_severity"])
    if chain_path.exists():
        full_chain = pd.read_csv(chain_path)
        chain_df   = full_chain[full_chain["year"] == year].copy()
        if chain_df.empty:
            log.warning(
                "No chain data for year %d — chain component will be NaN", year
            )

    chain_recent_df = compute_chain_recent(chain_df, n_years=1)

    # ── Composite score ───────────────────────────────────────────────────────
    rnr_df = compute_right_now_risk(
        stress_table=stress_table,
        runway_df=runway_table,
        stability_df=stability_table,
        chain_recent_df=chain_recent_df,
    )

    # ── Metadata columns ──────────────────────────────────────────────────────
    rnr_df["snapshot_year"]          = year
    rnr_df["scenario"]               = scenario
    rnr_df["method_version"]         = _METHOD_VERSION
    rnr_df["historical_brent_usd"]   = round(brent_price, 2)
    rnr_df["missing_components_count"] = (
        rnr_df["missing_components"]
        .apply(lambda x: len(x.split(";")) if x else 0)
    )

    log.info(
        "Snapshot complete: year=%d  scenario=%s  countries=%d  "
        "Brent=$%.2f  partial_rows=%d",
        year, scenario, len(rnr_df), brent_price,
        int((rnr_df["missing_components_count"] > 0).sum()),
    )
    return rnr_df


# ── Multi-year range ───────────────────────────────────────────────────────────

def run_backtest_range(
    year_start: int,
    year_end:   int,
    scenario:   str  = "base",
    **kwargs,
) -> pd.DataFrame:
    """Run run_backtest_snapshot for every year in [year_start, year_end].

    Args:
        year_start: First year (inclusive).
        year_end:   Last year (inclusive).
        scenario:   Scenario to apply to all years.
        **kwargs:   Forwarded to run_backtest_snapshot (paths, overrides).

    Returns:
        Concatenated DataFrame of all yearly snapshots, sorted by
        (snapshot_year, country_code_a3).
    """
    frames = []
    for year in range(year_start, year_end + 1):
        try:
            frames.append(run_backtest_snapshot(year=year, scenario=scenario, **kwargs))
        except ValueError as exc:
            log.warning("Skipping year %d: %s", year, exc)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["snapshot_year", "country_code_a3"]).reset_index(drop=True)


# ── Rank stability summary ─────────────────────────────────────────────────────

def summarize_rank_stability(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-country rank statistics across all snapshot years.

    Within each snapshot year, countries are ranked by right_now_risk_score
    (rank 1 = highest risk).  Then per-country rank statistics are computed
    across all years.

    Args:
        backtest_df: Output of run_backtest_range (multiple years).

    Returns:
        DataFrame with one row per country sorted by mean_rank ascending, with:
            country_code_a3, country_label (if present), mean_rank, min_rank,
            max_rank, std_rank, n_years, score_mean, score_std.
    """
    if backtest_df.empty or "right_now_risk_score" not in backtest_df.columns:
        return pd.DataFrame()

    df = backtest_df.dropna(subset=["right_now_risk_score"]).copy()

    df["rank_in_year"] = df.groupby("snapshot_year")["right_now_risk_score"].rank(
        ascending=False, method="min"
    )

    group_cols = ["country_code_a3"]
    if "country_label" in df.columns:
        group_cols = ["country_code_a3", "country_label"]

    summary = (
        df.groupby(group_cols)
        .agg(
            mean_rank  = ("rank_in_year",         "mean"),
            min_rank   = ("rank_in_year",         "min"),
            max_rank   = ("rank_in_year",         "max"),
            std_rank   = ("rank_in_year",         "std"),
            n_years    = ("snapshot_year",         "nunique"),
            score_mean = ("right_now_risk_score", "mean"),
            score_std  = ("right_now_risk_score", "std"),
        )
        .reset_index()
        .sort_values("mean_rank")
        .reset_index(drop=True)
    )
    summary["mean_rank"] = summary["mean_rank"].round(1)
    summary["std_rank"]  = summary["std_rank"].round(2)
    summary["score_mean"] = summary["score_mean"].round(4)
    summary["score_std"]  = summary["score_std"].round(4)
    return summary


# ── Export ─────────────────────────────────────────────────────────────────────

def export_backtest_outputs(
    backtest_df: pd.DataFrame,
    summary_df:  pd.DataFrame | None = None,
    output_dir:  Path = BACKTEST_DIR,
) -> list[Path]:
    """Write per-year CSVs and an optional summary CSV.

    Args:
        backtest_df: Combined backtest output (multiple years, one row per
                     country per year).
        summary_df:  Optional rank stability summary from
                     :func:`summarize_rank_stability`.
        output_dir:  Directory to write outputs (created if absent).

    Returns:
        List of Path objects for every file written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    for year, group in backtest_df.groupby("snapshot_year"):
        scenario = group["scenario"].iloc[0] if "scenario" in group.columns else "base"
        fname    = output_dir / f"backtest_{year}_{scenario}.csv"
        group.to_csv(fname, index=False)
        written.append(fname)

    if summary_df is not None and not summary_df.empty:
        fname = output_dir / "backtest_rank_stability.csv"
        summary_df.to_csv(fname, index=False)
        written.append(fname)

    log.info("Backtest outputs written: %d files to %s", len(written), output_dir)
    return written
