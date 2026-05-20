"""
Chain Transmission Severity Model — Addition 4.

Reads structural stage parameters from data/reference/chain_transmission.csv
and computes a per-country chain_transmission_severity score in [0, 1].

Formula
-------
    stage_mean                  = mean(stage1_oil_fiscal,
                                       stage2_fiscal_inflation,
                                       stage3_inflation_employment,
                                       stage4_employment_consumption,
                                       stage5_consumption_growth)
    chain_transmission_severity = min(1.0, stage_mean * amplification_factor)

Five stages capture the propagation path of an oil price shock:
  Stage 1  Oil price → fiscal revenue        (linkage strength)
  Stage 2  Fiscal pressure → inflation       (subsidy / price pass-through)
  Stage 3  Inflation → employment pressure   (labour-market vulnerability)
  Stage 4  Employment / wages → household consumption
  Stage 5  Consumption contraction → GDP growth feedback

amplification_factor < 1.0 = SWF buffers or economic diversification dampen the
chain.  amplification_factor > 1.0 = institutional weakness, conflict, or embedded
inflation amplify it.

Stage scores and amplification factors are expert estimates calibrated to:
  - IMF Article IV Consultations 2023
  - IMF Regional Economic Outlook MENA, October 2023
  - Coady et al. (IMF, 2015) — energy-subsidy pass-through estimates
  - World Bank Development Indicators 2022

Integration with Addition 5
----------------------------
The output CSV (outputs/tables/chain_transmission.csv) is read by
src.model.right_now_risk.compute_chain_recent, which aggregates
transmission_severity over the most recent N years and normalises [0, 1]
to produce the chain component of the Right Now Risk composite score.

Usage (from project root)
--------------------------
    python -m src.model.chain_transmission
    python -m src.model.chain_transmission --ref  data/reference/chain_transmission.csv
    python -m src.model.chain_transmission --output outputs/tables/chain_transmission.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import math

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

REF_PATH:    Path = Path("data/reference/chain_transmission.csv")
OUTPUT_PATH: Path = Path("outputs/tables/chain_transmission.csv")

_SNAPSHOT_YEAR: int = 2024

_STAGE_COLS: list[str] = [
    "stage1_oil_fiscal",
    "stage2_fiscal_inflation",
    "stage3_inflation_employment",
    "stage4_employment_consumption",
    "stage5_consumption_growth",
]

_REQUIRED_REF_COLS: frozenset[str] = frozenset(
    _STAGE_COLS
    + [
        "country_code",
        "country_code_a3",
        "country_name",
        "transmission_speed",
        "amplification_factor",
    ]
)

# Oil / gas exporters: fuel exports > 20% of merchandise exports (consistent
# with src.model.chain_model exporter classifier).
_EXPORTERS: frozenset[str] = frozenset({
    "SAU", "IRQ", "KWT", "ARE", "QAT", "OMN", "BHR", "DZA", "LBY", "IRN",
})

_MIN_OBS: int = 5  # Minimum paired observations required for pooled OLS fit
_MIN_OBS_REGIME: int = 4  # Minimum per-regime observations for regime-switching OLS
BRENT_REGIME_THRESHOLD: float = 70.0  # USD/bbl threshold separating low and high price regimes
_PANEL_PATH: Path = Path("data/processed/world_bank_panel.csv")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def load_chain_reference(path: Path = REF_PATH) -> pd.DataFrame:
    """Load the chain transmission reference CSV.

    Args:
        path: Path to chain_transmission.csv in data/reference/.

    Returns:
        DataFrame with one row per country containing stage scores,
        amplification_factor, transmission_speed, and provenance columns.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If any required column is absent from the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Chain transmission reference not found: {path}\n"
            "Expected: data/reference/chain_transmission.csv"
        )

    df = pd.read_csv(path)

    missing = _REQUIRED_REF_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"chain_transmission.csv is missing required columns: {sorted(missing)}"
        )

    for col in _STAGE_COLS + ["amplification_factor"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info("Chain reference loaded: %d countries from %s", len(df), path)
    return df


def fit_transmission_ols(
    ref_df: pd.DataFrame,
    panel_path: Path = _PANEL_PATH,
) -> pd.DataFrame:
    """Add empirical OLS stage scores to the chain reference DataFrame.

    Proxy mapping (from world_bank_panel.csv → stage):
      stage1 → Δ NY_GDP_PETR_RT_ZS  (oil-rents-% YoY pp change)
      stage2 → FP_CPI_TOTL_ZG        (CPI inflation rate %)
      stage5 → NY_GDP_MKTP_CD pct_change × 100  (nominal GDP growth proxy)
      stage3, stage4 → no proxy available; expert estimates kept in build_chain_table

    OLS model:  y_t = α + β × brent_pct_change_{t−1} + ε
    Score:  min(1.0, |β|)
    Requires at least _MIN_OBS = 5 non-null paired observations.

    Brent history uses the deterministic EIA/WB fallback table (no network
    call) so results are reproducible without internet access.

    Args:
        ref_df:     Output of :func:`load_chain_reference`.
        panel_path: Path to data/processed/world_bank_panel.csv.

    Returns:
        Copy of *ref_df* with new columns:
          empirical_stage1..5  (float, NaN where data insufficient)
          empirical_flag       (bool, True if ≥ 1 stage was fitted)
          data_quality_flag    (str, semicolon-separated issue codes)
    """
    panel_path = Path(panel_path)
    df = ref_df.copy()

    for s in range(1, 6):
        df[f"empirical_stage{s}"] = float("nan")
    df["empirical_flag"] = False
    df["data_quality_flag"] = ""

    if not panel_path.exists():
        log.warning("World Bank panel not found at %s — OLS fitting skipped.", panel_path)
        df["data_quality_flag"] = "panel_not_found"
        return df

    panel = pd.read_csv(panel_path)

    # Build Brent annual pct-change series from the hard-coded EIA/WB fallback
    # (deterministic — no yfinance call required for OLS fitting).
    from src.data.brent import _BRENT_HISTORY_FALLBACK  # noqa: PLC0415
    brent_prices = pd.Series(_BRENT_HISTORY_FALLBACK, dtype=float).sort_index()
    brent_pct_chg = brent_prices.pct_change().mul(100)

    # lag1[t] = brent_pct_chg[t−1]: the price shock in year t−1 predicts y in t
    lag1 = pd.Series(
        {yr: brent_pct_chg.get(yr - 1, float("nan")) for yr in brent_pct_chg.index},
        name="brent_pct_change_lag1",
        dtype=float,
    )

    def _ols_score(x: pd.Series, y: pd.Series) -> float:
        combined = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(combined) < _MIN_OBS:
            return float("nan")
        A = np.column_stack([np.ones(len(combined)), combined["x"].values])
        coef, *_ = np.linalg.lstsq(A, combined["y"].values, rcond=None)
        return float(min(1.0, abs(coef[1])))

    # Proxy spec: stage_index → (panel_col, transform applied to raw series)
    _PROXY: dict[int, tuple[str, object]] = {
        1: ("NY_GDP_PETR_RT_ZS", lambda s: s.diff()),
        2: ("FP_CPI_TOTL_ZG",    lambda s: s),
        5: ("NY_GDP_MKTP_CD",    lambda s: s.pct_change().mul(100)),
    }

    dq: dict[str, list[str]] = {a3: [] for a3 in df["country_code_a3"]}

    for a3 in df["country_code_a3"]:
        cdf = panel[panel["country_code_a3"] == a3].copy()
        if cdf.empty:
            for s in range(1, 6):
                dq[a3].append(f"stage{s}_no_panel_row")
            continue

        cdf = cdf.set_index("year")

        for s_idx, (col, transform) in _PROXY.items():
            if col not in cdf.columns:
                dq[a3].append(f"stage{s_idx}_col_missing")
                continue
            y_raw = pd.to_numeric(cdf[col], errors="coerce")
            y = transform(y_raw).rename("y")
            score = _ols_score(lag1, y)
            if np.isnan(score):
                dq[a3].append(f"stage{s_idx}_insufficient_data")
            else:
                df.loc[df["country_code_a3"] == a3, f"empirical_stage{s_idx}"] = score

        dq[a3].extend(["stage3_insufficient_data", "stage4_insufficient_data"])

    emp_cols = [f"empirical_stage{s}" for s in range(1, 6)]
    df["empirical_flag"] = df[emp_cols].notna().any(axis=1)
    df["data_quality_flag"] = df["country_code_a3"].map(
        lambda a3: ";".join(dq.get(a3, []))
    )

    n_fitted = int(df["empirical_flag"].sum())
    log.info("OLS fitting: %d/%d countries have ≥1 empirical stage.", n_fitted, len(df))
    return df


def fit_transmission_ols_regime(
    ref_df: pd.DataFrame,
    panel_path: Path = _PANEL_PATH,
) -> pd.DataFrame:
    """Add regime-specific OLS stage scores to the chain reference DataFrame.

    Splits the WB panel into low-price (annual Brent < BRENT_REGIME_THRESHOLD)
    and high-price (annual Brent >= BRENT_REGIME_THRESHOLD) regimes and fits
    separate OLS for each regime/country/stage.  Where a regime has fewer than
    _MIN_OBS_REGIME = 4 observations the regime column is left NaN; callers
    should fall back to pooled OLS (empirical_stage{s}) or expert estimates.

    Designed to be called after :func:`fit_transmission_ols` so that the pooled
    columns are available as a fallback in :func:`build_chain_table`.

    OLS model and proxy mapping are identical to :func:`fit_transmission_ols`.
    The only difference is that observations are split by the Brent price regime
    of each year *t* before fitting.

    Args:
        ref_df:     Output of :func:`fit_transmission_ols` (or
                    :func:`load_chain_reference`).
        panel_path: Path to data/processed/world_bank_panel.csv.

    Returns:
        Copy of *ref_df* with new columns:
          empirical_stage1_low .. empirical_stage5_low  (float, NaN where insufficient)
          empirical_stage1_high .. empirical_stage5_high (float, NaN where insufficient)
          regime_flag (str, semicolon-separated issue codes; empty string if none)
    """
    panel_path = Path(panel_path)
    df = ref_df.copy()

    for s in range(1, 6):
        df[f"empirical_stage{s}_low"]  = float("nan")
        df[f"empirical_stage{s}_high"] = float("nan")
    df["regime_flag"] = ""

    if not panel_path.exists():
        log.warning("Panel not found at %s — regime OLS skipped.", panel_path)
        df["regime_flag"] = "panel_not_found"
        return df

    panel = pd.read_csv(panel_path)

    from src.data.brent import _BRENT_HISTORY_FALLBACK  # noqa: PLC0415
    brent_prices = pd.Series(_BRENT_HISTORY_FALLBACK, dtype=float).sort_index()
    brent_pct_chg = brent_prices.pct_change().mul(100)

    lag1 = pd.Series(
        {yr: brent_pct_chg.get(yr - 1, float("nan")) for yr in brent_pct_chg.index},
        name="brent_pct_change_lag1",
        dtype=float,
    )

    low_years  = frozenset(yr for yr, p in _BRENT_HISTORY_FALLBACK.items() if p <  BRENT_REGIME_THRESHOLD)
    high_years = frozenset(yr for yr, p in _BRENT_HISTORY_FALLBACK.items() if p >= BRENT_REGIME_THRESHOLD)

    def _ols_score_subset(x: pd.Series, y: pd.Series, year_set: frozenset) -> float:
        combined = pd.DataFrame({"x": x, "y": y}).dropna()
        subset   = combined[combined.index.isin(year_set)]
        if len(subset) < _MIN_OBS_REGIME:
            return float("nan")
        A = np.column_stack([np.ones(len(subset)), subset["x"].values])
        coef, *_ = np.linalg.lstsq(A, subset["y"].values, rcond=None)
        return float(min(1.0, abs(coef[1])))

    _PROXY: dict[int, tuple[str, object]] = {
        1: ("NY_GDP_PETR_RT_ZS", lambda s: s.diff()),
        2: ("FP_CPI_TOTL_ZG",    lambda s: s),
        5: ("NY_GDP_MKTP_CD",    lambda s: s.pct_change().mul(100)),
    }

    for a3 in df["country_code_a3"]:
        cdf = panel[panel["country_code_a3"] == a3].copy()
        if cdf.empty:
            df.loc[df["country_code_a3"] == a3, "regime_flag"] = "no_panel_row"
            continue

        cdf = cdf.set_index("year")
        issues: list[str] = []

        for s_idx, (col, transform) in _PROXY.items():
            if col not in cdf.columns:
                issues.append(f"stage{s_idx}_col_missing")
                continue

            y_raw = pd.to_numeric(cdf[col], errors="coerce")
            y     = transform(y_raw).rename("y")

            score_low  = _ols_score_subset(lag1, y, low_years)
            score_high = _ols_score_subset(lag1, y, high_years)

            mask = df["country_code_a3"] == a3
            if not np.isnan(score_low):
                df.loc[mask, f"empirical_stage{s_idx}_low"]  = score_low
            else:
                issues.append(f"stage{s_idx}_insufficient_data_for_regime_low")

            if not np.isnan(score_high):
                df.loc[mask, f"empirical_stage{s_idx}_high"] = score_high
            else:
                issues.append(f"stage{s_idx}_insufficient_data_for_regime_high")

        if issues:
            df.loc[df["country_code_a3"] == a3, "regime_flag"] = ";".join(issues)

    n_low  = int(df[[f"empirical_stage{s}_low"  for s in (1, 2, 5)]].notna().any(axis=1).sum())
    n_high = int(df[[f"empirical_stage{s}_high" for s in (1, 2, 5)]].notna().any(axis=1).sum())
    log.info(
        "Regime OLS: %d/%d countries with low-regime scores, %d/%d with high-regime.",
        n_low, len(df), n_high, len(df),
    )
    return df


def compute_chain_severity(ref_df: pd.DataFrame) -> pd.DataFrame:
    """Compute chain_transmission_severity = mean(stage1..5) * amplification.

    The result is clamped to [0, 1].

    Args:
        ref_df: Output of :func:`load_chain_reference`.

    Returns:
        Copy of *ref_df* with new columns ``stage_mean`` and
        ``chain_transmission_severity`` in [0, 1].
    """
    df = ref_df.copy()
    df["stage_mean"] = df[_STAGE_COLS].mean(axis=1)
    amp = pd.to_numeric(df["amplification_factor"], errors="coerce")
    df["chain_transmission_severity"] = (df["stage_mean"] * amp).clip(0.0, 1.0)

    lo = df["chain_transmission_severity"].min()
    hi = df["chain_transmission_severity"].max()
    log.info(
        "Chain severity computed: %d countries  range [%.4f, %.4f]",
        len(df), lo, hi,
    )
    return df


def build_chain_table(
    ref_df: pd.DataFrame,
    brent_price: float | None = None,
) -> pd.DataFrame:
    """Build the full chain transmission output table.

    Adds severity scores, backward-compatibility aliases, and the
    ``is_exporter``, ``year``, and ``regime_used`` columns expected by
    downstream consumers.

    When regime-specific empirical columns are present (from
    :func:`fit_transmission_ols_regime`), the column appropriate to the
    current Brent price regime is selected.  Falls back to pooled OLS
    (empirical_stage{s}) when regime-specific columns are NaN, then to the
    expert estimate from the reference CSV.

    Additional columns added:
      - ``chain_transmission_severity``       composite [0,1] score
      - ``chain_transmission_severity_recent`` same value (static snapshot)
      - ``transmission_severity``             alias — required by right_now_risk.py
      - ``is_exporter``                       True for oil/gas net exporters
      - ``year``                              static snapshot year (2024)
      - ``regime_used``                       which coefficient set was applied
                                              ('low', 'high', 'pooled', 'expert')

    Args:
        ref_df:      Output of :func:`load_chain_reference` (optionally enriched
                     with empirical columns from OLS fitting functions).
        brent_price: Live Brent price (USD/bbl) used to select the active regime.
                     If None or NaN, the most recent year in
                     _BRENT_HISTORY_FALLBACK is used (deterministic fallback).

    Returns:
        DataFrame with one row per country, sorted by
        ``chain_transmission_severity`` descending.
    """
    from src.data.brent import _BRENT_HISTORY_FALLBACK  # noqa: PLC0415

    if brent_price is None or math.isnan(float(brent_price)):
        brent_price = float(_BRENT_HISTORY_FALLBACK[max(_BRENT_HISTORY_FALLBACK.keys())])

    regime = "high" if float(brent_price) >= BRENT_REGIME_THRESHOLD else "low"
    log.info("build_chain_table: Brent=%.2f → regime='%s'", brent_price, regime)

    blended = ref_df.copy()

    # Determine regime_used per country *before* blending so we record original sources.
    _emp_check = (1, 2, 5)
    regime_used: list[str] = []
    for _, row in blended.iterrows():
        has_regime = any(
            f"empirical_stage{s}_{regime}" in blended.columns
            and not pd.isna(row.get(f"empirical_stage{s}_{regime}"))
            for s in _emp_check
        )
        has_pooled = (
            not has_regime
            and any(
                f"empirical_stage{s}" in blended.columns
                and not pd.isna(row.get(f"empirical_stage{s}"))
                for s in _emp_check
            )
        )
        regime_used.append(
            regime if has_regime else ("pooled" if has_pooled else "expert")
        )
    blended["regime_used"] = regime_used

    # Blend stage columns: regime-specific empirical → pooled empirical → expert.
    for i, stage_col in enumerate(_STAGE_COLS, start=1):
        regime_col = f"empirical_stage{i}_{regime}"
        pooled_col = f"empirical_stage{i}"

        if regime_col in blended.columns:
            pooled = (
                blended[pooled_col]
                if pooled_col in blended.columns
                else pd.Series(float("nan"), index=blended.index)
            )
            blended[stage_col] = (
                blended[regime_col]
                .combine_first(pooled)
                .combine_first(blended[stage_col])
                .clip(0.0, 1.0)
            )
        elif pooled_col in blended.columns:
            blended[stage_col] = (
                blended[pooled_col].combine_first(blended[stage_col]).clip(0.0, 1.0)
            )

    df = compute_chain_severity(blended)

    df["is_exporter"] = df["country_code_a3"].isin(_EXPORTERS)
    df["year"]        = _SNAPSHOT_YEAR

    # Aliases for backward compatibility with right_now_risk.py and backtest.py
    df["transmission_severity"]              = df["chain_transmission_severity"]
    df["chain_transmission_severity_recent"] = df["chain_transmission_severity"]

    df = df.sort_values("chain_transmission_severity", ascending=False).reset_index(drop=True)

    _validate_chain_table(df)
    return df


def save_chain_table(df: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> Path:
    """Write the chain transmission output table to CSV.

    Column order is deterministic: identity → year → classification →
    stages → derived severity → speed/confidence.

    Args:
        df:          Output of :func:`build_chain_table`.
        output_path: Destination path; parent directory is created if absent.

    Returns:
        Resolved absolute path of the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preferred_order = [
        "country_code", "country_code_a3", "country_name",
        "year", "is_exporter",
        "stage1_oil_fiscal", "stage2_fiscal_inflation",
        "stage3_inflation_employment", "stage4_employment_consumption",
        "stage5_consumption_growth",
        "stage_mean", "amplification_factor",
        "chain_transmission_severity",
        "chain_transmission_severity_recent",
        "transmission_severity",
        "regime_used",
        "transmission_speed",
        "confidence", "is_estimate",
    ]
    col_order = [c for c in preferred_order if c in df.columns]
    # Append any remaining columns not in the preferred list
    col_order += [c for c in df.columns if c not in col_order]

    df[col_order].to_csv(output_path, index=False)
    log.info("Chain table saved: %d rows → %s", len(df), output_path)
    return output_path.resolve()


def run_chain_transmission(
    ref_path:    Path = REF_PATH,
    output_path: Path = OUTPUT_PATH,
    fit_ols:     bool = False,
    panel_path:  Path = _PANEL_PATH,
    brent_price: float | None = None,
) -> pd.DataFrame:
    """End-to-end chain transmission pipeline: load → (OLS fit) → compute → save.

    Args:
        ref_path:    Path to the reference CSV (data/reference/).
        output_path: Destination path for the output CSV (outputs/tables/).
        fit_ols:     If True, run pooled OLS and regime-switching OLS on
                     world_bank_panel.csv and write empirical columns back to
                     *ref_path* before building the output table.
        panel_path:  Path to the World Bank panel CSV used by OLS fitting.
        brent_price: Live Brent price (USD/bbl) passed to :func:`build_chain_table`
                     for regime selection.  Defaults to the most recent year in
                     _BRENT_HISTORY_FALLBACK when None or NaN.

    Returns:
        Computed chain transmission DataFrame (also written to *output_path*).

    Raises:
        FileNotFoundError: If *ref_path* does not exist.
        ValueError: If required columns are absent.
    """
    ref_df = load_chain_reference(ref_path)

    if fit_ols:
        ref_df = fit_transmission_ols(ref_df, panel_path)
        ref_df = fit_transmission_ols_regime(ref_df, panel_path)
        _save_empirical_to_ref(ref_df, ref_path)

    chain_df = build_chain_table(ref_df, brent_price=brent_price)
    save_chain_table(chain_df, output_path)
    _print_summary(chain_df)
    return chain_df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_empirical_to_ref(df: pd.DataFrame, ref_path: Path) -> None:
    """Merge empirical OLS columns back into the reference CSV in place."""
    emp_cols = [
        c for c in df.columns
        if c.startswith("empirical_") or c in ("data_quality_flag", "regime_flag")
    ]
    if not emp_cols:
        log.info("No empirical columns to save.")
        return

    ref_path = Path(ref_path)
    existing = pd.read_csv(ref_path)
    # Drop stale empirical columns so we get a clean merge
    existing = existing.drop(
        columns=[c for c in emp_cols if c in existing.columns], errors="ignore"
    )
    merged = existing.merge(
        df[["country_code_a3"] + emp_cols], on="country_code_a3", how="left"
    )
    merged.to_csv(ref_path, index=False)
    log.info("Empirical OLS columns saved to %s", ref_path)


def _validate_chain_table(df: pd.DataFrame) -> None:
    """Post-build guardrail checks. Logs warnings; never raises."""
    sev = df["chain_transmission_severity"].dropna()
    oob = sev[(sev < 0.0) | (sev > 1.0)]
    if not oob.empty:
        log.warning(
            "GUARDRAIL FAIL: %d severity value(s) outside [0,1]: %s",
            len(oob), oob.tolist(),
        )

    if len(df) != 14:
        log.warning("GUARDRAIL FAIL: expected 14 countries, got %d", len(df))

    dups = df["country_code_a3"][df["country_code_a3"].duplicated()]
    if not dups.empty:
        log.warning("GUARDRAIL FAIL: duplicate country_code_a3: %s", dups.tolist())

    if "chain_transmission_severity_recent" in df.columns:
        mismatch = (
            (df["chain_transmission_severity_recent"] - df["chain_transmission_severity"])
            .abs()
            .gt(1e-9)
            .sum()
        )
        if mismatch:
            log.warning(
                "GUARDRAIL FAIL: %d row(s) where recent != severity "
                "(should be 0 for static snapshot data)",
                mismatch,
            )

    if "regime_used" not in df.columns:
        log.warning("GUARDRAIL FAIL: regime_used column missing from chain table")
    else:
        _valid_regimes = {"low", "high", "pooled", "expert"}
        bad = df["regime_used"][~df["regime_used"].isin(_valid_regimes)].tolist()
        if bad:
            log.warning("GUARDRAIL FAIL: invalid regime_used values: %s", bad)

    log.info("Chain validation complete: %d countries", len(df))


def _print_summary(df: pd.DataFrame) -> None:
    """Print a concise per-country summary to stdout."""
    header = (
        f"\n{'Country':<26} {'A3':<5} {'Speed':<8} "
        f"{'StageMean':>10} {'Amplif':>7} {'Severity':>9}"
    )
    try:
        print(header)
        print("-" * 70)
        for _, row in df.iterrows():
            print(
                f"{row['country_name']:<26} {row['country_code_a3']:<5} "
                f"{row['transmission_speed']:<8} "
                f"{row['stage_mean']:10.4f} {row['amplification_factor']:7.2f} "
                f"{row['chain_transmission_severity']:9.4f}"
            )
        mean_sev = df["chain_transmission_severity"].mean()
        print(f"\n{'14-country mean':>57} {mean_sev:9.4f}")
    except UnicodeEncodeError:
        # Fallback for terminals without full Unicode support (e.g. Windows cp1252)
        log.info(
            "Chain summary: %d countries, severity range [%.4f, %.4f], mean %.4f",
            len(df),
            df["chain_transmission_severity"].min(),
            df["chain_transmission_severity"].max(),
            df["chain_transmission_severity"].mean(),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.model.chain_transmission",
        description=(
            "Addition 4 — Chain Transmission Severity Model.\n"
            "Reads structural stage parameters from a reference CSV and computes\n"
            "chain_transmission_severity = mean(stage1..5) × amplification_factor,\n"
            "clamped to [0, 1]."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--ref",
        metavar="PATH",
        default=str(REF_PATH),
        help=f"Path to the reference CSV (default: {REF_PATH}).",
    )
    p.add_argument(
        "--output",
        metavar="PATH",
        default=str(OUTPUT_PATH),
        help=f"Destination CSV (default: {OUTPUT_PATH}).",
    )
    p.add_argument(
        "--fit-ols",
        action="store_true",
        default=False,
        help=(
            "Fit OLS coefficients from the World Bank panel and save empirical stage "
            "scores to the reference CSV before building the output table."
        ),
    )
    p.add_argument(
        "--panel",
        metavar="PATH",
        default=str(_PANEL_PATH),
        help=f"Path to world_bank_panel.csv for OLS fitting (default: {_PANEL_PATH}).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run the chain transmission pipeline."""
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        run_chain_transmission(
            ref_path=Path(args.ref),
            output_path=Path(args.output),
            fit_ols=args.fit_ols,
            panel_path=Path(args.panel),
        )
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
