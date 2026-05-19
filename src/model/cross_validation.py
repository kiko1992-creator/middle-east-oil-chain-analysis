"""
IMF / World Bank Cross-Validation — Priority 2.

Compares the model's Right Now Risk composite scores against independent
benchmark risk tiers published by the IMF (Fiscal Monitor Oct 2023) and
the World Bank (Macro Poverty Outlook Fall 2023).

Methodology
-----------
1. Load benchmarks from data/reference/imf_wb_benchmarks.csv.
2. Load model scores from the Right Now Risk pipeline.
3. Map model composite scores to three risk tiers using data-driven tertile
   thresholds (so Low / Medium / High contain roughly equal numbers of
   countries — no hardcoded cut-offs).
4. Compute Spearman rank correlations between model ranks and the ordinal
   IMF / WB tier rankings.
5. Flag divergences: countries where model tier differs from IMF or WB tier.
6. Export to outputs/tables/cross_validation.csv.

Key functions
-------------
load_benchmarks          Load imf_wb_benchmarks.csv
map_model_to_tiers       Assign model tier from composite scores (tertile split)
compute_correlations     Spearman ρ between model and IMF / WB ordinals
identify_divergences     Countries where model ≠ benchmark tier
run_cross_validation     Orchestrate all steps and return merged DataFrame
export_cross_validation  Write to CSV

Usage (from project root)
--------------------------
    python -m src.model.cross_validation
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parents[2]

BENCHMARKS_PATH  = _ROOT / "data" / "reference" / "imf_wb_benchmarks.csv"
BREAKEVEN_PATH   = _ROOT / "data" / "reference" / "fiscal_breakeven.csv"
RESERVES_PATH    = _ROOT / "data" / "reference" / "swf_reserves.csv"
FOOD_PATH        = _ROOT / "data" / "reference" / "food_security.csv"
CHAIN_PATH       = _ROOT / "outputs" / "tables" / "chain_transmission.csv"
PANEL_PATH       = _ROOT / "data" / "processed" / "world_bank_panel.csv"
OUTPUT_PATH      = _ROOT / "outputs" / "tables" / "cross_validation.csv"

# Ordinal mappings for benchmark tiers (higher ordinal = higher risk)
_IMF_FM_ORDINAL: dict[str, int] = {"Low": 1, "Medium": 2, "High": 3}
_WB_MPO_ORDINAL: dict[str, int] = {"Stable": 1, "Watch": 2, "Stressed": 3}
_MODEL_TIER_LABELS: list[str]   = ["Low", "Medium", "High"]


# ── Correlation helpers (no scipy) ────────────────────────────────────────────

def _spearman_r(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation without scipy."""
    rx = pd.Series(x, dtype=float).rank()
    ry = pd.Series(y, dtype=float).rank()
    return float(np.corrcoef(rx.values, ry.values)[0, 1])


def _spearman_pvalue(r: float, n: int) -> float:
    """Two-tailed p-value via Fisher z-transform approximation (n ≥ 5)."""
    if n < 3 or abs(r) >= 1.0:
        return float("nan")
    z = math.atanh(r) * math.sqrt(max(n - 3, 1))
    return float(math.erfc(abs(z) / math.sqrt(2)))


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_benchmarks(path: Path = BENCHMARKS_PATH) -> pd.DataFrame:
    """Load the IMF / WB benchmark tiers reference CSV.

    Args:
        path: Path to imf_wb_benchmarks.csv.

    Returns:
        DataFrame with one row per country, imf_fm_risk_tier and wb_mpo_status
        columns, plus source and confidence metadata.

    Raises:
        FileNotFoundError: If the CSV does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmarks file not found: {path}")
    df = pd.read_csv(path)
    df["imf_fm_ordinal"] = df["imf_fm_risk_tier"].map(_IMF_FM_ORDINAL)
    df["wb_mpo_ordinal"]  = df["wb_mpo_status"].map(_WB_MPO_ORDINAL)
    log.info("Loaded %d benchmark rows from %s", len(df), path)
    return df


# ── Tier mapping ──────────────────────────────────────────────────────────────

def map_model_to_tiers(model_df: pd.DataFrame) -> pd.DataFrame:
    """Assign Low / Medium / High tier to each country from composite scores.

    Uses data-driven tertile thresholds (33rd and 67th percentiles of
    right_now_risk_score) so the tier boundaries adapt to the current
    score distribution rather than relying on hardcoded cut-offs.

    Args:
        model_df: DataFrame with country_code_a3 and right_now_risk_score.

    Returns:
        Input DataFrame with two added columns:
            model_tier        (str)  — "Low", "Medium", or "High"
            model_tier_ordinal (int) — 1, 2, or 3
            model_rank         (int) — rank by right_now_risk_score (1 = highest risk)

    Raises:
        ValueError: If right_now_risk_score column is absent.
    """
    if "right_now_risk_score" not in model_df.columns:
        raise ValueError("model_df must contain 'right_now_risk_score'.")

    df = model_df.copy()
    scores = df["right_now_risk_score"].dropna()

    low_thresh  = float(np.percentile(scores, 100 / 3))
    high_thresh = float(np.percentile(scores, 200 / 3))

    def _tier(s: float) -> str:
        if pd.isna(s):
            return ""
        if s <= low_thresh:
            return "Low"
        if s <= high_thresh:
            return "Medium"
        return "High"

    df["model_tier"]         = df["right_now_risk_score"].apply(_tier)
    df["model_tier_ordinal"] = df["model_tier"].map(
        {"Low": 1, "Medium": 2, "High": 3}
    ).fillna(0).astype(int)
    df["model_rank"] = (
        df["right_now_risk_score"]
        .rank(ascending=False, method="min", na_option="bottom")
        .astype(int)
    )
    return df


# ── Correlation computation ───────────────────────────────────────────────────

def compute_correlations(cv_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute Spearman correlations between model rank and benchmark tiers.

    Args:
        cv_df: Merged cross-validation DataFrame with model_rank,
               imf_fm_ordinal, and wb_mpo_ordinal columns.

    Returns:
        Nested dict:
            {
              "vs_imf_fm":  {"spearman_r": float, "spearman_p": float, "n": int},
              "vs_wb_mpo":  {"spearman_r": float, "spearman_p": float, "n": int},
            }
    """
    results: dict[str, dict[str, float]] = {}

    pairs = [
        ("vs_imf_fm", "model_rank", "imf_fm_ordinal"),
        ("vs_wb_mpo", "model_rank", "wb_mpo_ordinal"),
    ]
    for label, col_a, col_b in pairs:
        valid = cv_df.dropna(subset=[col_a, col_b])
        n = len(valid)
        if n < 3:
            results[label] = {"spearman_r": float("nan"), "spearman_p": float("nan"), "n": n}
            continue
        r = _spearman_r(valid[col_a].tolist(), valid[col_b].tolist())
        p = _spearman_pvalue(r, n)
        results[label] = {"spearman_r": round(r, 4), "spearman_p": round(p, 4), "n": n}

    return results


# ── Divergence identification ─────────────────────────────────────────────────

def identify_divergences(cv_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where the model tier diverges from IMF FM or WB MPO.

    A divergence is any country where model_tier ≠ imf_fm_risk_tier or
    model_tier ≠ wb_mpo_status.  The ordinal distance is also recorded.

    Args:
        cv_df: Merged cross-validation DataFrame.

    Returns:
        Subset of cv_df flagged for divergence, with additional columns:
            imf_divergence (bool), wb_divergence (bool),
            imf_ordinal_distance (int), wb_ordinal_distance (int).
    """
    df = cv_df.copy()

    if "model_tier" in df.columns and "imf_fm_risk_tier" in df.columns:
        df["imf_divergence"] = df["model_tier"] != df["imf_fm_risk_tier"]
        df["imf_ordinal_distance"] = (
            df["model_tier_ordinal"] - df["imf_fm_ordinal"]
        ).abs().fillna(0).astype(int)
    else:
        df["imf_divergence"] = False
        df["imf_ordinal_distance"] = 0

    if "model_tier" in df.columns and "wb_mpo_status" in df.columns:
        df["wb_divergence"] = df["model_tier"] != df["wb_mpo_status"]
        df["wb_ordinal_distance"] = (
            df["model_tier_ordinal"] - df["wb_mpo_ordinal"]
        ).abs().fillna(0).astype(int)
    else:
        df["wb_divergence"] = False
        df["wb_ordinal_distance"] = 0

    df["any_divergence"] = df["imf_divergence"] | df["wb_divergence"]
    return df


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_cross_validation(
    benchmarks_path: Path = BENCHMARKS_PATH,
    breakeven_path:  Path = BREAKEVEN_PATH,
    reserves_path:   Path = RESERVES_PATH,
    food_path:       Path = FOOD_PATH,
    chain_path:      Path = CHAIN_PATH,
    panel_path:      Path = PANEL_PATH,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Run the full cross-validation pipeline.

    Loads benchmarks, runs the Right Now Risk pipeline for current model
    scores, merges, assigns tiers, computes correlations, and flags
    divergences.

    Args:
        benchmarks_path: Path to imf_wb_benchmarks.csv.
        breakeven_path:  Path to fiscal_breakeven.csv.
        reserves_path:   Path to swf_reserves.csv.
        food_path:       Path to food_security.csv.
        chain_path:      Path to chain_transmission.csv (outputs).
        panel_path:      Path to world_bank_panel.csv.

    Returns:
        Tuple (cv_df, correlations) where:
            cv_df        — merged per-country DataFrame with all columns
            correlations — dict from :func:`compute_correlations`

    Raises:
        FileNotFoundError: If benchmarks file is missing.
    """
    from src.model.right_now_risk import run_right_now_risk

    benchmarks = load_benchmarks(benchmarks_path)

    rnr = run_right_now_risk(
        breakeven_path=breakeven_path,
        reserves_path=reserves_path,
        food_path=food_path,
        chain_path=chain_path,
        panel_path=panel_path,
        export_csv=None,
    )
    model_df = rnr["right_now_risk_df"]

    model_df = map_model_to_tiers(model_df)

    keep_model = [
        "country_code_a3", "country_label",
        "right_now_risk_score", "model_rank", "model_tier", "model_tier_ordinal",
        "fiscal_stress_score", "reserve_runway_risk",
        "social_stability_risk", "chain_transmission_severity_recent",
    ]
    keep_model = [c for c in keep_model if c in model_df.columns]

    keep_bench = [
        "country_code_a3",
        "imf_fm_risk_tier", "imf_fm_ordinal",
        "wb_mpo_status",    "wb_mpo_ordinal",
        "confidence",
    ]
    keep_bench = [c for c in keep_bench if c in benchmarks.columns]

    cv_df = model_df[keep_model].merge(benchmarks[keep_bench],
                                       on="country_code_a3", how="left")
    cv_df = identify_divergences(cv_df)

    correlations = compute_correlations(cv_df)

    n_div = int(cv_df["any_divergence"].sum())
    for label, corr in correlations.items():
        log.info(
            "Correlation %s: spearman_r=%.3f  p=%.4f  n=%d",
            label, corr["spearman_r"], corr["spearman_p"], corr["n"],
        )
    log.info("Divergences: %d / %d countries", n_div, len(cv_df))

    return cv_df, correlations


# ── Export ─────────────────────────────────────────────────────────────────────

def export_cross_validation(
    df: pd.DataFrame,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    """Write cross-validation results to CSV.

    Args:
        df:          Output of :func:`run_cross_validation`.
        output_path: Destination path (parent created if absent).

    Returns:
        Resolved absolute path of the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("Cross-validation exported: %d rows → %s", len(df), output_path)
    return output_path.resolve()


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.model.cross_validation",
        description=(
            "Priority 2 — IMF / WB Cross-Validation.\n"
            "Compares model Right Now Risk tiers against IMF FM and WB MPO benchmarks."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output", metavar="PATH", default=str(OUTPUT_PATH),
                   help=f"Destination CSV (default: {OUTPUT_PATH}).")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity (default: INFO).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        cv_df, correlations = run_cross_validation()
    except (FileNotFoundError, ValueError) as exc:
        log.error("Cross-validation failed: %s", exc)
        return 1

    export_cross_validation(cv_df, Path(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
