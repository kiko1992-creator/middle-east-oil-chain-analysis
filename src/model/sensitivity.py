"""
Sensitivity Analysis — Sprint 5.

Evaluates how Right Now Risk rankings change as component weights shift ±0.10
from their defaults, one component at a time (one-at-a-time / OAT design).

Default weights (fiscal=0.35, runway=0.25, social=0.20, chain=0.20):
  When one weight is varied, the remaining three are rescaled proportionally
  so that all four weights sum to 1.0.

  Formula for renormalization:
      remaining_new[k] = remaining_base[k] / sum(remaining_base) × (1 - new_weight)

Key functions
-------------
build_weight_grid        Generate the OAT scenario list
run_sensitivity          Compute Right Now Risk for every scenario
summarize_sensitivity    Per-country rank_volatility = std(rank across scenarios)
export_sensitivity       Write results to outputs/tables/sensitivity_results.csv

Usage (from project root)
--------------------------
    python -m src.model.sensitivity
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Default weights ────────────────────────────────────────────────────────────

_COMP_COLS: list[str] = [
    "fiscal_stress_score",
    "reserve_runway_risk",
    "social_stability_risk",
    "chain_transmission_severity_recent",
]

_BASE_WEIGHTS: dict[str, float] = {
    "fiscal_stress_score":                0.35,
    "reserve_runway_risk":                0.25,
    "social_stability_risk":              0.20,
    "chain_transmission_severity_recent": 0.20,
}

_SHORT_NAME: dict[str, str] = {
    "fiscal_stress_score":                "fiscal_w",
    "reserve_runway_risk":                "runway_w",
    "social_stability_risk":              "social_w",
    "chain_transmission_severity_recent": "chain_w",
}

_OAT_STEP  = 0.05
_OAT_DELTA = 0.10  # vary each weight ±0.10

_OUTPUT_PATH = Path("outputs/tables/sensitivity_results.csv")


# ── Weight grid builder ────────────────────────────────────────────────────────

def build_weight_grid(
    base_weights: dict[str, float] = _BASE_WEIGHTS,
    step: float = _OAT_STEP,
    delta: float = _OAT_DELTA,
) -> list[dict[str, float]]:
    """Generate a one-at-a-time (OAT) sensitivity weight grid.

    For each component, the weight is varied from (base - delta) to
    (base + delta) in increments of *step*, clamped to [0.01, 0.99].
    All other component weights are proportionally rescaled so that
    all four weights sum to 1.0.

    Args:
        base_weights: Default component weights dict.
        step:         Step size for weight variation (default 0.05).
        delta:        Max deviation from base (default 0.10).

    Returns:
        List of scenario dicts, each with keys in _SHORT_NAME values
        (fiscal_w, runway_w, social_w, chain_w) plus 'scenario_id'
        and 'varied_component'.  The base scenario is included once.
    """
    scenarios: list[dict[str, float]] = []

    # Base scenario
    base = {_SHORT_NAME[k]: v for k, v in base_weights.items()}
    base["scenario_id"]       = "base"
    base["varied_component"]  = "base"
    scenarios.append(base)

    comps = list(base_weights.keys())

    for varied_comp in comps:
        base_v = base_weights[varied_comp]
        others = {k: v for k, v in base_weights.items() if k != varied_comp}
        others_total = sum(others.values())

        levels = np.arange(
            max(0.01, base_v - delta),
            min(0.99, base_v + delta) + step / 2,
            step,
        )

        for level in levels:
            level = round(float(level), 4)
            if abs(level - base_v) < 1e-9:
                continue  # skip base level (already added)

            remaining = 1.0 - level
            # Renormalize other weights proportionally, then force exact sum
            w_new = {k: v / others_total * remaining for k, v in others.items()}
            w_new[varied_comp] = level
            total = sum(w_new.values())
            w_new = {k: v / total for k, v in w_new.items()}  # exact normalisation

            row = {_SHORT_NAME[k]: v for k, v in w_new.items()}
            row["scenario_id"]      = f"{_SHORT_NAME[varied_comp]}_{level:.2f}"
            row["varied_component"] = _SHORT_NAME[varied_comp]
            scenarios.append(row)

    log.info("Weight grid: %d scenarios (%d components × OAT ±%.2f step %.2f + base)",
             len(scenarios), len(comps), delta, step)
    return scenarios


# ── Per-scenario scorer ────────────────────────────────────────────────────────

def _score_row(
    row: pd.Series,
    weights: dict[str, float],
) -> float:
    """Compute weighted composite score for one country row, with NaN rescaling."""
    available: dict[str, tuple[float, float]] = {}
    for col, w in weights.items():
        val = row.get(col, float("nan"))
        if not pd.isna(val):
            available[col] = (float(val), w)

    if not available:
        return float("nan")

    total_w = sum(w for _, w in available.values())
    return float(
        min(1.0, max(0.0, sum(v * w / total_w for v, w in available.values())))
    )


# ── Main sensitivity function ──────────────────────────────────────────────────

def run_sensitivity(
    base_scores_df: pd.DataFrame,
    weight_grid: list[dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Compute Right Now Risk scores for every scenario in the weight grid.

    Args:
        base_scores_df: DataFrame with one row per country containing the
                        raw component scores (fiscal_stress_score,
                        reserve_runway_risk, social_stability_risk,
                        chain_transmission_severity_recent) and
                        country_code_a3 / country_label identity columns.
        weight_grid:    List of scenario dicts from :func:`build_weight_grid`.
                        Defaults to the OAT grid generated by that function.

    Returns:
        Long-format DataFrame with one row per (scenario, country).
        Columns: scenario_id, varied_component, fiscal_w, runway_w,
        social_w, chain_w, country_code_a3, country_label,
        right_now_risk_score, rank.
    """
    if weight_grid is None:
        weight_grid = build_weight_grid()

    required = {"country_code_a3"} | set(_COMP_COLS)
    missing  = required - set(base_scores_df.columns)
    if missing:
        raise ValueError(f"base_scores_df is missing required columns: {sorted(missing)}")

    frames: list[pd.DataFrame] = []

    for scenario in weight_grid:
        # Build component-col → weight mapping
        comp_weights = {
            col: scenario[_SHORT_NAME[col]]
            for col in _COMP_COLS
        }

        df = base_scores_df.copy()
        df["right_now_risk_score"] = df.apply(
            _score_row, axis=1, weights=comp_weights
        )
        df["rank"] = (
            df["right_now_risk_score"]
            .rank(ascending=False, method="min", na_option="bottom")
            .astype(int)
        )

        # Metadata columns
        df["scenario_id"]      = scenario["scenario_id"]
        df["varied_component"] = scenario["varied_component"]
        df["fiscal_w"]  = scenario["fiscal_w"]
        df["runway_w"]  = scenario["runway_w"]
        df["social_w"]  = scenario["social_w"]
        df["chain_w"]   = scenario["chain_w"]

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Column ordering
    id_cols    = ["scenario_id", "varied_component",
                  "fiscal_w", "runway_w", "social_w", "chain_w"]
    cntry_cols = ["country_code_a3", "country_label"]
    score_cols = ["right_now_risk_score", "rank"]
    extra      = [c for c in combined.columns
                  if c not in id_cols + cntry_cols + score_cols]
    out = combined[id_cols + cntry_cols + score_cols + extra].copy()

    n_scenarios = combined["scenario_id"].nunique()
    log.info(
        "Sensitivity: %d scenarios × %d countries = %d rows",
        n_scenarios, len(base_scores_df), len(out),
    )
    return out


# ── Rank volatility summary ────────────────────────────────────────────────────

def summarize_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-country rank volatility across all scenarios.

    rank_volatility = std(rank) across all scenario_ids for that country.

    Args:
        df: Output of :func:`run_sensitivity`.

    Returns:
        DataFrame with one row per country, sorted by rank_volatility
        descending (most sensitive first).  Columns:
            country_code_a3, country_label (if present),
            rank_volatility, rank_mean, rank_min, rank_max, n_scenarios,
            score_mean, score_std.
    """
    if df.empty or "rank" not in df.columns:
        return pd.DataFrame()

    grp_cols = ["country_code_a3"]
    if "country_label" in df.columns:
        grp_cols.append("country_label")

    summary = (
        df.groupby(grp_cols)
        .agg(
            rank_volatility = ("rank",                   "std"),
            rank_mean       = ("rank",                   "mean"),
            rank_min        = ("rank",                   "min"),
            rank_max        = ("rank",                   "max"),
            n_scenarios     = ("scenario_id",            "nunique"),
            score_mean      = ("right_now_risk_score",   "mean"),
            score_std       = ("right_now_risk_score",   "std"),
        )
        .reset_index()
    )
    summary["rank_volatility"] = summary["rank_volatility"].round(3)
    summary["rank_mean"]       = summary["rank_mean"].round(1)
    summary["score_mean"]      = summary["score_mean"].round(4)
    summary["score_std"]       = summary["score_std"].round(4)

    summary = summary.sort_values("rank_volatility", ascending=False).reset_index(drop=True)
    return summary


# ── Export ─────────────────────────────────────────────────────────────────────

def export_sensitivity(
    df: pd.DataFrame,
    output_path: Path = _OUTPUT_PATH,
) -> Path:
    """Write sensitivity results to a CSV file.

    Args:
        df:          Output of :func:`run_sensitivity`.
        output_path: Destination path (parent directory created if absent).

    Returns:
        Resolved absolute path of the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("Sensitivity results exported: %d rows → %s", len(df), output_path)
    return output_path.resolve()


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.model.sensitivity",
        description=(
            "Sprint 5 — Sensitivity Analysis.\n"
            "Varies each Right Now Risk component weight ±0.10 (OAT) and "
            "exports per-scenario scores to outputs/tables/sensitivity_results.csv."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output", metavar="PATH", default=str(_OUTPUT_PATH),
        help=f"Destination CSV (default: {_OUTPUT_PATH}).",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        from src.model.right_now_risk import run_right_now_risk
        results = run_right_now_risk(export_csv=None)
        base_df = results["right_now_risk_df"]
    except Exception as exc:
        log.error("Failed to build base scores: %s", exc)
        return 1

    try:
        grid = build_weight_grid()
        sens_df = run_sensitivity(base_df, weight_grid=grid)
    except Exception as exc:
        log.error("Sensitivity run failed: %s", exc)
        return 1

    export_sensitivity(sens_df, Path(args.output))

    summary = summarize_sensitivity(sens_df)
    if not summary.empty:
        most_stable   = summary.iloc[-1]["country_label"] if "country_label" in summary.columns else summary.iloc[-1]["country_code_a3"]
        most_volatile = summary.iloc[0]["country_label"]  if "country_label" in summary.columns else summary.iloc[0]["country_code_a3"]
        log.info(
            "Rank volatility: most stable=%s (σ=%.3f)  most volatile=%s (σ=%.3f)",
            most_stable,   summary.iloc[-1]["rank_volatility"],
            most_volatile, summary.iloc[0]["rank_volatility"],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
