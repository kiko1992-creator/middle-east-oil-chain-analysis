"""
Historical Risk Index — Sprint 4.

Runs the Right Now Risk pipeline for each year in a range (default 2015–2024)
using historical annual-average Brent prices, and produces a 140-row panel
(14 countries × 10 years) for trend analysis.

This is a CONDITIONAL backtest: fiscal breakeven, reserve, and food-security
reference data are the 2023 static estimates held fixed across all years.
Only the Brent price and chain transmission data vary per year.

Chain component note
--------------------
The chain_transmission.csv output is a static 2024 snapshot (year=2024).
For years prior to 2024 the chain component is NaN; weight rescaling
in compute_right_now_risk handles this gracefully.

Key functions
-------------
run_historical_index    Run snapshots for every year in [start, end]
export_historical_index Write the panel to outputs/tables/historical_risk_index.csv

Usage (from project root)
--------------------------
    python -m src.model.historical_index
    python -m src.model.historical_index --start 2015 --end 2024
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.model.backtest import (
    BREAKEVEN_PATH,
    CHAIN_PATH,
    FOOD_PATH,
    PANEL_PATH,
    RESERVES_PATH,
    run_backtest_snapshot,
)

log = logging.getLogger(__name__)

_DEFAULT_START = 2015
_DEFAULT_END   = 2024
_OUTPUT_PATH   = Path("outputs/tables/historical_risk_index.csv")

_KEEP_COLS = [
    "year",
    "country_code_a3",
    "country_label",
    "historical_brent_usd",
    "fiscal_stress_score",
    "reserve_runway_risk",
    "social_stability_risk",
    "chain_transmission_severity",
    "right_now_risk_score",
    "missing_components",
    "rescaled_weights",
]


# ── Public functions ───────────────────────────────────────────────────────────

def run_historical_index(
    start: int = _DEFAULT_START,
    end:   int = _DEFAULT_END,
    scenario: str = "base",
    breakeven_path: Path = BREAKEVEN_PATH,
    reserves_path:  Path = RESERVES_PATH,
    food_path:      Path = FOOD_PATH,
    chain_path:     Path = CHAIN_PATH,
    panel_path:     Path = PANEL_PATH,
) -> pd.DataFrame:
    """Run the Right Now Risk pipeline for every year in [start, end].

    Args:
        start:          First year to evaluate (default 2015).
        end:            Last year to evaluate (default 2024).
        scenario:       Backtest scenario — "base", "stress", or "optimistic".
        breakeven_path: Path to fiscal_breakeven.csv.
        reserves_path:  Path to swf_reserves.csv.
        food_path:      Path to food_security.csv.
        chain_path:     Path to chain_transmission.csv.
        panel_path:     Path to world_bank_panel.csv.

    Returns:
        DataFrame with one row per (year, country), sorted by (year,
        right_now_risk_score descending).  Columns follow the schema in
        _KEEP_COLS.  The chain component is NaN for years without a chain
        entry (pre-2024 given the static snapshot).
    """
    frames: list[pd.DataFrame] = []

    for yr in range(start, end + 1):
        try:
            snap = run_backtest_snapshot(
                year=yr,
                scenario=scenario,
                breakeven_path=breakeven_path,
                reserves_path=reserves_path,
                food_path=food_path,
                chain_path=chain_path,
                panel_path=panel_path,
            )
            snap["year"] = yr
            frames.append(snap)
            log.info("Historical index: year=%d  rows=%d", yr, len(snap))
        except ValueError as exc:
            log.warning("Skipping year %d: %s", yr, exc)

    if not frames:
        log.error("No snapshots produced for range %d–%d", start, end)
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Rename chain column to match schema
    if "chain_transmission_severity_recent" in combined.columns:
        combined = combined.rename(
            columns={"chain_transmission_severity_recent": "chain_transmission_severity"}
        )

    # Add country_label fallback if absent
    if "country_label" not in combined.columns and "country_name" in combined.columns:
        combined["country_label"] = combined["country_name"]

    # Select and order output columns
    out_cols = [c for c in _KEEP_COLS if c in combined.columns]
    extra    = [c for c in combined.columns if c not in out_cols]
    df_out   = combined[out_cols + extra].copy()

    df_out = (
        df_out
        .sort_values(["year", "right_now_risk_score"], ascending=[True, False])
        .reset_index(drop=True)
    )

    n_rows     = len(df_out)
    n_complete = int((df_out["missing_components"] == "").sum())
    log.info(
        "Historical index complete: %d rows (%d years × 14 countries)  "
        "%d complete / %d partial",
        n_rows, end - start + 1, n_complete, n_rows - n_complete,
    )
    return df_out


def export_historical_index(
    df: pd.DataFrame,
    output_path: Path = _OUTPUT_PATH,
) -> Path:
    """Write the historical risk index panel to a CSV.

    Args:
        df:          Output of :func:`run_historical_index`.
        output_path: Destination path (parent directory created if absent).

    Returns:
        Resolved absolute path of the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("Historical index exported: %d rows → %s", len(df), output_path)
    return output_path.resolve()


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.model.historical_index",
        description=(
            "Sprint 4 — Historical Risk Index.\n"
            "Runs the Right Now Risk pipeline for each year in a range and "
            "exports a panel CSV to outputs/tables/historical_risk_index.csv."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--start", type=int, default=_DEFAULT_START,
        help=f"First year to evaluate (default: {_DEFAULT_START}).",
    )
    p.add_argument(
        "--end", type=int, default=_DEFAULT_END,
        help=f"Last year to evaluate (default: {_DEFAULT_END}).",
    )
    p.add_argument(
        "--scenario", default="base",
        choices=["base", "stress", "optimistic"],
        help="Backtest scenario (default: base).",
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
        df = run_historical_index(start=args.start, end=args.end, scenario=args.scenario)
        if df.empty:
            log.error("Historical index produced no rows — check data paths.")
            return 1
        export_historical_index(df, Path(args.output))
    except Exception as exc:
        log.error("Historical index failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
