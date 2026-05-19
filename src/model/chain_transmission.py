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


def build_chain_table(ref_df: pd.DataFrame) -> pd.DataFrame:
    """Build the full chain transmission output table.

    Adds severity scores, backward-compatibility aliases, and the
    ``is_exporter`` and ``year`` columns expected by downstream consumers.

    Additional columns added:
      - ``chain_transmission_severity``       composite [0,1] score
      - ``chain_transmission_severity_recent`` same value (static snapshot)
      - ``transmission_severity``             alias — required by right_now_risk.py
      - ``is_exporter``                       True for oil/gas net exporters
      - ``year``                              static snapshot year (2024)

    Args:
        ref_df: Output of :func:`load_chain_reference`.

    Returns:
        DataFrame with one row per country, sorted by
        ``chain_transmission_severity`` descending.
    """
    df = compute_chain_severity(ref_df)

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
) -> pd.DataFrame:
    """End-to-end chain transmission pipeline: load → compute → save.

    Args:
        ref_path:    Path to the reference CSV (data/reference/).
        output_path: Destination path for the output CSV (outputs/tables/).

    Returns:
        Computed chain transmission DataFrame (also written to *output_path*).

    Raises:
        FileNotFoundError: If *ref_path* does not exist.
        ValueError: If required columns are absent.
    """
    ref_df   = load_chain_reference(ref_path)
    chain_df = build_chain_table(ref_df)
    save_chain_table(chain_df, output_path)
    _print_summary(chain_df)
    return chain_df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
        )
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
