"""
Reference data integrity validator.

Checks:
  1. source_id referential integrity  — every source_id_primary and
     source_id_secondary in each reference CSV must exist in source_registry.csv
  2. confidence enum validity         — only 'high', 'medium', 'low', '' (empty)
  3. base values present              — exporter rows must have non-null base values
  4. monotonic uncertainty bands      — low <= base <= high where all three present
  5. no duplicate country_code_a3     — each reference CSV must have one row per country

Usage:
    python -m src.data.validate_reference             # prints PASS/FAIL summary
    python -m src.data.validate_reference --strict    # exits 1 on any failure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parents[2]

REGISTRY_PATH     = _ROOT / "data" / "reference" / "source_registry.csv"
BREAKEVEN_PATH    = _ROOT / "data" / "reference" / "fiscal_breakeven.csv"
RESERVES_PATH     = _ROOT / "data" / "reference" / "swf_reserves.csv"
FOOD_PATH         = _ROOT / "data" / "reference" / "food_security.csv"
RETRO_PATH        = _ROOT / "data" / "reference" / "imf_weo_2020_outcomes.csv"
BENCHMARKS_PATH   = _ROOT / "data" / "reference" / "imf_wb_benchmarks.csv"

_VALID_CONFIDENCE   = {"high", "medium", "low", ""}
_VALID_FM_TIER      = {"Low", "Medium", "High"}
_VALID_MPO_STATUS   = {"Stable", "Watch", "Stressed"}
_N_COUNTRIES        = 14

# ── Helpers ────────────────────────────────────────────────────────────────────

_RESULTS: list[tuple[str, bool, str]] = []   # (label, passed, detail)


def _check(label: str, condition: bool, detail: str = "") -> bool:
    _RESULTS.append((label, condition, detail))
    sym = "PASS" if condition else "FAIL"
    suffix = f"  -- {detail}" if detail else ""
    print(f"{sym}  {label}{suffix}")
    return condition


def _load(path: Path, name: str) -> pd.DataFrame | None:
    if not path.exists():
        _check(f"file exists: {name}", False, str(path))
        return None
    _check(f"file exists: {name}", True)
    return pd.read_csv(path, keep_default_na=False)


# ── Check functions ────────────────────────────────────────────────────────────

def check_registry_self_consistent(reg: pd.DataFrame) -> None:
    """Registry: unique source_id, confidence_tier in allowed set."""
    dups = reg["source_id"][reg["source_id"].duplicated()].tolist()
    _check("registry: no duplicate source_id", len(dups) == 0, str(dups))

    bad = [v for v in reg["confidence_tier"] if v not in _VALID_CONFIDENCE]
    _check("registry: confidence_tier valid", len(bad) == 0, str(bad))


def check_source_referential_integrity(
    df: pd.DataFrame,
    name: str,
    valid_ids: set[str],
) -> None:
    """Every non-empty source_id_primary / source_id_secondary must be in registry."""
    for col in ("source_id_primary", "source_id_secondary"):
        if col not in df.columns:
            _check(f"{name}: {col} column present", False)
            continue
        _check(f"{name}: {col} column present", True)
        bad = [v for v in df[col] if v and v not in valid_ids]
        _check(f"{name}: {col} all resolve to registry", len(bad) == 0,
               f"unknown ids: {bad}" if bad else "")


def check_confidence_enum(df: pd.DataFrame, name: str) -> None:
    """confidence column must only contain high / medium / low / ''."""
    if "confidence" not in df.columns:
        _check(f"{name}: confidence column present", False)
        return
    bad = [v for v in df["confidence"] if v not in _VALID_CONFIDENCE]
    _check(f"{name}: confidence enum valid", len(bad) == 0,
           f"bad values: {bad}" if bad else "")


def check_no_duplicate_a3(df: pd.DataFrame, name: str) -> None:
    """country_code_a3 must be unique in every reference CSV."""
    if "country_code_a3" not in df.columns:
        _check(f"{name}: country_code_a3 present", False)
        return
    dups = df["country_code_a3"][df["country_code_a3"].duplicated()].tolist()
    _check(f"{name}: no duplicate country_code_a3", len(dups) == 0,
           f"dups: {dups}" if dups else "")


def check_monotonic_bands(
    df: pd.DataFrame,
    name: str,
    low_col: str,
    base_col: str,
    high_col: str,
) -> None:
    """Where all three band columns are non-null: low <= base <= high."""
    for col in (low_col, base_col, high_col):
        if col not in df.columns:
            _check(f"{name}: {col} present", False)
            return

    mask = (
        df[low_col].notna() & df[base_col].notna() & df[high_col].notna()
        & (df[low_col] != "") & (df[base_col] != "") & (df[high_col] != "")
    )
    sub = df.loc[mask].copy()
    if sub.empty:
        _check(f"{name}: {low_col}..{high_col} monotonic (no rows to check)", True)
        return

    lo  = pd.to_numeric(sub[low_col],  errors="coerce")
    ba  = pd.to_numeric(sub[base_col], errors="coerce")
    hi  = pd.to_numeric(sub[high_col], errors="coerce")

    lo_ok = (lo <= ba + 1e-9).all()
    hi_ok = (ba <= hi + 1e-9).all()

    violations = sub[~(lo <= ba + 1e-9) | ~(ba <= hi + 1e-9)]["country_code_a3"].tolist() \
        if "country_code_a3" in sub.columns else []
    _check(
        f"{name}: {low_col} <= {base_col} <= {high_col}",
        bool(lo_ok and hi_ok),
        f"violations: {violations}" if violations else f"checked {len(sub)} rows",
    )


def check_enum_column(
    df: pd.DataFrame,
    name: str,
    col: str,
    valid_values: set[str],
) -> None:
    """Column must only contain values from valid_values (empty string also allowed)."""
    if col not in df.columns:
        _check(f"{name}: {col} column present", False)
        return
    bad = [v for v in df[col] if str(v) not in valid_values and str(v) != ""]
    _check(f"{name}: {col} enum valid", len(bad) == 0,
           f"bad values: {bad}" if bad else "")


def check_rank_column_unique(
    df: pd.DataFrame,
    name: str,
    col: str,
    expected_n: int,
) -> None:
    """Rank column must be integers 1..expected_n with no duplicates."""
    if col not in df.columns:
        _check(f"{name}: {col} column present", False)
        return
    vals = pd.to_numeric(df[col], errors="coerce").dropna().astype(int).tolist()
    duplicates = [v for v in vals if vals.count(v) > 1]
    _check(f"{name}: {col} no duplicates", len(duplicates) == 0,
           f"duplicated ranks: {sorted(set(duplicates))}" if duplicates else "")
    expected = set(range(1, expected_n + 1))
    actual   = set(vals)
    missing  = expected - actual
    extra    = actual - expected
    _check(f"{name}: {col} covers 1..{expected_n}",
           not missing and not extra,
           f"missing={sorted(missing)} extra={sorted(extra)}" if (missing or extra) else "")


def check_base_values_present(
    df: pd.DataFrame,
    name: str,
    base_col: str,
    require_for: str = "exporter",   # country_type value that must have non-null base
) -> None:
    """Exporter rows (or all rows for food) must have a non-null, positive base value."""
    if base_col not in df.columns:
        _check(f"{name}: {base_col} present", False)
        return

    if "country_type" in df.columns and require_for:
        sub = df[df["country_type"] == require_for]
    else:
        sub = df

    numeric = pd.to_numeric(sub[base_col], errors="coerce")
    missing = sub[numeric.isna() | (numeric == 0)]
    n_miss  = len(missing)
    labels  = missing["country_code_a3"].tolist() if "country_code_a3" in missing.columns else []
    _check(
        f"{name}: {base_col} non-null/non-zero for {require_for or 'all'} rows",
        n_miss == 0,
        f"missing/zero in: {labels}" if n_miss else f"ok ({len(sub)} rows checked)",
    )


# ── Main validation run ────────────────────────────────────────────────────────

def run_all() -> bool:
    reg   = _load(REGISTRY_PATH,   "source_registry.csv")
    be    = _load(BREAKEVEN_PATH,  "fiscal_breakeven.csv")
    res   = _load(RESERVES_PATH,   "swf_reserves.csv")
    food  = _load(FOOD_PATH,       "food_security.csv")
    retro = _load(RETRO_PATH,      "imf_weo_2020_outcomes.csv")
    bench = _load(BENCHMARKS_PATH, "imf_wb_benchmarks.csv")

    if reg is None:
        print("\nCannot continue without source registry.")
        return False

    valid_ids = set(reg["source_id"].tolist())
    print(f"\n-- registry: {len(valid_ids)} source_ids loaded --")

    check_registry_self_consistent(reg)

    for df, name in [
        (be,    "fiscal_breakeven"),
        (res,   "swf_reserves"),
        (food,  "food_security"),
        (retro, "imf_weo_2020_outcomes"),
        (bench, "imf_wb_benchmarks"),
    ]:
        if df is None:
            continue
        print(f"\n-- {name} --")
        check_no_duplicate_a3(df, name)
        check_confidence_enum(df, name)
        check_source_referential_integrity(df, name, valid_ids)

    if be is not None:
        print("\n-- fiscal_breakeven uncertainty bands --")
        check_base_values_present(be, "fiscal_breakeven",
                                  "breakeven_base_usd", require_for="exporter")
        check_monotonic_bands(be, "fiscal_breakeven",
                              "breakeven_low_usd", "breakeven_base_usd", "breakeven_high_usd")

    if res is not None:
        print("\n-- swf_reserves uncertainty bands --")
        # Only check rows where concept applies (non-empty confidence = exporter/mixed)
        res_exp = res[res["confidence"].isin({"high", "medium", "low"})].copy()
        check_monotonic_bands(res_exp, "swf_reserves (exporters only)",
                              "liquid_buffer_low_usd_bn", "liquid_buffer_base_usd_bn",
                              "liquid_buffer_high_usd_bn")
        check_monotonic_bands(res_exp, "swf_reserves (exporters only)",
                              "monthly_burn_low_usd_bn", "monthly_burn_base_usd_bn",
                              "monthly_burn_high_usd_bn")

    if food is not None:
        print("\n-- food_security uncertainty bands --")
        check_monotonic_bands(food, "food_security",
                              "cereal_dependency_low", "cereal_dependency_base",
                              "cereal_dependency_high")

    if retro is not None:
        print("\n-- imf_weo_2020_outcomes domain checks --")
        check_rank_column_unique(retro, "imf_weo_2020_outcomes",
                                 "outcome_severity_rank", _N_COUNTRIES)

    if bench is not None:
        print("\n-- imf_wb_benchmarks domain checks --")
        check_enum_column(bench, "imf_wb_benchmarks",
                          "imf_fm_risk_tier", _VALID_FM_TIER)
        check_enum_column(bench, "imf_wb_benchmarks",
                          "wb_mpo_status", _VALID_MPO_STATUS)

    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    print(f"\n{'=' * 52}")
    print(f"TOTAL: {n_pass} PASS  {n_fail} FAIL  ({len(_RESULTS)} checks)")
    return n_fail == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate reference data integrity")
    parser.add_argument("--strict", action="store_true",
                        help="Exit with code 1 if any check fails")
    args = parser.parse_args()
    ok = run_all()
    if args.strict and not ok:
        sys.exit(1)
