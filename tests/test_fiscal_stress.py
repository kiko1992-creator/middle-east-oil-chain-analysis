"""
Deterministic unit tests for src.model.fiscal_stress.

No network calls. No random data. No pytest required.

Run from the project root:
    python tests/test_fiscal_stress.py

Exit codes:
    0  all assertions passed
    1  one or more assertions failed
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.model.fiscal_stress import (
    _AMBER_BUFFER,
    build_stress_table,
    classify_stress,
)

# ---------------------------------------------------------------------------
# Minimal assertion harness — collects ALL failures before printing summary.
# ---------------------------------------------------------------------------

_passed: list[str] = []
_failed: list[str] = []


def check(label: str, condition: bool) -> None:
    if condition:
        _passed.append(label)
        print(f"  PASS  {label}")
    else:
        _failed.append(label)
        print(f"  FAIL  {label}")


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


# ---------------------------------------------------------------------------
# Fixtures — pure in-memory DataFrames, no file I/O, no network.
# ---------------------------------------------------------------------------

def _one_row(country_type: str, breakeven: float) -> pd.DataFrame:
    """Minimal single-country DataFrame accepted by build_stress_table."""
    return pd.DataFrame([{
        "country_code":         "SA",
        "country_code_a3":      "SAU",
        "country_name":         "Saudi Arabia",
        "country_label":        "Saudi Arabia",
        "country_type":         country_type,
        "fiscal_breakeven_usd": breakeven,
        "confidence":           "medium",
        "is_estimate":          True,
    }])


_YTD_EMPTY = pd.DataFrame(columns=["Close"])

# 10 trading-day sample: 5 closes at 70 (below 80 breakeven) + 5 at 85 (above).
_YTD_GOOD = pd.DataFrame({"Close": [70.0] * 5 + [85.0] * 5})


# ===========================================================================
# Suite 1 — classify_stress: boundary correctness
#
# Reference breakeven = 80.  _AMBER_BUFFER = 15 (imported, not hard-coded).
# Expected transitions:
#   brent < 80            -> Red
#   80 <= brent < 95      -> Amber
#   brent >= 95           -> Green
# ===========================================================================

section("Suite 1 — classify_stress boundary values  (breakeven=80, type='exporter')")

BE = 80.0

check(
    "brent = breakeven - 0.01 (79.99)   -> Red",
    classify_stress(BE - 0.01, BE, "exporter") == "Red",
)
check(
    "brent = breakeven exactly (80.00)  -> Amber  [at BE is not below it]",
    classify_stress(BE, BE, "exporter") == "Amber",
)
check(
    "brent = breakeven + 14.99 (94.99)  -> Amber",
    classify_stress(BE + 14.99, BE, "exporter") == "Amber",
)
check(
    "brent = breakeven + 15.00 (95.00)  -> Green  [exact buffer boundary]",
    classify_stress(BE + _AMBER_BUFFER, BE, "exporter") == "Green",
)
check(
    "brent = breakeven + 15.01 (95.01)  -> Green",
    classify_stress(BE + 15.01, BE, "exporter") == "Green",
)

# Same boundaries hold for mixed_importer (Egypt-style)
check(
    "mixed_importer  brent < breakeven  -> Red",
    classify_stress(44.0, 45.0, "mixed_importer") == "Red",
)
check(
    "mixed_importer  brent >= breakeven + buffer  -> Green",
    classify_stress(61.0, 45.0, "mixed_importer") == "Green",
)


# ===========================================================================
# Suite 2 — classify_stress: NaN/nan input always returns Gray
#
# All three common NaN representations are tested.  Verifies that the fix
# (pd.isna guard) prevents NaN comparisons silently returning False and
# falling through to Green.
# ===========================================================================

section("Suite 2 — classify_stress(NaN brent) -> always Gray")

_nan_variants = [
    (float("nan"), "float('nan')"),
    (math.nan,     "math.nan    "),
    (np.nan,       "np.nan      "),
]
_nan_cases = [
    ("exporter",       80.0),
    ("mixed_importer", 45.0),
    ("exporter",       50.0),
]

for brent_val, brent_label in _nan_variants:
    for ctype, be in _nan_cases:
        check(
            f"classify_stress({brent_label}, be={be}, type={ctype!r})  -> Gray",
            classify_stress(brent_val, be, ctype) == "Gray",
        )

# Explicit anti-regression: NaN must never produce Red, Amber, or Green.
for verdict in ("Red", "Amber", "Green"):
    check(
        f"NaN brent never produces '{verdict}'",
        classify_stress(np.nan, 80.0, "exporter") != verdict,
    )


# ===========================================================================
# Suite 3 — classify_stress: net_importer always Gray
#
# The concept is not applicable regardless of price or breakeven value.
# ===========================================================================

section("Suite 3 — net_importer always Gray  (any price, any breakeven)")

_importer_prices = [0.0, 1.0, 50.0, 79.99, 80.0, 95.0, 999.0, np.nan]

for price in _importer_prices:
    check(
        f"net_importer  brent={price}  -> Gray",
        classify_stress(price, 80.0, "net_importer") == "Gray",
    )

# zero breakeven on a non-importer also maps to Gray (not applicable)
check(
    "breakeven=0  exporter  -> Gray  [zero signals N/A]",
    classify_stress(72.0, 0.0, "exporter") == "Gray",
)


# ===========================================================================
# Suite 4 — build_stress_table: empty YTD -> stress_days_ytd = NA, not 0
#
# When fetch_brent_ytd() fails and returns an empty DataFrame, the function
# must NOT leave stress_days_ytd=0 (which would imply "no stress this year").
# It must be pd.NA / NaN so the UI can display "N/A".
# ===========================================================================

section("Suite 4 — build_stress_table: total_trading_days=0 -> stress metrics NA")

# ── 4a: exporter, empty YTD ──────────────────────────────────────────────────
tbl_exp_empty = build_stress_table(_one_row("exporter", 80.0), 72.0, _YTD_EMPTY)
r = tbl_exp_empty.iloc[0]

check(
    "exporter + empty YTD   total_trading_days == 0",
    int(r["total_trading_days"]) == 0,
)
check(
    "exporter + empty YTD   stress_days_ytd is NA   (not 0)",
    pd.isna(r["stress_days_ytd"]),
)
check(
    "exporter + empty YTD   stress_share_ytd is NaN  (not 0.0)",
    pd.isna(r["stress_share_ytd"]),
)

# ── 4b: mixed_importer, empty YTD ────────────────────────────────────────────
tbl_mix_empty = build_stress_table(_one_row("mixed_importer", 45.0), 72.0, _YTD_EMPTY)
r_mix = tbl_mix_empty.iloc[0]

check(
    "mixed_importer + empty YTD   stress_days_ytd is NA  (not 0)",
    pd.isna(r_mix["stress_days_ytd"]),
)
check(
    "mixed_importer + empty YTD   stress_share_ytd is NaN  (not 0.0)",
    pd.isna(r_mix["stress_share_ytd"]),
)

# ── 4c: contrast — non-empty YTD computes real values, not NA ────────────────
# _YTD_GOOD has 5 closes at 70 (< 80) and 5 at 85 (>= 80).
tbl_exp_good = build_stress_table(_one_row("exporter", 80.0), 72.0, _YTD_GOOD)
r_good = tbl_exp_good.iloc[0]

check(
    "exporter + 10-day YTD   total_trading_days == 10",
    int(r_good["total_trading_days"]) == 10,
)
check(
    "exporter + 10-day YTD   stress_days_ytd == 5   (not NA)",
    not pd.isna(r_good["stress_days_ytd"]) and int(r_good["stress_days_ytd"]) == 5,
)
check(
    "exporter + 10-day YTD   stress_share_ytd == 0.5  (not NA)",
    not pd.isna(r_good["stress_share_ytd"])
    and abs(float(r_good["stress_share_ytd"]) - 0.5) < 1e-9,
)

# ── 4d: net_importer Gray rows stay NA regardless of YTD availability ─────────
tbl_imp_good = build_stress_table(_one_row("net_importer", 0.0), 72.0, _YTD_GOOD)
r_imp = tbl_imp_good.iloc[0]

check(
    "net_importer + good YTD   stress_status == Gray",
    r_imp["stress_status"] == "Gray",
)
check(
    "net_importer + good YTD   stress_days_ytd still NA",
    pd.isna(r_imp["stress_days_ytd"]),
)


# ===========================================================================
# Summary
# ===========================================================================

total = len(_passed) + len(_failed)
print(f"\n{'='*60}")
print(f"Results: {len(_passed)}/{total} passed,  {len(_failed)} failed")

if _failed:
    print("\nFailed assertions:")
    for label in _failed:
        print(f"  - {label}")
    sys.exit(1)

sys.exit(0)
