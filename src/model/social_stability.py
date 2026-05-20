"""
Social stability risk model for MENA countries.

Answers: "Given food import dependency, fiscal stress, and inflation
volatility, which MENA countries face the highest social stability risk?"

Social stability risk is a composite 0–1 indicator:

  social_stability_risk =
      0.5 × food_security_exposure
    + 0.3 × fiscal_stress_score
    + 0.2 × norm(inflation_volatility)

Component definitions
---------------------
food_security_exposure =
    0.6 × norm(food_imports_pct_merch_imports)
  + 0.4 × norm(cereal_import_dependency)

  Fallback: if cereal_import_dependency is absent for a country, the
  weight is redistributed fully to food_imports_pct (weight = 1.0).
  data_quality_flag is set; the score is still computed.

fiscal_stress_score (continuous):
    min(1, max(0, (fiscal_breakeven_usd - brent_live_usd) / fiscal_breakeven_usd))

  For net importers (Gray) and countries with breakeven ≤ 0: score = 0.0.
  When brent > breakeven the formula gives a negative value, clamped to 0.0.
  When brent << breakeven the formula approaches 1.0.

  This replaces a former categorical mapping (Red/Amber/Green → fixed scalars)
  with a continuous 0–1 score that captures the *degree* of fiscal stress, not
  just the threshold bucket.

inflation_volatility — std(FP_CPI_TOTL_ZG, 2000–2024) from the WB panel —
  is normalised with percentile winsorisation (p5–p95) before min-max scaling
  to prevent Lebanon's extreme CPI volatility from compressing all other
  countries to near-zero.  The normalization_method column records the method.

All norm() operations except inflation use standard min-max across all
14 MENA countries so every component lands in [0, 1].

Key functions
-------------
load_food_security          Load data/reference/food_security.csv
derive_inflation_vol        Compute std(CPI) per country from the WB panel
build_stability_table       Merge + normalise + score all countries
run_social_stability        End-to-end orchestrator

Data sources
------------
- Food imports % merch: World Bank WDI TM.VAL.FOOD.ZS.UN (~2022)
- Cereal import dependency: FAO FAOSTAT Food Balance Sheets (~2021)
- Inflation volatility:     World Bank FP.CPI.TOTL.ZG, std 2000-2024
- Fiscal stress scores:     src.model.fiscal_stress (Addition 1)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.model.fiscal_stress import (
    build_stress_table,
    fetch_brent_live,
    fetch_brent_ytd,
    load_breakeven,
)

log = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────────────

PANEL_PATH      = Path("data/processed/world_bank_panel.csv")
FOOD_PATH       = Path("data/reference/food_security.csv")
BREAKEVEN_PATH  = Path("data/reference/fiscal_breakeven.csv")
FX_CHANNEL_PATH = Path("data/reference/fx_channel.csv")

# ── Component weights ───────────────────────────────────────────────────────────

# Within food_security_exposure
_W_FOOD_IMPORTS = 0.6
_W_CEREAL_DEP   = 0.4

# Within social_stability_risk
_W_FOOD_EXP  = 0.5
_W_FISCAL    = 0.3
_W_INFLATION = 0.2

# ── Inflation normalisation parameters ─────────────────────────────────────────

_INFLATION_WINSOR_P_LOW  = 5.0
_INFLATION_WINSOR_P_HIGH = 95.0

# ── Risk driver labels ──────────────────────────────────────────────────────────

DRIVER_FOOD      = "Food security exposure"
DRIVER_FISCAL    = "Fiscal stress"
DRIVER_INFLATION = "Inflation volatility"
DRIVER_MIXED     = "Mixed"

_ALLOWED_DRIVERS = frozenset(
    {DRIVER_FOOD, DRIVER_FISCAL, DRIVER_INFLATION, DRIVER_MIXED}
)

# Required columns in the food security CSV.
_FOOD_REQUIRED: list[str] = [
    "country_code",
    "country_code_a3",
    "country_name",
    "country_label",
    "food_imports_pct_merch_imports",
    "cereal_import_dependency",
]


# ── Data loading ────────────────────────────────────────────────────────────────

_FX_REQUIRED: list[str] = ["country_code_a3", "fx_channel_relevant", "fx_channel_weight"]


def load_fx_channel(path: Path = FX_CHANNEL_PATH) -> pd.DataFrame:
    """Load and validate the FX channel reference CSV.

    The FX channel CSV records, for each country, whether the exchange rate
    channel is relevant for transmitting oil price shocks to social stability
    (i.e. whether currency depreciation amplifies import-price inflation), and
    the weight of that adjustment in the social stability score.

    GCC countries with hard USD pegs have fx_channel_relevant=False and
    fx_channel_weight=0.0.  Countries with managed floats or a history of
    large devaluations carry positive weights (up to 0.3).

    Args:
        path: Path to ``fx_channel.csv``.

    Returns:
        DataFrame with one row per country and all reference fields.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If required columns are absent.
    """
    if not path.exists():
        raise FileNotFoundError(f"FX channel CSV not found: {path}")

    df = pd.read_csv(path)

    missing = [c for c in _FX_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {missing}")

    df["fx_channel_relevant"] = df["fx_channel_relevant"].astype(str).str.lower().isin(
        ("true", "1", "yes")
    )
    df["fx_channel_weight"] = pd.to_numeric(df["fx_channel_weight"], errors="coerce").fillna(0.0)

    n_relevant = int(df["fx_channel_relevant"].sum())
    log.info("FX channel data loaded: %d countries, %d with active FX channel.", len(df), n_relevant)
    return df


def load_food_security(path: Path = FOOD_PATH) -> pd.DataFrame:
    """Load and validate the food security reference CSV.

    Args:
        path: Path to ``food_security.csv``.

    Returns:
        DataFrame with one row per country and all reference fields.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If required columns are absent.
    """
    if not path.exists():
        raise FileNotFoundError(f"Food security CSV not found: {path}")

    df = pd.read_csv(path)

    missing = [c for c in _FOOD_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {missing}")

    for col in ("food_imports_pct_merch_imports", "cereal_import_dependency"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info(
        "Food security data loaded: %d countries, cereal present for %d",
        len(df),
        int(df["cereal_import_dependency"].notna().sum()),
    )
    return df


# ── Inflation volatility derivation ─────────────────────────────────────────────

def derive_inflation_vol(panel_path: Path = PANEL_PATH) -> pd.DataFrame:
    """Compute per-country inflation volatility from the WB panel.

    Inflation volatility = std(FP_CPI_TOTL_ZG) across all non-null
    annual observations.  Requires at least 2 years; returns NaN otherwise.

    Consistent with the OCVI model's ``inflation_vol`` component —
    same panel, same column, same formula.

    Args:
        panel_path: Path to ``data/processed/world_bank_panel.csv``.

    Returns:
        DataFrame with columns:
        ``country_code_a3``, ``inflation_volatility``, ``yrs_inflation``.

    Raises:
        FileNotFoundError: If *panel_path* does not exist.
    """
    if not panel_path.exists():
        raise FileNotFoundError(f"Panel CSV not found: {panel_path}")

    panel = pd.read_csv(panel_path, usecols=["country_code_a3", "FP_CPI_TOTL_ZG"])

    records = []
    for code, grp in panel.groupby("country_code_a3"):
        cpi = grp["FP_CPI_TOTL_ZG"].dropna()
        records.append({
            "country_code_a3":      code,
            "inflation_volatility": cpi.std() if len(cpi) >= 2 else float("nan"),
            "yrs_inflation":        len(cpi),
        })

    df = pd.DataFrame(records)
    log.info("Inflation volatility derived for %d countries from panel.", len(df))
    return df


# ── Normalisation helpers ────────────────────────────────────────────────────────

def _minmax(series: pd.Series) -> pd.Series:
    """Min-max normalise *series* to [0, 1] (NaN preserved; uniform → 0.0)."""
    col_min, col_max = series.min(), series.max()
    if pd.isna(col_min) or pd.isna(col_max) or col_min == col_max:
        return series.where(series.isna(), 0.0)
    return (series - col_min) / (col_max - col_min)


def _minmax_winsorized(
    series: pd.Series,
    p_low: float = _INFLATION_WINSOR_P_LOW,
    p_high: float = _INFLATION_WINSOR_P_HIGH,
) -> tuple[pd.Series, str]:
    """Winsorise at (p_low, p_high) percentiles then apply min-max normalisation.

    Prevents a single outlier (e.g. Lebanon's extreme CPI volatility) from
    compressing all other normalised values to near zero.

    Args:
        series: Raw numeric series (NaN-tolerant).
        p_low:  Lower percentile cap (default 5).
        p_high: Upper percentile cap (default 95).

    Returns:
        Tuple ``(normalised_series, method_string)`` where the method string
        is suitable for the ``normalization_method`` column.
    """
    method = f"winsorize_p{int(p_low)}_p{int(p_high)}_minmax"
    vals = series.dropna()

    if len(vals) < 2:
        return series.where(series.isna(), 0.0), method

    lo = float(np.percentile(vals, p_low))
    hi = float(np.percentile(vals, p_high))

    if lo >= hi:
        return series.where(series.isna(), 0.0), method

    # pandas.Series.clip preserves NaN values.
    winsorized = series.clip(lower=lo, upper=hi)
    return (winsorized - lo) / (hi - lo), method


# ── Continuous fiscal stress score ───────────────────────────────────────────────

def compute_fiscal_stress_score(
    price_gap_usd: float,
    fiscal_breakeven_usd: float,
    stress_status: str,
) -> float:
    """Continuous fiscal stress score in [0, 1].

    Formula:
        min(1, max(0, (fiscal_breakeven_usd - brent_live_usd) / fiscal_breakeven_usd))

    Equivalently, since ``price_gap_usd = brent_live - fiscal_breakeven``:
        min(1, max(0, -price_gap_usd / fiscal_breakeven_usd))

    Interpretation:
        0.0 — no fiscal stress (brent at or above breakeven)
        1.0 — maximum stress (brent at zero; full breakeven deficit)
        Linear scaling between these extremes captures the *degree* of
        shortfall, not just a threshold bucket.

    Gray countries (net importers, breakeven N/A) receive 0.0 — they still
    accrue non-zero overall social stability risk via food exposure and
    inflation components.

    Args:
        price_gap_usd:        brent_live − fiscal_breakeven (USD/bbl).
        fiscal_breakeven_usd: Fiscal breakeven price (USD/bbl).
        stress_status:        Fiscal stress status from Addition 1.

    Returns:
        Float in [0.0, 1.0].
    """
    if stress_status == "Gray":
        return 0.0
    be = float(fiscal_breakeven_usd) if not pd.isna(fiscal_breakeven_usd) else 0.0
    pg = float(price_gap_usd) if not pd.isna(price_gap_usd) else float("nan")
    if be <= 0.0 or math.isnan(pg):
        return 0.0
    return min(1.0, max(0.0, -pg / be))


# ── Risk driver identification ───────────────────────────────────────────────────

def identify_risk_driver(
    contrib_food: float,
    contrib_fiscal: float,
    contrib_inflation: float,
) -> str:
    """Identify the dominant social stability risk driver.

    Returns the label of the highest weighted component contribution.
    Returns ``DRIVER_MIXED`` when:
      - all contributions are zero or NaN (total risk ≈ 0), OR
      - the top two contributions are within 5 % of total risk.

    Args:
        contrib_food:       0.5 × food_security_exposure
        contrib_fiscal:     0.3 × fiscal_stress_score
        contrib_inflation:  0.2 × inflation_volatility_norm

    Returns:
        One of ``DRIVER_FOOD``, ``DRIVER_FISCAL``, ``DRIVER_INFLATION``,
        ``DRIVER_MIXED``.
    """
    def _safe(v: float) -> float:
        return 0.0 if (pd.isna(v) or math.isnan(v)) else v

    contribs = {
        DRIVER_FOOD:      _safe(contrib_food),
        DRIVER_FISCAL:    _safe(contrib_fiscal),
        DRIVER_INFLATION: _safe(contrib_inflation),
    }
    total = sum(contribs.values())
    if total <= 0.0:
        return DRIVER_MIXED

    ranked = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
    top_label, top_val = ranked[0]
    _, second_val      = ranked[1]

    if (top_val - second_val) < 0.05 * total:
        return DRIVER_MIXED
    return top_label


# ── Core table assembly ──────────────────────────────────────────────────────────

def build_stability_table(
    food_df:       pd.DataFrame,
    stress_table:  pd.DataFrame,
    inflation_df:  pd.DataFrame,
) -> pd.DataFrame:
    """Merge all inputs, normalise components, and compute stability scores.

    Processing steps:

    1. Left-join food security + stress table + inflation on ``country_code_a3``.
    2. Check food provenance: set ``data_quality_flag`` where ``source_food``
       or ``source_cereal`` is missing.
    3. Min-max normalise food_imports_pct and cereal_import_dependency across
       all 14 countries.
    4. Compute ``food_security_exposure``:
       - Full formula (0.6 × norm_food + 0.4 × norm_cereal) when both present.
       - Fallback (1.0 × norm_food) when cereal is absent; ``data_quality_flag``
         is set.  No silent imputation.
    5. Compute continuous ``fiscal_stress_score`` via
       :func:`compute_fiscal_stress_score` (uses breakeven and live Brent from
       the stress table, not a categorical mapping).
    6. Normalise ``inflation_volatility`` with percentile winsorisation
       (:func:`_minmax_winsorized`, p5–p95) to prevent Lebanon's extreme
       volatility from compressing all other countries to near zero.
       Stores the method name in ``normalization_method``.
    7. Compute ``social_stability_risk``.
    8. Derive ``main_risk_driver`` via :func:`identify_risk_driver`.
    9. Flag countries with insufficient inflation data.

    Args:
        food_df:      Output of :func:`load_food_security`.
        stress_table: Output of ``src.model.fiscal_stress.build_stress_table``.
        inflation_df: Output of :func:`derive_inflation_vol`.

    Returns:
        DataFrame with one row per country, sorted by ``social_stability_risk``
        descending.
    """
    # Pull stress columns needed for fiscal score computation.
    _stress_cols = ["country_code_a3", "stress_status", "price_gap_usd",
                    "brent_live_usd", "fiscal_breakeven_usd"]
    stress_slim = stress_table[
        [c for c in _stress_cols if c in stress_table.columns]
    ].copy()

    df = (
        food_df
        .merge(stress_slim,  on="country_code_a3", how="left")
        .merge(inflation_df, on="country_code_a3", how="left")
    )

    df["stress_status"]       = df["stress_status"].fillna("Gray")
    df["price_gap_usd"]       = df.get("price_gap_usd",       pd.Series(float("nan"), index=df.index))
    df["brent_live_usd"]      = df.get("brent_live_usd",      pd.Series(float("nan"), index=df.index))
    df["fiscal_breakeven_usd"]= df.get("fiscal_breakeven_usd",pd.Series(float("nan"), index=df.index))

    # ── Step 2: food provenance flags ─────────────────────────────────────────
    flags: list[str] = [""] * len(df)

    for i, row in df.iterrows():
        src_food   = str(row.get("source_food",   ""))
        src_cereal = str(row.get("source_cereal", ""))
        if not src_food or src_food.lower() in ("nan", "n/a", ""):
            _append_flag(flags, i, "Food import source missing — provenance unverified")
        if not src_cereal or src_cereal.lower() in ("nan", "n/a", ""):
            _append_flag(flags, i, "Cereal source missing — provenance unverified")

    # ── Step 3: normalise food components ────────────────────────────────────
    df["norm_food_imports_pct"] = _minmax(df["food_imports_pct_merch_imports"])
    df["norm_cereal_dep"]       = _minmax(df["cereal_import_dependency"])

    # ── Step 4: food_security_exposure ────────────────────────────────────────
    food_exposure = []
    for i, row in df.iterrows():
        nf = float(row["norm_food_imports_pct"]) if not pd.isna(row["norm_food_imports_pct"]) else float("nan")
        nc = float(row["norm_cereal_dep"])        if not pd.isna(row["norm_cereal_dep"])        else float("nan")

        if pd.isna(nf) or math.isnan(nf):
            food_exposure.append(float("nan"))
            _append_flag(flags, i, "Food import data unavailable — food_security_exposure is NaN")
        elif pd.isna(nc) or math.isnan(nc):
            # Explicit fallback: cereal absent → full weight on food imports.
            food_exposure.append(nf)
            _append_flag(flags, i,
                "Cereal data unavailable — food exposure uses food imports only "
                "(weight rescaled to 1.0)")
        else:
            food_exposure.append(_W_FOOD_IMPORTS * nf + _W_CEREAL_DEP * nc)

    df["food_security_exposure"] = food_exposure

    # ── Step 5: continuous fiscal_stress_score ────────────────────────────────
    df["fiscal_stress_score"] = df.apply(
        lambda r: compute_fiscal_stress_score(
            float(r["price_gap_usd"])        if not pd.isna(r["price_gap_usd"])        else float("nan"),
            float(r["fiscal_breakeven_usd"]) if not pd.isna(r["fiscal_breakeven_usd"]) else float("nan"),
            str(r["stress_status"]),
        ),
        axis=1,
    )

    # ── Step 6: winsorised inflation normalisation ────────────────────────────
    inf_norm, norm_method = _minmax_winsorized(
        df["inflation_volatility"],
        p_low  = _INFLATION_WINSOR_P_LOW,
        p_high = _INFLATION_WINSOR_P_HIGH,
    )
    df["inflation_volatility_norm"] = inf_norm
    df["normalization_method"]       = norm_method

    # ── Step 7: social_stability_risk ─────────────────────────────────────────
    def _risk(row: pd.Series) -> float:
        fse = float(row["food_security_exposure"]) if not pd.isna(row["food_security_exposure"]) else float("nan")
        fss = float(row["fiscal_stress_score"])
        inv = float(row["inflation_volatility_norm"]) if not pd.isna(row["inflation_volatility_norm"]) else float("nan")

        if (pd.isna(fse) or math.isnan(fse)) and (pd.isna(inv) or math.isnan(inv)):
            return float("nan")

        fse_safe = fse if not (pd.isna(fse) or math.isnan(fse)) else 0.0
        inv_safe = inv if not (pd.isna(inv) or math.isnan(inv)) else 0.0
        return _W_FOOD_EXP * fse_safe + _W_FISCAL * fss + _W_INFLATION * inv_safe

    df["social_stability_risk"] = df.apply(_risk, axis=1)

    # ── Step 8: main_risk_driver ──────────────────────────────────────────────
    def _driver(row: pd.Series) -> str:
        fse = float(row["food_security_exposure"]) if not pd.isna(row["food_security_exposure"]) else 0.0
        fss = float(row["fiscal_stress_score"])
        inv = float(row["inflation_volatility_norm"]) if not pd.isna(row["inflation_volatility_norm"]) else 0.0
        return identify_risk_driver(
            _W_FOOD_EXP  * fse,
            _W_FISCAL    * fss,
            _W_INFLATION * inv,
        )

    df["main_risk_driver"] = df.apply(_driver, axis=1)

    # ── Step 9: inflation data quality flag ───────────────────────────────────
    for i, row in df.iterrows():
        if pd.isna(row.get("inflation_volatility")) or row.get("yrs_inflation", 25) < 2:
            _append_flag(flags, i, "Inflation data insufficient (< 2 years)")

    df["data_quality_flag"] = flags

    # Sort by descending risk (NaN last).
    df = df.sort_values("social_stability_risk", ascending=False, na_position="last")
    df = df.reset_index(drop=True)

    n_high = int((df["social_stability_risk"].fillna(0) >= 0.6).sum())
    n_warn = int((df["data_quality_flag"] != "").sum())
    log.info(
        "Stability table built: %d countries, %d with risk >= 0.6, %d with data warnings",
        len(df), n_high, n_warn,
    )
    return df


def _append_flag(flags: list[str], idx: int, message: str) -> None:
    """Append *message* to ``flags[idx]``, separated by '; ' if non-empty."""
    if flags[idx]:
        flags[idx] = f"{flags[idx]}; {message}"
    else:
        flags[idx] = message


def compute_fx_adjustment(df: pd.DataFrame, fx_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the FX channel adjustment to social stability risk scores.

    For countries where ``fx_channel_relevant`` is True, adds an FX-driven
    component that captures the pass-through of oil price shocks via currency
    depreciation to imported-goods inflation.

    Formula:
        fx_adjustment_value = fx_channel_weight × fiscal_stress_score

    Interpretation: when fiscal stress is high (Brent well below breakeven)
    *and* the country has a managed or freely floating exchange rate, currency
    depreciation amplifies import-price inflation beyond what is already
    captured by the direct food-import and inflation-volatility components.
    For GCC hard-peg countries, fx_channel_weight=0.0 so the adjustment is
    always zero.

    The adjusted ``social_stability_risk`` is clamped to [0, 1].

    Args:
        df:    Output of :func:`build_stability_table` containing
               ``social_stability_risk`` and ``fiscal_stress_score`` columns.
        fx_df: Output of :func:`load_fx_channel` (one row per country).

    Returns:
        Copy of *df* with three new columns:
          ``fx_adjusted``        (bool)  — True where fx_adjustment_value > 0
          ``fx_adjustment_value`` (float) — raw adjustment added to risk score
          ``social_stability_risk`` updated in place (original + adjustment,
          clamped to [0, 1])
    """
    df = df.copy()
    df = df.merge(
        fx_df[["country_code_a3", "fx_channel_relevant", "fx_channel_weight"]],
        on="country_code_a3",
        how="left",
    )

    df["fx_channel_relevant"] = df["fx_channel_relevant"].fillna(False).astype(bool)
    df["fx_channel_weight"]   = pd.to_numeric(
        df["fx_channel_weight"], errors="coerce"
    ).fillna(0.0)

    def _adj(row: pd.Series) -> float:
        if not row["fx_channel_relevant"]:
            return 0.0
        w   = float(row["fx_channel_weight"])
        fss = float(row["fiscal_stress_score"]) if not pd.isna(row["fiscal_stress_score"]) else 0.0
        return w * fss

    df["fx_adjustment_value"] = df.apply(_adj, axis=1)
    df["fx_adjusted"]         = df["fx_adjustment_value"] > 0.0
    df["social_stability_risk"] = (
        df["social_stability_risk"] + df["fx_adjustment_value"]
    ).clip(0.0, 1.0)

    n_adj = int(df["fx_adjusted"].sum())
    log.info(
        "FX adjustment applied: %d countries adjusted, max adjustment=%.4f",
        n_adj, df["fx_adjustment_value"].max(),
    )
    return df


# ── End-to-end orchestrator ──────────────────────────────────────────────────────

def run_social_stability(
    food_path:      Path = FOOD_PATH,
    breakeven_path: Path = BREAKEVEN_PATH,
    panel_path:     Path = PANEL_PATH,
    fx_path:        Path = FX_CHANNEL_PATH,
) -> dict:
    """End-to-end social stability pipeline.

    Args:
        food_path:      Path to ``food_security.csv``.
        breakeven_path: Path to ``fiscal_breakeven.csv``.
        panel_path:     Path to ``world_bank_panel.csv``.
        fx_path:        Path to ``fx_channel.csv``.

    Returns:
        dict with keys:
            ``'brent_live'``      : float
            ``'stress_table'``    : pd.DataFrame
            ``'stability_table'`` : pd.DataFrame (includes fx_adjusted columns)
    """
    breakeven_df  = load_breakeven(breakeven_path)
    brent_live    = fetch_brent_live()
    ytd_prices    = fetch_brent_ytd()
    stress_table  = build_stress_table(breakeven_df, brent_live, ytd_prices)
    inflation_df  = derive_inflation_vol(panel_path)
    food_df       = load_food_security(food_path)
    fx_df         = load_fx_channel(fx_path)
    stability_tbl = build_stability_table(food_df, stress_table, inflation_df)
    stability_tbl = compute_fx_adjustment(stability_tbl, fx_df)

    return {
        "brent_live":      brent_live,
        "stress_table":    stress_table,
        "stability_table": stability_tbl,
    }
