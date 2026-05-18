"""
Oil Price Shock Transmission Chain — multi-stage contagion model.

Chain formula
-------------
Six linked stages propagate an oil price shock through each MENA economy:

  Stage 1  Oil price → Fiscal revenue delta (pp of GDP)
    Exporters:  fiscal_delta = oil_rents_%GDP  ×  ΔP/P
    Importers:  fiscal_delta = −imports_%GDP  ×  oil_import_share  ×  ΔP/P

  Stage 2  Fiscal revenue → Government spending pressure (pp of GDP)
    spending_pressure = max(−fiscal_delta, 0)
    [Only negative fiscal shocks force austerity; positive shocks give space.]

  Stage 3  Spending pressure → Subsidy strain score  [0, 1]
    raw_strain     = norm(fossil_fuel_%) × norm(energy_per_capita) × norm(spending_pressure)
    subsidy_strain = min-max-norm(raw_strain)

  Stage 4  Subsidy strain → Consumer price pass-through factor  [0, 1]
    base_rate      = 0.15 (exporters, high domestic subsidies)
                   | 0.40 (importers, direct market exposure)
    passthrough    = base_rate × (1 + subsidy_strain × 0.50)
    [Coady et al. IMF 2015; MENA energy pricing reform studies 2018–2022]

  Stage 5  Pass-through → Oil-driven inflation component (pp)
    oil_inflation_pp = passthrough_factor × brent_yoy_%
    cpi_actual_pct   = observed FP_CPI_TOTL_ZG

  Stage 6  Inflation → Employment pressure index  [0, 1]
    raw_emp  = norm(|oil_inflation_pp|) × norm(spending_pressure) × (1 − norm(oil_rents/GDP))
    employment_pressure = min-max-norm(raw_emp)
    [High non-oil share → private sector more exposed to purchasing-power erosion]

Composite transmission severity  [0, 1]
    severity = 0.30×norm(|fiscal_delta|) + 0.20×norm(spending_pressure)
             + 0.15×subsidy_strain        + 0.15×passthrough_factor
             + 0.12×norm(|oil_inflation|)  + 0.08×employment_pressure

Panel inputs
------------
All six stages are derived from ``data/processed/world_bank_panel.csv``:

    Column                    Stage   Description
    ────────────────────────────────────────────────────────────────────
    NY_GDP_PETR_RT_ZS          1      Oil rents (% of GDP)
    BM_GSR_MRCH_CD             1      Merchandise imports (USD)
    NY_GDP_MKTP_CD             1,3,6  GDP at market prices (USD)
    EG_USE_COMM_FO_ZS          3      Fossil-fuel % of energy use
    EG_USE_PCAP_KG_OE          3      Energy use per capita (kg OE)
    TX_VAL_FUEL_ZS_UN          —      Fuel exports % (exporter classifier)
    FP_CPI_TOTL_ZG             5      Actual CPI inflation (%)

Brent crude price series (annual, USD/bbl) is embedded as a fallback
constant; the same values are used in ``app/pages/price_analysis.py``.
Oil import shares for net importers come from IEA/WITS country profiles
(2019–2022 avg), consistent with ``app/pages/supply_chain.py``.

Output
------
``outputs/tables/chain_transmission.csv`` — one row per country-year:

    country_code, country_code_a3, country_name, year,
    is_exporter,
    brent_price_usd, brent_yoy_pct,
    fiscal_delta_pp,
    spending_pressure_pp,
    subsidy_strain_score,
    passthrough_factor,
    oil_inflation_pp,
    cpi_actual_pct,
    employment_pressure_score,
    transmission_severity

Usage (from project root)::

    python -m src.model.chain_model
    python -m src.model.chain_model --panel data/processed/world_bank_panel.csv
    python -m src.model.chain_model --output outputs/tables/chain_transmission.csv
    python -m src.model.chain_model --log-level DEBUG

Exit codes:
    0  chain computed and saved successfully
    1  unrecoverable error (file not found, missing columns)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PANEL_PATH:  Path = Path("data/processed/world_bank_panel.csv")
OUTPUT_PATH: Path = Path("outputs/tables/chain_transmission.csv")

# Annual Brent crude price (EIA / World Bank, USD/bbl, 2000–2024).
# Matches the fallback series in app/pages/price_analysis.py.
_BRENT: dict[int, float] = {
    2000: 28.51, 2001: 24.44, 2002: 25.02, 2003: 28.85, 2004: 38.27,
    2005: 54.52, 2006: 65.14, 2007: 72.44, 2008: 96.94, 2009: 61.51,
    2010: 79.50, 2011: 111.26, 2012: 111.63, 2013: 108.66, 2014: 98.97,
    2015: 52.39, 2016: 43.55, 2017: 54.25, 2018: 71.69, 2019: 64.37,
    2020: 41.96, 2021: 70.68, 2022: 100.93, 2023: 82.17, 2024: 80.30,
}

# Oil/energy products as share of total merchandise imports for net importers.
# IEA country profiles / World Bank WITS (2019-2022 avg).
# Consistent with app/pages/supply_chain.py _OIL_IMPORT_SHARE.
_OIL_IMPORT_SHARE: dict[str, float] = {
    "JOR": 0.22,   # Jordan ~22%
    "LBN": 0.20,   # Lebanon ~20%
    "MAR": 0.25,   # Morocco ~25%
    "EGY": 0.13,   # Egypt ~13% (partial domestic producer)
}
# Default oil import share for importers not listed above
_OIL_IMPORT_SHARE_DEFAULT = 0.10

# Exporter classification: long-run average fuel exports exceeds this
# share of total merchandise exports (TX_VAL_FUEL_ZS_UN).
_EXPORTER_THRESHOLD = 20.0

# Hardcoded fallback exporter set (used only if TX_VAL_FUEL_ZS_UN is absent)
_KNOWN_EXPORTERS: frozenset[str] = frozenset({
    "DZA", "BHR", "IRN", "IRQ", "KWT", "LBY", "OMN", "QAT", "SAU", "ARE",
})

# Stage 4 consumer pass-through base rates (Coady et al., IMF 2015).
# Exporters have heavily subsidised domestic energy → lower pass-through.
_PASSTHROUGH_EXPORTER = 0.15
_PASSTHROUGH_IMPORTER = 0.40
# Subsidy amplifier: each unit of subsidy strain boosts pass-through by this.
_SUBSIDY_AMPLIFIER    = 0.50

# Stage 3 normalisation denominator: maximum possible product of
# (suez_import_pct × oil_import_share) = 30 × 0.25 = 7.5 (Morocco).
# Kept here for reference; Stage 3 uses within-dataset min-max instead.

# Severity composite weights (must sum to 1.0)
_SEVERITY_WEIGHTS: dict[str, float] = {
    "fiscal":      0.30,
    "spending":    0.20,
    "subsidy":     0.15,
    "passthrough": 0.15,
    "inflation":   0.12,
    "employment":  0.08,
}

# Required panel columns
_REQUIRED_COLS: list[str] = [
    "country_code", "country_code_a3", "country_name", "year",
    "NY_GDP_PETR_RT_ZS", "FP_CPI_TOTL_ZG", "NY_GDP_MKTP_CD",
]

# Output column order (canonical)
_OUTPUT_COLS: list[str] = [
    "country_code", "country_code_a3", "country_name", "year",
    "is_exporter",
    "brent_price_usd", "brent_yoy_pct",
    "fiscal_delta_pp",
    "spending_pressure_pp",
    "subsidy_strain_score",
    "passthrough_factor",
    "oil_inflation_pp",
    "cpi_actual_pct",
    "employment_pressure_score",
    "transmission_severity",
]


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def _norm(s: pd.Series) -> pd.Series:
    """Min-max normalise *s* to [0, 1].  Returns zeros when all values are equal.

    NaN values are preserved.  Normalisation is computed over the full
    passed Series (i.e. across all country-years simultaneously), which
    captures relative severity across the panel rather than within a
    single country.
    """
    mn = s.min(skipna=True)
    mx = s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.0, index=s.index)
    return ((s - mn) / (mx - mn)).clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_panel(path: Path) -> pd.DataFrame:
    """Load and validate the World Bank country-year panel.

    Args:
        path: Path to ``world_bank_panel.csv``.

    Returns:
        Raw panel DataFrame.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If any required column is absent.
    """
    if not path.exists():
        raise FileNotFoundError(f"Panel data not found: {path}")

    df = pd.read_csv(path)
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Panel missing required columns: {missing}")

    log.info(
        "Panel loaded: %d rows × %d cols  |  %d countries, years %d–%d",
        len(df), len(df.columns),
        df["country_code"].nunique(),
        int(df["year"].min()), int(df["year"].max()),
    )
    return df


# ---------------------------------------------------------------------------
# Pre-processing: Brent prices and exporter classification
# ---------------------------------------------------------------------------

def attach_brent(df: pd.DataFrame) -> pd.DataFrame:
    """Merge embedded annual Brent prices onto the panel and compute YoY change.

    The ``brent_yoy_pct`` column is computed from the *global* Brent series
    (not per-country), so every country in the same year sees the same price
    shock input at Stage 1.

    Args:
        df: Country-year panel.

    Returns:
        Panel with ``brent_price_usd`` and ``brent_yoy_pct`` columns appended.
    """
    brent_df = pd.DataFrame(
        [{"year": y, "brent_price_usd": p} for y, p in sorted(_BRENT.items())]
    )
    brent_df["brent_yoy_pct"] = brent_df["brent_price_usd"].pct_change().mul(100)

    df = df.merge(brent_df, on="year", how="left")
    matched = df["brent_price_usd"].notna().sum() // max(df["country_code"].nunique(), 1)
    log.debug("Brent series merged: %d years with price data per country.", matched)
    return df


def classify_exporters(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``is_exporter`` column based on long-run average fuel export share.

    A country is a net exporter if its **mean** fuel-export share of
    merchandise exports (``TX_VAL_FUEL_ZS_UN``) exceeds
    ``_EXPORTER_THRESHOLD`` (20 %).  The mean is taken across all years
    with valid data to provide a stable, shock-invariant classification.

    Falls back to a hardcoded exporter set when the column is absent.

    Args:
        df: Country-year panel with Brent columns already attached.

    Returns:
        Panel with boolean ``is_exporter`` column.
    """
    if "TX_VAL_FUEL_ZS_UN" in df.columns:
        avg_fuel = df.groupby("country_code_a3")["TX_VAL_FUEL_ZS_UN"].transform("mean")
        df["is_exporter"] = avg_fuel > _EXPORTER_THRESHOLD
    else:
        df["is_exporter"] = df["country_code_a3"].isin(_KNOWN_EXPORTERS)
        log.warning(
            "TX_VAL_FUEL_ZS_UN absent — using hardcoded exporter set (%d countries).",
            len(_KNOWN_EXPORTERS),
        )

    n_exp = df[df["is_exporter"]]["country_code_a3"].nunique()
    n_imp = df[~df["is_exporter"]]["country_code_a3"].nunique()
    log.info("Exporter classification: %d exporters, %d importers.", n_exp, n_imp)
    return df


# ---------------------------------------------------------------------------
# Stage 1 — Oil price → Fiscal revenue delta (pp of GDP)
# ---------------------------------------------------------------------------

def compute_stage1_fiscal(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 1: Estimate the direct fiscal revenue impact of the oil price change.

    For each country-year, the oil price YoY change (``brent_yoy_pct``) is
    translated into an estimated change in government oil-related revenue as
    a share of GDP.

    Exporter path (oil revenue gain/loss):
        ``fiscal_delta = oil_rents_pct_gdp × (brent_yoy_pct / 100)``

        Oil rents as % GDP (``NY_GDP_PETR_RT_ZS``) proxies the government's
        oil-revenue exposure.  A 30% price rise on 20% oil rents → +6 pp.

    Importer path (increased import bill):
        ``fiscal_delta = −imports_pct_gdp × oil_import_share × (brent_yoy_pct / 100)``

        The additional cost of imported oil is subtracted from fiscal space.
        ``oil_import_share`` is the oil-product fraction of total merchandise
        imports (IEA/WITS, 2019–2022 avg).

    Rows where ``brent_yoy_pct`` is NaN (first observed year) receive NaN.

    Args:
        df: Panel with Brent and exporter columns.

    Returns:
        Panel with ``fiscal_delta_pp`` column.
    """
    shock_frac = df["brent_yoy_pct"].div(100)

    # Exporter: oil rent income change
    oil_rents = df["NY_GDP_PETR_RT_ZS"].fillna(0.0)
    exp_delta = oil_rents * shock_frac

    # Importer: higher import bill as % of GDP
    if "BM_GSR_MRCH_CD" in df.columns and "NY_GDP_MKTP_CD" in df.columns:
        gdp_nonzero = df["NY_GDP_MKTP_CD"].replace(0, np.nan)
        imports_pct = df["BM_GSR_MRCH_CD"].div(gdp_nonzero).mul(100).fillna(0.0)
    else:
        imports_pct = pd.Series(20.0, index=df.index)
        log.warning("BM_GSR_MRCH_CD or NY_GDP_MKTP_CD absent — defaulting imports/GDP to 20%%.")

    oil_share = df["country_code_a3"].map(_OIL_IMPORT_SHARE).fillna(_OIL_IMPORT_SHARE_DEFAULT)
    imp_delta = -imports_pct * oil_share * shock_frac

    df["fiscal_delta_pp"] = np.where(df["is_exporter"], exp_delta, imp_delta)
    # Propagate NaN from missing price change
    df.loc[df["brent_yoy_pct"].isna(), "fiscal_delta_pp"] = np.nan

    valid = df["fiscal_delta_pp"].notna()
    log.info(
        "Stage 1 (fiscal delta): mean=%+.3f pp, min=%+.3f pp, max=%+.3f pp  (%d obs)",
        df.loc[valid, "fiscal_delta_pp"].mean(),
        df.loc[valid, "fiscal_delta_pp"].min(),
        df.loc[valid, "fiscal_delta_pp"].max(),
        valid.sum(),
    )
    return df


# ---------------------------------------------------------------------------
# Stage 2 — Fiscal revenue → Government spending pressure (pp of GDP)
# ---------------------------------------------------------------------------

def compute_stage2_spending(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 2: Derive forced government spending pressure from fiscal shock.

    Spending pressure is the non-negative portion of a negative fiscal shock:
        ``spending_pressure_pp = max(−fiscal_delta_pp, 0)``

    A positive fiscal delta (price-rise windfall for exporters) yields zero
    pressure — governments can maintain or expand spending.  Only negative
    shocks (oil-price crash for exporters; price-rise cost for importers)
    generate actual austerity pressure.

    Args:
        df: Panel with ``fiscal_delta_pp``.

    Returns:
        Panel with ``spending_pressure_pp`` column.
    """
    df["spending_pressure_pp"] = (-df["fiscal_delta_pp"]).clip(lower=0.0)
    log.info(
        "Stage 2 (spending pressure): mean=%.3f pp, max=%.3f pp",
        df["spending_pressure_pp"].mean(skipna=True),
        df["spending_pressure_pp"].max(skipna=True),
    )
    return df


# ---------------------------------------------------------------------------
# Stage 3 — Spending pressure → Subsidy strain score  [0, 1]
# ---------------------------------------------------------------------------

def compute_stage3_subsidy(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 3: Map spending pressure to energy-subsidy strain.

    Governments with large energy-subsidy programmes face greater strain
    when fiscal space tightens.  The strain is higher when:
    - Fossil fuels dominate energy consumption (``EG_USE_COMM_FO_ZS`` high)
    - Per-capita energy use is high (``EG_USE_PCAP_KG_OE`` high — heavier
      subsidy burden per dollar of fiscal revenue lost)
    - Spending pressure is high (Stage 2)

    All three components are normalised to [0, 1] before multiplication so
    the composite reflects simultaneous extremes rather than a single driver.

    Args:
        df: Panel with ``spending_pressure_pp``.

    Returns:
        Panel with ``subsidy_strain_score`` column (min-max normalised).
    """
    if "EG_USE_COMM_FO_ZS" in df.columns:
        ff_pct = df["EG_USE_COMM_FO_ZS"].fillna(df["EG_USE_COMM_FO_ZS"].median(skipna=True))
    else:
        ff_pct = pd.Series(80.0, index=df.index)
        log.warning("EG_USE_COMM_FO_ZS absent — defaulting fossil-fuel share to 80%%.")

    if "EG_USE_PCAP_KG_OE" in df.columns:
        energy_cap = df["EG_USE_PCAP_KG_OE"].fillna(df["EG_USE_PCAP_KG_OE"].median(skipna=True))
    else:
        energy_cap = pd.Series(2000.0, index=df.index)
        log.warning("EG_USE_PCAP_KG_OE absent — defaulting energy per capita to 2000 kg OE.")

    fossil_norm  = _norm(ff_pct)
    energy_norm  = _norm(energy_cap)
    spend_norm   = _norm(df["spending_pressure_pp"].fillna(0.0))

    raw_strain = fossil_norm * energy_norm * spend_norm
    df["subsidy_strain_score"] = _norm(raw_strain)

    log.info(
        "Stage 3 (subsidy strain): mean=%.3f, max=%.3f",
        df["subsidy_strain_score"].mean(skipna=True),
        df["subsidy_strain_score"].max(skipna=True),
    )
    return df


# ---------------------------------------------------------------------------
# Stage 4 — Subsidy strain → Consumer price pass-through factor  [0, 1]
# ---------------------------------------------------------------------------

def compute_stage4_passthrough(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 4: Estimate the fraction of the oil price change reaching consumers.

    Base pass-through rates reflect MENA-specific energy pricing structures:
    - Exporters (15%): heavily subsidised domestic fuel/energy markets absorb
      most of the oil price movement before it reaches consumers.
    - Importers (40%): more directly exposed to world market prices, though
      partial subsidies still buffer some pass-through.

    Subsidy strain amplifies the base rate: when fiscal tightening forces
    governments to reduce subsidies, a larger share of the world price is
    passed to households:
        ``passthrough = base × (1 + subsidy_strain × 0.50)``

    The result is clipped to [0, 1].

    Args:
        df: Panel with ``subsidy_strain_score`` and ``is_exporter``.

    Returns:
        Panel with ``passthrough_factor`` column.
    """
    base = np.where(df["is_exporter"], _PASSTHROUGH_EXPORTER, _PASSTHROUGH_IMPORTER)
    df["passthrough_factor"] = (
        base * (1.0 + df["subsidy_strain_score"].fillna(0.0) * _SUBSIDY_AMPLIFIER)
    ).clip(0.0, 1.0)

    log.info(
        "Stage 4 (pass-through): exporter mean=%.3f, importer mean=%.3f",
        df.loc[df["is_exporter"],  "passthrough_factor"].mean(skipna=True),
        df.loc[~df["is_exporter"], "passthrough_factor"].mean(skipna=True),
    )
    return df


# ---------------------------------------------------------------------------
# Stage 5 — Pass-through → Oil-driven inflation component (pp)
# ---------------------------------------------------------------------------

def compute_stage5_inflation(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 5: Compute the oil-driven component of consumer price inflation.

    ``oil_inflation_pp = passthrough_factor × brent_yoy_pct``

    The observed CPI (``FP_CPI_TOTL_ZG``) is also carried through as
    ``cpi_actual_pct`` for comparison.  This allows the dashboard to show
    what fraction of actual inflation was attributable to the oil channel
    in each country-year.

    Rows with missing Brent YoY receive NaN for ``oil_inflation_pp``.

    Args:
        df: Panel with ``passthrough_factor`` and ``brent_yoy_pct``.

    Returns:
        Panel with ``oil_inflation_pp`` and ``cpi_actual_pct`` columns.
    """
    df["oil_inflation_pp"] = df["passthrough_factor"] * df["brent_yoy_pct"].fillna(0.0)
    df.loc[df["brent_yoy_pct"].isna(), "oil_inflation_pp"] = np.nan
    df["cpi_actual_pct"] = df["FP_CPI_TOTL_ZG"]

    valid = df["oil_inflation_pp"].notna()
    log.info(
        "Stage 5 (oil inflation): mean=%+.2f pp, range [%+.2f, %+.2f]  (%d obs)",
        df.loc[valid, "oil_inflation_pp"].mean(),
        df.loc[valid, "oil_inflation_pp"].min(),
        df.loc[valid, "oil_inflation_pp"].max(),
        valid.sum(),
    )
    return df


# ---------------------------------------------------------------------------
# Stage 6 — Inflation → Employment pressure index  [0, 1]
# ---------------------------------------------------------------------------

def compute_stage6_employment(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 6: Derive an employment pressure proxy from upstream chain effects.

    The proxy combines three amplifiers:

    1. ``|oil_inflation_pp|``: Larger price shocks erode household purchasing
       power and real wages, increasing employment instability.
    2. ``spending_pressure_pp``: Government austerity reduces public-sector
       employment and transfers, amplifying labour market strain.
    3. ``1 − norm(oil_rents/GDP)``: Countries with lower oil-sector GDP
       share have a larger private non-oil workforce exposed to income shocks.
       High oil-rent economies absorb shocks via sovereign wealth or transfers.

    The three factors are normalised and multiplied, then the product is
    normalised to [0, 1] across the full panel.

    Args:
        df: Panel with ``oil_inflation_pp``, ``spending_pressure_pp``,
            and ``NY_GDP_PETR_RT_ZS``.

    Returns:
        Panel with ``employment_pressure_score`` column.
    """
    oil_rents_norm  = _norm(df["NY_GDP_PETR_RT_ZS"].fillna(0.0))
    non_oil_share   = 1.0 - oil_rents_norm     # high oil rents → lower non-oil exposure
    infl_abs_norm   = _norm(df["oil_inflation_pp"].fillna(0.0).abs())
    spend_norm      = _norm(df["spending_pressure_pp"].fillna(0.0))

    raw_emp = infl_abs_norm * spend_norm * non_oil_share
    df["employment_pressure_score"] = _norm(raw_emp)

    log.info(
        "Stage 6 (employment pressure): mean=%.3f, max=%.3f",
        df["employment_pressure_score"].mean(skipna=True),
        df["employment_pressure_score"].max(skipna=True),
    )
    return df


# ---------------------------------------------------------------------------
# Composite transmission severity  [0, 1]
# ---------------------------------------------------------------------------

def compute_transmission_severity(df: pd.DataFrame) -> pd.DataFrame:
    """Combine all six stage outputs into a single transmission severity score.

    Severity weights (sum to 1.0):
        fiscal      0.30  — largest driver; determines scale of all downstream
        spending    0.20  — forced austerity amplifies every subsequent stage
        subsidy     0.15  — determines how much of the fiscal shock is absorbed
        passthrough 0.15  — determines how much reaches consumers
        inflation   0.12  — observed consumer impact
        employment  0.08  — lagged; most uncertain of the six proxies

    All inputs are normalised to [0, 1] before weighting so each component
    contributes proportionally regardless of its natural scale.

    Args:
        df: Panel with all six stage outputs.

    Returns:
        Panel with ``transmission_severity`` column ∈ [0, 1].
    """
    w = _SEVERITY_WEIGHTS
    fiscal_n  = _norm(df["fiscal_delta_pp"].abs().fillna(0.0))
    spend_n   = _norm(df["spending_pressure_pp"].fillna(0.0))
    infl_n    = _norm(df["oil_inflation_pp"].abs().fillna(0.0))

    df["transmission_severity"] = (
        w["fiscal"]      * fiscal_n
        + w["spending"]  * spend_n
        + w["subsidy"]   * df["subsidy_strain_score"].fillna(0.0)
        + w["passthrough"] * df["passthrough_factor"].fillna(0.0)
        + w["inflation"] * infl_n
        + w["employment"] * df["employment_pressure_score"].fillna(0.0)
    )
    log.info(
        "Composite severity: mean=%.3f, max=%.3f (country-year: %s %d)",
        df["transmission_severity"].mean(skipna=True),
        df["transmission_severity"].max(skipna=True),
        df.loc[df["transmission_severity"].idxmax(), "country_name"],
        int(df.loc[df["transmission_severity"].idxmax(), "year"]),
    )
    return df


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_chain(panel: pd.DataFrame) -> pd.DataFrame:
    """Run all six transmission stages in sequence.

    Args:
        panel: Raw country-year panel from :func:`load_panel`.

    Returns:
        Enriched DataFrame with all chain stage columns appended.
    """
    df = panel.copy()
    df = attach_brent(df)
    df = classify_exporters(df)
    df = compute_stage1_fiscal(df)
    df = compute_stage2_spending(df)
    df = compute_stage3_subsidy(df)
    df = compute_stage4_passthrough(df)
    df = compute_stage5_inflation(df)
    df = compute_stage6_employment(df)
    df = compute_transmission_severity(df)
    return df


# ---------------------------------------------------------------------------
# Output formatting & saving
# ---------------------------------------------------------------------------

def save_chain(df: pd.DataFrame, output_path: Path) -> Path:
    """Write the transmission table to *output_path* as a UTF-8 CSV.

    Columns are written in the canonical order defined by ``_OUTPUT_COLS``.
    Rows are sorted by ``country_code``, then ``year``.  Parent directories
    are created if they do not exist.

    Args:
        df: Enriched panel from :func:`run_chain`.
        output_path: Destination ``.csv`` path.

    Returns:
        The resolved :class:`~pathlib.Path` of the written file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in _OUTPUT_COLS if c in df.columns]
    out  = df[cols].sort_values(["country_code", "year"]).reset_index(drop=True)
    out.to_csv(output_path, index=False, encoding="utf-8", float_format="%.6f")
    log.info(
        "Chain transmission saved → %s  (%d rows × %d cols).",
        output_path, len(out), len(out.columns),
    )
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_chain_summary(df: pd.DataFrame) -> None:
    """Log a country-level summary table of average chain metrics to INFO."""
    summary = (
        df.groupby(["country_code_a3", "country_name"])
        .agg(
            is_exporter            = ("is_exporter",              "first"),
            avg_fiscal_delta       = ("fiscal_delta_pp",          "mean"),
            avg_spending_press     = ("spending_pressure_pp",     "mean"),
            avg_subsidy_strain     = ("subsidy_strain_score",     "mean"),
            avg_passthrough        = ("passthrough_factor",       "mean"),
            avg_oil_inflation      = ("oil_inflation_pp",         "mean"),
            avg_employment_press   = ("employment_pressure_score","mean"),
            avg_severity           = ("transmission_severity",    "mean"),
        )
        .reset_index()
        .sort_values("avg_severity", ascending=False)
    )

    log.info("─── Oil Shock Transmission Chain — Country Severity (2000–2024 avg) ───")
    log.info(
        "  %-3s  %-24s  %-3s  %8s  %8s  %8s  %8s  %8s",
        "A3", "Country", "Typ",
        "Fiscal", "Spend", "Passthru", "OilInfl", "Severity",
    )
    log.info("  " + "─" * 78)
    for _, row in summary.iterrows():
        log.info(
            "  %-3s  %-24s  %-3s  %+7.2fpp  %7.2fpp  %8.3f  %+7.2fpp  %8.3f",
            row["country_code_a3"],
            row["country_name"],
            "EXP" if row["is_exporter"] else "IMP",
            row["avg_fiscal_delta"],
            row["avg_spending_press"],
            row["avg_passthrough"],
            row["avg_oil_inflation"],
            row["avg_severity"],
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_chain(
    panel_path:  Path = PANEL_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """End-to-end chain pipeline: load → run → save.

    Args:
        panel_path:  Path to the cleaned World Bank panel CSV.
        output_path: Destination CSV path for the chain transmission table.

    Returns:
        Enriched transmission :class:`~pandas.DataFrame`
        (also written to *output_path*).

    Raises:
        FileNotFoundError: If *panel_path* does not exist.
        ValueError: If required columns are absent from the panel.
    """
    panel  = load_panel(panel_path)
    result = run_chain(panel)
    _print_chain_summary(result)
    save_chain(result, output_path)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.model.chain_model",
        description=(
            "Oil price shock transmission chain for 14 MENA economies.\n"
            "Runs six linked stages: fiscal → spending → subsidy → passthrough "
            "→ inflation → employment."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--panel",
        metavar="PATH",
        default=str(PANEL_PATH),
        help=f"Path to the cleaned panel CSV (default: {PANEL_PATH}).",
    )
    p.add_argument(
        "--output",
        metavar="PATH",
        default=str(OUTPUT_PATH),
        help=f"Destination CSV path (default: {OUTPUT_PATH}).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, run the chain pipeline, print summary."""
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        build_chain(
            panel_path=Path(args.panel),
            output_path=Path(args.output),
        )
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
