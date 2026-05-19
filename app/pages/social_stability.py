"""
Social Stability Risk Monitor

Answers: "Which MENA countries face the greatest social stability risk,
combining food import dependency, fiscal stress, and inflation volatility?"

Social stability risk is a composite 0–1 indicator:
  social_stability_risk = 0.5 × food_security_exposure
                        + 0.3 × fiscal_stress_score
                        + 0.2 × norm(inflation_volatility)

Page sections
-------------
  KPI cards    — highest-risk country, median risk, high-risk count, warning count
  Bar chart    — countries ranked by social_stability_risk
  Scatter      — food exposure vs fiscal stress, bubble = inflation, colour = risk
  Full table   — all 14 countries with data_quality_flag column
  Methodology  — formulas, weight rationale, data caveats

Run standalone (from project root):
    streamlit run app/pages/social_stability.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# ── Make project root importable ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app.export import make_csv_download_button
from src.model.fiscal_stress import (
    build_stress_table,
    fetch_brent_live,
    fetch_brent_ytd,
    load_breakeven,
)
from src.model.social_stability import (
    _W_CEREAL_DEP,
    _W_FISCAL,
    _W_FOOD_EXP,
    _W_FOOD_IMPORTS,
    _W_INFLATION,
    DRIVER_FISCAL,
    DRIVER_FOOD,
    DRIVER_INFLATION,
    DRIVER_MIXED,
    build_stability_table,
    derive_inflation_vol,
    load_food_security,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Social Stability Risk",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT          = Path(__file__).resolve().parents[2]
_BREAKEVEN_CSV = _ROOT / "data" / "reference" / "fiscal_breakeven.csv"
_FOOD_CSV      = _ROOT / "data" / "reference" / "food_security.csv"
_PANEL_CSV     = _ROOT / "data" / "processed" / "world_bank_panel.csv"

# ── Visual constants ───────────────────────────────────────────────────────────
# Risk colour ramp (threshold-based, mirroring traffic-light palette).
_RISK_COLOUR: dict[str, str] = {
    "critical": "#7B0000",   # risk >= 0.7
    "high":     "#d62728",   # risk >= 0.5
    "medium":   "#ff7f0e",   # risk >= 0.3
    "low":      "#2ca02c",   # risk <  0.3
    "na":       "#aaaaaa",
}

# Driver colours for scatter annotation.
_DRIVER_COLOUR: dict[str, str] = {
    DRIVER_FOOD:      "#d62728",
    DRIVER_FISCAL:    "#ff7f0e",
    DRIVER_INFLATION: "#1f77b4",
    DRIVER_MIXED:     "#9467bd",
}

# High-risk threshold used in KPI and table highlighting.
_HIGH_RISK_THRESHOLD = 0.6


def _risk_colour(risk: float) -> str:
    if math.isnan(risk):
        return _RISK_COLOUR["na"]
    if risk >= 0.7:
        return _RISK_COLOUR["critical"]
    if risk >= 0.5:
        return _RISK_COLOUR["high"]
    if risk >= 0.3:
        return _RISK_COLOUR["medium"]
    return _RISK_COLOUR["low"]


def _risk_label(risk: float) -> str:
    if math.isnan(risk):
        return "N/A"
    if risk >= 0.7:
        return "Critical"
    if risk >= 0.5:
        return "High"
    if risk >= 0.3:
        return "Medium"
    return "Low"


# ── Cached data loading ────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading food security reference data…")
def _load_food() -> pd.DataFrame:
    return load_food_security(_FOOD_CSV)


@st.cache_data(show_spinner="Computing inflation volatility from panel…")
def _load_inflation() -> pd.DataFrame:
    return derive_inflation_vol(_PANEL_CSV)


@st.cache_data(show_spinner="Loading fiscal breakeven data…")
def _load_breakeven_df() -> pd.DataFrame:
    return load_breakeven(_BREAKEVEN_CSV)


@st.cache_data(ttl=3600, show_spinner="Fetching live Brent price…")
def _fetch_brent_live() -> float:
    return fetch_brent_live()


@st.cache_data(ttl=3600, show_spinner="Fetching YTD Brent history…")
def _fetch_brent_ytd() -> pd.DataFrame:
    return fetch_brent_ytd()


# ── Guards ─────────────────────────────────────────────────────────────────────

if not _BREAKEVEN_CSV.exists():
    st.error(
        f"Fiscal breakeven reference file not found: `{_BREAKEVEN_CSV.relative_to(_ROOT)}`  \n"
        "Run the data pipeline before launching this page."
    )
    st.stop()

if not _FOOD_CSV.exists():
    st.error(
        f"Food security reference file not found: `{_FOOD_CSV.relative_to(_ROOT)}`  \n"
        "Expected: `data/reference/food_security.csv`"
    )
    st.stop()

if not _PANEL_CSV.exists():
    st.error(
        f"World Bank panel not found: `{_PANEL_CSV.relative_to(_ROOT)}`  \n"
        "Run `python src/data/fetch_world_bank.py` and `python src/data/clean_world_bank.py`."
    )
    st.stop()

# ── Load data ──────────────────────────────────────────────────────────────────

food_df       = _load_food()
inflation_df  = _load_inflation()
breakeven_df  = _load_breakeven_df()
brent_live    = _fetch_brent_live()
ytd_prices    = _fetch_brent_ytd()
stress_table  = build_stress_table(breakeven_df, brent_live, ytd_prices)
df            = build_stability_table(food_df, stress_table, inflation_df)

_brent_ok = not math.isnan(brent_live)

# ── Header ─────────────────────────────────────────────────────────────────────

st.title("Social Stability Risk Monitor")
st.caption(
    "Composite 0–1 risk indicator: food security exposure, fiscal stress, "
    "and inflation volatility — 14 MENA countries.  "
    "All values are estimates; see Methodology."
)

if not _brent_ok:
    st.warning(
        "Live Brent price unavailable — fiscal stress scores are computed from "
        "last known price.  Check network connectivity or Yahoo Finance (BZ=F)."
    )
else:
    st.info(f"Live Brent: **${brent_live:.2f} / bbl**")

# ── KPI row ────────────────────────────────────────────────────────────────────

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

_valid       = df[df["social_stability_risk"].notna()]
_n_high      = int((_valid["social_stability_risk"] >= _HIGH_RISK_THRESHOLD).sum())
_n_warn      = int((df["data_quality_flag"] != "").sum())
_n_low_conf  = int((df.get("confidence", pd.Series(dtype=str)).str.lower() == "low").sum())
_median_risk = float(_valid["social_stability_risk"].median()) if len(_valid) else float("nan")

with kpi1:
    if len(_valid):
        top = _valid.iloc[0]
        st.metric(
            "Highest Risk Country",
            f"{top['country_label']}  ({float(top['social_stability_risk']):.2f})",
            help=f"Main driver: {top['main_risk_driver']}",
        )
    else:
        st.metric("Highest Risk Country", "N/A")

with kpi2:
    st.metric(
        "Median Risk Score",
        f"{_median_risk:.2f}" if not math.isnan(_median_risk) else "N/A",
        help="Median social_stability_risk across all 14 countries",
    )

with kpi3:
    st.metric(
        f"Risk >= {_HIGH_RISK_THRESHOLD:.1f}",
        str(_n_high),
        help=f"Countries with social_stability_risk at or above {_HIGH_RISK_THRESHOLD}",
    )

with kpi4:
    st.metric(
        "Data Warnings",
        str(_n_warn),
        help="Countries with missing cereal data, inflation data, or provenance gaps",
    )

with kpi5:
    st.metric(
        "Low-Confidence Rows",
        str(_n_low_conf),
        help="Countries where food/cereal data confidence is rated 'low' in source CSV",
    )

st.divider()

# ── Bar chart — ranked by social stability risk ────────────────────────────────

st.subheader("Countries Ranked by Social Stability Risk")

_chart_df = df[df["social_stability_risk"].notna()].copy()
_bar_colours = [_risk_colour(float(r)) for r in _chart_df["social_stability_risk"]]

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    x=_chart_df["social_stability_risk"],
    y=_chart_df["country_label"],
    orientation="h",
    marker_color=_bar_colours,
    customdata=np.column_stack([
        _chart_df["main_risk_driver"],
        _chart_df["food_security_exposure"].fillna(float("nan")),
        _chart_df["fiscal_stress_score"],
        _chart_df["inflation_volatility_norm"].fillna(float("nan")),
        _chart_df["data_quality_flag"],
    ]),
    hovertemplate=(
        "<b>%{y}</b><br>"
        "Social Stability Risk: %{x:.3f}<br>"
        "Main Driver: %{customdata[0]}<br>"
        "Food Security Exposure: %{customdata[1]:.3f}<br>"
        "Fiscal Stress Score: %{customdata[2]:.2f}<br>"
        "Inflation Volatility (norm): %{customdata[3]:.3f}<br>"
        "Data Flag: %{customdata[4]}"
        "<extra></extra>"
    ),
))

fig_bar.add_vline(
    x=_HIGH_RISK_THRESHOLD,
    line_dash="dot",
    line_color="#888888",
    annotation_text=f"High-risk threshold ({_HIGH_RISK_THRESHOLD})",
    annotation_position="top right",
)

fig_bar.update_layout(
    height=460,
    margin=dict(l=10, r=20, t=30, b=40),
    xaxis=dict(
        title="Social Stability Risk (0–1)",
        range=[0, 1.05],
        tickformat=".2f",
    ),
    yaxis=dict(
        autorange="reversed",
        title="",
    ),
    showlegend=False,
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis_gridcolor="#eeeeee",
)

st.plotly_chart(fig_bar, use_container_width=True)

# Colour legend
c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(f'<span style="color:{_RISK_COLOUR["critical"]}">&#9632;</span> Critical (>= 0.7)',  unsafe_allow_html=True)
c2.markdown(f'<span style="color:{_RISK_COLOUR["high"]}">&#9632;</span> High (0.5–0.7)',         unsafe_allow_html=True)
c3.markdown(f'<span style="color:{_RISK_COLOUR["medium"]}">&#9632;</span> Medium (0.3–0.5)',     unsafe_allow_html=True)
c4.markdown(f'<span style="color:{_RISK_COLOUR["low"]}">&#9632;</span> Low (< 0.3)',             unsafe_allow_html=True)
c5.markdown(f'<span style="color:{_RISK_COLOUR["na"]}">&#9632;</span> N/A',                      unsafe_allow_html=True)

st.divider()

# ── Scatter — food exposure vs fiscal stress (bubble = inflation) ───────────────

st.subheader("Food Exposure vs Fiscal Stress")
st.caption(
    "Bubble size = inflation volatility (normalised).  "
    "Colour = social stability risk score."
)

_scatter_df = df[
    df["food_security_exposure"].notna() & df["fiscal_stress_score"].notna()
].copy()

# Ensure bubble size is always visible (min size 4%).
_scatter_df["_bubble_size"] = (
    _scatter_df["inflation_volatility_norm"].fillna(0.0) * 0.95 + 0.05
)

fig_scatter = px.scatter(
    _scatter_df,
    x="fiscal_stress_score",
    y="food_security_exposure",
    size="_bubble_size",
    color="social_stability_risk",
    hover_name="country_label",
    hover_data={
        "main_risk_driver":       True,
        "fiscal_stress_score":    ":.2f",
        "food_security_exposure": ":.3f",
        "inflation_volatility":   ":.2f",
        "social_stability_risk":  ":.3f",
        "_bubble_size":           False,
    },
    color_continuous_scale="RdYlGn_r",
    range_color=[0.0, 1.0],
    size_max=55,
    labels={
        "fiscal_stress_score":    "Fiscal Stress Score (0=none, 1=severe)",
        "food_security_exposure": "Food Security Exposure (0–1)",
        "social_stability_risk":  "Social Stability Risk",
    },
)

fig_scatter.update_layout(
    height=480,
    margin=dict(l=10, r=10, t=30, b=40),
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(
        range=[-0.05, 1.15],
        gridcolor="#eeeeee",
        zeroline=True,
        zerolinecolor="#dddddd",
    ),
    yaxis=dict(
        range=[-0.05, 1.10],
        gridcolor="#eeeeee",
        zeroline=True,
        zerolinecolor="#dddddd",
    ),
    coloraxis_colorbar=dict(title="Risk"),
)

st.plotly_chart(fig_scatter, use_container_width=True)
st.caption(
    "Upper-right quadrant = high food exposure AND high fiscal stress.  "
    "Large bubbles = high inflation volatility.  "
    f"Bubble size: inflation volatility (norm); small minimum applied so all countries are visible."
)

st.divider()

# ── Full table ──────────────────────────────────────────────────────────────────

st.subheader("Full Country Table")

_display_cols = {
    "country_label":                  "Country",
    "social_stability_risk":          "Risk Score",
    "main_risk_driver":               "Main Driver",
    "food_security_exposure":         "Food Exposure",
    "food_imports_pct_merch_imports": "Food Imports % Merch",
    "cereal_import_dependency":       "Cereal Dep %",
    "fiscal_stress_score":            "Fiscal Score (cont.)",
    "stress_status":                  "Fiscal Status",
    "inflation_volatility":           "Infl. Vol (std %)",
    "inflation_volatility_norm":      "Infl. Norm (winsor.)",
    "confidence":                     "Confidence",
    "data_quality_flag":              "Data Flag",
}

_tbl_cols_available = [c for c in _display_cols.keys() if c in df.columns]
_tbl = df[_tbl_cols_available].rename(columns=_display_cols).copy()

# Format numeric columns.
for col, fmt in [
    ("Risk Score",              "{:.3f}"),
    ("Food Exposure",           "{:.3f}"),
    ("Food Imports % Merch",    "{:.1f}"),
    ("Cereal Dep %",            "{:.0f}"),
    ("Fiscal Score (cont.)",    "{:.3f}"),
    ("Infl. Vol (std %)",       "{:.2f}"),
    ("Infl. Norm (winsor.)",    "{:.3f}"),
]:
    if col in _tbl.columns:
        _tbl[col] = _tbl[col].apply(
            lambda v: fmt.format(float(v)) if pd.notna(v) and not (isinstance(v, float) and math.isnan(v)) else "N/A"
        )

st.dataframe(
    _tbl,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Risk Score":     st.column_config.TextColumn("Risk Score", width="small"),
        "Main Driver":    st.column_config.TextColumn("Main Driver", width="medium"),
        "Data Flag":      st.column_config.TextColumn("Data Flag", width="large"),
    },
)
make_csv_download_button(_tbl, "social_stability_table.csv", "Download table as CSV")

_n_warn_shown = int((df["data_quality_flag"] != "").sum())
if _n_warn_shown:
    st.caption(
        f"**{_n_warn_shown} countries have data quality flags** — see 'Data Flag' column.  "
        "Scores for flagged countries are still computed but should be treated with caution."
    )

st.divider()

# ── Methodology ────────────────────────────────────────────────────────────────

with st.expander("Methodology & Caveats", expanded=False):
    _norm_method = df["normalization_method"].iloc[0] if "normalization_method" in df.columns and len(df) else "winsorize_p5_p95_minmax"
    st.markdown(f"""
### Composite Social Stability Risk

**Formula:**
```
social_stability_risk =
    {_W_FOOD_EXP} × food_security_exposure
  + {_W_FISCAL}   × fiscal_stress_score
  + {_W_INFLATION} × inflation_volatility_norm
```

All components are in [0, 1].  Higher score = higher social stability risk.

---

### Food Security Exposure

```
food_security_exposure =
    {_W_FOOD_IMPORTS} × norm(food_imports_pct_merch_imports)
  + {_W_CEREAL_DEP}   × norm(cereal_import_dependency)
```

`norm()` = standard min-max across all 14 countries.

**Fallback:** if cereal_import_dependency is absent for a country, the weight
is redistributed fully to food_imports_pct (weight = 1.0).  A `data_quality_flag`
is set; the score is still computed but flagged explicitly — no silent imputation.

**Net importers (Jordan, Lebanon, Morocco) are included.**  Their social stability
risk is not forced to zero — food import shocks hit importers directly.

---

### Fiscal Stress Score — Continuous Formula

```
fiscal_stress_score =
    min(1, max(0, (fiscal_breakeven_usd - brent_live_usd) / fiscal_breakeven_usd))
```

This replaces a former categorical mapping (Red=1.0, Amber=0.5, Green=0.0) with
a continuous 0–1 score that captures the *degree* of fiscal shortfall:
- When Brent = breakeven → score = 0 (no stress)
- When Brent << breakeven → score approaches 1 (severe stress)
- When Brent > breakeven → negative value, clamped to 0

Net importers (Gray, fiscal_breakeven_usd = 0) receive 0.0.  They can still
accumulate non-zero overall risk via food exposure and inflation components.

---

### Inflation Volatility Normalisation

Raw inflation volatility = `std(FP.CPI.TOTL.ZG, 2000–2024)` from World Bank panel.

Normalisation method: **`{_norm_method}`**

Lebanon's CPI volatility (std ≈ 73 pp) is an extreme outlier that would compress
all other countries to near zero under standard min-max.  Percentile winsorisation
(p5–p95) caps values at the 5th and 95th percentile before applying min-max,
preserving meaningful differentiation across the rest of the sample.

The raw `inflation_volatility` column is unchanged; only `inflation_volatility_norm`
uses the winsorised scale.

---

### Main Risk Driver

Determined by the largest of the three weighted contributions:

| Label | Condition |
|---|---|
| `{DRIVER_FOOD}` | 0.5 × food_security_exposure is the largest |
| `{DRIVER_FISCAL}` | 0.3 × fiscal_stress_score is the largest |
| `{DRIVER_INFLATION}` | 0.2 × inflation_volatility_norm is the largest |
| `{DRIVER_MIXED}` | Top two are within 5 % of total risk, or all contributions = 0 |

---

### Data Sources

| Component | Source | Reference year | Indicator |
|---|---|---|---|
| Food imports % merch | World Bank WDI | 2022 est. | TM.VAL.FOOD.ZS.UN |
| Cereal import dependency | FAO FAOSTAT Food Balance Sheets | 2021 est. | Cereal import dep. ratio |
| Inflation volatility | World Bank WDI panel | 2000–2024 | FP.CPI.TOTL.ZG std |
| Fiscal stress score | IMF Article IV + yfinance BZ=F | 2023 est. + live | fiscal_breakeven.csv |

All values are preliminary estimates.  `is_estimate = True` in all reference CSVs.
Low-confidence rows are counted in the KPI row above.

---

### Caveats

- **Iran:** Cereal dependency ≈ 35% (lowest in sample); domestic wheat production
  partially offsets imports. Sanctions distort actual trade data — rated **low confidence**.
- **Lebanon:** Food imports % is the highest in sample; post-2019 economic crisis
  means data quality is poor — rated **low confidence**.
- **Libya:** Conflict-affected supply chains; dual-government structure adds further
  uncertainty — rated **low confidence**.
- **Fiscal score sensitivity:** The continuous formula is sensitive to the live Brent
  price. A $1/bbl change in Brent shifts fiscal scores for all non-Gray exporters.
- **No causal claims:** This is a relative risk indicator for 14 countries, not a
  predictive model.  Read alongside the Fiscal Stress and Reserve Runway monitors.
""")
