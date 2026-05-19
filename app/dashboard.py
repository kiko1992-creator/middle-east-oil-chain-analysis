"""
Middle East Oil Chain Analysis — Streamlit Dashboard

Loads:
    data/processed/world_bank_panel.csv   (country-year panel, 2000-2024)
    outputs/tables/ocvi_scores.csv        (OCVI country-level scores)

Run from project root:
    streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app.export import make_csv_download_button

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MENA Oil Chain Analysis",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT          = Path(__file__).resolve().parents[1]
PANEL_PATH     = _ROOT / "data" / "processed" / "world_bank_panel.csv"
OCVI_PATH      = _ROOT / "outputs" / "tables" / "ocvi_scores.csv"
CHAIN_PATH     = _ROOT / "outputs" / "tables" / "chain_transmission.csv"
BREAKEVEN_PATH = _ROOT / "data" / "reference" / "fiscal_breakeven.csv"
RESERVES_PATH  = _ROOT / "data" / "reference" / "swf_reserves.csv"
FOOD_PATH      = _ROOT / "data" / "reference" / "food_security.csv"
HIST_PATH      = _ROOT / "outputs" / "tables" / "historical_risk_index.csv"
SENS_PATH      = _ROOT / "outputs" / "tables" / "sensitivity_results.csv"
RETRO_PATH     = _ROOT / "outputs" / "tables" / "retrospective_2020.csv"
CV_PATH        = _ROOT / "outputs" / "tables" / "cross_validation.csv"

# Consistent 14-colour palette (one per country)
_PALETTE = px.colors.qualitative.D3 + px.colors.qualitative.Plotly

# World Bank long names → short display labels
_LABEL: dict[str, str] = {
    "Egypt, Arab Rep.":     "Egypt",
    "Iran, Islamic Rep.":   "Iran",
    "United Arab Emirates": "UAE",
}

def _label(name: str) -> str:
    return _LABEL.get(name, name)


# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading panel data…")
def load_panel() -> pd.DataFrame:
    df = pd.read_csv(PANEL_PATH)
    df["country_label"] = df["country_name"].map(_label)
    # Derive year-on-year GDP growth %
    df = df.sort_values(["country_code", "year"])
    df["gdp_growth_pct"] = (
        df.groupby("country_code")["NY_GDP_MKTP_CD"]
        .pct_change()
        .mul(100)
    )
    return df


@st.cache_data(show_spinner="Loading OCVI scores…")
def load_ocvi() -> pd.DataFrame:
    df = pd.read_csv(OCVI_PATH)
    df["country_label"] = df["country_name"].map(_label)
    return df


@st.cache_data(show_spinner="Loading chain transmission data…")
def load_chain() -> pd.DataFrame:
    df = pd.read_csv(CHAIN_PATH)
    df["country_label"] = df["country_name"].map(_label)
    return df


@st.cache_data(show_spinner="Loading historical risk index…")
def load_historical_index() -> pd.DataFrame:
    if not HIST_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(HIST_PATH)
    if "country_label" not in df.columns and "country_name" in df.columns:
        df["country_label"] = df["country_name"].map(_label)
    return df


@st.cache_data(show_spinner="Loading sensitivity results…")
def load_sensitivity() -> pd.DataFrame:
    if not SENS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(SENS_PATH)


@st.cache_data(show_spinner="Loading 2020 retrospective…")
def load_retrospective() -> pd.DataFrame:
    if not RETRO_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(RETRO_PATH)
    if "country_label" not in df.columns and "country_name" in df.columns:
        df["country_label"] = df["country_name"].apply(_label)
    return df


@st.cache_data(show_spinner="Loading cross-validation results…")
def load_cross_validation() -> pd.DataFrame:
    if not CV_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CV_PATH)


@st.cache_data(show_spinner="Running historical backtest…", ttl=3600)
def _run_bt(year: int, scenario: str) -> dict:
    """Cache-wrapped backtest snapshot.  Returns {'ok': True, 'df': ...} or {'ok': False, 'error': ...}."""
    from src.model.backtest import run_backtest_snapshot
    try:
        df = run_backtest_snapshot(year=year, scenario=scenario)
        return {"ok": True, "df": df}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@st.cache_data(ttl=3600, show_spinner="Running fiscal stress analysis…")
def load_fiscal_stress_data() -> dict:
    """Build the Right Now Risk composite table from all four additions."""
    from src.model.right_now_risk import run_right_now_risk
    try:
        return run_right_now_risk(
            breakeven_path=BREAKEVEN_PATH,
            reserves_path=RESERVES_PATH,
            food_path=FOOD_PATH,
            chain_path=CHAIN_PATH,
            panel_path=PANEL_PATH,
        )
    except Exception as exc:
        return {"error": str(exc)}


# EIA annual average Brent spot price 2022 — used as the shock baseline
_BRENT_2022_AVG: float = 100.93

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_brent() -> tuple[float | None, str]:
    """Fetch latest Brent Crude Futures close from Yahoo Finance (BZ=F)."""
    try:
        import yfinance as yf
        hist = yf.Ticker("BZ=F").history(period="5d")
        if hist.empty:
            return None, ""
        price = float(hist["Close"].dropna().iloc[-1])
        ts = hist.index[-1].strftime("%Y-%m-%d %H:%M UTC")
        return price, ts
    except Exception:
        return None, ""


# ── Sanity-check paths before anything else ───────────────────────────────────
for _p in (PANEL_PATH, OCVI_PATH):
    if not _p.exists():
        st.error(
            f"Required file not found: `{_p}`  \n"
            "Run the data pipeline first:  \n"
            "`python -m src.data.fetch_world_bank`  \n"
            "`python -m src.data.clean_world_bank`  \n"
            "`python -m src.model.vulnerability_index`"
        )
        st.stop()

panel = load_panel()
ocvi  = load_ocvi()
_chain_available = CHAIN_PATH.exists()
chain = load_chain() if _chain_available else None

_live_brent, _live_ts = fetch_live_brent()
_auto_shock_raw: float | None = (
    (_live_brent - _BRENT_2022_AVG) / _BRENT_2022_AVG * 100
    if _live_brent is not None else None
)
_auto_shock_snap: int | None = (
    int(max(-60, min(60, round(_auto_shock_raw / 5) * 5)))
    if _auto_shock_raw is not None else None
)

_all_labels   = sorted(panel["country_label"].unique())
_yr_min       = int(panel["year"].min())
_yr_max       = int(panel["year"].max())

# Consistent colour map: alphabetical order → same colour in every chart
_COLOUR_MAP: dict[str, str] = {
    label: _PALETTE[i % len(_PALETTE)]
    for i, label in enumerate(_all_labels)
}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛢️ MENA Oil Chain")
    st.markdown("---")

    # Country filter
    st.markdown("### Country filter")
    selected_countries = st.multiselect(
        "Select countries",
        options=_all_labels,
        default=_all_labels,
        help="Filters the Oil Rents time-series and GDP vs Inflation chart.",
    )
    if not selected_countries:
        selected_countries = _all_labels  # guard: never allow empty

    # Year range
    st.markdown("### Year range")
    year_range = st.slider(
        "Years",
        min_value=_yr_min,
        max_value=_yr_max,
        value=(_yr_min, _yr_max),
        step=1,
    )

    # Oil price shock
    st.markdown("### ⚡ Oil price shock")
    shock_pct = st.slider(
        "Price change (%)",
        min_value=-60,
        max_value=60,
        value=-30,
        step=5,
        format="%+d%%",
        help=(
            "Simulates a permanent shift in oil prices.  \n"
            "**First-order exposure** = Oil Rents % GDP × price change.  \n"
            "Negative = revenue loss · Positive = revenue gain."
        ),
    )

    # Live Brent price
    st.markdown("### 📡 Live Brent Price")
    if _live_brent is not None:
        st.metric(
            "BZ=F (Brent Futures)",
            f"${_live_brent:.2f}",
            f"{_auto_shock_raw:+.1f}% vs 2022 avg (${_BRENT_2022_AVG:.2f})",
            delta_color="off",
        )
        _use_live_shock = st.checkbox(
            "Auto-set shock from live price",
            value=False,
            help=(
                f"Overrides the slider to **{_auto_shock_snap:+d}%**  \n"
                f"live ${_live_brent:.2f} vs 2022 avg ${_BRENT_2022_AVG:.2f}"
            ),
        )
        if _use_live_shock:
            shock_pct = _auto_shock_snap
    else:
        st.caption("Live price unavailable — markets closed or API error.")
        _use_live_shock = False

    st.markdown("---")

    # Country Detail navigation
    st.markdown("### 🔍 Country Detail")
    _nav_country = st.selectbox(
        "Deep-dive country",
        options=["— select —"] + sorted(_all_labels),
        index=0,
        help="Opens the Country Detail page for the selected country.",
        key="nav_country_detail",
    )
    if _nav_country != "— select —":
        _detail_url = f"country_detail?country={_nav_country.replace(' ', '%20')}"
        st.page_link(
            "pages/country_detail.py",
            label=f"Open {_nav_country} detail →",
            icon="🔍",
        )

    st.markdown("---")
    st.caption("Source: World Bank Open Data  \nBuilt with Streamlit + Plotly")


# ── Utility: apply country + year filter to panel ─────────────────────────────
def _filter(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["country_label"].isin(selected_countries) &
        df["year"].between(year_range[0], year_range[1])
    ]


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_rank, tab_rents, tab_shock, tab_macro, tab_chain, tab_fiscal, tab_backtest, tab_hist, tab_sens, tab_retro, tab_cv = st.tabs([
    "📊  OCVI Rankings",
    "🛢️  Oil Rents % GDP",
    "⚡  Price Shock",
    "📈  GDP Growth vs Inflation",
    "⛓️  Chain Transmission",
    "🚨  Fiscal Stress",
    "🔁  Backtesting",
    "📉  Historical Risk Index",
    "🎛️  Sensitivity Analysis",
    "🔍  2020 Retrospective",
    "✅  Validation",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 · OCVI Rankings
# ═══════════════════════════════════════════════════════════════════════════════
with tab_rank:
    st.header("Oil Chain Vulnerability Index — Country Rankings")
    st.caption(
        "OCVI = **0.40** × oil_rents  +  **0.20** × exports/GDP  "
        "+  **0.15** × imports/GDP  +  **0.15** × GDP-growth volatility  "
        "+  **0.10** × inflation volatility  ·  min-max normalised across 14 countries"
    )

    left_col, right_col = st.columns([3, 2], gap="large")

    # ── Ranking table ──────────────────────────────────────────────────────────
    with left_col:
        tbl = (
            ocvi[[
                "ocvi_rank", "country_label", "ocvi_score",
                "oil_rents_pct_gdp", "exports_pct_gdp", "imports_pct_gdp",
                "gdp_growth_vol", "inflation_vol", "data_insufficient",
            ]]
            .sort_values("ocvi_rank")
            .copy()
        )
        tbl.columns = [
            "Rank", "Country", "OCVI Score",
            "Oil Rents % GDP", "Exports % GDP", "Imports % GDP",
            "GDP Vol", "Infl Vol", "⚠ Low data",
        ]

        st.dataframe(
            tbl,
            hide_index=True,
            use_container_width=True,
            column_config={
                "OCVI Score": st.column_config.ProgressColumn(
                    "OCVI Score",
                    format="%.3f",
                    min_value=0.0,
                    max_value=1.0,
                    help="0 = least vulnerable · 1 = most vulnerable",
                ),
                "Oil Rents % GDP": st.column_config.NumberColumn(
                    "Oil Rents % GDP", format="%.1f%%"
                ),
                "Exports % GDP": st.column_config.NumberColumn(
                    "Exports % GDP", format="%.1f%%"
                ),
                "Imports % GDP": st.column_config.NumberColumn(
                    "Imports % GDP", format="%.1f%%"
                ),
                "GDP Vol": st.column_config.NumberColumn(
                    "GDP Vol", format="%.4f",
                    help="Std of annual GDP growth rate (2000-2024)",
                ),
                "Infl Vol": st.column_config.NumberColumn(
                    "Infl Vol", format="%.2f",
                    help="Std of annual CPI inflation (2000-2024)",
                ),
                "⚠ Low data": st.column_config.CheckboxColumn(
                    "⚠ Low data",
                    help="< 10 valid years for at least one component",
                ),
            },
        )
        make_csv_download_button(tbl, "ocvi_rankings.csv", "Download rankings as CSV")

    # ── Horizontal bar chart ───────────────────────────────────────────────────
    with right_col:
        bar_df = ocvi.sort_values("ocvi_score").copy()
        bar_df["rank_label"] = "#" + bar_df["ocvi_rank"].astype(str)

        fig_bar = px.bar(
            bar_df,
            x="ocvi_score",
            y="country_label",
            orientation="h",
            color="ocvi_score",
            color_continuous_scale="RdYlGn_r",
            range_color=[0, 1],
            text="rank_label",
            labels={"ocvi_score": "OCVI Score", "country_label": ""},
            title="OCVI Score",
        )
        fig_bar.update_traces(
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>OCVI: %{x:.3f}<extra></extra>",
        )
        fig_bar.update_layout(
            coloraxis_showscale=False,
            height=500,
            xaxis=dict(range=[0, 1.1], title="OCVI Score"),
            yaxis_title=None,
            margin=dict(l=0, r=50, t=40, b=0),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Component breakdown radar ──────────────────────────────────────────────
    with st.expander("Component breakdown — normalised radar chart"):
        norm_cols = [
            "norm_oil_rents_pct_gdp", "norm_exports_pct_gdp",
            "norm_imports_pct_gdp", "norm_gdp_growth_vol", "norm_inflation_vol",
        ]
        axis_labels = [
            "Oil Rents", "Exports/GDP",
            "Imports/GDP", "GDP Vol", "Infl Vol",
        ]
        fig_radar = go.Figure()
        for _, row in ocvi.iterrows():
            vals = [row[c] for c in norm_cols]
            vals_closed = vals + [vals[0]]
            labels_closed = axis_labels + [axis_labels[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_closed,
                theta=labels_closed,
                fill="toself",
                opacity=0.25,
                name=row["country_label"],
                line=dict(color=_COLOUR_MAP[row["country_label"]]),
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=460,
            legend=dict(orientation="v"),
            title="Normalised OCVI Components (0=min · 1=max vulnerability)",
            margin=dict(l=40, r=40, t=60, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 · Oil Rents % GDP time-series
# ═══════════════════════════════════════════════════════════════════════════════
with tab_rents:
    st.header("Oil Rents as % of GDP — Time Series")
    st.caption(
        "**NY.GDP.PETR.RT.ZS** — value of crude oil production at world prices "
        "minus production costs, as a share of GDP.  "
        "Higher = greater oil-revenue dependency."
    )

    df_rents = _filter(panel).dropna(subset=["NY_GDP_PETR_RT_ZS"])

    if df_rents.empty:
        st.warning("No data for the current selection. Adjust the country or year filters.")
    else:
        fig_ts = px.line(
            df_rents,
            x="year",
            y="NY_GDP_PETR_RT_ZS",
            color="country_label",
            markers=True,
            labels={
                "year": "Year",
                "NY_GDP_PETR_RT_ZS": "Oil Rents (% of GDP)",
                "country_label": "Country",
            },
            title=f"Oil Rents % GDP · {year_range[0]}–{year_range[1]}",
            color_discrete_map=_COLOUR_MAP,
        )
        fig_ts.update_traces(marker_size=4, line_width=1.8)
        fig_ts.update_layout(
            height=500,
            hovermode="x unified",
            xaxis=dict(dtick=2),
            yaxis_title="Oil Rents (% of GDP)",
            legend=dict(
                orientation="v", yanchor="top", y=1,
                xanchor="left", x=1.02,
            ),
        )
        # Shade the 2008 and 2014 oil price crash periods
        for yr, label in [(2008, "GFC / oil crash"), (2014, "2014 price rout"), (2020, "COVID-19")]:
            if year_range[0] <= yr <= year_range[1]:
                fig_ts.add_vline(
                    x=yr, line_dash="dot", line_color="grey", opacity=0.5,
                    annotation_text=label, annotation_position="top",
                    annotation_font_size=10,
                )
        st.plotly_chart(fig_ts, use_container_width=True)

        # Summary statistics
        summary = (
            df_rents
            .groupby("country_label")["NY_GDP_PETR_RT_ZS"]
            .agg(
                Mean="mean", Median="median",
                Max="max", Min="min", Std="std",
            )
            .round(2)
            .reset_index()
            .rename(columns={"country_label": "Country"})
            .sort_values("Mean", ascending=False)
        )
        with st.expander("Summary statistics (selected period)"):
            st.dataframe(
                summary,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Mean":   st.column_config.NumberColumn(format="%.2f%%"),
                    "Median": st.column_config.NumberColumn(format="%.2f%%"),
                    "Max":    st.column_config.NumberColumn(format="%.2f%%"),
                    "Min":    st.column_config.NumberColumn(format="%.2f%%"),
                    "Std":    st.column_config.NumberColumn(format="%.2f"),
                },
            )
            make_csv_download_button(summary, "oil_rents_summary.csv", "Download summary as CSV")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 · Oil Price Shock
# ═══════════════════════════════════════════════════════════════════════════════
with tab_shock:
    st.header("First-Order Oil Price Shock Exposure")

    # Live price banner
    if _live_brent is not None:
        _banner = (
            f"📡 **Live Brent (BZ=F):** ${_live_brent:.2f}"
            f"  ·  **vs 2022 avg (${_BRENT_2022_AVG:.2f}):** {_auto_shock_raw:+.1f}%"
            f"  ·  *{_live_ts}*"
        )
        if _use_live_shock:
            st.success(_banner + f"  ·  **Shock auto-set to {shock_pct:+d}%**")
        else:
            st.info(_banner + "  ·  Enable *Auto-set shock* in the sidebar to use this value.")

    # Scenario narrative
    _dir = "decline" if shock_pct < 0 else "surge" if shock_pct > 0 else "no change"
    _sign = "loss" if shock_pct < 0 else "gain" if shock_pct > 0 else "–"
    st.markdown(
        f"**Scenario:** {abs(shock_pct)}% oil price **{_dir}**  \n"
        f"*First-order revenue exposure ≈ Oil Rents % GDP × price change.  "
        f"This is a linear approximation before multiplier or substitution effects.*"
    )

    shock_fraction = shock_pct / 100.0

    # Average GDP per country over full panel (used for USD impact)
    avg_gdp = (
        panel.groupby("country_code")["NY_GDP_MKTP_CD"]
        .mean()
        .reset_index()
        .rename(columns={"NY_GDP_MKTP_CD": "avg_gdp_usd"})
    )
    shock_df = (
        ocvi[["country_code", "country_label", "oil_rents_pct_gdp"]]
        .merge(avg_gdp, on="country_code")
    )
    shock_df["exposure_pp_gdp"] = shock_df["oil_rents_pct_gdp"] * shock_fraction
    shock_df["impact_usd_bn"]   = (
        shock_df["oil_rents_pct_gdp"] / 100.0
        * shock_fraction
        * shock_df["avg_gdp_usd"]
        / 1e9
    )
    shock_df = shock_df.sort_values("exposure_pp_gdp", ascending=True)

    # ── Layout ─────────────────────────────────────────────────────────────────
    c_chart, c_table = st.columns([3, 2], gap="large")

    with c_chart:
        # Colour: negative = red shades, positive = green shades, zero = grey
        if shock_pct < 0:
            cscale = "RdYlGn"        # sorted ascending → leftmost bars are most-red
        elif shock_pct > 0:
            cscale = "RdYlGn_r"
        else:
            cscale = "Greys"

        fig_shock = px.bar(
            shock_df,
            x="exposure_pp_gdp",
            y="country_label",
            orientation="h",
            color="exposure_pp_gdp",
            color_continuous_scale=cscale,
            text=shock_df["exposure_pp_gdp"].map(lambda v: f"{v:+.1f} pp"),
            labels={
                "exposure_pp_gdp": f"Revenue impact (percentage points of GDP)",
                "country_label": "",
            },
            title=f"First-Order GDP Revenue Impact · Oil price {shock_pct:+d}%",
        )
        fig_shock.update_traces(
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Impact: %{x:+.2f} pp of GDP<extra></extra>"
            ),
        )
        fig_shock.add_vline(x=0, line_color="black", line_width=1)
        fig_shock.update_layout(
            coloraxis_showscale=False,
            height=500,
            margin=dict(l=0, r=70, t=50, b=0),
            xaxis_title="Impact (percentage points of GDP)",
        )
        st.plotly_chart(fig_shock, use_container_width=True)

    with c_table:
        st.subheader("Country-level impact")
        disp = shock_df.sort_values("exposure_pp_gdp")[
            ["country_label", "oil_rents_pct_gdp", "exposure_pp_gdp", "impact_usd_bn"]
        ].copy()
        disp.columns = ["Country", "Oil Rents % GDP", "Impact (pp GDP)", "Impact (USD bn)"]

        st.dataframe(
            disp,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Oil Rents % GDP": st.column_config.NumberColumn(format="%.1f%%"),
                "Impact (pp GDP)": st.column_config.NumberColumn(format="%+.2f"),
                "Impact (USD bn)": st.column_config.NumberColumn(format="$%+.1f bn"),
            },
        )
        make_csv_download_button(disp, "price_shock_table.csv", "Download table as CSV")

        # KPI cards
        st.markdown("---")
        if shock_pct != 0:
            # Most impacted = largest absolute exposure
            worst = shock_df.loc[shock_df["exposure_pp_gdp"].abs().idxmax()]
            st.metric(
                label=f"Most exposed country",
                value=worst["country_label"],
                delta=f"{worst['exposure_pp_gdp']:+.1f} pp of GDP",
                delta_color="inverse",
            )

        total_impact = shock_df["impact_usd_bn"].sum()
        st.metric(
            label="Combined MENA-14 revenue impact",
            value=f"${total_impact:+.1f} bn",
            delta=f"vs. baseline — {_sign}",
            delta_color="inverse" if shock_pct < 0 else "normal",
        )

        most_resilient = shock_df.loc[shock_df["oil_rents_pct_gdp"].idxmin()]
        st.metric(
            label="Least exposed country",
            value=most_resilient["country_label"],
            delta=f"{most_resilient['oil_rents_pct_gdp']:.1f}% oil rents/GDP",
            delta_color="off",
        )

    # ── Methodology note ───────────────────────────────────────────────────────
    with st.expander("Methodology — what 'first-order exposure' means"):
        st.markdown("""
**Formula:**  `Exposure (pp GDP) = Oil Rents % GDP × ΔPrice/Price`

**Assumptions and limitations:**
- Uses the *long-run average* Oil Rents % GDP (2000–2024) as the base.
- Linear approximation: assumes production volumes and costs are fixed.
- Does **not** model second-order effects (exchange rate pass-through,
  fiscal multiplier, sovereign wealth fund buffers, or OPEC quota responses).
- Countries with low or zero oil rents (Morocco, Jordan, Lebanon) appear
  near-zero not because they are immune, but because they are *import-side*
  exposed — a price surge increases their import bill rather than their revenue.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 · GDP Growth vs Inflation
# ═══════════════════════════════════════════════════════════════════════════════
with tab_macro:
    st.header("GDP Growth vs Inflation")
    st.caption(
        "**X-axis:** Year-on-year GDP growth (%) derived from NY.GDP.MKTP.CD  ·  "
        "**Y-axis:** CPI inflation (%) from FP.CPI.TOTL.ZG  ·  "
        "**Bubble size:** GDP (USD)  ·  Each dot = one country-year observation.  "
        "Extreme outliers clipped on both axes; true values shown in hover tooltips."
    )

    df_macro = _filter(panel).dropna(subset=["gdp_growth_pct", "FP_CPI_TOTL_ZG"]).copy()

    if df_macro.empty:
        st.warning("No data for the current selection. Adjust the filters.")
    else:
        # ── Clip Y-axis (inflation) at p99 — Lebanon hyperinflation ──────────
        p99_inf = df_macro["FP_CPI_TOTL_ZG"].quantile(0.99)
        clipped_inf_n = (df_macro["FP_CPI_TOTL_ZG"] > p99_inf).sum()
        df_macro["infl_clipped"] = df_macro["FP_CPI_TOTL_ZG"].clip(upper=p99_inf)

        # ── Clip X-axis (GDP growth) at p2/p98 — Libya/Iraq war-rebound ─────
        # Without clipping, Libya 2012 (+92%) and Iraq 2004 (+67%) collapse
        # all other 12 countries into a narrow unreadable band.
        p2_g  = df_macro["gdp_growth_pct"].quantile(0.02)
        p98_g = df_macro["gdp_growth_pct"].quantile(0.98)
        clipped_g_n = (
            (df_macro["gdp_growth_pct"] < p2_g) |
            (df_macro["gdp_growth_pct"] > p98_g)
        ).sum()
        df_macro["growth_clipped"] = df_macro["gdp_growth_pct"].clip(
            lower=p2_g, upper=p98_g
        )

        # ── Info banners ──────────────────────────────────────────────────────
        msgs = []
        if clipped_inf_n > 0:
            msgs.append(
                f"{clipped_inf_n} point(s) with inflation > {p99_inf:.0f}% "
                f"clipped on Y-axis (Lebanon 2020-2023, max 221%)"
            )
        if clipped_g_n > 0:
            msgs.append(
                f"{clipped_g_n} point(s) with GDP growth outside "
                f"[{p2_g:.0f}%, {p98_g:.0f}%] clipped on X-axis "
                f"(Libya/Iraq conflict rebounds)"
            )
        if msgs:
            st.info("ℹ️  " + "  ·  ".join(msgs) + ".  True values visible in hover tooltips.")

        # Bubble size proportional to GDP; guard against NaN/zero
        df_macro["gdp_bn"] = df_macro["NY_GDP_MKTP_CD"].fillna(0).clip(lower=1e8) / 1e9

        fig_macro = px.scatter(
            df_macro,
            x="growth_clipped",
            y="infl_clipped",
            color="country_label",
            size="gdp_bn",
            size_max=35,
            hover_name="country_label",
            hover_data={
                "year": True,
                "gdp_growth_pct": ":.1f",   # true (unclipped) GDP growth
                "FP_CPI_TOTL_ZG": ":.1f",   # true (unclipped) inflation
                "growth_clipped": False,
                "infl_clipped":   False,
                "gdp_bn": ":.0f",
            },
            labels={
                "growth_clipped":  f"GDP Growth (%, clipped p2–p98)",
                "infl_clipped":    f"Inflation — CPI (%, clipped p99)",
                "country_label":   "Country",
                "gdp_bn":          "GDP (USD bn)",
                "gdp_growth_pct":  "True GDP Growth (%)",
                "FP_CPI_TOTL_ZG":  "True Inflation (%)",
                "year":            "Year",
            },
            title=(
                f"GDP Growth vs Inflation · "
                f"{year_range[0]}–{year_range[1]} · "
                f"bubble size = GDP"
            ),
            color_discrete_map=_COLOUR_MAP,
        )

        # Reference lines + quadrant labels
        fig_macro.add_hline(y=0,  line_dash="solid", line_color="lightgrey", line_width=1)
        fig_macro.add_vline(x=0,  line_dash="solid", line_color="lightgrey", line_width=1)
        fig_macro.add_hline(
            y=10, line_dash="dash", line_color="red", opacity=0.4, line_width=1,
            annotation_text="10% inflation threshold",
            annotation_position="top right",
            annotation_font_size=10,
        )

        fig_macro.update_layout(
            height=580,
            hovermode="closest",
            legend=dict(
                orientation="v", yanchor="top", y=1,
                xanchor="left", x=1.02,
                font_size=11,
            ),
            xaxis_title="GDP Growth (%)",
            yaxis_title="CPI Inflation (%)",
        )
        st.plotly_chart(fig_macro, use_container_width=True)

        # Quadrant counts
        q_data = df_macro[df_macro["FP_CPI_TOTL_ZG"].notna() & df_macro["gdp_growth_pct"].notna()]
        q1 = ((q_data["gdp_growth_pct"] >= 0) & (q_data["FP_CPI_TOTL_ZG"] >= 0)).sum()
        q2 = ((q_data["gdp_growth_pct"] <  0) & (q_data["FP_CPI_TOTL_ZG"] >= 0)).sum()
        q3 = ((q_data["gdp_growth_pct"] <  0) & (q_data["FP_CPI_TOTL_ZG"] <  0)).sum()
        q4 = ((q_data["gdp_growth_pct"] >= 0) & (q_data["FP_CPI_TOTL_ZG"] <  0)).sum()

        with st.expander("Quadrant breakdown"):
            qc1, qc2, qc3, qc4 = st.columns(4)
            qc1.metric("↗ Growth + Inflation",  f"{q1} obs", "Expansion / overheating")
            qc2.metric("↖ Contraction + Inflation", f"{q2} obs", "Stagflation / crisis")
            qc3.metric("↙ Contraction + Deflation", f"{q3} obs", "Recession / slump")
            qc4.metric("↘ Growth + Low Inflation",  f"{q4} obs", "Goldilocks zone")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 · Chain Transmission  (Addition 4 — static reference model)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chain:
    st.header("Oil Price Shock — Chain Transmission Severity")
    st.caption(
        "**Transmission chain:** Oil Price → Fiscal Revenue (Stage 1) → "
        "Inflation (Stage 2) → Employment (Stage 3) → "
        "Consumption (Stage 4) → Growth (Stage 5)  ·  "
        "Severity = mean(stage 1–5) × amplification factor, clamped [0, 1]."
    )

    if not _chain_available or chain is None:
        st.warning(
            "Chain transmission data not found.  \n"
            "Run: `python -m src.model.chain_transmission` from the project root."
        )
    else:
        # Use the most recent year in the file (static snapshot = 2024)
        _ct_year = int(chain["year"].max()) if "year" in chain.columns else 2024
        _chain_snap = (
            chain[chain["year"] == _ct_year].copy()
            if "year" in chain.columns else chain.copy()
        )
        _sev_col = (
            "chain_transmission_severity"
            if "chain_transmission_severity" in _chain_snap.columns
            else "transmission_severity"
        )

        # ── KPI row ────────────────────────────────────────────────────────────
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        _ct_top  = _chain_snap.loc[_chain_snap[_sev_col].idxmax()]
        _ct_mean = _chain_snap[_sev_col].mean()
        _ct_fast = (
            int((_chain_snap["transmission_speed"] == "fast").sum())
            if "transmission_speed" in _chain_snap.columns else 0
        )
        kpi1.metric(
            "Highest severity",
            _ct_top.get("country_label", _ct_top.get("country_name", "N/A")),
            f"{_ct_top[_sev_col]:.3f}",
            delta_color="inverse",
        )
        kpi2.metric("Mean severity (14 countries)", f"{_ct_mean:.3f}")
        kpi3.metric(
            "Fast-transmission countries",
            f"{_ct_fast} / {len(_chain_snap)}",
        )
        kpi4.metric(
            "Countries covered",
            f"{_chain_snap['country_code_a3'].nunique()}",
            f"Reference year {_ct_year}",
            delta_color="off",
        )

        st.markdown("---")

        # ── Ranked bar chart ────────────────────────────────────────────────────
        st.subheader("Country Rankings by Transmission Severity")

        _speed_map = {"fast": "#d62728", "medium": "#ff7f0e", "slow": "#2ca02c"}
        _bar_df    = _chain_snap.sort_values(_sev_col).copy()
        _bar_label = "country_label" if "country_label" in _bar_df.columns else "country_name"

        fig_ct_bar = go.Figure()
        for _spd, _col in _speed_map.items():
            _sub = (
                _bar_df[_bar_df["transmission_speed"] == _spd]
                if "transmission_speed" in _bar_df.columns
                else _bar_df
            )
            if _sub.empty:
                continue
            fig_ct_bar.add_trace(go.Bar(
                x=_sub[_sev_col],
                y=_sub[_bar_label],
                orientation="h",
                name=_spd.capitalize(),
                marker_color=_col,
                text=_sub[_sev_col].map(lambda v: f"{v:.3f}"),
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>Severity: %{x:.4f}<br>"
                    f"Speed: {_spd}<extra></extra>"
                ),
            ))
        fig_ct_bar.update_layout(
            barmode="overlay",
            height=460,
            xaxis=dict(range=[0, 1.08], title="Chain Transmission Severity"),
            yaxis_title=None,
            legend=dict(title="Speed", orientation="h", y=-0.13, font_size=12),
            margin=dict(l=10, r=90, t=10, b=60),
        )
        st.plotly_chart(fig_ct_bar, use_container_width=True)

        st.markdown("---")

        # ── Stage heatmap — countries × stages ─────────────────────────────────
        _ct_stage_cols = [
            c for c in [
                "stage1_oil_fiscal", "stage2_fiscal_inflation",
                "stage3_inflation_employment", "stage4_employment_consumption",
                "stage5_consumption_growth",
            ]
            if c in _chain_snap.columns
        ]
        if _ct_stage_cols:
            st.subheader("Stage Scores — Countries × Stages")
            st.caption(
                "Each cell = structural score [0, 1].  "
                "Red = strong transmission; green = buffered."
            )
            _ct_stage_labels = {
                "stage1_oil_fiscal":            "Stage 1 · Oil->Fiscal",
                "stage2_fiscal_inflation":      "Stage 2 · Fiscal->Inflation",
                "stage3_inflation_employment":  "Stage 3 · Inflation->Employment",
                "stage4_employment_consumption":"Stage 4 · Employment->Consumption",
                "stage5_consumption_growth":    "Stage 5 · Consumption->Growth",
            }
            _ct_heat = (
                _chain_snap.set_index(_bar_label)[_ct_stage_cols]
                .rename(columns=_ct_stage_labels)
            )
            _ct_order = (
                _chain_snap.sort_values(_sev_col, ascending=False)[_bar_label].tolist()
            )
            _ct_heat = _ct_heat.loc[[c for c in _ct_order if c in _ct_heat.index]]

            fig_ct_heat = go.Figure(go.Heatmap(
                z=_ct_heat.values,
                x=_ct_heat.columns.tolist(),
                y=_ct_heat.index.tolist(),
                colorscale="RdYlGn_r",
                zmin=0.0, zmax=1.0,
                colorbar=dict(title="Score", thickness=14),
                hovertemplate="<b>%{y}</b><br>%{x}<br>Score: %{z:.2f}<extra></extra>",
                text=_ct_heat.round(2).values,
                texttemplate="%{z:.2f}",
                textfont=dict(size=10),
            ))
            fig_ct_heat.update_layout(
                height=430,
                margin=dict(l=10, r=10, t=10, b=90),
                xaxis=dict(tickangle=-20, side="bottom"),
                yaxis_title=None,
            )
            st.plotly_chart(fig_ct_heat, use_container_width=True)
            st.markdown("---")

        # ── Full data table ─────────────────────────────────────────────────────
        with st.expander("Full chain data — all 14 countries"):
            _ct_show = {
                _bar_label:                      "Country",
                "transmission_speed":            "Speed",
                "amplification_factor":          "Amplif.",
                "stage1_oil_fiscal":             "Stage 1",
                "stage2_fiscal_inflation":       "Stage 2",
                "stage3_inflation_employment":   "Stage 3",
                "stage4_employment_consumption": "Stage 4",
                "stage5_consumption_growth":     "Stage 5",
                _sev_col:                        "Severity",
                "confidence":                    "Confidence",
            }
            _ct_show = {k: v for k, v in _ct_show.items() if k in _chain_snap.columns}
            _ct_disp = (
                _chain_snap.sort_values(_sev_col, ascending=False)
                [list(_ct_show.keys())]
                .rename(columns=_ct_show)
                .round(4)
            )
            st.dataframe(
                _ct_disp,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Severity": st.column_config.ProgressColumn(
                        "Severity", format="%.3f", min_value=0.0, max_value=1.0,
                    ),
                    "Amplif.": st.column_config.NumberColumn(format="%.2f"),
                },
            )
            make_csv_download_button(_ct_disp, "chain_transmission_table.csv", "Download table as CSV")

        # ── Methodology ─────────────────────────────────────────────────────────
        with st.expander("Methodology"):
            st.markdown(
                "**Formula:** `severity = min(1.0, mean(stage 1–5) × amplification_factor)`  \n\n"
                "**Stages:** Oil→Fiscal (1) · Fiscal→Inflation (2) · "
                "Inflation→Employment (3) · Employment→Consumption (4) · "
                "Consumption→Growth (5).  \n\n"
                "**Amplification factor** < 1.0 = large SWF or diversified economy buffers "
                "the chain (Kuwait 0.78, UAE 0.80, Qatar 0.82, Saudi Arabia 0.90).  "
                "> 1.0 = conflict / embedded inflation / institutional fragility amplifies "
                "it (Algeria 1.18, Iraq 1.25, Lebanon 1.32, Libya 1.38).  \n\n"
                "**Data:** IMF Article IV 2023, IMF REO MENA Oct 2023, "
                "Coady et al. IMF 2015 (subsidy pass-through).  "
                "All scores are expert estimates (`is_estimate=True`).  \n\n"
                "**Integration:** `chain_transmission_severity_recent` feeds the "
                "**0.20** weight in the Right Now Risk composite (Addition 5)."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 · Right Now Risk — Fiscal Stress
# ═══════════════════════════════════════════════════════════════════════════════
with tab_fiscal:
    st.header("Right Now Risk — Fiscal & Stability Assessment")
    st.caption(
        "Composite score = **0.35** × fiscal stress  +  **0.25** × reserve runway risk  "
        "+  **0.20** × social stability risk  +  **0.20** × chain transmission severity  ·  "
        "Missing components trigger proportional weight rescaling."
    )

    _fiscal_data = load_fiscal_stress_data()

    if "error" in _fiscal_data:
        st.warning(
            f"Fiscal stress analysis unavailable: {_fiscal_data['error']}  \n"
            "Ensure all reference CSVs exist:  \n"
            "`data/reference/fiscal_breakeven.csv`  \n"
            "`data/reference/swf_reserves.csv`  \n"
            "`data/reference/food_security.csv`"
        )
    else:
        _rnr_df      = _fiscal_data["right_now_risk_df"]
        _stress_tbl  = _fiscal_data["stress_table"]
        _runway_tbl  = _fiscal_data["runway_table"]
        _fs_brent    = _fiscal_data.get("brent_live", float("nan"))

        # ── KPI row ────────────────────────────────────────────────────────────
        kf1, kf2, kf3, kf4 = st.columns(4)

        import math as _math
        _brent_str = f"${_fs_brent:.2f}" if not _math.isnan(_fs_brent) else "N/A"
        kf1.metric(
            "Brent Today (BZ=F)",
            _brent_str,
            delta=None,
        )

        _n_below_be = int((_stress_tbl["stress_status"] == "Red").sum()) if not _stress_tbl.empty else 0
        kf2.metric(
            "Countries Below Breakeven",
            str(_n_below_be),
            "fiscal deficit territory",
            delta_color="inverse",
        )

        _runway_active = _runway_tbl[
            ~_runway_tbl["runway_status"].isin(["Gray"])
        ].copy() if not _runway_tbl.empty else pd.DataFrame()

        if not _runway_active.empty and _runway_active["reserve_runway_months"].notna().any():
            _shortest_idx = _runway_active["reserve_runway_months"].idxmin()
            _shortest_row = _runway_active.loc[_shortest_idx]
            _runway_lbl   = _shortest_row.get("country_label", _shortest_row.get("country_code_a3", "?"))
            _runway_mo    = _shortest_row["reserve_runway_months"]
            kf3.metric(
                "Shortest Runway (stressed)",
                _runway_lbl,
                f"{_runway_mo:.0f} months",
                delta_color="inverse",
            )
        else:
            kf3.metric("Shortest Runway (stressed)", "—", "no stressed countries")

        if not _rnr_df.empty and not _rnr_df["right_now_risk_score"].isna().all():
            _top_row   = _rnr_df.dropna(subset=["right_now_risk_score"]).iloc[0]
            _top_lbl   = _top_row.get("country_label", _top_row.get("country_code_a3", "?"))
            _top_score = _top_row["right_now_risk_score"]
            kf4.metric(
                "Highest Risk Country",
                _top_lbl,
                f"score {_top_score:.3f}",
                delta_color="inverse",
            )
        else:
            kf4.metric("Highest Risk Country", "—", "data unavailable")

        st.markdown("---")

        # ── Rescaled-weights transparency banner ───────────────────────────────
        _n_rescaled = int((_rnr_df["missing_components"].fillna("") != "").sum())
        if _n_rescaled > 0:
            _rescaled_pct = _n_rescaled / len(_rnr_df) * 100
            st.warning(
                f"**Weight rescaling active** — {_n_rescaled} of {len(_rnr_df)} "
                f"countries ({_rescaled_pct:.0f}%) had one or more components "
                f"missing and used proportionally rescaled weights. "
                f"See the **Missing** column in the Country Detail table for per-country detail."
            )

        # ── Bar chart ──────────────────────────────────────────────────────────
        st.subheader("Right Now Risk Score — Country Ranking")

        _bar_df = (
            _rnr_df
            .dropna(subset=["right_now_risk_score"])
            .sort_values("right_now_risk_score", ascending=True)
            .copy()
        )
        if not _bar_df.empty:
            _bar_df["_label"] = _bar_df.get(
                "country_label",
                _bar_df.get("country_code_a3", _bar_df.index.astype(str))
            )
            _bar_df["_driver_tag"] = _bar_df["primary_driver"].fillna("Mixed")

            fig_rnr = px.bar(
                _bar_df,
                x="right_now_risk_score",
                y="_label",
                orientation="h",
                color="right_now_risk_score",
                color_continuous_scale="RdYlGn_r",
                range_color=[0.0, 1.0],
                text=_bar_df["right_now_risk_score"].map(lambda v: f"{v:.3f}"),
                hover_data={
                    "right_now_risk_score": ":.3f",
                    "_driver_tag": True,
                    "fiscal_stress_score": ":.3f" if "fiscal_stress_score" in _bar_df.columns else False,
                    "reserve_runway_risk": ":.3f" if "reserve_runway_risk" in _bar_df.columns else False,
                    "social_stability_risk": ":.3f" if "social_stability_risk" in _bar_df.columns else False,
                    "chain_transmission_severity_recent": ":.3f" if "chain_transmission_severity_recent" in _bar_df.columns else False,
                    "_label": False,
                },
                labels={
                    "right_now_risk_score": "Right Now Risk Score",
                    "_label": "",
                    "_driver_tag": "Primary driver",
                },
                title="Right Now Risk Score (0 = lowest · 1 = highest)",
            )
            fig_rnr.update_traces(textposition="outside")
            fig_rnr.update_layout(
                coloraxis_showscale=False,
                height=500,
                margin=dict(l=0, r=70, t=50, b=0),
                xaxis=dict(range=[0, 1.12], title="Right Now Risk Score"),
            )
            st.plotly_chart(fig_rnr, use_container_width=True)

        st.markdown("---")

        # ── Ranked table ───────────────────────────────────────────────────────
        st.subheader("Country Detail")

        _tbl_src = _rnr_df.copy()

        # Data-quality badges for low-confidence and partially-missing rows
        _n_low_conf   = int((_tbl_src.get("confidence", pd.Series(dtype=str)).str.lower() == "low").sum())
        _n_partial     = int((_tbl_src["missing_components"].fillna("") != "").sum())
        if _n_low_conf > 0 or _n_partial > 0:
            _badge_parts = []
            if _n_low_conf > 0:
                _badge_parts.append(f"**{_n_low_conf}** low-confidence row(s)")
            if _n_partial > 0:
                _badge_parts.append(f"**{_n_partial}** row(s) with missing components (weights rescaled)")
            st.info("Data quality: " + " · ".join(_badge_parts))

        _display_cols = {
            "country_label":                        "Country",
            "stress_status":                        "Fiscal Status",
            "runway_status":                        "Runway Status",
            "fiscal_stress_score":                  "Fiscal Score",
            "reserve_runway_risk":                  "Runway Risk",
            "social_stability_risk":                "Social Risk",
            "chain_transmission_severity_recent":   "Chain Severity",
            "right_now_risk_score":                 "Right Now Risk",
            "primary_driver":                       "Primary Driver",
            "confidence":                           "Data Confidence",
            "missing_components":                   "Missing",
        }
        _tbl_cols_available = [c for c in _display_cols if c in _tbl_src.columns]
        _tbl_disp = (
            _tbl_src[_tbl_cols_available]
            .rename(columns=_display_cols)
            .round(3)
        )

        _col_config: dict = {
            "Right Now Risk": st.column_config.ProgressColumn(
                "Right Now Risk", format="%.3f", min_value=0.0, max_value=1.0,
                help="Composite 0-1 score; 1 = highest risk",
            ),
            "Fiscal Score": st.column_config.ProgressColumn(
                "Fiscal Score", format="%.3f", min_value=0.0, max_value=1.0,
                help="Continuous fiscal stress: (breakeven - brent) / breakeven, clamped to [0,1]",
            ),
            "Runway Risk": st.column_config.ProgressColumn(
                "Runway Risk", format="%.3f", min_value=0.0, max_value=1.0,
                help="Reserve runway risk: 1=<=6 months, 0=>=36 months (Gray=0)",
            ),
            "Social Risk": st.column_config.ProgressColumn(
                "Social Risk", format="%.3f", min_value=0.0, max_value=1.0,
                help="Social stability risk (Addition 3)",
            ),
            "Chain Severity": st.column_config.ProgressColumn(
                "Chain Severity", format="%.3f", min_value=0.0, max_value=1.0,
                help="Normalised mean transmission severity, most recent 3 years",
            ),
        }
        st.dataframe(
            _tbl_disp,
            hide_index=True,
            use_container_width=True,
            column_config=_col_config,
        )
        make_csv_download_button(_tbl_disp, "right_now_risk_scores.csv", "Download scores as CSV")

        # ── Methodology expander ───────────────────────────────────────────────
        with st.expander("Methodology — Right Now Risk score"):
            st.markdown("""
**Composite formula (default weights):**
```
right_now_risk_score =
    0.35 × fiscal_stress_score
  + 0.25 × reserve_runway_risk
  + 0.20 × social_stability_risk
  + 0.20 × chain_transmission_severity_recent
```

**Component definitions:**

| Component | Source | Formula |
|-----------|--------|---------|
| **Fiscal stress score** | Addition 1 — fiscal breakeven | `min(1, max(0, (breakeven − brent) / breakeven))` |
| **Reserve runway risk** | Addition 2 — SWF/FX reserves | Linear: 0 at ≥36 months, 1 at ≤6 months; 0 if Gray (not stressed) |
| **Social stability risk** | Addition 3 — food/fiscal/inflation | `0.5×food_exposure + 0.3×fiscal_score + 0.2×inflation_norm` |
| **Chain transmission** | Chain model — transmission_severity | Mean severity, most recent 3 years, min-max normalised |

**Fallback policy:**
If any component is NaN for a country, the remaining component weights
are rescaled proportionally to sum to 1.0.  The `Missing` column shows
which components were absent; `rescaled_weights` shows the actual weights used.
No country is silently dropped from the table.

**Stress classifications:**
- **Red** fiscal status: Brent < breakeven (government in deficit)
- **Amber**: Brent within $15/bbl above breakeven
- **Green**: comfortable fiscal headroom (≥$15 above breakeven)
- **Gray**: net importer or concept not applicable
""")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 · Backtesting
# ═══════════════════════════════════════════════════════════════════════════════
with tab_backtest:
    st.header("Right Now Risk — Historical Backtest")
    st.caption(
        "**Conditional backtest:** the 2023 reference data (fiscal breakeven, reserves, "
        "food security) is held fixed while the annual-average Brent price varies to a "
        "historical year.  Chain component is absent for pre-2024 years — weights rescale "
        "proportionally across the remaining three components."
    )

    # ── Controls (inside tab — not in global sidebar) ─────────────────────────
    _bt_c1, _bt_c2, _bt_c3 = st.columns([1, 2, 1])

    with _bt_c1:
        _bt_year = st.selectbox(
            "Backtest year",
            options=list(range(2010, 2024)),
            index=10,          # default 2020 (COVID collapse year)
            key="bt_year",
            help=(
                "Historical annual-average Brent price for this year is used as the "
                "live Brent input.  Fetched from yfinance; falls back to an embedded "
                "EIA / World Bank reference table."
            ),
        )

    with _bt_c2:
        _bt_scenario_opts: dict[str, str] = {
            "Base (2023 reference values)":               "base",
            "Stress (high breakeven + low buffer)":       "stress",
            "Favorable (low breakeven + high buffer)":    "optimistic",
        }
        _bt_scenario_label = st.selectbox(
            "Scenario",
            options=list(_bt_scenario_opts.keys()),
            index=0,
            key="bt_scenario",
            help=(
                "**Base:** 2023 reference columns unchanged.  \n"
                "**Stress:** `breakeven_high_usd` + `liquid_buffer_low_usd_bn` "
                "+ `monthly_burn_high_usd_bn`.  \n"
                "**Favorable:** `breakeven_low_usd` + `liquid_buffer_high_usd_bn` "
                "+ `monthly_burn_low_usd_bn`."
            ),
        )
        _bt_scenario = _bt_scenario_opts[_bt_scenario_label]

    with _bt_c3:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        _bt_clicked = st.button("Run backtest", type="primary", key="bt_run_btn")

    # Persist the last-run selection across re-renders via session state
    if _bt_clicked:
        st.session_state["_bt_run_year"]     = _bt_year
        st.session_state["_bt_run_scenario"] = _bt_scenario

    _bt_active_year     = st.session_state.get("_bt_run_year",     None)
    _bt_active_scenario = st.session_state.get("_bt_run_scenario", "base")

    if _bt_active_year is None:
        st.info(
            "Select a year and scenario above, then click **Run backtest**.  \n"
            "The model applies the historical annual-average Brent price to the "
            "2023 reference data and recomputes all four Right Now Risk components."
        )
    else:
        _bt_result = _run_bt(_bt_active_year, _bt_active_scenario)

        if not _bt_result["ok"]:
            st.error(
                f"Backtest failed: {_bt_result['error']}  \n"
                "Check that reference CSVs exist and that yfinance or the "
                "fallback Brent table covers the selected year."
            )
        else:
            _bt_df    = _bt_result["df"].copy()
            _bt_brent = float(_bt_df["historical_brent_usd"].iloc[0])
            _bt_n_rsc = int((_bt_df["missing_components_count"] > 0).sum())

            # ── Historical Brent banner ──────────────────────────────────────
            st.success(
                f"**{_bt_active_year}** · "
                f"Annual-average Brent **${_bt_brent:.2f}/bbl** · "
                f"Scenario: **{_bt_active_scenario}** · "
                f"Rescaled weights: **{_bt_n_rsc} / 14** countries"
            )

            st.markdown("---")

            # ── 1. Coverage indicator ────────────────────────────────────────
            st.subheader("Component coverage")

            _chain_missing_any = (
                _bt_df["missing_components"]
                .str.contains("chain_transmission", na=False)
                .any()
            )
            _cov_cols = st.columns(4)
            _cov_items = [
                (
                    "Fiscal stress",
                    True,
                    "2023 fiscal breakeven reference",
                ),
                (
                    "Reserve runway",
                    True,
                    "2023 SWF / FX reserve reference",
                ),
                (
                    "Social stability",
                    True,
                    "2023 food security + WB inflation panel",
                ),
                (
                    "Chain transmission",
                    not _chain_missing_any,
                    "chain_transmission.csv year match"
                    if not _chain_missing_any
                    else f"No chain data for {_bt_active_year} — weight rescaled to 0",
                ),
            ]
            for _cov_col, (_lbl, _ok, _detail) in zip(_cov_cols, _cov_items):
                _cov_col.metric(
                    ("✅ " if _ok else "⚠️ ") + _lbl,
                    "Available" if _ok else "Rescaled",
                    _detail,
                    delta_color="off",
                )

            st.markdown("---")

            # ── 2. Results table ─────────────────────────────────────────────
            st.subheader(
                f"Right Now Risk scores — {_bt_active_year} "
                f"(Brent ${_bt_brent:.2f}, {_bt_active_scenario})"
            )

            _bt_col_map: dict[str, str] = {
                "country_label":                      "Country",
                "right_now_risk_score":               "Right Now Risk",
                "fiscal_stress_score":                "Fiscal",
                "reserve_runway_risk":                "Runway",
                "social_stability_risk":              "Social",
                "chain_transmission_severity_recent": "Chain",
                "primary_driver":                     "Driver",
                "missing_components_count":           "Missing#",
                "rescaled_weights":                   "Weights used",
            }
            _bt_col_map = {k: v for k, v in _bt_col_map.items() if k in _bt_df.columns}
            _bt_tbl = (
                _bt_df.sort_values("right_now_risk_score", ascending=False)
                [list(_bt_col_map.keys())]
                .rename(columns=_bt_col_map)
                .round(4)
            )
            st.dataframe(
                _bt_tbl,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Right Now Risk": st.column_config.ProgressColumn(
                        "Right Now Risk", format="%.3f", min_value=0.0, max_value=1.0,
                        help="Composite score; rescaled weights when chain absent",
                    ),
                    "Fiscal": st.column_config.ProgressColumn(
                        "Fiscal", format="%.3f", min_value=0.0, max_value=1.0,
                    ),
                    "Runway": st.column_config.ProgressColumn(
                        "Runway", format="%.3f", min_value=0.0, max_value=1.0,
                    ),
                    "Social": st.column_config.ProgressColumn(
                        "Social", format="%.3f", min_value=0.0, max_value=1.0,
                    ),
                    "Chain": st.column_config.ProgressColumn(
                        "Chain", format="%.3f", min_value=0.0, max_value=1.0,
                    ),
                    "Missing#": st.column_config.NumberColumn(
                        "Missing#", format="%d", help="Number of components absent (weight rescaling active)",
                    ),
                },
            )
            make_csv_download_button(_bt_tbl, "backtest_scores.csv", "Download backtest as CSV")

            st.markdown("---")

            # ── 3. Comparison chart: backtest vs live ────────────────────────
            st.subheader("Backtest vs Current Live — Right Now Risk")
            st.caption(
                f"Live scores use today's Brent (${_live_brent:.2f}/bbl if available); "
                f"backtest scores use the {_bt_active_year} annual average "
                f"(${_bt_brent:.2f}/bbl).  "
                "Reference data (breakeven, reserves, food) is identical in both — "
                "only the Brent price and scenario bands differ."
            )

            _live_data = load_fiscal_stress_data()
            if "error" not in _live_data and "right_now_risk_df" in _live_data:
                _live_rnr = (
                    _live_data["right_now_risk_df"]
                    [["country_code_a3", "country_label", "right_now_risk_score"]]
                    .copy()
                    .rename(columns={"right_now_risk_score": "live_score"})
                )
                _bt_rnr = (
                    _bt_df[["country_code_a3", "right_now_risk_score"]]
                    .copy()
                    .rename(columns={"right_now_risk_score": "bt_score"})
                )
                _cmp = (
                    _live_rnr
                    .merge(_bt_rnr, on="country_code_a3", how="inner")
                    .sort_values("bt_score", ascending=False)
                )

                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Bar(
                    x=_cmp["country_label"],
                    y=_cmp["bt_score"],
                    name=f"{_bt_active_year} backtest",
                    marker_color="#e45756",
                    hovertemplate="<b>%{x}</b><br>Backtest: %{y:.3f}<extra></extra>",
                ))
                fig_cmp.add_trace(go.Bar(
                    x=_cmp["country_label"],
                    y=_cmp["live_score"],
                    name="Live (today)",
                    marker_color="#4c78a8",
                    hovertemplate="<b>%{x}</b><br>Live: %{y:.3f}<extra></extra>",
                ))
                fig_cmp.update_layout(
                    barmode="group",
                    height=420,
                    xaxis=dict(title=None, tickangle=-30),
                    yaxis=dict(title="Right Now Risk", range=[0, 1.0]),
                    legend=dict(orientation="h", y=-0.22, font_size=12),
                    margin=dict(l=10, r=10, t=10, b=90),
                )
                st.plotly_chart(fig_cmp, use_container_width=True)

                # Delta table inside expander
                _cmp["delta"] = (_cmp["bt_score"] - _cmp["live_score"]).round(4)
                _delta_disp = (
                    _cmp[["country_label", "bt_score", "live_score", "delta"]]
                    .sort_values("delta", ascending=False)
                    .rename(columns={
                        "country_label": "Country",
                        "bt_score":      f"{_bt_active_year} score",
                        "live_score":    "Live score",
                        "delta":         "Delta (BT minus Live)",
                    })
                    .round(4)
                )
                with st.expander("Score delta table (backtest minus live)"):
                    st.dataframe(_delta_disp, hide_index=True, use_container_width=True)
                    make_csv_download_button(_delta_disp, "backtest_delta.csv", "Download delta as CSV")
            else:
                st.info(
                    "Live Right Now Risk scores unavailable — "
                    "comparison chart cannot be drawn."
                )

            st.markdown("---")

            # ── 4. Methodology expander ──────────────────────────────────────
            with st.expander("What this backtest does and does not test"):
                st.markdown(f"""
**What it tests**

The backtest asks: *"What would the Right Now Risk score have been for each MENA country
if oil had been priced at the {_bt_active_year} annual average (${_bt_brent:.2f}/bbl)?"*

It varies:
- **Brent crude price** — historical annual average sourced from yfinance (`BZ=F`);
  falls back to an embedded EIA / World Bank reference table for years where yfinance
  data is unavailable.
- **Scenario uncertainty bands** — `stress` substitutes `breakeven_high_usd`,
  `liquid_buffer_low_usd_bn`, and `monthly_burn_high_usd_bn` from the reference CSVs;
  `favorable` (optimistic) substitutes the low/high counterparts.
- **Chain transmission** — filtered to rows where `year == {_bt_active_year}` in
  `outputs/tables/chain_transmission.csv`.  The current file is a 2024 static snapshot,
  so chain data is absent for all pre-2024 years; the remaining three components receive
  proportionally rescaled weights (fiscal 0.35→0.47, runway 0.25→0.33, social 0.20→0.27).

It holds constant (2023 reference):
- **Fiscal breakeven estimates** — IMF Article IV preliminary estimates.
- **Reserve / SWF figures** — SWF annual reports and central bank bulletins.
- **Food security** — World Bank WDI 2022 / FAO FAOSTAT 2021.

**What it does not test**

- **True historical state**: a country that reformed its subsidy system in 2018 still
  appears with 2023 breakeven and reserve figures.
- **Intra-year price volatility**: annual averages mask within-year swings
  (e.g. 2008: $96 average conceals the $147 peak and $32 trough).
- **Policy responses**: exchange-rate adjustments, monetary tightening, emergency
  borrowing, and non-linear subsidy reform dynamics are not modelled.
- **Political transitions**: leadership changes and conflict escalations that altered
  fiscal capacity between the reference year and the snapshot year.

**Interpreting the comparison chart**

Both the backtest and live scores use the same model structure and 2023 reference data.
The difference reflects **oil-price sensitivity** (and scenario effects) only — it does
not represent an actual change in a country's underlying fiscal position.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8 · Historical Risk Index  (Sprint 4)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_hist:
    st.header("Historical Risk Index — 2015–2024")
    st.caption(
        "Conditional backtest panel: Right Now Risk score per country per year, "
        "holding 2023 reference data fixed while varying the historical Brent price.  "
        "Chain component available for 2024 only; earlier years use rescaled weights "
        "(fiscal 47% · runway 33% · social 27%)."
    )

    _hist_df = load_historical_index()

    if _hist_df.empty:
        st.warning(
            "Historical risk index not found.  \n"
            "Run from the project root:  \n"
            "`python -m src.model.historical_index`"
        )
    else:
        _h_years   = sorted(_hist_df["year"].unique())
        _h_labels  = sorted(_hist_df["country_label"].unique())
        _h_yr_min  = int(min(_h_years))
        _h_yr_max  = int(max(_h_years))

        # ── KPI row ────────────────────────────────────────────────────────────
        hk1, hk2, hk3, hk4 = st.columns(4)

        _h_2024 = _hist_df[_hist_df["year"] == _h_yr_max]
        _h_2015 = _hist_df[_hist_df["year"] == _h_yr_min]

        _h_top24  = _h_2024.loc[_h_2024["right_now_risk_score"].idxmax()]
        _h_mean24 = _h_2024["right_now_risk_score"].mean()
        _h_mean15 = _h_2015["right_now_risk_score"].mean()
        _h_brent_range = (
            f"${_hist_df['historical_brent_usd'].min():.0f}–"
            f"${_hist_df['historical_brent_usd'].max():.0f}/bbl"
        ) if "historical_brent_usd" in _hist_df.columns else "—"

        hk1.metric(
            "Highest risk (2024)",
            _h_top24.get("country_label", "—"),
            f"{_h_top24['right_now_risk_score']:.3f}",
            delta_color="inverse",
        )
        hk2.metric("Mean score (2024)", f"{_h_mean24:.3f}")
        hk3.metric(
            "Mean score change",
            f"{_h_mean24:.3f} ({_h_yr_max})",
            f"{_h_mean24 - _h_mean15:+.3f} vs {_h_yr_min}",
            delta_color="inverse",
        )
        hk4.metric("Brent range covered", _h_brent_range, f"{_h_yr_min}–{_h_yr_max}")

        st.markdown("---")

        # ── 1. Line chart — risk score over time per country ──────────────────
        st.subheader("Right Now Risk Score — Country Trends (2015–2024)")

        _h_line_filter = st.multiselect(
            "Filter countries (leave empty to show all)",
            options=_h_labels,
            default=[],
            key="hist_line_filter",
        )
        _h_plot_df = (
            _hist_df[_hist_df["country_label"].isin(_h_line_filter)]
            if _h_line_filter else _hist_df
        )

        fig_hist_line = px.line(
            _h_plot_df.sort_values("year"),
            x="year",
            y="right_now_risk_score",
            color="country_label",
            color_discrete_map=_COLOUR_MAP,
            markers=True,
            labels={
                "year": "Year",
                "right_now_risk_score": "Right Now Risk",
                "country_label": "Country",
            },
            hover_data={
                "historical_brent_usd": ":.2f" if "historical_brent_usd" in _h_plot_df.columns else False,
                "right_now_risk_score": ":.3f",
            },
        )
        fig_hist_line.update_layout(
            height=460,
            yaxis=dict(range=[0, 1.0], title="Right Now Risk Score"),
            xaxis=dict(title=None, dtick=1),
            legend=dict(
                title="Country", orientation="v",
                yanchor="top", y=1, xanchor="left", x=1.02,
                font_size=11,
            ),
            margin=dict(l=10, r=10, t=10, b=40),
        )
        st.plotly_chart(fig_hist_line, use_container_width=True)

        st.markdown("---")

        # ── 2. Heatmap — countries × years ───────────────────────────────────
        st.subheader("Heatmap — Right Now Risk by Country and Year")
        st.caption("Colour scale: green = low risk · red = high risk.")

        _h_pivot = (
            _hist_df
            .pivot(index="country_label", columns="year", values="right_now_risk_score")
            .fillna(float("nan"))
        )
        _h_avg_score = _hist_df.groupby("country_label")["right_now_risk_score"].mean()
        _h_heat_order = _h_avg_score.sort_values(ascending=False).index.tolist()
        _h_pivot = _h_pivot.loc[[c for c in _h_heat_order if c in _h_pivot.index]]

        fig_hist_heat = go.Figure(go.Heatmap(
            z=_h_pivot.values,
            x=[str(y) for y in _h_pivot.columns],
            y=_h_pivot.index.tolist(),
            colorscale="RdYlGn_r",
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="Score", thickness=14),
            hovertemplate="<b>%{y}</b> · %{x}<br>Risk: %{z:.3f}<extra></extra>",
            text=_h_pivot.round(3).values,
            texttemplate="%{z:.3f}",
            textfont=dict(size=9),
        ))
        fig_hist_heat.update_layout(
            height=460,
            margin=dict(l=10, r=10, t=10, b=60),
            xaxis=dict(title="Year", tickangle=0),
            yaxis_title=None,
        )
        st.plotly_chart(fig_hist_heat, use_container_width=True)

        st.markdown("---")

        # ── 3. Rank shift table: 2015 rank vs 2024 rank vs today ─────────────
        st.subheader("Rank Shift — 2015 → 2024 → Today")
        st.caption(
            "Rank 1 = highest risk.  Today's rank uses the live Right Now Risk scores "
            "(same Brent as Tab 6)."
        )

        def _rank_col(df_yr: pd.DataFrame, label: str) -> pd.Series:
            return (
                df_yr.set_index("country_label")["right_now_risk_score"]
                .rank(ascending=False, method="min")
                .astype(int)
                .rename(label)
            )

        _r15  = _rank_col(_h_2015, f"Rank {_h_yr_min}")
        _r24  = _rank_col(_h_2024, f"Rank {_h_yr_max}")

        _rank_tbl = pd.concat([_r15, _r24], axis=1).reset_index()
        _rank_tbl.columns = ["Country", f"Rank {_h_yr_min}", f"Rank {_h_yr_max}"]
        _rank_tbl["Shift"] = _rank_tbl[f"Rank {_h_yr_min}"] - _rank_tbl[f"Rank {_h_yr_max}"]

        _live_data_h = load_fiscal_stress_data()
        if "right_now_risk_df" in _live_data_h:
            _live_rnr_h = _live_data_h["right_now_risk_df"].copy()
            _live_rnr_h["country_label"] = _live_rnr_h.get(
                "country_label",
                _live_rnr_h.get("country_code_a3", pd.Series(dtype=str))
            )
            _r_today = (
                _live_rnr_h
                .set_index("country_label")["right_now_risk_score"]
                .rank(ascending=False, method="min")
                .astype(int)
                .rename("Rank Today")
            )
            _rank_tbl = _rank_tbl.merge(
                _r_today.reset_index(), on="Country", how="left"
            )
            _rank_tbl["Shift vs Today"] = (
                _rank_tbl[f"Rank {_h_yr_min}"] - _rank_tbl["Rank Today"]
            )

        _rank_tbl = _rank_tbl.sort_values(f"Rank {_h_yr_max}").reset_index(drop=True)
        st.dataframe(
            _rank_tbl,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Shift": st.column_config.NumberColumn(
                    "Shift 2015→2024",
                    help="Positive = moved up in risk ranking (became relatively riskier); "
                         "Negative = moved down (became relatively less risky).",
                    format="%+d",
                ),
                "Shift vs Today": st.column_config.NumberColumn(
                    "Shift 2015→Today",
                    format="%+d",
                ),
            },
        )
        make_csv_download_button(_rank_tbl, "historical_rank_shifts.csv", "Download rank shifts as CSV")

        st.markdown("---")

        # ── 4. Methodology expander ───────────────────────────────────────────
        with st.expander("Methodology — Historical Risk Index"):
            st.markdown("""
**Source data**

Generated by `python -m src.model.historical_index` using the base scenario.
Output: `outputs/tables/historical_risk_index.csv` (140 rows = 14 countries × 10 years).

**What varies per year**

| Variable | Source |
|----------|--------|
| Brent crude price | yfinance `BZ=F` annual average; EIA/WB fallback table |
| Chain transmission | `chain_transmission.csv` filtered to `year == <snapshot_year>` |

**What is held constant (2023 reference)**

- Fiscal breakeven estimates — IMF Article IV 2023
- Reserve / SWF figures — SWF annual reports 2023
- Food security — World Bank WDI 2022 / FAO FAOSTAT 2021

**Chain component coverage**

The `chain_transmission.csv` is a static 2024 snapshot.  For years 2015–2023 the
chain component is NaN; the remaining three components are rescaled proportionally:
fiscal 0.35→0.47, runway 0.25→0.33, social 0.20→0.27.  The `missing_components`
column in the source CSV records this for every row.

**Rank shift interpretation**

Positive shift (e.g. +3) means the country rose 3 places in the risk ranking —
it became *relatively more risky* compared to the reference year.
Negative shift means it became relatively less risky.
Shifts reflect both oil-price sensitivity and structural differences between countries.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 9 · Sensitivity Analysis  (Sprint 5)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_sens:
    st.header("Sensitivity Analysis — Component Weight Variation")
    st.caption(
        "One-at-a-time (OAT) analysis: each of the four Right Now Risk component "
        "weights is varied ±0.10 from its default in 0.05 steps.  The remaining "
        "weights are proportionally renormalized so all four always sum to 1.0."
    )

    _sens_df = load_sensitivity()

    if _sens_df.empty:
        st.warning(
            "Sensitivity results not found.  \n"
            "Run from the project root:  \n"
            "`python -m src.model.sensitivity`"
        )
    else:
        from src.model.sensitivity import summarize_sensitivity, _BASE_WEIGHTS, _SHORT_NAME

        _sum_df = summarize_sensitivity(_sens_df)

        # ── KPI row ────────────────────────────────────────────────────────────
        sk1, sk2, sk3, sk4 = st.columns(4)

        _stable   = _sum_df.iloc[-1]
        _volatile = _sum_df.iloc[0]
        _n_scen   = int(_sens_df["scenario_id"].nunique())

        sk1.metric(
            "Most stable country",
            _stable.get("country_label", _stable["country_code_a3"]),
            f"σ = {_stable['rank_volatility']:.3f}",
        )
        sk2.metric(
            "Most sensitive country",
            _volatile.get("country_label", _volatile["country_code_a3"]),
            f"σ = {_volatile['rank_volatility']:.3f}",
            delta_color="inverse",
        )
        sk3.metric("Scenarios tested", str(_n_scen), "OAT ±0.10 step 0.05")
        sk4.metric(
            "Countries analysed",
            str(_sens_df["country_code_a3"].nunique()),
        )

        st.markdown("---")

        # ── 1. Rank volatility bar chart ──────────────────────────────────────
        st.subheader("Rank Volatility by Country")
        st.caption(
            "σ(rank) across all 17 OAT scenarios.  Higher = rank changes "
            "more when component weights shift."
        )

        fig_vol = px.bar(
            _sum_df.sort_values("rank_volatility", ascending=True),
            x="rank_volatility",
            y="country_label" if "country_label" in _sum_df.columns else "country_code_a3",
            orientation="h",
            color="rank_volatility",
            color_continuous_scale="RdYlGn_r",
            range_color=[0, _sum_df["rank_volatility"].max() * 1.05],
            text=_sum_df.sort_values("rank_volatility")["rank_volatility"].map(lambda v: f"{v:.3f}"),
            labels={
                "rank_volatility": "Rank Volatility σ",
                "country_label": "",
                "country_code_a3": "",
            },
        )
        fig_vol.update_traces(textposition="outside")
        fig_vol.update_layout(
            height=460,
            coloraxis_showscale=False,
            xaxis=dict(title="Rank volatility (std dev of rank)", range=[0, _sum_df["rank_volatility"].max() * 1.2]),
            margin=dict(l=10, r=80, t=10, b=40),
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        st.markdown("---")

        # ── 2. Interactive slider — live-recomputed bar chart ─────────────────
        st.subheader("Interactive Weight Sensitivity")
        st.caption(
            "Adjust the fiscal stress weight below.  "
            "The remaining three weights are proportionally renormalized.  "
            "The chart shows the resulting Right Now Risk score ranking."
        )

        _s_fiscal_w = st.slider(
            "Fiscal stress weight",
            min_value=0.15,
            max_value=0.55,
            value=0.35,
            step=0.05,
            format="%.2f",
            key="sens_fiscal_slider",
            help=(
                "Default = 0.35.  "
                "Remaining weights (runway=0.25, social=0.20, chain=0.20) "
                "are renormalized proportionally."
            ),
        )

        # Renormalize other weights
        _others_base = {k: v for k, v in _BASE_WEIGHTS.items()
                        if k != "fiscal_stress_score"}
        _others_total = sum(_others_base.values())
        _remaining = 1.0 - _s_fiscal_w
        _live_weights = {k: v / _others_total * _remaining for k, v in _others_base.items()}
        _live_weights["fiscal_stress_score"] = _s_fiscal_w

        # Load the live base component scores
        _sens_live_data = load_fiscal_stress_data()
        if "right_now_risk_df" in _sens_live_data:
            _base_scores = _sens_live_data["right_now_risk_df"].copy()

            _comp_cols = [
                "fiscal_stress_score", "reserve_runway_risk",
                "social_stability_risk", "chain_transmission_severity_recent",
            ]

            def _apply_w(row):
                av = {c: (row[c], _live_weights[c]) for c in _comp_cols
                      if c in row.index and not pd.isna(row[c])}
                if not av:
                    return float("nan")
                tw = sum(w for _, w in av.values())
                return min(1.0, max(0.0, sum(v * w / tw for v, w in av.values())))

            _base_scores["adj_score"] = _base_scores.apply(_apply_w, axis=1)
            _base_scores["adj_rank"] = (
                _base_scores["adj_score"]
                .rank(ascending=False, method="min")
                .astype(int)
            )

            _lbl_col = "country_label" if "country_label" in _base_scores.columns else "country_code_a3"

            _adj_bar = _base_scores.sort_values("adj_score", ascending=True)

            # Side-by-side: default vs adjusted weight scores
            _default_scores = _base_scores[["right_now_risk_score", _lbl_col]].copy()

            _weights_str = (
                f"fiscal={_s_fiscal_w:.2f}  ·  "
                f"runway={_live_weights['reserve_runway_risk']:.2f}  ·  "
                f"social={_live_weights['social_stability_risk']:.2f}  ·  "
                f"chain={_live_weights['chain_transmission_severity_recent']:.2f}"
            )
            st.caption(f"Active weights → {_weights_str}")

            fig_adj = go.Figure()
            fig_adj.add_trace(go.Bar(
                x=_adj_bar["adj_score"],
                y=_adj_bar[_lbl_col],
                orientation="h",
                name=f"Adjusted (fiscal={_s_fiscal_w:.2f})",
                marker_color="#e45756",
                text=_adj_bar["adj_score"].map(lambda v: f"{v:.3f}"),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Adjusted: %{x:.3f}<extra></extra>",
            ))
            fig_adj.add_trace(go.Bar(
                x=_adj_bar["right_now_risk_score"],
                y=_adj_bar[_lbl_col],
                orientation="h",
                name="Default weights",
                marker_color="#4c78a8",
                opacity=0.55,
                hovertemplate="<b>%{y}</b><br>Default: %{x:.3f}<extra></extra>",
            ))
            fig_adj.update_layout(
                barmode="overlay",
                height=460,
                xaxis=dict(range=[0, 1.0], title="Right Now Risk Score"),
                yaxis_title=None,
                legend=dict(orientation="h", y=-0.18, font_size=12),
                margin=dict(l=10, r=80, t=10, b=70),
            )
            st.plotly_chart(fig_adj, use_container_width=True)
        else:
            st.info(
                "Live component scores unavailable — "
                "interactive chart cannot be rendered."
            )

        st.markdown("---")

        # ── 3. Scenario detail table ──────────────────────────────────────────
        with st.expander("Scenario detail — all 17 OAT scenarios"):
            _pivot_scen = (
                _sens_df
                .pivot(
                    index="scenario_id",
                    columns="country_label" if "country_label" in _sens_df.columns else "country_code_a3",
                    values="rank",
                )
                .astype(int)
            )
            st.dataframe(_pivot_scen, use_container_width=True)
            make_csv_download_button(
                _pivot_scen.reset_index(),
                "sensitivity_pivot.csv",
                "Download pivot as CSV",
            )

        # ── 4. Methodology expander ───────────────────────────────────────────
        with st.expander("Methodology — Sensitivity Analysis"):
            st.markdown("""
**Design: one-at-a-time (OAT)**

Each of the four component weights is varied independently while the other
three are proportionally renormalized so all weights sum to 1.0:

```
when fiscal_w = new_value:
    runway_w_adj = 0.25 / (0.25 + 0.20 + 0.20) × (1 − new_value)
    social_w_adj = 0.20 / (0.25 + 0.20 + 0.20) × (1 − new_value)
    chain_w_adj  = 0.20 / (0.25 + 0.20 + 0.20) × (1 − new_value)
```

**Weight grid**

| Component | Default | Levels tested |
|-----------|---------|---------------|
| Fiscal stress | 0.35 | 0.25, 0.30, 0.40, 0.45 |
| Reserve runway | 0.25 | 0.15, 0.20, 0.30, 0.35 |
| Social stability | 0.20 | 0.10, 0.15, 0.25, 0.30 |
| Chain transmission | 0.20 | 0.10, 0.15, 0.25, 0.30 |

Total: 16 non-base + 1 base = **17 scenarios × 14 countries = 238 rows.**

**Rank volatility**

σ(rank) across all 17 scenarios per country.  A value of 0 means the
country's rank never changes regardless of weight choice; high σ means
the country's ranking is sensitive to how the model is parameterised.

**Source data**

`outputs/tables/sensitivity_results.csv` — regenerate with:
`python -m src.model.sensitivity`

Base component scores come from the live Right Now Risk pipeline
(`run_right_now_risk()` with today's Brent price).
""")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 10 · 2020 Oil Crash Retrospective
# ═══════════════════════════════════════════════════════════════════════════════
with tab_retro:
    st.header("2020 Oil Crash Retrospective")
    st.caption(
        "How well did the pre-crisis (2019) model rankings predict actual 2020 economic outcomes?  "
        "Source: IMF World Economic Outlook April 2021 · World Bank Global Economic Prospects Jan 2021."
    )

    _retro_df = load_retrospective()

    if _retro_df.empty:
        st.info(
            "Retrospective data not found.  \n"
            "Run: `python -m src.model.retrospective`  \n"
            "(requires `outputs/tables/historical_risk_index.csv`)"
        )
    else:
        # ── KPI row ────────────────────────────────────────────────────────────
        try:
            from src.model.retrospective import summarize_retrospective as _sum_retro
            _retro_summary = _sum_retro(_retro_df)
            rk1, rk2, rk3, rk4 = st.columns(4)
            rk1.metric("Spearman ρ (pre-crisis rank vs outcome)",
                       f"{_retro_summary['spearman_r']:.3f}",
                       help="Rank correlation between 2019 model rank and 2020 GDP severity rank. "
                            "1.0 = perfect prediction.")
            rk2.metric("Hit rate (±3 rank positions)",
                       f"{_retro_summary['hit_rate']:.0%}",
                       help=f"% of countries where |model_rank - outcome_rank| ≤ {_retro_summary['hit_threshold']}")
            rk3.metric("Mean absolute rank error",
                       f"{_retro_summary['rank_error_mean']:.1f}",
                       help="Average |model_rank_2019 − outcome_severity_rank_2020|")
            rk4.metric("Top-5 precision / recall",
                       f"{_retro_summary['top5_precision']:.0%} / {_retro_summary['top5_recall']:.0%}",
                       help="Among countries model placed in top-5 highest risk, how many were actually "
                            "in the worst-5 outcomes (precision); and of the actual worst-5, how many "
                            "did the model flag (recall).")
        except Exception:
            pass

        st.markdown("---")

        # ── Scatter: model rank 2019 vs outcome severity rank 2020 ────────────
        st.subheader("Pre-Crisis Model Rank vs 2020 Outcome Severity")
        st.caption(
            "Each bubble is a country.  X-axis = model's 2019 Right Now Risk rank "
            "(1 = most at-risk predicted).  Y-axis = actual 2020 GDP severity rank "
            "(1 = worst outcome, e.g. Libya −59.7%).  "
            "Points on the diagonal = perfect prediction."
        )

        _r_have = "model_rank_precris" in _retro_df.columns and "outcome_severity_rank" in _retro_df.columns
        if _r_have:
            _label_col = "country_label" if "country_label" in _retro_df.columns else "country_code_a3"
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=_retro_df["model_rank_precris"],
                y=_retro_df["outcome_severity_rank"],
                mode="markers+text",
                text=_retro_df[_label_col],
                textposition="top center",
                textfont=dict(size=10),
                marker=dict(
                    size=14,
                    color=_retro_df["rank_error"] if "rank_error" in _retro_df.columns else "#1f77b4",
                    colorscale="RdYlGn_r",
                    colorbar=dict(title="Rank error", thickness=12),
                    showscale=True,
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Model rank 2019: %{x}<br>"
                    "Outcome rank 2020: %{y}<br>"
                    "<extra></extra>"
                ),
            ))
            # Perfect prediction diagonal
            _max_r = max(
                _retro_df["model_rank_precris"].max(),
                _retro_df["outcome_severity_rank"].max(),
            ) + 0.5
            fig_scatter.add_trace(go.Scatter(
                x=[0.5, _max_r], y=[0.5, _max_r],
                mode="lines",
                line=dict(dash="dash", color="grey", width=1),
                showlegend=False,
                hoverinfo="skip",
            ))
            fig_scatter.update_layout(
                height=480,
                xaxis=dict(title="Model rank 2019 (1 = highest predicted risk)", dtick=1),
                yaxis=dict(title="Actual 2020 outcome severity rank (1 = worst)", dtick=1),
                margin=dict(l=10, r=10, t=10, b=60),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("---")

        # ── Data table ─────────────────────────────────────────────────────────
        with st.expander("Full retrospective data table"):
            _retro_show = {
                "country_label":           "Country",
                "outcome_severity_rank":   "Outcome Rank 2020",
                "gdp_growth_2020_pct":     "GDP Growth 2020 (%)",
                "fiscal_balance_2020_pct_gdp": "Fiscal Balance 2020 (% GDP)",
                "imf_emergency_program":   "IMF Emergency Program",
                "model_rank_precris":      "Model Rank 2019",
                "right_now_risk_score":    "Model Score 2019",
                "rank_error":              "Rank Error",
                "hit":                     "Hit (±3)",
                "confidence":              "Confidence",
            }
            _retro_show = {k: v for k, v in _retro_show.items() if k in _retro_df.columns}
            _retro_disp = (
                _retro_df[list(_retro_show.keys())]
                .rename(columns=_retro_show)
                .round({"GDP Growth 2020 (%)": 1, "Model Score 2019": 3, "Rank Error": 1})
            )
            st.dataframe(_retro_disp, hide_index=True, use_container_width=True)
            make_csv_download_button(_retro_disp, "retrospective_2020.csv", "Download as CSV")

        # ── Methodology ────────────────────────────────────────────────────────
        with st.expander("Methodology — 2020 Retrospective"):
            st.markdown("""
**Pre-crisis snapshot year:** 2019 (last full year before the crisis).

**Outcome measure:** GDP growth rate (%) from IMF WEO April 2021 Table A7.
Countries ranked 1 (worst, e.g. Libya −59.7%) to 14 (best, Egypt +3.6%).

**Correlation metric:** Spearman rank correlation ρ between the model's 2019
Right Now Risk rank and the 2020 outcome severity rank.  A perfect model would
produce ρ = 1.0.

**Hit rate:** Proportion of countries where |model_rank_2019 − outcome_rank_2020| ≤ 3.

**Important caveat:** The 2020 shock was driven partly by factors the model
could not have predicted from structural data alone:
- Libya's oil export blockade (Jan–Sep 2020) amplified an already high-risk score.
- Lebanon's financial collapse pre-dated COVID and was not primarily oil-driven.
- Morocco's drought compounded the COVID shock.
- Kuwait's fiscal hit was severe but buffered by KIA SWF assets.

**Data source:** `data/reference/imf_weo_2020_outcomes.csv`
Regenerate output: `python -m src.model.retrospective`
""")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 11 · IMF / WB Cross-Validation
# ═══════════════════════════════════════════════════════════════════════════════
with tab_cv:
    st.header("IMF / World Bank Cross-Validation")
    st.caption(
        "Compares model Right Now Risk tiers against independent benchmarks from the "
        "IMF Fiscal Monitor (Oct 2023) and World Bank Macro Poverty Outlook (Fall 2023)."
    )

    _cv_df = load_cross_validation()

    if _cv_df.empty:
        st.info(
            "Cross-validation data not found.  \n"
            "Run: `python -m src.model.cross_validation`"
        )
    else:
        # ── KPI row ────────────────────────────────────────────────────────────
        _cv_n_div = int(_cv_df.get("any_divergence", pd.Series([False] * len(_cv_df))).sum()) \
            if "any_divergence" in _cv_df.columns else None
        _cv_imf_r = _cv_df[["model_rank", "imf_fm_ordinal"]].dropna() \
            if "model_rank" in _cv_df.columns and "imf_fm_ordinal" in _cv_df.columns \
            else pd.DataFrame()
        _cv_wb_r  = _cv_df[["model_rank", "wb_mpo_ordinal"]].dropna() \
            if "model_rank" in _cv_df.columns and "wb_mpo_ordinal" in _cv_df.columns \
            else pd.DataFrame()

        import math as _math
        import numpy as _np

        def _sr(x, y):
            rx = pd.Series(x, dtype=float).rank()
            ry = pd.Series(y, dtype=float).rank()
            return float(_np.corrcoef(rx.values, ry.values)[0, 1]) if len(x) > 2 else float("nan")

        _r_imf = _sr(_cv_imf_r["model_rank"].tolist(), _cv_imf_r["imf_fm_ordinal"].tolist()) \
            if not _cv_imf_r.empty else float("nan")
        _r_wb  = _sr(_cv_wb_r["model_rank"].tolist(),  _cv_wb_r["wb_mpo_ordinal"].tolist()) \
            if not _cv_wb_r.empty else float("nan")

        cv1, cv2, cv3, cv4 = st.columns(4)
        cv1.metric("Spearman ρ vs IMF FM tier",
                   f"{_r_imf:.3f}" if not _math.isnan(_r_imf) else "N/A",
                   help="Rank correlation between model rank and IMF Fiscal Monitor risk tier ordinal. "
                        "Positive = model agrees with IMF on relative ordering.")
        cv2.metric("Spearman ρ vs WB MPO status",
                   f"{_r_wb:.3f}" if not _math.isnan(_r_wb) else "N/A",
                   help="Rank correlation between model rank and WB Macro Poverty Outlook status ordinal.")
        cv3.metric("Countries with divergence",
                   f"{_cv_n_div} / {len(_cv_df)}" if _cv_n_div is not None else "N/A",
                   help="Countries where model tier disagrees with IMF FM or WB MPO classification.")
        cv4.metric("Reference year", "2023",
                   help="IMF Fiscal Monitor Oct 2023 · WB Macro Poverty Outlook Fall 2023")

        st.markdown("---")

        # ── Tier comparison table ──────────────────────────────────────────────
        st.subheader("Tier Comparison — Model vs IMF FM vs WB MPO")

        _cv_show = {
            "country_label":      "Country",
            "right_now_risk_score": "Model Score",
            "model_rank":         "Model Rank",
            "model_tier":         "Model Tier",
            "imf_fm_risk_tier":   "IMF FM Tier",
            "wb_mpo_status":      "WB MPO Status",
            "imf_divergence":     "IMF Divergence",
            "wb_divergence":      "WB Divergence",
        }
        _cv_show = {k: v for k, v in _cv_show.items() if k in _cv_df.columns}
        _cv_disp = (
            _cv_df.sort_values("model_rank" if "model_rank" in _cv_df.columns else "right_now_risk_score")
            [list(_cv_show.keys())]
            .rename(columns=_cv_show)
        )
        if "Model Score" in _cv_disp.columns:
            _cv_disp["Model Score"] = _cv_disp["Model Score"].round(3)

        st.dataframe(
            _cv_disp,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Model Score": st.column_config.ProgressColumn(
                    "Model Score", format="%.3f", min_value=0.0, max_value=1.0,
                ),
                "IMF Divergence": st.column_config.CheckboxColumn(
                    "IMF Divergence",
                    help="Model tier ≠ IMF Fiscal Monitor tier",
                ),
                "WB Divergence": st.column_config.CheckboxColumn(
                    "WB Divergence",
                    help="Model tier ≠ WB MPO status",
                ),
            },
        )
        make_csv_download_button(_cv_disp, "cross_validation.csv", "Download comparison as CSV")

        st.markdown("---")

        # ── Divergence detail ─────────────────────────────────────────────────
        if "any_divergence" in _cv_df.columns:
            _div_rows = _cv_df[_cv_df["any_divergence"]].copy()
            if not _div_rows.empty:
                st.subheader(f"Divergence Detail ({len(_div_rows)} countries)")
                st.caption(
                    "These countries have at least one benchmark tier that differs from "
                    "the model's tertile-based tier.  Review the notes column for context."
                )
                _div_show = {
                    "country_label":        "Country",
                    "model_tier":           "Model Tier",
                    "imf_fm_risk_tier":     "IMF FM Tier",
                    "wb_mpo_status":        "WB MPO Status",
                    "imf_ordinal_distance": "IMF Distance",
                    "wb_ordinal_distance":  "WB Distance",
                    "right_now_risk_score": "Model Score",
                }
                _div_show = {k: v for k, v in _div_show.items() if k in _div_rows.columns}
                st.dataframe(
                    _div_rows[list(_div_show.keys())].rename(columns=_div_show).round(3),
                    hide_index=True,
                    use_container_width=True,
                )

        # ── Methodology ────────────────────────────────────────────────────────
        with st.expander("Methodology — Cross-Validation"):
            st.markdown("""
**Tier mapping**

The model's composite Right Now Risk score is divided into three tiers using
data-driven tertile thresholds (33rd and 67th percentiles of the current score
distribution), giving roughly equal-sized groups:
- **Low**: bottom third by score
- **Medium**: middle third
- **High**: top third

**Benchmark sources**
- **IMF Fiscal Monitor (Oct 2023)** — Table A8, MENA region fiscal vulnerability classification.
- **WB Macro Poverty Outlook (Fall 2023)** — Country notes annex, Stable / Watch / Stressed.

Both benchmarks carry `source_id_primary = IMF_FM_OCT2023` or `WB_MPO_2023`
in `data/reference/imf_wb_benchmarks.csv`.

**Correlation metric**

Spearman rank correlation ρ between model rank (1 = highest risk) and benchmark
tier ordinal (Low=1, Medium=2, High=3 for IMF FM; Stable=1, Watch=2, Stressed=3
for WB MPO).  A positive ρ means the model broadly agrees with the benchmark
on which countries are more vs less at risk.

**Divergences**

A divergence is recorded when model_tier ≠ imf_fm_risk_tier or model_tier ≠ wb_mpo_status.
The ordinal distance (|model_ordinal − benchmark_ordinal|) measures severity.

**Regenerate:** `python -m src.model.cross_validation`
""")
