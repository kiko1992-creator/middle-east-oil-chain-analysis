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
    st.caption("Source: World Bank Open Data  \nBuilt with Streamlit + Plotly")


# ── Utility: apply country + year filter to panel ─────────────────────────────
def _filter(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["country_label"].isin(selected_countries) &
        df["year"].between(year_range[0], year_range[1])
    ]


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_rank, tab_rents, tab_shock, tab_macro, tab_chain, tab_fiscal = st.tabs([
    "📊  OCVI Rankings",
    "🛢️  Oil Rents % GDP",
    "⚡  Price Shock",
    "📈  GDP Growth vs Inflation",
    "⛓️  Chain Transmission",
    "🚨  Fiscal Stress",
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
