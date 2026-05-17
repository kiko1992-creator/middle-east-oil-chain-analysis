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
_ROOT       = Path(__file__).resolve().parents[1]
PANEL_PATH  = _ROOT / "data" / "processed" / "world_bank_panel.csv"
OCVI_PATH   = _ROOT / "outputs" / "tables" / "ocvi_scores.csv"

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

_all_labels   = sorted(panel["country_label"].unique())
_yr_min       = int(panel["year"].min())
_yr_max       = int(panel["year"].max())


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

    st.markdown("---")
    st.caption("Source: World Bank Open Data  \nBuilt with Streamlit + Plotly")


# ── Utility: apply country + year filter to panel ─────────────────────────────
def _filter(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["country_label"].isin(selected_countries) &
        df["year"].between(year_range[0], year_range[1])
    ]


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_rank, tab_rents, tab_shock, tab_macro = st.tabs([
    "📊  OCVI Rankings",
    "🛢️  Oil Rents % GDP",
    "⚡  Price Shock",
    "📈  GDP Growth vs Inflation",
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
        for i, row in ocvi.iterrows():
            vals = [row[c] for c in norm_cols]
            vals_closed = vals + [vals[0]]
            labels_closed = axis_labels + [axis_labels[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_closed,
                theta=labels_closed,
                fill="toself",
                opacity=0.25,
                name=row["country_label"],
                line=dict(color=_PALETTE[i % len(_PALETTE)]),
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
            color_discrete_sequence=_PALETTE,
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
        "**Bubble size:** GDP (USD)  ·  Each dot = one country-year observation."
    )

    df_macro = _filter(panel).dropna(subset=["gdp_growth_pct", "FP_CPI_TOTL_ZG"]).copy()

    if df_macro.empty:
        st.warning("No data for the current selection. Adjust the filters.")
    else:
        # Clip extreme inflation for axis readability; annotate clipping
        p99 = df_macro["FP_CPI_TOTL_ZG"].quantile(0.99)
        clipped_n = (df_macro["FP_CPI_TOTL_ZG"] > p99).sum()
        df_macro["infl_clipped"] = df_macro["FP_CPI_TOTL_ZG"].clip(upper=p99)

        if clipped_n > 0:
            st.info(
                f"ℹ️  {clipped_n} observation(s) with inflation > {p99:.0f}% clipped on the "
                f"Y-axis for readability (Lebanon hyperinflation period). "
                f"Full values visible in hover tooltips."
            )

        # Bubble size proportional to GDP; guard against NaN/zero
        df_macro["gdp_bn"] = df_macro["NY_GDP_MKTP_CD"].fillna(0).clip(lower=1e8) / 1e9

        fig_macro = px.scatter(
            df_macro,
            x="gdp_growth_pct",
            y="infl_clipped",
            color="country_label",
            size="gdp_bn",
            size_max=35,
            hover_name="country_label",
            hover_data={
                "year": True,
                "gdp_growth_pct": ":.1f",
                "FP_CPI_TOTL_ZG": ":.1f",   # show true (unclipped) value in tooltip
                "infl_clipped": False,
                "gdp_bn": ":.0f",
            },
            labels={
                "gdp_growth_pct": "GDP Growth (%)",
                "infl_clipped":   "Inflation — CPI (%, clipped at p99)",
                "country_label":  "Country",
                "gdp_bn":         "GDP (USD bn)",
                "FP_CPI_TOTL_ZG": "True Inflation (%)",
                "year":           "Year",
            },
            title=(
                f"GDP Growth vs Inflation · "
                f"{year_range[0]}–{year_range[1]} · "
                f"bubble size = GDP"
            ),
            color_discrete_sequence=_PALETTE,
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
