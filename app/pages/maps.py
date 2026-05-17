"""
Maps — Geospatial OCVI and Oil Rents % GDP choropleth maps.

Two Plotly choropleth maps zoomed to the MENA region:
  1. Oil Chain Vulnerability Index (OCVI) — long-run country scores
  2. Oil Rents % GDP — annual value for a user-selected year

Run standalone (from project root):
    streamlit run app/pages/maps.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).resolve().parents[2]
OCVI_PATH  = _ROOT / "outputs" / "tables" / "ocvi_scores.csv"
PANEL_PATH = _ROOT / "data" / "processed" / "world_bank_panel.csv"

_LABEL: dict[str, str] = {
    "Egypt, Arab Rep.":     "Egypt",
    "Iran, Islamic Rep.":   "Iran",
    "United Arab Emirates": "UAE",
}

def _label(name: str) -> str:
    return _LABEL.get(name, name)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading OCVI scores…")
def load_ocvi() -> pd.DataFrame:
    df = pd.read_csv(OCVI_PATH)
    df["country_label"] = df["country_name"].map(_label)
    return df


@st.cache_data(show_spinner="Loading panel data…")
def load_panel() -> pd.DataFrame:
    return pd.read_csv(PANEL_PATH)


# ── Guard: check files exist ──────────────────────────────────────────────────
for _p in (OCVI_PATH, PANEL_PATH):
    if not _p.exists():
        st.error(
            f"Required file not found: `{_p}`  \n"
            "Run the data pipeline first:  \n"
            "`python -m src.data.fetch_world_bank`  \n"
            "`python -m src.data.clean_world_bank`  \n"
            "`python -m src.model.vulnerability_index`"
        )
        st.stop()

ocvi  = load_ocvi()
panel = load_panel()

# ── Shared geo layout applied to both maps ────────────────────────────────────
_GEO = dict(
    showcoastlines=True,
    coastlinecolor="white",
    showland=True,
    landcolor="#f0ede6",
    showocean=True,
    oceancolor="#daeef5",
    showlakes=False,
    showrivers=False,
    showcountries=True,
    countrycolor="white",
    countrywidth=0.5,
    lataxis_range=[10, 42],
    lonaxis_range=[-20, 65],
    projection_type="natural earth",
)

# ─────────────────────────────────────────────────────────────────────────────
st.title("MENA — Geospatial Overview")
st.caption(
    "Choropleth maps covering the 14-country MENA panel (2000–2024).  "
    "Only the 14 study countries are coloured; the rest of the world is unshaded."
)


# ═══════════════════════════════════════════════════════════════════════════════
# MAP 1 · OCVI
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Map 1 — Oil Chain Vulnerability Index (OCVI)")
st.caption(
    "Long-run composite score (2000–2024 average across 5 components).  "
    "**Red** = highest vulnerability · **Green** = lowest.  "
    "⚠ UAE and Iran are flagged for < 10 years of exports/imports data."
)

fig_ocvi = px.choropleth(
    ocvi,
    locations="country_code_a3",
    color="ocvi_score",
    hover_name="country_label",
    hover_data={
        "ocvi_rank":         True,
        "ocvi_score":        ":.3f",
        "oil_rents_pct_gdp": ":.1f",
        "data_insufficient": True,
        "country_code_a3":   False,
    },
    color_continuous_scale="RdYlGn_r",
    range_color=[0, 1],
    labels={
        "ocvi_score":        "OCVI Score",
        "ocvi_rank":         "Rank",
        "oil_rents_pct_gdp": "Oil Rents % GDP",
        "data_insufficient": "⚠ Low data",
    },
)

fig_ocvi.update_geos(**_GEO)
fig_ocvi.update_layout(
    height=500,
    margin=dict(l=0, r=0, t=10, b=0),
    coloraxis_colorbar=dict(
        title="OCVI",
        tickvals=[0, 0.25, 0.5, 0.75, 1.0],
        ticktext=["0 — low", "0.25", "0.50", "0.75", "1.0 — high"],
        len=0.65,
        thickness=14,
    ),
)

st.plotly_chart(fig_ocvi, use_container_width=True)

# Ranking table beneath the map
with st.expander("Full rankings table"):
    tbl = (
        ocvi[[
            "ocvi_rank", "country_label", "ocvi_score",
            "oil_rents_pct_gdp", "exports_pct_gdp", "imports_pct_gdp",
            "data_insufficient",
        ]]
        .sort_values("ocvi_rank")
        .copy()
    )
    tbl.columns = [
        "Rank", "Country", "OCVI Score",
        "Oil Rents % GDP", "Exports % GDP", "Imports % GDP", "⚠ Low data",
    ]
    st.dataframe(
        tbl,
        hide_index=True,
        use_container_width=True,
        column_config={
            "OCVI Score": st.column_config.ProgressColumn(
                format="%.3f", min_value=0.0, max_value=1.0,
            ),
            "Oil Rents % GDP": st.column_config.NumberColumn(format="%.1f%%"),
            "Exports % GDP":   st.column_config.NumberColumn(format="%.1f%%"),
            "Imports % GDP":   st.column_config.NumberColumn(format="%.1f%%"),
            "⚠ Low data":      st.column_config.CheckboxColumn(),
        },
    )

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# MAP 2 · Oil Rents % GDP (annual, year-selectable)
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Map 2 — Oil Rents as % of GDP")
st.caption(
    "**NY.GDP.PETR.RT.ZS** — value of crude oil production at world prices minus "
    "total production costs, as a share of GDP.  Use the slider to step through years."
)

available_years = sorted(
    panel.dropna(subset=["NY_GDP_PETR_RT_ZS"])["year"].unique()
)
selected_year = st.select_slider(
    "Select year",
    options=available_years,
    value=available_years[-1],
    key="maps_rents_year",
)

rents_df = (
    panel[panel["year"] == selected_year]
    [["country_code_a3", "country_name", "NY_GDP_PETR_RT_ZS", "NY_GDP_MKTP_CD"]]
    .dropna(subset=["NY_GDP_PETR_RT_ZS"])
    .copy()
)
rents_df["country_label"] = rents_df["country_name"].map(_label)
rents_df["gdp_usd_bn"]    = rents_df["NY_GDP_MKTP_CD"].fillna(0) / 1e9

if rents_df.empty:
    st.warning(f"No oil rents data available for {selected_year}.")
else:
    n_missing = 14 - len(rents_df)
    if n_missing > 0:
        st.info(
            f"{n_missing} of 14 countries have no oil rents data for {selected_year} "
            "and appear unshaded on the map."
        )

    fig_rents = px.choropleth(
        rents_df,
        locations="country_code_a3",
        color="NY_GDP_PETR_RT_ZS",
        hover_name="country_label",
        hover_data={
            "NY_GDP_PETR_RT_ZS": ":.1f",
            "gdp_usd_bn":        ":.0f",
            "country_code_a3":   False,
        },
        color_continuous_scale="YlOrRd",
        labels={
            "NY_GDP_PETR_RT_ZS": "Oil Rents % GDP",
            "gdp_usd_bn":        "GDP (USD bn)",
        },
        title=f"Oil Rents % GDP · {selected_year}",
    )

    fig_rents.update_geos(**_GEO)
    fig_rents.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_colorbar=dict(
            title="Oil Rents (% GDP)",
            len=0.65,
            thickness=14,
        ),
    )

    st.plotly_chart(fig_rents, use_container_width=True)

    # Year-on-year change table
    prev_year = selected_year - 1
    if prev_year in panel["year"].values:
        prev_df = (
            panel[panel["year"] == prev_year]
            [["country_code_a3", "NY_GDP_PETR_RT_ZS"]]
            .rename(columns={"NY_GDP_PETR_RT_ZS": "prev_rents"})
        )
        compare = rents_df.merge(prev_df, on="country_code_a3", how="left")
        compare["yoy_change_pp"] = compare["NY_GDP_PETR_RT_ZS"] - compare["prev_rents"]

        with st.expander(f"Year-on-year change vs {prev_year}"):
            disp = (
                compare[[
                    "country_label", "prev_rents",
                    "NY_GDP_PETR_RT_ZS", "yoy_change_pp",
                ]]
                .sort_values("yoy_change_pp", ascending=False)
                .copy()
            )
            disp.columns = [
                "Country",
                f"Oil Rents {prev_year} (%)",
                f"Oil Rents {selected_year} (%)",
                "YoY Change (pp)",
            ]
            st.dataframe(
                disp,
                hide_index=True,
                use_container_width=True,
                column_config={
                    f"Oil Rents {prev_year} (%)":    st.column_config.NumberColumn(format="%.1f%%"),
                    f"Oil Rents {selected_year} (%)": st.column_config.NumberColumn(format="%.1f%%"),
                    "YoY Change (pp)": st.column_config.NumberColumn(format="%+.2f"),
                },
            )
