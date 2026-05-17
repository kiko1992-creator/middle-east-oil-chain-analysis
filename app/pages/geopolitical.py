"""
Iran Conflict Scenario — Geopolitical Stress Test

Models the first-order economic impact of an Iran conflict on all 14 MENA
countries through two levers:
  - Oil price shock % (direct war premium)
  - Strait of Hormuz disruption % (volume blockage for Gulf exporters)

For exporters: revenue impact = oil_rents × (volume_retained × price_factor − 1)
For importers: cost shock = imports_pct_gdp × oil_import_share × effective_price_change

Run standalone (from project root):
    streamlit run app/pages/geopolitical.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

# ── Domain knowledge (not in CSV files) ───────────────────────────────────────
# Fraction of each exporter's oil shipped through the Strait of Hormuz.
# Sources: EIA (2023), IEA Oil Market Report, country-specific pipeline data.
_HORMUZ_EXPORT_PCT: dict[str, int] = {
    "IRN": 100,  # Kharg Island — all crude exports transit Hormuz
    "IRQ": 90,   # Basrah Oil Terminal (~90%); remainder via Kirkuk–Ceyhan (Turkey)
    "KWT": 100,  # No alternative export pipeline
    "QAT": 100,  # No bypass capacity for LNG or crude
    "BHR": 85,   # Sitra terminal; small fraction re-exported via Saudi pipelines
    "SAU": 65,   # East-West pipeline to Yanbu Red Sea terminal bypasses ~35%
    "ARE": 75,   # ADCO/IPIC pipeline to Fujairah (Gulf of Oman) bypasses ~25%
    "OMN": 0,    # Mina Al Fahal on Gulf of Oman — physically outside Hormuz
    "DZA": 0,    # Mediterranean coast (Arzew / Skikda)
    "LBY": 0,    # Mediterranean coast (Es Sider / Ras Lanuf)
    "EGY": 0,    # Red Sea / Mediterranean (Ain Sukhna, Sidi Kerir)
    "JOR": 0,    # Red Sea — Aqaba
    "LBN": 0,    # Mediterranean
    "MAR": 0,    # Atlantic / Mediterranean
}

# For net importers: oil & energy products as a share of total merchandise imports.
# Sources: World Bank WITS, IEA country profiles (2019–2022 avg).
_OIL_IMPORT_SHARE: dict[str, float] = {
    "JOR": 0.22,  # Jordan: ~22% of imports are petroleum products
    "LBN": 0.20,  # Lebanon
    "MAR": 0.25,  # Morocco
    "EGY": 0.13,  # Egypt (partial domestic producer; net importer since ~2010)
}

# Strait of Hormuz baseline daily flow (from config/countries.yaml)
_HORMUZ_MBD = 21.0

# A country is classified as a net oil exporter if fuel exports exceed this
# share of its total merchandise exports (TX_VAL_FUEL_ZS_UN, long-run avg).
_EXPORTER_THRESHOLD_PCT = 20.0


# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    ocvi  = pd.read_csv(OCVI_PATH)
    panel = pd.read_csv(PANEL_PATH)
    ocvi["country_label"] = ocvi["country_name"].map(_label)
    return ocvi, panel


@st.cache_data(show_spinner="Building scenario base…")
def build_base(ocvi: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Compute long-run per-country averages used in every scenario calculation."""
    avg_gdp = (
        panel.groupby("country_code_a3")["NY_GDP_MKTP_CD"]
        .mean().rename("avg_gdp_usd")
    )
    avg_fuel_pct = (
        panel.groupby("country_code_a3")["TX_VAL_FUEL_ZS_UN"]
        .mean().rename("avg_fuel_export_pct")
    )
    avg_imports = (
        panel.groupby("country_code_a3")["BM_GSR_MRCH_CD"]
        .mean().rename("avg_imports_usd")
    )

    base = (
        ocvi[[
            "country_code", "country_code_a3", "country_name", "country_label",
            "oil_rents_pct_gdp", "exports_pct_gdp", "imports_pct_gdp", "ocvi_rank",
        ]]
        .merge(avg_gdp,      left_on="country_code_a3", right_index=True, how="left")
        .merge(avg_fuel_pct, left_on="country_code_a3", right_index=True, how="left")
        .merge(avg_imports,  left_on="country_code_a3", right_index=True, how="left")
    )

    base["hormuz_export_pct"] = base["country_code_a3"].map(_HORMUZ_EXPORT_PCT).fillna(0)
    base["oil_import_share"]  = base["country_code_a3"].map(_OIL_IMPORT_SHARE).fillna(0)
    base["is_exporter"]       = base["avg_fuel_export_pct"] > _EXPORTER_THRESHOLD_PCT
    base["hormuz_dependent"]  = base["hormuz_export_pct"] > 0
    base["bypass_pct"]        = 100 - base["hormuz_export_pct"]

    return base


# ── Guard: check files ────────────────────────────────────────────────────────
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

ocvi, panel = load_data()
base = build_base(ocvi, panel)

# ─────────────────────────────────────────────────────────────────────────────
st.title("Iran Conflict Scenario — Geopolitical Stress Test")
st.markdown(
    """
    Iran controls the **northern shoreline** of the Strait of Hormuz,
    through which ~**21 mb/d** (~21% of global oil trade) transits daily.
    A conflict would simultaneously trigger a **price spike** and cut
    **export volumes** for every Gulf producer dependent on Hormuz routing.

    Adjust the two levers below to stress-test all 14 MENA economies.
    """
)

# ── Scenario controls ─────────────────────────────────────────────────────────
c_s1, c_s2 = st.columns(2)
with c_s1:
    price_shock = st.slider(
        "Oil price shock (%)",
        min_value=-60, max_value=100, value=40, step=5,
        format="%+d%%",
        help=(
            "Direct war-premium price change.  Historical benchmarks: "
            "Gulf War 1990 +130%, Gulf War 2003 +30%, Russia-Ukraine 2022 +60%."
        ),
    )
with c_s2:
    hormuz_disruption = st.slider(
        "Strait of Hormuz disruption (%)",
        min_value=0, max_value=100, value=30, step=5,
        format="%d%%",
        help=(
            "Share of Hormuz-routed oil volume blocked or severely disrupted.  "
            "30% ≈ partial mine / shipping-risk scenario; "
            "100% ≈ full closure (historically unprecedented)."
        ),
    )

# ── Calculations ──────────────────────────────────────────────────────────────
price_factor     = 1.0 + price_shock / 100.0
hormuz_frac      = hormuz_disruption / 100.0
# Hormuz supply crunch adds a further price premium (0.25 pp per 1 pp disruption).
# Derived from historical Hormuz tension episodes (2012 Iran sanctions, 2019 tanker
# seizures); documented in EIA Strait of Hormuz country analysis (2019).
hormuz_price_premium = hormuz_disruption * 0.25
effective_price_change = price_shock + hormuz_price_premium

df = base.copy()

# Exporters: fraction of normal volume still shippable
df["volume_retained"] = 1.0 - (df["hormuz_export_pct"] / 100.0) * hormuz_frac

# Revenue impact (pp of GDP): combines price effect and volume loss
df["revenue_impact_pp"] = (
    df["oil_rents_pct_gdp"] * (df["volume_retained"] * price_factor - 1.0)
)
df["revenue_impact_usd_bn"] = (
    df["revenue_impact_pp"] / 100.0 * df["avg_gdp_usd"] / 1e9
)

# Importers: oil import cost shock (pp of GDP)
df["import_shock_pp"] = (
    df["imports_pct_gdp"] * df["oil_import_share"] * effective_price_change / 100.0
)
df["import_shock_usd_bn"] = (
    df["import_shock_pp"] / 100.0 * df["avg_gdp_usd"] / 1e9
)

# Net impact for map: exporters use revenue_impact; importers negate import_shock
df["net_impact_pp"] = df.apply(
    lambda r: r["revenue_impact_pp"] if r["is_exporter"] else -r["import_shock_pp"],
    axis=1,
)

exporter_df = df[df["is_exporter"]].copy()
importer_df = df[~df["is_exporter"]].copy()

# ── KPI metrics ───────────────────────────────────────────────────────────────
mbd_at_risk      = _HORMUZ_MBD * hormuz_frac
total_rev_impact = exporter_df["revenue_impact_usd_bn"].sum()
total_imp_shock  = importer_df["import_shock_usd_bn"].sum()
n_hormuz         = int(df["hormuz_dependent"].sum())

st.markdown("---")
k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Oil at risk (Hormuz)",
    f"{mbd_at_risk:.1f} mb/d",
    f"{hormuz_disruption}% of {_HORMUZ_MBD:.0f} mb/d daily flow",
    delta_color="inverse",
)
k2.metric(
    "Exporter revenue impact",
    f"${total_rev_impact:+.1f} bn",
    "combined MENA-14 exporters",
    delta_color="normal" if total_rev_impact >= 0 else "inverse",
)
k3.metric(
    "Importer cost shock",
    f"${total_imp_shock:.1f} bn",
    f"+{effective_price_change:.0f}% effective oil price",
    delta_color="inverse",
)
k4.metric(
    "Hormuz-exposed exporters",
    f"{n_hormuz} / {len(exporter_df)}",
    "Gulf producers routing via Hormuz",
    delta_color="off",
)
st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 · Exporter / Importer split charts
# ═══════════════════════════════════════════════════════════════════════════════
col_exp, col_imp = st.columns(2, gap="large")

# ── Exporters ─────────────────────────────────────────────────────────────────
with col_exp:
    st.subheader("Exporter Revenue Impact")
    st.caption(
        "Revenue change as **pp of GDP**.  "
        "Hormuz-dependent exporters lose volume AND face the price effect.  "
        "Algeria, Libya and Oman route outside Hormuz — pure price gain."
    )

    exp_sorted = exporter_df.sort_values("revenue_impact_pp").copy()
    exp_sorted["direction"] = exp_sorted["revenue_impact_pp"].apply(
        lambda v: "Gain" if v >= 0 else "Loss"
    )
    exp_sorted["bar_label"] = exp_sorted["revenue_impact_pp"].map(lambda v: f"{v:+.1f} pp")

    fig_exp = px.bar(
        exp_sorted,
        x="revenue_impact_pp",
        y="country_label",
        orientation="h",
        color="direction",
        color_discrete_map={"Gain": "#2ca02c", "Loss": "#d62728"},
        text="bar_label",
        hover_name="country_label",
        hover_data={
            "oil_rents_pct_gdp":  ":.1f",
            "hormuz_export_pct":  ":.0f",
            "volume_retained":    ":.2f",
            "revenue_impact_usd_bn": ":.1f",
            "direction":          False,
            "bar_label":          False,
        },
        labels={
            "revenue_impact_pp":     "Revenue Impact (pp GDP)",
            "country_label":         "",
            "oil_rents_pct_gdp":     "Oil Rents % GDP",
            "hormuz_export_pct":     "Hormuz Routing %",
            "volume_retained":       "Volume Retained (fraction)",
            "revenue_impact_usd_bn": "Impact (USD bn)",
        },
    )
    fig_exp.add_vline(x=0, line_color="black", line_width=1)
    fig_exp.update_traces(textposition="outside")
    fig_exp.update_layout(
        height=440,
        showlegend=False,
        margin=dict(l=0, r=80, t=10, b=0),
        xaxis_title="Revenue impact (pp of GDP)",
        yaxis_title=None,
    )
    st.plotly_chart(fig_exp, use_container_width=True)

# ── Importers ─────────────────────────────────────────────────────────────────
with col_imp:
    st.subheader("Importer Cost Shock")
    st.caption(
        f"Additional oil import bill as **pp of GDP** at **+{effective_price_change:.0f}%** "
        f"effective price rise (direct {price_shock:+d}% + "
        f"{hormuz_price_premium:.0f}% Hormuz premium).  "
        "Larger bar = greater strain on current account."
    )

    imp_sorted = importer_df.sort_values("import_shock_pp", ascending=False).copy()
    imp_sorted["bar_label"] = imp_sorted["import_shock_pp"].map(lambda v: f"+{v:.2f} pp")
    imp_sorted["oil_import_pct"] = imp_sorted["oil_import_share"] * 100

    fig_imp = px.bar(
        imp_sorted,
        x="import_shock_pp",
        y="country_label",
        orientation="h",
        color="import_shock_pp",
        color_continuous_scale="Reds",
        range_color=[0, imp_sorted["import_shock_pp"].max() * 1.2 + 0.01],
        text="bar_label",
        hover_name="country_label",
        hover_data={
            "imports_pct_gdp":      ":.1f",
            "oil_import_pct":       ":.0f",
            "import_shock_usd_bn":  ":.2f",
            "bar_label":            False,
        },
        labels={
            "import_shock_pp":     "Cost Shock (pp GDP)",
            "country_label":       "",
            "imports_pct_gdp":     "Total Imports % GDP",
            "oil_import_pct":      "Oil Share of Imports %",
            "import_shock_usd_bn": "Cost Shock (USD bn)",
        },
    )
    fig_imp.update_traces(textposition="outside")
    fig_imp.update_layout(
        height=440,
        coloraxis_showscale=False,
        margin=dict(l=0, r=80, t=10, b=0),
        xaxis_title="Import cost shock (pp of GDP)",
        yaxis_title=None,
    )
    st.plotly_chart(fig_imp, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 · Hormuz dependency table
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("Hormuz Dependency & Bypass Capacity")
st.caption(
    f"~{_HORMUZ_MBD:.0f} mb/d transits the Strait daily (2023).  "
    f"At {hormuz_disruption}% disruption: **{mbd_at_risk:.1f} mb/d** at risk.  "
    "Countries with bypass capacity retain partial export volume."
)

hormuz_display = (
    df[df["is_exporter"]]
    [[
        "country_label", "hormuz_export_pct", "bypass_pct",
        "oil_rents_pct_gdp", "volume_retained",
        "revenue_impact_pp", "revenue_impact_usd_bn",
    ]]
    .sort_values("hormuz_export_pct", ascending=False)
    .copy()
)
hormuz_display["volume_retained_pct"] = (hormuz_display["volume_retained"] * 100).round(1)
hormuz_display = hormuz_display.drop(columns=["volume_retained"])
hormuz_display.columns = [
    "Country", "Hormuz Routing (%)", "Bypass Capacity (%)",
    "Oil Rents % GDP", "Volume Retained (%)",
    "Revenue Impact (pp GDP)", "Revenue Impact (USD bn)",
]

st.dataframe(
    hormuz_display,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Hormuz Routing (%)": st.column_config.ProgressColumn(
            "Hormuz Routing (%)",
            format="%.0f%%",
            min_value=0, max_value=100,
            help="% of oil exports that must transit Strait of Hormuz",
        ),
        "Bypass Capacity (%)": st.column_config.ProgressColumn(
            "Bypass Capacity (%)",
            format="%.0f%%",
            min_value=0, max_value=100,
            help="% of exports that can bypass Hormuz via pipeline or alternative port",
        ),
        "Oil Rents % GDP":         st.column_config.NumberColumn(format="%.1f%%"),
        "Volume Retained (%)":     st.column_config.NumberColumn(format="%.1f%%"),
        "Revenue Impact (pp GDP)": st.column_config.NumberColumn(format="%+.2f"),
        "Revenue Impact (USD bn)": st.column_config.NumberColumn(format="$%+.1f bn"),
    },
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 · Net impact map
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("Net Impact Map")
st.caption(
    "Green = net revenue gain (exporters benefiting from price shock outweighs volume loss).  "
    "Red = net loss (Hormuz volume cut dominates, or importer cost shock).  "
    "Scale: percentage points of GDP."
)

# Symmetric colour scale centred on zero
abs_max = df["net_impact_pp"].abs().max()

fig_map = px.choropleth(
    df,
    locations="country_code_a3",
    color="net_impact_pp",
    hover_name="country_label",
    hover_data={
        "net_impact_pp":      ":.2f",
        "hormuz_export_pct":  ":.0f",
        "is_exporter":        True,
        "country_code_a3":    False,
    },
    color_continuous_scale="RdYlGn",
    range_color=[-abs_max, abs_max],
    labels={
        "net_impact_pp":     "Net Impact (pp GDP)",
        "hormuz_export_pct": "Hormuz Routing %",
        "is_exporter":       "Net exporter",
    },
)
fig_map.update_geos(
    showcoastlines=True, coastlinecolor="white",
    showland=True, landcolor="#f0ede6",
    showocean=True, oceancolor="#daeef5",
    showlakes=False, showrivers=False,
    showcountries=True, countrycolor="white", countrywidth=0.5,
    lataxis_range=[10, 42], lonaxis_range=[-20, 65],
    projection_type="natural earth",
)
fig_map.update_layout(
    height=440,
    margin=dict(l=0, r=0, t=10, b=0),
    coloraxis_colorbar=dict(
        title="pp of GDP",
        len=0.65, thickness=14,
    ),
)
st.plotly_chart(fig_map, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 · Combined impact table
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("Full scenario results table (all 14 countries)"):
    full_tbl = df[[
        "country_label", "is_exporter", "hormuz_dependent",
        "oil_rents_pct_gdp", "hormuz_export_pct",
        "revenue_impact_pp", "revenue_impact_usd_bn",
        "import_shock_pp", "import_shock_usd_bn",
        "net_impact_pp",
    ]].sort_values("net_impact_pp", ascending=False).copy()
    full_tbl.columns = [
        "Country", "Net Exporter", "Hormuz Exposed",
        "Oil Rents % GDP", "Hormuz Routing %",
        "Rev. Impact (pp GDP)", "Rev. Impact (USD bn)",
        "Import Shock (pp GDP)", "Import Shock (USD bn)",
        "Net Impact (pp GDP)",
    ]
    st.dataframe(
        full_tbl,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Net Exporter":          st.column_config.CheckboxColumn(),
            "Hormuz Exposed":        st.column_config.CheckboxColumn(),
            "Oil Rents % GDP":       st.column_config.NumberColumn(format="%.1f%%"),
            "Hormuz Routing %":      st.column_config.NumberColumn(format="%.0f%%"),
            "Rev. Impact (pp GDP)":  st.column_config.NumberColumn(format="%+.2f"),
            "Rev. Impact (USD bn)":  st.column_config.NumberColumn(format="$%+.1f bn"),
            "Import Shock (pp GDP)": st.column_config.NumberColumn(format="%.2f"),
            "Import Shock (USD bn)": st.column_config.NumberColumn(format="$%.2f bn"),
            "Net Impact (pp GDP)":   st.column_config.NumberColumn(format="%+.2f"),
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Methodology
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("Model assumptions & methodology"):
    st.markdown(f"""
**Exporter revenue formula (first-order approximation):**
```
volume_retained  = 1 − (hormuz_routing_% ÷ 100) × (disruption_% ÷ 100)
revenue_impact   = oil_rents_%_GDP × ( volume_retained × (1 + price_shock/100) − 1 )
```
Uses long-run average **Oil Rents % GDP** (NY.GDP.PETR.RT.ZS, 2000–2024 avg).
A country with 0% Hormuz routing retains all volume and sees a pure price gain.

**Importer cost shock formula:**
```
effective_price_change = price_shock + hormuz_disruption × 0.25
import_shock = imports_%_GDP × oil_import_share × effective_price_change ÷ 100
```
The **0.25× Hormuz premium multiplier** reflects the historical relationship between
Hormuz tension and global price spikes (2012 Iran sanctions, 2019 tanker seizures;
EIA Strait of Hormuz analysis, 2019).

**Exporter classification:** countries where fuel exports (TX.VAL.FUEL.ZS.UN) exceed
**{_EXPORTER_THRESHOLD_PCT:.0f}%** of total merchandise exports (long-run average).

**Hormuz routing fractions:**
| Country | Routing % | Notes |
|---|---|---|
| Iran | 100% | All exports via Kharg Island |
| Iraq | 90% | ~10% via Kirkuk–Ceyhan pipeline (Turkey) |
| Kuwait | 100% | No bypass capacity |
| Qatar | 100% | No bypass capacity |
| Bahrain | 85% | Minor alternative routing |
| Saudi Arabia | 65% | East-West pipeline to Yanbu Red Sea terminal |
| UAE | 75% | ADCO/IPIC pipeline to Fujairah (Gulf of Oman) |
| Oman | 0% | Mina Al Fahal is on the Gulf of Oman, east of the strait |
| Algeria, Libya | 0% | Mediterranean coast |
| Egypt, Jordan, Lebanon, Morocco | 0% | Non-Hormuz routing |

**Oil import shares (importers):**
Jordan 22% · Lebanon 20% · Morocco 25% · Egypt 13%
(World Bank WITS / IEA country profiles, 2019–2022 avg)

**Limitations:** Linear first-order model only.  Not modelled:
OPEC production quota responses, sovereign wealth fund buffers, exchange rate
pass-through, insurance cost surges, rerouting time delays, second-order
fiscal multiplier effects, or demand destruction.
""")
