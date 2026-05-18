"""
Oil Supply Chain — Trade Dependency & Chokepoint Exposure

Three analytical views:
  1. Trade chains       — fuel export concentration, trade balance dynamics (2000–2024)
  2. Chokepoint exposure — Hormuz (exporters) + Suez Canal (importers) routing risk matrix
  3. Import vulnerability — imports/GDP, fossil-fuel energy dependency, composite score

Run standalone (from project root):
    streamlit run app/pages/supply_chain.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Analysis",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).resolve().parents[2]
PANEL_PATH = _ROOT / "data" / "processed" / "world_bank_panel.csv"
OCVI_PATH  = _ROOT / "outputs" / "tables" / "ocvi_scores.csv"

_PALETTE = px.colors.qualitative.D3 + px.colors.qualitative.Plotly

_LABEL: dict[str, str] = {
    "Egypt, Arab Rep.":     "Egypt",
    "Iran, Islamic Rep.":   "Iran",
    "United Arab Emirates": "UAE",
}
def _label(name: str) -> str:
    return _LABEL.get(name, name)


# ── Domain knowledge: chokepoint routing fractions ────────────────────────────
#
# OIL EXPORT routing via Strait of Hormuz (% of each country's crude exports).
# Sources: EIA (2023), IEA Oil Market Report, country pipeline data.
_HORMUZ_EXPORT_PCT: dict[str, int] = {
    "IRN": 100,  # All exports via Kharg Island — no alternative
    "IRQ": 90,   # Basrah Oil Terminal; ~10% via Kirkuk–Ceyhan pipeline (Turkey)
    "KWT": 100,  # No bypass capacity
    "QAT": 100,  # No bypass for LNG or crude
    "BHR": 85,   # Minor rerouting via Saudi Arabia
    "SAU": 65,   # East-West pipeline to Yanbu Red Sea terminal (35% bypass)
    "ARE": 75,   # ADCO/IPIC pipeline to Fujairah, Gulf of Oman (~25% bypass)
    "OMN": 0,    # Mina Al Fahal is on the Gulf of Oman — outside the strait
    "DZA": 0,    # Mediterranean coast (Arzew / Skikda)
    "LBY": 0,    # Mediterranean coast (Es Sider / Ras Lanuf)
    "EGY": 0,    # Red Sea / Mediterranean (Ain Sukhna, Sidi Kerir)
    "JOR": 0,
    "LBN": 0,
    "MAR": 0,
}

# OIL EXPORT routing via Suez Canal (% of each exporter's crude/product exports).
# Sources: EIA Suez Canal analysis (2023), UNCTAD shipping statistics.
_SUEZ_EXPORT_PCT: dict[str, int] = {
    "SAU": 15,   # Yanbu Red Sea terminal → Suez → European/Atlantic markets
    "OMN": 10,   # Gulf of Oman → Indian Ocean → Suez to Atlantic
    "QAT": 5,    # Some LNG/condensate routed to Europe via Suez
    "ARE": 5,    # Fujairah exports to European markets via Suez
    "IRQ": 0,    # Ceyhan → Mediterranean direct; Basrah via Hormuz
    "IRN": 0,    # No Red Sea access; all via Hormuz
    "KWT": 0,
    "BHR": 0,
    "DZA": 0,    # Mediterranean coast — direct to Southern Europe, no Suez transit
    "LBY": 0,    # Mediterranean coast — direct to Southern Europe
    "EGY": 0,    # Has SUMED pipeline parallel bypass
    "JOR": 0,
    "LBN": 0,
    "MAR": 0,
}

# OIL IMPORT routing via Suez Canal for net importers (% of petroleum product imports).
# Sources: UNCTAD, IEA country profiles (2019–2022 avg).
_SUEZ_IMPORT_PCT: dict[str, int] = {
    "MAR": 30,   # Atlantic coast; substantial Asian-sourced product imports
    "LBN": 25,   # Receives products from Asia/East Africa via Suez
    "JOR": 20,   # Some petroleum products from Asia arrive via Suez/Aqaba Red Sea
    "EGY": 5,    # Domestic production + SUMED bypass; low Suez import dependence
    "IRN": 0, "IRQ": 0, "KWT": 0, "QAT": 0,
    "BHR": 0, "SAU": 0, "ARE": 0, "OMN": 0,
    "DZA": 0, "LBY": 0,
}

# Oil/energy products as share of total merchandise imports (for net importers).
# Sources: World Bank WITS, IEA country profiles (2019–2022 avg).
_OIL_IMPORT_SHARE: dict[str, float] = {
    "JOR": 0.22,  # ~22% of merchandise imports are petroleum products
    "LBN": 0.20,
    "MAR": 0.25,
    "EGY": 0.13,  # Partial domestic producer; net importer since ~2010
}

_HORMUZ_MBD          = 21.0   # Strait of Hormuz daily flow, mb/d (EIA 2023)
_SUEZ_MBD            = 4.5    # Suez Canal daily oil flow, mb/d (EIA 2023)
_EXPORTER_THRESHOLD  = 20.0   # Fuel exports % of merch exports → classified as exporter


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading panel data…")
def load_panel() -> pd.DataFrame:
    df = pd.read_csv(PANEL_PATH)
    df["country_label"] = df["country_name"].map(_label)
    df = df.sort_values(["country_code", "year"])
    df["imports_pct_gdp_ts"] = df["BM_GSR_MRCH_CD"] / df["NY_GDP_MKTP_CD"] * 100
    df["exports_pct_gdp_ts"] = df["BX_GSR_TOTL_CD"] / df["NY_GDP_MKTP_CD"] * 100
    df["trade_balance_pct"]  = df["exports_pct_gdp_ts"] - df["imports_pct_gdp_ts"]
    return df


@st.cache_data(show_spinner="Loading OCVI scores…")
def load_ocvi() -> pd.DataFrame:
    df = pd.read_csv(OCVI_PATH)
    df["country_label"] = df["country_name"].map(_label)
    return df


@st.cache_data(show_spinner="Building chokepoint matrix…")
def build_chokepoint(ocvi: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Assemble per-country static chokepoint metrics."""
    avg_fuel = (
        panel.groupby("country_code_a3")["TX_VAL_FUEL_ZS_UN"]
        .mean().rename("avg_fuel_export_pct")
    )
    avg_gdp = (
        panel.groupby("country_code_a3")["NY_GDP_MKTP_CD"]
        .mean().rename("avg_gdp_usd")
    )
    cp = (
        ocvi[[
            "country_code", "country_code_a3", "country_name", "country_label",
            "oil_rents_pct_gdp", "exports_pct_gdp", "imports_pct_gdp",
        ]]
        .merge(avg_fuel, left_on="country_code_a3", right_index=True, how="left")
        .merge(avg_gdp,  left_on="country_code_a3", right_index=True, how="left")
    )

    cp["hormuz_export_pct"] = cp["country_code_a3"].map(_HORMUZ_EXPORT_PCT).fillna(0)
    cp["suez_export_pct"]   = cp["country_code_a3"].map(_SUEZ_EXPORT_PCT).fillna(0)
    cp["suez_import_pct"]   = cp["country_code_a3"].map(_SUEZ_IMPORT_PCT).fillna(0)
    cp["oil_import_share"]  = cp["country_code_a3"].map(_OIL_IMPORT_SHARE).fillna(0)
    cp["is_exporter"]       = cp["avg_fuel_export_pct"] > _EXPORTER_THRESHOLD
    cp["hormuz_bypass_pct"] = 100 - cp["hormuz_export_pct"]

    # Revenue at risk if Hormuz closed 100%
    cp["hormuz_rev_at_risk_bn"] = (
        cp["hormuz_export_pct"] / 100.0
        * cp["oil_rents_pct_gdp"] / 100.0
        * cp["avg_gdp_usd"] / 1e9
    )

    # Composite chokepoint exposure scores (0–100 scale)
    # Exporters: Hormuz weighted higher because it carries ~5× Suez flow
    cp["chokepoint_exp_score"] = (
        cp["hormuz_export_pct"] * 0.70 + cp["suez_export_pct"] * 0.30
    )
    # Importers: Suez import % × oil share of imports (normalised to 0-100)
    # Max product (Morocco 30% × 0.25 share = 7.5) normalises to 100
    cp["chokepoint_imp_score"] = (
        (cp["suez_import_pct"] * cp["oil_import_share"]) / 7.5 * 100
    ).clip(upper=100)
    cp["chokepoint_score"] = cp.apply(
        lambda r: r["chokepoint_exp_score"] if r["is_exporter"] else r["chokepoint_imp_score"],
        axis=1,
    )
    return cp


# ── Guards ─────────────────────────────────────────────────────────────────────
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
cp    = build_chokepoint(ocvi, panel)

_all_labels = sorted(panel["country_label"].unique())
_COLOUR_MAP: dict[str, str] = {
    lbl: _PALETTE[i % len(_PALETTE)] for i, lbl in enumerate(_all_labels)
}
_yr_min = int(panel["year"].min())
_yr_max = int(panel["year"].max())

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚢 Supply Chain")
    st.markdown("---")

    st.markdown("### Country filter")
    selected_countries: list[str] = st.multiselect(
        "Select countries",
        options=_all_labels,
        default=_all_labels,
    )
    if not selected_countries:
        selected_countries = _all_labels

    st.markdown("### Year range")
    year_range: tuple[int, int] = st.slider(
        "Years",
        min_value=_yr_min,
        max_value=_yr_max,
        value=(_yr_min, _yr_max),
        step=1,
    )

    st.markdown("---")
    st.caption("Sources: World Bank, EIA, IEA, UNCTAD  \nBuilt with Streamlit + Plotly")


# ── Filter helpers ─────────────────────────────────────────────────────────────
def _filter(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["country_label"].isin(selected_countries) &
        df["year"].between(year_range[0], year_range[1])
    ]

def _min_max_norm(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else pd.Series(0.0, index=s.index)


# ── Page header ────────────────────────────────────────────────────────────────
st.title("Oil Supply Chain — Trade Dependency & Chokepoint Exposure")
st.caption(
    "Structural oil trade dependency for 14 MENA economies: "
    "fuel export concentration, Strait of Hormuz and Suez Canal routing risk, "
    "and import vulnerability from World Bank panel data (2000–2024)."
)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_trade, tab_choke, tab_vuln = st.tabs([
    "🔗  Trade Dependency Chains",
    "🚢  Chokepoint Exposure",
    "📦  Import Vulnerability",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Trade Dependency Chains
# ═══════════════════════════════════════════════════════════════════════════════
with tab_trade:
    st.header("Trade Dependency Chains")
    st.caption(
        "**TX.VAL.FUEL.ZS.UN** — fuel exports as % of merchandise exports.  "
        "Near-100% values = single-commodity dependence on oil/gas; "
        "a falling trend signals diversification."
    )

    df_fuel = _filter(panel).dropna(subset=["TX_VAL_FUEL_ZS_UN"]).copy()

    if df_fuel.empty:
        st.warning("No trade data for the current selection.")
    else:
        # ── Fuel export concentration time series ──────────────────────────────
        fig_fuel = px.line(
            df_fuel,
            x="year",
            y="TX_VAL_FUEL_ZS_UN",
            color="country_label",
            markers=True,
            labels={
                "year":                "Year",
                "TX_VAL_FUEL_ZS_UN":   "Fuel Exports (% of merchandise exports)",
                "country_label":       "Country",
            },
            title=f"Fuel Export Concentration · {year_range[0]}–{year_range[1]}",
            color_discrete_map=_COLOUR_MAP,
        )
        fig_fuel.update_traces(marker_size=4, line_width=1.8)
        for yr, lbl in [(2008, "2008"), (2014, "2014"), (2020, "2020")]:
            if year_range[0] <= yr <= year_range[1]:
                fig_fuel.add_vline(
                    x=yr, line_dash="dot", line_color="grey",
                    opacity=0.50, annotation_text=lbl,
                    annotation_position="top", annotation_font_size=9,
                )
        fig_fuel.update_layout(
            height=480,
            hovermode="x unified",
            xaxis=dict(dtick=2),
            yaxis=dict(title="Fuel Exports (% of merchandise)", range=[0, 110]),
            legend=dict(
                orientation="v", yanchor="top", y=1,
                xanchor="left", x=1.02,
            ),
        )
        st.plotly_chart(fig_fuel, use_container_width=True)

        # ── Trade balance time series ──────────────────────────────────────────
        st.subheader("Trade Balance Over Time")
        st.caption(
            "Goods exports minus merchandise imports as % of GDP "
            "(BX.GSR.TOTL.CD − BM.GSR.MRCH.CD) / NY.GDP.MKTP.CD.  "
            "Positive = trade surplus; negative = deficit."
        )
        df_bal = _filter(panel).dropna(subset=["trade_balance_pct"]).copy()
        if not df_bal.empty:
            fig_bal = px.line(
                df_bal,
                x="year",
                y="trade_balance_pct",
                color="country_label",
                markers=True,
                labels={
                    "year":               "Year",
                    "trade_balance_pct":  "Trade Balance (% GDP)",
                    "country_label":      "Country",
                },
                title=f"Trade Balance (Exports − Imports, % GDP) · {year_range[0]}–{year_range[1]}",
                color_discrete_map=_COLOUR_MAP,
            )
            fig_bal.add_hline(y=0, line_color="black", line_width=1)
            fig_bal.update_traces(marker_size=4, line_width=1.8)
            for yr, lbl in [(2008, "2008"), (2014, "2014"), (2020, "2020")]:
                if year_range[0] <= yr <= year_range[1]:
                    fig_bal.add_vline(
                        x=yr, line_dash="dot", line_color="grey",
                        opacity=0.50, annotation_text=lbl,
                        annotation_position="top", annotation_font_size=9,
                    )
            fig_bal.update_layout(
                height=420,
                hovermode="x unified",
                xaxis=dict(dtick=2),
                yaxis_title="Trade Balance (% GDP)",
                legend=dict(
                    orientation="v", yanchor="top", y=1,
                    xanchor="left", x=1.02,
                ),
            )
            st.plotly_chart(fig_bal, use_container_width=True)

        # ── Long-run summary table ─────────────────────────────────────────────
        with st.expander("Long-run fuel export concentration — summary statistics"):
            avg_tbl = (
                df_fuel.groupby("country_label")["TX_VAL_FUEL_ZS_UN"]
                .agg(Mean="mean", Max="max", Min="min", Std="std")
                .round(1)
                .reset_index()
                .rename(columns={"country_label": "Country"})
                .sort_values("Mean", ascending=False)
            )
            st.dataframe(
                avg_tbl,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Mean": st.column_config.ProgressColumn(
                        "Mean %", format="%.1f%%", min_value=0, max_value=100,
                    ),
                    "Max": st.column_config.NumberColumn(format="%.1f%%"),
                    "Min": st.column_config.NumberColumn(format="%.1f%%"),
                    "Std": st.column_config.NumberColumn(
                        "Std Dev", format="%.1f",
                        help="Higher = more volatile fuel export share over time",
                    ),
                },
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Chokepoint Exposure
# ═══════════════════════════════════════════════════════════════════════════════
with tab_choke:
    st.header("Chokepoint Exposure — Hormuz & Suez")
    st.caption(
        "**Strait of Hormuz** (~21 mb/d) is the primary export chokepoint for Gulf producers.  "
        "**Suez Canal** (~4.5 mb/d) is a secondary route for Saudi Red Sea exports "
        "and an import route for Mediterranean/Atlantic importers."
    )

    cp_sel = cp[cp["country_label"].isin(selected_countries)].copy()

    # ── KPI row ────────────────────────────────────────────────────────────────
    n_hormuz   = int((cp_sel["hormuz_export_pct"] > 0).sum())
    n_suez_exp = int((cp_sel["suez_export_pct"] > 0).sum())
    n_suez_imp = int((cp_sel["suez_import_pct"] > 0).sum())
    total_rev_risk = cp_sel["hormuz_rev_at_risk_bn"].sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Hormuz-exposed exporters", str(n_hormuz),     f"{_HORMUZ_MBD:.0f} mb/d through strait")
    k2.metric("Suez-routing exporters",   str(n_suez_exp),   f"{_SUEZ_MBD:.1f} mb/d via Suez")
    k3.metric("Suez-dependent importers", str(n_suez_imp),   "oil product imports")
    k4.metric(
        "Hormuz revenue at risk",
        f"${total_rev_risk:.1f} bn",
        "at full strait closure",
        delta_color="inverse",
    )
    st.markdown("---")

    col_bubble, col_imp = st.columns([3, 2], gap="large")

    # ── Bubble scatter: Hormuz vs Suez (exporters) ────────────────────────────
    with col_bubble:
        st.subheader("Dual Chokepoint Exposure Matrix — Exporters")
        st.caption(
            "**X** = Hormuz routing %.  "
            "**Y** = Suez Canal routing %.  "
            "**Bubble size** = Oil Rents % GDP.  "
            "Top-right = dual dependency; bottom-left = well-routed."
        )
        exp_cp = cp_sel[cp_sel["is_exporter"]].copy()
        if not exp_cp.empty:
            fig_bubble = px.scatter(
                exp_cp,
                x="hormuz_export_pct",
                y="suez_export_pct",
                size="oil_rents_pct_gdp",
                size_max=55,
                color="country_label",
                text="country_label",
                hover_name="country_label",
                hover_data={
                    "hormuz_export_pct":     ":.0f",
                    "suez_export_pct":       ":.0f",
                    "hormuz_bypass_pct":     ":.0f",
                    "oil_rents_pct_gdp":     ":.1f",
                    "hormuz_rev_at_risk_bn": ":.1f",
                },
                labels={
                    "hormuz_export_pct":     "Hormuz Routing (%)",
                    "suez_export_pct":       "Suez Routing (%)",
                    "hormuz_bypass_pct":     "Hormuz Bypass (%)",
                    "oil_rents_pct_gdp":     "Oil Rents % GDP",
                    "hormuz_rev_at_risk_bn": "Revenue at Risk (USD bn)",
                },
                color_discrete_map=_COLOUR_MAP,
                title="Exporter Chokepoint Routing Matrix",
            )
            fig_bubble.update_traces(
                textposition="top center",
                textfont_size=10,
                marker=dict(opacity=0.75, line=dict(width=1, color="white")),
            )
            # Quadrant dividers at 50% Hormuz / 10% Suez
            fig_bubble.add_vline(
                x=50, line_dash="dot", line_color="grey", opacity=0.4,
                annotation_text="50% Hormuz", annotation_font_size=9,
                annotation_position="bottom",
            )
            fig_bubble.add_hline(
                y=8, line_dash="dot", line_color="grey", opacity=0.4,
                annotation_text="8% Suez", annotation_font_size=9,
                annotation_position="right",
            )
            fig_bubble.update_layout(
                height=480,
                showlegend=False,
                xaxis=dict(title="Hormuz Routing (%)", range=[-5, 115]),
                yaxis=dict(title="Suez Canal Routing (%)", range=[-1, 22]),
            )
            st.plotly_chart(fig_bubble, use_container_width=True)
        else:
            st.info("No exporters in the current country selection.")

    # ── Importer Suez dependency bar ──────────────────────────────────────────
    with col_imp:
        st.subheader("Importer Suez Dependency")
        st.caption(
            "% of petroleum product imports that arrive via the Suez Canal.  "
            "A closure or tariff shock directly raises landed cost for these countries."
        )
        imp_cp = cp_sel[~cp_sel["is_exporter"]].copy()
        if not imp_cp.empty:
            imp_sorted = imp_cp.sort_values("suez_import_pct", ascending=True)
            fig_imp_bar = px.bar(
                imp_sorted,
                x="suez_import_pct",
                y="country_label",
                orientation="h",
                color="suez_import_pct",
                color_continuous_scale="Blues",
                range_color=[0, 35],
                text=imp_sorted["suez_import_pct"].map(lambda v: f"{v}%"),
                labels={
                    "suez_import_pct": "Suez Import Routing (%)",
                    "country_label":   "",
                },
                title="Suez Dependency — Net Importers",
            )
            fig_imp_bar.update_traces(textposition="outside")
            fig_imp_bar.update_layout(
                height=260,
                coloraxis_showscale=False,
                xaxis=dict(range=[0, 42], title="% of oil imports via Suez"),
                margin=dict(l=0, r=60, t=40, b=0),
            )
            st.plotly_chart(fig_imp_bar, use_container_width=True)

            # Oil import share context
            imp_detail = imp_cp[["country_label", "suez_import_pct", "oil_import_share", "imports_pct_gdp"]].copy()
            imp_detail["oil_import_share_pct"] = imp_detail["oil_import_share"] * 100
            imp_detail["suez_oil_import_cost"] = (
                imp_detail["suez_import_pct"] / 100 * imp_detail["oil_import_share_pct"]
            )
            imp_detail.columns = [
                "Country", "Suez Routing (%)", "Oil / Total Imports (%)",
                "Imports % GDP", "Suez Oil Cost Exposure",
            ]
            st.dataframe(
                imp_detail,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Suez Routing (%)":        st.column_config.NumberColumn(format="%.0f%%"),
                    "Oil / Total Imports (%)": st.column_config.NumberColumn(format="%.1f%%"),
                    "Imports % GDP":           st.column_config.NumberColumn(format="%.1f%%"),
                    "Suez Oil Cost Exposure":  st.column_config.NumberColumn(
                        format="%.2f",
                        help="Suez routing % × oil import share — combined exposure factor",
                    ),
                },
            )
        else:
            st.info("No importers in the current country selection.")

    # ── Hormuz routing & bypass table ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("Hormuz Routing & Bypass Capacity — All Exporters")

    exp_tbl = (
        cp_sel[cp_sel["is_exporter"]]
        .sort_values("hormuz_export_pct", ascending=False)
        [[
            "country_label", "hormuz_export_pct", "hormuz_bypass_pct",
            "suez_export_pct", "oil_rents_pct_gdp", "hormuz_rev_at_risk_bn",
        ]]
        .copy()
    )
    exp_tbl.columns = [
        "Country", "Hormuz Routing (%)", "Hormuz Bypass (%)",
        "Suez Routing (%)", "Oil Rents % GDP", "Rev at Risk (USD bn)",
    ]
    st.dataframe(
        exp_tbl,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Hormuz Routing (%)": st.column_config.ProgressColumn(
                "Hormuz Routing (%)", format="%.0f%%", min_value=0, max_value=100,
                help="% of oil exports that must transit Strait of Hormuz",
            ),
            "Hormuz Bypass (%)": st.column_config.ProgressColumn(
                "Hormuz Bypass (%)", format="%.0f%%", min_value=0, max_value=100,
                help="% of exports that can bypass Hormuz via pipeline or alternative port",
            ),
            "Suez Routing (%)":     st.column_config.NumberColumn(format="%.0f%%"),
            "Oil Rents % GDP":      st.column_config.NumberColumn(format="%.1f%%"),
            "Rev at Risk (USD bn)": st.column_config.NumberColumn(
                format="$%.1f bn",
                help="Annual oil revenue at risk if Hormuz closes completely",
            ),
        },
    )

    with st.expander("Data sources & routing notes"):
        st.markdown(f"""
**Strait of Hormuz** baseline: ~**{_HORMUZ_MBD:.0f} mb/d** (~21% of global oil trade)

| Country | Hormuz % | Route notes |
|---|---|---|
| Iran | 100% | All exports via Kharg Island — no alternative |
| Iraq | 90% | ~10% via Kirkuk–Ceyhan pipeline to Turkey |
| Kuwait, Qatar | 100% | No bypass capacity |
| Bahrain | 85% | Minor alternative routing via Saudi Arabia |
| Saudi Arabia | 65% | East-West pipeline to Yanbu Red Sea terminal (35% bypass) |
| UAE | 75% | ADCO/IPIC pipeline to Fujairah, Gulf of Oman (25% bypass) |
| Oman | 0% | Mina Al Fahal on Gulf of Oman — outside the strait |
| Algeria, Libya | 0% | Mediterranean coast — direct to Southern Europe |
| Egypt, Jordan, Lebanon, Morocco | 0% | Non-Hormuz routing |

**Suez Canal** baseline: ~**{_SUEZ_MBD:.1f} mb/d** oil (EIA 2023, UNCTAD)

Export routing: Saudi Arabia 15% (Yanbu → Suez) · Oman 10% · Qatar/UAE 5%
Import routing: Morocco 30% · Lebanon 25% · Jordan 20% (of petroleum product imports)

*Sources: EIA Strait of Hormuz & Suez Canal country analyses (2023),
IEA Oil Market Report, World Bank WITS, UNCTAD shipping statistics.*
""")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Import Vulnerability
# ═══════════════════════════════════════════════════════════════════════════════
with tab_vuln:
    st.header("Import Vulnerability")
    st.caption(
        "Three lenses: merchandise import dependency (imports/GDP), "
        "fossil-fuel energy dependency (% of energy from fossil fuels), "
        "and a composite import vulnerability score combining both with chokepoint routing."
    )

    df_vuln = _filter(panel).copy()

    # ── Imports/GDP time series ────────────────────────────────────────────────
    st.subheader("Merchandise Imports / GDP")
    df_imp = df_vuln.dropna(subset=["imports_pct_gdp_ts"])
    if not df_imp.empty:
        fig_imp = px.line(
            df_imp,
            x="year",
            y="imports_pct_gdp_ts",
            color="country_label",
            markers=True,
            labels={
                "year":               "Year",
                "imports_pct_gdp_ts": "Imports (% of GDP)",
                "country_label":      "Country",
            },
            title=f"Merchandise Imports / GDP · {year_range[0]}–{year_range[1]}",
            color_discrete_map=_COLOUR_MAP,
        )
        fig_imp.update_traces(marker_size=4, line_width=1.8)
        for yr, lbl in [(2008, "2008"), (2014, "2014"), (2020, "2020")]:
            if year_range[0] <= yr <= year_range[1]:
                fig_imp.add_vline(
                    x=yr, line_dash="dot", line_color="grey",
                    opacity=0.50, annotation_text=lbl,
                    annotation_position="top", annotation_font_size=9,
                )
        fig_imp.update_layout(
            height=420,
            hovermode="x unified",
            xaxis=dict(dtick=2),
            yaxis_title="Imports (% of GDP)",
            legend=dict(
                orientation="v", yanchor="top", y=1,
                xanchor="left", x=1.02,
            ),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # ── Fossil fuel energy dependency ─────────────────────────────────────────
    st.subheader("Fossil Fuel Energy Dependency")
    st.caption(
        "**EG.USE.COMM.FO.ZS** — fossil fuels as % of total energy consumption.  "
        "Near-100% means the entire energy matrix depends on fossil fuels; "
        "any oil import disruption propagates across the whole economy."
    )
    df_ff = df_vuln.dropna(subset=["EG_USE_COMM_FO_ZS"])
    if not df_ff.empty:
        fig_ff = px.line(
            df_ff,
            x="year",
            y="EG_USE_COMM_FO_ZS",
            color="country_label",
            markers=True,
            labels={
                "year":               "Year",
                "EG_USE_COMM_FO_ZS":  "Fossil Fuel Energy Use (%)",
                "country_label":      "Country",
            },
            title=f"Fossil Fuel Share of Energy Consumption · {year_range[0]}–{year_range[1]}",
            color_discrete_map=_COLOUR_MAP,
        )
        fig_ff.update_traces(marker_size=4, line_width=1.8)
        fig_ff.update_layout(
            height=400,
            hovermode="x unified",
            xaxis=dict(dtick=2),
            yaxis=dict(title="Fossil Fuel % of Energy", range=[0, 105]),
            legend=dict(
                orientation="v", yanchor="top", y=1,
                xanchor="left", x=1.02,
            ),
        )
        st.plotly_chart(fig_ff, use_container_width=True)

    # ── Composite import vulnerability score ──────────────────────────────────
    st.subheader("Composite Import Vulnerability Score")
    st.caption(
        "Score = 0.35 × norm(Imports/GDP)  +  0.30 × norm(Fossil Fuel %)  "
        "+  0.25 × norm(Chokepoint Score)  +  0.10 × norm(Oil Import Share).  "
        "All components min-max normalised across selected countries."
    )

    cp_snap = cp[cp["country_label"].isin(selected_countries)].copy()

    avg_imp_gdp = (
        panel[panel["country_label"].isin(selected_countries)]
        .groupby("country_label")["imports_pct_gdp_ts"]
        .mean()
        .reset_index(name="avg_imports_pct_gdp")
    )
    avg_ff_pct = (
        panel[panel["country_label"].isin(selected_countries)]
        .groupby("country_label")["EG_USE_COMM_FO_ZS"]
        .mean()
        .reset_index(name="avg_fossil_fuel_pct")
    )

    snap = (
        cp_snap
        .merge(avg_imp_gdp, on="country_label", how="left")
        .merge(avg_ff_pct,  on="country_label", how="left")
    )

    snap["norm_imports_gdp"]   = _min_max_norm(snap["avg_imports_pct_gdp"].fillna(0))
    snap["norm_fossil_fuel"]   = _min_max_norm(snap["avg_fossil_fuel_pct"].fillna(0))
    snap["norm_chokepoint"]    = _min_max_norm(snap["chokepoint_score"])
    snap["norm_oil_imp_share"] = _min_max_norm(snap["oil_import_share"])

    snap["import_vuln_score"] = (
        0.35 * snap["norm_imports_gdp"]
        + 0.30 * snap["norm_fossil_fuel"]
        + 0.25 * snap["norm_chokepoint"]
        + 0.10 * snap["norm_oil_imp_share"]
    )

    snap_sorted = snap.sort_values("import_vuln_score", ascending=True)

    fig_vuln = px.bar(
        snap_sorted,
        x="import_vuln_score",
        y="country_label",
        orientation="h",
        color="import_vuln_score",
        color_continuous_scale="YlOrRd",
        range_color=[0, 1],
        text=snap_sorted["import_vuln_score"].map(lambda v: f"{v:.2f}"),
        labels={
            "import_vuln_score": "Import Vulnerability Score",
            "country_label":     "",
        },
        title="Import Vulnerability Score (0 = least exposed · 1 = most exposed)",
    )
    fig_vuln.update_traces(textposition="outside")
    fig_vuln.update_layout(
        height=500,
        coloraxis_showscale=False,
        xaxis=dict(range=[0, 1.15], title="Score (composite, 0–1)"),
        margin=dict(l=0, r=60, t=50, b=0),
    )
    st.plotly_chart(fig_vuln, use_container_width=True)

    # ── Component breakdown table ──────────────────────────────────────────────
    with st.expander("Component breakdown — all selected countries"):
        snap_disp = snap_sorted[[
            "country_label", "is_exporter",
            "avg_imports_pct_gdp", "avg_fossil_fuel_pct",
            "chokepoint_score", "oil_import_share",
            "import_vuln_score",
        ]].copy()
        snap_disp["oil_import_share"] = snap_disp["oil_import_share"] * 100
        snap_disp.columns = [
            "Country", "Net Exporter",
            "Imports / GDP (%)", "Fossil Fuel Energy (%)",
            "Chokepoint Score (0–100)", "Oil Import Share (%)",
            "Vuln Score",
        ]
        st.dataframe(
            snap_disp.sort_values("Vuln Score", ascending=False),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Net Exporter":              st.column_config.CheckboxColumn(),
                "Imports / GDP (%)":         st.column_config.NumberColumn(format="%.1f%%"),
                "Fossil Fuel Energy (%)":    st.column_config.NumberColumn(format="%.1f%%"),
                "Chokepoint Score (0–100)":  st.column_config.NumberColumn(format="%.1f"),
                "Oil Import Share (%)":      st.column_config.NumberColumn(format="%.1f%%"),
                "Vuln Score": st.column_config.ProgressColumn(
                    "Vuln Score", format="%.2f", min_value=0, max_value=1,
                ),
            },
        )

    with st.expander("Methodology — composite import vulnerability score"):
        st.markdown("""
**Formula (all components min-max normalised to [0,1] across selected countries):**
```
import_vuln_score =
    0.35 × norm(Imports / GDP)            [BM.GSR.MRCH.CD / NY.GDP.MKTP.CD]
  + 0.30 × norm(Fossil Fuel Energy %)     [EG.USE.COMM.FO.ZS]
  + 0.25 × norm(Chokepoint Score)         [Hormuz/Suez routing-weighted]
  + 0.10 × norm(Oil Import Share)         [IEA/WITS 2019–2022 avg]
```

**Chokepoint Score** for exporters = 0.70 × Hormuz routing % + 0.30 × Suez routing %
**Chokepoint Score** for importers = Suez import routing % × oil import share × scale factor

Weights reflect that import/GDP (35%) and fossil fuel dependency (30%) are
structural drivers, while chokepoint routing (25%) amplifies shock transmission
and oil import share (10%) is a point-in-time composition factor.

*Data sources: World Bank Open Data, EIA Strait of Hormuz & Suez Canal analyses (2023),
IEA country profiles, UNCTAD shipping statistics.*
""")
