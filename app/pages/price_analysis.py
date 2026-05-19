"""
Brent Crude Oil Price Analysis

Fetches annual Brent crude price data via yfinance (BZ=F), then analyses:
  - Historical price with 2008 / 2014 / 2020 crash highlights
  - Rolling 3-year annualised volatility
  - Pearson correlation between Brent price and GDP growth per country

Run standalone (from project root):
    streamlit run app/pages/price_analysis.py
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

from src.data.brent import (
    calculate_returns,
    calculate_rolling_volatility,
    fetch_brent_history,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brent Price Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).resolve().parents[2]
PANEL_PATH = _ROOT / "data" / "processed" / "world_bank_panel.csv"

_PALETTE = px.colors.qualitative.D3 + px.colors.qualitative.Plotly

_LABEL: dict[str, str] = {
    "Egypt, Arab Rep.":     "Egypt",
    "Iran, Islamic Rep.":   "Iran",
    "United Arab Emirates": "UAE",
}

def _label(name: str) -> str:
    return _LABEL.get(name, name)


@st.cache_data(show_spinner="Fetching Brent crude prices…", ttl=86_400)
def load_brent() -> tuple[pd.DataFrame, bool]:
    """Return (df[year, price_usd], live_ok) via src.data.brent.fetch_brent_history."""
    return fetch_brent_history()


@st.cache_data(show_spinner="Loading panel data…")
def load_panel() -> pd.DataFrame:
    df = pd.read_csv(PANEL_PATH)
    df["country_label"] = df["country_name"].map(_label)
    df = df.sort_values(["country_code", "year"])
    df["gdp_growth_pct"] = (
        df.groupby("country_code")["NY_GDP_MKTP_CD"]
        .pct_change()
        .mul(100)
    )
    return df


# ── Guard ──────────────────────────────────────────────────────────────────────
if not PANEL_PATH.exists():
    st.error(
        f"Panel data not found: `{PANEL_PATH}`  \n"
        "Run the pipeline first:  \n"
        "`python -m src.data.fetch_world_bank`  \n"
        "`python -m src.data.clean_world_bank`"
    )
    st.stop()

# ── Load ───────────────────────────────────────────────────────────────────────
brent_all, _api_ok = load_brent()
panel              = load_panel()

# Derived price columns (computed on the full series before year-range filtering)
brent_all = brent_all.copy()
brent_all["return_pct"] = calculate_returns(brent_all["price_usd"])
brent_all["vol_3yr"]    = calculate_rolling_volatility(brent_all["return_pct"], window=3)

# Consistent country colour map
_all_labels = sorted(panel["country_label"].unique())
_COLOUR_MAP: dict[str, str] = {
    lbl: _PALETTE[i % len(_PALETTE)] for i, lbl in enumerate(_all_labels)
}

# Major oil price crash events
_CRASHES = [
    {"year": 2008, "label": "GFC (2008)",         "color": "#d62728"},
    {"year": 2014, "label": "Supply glut (2014)",  "color": "#ff7f0e"},
    {"year": 2020, "label": "COVID-19 (2020)",     "color": "#9467bd"},
]

_yr_min = int(brent_all["year"].min())
_yr_max = int(brent_all["year"].max())

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Brent Price Analysis")
    st.markdown("---")

    st.markdown("### Year range")
    year_range: tuple[int, int] = st.slider(
        "Years",
        min_value=_yr_min,
        max_value=_yr_max,
        value=(_yr_min, _yr_max),
        step=1,
    )

    st.markdown("### Country filter")
    st.caption("Used in GDP Correlation tab")
    selected_countries: list[str] = st.multiselect(
        "Select countries",
        options=_all_labels,
        default=_all_labels,
    )
    if not selected_countries:
        selected_countries = _all_labels

    st.markdown("---")
    if _api_ok:
        st.caption("Source: Yahoo Finance · BZ=F (annual averages)")
    else:
        st.caption("Source: EIA/World Bank historical data (2000–2024)")
    st.caption("Built with Streamlit + Plotly")

# ── Filter Brent to selected year range ───────────────────────────────────────
brent = brent_all[brent_all["year"].between(year_range[0], year_range[1])].copy()

# ── Page title ─────────────────────────────────────────────────────────────────
st.title("Brent Crude Oil — Historical Price Analysis")
if not _api_ok:
    st.caption("Using historical reference data (EIA/World Bank, 2000–2024). Live data unavailable.")
else:
    st.caption("Data: Yahoo Finance · BZ=F (annual averages via yfinance)")

# ── Helper: stamp crash vertical lines on any figure ──────────────────────────
def _add_crashes(fig: go.Figure) -> None:
    for c in _CRASHES:
        if year_range[0] <= c["year"] <= year_range[1]:
            fig.add_vline(
                x=c["year"],
                line_dash="dash",
                line_color=c["color"],
                opacity=0.65,
                line_width=1.6,
                annotation_text=c["label"],
                annotation_position="top",
                annotation_font_size=10,
                annotation_font_color=c["color"],
            )


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_price, tab_vol, tab_corr = st.tabs([
    "📈  Historical Price",
    "📊  Rolling Volatility",
    "🔗  GDP Correlation",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Historical Brent Price
# ═══════════════════════════════════════════════════════════════════════════════
with tab_price:
    st.header("Historical Brent Crude Price (USD/bbl)")
    st.caption(
        "Annual average Brent crude price in nominal USD per barrel.  "
        "Dashed lines mark the three major price collapses."
    )

    if brent.empty:
        st.warning("No price data for the selected year range.")
    else:
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=brent["year"],
            y=brent["price_usd"],
            mode="lines+markers",
            name="Brent (USD/bbl)",
            line=dict(color="#1f77b4", width=2.5),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor="rgba(31,119,180,0.10)",
            hovertemplate="<b>%{x}</b><br>Brent: $%{y:.2f}/bbl<extra></extra>",
        ))
        _add_crashes(fig_price)
        fig_price.update_layout(
            height=480,
            title=f"Brent Crude Annual Price · {year_range[0]}–{year_range[1]}",
            xaxis=dict(title="Year", dtick=2),
            yaxis=dict(title="USD / barrel", rangemode="tozero"),
            hovermode="x unified",
            margin=dict(l=0, r=20, t=50, b=0),
        )
        st.plotly_chart(fig_price, use_container_width=True)

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        min_row = brent.loc[brent["price_usd"].idxmin()]
        max_row = brent.loc[brent["price_usd"].idxmax()]
        last    = brent.iloc[-1]
        prev    = brent.iloc[-2] if len(brent) > 1 else last
        yoy_pct = (last["price_usd"] - prev["price_usd"]) / prev["price_usd"] * 100
        k1.metric("Min price",  f"${min_row['price_usd']:.2f}", f"{int(min_row['year'])}")
        k2.metric("Max price",  f"${max_row['price_usd']:.2f}", f"{int(max_row['year'])}")
        k3.metric("Period avg", f"${brent['price_usd'].mean():.2f}")
        k4.metric(
            f"Latest ({int(last['year'])})",
            f"${last['price_usd']:.2f}",
            f"{yoy_pct:+.1f}% YoY",
            delta_color="normal" if yoy_pct >= 0 else "inverse",
        )

        # Peak-to-trough drawdown table
        with st.expander("Crash peak-to-trough drawdowns"):
            rows = []
            for crash in _CRASHES:
                yr      = crash["year"]
                window  = brent[brent["year"].between(yr - 3, yr + 2)]
                if window.empty:
                    continue
                pk = window.loc[window["price_usd"].idxmax()]
                tr = window.loc[window["price_usd"].idxmin()]
                rows.append({
                    "Event":                  crash["label"],
                    "Peak year":              int(pk["year"]),
                    "Peak (USD/bbl)":         pk["price_usd"],
                    "Trough year":            int(tr["year"]),
                    "Trough (USD/bbl)":       tr["price_usd"],
                    "Drawdown (%)":
                        (tr["price_usd"] - pk["price_usd"]) / pk["price_usd"] * 100,
                })
            if rows:
                st.dataframe(
                    pd.DataFrame(rows),
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Peak (USD/bbl)":   st.column_config.NumberColumn(format="$%.2f"),
                        "Trough (USD/bbl)": st.column_config.NumberColumn(format="$%.2f"),
                        "Drawdown (%)":     st.column_config.NumberColumn(format="%.1f%%"),
                    },
                )

        with st.expander("Annual price & return data"):
            tbl = brent[["year", "price_usd", "return_pct"]].copy()
            tbl.columns = ["Year", "Price (USD/bbl)", "Annual Return (%)"]
            st.dataframe(
                tbl,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Price (USD/bbl)":    st.column_config.NumberColumn(format="$%.2f"),
                    "Annual Return (%)":  st.column_config.NumberColumn(format="%+.1f%%"),
                },
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Rolling 3-Year Volatility
# ═══════════════════════════════════════════════════════════════════════════════
with tab_vol:
    st.header("Rolling 3-Year Price Volatility")
    st.caption(
        "**Volatility** = rolling 3-year standard deviation of annual price returns (%).  "
        "Captures how uncertain the oil price environment has been for budget planners."
    )

    vol_df = brent.dropna(subset=["vol_3yr"]).copy()

    if vol_df.empty:
        st.warning("Insufficient data — select at least 4 years to compute 3-year rolling volatility.")
    else:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=vol_df["year"],
            y=vol_df["vol_3yr"],
            mode="lines+markers",
            name="3yr volatility",
            line=dict(color="#d62728", width=2.5),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor="rgba(214,39,40,0.10)",
            hovertemplate="<b>%{x}</b><br>Volatility: %{y:.1f}%<extra></extra>",
        ))
        _add_crashes(fig_vol)
        fig_vol.update_layout(
            height=460,
            title=f"Rolling 3-Year Brent Volatility · {year_range[0]}–{year_range[1]}",
            xaxis=dict(title="Year", dtick=2),
            yaxis=dict(title="Std dev of annual returns (%)", rangemode="tozero"),
            hovermode="x unified",
            margin=dict(l=0, r=20, t=50, b=0),
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        v1, v2, v3, v4 = st.columns(4)
        pk_vol = vol_df.loc[vol_df["vol_3yr"].idxmax()]
        lo_vol = vol_df.loc[vol_df["vol_3yr"].idxmin()]
        v1.metric("Mean volatility",  f"{vol_df['vol_3yr'].mean():.1f}%")
        v2.metric("Peak volatility",  f"{pk_vol['vol_3yr']:.1f}%", f"{int(pk_vol['year'])}")
        v3.metric("Min volatility",   f"{lo_vol['vol_3yr']:.1f}%",  f"{int(lo_vol['year'])}")
        v4.metric("Latest 3yr vol",   f"{vol_df.iloc[-1]['vol_3yr']:.1f}%")

        with st.expander("Price & volatility data table"):
            vt = vol_df[["year", "price_usd", "return_pct", "vol_3yr"]].copy()
            vt.columns = ["Year", "Price (USD/bbl)", "Annual Return (%)", "3yr Volatility (%)"]
            st.dataframe(
                vt,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Price (USD/bbl)":     st.column_config.NumberColumn(format="$%.2f"),
                    "Annual Return (%)":   st.column_config.NumberColumn(format="%+.1f%%"),
                    "3yr Volatility (%)":  st.column_config.NumberColumn(format="%.1f%%"),
                },
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GDP Correlation
# ═══════════════════════════════════════════════════════════════════════════════
with tab_corr:
    st.header("Brent Price vs GDP Growth — Country Correlations")
    st.caption(
        "Pearson r between annual Brent crude price and year-on-year GDP growth.  "
        "Positive r → oil-export revenues amplify economic growth; "
        "negative r → import-side squeeze dominates."
    )

    panel_yr = panel[
        panel["country_label"].isin(selected_countries) &
        panel["year"].between(year_range[0], year_range[1])
    ].copy()

    merged = panel_yr.merge(brent[["year", "price_usd"]], on="year", how="inner")
    merged = merged.dropna(subset=["gdp_growth_pct", "price_usd"])

    if merged.empty:
        st.warning("No overlapping data for the current selection.")
    else:
        # ── Pearson r per country ──────────────────────────────────────────────
        corrs = (
            merged.groupby("country_label")
            .apply(lambda g: g["price_usd"].corr(g["gdp_growth_pct"]))
            .reset_index()
            .rename(columns={0: "pearson_r"})
            .sort_values("pearson_r", ascending=True)
        )

        fig_corr = px.bar(
            corrs,
            x="pearson_r",
            y="country_label",
            orientation="h",
            color="pearson_r",
            color_continuous_scale="RdYlGn",
            range_color=[-1, 1],
            text=corrs["pearson_r"].map(lambda v: f"{v:+.2f}"),
            labels={"pearson_r": "Pearson r", "country_label": ""},
            title=f"Brent Price – GDP Growth Correlation · {year_range[0]}–{year_range[1]}",
        )
        fig_corr.add_vline(x=0, line_color="black", line_width=1)
        fig_corr.update_traces(
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>r = %{x:+.3f}<extra></extra>",
        )
        fig_corr.update_layout(
            height=480,
            coloraxis_showscale=False,
            xaxis=dict(range=[-1.3, 1.3], title="Pearson r  (−1 = inverse · +1 = direct)"),
            yaxis_title=None,
            margin=dict(l=0, r=70, t=50, b=0),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # ── Scatter: Brent price vs GDP growth ────────────────────────────────
        st.subheader("Scatter — Brent Price vs GDP Growth")

        # Clip GDP growth extremes (war-rebuild outliers collapse the visible range)
        p2  = merged["gdp_growth_pct"].quantile(0.02)
        p98 = merged["gdp_growth_pct"].quantile(0.98)
        n_clip = int(((merged["gdp_growth_pct"] < p2) | (merged["gdp_growth_pct"] > p98)).sum())
        merged["gdp_growth_clipped"] = merged["gdp_growth_pct"].clip(lower=p2, upper=p98)
        if n_clip > 0:
            st.info(
                f"{n_clip} point(s) with GDP growth outside [{p2:.0f}%, {p98:.0f}%] "
                "clipped for readability. True values visible in hover tooltips.",
                icon="ℹ️",
            )

        fig_scatter = px.scatter(
            merged,
            x="price_usd",
            y="gdp_growth_clipped",
            color="country_label",
            hover_name="country_label",
            hover_data={
                "year": True,
                "gdp_growth_pct": ":.1f",
                "price_usd": ":.2f",
                "gdp_growth_clipped": False,
            },
            labels={
                "price_usd":           "Brent Price (USD/bbl)",
                "gdp_growth_clipped":  "GDP Growth (%, clipped p2–p98)",
                "country_label":       "Country",
                "year":                "Year",
                "gdp_growth_pct":      "True GDP Growth (%)",
            },
            title=f"Brent Price vs GDP Growth · {year_range[0]}–{year_range[1]}",
            color_discrete_map=_COLOUR_MAP,
        )

        # Overall OLS trendline (numpy, no statsmodels required)
        xy = merged[["price_usd", "gdp_growth_clipped"]].dropna()
        if len(xy) >= 2:
            coeffs = np.polyfit(xy["price_usd"], xy["gdp_growth_clipped"], 1)
            x_line = np.linspace(float(xy["price_usd"].min()), float(xy["price_usd"].max()), 200)
            y_line = np.polyval(coeffs, x_line)
            fig_scatter.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=f"OLS trend (slope {coeffs[0]:+.3f})",
                line=dict(color="black", width=1.5, dash="dash"),
                hoverinfo="skip",
            ))

        # Vertical markers at crash-year average Brent price
        for crash in _CRASHES:
            pts = merged[merged["year"] == crash["year"]]
            if not pts.empty:
                fig_scatter.add_vline(
                    x=float(pts["price_usd"].mean()),
                    line_dash="dot",
                    line_color=crash["color"],
                    opacity=0.45,
                    annotation_text=crash["label"],
                    annotation_position="top right",
                    annotation_font_size=9,
                    annotation_font_color=crash["color"],
                )

        fig_scatter.add_hline(y=0, line_color="lightgrey", line_width=1)
        fig_scatter.update_layout(
            height=540,
            hovermode="closest",
            legend=dict(
                orientation="v", yanchor="top", y=1,
                xanchor="left", x=1.02, font_size=11,
            ),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # ── Correlation detail table ───────────────────────────────────────────
        with st.expander("Correlation table — all selected countries"):
            disp = corrs.sort_values("pearson_r", ascending=False).copy()
            disp.columns = ["Country", "Pearson r"]
            st.dataframe(
                disp,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Pearson r": st.column_config.NumberColumn(format="%+.3f"),
                },
            )
