"""
Sovereign Wealth Fund & FX Reserve Runway Monitor

Answers: "For governments currently under fiscal stress, how many months
can they sustain spending from liquid reserves?"

Reserve runway = liquid buffer (accessible FX reserves + deployable SWF)
                 divided by estimated monthly fiscal burn rate.

Runway is only shown for countries currently in fiscal stress (Red or Amber
from the Addition 1 breakeven monitor).  Countries above breakeven (Green)
or net importers (Gray) show Gray — their reserves are not under pressure
at the current price.

Page sections
-------------
  KPI cards    — shortest runway, # under 12 months, median for stressed, # gray
  Runway chart — horizontal bar chart ranked by months with status colours
  Full table   — all 14 countries with fiscal gap + runway fields merged
  Methodology  — formulas, buffer definitions, caveats

Run standalone (from project root):
    streamlit run app/pages/reserve_runway.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# ── Make project root importable so we can use src.model.* ────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app.export import make_csv_download_button
from src.model.fiscal_stress import (
    build_stress_table,
    fetch_brent_live,
    fetch_brent_ytd,
    load_breakeven,
)
from src.model.reserve_runway import (
    _AMBER_THRESHOLD,
    _CRITICAL_THRESHOLD,
    _RED_THRESHOLD,
    build_runway_table,
    load_reserves,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Reserve Runway Monitor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT          = Path(__file__).resolve().parents[2]
_BREAKEVEN_CSV = _ROOT / "data" / "reference" / "fiscal_breakeven.csv"
_RESERVES_CSV  = _ROOT / "data" / "reference" / "swf_reserves.csv"

# ── Visual constants ───────────────────────────────────────────────────────────
# Five-level runway traffic-light palette — superset of Addition 1's four levels.
_RUNWAY_COLOUR: dict[str, str] = {
    "Critical": "#7B0000",   # very dark red
    "Red":      "#d62728",   # matplotlib red
    "Amber":    "#ff7f0e",   # matplotlib orange
    "Green":    "#2ca02c",   # matplotlib green
    "Gray":     "#aaaaaa",   # neutral grey (not applicable)
}

_RUNWAY_ICON: dict[str, str] = {
    "Critical": "⛔",
    "Red":      "🔴",
    "Amber":    "🟡",
    "Green":    "🟢",
    "Gray":     "⚫",
}

# Fiscal stress icon — reused from Addition 1 for the merged table column.
_STRESS_ICON: dict[str, str] = {
    "Red":   "🔴",
    "Amber": "🟡",
    "Green": "🟢",
    "Gray":  "⚫",
}


# ── Cached data loading ────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading breakeven reference data…")
def _load_breakeven() -> pd.DataFrame:
    """Load static fiscal breakeven CSV — no TTL."""
    return load_breakeven(_BREAKEVEN_CSV)


@st.cache_data(show_spinner="Loading reserve reference data…")
def _load_reserves() -> pd.DataFrame:
    """Load static SWF/reserves CSV — no TTL."""
    return load_reserves(_RESERVES_CSV)


@st.cache_data(show_spinner="Fetching live Brent price…", ttl=3600)
def _load_brent_live() -> float:
    """Fetch latest Brent close; cached for 1 hour."""
    return fetch_brent_live()


@st.cache_data(show_spinner="Loading YTD Brent history…", ttl=3600)
def _load_brent_ytd() -> pd.DataFrame:
    """Fetch year-to-date daily Brent closes; cached for 1 hour."""
    return fetch_brent_ytd()


# ── Guards — fail early if required reference files are missing ────────────────
for _csv in (_BREAKEVEN_CSV, _RESERVES_CSV):
    if not _csv.exists():
        st.error(
            f"Required file not found: `{_csv.relative_to(_ROOT)}`  \n"
            "Ensure both reference CSVs are present under `data/reference/`."
        )
        st.stop()

# ── Load all data ──────────────────────────────────────────────────────────────
breakeven_df = _load_breakeven()
reserves_df  = _load_reserves()
brent_live   = _load_brent_live()
ytd_prices   = _load_brent_ytd()

# Build fiscal stress table (Addition 1 logic — not duplicated here).
stress_df  = build_stress_table(breakeven_df, brent_live, ytd_prices)

# Build runway table (merges reserves + stress; computes runway months + status).
runway_df  = build_runway_table(reserves_df, stress_df)

_brent_ok  = not math.isnan(brent_live)

# Convenience subsets.
_active    = runway_df[runway_df["runway_status"] != "Gray"]   # stressed countries with runway
_gray_rows = runway_df[runway_df["runway_status"] == "Gray"]


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Reserve Runway")
    st.markdown("---")

    # Toggle: show Gray (non-stressed / N/A) rows in table and chart
    show_gray: bool = st.checkbox(
        "Show non-stressed / N/A countries (Gray)",
        value=False,
        help="Gray = currently above fiscal breakeven (no depletion risk at this price) "
             "or net importer (concept not applicable).",
    )

    # Country filter
    all_labels = sorted(runway_df["country_label"].tolist())
    selected: list[str] = st.multiselect(
        "Filter countries",
        options=all_labels,
        default=all_labels,
    )
    if not selected:
        selected = all_labels

    st.markdown("---")
    st.caption(
        "**Sources:** SWF annual reports, IMF Article IV Consultations, "
        "central bank statistical bulletins, 2023.  \n"
        "All buffer and burn figures are **preliminary estimates**.  \n"
        "Live Brent: Yahoo Finance (BZ=F).  \n"
        "Built with Streamlit + Plotly."
    )


# ── Apply filters ──────────────────────────────────────────────────────────────
display_df = runway_df[runway_df["country_label"].isin(selected)].copy()
if not show_gray:
    display_df = display_df[display_df["runway_status"] != "Gray"]


# ── Page header ────────────────────────────────────────────────────────────────
st.title("Sovereign Wealth Fund & FX Reserve Runway")
st.caption(
    "Reserve runway = accessible liquid buffer ÷ estimated monthly burn rate.  "
    "Runway is classified only for countries **currently under fiscal stress** "
    f"(Red or Amber in the Fiscal Breakeven monitor).  "
    "Countries above their breakeven (Green) or net importers (Gray) are shown "
    "as Gray — their reserves are not under pressure at the current Brent price.  "
    "All buffer and burn estimates are **preliminary** (see Methodology)."
)

if not _brent_ok:
    st.warning(
        "Could not fetch live Brent price (Yahoo Finance BZ=F unavailable).  "
        "Fiscal stress classifications are unavailable — all runway statuses "
        "may show as Gray."
    )

st.markdown("---")


# ── KPI cards ──────────────────────────────────────────────────────────────────
# Four headline metrics for the most urgent question: "who's running low?"

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

# KPI 1 — shortest runway among actively stressed countries
with kpi1:
    if _brent_ok and not _active.empty:
        shortest_idx     = _active["reserve_runway_months"].idxmin()
        shortest_months  = float(_active.loc[shortest_idx, "reserve_runway_months"])
        shortest_country = str(_active.loc[shortest_idx, "country_label"])
        shortest_status  = str(_active.loc[shortest_idx, "runway_status"])
        icon = _RUNWAY_ICON.get(shortest_status, "")
        st.metric(
            label="Shortest Runway",
            value=f"{icon} {shortest_months:.0f} mo",
            delta=shortest_country,
            delta_color="inverse",
            help="Country with the fewest months of liquid-buffer runway "
                 "among governments currently in fiscal stress.",
        )
    else:
        st.metric("Shortest Runway", "N/A")

# KPI 2 — count of countries with fewer than 12 months runway (Critical + Red)
with kpi2:
    under_12 = int(
        runway_df["runway_status"].isin(["Critical", "Red"]).sum()
    )
    if _brent_ok:
        st.metric(
            label=f"Under {_RED_THRESHOLD:.0f} Months",
            value=str(under_12),
            delta=f"Critical + Red",
            delta_color="inverse" if under_12 > 0 else "normal",
            help=f"Countries with fewer than {_RED_THRESHOLD:.0f} months of "
                 "reserve runway while under fiscal stress.",
        )
    else:
        st.metric(f"Under {_RED_THRESHOLD:.0f} Months", "N/A")

# KPI 3 — median runway for stressed exporters (non-Gray active countries)
with kpi3:
    if _brent_ok and not _active.empty:
        median_months = _active["reserve_runway_months"].median(skipna=True)
        if math.isnan(median_months):
            st.metric("Median Stressed Runway", "N/A")
        else:
            st.metric(
                label="Median Stressed Runway",
                value=f"{median_months:.0f} mo",
                help="Median reserve runway across all countries currently "
                     "under fiscal stress (Red or Amber fiscal status).",
            )
    else:
        st.metric("Median Stressed Runway", "N/A")

# KPI 4 — count of Gray (non-stressed or N/A) countries
with kpi4:
    n_gray_total = int((runway_df["runway_status"] == "Gray").sum())
    st.metric(
        label="Not Under Stress / N/A",
        value=str(n_gray_total),
        help="Countries currently above their fiscal breakeven "
             "(no reserve depletion risk at this Brent price) "
             "plus net importers where the concept is not applicable.",
    )

st.markdown("---")


# ── Runway bar chart ───────────────────────────────────────────────────────────
st.subheader("Reserve Runway by Country")
st.caption(
    "**Only countries under current fiscal stress are shown** (Gray excluded by default).  "
    "Bars show months of liquid-buffer runway at the estimated burn rate.  "
    "Toggle 'Show non-stressed / N/A countries' in the sidebar to include Gray rows."
)

# Chart data: filtered display_df, only non-Gray rows, sorted worst-first.
chart_df = display_df[display_df["runway_status"] != "Gray"].dropna(
    subset=["reserve_runway_months"]
)

if chart_df.empty:
    st.info(
        "No stressed-country runway data available for the current selection.  "
        "All selected countries may be above their fiscal breakeven at this price."
    )
else:
    chart_df = chart_df.sort_values("reserve_runway_months", ascending=True)
    bar_colours = [_RUNWAY_COLOUR[s] for s in chart_df["runway_status"]]

    fig_run = go.Figure()
    fig_run.add_trace(
        go.Bar(
            x=chart_df["reserve_runway_months"],
            y=chart_df["country_label"],
            orientation="h",
            marker_color=bar_colours,
            text=chart_df["reserve_runway_months"].map(lambda v: f"{v:.0f} mo"),
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Runway: %{x:.1f} months<br>"
                "<extra></extra>"
            ),
        )
    )

    # Reference lines for each status boundary
    max_shown = chart_df["reserve_runway_months"].max()
    x_limit   = max(max_shown * 1.15, _AMBER_THRESHOLD * 1.2)

    for threshold, label, colour in [
        (_CRITICAL_THRESHOLD, f"{_CRITICAL_THRESHOLD:.0f} mo — Critical", "#7B0000"),
        (_RED_THRESHOLD,      f"{_RED_THRESHOLD:.0f} mo — Red boundary",  "#d62728"),
        (_AMBER_THRESHOLD,    f"{_AMBER_THRESHOLD:.0f} mo — Amber/Green", "#ff7f0e"),
    ]:
        if threshold <= x_limit:
            fig_run.add_vline(
                x=threshold,
                line_dash="dot",
                line_color=colour,
                opacity=0.55,
                annotation_text=label,
                annotation_position="top",
                annotation_font_size=9,
                annotation_font_color=colour,
            )

    fig_run.update_layout(
        height=max(300, len(chart_df) * 42),
        xaxis=dict(
            title="Reserve Runway (months at estimated burn rate)",
            range=[0, x_limit],
        ),
        yaxis=dict(title=""),
        margin=dict(l=0, r=90, t=30, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig_run, use_container_width=True)

    # Colour legend
    legend_cols = st.columns(5)
    for col, (status, colour, label) in zip(
        legend_cols,
        [
            ("Critical", "#7B0000", f"Critical  < {_CRITICAL_THRESHOLD:.0f} mo"),
            ("Red",      "#d62728", f"Red  {_CRITICAL_THRESHOLD:.0f}–{_RED_THRESHOLD:.0f} mo"),
            ("Amber",    "#ff7f0e", f"Amber  {_RED_THRESHOLD:.0f}–{_AMBER_THRESHOLD:.0f} mo"),
            ("Green",    "#2ca02c", f"Green  > {_AMBER_THRESHOLD:.0f} mo"),
            ("Gray",     "#aaaaaa", "Gray  not applicable"),
        ],
    ):
        col.markdown(
            f'<span style="color:{colour}; font-weight:bold;">■</span> {label}',
            unsafe_allow_html=True,
        )

st.markdown("---")


# ── Full merged table ──────────────────────────────────────────────────────────
st.subheader("Full Country Table — Fiscal Gap & Reserve Runway")
st.caption(
    "Merges the live fiscal breakeven gap (from Addition 1) with reserve "
    "buffer and runway data.  All 14 MENA economies shown when Gray filter is on."
)

tbl = display_df.copy()

# Add icon columns for both runway and fiscal stress status.
tbl["Runway"]         = tbl["runway_status"].map(_RUNWAY_ICON)
tbl["Fiscal Status"]  = tbl["stress_status"].map(_STRESS_ICON)

# Format runway months: number or "N/A" for Gray rows.
def _fmt_months(row: pd.Series) -> str:
    if row["runway_status"] == "Gray":
        return "N/A"
    v = row["reserve_runway_months"]
    return "N/A" if (pd.isna(v) or math.isnan(v)) else f"{v:.0f}"

# Format price gap with sign or N/A.
def _fmt_gap(row: pd.Series) -> str:
    if row["stress_status"] == "Gray":
        return "N/A"
    v = row.get("price_gap_usd", float("nan"))
    if pd.isna(v) or math.isnan(float(v)):
        return "N/A"
    sign = "+" if float(v) >= 0 else ""
    return f"{sign}${float(v):.0f}"

tbl["Runway (mo)"]   = tbl.apply(_fmt_months, axis=1)
tbl["Price Gap"]     = tbl.apply(_fmt_gap, axis=1)

# Columns to display and their renamed headers.
_SHOW_COLS = {
    "Runway":                      "Runway",
    "country_label":               "Country",
    "country_type":                "Type",
    "Fiscal Status":               "Fiscal Status",
    "Price Gap":                   "Price Gap ($/bbl)",
    "liquid_buffer_usd_bn":        "Liquid Buffer ($bn)",
    "estimated_monthly_burn_usd_bn": "Monthly Burn ($bn)",
    "Runway (mo)":                 "Runway (months)",
    "confidence":                  "Confidence",
}

disp_tbl = (
    tbl[[c for c in _SHOW_COLS if c in tbl.columns]]
    .rename(columns=_SHOW_COLS)
)

st.dataframe(
    disp_tbl,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Runway": st.column_config.TextColumn(
            "Runway",
            width="small",
            help="Reserve runway status: ⛔ Critical / 🔴 Red / 🟡 Amber / 🟢 Green / ⚫ Gray",
        ),
        "Country": st.column_config.TextColumn("Country", width="medium"),
        "Type": st.column_config.TextColumn(
            "Type",
            width="small",
            help="exporter | mixed_importer | net_importer",
        ),
        "Fiscal Status": st.column_config.TextColumn(
            "Fiscal",
            width="small",
            help="Fiscal breakeven stress status from Addition 1.",
        ),
        "Price Gap ($/bbl)": st.column_config.TextColumn(
            "Price Gap",
            help="Live Brent minus fiscal breakeven ($/bbl).  Negative = below breakeven.",
        ),
        "Liquid Buffer ($bn)": st.column_config.NumberColumn(
            "Buffer ($bn)",
            format="$%.0f bn",
            help="Accessible FX reserves + deployable SWF assets (USD bn).  "
                 "Excludes constitutionally ring-fenced or illiquid SWF tranches.",
        ),
        "Monthly Burn ($bn)": st.column_config.NumberColumn(
            "Burn ($bn/mo)",
            format="$%.1f bn",
            help="Estimated monthly reserve drawdown when oil is below fiscal breakeven.",
        ),
        "Runway (months)": st.column_config.TextColumn(
            "Runway",
            help="Months of liquid buffer remaining at the estimated burn rate.  "
                 "N/A for non-stressed or net-importer countries.",
        ),
        "Confidence": st.column_config.TextColumn(
            "Confidence",
            help="Confidence in the buffer and burn estimates: high / medium / low / na.",
        ),
    },
)
make_csv_download_button(disp_tbl, "reserve_runway_table.csv", "Download table as CSV")

# Detail expander: full reference data including SWF + FX breakdown.
with st.expander("Full reference data — SWF and FX reserve breakdown"):
    detail_cols = {
        "country_label":               "Country",
        "swf_assets_usd_bn":           "SWF Assets ($bn)",
        "fx_reserves_usd_bn":          "FX Reserves ($bn)",
        "liquid_buffer_usd_bn":        "Liquid Buffer ($bn)",
        "estimated_monthly_burn_usd_bn": "Monthly Burn ($bn)",
        "burn_rate_method":            "Burn Rate Method",
        "source_swf":                  "Source: SWF",
        "source_fx":                   "Source: FX",
        "source_year":                 "Year",
        "confidence":                  "Confidence",
    }
    detail_df = (
        runway_df[[c for c in detail_cols if c in runway_df.columns]]
        .rename(columns=detail_cols)
    )
    st.dataframe(
        detail_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "SWF Assets ($bn)":    st.column_config.NumberColumn(format="$%.0f bn"),
            "FX Reserves ($bn)":   st.column_config.NumberColumn(format="$%.0f bn"),
            "Liquid Buffer ($bn)": st.column_config.NumberColumn(format="$%.0f bn"),
            "Monthly Burn ($bn)":  st.column_config.NumberColumn(format="$%.1f bn"),
        },
    )
    make_csv_download_button(detail_df, "reserve_runway_detail.csv", "Download detail as CSV")

st.markdown("---")


# ── Methodology expander ───────────────────────────────────────────────────────
with st.expander("Methodology — formulas, buffer definitions, and caveats"):
    st.markdown(f"""
### Reserve Runway Formula

```
reserve_runway_months = liquid_buffer_usd_bn / estimated_monthly_burn_usd_bn
```

### Runway Status Thresholds

| Status | Condition | Interpretation |
|--------|-----------|---------------|
| ⛔ **Critical** | Runway < {_CRITICAL_THRESHOLD:.0f} months | Imminent depletion risk; government likely needs emergency financing or forced austerity |
| 🔴 **Red**      | {_CRITICAL_THRESHOLD:.0f} ≤ Runway < {_RED_THRESHOLD:.0f} months | Very short runway; severe pressure without price recovery or external support |
| 🟡 **Amber**    | {_RED_THRESHOLD:.0f} ≤ Runway < {_AMBER_THRESHOLD:.0f} months | Limited buffer — 1 to 3 years at current burn; watchlist territory |
| 🟢 **Green**    | Runway ≥ {_AMBER_THRESHOLD:.0f} months | Comfortable buffer at current burn rate; no near-term depletion risk |
| ⚫ **Gray**     | Fiscal stress Green or Gray | Not under stress at this Brent price; runway not the pressing question |

### Liquid Buffer Definition

The **liquid buffer** is the subset of a government's financial assets that
can realistically be deployed on a 0–12 month horizon to cover a fiscal
deficit.  It is **not** the total SWF AUM.

**Included:**
- Central bank FX reserves (most liquid)
- Government Revenue Stabilisation / General Reserve Fund allocations
- Accessible government deposits at commercial banks

**Excluded:**
- Kuwait RFFG: constitutionally protected; requires parliament super-majority to touch
- ADIA, Mubadala, ADQ: long-term equity / infrastructure — typically 7–20 year horizons
- PIF (Saudi): long-term project equity; not a fiscal buffer
- QIA majority tranche: long-term global equity
- Iran NDFI + CBI: largely frozen or sanctioned; excluded from accessible buffer
- GCC Development Fund grants/loans (Bahrain): off-balance-sheet, not guaranteed

### Monthly Burn Rate Estimation

The **estimated monthly burn** represents the rate at which the liquid buffer
is drawn down when oil trades below the fiscal breakeven.  It is derived from
IMF Article IV Consultations and Regional Economic Outlook estimates of the
fiscal deficit at a "stress scenario" oil price, annualised and divided by 12.

Burn rates are **static estimates** — they do not dynamically adjust with the
live Brent price.  They reflect a representative fiscal-stress scenario rather
than the precise current-month drawdown.

### Important Caveats

- All buffer and burn figures are **preliminary estimates** from public sources.
  Governments do not publish real-time reserve drawdown rates.
- **Iran**: severely constrained by sanctions; frozen assets are excluded;
  actual accessible buffer is likely lower than stated.
- **Libya**: dual-government structure makes the consolidated figure highly
  uncertain; assets may be pledged, earmarked, or disputed.
- **Bahrain**: GCC Development Fund provides implicit backstop that is not
  reflected in the liquid buffer figure — effective runway may be longer.
- **Kuwait**: RFFG ring-fence means the $760bn KIA headline figure is
  misleading for fiscal-sustainability analysis; effective buffer is ~$60bn.
- **Algeria**: FRR (Fonds de Régulation des Recettes) essentially depleted
  by 2017; the buffer is now entirely FX reserves at the Banque d'Algérie.
- Runway months assume the burn rate is constant — in practice, governments
  adjust spending and borrowing dynamically, so actual depletion may be
  faster or slower.

*Sources: SWF Institute, IMF Article IV Consultations (2023), IMF Regional
Economic Outlook — Middle East and Central Asia (Oct 2023), central bank
statistical bulletins (2023), SWF and sovereign fund annual reports (2023).*
""")
