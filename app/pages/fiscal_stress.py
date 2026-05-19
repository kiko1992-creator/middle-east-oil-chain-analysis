"""
Fiscal Breakeven Stress Monitor

Answers: "At current Brent price, which MENA governments are above or
below their fiscal breakeven?"

The fiscal breakeven is the oil price (USD/bbl) a government needs to
balance its budget.  A price below breakeven means the government must
draw down reserves, cut spending, or borrow — fiscal stress.

Page sections
-------------
  KPI cards    — live Brent, count below breakeven, worst gap, avg stress
  Status table — traffic-light view per country (Red/Amber/Green/Gray)
  Gap chart    — horizontal bar chart of price gap coloured by status
  Methodology  — formulas, thresholds, data caveats

Run standalone (from project root):
    streamlit run app/pages/fiscal_stress.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# ── Make project root importable so we can import src.model.* ─────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.model.fiscal_stress import (
    _AMBER_BUFFER,
    build_stress_table,
    fetch_brent_live,
    fetch_brent_ytd,
    load_breakeven,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fiscal Breakeven Stress",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT          = Path(__file__).resolve().parents[2]
_BREAKEVEN_CSV = _ROOT / "data" / "reference" / "fiscal_breakeven.csv"

# ── Visual constants ───────────────────────────────────────────────────────────
# Traffic-light colour palette — consistent across table and chart.
_STATUS_COLOUR: dict[str, str] = {
    "Red":   "#d62728",   # matplotlib red
    "Amber": "#ff7f0e",   # matplotlib orange
    "Green": "#2ca02c",   # matplotlib green
    "Gray":  "#aaaaaa",   # neutral grey for N/A countries
}

# Emoji indicator shown in the status table column.
_STATUS_ICON: dict[str, str] = {
    "Red":   "🔴",
    "Amber": "🟡",
    "Green": "🟢",
    "Gray":  "⚫",
}


# ── Cached data loading ────────────────────────────────────────────────────────
# We cache each network call separately so re-runs only hit yfinance
# when the TTL expires, not on every Streamlit interaction.

@st.cache_data(show_spinner="Loading breakeven reference data…")
def _load_breakeven() -> pd.DataFrame:
    """Load the static fiscal breakeven CSV (no TTL — file rarely changes)."""
    return load_breakeven(_BREAKEVEN_CSV)


@st.cache_data(show_spinner="Fetching live Brent price…", ttl=3600)
def _load_brent_live() -> float:
    """Fetch latest Brent close; cached for 1 hour to avoid rate-limiting."""
    return fetch_brent_live()


@st.cache_data(show_spinner="Loading YTD Brent history…", ttl=3600)
def _load_brent_ytd() -> pd.DataFrame:
    """Fetch year-to-date daily Brent closes; cached for 1 hour."""
    return fetch_brent_ytd()


# ── Guard — fail early if the reference file is missing ───────────────────────
if not _BREAKEVEN_CSV.exists():
    st.error(
        f"Required file not found: `{_BREAKEVEN_CSV.relative_to(_ROOT)}`  \n"
        "The fiscal breakeven reference CSV should be at  \n"
        "`data/reference/fiscal_breakeven.csv`"
    )
    st.stop()

# ── Load all data ──────────────────────────────────────────────────────────────
breakeven_df = _load_breakeven()
brent_live   = _load_brent_live()
ytd_prices   = _load_brent_ytd()

# Build the full stress table (pure Python — not cached separately so it
# always reflects the live price loaded above).
stress_df = build_stress_table(breakeven_df, brent_live, ytd_prices)

# Convenience subsets used throughout the page
_exporters    = stress_df[stress_df["country_type"].isin(["exporter", "mixed_importer"])]
_non_gray     = stress_df[stress_df["stress_status"] != "Gray"]
_brent_ok     = not math.isnan(brent_live)   # False when yfinance fetch failed


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏛️ Fiscal Stress")
    st.markdown("---")

    # Toggle: show or hide net-importer countries (Gray rows)
    show_importers: bool = st.checkbox(
        "Show net importers (Gray)",
        value=False,
        help="Net importers (Jordan, Lebanon, Morocco) are shown in Gray — "
             "the fiscal-breakeven concept does not apply to them.",
    )

    # Country filter — starts with all selected
    all_labels = sorted(stress_df["country_label"].tolist())
    selected: list[str] = st.multiselect(
        "Filter countries",
        options=all_labels,
        default=all_labels,
    )
    if not selected:
        selected = all_labels  # never leave empty

    st.markdown("---")
    st.caption(
        "**Sources:** IMF Article IV Consultations, "
        "IMF Regional Economic Outlook (MENA), 2023.  \n"
        "All breakeven figures are **preliminary estimates**.  \n"
        "Live Brent: Yahoo Finance (BZ=F).  \n"
        "Built with Streamlit + Plotly."
    )


# ── Apply filters ──────────────────────────────────────────────────────────────
display_df = stress_df[stress_df["country_label"].isin(selected)].copy()
if not show_importers:
    display_df = display_df[display_df["stress_status"] != "Gray"]


# ── Page header ────────────────────────────────────────────────────────────────
st.title("Fiscal Breakeven Stress Monitor")
st.caption(
    "IMF-style fiscal breakeven monitor for 14 MENA economies.  "
    "The **fiscal breakeven** is the Brent crude price at which a government "
    "balances its budget.  Trading below it signals fiscal stress — reserve "
    "drawdown, spending cuts, or borrowing.  "
    "All breakeven figures are **preliminary estimates** (see Methodology)."
)

# Show a warning banner when the live price could not be fetched
if not _brent_ok:
    st.warning(
        "⚠️ Could not fetch live Brent price from Yahoo Finance (BZ=F).  "
        "Stress classifications and gap metrics are unavailable.  "
        "The breakeven reference table is still shown below."
    )

st.markdown("---")


# ── KPI cards ──────────────────────────────────────────────────────────────────
# Four headline numbers that answer the most common questions at a glance.

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

# KPI 1 — live Brent price
with kpi1:
    if _brent_ok:
        st.metric(
            label="Live Brent (BZ=F)",
            value=f"${brent_live:.2f}",
            help="Most recent Brent Crude Futures close price (USD/bbl) from Yahoo Finance.",
        )
    else:
        st.metric("Live Brent (BZ=F)", "N/A", help="Price fetch failed.")

# KPI 2 — count of exporters currently below their breakeven (Red)
with kpi2:
    n_red = int((_exporters["stress_status"] == "Red").sum())
    n_exporters = len(_exporters)
    if _brent_ok:
        st.metric(
            label="Below Breakeven",
            value=f"{n_red} / {n_exporters}",
            delta=f"{n_red} in Red" if n_red > 0 else "None in Red",
            delta_color="inverse",
            help="Number of oil-exporting governments whose fiscal breakeven "
                 "exceeds the current Brent price.",
        )
    else:
        st.metric("Below Breakeven", "N/A")

# KPI 3 — worst price gap (most negative = most stressed exporter)
with kpi3:
    if _brent_ok and not _non_gray.empty:
        worst_idx = _non_gray["price_gap_usd"].idxmin()
        worst_gap = float(_non_gray.loc[worst_idx, "price_gap_usd"])
        worst_country = str(_non_gray.loc[worst_idx, "country_label"])
        sign = "+" if worst_gap >= 0 else ""
        st.metric(
            label="Worst Gap",
            value=f"{sign}${worst_gap:.0f}",
            delta=worst_country,
            delta_color="inverse" if worst_gap < 0 else "normal",
            help="Country with the largest negative price gap "
                 "(Brent − breakeven), i.e., deepest below breakeven.",
        )
    else:
        st.metric("Worst Gap", "N/A")

# KPI 4 — average YTD stress share across exporting countries
# (fraction of trading days this year that Brent spent below each exporter's breakeven)
with kpi4:
    if _brent_ok and len(_exporters) > 0:
        avg_stress = _exporters["stress_share_ytd"].mean(skipna=True)
        if math.isnan(avg_stress):
            st.metric("Avg Exporter Stress YTD", "N/A")
        else:
            st.metric(
                label="Avg Exporter Stress YTD",
                value=f"{avg_stress:.0%}",
                help="Average share of YTD trading days where Brent closed "
                     "below each exporter's fiscal breakeven.",
            )
    else:
        st.metric("Avg Exporter Stress YTD", "N/A")

st.markdown("---")


# ── Traffic-light status table ─────────────────────────────────────────────────
st.subheader("Country Fiscal Stress — Traffic-Light View")
st.caption(
    f"**Red** = Brent below fiscal breakeven (government in deficit at current price).  "
    f"**Amber** = within ${_AMBER_BUFFER:.0f} buffer above breakeven (thin headroom).  "
    f"**Green** = at least ${_AMBER_BUFFER:.0f} above breakeven (comfortable headroom).  "
    "**Gray** = not applicable (net importer or breakeven not set)."
)

# Build the display table: select and rename columns for readability
tbl = display_df.copy()

# Add a single-character icon column for the traffic-light signal
tbl["Status"] = tbl["stress_status"].map(_STATUS_ICON)

# Format the price-gap column: show sign and N/A for Gray
def _fmt_gap(row: pd.Series) -> str:
    """Format price_gap_usd with sign, or return N/A for Gray countries."""
    if row["stress_status"] == "Gray":
        return "N/A"
    v = row["price_gap_usd"]
    if math.isnan(v):
        return "N/A"
    sign = "+" if v >= 0 else ""
    return f"{sign}${v:.0f}"

tbl["Gap"] = tbl.apply(_fmt_gap, axis=1)

# Format stress days as int or N/A
def _fmt_stress_days(row: pd.Series) -> str:
    if row["stress_status"] == "Gray":
        return "N/A"
    v = row["stress_days_ytd"]
    return "N/A" if pd.isna(v) else str(int(v))

def _fmt_stress_share(row: pd.Series) -> str:
    if row["stress_status"] == "Gray":
        return "N/A"
    v = row["stress_share_ytd"]
    return "N/A" if (pd.isna(v) or math.isnan(v)) else f"{v:.0%}"

tbl["Stress Days YTD"] = tbl.apply(_fmt_stress_days, axis=1)
tbl["Stress Share YTD"] = tbl.apply(_fmt_stress_share, axis=1)

# Select and rename columns for the final display
table_cols = {
    "Status":                    "Status",
    "country_label":             "Country",
    "country_type":              "Type",
    "fiscal_breakeven_usd":      "Breakeven ($/bbl)",
    "brent_live_usd":            "Brent Live ($/bbl)",
    "Gap":                       "Price Gap ($/bbl)",
    "Stress Days YTD":           "Stress Days YTD",
    "Stress Share YTD":          "Stress Share YTD",
    "confidence":                "Confidence",
}
disp_tbl = tbl[[c for c in table_cols if c in tbl.columns]].rename(columns=table_cols)

st.dataframe(
    disp_tbl,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Status": st.column_config.TextColumn(
            "Status", width="small",
            help="Traffic-light status: 🔴 Red / 🟡 Amber / 🟢 Green / ⚫ Gray (N/A)",
        ),
        "Country": st.column_config.TextColumn("Country", width="medium"),
        "Type": st.column_config.TextColumn(
            "Type", width="small",
            help="exporter | mixed_importer | net_importer",
        ),
        "Breakeven ($/bbl)": st.column_config.NumberColumn(
            "Breakeven ($/bbl)", format="$%.0f",
            help="Estimated Brent price at which the government budget balances.",
        ),
        "Brent Live ($/bbl)": st.column_config.NumberColumn(
            "Brent ($/bbl)", format="$%.2f",
        ),
        "Price Gap ($/bbl)": st.column_config.TextColumn(
            "Price Gap ($/bbl)",
            help="Brent − Breakeven.  Positive = above breakeven; "
                 "negative = government in deficit at current price.",
        ),
        "Stress Days YTD": st.column_config.TextColumn(
            "Stress Days YTD",
            help="Trading days this calendar year where Brent closed below the breakeven.",
        ),
        "Stress Share YTD": st.column_config.TextColumn(
            "Stress Share YTD",
            help="Stress days / total YTD trading days.  Higher = more persistent stress.",
        ),
        "Confidence": st.column_config.TextColumn(
            "Confidence",
            help="Confidence in the breakeven estimate: high / medium / low / na.",
        ),
    },
)

st.markdown("---")


# ── Horizontal bar chart — price gap by country ────────────────────────────────
st.subheader("Price Gap vs Fiscal Breakeven")
st.caption(
    "**Price Gap = Live Brent − Fiscal Breakeven (USD/bbl).**  "
    "Bars to the left of zero (negative) mean Brent is below breakeven — "
    "fiscal stress zone.  Colour follows the same traffic-light scheme.  "
    "Net importers (Gray / N/A) are excluded from this chart."
)

# Filter out Gray countries (gap is N/A for them)
chart_df = display_df[display_df["stress_status"] != "Gray"].copy()
chart_df = chart_df.dropna(subset=["price_gap_usd"])

if chart_df.empty:
    st.info("No exporter data available for the current country selection.")
else:
    # Sort: worst gap (most negative) at the bottom so it stands out
    chart_df = chart_df.sort_values("price_gap_usd", ascending=True)

    # Build colour list matched to row order
    bar_colours = [_STATUS_COLOUR[s] for s in chart_df["stress_status"]]

    fig_gap = go.Figure()
    fig_gap.add_trace(
        go.Bar(
            x=chart_df["price_gap_usd"],
            y=chart_df["country_label"],
            orientation="h",
            marker_color=bar_colours,
            text=chart_df.apply(
                lambda r: f"+${r['price_gap_usd']:.0f}"
                          if r["price_gap_usd"] >= 0
                          else f"-${abs(r['price_gap_usd']):.0f}",
                axis=1,
            ),
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Price gap: %{x:+.1f} $/bbl<br>"
                "<extra></extra>"
            ),
        )
    )

    # Zero line — the break-even boundary
    fig_gap.add_vline(
        x=0,
        line_color="black",
        line_width=1.5,
        annotation_text="Breakeven",
        annotation_position="top",
        annotation_font_size=10,
    )

    # Amber zone shading: 0 to +15 is the thin-buffer area
    if _brent_ok:
        fig_gap.add_vrect(
            x0=0,
            x1=_AMBER_BUFFER,
            fillcolor="rgba(255, 127, 14, 0.08)",   # faint orange
            line_width=0,
            annotation_text="Amber buffer",
            annotation_position="top right",
            annotation_font_size=9,
            annotation_font_color="#ff7f0e",
        )

    fig_gap.update_layout(
        height=max(280, len(chart_df) * 38),
        xaxis=dict(
            title="Price Gap (Live Brent − Breakeven, USD/bbl)",
            zeroline=False,
        ),
        yaxis=dict(title=""),
        margin=dict(l=0, r=80, t=30, b=40),
        showlegend=False,
    )

    st.plotly_chart(fig_gap, use_container_width=True)

    # Legend key for the colour scheme
    legend_cols = st.columns(4)
    for col, (status, colour) in zip(
        legend_cols,
        [("Red — below breakeven", "#d62728"),
         (f"Amber — 0–${_AMBER_BUFFER:.0f} above", "#ff7f0e"),
         (f"Green — >${_AMBER_BUFFER:.0f} above", "#2ca02c"),
         ("Gray — N/A (importers)", "#aaaaaa")],
    ):
        col.markdown(
            f'<span style="color:{colour}; font-weight:bold;">■</span> {status}',
            unsafe_allow_html=True,
        )

st.markdown("---")


# ── Methodology expander ───────────────────────────────────────────────────────
with st.expander("Methodology — formulas, thresholds, and caveats"):
    st.markdown(f"""
### Stress Classification

| Status | Condition | Interpretation |
|--------|-----------|---------------|
| 🔴 **Red**   | Brent < Breakeven | Government in fiscal deficit at current oil price; must draw reserves, cut spending, or borrow |
| 🟡 **Amber** | Breakeven ≤ Brent < Breakeven + ${_AMBER_BUFFER:.0f} | Nominally in surplus but with only a thin buffer; vulnerable to short-term price dips |
| 🟢 **Green** | Brent ≥ Breakeven + ${_AMBER_BUFFER:.0f} | Comfortable fiscal headroom at current price |
| ⚫ **Gray**  | net_importer or breakeven = 0 | Fiscal breakeven concept not applicable for net importers |

### Metrics

```
price_gap_usd     = brent_live − fiscal_breakeven_usd

stress_days_ytd   = number of YTD trading days where
                    daily_close(BZ=F) < fiscal_breakeven_usd

stress_share_ytd  = stress_days_ytd / total_ytd_trading_days
```

### Data sources and confidence levels

**Breakeven estimates** are derived from IMF Article IV Consultations
and IMF Regional Economic Outlooks (MENA edition) for the most recent
available year (typically 2022–2023).  **All figures are preliminary
estimates** and should not be cited as official government disclosures.

Confidence levels reflect data availability and transparency:

| Confidence | Countries | Notes |
|------------|-----------|-------|
| **high**   | Kuwait, UAE, Qatar | Frequently cited in IMF staff reports; narrow range of estimates in the literature |
| **medium** | Saudi Arabia, Iraq, Oman, Bahrain | Cited in IMF reports but some uncertainty remains |
| **low**    | Algeria, Libya, Iran, Egypt | Political fragmentation, sanctions, or opaque fiscal frameworks increase uncertainty |
| **na**     | Jordan, Lebanon, Morocco | Net importers; concept not applicable |

### Caveats

- Breakeven prices reflect **fiscal breakeven**, not production cost breakeven.
  They include all government expenditure financed by oil revenues (wages,
  subsidies, transfers, debt service).
- Egypt is classified **mixed_importer** — it is a marginal net oil importer
  but a significant LNG exporter; its breakeven is less comparable to pure
  exporters and should be interpreted with caution.
- Iran's estimate is especially uncertain due to sanctions, a parallel
  exchange rate, and limited IMF access to data.
- Libya's estimate is uncertain due to the dual-government structure;
  the figure represents a notional consolidated breakeven.
- Live Brent price is the **Brent Crude Futures** front-month contract
  (BZ=F on Yahoo Finance).  The daily close is used, not intraday.
- Breakeven estimates are **point-in-time** (2023) and do not update
  automatically — fiscal reform, subsidy changes, or new spending
  commitments can shift a country's actual breakeven meaningfully.

*Sources: IMF Article IV Consultations (2023), IMF Regional Economic
Outlook — Middle East and Central Asia (Oct 2023),
World Bank Macro Poverty Outlook (2023), Yahoo Finance (BZ=F).*
""")
