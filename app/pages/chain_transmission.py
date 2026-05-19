"""
Chain Transmission Severity — Addition 4

Answers: "How strongly does an oil price shock propagate through each MENA
economy, and which structural stages amplify or dampen the chain?"

Five structural stages capture the full transmission path:
  Stage 1  Oil price → fiscal revenue linkage
  Stage 2  Fiscal pressure → inflation (subsidy / price pass-through)
  Stage 3  Inflation → employment vulnerability
  Stage 4  Employment / wages → household consumption impact
  Stage 5  Consumption contraction → GDP growth feedback

Severity formula:
    chain_transmission_severity = min(1.0, mean(stage1..5) × amplification_factor)

Page sections
-------------
  KPI cards     — highest-severity country, mean severity, # fast-transmission
  Bar chart     — countries ranked by chain_transmission_severity, coloured by speed
  Stage heatmap — countries × stages structural scores
  Data table    — full detail with confidence flag
  Methodology   — formula, stage definitions, amplification rationale, caveats

Run standalone (from project root):
    streamlit run app/pages/chain_transmission.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app.export import make_csv_download_button

from src.model.chain_transmission import (
    _SNAPSHOT_YEAR,
    _STAGE_COLS,
    build_chain_table,
    load_chain_reference,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT    = Path(__file__).resolve().parents[2]
_REF     = _ROOT / "data" / "reference" / "chain_transmission.csv"
_OUT     = _ROOT / "outputs" / "tables" / "chain_transmission.csv"

# ── Constants ──────────────────────────────────────────────────────────────────
_STAGE_LABELS: dict[str, str] = {
    "stage1_oil_fiscal":            "Stage 1 · Oil → Fiscal",
    "stage2_fiscal_inflation":      "Stage 2 · Fiscal → Inflation",
    "stage3_inflation_employment":  "Stage 3 · Inflation → Employment",
    "stage4_employment_consumption":"Stage 4 · Employment → Consumption",
    "stage5_consumption_growth":    "Stage 5 · Consumption → Growth",
}

_SPEED_COLOUR: dict[str, str] = {
    "fast":   "#d62728",
    "medium": "#ff7f0e",
    "slow":   "#2ca02c",
}

_NAME_MAP: dict[str, str] = {
    "Egypt, Arab Rep.":     "Egypt",
    "Iran, Islamic Rep.":   "Iran",
    "United Arab Emirates": "UAE",
}


def _short(name: str) -> str:
    return _NAME_MAP.get(name, name)


# ── Data loader ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading chain transmission data…")
def _load() -> pd.DataFrame:
    """Load chain output CSV, falling back to recomputing from reference."""
    if _OUT.exists():
        df = pd.read_csv(_OUT)
    elif _REF.exists():
        df = build_chain_table(load_chain_reference(_REF))
    else:
        return pd.DataFrame()

    df["country_label"] = df["country_name"].apply(_short)
    return df


# ── Page ───────────────────────────────────────────────────────────────────────

st.title("⛓️  Chain Transmission Severity")
st.caption(
    "**Transmission chain:** Oil Price → Fiscal Revenue (1) → Inflation (2) → "
    "Employment (3) → Consumption (4) → Growth (5)  ·  "
    "`severity = mean(stage 1–5) × amplification factor`, clamped [0, 1]."
)

df = _load()

if df.empty:
    st.error(
        "Chain transmission data not found.  \n"
        "Run from the project root:  \n"
        "`python -m src.model.chain_transmission`"
    )
    st.stop()

# Latest year in file (static snapshot = 2024)
_year = int(df["year"].max()) if "year" in df.columns else _SNAPSHOT_YEAR
df_snap = df[df["year"] == _year].copy() if "year" in df.columns else df.copy()

_sev_col = (
    "chain_transmission_severity"
    if "chain_transmission_severity" in df_snap.columns
    else "transmission_severity"
)

# ── KPI row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

_top    = df_snap.loc[df_snap[_sev_col].idxmax()]
_mean   = df_snap[_sev_col].mean()
_n_fast = int((df_snap["transmission_speed"] == "fast").sum()) if "transmission_speed" in df_snap.columns else 0

k1.metric(
    "Highest severity",
    _top.get("country_label", _top.get("country_name", "N/A")),
    f"{_top[_sev_col]:.3f}",
    delta_color="inverse",
)
k2.metric("Mean severity (14 countries)", f"{_mean:.3f}")
k3.metric("Fast-transmission countries", f"{_n_fast} / {len(df_snap)}")
k4.metric("Reference year", str(_year), "Static structural estimate")

st.markdown("---")

# ── Ranked bar chart ───────────────────────────────────────────────────────────
st.subheader("Country Rankings by Chain Transmission Severity")

bar_df = df_snap.sort_values(_sev_col).copy()

fig_bar = go.Figure()
for speed, colour in _SPEED_COLOUR.items():
    sub = bar_df[bar_df["transmission_speed"] == speed] if "transmission_speed" in bar_df.columns else bar_df
    if sub.empty:
        continue
    fig_bar.add_trace(go.Bar(
        x=sub[_sev_col],
        y=sub["country_label"],
        orientation="h",
        name=speed.capitalize(),
        marker_color=colour,
        text=sub[_sev_col].map(lambda v: f"{v:.3f}"),
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Severity: %{x:.4f}<br>"
            f"Transmission speed: {speed}<extra></extra>"
        ),
    ))

fig_bar.update_layout(
    barmode="overlay",
    height=480,
    xaxis=dict(range=[0, 1.08], title="Chain Transmission Severity"),
    yaxis_title=None,
    legend=dict(title="Transmission speed", orientation="h", y=-0.13, font_size=12),
    margin=dict(l=10, r=90, t=10, b=60),
)
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ── Stage heatmap — countries × stages ────────────────────────────────────────
st.subheader("Stage Scores — Countries × Stages")
st.caption(
    "Each cell is a structural score in [0, 1] for that transmission stage in that country.  "
    "Red = strong transmission; green = weak / buffered."
)

_stage_cols_present = [c for c in _STAGE_COLS if c in df_snap.columns]

if _stage_cols_present:
    heat_df = (
        df_snap.set_index("country_label")[_stage_cols_present]
        .rename(columns=_STAGE_LABELS)
    )
    _severity_order = (
        df_snap.sort_values(_sev_col, ascending=False)["country_label"].tolist()
    )
    heat_df = heat_df.loc[[c for c in _severity_order if c in heat_df.index]]

    fig_heat = go.Figure(go.Heatmap(
        z=heat_df.values,
        x=heat_df.columns.tolist(),
        y=heat_df.index.tolist(),
        colorscale="RdYlGn_r",
        zmin=0.0,
        zmax=1.0,
        colorbar=dict(title="Score", thickness=14),
        hovertemplate="<b>%{y}</b><br>%{x}<br>Score: %{z:.2f}<extra></extra>",
        text=heat_df.round(2).values,
        texttemplate="%{z:.2f}",
        textfont=dict(size=10),
    ))
    fig_heat.update_layout(
        height=440,
        margin=dict(l=10, r=10, t=10, b=90),
        xaxis=dict(tickangle=-22, side="bottom"),
        yaxis_title=None,
    )
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.info("Stage score columns not found in data. Re-run `python -m src.model.chain_transmission`.")

st.markdown("---")

# ── Full data table ────────────────────────────────────────────────────────────
with st.expander("Full data table — all 14 countries"):
    _show = {
        "country_label":              "Country",
        "transmission_speed":         "Speed",
        "amplification_factor":       "Amplif.",
        "stage1_oil_fiscal":          "Stage 1",
        "stage2_fiscal_inflation":    "Stage 2",
        "stage3_inflation_employment":"Stage 3",
        "stage4_employment_consumption": "Stage 4",
        "stage5_consumption_growth":  "Stage 5",
        _sev_col:                     "Severity",
        "confidence":                 "Confidence",
    }
    _show = {k: v for k, v in _show.items() if k in df_snap.columns}
    disp = (
        df_snap.sort_values(_sev_col, ascending=False)
        [list(_show.keys())]
        .rename(columns=_show)
        .round(4)
    )
    st.dataframe(
        disp,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Severity": st.column_config.ProgressColumn(
                "Severity", format="%.3f", min_value=0.0, max_value=1.0,
            ),
            "Amplif.": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    make_csv_download_button(disp, "chain_transmission_table.csv", "Download table as CSV")

# ── Methodology ────────────────────────────────────────────────────────────────
with st.expander("Methodology"):
    st.markdown("""
**Formula**

```
stage_mean                  = mean(stage1_oil_fiscal, stage2_fiscal_inflation,
                                   stage3_inflation_employment,
                                   stage4_employment_consumption,
                                   stage5_consumption_growth)
chain_transmission_severity = min(1.0, stage_mean × amplification_factor)
```

**Stage definitions**

| Stage | Transmission path | Key drivers |
|-------|-------------------|-------------|
| Stage 1 | Oil price → Fiscal revenue | Hydrocarbon share of government revenue; exporters score 0.7–0.95, importers 0.1–0.4 |
| Stage 2 | Fiscal pressure → Inflation | Subsidy pass-through rate; countries with extensive price controls score higher |
| Stage 3 | Inflation → Employment | Real-wage erosion and labour-market flexibility; high youth unemployment raises score |
| Stage 4 | Employment → Consumption | Household spending sensitivity; large public-sector employment dampens this stage |
| Stage 5 | Consumption → Growth | Strength of domestic demand channel relative to hydrocarbon revenue in GDP |

**Amplification factor**

| Range | Meaning | Examples |
|-------|---------|---------|
| 0.78–0.82 | Large SWF or diversified economy buffers the shock | Kuwait (0.78), UAE (0.80), Qatar (0.82) |
| 0.88–0.94 | Moderate buffers or partial diversification | Morocco (0.88), Saudi Arabia (0.90), Oman (0.94) |
| 1.04–1.12 | Limited fiscal space, high subsidy dependency | Jordan (1.04), Egypt & Iran (1.08), Bahrain (1.12) |
| 1.18–1.38 | Institutional weakness, conflict, or embedded inflation | Algeria (1.18), Iraq (1.25), Lebanon (1.32), Libya (1.38) |

**Data sources**

Stage scores are expert estimates calibrated to IMF Article IV Consultations 2023,
IMF Regional Economic Outlook MENA October 2023, and Coady et al. (IMF 2015) on
energy subsidy pass-through rates.  All rows carry `is_estimate = True`.

Confidence tiers: *high* = well-documented GCC states (Saudi Arabia, Kuwait, UAE,
Qatar); *medium* = Oman, Bahrain, Algeria, Iraq, Egypt, Jordan, Morocco;
*low* = Libya, Iran, Lebanon (data limitations, conflict, or sanctions).

**Integration with Right Now Risk (Addition 5)**

`chain_transmission_severity_recent` feeds the **0.20** weight in the Right Now
Risk composite score.  For this static snapshot the recent value equals the
composite severity.  When the model is extended to a time series, the recent
value will be the min-max normalised mean over the most recent 3 calendar years.
""")
