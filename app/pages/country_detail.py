"""
Country Detail — Priority 3B

Deep-dive view for a single MENA economy, covering all four Right Now Risk
components, the historical trend, and data sources from the source registry.

Page sections
-------------
  1. Header + composite risk badge
  2. Four component score cards (fiscal / runway / social / chain)
  3. Historical risk trend line (2015–2024)
  4. Weighted contribution bar chart
  5. Benchmark comparison (IMF FM tier vs WB MPO status vs model tier)
  6. Data sources from source_registry.csv
  7. Raw component data expander + download button

Run standalone (from project root):
    streamlit run app/pages/country_detail.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app.export import make_csv_download_button

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT         = Path(__file__).resolve().parents[2]
_BREAKEVEN    = _ROOT / "data" / "reference" / "fiscal_breakeven.csv"
_RESERVES     = _ROOT / "data" / "reference" / "swf_reserves.csv"
_FOOD         = _ROOT / "data" / "reference" / "food_security.csv"
_CHAIN_OUT    = _ROOT / "outputs" / "tables" / "chain_transmission.csv"
_PANEL        = _ROOT / "data" / "processed" / "world_bank_panel.csv"
_HIST         = _ROOT / "outputs" / "tables" / "historical_risk_index.csv"
_REGISTRY     = _ROOT / "data" / "reference" / "source_registry.csv"
_BENCHMARKS   = _ROOT / "data" / "reference" / "imf_wb_benchmarks.csv"

_RISK_COLOURS = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
_TIER_BADGE   = {"Low": "#2ca02c", "Medium": "#ff7f0e", "High": "#d62728"}

_NAME_MAP: dict[str, str] = {
    "Egypt, Arab Rep.":     "Egypt",
    "Iran, Islamic Rep.":   "Iran",
    "United Arab Emirates": "UAE",
}

def _short(name: str) -> str:
    return _NAME_MAP.get(name, name)


# ── Loaders ────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_rnr() -> dict:
    from src.model.right_now_risk import run_right_now_risk
    try:
        return run_right_now_risk(
            breakeven_path=_BREAKEVEN,
            reserves_path=_RESERVES,
            food_path=_FOOD,
            chain_path=_CHAIN_OUT,
            panel_path=_PANEL,
            export_csv=None,
        )
    except Exception as exc:
        return {"error": str(exc)}


@st.cache_data(show_spinner=False)
def _load_hist() -> pd.DataFrame:
    if not _HIST.exists():
        return pd.DataFrame()
    df = pd.read_csv(_HIST)
    if "country_label" not in df.columns and "country_name" in df.columns:
        df["country_label"] = df["country_name"].apply(_short)
    return df


@st.cache_data(show_spinner=False)
def _load_registry() -> pd.DataFrame:
    if not _REGISTRY.exists():
        return pd.DataFrame()
    return pd.read_csv(_REGISTRY)


@st.cache_data(show_spinner=False)
def _load_benchmarks() -> pd.DataFrame:
    if not _BENCHMARKS.exists():
        return pd.DataFrame()
    return pd.read_csv(_BENCHMARKS)


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Country Detail — MENA Oil Chain",
    page_icon="🔍",
    layout="wide",
)

# ── Country selector ───────────────────────────────────────────────────────────
rnr_result = _load_rnr()

if "error" in rnr_result:
    st.error(
        f"Failed to load Right Now Risk data: {rnr_result['error']}  \n"
        "Ensure the data pipeline has been run from the project root."
    )
    st.stop()

rnr_df = rnr_result["right_now_risk_df"].copy()
if "country_label" not in rnr_df.columns and "country_name" in rnr_df.columns:
    rnr_df["country_label"] = rnr_df["country_name"].apply(_short)

country_options = sorted(rnr_df["country_label"].dropna().unique().tolist())

# Allow pre-selection from query params (for sidebar navigation)
_qp = st.query_params.get("country", country_options[0] if country_options else "")
_default_idx = country_options.index(_qp) if _qp in country_options else 0

selected = st.selectbox(
    "Select country",
    options=country_options,
    index=_default_idx,
    help="Choose a country for a full component breakdown.",
)

row = rnr_df[rnr_df["country_label"] == selected]
if row.empty:
    st.warning(f"No data found for {selected}.")
    st.stop()
row = row.iloc[0]

st.markdown("---")

# ── 1. Header + composite risk badge ──────────────────────────────────────────
composite = row.get("right_now_risk_score", float("nan"))
rank_val  = row.get("right_now_risk_rank", None)

col_hdr, col_badge = st.columns([3, 1])
with col_hdr:
    st.title(f"🔍 {selected}")
    st.caption(
        "Right Now Risk composite score and component breakdown.  "
        "Scores in **[0, 1]** — higher = more at risk."
    )
with col_badge:
    if pd.notna(composite):
        colour = (
            "#d62728" if composite >= 0.60 else
            "#ff7f0e" if composite >= 0.35 else
            "#2ca02c"
        )
        tier_label = (
            "High Risk" if composite >= 0.60 else
            "Medium Risk" if composite >= 0.35 else
            "Low Risk"
        )
        st.markdown(
            f"""
            <div style="background:{colour};border-radius:10px;padding:18px;text-align:center">
              <span style="color:white;font-size:2rem;font-weight:bold">{composite:.3f}</span><br>
              <span style="color:white;font-size:0.9rem">{tier_label}</span>
              {"<br><span style='color:white;font-size:0.8rem'>Rank #" + str(rank_val) + " / 14</span>" if rank_val else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")

# ── 2. Four component score cards ─────────────────────────────────────────────
st.subheader("Component Scores")

components = [
    ("fiscal_stress_score",                "Fiscal Stress",       "0.35",
     "Brent below fiscal breakeven → government runs a deficit at current prices."),
    ("reserve_runway_risk",                "Reserve Runway Risk", "0.25",
     "How many months of spending can be funded from liquid FX / SWF reserves."),
    ("social_stability_risk",              "Social Stability",    "0.20",
     "Food import dependency + inflation volatility + fiscal pass-through."),
    ("chain_transmission_severity_recent", "Chain Transmission",  "0.20",
     "Structural severity of oil shock propagation through the economy."),
]

c1, c2, c3, c4 = st.columns(4)
for col, (field, label, weight, tip) in zip([c1, c2, c3, c4], components):
    val = row.get(field, float("nan"))
    with col:
        if pd.notna(val):
            col.metric(label=f"{label} (w={weight})", value=f"{val:.3f}",
                       help=tip, delta=None)
            col.progress(float(val), text=None)
        else:
            col.metric(label=f"{label} (w={weight})", value="N/A", help=tip)
            col.caption("Component not available — weight rescaled.")

st.markdown("---")

# ── 3. Historical risk trend ───────────────────────────────────────────────────
hist_df = _load_hist()

if not hist_df.empty and "country_label" in hist_df.columns:
    st.subheader("Historical Right Now Risk (2015–2024)")
    cdf = hist_df[hist_df["country_label"] == selected].copy()
    cdf = cdf.sort_values("year")

    if not cdf.empty:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=cdf["year"],
            y=cdf["right_now_risk_score"],
            mode="lines+markers",
            name="Right Now Risk",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=7),
            hovertemplate="Year %{x}<br>Score: %{y:.3f}<extra></extra>",
        ))
        # Component area fills
        comp_colours = {
            "fiscal_stress_score":   "rgba(214,39,40,0.15)",
            "reserve_runway_risk":   "rgba(255,127,14,0.15)",
            "social_stability_risk": "rgba(44,160,44,0.15)",
            "chain_transmission_severity": "rgba(148,103,189,0.15)",
        }
        comp_names = {
            "fiscal_stress_score":   "Fiscal Stress",
            "reserve_runway_risk":   "Reserve Runway",
            "social_stability_risk": "Social Stability",
            "chain_transmission_severity": "Chain (when available)",
        }
        for comp, colour in comp_colours.items():
            if comp in cdf.columns:
                fig_trend.add_trace(go.Scatter(
                    x=cdf["year"], y=cdf[comp],
                    mode="lines", name=comp_names[comp],
                    line=dict(color=colour.replace("0.15", "0.8"), width=1, dash="dot"),
                    fill=None, opacity=0.6,
                    hovertemplate=f"{comp_names[comp]}: %{{y:.3f}}<extra></extra>",
                ))

        fig_trend.update_layout(
            height=320,
            xaxis_title="Year",
            yaxis=dict(range=[0, 1.05], title="Score"),
            legend=dict(orientation="h", y=-0.25, font_size=11),
            margin=dict(l=10, r=10, t=10, b=60),
            hovermode="x unified",
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info(f"No historical data available for {selected}.")
else:
    st.info(
        "Historical index not found.  "
        "Run: `python -m src.model.historical_index`"
    )

st.markdown("---")

# ── 4. Weighted contribution bar chart ────────────────────────────────────────
st.subheader("Weighted Contribution to Composite Score")

_weights = {
    "Fiscal Stress":    ("fiscal_stress_score",                0.35),
    "Reserve Runway":   ("reserve_runway_risk",                0.25),
    "Social Stability": ("social_stability_risk",              0.20),
    "Chain Transmission": ("chain_transmission_severity_recent", 0.20),
}

contrib_rows = []
for label, (field, w) in _weights.items():
    val = row.get(field, float("nan"))
    contrib = val * w if pd.notna(val) else float("nan")
    contrib_rows.append({"Component": label, "Score": val, "Weight": w, "Contribution": contrib})
contrib_df = pd.DataFrame(contrib_rows)

if not contrib_df["Contribution"].isna().all():
    fig_contrib = go.Figure(go.Bar(
        x=contrib_df["Contribution"].fillna(0),
        y=contrib_df["Component"],
        orientation="h",
        marker_color=["#d62728", "#ff7f0e", "#2ca02c", "#9467bd"],
        text=contrib_df["Contribution"].apply(
            lambda v: f"{v:.3f}" if pd.notna(v) else "N/A"
        ),
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Contribution: %{x:.4f}<br>"
            "<extra></extra>"
        ),
    ))
    fig_contrib.update_layout(
        height=260,
        xaxis=dict(range=[0, 0.50], title="Contribution to composite score"),
        yaxis_title=None,
        margin=dict(l=10, r=60, t=10, b=10),
    )
    st.plotly_chart(fig_contrib, use_container_width=True)

st.markdown("---")

# ── 5. Benchmark comparison ───────────────────────────────────────────────────
benchmarks = _load_benchmarks()
a3 = row.get("country_code_a3", None)

if not benchmarks.empty and a3 is not None:
    bench_row = benchmarks[benchmarks["country_code_a3"] == a3]
    if not bench_row.empty:
        st.subheader("Benchmark Comparison")
        bench_row = bench_row.iloc[0]
        imf_tier = bench_row.get("imf_fm_risk_tier", "N/A")
        wb_status = bench_row.get("wb_mpo_status", "N/A")

        # Map composite score to model tier (tertile of current scores)
        all_scores = rnr_df["right_now_risk_score"].dropna()
        t33 = float(all_scores.quantile(1/3))
        t67 = float(all_scores.quantile(2/3))
        model_tier = (
            "High"   if composite >= t67 else
            "Medium" if composite >= t33 else
            "Low"
        ) if pd.notna(composite) else "N/A"

        b1, b2, b3 = st.columns(3)
        b1.metric("Model tier (tertile)", model_tier)
        b2.metric("IMF Fiscal Monitor 2023", imf_tier, help="Source: IMF_FM_OCT2023")
        b3.metric("WB Macro Poverty Outlook 2023", wb_status, help="Source: WB_MPO_2023")

        _bench_notes = bench_row.get("notes", "")
        if _bench_notes and pd.notna(_bench_notes):
            st.caption(f"Benchmark notes: {_bench_notes}")

        st.markdown("---")

# ── 6. Data sources ───────────────────────────────────────────────────────────
st.subheader("Data Sources")
registry = _load_registry()

if not registry.empty:
    # Collect source IDs used across all reference files for this country
    source_ids: set[str] = set()
    for ref_path in [_BREAKEVEN, _RESERVES, _FOOD]:
        if ref_path.exists():
            ref_df = pd.read_csv(ref_path)
            if a3 is not None and "country_code_a3" in ref_df.columns:
                cref = ref_df[ref_df["country_code_a3"] == a3]
                for col in ("source_id_primary", "source_id_secondary"):
                    if col in cref.columns:
                        source_ids.update(v for v in cref[col].tolist() if v and pd.notna(v))

    if source_ids:
        src_df = registry[registry["source_id"].isin(source_ids)][
            ["source_id", "source_name", "organization", "publication_year",
             "confidence_tier", "url"]
        ].reset_index(drop=True)
        st.dataframe(src_df, hide_index=True, use_container_width=True)
    else:
        st.caption("No source mapping found for this country in reference files.")
else:
    st.caption("Source registry not found.")

st.markdown("---")

# ── 7. Raw component data expander ────────────────────────────────────────────
with st.expander("Raw component data"):
    all_cols = [
        "country_code_a3", "country_label",
        "right_now_risk_score", "right_now_risk_rank",
        "fiscal_stress_score", "reserve_runway_risk",
        "social_stability_risk", "chain_transmission_severity_recent",
        "missing_components", "rescaled_weights",
    ]
    present = [c for c in all_cols if c in rnr_df.columns]
    raw_row = rnr_df[rnr_df["country_label"] == selected][present].copy()
    st.dataframe(raw_row.T.rename(columns={raw_row.index[0]: "Value"}),
                 use_container_width=True)
    make_csv_download_button(
        raw_row,
        filename=f"country_detail_{selected.replace(' ', '_').lower()}.csv",
        label="Download country data as CSV",
    )
