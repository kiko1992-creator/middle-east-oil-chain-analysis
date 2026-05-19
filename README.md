# Middle East Oil Chain Analysis

Scenario-based analysis of oil dependency and economic-chain vulnerability
in 14 MENA economies (Bahrain, Egypt, Iran, Iraq, Jordan, Kuwait, Lebanon,
Libya, Morocco, Oman, Qatar, Saudi Arabia, UAE, Algeria).

---

## Research question

> **Which MENA governments are most at risk right now — and why?**

The project answers this by combining four distinct lenses:

1. **Fiscal stress** — how far below their budget breakeven each government
   currently sits, given the live Brent crude price.
2. **Reserve runway** — how many months of spending the country can sustain
   from liquid FX / SWF reserves before fiscal measures become unavoidable.
3. **Social stability risk** — food import dependency, inflation volatility,
   and the fiscal squeeze transmitted to households.
4. **Chain transmission severity** — how strongly historical oil price shocks
   propagated through each economy's fiscal → spending → subsidy → inflation chain.

---

## Right Now Risk formula

```
right_now_risk_score =
    0.35 × fiscal_stress_score
  + 0.25 × reserve_runway_risk
  + 0.20 × social_stability_risk
  + 0.20 × chain_transmission_severity_recent
```

All four components are normalised to **[0, 1]**.  The composite score is
also clamped to [0, 1] after weighting.

### Component definitions

| Component | Range | Formula |
|-----------|-------|---------|
| **fiscal_stress_score** | [0, 1] | `min(1, max(0, (breakeven_usd − brent_live) / breakeven_usd))` |
| **reserve_runway_risk** | [0, 1] | Linear: 1.0 at ≤ 6 months, 0.0 at ≥ 36 months; 0.0 for Gray (not stressed) |
| **social_stability_risk** | [0, 1] | `0.5 × food_exposure + 0.3 × fiscal_score + 0.2 × inflation_vol_norm` |
| **chain_transmission_severity_recent** | [0, 1] | Min-max normalised mean of `transmission_severity` over the most recent 3 calendar years |

### social_stability_risk sub-formula

```
food_security_exposure   = 0.6 × norm(food_imports_pct) + 0.4 × norm(cereal_import_dep)
inflation_vol_norm       = winsorize(p5–p95) then min-max normalise
social_stability_risk    = 0.5 × food_security_exposure
                         + 0.3 × fiscal_stress_score
                         + 0.2 × inflation_vol_norm
```

---

## Fallback / rescaled-weight policy

If any component is **NaN** for a country (due to missing data or no chain
CSV entry), the remaining available component weights are rescaled
proportionally to sum to 1.0.  **No country is silently dropped.**

Every output row records:

- `missing_components` — semicolon-separated list of missing component names
  with a bracketed reason (e.g. `chain_transmission_severity_recent [country absent from chain CSV]`)
- `rescaled_weights` — the actual per-row weights used as a JSON-style dict
- `component_weights_base` — the hardcoded default weights (in CSV export)
- `component_weights_effective` — alias for `rescaled_weights` (in CSV export)

---

## Data provenance

| Dataset | Columns used | Source | Notes |
|---------|-------------|--------|-------|
| `data/reference/fiscal_breakeven.csv` | `fiscal_breakeven_usd`, `country_type` | IMF Article IV / Regional Economic Outlook 2023 | Preliminary estimates; see `confidence` field |
| `data/reference/swf_reserves.csv` | `liquid_buffer_usd_bn`, `estimated_monthly_burn_usd_bn` | SWF annual reports, IMF Article IV, central bank bulletins 2023 | Liquid/accessible portion only; major illiquid SWFs excluded |
| `data/reference/food_security.csv` | `food_imports_pct_merch_imports`, `cereal_import_dependency` | World Bank TM.VAL.FOOD.ZS.UN; FAO FAOSTAT 2020–2023 | `is_estimate=True` for all rows |
| `data/processed/world_bank_panel.csv` | `FP_CPI_TOTL_ZG` (2000–2024 for inflation vol) | World Bank Open Data API | Fetched via `src/data/fetch_world_bank.py` |
| `outputs/tables/chain_transmission.csv` | `transmission_severity`, `year` | Computed by `src/model/chain_model.py` | 2000–2024; 14 countries × 25 years |
| Live Brent price | `brent_live_usd` | Yahoo Finance ticker `BZ=F` via yfinance | Cached hourly in the dashboard; `float('nan')` on failure |

---

## Normalization methods

| Component | Method | Outlier handling |
|-----------|--------|-----------------|
| fiscal_stress_score | Continuous formula (no normalisation step) | Clamped to [0, 1] |
| reserve_runway_risk | Linear interpolation | Floor 6 mo, ceiling 36 mo |
| food_imports_pct / cereal_dep | Min-max across 14 countries | None |
| inflation volatility | p5–p95 winsorisation, then min-max | Lebanon (σ = 73 pp) capped at p95 ≈ 35 pp |
| chain_transmission_severity_recent | Min-max across 14 countries | None |

---

## Limitations

- **Brent proxy** — `BZ=F` (Brent Futures) is used as a real-time price
  proxy.  It may diverge slightly from spot Brent and is unavailable outside
  market hours.  All downstream scores use the same price snapshot from a
  single yfinance call to avoid cross-component inconsistency.

- **Estimate confidence** — fiscal breakeven and reserve data are 2023
  preliminary estimates.  Countries flagged `confidence=low` (Iran, Lebanon,
  Libya) have restricted data access or significant estimation uncertainty.

- **Missing-component behaviour** — when a component is NaN, weights are
  rescaled rather than zero-filled.  This preserves proportionality but means
  the score is not directly comparable to rows with all four components.

- **Chain data coverage** — `chain_transmission.csv` covers 2000–2024.
  The recent-3-year window uses calendar years present in the file, not
  necessarily the last 3 calendar years from today.

- **Static reference data** — breakeven, reserves, and food security CSVs are
  point-in-time reference tables.  They are not automatically refreshed.

- **First-order model** — the chain transmission model is a linear
  approximation.  It does not capture exchange-rate feedback, monetary policy
  response, or non-linear subsidy reform dynamics.

---

## Source governance

All reference assumptions are traceable to a stable source ID in
`data/reference/source_registry.csv`.  Each row in the reference CSVs carries
`source_id_primary` and `source_id_secondary` columns that map to registry
entries.  The registry records organization, publication year, retrieval date,
URL, and `confidence_tier` (high / medium / low).

Run the integrity check at any time:

```bash
python -m src.data.validate_reference          # PASS/FAIL summary
python -m src.data.validate_reference --strict  # exits 1 on any failure
```

The validator checks: unique source IDs, referential integrity (every source ID
in a reference CSV resolves to a registry row), confidence enum validity,
monotonic uncertainty bands (low ≤ base ≤ high), no duplicate `country_code_a3`,
and non-null base values for all exporter rows.

---

## Uncertainty bands

Each reference CSV carries three uncertainty columns per key assumption:

| File | Columns |
|------|---------|
| `fiscal_breakeven.csv` | `breakeven_low_usd`, `breakeven_base_usd`, `breakeven_high_usd` |
| `swf_reserves.csv` | `liquid_buffer_low/base/high_usd_bn`, `monthly_burn_low/base/high_usd_bn` |
| `food_security.csv` | `cereal_dependency_low`, `cereal_dependency_base`, `cereal_dependency_high` |

The live model uses only the `base` columns — existing behavior is unchanged.
The `low`/`high` columns are consumed by the backtesting scenario framework
(`stress` and `optimistic` scenarios).

Band widths reflect source confidence:

| Tier | Breakeven | Liquid buffer | Burn rate | Cereal dep. |
|------|-----------|--------------|-----------|-------------|
| high | from notes | ±15% | ±20% | ±5 pp |
| medium | from notes | ±25% | ±30% | ±10 pp |
| low | from notes | ±40% | ±50% | ±20 pp |

Countries with `confidence=""` (net importers: JOR, LBN, MAR) have null
uncertainty bands — the formula returns 0 fiscal stress for them regardless.

---

## Backtesting (initial scaffolding)

`src/model/backtest.py` provides conditional backtesting across historical Brent
price environments.  It is *conditional* — the 2023 reference estimates are held
fixed; only the annual-average Brent price and chain transmission severity vary.

```python
from src.model.backtest import run_backtest_range, summarize_rank_stability, export_backtest_outputs

# Base scenario, 2008–2022
df = run_backtest_range(2008, 2022, scenario="base")

# Stress scenario: high breakeven + low liquid buffer + high burn
df_stress = run_backtest_range(2015, 2016, scenario="stress")

# Export per-year CSVs + rank stability summary
summary = summarize_rank_stability(df)
export_backtest_outputs(df, summary_df=summary)
```

Three scenarios are supported:

| Scenario | Breakeven | Liquid buffer | Burn |
|----------|-----------|--------------|------|
| `base` | `breakeven_base_usd` | `liquid_buffer_base_usd_bn` | `monthly_burn_base_usd_bn` |
| `stress` | `breakeven_high_usd` | `liquid_buffer_low_usd_bn` | `monthly_burn_high_usd_bn` |
| `optimistic` | `breakeven_low_usd` | `liquid_buffer_high_usd_bn` | `monthly_burn_low_usd_bn` |

See `docs/backtesting_plan.md` for target periods, face-validity criteria, and
known limitations.

---

## Project structure

```
middle-east-oil-chain-analysis/
├── app/
│   ├── dashboard.py              # Main Streamlit dashboard (6 tabs)
│   └── pages/
│       ├── price_analysis.py     # Brent price history & volatility
│       ├── supply_chain.py       # Trade route & Suez exposure
│       ├── social_stability.py   # Social stability risk (Addition 3)
│       └── fiscal_stress.py      # Fiscal stress deep-dive (Addition 1)
├── docs/
│   └── backtesting_plan.md       # Target periods, success criteria, limitations
├── src/
│   ├── data/
│   │   ├── brent.py              # Single source of truth for Brent prices
│   │   ├── fetch_world_bank.py
│   │   ├── clean_world_bank.py
│   │   └── validate_reference.py # Reference data integrity checker
│   └── model/
│       ├── vulnerability_index.py  # OCVI (Addition 0)
│       ├── chain_model.py          # Chain transmission model
│       ├── fiscal_stress.py        # Fiscal breakeven stress (Addition 1)
│       ├── reserve_runway.py       # Reserve runway model (Addition 2)
│       ├── social_stability.py     # Social stability risk (Addition 3)
│       ├── right_now_risk.py       # Right Now Risk composite (Addition 5)
│       └── backtest.py             # Conditional backtesting scaffolding
├── data/
│   ├── processed/world_bank_panel.csv
│   └── reference/
│       ├── source_registry.csv     # Stable source IDs + confidence tiers
│       ├── fiscal_breakeven.csv    # Includes low/base/high uncertainty bands
│       ├── swf_reserves.csv        # Includes low/base/high uncertainty bands
│       └── food_security.csv       # Includes low/base/high uncertainty bands
└── outputs/tables/
    ├── ocvi_scores.csv
    ├── chain_transmission.csv
    ├── right_now_risk_scores.csv   # Auto-exported on each pipeline run
    └── backtest/                   # Per-year backtest CSVs + rank stability
```

---

## Running the dashboard

```bash
# From the project root:
streamlit run app/dashboard.py
```

Requirements: `streamlit`, `pandas`, `plotly`, `yfinance`, `numpy`.

To regenerate the data pipeline from scratch:

```bash
python -m src.data.fetch_world_bank
python -m src.data.clean_world_bank
python -m src.model.vulnerability_index
python -m src.model.chain_model
```

---

## Screenshots

> _Add dashboard screenshots here once deployed._

| Tab | Description |
|-----|-------------|
| OCVI Rankings | Country vulnerability ranking with radar chart |
| Oil Rents % GDP | Time-series of oil revenue dependency |
| Price Shock | First-order fiscal exposure to a price change |
| GDP Growth vs Inflation | Macro scatter with bubble size = GDP |
| Chain Transmission | Severity heatmap and stage breakdown |
| Fiscal Stress | Right Now Risk composite + component detail |
