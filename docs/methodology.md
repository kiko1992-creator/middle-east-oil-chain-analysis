# MENA Oil Chain Analysis — Methodology

This document describes the data sources, model formulas, normalisation
methods, validation results, and known limitations of the Right Now Risk
composite score.  All formulas are expressed as plain code blocks; no LaTeX
is used.  Numbers are drawn directly from the model source files and
validated output CSVs.

---

## 1. Research Question

> **Which MENA governments are most at risk right now — and why?**

The model answers this by combining four quantitative lenses into a single
composite score, `right_now_risk_score`, ranging from 0 (least at risk) to
1 (most at risk).  The score is designed to reflect *structural* fiscal and
social vulnerability given the *current* Brent crude oil price — not a
forecast of specific events.

---

## 2. Country Coverage

The analysis covers 14 MENA economies:

| # | Country | ISO A3 | Classification |
|---|---------|--------|----------------|
| 1 | Algeria | DZA | Oil exporter |
| 2 | Bahrain | BHR | Oil exporter |
| 3 | Egypt | EGY | Net importer |
| 4 | Iran | IRN | Oil exporter |
| 5 | Iraq | IRQ | Oil exporter |
| 6 | Jordan | JOR | Net importer |
| 7 | Kuwait | KWT | Oil exporter |
| 8 | Lebanon | LBN | Net importer |
| 9 | Libya | LBY | Oil exporter |
| 10 | Morocco | MAR | Net importer |
| 11 | Oman | OMN | Oil exporter |
| 12 | Qatar | QAT | Oil exporter |
| 13 | Saudi Arabia | SAU | Oil exporter |
| 14 | United Arab Emirates | ARE | Oil exporter |

Ten countries are classified as oil exporters (fuel exports > 20% of
merchandise exports); four are net importers for whom oil-revenue stress
concepts are not directly applicable.

---

## 3. Composite Risk Formula

### Right Now Risk Score

```
right_now_risk_score =
    0.35 × fiscal_stress_score
  + 0.25 × reserve_runway_risk
  + 0.20 × social_stability_risk
  + 0.20 × chain_transmission_severity_recent
```

All four components are normalised to **[0, 1]**.  The composite score is
also clamped to [0, 1] after weighting.

| Component | Weight | Range |
|-----------|--------|-------|
| `fiscal_stress_score` | 0.35 | [0, 1] |
| `reserve_runway_risk` | 0.25 | [0, 1] |
| `social_stability_risk` | 0.20 | [0, 1] |
| `chain_transmission_severity_recent` | 0.20 | [0, 1] |

### Fallback and Weight-Rescaling Policy

When a component is unavailable for a country (due to missing data or the
country being absent from a reference file), the remaining available
component weights are rescaled proportionally so they sum to 1.0.  **No
country is silently dropped.**

Every output row records:

- `missing_components` — semicolon-separated list of absent components with
  a bracketed reason (e.g. `chain_transmission_severity_recent [country absent from chain CSV]`)
- `rescaled_weights` — the actual per-row weights used, serialised as a
  JSON-style dict

---

## 4. Component Methodologies

### 4.1 Fiscal Stress Score

**Question answered:** At the current Brent price, how far is this
government below its fiscal breakeven?

The fiscal breakeven is the oil price (USD/bbl) at which a government's
budget is in balance given current spending commitments.  Exporters trading
below their breakeven face fiscal deficit — they must draw down reserves,
cut spending, or borrow.

**Formula (continuous):**

```
fiscal_stress_score = min(1.0, max(0.0,
    (fiscal_breakeven_usd - brent_live_usd) / fiscal_breakeven_usd
))
```

When `brent_live_usd > fiscal_breakeven_usd` the formula produces a
negative value, clamped to 0.0.  When Brent is far below the breakeven the
score approaches 1.0.

Net importers and countries with `fiscal_breakeven_usd = 0` receive a score
of 0.0 (classified Gray — the concept is not applicable).

**Stress classification thresholds (for dashboard display only):**

| Status | Condition |
|--------|-----------|
| Red | Brent < breakeven |
| Amber | breakeven ≤ Brent < breakeven + $15 |
| Green | Brent ≥ breakeven + $15 |
| Gray | Net importer or breakeven = 0 |

**Data source:** `data/reference/fiscal_breakeven.csv`
(IMF Article IV Consultations / Regional Economic Outlook 2023).

---

### 4.2 Reserve Runway Risk

**Question answered:** For governments currently under fiscal stress, how
many months can they sustain current spending from liquid reserves?

**Runway calculation:**

```
reserve_runway_months = liquid_buffer_usd_bn / estimated_monthly_burn_usd_bn
```

`liquid_buffer_usd_bn` is the sum of accessible central-bank FX reserves
and the liquid/deployable portion of sovereign wealth funds.  Long-term
illiquid holdings are excluded (e.g. ADIA, KIA RFFG, QIA equity portfolios,
Iran NDFI under sanctions).

**Risk conversion (linear interpolation):**

```
if reserve_runway_months <= 6:
    reserve_runway_risk = 1.0          # Critical
elif reserve_runway_months >= 36:
    reserve_runway_risk = 0.0          # Comfortable
else:
    reserve_runway_risk = 1.0 - (reserve_runway_months - 6) / (36 - 6)
```

Countries not under fiscal stress (Green or Gray fiscal classification)
receive `reserve_runway_risk = 0.0` — the runway concept is not the
pressing concern when oil is above the breakeven.

**Data source:** `data/reference/swf_reserves.csv`
(SWF annual reports, IMF Article IV Consultations, central bank
statistical bulletins, 2023).

---

### 4.3 Social Stability Risk

**Question answered:** Given food import dependency, fiscal pressure passed
through to households, and inflation volatility, how exposed is the
population to social stress?

**Top-level formula:**

```
social_stability_risk =
    0.5 × food_security_exposure
  + 0.3 × fiscal_stress_score
  + 0.2 × norm(inflation_volatility)
```

#### Food Security Exposure Sub-formula

```
food_security_exposure =
    0.6 × norm(food_imports_pct_merch_imports)
  + 0.4 × norm(cereal_import_dependency)
```

Both `norm()` calls are standard min-max normalisation across all 14 MENA
countries, so the result lies in [0, 1].

If `cereal_import_dependency` is absent for a country, the weight is
redistributed fully to `food_imports_pct_merch_imports` (weight = 1.0), and
a `data_quality_flag` is set on that row.

**Data sources:**
- `food_imports_pct_merch_imports`: World Bank WDI indicator
  `TM.VAL.FOOD.ZS.UN` (~2022)
- `cereal_import_dependency`: FAO FAOSTAT Food Balance Sheets (~2021)

#### Inflation Volatility Normalisation

Inflation volatility is computed as the standard deviation of the World Bank
annual CPI growth rate (`FP.CPI.TOTL.ZG`) over 2000–2024.  Raw values are
then **winsorised at the 5th and 95th percentile** across all 14 countries
before min-max normalisation.

Winsorisation is applied because Lebanon's CPI volatility (σ ≈ 73 percentage
points) would otherwise compress all other countries to near-zero after
min-max scaling.  The winsorised 95th percentile is approximately 35 pp.

```
inflation_vol_winsorized = clip(inflation_vol,
                                p5_across_countries,
                                p95_across_countries)
norm_inflation_vol = (inflation_vol_winsorized - min_winsorized)
                   / (max_winsorized - min_winsorized)
```

**Data source:** `data/processed/world_bank_panel.csv`
(World Bank Open Data API, fetched by `src/data/fetch_world_bank.py`).

---

### 4.4 Chain Transmission Severity

**Question answered:** How strongly do oil price shocks propagate through
each economy's fiscal → spending → subsidy → inflation → employment chain?

#### Stage Definitions

Five sequential stages capture the propagation path:

| Stage | Link | Description |
|-------|------|-------------|
| 1 | Oil price → Fiscal revenue | Degree to which oil revenue tracks price (linkage strength) |
| 2 | Fiscal pressure → Inflation | Subsidy and price pass-through to consumer prices |
| 3 | Inflation → Employment | Labour-market vulnerability to price shocks |
| 4 | Employment / wages → Household consumption | Income transmission to consumption |
| 5 | Consumption contraction → GDP growth | Final feedback to economic output |

**Severity formula:**

```
stage_mean                  = mean(stage1, stage2, stage3, stage4, stage5)
chain_transmission_severity = min(1.0, stage_mean × amplification_factor)
```

`amplification_factor < 1.0` reflects dampening effects (SWF buffers,
economic diversification).  `amplification_factor > 1.0` reflects
amplifying factors (institutional weakness, embedded inflation, conflict).

#### OLS Regression Approach (--fit-ols)

When run with `--fit-ols`, the pipeline fits empirical stage scores from the
World Bank panel using simple OLS regression:

```
y_t = α + β × brent_pct_change_{t-1} + ε

empirical_stage_score = min(1.0, |β|)
```

Proxy mapping from World Bank indicators:

| Stage | Column | World Bank indicator | Proxy |
|-------|--------|---------------------|-------|
| Stage 1 (`stage1_oil_fiscal`) | `empirical_stage1` | `NY_GDP_PETR_RT_ZS` | YoY pp change in oil rents % GDP |
| Stage 2 (`stage2_fiscal_inflation`) | `empirical_stage2` | `FP_CPI_TOTL_ZG` | CPI inflation rate (%) |
| Stage 5 (`stage5_consumption_growth`) | `empirical_stage5` | `NY_GDP_MKTP_CD` | Nominal GDP growth (%) |
| Stage 3 (`stage3_inflation_employment`) | — | No proxy available | Expert estimate retained |
| Stage 4 (`stage4_employment_consumption`) | — | No proxy available | Expert estimate retained |

A minimum of 5 non-null paired observations is required for OLS fitting.
Where data are insufficient, the expert estimate is kept.

The model **blends** empirical and expert estimates: empirical scores are
used where available; expert estimates fill the gaps.

#### Expert Estimate Calibration

Expert stage scores and amplification factors are calibrated to:
- IMF Article IV Consultations 2023
- IMF Regional Economic Outlook MENA, October 2023
- Coady et al. (IMF, 2015) — energy-subsidy pass-through estimates
- World Bank Development Indicators 2022

**Data source:** `data/reference/chain_transmission.csv`

---

## 5. Normalization Methods

| Component | Method | Outlier handling |
|-----------|--------|-----------------|
| `fiscal_stress_score` | Continuous breakeven formula | Clamped to [0, 1]; net importers set to 0.0 |
| `reserve_runway_risk` | Linear interpolation, floor 6 mo, ceiling 36 mo | Gray countries set to 0.0 |
| `food_imports_pct_merch_imports` | Min-max across 14 countries | None |
| `cereal_import_dependency` | Min-max across 14 countries | None |
| `inflation_volatility` | p5–p95 winsorisation, then min-max | Lebanon σ ≈ 73 pp capped at p95 ≈ 35 pp |
| `chain_transmission_severity_recent` | Min-max across 14 countries (3-year rolling mean) | None |

---

## 6. Data Sources

| File | Key columns | Primary source | Coverage | Confidence |
|------|-------------|----------------|----------|------------|
| `data/reference/fiscal_breakeven.csv` | `fiscal_breakeven_usd`, `country_type` | IMF Article IV / Regional Economic Outlook 2023 | 14 countries | Low for IRN, LBN, LBY |
| `data/reference/swf_reserves.csv` | `liquid_buffer_usd_bn`, `estimated_monthly_burn_usd_bn` | SWF annual reports, IMF Article IV, central bank bulletins 2023 | 14 countries | Low for IRN, LBN, LBY |
| `data/reference/food_security.csv` | `food_imports_pct_merch_imports`, `cereal_import_dependency` | World Bank TM.VAL.FOOD.ZS.UN; FAO FAOSTAT | ~2021–2022 | Estimated for all rows |
| `data/processed/world_bank_panel.csv` | `FP_CPI_TOTL_ZG`, `NY_GDP_PETR_RT_ZS`, `NY_GDP_MKTP_CD` | World Bank Open Data API | 2000–2024 | High |
| `data/reference/chain_transmission.csv` | `stage1`–`stage5`, `amplification_factor` | Expert estimates + OLS from WB panel | 2024 snapshot | Varies by country |
| Live Brent price | `brent_live_usd` | Yahoo Finance ticker `BZ=F` via yfinance | Real-time | High; unavailable outside market hours |
| `data/reference/imf_weo_2020_outcomes.csv` | `gdp_growth_2020_pct`, `fiscal_balance_2020_pct_gdp`, `outcome_severity_rank` | IMF World Economic Outlook April 2021 Table A7; WB GEP Jan 2021 | 2020 outcomes | Low for IRN, LBN, LBY |
| `data/reference/imf_wb_benchmarks.csv` | `imf_fm_risk_tier`, `wb_mpo_status` | IMF Fiscal Monitor Oct 2023; WB Macro Poverty Outlook Fall 2023 | 2023 benchmarks | Low for IRN, LBN, LBY |

All reference assumptions are traceable to a stable source ID in
`data/reference/source_registry.csv` (19 source IDs across 6 reference
files).  Every reference CSV row carries `source_id_primary` and
`source_id_secondary` columns that resolve to registry entries recording
organisation, publication year, retrieval date, URL, and confidence tier.

---

## 7. Validation

### 7.1 2020 Oil Crash Retrospective

The retrospective tests whether the model's **2019 pre-crisis composite
score** predicted the severity of actual 2020 economic outcomes.

**Setup:**

- **Pre-crisis snapshot year:** 2019 (annual average Brent = $64.37/bbl)
- **Crisis year:** 2020 (annual average Brent = $41.96/bbl; April low ~$18/bbl)
- **Outcome measure:** IMF WEO April 2021 GDP growth rate, ranked 1 (worst,
  Libya −59.7%) to 14 (best, Egypt +3.6%)
- **Model input:** `right_now_risk_score` from the 2019 historical index
  snapshot, ranked 1 (highest predicted risk) to 14 (lowest)
- **Correlation metric:** Spearman rank correlation ρ

**Results:**

| Metric | Value |
|--------|-------|
| Spearman ρ | 0.108 (Weak) |
| p-value | 0.72 |
| Mean absolute rank error | 4.4 positions |
| Top-5 correct predictions | 2 / 5 |

**Known structural misses:**

The low Spearman ρ reflects structural events that the model could not
predict from 2019 fiscal data alone:

| Country | Model rank 2019 | Outcome rank 2020 | Rank error | Structural explanation |
|---------|----------------|-------------------|------------|----------------------|
| Iran | 1 | 13 | 12 | Sanctions collapsed oil exports; GDP contraction was already priced in by 2019; the COVID shock added little marginal stress vs the sanctions baseline |
| Morocco | 14 | 5 | 9 | Severe drought compounded COVID; fiscal breakeven and reserves were among the strongest in the sample |
| Kuwait | 10 | 4 | 6 | Fiscal hit was severe (−33% fiscal balance) but KIA SWF buffered the outcome; model underweighted Kuwait's oil-revenue dependency |
| Algeria | 3 | 9 | 6 | FX reserve drawdown was manageable; hydrocarbons still funded the budget; outcome less severe than the pre-crisis fiscal position implied |
| Bahrain | 2 | 7 | 5 | GCC financial support package sustained the external position; model did not capture the backstop guarantee |
| Lebanon | 7 | 2 | 5 | Financial sector collapse pre-dated COVID and was driven by a banking crisis and political deadlock, not the oil shock |

**Interpretation:** The 2020 shock was an extreme, multi-factor event
(COVID demand collapse, OPEC+ output cuts, geopolitical blockades, pre-existing
financial crises) that combined structural vulnerability with idiosyncratic
shocks the model is not designed to forecast.  Iran and Lebanon in particular
were already in crisis by 2019 for non-oil reasons.

---

### 7.2 IMF / WB Cross-Validation

The cross-validation compares the model's current composite score against
two independent risk-tier classifications from 2023.

**Setup:**

- **Model tiers:** `right_now_risk_score` divided into three tertile-based
  tiers (33rd and 67th percentile thresholds → Low / Medium / High)
- **IMF benchmark:** IMF Fiscal Monitor October 2023 — Low / Medium / High
  fiscal vulnerability
- **WB benchmark:** World Bank Macro Poverty Outlook Fall 2023 — Stable /
  Watch / Stressed

**Ordinal encoding:**

| IMF Fiscal Monitor | Ordinal | WB Macro Poverty Outlook | Ordinal |
|--------------------|---------|--------------------------|---------|
| Low | 1 | Stable | 1 |
| Medium | 2 | Watch | 2 |
| High | 3 | Stressed | 3 |

**Results:**

| Metric | Value |
|--------|-------|
| Kendall τ-b vs IMF FM | -0.680 |
| Kendall τ-b vs WB MPO | -0.621 |
| Exact tier match vs IMF FM | 9 / 14 (64%) |
| Top-5 model vs IMF High | 5 / 5 overlap |

**Sign convention:** `model_rank = 1` denotes the highest-risk country;
`imf_fm_ordinal = 3` (High) also denotes the highest-risk country.  The two
scales point in **opposite numeric directions**, so a negative Kendall τ
indicates *agreement* — countries the model ranks highest are the same
countries the IMF classifies as High risk.  The five countries ranked 1–5 by
the model (Iran, Lebanon, Bahrain, Algeria, Iraq) all carry IMF High tier
classification.

**Tier-level divergences (9 countries agree exactly with IMF; 5 diverge):**

| Country | Model Tier | IMF Tier | WB Stress | Note |
|---------|-----------|----------|-----------|------|
| Libya | Medium | High | Stressed | Oil blockade history and dual-government raise confidence concerns; model may under-weight geopolitical instability |
| Jordan | Medium | High | Watch | IMF FM accounts for large debt / GDP and Syrian refugee fiscal burden; model scores Jordan lower on structural grounds |
| Egypt | Medium | High | Stressed | IMF and WB reflect active SBA / EFF programs and high external financing needs; model score is lower due to partial diversification |
| Saudi Arabia | Low | Medium | Watch | IMF accounts for Vision 2030 transition risk; model treats current reserves and low breakeven as dominant |
| Morocco | Low | Medium | Watch | IMF weight on drought and earthquake fiscal impact; model shows strongest reserves among non-GCC countries |

---

## 8. Uncertainty Bands

Each reference CSV carries three uncertainty columns (low / base / high) per
key assumption, enabling stress and optimistic scenario testing:

| File | Columns |
|------|---------|
| `fiscal_breakeven.csv` | `breakeven_low_usd`, `breakeven_base_usd`, `breakeven_high_usd` |
| `swf_reserves.csv` | `liquid_buffer_low_usd_bn`, `liquid_buffer_base_usd_bn`, `liquid_buffer_high_usd_bn` |
| `swf_reserves.csv` | `monthly_burn_low_usd_bn`, `monthly_burn_base_usd_bn`, `monthly_burn_high_usd_bn` |
| `food_security.csv` | `cereal_dependency_low`, `cereal_dependency_base`, `cereal_dependency_high` |

The live model uses only the `base` columns.  The `low` / `high` columns are
consumed by the backtesting scenario framework (`stress` and `optimistic` scenarios).

Band widths are calibrated to source confidence tier:

| Tier | Breakeven | Liquid buffer | Burn rate | Cereal dep. |
|------|-----------|---------------|-----------|-------------|
| High | Derived from notes | ±15% | ±20% | ±5 pp |
| Medium | Derived from notes | ±25% | ±30% | ±10 pp |
| Low | Derived from notes | ±40% | ±50% | ±20 pp |

Countries with `confidence = ""` (net importers: Jordan, Lebanon, Morocco)
have null uncertainty bands — fiscal stress is 0 for them regardless.

---

## 9. Limitations

### Static reference data

`fiscal_breakeven.csv`, `swf_reserves.csv`, and `food_security.csv` are
point-in-time estimates from 2023 reference publications.  They are not
automatically updated.  The World Bank panel (`world_bank_panel.csv`) refreshes
weekly via GitHub Actions; the reference CSVs require manual revision when
new IMF Article IV or SWF annual-report data are published.

### First-order linear model

The chain transmission model is a linear approximation.  It does not capture:
- Exchange-rate feedback loops
- Monetary policy response (rate changes, capital controls)
- Non-linear subsidy reform dynamics (one-off removal vs. gradual phase-out)
- Second-round effects (inflation expectations, wage-price spirals)

### Lebanon inflation outlier handling

Lebanon's CPI standard deviation over 2000–2024 is approximately 73 percentage
points — roughly six times the next-highest country.  Inflation volatility is
winsorised at the 95th percentile (~35 pp) before min-max normalisation to
prevent Lebanon from compressing all other countries to near-zero.  This means
Lebanon's inflation contribution is capped, and the model under-represents
the severity of its inflation crisis.

### Iran, Libya, Lebanon — low confidence data

All three countries carry `confidence = "low"` across fiscal breakeven,
reserve, and benchmark CSVs.  For Iran, sanctions severely restrict data
access; official statistics are proxy-estimated.  For Libya, dual-government
and oil-blockade history make revenue and expenditure figures unreliable.  For
Lebanon, financial-sector collapse and banking controls make FX and fiscal
data highly uncertain.  All three countries should be interpreted with caution.

### Chain transmission static snapshot

`chain_transmission.csv` is generated as a single 2024 snapshot.  The
historical index (2015–2024) uses this static snapshot for all years prior to
2024, meaning the chain component does not vary historically.  The weight-
rescaling fallback handles years where this component is NaN, but the
historical trend analysis reflects primarily the fiscal and reserve components.

---

## 10. Reproducibility

### Full pipeline

All outputs can be regenerated from scratch with a single command from the
project root:

```bash
python reproduce_all.py
```

This executes 10 steps in dependency order, timestamps each step, and
prints a PASS / FAIL summary.  The script continues past any failed step
and exits with code 1 if any step fails.

| Step | Module | Purpose |
|------|--------|---------|
| 1 | `src.data.fetch_world_bank` | Download World Bank panel (2000–2024) |
| 2 | `src.data.clean_world_bank` | Clean and validate panel |
| 3 | `src.model.vulnerability_index` | OCVI country rankings |
| 4 | `src.model.chain_transmission --fit-ols` | Chain severity + OLS fitting |
| 5 | `src.model.historical_index` | 2015–2024 risk panel (140 rows) |
| 6 | `src.model.right_now_risk` | Right Now Risk composite (live Brent) |
| 7 | `src.model.retrospective` | 2020 oil crash retrospective |
| 8 | `src.model.cross_validation` | IMF / WB cross-validation |
| 9 | `src.model.sensitivity` | One-at-a-time weight sensitivity |
| 10 | `src.data.validate_reference --strict` | Reference data integrity check |

### Reference data integrity check

The validator runs 47 automated checks across all six reference CSVs:

```bash
python -m src.data.validate_reference          # summary only
python -m src.data.validate_reference --strict  # exits 1 on any failure
```

Checks include: unique `country_code_a3` per file, referential integrity
(every `source_id_primary` / `source_id_secondary` resolves to a
`source_registry.csv` row), confidence-tier enum validity, monotonic
uncertainty bands (`low ≤ base ≤ high`), non-null base values for all
exporter rows, and domain-specific checks (e.g. `outcome_severity_rank`
covers 1–14 with no duplicates).

The GitHub Actions weekly refresh workflow (`refresh_data.yml`) runs
`validate_reference --strict` as the final gate — if validation fails,
no data is committed.
