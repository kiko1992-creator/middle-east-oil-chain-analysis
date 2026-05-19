# Backtesting Plan — Right Now Risk Model

## Purpose

The backtest answers one question: **does the Right Now Risk score behave plausibly
across materially different oil-price environments?**

This is a *conditional* backtest — not a true historical simulation.  The fiscal
breakeven estimates, reserve figures, and food security data are fixed at their
2023 reference values.  Only the annual-average Brent crude price and the chain
transmission severity (where available for the target year) vary.

The value is not precision: it is face validity.  When Brent was $38 in 2020,
high-breakeven exporters (Iraq, Algeria, Libya) should score markedly higher
than when Brent was $111 in 2022.  If the model inverts that relationship, the
weights or formula are wrong.

---

## Target Periods

| Period | Avg Brent (USD) | Event | What to test |
|--------|----------------|-------|--------------|
| **2008** | ~$97 | Pre-GFC spike then collapse | High fiscal stress mid-year; scores near peak |
| **2015–2016** | ~$52 / ~$44 | Saudi-led supply glut | Persistent sub-breakeven for most exporters; Iraq/Libya near max |
| **2020** | ~$42 | COVID demand collapse | Scores near 2015–16 floor; Kuwait/Qatar buffered by reserves |
| **2022** | ~$101 | Post-Ukraine rally | Most exporters at or above breakeven; fiscal stress near 0 |

These four windows cover a full price cycle (collapse → floor → recovery →
spike) and provide the minimum needed to verify rank stability.

---

## Run Configuration

```python
from src.model.backtest import run_backtest_range, summarize_rank_stability, export_backtest_outputs

# Full cycle (uses annual-average Brent from yfinance / fallback table)
df = run_backtest_range(2008, 2022, scenario="base")

# Stress scenario: high breakeven + low reserve buffer + high burn
df_stress = run_backtest_range(2015, 2016, scenario="stress")

# Optimistic: low breakeven + high buffer + low burn
df_opt = run_backtest_range(2020, 2020, scenario="optimistic")
```

Each scenario produces one row per country per year with metadata columns:
`snapshot_year`, `scenario`, `method_version`, `historical_brent_usd`,
`missing_components_count`.

---

## Success Criteria

### 1. Rank plausibility (required)

In the 2015–2016 sub-$55 Brent window:
- **Iraq** (breakeven ~$130), **Algeria** (breakeven ~$135), **Libya** (breakeven ~$115)
  must rank in the **top 4** by `right_now_risk_score`.
- **Kuwait** (breakeven ~$50, large buffer) must rank **outside the top 3**.
- **Qatar** (breakeven ~$55, large buffer) must rank **outside the top 3**.

In the 2022 $101 Brent window:
- **Iraq**, **Algeria**, **Libya** scores must be materially lower than their 2015–2016
  scores (score drop ≥ 0.10).
- The overall score distribution must compress toward 0 for exporters.

### 2. Monotonicity (required)

For every oil exporter, the mean 2015–2016 score must exceed the mean 2022 score.
Net importers are exempt (fiscal stress formula returns 0 when breakeven is null).

### 3. Scenario ordering (required)

For the same year and country:  `score_stress ≥ score_base ≥ score_optimistic`.
A single violation (due to numerical precision) is acceptable; systematic inversion
is a model defect.

### 4. Missing-component coverage (required)

In years before chain data begins (2000–1999), `missing_components_count` must be 1
for all exporters (chain absent).  The weight-rescaling logic must keep all 14
countries in the output.

### 5. Rank stability (informational)

`summarize_rank_stability` should show:
- `std_rank ≤ 3` for Iraq, Algeria (high-breakeven countries should be consistently
  near the top regardless of price level above their breakeven).
- `std_rank` largest for borderline countries (Bahrain, Egypt) where price proximity
  to breakeven produces high rank sensitivity.

---

## Output Files

| File | Contents |
|------|----------|
| `outputs/tables/backtest/backtest_{year}_base.csv` | Per-country scores for one year, base scenario |
| `outputs/tables/backtest/backtest_rank_stability.csv` | Cross-year rank statistics per country |

---

## Known Limitations

1. **No historical reference data** — breakeven and reserve figures are frozen at
   2023.  A country that nationalized its SWF in 2015 (Libya) appears with its 2023
   reserve figure.  This is intentional and documented.

2. **Chain data coverage** — `chain_transmission.csv` covers 2000–2024.  For years
   outside that range, chain severity is NaN and weights are rescaled.

3. **Annual-average Brent** — the backtest uses annual averages, not month-by-month
   prices.  Intra-year volatility (e.g., the 2008 $147→$32 swing) is invisible.

4. **Interpretation** — a high backtest score in 2015 does not mean the country was
   actually in fiscal crisis in 2015; it means the model *would have flagged it* given
   those prices and the 2023 reference estimates.

---

## Status

| Item | Status |
|------|--------|
| `src/model/backtest.py` | Complete |
| `data/reference/source_registry.csv` | Complete |
| Uncertainty bands in reference CSVs | Complete |
| Validation script | Complete |
| docs/backtesting_plan.md (this file) | Complete |
| Dashboard integration of backtest tab | Planned (Sprint 2) |
| True historical simulation (time-varying reference data) | Out of scope |
