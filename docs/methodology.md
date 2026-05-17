# Methodology

## 1. Data Sources

- **Production & exports**: OPEC Monthly Oil Market Report, EIA International Energy Statistics
- **Prices**: Bloomberg commodity feeds (Brent, WTI, Dubai/Oman)
- **Geopolitical events**: ACLED, Global Conflict Tracker
- **Geospatial**: Natural Earth, UN OCHA shapefiles

## 2. Supply Chain Analysis

Net export volumes are derived by subtracting domestic consumption estimates from reported production.
Chokepoint analysis flags transit routes carrying > 20 % of total modelled shipment volume.

## 3. Price Analysis

Rolling 30-day averages smooth short-term noise.
Annualised volatility uses daily log-returns scaled by √252.
Correlations between regional benchmarks are computed on a trailing 12-month window.

## 4. Geopolitical Risk Scoring

Events are scored 1–5 on severity (supply disruption potential).
Normalised risk scores are merged to price series using an as-of join so that each price
observation is paired with the most recent risk event.

## 5. Outputs

All figures are exported to `outputs/figures/` in both PNG and interactive HTML formats.
Summary tables are written to `outputs/tables/` as CSV.
