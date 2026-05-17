# Data Dictionary

## Production & Trade

| Field | Type | Unit | Description |
|---|---|---|---|
| `country` | string | — | Producing/exporting country name |
| `date` | date | YYYY-MM-DD | Observation date |
| `volume_mbd` | float | million barrels/day | Crude oil volume |
| `commodity` | string | — | crude_oil / natural_gas / refined_products |
| `route` | string | — | Named shipping route (e.g. Strait of Hormuz) |

## Price Benchmarks

| Field | Type | Unit | Description |
|---|---|---|---|
| `brent_usd` | float | USD/bbl | Brent Crude spot price |
| `wti_usd` | float | USD/bbl | WTI spot price |
| `dubai_usd` | float | USD/bbl | Dubai/Oman benchmark |

## Geopolitical Events

| Field | Type | Unit | Description |
|---|---|---|---|
| `event_id` | string | — | Unique event identifier |
| `date` | date | YYYY-MM-DD | Event date |
| `country` | string | — | Affected country |
| `severity` | int | 1–5 | Severity score (1 = low, 5 = critical) |
| `description` | string | — | Short event description |
