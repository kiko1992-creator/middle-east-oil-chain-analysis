"""
Fetch IMF World Economic Outlook April 2026 data for 14 MENA countries.

Downloads five macroeconomic indicators for years 2015-2026 from the IMF
External Datamapper API and writes a long-format panel CSV.

Indicators
----------
  NGDP_RPCH    GDP growth rate (%)
  PCPIPCH      Inflation rate (%)
  GGXCNL_NGDP  Fiscal balance % GDP
  BCA_NGDPD    Current account % GDP
  LUR          Unemployment rate (%)

Coverage
--------
  14 MENA economies: BHR DZA EGY IRQ IRN JOR KWT LBN LBY MAR OMN QAT SAU ARE
  Years 2015-2026; 2025-2026 marked is_estimate=True (IMF estimates/projections)

API notes
---------
  The IMF datamapper API (https://www.imf.org/external/datamapper/api/v1/)
  returns all 229 countries for each indicator regardless of the URL path.
  Country filtering is applied locally after the response is received.
  The ?periods= query parameter restricts the year columns returned.

Resilience
----------
  This module is designed for graceful degradation:
  - Missing Python dependencies: logs a warning, writes empty schema CSV, exits 0.
  - Network errors: retries once, then logs warning and skips that indicator.
  - Non-200 HTTP responses: logs warning, returns empty dict for that indicator.
  - Malformed API response: explicit guards at each nesting level log missing keys.
  - Complete fetch failure: writes an empty panel CSV with correct schema columns
    so downstream code never receives a FileNotFoundError.
  Downstream modules fall back to World Bank panel data when this file is absent.

On API failure: logs a warning and returns an empty DataFrame.  Never raises.

Usage (from project root)
--------------------------
    python -m src.data.fetch_imf_weo
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — missing packages cause graceful degradation, not failure
# ---------------------------------------------------------------------------

_pandas_imported: bool
_requests_imported: bool
_pandas_error: str = ""
_requests_error: str = ""

try:
    import pandas as pd
    _pandas_imported = True
except ImportError as _exc:
    _pandas_imported = False
    _pandas_error = str(_exc)

try:
    import requests
    _requests_imported = True
except ImportError as _exc:
    _requests_imported = False
    _requests_error = str(_exc)

_DEPS_OK: bool = _pandas_imported and _requests_imported

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_API_BASE: str = "https://www.imf.org/external/datamapper/api/v1"

_INDICATORS: dict[str, str] = {
    "NGDP_RPCH":    "GDP growth rate (%)",
    "PCPIPCH":      "Inflation rate (%)",
    "GGXCNL_NGDP":  "Fiscal balance % GDP",
    "BCA_NGDPD":    "Current account % GDP",
    "LUR":          "Unemployment rate (%)",
}

# ISO 3166-1 alpha-3 codes used by the IMF datamapper API.
_MENA_ISO3: dict[str, str] = {
    "BHR": "Bahrain",
    "DZA": "Algeria",
    "EGY": "Egypt",
    "IRQ": "Iraq",
    "IRN": "Iran",
    "JOR": "Jordan",
    "KWT": "Kuwait",
    "LBN": "Lebanon",
    "LBY": "Libya",
    "MAR": "Morocco",
    "OMN": "Oman",
    "QAT": "Qatar",
    "SAU": "Saudi Arabia",
    "ARE": "United Arab Emirates",
}

_YEARS: list[int] = list(range(2015, 2027))    # 2015-2026 inclusive
_ESTIMATE_FROM_YEAR: int = 2025                  # 2025+ are IMF estimates/projections
_SOURCE_ID: str = "IMF_WEO_2026"
_REQUEST_TIMEOUT: int = 30                       # seconds per indicator request

_RAW_DIR:     Path = Path("data/raw/imf_weo")
_OUTPUT_PATH: Path = Path("data/processed/imf_weo_panel.csv")

# Minimum countries that must have data for the fetch to log INFO (vs WARNING)
_MIN_COUNTRIES_EXPECTED: int = 10

# Schema columns for the output panel CSV (including schema_valid provenance flag).
_PANEL_COLS: list[str] = [
    "country_code_a3", "year", "indicator", "value",
    "is_estimate", "source_id", "schema_valid",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_empty_panel(output_path: Path) -> "pd.DataFrame":
    """Write an empty panel CSV with the correct schema columns.

    Ensures downstream code never gets a FileNotFoundError even when the
    IMF API is unavailable or dependencies are missing.

    Args:
        output_path: Destination path for the empty CSV.

    Returns:
        Empty DataFrame with correct columns, or None if pandas is unavailable.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if _pandas_imported:
        df: pd.DataFrame = pd.DataFrame(columns=_PANEL_COLS)
        df.to_csv(output_path, index=False)
        return df
    # pandas unavailable — write header-only CSV manually
    output_path.write_text(",".join(_PANEL_COLS) + "\n", encoding="utf-8")
    return None  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def fetch_indicator(indicator: str) -> dict[str, dict[str, float]]:
    """Fetch one WEO indicator for years 2015-2026; return MENA-filtered data.

    The API returns all 229 countries regardless of the URL path; this
    function filters the response to only the 14 MENA ISO-3 codes defined
    in *_MENA_ISO3*.

    Retries once on any network error before giving up.  Applies explicit key
    guards at every nesting level of the API response and logs missing keys.

    Args:
        indicator: WEO indicator code (e.g. ``'PCPIPCH'``).

    Returns:
        ``{iso3: {year_str: value}}`` for MENA countries that have data.
        Empty dict on any network or parse error.
    """
    if not _requests_imported:
        log.warning("Skipping %s — 'requests' package not available.", indicator)
        return {}

    periods_param = ",".join(str(y) for y in _YEARS)
    url = f"{_API_BASE}/{indicator}?periods={periods_param}"

    response = None
    for attempt in (1, 2):
        try:
            response = requests.get(url, timeout=_REQUEST_TIMEOUT)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as exc:
            if attempt == 2:
                log.warning(
                    "Network error fetching %s after retry: %s", indicator, exc
                )
                return {}
            log.warning(
                "Network error fetching %s (attempt %d): %s — retrying...",
                indicator, attempt, exc,
            )
        except Exception as exc:
            log.warning("Unexpected error fetching %s: %s", indicator, exc)
            return {}

    if response is None:
        return {}

    # --- Parse JSON ---
    try:
        payload = response.json()
    except ValueError as exc:
        log.warning("JSON parse error for %s: %s", indicator, exc)
        return {}

    # --- Guard: "values" key must exist at top level ---
    if "values" not in payload:
        log.warning(
            "%s: 'values' key missing from API response (top-level keys: %s).",
            indicator, list(payload.keys()),
        )
        return {}

    values_block = payload["values"]

    # --- Guard: indicator key must exist inside "values" ---
    if indicator not in values_block:
        log.warning(
            "%s: indicator key absent from 'values' block "
            "(first 10 keys shown: %s).",
            indicator, list(values_block.keys())[:10],
        )
        return {}

    all_country_data: dict[str, object] = values_block[indicator]
    mena_data: dict[str, dict[str, float]] = {}

    for iso3 in _MENA_ISO3:
        # --- Guard: country key must exist ---
        if iso3 not in all_country_data:
            log.warning(
                "%s: country %s absent from indicator block — skipping.",
                indicator, iso3,
            )
            continue
        country_block = all_country_data[iso3]
        # --- Guard: country block must be a dict (year_str → value) ---
        if not isinstance(country_block, dict):
            log.warning(
                "%s: country %s data has unexpected type %s — skipping.",
                indicator, iso3, type(country_block).__name__,
            )
            continue
        # Year-level validation deferred to build_weo_panel where
        # bad (year_str, value) pairs are skipped with a warning.
        mena_data[iso3] = country_block  # type: ignore[assignment]

    n_countries = len(mena_data)
    if n_countries < _MIN_COUNTRIES_EXPECTED:
        log.warning(
            "%s: only %d/%d MENA countries found in API response.",
            indicator, n_countries, len(_MENA_ISO3),
        )
    else:
        log.info("%s: %d MENA countries fetched.", indicator, n_countries)

    return mena_data


def save_raw(indicator: str, data: dict[str, dict[str, float]]) -> None:
    """Save MENA-filtered indicator data as JSON to data/raw/imf_weo/.

    Args:
        indicator: WEO indicator code.
        data:      MENA-filtered ``{iso3: {year_str: value}}`` dict.
    """
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RAW_DIR / f"{indicator}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    log.info("Raw data saved: %s", out_path)


# ---------------------------------------------------------------------------
# Panel assembly
# ---------------------------------------------------------------------------

def build_weo_panel(
    raw: dict[str, dict[str, dict[str, float]]],
) -> "pd.DataFrame":
    """Convert raw per-indicator dicts to a long-format panel DataFrame.

    Each output row carries a ``schema_valid`` column: ``True`` when the
    year and value parsed without error; rows that cannot be parsed are
    skipped with a warning log rather than appearing as invalid records.

    Args:
        raw: ``{indicator: {iso3: {year_str: value}}}``

    Returns:
        Long-format DataFrame with columns:
        ``country_code_a3``, ``year``, ``indicator``, ``value``,
        ``is_estimate``, ``source_id``, ``schema_valid``.
        Empty DataFrame (with those columns) when *raw* yields no records.
    """
    records: list[dict] = []
    for indicator, country_data in raw.items():
        for iso3, year_data in country_data.items():
            if not isinstance(year_data, dict):
                log.warning(
                    "build_weo_panel: %s/%s has unexpected type %s — skipping.",
                    indicator, iso3, type(year_data).__name__,
                )
                continue
            for year_str, value in year_data.items():
                try:
                    year = int(year_str)
                    val  = float(value)
                except (ValueError, TypeError) as exc:
                    log.warning(
                        "build_weo_panel: skipping %s/%s year=%s value=%r (%s)",
                        indicator, iso3, year_str, value, exc,
                    )
                    continue
                records.append({
                    "country_code_a3": iso3,
                    "year":            year,
                    "indicator":       indicator,
                    "value":           val,
                    "is_estimate":     year >= _ESTIMATE_FROM_YEAR,
                    "source_id":       _SOURCE_ID,
                    "schema_valid":    True,
                })

    if not records:
        log.warning("No WEO data records to assemble — panel will be empty.")
        return pd.DataFrame(columns=_PANEL_COLS)

    df = (
        pd.DataFrame(records)
        .sort_values(["indicator", "country_code_a3", "year"])
        .reset_index(drop=True)
    )
    return df


# ---------------------------------------------------------------------------
# End-to-end orchestrator
# ---------------------------------------------------------------------------

def run_fetch_weo(
    output_path: Path = _OUTPUT_PATH,
    raw_dir:     Path = _RAW_DIR,
) -> "pd.DataFrame":
    """Fetch all WEO indicators, save raw JSON files, and write the panel CSV.

    Designed for graceful degradation: if the IMF API is unavailable or
    dependencies are missing, writes an empty panel CSV with correct schema
    columns so downstream code never receives a FileNotFoundError.

    On partial failure (some indicators unavailable), the panel is built
    from whatever data was successfully fetched; rows for missing
    indicators/countries are simply absent.

    Args:
        output_path: Destination for ``imf_weo_panel.csv``.
        raw_dir:     Directory for per-indicator raw JSON files.

    Returns:
        Assembled panel DataFrame (also written to *output_path*).
        Returns an empty DataFrame on any failure.
    """
    log.info("IMF WEO fetch starting — failure will not block pipeline")

    output_path = Path(output_path)

    if not _DEPS_OK:
        if not _pandas_imported:
            log.warning("'pandas' not installed (%s) — writing empty WEO panel.", _pandas_error)
        if not _requests_imported:
            log.warning("'requests' not installed (%s) — writing empty WEO panel.", _requests_error)
        empty = _write_empty_panel(output_path)
        return empty if empty is not None else pd.DataFrame(columns=_PANEL_COLS)

    try:
        raw: dict[str, dict[str, dict[str, float]]] = {}

        for indicator in _INDICATORS:
            log.info("Fetching %s — %s", indicator, _INDICATORS[indicator])
            data = fetch_indicator(indicator)
            if data:
                save_raw(indicator, data)
            else:
                log.warning("No data returned for %s — skipping raw save.", indicator)
            raw[indicator] = data

        panel = build_weo_panel(raw)

        if panel.empty:
            log.warning("WEO panel assembled empty — writing schema-only CSV.")
            _write_empty_panel(output_path)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            panel.to_csv(output_path, index=False)

            n_countries   = int(panel["country_code_a3"].nunique())
            n_records     = len(panel)
            years_present = sorted(panel["year"].unique())
            year_range    = (
                f"{years_present[0]}–{years_present[-1]}"
                if years_present else "none"
            )
            log.info(
                "IMF WEO fetch complete: %d records, %d countries, years %s → %s",
                n_records, n_countries, year_range, output_path,
            )

        return panel

    except Exception as exc:
        log.warning(
            "Unexpected error in WEO fetch — writing empty panel: %s", exc
        )
        empty = _write_empty_panel(output_path)
        return empty if empty is not None else pd.DataFrame(columns=_PANEL_COLS)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.data.fetch_imf_weo",
        description=(
            "Fetch IMF World Economic Outlook April 2026 data for 14 MENA countries.\n"
            "Saves raw JSON per indicator to data/raw/imf_weo/ and a long-format\n"
            "panel CSV to data/processed/imf_weo_panel.csv.\n\n"
            "Pipeline-safe: always exits 0 even if the IMF API is unavailable."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output", metavar="PATH", default=str(_OUTPUT_PATH),
        help=f"Destination for the panel CSV (default: {_OUTPUT_PATH}).",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run the WEO fetch pipeline.

    Always returns 0 — the pipeline must not fail when the IMF API is down.
    """
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not _pandas_imported:
        log.warning("'pandas' not installed (%s) — cannot run WEO fetch.", _pandas_error)
    if not _requests_imported:
        log.warning("'requests' not installed (%s) — cannot run WEO fetch.", _requests_error)

    panel = run_fetch_weo(output_path=Path(args.output))

    if _pandas_imported and panel is not None and not panel.empty:
        n_countries = int(panel["country_code_a3"].nunique())
        n_records   = len(panel)
        years_present = sorted(panel["year"].unique())
        year_range = (
            f"{years_present[0]}–{years_present[-1]}"
            if years_present else "none"
        )
        print(
            f"IMF WEO panel: {n_records} records, "
            f"{n_countries}/14 MENA countries, "
            f"years {year_range}."
        )
    else:
        print("IMF WEO panel: 0 records (empty panel written — downstream uses WB fallback).")
        log.warning("WEO panel is empty — downstream models will use WB panel fallback.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
