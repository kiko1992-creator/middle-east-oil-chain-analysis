"""
Fetch all 7 World Bank indicators for the 14 MENA countries.

Reads country and indicator definitions from ``config/countries.yaml`` and
``config/indicators.yaml`` respectively.  Makes one API call per indicator
(batching all 14 country codes in a single request), handles pagination
automatically, and writes one JSON file per indicator under
``data/raw/world_bank/``.

Each output file is a JSON envelope::

    {
      "meta":    { indicator, countries, date_range, fetched_at, ... },
      "records": [ <raw World Bank API records> ]
    }

Usage (run from project root)::

    python -m src.data.fetch_world_bank
    python -m src.data.fetch_world_bank --dry-run
    python -m src.data.fetch_world_bank --indicator NY.GDP.MKTP.CD
    python -m src.data.fetch_world_bank --output-dir /tmp/wb_raw

Exit codes:
    0  all indicators fetched and saved successfully
    1  one or more indicators failed after retries
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (overridden by config/indicators.yaml fetch_defaults)
# ---------------------------------------------------------------------------

BASE_URL = "https://api.worldbank.org/v2"

_CONFIG_COUNTRIES: Path = Path("config/countries.yaml")
_CONFIG_INDICATORS: Path = Path("config/indicators.yaml")
_OUTPUT_DIR: Path = Path("data/raw/world_bank")

_TIMEOUT: int = 30       # per-request socket timeout (seconds)
_REQUEST_DELAY: float = 0.5   # polite pause between successive indicator calls
_MAX_RETRIES: int = 3
_RETRY_BACKOFF: int = 5  # initial back-off; doubles on each retry


# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    """Load *path* as YAML and return the parsed dict.

    Raises:
        FileNotFoundError: if *path* does not exist.
        yaml.YAMLError: if the file cannot be parsed.
    """
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_countries(path: Path = _CONFIG_COUNTRIES) -> list[dict]:
    """Return the ``mena_countries`` list from *config/countries.yaml*.

    Each element is a dict with at minimum the keys ``code`` (ISO alpha-2)
    and ``name``.
    """
    cfg = _load_yaml(path)
    countries: list[dict] = cfg["mena_countries"]
    log.debug("Loaded %d countries from %s.", len(countries), path)
    return countries


def load_indicators(
    path: Path = _CONFIG_INDICATORS,
) -> tuple[list[dict], dict]:
    """Return ``(indicators, fetch_defaults)`` from *config/indicators.yaml*.

    ``indicators`` is the ``world_bank_indicators`` list; each element has at
    minimum the key ``code``.  ``fetch_defaults`` is the ``fetch_defaults``
    mapping (may be empty if the key is absent).
    """
    cfg = _load_yaml(path)
    indicators: list[dict] = cfg["world_bank_indicators"]
    defaults: dict = cfg.get("fetch_defaults", {})
    log.debug("Loaded %d indicators from %s.", len(indicators), path)
    return indicators, defaults


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _build_country_query(countries: list[dict]) -> str:
    """Join ISO alpha-2 codes with semicolons for the World Bank API.

    Example: ``"DZ;BH;EG;IR;IQ;JO;KW;LB;LY;MA;OM;QA;SA;AE"``
    """
    return ";".join(c["code"] for c in countries)


def _indicator_to_filename(indicator_code: str) -> str:
    """Sanitise an indicator code for use as a filename.

    ``"NY.GDP.MKTP.CD"``  →  ``"NY_GDP_MKTP_CD.json"``
    """
    return indicator_code.replace(".", "_") + ".json"


def _get_with_retry(
    session: requests.Session,
    url: str,
    params: dict,
    max_retries: int = _MAX_RETRIES,
    initial_backoff: int = _RETRY_BACKOFF,
) -> requests.Response:
    """Issue a GET request with exponential back-off on transient errors.

    Retries on :class:`~requests.ConnectionError` and
    :class:`~requests.Timeout`; does **not** retry on HTTP 4xx/5xx (those
    are raised immediately via :meth:`~requests.Response.raise_for_status`).

    Args:
        session: Active :class:`~requests.Session`.
        url: Full request URL.
        params: Query-string parameters dict.
        max_retries: Maximum number of attempts before re-raising.
        initial_backoff: Seconds to wait before the first retry; doubles
            after each subsequent failure.

    Returns:
        The successful :class:`~requests.Response` object.

    Raises:
        requests.ConnectionError | requests.Timeout: after all retries are
            exhausted.
        requests.HTTPError: immediately on a non-2xx HTTP status.
    """
    delay = initial_backoff
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp

        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            log.warning(
                "Attempt %d/%d failed (%s) – retrying in %ds.",
                attempt,
                max_retries,
                exc,
                delay,
            )
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2

        except requests.HTTPError as exc:
            log.error("HTTP %s for %s", exc.response.status_code, url)
            raise

    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Core fetch logic
# ---------------------------------------------------------------------------

def fetch_indicator_records(
    session: requests.Session,
    country_query: str,
    indicator_code: str,
    date_range: str,
    per_page: int,
) -> list[dict]:
    """Fetch every paginated page for one indicator across the given countries.

    The World Bank API returns a 2-element JSON array::

        [ {"page": 1, "pages": 3, "per_page": "500", "total": 1200}, [...] ]

    This function iterates all pages and returns a flat merged list.

    Args:
        session: Active :class:`~requests.Session` (connection-pooled).
        country_query: Semicolon-separated ISO alpha-2 codes.
        indicator_code: World Bank indicator code, e.g. ``"NY.GDP.MKTP.CD"``.
        date_range: Date range string accepted by the API, e.g. ``"2000:2024"``.
        per_page: Records per page (World Bank hard cap is 32 500).

    Returns:
        Flat list of raw record dicts as returned by the API.  Returns an
        empty list when the API signals no data (``records == null``).

    Raises:
        ValueError: If the API response structure is unexpected.
        requests.HTTPError: On a non-2xx HTTP response (not retried).
    """
    url = f"{BASE_URL}/country/{country_query}/indicator/{indicator_code}"
    all_records: list[dict] = []
    page = 1

    while True:
        params: dict[str, Any] = {
            "format": "json",
            "per_page": per_page,
            "date": date_range,
            "page": page,
        }

        log.debug("GET %s  (page=%d)", url, page)
        resp = _get_with_retry(session, url, params)

        payload = resp.json()

        if not isinstance(payload, list) or len(payload) != 2:
            raise ValueError(
                f"Unexpected World Bank API response structure for "
                f"indicator '{indicator_code}', page {page}: {payload!r}"
            )

        api_meta, records = payload

        if records is None:
            log.warning(
                "Indicator '%s': API returned null data on page %d – "
                "no observations available for this country/date combination.",
                indicator_code,
                page,
            )
            break

        all_records.extend(records)

        total_pages: int = int(api_meta.get("pages", 1))
        log.debug(
            "Indicator '%s': page %d/%d fetched %d records (%d total so far).",
            indicator_code,
            page,
            total_pages,
            len(records),
            len(all_records),
        )

        if page >= total_pages:
            break

        page += 1
        time.sleep(_REQUEST_DELAY)  # be polite between pages of the same indicator

    return all_records


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_raw_response(
    records: list[dict],
    indicator: dict,
    countries: list[dict],
    date_range: str,
    output_dir: Path,
) -> Path:
    """Serialise *records* inside a metadata envelope and write to disk.

    The output filename is derived from the indicator code with dots replaced
    by underscores, e.g. ``NY_GDP_MKTP_CD.json``.

    Args:
        records: Flat list of API records returned by :func:`fetch_indicator_records`.
        indicator: Indicator config dict (must contain ``code`` and ``name``).
        countries: Country config dicts used during the fetch.
        date_range: Date range string used in the API call.
        output_dir: Directory to write into; created if it does not exist.

    Returns:
        The :class:`~pathlib.Path` of the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    envelope: dict[str, Any] = {
        "meta": {
            "indicator_code": indicator["code"],
            "indicator_name": indicator["name"],
            "indicator_category": indicator.get("category"),
            "indicator_unit": indicator.get("unit"),
            "countries": [
                {
                    "code": c["code"],
                    "code_a3": c["code_a3"],
                    "name": c["name"],
                }
                for c in countries
            ],
            "date_range": date_range,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "total_records": len(records),
            "source": "World Bank Open Data",
            "api_base_url": BASE_URL,
        },
        "records": records,
    }

    out_path = output_dir / _indicator_to_filename(indicator["code"])
    out_path.write_text(
        json.dumps(envelope, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Saved %d records → %s", len(records), out_path)
    return out_path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def fetch_all(
    countries_config: Path = _CONFIG_COUNTRIES,
    indicators_config: Path = _CONFIG_INDICATORS,
    output_dir: Path = _OUTPUT_DIR,
    indicator_filter: str | None = None,
    dry_run: bool = False,
) -> dict[str, bool]:
    """Fetch every configured indicator for all MENA countries.

    Iterates each indicator in ``config/indicators.yaml`` (or only the one
    matching *indicator_filter*), batches all 14 country codes into a single
    API call, handles pagination, and writes the results to *output_dir*.

    Args:
        countries_config: Path to the countries YAML config file.
        indicators_config: Path to the indicators YAML config file.
        output_dir: Root directory for raw JSON output.
        indicator_filter: When provided, fetch only this indicator code.
        dry_run: Log the URLs that would be called without making any
            network requests or writing any files.

    Returns:
        Dict mapping each indicator code to ``True`` (success / dry-run) or
        ``False`` (failed after all retries).
    """
    countries = load_countries(countries_config)
    indicators, defaults = load_indicators(indicators_config)

    date_range: str = str(defaults.get("date_range", "2000:2024"))
    per_page: int = int(defaults.get("per_page", 500))
    country_query = _build_country_query(countries)

    if indicator_filter:
        indicators = [i for i in indicators if i["code"] == indicator_filter]
        if not indicators:
            log.error(
                "Indicator '%s' not found in %s.", indicator_filter, indicators_config
            )
            return {}

    log.info(
        "Starting fetch: %d indicator(s) × %d countries | date range: %s.",
        len(indicators),
        len(countries),
        date_range,
    )

    results: dict[str, bool] = {}

    with requests.Session() as session:
        session.headers.update(
            {"User-Agent": "middle-east-oil-chain-analysis/0.1 (github.com)"}
        )

        for idx, indicator in enumerate(indicators, start=1):
            code: str = indicator["code"]
            log.info(
                "[%d/%d] %s — %s",
                idx,
                len(indicators),
                code,
                indicator["name"],
            )

            if dry_run:
                url = (
                    f"{BASE_URL}/country/{country_query}/indicator/{code}"
                    f"?format=json&per_page={per_page}&date={date_range}&page=1"
                )
                log.info("[DRY RUN] Would GET: %s", url)
                results[code] = True
                continue

            try:
                records = fetch_indicator_records(
                    session=session,
                    country_query=country_query,
                    indicator_code=code,
                    date_range=date_range,
                    per_page=per_page,
                )
                save_raw_response(
                    records=records,
                    indicator=indicator,
                    countries=countries,
                    date_range=date_range,
                    output_dir=output_dir,
                )
                results[code] = True

            except requests.HTTPError as exc:
                log.error(
                    "HTTP error for indicator '%s' (status %s). Skipping.",
                    code,
                    exc.response.status_code,
                )
                results[code] = False

            except (requests.ConnectionError, requests.Timeout) as exc:
                log.error(
                    "Network error for indicator '%s' after %d retries: %s. Skipping.",
                    code,
                    _MAX_RETRIES,
                    exc,
                )
                results[code] = False

            except ValueError as exc:
                log.error(
                    "Malformed API response for indicator '%s': %s. Skipping.",
                    code,
                    exc,
                )
                results[code] = False

            except Exception as exc:  # noqa: BLE001 – catch-all so one failure never aborts the run
                log.exception(
                    "Unexpected error for indicator '%s': %s. Skipping.", code, exc
                )
                results[code] = False

            # Pause before the next indicator to avoid hammering the API
            if idx < len(indicators):
                time.sleep(_REQUEST_DELAY)

    _log_summary(results)
    return results


def _log_summary(results: dict[str, bool]) -> None:
    """Emit a pass/fail summary table to the log."""
    passed = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]
    log.info(
        "─── Fetch summary: %d succeeded, %d failed ───", len(passed), len(failed)
    )
    for code in passed:
        log.info("  ✓  %s", code)
    for code in failed:
        log.error("  ✗  %s", code)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.data.fetch_world_bank",
        description="Fetch World Bank indicators for all 14 MENA countries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--indicator",
        metavar="CODE",
        default=None,
        help=(
            "Fetch a single indicator code instead of all seven "
            "(e.g. NY.GDP.MKTP.CD)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        metavar="PATH",
        default=str(_OUTPUT_DIR),
        help=f"Directory for raw JSON output files (default: {_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print API URLs without making network requests or writing files.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, run the fetch, and return an exit code.

    Returns:
        ``0`` if every indicator succeeded (or dry-run), ``1`` if any failed.
    """
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    results = fetch_all(
        output_dir=Path(args.output_dir),
        indicator_filter=args.indicator,
        dry_run=args.dry_run,
    )

    failed = [k for k, v in results.items() if not v]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
