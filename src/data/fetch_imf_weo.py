"""Fetch IMF WEO April 2026 indicators for 14 MENA countries.

Outputs:
- raw JSON per-indicator under data/raw/imf_weo/
- long panel CSV at data/processed/imf_weo_panel.csv
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

API_BASE = "https://www.imf.org/external/datamapper/api/v1"
RAW_DIR = Path("data/raw/imf_weo")
OUT_CSV = Path("data/processed/imf_weo_panel.csv")
SOURCE_REGISTRY = Path("data/reference/source_registry.csv")

COUNTRIES = ["BH", "DZ", "EG", "IQ", "IR", "JO", "KW", "LB", "LY", "MA", "OM", "QA", "SA", "AE"]
ISO2_TO_A3 = {"BH": "BHR", "DZ": "DZA", "EG": "EGY", "IQ": "IRQ", "IR": "IRN", "JO": "JOR", "KW": "KWT", "LB": "LBN", "LY": "LBY", "MA": "MAR", "OM": "OMN", "QA": "QAT", "SA": "SAU", "AE": "ARE"}
INDICATORS = ["NGDP_RPCH", "PCPIPCH", "GGXCNL_NGDP", "BCA_NGDPD", "LUR"]
YEARS = set(range(2015, 2027))


def _register_source() -> None:
    if not SOURCE_REGISTRY.exists():
        return
    df = pd.read_csv(SOURCE_REGISTRY)
    if "source_id" not in df.columns or "source_name" not in df.columns:
        return
    if (df["source_id"] == "IMF_WEO_2026").any():
        return
    row = {
        "source_id": "IMF_WEO_2026",
        "source_name": "IMF World Economic Outlook April 2026",
        "organization": "IMF",
        "publication_year": 2026,
        "retrieval_date": pd.Timestamp.utcnow().date().isoformat(),
        "url": "https://www.imf.org/en/Publications/WEO",
        "used_for": "imf_weo_panel; inflation_volatility_fallback; fiscal_balance_context",
        "confidence_tier": "high",
        "notes": "Datamapper API pull for 2015-2026; 2025-2026 flagged as estimates.",
    }
    for col in df.columns:
        row.setdefault(col, "")
    df = pd.concat([df, pd.DataFrame([row])[df.columns]], ignore_index=True)
    df.to_csv(SOURCE_REGISTRY, index=False)
    log.info("Registered source IMF_WEO_2026 in source_registry.csv")


def _fetch_indicator(indicator: str) -> dict:
    url = f"{API_BASE}/{indicator}/{'/'.join([])}"
    # IMF endpoint expects indicator/country list in path
    url = f"{API_BASE}/{indicator}/{','.join(COUNTRIES)}"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        log.warning("IMF API fetch failed for %s: %s", indicator, exc)
        return {}

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / f"{indicator}.json"
    raw_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def _extract_rows(indicator: str, payload: dict) -> list[dict]:
    rows: list[dict] = []
    values = payload.get("values", {}).get(indicator, {}) if isinstance(payload, dict) else {}
    for iso2, year_map in values.items():
        if not isinstance(year_map, dict):
            continue
        for year_str, val in year_map.items():
            try:
                year = int(year_str)
            except ValueError:
                continue
            if year not in YEARS:
                continue
            try:
                value = float(val)
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "country_code_a3": ISO2_TO_A3.get(iso2, iso2),
                    "year": year,
                    "indicator": indicator,
                    "value": value,
                    "is_estimate": year >= 2025,
                    "source_id": "IMF_WEO_2026",
                }
            )
    return rows


def fetch_imf_weo_panel() -> pd.DataFrame:
    all_rows: list[dict] = []
    for ind in INDICATORS:
        payload = _fetch_indicator(ind)
        if not payload:
            continue
        all_rows.extend(_extract_rows(ind, payload))

    if not all_rows:
        log.warning("No IMF WEO rows fetched; returning empty DataFrame")
        return pd.DataFrame(
            columns=["country_code_a3", "year", "indicator", "value", "is_estimate", "source_id"]
        )

    df = pd.DataFrame(all_rows).sort_values(["country_code_a3", "indicator", "year"])
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    _register_source()
    df = fetch_imf_weo_panel()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    log.info("Wrote %d rows to %s", len(df), OUT_CSV)


if __name__ == "__main__":
    main()
