"""
Right-Now Risk composite model — combines fiscal stress, reserve runway,
social stability, and chain transmission into a single per-country score.

Composite formula (default weights):
    right_now_risk_score =
        0.35 * fiscal_stress_score            (Addition 1)
        0.25 * reserve_runway_risk            (Addition 2)
        0.20 * social_stability_risk          (Addition 3)
        0.20 * chain_transmission_severity_recent  (existing model)

Fallback policy:
    When any component value is NaN for a country, the available
    component weights are rescaled proportionally to sum to 1.0.
    The missing component names and the actual weights used are
    recorded in the output (missing_components, rescaled_weights).
    A country is never silently dropped.

Key functions
-------------
load_component_tables           Orchestrate loading of all four sources
compute_reserve_runway_risk     Convert runway months to [0,1] risk score
compute_chain_recent            Mean severity for most recent N years, normalised
compute_right_now_risk          Assemble the composite score table
identify_primary_driver         Return the dominant component label
run_right_now_risk              End-to-end orchestrator

Driver labels
-------------
    DRIVER_FISCAL    = "Fiscal stress"
    DRIVER_RUNWAY    = "Reserve runway"
    DRIVER_SOCIAL    = "Social stability"
    DRIVER_CHAIN     = "Chain transmission"
    DRIVER_MIXED     = "Mixed"
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.brent import fetch_brent_ytd, fetch_live_brent
from src.model.fiscal_stress import build_stress_table, load_breakeven
from src.model.reserve_runway import build_runway_table, load_reserves
from src.model.social_stability import (
    build_stability_table,
    derive_inflation_vol,
    load_food_security,
)

log = logging.getLogger(__name__)

# ── Default paths ──────────────────────────────────────────────────────────────

BREAKEVEN_PATH = Path("data/reference/fiscal_breakeven.csv")
RESERVES_PATH  = Path("data/reference/swf_reserves.csv")
FOOD_PATH      = Path("data/reference/food_security.csv")
CHAIN_PATH     = Path("outputs/tables/chain_transmission.csv")
PANEL_PATH     = Path("data/processed/world_bank_panel.csv")

# ── Composite weights ──────────────────────────────────────────────────────────

_W_FISCAL = 0.35
_W_RUNWAY = 0.25
_W_SOCIAL = 0.20
_W_CHAIN  = 0.20

_COMPONENT_WEIGHTS: dict[str, float] = {
    "fiscal_stress_score":                _W_FISCAL,
    "reserve_runway_risk":                _W_RUNWAY,
    "social_stability_risk":              _W_SOCIAL,
    "chain_transmission_severity_recent": _W_CHAIN,
}

# ── Driver labels ──────────────────────────────────────────────────────────────

DRIVER_FISCAL  = "Fiscal stress"
DRIVER_RUNWAY  = "Reserve runway"
DRIVER_SOCIAL  = "Social stability"
DRIVER_CHAIN   = "Chain transmission"
DRIVER_MIXED   = "Mixed"

_ALLOWED_DRIVERS = frozenset({
    DRIVER_FISCAL, DRIVER_RUNWAY, DRIVER_SOCIAL, DRIVER_CHAIN, DRIVER_MIXED,
})

# Months thresholds for runway-to-risk conversion.
_RUNWAY_RISK_FLOOR_MONTHS   = 6.0   # <= this -> risk = 1.0
_RUNWAY_RISK_CEILING_MONTHS = 36.0  # >= this -> risk = 0.0

# Gap threshold: if top-two driver contributions are within 5% of total,
# classify as Mixed rather than a single dominant driver.
_DRIVER_TIE_THRESHOLD = 0.05

# Human-readable normalization method string embedded in CSV exports.
_COMPONENT_NORM_METHODS = (
    "fiscal:continuous_breakeven_formula; "
    "runway:linear_interpolation_6_36mo; "
    "social:minmax_winsorized_p5p95; "
    "chain:minmax_3yr_rolling_mean"
)

# Default path for the reproducibility CSV snapshot.
_DEFAULT_EXPORT_PATH = Path("outputs/tables/right_now_risk_scores.csv")


# ── Runway risk conversion ─────────────────────────────────────────────────────

def compute_reserve_runway_risk(runway_df: pd.DataFrame) -> pd.DataFrame:
    """Convert reserve runway months to a [0,1] risk score per country.

    Gray runway (country not under fiscal stress) and NaN runway both
    map to 0.0 — the concept is not applicable, not "zero risk".

    Linear interpolation between _RUNWAY_RISK_FLOOR_MONTHS (risk=1.0)
    and _RUNWAY_RISK_CEILING_MONTHS (risk=0.0).

    Args:
        runway_df: Output of src.model.reserve_runway.build_runway_table.

    Returns:
        DataFrame with columns:
            country_code_a3, reserve_runway_risk,
            reserve_runway_months, runway_status.
    """
    df = runway_df[["country_code_a3", "reserve_runway_months", "runway_status"]].copy()

    def _risk(row) -> float:
        status = str(row["runway_status"])
        months = row["reserve_runway_months"]
        if status == "Gray":
            return 0.0
        if pd.isna(months):
            return 0.0
        m = float(months)
        if math.isnan(m):
            return 0.0
        if m <= _RUNWAY_RISK_FLOOR_MONTHS:
            return 1.0
        if m >= _RUNWAY_RISK_CEILING_MONTHS:
            return 0.0
        return 1.0 - (m - _RUNWAY_RISK_FLOOR_MONTHS) / (
            _RUNWAY_RISK_CEILING_MONTHS - _RUNWAY_RISK_FLOOR_MONTHS
        )

    df["reserve_runway_risk"] = df.apply(_risk, axis=1)
    return df


# ── Chain transmission — recent average ───────────────────────────────────────

def compute_chain_recent(
    chain_df: pd.DataFrame,
    n_years: int = 3,
) -> pd.DataFrame:
    """Mean transmission_severity for the most recent N years, normalised to [0,1].

    Takes the last N calendar years present in the data (not necessarily
    contiguous), computes a per-country mean, then min-max normalises
    across all countries.

    Args:
        chain_df: Chain transmission DataFrame with columns
                  ['country_code_a3', 'year', 'transmission_severity'].
        n_years:  Look-back window in years (default 3).

    Returns:
        DataFrame with columns:
            country_code_a3, chain_transmission_severity_recent, chain_severity_raw.
        Returns an empty DataFrame on missing input.
    """
    _empty = pd.DataFrame(
        columns=["country_code_a3", "chain_transmission_severity_recent", "chain_severity_raw"]
    )
    if chain_df.empty or "transmission_severity" not in chain_df.columns:
        return _empty
    if "country_code_a3" not in chain_df.columns:
        log.warning("chain_df missing country_code_a3 — skipping chain component")
        return _empty

    max_year = int(chain_df["year"].max())
    cutoff   = max_year - n_years + 1
    recent   = chain_df[chain_df["year"] >= cutoff]

    if recent.empty:
        return _empty

    agg = (
        recent
        .groupby("country_code_a3")["transmission_severity"]
        .mean()
        .reset_index()
        .rename(columns={"transmission_severity": "chain_severity_raw"})
    )

    lo = agg["chain_severity_raw"].min()
    hi = agg["chain_severity_raw"].max()
    if hi > lo:
        agg["chain_transmission_severity_recent"] = (
            (agg["chain_severity_raw"] - lo) / (hi - lo)
        ).clip(0.0, 1.0)
    else:
        agg["chain_transmission_severity_recent"] = 0.0

    log.info(
        "Chain recent (%d-yr mean): %d countries  max_year=%d  raw [%.3f, %.3f]",
        n_years, len(agg), max_year, lo, hi,
    )
    return agg


# ── Driver identification ──────────────────────────────────────────────────────

def identify_primary_driver(
    contrib_fiscal: float,
    contrib_runway: float,
    contrib_social: float,
    contrib_chain:  float,
) -> str:
    """Return the label of the dominant component.

    Returns DRIVER_MIXED when:
      - total is zero or negative, OR
      - the gap between the top two contributions is < 5% of total.

    Args:
        contrib_fiscal:  Weighted fiscal stress contribution.
        contrib_runway:  Weighted reserve runway contribution.
        contrib_social:  Weighted social stability contribution.
        contrib_chain:   Weighted chain transmission contribution.

    Returns:
        One of DRIVER_FISCAL, DRIVER_RUNWAY, DRIVER_SOCIAL,
        DRIVER_CHAIN, DRIVER_MIXED.
    """
    contribs = {
        DRIVER_FISCAL: contrib_fiscal,
        DRIVER_RUNWAY: contrib_runway,
        DRIVER_SOCIAL: contrib_social,
        DRIVER_CHAIN:  contrib_chain,
    }
    total = sum(contribs.values())
    if total <= 0.0:
        return DRIVER_MIXED

    sorted_vals = sorted(contribs.values(), reverse=True)
    gap = sorted_vals[0] - sorted_vals[1]
    if gap / total < _DRIVER_TIE_THRESHOLD:
        return DRIVER_MIXED

    return max(contribs, key=contribs.__getitem__)


# ── Output validation ──────────────────────────────────────────────────────────

def _validate_output(df: pd.DataFrame) -> None:
    """Post-assembly guardrail checks.  Logs warnings on violations; never raises.

    Checks:
        - right_now_risk_score in [0, 1] for every non-NaN value
        - exactly 14 countries (MENA-14 coverage)
        - no duplicate country_code_a3
        - all primary_driver values in _ALLOWED_DRIVERS
        - missing_components column is never null (may be empty string)
    """
    valid_scores = df["right_now_risk_score"].dropna()
    oob = valid_scores[(valid_scores < 0.0) | (valid_scores > 1.0)]
    if not oob.empty:
        log.warning("GUARDRAIL FAIL: %d score(s) outside [0,1]: %s", len(oob), oob.tolist())

    n = len(df)
    if n != 14:
        log.warning("GUARDRAIL FAIL: expected 14 countries, got %d", n)

    if "country_code_a3" in df.columns:
        dup = df["country_code_a3"][df["country_code_a3"].duplicated()]
        if not dup.empty:
            log.warning("GUARDRAIL FAIL: duplicate country_code_a3: %s", dup.tolist())

    if "primary_driver" in df.columns:
        bad = [d for d in df["primary_driver"].dropna() if d not in _ALLOWED_DRIVERS]
        if bad:
            log.warning("GUARDRAIL FAIL: invalid primary_driver values: %s", bad)

    if "missing_components" in df.columns:
        n_null = int(df["missing_components"].isna().sum())
        if n_null > 0:
            log.warning(
                "GUARDRAIL FAIL: %d row(s) have null missing_components "
                "(expected empty string or reason string)",
                n_null,
            )

    log.info("Output validation complete (see any GUARDRAIL FAIL lines above)")


# ── Component table loader ─────────────────────────────────────────────────────

def load_component_tables(
    breakeven_path: Path = BREAKEVEN_PATH,
    reserves_path:  Path = RESERVES_PATH,
    food_path:      Path = FOOD_PATH,
    chain_path:     Path = CHAIN_PATH,
    panel_path:     Path = PANEL_PATH,
    brent_live:     float | None = None,
) -> dict:
    """Load and build all four component tables for the Right Now Risk model.

    Fetches live Brent once and passes it to every downstream builder so
    all components use the same price snapshot.

    Args:
        breakeven_path: Path to fiscal_breakeven.csv.
        reserves_path:  Path to swf_reserves.csv.
        food_path:      Path to food_security.csv.
        chain_path:     Path to chain_transmission.csv.
        panel_path:     Path to world_bank_panel.csv.
        brent_live:     Override live price (useful for testing/offline).

    Returns:
        dict with keys:
            'brent_live'      : float
            'stress_table'    : pd.DataFrame  (Addition 1)
            'runway_table'    : pd.DataFrame  (Addition 2)
            'stability_table' : pd.DataFrame  (Addition 3)
            'chain_df'        : pd.DataFrame  (chain transmission, raw)

    Raises:
        FileNotFoundError: If any required CSV does not exist.
        ValueError: If required columns are absent.
    """
    if brent_live is None:
        brent_live = fetch_live_brent()

    ytd_prices   = fetch_brent_ytd()
    breakeven_df = load_breakeven(breakeven_path)
    stress_table = build_stress_table(breakeven_df, brent_live, ytd_prices)

    reserves_df     = load_reserves(reserves_path)
    runway_table    = build_runway_table(reserves_df, stress_table)

    food_df         = load_food_security(food_path)
    inflation_df    = derive_inflation_vol(panel_path)
    stability_table = build_stability_table(food_df, stress_table, inflation_df)

    if chain_path.exists():
        chain_df = pd.read_csv(chain_path)
    else:
        log.warning("Chain transmission CSV not found: %s — chain component will be NaN", chain_path)
        chain_df = pd.DataFrame(
            columns=["country_code_a3", "year", "transmission_severity"]
        )

    log.info(
        "Component tables loaded: brent=$%.2f  "
        "stress=%d rows  runway=%d rows  stability=%d rows  chain=%d rows",
        brent_live,
        len(stress_table), len(runway_table), len(stability_table), len(chain_df),
    )
    return {
        "brent_live":      brent_live,
        "stress_table":    stress_table,
        "runway_table":    runway_table,
        "stability_table": stability_table,
        "chain_df":        chain_df,
    }


# ── Composite score assembly ───────────────────────────────────────────────────

def compute_right_now_risk(
    stress_table:    pd.DataFrame,
    runway_df:       pd.DataFrame,
    stability_df:    pd.DataFrame,
    chain_recent_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the per-country Right Now Risk composite score.

    Each country receives a score in [0,1].  When any component is missing
    (NaN) for a row, the remaining weights are rescaled proportionally to
    sum to 1.0.  Missing components and actual weights are recorded per row.

    The fiscal_stress_score comes from stability_df (computed via
    compute_fiscal_stress_score in Addition 3) so all four additions share
    the same Brent snapshot.

    Args:
        stress_table:    Fiscal stress table (Addition 1 output).
        runway_df:       Reserve runway table (Addition 2 output).
        stability_df:    Social stability table (Addition 3 output).
        chain_recent_df: Per-country recent chain severity (compute_chain_recent).

    Returns:
        DataFrame sorted by right_now_risk_score descending, with columns:
            country_code_a3, country_label, stress_status, runway_status,
            fiscal_stress_score, reserve_runway_risk, social_stability_risk,
            chain_transmission_severity_recent, chain_severity_raw,
            right_now_risk_score, primary_driver, missing_components,
            rescaled_weights, brent_live_usd, fiscal_breakeven_usd,
            price_gap_usd, confidence, reserve_runway_months.
    """
    # Build runway risk scores
    runway_risk_df = compute_reserve_runway_risk(runway_df)

    # Slim each source to the columns we need
    stress_slim = stress_table[[
        c for c in [
            "country_code_a3", "country_label", "stress_status",
            "brent_live_usd", "fiscal_breakeven_usd", "price_gap_usd",
        ]
        if c in stress_table.columns
    ]].copy()

    runway_slim = runway_risk_df[[
        "country_code_a3", "reserve_runway_risk",
        "reserve_runway_months", "runway_status",
    ]].copy()

    stability_slim = stability_df[[
        c for c in [
            "country_code_a3", "fiscal_stress_score",
            "social_stability_risk", "confidence",
        ]
        if c in stability_df.columns
    ]].copy()

    chain_slim = chain_recent_df[[
        c for c in [
            "country_code_a3",
            "chain_transmission_severity_recent",
            "chain_severity_raw",
        ]
        if c in chain_recent_df.columns
    ]].copy()

    # Merge onto stress table as base (left joins preserve all 14 countries)
    df = stress_slim.copy()
    df = df.merge(runway_slim,    on="country_code_a3", how="left")
    df = df.merge(stability_slim, on="country_code_a3", how="left")
    df = df.merge(chain_slim,     on="country_code_a3", how="left")

    # Ensure component columns exist even if joins returned nothing
    for col in _COMPONENT_WEIGHTS:
        if col not in df.columns:
            df[col] = float("nan")

    # Verify chain component populated when source data was available
    _chain_has_data = (
        not chain_slim.empty
        and "chain_transmission_severity_recent" in chain_slim.columns
    )
    if _chain_has_data:
        n_chain_pop = df["chain_transmission_severity_recent"].notna().sum()
        if n_chain_pop == 0:
            log.warning(
                "chain_transmission_severity_recent is NaN for all countries despite "
                "chain_recent_df having %d rows — check country_code_a3 key alignment",
                len(chain_slim),
            )
        else:
            log.info(
                "chain_transmission_severity_recent populated for %d/%d countries",
                n_chain_pop, len(df),
            )

    # Pre-compute chain country set for per-row reason generation
    _chain_a3_set: set[str] = (
        set(chain_slim["country_code_a3"].tolist()) if _chain_has_data else set()
    )

    # Per-row composite score with weight rescaling fallback
    scores          = []
    primary_drivers = []
    missing_list    = []
    weights_used    = []

    for _, row in df.iterrows():
        available: dict[str, tuple[float, float]] = {}
        missing: list[str] = []

        a3 = str(row.get("country_code_a3", ""))

        for col, w in _COMPONENT_WEIGHTS.items():
            val = row.get(col, float("nan"))
            if pd.isna(val):
                # Include a specific reason for the chain component so the
                # missing_components string is self-explanatory in the CSV and UI.
                if col == "chain_transmission_severity_recent":
                    if not _chain_has_data:
                        missing.append(col + " [no chain data available]")
                    elif a3 not in _chain_a3_set:
                        missing.append(col + " [country absent from chain CSV]")
                    else:
                        missing.append(col + " [normalization produced NaN]")
                else:
                    missing.append(col)
            else:
                available[col] = (float(val), w)

        if not available:
            scores.append(float("nan"))
            primary_drivers.append(DRIVER_MIXED)
            missing_list.append("; ".join(missing))
            weights_used.append("{}")
            continue

        total_w  = sum(w for _, w in available.values())
        rescaled = {col: w / total_w for col, (_, w) in available.items()}

        score = sum(val * rescaled[col] for col, (val, _) in available.items())
        scores.append(min(1.0, max(0.0, score)))
        missing_list.append("; ".join(missing) if missing else "")
        weights_used.append(str({k: round(v, 4) for k, v in rescaled.items()}))

        # Weighted contributions for driver identification
        def _contrib(key: str) -> float:
            if key not in available:
                return 0.0
            return available[key][0] * rescaled[key]

        primary_drivers.append(identify_primary_driver(
            contrib_fiscal=_contrib("fiscal_stress_score"),
            contrib_runway=_contrib("reserve_runway_risk"),
            contrib_social=_contrib("social_stability_risk"),
            contrib_chain =_contrib("chain_transmission_severity_recent"),
        ))

    df["right_now_risk_score"] = scores
    df["primary_driver"]       = primary_drivers
    df["missing_components"]   = missing_list
    df["rescaled_weights"]     = weights_used

    df = df.sort_values("right_now_risk_score", ascending=False).reset_index(drop=True)

    n_complete = int((df["missing_components"] == "").sum())
    n_missing_chain = int(
        df["missing_components"]
        .str.contains("chain_transmission_severity_recent", na=False)
        .sum()
    )
    log.info(
        "Right Now Risk table: %d countries  %d complete  %d partial  "
        "count_rows_missing_chain=%d",
        len(df), n_complete, len(df) - n_complete, n_missing_chain,
    )
    _validate_output(df)
    return df


# ── End-to-end orchestrator ────────────────────────────────────────────────────

def run_right_now_risk(
    breakeven_path: Path = BREAKEVEN_PATH,
    reserves_path:  Path = RESERVES_PATH,
    food_path:      Path = FOOD_PATH,
    chain_path:     Path = CHAIN_PATH,
    panel_path:     Path = PANEL_PATH,
    chain_n_years:  int  = 3,
    export_csv:     Path | None = _DEFAULT_EXPORT_PATH,
) -> dict:
    """End-to-end Right Now Risk pipeline.

    Fetches live Brent, builds all four component tables, computes the
    chain transmission recent average, assembles the composite score,
    and optionally exports a reproducibility CSV snapshot.

    Args:
        breakeven_path: Path to fiscal_breakeven.csv.
        reserves_path:  Path to swf_reserves.csv.
        food_path:      Path to food_security.csv.
        chain_path:     Path to chain_transmission.csv.
        panel_path:     Path to world_bank_panel.csv.
        chain_n_years:  Look-back window for chain severity average (default 3).
        export_csv:     Path for the reproducibility CSV snapshot.
                        Defaults to outputs/tables/right_now_risk_scores.csv.
                        Pass ``None`` to skip export.

    Returns:
        dict with keys:
            'brent_live'        : float
            'stress_table'      : pd.DataFrame
            'runway_table'      : pd.DataFrame
            'stability_table'   : pd.DataFrame
            'chain_recent_df'   : pd.DataFrame
            'right_now_risk_df' : pd.DataFrame

    Raises:
        FileNotFoundError: If any required CSV does not exist.
        ValueError: If required columns are absent from a CSV.
    """
    components = load_component_tables(
        breakeven_path=breakeven_path,
        reserves_path=reserves_path,
        food_path=food_path,
        chain_path=chain_path,
        panel_path=panel_path,
    )

    chain_recent_df = compute_chain_recent(
        components["chain_df"], n_years=chain_n_years
    )

    right_now_risk_df = compute_right_now_risk(
        stress_table=components["stress_table"],
        runway_df=components["runway_table"],
        stability_df=components["stability_table"],
        chain_recent_df=chain_recent_df,
    )

    # ── Optional reproducibility CSV export ───────────────────────────────────
    if export_csv is not None:
        export_path = Path(export_csv)
        try:
            export_path.parent.mkdir(parents=True, exist_ok=True)
            _snapshot_ts   = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            _weights_base  = json.dumps({
                "fiscal_stress_score":                _W_FISCAL,
                "reserve_runway_risk":                _W_RUNWAY,
                "social_stability_risk":              _W_SOCIAL,
                "chain_transmission_severity_recent": _W_CHAIN,
            })
            export_df = right_now_risk_df.copy()
            export_df["snapshot_timestamp_utc"]      = _snapshot_ts
            export_df["live_brent_used"]             = components["brent_live"]
            export_df["component_weights_base"]      = _weights_base
            export_df["component_weights_effective"] = export_df["rescaled_weights"]
            export_df["normalization_method"]        = _COMPONENT_NORM_METHODS
            export_df.to_csv(export_path, index=False)
            log.info("Reproducibility snapshot exported: %s", export_path)
        except Exception as exc:
            log.warning("CSV export failed (%s) — continuing without export", exc)

    return {
        "brent_live":        components["brent_live"],
        "stress_table":      components["stress_table"],
        "runway_table":      components["runway_table"],
        "stability_table":   components["stability_table"],
        "chain_recent_df":   chain_recent_df,
        "right_now_risk_df": right_now_risk_df,
    }
