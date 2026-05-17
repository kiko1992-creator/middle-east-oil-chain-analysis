"""
Build a clean country-year panel from the raw World Bank JSON files.

Reads every ``*.json`` file under ``data/raw/world_bank/``, extracts the
``records`` array from each envelope, and pivots them into a single wide
panel DataFrame indexed by ``(country_code, year)``.

Missing observations are preserved as ``NaN`` — values are **never filled,
interpolated, or invented**.  Two audit columns are appended so that any
downstream code can immediately see which cells are absent:

* ``missing_count``      – integer: how many of the 7 indicators are NaN for
                           that country-year row.
* ``missing_indicators`` – string: comma-separated indicator codes that are
                           NaN (empty string when the row is complete).

Output
------
``data/processed/world_bank_panel.csv``

    Columns (in order):
        country_code, country_code_a3, country_name, year,
        <7 indicator columns>,
        missing_count, missing_indicators

Usage (from project root)::

    python -m src.data.clean_world_bank
    python -m src.data.clean_world_bank --raw-dir data/raw/world_bank
    python -m src.data.clean_world_bank --output data/processed/panel.csv

Exit codes:
    0  panel built and saved successfully
    1  unrecoverable error (no files found, I/O failure, etc.)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_RAW_DIR: Path = Path("data/raw/world_bank")
_OUTPUT_PATH: Path = Path("data/processed/world_bank_panel.csv")

# Canonical column order for identity fields
_ID_COLS = ["country_code", "country_code_a3", "country_name", "year"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw_file(path: Path) -> tuple[dict, pd.DataFrame]:
    """Load one raw World Bank JSON envelope and return ``(meta, records_df)``.

    ``records_df`` has exactly five columns::

        indicator_code | country_code | country_code_a3 | country_name | year | value

    Rows whose ``value`` field is ``null`` in the source JSON are kept with
    ``NaN`` in the ``value`` column — no rows are dropped at this stage.

    Args:
        path: Path to a ``*.json`` file written by :mod:`src.data.fetch_world_bank`.

    Returns:
        A two-tuple ``(meta_dict, records_dataframe)``.

    Raises:
        KeyError: If the JSON envelope is missing the ``meta`` or ``records`` key.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    log.debug("Loading %s", path)
    with path.open(encoding="utf-8") as fh:
        envelope = json.load(fh)

    meta: dict = envelope["meta"]
    raw_records: list[dict] = envelope["records"]

    indicator_code: str = meta["indicator_code"]

    if not raw_records:
        log.warning("File %s contains zero records — skipping.", path.name)
        return meta, pd.DataFrame(
            columns=["indicator_code", "country_code", "country_code_a3", "country_name", "year", "value"]
        )

    rows = []
    for rec in raw_records:
        rows.append(
            {
                "indicator_code": indicator_code,
                "country_code": rec["country"]["id"],
                "country_code_a3": rec["countryiso3code"],
                "country_name": rec["country"]["value"],
                "year": int(rec["date"]),
                # null in JSON → NaN in pandas automatically via float(None) path;
                # we keep None here and let pandas handle the dtype on DataFrame creation
                "value": rec["value"],
            }
        )

    records_df = pd.DataFrame(rows)
    # Ensure year is int even when mixed with NaN-carrying rows
    records_df["year"] = records_df["year"].astype(int)

    null_n = records_df["value"].isna().sum()
    log.debug(
        "  %s: %d records, %d null (%.1f%%)",
        indicator_code,
        len(records_df),
        null_n,
        null_n / len(records_df) * 100,
    )

    return meta, records_df


def load_all_raw(raw_dir: Path) -> tuple[list[dict], pd.DataFrame]:
    """Load every ``*.json`` file in *raw_dir* and concatenate into one long DataFrame.

    Args:
        raw_dir: Directory that contains the raw World Bank JSON files.

    Returns:
        A two-tuple ``(list_of_meta_dicts, long_dataframe)`` where the
        long DataFrame has columns
        ``indicator_code, country_code, country_code_a3, country_name, year, value``.

    Raises:
        FileNotFoundError: If *raw_dir* does not exist.
        RuntimeError: If no ``*.json`` files are found in *raw_dir*.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    json_files = sorted(raw_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No *.json files found in {raw_dir}")

    log.info("Found %d JSON file(s) in %s.", len(json_files), raw_dir)

    all_meta: list[dict] = []
    frames: list[pd.DataFrame] = []

    for path in json_files:
        meta, df = load_raw_file(path)
        all_meta.append(meta)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError("All JSON files were empty — cannot build panel.")

    long_df = pd.concat(frames, ignore_index=True)
    log.info(
        "Concatenated long DataFrame: %d rows across %d indicator(s).",
        len(long_df),
        long_df["indicator_code"].nunique(),
    )
    return all_meta, long_df


# ---------------------------------------------------------------------------
# Pivoting
# ---------------------------------------------------------------------------

def _safe_indicator_colname(indicator_code: str) -> str:
    """Convert ``"NY.GDP.MKTP.CD"`` → ``"NY_GDP_MKTP_CD"`` for use as a column name."""
    return indicator_code.replace(".", "_")


def pivot_to_panel(long_df: pd.DataFrame, all_meta: list[dict]) -> pd.DataFrame:
    """Pivot the long DataFrame into a wide country-year panel.

    Each indicator becomes its own column.  The complete set of
    (country, year) pairs forms the index so that rows missing from a
    particular indicator's data are **explicitly represented as NaN**
    rather than silently absent.

    Args:
        long_df: Concatenated long DataFrame from :func:`load_all_raw`.
        all_meta: List of ``meta`` dicts (one per indicator file), used to
                  build a clean column-label mapping.

    Returns:
        Wide panel DataFrame with ``country_code`` and ``year`` as the first
        two index-defining columns, one column per indicator, plus the
        ``country_code_a3`` and ``country_name`` identity columns.
    """
    # Map indicator code → safe column name
    col_map: dict[str, str] = {
        m["indicator_code"]: _safe_indicator_colname(m["indicator_code"])
        for m in all_meta
    }

    # Build the complete (country_code, year) grid so every combination is
    # present — even if a given indicator has no record for that cell.
    all_countries = (
        long_df[["country_code", "country_code_a3", "country_name"]]
        .drop_duplicates()
    )
    all_years = pd.DataFrame(
        {"year": sorted(long_df["year"].unique())}
    )
    # Cross-join: every country × every year
    full_grid = all_countries.merge(all_years, how="cross")

    log.debug(
        "Panel grid: %d countries × %d years = %d rows.",
        all_countries.shape[0],
        all_years.shape[0],
        full_grid.shape[0],
    )

    # Pivot each indicator separately and left-join onto the grid
    panel = full_grid.copy()

    for indicator_code, col_name in col_map.items():
        subset = long_df[long_df["indicator_code"] == indicator_code][
            ["country_code", "year", "value"]
        ].rename(columns={"value": col_name})

        panel = panel.merge(subset, on=["country_code", "year"], how="left")

        merged_nulls = panel[col_name].isna().sum()
        log.debug(
            "  After merging '%s' → '%s': %d NaN cells.",
            indicator_code,
            col_name,
            merged_nulls,
        )

    # Sort for readability: country alphabetically, then year ascending
    panel = panel.sort_values(["country_name", "year"]).reset_index(drop=True)

    return panel


# ---------------------------------------------------------------------------
# Missing-value annotation
# ---------------------------------------------------------------------------

def annotate_missing(panel: pd.DataFrame, indicator_col_names: list[str]) -> pd.DataFrame:
    """Append ``missing_count`` and ``missing_indicators`` audit columns.

    These columns make the extent of missingness immediately visible to any
    reader of the CSV without requiring a separate null-check step.

    ``missing_count`` is an integer (0 = fully observed row).
    ``missing_indicators`` is a comma-separated string of the column names
    that are NaN, or an empty string for complete rows.

    Values are **never modified** — this function only adds metadata.

    Args:
        panel: Wide panel DataFrame produced by :func:`pivot_to_panel`.
        indicator_col_names: Ordered list of the indicator column names
                             (underscore-sanitised codes).

    Returns:
        The same DataFrame with two new columns appended in-place.
    """
    indicator_cols = [c for c in indicator_col_names if c in panel.columns]
    missing_mask = panel[indicator_cols].isna()

    panel["missing_count"] = missing_mask.sum(axis=1).astype(int)
    panel["missing_indicators"] = missing_mask.apply(
        lambda row: ",".join(col for col in indicator_cols if row[col]),
        axis=1,
    )

    fully_observed = (panel["missing_count"] == 0).sum()
    partially_missing = (panel["missing_count"].between(1, len(indicator_cols) - 1)).sum()
    fully_missing = (panel["missing_count"] == len(indicator_cols)).sum()

    log.info(
        "Missing-value summary across %d rows:",
        len(panel),
    )
    log.info("  Fully observed (0 missing):         %d rows", fully_observed)
    log.info("  Partially missing (1–6 indicators): %d rows", partially_missing)
    log.info("  All indicators missing:             %d rows", fully_missing)

    return panel


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_panel(panel: pd.DataFrame, output_path: Path) -> Path:
    """Write the panel to *output_path* as a UTF-8 CSV with a header row.

    The parent directory is created if it does not already exist.

    Args:
        panel: Cleaned wide panel DataFrame.
        output_path: Destination ``.csv`` path.

    Returns:
        The resolved :class:`~pathlib.Path` of the written file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False, encoding="utf-8")
    log.info(
        "Panel saved → %s  (%d rows × %d columns).",
        output_path,
        len(panel),
        len(panel.columns),
    )
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_panel(
    raw_dir: Path = _RAW_DIR,
    output_path: Path = _OUTPUT_PATH,
) -> pd.DataFrame:
    """End-to-end pipeline: load → pivot → annotate → save.

    Args:
        raw_dir: Directory containing the raw World Bank JSON files.
        output_path: Destination CSV path for the processed panel.

    Returns:
        The final wide panel :class:`~pandas.DataFrame`.

    Raises:
        FileNotFoundError: If *raw_dir* does not exist.
        RuntimeError: If no usable JSON files are found.
    """
    # 1. Load
    all_meta, long_df = load_all_raw(raw_dir)

    # 2. Pivot to wide panel
    panel = pivot_to_panel(long_df, all_meta)

    # 3. Annotate missing values
    indicator_col_names = [
        _safe_indicator_colname(m["indicator_code"]) for m in all_meta
    ]
    panel = annotate_missing(panel, indicator_col_names)

    # 4. Enforce canonical column order
    id_cols = [c for c in _ID_COLS if c in panel.columns]
    indicator_cols = [c for c in indicator_col_names if c in panel.columns]
    audit_cols = ["missing_count", "missing_indicators"]
    ordered_cols = id_cols + indicator_cols + audit_cols
    panel = panel[ordered_cols]

    # 5. Save
    save_panel(panel, output_path)

    return panel


# ---------------------------------------------------------------------------
# Diagnostics helper (used by __main__ to print a readable summary)
# ---------------------------------------------------------------------------

def _print_summary(panel: pd.DataFrame, indicator_cols: list[str]) -> None:
    """Log a per-indicator null-rate table to INFO."""
    log.info("─── Per-indicator null rates ───")
    log.info("  %-30s  %6s / %-6s  (%s)", "Indicator", "NaN", "Total", "Rate")
    for col in indicator_cols:
        if col not in panel.columns:
            continue
        total = len(panel)
        null_n = panel[col].isna().sum()
        log.info("  %-30s  %6d / %-6d  (%.1f%%)", col, null_n, total, null_n / total * 100)

    log.info("─── Country coverage ───")
    coverage = (
        panel.groupby("country_name")["missing_count"]
        .agg(["mean", "max"])
        .rename(columns={"mean": "avg_missing", "max": "max_missing"})
        .round(2)
    )
    for country, row in coverage.iterrows():
        log.info("  %-30s  avg_missing=%.2f  max_missing=%d", country, row["avg_missing"], row["max_missing"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.data.clean_world_bank",
        description="Build a clean country-year panel from raw World Bank JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--raw-dir",
        metavar="PATH",
        default=str(_RAW_DIR),
        help=f"Directory containing raw JSON files (default: {_RAW_DIR}).",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=str(_OUTPUT_PATH),
        help=f"Destination CSV path (default: {_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, run the pipeline, print a summary, return exit code."""
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        panel = build_panel(
            raw_dir=Path(args.raw_dir),
            output_path=Path(args.output),
        )
    except (FileNotFoundError, RuntimeError) as exc:
        log.error("%s", exc)
        return 1

    indicator_cols = [
        c for c in panel.columns
        if c not in _ID_COLS + ["missing_count", "missing_indicators"]
    ]
    _print_summary(panel, indicator_cols)
    return 0


if __name__ == "__main__":
    sys.exit(main())
