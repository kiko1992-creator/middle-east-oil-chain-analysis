"""
CSV export utility for Streamlit pages — Priority 3A.

Provides a single helper that renders a Streamlit download button for any
DataFrame, stamping each export with the model version read from
config/settings.yaml.

Usage
-----
    from src.app.export import make_csv_download_button

    make_csv_download_button(df, filename="my_table.csv", label="Download table")
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Config path ────────────────────────────────────────────────────────────────

_ROOT        = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _ROOT / "config" / "settings.yaml"

_UNKNOWN_VERSION = "unknown"


def _read_model_version() -> str:
    """Read project.version from config/settings.yaml.

    Returns "unknown" on any read or parse failure so the dashboard never
    crashes due to a missing config file.
    """
    if not _CONFIG_PATH.exists():
        return _UNKNOWN_VERSION
    try:
        import yaml  # type: ignore[import-untyped]
        with _CONFIG_PATH.open(encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return str(cfg.get("project", {}).get("version", _UNKNOWN_VERSION))
    except Exception:
        return _UNKNOWN_VERSION


# Cache at module level so every button in the session reuses the same value
_MODEL_VERSION: str = _read_model_version()


def make_csv_download_button(
    df: pd.DataFrame,
    filename: str,
    label: str = "Download CSV",
    help: str | None = None,
) -> None:
    """Render a Streamlit download button that exports *df* as a UTF-8 CSV.

    The filename is prefixed with the model version from config/settings.yaml
    so downloaded files are self-documenting (e.g. ``v0.1.0_my_table.csv``).

    Args:
        df:       DataFrame to export.
        filename: Base filename including ``.csv`` extension.
        label:    Button label shown to the user.
        help:     Optional tooltip text.
    """
    versioned_name = f"v{_MODEL_VERSION}_{filename}"
    export_df = df.copy()
    export_df["export_timestamp"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    export_df["model_version"] = _MODEL_VERSION
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=versioned_name,
        mime="text/csv",
        help=help,
    )
