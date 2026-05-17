"""Functions for loading raw data from various sources."""

import pandas as pd
from pathlib import Path


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_excel(path: str | Path, sheet_name: str = 0) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name)


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
