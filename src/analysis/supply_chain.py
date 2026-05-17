"""Supply chain flow analysis: production, transit, and export volumes."""

import pandas as pd


def compute_net_exports(production: pd.Series, domestic_consumption: pd.Series) -> pd.Series:
    return production - domestic_consumption


def aggregate_by_country(df: pd.DataFrame, value_col: str = "volume_mbd") -> pd.DataFrame:
    return df.groupby("country")[value_col].sum().reset_index()


def identify_chokepoints(df: pd.DataFrame, route_col: str = "route") -> pd.DataFrame:
    """Return routes that appear in > 20 % of total shipments."""
    counts = df[route_col].value_counts(normalize=True)
    return counts[counts > 0.20].reset_index().rename(columns={"index": route_col, route_col: "share"})
