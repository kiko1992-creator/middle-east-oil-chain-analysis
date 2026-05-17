"""Data cleaning and transformation pipelines."""

import pandas as pd


def normalize_country_names(df: pd.DataFrame, col: str = "country") -> pd.DataFrame:
    aliases = {
        "KSA": "Saudi Arabia",
        "UAE": "United Arab Emirates",
        "U.A.E.": "United Arab Emirates",
    }
    df[col] = df[col].replace(aliases)
    return df


def resample_to_monthly(df: pd.DataFrame, date_col: str = "date", value_col: str = "value") -> pd.DataFrame:
    df[date_col] = pd.to_datetime(df[date_col])
    return df.set_index(date_col)[value_col].resample("MS").mean().reset_index()


def remove_outliers_iqr(df: pd.DataFrame, col: str, factor: float = 1.5) -> pd.DataFrame:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    mask = df[col].between(q1 - factor * iqr, q3 + factor * iqr)
    return df[mask].copy()
