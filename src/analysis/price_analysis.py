"""Oil price time-series analysis: trends, correlations, and benchmarks."""

import pandas as pd
import numpy as np


def compute_rolling_average(series: pd.Series, window: int = 30) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def price_correlation_matrix(df: pd.DataFrame, benchmarks: list[str]) -> pd.DataFrame:
    return df[benchmarks].corr()


def annualised_volatility(series: pd.Series, periods_per_year: int = 252) -> float:
    log_returns = np.log(series / series.shift(1)).dropna()
    return float(log_returns.std() * np.sqrt(periods_per_year))
