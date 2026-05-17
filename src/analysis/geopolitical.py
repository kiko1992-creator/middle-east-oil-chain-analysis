"""Geopolitical risk scoring and event-impact analysis."""

import pandas as pd


def score_events(events_df: pd.DataFrame, weight_col: str = "severity") -> pd.DataFrame:
    events_df = events_df.copy()
    events_df["risk_score"] = events_df[weight_col] / events_df[weight_col].max()
    return events_df


def merge_risk_with_prices(prices_df: pd.DataFrame, risk_df: pd.DataFrame, on: str = "date") -> pd.DataFrame:
    prices_df[on] = pd.to_datetime(prices_df[on])
    risk_df[on] = pd.to_datetime(risk_df[on])
    return pd.merge_asof(prices_df.sort_values(on), risk_df.sort_values(on), on=on)
