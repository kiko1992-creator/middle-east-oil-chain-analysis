"""Time-series and statistical chart builders (Plotly)."""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def line_chart(df: pd.DataFrame, x: str, y: str, color: str | None = None, title: str = "") -> go.Figure:
    return px.line(df, x=x, y=y, color=color, title=title, template="plotly_white")


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str = "") -> go.Figure:
    return px.bar(df, x=x, y=y, title=title, template="plotly_white")


def correlation_heatmap(corr_matrix: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    return px.imshow(corr_matrix, text_auto=True, title=title, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
