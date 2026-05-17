"""Price analysis dashboard page."""

import streamlit as st


def render() -> None:
    st.header("Oil Price Analysis")
    st.info("Upload price data to visualise benchmarks and volatility metrics.")
