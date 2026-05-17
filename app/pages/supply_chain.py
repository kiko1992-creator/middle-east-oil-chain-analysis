"""Supply chain dashboard page."""

import streamlit as st


def render() -> None:
    st.header("Supply Chain Analysis")
    st.info("Load data via the sidebar and run the supply chain pipeline to populate this view.")
