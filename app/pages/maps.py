"""Geospatial maps dashboard page."""

import streamlit as st
from streamlit_folium import st_folium

from src.visualization.maps import base_map


def render() -> None:
    st.header("Supply Route Maps")
    m = base_map()
    st_folium(m, width=900, height=550)
