"""Geospatial map builders (Folium / Plotly)."""

import folium
import pandas as pd


MIDDLE_EAST_CENTER = (25.0, 45.0)


def base_map(zoom: int = 5) -> folium.Map:
    return folium.Map(location=MIDDLE_EAST_CENTER, zoom_start=zoom, tiles="OpenStreetMap")


def add_flow_lines(m: folium.Map, flows_df: pd.DataFrame) -> folium.Map:
    """Overlay supply route lines on a folium map.

    flows_df must have columns: origin_lat, origin_lon, dest_lat, dest_lon, volume.
    """
    for _, row in flows_df.iterrows():
        weight = max(1, int(row["volume"] / flows_df["volume"].max() * 8))
        folium.PolyLine(
            locations=[(row["origin_lat"], row["origin_lon"]), (row["dest_lat"], row["dest_lon"])],
            weight=weight,
            color="#e05c00",
            opacity=0.7,
        ).add_to(m)
    return m
