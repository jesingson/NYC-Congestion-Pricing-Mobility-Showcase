from __future__ import annotations

from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).parent

st.set_page_config(
    page_title="NYC Congestion Pricing Mobility Showcase",
    layout="wide",
)

pages = {
    "Overview": [
        st.Page("views/home.py", title="Home"),
    ],
    "Raw Data Explorer": [
        st.Page(
            "views/raw_01_temporal_did_mobility_change.py",
            title="Temporal Overview",
        ),
        st.Page(
            "views/raw_02_temporal_when_did_it_change.py",
            title="Time-of-Day Patterns",
        ),
        st.Page(
            "views/raw_03_spatial_where_did_it_change.py",
            title="Spatial Patterns",
        ),
        st.Page(
            "views/raw_04_zone_rankings_changed_most.py",
            title="Zone Rankings",
        ),
        st.Page(
            "views/raw_05_mode_relationships_move_together.py",
            title="Mode Relationships",
        ),
        st.Page(
            "views/raw_06_mode_divergences_disagree.py",
            title="Mode Divergences",
        ),
        st.Page(
            "views/raw_07_zone_profile_what_happened_here.py",
            title="Zone Profile",
        ),
    ],
    "Future Layers": [
        st.Page(
            "views/future_mobility_discovery.py",
            title="Mobility Discovery",
        ),
        st.Page(
            "views/future_anomaly_prediction.py",
            title="Anomaly Detection",
        ),
        st.Page(
            "views/future_forecasting_counterfactual.py",
            title="Forecasting",
        ),
    ],
}

pg = st.navigation(pages)
pg.run()