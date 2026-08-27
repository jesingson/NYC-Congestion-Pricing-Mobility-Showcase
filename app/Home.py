from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="NYC Congestion Pricing Mobility Showcase",
    layout="wide",
)

pages = {
    "Overview": [
        st.Page(
            "views/home.py",
            title="Home",
            icon=":material/home:",
        ),
    ],

    "Mobility Patterns": [
        st.Page(
            "views/raw_01_temporal_did_mobility_change.py",
            title="Temporal Overview",
            icon=":material/show_chart:",
        ),
        st.Page(
            "views/raw_02_temporal_when_did_it_change.py",
            title="Time-of-Day Patterns",
            icon=":material/schedule:",
        ),
        st.Page(
            "views/raw_03_spatial_where_did_it_change.py",
            title="Spatial Patterns",
            icon=":material/map:",
        ),
        st.Page(
            "views/raw_04_zone_rankings_changed_most.py",
            title="Zone Rankings",
            icon=":material/leaderboard:",
        ),
        st.Page(
            "views/raw_07_zone_profile_what_happened_here.py",
            title="Zone Profile",
            icon=":material/location_on:",
        ),
    ],

    "Relationships & Dynamics": [
        st.Page(
            "views/raw_05_mode_relationships_move_together.py",
            title="Mode Relationships",
            icon=":material/hub:",
        ),
        st.Page(
            "views/raw_06_mode_divergences_disagree.py",
            title="Mode Divergences",
            icon=":material/compare_arrows:",
        ),
        st.Page(
            "views/raw_08_rolling_mode_relationships.py",
            title="Rolling Relationships",
            icon=":material/timeline:",
        ),
        st.Page(
            "views/raw_09_weather_relationships.py",
            title="Weather Relationships",
            icon=":material/cloud:",
        ),
        st.Page(
            "views/raw_10_mobility_pulse.py",
            title="Mobility Pulse",
            icon=":material/travel_explore:",
        ),
        st.Page(
            "views/raw_11_cyclical_time.py",
            title="Cyclical Time",
            icon=":material/cycle:",
        ),
    ],

    "Diagnostics": [
        st.Page(
            "views/raw_12_anomaly_explorer.py",
            title="Anomaly Explorer",
            icon=":material/warning:",
        ),
    ],

    "Planned Work": [
        st.Page(
            "views/future_forecasting_counterfactual.py",
            title="Forecasting",
            icon=":material/query_stats:",
        ),
    ],
}

pg = st.navigation(pages)
pg.run()