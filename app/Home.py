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

    "Mobility Types & Drivers": [
        st.Page(
            "views/raw_16_mobility_environment_profiler.py",
            title="Mobility Environment Profiler",
            icon=":material/account_tree:",
        ),
        st.Page(
            "views/raw_26_mobility_day_types.py",
            title="Mobility Day Types",
            icon=":material/calendar_month:",
        ),
        st.Page(
            "views/raw_27_mobility_drivers.py",
            title="Mobility Drivers",
            icon=":material/stacked_line_chart:",
        ),
    ],

    "Stress & Anomalies": [
        st.Page(
            "views/raw_12_stress_anomalies_over_time.py",
            title="Stress Anomaly Temporal Explorer",
            icon=":material/monitoring:",
        ),
        st.Page(
            "views/raw_13_stress_anomaly_spatial_explorer.py",
            title="Stress Anomaly Spatial Explorer",
            icon=":material/location_searching:",
        ),
        st.Page(
            "views/raw_14_mobility_stress_patterns.py",
            title="Stress Anomaly Patterns",
            icon=":material/analytics:",
        ),
        st.Page(
            "views/raw_15_stress_anomaly_event_profiler.py",
            title="Stress Anomaly Deep Dive",
            icon=":material/troubleshoot:",
        ),
        st.Page(
            "views/raw_17_weather_stress_episode_explorer.py",
            title="Weather and Stress Episodes",
            icon=":material/weather_mix:",
        ),
    ],

    "Forecasting & Reliability": [
        st.Page(
            "views/raw_18_forecast_scorecard.py",
            title="Forecast Scorecard",
            icon=":material/query_stats:",
        ),
        st.Page(
            "views/raw_19_forecast_reliability.py",
            title="Forecast Reliability",
            icon=":material/verified:",
        ),
        st.Page(
            "views/raw_20_forecast_wins_misses.py",
            title="Forecast Wins & Misses",
            icon=":material/balance:",
        ),
    ],

    "No-CP Counterfactual Mobility": [
        st.Page(
            "views/raw_21_counterfactual_overview.py",
            title="Counterfactual Overview",
            icon=":material/alt_route:",
        ),
        st.Page(
            "views/raw_22_counterfactual_geography.py",
            title="Counterfactual Geography",
            icon=":material/layers:",
        ),
        st.Page(
            "views/raw_23_multimodal_counterfactuals.py",
            title="Multimodal Counterfactuals",
            icon=":material/route:",
        ),
        st.Page(
            "views/raw_24_counterfactual_robustness.py",
            title="Counterfactual Robustness",
            icon=":material/fact_check:",
        ),
        st.Page(
            "views/raw_25_counterfactual_calibration.py",
            title="Gap Calibration",
            icon=":material/straighten:",
        ),
    ],

    "About the Project": [
        st.Page(
            "views/about_source_pipeline.py",
            title="About the Source Pipeline",
            icon=":material/account_tree:",
        ),
    ],
}

pg = st.navigation(pages)
pg.run()
