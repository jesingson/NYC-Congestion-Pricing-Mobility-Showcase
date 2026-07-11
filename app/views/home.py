from __future__ import annotations

import streamlit as st

from app.data_access.loaders import CORE_METRICS, get_data_inventory
from app.utils.project_branding import inject_app_css

inject_app_css()

st.title("NYC Congestion Pricing Mobility Showcase")

st.markdown(
    """
    A question-led mobility showcase exploring how NYC travel patterns changed around the
    January 2025 congestion-pricing launch.

    The current release focuses on the **Raw Data Explorer**: temporal patterns, spatial patterns,
    zone-level profiles, and multimodal relationships built from the processed 1.3.1 mobility tables.
    """
)

st.subheader("How to use this app")

st.markdown(
    """
    Each page follows the same pattern:

    **Frozen answer view** — a curated default visualization that answers the page question.  
    **Explore this view** — controls that let you vary the metric, geography, period, or temporal bucket.
    """
)

st.subheader("Current raw-data questions")

st.markdown(
    """
    - **Did mobility change?**
    - **When did it change?**
    - **Where did it change?**
    - **Which zones changed most?**
    - **Do modes move together?**
    - **Where do modes disagree?**
    - **What happened in this Taxi Zone?**
    """
)

st.subheader("Core metrics")

st.markdown(
    """
    The app currently centers on six trusted core metrics:
    """
)

for metric in CORE_METRICS:
    st.markdown(f"- `{metric}`")

with st.expander("Local data inventory / app diagnostics", expanded=False):
    st.dataframe(get_data_inventory(), use_container_width=True)

with st.expander("Notes on excluded or contextual signals", expanded=False):
    st.markdown(
        """
        - `bus_trip_count` is not part of the core app backbone because it is heavily tied to scheduled service structure.
        - `traffic_volume` is not part of the core app backbone because traffic observations are sparse and uneven.
        - Bridge/Tunnel and Weather are contextual layers, not primary raw-data explorer metrics.
        """
    )