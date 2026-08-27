from __future__ import annotations

import streamlit as st

from app.utils.project_branding import inject_app_css


inject_app_css()

st.caption("PAGE 12")
st.title("Anomaly Explorer")

st.write(
    "Investigate where unusual mobility events occurred, what type of stress they represent, "
    "and which metrics contributed to each event."
)

st.info(
    "This page is being rebuilt around the final 3.3.6 anomaly surface. "
    "Interactive anomaly diagnostics are coming next."
)

st.markdown("### Planned diagnostic views")
st.markdown(
    """
    - Explore anomalies by Taxi Zone and time period.
    - Distinguish congestion-oriented, demand-oriented, combined, and mixed stress events.
    - Identify the mobility metrics and modes that contributed to each anomaly.
    - Compare observed values with expected values and standardized residuals.
    - Examine anomaly frequency, recurrence, persistence, and pre/post congestion-pricing patterns.
    """
)