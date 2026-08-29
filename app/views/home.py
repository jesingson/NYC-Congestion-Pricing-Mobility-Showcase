from __future__ import annotations

import streamlit as st

from app.data_access.loaders import get_data_inventory
from app.utils.project_branding import inject_app_css

inject_app_css()

st.title("NYC Congestion Pricing Mobility Showcase")

st.markdown(
    """
    Explore how travel across New York City changed around the January 2025
    congestion-pricing launch. The Showcase connects citywide trends, Taxi Zone
    patterns, mobility environments, relationships among transportation modes, and
    stress anomalies in one question-led experience.
    """
)

st.subheader("Choose an investigative path")

st.markdown(
    """
    **Mobility patterns** — See when and where demand, speed, and travel duration
    changed; compare Taxi Zones; or investigate the mobility environment that best
    describes a neighborhood.

    **Relationships and dynamics** — Examine whether modes move together, where they
    diverge, how those relationships evolve, and how mobility varies with weather and
    recurring time cycles.

    **Stress-anomaly diagnostics** — Find when and where mobility stress appeared,
    understand which modes combined, and inspect the observed-versus-expected evidence
    behind an individual event.
    """
)

st.subheader("How to use the pages")

st.markdown(
    """
    Most pages begin with a curated view that answers the page’s central question.
    Continue into the explorer to change the metric, geography, period, daypart, or
    comparison. Insight cards and interpretation boxes update with the current
    selection.

    Start broad with a temporal or spatial overview, then move to a Taxi Zone,
    mobility-environment, or anomaly deep dive when something warrants closer review.
    """
)

st.subheader("Mobility evidence in the Showcase")

st.markdown(
    """
    The analysis covers **Taxi, high-volume for-hire vehicles, Subway, and Bus** using
    trip activity, ridership, transfers, average speed, and average trip duration where
    each measure is available. Weather, policy geography, borough, and learned mobility
    environments provide context rather than interchangeable transportation measures.

    A **stress anomaly** is an observation with stress-aligned evidence: unusually high
    demand, slower movement, or longer travel duration relative to its modeled
    expectation. It identifies an unusual pattern—not its cause.
    """
)

with st.expander("Local data inventory / app diagnostics", expanded=False):
    st.dataframe(get_data_inventory(), width="stretch")

with st.expander("Notes on excluded or contextual signals", expanded=False):
    st.markdown(
        """
        - Bus trip activity is interpreted alongside scheduled-service structure rather
          than treated as interchangeable with for-hire demand.
        - Traffic-volume observations remain too sparse and uneven for the primary
          citywide mobility comparisons.
        - Weather and policy geography provide context; they do not by themselves
          explain why a mobility pattern occurred.
        """
    )
