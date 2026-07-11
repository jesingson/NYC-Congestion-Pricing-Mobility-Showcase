from __future__ import annotations

import streamlit as st

from app.utils.project_branding import inject_app_css


def render_question_page(
    *,
    section: str,
    question: str,
    page_goal: str,
    frozen_view: str,
    explore_controls: list[str],
    notebook_source: str,
) -> None:
    """Render the common structure for question-led exhibit pages."""
    inject_app_css()

    st.caption("Raw Data Explorer")
    st.title(f"{section}: {question}")

    st.markdown(page_goal)

    st.divider()

    st.subheader("Frozen answer view")
    st.markdown(
        """
        This section will show the curated default view that directly answers the page question.
        """
    )
    st.info(frozen_view)

    st.divider()

    st.subheader("Explore this view")
    st.markdown(
        """
        This section will reuse the same visual logic, but expose controls so users can explore
        nearby versions of the question.
        """
    )

    with st.expander("Planned interaction controls", expanded=True):
        for control in explore_controls:
            st.markdown(f"- {control}")

    with st.expander("Notebook source / design rationale", expanded=False):
        st.markdown(notebook_source)


def render_future_page(*, title: str, description: str) -> None:
    """Render a lightweight placeholder for future app layers."""
    inject_app_css()

    st.caption("Future Layer")
    st.title(title)
    st.markdown(description)
    st.info(
        "This page will be added after the underlying notebooks produce app-ready outputs."
    )