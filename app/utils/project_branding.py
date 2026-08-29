"""Shared visual branding constants and helpers for the NYC mobility showcase app."""

from __future__ import annotations

import plotly.graph_objects as go


BRAND_COLORS = {
    "dark_teal": "#006D77",
    "seafoam": "#83C5BE",
    "ice": "#EDF6F9",
    "pale_peach": "#FFDDD2",
    "terracotta": "#E29578",
}

BRAND_COLOR_SEQUENCE = [
    BRAND_COLORS["dark_teal"],
    BRAND_COLORS["terracotta"],
    BRAND_COLORS["seafoam"],
    BRAND_COLORS["pale_peach"],
    BRAND_COLORS["ice"],
]

BRAND_DIVERGING_SEQUENCE = [
    BRAND_COLORS["dark_teal"],
    BRAND_COLORS["seafoam"],
    BRAND_COLORS["ice"],
    BRAND_COLORS["pale_peach"],
    BRAND_COLORS["terracotta"],
]

BRAND_MAP_COLORS = {
    "primary": BRAND_COLORS["dark_teal"],
    "secondary": BRAND_COLORS["terracotta"],
    "supporting": BRAND_COLORS["seafoam"],
    "background": BRAND_COLORS["ice"],
    "soft_highlight": BRAND_COLORS["pale_peach"],
}

BRAND_PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "white",
        "plot_bgcolor": BRAND_COLORS["ice"],
        "font": {
            "color": BRAND_COLORS["dark_teal"],
            "size": 13,
        },
        "title": {
            "font": {
                "color": BRAND_COLORS["dark_teal"],
                "size": 18,
            }
        },
        "colorway": BRAND_COLOR_SEQUENCE,
        "margin": {"l": 50, "r": 30, "t": 60, "b": 50},
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"color": BRAND_COLORS["dark_teal"]},
        },
        "xaxis": {
            "gridcolor": "rgba(0, 109, 119, 0.18)",
            "zerolinecolor": "rgba(0, 109, 119, 0.25)",
            "tickfont": {"color": BRAND_COLORS["dark_teal"]},
            "title": {"font": {"color": BRAND_COLORS["dark_teal"]}},
        },
        "yaxis": {
            "gridcolor": "rgba(0, 109, 119, 0.18)",
            "zerolinecolor": "rgba(0, 109, 119, 0.25)",
            "tickfont": {"color": BRAND_COLORS["dark_teal"]},
            "title": {"font": {"color": BRAND_COLORS["dark_teal"]}},
        },
    }
}


def apply_branding(fig: go.Figure) -> go.Figure:
    """Apply shared Plotly branding to a figure."""
    fig.update_layout(**BRAND_PLOTLY_TEMPLATE["layout"])
    return fig


def inject_app_css() -> None:
    """Inject lightweight Streamlit CSS using the app brand palette."""
    import streamlit as st

    st.markdown(
        f"""
        <style>
        h1, h2, h3, h4, h5, h6 {{
            color: {BRAND_COLORS["dark_teal"]};
            font-weight: 750;
        }}

        .app-subtitle {{
            color: #335c67;
            font-size: 1.05rem;
            line-height: 1.45;
            margin-bottom: 1.25rem;
        }}

        /* Sidebar container */
        [data-testid="stSidebar"] {{
            background-color: #F7FBFC;
            border-right: 1px solid rgba(0, 109, 119, 0.18);
        }}

        /* Sidebar nav text */
        [data-testid="stSidebar"] a,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] p {{
            color: #003F46 !important;
            font-weight: 600;
        }}

        /* Selected sidebar nav item */
        [data-testid="stSidebar"] [aria-current="page"] {{
            background-color: {BRAND_COLORS["seafoam"]} !important;
            color: #003F46 !important;
            border-radius: 0.45rem;
            font-weight: 700;
        }}

        .metric-card {{
            background-color: {BRAND_COLORS["ice"]};
            border: 1px solid {BRAND_COLORS["seafoam"]};
            border-radius: 0.75rem;
            padding: 1rem;
            margin-bottom: 0.75rem;
        }}

        .soft-callout {{
            background-color: {BRAND_COLORS["pale_peach"]};
            border-left: 5px solid {BRAND_COLORS["terracotta"]};
            border-radius: 0.5rem;
            padding: 0.85rem 1rem;
            margin: 1rem 0;
            color: #003F46;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_chart_insight(text: str) -> None:
    """Render the standard data-derived takeaway directly below a visual."""
    import streamlit as st

    cleaned_text = (
        str(text)
        .replace("<strong>", "**")
        .replace("</strong>", "**")
        .replace("<br>", "  \n")
        .replace("<br/>", "  \n")
        .strip()
    )
    if not cleaned_text:
        return

    st.info(f"**Takeaway —** {cleaned_text}")
