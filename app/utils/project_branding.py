"""Shared visual branding constants and helpers for the NYC mobility showcase app."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from html import escape

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


# ---------------------------------------------------------------------
# Shared exploration-area contract
# ---------------------------------------------------------------------
#
# WHY:
# Pale peach is reserved as the background for reader-led exploration.
# It is uncommon as a large Showcase surface, so it quietly signals a
# change from curated storytelling to interactive investigation.
#
# The wash is intentionally faint. Controls, charts, expanders, and Takeaway
# cards remain opaque so the exploration color acts as a section-level cue
# rather than tinting every component inside it.
EXPLORATION_KEY_SUFFIX = "_exploration_area"

EXPLORATION_WASH = "rgba(255, 221, 210, 0.20)"
EXPLORATION_BORDER = "rgba(226, 149, 120, 0.38)"
EXPLORATION_CONTROL_BORDER = "rgba(0, 109, 119, 0.26)"
EXPLORATION_CONTROL_FOCUS = "rgba(0, 109, 119, 0.16)"
EXPLORATION_CHART_BORDER = "rgba(226, 149, 120, 0.30)"
EXPLORATION_EXPANDER_BORDER = "rgba(226, 149, 120, 0.24)"

TAKEAWAY_BACKGROUND = "#E7F2FF"
TAKEAWAY_BORDER = "#BCD8F0"
TAKEAWAY_TEXT = "#075A9C"


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

        /* -------------------------------------------------------------
           Interactive-exploration regions
           -------------------------------------------------------------
           WHY:
           The faint pale-peach field marks a change in interaction mode.
           Child components stay opaque so they remain visually crisp and
           the wash reads as a section boundary rather than a data encoding.
        */

        div[class*="st-key-"][class*="{EXPLORATION_KEY_SUFFIX}"] {{
            background: {EXPLORATION_WASH};
            border-radius: 0.72rem;
        }}

        div[class*="st-key-"][class*="{EXPLORATION_KEY_SUFFIX}"]
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: {EXPLORATION_WASH};
            border-color: {EXPLORATION_BORDER} !important;
            border-top: 2px solid rgba(226, 149, 120, 0.50) !important;
            border-radius: 0.72rem;
        }}

        /*
        Keep Plotly charts neutral inside the warm exploration field.

        White chart paper and the existing ice plotting surface preserve the
        chart hierarchy. The hairline peach edge simply nests the chart within
        the exploration region.
        */
        div[class*="st-key-"][class*="{EXPLORATION_KEY_SUFFIX}"]
        [data-testid="stPlotlyChart"] {{
            background: #FFFFFF;
            border: 1px solid {EXPLORATION_CHART_BORDER};
            border-radius: 0.50rem;
            box-sizing: border-box;
            overflow: hidden;
        }}

        /*
        Make select controls read as real controls rather than washed-out
        extensions of the exploration surface.

        Streamlit 1.59+ no longer uses BaseWeb for st.selectbox. The current
        React Aria implementation renders the visible control as a role="group"
        inside [data-testid="stSelectbox"], with a transparent combobox input.
        Styling that group therefore changes the actual surface the reader sees.
        */
        div[class*="st-key-"][class*="{EXPLORATION_KEY_SUFFIX}"]
        [data-testid="stSelectbox"] [role="group"] {{
            background-color: #FFFFFF !important;
            border: 1px solid rgba(0, 109, 119, 0.34) !important;
            border-radius: 0.50rem !important;
            box-shadow: none !important;
        }}

        div[class*="st-key-"][class*="{EXPLORATION_KEY_SUFFIX}"]
        [data-testid="stSelectbox"] input[role="combobox"] {{
            background-color: transparent !important;
        }}

        div[class*="st-key-"][class*="{EXPLORATION_KEY_SUFFIX}"]
        [data-testid="stSelectbox"] [role="group"]:focus-within {{
            border-color: {BRAND_COLORS["dark_teal"]} !important;
            box-shadow: 0 0 0 2px {EXPLORATION_CONTROL_FOCUS} !important;
        }}

        /*
        Backward-compatible fallback for Streamlit versions that still use
        BaseWeb Select. This does not affect the current React Aria control.
        */
        div[class*="st-key-"][class*="{EXPLORATION_KEY_SUFFIX}"]
        div[data-baseweb="select"] > div {{
            background-color: #FFFFFF !important;
            border: 1px solid rgba(0, 109, 119, 0.34) !important;
            border-radius: 0.50rem !important;
            box-shadow: none !important;
        }}

        div[class*="st-key-"][class*="{EXPLORATION_KEY_SUFFIX}"]
        [data-testid="stNumberInput"] input,
        div[class*="st-key-"][class*="{EXPLORATION_KEY_SUFFIX}"]
        [data-testid="stTextInput"] input,
        div[class*="st-key-"][class*="{EXPLORATION_KEY_SUFFIX}"]
        [data-testid="stDateInput"] input {{
            background-color: #FFFFFF !important;
            border: 1px solid {EXPLORATION_CONTROL_BORDER} !important;
            border-radius: 0.50rem !important;
        }}

        /*
        Expanders are analytical controls too. Keeping their header and body
        white prevents Streamlit's neutral gray surface from muddying the
        warm exploration background.
        */
        div[class*="st-key-"][class*="{EXPLORATION_KEY_SUFFIX}"]
        [data-testid="stExpander"] {{
            background-color: #FFFFFF !important;
            border-color: {EXPLORATION_EXPANDER_BORDER} !important;
        }}

        div[class*="st-key-"][class*="{EXPLORATION_KEY_SUFFIX}"]
        [data-testid="stExpander"] details,
        div[class*="st-key-"][class*="{EXPLORATION_KEY_SUFFIX}"]
        [data-testid="stExpander"] summary {{
            background-color: #FFFFFF !important;
        }}

        /*
        Preserve the familiar blue Takeaway treatment inside the peach field.

        Streamlit's default info card can be translucent, which allows the
        warm parent background to shift it toward lavender-gray. An opaque
        blue surface keeps Takeaways visually distinct and consistent.
        */
        div[class*="st-key-"][class*="{EXPLORATION_KEY_SUFFIX}"]
        [data-testid="stAlert"] {{
            background-color: {TAKEAWAY_BACKGROUND} !important;
            border: 1px solid {TAKEAWAY_BORDER} !important;
        }}

        div[class*="st-key-"][class*="{EXPLORATION_KEY_SUFFIX}"]
        [data-testid="stAlert"] p,
        div[class*="st-key-"][class*="{EXPLORATION_KEY_SUFFIX}"]
        [data-testid="stAlert"] strong {{
            color: {TAKEAWAY_TEXT} !important;
        }}

        .showcase-exploration-header {{
            border-bottom: 1px solid rgba(226, 149, 120, 0.26);
            margin: 0 0 1.15rem 0;
            padding: 0.10rem 0 0.95rem 0;
        }}

        .showcase-exploration-kicker {{
            color: {BRAND_COLORS["dark_teal"]};
            font-size: 0.75rem;
            font-weight: 750;
            letter-spacing: 0.09em;
            margin-bottom: 0.20rem;
            text-transform: uppercase;
        }}

        .showcase-exploration-title {{
            color: #003F46;
            font-size: 1.28rem;
            font-weight: 750;
            line-height: 1.25;
            margin-bottom: 0.28rem;
        }}

        .showcase-exploration-copy {{
            color: #243238;
            font-size: 0.98rem;
            line-height: 1.5;
            margin: 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@contextmanager
def exploration_section(
    *,
    key: str,
    title: str,
    description: str,
    kicker: str = "Interactive exploration",
) -> Iterator[None]:
    """
    Render one visually distinct reader-controlled exploration region.

    The enclosing Streamlit key intentionally follows a shared naming contract
    so inject_app_css() can apply the same subtle treatment on every Showcase
    page without page-specific CSS.
    """
    import streamlit as st

    if not key.endswith(EXPLORATION_KEY_SUFFIX):
        raise ValueError(
            "Exploration-section keys must end with "
            f"'{EXPLORATION_KEY_SUFFIX}' so shared branding can identify them."
        )

    safe_kicker = escape(str(kicker))
    safe_title = escape(str(title))
    safe_description = escape(str(description))

    with st.container(
        border=True,
        key=key,
    ):
        st.markdown(
            f"""
            <div class="showcase-exploration-header">
                <div class="showcase-exploration-kicker">
                    {safe_kicker}
                </div>
                <div class="showcase-exploration-title">
                    {safe_title}
                </div>
                <p class="showcase-exploration-copy">
                    {safe_description}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        yield


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
