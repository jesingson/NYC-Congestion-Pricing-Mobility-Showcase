"""
Raw 23 — How did the modes differ across NYC?

The page compares Taxi demand, Taxi speed, FHVHV demand, FHVHV speed, and
Subway ridership against the estimated no-congestion-pricing path on a common
Pre-CP forecast-error scale.
"""

from __future__ import annotations

import base64
import re
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from app.data_access.counterfactuals import (
    GROUPINGS,
    HORIZONS,
    METRIC_LABELS,
    METRIC_MODALITY,
    METRIC_ORDER,
    PERIODS,
    load_primary_counterfactual_surface,
    get_child_taxi_zone_profiles,
    get_complete_profiles,
    get_counterfactual_profiles,
    get_profile_label,
    get_profile_label_column,
    get_same_borough_taxi_zone_profiles,
    get_same_level_peers,
    get_strongest_profiles,
    load_counterfactual_runtime_qa,
)
from app.utils.project_branding import (
    BRAND_COLORS,
    apply_branding,
    exploration_section,
    inject_app_css,
    render_chart_insight,
)


# ---------------------------------------------------------------------
# Page contract
# ---------------------------------------------------------------------

PAGE_CAPTION = "MULTIMODAL COUNTERFACTUALS PROFILER"
PAGE_TITLE = "How did the modes differ across NYC?"
PAGE_SUBTITLE = (
    "Compare the estimated no-congestion-pricing world with what NYC actually "
    "observed across Taxi, FHVHV, and Subway mobility."
)

APP_DIR = Path(__file__).resolve().parents[1]
IMAGE_DIR = APP_DIR / "images"

SVG_PATHS = {
    "taxi": IMAGE_DIR / "taxi_duotone.svg",
    "fhvhv": IMAGE_DIR / "fhvhv_duotone.svg",
    "subway": IMAGE_DIR / "subway_duotone.svg",
}

METRIC_SHORT = {
    "taxi_trip_count": "Taxi trips",
    "taxi_avg_trip_speed": "Taxi speed",
    "fhvhv_trip_count": "FHVHV trips",
    "fhvhv_avg_trip_speed": "FHVHV speed",
    "subway_ridership": "Subway",
}


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------

inject_app_css()

st.markdown(
    f"""
    <style>
    .raw23-card {{
        width: 100%;
        box-sizing: border-box;
        overflow: hidden;
        border: 1px solid rgba(0, 109, 119, 0.17);
        border-radius: 0.9rem;
        background: white;
        padding: 0.95rem 1rem 0.8rem 1rem;
        min-height: 17rem;
    }}

    .raw23-card-title {{
        color: #003F46;
        font-size: 1rem;
        font-weight: 760;
        line-height: 1.25;
        margin-bottom: 0.22rem;
    }}

    .raw23-card-subtitle {{
        color: #657B81;
        font-size: 0.73rem;
        line-height: 1.35;
        margin-bottom: 0.68rem;
    }}

    .raw23-axis-header,
    .raw23-row,
    .raw23-axis-footer {{
        width: 100%;
        box-sizing: border-box;
        display: grid;
        grid-template-columns: 1.4rem 5.9rem minmax(0, 1fr) 4rem;
        align-items: center;
        column-gap: 0.4rem;
    }}

    .raw23-axis-header {{
        margin-bottom: 0.1rem;
    }}

    .raw23-axis-label {{
        color: #64797F;
        font-size: 0.64rem;
        font-weight: 700;
    }}

    .raw23-axis-label.measure {{
        grid-column: 2;
    }}

    .raw23-axis-label.scale {{
        grid-column: 3;
        text-align: center;
    }}

    .raw23-row {{
        margin: 0.4rem 0;
    }}

    .raw23-row > * {{
        min-width: 0;
    }}

    .raw23-left-icon {{
        width: 1.12rem;
        height: 1.12rem;
        display: block;
        margin: auto;
    }}

    .raw23-mode-label {{
        color: #4D656C;
        font-size: 0.73rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}

    .raw23-track {{
        width: 100%;
        min-width: 0;
        position: relative;
        height: 1.42rem;
        border-radius: 0.33rem;
        background:
            linear-gradient(
                to right,
                rgba(0, 109, 119, 0.065) 0%,
                rgba(0, 109, 119, 0.023) 49.35%,
                rgba(226, 149, 120, 0.023) 50.65%,
                rgba(226, 149, 120, 0.065) 100%
            );
        overflow: hidden;
    }}

    .raw23-zero {{
        position: absolute;
        left: 50%;
        top: 0;
        bottom: 0;
        border-left: 1px solid rgba(0, 63, 70, 0.62);
        z-index: 4;
    }}

    .raw23-window {{
        position: absolute;
        top: 0;
        height: 100%;
        overflow: hidden;
        z-index: 3;
    }}

    .raw23-window.negative {{
        right: 50%;
    }}

    .raw23-window.positive {{
        left: 50%;
    }}

    .raw23-run {{
        position: absolute;
        top: 0.10rem;
        height: 1.22rem;
        display: flex;
        align-items: center;
        gap: 0.05rem;
        white-space: nowrap;
    }}

    .raw23-window.negative .raw23-run {{
        right: 0;
        flex-direction: row-reverse;
    }}

    .raw23-window.positive .raw23-run {{
        left: 0;
    }}

    .raw23-mark-icon {{
        width: 0.92rem;
        height: 0.92rem;
        flex: 0 0 auto;
        display: inline-block;
    }}

    .raw23-value {{
        color: #496168;
        font-size: 0.70rem;
        font-variant-numeric: tabular-nums;
        text-align: right;
        white-space: nowrap;
    }}

    .raw23-axis-footer {{
        margin-top: 0.18rem;
    }}

    .raw23-axis-ticks {{
        grid-column: 3;
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        color: #6A7E84;
        font-size: 0.64rem;
        font-variant-numeric: tabular-nums;
    }}

    .raw23-axis-ticks span:nth-child(1) {{
        text-align: left;
    }}

    .raw23-axis-ticks span:nth-child(2) {{
        text-align: center;
    }}

    .raw23-axis-ticks span:nth-child(3) {{
        text-align: right;
    }}

    .raw23-card-caption {{
        color: #687D83;
        font-size: 0.69rem;
        line-height: 1.35;
        margin-top: 0.45rem;
    }}

    .raw23-direction-box {{
        border-left: 4px solid {BRAND_COLORS["dark_teal"]};
        background: rgba(237, 246, 249, 0.72);
        border-radius: 0 0.7rem 0.7rem 0;
        padding: 0.76rem 0.9rem;
        color: #31535B;
        font-size: 0.9rem;
        line-height: 1.5;
        margin: 0.6rem 0 0.95rem 0;
    }}

    .raw23-card-insight {{
        border-left: 3px solid #006D77;
        border-radius: 0 0.48rem 0.48rem 0;
        background: rgba(237, 246, 249, 0.82);
        color: #35565E;
        font-size: 0.74rem;
        line-height: 1.38;
        padding: 0.48rem 0.60rem;
        margin: 0.34rem 0 0.88rem 0;
    }}


    .raw23-tab-intro {{
        color: #526A71;
        font-size: 0.88rem;
        line-height: 1.5;
        margin: 0.2rem 0 0.65rem 0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Asset helpers
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_svg_assets() -> dict[str, str]:
    """Load local branded SVGs as data URIs."""
    result = {}

    for key, path in SVG_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Expected Raw 23 SVG asset was not found: {path}"
            )

        encoded = base64.b64encode(
            path.read_text(
                encoding="utf-8"
            )
            .strip()
            .encode("utf-8")
        ).decode("ascii")

        result[key] = (
            "data:image/svg+xml;base64,"
            + encoded
        )

    return result


def nice_scale_limit(
    profiles: pd.DataFrame,
) -> float:
    """Return one unclipped symmetric scale covering every displayed value."""
    if profiles.empty:
        return 1.0

    values = (
        profiles[
            METRIC_ORDER
        ]
        .abs()
        .to_numpy(
            dtype=float
        )
        .ravel()
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return 1.0

    maximum = float(
        values.max()
    )

    if maximum <= 1:
        return 1.0
    if maximum <= 10:
        return float(
            np.ceil(maximum)
        )
    if maximum <= 25:
        return float(
            np.ceil(maximum / 5.0) * 5.0
        )

    return float(
        np.ceil(maximum / 10.0) * 10.0
    )


def icon_img_html(
    data_uri: str,
    *,
    css_class: str,
) -> str:
    """Render one local SVG data URI."""
    return (
        f'<img src="{data_uri}" '
        f'class="{css_class}" alt="">'
    )


def icon_run_html(
    data_uri: str,
    *,
    repeats: int = 90,
) -> str:
    """Build a long icon strip revealed by the quantitative window."""
    return "".join(
        icon_img_html(
            data_uri,
            css_class="raw23-mark-icon",
        )
        for _ in range(repeats)
    )


# ---------------------------------------------------------------------
# Interpretation helpers
# ---------------------------------------------------------------------

def contrast_clause(
    metric: str,
    value: float,
) -> str:
    """Translate one signed gap into plain mobility language."""
    if metric == "taxi_trip_count":
        if value < 0:
            return "fewer Taxi trips than observed"
        if value > 0:
            return "more Taxi trips than observed"
        return "about the same number of Taxi trips as observed"

    if metric == "fhvhv_trip_count":
        if value < 0:
            return "fewer FHVHV trips than observed"
        if value > 0:
            return "more FHVHV trips than observed"
        return "about the same number of FHVHV trips as observed"

    if metric == "subway_ridership":
        if value < 0:
            return "lower Subway ridership than observed"
        if value > 0:
            return "higher Subway ridership than observed"
        return "about the same Subway ridership as observed"

    if metric == "taxi_avg_trip_speed":
        if value < 0:
            return "slower Taxi speeds than observed"
        if value > 0:
            return "faster Taxi speeds than observed"
        return "about the same Taxi speeds as observed"

    if metric == "fhvhv_avg_trip_speed":
        if value < 0:
            return "slower FHVHV speeds than observed"
        if value > 0:
            return "faster FHVHV speeds than observed"
        return "about the same FHVHV speeds as observed"

    if value < 0:
        return "a lower value than observed"
    if value > 0:
        return "a higher value than observed"

    return "about the same value as observed"


def relation_phrase(
    metric: str,
    value: float,
) -> str:
    """Spell out the no-CP-versus-observed contrast in plain English."""
    magnitude = abs(
        float(value)
    )

    clause = contrast_clause(
        metric,
        float(value),
    )

    if float(value) == 0:
        return (
            f"The estimated no-CP world shows {clause}."
        )

    return (
        f"The estimated no-CP world shows {clause} by "
        f"{magnitude:.3f} Pre-CP MAE units."
    )


def strongest_metric(
    profile: pd.Series,
) -> tuple[str | None, float]:
    """Return the profile's largest absolute available component."""
    available = {
        metric: float(
            profile[
                metric
            ]
        )
        for metric in METRIC_ORDER
        if pd.notna(
            profile.get(
                metric
            )
        )
    }

    if not available:
        return None, np.nan

    metric = max(
        available,
        key=lambda key: abs(
            available[
                key
            ]
        ),
    )

    return metric, available[
        metric
    ]


def profile_editorial_sentence(
    profile: pd.Series,
    *,
    label: str,
) -> str:
    """Explain the largest component without making readers decode the sign."""
    metric, value = strongest_metric(
        profile
    )

    if metric is None:
        return (
            f"{label} does not have an available multimodal profile."
        )

    return (
        f"**{label}:** {relation_phrase(metric, value)} "
        f"This is the largest standardized difference in its five-measure profile."
    )


def set_editorial_insight(
    profiles: pd.DataFrame,
    *,
    grouping_name: str,
) -> str:
    """Lead with the strongest profile and translate its sign explicitly."""
    complete = get_complete_profiles(
        profiles
    )

    if complete.empty:
        return (
            "No complete five-measure profiles are available for this view."
        )

    strongest = complete.sort_values(
        "multimodal_rms",
        ascending=False,
    ).iloc[0]

    label_column = get_profile_label_column(
        grouping_name
    )

    label = get_profile_label(
        grouping_name,
        strongest[
            label_column
        ],
    )

    return profile_editorial_sentence(
        strongest,
        label=label,
    )


def profile_card_insight(
    profile: pd.Series,
    *,
    label: str,
) -> str:
    """Explain the two largest components of one card without sign decoding."""
    available = [
        (
            metric,
            float(profile[metric]),
        )
        for metric in METRIC_ORDER
        if pd.notna(
            profile.get(metric)
        )
    ]

    if not available:
        return (
            f"**{label}:** No mobility measures are available."
        )

    ranked = sorted(
        available,
        key=lambda item: abs(
            item[1]
        ),
        reverse=True,
    )

    first_metric, first_value = ranked[0]

    first_sentence = (
        f"**{label}:** The estimated no-CP world shows "
        f"{contrast_clause(first_metric, first_value)} by "
        f"{abs(first_value):.3f} Pre-CP MAE units — the largest gap "
        "in this profile."
    )

    if len(ranked) == 1:
        return first_sentence

    second_metric, second_value = ranked[1]

    return (
        first_sentence
        + " The next-largest difference is "
        + f"{contrast_clause(second_metric, second_value)} by "
        + f"{abs(second_value):.3f} MAE units."
    )


def render_profile_card_insight(
    profile: pd.Series,
    *,
    label: str,
) -> None:
    """Render the compact data-driven interpretation beneath one card."""
    insight = profile_card_insight(
        profile,
        label=label,
    )

    safe = escape(
        insight.replace(
            "**",
            "",
        )
    )

    st.markdown(
        f'<div class="raw23-card-insight">{safe}</div>',
        unsafe_allow_html=True,
    )


def group_story_insight(
    profiles: pd.DataFrame,
    *,
    grouping_name: str,
) -> str:
    """Tell the main story of one displayed comparison set."""
    complete = get_complete_profiles(
        profiles
    )

    if complete.empty:
        return (
            "No complete five-measure profiles are available for this view."
        )

    negative_metrics = []
    positive_metrics = []

    for metric in METRIC_ORDER:
        values = pd.to_numeric(
            complete[
                metric
            ],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        if values.lt(0).all():
            negative_metrics.append(
                metric
            )
        elif values.gt(0).all():
            positive_metrics.append(
                metric
            )

    clauses = []

    if negative_metrics:
        readable = ", ".join(
            METRIC_LABELS[
                metric
            ]
            for metric in negative_metrics
        )
        clauses.append(
            f"the estimated no-CP world has lower {readable} than the "
            "observed post-CP world"
        )

    if positive_metrics:
        readable = ", ".join(
            METRIC_LABELS[
                metric
            ]
            for metric in positive_metrics
        )
        clauses.append(
            f"the estimated no-CP world has higher {readable} than observed"
        )

    label_column = get_profile_label_column(
        grouping_name
    )

    strongest = complete.sort_values(
        "multimodal_rms",
        ascending=False,
    ).iloc[0]

    weakest = complete.sort_values(
        "multimodal_rms",
        ascending=True,
    ).iloc[0]

    strongest_label = get_profile_label(
        grouping_name,
        strongest[
            label_column
        ],
    )

    weakest_label = get_profile_label(
        grouping_name,
        weakest[
            label_column
        ],
    )

    metric, value = strongest_metric(
        strongest
    )

    story_parts = []

    if clauses:
        story_parts.append(
            "Across every complete profile shown, "
            + "; ".join(
                clauses
            )
            + "."
        )

    story_parts.append(
        f"{strongest_label} has the largest overall five-measure separation. "
        f"There, {relation_phrase(metric, value).replace('The estimated no-CP world shows ', '').rstrip('.')}."
    )

    if strongest_label != weakest_label:
        story_parts.append(
            f"{weakest_label} has the smallest overall separation among the "
            "complete profiles shown."
        )

    return " ".join(
        story_parts
    )


def slice_editorial_insights(
    profiles: pd.DataFrame,
    *,
    grouping_name: str,
) -> list[str]:
    """Build several data-driven plain-language observations for one slice."""
    complete = get_complete_profiles(
        profiles
    )

    if complete.empty:
        return [
            "No complete five-measure profiles are available for this selection."
        ]

    label_column = get_profile_label_column(
        grouping_name
    )

    strongest = complete.sort_values(
        "multimodal_rms",
        ascending=False,
    ).iloc[0]

    strongest_label = get_profile_label(
        grouping_name,
        strongest[
            label_column
        ],
    )

    metric, value = strongest_metric(
        strongest
    )

    insights = [
        (
            f"**Largest overall profile — {strongest_label}:** "
            + relation_phrase(
                metric,
                value,
            )
        )
    ]

    shared_negative = []
    shared_positive = []

    for current_metric in METRIC_ORDER:
        values = pd.to_numeric(
            complete[
                current_metric
            ],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        if values.lt(0).all():
            shared_negative.append(
                current_metric
            )
        elif values.gt(0).all():
            shared_positive.append(
                current_metric
            )

    if shared_negative:
        readable = ", ".join(
            METRIC_LABELS[
                metric_name
            ]
            for metric_name in shared_negative
        )

        insights.append(
            f"**Shared negative contrast:** {readable} are lower in the "
            "estimated no-CP world than in the observed post-CP world across "
            "every complete profile in this selection."
        )

    if shared_positive:
        readable = ", ".join(
            METRIC_LABELS[
                metric_name
            ]
            for metric_name in shared_positive
        )

        insights.append(
            f"**Shared positive contrast:** {readable} are higher in the "
            "estimated no-CP world than in the observed post-CP world across "
            "every complete profile in this selection."
        )

    mixed = complete.sort_values(
        [
            "opposite_sign_pairs",
            "multimodal_rms",
        ],
        ascending=False,
    ).iloc[0]

    mixed_label = get_profile_label(
        grouping_name,
        mixed[
            label_column
        ],
    )

    insights.append(
        f"**Most mixed profile — {mixed_label}:** its five measures contain "
        f"{int(mixed['opposite_sign_pairs'])} opposite-direction metric pairs."
    )

    return insights


def metric_tooltip(
    metric: str,
    value: float,
) -> str:
    """Create row-level hover text with the contrast spelled out."""
    plain = relation_phrase(
        metric,
        value,
    )

    return (
        f"{METRIC_LABELS[metric]}&#10;"
        f"{plain}&#10;"
        f"Signed gap: {value:+.3f} Pre-CP MAE units&#10;"
        "Gap = estimated no-CP − observed"
    )




# ---------------------------------------------------------------------
# Card rendering
# ---------------------------------------------------------------------

def profile_card_html(
    profile: pd.Series,
    *,
    title: str,
    scale_limit: float,
    svg_assets: dict[str, str],
) -> str:
    """Build one unclipped, explicitly labeled fingerprint card."""
    rows = []

    for metric in METRIC_ORDER:
        value = pd.to_numeric(
            pd.Series(
                [
                    profile.get(
                        metric
                    )
                ]
            ),
            errors="coerce",
        ).iloc[0]

        icon_uri = svg_assets[
            METRIC_MODALITY[
                metric
            ]
        ]

        left_icon = icon_img_html(
            icon_uri,
            css_class="raw23-left-icon",
        )

        if pd.isna(
            value
        ):
            track_html = (
                '<div class="raw23-track">'
                '<div class="raw23-zero"></div>'
                '</div>'
            )
            value_html = "—"

        else:
            numeric_value = float(
                value
            )

            width_pct = (
                50.0
                * abs(
                    numeric_value
                )
                / scale_limit
                if scale_limit > 0
                else 0.0
            )

            direction = (
                "negative"
                if numeric_value < 0
                else "positive"
            )

            track_html = (
                f'<div class="raw23-track" '
                f'title="{metric_tooltip(metric, numeric_value)}">'
                '<div class="raw23-zero"></div>'
                f'<div class="raw23-window {direction}" '
                f'style="width:{width_pct:.3f}%;">'
                f'<div class="raw23-run {direction}">'
                f'{icon_run_html(icon_uri)}'
                '</div>'
                '</div>'
                '</div>'
            )

            value_html = (
                f"{numeric_value:+.3f}"
            )

        rows.append(
            (
                '<div class="raw23-row">'
                f'<div>{left_icon}</div>'
                f'<div class="raw23-mode-label">'
                f'{escape(METRIC_SHORT[metric])}'
                '</div>'
                f'{track_html}'
                f'<div class="raw23-value">'
                f'{escape(value_html)}'
                '</div>'
                '</div>'
            )
        )

    return (
        '<div class="raw23-card">'
        f'<div class="raw23-card-title">{escape(title)}</div>'
        '<div class="raw23-card-subtitle">'
        'Estimated no-CP world minus observed post-CP world'
        '</div>'
        '<div class="raw23-axis-header">'
        '<div class="raw23-axis-label measure">Measure</div>'
        '<div class="raw23-axis-label scale">'
        'No-CP − observed · Pre-CP MAE units'
        '</div>'
        '</div>'
        f'{"".join(rows)}'
        '<div class="raw23-axis-footer">'
        '<div class="raw23-axis-ticks">'
        f'<span>−{scale_limit:.0f}</span>'
        '<span>0</span>'
        f'<span>+{scale_limit:.0f}</span>'
        '</div>'
        '</div>'
        '<div class="raw23-card-caption">'
        'Left = estimated no-CP lower than observed · '
        'Right = estimated no-CP higher than observed'
        '</div>'
        '</div>'
    )


def render_profile_card(
    profile: pd.Series,
    *,
    title: str,
    scale_limit: float,
    svg_assets: dict[str, str],
) -> None:
    """Render one multimodal fingerprint."""
    st.markdown(
        profile_card_html(
            profile,
            title=title,
            scale_limit=scale_limit,
            svg_assets=svg_assets,
        ),
        unsafe_allow_html=True,
    )


def render_card_grid(
    profiles: pd.DataFrame,
    *,
    grouping_name: str,
    svg_assets: dict[str, str],
    columns_per_row: int = 2,
) -> None:
    """Render one comparison set on a single unclipped scale."""
    if profiles.empty:
        st.info(
            "No complete profiles are available for this comparison."
        )
        return

    scale_limit = nice_scale_limit(
        profiles
    )

    label_column = get_profile_label_column(
        grouping_name
    )

    columns = st.columns(
        columns_per_row
    )

    for index, (_, row) in enumerate(
        profiles.iterrows()
    ):
        with columns[
            index % columns_per_row
        ]:
            card_label = get_profile_label(
                grouping_name,
                row[
                    label_column
                ],
            )

            render_profile_card(
                row,
                title=card_label,
                scale_limit=scale_limit,
                svg_assets=svg_assets,
            )

            render_profile_card_insight(
                row,
                label=card_label,
            )

# ---------------------------------------------------------------------
# Multimodal Counterfactual Braid helpers
# ---------------------------------------------------------------------

BRAID_HORIZON_COLORS = {
    1: BRAND_COLORS["dark_teal"],
    2: BRAND_COLORS["seafoam"],
    5: BRAND_COLORS["terracotta"],
}

BRAID_METRIC_SHORT = {
    "taxi_trip_count": "Taxi trips",
    "taxi_avg_trip_speed": "Taxi speed",
    "fhvhv_trip_count": "FHVHV trips",
    "fhvhv_avg_trip_speed": "FHVHV speed",
    "subway_ridership": "Subway",
}

BRAID_DAY_TYPE_BUCKETS = {
    "All days": {
        "weekday_overnight", "weekday_am_peak", "weekday_midday",
        "weekday_pm_peak", "weekday_evening", "weekend_overnight",
        "weekend_am_peak", "weekend_midday", "weekend_pm_peak",
        "weekend_evening",
    },
    "Weekdays": {
        "weekday_overnight", "weekday_am_peak", "weekday_midday",
        "weekday_pm_peak", "weekday_evening",
    },
    "Weekends": {
        "weekend_overnight", "weekend_am_peak", "weekend_midday",
        "weekend_pm_peak", "weekend_evening",
    },
}

BRAID_DAYPART_BUCKETS = {
    "All dayparts": set().union(*BRAID_DAY_TYPE_BUCKETS.values()),
    "Overnight": {"weekday_overnight", "weekend_overnight"},
    "AM peak": {"weekday_am_peak", "weekend_am_peak"},
    "Midday": {"weekday_midday", "weekend_midday"},
    "PM peak": {"weekday_pm_peak", "weekend_pm_peak"},
    "Evening": {"weekday_evening", "weekend_evening"},
}

@st.cache_data(show_spinner=False)
def build_weekly_multimodal_braid(
    grouping_name: str,
    group_value: object,
    day_type: str = "All days",
    daypart: str = "All dayparts",
) -> pd.DataFrame:
    """
    Build one geography's five-metric counterfactual profile by policy week.

    WHY:
    Raw 23's storyline uses the frozen row-level gap in Pre-CP MAE units.
    Filtering geography and temporal buckets during the Parquet read preserves
    that exact contract while avoiding a full Raw 23 runtime materialization.
    """
    if grouping_name not in GROUPINGS:
        raise ValueError(f"Unsupported grouping: {grouping_name}")

    valid_buckets = tuple(
        sorted(
            BRAID_DAY_TYPE_BUCKETS[day_type]
            & BRAID_DAYPART_BUCKETS[daypart]
        )
    )

    geography_filters = {
        "taxi_zone_id": None,
        "borough": None,
        "cbd_spatial_category": None,
        "pre_cp_mobility_environment": None,
    }

    group_column = GROUPINGS[grouping_name]["group_id"]

    if group_column == "taxi_zone_id":
        geography_filters["taxi_zone_id"] = int(float(group_value))
    elif group_column in geography_filters:
        geography_filters[group_column] = str(group_value)
    else:
        raise ValueError(
            f"Unsupported Raw 23 storyline geography: {grouping_name}"
        )

    scoped = load_primary_counterfactual_surface(
        temporal_buckets=valid_buckets,
        **geography_filters,
    )

    if scoped.empty:
        return pd.DataFrame()

    # Policy-relative weeks begin Sunday Jan. 5, 2025.
    days_since_launch = (
        scoped["target_date"]
        - pd.Timestamp("2025-01-05")
    ).dt.days

    scoped["week_start"] = (
        pd.Timestamp("2025-01-05")
        + pd.to_timedelta(
            (days_since_launch // 7) * 7,
            unit="D",
        )
    )

    weekly = (
        scoped.groupby(
            ["week_start", "metric", "horizon"],
            observed=True,
            dropna=False,
        )
        .agg(
            gap_mae_units=("counterfactual_gap_mae_units", "mean"),
            support_rows=("counterfactual_gap_mae_units", "count"),
        )
        .reset_index()
    )

    weekly["horizon"] = pd.to_numeric(
        weekly["horizon"],
        errors="coerce",
    ).astype("Int64")

    return (
        weekly.loc[
            weekly["metric"].isin(METRIC_ORDER)
            & weekly["horizon"].isin(HORIZONS)
            & weekly["gap_mae_units"].notna()
        ]
        .sort_values(["metric", "horizon", "week_start"])
        .reset_index(drop=True)
    )


def filter_weekly_storyline_period(
    weekly: pd.DataFrame,
    period: str,
) -> pd.DataFrame:
    """
    Apply the explorer's existing period choice to the weekly storyline.

    WHY: the timeline should obey the same reader-selected time window as the
    fingerprint and peer views instead of introducing another set of controls.
    """
    if weekly.empty:
        return weekly.copy()

    scoped = weekly.copy()
    scoped["week_start"] = pd.to_datetime(
        scoped["week_start"],
        errors="coerce",
    )

    label = str(period).strip().lower()
    weeks = pd.Index(
        sorted(scoped["week_start"].dropna().unique())
    )

    if "full" in label:
        return scoped

    if label == "2025" or ("2025" in label and "first" not in label):
        return scoped.loc[
            scoped["week_start"].dt.year.eq(2025)
        ].copy()

    if "2026" in label:
        return scoped.loc[
            scoped["week_start"].dt.year.eq(2026)
        ].copy()

    number_match = re.search(r"(\d+)", label)
    number = int(number_match.group(1)) if number_match else None

    if "first" in label and "week" in label and number:
        keep_weeks = set(weeks[:number])
        return scoped.loc[
            scoped["week_start"].isin(keep_weeks)
        ].copy()

    if "final" in label and "week" in label and number:
        keep_weeks = set(weeks[-number:])
        return scoped.loc[
            scoped["week_start"].isin(keep_weeks)
        ].copy()

    if "first" in label and "3 month" in label:
        keep_weeks = set(weeks[:13])
        return scoped.loc[
            scoped["week_start"].isin(keep_weeks)
        ].copy()

    if "first" in label and "6 month" in label:
        keep_weeks = set(weeks[:26])
        return scoped.loc[
            scoped["week_start"].isin(keep_weeks)
        ].copy()

    return scoped


def build_multimodal_braid_figure(
    weekly: pd.DataFrame,
    *,
    geography_label: str,
    selected_horizon: int,
    period_label: str,
) -> go.Figure:
    """
    Render five mobility measures together on Raw 23's shared MAE scale.

    WHY: the explorer already chooses horizon and period. The selected horizon
    drives each storyline, while the faint all-horizon range stays as context.
    """
    metric_colors = {
        "taxi_trip_count": BRAND_COLORS["terracotta"],
        "taxi_avg_trip_speed": BRAND_COLORS["seafoam"],
        "fhvhv_trip_count": BRAND_COLORS["dark_teal"],
        "fhvhv_avg_trip_speed": BRAND_COLORS["pale_peach"],
        "subway_ridership": BRAND_COLORS["dark_teal"],
    }

    metric_dashes = {
        "taxi_trip_count": "solid",
        "taxi_avg_trip_speed": "dot",
        "fhvhv_trip_count": "solid",
        "fhvhv_avg_trip_speed": "dash",
        "subway_ridership": "longdash",
    }

    figure = go.Figure()
    min_week = weekly["week_start"].min()
    max_week = weekly["week_start"].max()
    all_primary = []

    figure.add_hline(
        y=0,
        line_width=1.5,
        line_dash="dash",
        line_color=BRAND_COLORS["dark_teal"],
        opacity=0.55,
    )

    for metric in METRIC_ORDER:
        metric_frame = weekly.loc[
            weekly["metric"].eq(metric),
            ["week_start", "horizon", "gap_mae_units", "support_rows"],
        ].copy()

        horizon_matrix = (
            metric_frame.pivot_table(
                index="week_start",
                columns="horizon",
                values="gap_mae_units",
                aggfunc="mean",
            )
            .reindex(columns=HORIZONS)
            .sort_index()
        )

        if selected_horizon not in horizon_matrix.columns:
            continue

        primary = horizon_matrix[selected_horizon]
        horizon_min = horizon_matrix.min(axis=1, skipna=True)
        horizon_max = horizon_matrix.max(axis=1, skipna=True)
        spread = horizon_max - horizon_min
        all_primary.append(primary.rename(metric))

        support = (
            metric_frame.loc[
                metric_frame["horizon"].eq(selected_horizon)
            ]
            .groupby("week_start", observed=True)["support_rows"]
            .sum()
            .reindex(horizon_matrix.index)
        )

        color = metric_colors[metric]

        figure.add_trace(
            go.Scatter(
                x=horizon_matrix.index,
                y=horizon_min,
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
                legendgroup=metric,
            )
        )

        figure.add_trace(
            go.Scatter(
                x=horizon_matrix.index,
                y=horizon_max,
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(131, 197, 190, 0.10)",
                hoverinfo="skip",
                showlegend=False,
                legendgroup=metric,
            )
        )

        hover_rows = []

        for week in horizon_matrix.index:
            values = horizon_matrix.loc[week]
            horizon_text = " · ".join(
                (
                    f"h={horizon}: {values[horizon]:+.3f}"
                    if pd.notna(values[horizon])
                    else f"h={horizon}: —"
                )
                for horizon in HORIZONS
            )

            hover_rows.append(
                [
                    (
                        f"{primary.loc[week]:+.3f}"
                        if pd.notna(primary.loc[week])
                        else "—"
                    ),
                    f"{spread.loc[week]:.3f}",
                    horizon_text,
                    (
                        f"{int(support.loc[week]):,}"
                        if pd.notna(support.loc[week])
                        else "—"
                    ),
                ]
            )

        figure.add_trace(
            go.Scatter(
                x=horizon_matrix.index,
                y=primary,
                mode="lines",
                name=BRAID_METRIC_SHORT[metric],
                legendgroup=metric,
                line={
                    "color": color,
                    "width": 3.1,
                    "dash": metric_dashes[metric],
                    "shape": "spline",
                    "smoothing": 0.30,
                },
                customdata=np.asarray(
                    hover_rows,
                    dtype=object,
                ),
                hovertemplate=(
                    f"<b>{BRAID_METRIC_SHORT[metric]}</b>"
                    "<br>Week of %{x|%b %d, %Y}"
                    f"<br>Selected horizon: h={selected_horizon}"
                    "<br>No-CP − observed: %{customdata[0]} Pre-CP MAE"
                    "<br>All-horizon spread: %{customdata[1]} MAE"
                    "<br>%{customdata[2]}"
                    "<br>Supporting rows: %{customdata[3]}"
                    "<extra></extra>"
                ),
                showlegend=True,
            )
        )

    if all_primary:
        primary_frame = pd.concat(
            all_primary,
            axis=1,
        )
        finite_values = primary_frame.stack().dropna()
    else:
        finite_values = pd.Series(dtype=float)

    if finite_values.empty:
        y_min, y_max = -1.0, 1.0
    else:
        y_min = float(finite_values.min())
        y_max = float(finite_values.max())
        y_span = max(y_max - y_min, 1.0)
        y_min -= 0.10 * y_span
        y_max += 0.10 * y_span

    figure.update_layout(
        title={
            "text": (
                f"Weekly multimodal storyline · {geography_label} · "
                f"h={selected_horizon}"
            ),
            "x": 0,
            "xanchor": "left",
            "font": {
                "size": 18,
                "color": BRAND_COLORS["dark_teal"],
            },
        },
        height=475,
        margin={
            "l": 85,
            "r": 30,
            "t": 82,
            "b": 65,
        },
        hovermode="closest",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 11},
        },
    )

    figure.update_xaxes(
        title=None,
        range=[min_week, max_week],
        showgrid=True,
        gridcolor="rgba(131, 197, 190, 0.20)",
        tickformat="%b<br>%Y",
        hoverformat="%b %d, %Y",
        linecolor=BRAND_COLORS["seafoam"],
        tickfont={
            "color": BRAND_COLORS["dark_teal"],
        },
    )

    figure.update_yaxes(
        title={
            "text": "No-CP − observed · Pre-CP MAE units",
            "font": {
                "color": BRAND_COLORS["dark_teal"],
            },
        },
        range=[y_min, y_max],
        showgrid=True,
        gridcolor="rgba(131, 197, 190, 0.20)",
        zeroline=False,
        tickfont={
            "color": BRAND_COLORS["dark_teal"],
        },
    )

    apply_branding(figure)

    return figure


def multimodal_braid_insight(
    weekly: pd.DataFrame,
    *,
    selected_horizon: int,
) -> str:
    """Summarize the strongest standardized separation for the visible horizon."""
    if weekly.empty:
        return ""

    visible = weekly.loc[
        weekly["horizon"].eq(selected_horizon)
    ].copy()

    if visible.empty:
        return ""

    summary = (
        visible.groupby(
            "metric",
            observed=True,
        )["gap_mae_units"]
        .agg(
            mean_gap="mean",
            mean_abs_gap=lambda values: float(
                np.mean(
                    np.abs(values)
                )
            ),
        )
        .reset_index()
    )

    strongest = summary.loc[
        summary["mean_abs_gap"].idxmax()
    ]
    metric = strongest["metric"]
    signed_gap = float(
        strongest["mean_gap"]
    )

    return (
        f"For the displayed weeks at **h={selected_horizon}**, "
        f"**{BRAID_METRIC_SHORT[metric]}** has the largest average "
        f"standardized gap. {relation_phrase(metric, signed_gap)}"
    )


# ---------------------------------------------------------------------
# Load + hidden QA
# ---------------------------------------------------------------------

with st.spinner("Loading multimodal counterfactuals..."):
    runtime_qa = load_counterfactual_runtime_qa()
    svg_assets = load_svg_assets()


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.caption(
    PAGE_CAPTION
)

st.title(
    PAGE_TITLE
)

st.markdown(
    f'<p class="app-subtitle">{PAGE_SUBTITLE}</p>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Static editorial overview
# ---------------------------------------------------------------------

hero_profiles = get_counterfactual_profiles(
    grouping_name="Policy geography",
    horizon=1,
    period="Full post-CP period",
    day_type="All days",
    daypart="All dayparts",
)

st.divider()

st.subheader(
    "1. How did the five mobility measures differ across policy geographies?"
)

st.markdown(
    "The fixed overview compares **CBD, adjacent, gateway, and non-CBD geographies** "
    "over the full post-CP period at **h=1**. Each fingerprint puts Taxi demand, Taxi "
    "speed, FHVHV demand, FHVHV speed, and Subway ridership on the same Pre-CP "
    "forecast-error scale, so direction and relative magnitude can be compared even "
    "though the measures use different native units."
)

st.caption(
    "Fixed overview · full post-CP period · h=1 · all days and dayparts"
)

hero_compare_tab, hero_inside_tab = st.tabs(
    [
        "Compare policy geographies",
        "Look inside the policy geographies",
    ]
)

with hero_compare_tab:
    st.markdown(
        '<div class="raw23-tab-intro">'
        'These are peers at the same level: CBD, adjacent, gateway, and '
        'non-CBD geographies.'
        '</div>',
        unsafe_allow_html=True,
    )

    render_card_grid(
        hero_profiles,
        grouping_name="Policy geography",
        svg_assets=svg_assets,
    )

    render_chart_insight(
        group_story_insight(
            hero_profiles,
            grouping_name="Policy geography",
        )
    )

with hero_inside_tab:
    st.markdown(
        '<div class="raw23-tab-intro">'
        'Each column below looks inside one policy geography and surfaces its '
        'strongest complete Taxi-Zone profile.'
        '</div>',
        unsafe_allow_html=True,
    )

    child_cards = []

    for _, policy_row in hero_profiles.iterrows():
        policy_value = policy_row[
            "cbd_spatial_category"
        ]

        children = (
            get_child_taxi_zone_profiles(
                parent_grouping="Policy geography",
                selected_value=policy_value,
                horizon=1,
                period="Full post-CP period",
                day_type="All days",
                daypart="All dayparts",
            )
        )

        strongest_child = get_strongest_profiles(
            children,
            count=1,
        )

        if not strongest_child.empty:
            strongest_child = strongest_child.copy()
            strongest_child[
                "_parent_label"
            ] = get_profile_label(
                "Policy geography",
                policy_value,
            )
            child_cards.append(
                strongest_child
            )

    if child_cards:
        hero_children = pd.concat(
            child_cards,
            ignore_index=True,
        )

        scale_limit = nice_scale_limit(
            hero_children
        )

        columns = st.columns(
            2
        )

        for index, (_, row) in enumerate(
            hero_children.iterrows()
        ):
            with columns[
                index % 2
            ]:
                parent_label = row[
                    "_parent_label"
                ]

                card_label = (
                    f"{row['zone']} · {parent_label}"
                )

                render_profile_card(
                    row,
                    title=card_label,
                    scale_limit=scale_limit,
                    svg_assets=svg_assets,
                )

                render_profile_card_insight(
                    row,
                    label=str(
                        row[
                            "zone"
                        ]
                    ),
                )

        render_chart_insight(
            group_story_insight(
                hero_children,
                grouping_name="Taxi Zone",
            )
        )

    else:
        st.info(
            "No complete Taxi-Zone child profiles are available for this overview."
        )


st.markdown(
    "### How did the CBD multimodal pattern unfold?"
)

st.markdown(
    """
The fingerprints above give the quickest summary of how the five mobility
measures differed from their estimated no-congestion-pricing paths. This
**frozen CBD example** teaches the temporal view before you reach the
interactive explorer.

Read the chart from left to right. Each line is one mobility measure at
**h=1**, and all five share the same Pre-CP forecast-error scale. **Zero** means
observed mobility matched the estimated no-CP path. Above zero, the no-CP
estimate was higher than observed; below zero, observed mobility was higher.

Look for **persistent separation, convergence, crossings, and sudden changes
in ordering**. The faint envelope around each line shows the range across
h=1, h=2, and h=5, so wider bands flag weeks where the counterfactual estimate
depends more on forecast horizon.
    """
)

# Resolve the frozen CBD through the same profile contract used by the explorer.
# WHY: reader-facing labels are not guaranteed to be the underlying group IDs.
hero_grouping = "Policy geography"
hero_group_id = GROUPINGS[hero_grouping]["group_id"]
hero_label_column = get_profile_label_column(
    hero_grouping
)

hero_cbd_matches = hero_profiles.loc[
    hero_profiles[hero_label_column]
    .map(
        lambda value: get_profile_label(
            hero_grouping,
            value,
        )
    )
    .astype(str)
    .str.strip()
    .str.casefold()
    .eq("cbd")
].copy()

if hero_cbd_matches.empty:
    hero_storyline_weekly = pd.DataFrame()
else:
    hero_cbd_value = hero_cbd_matches.iloc[0][
        hero_group_id
    ]

    hero_storyline_weekly = build_weekly_multimodal_braid(
        grouping_name=hero_grouping,
        group_value=hero_cbd_value,
        day_type="All days",
        daypart="All dayparts",
    )

    hero_storyline_weekly = filter_weekly_storyline_period(
        hero_storyline_weekly,
        "Full post-CP period",
    )

if hero_storyline_weekly.empty:
    st.info(
        "The frozen CBD profile resolved successfully, but no weekly rows "
        "were returned for this overview. Check the weekly surface contract."
    )
else:
    hero_storyline_figure = build_multimodal_braid_figure(
        hero_storyline_weekly,
        geography_label="CBD",
        selected_horizon=1,
        period_label="Full post-CP period",
    )

    st.plotly_chart(
        hero_storyline_figure,
        width="stretch",
        config={
            "displayModeBar": False,
            "responsive": True,
        },
        key="raw23_multimodal_counterfactual_storyline_hero",
    )

    hero_storyline_takeaway = multimodal_braid_insight(
        hero_storyline_weekly,
        selected_horizon=1,
    )

    if hero_storyline_takeaway:
        render_chart_insight(
            hero_storyline_takeaway
        )

st.caption(
    "Frozen example · CBD · full post-CP period · h=1 · all days and dayparts. "
    "Above zero = estimated no-CP higher than observed; below zero = observed "
    "higher than estimated no-CP."
)


# ---------------------------------------------------------------------
# Interactive explorer
# ---------------------------------------------------------------------

st.divider()


@st.fragment
def _render_interactive_explorer() -> None:
    """Render Raw 23's explorer without rerunning the static overview."""

    with exploration_section(
        key="raw23_exploration_area",
        title="2. Explore the multimodal counterfactuals",
        description=(
            "Choose a geography level, place, horizon, and time slice. The weekly "
            "storyline, focus fingerprint, peer comparison, drill-down comparison, "
            "and written interpretation all update together."
        ),
    ):
        control_1, control_2, control_3 = st.columns(
            [
                1.25,
                0.65,
                1.1,
            ]
        )

        with control_1:
            selected_grouping = st.selectbox(
                "Geography level",
                options=list(
                    GROUPINGS
                ),
                index=0,
                key="raw23_grouping",
            )

        with control_2:
            selected_horizon = st.selectbox(
                "Horizon",
                options=HORIZONS,
                index=0,
                format_func=lambda value: f"h={value}",
                key="raw23_horizon",
            )

        with control_3:
            selected_period = st.selectbox(
                "Period",
                options=list(
                    PERIODS
                ),
                index=0,
                key="raw23_period",
            )

        control_4, control_5 = st.columns(
            2
        )

        with control_4:
            selected_day_type = st.selectbox(
                "Day type",
                options=[
                    "All days",
                    "Weekdays",
                    "Weekends",
                ],
                index=0,
                key="raw23_day_type",
            )

        with control_5:
            selected_daypart = st.selectbox(
                "Daypart",
                options=[
                    "All dayparts",
                    "Overnight",
                    "AM peak",
                    "Midday",
                    "PM peak",
                    "Evening",
                ],
                index=0,
                key="raw23_daypart",
            )

        selected_profiles = get_counterfactual_profiles(
            grouping_name=selected_grouping,
            horizon=selected_horizon,
            period=selected_period,
            day_type=selected_day_type,
            daypart=selected_daypart,
        )

        if selected_profiles.empty:
            st.info(
                "No profiles are available for this selection."
            )

        else:
            label_column = (
                get_profile_label_column(
                    selected_grouping
                )
            )

            # WHY: "Unknown" is useful upstream for completeness/QA, but it is not
            # a meaningful reader-facing geography choice.
            selected_profiles = selected_profiles.loc[
                selected_profiles[
                    label_column
                ]
                .astype(str)
                .str.strip()
                .str.lower()
                .ne("unknown")
            ].copy()

            if selected_profiles.empty:
                st.info(
                    "No reader-facing geography profiles are available for this selection."
                )
                st.stop()

            group_id_column = GROUPINGS[
                selected_grouping
            ]["group_id"]

            complete = get_complete_profiles(
                selected_profiles
            )

            default_value = selected_profiles.iloc[0][
                group_id_column
            ]

            if not complete.empty:
                strongest = complete.sort_values(
                    "multimodal_rms",
                    ascending=False,
                ).iloc[0]

                default_value = strongest[
                    group_id_column
                ]

            option_values = (
                selected_profiles[
                    group_id_column
                ]
                .dropna()
                .drop_duplicates()
                .tolist()
            )

            previous_grouping = st.session_state.get(
                "raw23_previous_geography_level"
            )

            if previous_grouping != selected_grouping:
                # WHY: geography IDs mean different things across grouping levels.
                # Reset once when the level changes; within a level, the stable
                # semantic geography value survives ordinary filter reruns.
                st.session_state[
                    "raw23_focus_profile"
                ] = default_value

                st.session_state[
                    "raw23_previous_geography_level"
                ] = selected_grouping

            elif (
                "raw23_focus_profile"
                not in st.session_state
                or st.session_state[
                    "raw23_focus_profile"
                ]
                not in option_values
            ):
                # WHY: a temporal filter can remove a previously selected place.
                st.session_state[
                    "raw23_focus_profile"
                ] = default_value

            selected_value = st.selectbox(
                "Focus profile",
                options=option_values,
                format_func=lambda value: get_profile_label(
                    selected_grouping,
                    selected_profiles.loc[
                        selected_profiles[
                            group_id_column
                        ].eq(
                            value
                        ),
                        label_column,
                    ].iloc[0],
                ),
                key="raw23_focus_profile",
            )

            selected_profile = selected_profiles.loc[
                selected_profiles[
                    group_id_column
                ].eq(
                    selected_value
                )
            ].iloc[0]

            selected_label = get_profile_label(
                selected_grouping,
                selected_profile[
                    label_column
                ],
            )

            st.markdown(
                "##### Focus fingerprint"
            )

            focus_scale = nice_scale_limit(
                pd.DataFrame(
                    [
                        selected_profile[
                            METRIC_ORDER
                        ]
                    ]
                )
            )

            render_profile_card(
                selected_profile,
                title=selected_label,
                scale_limit=focus_scale,
                svg_assets=svg_assets,
            )

            render_profile_card_insight(
                selected_profile,
                label=selected_label,
            )


            st.markdown(
                "##### How did this multimodal pattern unfold?"
            )

            st.markdown(
                """
    Now explore the **same weekly view introduced in the frozen example above**
    using your current geography, period, day-type, daypart, and horizon selections.
                """
            )

            storyline_weekly = build_weekly_multimodal_braid(
                grouping_name=selected_grouping,
                group_value=selected_value,
                day_type=selected_day_type,
                daypart=selected_daypart,
            )

            storyline_weekly = filter_weekly_storyline_period(
                storyline_weekly,
                selected_period,
            )

            if storyline_weekly.empty:
                st.info(
                    "No weekly counterfactual observations are available "
                    "for this selection."
                )
            else:
                storyline_figure = build_multimodal_braid_figure(
                    storyline_weekly,
                    geography_label=selected_label,
                    selected_horizon=selected_horizon,
                    period_label=selected_period,
                )

                st.plotly_chart(
                    storyline_figure,
                    width="stretch",
                    config={
                        "displayModeBar": False,
                        "responsive": True,
                    },
                    key="raw23_multimodal_counterfactual_storyline",
                )

                st.caption(
                    f"{selected_period} · h={selected_horizon}. "
                    "Above zero = estimated no-CP higher than observed; below zero "
                    "= observed higher than estimated no-CP."
                )


                storyline_takeaway = multimodal_braid_insight(
                    storyline_weekly,
                    selected_horizon=selected_horizon,
                )

                if storyline_takeaway:
                    render_chart_insight(
                        storyline_takeaway
                    )

                st.caption(
                    "These are model-based counterfactual comparisons, not direct "
                    "proof that congestion pricing caused the full observed difference."
                )

            if selected_grouping == "Taxi Zone":
                peer_tab_label = (
                    "Nearby Taxi Zones"
                )
                child_tab_label = (
                    "Other Taxi Zones in the same borough"
                )

            else:
                peer_tab_label = (
                    f"Compare {selected_grouping.lower()} peers"
                )
                child_tab_label = (
                    f"Taxi Zones inside {selected_label}"
                )

            peer_tab, child_tab = st.tabs(
                [
                    peer_tab_label,
                    child_tab_label,
                ]
            )

            with peer_tab:
                peers = get_same_level_peers(
                    grouping_name=selected_grouping,
                    selected_value=selected_value,
                    horizon=selected_horizon,
                    period=selected_period,
                    day_type=selected_day_type,
                    daypart=selected_daypart,
                )

                if selected_grouping == "Taxi Zone":
                    st.markdown(
                        '<div class="raw23-tab-intro">'
                        'These are Taxi Zones connected to the selected zone in the '
                        'project’s transportation-aware Taxi Zone network.'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="raw23-tab-intro">'
                        'These are comparable profiles at the same geographic level.'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                peer_display = (
                    get_strongest_profiles(
                        peers,
                        count=4,
                    )
                    if len(
                        get_complete_profiles(
                            peers
                        )
                    ) > 4
                    else get_complete_profiles(
                        peers
                    )
                )

                if peer_display.empty:
                    st.info(
                        "No complete peer profiles are available for this selection."
                    )
                else:
                    render_card_grid(
                        peer_display,
                        grouping_name=selected_grouping,
                        svg_assets=svg_assets,
                    )


            with child_tab:
                if selected_grouping == "Taxi Zone":
                    children = (
                        get_same_borough_taxi_zone_profiles(
                            selected_zone_id=int(
                                selected_value
                            ),
                            horizon=selected_horizon,
                            period=selected_period,
                            day_type=selected_day_type,
                            daypart=selected_daypart,
                        )
                    )

                    st.markdown(
                        '<div class="raw23-tab-intro">'
                        'This second view broadens the comparison from immediate '
                        'network neighbors to other Taxi Zones in the same borough.'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                else:
                    children = (
                        get_child_taxi_zone_profiles(
                            parent_grouping=selected_grouping,
                            selected_value=selected_value,
                            horizon=selected_horizon,
                            period=selected_period,
                            day_type=selected_day_type,
                            daypart=selected_daypart,
                        )
                    )

                    st.markdown(
                        '<div class="raw23-tab-intro">'
                        'This view drills inside the selected geography and surfaces '
                        'the Taxi Zones with the largest complete multimodal profiles.'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                child_display = get_strongest_profiles(
                    children,
                    count=4,
                )

                if child_display.empty:
                    st.info(
                        "No complete Taxi-Zone profiles are available inside this selection."
                    )

                else:
                    render_card_grid(
                        child_display,
                        grouping_name="Taxi Zone",
                        svg_assets=svg_assets,
                    )



            with st.expander(
                "What this selection means",
                expanded=False,
            ):
                st.markdown(
                    profile_editorial_sentence(
                        selected_profile,
                        label=selected_label,
                    )
                )

                st.markdown(
                    "**Across the current grouping:**"
                )

                for insight in slice_editorial_insights(
                    selected_profiles,
                    grouping_name=selected_grouping,
                ):
                    st.markdown(
                        f"- {insight}"
                    )

                st.caption(
                    "These statements update with the selected geography, horizon, "
                    "period, day type, and daypart."
                )

_render_interactive_explorer()

# ---------------------------------------------------------------------
# Closing synthesis
# ---------------------------------------------------------------------

st.divider()

st.markdown("### What this page establishes")
st.markdown(
    "The counterfactual gap was multimodal rather than a single systemwide number. "
    "Taxi demand, Taxi speed, FHVHV demand, FHVHV speed, and Subway ridership could "
    "differ from their estimated no-CP paths by different amounts—and sometimes in "
    "different directions—within the same geography. Putting all five on one common "
    "forecast-error scale makes those cross-mode contrasts visible without pretending "
    "their native units are interchangeable."
)

with st.expander("How this page works", expanded=False):
    st.markdown(
        """
        **1. Keep one sign convention everywhere.** The plotted difference is always
        **estimated no-CP − observed post-CP**. Positive means the no-CP estimate is
        higher; negative means observed mobility is higher. For demand this means
        more/fewer trips or riders; for speed it means faster/slower movement.

        **2. Standardize magnitude with each forecast's own Pre-CP error.** Distance
        from zero is measured in **Pre-CP MAE units**. A value one unit from zero is a
        gap as large as that forecasting job's typical absolute Pre-CP error. This
        makes trips, speeds, and ridership visually comparable; it is not a
        significance test.

        **3. Use fingerprints for cross-mode shape and the weekly braid for timing.**
        Fingerprints summarize the selected period. The weekly view shows when the
        five measures separated, converged, crossed, or changed ordering.

        **4. Keep forecast horizon separate from statistical uncertainty.** The faint
        weekly envelope is the range across **h=1, h=2, and h=5**. It shows horizon
        sensitivity, not a confidence interval.

        **5. Compare geography at several levels without changing the metric logic.**
        Policy geography, borough, mobility environment, and Taxi Zone views all use
        the same signed, Pre-CP-error-scaled counterfactual definition.
        """
    )

st.caption(
    "Evidence scope: forecast-based comparisons between observed post-launch mobility "
    "and the estimated no-congestion-pricing path. Cross-mode distance from zero is "
    "scaled by Pre-CP forecast error and is not statistical significance; the displayed "
    "gaps should not be read as direct proof that congestion pricing caused the entire "
    "observed difference."
)

