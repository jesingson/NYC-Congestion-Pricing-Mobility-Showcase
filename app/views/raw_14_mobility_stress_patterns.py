from __future__ import annotations

import re
from collections.abc import Sequence
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.data_access.anomalies import (
    load_stress_anomaly_events_runtime,
    load_stress_anomaly_observation_universe,
)
from app.data_access.loaders import (
    CONGESTION_PRICING_START_DATE,
)
from app.utils.project_branding import (
    BRAND_COLORS,
    exploration_section,
    inject_app_css,
    render_chart_insight,
)


inject_app_css()


def _render_filtered_storyline(
    scoped_events: pd.DataFrame,
    scoped_eligible: pd.DataFrame,
    *,
    stress_family: str,
) -> None:
    """
    Render the storyline inside Raw 14's existing Interactive Explorer.

    WHY: the UpSet chart and storyline should describe the same filtered event
    universe. Page-level stress-family, date, and temporal-bucket controls are
    therefore resolved before this function is called. Only storyline-specific
    display controls live here.
    """
    st.markdown("### When did these modes share the same stress?")

    st.caption(
        "The UpSet view above shows which modality combinations characterize "
        "the current selection. This storyline keeps those same filters and "
        "adds time; the controls below only change which intersection sizes "
        "and how many weekly events are drawn."
    )

    control_left, control_right = st.columns([1.25, 1.0])

    with control_left:
        intersection_sizes = _intersection_size_control(
            key_prefix="raw14_filtered_storyline",
            default=(2, 3, 4),
        )

    with control_right:
        event_limit = st.slider(
            "Weekly events to display",
            min_value=8,
            max_value=40,
            value=20,
            step=1,
            key="raw14_filtered_storyline_event_limit",
            help=(
                "Controls visual density. A small number of the strongest "
                "higher-order events are preserved before remaining positions "
                "are filled by incidence."
            ),
        )

    if not intersection_sizes:
        st.info(
            "Choose at least one intersection size to display the storyline."
        )
        return

    required_storyline_columns = {
        "signature_label",
        "modality_signature",
    }
    missing_storyline_columns = (
        required_storyline_columns
        - set(scoped_events.columns)
    )

    if missing_storyline_columns:
        raise RuntimeError(
            "Filtered storyline events are missing the signature enrichment "
            f"required by the weekly storyline: "
            f"{sorted(missing_storyline_columns)}"
        )

    weekly = _weekly_storyline_surface(
        scoped_events,
        scoped_eligible,
    )

    if weekly.empty:
        st.info(
            "No shared weekly stress combinations match the current "
            "explorer filters and selected intersection sizes."
        )
        return

    figure = _build_stress_storyline_figure(
        weekly,
        family_label=stress_family,
        max_events=event_limit,
        modality_counts=intersection_sizes,
    )

    st.plotly_chart(
        figure,
        width="stretch",
        config={
            "displayModeBar": False,
            "responsive": True,
        },
        key="raw14_filtered_storyline_chart",
    )

    takeaway = _storyline_takeaway(
        weekly,
        max_events=event_limit,
        family_label=stress_family,
        modality_counts=intersection_sizes,
    )

    if takeaway:
        render_chart_insight(takeaway)

    sizes = ", ".join(
        f"{value}-way"
        for value in intersection_sizes
    )
    st.caption(
        f"Current explorer selection · {sizes} intersections · "
        f"up to {event_limit} weekly events · "
        "line thickness = incidence per 1,000 eligible observations."
    )


# =============================================================================
# Page 14 · Phase 1 · hero re-scouting
#
# Temporary scouting mode produces one copy-pasteable report, then stops before
# the existing positive-demand hero. Phase 2 will replace that hero only after
# the All / Congestion / Demand / Both evidence is reviewed.
# =============================================================================


PAGE_CAPTION = "MOBILITY STRESS PATTERNS"
PAGE_TITLE = "How did mobility stress differ across modes, time, and place?"
PHASE_1_SCOUTING_MODE = False

HERO_INTERSECTION_COUNT = 6

DEMAND_METRIC_TO_MODALITY = {
    "taxi_trip_count": "Taxi",
    "fhvhv_trip_count": "FHVHV",
    "subway_ridership": "Subway",
    "subway_transfers": "Subway",
    "bus_trip_count": "Bus",
}

DEMAND_MODALITIES = (
    "Taxi",
    "FHVHV",
    "Subway",
    "Bus",
)

DEMAND_MODALITY_ORDER = {
    modality: index
    for index, modality in enumerate(
        DEMAND_MODALITIES
    )
}


STRESS_FAMILY_OPTIONS = (
    "Congestion",
    "Demand",
    "Both",
    "All",
)

MATCH_RULE_ANY = "At least one selected mode — OR"
MATCH_RULE_ALL = "All selected modes — AND"
MATCH_RULE_OPTIONS = (
    MATCH_RULE_ANY,
    MATCH_RULE_ALL,
)

MEASURE_OPTIONS = (
    "Composition share",
    "Stress anomalies per 1,000 eligible observations",
)

TIME_SCOPE_OPTIONS = (
    "Full study",
    "Pre-CP",
    "Post-CP",
    "Custom range",
)

TEMPORAL_BUCKET_ORDER = (
    "weekday_am_peak",
    "weekday_midday",
    "weekday_pm_peak",
    "weekday_evening",
    "weekday_overnight",
    "weekend_am_peak",
    "weekend_midday",
    "weekend_pm_peak",
    "weekend_evening",
    "weekend_overnight",
)

TEMPORAL_BUCKET_LABELS = {
    "weekday_am_peak": "Weekday AM peak",
    "weekday_midday": "Weekday midday",
    "weekday_pm_peak": "Weekday PM peak",
    "weekday_evening": "Weekday evening",
    "weekday_overnight": "Weekday overnight",
    "weekend_am_peak": "Weekend AM peak",
    "weekend_midday": "Weekend midday",
    "weekend_pm_peak": "Weekend PM peak",
    "weekend_evening": "Weekend evening",
    "weekend_overnight": "Weekend overnight",
}


METRIC_DISPLAY_LABELS = {
    "taxi_trip_count": "Taxi trips",
    "taxi_avg_trip_speed": "Taxi avg. speed",
    "taxi_avg_trip_duration": "Taxi avg. duration",
    "fhvhv_trip_count": "FHVHV trips",
    "fhvhv_avg_trip_speed": "FHVHV avg. speed",
    "fhvhv_avg_trip_duration": "FHVHV avg. duration",
    "subway_ridership": "Subway ridership",
    "subway_transfers": "Subway transfers",
    "bus_trip_count": "Bus trips",
    "avg_bus_speed": "Bus avg. speed",
}

POLICY_PERIOD_ORDER = (
    "Pre-CP",
    "Post-CP",
)

POLICY_GEOGRAPHY_ORDER = (
    "CBD",
    "Gateway + adjacent",
    "Outside",
)

MOBILITY_ENVIRONMENT_ORDER = (
    "Transit-Rich Outer Boroughs",
    "Lower-Transit Neighborhoods",
    "Long-Trip Fast-Mobility Zones",
    "Urban Activity Core",
    "Staten Island Fast-Mobility",
)

MOBILITY_ENVIRONMENT_SHORT_LABELS = {
    "Transit-Rich Outer Boroughs": "Transit-Rich<br>Outer Boroughs",
    "Lower-Transit Neighborhoods": "Lower-Transit<br>Neighborhoods",
    "Long-Trip Fast-Mobility Zones": "Long-Trip<br>Fast-Mobility",
    "Urban Activity Core": "Urban Activity<br>Core",
    "Staten Island Fast-Mobility": "Staten Island<br>Fast-Mobility",
}

PANEL_COLORS = {
    "Pre-CP": BRAND_COLORS["dark_teal"],
    "Post-CP": BRAND_COLORS["terracotta"],
    "CBD": BRAND_COLORS["dark_teal"],
    "Gateway + adjacent": BRAND_COLORS["seafoam"],
    "Outside": BRAND_COLORS["terracotta"],
    "Transit-Rich Outer Boroughs": BRAND_COLORS["dark_teal"],
    "Lower-Transit Neighborhoods": BRAND_COLORS["terracotta"],
    "Long-Trip Fast-Mobility Zones": BRAND_COLORS["seafoam"],
    "Urban Activity Core": "#4F8F92",
    "Staten Island Fast-Mobility": "#C77E63",
}

INACTIVE_DOT_COLOR = "rgba(120, 145, 150, 0.20)"
GRID_COLOR = "rgba(0, 109, 119, 0.12)"
TEXT_COLOR = "#003F46"


# -----------------------------------------------------------------------------
# Normalization helpers
# -----------------------------------------------------------------------------
def _normalize_period(
    values: pd.Series,
) -> pd.Series:
    normalized = (
        values.astype("string")
        .str.strip()
        .str.lower()
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )

    result = pd.Series(
        "Unknown",
        index=values.index,
        dtype="string",
    )

    result.loc[
        normalized.eq("pre_cp").fillna(False)
    ] = "Pre-CP"

    result.loc[
        normalized.eq("post_cp").fillna(False)
    ] = "Post-CP"

    return result


def _normalize_policy_geography(
    policy_values: pd.Series,
    cbd_values: pd.Series,
) -> pd.Series:
    policy = (
        policy_values.astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )

    cbd = (
        cbd_values.astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )

    combined = policy.where(
        policy.ne(""),
        cbd,
    )

    result = pd.Series(
        "Unknown",
        index=policy_values.index,
        dtype="string",
    )

    result.loc[
        (
            combined.eq("cbd")
            | cbd.eq("cbd")
        ).fillna(False)
    ] = "CBD"

    result.loc[
        (
            combined.isin(
                {
                    "gateway_or_adjacent",
                    "gateway+adjacent",
                    "gateway_+_adjacent",
                    "gateway",
                    "gateway_to_cbd",
                    "adjacent",
                    "adjacent_to_cbd",
                }
            )
            | cbd.isin(
                {
                    "gateway",
                    "gateway_to_cbd",
                    "adjacent",
                    "adjacent_to_cbd",
                }
            )
        ).fillna(False)
    ] = "Gateway + adjacent"

    result.loc[
        (
            combined.isin(
                {
                    "outside",
                    "non_cbd",
                    "noncbd",
                }
            )
            | cbd.isin(
                {
                    "outside",
                    "non_cbd",
                    "noncbd",
                }
            )
        ).fillna(False)
    ] = "Outside"

    return result


def _normalize_environment(
    values: pd.Series,
) -> pd.Series:
    return (
        values.astype("string")
        .fillna("Unknown")
        .str.strip()
        .replace("", "Unknown")
    )


# -----------------------------------------------------------------------------
# Modality-signature helpers
# -----------------------------------------------------------------------------
def _signature_label(
    signature: tuple[str, ...],
) -> str:
    if not signature:
        return "(none)"

    return " + ".join(
        signature
    )


def _modality_signature(
    driver_text: object,
) -> tuple[str, ...]:
    """
    Collapse demand-metric drivers to the participating mobility modes.

    Example:
        subway_ridership + subway_transfers -> ("Subway",)
        taxi_trip_count + subway_transfers  -> ("Taxi", "Subway")
    """
    if pd.isna(driver_text):
        return tuple()

    text = str(
        driver_text
    )

    matched_modalities = {
        modality
        for metric, modality in (
            DEMAND_METRIC_TO_MODALITY.items()
        )
        if re.search(
            rf"(?<![A-Za-z0-9_])"
            rf"{re.escape(metric)}"
            rf"(?![A-Za-z0-9_])",
            text,
        )
    }

    return tuple(
        sorted(
            matched_modalities,
            key=DEMAND_MODALITY_ORDER.get,
        )
    )



def _driver_tokens(
    driver_text: object,
) -> tuple[str, ...]:
    if pd.isna(driver_text):
        return tuple()

    return tuple(
        token.lower()
        for token in re.findall(
            r"[A-Za-z][A-Za-z0-9_]*",
            str(driver_text),
        )
    )


def _stress_family_signature(
    driver_text: object,
    stress_family: str,
) -> tuple[str, ...]:
    """
    Collapse the relevant metric drivers to modality-level stress signatures.

    Demand:
        trip/ridership/transfer metrics only.
    Congestion:
        speed/duration metrics only.
    All:
        every recognized mobility metric driver.
    """
    tokens = _driver_tokens(
        driver_text
    )

    modalities: set[str] = set()

    for token in tokens:
        is_taxi = token.startswith(
            "taxi_"
        )
        is_fhvhv = token.startswith(
            "fhvhv_"
        )
        is_subway = token.startswith(
            "subway_"
        )
        is_bus = (
            token.startswith("bus_")
            or "bus_speed" in token
            or token == "avg_bus_speed"
        )

        is_demand_metric = (
            token in DEMAND_METRIC_TO_MODALITY
        )

        is_congestion_metric = (
            (
                is_taxi
                or is_fhvhv
            )
            and (
                "speed" in token
                or "duration" in token
            )
        ) or (
            is_bus
            and "speed" in token
        )

        include = False

        if stress_family == "Demand":
            include = is_demand_metric
        elif stress_family == "Congestion":
            include = is_congestion_metric
        elif stress_family in {"All", "Both"}:
            include = (
                is_taxi
                or is_fhvhv
                or is_subway
                or is_bus
            )
        else:
            raise ValueError(
                f"Unsupported stress family: {stress_family}"
            )

        if not include:
            continue

        if is_taxi:
            modalities.add("Taxi")
        elif is_fhvhv:
            modalities.add("FHVHV")
        elif is_subway:
            modalities.add("Subway")
        elif is_bus:
            modalities.add("Bus")

    return tuple(
        sorted(
            modalities,
            key=DEMAND_MODALITY_ORDER.get,
        )
    )


def _modalities_for_family(
    stress_family: str,
) -> tuple[str, ...]:
    if stress_family == "Congestion":
        return (
            "Taxi",
            "FHVHV",
            "Bus",
        )

    return DEMAND_MODALITIES



def _metric_tokens_for_family(
    driver_text: object,
    stress_family: str,
) -> tuple[str, ...]:
    """Return unique underlying metric drivers relevant to the chosen family."""
    tokens = _driver_tokens(
        driver_text
    )

    selected: set[str] = set()

    for token in tokens:
        is_taxi = token.startswith(
            "taxi_"
        )
        is_fhvhv = token.startswith(
            "fhvhv_"
        )
        is_subway = token.startswith(
            "subway_"
        )
        is_bus = (
            token.startswith("bus_")
            or "bus_speed" in token
            or token == "avg_bus_speed"
        )

        is_demand_metric = (
            token in DEMAND_METRIC_TO_MODALITY
        )

        is_congestion_metric = (
            (
                is_taxi
                or is_fhvhv
            )
            and (
                "speed" in token
                or "duration" in token
            )
        ) or (
            is_bus
            and "speed" in token
        )

        include = False

        if stress_family == "Demand":
            include = is_demand_metric
        elif stress_family == "Congestion":
            include = is_congestion_metric
        elif stress_family in {"All", "Both"}:
            include = (
                is_taxi
                or is_fhvhv
                or is_subway
                or is_bus
            )
        else:
            raise ValueError(
                f"Unsupported stress family: {stress_family}"
            )

        if include:
            selected.add(
                token
            )

    return tuple(
        sorted(
            selected
        )
    )


def _metric_display_label(
    metric: str,
) -> str:
    return METRIC_DISPLAY_LABELS.get(
        metric,
        metric.replace(
            "_",
            " ",
        ).title(),
    )


def _metric_stress_role(metric: str) -> str:
    return (
        "Demand"
        if metric in DEMAND_METRIC_TO_MODALITY
        else "Congestion"
    )



def _metric_modality(
    metric: str,
) -> str | None:
    lower = metric.lower()

    if lower.startswith("taxi_"):
        return "Taxi"

    if lower.startswith("fhvhv_"):
        return "FHVHV"

    if lower.startswith("subway_"):
        return "Subway"

    if (
        lower.startswith("bus_")
        or lower == "avg_bus_speed"
    ):
        return "Bus"

    return None


def _defining_metric_breakdown(
    selected_events: pd.DataFrame,
    *,
    stress_family: str,
    selected_modalities: Sequence[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Return only driver metrics that define the selected family × modalities.

    Examples:
      Demand + Taxi       -> Taxi trips
      Demand + Subway     -> Subway ridership / transfers
      Congestion + Bus    -> Bus avg. speed
      Congestion + Taxi   -> Taxi speed / duration

    Shares are non-exclusive when a modality has multiple defining metrics.
    """
    if (
        selected_events.empty
        or not selected_modalities
    ):
        return pd.DataFrame(
            columns=[
                "metric",
                "metric_label",
                "event_count",
                "event_share",
            ]
        )

    selected_modality_set = set(
        selected_modalities
    )

    per_event_metrics = (
        selected_events[
            "stress_metric_driver_list"
        ].map(
            lambda value: tuple(
                metric
                for metric in (
                    _metric_tokens_for_family(
                        value,
                        stress_family,
                    )
                )
                if _metric_modality(
                    metric
                )
                in selected_modality_set
            )
        )
    )

    exploded = (
        per_event_metrics.explode()
        .dropna()
    )

    if exploded.empty:
        return pd.DataFrame(
            columns=[
                "metric",
                "metric_label",
                "event_count",
                "event_share",
            ]
        )

    result = (
        exploded.value_counts()
        .rename_axis(
            "metric"
        )
        .rename(
            "event_count"
        )
        .reset_index()
    )

    result[
        "event_share"
    ] = (
        result[
            "event_count"
        ]
        / len(
            selected_events
        )
    )

    result[
        "metric_label"
    ] = result[
        "metric"
    ].map(
        _metric_display_label
    )

    if stress_family == "All":
        result["metric_label"] = (
            result["metric_label"]
            + " ("
            + result["metric"].map(_metric_stress_role)
            + ")"
        )

    return result.head(
        top_n
    )


def _defining_metric_mix_text(
    selected_events: pd.DataFrame,
    *,
    stress_family: str,
    selected_modalities: Sequence[str],
    top_n: int = 5,
) -> str:
    breakdown = (
        _defining_metric_breakdown(
            selected_events,
            stress_family=(
                stress_family
            ),
            selected_modalities=(
                selected_modalities
            ),
            top_n=(
                top_n
            ),
        )
    )

    if breakdown.empty:
        return "No defining metric detail"

    return " · ".join(
        (
            f"{row.metric_label}: "
            f"{row.event_share * 100:.1f}%"
        )
        for row in (
            breakdown.itertuples()
        )
    )


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------
@st.cache_data(
    show_spinner=(
        "Loading mobility stress anomalies..."
    )
)

@st.cache_data(
    show_spinner=(
        "Loading mobility stress anomalies..."
    )
)
def _load_all_stress_events() -> pd.DataFrame:
    """
    Load the compact selected-event runtime prepared offline.

    WHY:
    Period, geography, mobility environment, stress family, and modality
    signatures are stable event attributes. They are built once in the shared
    runtime instead of reconstructed during every Streamlit page load.
    """
    events = (
        load_stress_anomaly_events_runtime()
        .copy()
    )

    required_columns = {
        "comparison_event_id",
        "taxi_zone_id",
        "date",
        "temporal_bucket",
        "period_group",
        "borough",
        "geography_group",
        "environment_group",
        "stress_metric_driver_list",
        "has_positive_demand_shock",
        "has_congestion_oriented",
        "stress_family",
        "modality_signature",
        "demand_modality_signature",
        "congestion_modality_signature",
    }

    missing = sorted(
        required_columns.difference(
            events.columns
        )
    )

    if missing:
        raise ValueError(
            "The shared stress-anomaly event runtime is missing required "
            "Raw 14 fields: "
            + ", ".join(missing)
        )

    return events.reset_index(
        drop=True
    )

@st.cache_data(
    show_spinner=False
)
def _load_demand_stress_events() -> pd.DataFrame:
    """
    Return selected events containing positive demand stress.

    WHY:
    Demand-specific modality participation is precomputed offline and decoded
    once by the shared loader, avoiding repeated metric-driver parsing.
    """
    events = (
        _load_all_stress_events()
        .copy()
    )

    events = events[
        events[
            "has_positive_demand_shock"
        ]
        .fillna(False)
        .astype(bool)
    ].copy()

    events["modality_signature"] = (
        events[
            "demand_modality_signature"
        ]
    )

    events = events[
        events[
            "modality_signature"
        ].map(bool)
    ].copy()

    return events.reset_index(
        drop=True
    )


@st.cache_data(
    show_spinner=(
        "Loading eligible observation denominator..."
    )
)
def _load_eligible_observation_context() -> pd.DataFrame:
    """
    Load the validated full observation universe with context already attached.

    The saved 3.3.6 incidence surface uses every Taxi Zone × date × daypart
    observation as its denominator. Geography and period-specific mobility
    environment are prepared offline in the shared runtime.
    """
    eligible = (
        load_stress_anomaly_observation_universe()
        .copy()
    )

    required_columns = {
        "taxi_zone_id",
        "date",
        "daypart",
        "day_type",
        "temporal_bucket",
        "period_group",
        "borough",
        "geography_group",
        "environment_group",
        "selected_finalist_flag",
    }

    missing = sorted(
        required_columns.difference(
            eligible.columns
        )
    )

    if missing:
        raise ValueError(
            "The shared stress-anomaly observation universe is missing "
            "required Raw 14 fields: "
            + ", ".join(missing)
        )

    return eligible.reset_index(
        drop=True
    )

# -----------------------------------------------------------------------------
# Comparison summaries
# -----------------------------------------------------------------------------
def _panel_summary(
    events: pd.DataFrame,
    *,
    group_column: str,
    group_label: str,
    intersection_count: int = HERO_INTERSECTION_COUNT,
) -> pd.DataFrame:
    """
    Rank this panel's own modality combinations from most to least common.

    The denominator for event_share is all positive-demand stress anomalies in
    this panel, not only the displayed top intersections.
    """
    panel = events[
        events[
            group_column
        ].eq(
            group_label
        )
    ].copy()

    panel_total = int(
        len(panel)
    )

    ranked = (
        panel.groupby(
            "modality_signature",
            observed=True,
            dropna=False,
        )
        .size()
        .rename(
            "event_count"
        )
        .reset_index()
        .sort_values(
            [
                "event_count",
                "modality_signature",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .head(
            intersection_count
        )
        .reset_index(
            drop=True
        )
    )

    if ranked.empty:
        return pd.DataFrame(
            columns=[
                "intersection_order",
                "signature",
                "signature_label",
                "event_count",
                "event_share",
                "panel_total",
                "defining_metric_mix",
            ]
        )

    ranked[
        "intersection_order"
    ] = np.arange(
        len(
            ranked
        )
    )

    ranked[
        "signature"
    ] = ranked[
        "modality_signature"
    ]

    ranked[
        "signature_label"
    ] = ranked[
        "signature"
    ].map(
        _signature_label
    )

    ranked[
        "event_share"
    ] = (
        ranked[
            "event_count"
        ]
        / panel_total
        if panel_total
        else np.nan
    )

    ranked[
        "panel_total"
    ] = panel_total

    defining_metric_mix: list[str] = []

    for signature in ranked[
        "signature"
    ]:
        signature_events = panel[
            panel[
                "modality_signature"
            ].map(
                lambda value: (
                    value == signature
                )
            )
        ]

        defining_metric_mix.append(
            _defining_metric_mix_text(
                signature_events,
                stress_family="Demand",
                selected_modalities=(
                    signature
                ),
            )
        )

    ranked[
        "defining_metric_mix"
    ] = defining_metric_mix

    return ranked[
        [
            "intersection_order",
            "signature",
            "signature_label",
            "event_count",
            "event_share",
            "panel_total",
            "defining_metric_mix",
        ]
    ]


def _comparison_summaries(
    events: pd.DataFrame,
    *,
    group_column: str,
    group_order: Sequence[str],
) -> dict[str, pd.DataFrame]:
    return {
        group_label: (
            _panel_summary(
                events,
                group_column=(
                    group_column
                ),
                group_label=(
                    group_label
                ),
            )
        )
        for group_label in group_order
    }


def _shared_y_max(
    summaries: dict[
        str,
        pd.DataFrame,
    ],
) -> float:
    max_share = max(
        (
            float(
                summary[
                    "event_share"
                ].max()
            )
            if not summary.empty
            else 0.0
        )
        for summary in (
            summaries.values()
        )
    )

    if max_share <= 0:
        return 0.10

    # Add room for labels, then round upward to a clean 10-point step.
    return min(
        1.0,
        max(
            0.10,
            np.ceil(
                (
                    max_share
                    * 100
                    + 7
                )
                / 10
            )
            * 10
            / 100,
        ),
    )


# -----------------------------------------------------------------------------
# UpSet-style chart
# -----------------------------------------------------------------------------
def _build_upset_panel(
    summary: pd.DataFrame,
    *,
    panel_label: str,
    panel_color: str,
    y_max: float,
    show_modality_labels: bool,
) -> go.Figure:
    """Build one compact, independently ranked modality-level UpSet panel."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[
            0.60,
            0.40,
        ],
        vertical_spacing=0.03,
    )

    signatures = summary[
        "signature"
    ].tolist()

    x_values = list(
        range(
            len(
                signatures
            )
        )
    )

    shares = (
        summary[
            "event_share"
        ]
        .fillna(0.0)
        .astype(float)
        .tolist()
    )

    counts = (
        summary[
            "event_count"
        ]
        .fillna(0)
        .astype(int)
        .tolist()
    )

    labels = (
        summary[
            "signature_label"
        ]
        .astype(str)
        .tolist()
    )

    panel_total = (
        int(
            summary[
                "panel_total"
            ].iloc[0]
        )
        if not summary.empty
        else 0
    )

    customdata = np.column_stack(
        [
            labels,
            counts,
            [
                panel_total
            ]
            * len(
                summary
            ),
            summary[
                "defining_metric_mix"
            ]
            .astype(str)
            .tolist(),
        ]
    )

    fig.add_trace(
        go.Bar(
            x=x_values,
            y=shares,
            marker={
                "color": panel_color,
                "line": {
                    "color": "white",
                    "width": 0.7,
                },
            },
            text=[
                (
                    f"{share * 100:.1f}%"
                    if share > 0
                    else ""
                )
                for share in shares
            ],
            textposition="outside",
            cliponaxis=False,
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Share of positive-demand stress anomalies: "
                "%{y:.1%}<br>"
                "Stress anomalies: %{customdata[1]:,}<br>"
                "Panel total: %{customdata[2]:,}<br>"
                "<br><b>Defining demand metrics</b><br>"
                "%{customdata[3]}"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    modality_y = {
        modality: (
            len(
                DEMAND_MODALITIES
            )
            - 1
            - index
        )
        for index, modality in enumerate(
            DEMAND_MODALITIES
        )
    }

    for x_index, signature in enumerate(
        signatures
    ):
        active_positions = [
            modality_y[
                modality
            ]
            for modality in (
                signature
            )
        ]

        if len(
            active_positions
        ) >= 2:
            fig.add_trace(
                go.Scatter(
                    x=[
                        x_index,
                        x_index,
                    ],
                    y=[
                        min(
                            active_positions
                        ),
                        max(
                            active_positions
                        ),
                    ],
                    mode="lines",
                    line={
                        "color": panel_color,
                        "width": 2.2,
                    },
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        inactive_modalities = [
            modality
            for modality in (
                DEMAND_MODALITIES
            )
            if modality not in signature
        ]

        fig.add_trace(
            go.Scatter(
                x=[
                    x_index
                ]
                * len(
                    inactive_modalities
                ),
                y=[
                    modality_y[
                        modality
                    ]
                    for modality in (
                        inactive_modalities
                    )
                ],
                mode="markers",
                marker={
                    "size": 7,
                    "color": (
                        INACTIVE_DOT_COLOR
                    ),
                    "line": {
                        "width": 0,
                    },
                },
                hoverinfo="skip",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[
                    x_index
                ]
                * len(
                    signature
                ),
                y=[
                    modality_y[
                        modality
                    ]
                    for modality in (
                        signature
                    )
                ],
                mode="markers",
                marker={
                    "size": 9,
                    "color": panel_color,
                    "line": {
                        "color": "white",
                        "width": 0.8,
                    },
                },
                text=[
                    (
                        f"<b>Intersection {x_index + 1}: "
                        f"{_signature_label(signature)}</b><br>"
                        f"Includes {', '.join(signature)}<br>"
                        f"Stress anomalies: {counts[x_index]:,}<br>"
                        f"Share of this panel: {shares[x_index]:.1%}"
                    )
                ] * len(signature),
                hovertemplate=(
                    "%{text}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(
        row=1,
        col=1,
        range=[
            0,
            y_max,
        ],
        tickformat=".0%",
        title_text=(
            "% of demand-stress anomalies"
            if show_modality_labels
            else ""
        ),
        gridcolor=GRID_COLOR,
        zeroline=False,
        tickfont={
            "size": 10,
            "color": TEXT_COLOR,
        },
        title_font={
            "size": 11,
            "color": TEXT_COLOR,
        },
    )

    modality_tickvals = [
        modality_y[
            modality
        ]
        for modality in (
            DEMAND_MODALITIES
        )
    ]

    fig.update_yaxes(
        row=2,
        col=1,
        range=[
            -0.5,
            len(
                DEMAND_MODALITIES
            )
            - 0.5,
        ],
        tickmode="array",
        tickvals=(
            modality_tickvals
        ),
        ticktext=(
            list(
                DEMAND_MODALITIES
            )
            if show_modality_labels
            else [
                ""
            ]
            * len(
                DEMAND_MODALITIES
            )
        ),
        showgrid=True,
        gridcolor=(
            "rgba(0, 109, 119, 0.07)"
        ),
        zeroline=False,
        tickfont={
            "size": 10,
            "color": TEXT_COLOR,
        },
    )

    fig.update_xaxes(
        row=1,
        col=1,
        tickmode="array",
        tickvals=x_values,
        ticktext=[
            str(
                index + 1
            )
            for index in (
                x_values
            )
        ],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    )

    fig.update_xaxes(
        row=2,
        col=1,
        tickmode="array",
        tickvals=x_values,
        ticktext=[
            str(
                index + 1
            )
            for index in (
                x_values
            )
        ],
        title_text="Intersection",
        showgrid=False,
        zeroline=False,
        tickfont={
            "size": 9,
            "color": TEXT_COLOR,
        },
        title_font={
            "size": 10,
            "color": TEXT_COLOR,
        },
    )

    fig.update_layout(
        height=390,
        margin={
            "l": (
                90
                if show_modality_labels
                else 8
            ),
            "r": 6,
            "t": 52,
            "b": 20,
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={
            "color": TEXT_COLOR,
            "size": 11,
        },
        title={
            "text": panel_label,
            "x": 0.5,
            "xanchor": "center",
            "font": {
                "size": 14,
                "color": (
                    BRAND_COLORS[
                        "dark_teal"
                    ]
                ),
            },
        },
        hoverlabel={
            "bgcolor": "white",
            "font": {
                "color": TEXT_COLOR,
            },
        },
        bargap=0.22,
    )

    return fig


def _render_upset_comparison(
    events: pd.DataFrame,
    *,
    group_column: str,
    group_order: Sequence[str],
    chart_key_prefix: str,
    group_display_labels: dict[
        str,
        str,
    ] | None = None,
) -> None:
    summaries = (
        _comparison_summaries(
            events,
            group_column=(
                group_column
            ),
            group_order=(
                group_order
            ),
        )
    )

    y_max = _shared_y_max(
        summaries
    )

    columns = st.columns(
        len(
            group_order
        ),
        gap="small",
    )

    for index, (
        column,
        group_label,
    ) in enumerate(
        zip(
            columns,
            group_order,
            strict=True,
        )
    ):
        display_label = (
            group_display_labels.get(
                group_label,
                group_label,
            )
            if group_display_labels
            else group_label
        )

        with column:
            st.plotly_chart(
                _build_upset_panel(
                    summaries[
                        group_label
                    ],
                    panel_label=(
                        display_label
                    ),
                    panel_color=(
                        PANEL_COLORS.get(
                            group_label,
                            BRAND_COLORS[
                                "dark_teal"
                            ],
                        )
                    ),
                    y_max=y_max,
                    show_modality_labels=(
                        index == 0
                    ),
                ),
                width="stretch",
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
                key=(
                    f"{chart_key_prefix}_"
                    f"{index}"
                ),
            )


# -----------------------------------------------------------------------------
# Insight helpers
# -----------------------------------------------------------------------------
def _group_signature_share(
    events: pd.DataFrame,
    *,
    group_column: str,
    group_label: str,
    signature: tuple[str, ...],
) -> float:
    """Return one exact modality-signature share from the full panel."""
    panel = events[
        events[
            group_column
        ].eq(
            group_label
        )
    ]

    if panel.empty:
        return np.nan

    match_count = int(
        panel[
            "modality_signature"
        ].map(
            lambda value: (
                value == signature
            )
        ).sum()
    )

    return (
        match_count
        / len(
            panel
        )
    )


def _pct(
    value: float,
) -> str:
    if pd.isna(
        value
    ):
        return "n/a"

    return (
        f"{value * 100:.1f}%"
    )


# -----------------------------------------------------------------------------
# Phase 3 custom explorer helpers
# -----------------------------------------------------------------------------
def _stress_family_filter(
    events: pd.DataFrame,
    stress_family: str,
) -> pd.DataFrame:
    """
    Filter events to one stress family using precomputed modality signatures.

    WHY:
    The shared runtime already derives All, Demand, and Congestion modality
    participation offline. Reusing those tuple columns avoids reparsing every
    event's serialized metric-driver list on each Streamlit rerun.
    """
    congestion = (
        events["has_congestion_oriented"]
        .fillna(False)
        .astype(bool)
    )

    demand = (
        events["has_positive_demand_shock"]
        .fillna(False)
        .astype(bool)
    )

    if stress_family == "Demand":
        mask = demand
        signature_column = "demand_modality_signature"

    elif stress_family == "Congestion":
        mask = congestion
        signature_column = "congestion_modality_signature"

    elif stress_family == "Both":
        mask = congestion & demand
        signature_column = "modality_signature"

    elif stress_family == "All":
        mask = pd.Series(
            True,
            index=events.index,
        )
        signature_column = "modality_signature"

    else:
        raise ValueError(
            f"Unsupported stress family: {stress_family}"
        )

    scoped = events.loc[
        mask
    ].copy()

    # WHY: downstream UpSet/storyline helpers expect one canonical
    # `modality_signature` column regardless of the selected stress family.
    scoped["modality_signature"] = (
        scoped[signature_column]
    )

    return scoped.loc[
        scoped["modality_signature"].map(bool)
    ].copy()


def _temporal_bucket_options(
    events: pd.DataFrame,
) -> list[str]:
    observed = {
        str(value)
        for value in events[
            "temporal_bucket"
        ]
        .dropna()
        .unique()
    }

    ordered = [
        value
        for value in TEMPORAL_BUCKET_ORDER
        if value in observed
    ]

    ordered.extend(
        sorted(
            observed.difference(
                ordered
            )
        )
    )

    return ordered


def _temporal_bucket_display(
    value: str,
) -> str:
    if value == "All temporal buckets":
        return value

    return TEMPORAL_BUCKET_LABELS.get(
        value,
        value.replace(
            "_",
            " ",
        ).title(),
    )


def _resolve_date_window(
    events: pd.DataFrame,
    time_scope: str,
    custom_range: tuple[
        object,
        object,
    ] | list[object] | object | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    study_start = pd.Timestamp(
        events[
            "date"
        ].min()
    ).normalize()

    study_end = pd.Timestamp(
        events[
            "date"
        ].max()
    ).normalize()

    cp_start = pd.Timestamp(
        CONGESTION_PRICING_START_DATE
    ).normalize()

    if time_scope == "Full study":
        return (
            study_start,
            study_end,
        )

    if time_scope == "Pre-CP":
        return (
            study_start,
            min(
                study_end,
                cp_start
                - pd.Timedelta(
                    days=1
                ),
            ),
        )

    if time_scope == "Post-CP":
        return (
            max(
                study_start,
                cp_start,
            ),
            study_end,
        )

    if time_scope != "Custom range":
        raise ValueError(
            f"Unsupported time scope: {time_scope}"
        )

    if isinstance(
        custom_range,
        (
            tuple,
            list,
        ),
    ):
        if len(
            custom_range
        ) == 2:
            start_date = pd.Timestamp(
                custom_range[0]
            )
            end_date = pd.Timestamp(
                custom_range[1]
            )

            return (
                min(
                    start_date,
                    end_date,
                ).normalize(),
                max(
                    start_date,
                    end_date,
                ).normalize(),
            )

    selected = pd.Timestamp(
        custom_range
    ).normalize()

    return (
        selected,
        selected,
    )


def _filter_date_and_bucket(
    frame: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    temporal_bucket: str,
) -> pd.DataFrame:
    scoped = frame[
        frame[
            "date"
        ].between(
            start_date,
            end_date,
            inclusive="both",
        )
    ].copy()

    if temporal_bucket != "All temporal buckets":
        scoped = scoped[
            scoped[
                "temporal_bucket"
            ]
            .astype(str)
            .eq(
                temporal_bucket
            )
        ].copy()

    return scoped





def _custom_panel_summary(
    events: pd.DataFrame,
    denominator: pd.DataFrame | None,
    *,
    group_column: str,
    group_label: str,
    measure: str,
    stress_family: str,
) -> pd.DataFrame:
    panel = events[
        events[
            group_column
        ].eq(
            group_label
        )
    ].copy()

    panel_total = int(
        len(
            panel
        )
    )

    eligible_total = (
        int(
            denominator[
                denominator[
                    group_column
                ].eq(
                    group_label
                )
            ].shape[0]
        )
        if denominator is not None
        else 0
    )

    ranked = (
        panel.groupby(
            "modality_signature",
            observed=True,
            dropna=False,
        )
        .size()
        .rename(
            "event_count"
        )
        .reset_index()
    )

    if ranked.empty:
        return pd.DataFrame(
            columns=[
                "signature",
                "signature_label",
                "event_count",
                "event_share",
                "eligible_total",
                "event_rate_per_1000",
                "display_value",
                "defining_metric_mix",
            ]
        )

    ranked[
        "signature"
    ] = ranked[
        "modality_signature"
    ]

    ranked[
        "signature_label"
    ] = ranked[
        "signature"
    ].map(
        _signature_label
    )

    ranked[
        "event_share"
    ] = (
        ranked[
            "event_count"
        ]
        / panel_total
        if panel_total
        else np.nan
    )

    ranked[
        "eligible_total"
    ] = eligible_total

    ranked[
        "event_rate_per_1000"
    ] = (
        ranked[
            "event_count"
        ]
        / eligible_total
        * 1000
        if eligible_total
        else np.nan
    )

    ranked[
        "display_value"
    ] = np.where(
        measure
        == "Composition share",
        ranked[
            "event_share"
        ],
        ranked[
            "event_rate_per_1000"
        ],
    )

    ranked = (
        ranked.sort_values(
            [
                "display_value",
                "event_count",
                "signature_label",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .head(
            HERO_INTERSECTION_COUNT
        )
        .reset_index(
            drop=True
        )
    )

    defining_metric_mix: list[str] = []

    for signature in ranked[
        "signature"
    ]:
        signature_events = panel[
            panel[
                "modality_signature"
            ].map(
                lambda value: (
                    value == signature
                )
            )
        ]

        defining_metric_mix.append(
            _defining_metric_mix_text(
                signature_events,
                stress_family=(
                    stress_family
                ),
                selected_modalities=(
                    signature
                ),
            )
        )

    ranked[
        "defining_metric_mix"
    ] = defining_metric_mix

    return ranked


def _custom_summaries(
    events: pd.DataFrame,
    denominator: pd.DataFrame | None,
    *,
    group_column: str,
    group_order: Sequence[str],
    measure: str,
    stress_family: str,
) -> dict[str, pd.DataFrame]:
    return {
        group_label: (
            _custom_panel_summary(
                events,
                denominator,
                group_column=(
                    group_column
                ),
                group_label=(
                    group_label
                ),
                measure=(
                    measure
                ),
                stress_family=(
                    stress_family
                ),
            )
        )
        for group_label in group_order
    }


def _custom_y_max(
    summaries: dict[
        str,
        pd.DataFrame,
    ],
    measure: str,
) -> float:
    max_value = max(
        (
            float(
                summary[
                    "display_value"
                ].max()
            )
            if not summary.empty
            else 0.0
        )
        for summary in (
            summaries.values()
        )
    )

    if measure == "Composition share":
        if max_value <= 0:
            return 0.10

        return min(
            1.0,
            max(
                0.10,
                np.ceil(
                    (
                        max_value
                        * 100
                        + 7
                    )
                    / 10
                )
                * 10
                / 100,
            ),
        )

    if max_value <= 0:
        return 1.0

    magnitude = 10 ** np.floor(
        np.log10(
            max_value
        )
    )

    step = magnitude / 2

    return float(
        np.ceil(
            (
                max_value
                * 1.15
            )
            / step
        )
        * step
    )


def _build_custom_upset_panel(
    summary: pd.DataFrame,
    *,
    panel_label: str,
    panel_color: str,
    y_max: float,
    show_modality_labels: bool,
    modalities: Sequence[str],
    measure: str,
    stress_family: str,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[
            0.60,
            0.40,
        ],
        vertical_spacing=0.03,
    )

    signatures = summary[
        "signature"
    ].tolist()

    x_values = list(
        range(
            len(
                signatures
            )
        )
    )

    values = (
        summary[
            "display_value"
        ]
        .fillna(0.0)
        .astype(float)
        .tolist()
    )

    customdata = np.column_stack(
        [
            summary[
                "signature_label"
            ]
            .astype(str)
            .tolist(),
            summary[
                "event_count"
            ]
            .fillna(0)
            .astype(int)
            .tolist(),
            summary[
                "event_share"
            ]
            .fillna(0.0)
            .astype(float)
            .tolist(),
            summary[
                "eligible_total"
            ]
            .fillna(0)
            .astype(int)
            .tolist(),
            summary[
                "event_rate_per_1000"
            ]
            .fillna(0.0)
            .astype(float)
            .tolist(),
            summary[
                "defining_metric_mix"
            ]
            .astype(str)
            .tolist(),
        ]
    )

    is_share = (
        measure
        == "Composition share"
    )

    fig.add_trace(
        go.Bar(
            x=x_values,
            y=values,
            marker={
                "color": panel_color,
                "line": {
                    "color": "white",
                    "width": 0.7,
                },
            },
            text=[
                (
                    f"{value * 100:.1f}%"
                    if is_share
                    else f"{value:.1f}"
                )
                for value in values
            ],
            textposition="outside",
            cliponaxis=False,
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                + (
                    "Composition share: %{customdata[2]:.1%}<br>"
                    if is_share
                    else "Events per 1,000 eligible observations: "
                    "%{customdata[4]:.2f}<br>"
                )
                + "Stress anomalies: %{customdata[1]:,}<br>"
                + (
                    "Eligible observations: %{customdata[3]:,}<br>"
                    if not is_share
                    else ""
                )
                + "<br><b>"
                + (
                    "Defining metrics"
                    if stress_family in {"All", "Both"}
                    else (
                        "Defining "
                        + stress_family.lower()
                        + " metrics"
                    )
                )
                + "</b><br>"
                + "%{customdata[5]}"
                + "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    modality_y = {
        modality: (
            len(
                modalities
            )
            - 1
            - index
        )
        for index, modality in enumerate(
            modalities
        )
    }

    for x_index, signature in enumerate(
        signatures
    ):
        active_positions = [
            modality_y[
                modality
            ]
            for modality in (
                signature
            )
            if modality in modality_y
        ]

        if len(
            active_positions
        ) >= 2:
            fig.add_trace(
                go.Scatter(
                    x=[
                        x_index,
                        x_index,
                    ],
                    y=[
                        min(
                            active_positions
                        ),
                        max(
                            active_positions
                        ),
                    ],
                    mode="lines",
                    line={
                        "color": panel_color,
                        "width": 2.2,
                    },
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        inactive_modalities = [
            modality
            for modality in (
                modalities
            )
            if modality not in signature
        ]

        fig.add_trace(
            go.Scatter(
                x=[
                    x_index
                ]
                * len(
                    inactive_modalities
                ),
                y=[
                    modality_y[
                        modality
                    ]
                    for modality in (
                        inactive_modalities
                    )
                ],
                mode="markers",
                marker={
                    "size": 7,
                    "color": (
                        INACTIVE_DOT_COLOR
                    ),
                    "line": {
                        "width": 0,
                    },
                },
                hoverinfo="skip",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        active_modalities = [
            modality
            for modality in (
                signature
            )
            if modality in modality_y
        ]

        fig.add_trace(
            go.Scatter(
                x=[
                    x_index
                ]
                * len(
                    active_modalities
                ),
                y=[
                    modality_y[
                        modality
                    ]
                    for modality in (
                        active_modalities
                    )
                ],
                mode="markers",
                marker={
                    "size": 9,
                    "color": panel_color,
                    "line": {
                        "color": "white",
                        "width": 0.8,
                    },
                },
                text=[
                    (
                        f"<b>Intersection {x_index + 1}: "
                        f"{_signature_label(signature)}</b><br>"
                        f"Includes {', '.join(active_modalities)}<br>"
                        f"Stress anomalies: {int(summary.iloc[x_index]['event_count']):,}<br>"
                        f"Share of this selection: "
                        f"{float(summary.iloc[x_index]['event_share']):.1%}"
                    )
                ] * len(active_modalities),
                hovertemplate=(
                    "%{text}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(
        row=1,
        col=1,
        range=[
            0,
            y_max,
        ],
        tickformat=(
            ".0%"
            if is_share
            else ".1f"
        ),
        title_text=(
            (
                "% of stress anomalies"
                if is_share
                else "Stress anomalies / 1,000"
            )
            if show_modality_labels
            else ""
        ),
        gridcolor=GRID_COLOR,
        zeroline=False,
        tickfont={
            "size": 10,
            "color": TEXT_COLOR,
        },
        title_font={
            "size": 11,
            "color": TEXT_COLOR,
        },
    )

    modality_tickvals = [
        modality_y[
            modality
        ]
        for modality in (
            modalities
        )
    ]

    fig.update_yaxes(
        row=2,
        col=1,
        range=[
            -0.5,
            len(
                modalities
            )
            - 0.5,
        ],
        tickmode="array",
        tickvals=(
            modality_tickvals
        ),
        ticktext=(
            list(
                modalities
            )
            if show_modality_labels
            else [
                ""
            ]
            * len(
                modalities
            )
        ),
        showgrid=True,
        gridcolor=(
            "rgba(0, 109, 119, 0.07)"
        ),
        zeroline=False,
        tickfont={
            "size": 10,
            "color": TEXT_COLOR,
        },
    )

    fig.update_xaxes(
        row=1,
        col=1,
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    )

    fig.update_xaxes(
        row=2,
        col=1,
        tickmode="array",
        tickvals=x_values,
        ticktext=[
            str(
                index + 1
            )
            for index in (
                x_values
            )
        ],
        title_text="Intersection",
        showgrid=False,
        zeroline=False,
        tickfont={
            "size": 9,
            "color": TEXT_COLOR,
        },
        title_font={
            "size": 10,
            "color": TEXT_COLOR,
        },
    )

    fig.update_layout(
        height=390,
        margin={
            "l": (
                90
                if show_modality_labels
                else 8
            ),
            "r": 6,
            "t": 52,
            "b": 20,
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={
            "color": TEXT_COLOR,
            "size": 11,
        },
        title={
            "text": panel_label,
            "x": 0.5,
            "xanchor": "center",
            "font": {
                "size": 14,
                "color": (
                    BRAND_COLORS[
                        "dark_teal"
                    ]
                ),
            },
        },
        hoverlabel={
            "bgcolor": "white",
            "font": {
                "color": TEXT_COLOR,
            },
        },
        bargap=0.22,
    )

    return fig


def _render_custom_comparison(
    events: pd.DataFrame,
    denominator: pd.DataFrame | None,
    *,
    group_column: str,
    group_order: Sequence[str],
    stress_family: str,
    measure: str,
    chart_key_prefix: str,
    group_display_labels: dict[
        str,
        str,
    ] | None = None,
) -> dict[str, pd.DataFrame]:
    summaries = (
        _custom_summaries(
            events,
            denominator,
            group_column=(
                group_column
            ),
            group_order=(
                group_order
            ),
            measure=(
                measure
            ),
            stress_family=(
                stress_family
            ),
        )
    )

    y_max = _custom_y_max(
        summaries,
        measure,
    )

    modalities = (
        _modalities_for_family(
            stress_family
        )
    )

    columns = st.columns(
        len(
            group_order
        ),
        gap="small",
    )

    for index, (
        column,
        group_label,
    ) in enumerate(
        zip(
            columns,
            group_order,
            strict=True,
        )
    ):
        display_label = (
            group_display_labels.get(
                group_label,
                group_label,
            )
            if group_display_labels
            else group_label
        )

        with column:
            if summaries[
                group_label
            ].empty:
                st.markdown(
                    f"**{display_label}**"
                )
                st.caption(
                    "No stress anomalies in this scope."
                )
                continue

            st.plotly_chart(
                _build_custom_upset_panel(
                    summaries[
                        group_label
                    ],
                    panel_label=(
                        display_label
                    ),
                    panel_color=(
                        PANEL_COLORS.get(
                            group_label,
                            BRAND_COLORS[
                                "dark_teal"
                            ],
                        )
                    ),
                    y_max=y_max,
                    show_modality_labels=(
                        index == 0
                    ),
                    modalities=(
                        modalities
                    ),
                    measure=(
                        measure
                    ),
                    stress_family=(
                        stress_family
                    ),
                ),
                width="stretch",
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
                key=(
                    f"{chart_key_prefix}_"
                    f"{index}"
                ),
            )

    return summaries


def _top_pattern_caption(
    summaries: dict[
        str,
        pd.DataFrame,
    ],
    *,
    measure: str,
) -> str:
    parts: list[str] = []

    for panel_label, summary in (
        summaries.items()
    ):
        if summary.empty:
            continue

        top = summary.iloc[0]

        if measure == "Composition share":
            value_text = (
                f"{float(top['display_value']) * 100:.1f}%"
            )
        else:
            value_text = (
                f"{float(top['display_value']):.1f}/1,000"
            )

        parts.append(
            f"**{panel_label}:** "
            f"{top['signature_label']} ({value_text})"
        )

    return " · ".join(
        parts
    )



def _clean_panel_label(
    panel_label: str,
    group_display_labels: dict[
        str,
        str,
    ] | None,
) -> str:
    display = (
        group_display_labels.get(
            panel_label,
            panel_label,
        )
        if group_display_labels
        else panel_label
    )

    return str(
        display
    ).replace(
        "<br>",
        " ",
    )


def _pattern_name_for_sentence(
    signature: tuple[str, ...],
) -> str:
    label = _signature_label(
        signature
    )

    if len(
        signature
    ) == 1:
        return (
            f"{label}-only"
        )

    return label


def _format_measure_value(
    value: float,
    measure: str,
) -> str:
    if pd.isna(
        value
    ):
        return "n/a"

    if measure == "Composition share":
        return (
            f"{value * 100:.1f}%"
        )

    return (
        f"{value:.1f}/1,000"
    )


def _full_pattern_measure_table(
    events: pd.DataFrame,
    denominator: pd.DataFrame | None,
    *,
    group_column: str,
    group_order: Sequence[str],
    measure: str,
) -> pd.DataFrame:
    """
    Build every exact modality-pattern value across all comparison panels.

    Unlike the UpSet display, this table is not truncated to the top six.
    It powers the dynamic summary cards and largest-gap insight.
    """
    records: list[
        dict[str, object]
    ] = []

    for panel_label in group_order:
        panel_events = events[
            events[
                group_column
            ].eq(
                panel_label
            )
        ]

        panel_event_total = int(
            len(
                panel_events
            )
        )

        eligible_total = (
            int(
                denominator[
                    denominator[
                        group_column
                    ].eq(
                        panel_label
                    )
                ].shape[0]
            )
            if denominator is not None
            else 0
        )

        counts = (
            panel_events[
                "modality_signature"
            ]
            .value_counts()
        )

        for signature, event_count in (
            counts.items()
        ):
            event_share = (
                int(
                    event_count
                )
                / panel_event_total
                if panel_event_total
                else np.nan
            )

            event_rate_per_1000 = (
                int(
                    event_count
                )
                / eligible_total
                * 1000
                if eligible_total
                else np.nan
            )

            display_value = (
                event_share
                if measure
                == "Composition share"
                else event_rate_per_1000
            )

            records.append(
                {
                    "panel_label": (
                        panel_label
                    ),
                    "signature": (
                        signature
                    ),
                    "signature_label": (
                        _signature_label(
                            signature
                        )
                    ),
                    "event_count": int(
                        event_count
                    ),
                    "panel_event_total": (
                        panel_event_total
                    ),
                    "eligible_total": (
                        eligible_total
                    ),
                    "event_share": (
                        event_share
                    ),
                    "event_rate_per_1000": (
                        event_rate_per_1000
                    ),
                    "display_value": (
                        display_value
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


def _custom_summary_insight(
    events: pd.DataFrame,
    denominator: pd.DataFrame | None,
    *,
    group_column: str,
    group_order: Sequence[str],
    group_display_labels: dict[
        str,
        str,
    ] | None,
    measure: str,
) -> dict[str, object] | None:
    if events.empty:
        return None

    full_patterns = (
        _full_pattern_measure_table(
            events,
            denominator,
            group_column=(
                group_column
            ),
            group_order=(
                group_order
            ),
            measure=(
                measure
            ),
        )
    )

    if full_patterns.empty:
        return None

    # Most involved mode = inclusive participation in the scoped event set.
    modality_counts = {
        modality: int(
            events[
                "modality_signature"
            ].map(
                lambda signature: (
                    modality
                    in signature
                )
            ).sum()
        )
        for modality in DEMAND_MODALITIES
    }

    most_involved_mode = max(
        modality_counts,
        key=(
            modality_counts.get
        ),
    )

    most_involved_count = (
        modality_counts[
            most_involved_mode
        ]
    )

    most_involved_share = (
        most_involved_count
        / len(
            events
        )
    )

    # Highest exact-pattern value in any panel.
    leading_row = (
        full_patterns.sort_values(
            [
                "display_value",
                "event_count",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .iloc[0]
    )

    leading_panel = (
        str(
            leading_row[
                "panel_label"
            ]
        )
    )

    leading_signature = tuple(
        leading_row[
            "signature"
        ]
    )

    leading_value = float(
        leading_row[
            "display_value"
        ]
    )

    # Find the exact modality combination whose value varies most across
    # valid comparison panels. A missing pattern in an otherwise valid panel
    # is a true zero.
    if measure == "Composition share":
        valid_panels = [
            panel_label
            for panel_label in (
                group_order
            )
            if int(
                events[
                    events[
                        group_column
                    ].eq(
                        panel_label
                    )
                ].shape[0]
            )
            > 0
        ]
    else:
        valid_panels = [
            panel_label
            for panel_label in (
                group_order
            )
            if (
                denominator is not None
                and int(
                    denominator[
                        denominator[
                            group_column
                        ].eq(
                            panel_label
                        )
                    ].shape[0]
                )
                > 0
            )
        ]

    gap_signature: tuple[
        str,
        ...
    ] | None = None
    gap_min_panel: str | None = None
    gap_max_panel: str | None = None
    gap_min_value = np.nan
    gap_max_value = np.nan
    gap_value = np.nan

    if len(
        valid_panels
    ) >= 2:
        signatures = (
            full_patterns[
                "signature"
            ]
            .drop_duplicates()
            .tolist()
        )

        best_key: tuple[
            float,
            float,
        ] | None = None

        for signature in signatures:
            signature_rows = full_patterns[
                full_patterns[
                    "signature"
                ].map(
                    lambda value: (
                        value
                        == signature
                    )
                )
            ]

            panel_values = {
                panel_label: 0.0
                for panel_label in (
                    valid_panels
                )
            }

            for _, row in (
                signature_rows.iterrows()
            ):
                panel_label = str(
                    row[
                        "panel_label"
                    ]
                )

                if panel_label in (
                    panel_values
                ):
                    panel_values[
                        panel_label
                    ] = float(
                        row[
                            "display_value"
                        ]
                    )

            current_min_panel = min(
                panel_values,
                key=(
                    panel_values.get
                ),
            )

            current_max_panel = max(
                panel_values,
                key=(
                    panel_values.get
                ),
            )

            current_min = (
                panel_values[
                    current_min_panel
                ]
            )

            current_max = (
                panel_values[
                    current_max_panel
                ]
            )

            current_gap = (
                current_max
                - current_min
            )

            # Tie-break toward patterns with the larger observed peak.
            current_key = (
                current_gap,
                current_max,
            )

            if (
                best_key is None
                or current_key
                > best_key
            ):
                best_key = (
                    current_key
                )
                gap_signature = tuple(
                    signature
                )
                gap_min_panel = (
                    current_min_panel
                )
                gap_max_panel = (
                    current_max_panel
                )
                gap_min_value = (
                    current_min
                )
                gap_max_value = (
                    current_max
                )
                gap_value = (
                    current_gap
                )

    return {
        "stress_event_count": int(
            len(
                events
            )
        ),
        "most_involved_mode": (
            most_involved_mode
        ),
        "most_involved_share": (
            most_involved_share
        ),
        "leading_signature": (
            leading_signature
        ),
        "leading_panel": (
            leading_panel
        ),
        "leading_value": (
            leading_value
        ),
        "gap_signature": (
            gap_signature
        ),
        "gap_min_panel": (
            gap_min_panel
        ),
        "gap_max_panel": (
            gap_max_panel
        ),
        "gap_min_value": (
            gap_min_value
        ),
        "gap_max_value": (
            gap_max_value
        ),
        "gap_value": (
            gap_value
        ),
        "group_display_labels": (
            group_display_labels
        ),
        "measure": (
            measure
        ),
    }


def _render_custom_summary_cards(
    insight: dict[
        str,
        object
    ],
) -> None:
    measure = str(
        insight[
            "measure"
        ]
    )

    display_labels = insight[
        "group_display_labels"
    ]

    leading_signature = tuple(
        insight[
            "leading_signature"
        ]
    )

    leading_panel = (
        _clean_panel_label(
            str(
                insight[
                    "leading_panel"
                ]
            ),
            display_labels,
        )
    )

    gap_signature = (
        tuple(
            insight[
                "gap_signature"
            ]
        )
        if insight[
            "gap_signature"
        ]
        is not None
        else None
    )

    card1, card2, card3, card4 = (
        st.columns(4)
    )

    with card1:
        st.metric(
            "Stress anomalies in scope",
            f"{int(insight['stress_event_count']):,}",
        )
        st.caption(
            "Stress anomalies matching the current controls."
        )

    with card2:
        st.metric(
            "Most involved mode",
            (
                f"{insight['most_involved_mode']} "
                f"({float(insight['most_involved_share']) * 100:.1f}%)"
            ),
        )
        st.caption(
            "Inclusive share of scoped stress anomalies."
        )

    with card3:
        st.metric(
            "Leading exact pattern",
            _pattern_name_for_sentence(
                leading_signature
            ),
        )
        st.caption(
            f"{leading_panel} · "
            f"{_format_measure_value(float(insight['leading_value']), measure)}"
        )

    with card4:
        if (
            gap_signature
            is None
            or pd.isna(
                insight[
                    "gap_value"
                ]
            )
        ):
            st.metric(
                "Largest panel gap",
                "n/a",
            )
            st.caption(
                "Not enough populated panels to compare."
            )
        else:
            gap_value = float(
                insight[
                    "gap_value"
                ]
            )

            if measure == "Composition share":
                gap_text = (
                    f"{gap_value * 100:.1f} pp"
                )
            else:
                gap_text = (
                    f"{gap_value:.1f}/1,000"
                )

            min_panel = (
                _clean_panel_label(
                    str(
                        insight[
                            "gap_min_panel"
                        ]
                    ),
                    display_labels,
                )
            )

            max_panel = (
                _clean_panel_label(
                    str(
                        insight[
                            "gap_max_panel"
                        ]
                    ),
                    display_labels,
                )
            )

            st.metric(
                "Largest panel gap",
                gap_text,
            )
            st.caption(
                f"{_pattern_name_for_sentence(gap_signature)} · "
                f"{min_panel} → {max_panel}"
            )


def _custom_tell_me_insight(
    insight: dict[
        str,
        object
    ],
) -> str:
    measure = str(
        insight[
            "measure"
        ]
    )

    display_labels = insight[
        "group_display_labels"
    ]

    gap_signature = (
        tuple(
            insight[
                "gap_signature"
            ]
        )
        if insight[
            "gap_signature"
        ]
        is not None
        else None
    )

    most_mode = str(
        insight[
            "most_involved_mode"
        ]
    )

    most_share = float(
        insight[
            "most_involved_share"
        ]
    )

    if gap_signature is None:
        return (
            f"**{most_mode} is the mode most often involved in this scope**, "
            f"appearing in **{most_share * 100:.1f}%** of stress anomalies."
        )

    min_panel = (
        _clean_panel_label(
            str(
                insight[
                    "gap_min_panel"
                ]
            ),
            display_labels,
        )
    )

    max_panel = (
        _clean_panel_label(
            str(
                insight[
                    "gap_max_panel"
                ]
            ),
            display_labels,
        )
    )

    min_value = float(
        insight[
            "gap_min_value"
        ]
    )

    max_value = float(
        insight[
            "gap_max_value"
        ]
    )

    gap_value = float(
        insight[
            "gap_value"
        ]
    )

    pattern_text = (
        _pattern_name_for_sentence(
            gap_signature
        )
    )

    if measure == "Composition share":
        comparison_text = (
            f"**{pattern_text} stress anomalies are the clearest separator across these "
            f"panels.** It ranges from **{min_value * 100:.1f}% in "
            f"{min_panel}** to **{max_value * 100:.1f}% in {max_panel}**—a "
            f"**{gap_value * 100:.1f} percentage-point gap**."
        )
    else:
        comparison_text = (
            f"**{pattern_text} stress anomalies show the largest frequency difference "
            f"across these panels.** It ranges from **{min_value:.1f} events "
            f"per 1,000 in {min_panel}** to **{max_value:.1f} per 1,000 in "
            f"{max_panel}**—a **{gap_value:.1f}-per-1,000 gap**."
        )

    return (
        comparison_text
        + " "
        + f"Across the full scope, **{most_mode}** is the mode most often "
        + f"involved, appearing in **{most_share * 100:.1f}%** of stress anomalies."
    )


# -----------------------------------------------------------------------------
# Phase 5 modality drill-down helpers
# -----------------------------------------------------------------------------
def _filter_selected_modalities(
    events: pd.DataFrame,
    *,
    selected_modalities: Sequence[str],
    match_rule: str,
) -> pd.DataFrame:
    """Return events matching either any or every selected modality.

    Both rules retain the event's complete modality signature so downstream
    charts can show which additional modes participated.
    """
    panel = events.copy()

    selected_set = set(
        selected_modalities
    )

    if not selected_set:
        return panel.iloc[
            0:0
        ].copy()

    if match_rule == MATCH_RULE_ANY:
        mask = panel["modality_signature"].map(
            lambda value: not selected_set.isdisjoint(set(value))
        )
    elif match_rule == MATCH_RULE_ALL:
        mask = panel["modality_signature"].map(
            lambda value: selected_set.issubset(set(value))
        )
    else:
        raise ValueError(f"Unsupported modality match rule: {match_rule}")

    return panel[mask].copy()


def _available_modalities(
    events: pd.DataFrame,
    *,
    stress_family: str,
) -> list[str]:
    present = {
        modality
        for signature in events[
            "modality_signature"
        ]
        for modality in signature
    }

    return [
        modality
        for modality in (
            _modalities_for_family(
                stress_family
            )
        )
        if modality in present
    ]


def _natural_join(
    values: Sequence[str],
    *,
    conjunction: str,
) -> str:
    items = [str(value) for value in values]

    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {conjunction} {items[1]}"

    return ", ".join(items[:-1]) + f", {conjunction} {items[-1]}"


def _clear_drilldown_geography_values() -> None:
    """Prevent a value from one segmentation from carrying into another."""
    for key in (
        "raw14_detail_policy_geography",
        "raw14_detail_borough",
        "raw14_detail_cluster",
    ):
        st.session_state.pop(key, None)



def _zone_display_series(
    events: pd.DataFrame,
) -> pd.Series:
    if (
        "zone" in events.columns
        and events[
            "zone"
        ].notna().any()
    ):
        zone_text = (
            events[
                "zone"
            ]
            .astype("string")
            .fillna("")
            .str.strip()
        )

        if (
            "borough" in events.columns
            and events[
                "borough"
            ].notna().any()
        ):
            borough_text = (
                events[
                    "borough"
                ]
                .astype("string")
                .fillna("")
                .str.strip()
            )

            return pd.Series(
                np.where(
                    borough_text.ne(""),
                    zone_text
                    + " · "
                    + borough_text,
                    zone_text,
                ),
                index=events.index,
                dtype="string",
            ).replace(
                "",
                pd.NA,
            ).fillna(
                "Taxi Zone "
                + events[
                    "taxi_zone_id"
                ]
                .astype(str)
            )

        return zone_text.replace(
            "",
            pd.NA,
        ).fillna(
            "Taxi Zone "
            + events[
                "taxi_zone_id"
            ]
            .astype(str)
        )

    return (
        "Taxi Zone "
        + events[
            "taxi_zone_id"
        ]
        .astype(str)
    )


def _horizontal_share_chart(
    frame: pd.DataFrame,
    *,
    category_column: str,
    value_column: str,
    hover_count_column: str,
    value_title: str,
    key_color: str,
    height: int = 390,
) -> go.Figure:
    if frame.empty:
        return go.Figure()

    chart_frame = (
        frame.sort_values(
            value_column,
            ascending=False,
        )
        .copy()
    )

    fig = go.Figure(
        go.Bar(
            x=chart_frame[
                value_column
            ],
            y=chart_frame[
                category_column
            ],
            orientation="h",
            marker={
                "color": key_color,
            },
            customdata=np.column_stack(
                [
                    chart_frame[
                        hover_count_column
                    ],
                ]
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                + value_title
                + ": %{x:.1%}<br>"
                + "Stress anomalies: %{customdata[0]:,}"
                + "<extra></extra>"
            ),
        )
    )

    maximum_share = float(chart_frame[value_column].max())
    share_tick_format = ".1%" if maximum_share < 0.02 else ".0%"

    fig.update_layout(
        height=height,
        margin={
            "l": 6,
            "r": 8,
            "t": 8,
            "b": 54,
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        font={
            "color": TEXT_COLOR,
            "size": 10,
        },
    )

    fig.update_xaxes(
        tickformat=share_tick_format,
        nticks=5,
        title_text=value_title,
        title_standoff=12,
        gridcolor=GRID_COLOR,
        zeroline=False,
        automargin=True,
    )

    fig.update_yaxes(
        autorange="reversed",
        title_text="",
        automargin=True,
    )

    return fig


def _metric_driver_breakdown(
    selected_events: pd.DataFrame,
    *,
    stress_family: str,
    selected_modalities: Sequence[str],
    top_n: int = 10,
) -> pd.DataFrame:
    return _defining_metric_breakdown(
        selected_events,
        stress_family=(
            stress_family
        ),
        selected_modalities=(
            selected_modalities
        ),
        top_n=(
            top_n
        ),
    )



def _temporal_bucket_breakdown(
    selected_events: pd.DataFrame,
    *,
    top_n: int = 8,
) -> pd.DataFrame:
    if selected_events.empty:
        return pd.DataFrame(
            columns=[
                "temporal_bucket",
                "bucket_label",
                "event_count",
                "event_share",
            ]
        )

    result = (
        selected_events[
            "temporal_bucket"
        ]
        .astype("string")
        .value_counts()
        .rename_axis(
            "temporal_bucket"
        )
        .rename(
            "event_count"
        )
        .reset_index()
    )

    result[
        "event_share"
    ] = (
        result[
            "event_count"
        ]
        / len(
            selected_events
        )
    )

    result[
        "bucket_label"
    ] = result[
        "temporal_bucket"
    ].map(
        _temporal_bucket_display
    )

    return result.head(
        top_n
    )


def _zone_breakdown(
    selected_events: pd.DataFrame,
    *,
    top_n: int = 8,
) -> pd.DataFrame:
    if selected_events.empty:
        return pd.DataFrame(
            columns=[
                "zone_label",
                "event_count",
                "event_share",
            ]
        )

    zone_labels = (
        _zone_display_series(
            selected_events
        )
    )

    result = (
        zone_labels.value_counts()
        .rename_axis(
            "zone_label"
        )
        .rename(
            "event_count"
        )
        .reset_index()
    )

    result[
        "event_share"
    ] = (
        result[
            "event_count"
        ]
        / len(
            selected_events
        )
    )

    return result.head(
        top_n
    )


def _format_strength(
    value: float | int | None,
) -> str:
    if value is None:
        return "n/a"

    try:
        numeric = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return "n/a"

    if np.isnan(
        numeric
    ):
        return "n/a"

    return f"{numeric:.2f}"


# =============================================================================
# Phase 1 hero re-scouting
# =============================================================================
SCOUT_FAMILIES = (
    "All",
    "Congestion",
    "Demand",
    "Both",
)


def _scout_family_events(
    events: pd.DataFrame,
    family: str,
) -> tuple[pd.DataFrame, int]:
    """
    Return one family with its precomputed modality signature.

    WHY:
    Hero scouting, weekly storylines, and explorer summaries all use the same
    family-specific signatures. Reusing runtime columns avoids repeated regex
    parsing and metric-to-modality reconstruction.
    """
    congestion = (
        events["has_congestion_oriented"]
        .fillna(False)
        .astype(bool)
    )

    demand = (
        events["has_positive_demand_shock"]
        .fillna(False)
        .astype(bool)
    )

    if family == "All":
        mask = pd.Series(
            True,
            index=events.index,
        )
        signature_column = "modality_signature"

    elif family == "Congestion":
        mask = congestion
        signature_column = "congestion_modality_signature"

    elif family == "Demand":
        mask = demand
        signature_column = "demand_modality_signature"

    elif family == "Both":
        mask = congestion & demand
        signature_column = "modality_signature"

    else:
        raise ValueError(
            f"Unsupported scouting family: {family}"
        )

    scoped = events.loc[
        mask
    ].copy()

    raw_count = len(scoped)

    scoped["modality_signature"] = (
        scoped[signature_column]
    )

    scoped = scoped.loc[
        scoped["modality_signature"].map(bool)
    ].copy()

    scoped["signature_label"] = (
        scoped["modality_signature"]
        .map(_signature_label)
    )

    return scoped, raw_count


def _signature_distribution(
    events: pd.DataFrame,
) -> pd.Series:
    """Return a complete modality-signature probability distribution."""
    if events.empty:
        return pd.Series(dtype=float)
    return events["signature_label"].value_counts(normalize=True).sort_index()


def _total_variation_distance(
    left: pd.Series,
    right: pd.Series,
) -> float:
    """Measure composition separation from 0 (same) to 1 (disjoint)."""
    labels = left.index.union(right.index)
    return float(
        0.5
        * (
            left.reindex(labels, fill_value=0.0)
            - right.reindex(labels, fill_value=0.0)
        )
        .abs()
        .sum()
    )


def _group_separation_summary(
    events: pd.DataFrame,
    *,
    group_column: str,
    group_order: Sequence[str],
) -> dict[str, object]:
    """Summarize pairwise signature-composition separation across panels."""
    distributions = {
        label: _signature_distribution(
            events.loc[events[group_column].eq(label)]
        )
        for label in group_order
        if not events.loc[events[group_column].eq(label)].empty
    }
    pairs: list[tuple[str, str, float]] = []
    for left_label, right_label in combinations(distributions, 2):
        pairs.append(
            (
                left_label,
                right_label,
                _total_variation_distance(
                    distributions[left_label],
                    distributions[right_label],
                ),
            )
        )
    if not pairs:
        return {
            "mean_tvd": np.nan,
            "max_tvd": np.nan,
            "most_separated_pair": "n/a",
        }
    strongest = max(pairs, key=lambda item: item[2])
    return {
        "mean_tvd": float(np.mean([item[2] for item in pairs])),
        "max_tvd": float(strongest[2]),
        "most_separated_pair": f"{strongest[0]} vs {strongest[1]}",
    }


def _top_signatures_by_group(
    events: pd.DataFrame,
    *,
    family: str,
    segmentation: str,
    group_column: str,
    group_order: Sequence[str],
    top_n: int = 3,
) -> pd.DataFrame:
    """Return compact per-panel signature leaders for the scouting report."""
    rows: list[dict[str, object]] = []
    for group_label in group_order:
        panel = events.loc[events[group_column].eq(group_label)]
        counts = panel["signature_label"].value_counts().head(top_n)
        panel_total = len(panel)
        for rank, (signature, count) in enumerate(counts.items(), start=1):
            rows.append(
                {
                    "family": family,
                    "segmentation": segmentation,
                    "group": group_label,
                    "rank": rank,
                    "signature": signature,
                    "events": int(count),
                    "composition_share": count / panel_total if panel_total else np.nan,
                }
            )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Scouting replacement Raw 14 hero stories...")
def _build_phase_1_scouting_report() -> str:
    """Build one copy-pasteable report for choosing the replacement hero."""
    events = _load_all_stress_events()
    eligible = _load_eligible_observation_context()
    family_frames: dict[str, pd.DataFrame] = {}
    family_raw_counts: dict[str, int] = {}

    for family in SCOUT_FAMILIES:
        family_frame, raw_count = _scout_family_events(events, family)
        family_frames[family] = family_frame
        family_raw_counts[family] = raw_count

    congestion = events["has_congestion_oriented"].fillna(False).astype(bool)
    demand = events["has_positive_demand_shock"].fillna(False).astype(bool)
    exclusive_mix = pd.DataFrame(
        [
            {"stress_family_exclusive": "Congestion-only", "events": int((congestion & ~demand).sum())},
            {"stress_family_exclusive": "Demand-only", "events": int((~congestion & demand).sum())},
            {"stress_family_exclusive": "Both", "events": int((congestion & demand).sum())},
            {"stress_family_exclusive": "Neither flag", "events": int((~congestion & ~demand).sum())},
        ]
    )
    exclusive_mix["composition_share"] = (
        exclusive_mix["events"] / exclusive_mix["events"].sum()
    )

    family_coverage_rows: list[dict[str, object]] = []
    policy_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    shift_rows: list[dict[str, object]] = []
    group_leaders: list[pd.DataFrame] = []

    eligible_period_counts = eligible["period_group"].value_counts()

    for family, family_events in family_frames.items():
        raw_count = family_raw_counts[family]
        family_coverage_rows.append(
            {
                "family": family,
                "flagged_events": raw_count,
                "events_with_recognized_signature": len(family_events),
                "signature_coverage": len(family_events) / raw_count if raw_count else np.nan,
                "distinct_signatures": family_events["signature_label"].nunique(),
            }
        )

        for period in POLICY_PERIOD_ORDER:
            panel = family_events.loc[family_events["period_group"].eq(period)]
            eligible_count = int(eligible_period_counts.get(period, 0))
            policy_rows.append(
                {
                    "family": family,
                    "period": period,
                    "eligible_observations": eligible_count,
                    "stress_anomalies": len(panel),
                    "incidence_per_1k": len(panel) / eligible_count * 1000 if eligible_count else np.nan,
                    "leading_signature": (
                        panel["signature_label"].value_counts().index[0]
                        if not panel.empty
                        else "n/a"
                    ),
                    "leading_signature_share": (
                        panel["signature_label"].value_counts(normalize=True).iloc[0]
                        if not panel.empty
                        else np.nan
                    ),
                }
            )

        pre_distribution = _signature_distribution(
            family_events.loc[family_events["period_group"].eq("Pre-CP")]
        )
        post_distribution = _signature_distribution(
            family_events.loc[family_events["period_group"].eq("Post-CP")]
        )
        signature_labels = pre_distribution.index.union(post_distribution.index)
        signature_shift = pd.DataFrame(
            {
                "family": family,
                "signature": signature_labels,
                "pre_share": pre_distribution.reindex(signature_labels, fill_value=0.0).values,
                "post_share": post_distribution.reindex(signature_labels, fill_value=0.0).values,
            }
        )
        signature_shift["post_minus_pre_share"] = (
            signature_shift["post_share"] - signature_shift["pre_share"]
        )
        signature_shift["absolute_share_shift"] = signature_shift[
            "post_minus_pre_share"
        ].abs()
        shift_rows.extend(
            signature_shift.nlargest(8, "absolute_share_shift").to_dict("records")
        )

        geography_separation = _group_separation_summary(
            family_events,
            group_column="geography_group",
            group_order=POLICY_GEOGRAPHY_ORDER,
        )
        environment_separation = _group_separation_summary(
            family_events,
            group_column="environment_group",
            group_order=MOBILITY_ENVIRONMENT_ORDER,
        )
        overall_distribution = _signature_distribution(family_events)
        candidate_rows.append(
            {
                "family": family,
                "events": len(family_events),
                "leading_signature": overall_distribution.idxmax() if not overall_distribution.empty else "n/a",
                "leading_signature_share": overall_distribution.max() if not overall_distribution.empty else np.nan,
                "policy_period_tvd": _total_variation_distance(pre_distribution, post_distribution),
                "policy_geography_mean_tvd": geography_separation["mean_tvd"],
                "policy_geography_max_tvd": geography_separation["max_tvd"],
                "policy_geography_strongest_pair": geography_separation["most_separated_pair"],
                "environment_mean_tvd": environment_separation["mean_tvd"],
                "environment_max_tvd": environment_separation["max_tvd"],
                "environment_strongest_pair": environment_separation["most_separated_pair"],
            }
        )

        group_leaders.extend(
            [
                _top_signatures_by_group(
                    family_events,
                    family=family,
                    segmentation="Policy period",
                    group_column="period_group",
                    group_order=POLICY_PERIOD_ORDER,
                ),
                _top_signatures_by_group(
                    family_events,
                    family=family,
                    segmentation="Policy geography",
                    group_column="geography_group",
                    group_order=POLICY_GEOGRAPHY_ORDER,
                ),
                _top_signatures_by_group(
                    family_events,
                    family=family,
                    segmentation="Mobility environment",
                    group_column="environment_group",
                    group_order=MOBILITY_ENVIRONMENT_ORDER,
                ),
            ]
        )

    coverage = pd.DataFrame(family_coverage_rows)
    policy_summary = pd.DataFrame(policy_rows)
    candidates = pd.DataFrame(candidate_rows)
    shifts = pd.DataFrame(shift_rows).sort_values(
        ["family", "absolute_share_shift"], ascending=[True, False]
    )
    leaders = pd.concat(group_leaders, ignore_index=True)

    for frame, columns in [
        (exclusive_mix, ["composition_share"]),
        (coverage, ["signature_coverage"]),
        (policy_summary, ["incidence_per_1k", "leading_signature_share"]),
        (candidates, ["leading_signature_share", "policy_period_tvd", "policy_geography_mean_tvd", "policy_geography_max_tvd", "environment_mean_tvd", "environment_max_tvd"]),
        (shifts, ["pre_share", "post_share", "post_minus_pre_share", "absolute_share_shift"]),
        (leaders, ["composition_share"]),
    ]:
        frame[columns] = frame[columns].round(3)

    qa = pd.DataFrame(
        [
            {"check": "Selected stress-anomaly rows", "result": len(events)},
            {"check": "Distinct event IDs", "result": events["comparison_event_id"].nunique()},
            {"check": "Duplicate event IDs", "result": events["comparison_event_id"].duplicated().sum()},
            {"check": "First event date", "result": events["date"].min().date()},
            {"check": "Last event date", "result": events["date"].max().date()},
            {"check": "Eligible observation rows", "result": len(eligible)},
        ]
    )

    sections = [
        "RAW 14 — PHASE 1 HERO RE-SCOUTING REPORT",
        "",
        "## CONTRACT QA",
        qa.to_csv(index=False).strip(),
        "",
        "## EXCLUSIVE STRESS-FAMILY MIX",
        exclusive_mix.to_csv(index=False).strip(),
        "",
        "## SIGNATURE COVERAGE",
        coverage.to_csv(index=False).strip(),
        "",
        "## FAMILY × POLICY PERIOD INCIDENCE AND LEADING SIGNATURE",
        policy_summary.to_csv(index=False).strip(),
        "",
        "## CANDIDATE HERO SEPARATION SCORES",
        candidates.to_csv(index=False).strip(),
        "",
        "## LARGEST PRE/POST SIGNATURE-SHARE SHIFTS",
        shifts.to_csv(index=False).strip(),
        "",
        "## TOP SIGNATURES BY FAMILY × SEGMENTATION × GROUP",
        leaders.to_csv(index=False).strip(),
        "",
        "## HOW TO READ THE SCORES",
        "TVD is total-variation distance in full modality-signature composition: 0 means identical distributions and 1 means no overlap. Policy-period TVD compares Pre-CP with Post-CP. Geography and environment mean TVD average every available pair; max TVD identifies the strongest pair. Hero choice should balance thesis importance, event volume, signature coverage, and separation—not maximize TVD alone.",
        "",
        "## HERO HIERARCHY CONSTRAINT",
        "All stress or Congestion should lead. Demand may retain one focused hero tab when its modality story is distinctive, but it should not define the whole page.",
    ]
    return "\n".join(sections)



# =============================================================================
# Weekly stress-storyline prototype
# =============================================================================
STORYLINE_MODALITY_ORDER = (
    "Taxi",
    "FHVHV",
    "Subway",
    "Bus",
)

STORYLINE_MODALITY_Y = {
    "Taxi": 3.0,
    "FHVHV": 2.0,
    "Subway": 1.0,
    "Bus": 0.0,
}

STORYLINE_MODALITY_COLORS = {
    "Taxi": BRAND_COLORS["terracotta"],
    "FHVHV": BRAND_COLORS["dark_teal"],
    "Subway": BRAND_COLORS["seafoam"],
    "Bus": BRAND_COLORS["pale_peach"],
}


def _weekly_storyline_surface(
    family_events: pd.DataFrame,
    eligible: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate exact modality combinations to week.

    WHY: daily events are too dense for a storyline, while monthly aggregation
    can erase short-lived combinations. Weekly is the first defensible grain to
    test, not a permanent assumption.
    """
    if family_events.empty:
        return pd.DataFrame()

    weekly = family_events.copy()
    weekly["week_start"] = (
        weekly["date"]
        - pd.to_timedelta(
            weekly["date"].dt.weekday,
            unit="D",
        )
    )

    # WHY: the canonical event loader does not require directional strength.
    # Keep this prototype on fields guaranteed by Raw 14's event contract.
    weekly_summary = (
        weekly.groupby(
            [
                "week_start",
                "modality_signature",
                "signature_label",
            ],
            observed=True,
            dropna=False,
        )
        .agg(
            event_count=(
                "comparison_event_id",
                "nunique",
            ),
        )
        .reset_index()
    )

    weekly_totals = (
        weekly.groupby(
            "week_start",
            observed=True,
        )["comparison_event_id"]
        .nunique()
        .rename("weekly_stress_events")
        .reset_index()
    )

    eligible_weekly = eligible.copy()
    eligible_weekly["week_start"] = (
        eligible_weekly["date"]
        - pd.to_timedelta(
            eligible_weekly["date"].dt.weekday,
            unit="D",
        )
    )

    eligible_counts = (
        eligible_weekly.groupby(
            "week_start",
            observed=True,
        )
        .size()
        .rename("eligible_observations")
        .reset_index()
    )

    weekly_summary = (
        weekly_summary
        .merge(
            weekly_totals,
            on="week_start",
            how="left",
            validate="many_to_one",
        )
        .merge(
            eligible_counts,
            on="week_start",
            how="left",
            validate="many_to_one",
        )
    )

    weekly_summary["incidence_per_1k"] = (
        1000.0
        * weekly_summary["event_count"]
        / weekly_summary["eligible_observations"]
    )

    weekly_summary["weekly_stress_share"] = (
        weekly_summary["event_count"]
        / weekly_summary["weekly_stress_events"]
    )

    return weekly_summary.sort_values(
        [
            "week_start",
            "event_count",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(drop=True)


def _thickness_scout(
    weekly_surface: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare candidate thickness measures on the same weekly combinations.

    WHY: thickness only earns a place if its dynamic range is visible without
    allowing a few extreme weeks to dominate the whole graphic.
    """
    candidates = {
        "Raw event count": "event_count",
        "Incidence per 1,000 eligible": "incidence_per_1k",
        "Share of that week's stress": "weekly_stress_share",
    }

    rows = []

    for label, column in candidates.items():
        values = pd.to_numeric(
            weekly_surface[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        p10 = float(values.quantile(0.10))
        p50 = float(values.quantile(0.50))
        p90 = float(values.quantile(0.90))
        p95 = float(values.quantile(0.95))
        maximum = float(values.max())

        rows.append(
            {
                "candidate": label,
                "p10": p10,
                "median": p50,
                "p90": p90,
                "p95": p95,
                "max": maximum,
                "p90_to_median": (
                    p90 / p50
                    if p50 > 0
                    else np.nan
                ),
                "max_to_median": (
                    maximum / p50
                    if p50 > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def _width_from_incidence(
    values: pd.Series,
) -> pd.Series:
    """
    Map incidence to a restrained visible width.

    WHY: percentile clipping preserves relative prevalence without letting one
    extreme combination turn into an oversized ribbon.
    """
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).fillna(0.0)

    if numeric.empty:
        return numeric

    low = float(numeric.quantile(0.10))
    high = float(numeric.quantile(0.95))

    if not np.isfinite(high) or high <= low:
        return pd.Series(
            4.0,
            index=numeric.index,
        )

    clipped = numeric.clip(
        lower=low,
        upper=high,
    )

    return (
        2.5
        + 7.5
        * (clipped - low)
        / (high - low)
    )


def _storyline_display_events(
    weekly_surface: pd.DataFrame,
    *,
    max_events: int,
    modality_counts: tuple[int, ...] = (2, 3, 4),
) -> pd.DataFrame:
    """Select events while preserving rare higher-order convergence."""
    if weekly_surface.empty:
        return weekly_surface.copy()

    allowed = {int(value) for value in modality_counts}
    candidates = weekly_surface.copy()
    candidates["modality_count"] = candidates["modality_signature"].map(len)
    candidates = candidates.loc[
        candidates["modality_count"].isin(allowed)
    ].copy()

    if candidates.empty:
        return candidates

    reserved_parts = []

    # WHY: pure Top-N incidence hid nearly all 3-way events and every 4-way
    # event in the audit. Reserve limited representation without changing
    # thickness, which still communicates actual prevalence.
    if 4 in allowed:
        four_way = candidates.loc[
            candidates["modality_count"].eq(4)
        ].nlargest(1, "incidence_per_1k")
        if not four_way.empty:
            reserved_parts.append(four_way)

    if 3 in allowed:
        three_way = candidates.loc[
            candidates["modality_count"].eq(3)
        ].nlargest(min(3, max_events), "incidence_per_1k")
        if not three_way.empty:
            reserved_parts.append(three_way)

    reserved = (
        pd.concat(reserved_parts, ignore_index=False)
        .drop_duplicates(subset=["week_start", "modality_signature"])
        if reserved_parts
        else candidates.iloc[0:0].copy()
    )

    reserved_keys = set(zip(
        pd.to_datetime(reserved["week_start"]),
        reserved["modality_signature"].map(tuple),
    ))

    remainder = candidates.loc[
        [
            (pd.Timestamp(week), tuple(signature)) not in reserved_keys
            for week, signature in zip(
                candidates["week_start"],
                candidates["modality_signature"],
            )
        ]
    ].copy()

    fill = remainder.nlargest(
        max(0, int(max_events) - len(reserved)),
        "incidence_per_1k",
    )

    selected = (
        pd.concat([reserved, fill], ignore_index=True)
        .drop_duplicates(subset=["week_start", "modality_signature"])
        .sort_values(
            ["week_start", "incidence_per_1k"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )

    selected["line_width"] = _width_from_incidence(
        selected["incidence_per_1k"]
    )
    return selected

def _build_stress_storyline_figure(
    weekly_surface: pd.DataFrame,
    *,
    family_label: str,
    max_events: int,
    modality_counts: tuple[int, ...] = (2, 3, 4),
) -> go.Figure:
    """
    Prototype a storyline where modes converge only for shared stress events.

    The y positions are categorical home lanes, not a continuous measurement.
    Ribbon thickness encodes weekly incidence per 1,000 eligible observations.
    """
    display_events = _storyline_display_events(
        weekly_surface,
        max_events=max_events,
        modality_counts=modality_counts,
    )

    figure = go.Figure()

    if weekly_surface.empty:
        return figure

    x_min = weekly_surface["week_start"].min()
    x_max = weekly_surface["week_start"].max()

    # Stable home lanes: no vertical jitter carries quantitative meaning.
    for modality in STORYLINE_MODALITY_ORDER:
        y_home = STORYLINE_MODALITY_Y[modality]

        figure.add_trace(
            go.Scatter(
                x=[
                    x_min,
                    x_max,
                ],
                y=[
                    y_home,
                    y_home,
                ],
                mode="lines",
                name=modality,
                line={
                    "color": STORYLINE_MODALITY_COLORS[modality],
                    "width": 2.2,
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Each selected multi-mode weekly event pulls only participating modes
    # toward a shared midpoint. The central striped flag makes membership
    # readable even when several lanes converge.
    approach_days = pd.Timedelta(days=4)

    for _, event in display_events.iterrows():
        week = pd.Timestamp(
            event["week_start"]
        )
        signature = tuple(
            event["modality_signature"]
        )
        participant_y = [
            STORYLINE_MODALITY_Y[modality]
            for modality in signature
        ]
        merge_y = float(
            np.mean(participant_y)
        )
        width = float(
            event["line_width"]
        )

        hover = (
            f"<b>{event['signature_label']}</b>"
            f"<br>Week of {week:%b %d, %Y}"
            f"<br>{family_label} stress"
            f"<br>Events: {int(event['event_count']):,}"
            f"<br>Incidence: {event['incidence_per_1k']:.2f} per 1,000"
            f"<br>Share of weekly stress: {event['weekly_stress_share'] * 100:.1f}%"
            "<extra></extra>"
        )

        for modality in signature:
            y_home = STORYLINE_MODALITY_Y[
                modality
            ]

            figure.add_trace(
                go.Scatter(
                    x=[
                        week - approach_days,
                        week,
                        week + approach_days,
                    ],
                    y=[
                        y_home,
                        merge_y,
                        y_home,
                    ],
                    mode="lines",
                    line={
                        "color": STORYLINE_MODALITY_COLORS[modality],
                        "width": width,
                        "shape": "spline",
                        "smoothing": 0.75,
                    },
                    hovertemplate=hover,
                    showlegend=False,
                )
            )

        # Banded flag: make exact modality membership visually unmistakable.
        # WHY: the first prototype used a ~3-day-wide flag, which collapsed to
        # only a few screen pixels across a three-year timeline.
        flag_half_width = pd.Timedelta(days=6)
        band_height = 0.18
        total_height = (
            band_height
            * len(signature)
        )
        band_bottom = (
            merge_y
            - total_height / 2
        )
        band_top = (
            merge_y
            + total_height / 2
        )

        # A white-backed outline separates the flag from the converging paths.
        figure.add_shape(
            type="rect",
            x0=week - flag_half_width,
            x1=week + flag_half_width,
            y0=band_bottom - 0.035,
            y1=band_top + 0.035,
            line={
                "color": BRAND_COLORS["dark_teal"],
                "width": 1.4,
            },
            fillcolor="rgba(255,255,255,0.96)",
            layer="above",
        )

        for band_index, modality in enumerate(signature):
            y0 = (
                band_bottom
                + band_index * band_height
            )
            y1 = y0 + band_height

            figure.add_shape(
                type="rect",
                x0=week - flag_half_width,
                x1=week + flag_half_width,
                y0=y0,
                y1=y1,
                line={
                    "color": "white",
                    "width": 1.0,
                },
                fillcolor=STORYLINE_MODALITY_COLORS[
                    modality
                ],
                layer="above",
            )

        # A visible outlined center marker reinforces the merge point and also
        # provides a generous hover target without inventing another encoding.
        figure.add_trace(
            go.Scatter(
                x=[week],
                y=[merge_y],
                mode="markers",
                marker={
                    "size": max(
                        15.0,
                        width * 2.0,
                    ),
                    "color": "rgba(255,255,255,0.01)",
                    "line": {
                        "color": BRAND_COLORS["dark_teal"],
                        "width": 1.3,
                    },
                },
                hovertemplate=hover,
                showlegend=False,
            )
        )

    cp_start = pd.Timestamp(
        CONGESTION_PRICING_START_DATE
    )

    if x_min <= cp_start <= x_max:
        figure.add_vline(
            x=cp_start,
            line_width=1.2,
            line_dash="dot",
            line_color=BRAND_COLORS["terracotta"],
            opacity=0.75,
        )

        figure.add_annotation(
            x=cp_start,
            y=1.0,
            yref="paper",
            text="Jan 5, 2025 · congestion pricing begins",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font={
                "size": 10,
                "color": BRAND_COLORS["terracotta"],
            },
        )

    figure.update_layout(
        title={
            "text": (
                f"How did modes come together during {family_label.lower()} "
                "stress?"
            ),
            "x": 0,
            "xanchor": "left",
            "font": {
                "size": 18,
                "color": BRAND_COLORS["dark_teal"],
            },
        },
        height=570,
        margin={
            "l": 90,
            "r": 35,
            "t": 75,
            "b": 55,
        },
        hovermode="closest",
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor=BRAND_COLORS["ice"],
    )

    figure.update_xaxes(
        title=None,
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
        title=None,
        tickmode="array",
        tickvals=[
            STORYLINE_MODALITY_Y[modality]
            for modality in STORYLINE_MODALITY_ORDER
        ],
        ticktext=list(
            STORYLINE_MODALITY_ORDER
        ),
        range=[
            -0.55,
            3.55,
        ],
        showgrid=False,
        zeroline=False,
        tickfont={
            "color": BRAND_COLORS["dark_teal"],
        },
    )

    return figure



def _storyline_takeaway(
    weekly_surface: pd.DataFrame,
    *,
    max_events: int,
    family_label: str,
    modality_counts: tuple[int, ...] = (2, 3, 4),
) -> str:
    """Explain the most prominent shared mode pattern among displayed stress weeks."""
    displayed = _storyline_display_events(
        weekly_surface,
        max_events=max_events,
        modality_counts=modality_counts,
    )

    if displayed.empty:
        return ""

    strongest = displayed.loc[
        displayed["incidence_per_1k"].idxmax()
    ]

    week = pd.Timestamp(
        strongest["week_start"]
    )

    return (
        f"Among the displayed **{family_label.lower()}-stress** weeks, "
        f"**{strongest['signature_label']}** is the most prominent shared mode "
        f"pattern. Its strongest episode occurred in the week of "
        f"**{week:%b %d, %Y}**, reaching "
        f"**{float(strongest['incidence_per_1k']):.2f} stress events per "
        "1,000 eligible observations**."
    )


def _shared_modality_coverage_audit(
    all_events: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    max_events: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check whether the current Top-N storyline hides 3-/4-mode weeks."""
    coverage_rows = []
    example_frames = []

    for family_name in ["All", "Congestion", "Demand", "Both"]:
        family_events, _ = _scout_family_events(all_events, family_name)
        weekly = _weekly_storyline_surface(family_events, eligible)
        if weekly.empty:
            continue

        weekly = weekly.copy()
        weekly["modality_count"] = weekly["modality_signature"].map(len)
        displayed = _storyline_display_events(weekly, max_events=max_events)

        displayed_keys = (
            set(zip(
                pd.to_datetime(displayed["week_start"]),
                displayed["modality_signature"].map(tuple),
            ))
            if not displayed.empty else set()
        )

        for modality_count in [2, 3, 4]:
            subset = weekly.loc[
                weekly["modality_count"].eq(modality_count)
            ].copy()

            if subset.empty:
                coverage_rows.append({
                    "family": family_name,
                    "modalities": modality_count,
                    "available_week_combinations": 0,
                    "distinct_signatures": 0,
                    "displayed_in_top_n": 0,
                    "display_share_pct": 0.0,
                    "median_incidence_per_1k": np.nan,
                    "max_incidence_per_1k": np.nan,
                })
                continue

            subset["_displayed"] = [
                (pd.Timestamp(week), tuple(signature)) in displayed_keys
                for week, signature in zip(
                    subset["week_start"],
                    subset["modality_signature"],
                )
            ]
            available = int(len(subset))
            retained = int(subset["_displayed"].sum())

            coverage_rows.append({
                "family": family_name,
                "modalities": modality_count,
                "available_week_combinations": available,
                "distinct_signatures": int(
                    subset["signature_label"].nunique()
                ),
                "displayed_in_top_n": retained,
                "display_share_pct": 100.0 * retained / available,
                "median_incidence_per_1k": float(
                    subset["incidence_per_1k"].median()
                ),
                "max_incidence_per_1k": float(
                    subset["incidence_per_1k"].max()
                ),
            })

        unusual = weekly.loc[weekly["modality_count"].ge(3)].copy()
        if not unusual.empty:
            unusual["_displayed"] = [
                (pd.Timestamp(week), tuple(signature)) in displayed_keys
                for week, signature in zip(
                    unusual["week_start"],
                    unusual["modality_signature"],
                )
            ]
            unusual["family"] = family_name
            example_frames.append(
                unusual.nlargest(5, "incidence_per_1k")[
                    [
                        "family", "week_start", "modality_count",
                        "signature_label", "event_count",
                        "incidence_per_1k", "_displayed",
                    ]
                ].rename(columns={"_displayed": "displayed_in_top_n"})
            )

    coverage = pd.DataFrame(coverage_rows)
    if not coverage.empty:
        for column in [
            "display_share_pct",
            "median_incidence_per_1k",
            "max_incidence_per_1k",
        ]:
            coverage[column] = pd.to_numeric(
                coverage[column], errors="coerce"
            ).round(3)

    examples = (
        pd.concat(example_frames, ignore_index=True)
        if example_frames
        else pd.DataFrame(columns=[
            "family", "week_start", "modality_count", "signature_label",
            "event_count", "incidence_per_1k", "displayed_in_top_n",
        ])
    )
    if not examples.empty:
        examples["incidence_per_1k"] = pd.to_numeric(
            examples["incidence_per_1k"], errors="coerce"
        ).round(3)

    return coverage, examples

def _intersection_size_control(
    *,
    key_prefix: str,
    default: tuple[int, ...] = (2, 3, 4),
) -> tuple[int, ...]:
    """Choose 2-, 3-, and/or 4-way shared-stress intersections."""
    selected = st.multiselect(
        "Show intersections with",
        options=[2, 3, 4],
        default=list(default),
        format_func=lambda value: f"{value}-way",
        key=f"{key_prefix}_intersection_sizes",
        help=(
            "2-way events join two modes. 3-way and 4-way events show rarer "
            "higher-order convergence."
        ),
    )
    return tuple(sorted(int(value) for value in selected))


def _render_storyline_hero(
    all_events: pd.DataFrame,
    eligible: pd.DataFrame,
) -> None:
    """Teach the storyline once with a frozen, reproducible configuration."""
    st.divider()
    st.markdown("## When did mobility modes share the same stress?")
    st.markdown(
        """
The UpSet views summarize **which mobility modes appeared together** in retained
stress anomalies. This storyline adds **when**.

Each mode stays on a fixed horizontal **home lane** until it participates in a
shared weekly stress pattern. Participating modes bend toward the same event
and meet at a **striped flag**; each stripe identifies one participating mode.
**Thicker paths mean the exact combination was more prevalent**, measured as
stress events per 1,000 eligible observations.

This frozen example includes **2-, 3-, and 4-way intersections**. Higher-order
intersections are rarer, so the display preserves limited representation of
the strongest 3-/4-way events before filling the remaining positions by
incidence. Their thickness still reflects their actual prevalence.
        """
    )

    family = "All"
    event_limit = 20
    intersection_sizes = (2, 3, 4)

    family_events, _ = _scout_family_events(all_events, family)
    weekly = _weekly_storyline_surface(family_events, eligible)

    figure = _build_stress_storyline_figure(
        weekly,
        family_label=family,
        max_events=event_limit,
        modality_counts=intersection_sizes,
    )
    st.plotly_chart(
        figure,
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
        key="raw14_stress_storyline_hero",
    )

    takeaway = _storyline_takeaway(
        weekly,
        max_events=event_limit,
        family_label=family,
        modality_counts=intersection_sizes,
    )
    if takeaway:
        render_chart_insight(takeaway)

    st.caption(
        "Frozen example · All retained stress · 20 displayed weekly events · "
        "2-, 3-, and 4-way intersections · thickness = incidence per 1,000 "
        "eligible observations."
    )




# =============================================================================
# Page
# =============================================================================
st.caption(
    PAGE_CAPTION
)

st.title(
    PAGE_TITLE
)

if PHASE_1_SCOUTING_MODE:
    st.write(
        "Temporary Phase 1 view: compare All, Congestion, Demand, and Both "
        "stress families to select the replacement hero and supporting tabs."
    )
    scouting_report = _build_phase_1_scouting_report()
    st.download_button(
        "Download scouting report",
        data=scouting_report,
        file_name="raw_14_phase_1_hero_rescouting_report.txt",
        mime="text/plain",
        width="stretch",
    )
    st.code(scouting_report, language="text", wrap_lines=False)
    st.stop()

st.write(
    "Knowing when or where stress appeared still leaves an important question: "
    "**which mobility modes were involved together?** This page treats each retained "
    "stress anomaly as a combination of Taxi, FHVHV, Subway, and Bus participation, "
    "then compares those combinations across policy periods, mobility environments, "
    "and time. Congestion and Demand are inclusive views; **Both** isolates events "
    "that carried both signals."
)

hero_events = _load_all_stress_events()
hero_eligible = _load_eligible_observation_context()
congestion_events, _ = _scout_family_events(hero_events, "Congestion")
demand_events, _ = _scout_family_events(hero_events, "Demand")
both_events, _ = _scout_family_events(hero_events, "Both")


def _hero_period_incidence(family_events: pd.DataFrame, period: str) -> float:
    numerator = family_events.loc[
        family_events["period_group"].eq(period), "comparison_event_id"
    ].nunique()
    # The eligible-context loader intentionally reads only the canonical
    # Taxi Zone × date × temporal-bucket grain. Each row is one eligible
    # observation; comparison_event_id is not part of this denominator table.
    denominator = int(hero_eligible["period_group"].eq(period).sum())
    return 1000.0 * numerator / denominator if denominator else np.nan


def _hero_signature_share(
    family_events: pd.DataFrame,
    period: str,
    signature: tuple[str, ...],
) -> float:
    return _group_signature_share(
        family_events,
        group_column="period_group",
        group_label=period,
        signature=signature,
    )


congestion_pre = _hero_period_incidence(congestion_events, "Pre-CP")
congestion_post = _hero_period_incidence(congestion_events, "Post-CP")
demand_pre = _hero_period_incidence(demand_events, "Pre-CP")
demand_post = _hero_period_incidence(demand_events, "Post-CP")
both_pre = _hero_period_incidence(both_events, "Pre-CP")
both_post = _hero_period_incidence(both_events, "Post-CP")
bus_pre_share = _hero_signature_share(congestion_events, "Pre-CP", ("Bus",))
bus_post_share = _hero_signature_share(congestion_events, "Post-CP", ("Bus",))


def _relative_change(pre_value: float, post_value: float) -> float:
    return 100.0 * (post_value / pre_value - 1.0) if pre_value else np.nan


st.markdown("### Which modes tended to share the same stress events?")
st.markdown(
    "The UpSet views below show **exact modality combinations**. Filled dots identify "
    "the participating modes; a vertical connector means those modes appeared in the "
    "same stress event. Bar height shows that combination's share within the displayed "
    "stress family, so the chart answers *what kinds of multimodal stress made up the set?*"
)

st.markdown("**Choose a comparison story**")
congestion_tab, compound_tab, demand_tab = st.tabs(
    [
        "Congestion retreat",
        "Compound stress by environment",
        "Demand surge",
    ]
)

with congestion_tab:
    st.subheader("How did congestion-involved stress change?")
    _render_custom_comparison(
        congestion_events,
        hero_eligible,
        group_column="period_group",
        group_order=POLICY_PERIOD_ORDER,
        stress_family="Congestion",
        measure="Composition share",
        chart_key_prefix="raw14_hero_congestion",
    )
    render_chart_insight(
        "Congestion-involved anomaly incidence fell from "
        f"**{congestion_pre:.1f} to {congestion_post:.1f} per 1,000** "
        f"({_relative_change(congestion_pre, congestion_post):+.1f}%). At the "
        "same time, Bus-only anomalies grew from "
        f"**{bus_pre_share * 100:.1f}% to {bus_post_share * 100:.1f}%** of the "
        "congestion-involved family, so the smaller Post-CP set is more "
        "concentrated in Bus stress."
    )

with compound_tab:
    st.subheader("How did compound stress differ by mobility environment?")
    _render_custom_comparison(
        both_events,
        hero_eligible,
        group_column="environment_group",
        group_order=MOBILITY_ENVIRONMENT_ORDER,
        stress_family="All",
        measure="Composition share",
        chart_key_prefix="raw14_hero_compound",
        group_display_labels=MOBILITY_ENVIRONMENT_SHORT_LABELS,
    )
    environment_separation = _group_separation_summary(
        both_events,
        group_column="environment_group",
        group_order=MOBILITY_ENVIRONMENT_ORDER,
    )
    strongest_pair = str(environment_separation["most_separated_pair"])
    max_tvd = float(environment_separation["max_tvd"])
    render_chart_insight(
        "Compound anomalies increased from "
        f"**{both_pre:.1f} to {both_post:.1f} per 1,000** "
        f"({_relative_change(both_pre, both_post):+.1f}%). But the modes involved "
        f"differ substantially across mobility environments: **{strongest_pair}** "
        "shows the sharpest contrast in which modes participate together. "
        "Simultaneous demand and congestion stress therefore takes different forms "
        "across NYC rather than following one citywide pattern."
    )

with demand_tab:
    st.subheader("How did demand-involved stress change?")
    _render_custom_comparison(
        demand_events,
        hero_eligible,
        group_column="period_group",
        group_order=POLICY_PERIOD_ORDER,
        stress_family="Demand",
        measure="Composition share",
        chart_key_prefix="raw14_hero_demand",
    )
    pre_demand_leader = _signature_distribution(
        demand_events[demand_events["period_group"].eq("Pre-CP")]
    )
    post_demand_leader = _signature_distribution(
        demand_events[demand_events["period_group"].eq("Post-CP")]
    )
    pre_label = str(pre_demand_leader.idxmax()) if not pre_demand_leader.empty else "n/a"
    post_label = str(post_demand_leader.idxmax()) if not post_demand_leader.empty else "n/a"
    render_chart_insight(
        "Demand-involved anomaly incidence rose from "
        f"**{demand_pre:.1f} to {demand_post:.1f} per 1,000** "
        f"({_relative_change(demand_pre, demand_post):+.1f}%). The most common "
        f"mode pattern shifted from **{pre_label}** Pre-CP to "
        f"**{post_label}** Post-CP."
    )

card1, card2, card3, card4 = st.columns(4)
with card1:
    st.metric(
        "Congestion-involved incidence",
        f"{congestion_post:.1f} / 1K",
        f"{_relative_change(congestion_pre, congestion_post):+.1f}% vs Pre-CP",
        delta_color="off",
        help=(
            "Congestion-involved anomalies per 1,000 eligible observations "
            "Post-CP. Compound events are included when they carry congestion stress."
        ),
    )
with card2:
    st.metric(
        "Bus-only share",
        f"{bus_post_share * 100:.1f}%",
        f"{(bus_post_share - bus_pre_share) * 100:+.1f} pp vs Pre-CP",
        delta_color="off",
    )
with card3:
    st.metric(
        "Demand-involved incidence",
        f"{demand_post:.1f} / 1K",
        f"{_relative_change(demand_pre, demand_post):+.1f}% vs Pre-CP",
        delta_color="off",
        help=(
            "Demand-involved anomalies per 1,000 eligible observations Post-CP. "
            "Compound events are included when they carry demand stress."
        ),
    )
with card4:
    st.metric(
        "Compound incidence",
        f"{both_post:.1f} / 1K",
        f"{_relative_change(both_pre, both_post):+.1f}% vs Pre-CP",
        delta_color="off",
        help=(
            "Anomalies carrying both congestion and demand stress per 1,000 "
            "eligible observations Post-CP."
        ),
    )



_render_storyline_hero(
    hero_events,
    hero_eligible,
)


with exploration_section(
    key="raw14_exploration_area",
    title="Build your own stress-pattern comparison",
    description=(
        "Choose one comparison dimension, then narrow the stress family, "
        "time scope, time-of-week bucket, and measure. Policy geography and "
        "mobility environment remain alternative—not nested—ways to segment space."
    ),
):
    all_events = (
        _load_all_stress_events()
    )

    study_start = pd.Timestamp(
        all_events[
            "date"
        ].min()
    ).date()

    study_end = pd.Timestamp(
        all_events[
            "date"
        ].max()
    ).date()

    compare_by = st.selectbox(
        "Compare by",
        options=[
            "Policy period",
            "Policy geography",
            "Mobility environment",
        ],
        index=0,
        key="raw14_compare_by",
    )

    control_col1, control_col2, control_col3 = (
        st.columns(3)
    )

    with control_col1:
        stress_family = st.selectbox(
            "Stress family",
            options=list(
                STRESS_FAMILY_OPTIONS
            ),
            index=0,
            key="raw14_stress_family",
            help=(
                "Demand uses trips, ridership, and transfers. Congestion uses "
                "average speed and duration; Subway has no congestion metric. "
                "Congestion and Demand are inclusive, Both isolates events carrying "
                "both signals, and All uses every recognized mobility driver."
            ),
        )

    with control_col2:
        measure = st.selectbox(
            "Measure",
            options=list(
                MEASURE_OPTIONS
            ),
            index=0,
            key="raw14_measure",
            help=(
                "Composition share asks what kinds of stress anomalies make up each "
                "panel. Events per 1,000 eligible observations asks how often each "
                "stress pattern occurred among all eligible Taxi Zone × date × "
                "daypart observations in that panel."
            ),
        )

    bucket_options = [
        "All temporal buckets",
        *_temporal_bucket_options(
            all_events
        ),
    ]

    with control_col3:
        temporal_bucket = st.selectbox(
            "Temporal bucket",
            options=bucket_options,
            index=0,
            format_func=(
                _temporal_bucket_display
            ),
            key="raw14_temporal_bucket",
        )


    group_column: str
    group_order: Sequence[str]
    group_display_labels: dict[
        str,
        str,
    ] | None = None
    time_scope = "Full study"
    custom_date_range: object | None = None

    if compare_by == "Policy period":
        group_column = (
            "period_group"
        )
        group_order = (
            POLICY_PERIOD_ORDER
        )

        st.caption(
            "Policy-period comparison uses the full available Pre-CP and Post-CP "
            "windows citywide."
        )

    elif compare_by == "Policy geography":
        group_column = (
            "geography_group"
        )
        group_order = (
            POLICY_GEOGRAPHY_ORDER
        )

        time_scope = st.selectbox(
            "Time scope",
            options=list(
                TIME_SCOPE_OPTIONS
            ),
            index=0,
            key="raw14_geo_time_scope",
        )

        if time_scope == "Custom range":
            custom_date_range = st.date_input(
                "Custom date range",
                value=(
                    study_start,
                    study_end,
                ),
                min_value=(
                    study_start
                ),
                max_value=(
                    study_end
                ),
                key="raw14_geo_custom_dates",
            )

    else:
        group_column = (
            "environment_group"
        )
        group_order = (
            MOBILITY_ENVIRONMENT_ORDER
        )
        group_display_labels = (
            MOBILITY_ENVIRONMENT_SHORT_LABELS
        )

        time_scope = st.selectbox(
            "Time scope",
            options=list(
                TIME_SCOPE_OPTIONS
            ),
            index=0,
            key="raw14_env_time_scope",
        )

        if time_scope == "Custom range":
            custom_date_range = st.date_input(
                "Custom date range",
                value=(
                    study_start,
                    study_end,
                ),
                min_value=(
                    study_start
                ),
                max_value=(
                    study_end
                ),
                key="raw14_env_custom_dates",
            )


    start_date, end_date = (
        _resolve_date_window(
            all_events,
            time_scope,
            custom_date_range,
        )
    )

    scoped_events = (
        _stress_family_filter(
            all_events,
            stress_family,
        )
    )

    scoped_events = (
        _filter_date_and_bucket(
            scoped_events,
            start_date=start_date,
            end_date=end_date,
            temporal_bucket=(
                temporal_bucket
            ),
        )
    )

    # The independent modality drilldown always starts from the complete event
    # universe produced by the page-level family, date, and temporal controls.
    # Comparison-panel exclusions are applied only to the comparison above it.
    drilldown_base_events = scoped_events.copy()

    # Remove Unknown only when policy geography itself defines the panels.
    if (
        group_column
        == "geography_group"
    ):
        scoped_events = scoped_events[
            scoped_events[
                "geography_group"
            ].isin(
                POLICY_GEOGRAPHY_ORDER
            )
        ].copy()


    denominator: pd.DataFrame | None = None
    drilldown_base_denominator: pd.DataFrame | None = None

    if (
        measure
        == "Stress anomalies per 1,000 eligible observations"
    ):
        denominator = (
            _load_eligible_observation_context()
        )

        denominator = (
            _filter_date_and_bucket(
                denominator,
                start_date=(
                    start_date
                ),
                end_date=(
                    end_date
                ),
                temporal_bucket=(
                    temporal_bucket
                ),
            )
        )

        drilldown_base_denominator = denominator.copy()

        if (
            group_column
            == "geography_group"
        ):
            denominator = denominator[
                denominator[
                    "geography_group"
                ].isin(
                    POLICY_GEOGRAPHY_ORDER
                )
            ].copy()


    # The storyline always needs the eligible observation universe because
    # line thickness is incidence per 1,000. Reuse the explorer's existing
    # date and temporal-bucket filters even when the UpSet itself is showing
    # composition share rather than incidence.
    storyline_eligible = _load_eligible_observation_context()

    storyline_eligible = _filter_date_and_bucket(
        storyline_eligible,
        start_date=start_date,
        end_date=end_date,
        temporal_bucket=temporal_bucket,
    )

    # Match the same policy-geography panel contract used by the UpSet view.
    if group_column == "geography_group":
        storyline_eligible = storyline_eligible[
            storyline_eligible["geography_group"].isin(
                POLICY_GEOGRAPHY_ORDER
            )
        ].copy()


    date_caption = (
        f"{start_date:%b %d, %Y} – "
        f"{end_date:%b %d, %Y}"
    )

    st.markdown(
        f"### {stress_family} stress anomalies · {compare_by}"
    )

    st.caption(
        f"{date_caption}"
        f" · {_temporal_bucket_display(temporal_bucket)}"
        f" · {measure}"
    )

    if scoped_events.empty:
        st.warning(
            "No stress anomalies match this custom scope."
        )
    else:
        custom_summaries = (
            _render_custom_comparison(
                scoped_events,
                denominator,
                group_column=(
                    group_column
                ),
                group_order=(
                    group_order
                ),
                stress_family=(
                    stress_family
                ),
                measure=(
                    measure
                ),
                chart_key_prefix=(
                    "raw14_custom"
                ),
                group_display_labels=(
                    group_display_labels
                ),
            )
        )

        custom_insight = (
            _custom_summary_insight(
                scoped_events,
                denominator,
                group_column=(
                    group_column
                ),
                group_order=(
                    group_order
                ),
                group_display_labels=(
                    group_display_labels
                ),
                measure=(
                    measure
                ),
            )
        )

        if custom_insight:
            render_chart_insight(
                _custom_tell_me_insight(custom_insight)
            )

            st.markdown(
                "**Custom view summary**"
            )

            _render_custom_summary_cards(
                custom_insight
            )

        top_caption = (
            _top_pattern_caption(
                custom_summaries,
                measure=(
                    measure
                ),
            )
        )

        if top_caption:
            st.caption(
                "Panel leaders · "
                + top_caption
            )

        if (
            measure
            == "Stress anomalies per 1,000 eligible observations"
        ):
            st.caption(
                "Incidence is normalized by every eligible Taxi Zone × date × "
                "daypart observation in the current scope—not only observations "
                "flagged for stress. When the comparison uses policy geography, "
                "Taxi Zones 264 and 265 are omitted because no policy-area "
                "category is available for them."
            )



        # -----------------------------------------------------------------
        # Storyline · same filtered universe as the UpSet comparison above
        # -----------------------------------------------------------------
        st.divider()

        storyline_events, _ = _scout_family_events(
            scoped_events,
            stress_family,
        )

        _render_filtered_storyline(
            storyline_events,
            storyline_eligible,
            stress_family=stress_family,
        )

        # -----------------------------------------------------------------
        # Phase 5 · Independent modality drill-down
        # -----------------------------------------------------------------
        st.divider()

        st.header("Explore selected modalities")

        st.caption(
            "Start with the full stress-anomaly universe allowed by the controls "
            "above, or narrow it using one geographic segmentation. Then choose "
            "the modes you want to investigate. The combination chart preserves "
            "every mode attached to each matching event, including unselected modes."
        )

        with st.expander(
            "Which metrics define Demand and Congestion stress?",
            expanded=False,
        ):
            st.markdown(
                """
                | Mode | Demand evidence | Congestion evidence |
                |---|---|---|
                | Taxi | Trips | Average speed, average duration |
                | FHVHV | Trips | Average speed, average duration |
                | Subway | Ridership, transfers | None |
                | Bus | Trips | Average speed |
                """
            )
            st.caption(
                "A mode participates when at least one driver metric belonging to "
                "the selected stress family appears in the event's reconciled "
                "driver list."
            )

        if not drilldown_base_events.empty:
            geography_col1, geography_col2 = st.columns(2)

            with geography_col1:
                drilldown_geography_scheme = st.selectbox(
                    "View geography by",
                    options=[
                        "All geographies",
                        "Policy geography",
                        "Borough",
                        "Mobility environment",
                    ],
                    index=0,
                    key="raw14_detail_geography_scheme",
                    on_change=_clear_drilldown_geography_values,
                    help=(
                        "Choose one geographic segmentation system. Policy "
                        "geography, Borough, and Mobility cluster are alternatives; "
                        "they are never combined."
                    ),
                )

            drilldown_events = drilldown_base_events.copy()
            drilldown_denominator = (
                drilldown_base_denominator.copy()
                if drilldown_base_denominator is not None
                else None
            )
            geography_display = "All geographies"
            geography_column: str | None = None
            geography_value: str | None = None

            if drilldown_geography_scheme != "All geographies":
                if drilldown_geography_scheme == "Policy geography":
                    geography_column = "geography_group"
                    geography_options = [
                        value
                        for value in POLICY_GEOGRAPHY_ORDER
                        if drilldown_events["geography_group"].eq(value).any()
                    ]
                    geography_format = lambda value: value
                    geography_key = "raw14_detail_policy_geography"
                elif drilldown_geography_scheme == "Borough":
                    geography_column = "borough"
                    geography_options = sorted(
                        value
                        for value in drilldown_events["borough"].dropna().astype(str).unique()
                        if value.strip() and value != "Unknown"
                    )
                    geography_format = lambda value: value
                    geography_key = "raw14_detail_borough"
                else:
                    geography_column = "environment_group"
                    available_clusters = set(
                        drilldown_events["environment_group"].dropna().astype(str)
                    )
                    geography_options = [
                        value
                        for value in MOBILITY_ENVIRONMENT_ORDER
                        if value in available_clusters
                    ]
                    geography_format = lambda value: (
                        MOBILITY_ENVIRONMENT_SHORT_LABELS.get(value, value).replace("<br>", " ")
                    )
                    geography_key = "raw14_detail_cluster"

                with geography_col2:
                    geography_value = st.selectbox(
                        "Geography value",
                        options=geography_options,
                        format_func=geography_format,
                        key=geography_key,
                    )

                geography_display = geography_format(geography_value)
                drilldown_events = drilldown_events[
                    drilldown_events[geography_column].astype(str).eq(str(geography_value))
                ].copy()
                if drilldown_denominator is not None:
                    drilldown_denominator = drilldown_denominator[
                        drilldown_denominator[geography_column].astype(str).eq(str(geography_value))
                    ].copy()
            else:
                with geography_col2:
                    st.caption(
                        "No geographic filter applied. The drilldown uses the full "
                        "stress-anomaly universe allowed by the page-level controls."
                    )

            panel_modalities = _available_modalities(
                drilldown_events,
                stress_family=stress_family,
            )

            signature_counts = drilldown_events["modality_signature"].value_counts()
            top_signature = list(signature_counts.index[0]) if not signature_counts.empty else []

            # Start with one mode from the leading pattern so the initial view is
            # immediately readable and does not imply a broad multimode requirement.
            default_modalities = [
                modality
                for modality in top_signature[:1]
                if modality in panel_modalities
            ]

            if (
                not default_modalities
                and panel_modalities
            ):
                default_modalities = [
                    panel_modalities[0]
                ]

            mode_col1, mode_col2 = st.columns(2)

            with mode_col1:
                selected_modalities = st.multiselect(
                    "Modes to include",
                    options=panel_modalities,
                    default=(
                        default_modalities
                    ),
                    key="raw14_detail_modalities",
                    help=(
                        "With the default match rule, an event is included when "
                        "at least one selected mode participates. Change the "
                        "match rule to require every selected mode."
                    ),
                )

            with mode_col2:
                match_rule = st.selectbox(
                    "Mode matching rule",
                    options=MATCH_RULE_OPTIONS,
                    index=0,
                    key="raw14_detail_match_rule",
                    help=(
                        "OR includes an event when at least one selected mode "
                        "appears. AND requires all selected modes; modes not "
                        "selected may still appear."
                    ),
                )

            if not selected_modalities:
                st.info(
                    "Select at least one modality to inspect."
                )
            else:
                selected_pattern_events = (
                    _filter_selected_modalities(
                        drilldown_events,
                        selected_modalities=(
                            selected_modalities
                        ),
                        match_rule=match_rule,
                    )
                )

                selected_event_count = int(
                    len(
                        selected_pattern_events
                    )
                )

                panel_event_count = int(len(drilldown_events))

                panel_share = (
                    selected_event_count
                    / panel_event_count
                    if panel_event_count
                    else np.nan
                )

                unique_zone_count = int(
                    selected_pattern_events[
                        "taxi_zone_id"
                    ].nunique()
                )

                median_strength = (
                    selected_pattern_events[
                        "event_median_directional_strength"
                    ]
                    .median()
                    if (
                        "event_median_directional_strength"
                        in selected_pattern_events.columns
                    )
                    else np.nan
                )

                driver_breakdown = (
                    _metric_driver_breakdown(
                        selected_pattern_events,
                        stress_family=(
                            stress_family
                        ),
                        selected_modalities=(
                            selected_modalities
                        ),
                    )
                )

                scoped_event_count = int(len(drilldown_base_events))
                is_any_rule = match_rule == MATCH_RULE_ANY
                modality_text = _natural_join(
                    selected_modalities,
                    conjunction=("or" if is_any_rule else "and"),
                )
                funnel_parts = [f"Current scope: **{scoped_event_count:,}**"]
                if drilldown_geography_scheme != "All geographies":
                    funnel_parts.append(
                        f"{geography_display}: **{panel_event_count:,}**"
                    )
                funnel_parts.append(f"Mode filter: **{selected_event_count:,}**")
                st.caption(" → ".join(funnel_parts))

                if len(selected_modalities) == 1:
                    rule_explanation = (
                        f"A stress anomaly is included when **{selected_modalities[0]}** "
                        "appears in its driver evidence. With one selected mode, "
                        "the OR and AND rules return the same events."
                    )
                elif is_any_rule:
                    rule_explanation = (
                        "A stress anomaly is included when **"
                        + modality_text.replace(" or ", " OR ")
                        + "** appears in its driver evidence."
                    )
                else:
                    rule_explanation = (
                        "A stress anomaly is included only when **"
                        + modality_text.replace(" and ", " AND ")
                        + "** all appear in its driver evidence. Other mobility "
                        "modes may also appear."
                    )

                st.info(
                    f"**{selected_event_count:,} stress anomalies included.** "
                    + rule_explanation
                )

                metric_col1, metric_col2, metric_col3, metric_col4 = (
                    st.columns(4)
                )

                metric_col1.metric(
                    "Included stress anomalies",
                    f"{selected_event_count:,}",
                )

                metric_col2.metric(
                    "Share of geography scope",
                    (
                        f"{panel_share * 100:.1f}%"
                        if not pd.isna(
                            panel_share
                        )
                        else "n/a"
                    ),
                )

                metric_col3.metric(
                    "Taxi Zones involved",
                    f"{unique_zone_count:,}",
                )

                metric_col4.metric(
                    "Median directional strength",
                    _format_strength(
                        median_strength
                    ),
                )

                if (
                    drilldown_denominator is not None
                    and measure
                    == "Stress anomalies per 1,000 eligible observations"
                ):
                    panel_eligible_count = int(len(drilldown_denominator))

                    selected_incidence = (
                        selected_event_count
                        / panel_eligible_count
                        * 1000
                        if panel_eligible_count
                        else np.nan
                    )

                    if not pd.isna(
                        selected_incidence
                    ):
                        st.caption(
                            "Selected-modality stress-anomaly incidence: "
                            f"**{selected_incidence:.2f} events per 1,000 "
                            "eligible observations**."
                        )

                bucket_breakdown = (
                    _temporal_bucket_breakdown(
                        selected_pattern_events
                    )
                )

                zone_breakdown = (
                    _zone_breakdown(
                        selected_pattern_events
                    )
                )

                exact_signature_count = int(
                    selected_pattern_events["modality_signature"].nunique()
                )
                selected_pattern_events = selected_pattern_events.copy()
                selected_pattern_events["_drilldown_group"] = "Selection"
                selected_combination_summary = _custom_panel_summary(
                    selected_pattern_events,
                    None,
                    group_column="_drilldown_group",
                    group_label="Selection",
                    measure="Composition share",
                    stress_family=stress_family,
                )
                combination_chart_is_informative = (
                    exact_signature_count > 1
                    and not selected_combination_summary.empty
                )

                driver_chart_is_informative = (
                    not driver_breakdown.empty
                    and not np.isclose(
                        driver_breakdown["event_share"].astype(float),
                        1.0,
                    ).all()
                )

                bucket_chart_is_informative = (
                    temporal_bucket == "All temporal buckets"
                    and selected_pattern_events["temporal_bucket"].nunique() > 1
                )

                drill_col1, drill_col2, drill_col3 = (
                    st.columns(
                        3,
                        gap="medium",
                    )
                )

                with drill_col1:
                    if combination_chart_is_informative:
                        st.markdown("**How the selected modes combine**")
                        combination_y_max = _custom_y_max(
                            {"Selection": selected_combination_summary},
                            "Composition share",
                        )
                        st.plotly_chart(
                            _build_custom_upset_panel(
                                selected_combination_summary,
                                panel_label="Matching events",
                                panel_color=BRAND_COLORS["dark_teal"],
                                y_max=combination_y_max,
                                show_modality_labels=True,
                                modalities=_modalities_for_family(stress_family),
                                measure="Composition share",
                                stress_family=stress_family,
                            ),
                            width="stretch",
                            config={"displayModeBar": False},
                            key="raw14_detail_combinations",
                        )
                        combination_leader = selected_combination_summary.iloc[0]
                        render_chart_insight(
                            "The leading exact combination is "
                            f"**{combination_leader['signature_label']}**, accounting "
                            f"for **{float(combination_leader['event_share']) * 100:.1f}%** "
                            "of included stress anomalies."
                        )
                        if exact_signature_count > len(selected_combination_summary):
                            st.caption(
                                f"Showing the six leading combinations out of "
                                f"{exact_signature_count:,} represented in this match."
                            )
                    else:
                        st.markdown("**Driver evidence within the match**")

                        if driver_breakdown.empty:
                            st.caption(
                                "No defining-driver detail is available for this "
                                "selection."
                            )
                        elif not driver_chart_is_informative:
                            st.caption(
                                "Not charted: this match leaves one exact modality "
                                "combination, and each selected mode maps directly "
                                "to the displayed driver metric. Every bar would be "
                                "100% by construction."
                            )
                        else:
                            st.plotly_chart(
                                _horizontal_share_chart(
                                    driver_breakdown,
                                    category_column="metric_label",
                                    value_column="event_share",
                                    hover_count_column="event_count",
                                    value_title="Share of selection",
                                    key_color=BRAND_COLORS["dark_teal"],
                                ),
                                width="stretch",
                                config={"displayModeBar": False},
                                key="raw14_detail_metrics",
                            )
                            metric_leader = driver_breakdown.iloc[0]
                            render_chart_insight(
                                f"**{metric_leader['metric_label']}** is the most common "
                                "defining driver in this selection, appearing in "
                                f"**{float(metric_leader['event_share']) * 100:.1f}%** "
                                "of included stress anomalies."
                            )

                with drill_col2:
                    st.markdown(
                        "**Top temporal buckets**"
                    )

                    if bucket_breakdown.empty:
                        st.caption(
                            "No temporal-bucket detail is available for this "
                            "selection."
                        )
                    elif temporal_bucket != "All temporal buckets":
                        st.caption(
                            "Not charted: the current page scope already fixes the "
                            f"temporal bucket at **{_temporal_bucket_display(temporal_bucket)}**."
                        )
                    elif not bucket_chart_is_informative:
                        only_bucket = _temporal_bucket_display(
                            str(bucket_breakdown.iloc[0]["temporal_bucket"])
                        )
                        st.caption(
                            "Not charted: all matching anomalies happen to fall in "
                            f"**{only_bucket}** within the current scope."
                        )
                    else:
                        st.plotly_chart(
                            _horizontal_share_chart(
                                bucket_breakdown,
                                category_column=(
                                    "bucket_label"
                                ),
                                value_column=(
                                    "event_share"
                                ),
                                hover_count_column=(
                                    "event_count"
                                ),
                                value_title=(
                                    "Share of selection"
                                ),
                                key_color=(
                                    BRAND_COLORS[
                                        "terracotta"
                                    ]
                                ),
                            ),
                            width="stretch",
                            config={
                                "displayModeBar": False,
                            },
                            key="raw14_detail_buckets",
                        )
                        bucket_leader = bucket_breakdown.iloc[0]
                        render_chart_insight(
                            f"**{bucket_leader['bucket_label']}** contains the "
                            "largest share of this selection at "
                            f"**{float(bucket_leader['event_share']) * 100:.1f}%**."
                        )

                with drill_col3:
                    st.markdown(
                        "**Top Taxi Zones**"
                    )

                    if zone_breakdown.empty:
                        st.caption(
                            "No Taxi Zone detail is available for this selection."
                        )
                    else:
                        st.plotly_chart(
                            _horizontal_share_chart(
                                zone_breakdown,
                                category_column=(
                                    "zone_label"
                                ),
                                value_column=(
                                    "event_share"
                                ),
                                hover_count_column=(
                                    "event_count"
                                ),
                                value_title=(
                                    "Share of selection"
                                ),
                                key_color=(
                                    BRAND_COLORS[
                                        "seafoam"
                                    ]
                                ),
                            ),
                            width="stretch",
                            config={
                                "displayModeBar": False,
                            },
                            key="raw14_detail_zones",
                        )
                        zone_leader = zone_breakdown.iloc[0]
                        render_chart_insight(
                            f"**{zone_leader['zone_label']}** is the leading Taxi "
                            "Zone in this selection, accounting for "
                            f"**{float(zone_leader['event_share']) * 100:.1f}%** "
                            "of selected anomalies."
                        )

                if driver_chart_is_informative and not combination_chart_is_informative:
                    st.caption(
                        "Defining-driver shares are restricted to the selected stress "
                        "family and selected modalities. They can overlap because one "
                        "event can contain more than one defining metric—for example, "
                        "Subway ridership and transfers or Taxi speed and duration."
                    )

# =============================================================================
# Closing synthesis
# =============================================================================
st.divider()

st.markdown("### What this page establishes")
st.markdown(
    "Mobility stress was not a single systemwide condition. The modes involved in "
    "stress events changed with the stress family, policy period, mobility environment, "
    "and week. Looking at exact modality combinations shows whether unusual conditions "
    "were isolated to one mode or shared across several, while the storyline adds the "
    "missing temporal dimension: when those multimodal combinations actually occurred."
)

with st.expander("How this page works", expanded=False):
    st.markdown(
        """
        **1. Define one stress event at a consistent grain.** A mobility stress anomaly
        is one Taxi Zone × date × daypart observation retained after support from the
        project's anomaly-detection methods.

        **2. Separate demand and congestion signals.** Demand stress reflects positive
        shocks in trips, ridership, or transfers. Congestion stress reflects
        congestion-oriented movement in speed or duration. The **Congestion** and
        **Demand** views are inclusive, so a compound event can appear in both;
        **Both** isolates events carrying both signals.

        **3. Collapse metric evidence to mobility modes.** Defining metrics are grouped
        into **Taxi, FHVHV, Subway, and Bus** so the UpSet views compare exact modality
        combinations rather than individual measurements.

        **4. Read UpSet bars as composition, not total system volume.** Filled dots show
        the modes in an exact intersection, and the top bar shows that combination's
        share within the displayed family. Each panel ranks its own six most common
        combinations, so the displayed bars need not sum to 100%.

        **5. Use incidence when the question is how common stress was.** Incidence is
        expressed per **1,000 eligible observations**, which keeps scopes with different
        amounts of usable data comparable.

        **6. Add time with the shared-stress storyline.** Each mode has a home lane.
        Lines converge when multiple modes participate in the same weekly stress
        combination, and line thickness represents incidence per 1,000 eligible
        observations. The displayed set preserves strong higher-order combinations
        before filling remaining positions by incidence so uncommon multimodal structure
        is not automatically hidden.
        """
    )

st.caption(
    "Evidence scope: retained mobility stress anomalies and their defining modality "
    "combinations. Differences across the January 2025 policy boundary describe how "
    "observed stress patterns changed; they do not establish that congestion pricing "
    "caused those changes."
)
