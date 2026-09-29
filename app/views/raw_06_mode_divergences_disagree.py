from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data_access.loaders import CORE_METRICS
from app.data_access.mobility_environments import (
    attach_mobility_regime_cluster_context,
    format_mobility_regime_cluster_label,
    get_mobility_regime_cluster_options,
)
from app.data_access.spatial_aggregations import (
    ALL_TEMPORAL_BUCKETS_LABEL,
    get_zone_pre_post_metric_summary,
)
from app.data_access.spatial_visuals import (
    add_reliability_flags,
)
from app.utils.project_branding import (
    BRAND_COLORS,
    apply_branding,
    exploration_section,
    inject_app_css,
    render_chart_insight,
)


inject_app_css()

TAXI_METRIC = "taxi_trip_count"
FHVHV_METRIC = "fhvhv_trip_count"

HERO_TOP_N = 15

METRIC_DISPLAY_LABELS = {
    "taxi_trip_count": "Taxi Trips",
    "taxi_avg_trip_speed": "Taxi Average Speed",
    "fhvhv_trip_count": "FHVHV Trips",
    "fhvhv_avg_trip_speed": "FHVHV Average Speed",
    "subway_ridership": "Subway Ridership",
    "avg_bus_speed": "Bus Average Speed",
}

TEMPORAL_BUCKET_OPTIONS = [
    ALL_TEMPORAL_BUCKETS_LABEL,
    "weekday_overnight",
    "weekday_am_peak",
    "weekday_midday",
    "weekday_pm_peak",
    "weekday_evening",
    "weekend_overnight",
    "weekend_am_peak",
    "weekend_midday",
    "weekend_pm_peak",
    "weekend_evening",
]

TEMPORAL_BUCKET_LABELS = {
    ALL_TEMPORAL_BUCKETS_LABEL: "Overall",
    "weekday_overnight": "Weekday · Overnight",
    "weekday_am_peak": "Weekday · AM peak",
    "weekday_midday": "Weekday · Midday",
    "weekday_pm_peak": "Weekday · PM peak",
    "weekday_evening": "Weekday · Evening",
    "weekend_overnight": "Weekend · Overnight",
    "weekend_am_peak": "Weekend · AM peak",
    "weekend_midday": "Weekend · Midday",
    "weekend_pm_peak": "Weekend · PM peak",
    "weekend_evening": "Weekend · Evening",
}

GEOGRAPHY_FILTER_OPTIONS = [
    "All Taxi Zones",
    "Borough",
    "Geo-policy group",
    "Mobility regime cluster",
]

DIRECTION_OPTIONS = [
    "Both opposite-direction patterns",
    "Metric A up · Metric B down",
    "Metric A down · Metric B up",
]

TOP_N_OPTIONS = [
    10,
    15,
    20,
    25,
]

BOROUGH_ORDER = [
    "Manhattan",
    "Brooklyn",
    "Queens",
    "Bronx",
    "Staten Island",
    "Unknown",
]

GEO_POLICY_ORDER = [
    "CBD",
    "Gateway",
    "Adjacent",
    "Non-CBD",
    "Unknown",
]

MOBILITY_REGIME_CLUSTER_ORDER = get_mobility_regime_cluster_options()

METRIC_A_COLOR = BRAND_COLORS["dark_teal"]
METRIC_B_COLOR = BRAND_COLORS["terracotta"]
CONNECTOR_COLOR = "rgba(80, 80, 80, 0.34)"
ZERO_LINE_COLOR = "rgba(40, 40, 40, 0.58)"
GRID_COLOR = "rgba(0, 109, 119, 0.10)"


SAVED_VIEWS = {
    "Taxi vs FHVHV demand": {
        "description": (
            "The default divergence story: Taxi Trips versus FHVHV Trips "
            "across all eligible Taxi Zones."
        ),
        "metric_a": TAXI_METRIC,
        "metric_b": FHVHV_METRIC,
        "temporal_bucket": ALL_TEMPORAL_BUCKETS_LABEL,
        "top_n": 15,
        "direction_filter": "Both opposite-direction patterns",
        "geography_filter": "All Taxi Zones",
        "geography_value": None,
        "minimum_divergence": 0.0,
    },
    "Taxi vs subway demand": {
        "description": (
            "Compare where Taxi Trips and Subway Ridership moved in "
            "opposite directions."
        ),
        "metric_a": TAXI_METRIC,
        "metric_b": "subway_ridership",
        "temporal_bucket": ALL_TEMPORAL_BUCKETS_LABEL,
        "top_n": 15,
        "direction_filter": "Both opposite-direction patterns",
        "geography_filter": "All Taxi Zones",
        "geography_value": None,
        "minimum_divergence": 0.0,
    },
    "Surface speeds at the PM peak": {
        "description": (
            "Compare Taxi and FHVHV average-speed divergences during the "
            "weekday PM peak."
        ),
        "metric_a": "taxi_avg_trip_speed",
        "metric_b": "fhvhv_avg_trip_speed",
        "temporal_bucket": "weekday_pm_peak",
        "top_n": 15,
        "direction_filter": "Both opposite-direction patterns",
        "geography_filter": "All Taxi Zones",
        "geography_value": None,
        "minimum_divergence": 0.0,
    },
    "Taxi demand vs Taxi speed": {
        "description": (
            "Find places where Taxi demand and Taxi average speed moved "
            "in opposite directions during the weekday PM peak."
        ),
        "metric_a": TAXI_METRIC,
        "metric_b": "taxi_avg_trip_speed",
        "temporal_bucket": "weekday_pm_peak",
        "top_n": 15,
        "direction_filter": "Both opposite-direction patterns",
        "geography_filter": "All Taxi Zones",
        "geography_value": None,
        "minimum_divergence": 0.0,
    },
    "Manhattan demand divergences": {
        "description": (
            "Focus the default Taxi–FHVHV demand comparison on Manhattan."
        ),
        "metric_a": TAXI_METRIC,
        "metric_b": FHVHV_METRIC,
        "temporal_bucket": ALL_TEMPORAL_BUCKETS_LABEL,
        "top_n": 15,
        "direction_filter": "Both opposite-direction patterns",
        "geography_filter": "Borough",
        "geography_value": "Manhattan",
        "minimum_divergence": 0.0,
    },
}

SAVED_VIEW_OPTIONS = [
    *SAVED_VIEWS.keys(),
    "Custom",
]


# ---------------------------------------------------------------------
# Saved-view helpers
# ---------------------------------------------------------------------
def _apply_saved_view(
    saved_view_name: str,
) -> None:
    if saved_view_name == "Custom":
        return

    configuration = SAVED_VIEWS[
        saved_view_name
    ]

    st.session_state[
        "raw06_metric_a"
    ] = configuration["metric_a"]

    st.session_state[
        "raw06_metric_b"
    ] = configuration["metric_b"]

    st.session_state[
        "raw06_temporal_bucket"
    ] = configuration[
        "temporal_bucket"
    ]

    st.session_state[
        "raw06_top_n"
    ] = configuration["top_n"]

    st.session_state[
        "raw06_direction_filter"
    ] = configuration[
        "direction_filter"
    ]

    st.session_state[
        "raw06_geography_filter"
    ] = configuration[
        "geography_filter"
    ]

    st.session_state[
        "raw06_minimum_divergence"
    ] = configuration[
        "minimum_divergence"
    ]

    geography_value = configuration[
        "geography_value"
    ]

    if (
        configuration[
            "geography_filter"
        ]
        == "Borough"
    ):
        st.session_state[
            "raw06_borough_value"
        ] = geography_value

    elif (
        configuration[
            "geography_filter"
        ]
        == "Geo-policy group"
    ):
        st.session_state[
            "raw06_policy_value"
        ] = geography_value


def _mark_saved_view_custom() -> None:
    st.session_state[
        "raw06_saved_view"
    ] = "Custom"


def _direction_label(
    generic_value: str,
    *,
    metric_a_label: str,
    metric_b_label: str,
) -> str:
    labels = {
        "Both opposite-direction patterns": (
            "Both opposite-direction patterns"
        ),
        "Metric A up · Metric B down": (
            f"{metric_a_label} up · "
            f"{metric_b_label} down"
        ),
        "Metric A down · Metric B up": (
            f"{metric_a_label} down · "
            f"{metric_b_label} up"
        ),
    }

    return labels.get(
        generic_value,
        generic_value,
    )


# ---------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------
def _metric_label(
    metric: str,
) -> str:
    return METRIC_DISPLAY_LABELS.get(
        metric,
        metric.replace("_", " ").title(),
    )


def _format_geography_term(value: str) -> str:
    """Translate internal geography keys into reader-facing terminology."""
    mapping = {
        "Geo-policy group": "Policy geography",
        "Mobility regime cluster": "Mobility environment",
    }
    return mapping.get(value, value)


def _normalize_borough(
    value: object,
) -> str:
    if pd.isna(value):
        return "Unknown"

    text = str(value).strip()

    aliases = {
        "MN": "Manhattan",
        "BK": "Brooklyn",
        "QN": "Queens",
        "BX": "Bronx",
        "SI": "Staten Island",
    }

    return aliases.get(
        text,
        text or "Unknown",
    )


def _normalize_geo_policy(
    value: object,
) -> str:
    if pd.isna(value):
        return "Unknown"

    normalized = (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    labels = {
        "cbd": "CBD",
        "gateway": "Gateway",
        "gateway to cbd": "Gateway",
        "adjacent": "Adjacent",
        "adjacent to cbd": "Adjacent",
        "non cbd": "Non-CBD",
        "noncbd": "Non-CBD",
    }

    return labels.get(
        normalized,
        str(value).strip() or "Unknown",
    )


def _format_number(
    value: object,
) -> str:
    if pd.isna(value):
        return "Unavailable"

    numeric = float(value)

    if abs(numeric) >= 1_000_000:
        return f"{numeric / 1_000_000:,.2f}M"

    if abs(numeric) >= 1_000:
        return f"{numeric / 1_000:,.1f}K"

    return f"{numeric:,.1f}"


def _format_percent(
    value: object,
) -> str:
    if pd.isna(value):
        return "Unavailable"

    return f"{float(value):+,.2f}%"


def _apply_chart_branding(
    fig: go.Figure,
) -> go.Figure:
    fig.update_layout(
        title={
            "text": " ",
            "x": 0,
            "y": 1,
        }
    )

    fig = apply_branding(fig)

    fig.update_layout(
        title={
            "text": "",
            "x": 0,
            "y": 1,
            "pad": {
                "t": 0,
                "b": 0,
                "l": 0,
                "r": 0,
            },
        }
    )

    clean_annotations = []

    for annotation in fig.layout.annotations or []:
        text = str(
            getattr(annotation, "text", "")
        ).strip()

        if text.lower() in {
            "",
            "undefined",
            "none",
            "nan",
        }:
            continue

        clean_annotations.append(
            annotation
        )

    fig.update_layout(
        annotations=clean_annotations,
    )

    return fig


def _ordered_values(
    values: pd.Series,
    preferred_order: list[str],
) -> list[str]:
    represented = set(
        values.dropna()
        .astype(str)
        .tolist()
    )

    ordered = [
        value
        for value in preferred_order
        if value in represented
    ]

    return ordered + sorted(
        represented.difference(
            ordered
        )
    )


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_metric_pair_source(
    metric_a: str,
    metric_b: str,
    temporal_bucket: str,
) -> pd.DataFrame:
    source = get_zone_pre_post_metric_summary(
        metrics=[
            metric_a,
            metric_b,
        ],
        temporal_bucket=temporal_bucket,
    ).copy()

    return attach_mobility_regime_cluster_context(
        source,
        assignment_period="post_cp",
    )


def _prepare_metric_pair_divergence(
    source: pd.DataFrame,
    *,
    metric_a: str,
    metric_b: str,
) -> pd.DataFrame:
    required_columns = {
        "taxi_zone_id",
        "zone",
        "borough",
        "cbd_spatial_category",
        "mobility_regime_cluster_label",
        "mobility_regime_cluster_name",
        "metric",
        "pre_daily_average",
        "post_daily_average",
        "absolute_change",
        "percent_change",
        "has_both_periods",
    }

    missing = sorted(
        required_columns.difference(
            source.columns
        )
    )

    if missing:
        raise KeyError(
            "Missing required source columns: "
            + ", ".join(missing)
        )

    flagged = add_reliability_flags(
        source.copy()
    )

    reliability_columns = {
        "taxi_zone_id",
        "metric",
        "eligible_for_percent_change",
    }

    missing_reliability = sorted(
        reliability_columns.difference(
            flagged.columns
        )
    )

    if missing_reliability:
        raise KeyError(
            "Reliability output is missing required columns: "
            + ", ".join(
                missing_reliability
            )
        )

    selected = flagged[
        flagged["metric"].isin(
            [
                metric_a,
                metric_b,
            ]
        )
    ].copy()

    selected[
        "eligible_for_percent_change"
    ] = (
        selected[
            "eligible_for_percent_change"
        ]
        .fillna(False)
        .astype(bool)
    )

    metadata = (
        selected[
            [
                "taxi_zone_id",
                "zone",
                "borough",
                "cbd_spatial_category",
                "mobility_regime_cluster_label",
                "mobility_regime_cluster_name",
            ]
        ]
        .drop_duplicates(
            subset="taxi_zone_id"
        )
        .copy()
    )

    value_columns = [
        "pre_daily_average",
        "post_daily_average",
        "absolute_change",
        "percent_change",
        "has_both_periods",
        "eligible_for_percent_change",
    ]

    metric_frames: list[pd.DataFrame] = []

    for metric, prefix in [
        (metric_a, "a"),
        (metric_b, "b"),
    ]:
        metric_frame = (
            selected[
                selected["metric"].eq(
                    metric
                )
            ][
                [
                    "taxi_zone_id",
                    *value_columns,
                ]
            ]
            .copy()
            .rename(
                columns={
                    column: (
                        f"{prefix}_{column}"
                    )
                    for column in value_columns
                }
            )
        )

        metric_frames.append(
            metric_frame
        )

    if any(
        frame.empty
        for frame in metric_frames
    ):
        return pd.DataFrame()

    wide = metric_frames[0].merge(
        metric_frames[1],
        on="taxi_zone_id",
        how="inner",
        validate="one_to_one",
    )

    wide = wide.merge(
        metadata,
        on="taxi_zone_id",
        how="left",
        validate="one_to_one",
    )

    wide["borough"] = (
        wide["borough"]
        .map(_normalize_borough)
    )

    wide["geo_policy_group"] = (
        wide["cbd_spatial_category"]
        .map(_normalize_geo_policy)
    )

    wide["mobility_regime_cluster_label"] = pd.to_numeric(
        wide["mobility_regime_cluster_label"],
        errors="coerce",
    ).astype("Int64")

    complete_and_eligible = (
        wide["a_has_both_periods"]
        .fillna(False)
        .astype(bool)
        & wide["b_has_both_periods"]
        .fillna(False)
        .astype(bool)
        & wide[
            "a_eligible_for_percent_change"
        ]
        .fillna(False)
        .astype(bool)
        & wide[
            "b_eligible_for_percent_change"
        ]
        .fillna(False)
        .astype(bool)
        & wide[
            [
                "a_percent_change",
                "b_percent_change",
            ]
        ]
        .notna()
        .all(axis=1)
    )

    wide = wide[
        complete_and_eligible
    ].copy()

    a_sign = np.sign(
        wide["a_percent_change"]
    )

    b_sign = np.sign(
        wide["b_percent_change"]
    )

    wide["opposite_direction"] = (
        a_sign.ne(b_sign)
        & a_sign.ne(0)
        & b_sign.ne(0)
    )

    wide["divergence_gap"] = (
        wide["a_percent_change"]
        - wide["b_percent_change"]
    )

    wide["absolute_divergence"] = (
        wide["divergence_gap"].abs()
    )

    wide["direction_relationship"] = np.select(
        [
            wide["a_percent_change"].gt(0)
            & wide["b_percent_change"].lt(0),
            wide["a_percent_change"].lt(0)
            & wide["b_percent_change"].gt(0),
        ],
        [
            "Metric A up · Metric B down",
            "Metric A down · Metric B up",
        ],
        default="Not opposite",
    )

    return (
        wide[
            wide["opposite_direction"]
        ]
        .sort_values(
            [
                "absolute_divergence",
                "zone",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def _filter_divergence_data(
    data: pd.DataFrame,
    *,
    geography_filter: str,
    geography_value: str | None,
    direction_filter: str,
    minimum_divergence: float,
) -> pd.DataFrame:
    filtered = data.copy()

    if (
        geography_filter == "Borough"
        and geography_value is not None
    ):
        filtered = filtered[
            filtered["borough"].eq(
                geography_value
            )
        ].copy()

    elif (
        geography_filter
        == "Geo-policy group"
        and geography_value is not None
    ):
        filtered = filtered[
            filtered[
                "geo_policy_group"
            ].eq(
                geography_value
            )
        ].copy()

    elif (
        geography_filter == "Mobility regime cluster"
        and geography_value is not None
    ):
        filtered = filtered[
            filtered[
                "mobility_regime_cluster_label"
            ].astype("Int64").eq(int(geography_value))
        ].copy()

    if (
        direction_filter
        == "Metric A up · Metric B down"
    ):
        filtered = filtered[
            filtered[
                "direction_relationship"
            ].eq(
                "Metric A up · Metric B down"
            )
        ].copy()

    elif (
        direction_filter
        == "Metric A down · Metric B up"
    ):
        filtered = filtered[
            filtered[
                "direction_relationship"
            ].eq(
                "Metric A down · Metric B up"
            )
        ].copy()

    filtered = filtered[
        filtered[
            "absolute_divergence"
        ].ge(
            minimum_divergence
        )
    ].copy()

    return (
        filtered.sort_values(
            [
                "absolute_divergence",
                "zone",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def _axis_range(
    data: pd.DataFrame,
) -> list[float]:
    values = pd.concat(
        [
            data["a_percent_change"],
            data["b_percent_change"],
        ],
        ignore_index=True,
    )

    minimum = float(
        min(
            values.min(),
            0,
        )
    )

    maximum = float(
        max(
            values.max(),
            0,
        )
    )

    span = max(
        maximum - minimum,
        25,
    )

    padding = span * 0.12

    return [
        minimum - padding,
        maximum + padding,
    ]




# ---------------------------------------------------------------------
# Recurring inverse-pattern evidence
# ---------------------------------------------------------------------
RECURRENCE_BUCKETS = [
    bucket
    for bucket in TEMPORAL_BUCKET_OPTIONS
    if bucket != ALL_TEMPORAL_BUCKETS_LABEL
]


def _prepare_metric_pair_evidence(
    source: pd.DataFrame,
    *,
    metric_a: str,
    metric_b: str,
) -> pd.DataFrame:
    """Prepare all eligible zone pairs, retaining opposite-direction status."""
    required_columns = {
        "taxi_zone_id",
        "zone",
        "borough",
        "cbd_spatial_category",
        "mobility_regime_cluster_label",
        "mobility_regime_cluster_name",
        "metric",
        "pre_daily_average",
        "post_daily_average",
        "absolute_change",
        "percent_change",
        "has_both_periods",
    }

    missing = sorted(
        required_columns.difference(source.columns)
    )

    if missing:
        raise KeyError(
            "Missing required source columns: "
            + ", ".join(missing)
        )

    flagged = add_reliability_flags(
        source.copy()
    )

    selected = flagged[
        flagged["metric"].isin(
            [
                metric_a,
                metric_b,
            ]
        )
    ].copy()

    selected["eligible_for_percent_change"] = (
        selected["eligible_for_percent_change"]
        .fillna(False)
        .astype(bool)
    )

    metadata = (
        selected[
            [
                "taxi_zone_id",
                "zone",
                "borough",
                "cbd_spatial_category",
                "mobility_regime_cluster_label",
                "mobility_regime_cluster_name",
            ]
        ]
        .drop_duplicates(
            subset="taxi_zone_id"
        )
        .copy()
    )

    value_columns = [
        "pre_daily_average",
        "post_daily_average",
        "absolute_change",
        "percent_change",
        "has_both_periods",
        "eligible_for_percent_change",
    ]

    metric_frames: list[pd.DataFrame] = []

    for metric, prefix in [
        (metric_a, "a"),
        (metric_b, "b"),
    ]:
        metric_frame = (
            selected[
                selected["metric"].eq(metric)
            ][
                [
                    "taxi_zone_id",
                    *value_columns,
                ]
            ]
            .copy()
            .rename(
                columns={
                    column: f"{prefix}_{column}"
                    for column in value_columns
                }
            )
        )

        metric_frames.append(metric_frame)

    if any(
        frame.empty
        for frame in metric_frames
    ):
        return pd.DataFrame()

    wide = metric_frames[0].merge(
        metric_frames[1],
        on="taxi_zone_id",
        how="inner",
        validate="one_to_one",
    )

    wide = wide.merge(
        metadata,
        on="taxi_zone_id",
        how="left",
        validate="one_to_one",
    )

    wide["borough"] = (
        wide["borough"]
        .map(_normalize_borough)
    )

    wide["geo_policy_group"] = (
        wide["cbd_spatial_category"]
        .map(_normalize_geo_policy)
    )

    complete_and_eligible = (
        wide["a_has_both_periods"]
        .fillna(False)
        .astype(bool)
        & wide["b_has_both_periods"]
        .fillna(False)
        .astype(bool)
        & wide["a_eligible_for_percent_change"]
        .fillna(False)
        .astype(bool)
        & wide["b_eligible_for_percent_change"]
        .fillna(False)
        .astype(bool)
        & wide[
            [
                "a_percent_change",
                "b_percent_change",
            ]
        ]
        .notna()
        .all(axis=1)
    )

    wide = wide[
        complete_and_eligible
    ].copy()

    a_sign = np.sign(
        wide["a_percent_change"]
    )
    b_sign = np.sign(
        wide["b_percent_change"]
    )

    wide["opposite_direction"] = (
        a_sign.ne(b_sign)
        & a_sign.ne(0)
        & b_sign.ne(0)
    )

    wide["divergence_gap"] = (
        wide["a_percent_change"]
        - wide["b_percent_change"]
    )

    wide["absolute_divergence"] = (
        wide["divergence_gap"].abs()
    )

    wide["direction_relationship"] = np.select(
        [
            wide["a_percent_change"].gt(0)
            & wide["b_percent_change"].lt(0),
            wide["a_percent_change"].lt(0)
            & wide["b_percent_change"].gt(0),
        ],
        [
            "Metric A up · Metric B down",
            "Metric A down · Metric B up",
        ],
        default="Same direction",
    )

    return wide.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _build_recurrence_evidence(
    metric_a: str,
    metric_b: str,
    minimum_divergence: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build recurrence summaries using the same gap threshold as the explorer."""
    bucket_frames: list[pd.DataFrame] = []

    for temporal_bucket in RECURRENCE_BUCKETS:
        source = _load_metric_pair_source(
            metric_a,
            metric_b,
            temporal_bucket,
        )

        if source.empty:
            continue

        bucket_data = _prepare_metric_pair_evidence(
            source,
            metric_a=metric_a,
            metric_b=metric_b,
        )

        if bucket_data.empty:
            continue

        bucket_data["temporal_bucket"] = temporal_bucket
        bucket_data["temporal_bucket_label"] = (
            TEMPORAL_BUCKET_LABELS[
                temporal_bucket
            ]
        )
        bucket_data["day_type"] = np.where(
            bucket_data["temporal_bucket"]
            .str.startswith("weekday"),
            "Weekday",
            "Weekend",
        )

        bucket_frames.append(bucket_data)

    if not bucket_frames:
        return pd.DataFrame(), pd.DataFrame()

    bucket_evidence = pd.concat(
        bucket_frames,
        ignore_index=True,
    )

    rows: list[dict[str, object]] = []

    for _, zone_df in bucket_evidence.groupby(
        "taxi_zone_id",
        sort=False,
        observed=True,
    ):
        first = zone_df.iloc[0]
        # A bucket counts as recurring only when it is opposite-direction
        # AND clears the reader's shared minimum-divergence threshold. This
        # keeps the recurrence tab consistent with the largest-divergence tab.
        inverse = zone_df[
            zone_df["opposite_direction"]
            & zone_df["absolute_divergence"].ge(minimum_divergence)
        ].copy()

        eligible_bucket_count = int(
            zone_df["temporal_bucket"].nunique()
        )
        inverse_bucket_count = int(
            inverse["temporal_bucket"].nunique()
        )

        a_up_b_down_count = int(
            inverse["direction_relationship"]
            .eq("Metric A up · Metric B down")
            .sum()
        )
        a_down_b_up_count = int(
            inverse["direction_relationship"]
            .eq("Metric A down · Metric B up")
            .sum()
        )

        if a_up_b_down_count > a_down_b_up_count:
            dominant_direction = (
                "Metric A up · Metric B down"
            )
        elif a_down_b_up_count > a_up_b_down_count:
            dominant_direction = (
                "Metric A down · Metric B up"
            )
        elif inverse_bucket_count:
            dominant_direction = "Mixed inverse directions"
        else:
            dominant_direction = "No inverse pattern"

        weekday_inverse_count = int(
            inverse["day_type"]
            .eq("Weekday")
            .sum()
        )
        weekend_inverse_count = int(
            inverse["day_type"]
            .eq("Weekend")
            .sum()
        )

        if weekday_inverse_count and weekend_inverse_count:
            recurrence_scope = "Weekday and weekend"
        elif weekday_inverse_count:
            recurrence_scope = "Weekday only"
        elif weekend_inverse_count:
            recurrence_scope = "Weekend only"
        else:
            recurrence_scope = "No recurring inverse pattern"

        rows.append(
            {
                "taxi_zone_id": first["taxi_zone_id"],
                "zone": first["zone"],
                "borough": first["borough"],
                "geo_policy_group": first["geo_policy_group"],
                "mobility_regime_cluster_label": first[
                    "mobility_regime_cluster_label"
                ],
                "mobility_regime_cluster_name": first[
                    "mobility_regime_cluster_name"
                ],
                "eligible_bucket_count": eligible_bucket_count,
                "inverse_bucket_count": inverse_bucket_count,
                "inverse_bucket_share": (
                    inverse_bucket_count
                    / eligible_bucket_count
                    if eligible_bucket_count
                    else np.nan
                ),
                "weekday_inverse_count": weekday_inverse_count,
                "weekend_inverse_count": weekend_inverse_count,
                "dominant_direction": dominant_direction,
                "recurrence_scope": recurrence_scope,
                "median_inverse_divergence": (
                    float(
                        inverse["absolute_divergence"]
                        .median()
                    )
                    if not inverse.empty
                    else np.nan
                ),
                "maximum_inverse_divergence": (
                    float(
                        inverse["absolute_divergence"]
                        .max()
                    )
                    if not inverse.empty
                    else np.nan
                ),
                "mean_inverse_divergence": (
                    float(
                        inverse["absolute_divergence"]
                        .mean()
                    )
                    if not inverse.empty
                    else np.nan
                ),
            }
        )

    summary = pd.DataFrame(rows)

    summary["recurrence_score"] = (
        summary["inverse_bucket_share"].fillna(0)
        * summary["median_inverse_divergence"].fillna(0)
    )

    summary = summary.sort_values(
        [
            "inverse_bucket_count",
            "inverse_bucket_share",
            "median_inverse_divergence",
            "zone",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    return summary, bucket_evidence


def _filter_recurrence_summary(
    summary: pd.DataFrame,
    *,
    geography_filter: str,
    geography_value: str | None,
    minimum_recurring_buckets: int,
) -> pd.DataFrame:
    filtered = summary.copy()

    if (
        geography_filter == "Borough"
        and geography_value is not None
    ):
        filtered = filtered[
            filtered["borough"].eq(
                geography_value
            )
        ].copy()

    elif (
        geography_filter == "Geo-policy group"
        and geography_value is not None
    ):
        filtered = filtered[
            filtered["geo_policy_group"].eq(
                geography_value
            )
        ].copy()

    elif (
        geography_filter == "Mobility regime cluster"
        and geography_value is not None
    ):
        filtered = filtered[
            filtered[
                "mobility_regime_cluster_label"
            ].astype("Int64").eq(int(geography_value))
        ].copy()

    filtered = filtered[
        filtered["inverse_bucket_count"].ge(
            minimum_recurring_buckets
        )
    ].copy()

    return filtered.sort_values(
        [
            "inverse_bucket_count",
            "inverse_bucket_share",
            "median_inverse_divergence",
            "zone",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)


def build_recurrence_dot_plot(
    data: pd.DataFrame,
    *,
    metric_a_label: str,
    metric_b_label: str,
    height: int,
) -> go.Figure:
    """Rank zones by recurrence share while encoding divergence magnitude."""
    plot_data = (
        data.copy()
        .sort_values(
            [
                "inverse_bucket_share",
                "inverse_bucket_count",
                "median_inverse_divergence",
            ],
            ascending=True,
        )
        .reset_index(drop=True)
    )

    marker_size = (
        10
        + 18
        * (
            plot_data["median_inverse_divergence"]
            / max(
                float(
                    plot_data["median_inverse_divergence"]
                    .max()
                ),
                1.0,
            )
        )
    )

    customdata = np.column_stack(
        [
            plot_data["zone"],
            plot_data["borough"],
            plot_data["geo_policy_group"],
            plot_data["mobility_regime_cluster_label"].map(
                format_mobility_regime_cluster_label
            ),
            plot_data["inverse_bucket_count"],
            plot_data["eligible_bucket_count"],
            plot_data["median_inverse_divergence"].map(
                lambda value: f"{value:,.1f} pp"
            ),
            plot_data["maximum_inverse_divergence"].map(
                lambda value: f"{value:,.1f} pp"
            ),
            plot_data["recurrence_scope"],
            plot_data["dominant_direction"].map(
                lambda value: _direction_label(
                    value,
                    metric_a_label=metric_a_label,
                    metric_b_label=metric_b_label,
                )
            ),
        ]
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_data["inverse_bucket_share"] * 100,
            y=plot_data["zone"],
            mode="markers+text",
            marker={
                "size": marker_size,
                "color": plot_data["inverse_bucket_count"],
                "colorscale": [
                    [0.0, BRAND_COLORS["pale_peach"]],
                    [1.0, BRAND_COLORS["dark_teal"]],
                ],
                "cmin": 1,
                "cmax": len(RECURRENCE_BUCKETS),
                "line": {
                    "color": "white",
                    "width": 1,
                },
                "colorbar": {
                    "title": {
                        "text": "Inverse<br>buckets",
                    },
                    "tickmode": "linear",
                    "dtick": 1,
                },
            },
            text=plot_data["inverse_bucket_count"].map(
                lambda value: f"{value}/{len(RECURRENCE_BUCKETS)}"
            ),
            textposition="middle right",
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Borough: %{customdata[1]}<br>"
                "Policy geography: %{customdata[2]}<br>"
                "Mobility environment: %{customdata[3]}<br>"
                "Inverse buckets: %{customdata[4]} of %{customdata[5]}<br>"
                "Median inverse divergence: %{customdata[6]}<br>"
                "Maximum inverse divergence: %{customdata[7]}<br>"
                "Scope: %{customdata[8]}<br>"
                "Dominant pattern: %{customdata[9]}"
                "<extra></extra>"
            ),
            cliponaxis=False,
            showlegend=False,
        )
    )

    fig.update_xaxes(
        title_text=(
            "Share of eligible temporal buckets with opposite movement"
        ),
        ticksuffix="%",
        range=[0, 108],
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
    )

    fig.update_yaxes(
        title_text="",
        automargin=True,
    )

    fig = _apply_chart_branding(
        fig
    )

    fig.update_layout(
        height=height,
        margin={
            "l": 12,
            "r": 105,
            "t": 35,
            "b": 80,
        },
        hovermode="closest",
    )

    return fig


def _build_recurrence_summary_cards(
    data: pd.DataFrame,
) -> dict[str, object]:
    if data.empty:
        return {
            "zones": 0,
            "top_zone": None,
            "top_count": 0,
            "both_scope_count": 0,
            "median_divergence": np.nan,
        }

    top = data.iloc[0]

    return {
        "zones": int(len(data)),
        "top_zone": str(top["zone"]),
        "top_count": int(top["inverse_bucket_count"]),
        "both_scope_count": int(
            data["recurrence_scope"]
            .eq("Weekday and weekend")
            .sum()
        ),
        "median_divergence": float(
            data["median_inverse_divergence"]
            .median()
        ),
    }


def _render_recurrence_cards(
    summary: dict[str, object],
) -> None:
    card1, card2, card3, card4 = st.columns(4)

    card1.metric(
        "Recurring zones",
        f"{summary['zones']:,}",
    )
    card2.metric(
        "Most recurring zone",
        (
            f"{summary['top_count']} of "
            f"{len(RECURRENCE_BUCKETS)} buckets"
        ),
        summary["top_zone"],
    )
    card3.metric(
        "Weekday and weekend",
        f"{summary['both_scope_count']:,} zones",
    )
    card4.metric(
        "Median divergence",
        (
            f"{summary['median_divergence']:.1f} pp"
            if pd.notna(
                summary["median_divergence"]
            )
            else "Unavailable"
        ),
    )


def _build_recurrence_takeaway(
    data: pd.DataFrame,
    *,
    metric_a_label: str,
    metric_b_label: str,
) -> str:
    if data.empty:
        return (
            "No zones met the selected recurrence and divergence requirements."
        )

    top = data.iloc[0]

    both_scope_count = int(
        data["recurrence_scope"]
        .eq("Weekday and weekend")
        .sum()
    )

    dominant_label = _direction_label(
        str(top["dominant_direction"]),
        metric_a_label=metric_a_label,
        metric_b_label=metric_b_label,
    )

    if top["inverse_bucket_count"] >= 8:
        persistence_phrase = (
            "recurs across nearly the full time-of-week profile"
        )
    elif top["inverse_bucket_count"] >= 5:
        persistence_phrase = (
            "recurs across a majority of time-of-week buckets"
        )
    else:
        persistence_phrase = (
            "appears in a smaller subset of time-of-week buckets"
        )

    return (
        f"**{top['zone']}** shows the strongest recurrence: opposite movement "
        f"appears in **{int(top['inverse_bucket_count'])} of "
        f"{int(top['eligible_bucket_count'])} eligible buckets** and "
        f"{persistence_phrase}. Its dominant pattern is "
        f"**{dominant_label}**, with a median inverse divergence of "
        f"**{float(top['median_inverse_divergence']):.1f} percentage points**. "
        f"Across the filtered results, **{both_scope_count} zones** show inverse "
        "movement in both weekday and weekend buckets. Recurrence strengthens "
        "the descriptive evidence of a repeated divergence pattern, but it does "
        "not establish substitution or causality."
    )


def build_selected_zone_evidence_chart(
    zone_evidence: pd.DataFrame,
    *,
    metric_a_label: str,
    metric_b_label: str,
) -> go.Figure:
    """Show both metric changes across all eligible buckets for one zone."""
    plot_data = zone_evidence.copy()

    plot_data["bucket_order"] = (
        plot_data["temporal_bucket"]
        .map(
            {
                bucket: index
                for index, bucket in enumerate(
                    RECURRENCE_BUCKETS
                )
            }
        )
    )

    plot_data = plot_data.sort_values(
        "bucket_order"
    )

    plot_data["display_direction_relationship"] = (
        plot_data["direction_relationship"].map(
            lambda value: _direction_label(
                str(value),
                metric_a_label=metric_a_label,
                metric_b_label=metric_b_label,
            )
        )
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=plot_data["temporal_bucket_label"],
            y=plot_data["a_percent_change"],
            name=metric_a_label,
            marker_color=METRIC_A_COLOR,
            customdata=plot_data[
                [
                    "display_direction_relationship",
                    "absolute_divergence",
                ]
            ],
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"{metric_a_label}: %{{y:.3f}}%<br>"
                "Pattern: %{customdata[0]}<br>"
                "Divergence: %{customdata[1]:.3f} pp"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Bar(
            x=plot_data["temporal_bucket_label"],
            y=plot_data["b_percent_change"],
            name=metric_b_label,
            marker_color=METRIC_B_COLOR,
            customdata=plot_data[
                [
                    "display_direction_relationship",
                    "absolute_divergence",
                ]
            ],
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"{metric_b_label}: %{{y:.3f}}%<br>"
                "Pattern: %{customdata[0]}<br>"
                "Divergence: %{customdata[1]:.3f} pp"
                "<extra></extra>"
            ),
        )
    )

    fig.add_hline(
        y=0,
        line_color=ZERO_LINE_COLOR,
        line_width=1.5,
    )

    fig.update_xaxes(
        title_text="",
        tickangle=-35,
    )
    fig.update_yaxes(
        title_text="Pre- to post-CP percent change",
        ticksuffix="%",
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
    )

    fig = _apply_chart_branding(
        fig
    )

    fig.update_layout(
        barmode="group",
        height=560,
        margin={
            "l": 35,
            "r": 25,
            "t": 65,
            "b": 120,
        },
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.08,
            "yanchor": "bottom",
        },
    )

    return fig


def _build_recurrence_detail_table(
    data: pd.DataFrame,
    *,
    metric_a_label: str,
    metric_b_label: str,
) -> pd.DataFrame:
    display = data[
        [
            "zone",
            "borough",
            "geo_policy_group",
            "mobility_regime_cluster_label",
            "inverse_bucket_count",
            "eligible_bucket_count",
            "inverse_bucket_share",
            "weekday_inverse_count",
            "weekend_inverse_count",
            "dominant_direction",
            "recurrence_scope",
            "median_inverse_divergence",
            "maximum_inverse_divergence",
        ]
    ].copy()

    display["inverse_bucket_share"] = (
        display["inverse_bucket_share"] * 100
    )
    display["mobility_regime_cluster_label"] = (
        display["mobility_regime_cluster_label"]
        .map(format_mobility_regime_cluster_label)
    )
    display["dominant_direction"] = (
        display["dominant_direction"].map(
            lambda value: _direction_label(
                str(value),
                metric_a_label=metric_a_label,
                metric_b_label=metric_b_label,
            )
        )
    )

    return display.rename(
        columns={
            "zone": "Taxi Zone",
            "borough": "Borough",
            "geo_policy_group": "Policy geography",
            "mobility_regime_cluster_label": "Mobility environment",
            "inverse_bucket_count": "Inverse buckets",
            "eligible_bucket_count": "Eligible buckets",
            "inverse_bucket_share": "Inverse bucket share",
            "weekday_inverse_count": "Weekday inverse buckets",
            "weekend_inverse_count": "Weekend inverse buckets",
            "dominant_direction": "Dominant inverse pattern",
            "recurrence_scope": "Recurrence scope",
            "median_inverse_divergence": "Median divergence",
            "maximum_inverse_divergence": "Maximum divergence",
        }
    )


# ---------------------------------------------------------------------
# Dumbbell chart
# ---------------------------------------------------------------------
def build_dumbbell_chart(
    data: pd.DataFrame,
    *,
    metric_a_label: str,
    metric_b_label: str,
    show_endpoint_labels: bool,
    height: int,
) -> go.Figure:
    plot_data = (
        data.copy()
        .sort_values(
            "absolute_divergence",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    fig = go.Figure()

    connector_x: list[
        float | None
    ] = []

    connector_y: list[
        str | None
    ] = []

    for _, row in plot_data.iterrows():
        connector_x.extend(
            [
                float(
                    row[
                        "a_percent_change"
                    ]
                ),
                float(
                    row[
                        "b_percent_change"
                    ]
                ),
                None,
            ]
        )

        connector_y.extend(
            [
                str(row["zone"]),
                str(row["zone"]),
                None,
            ]
        )

    fig.add_trace(
        go.Scatter(
            x=connector_x,
            y=connector_y,
            mode="lines",
            line={
                "color": CONNECTOR_COLOR,
                "width": 3,
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )

    a_customdata = np.column_stack(
        [
            plot_data["zone"],
            plot_data["borough"],
            plot_data[
                "geo_policy_group"
            ],
            plot_data[
                "a_pre_daily_average"
            ].map(_format_number),
            plot_data[
                "a_post_daily_average"
            ].map(_format_number),
            plot_data[
                "a_absolute_change"
            ].map(_format_number),
            plot_data[
                "a_percent_change"
            ].map(_format_percent),
            plot_data[
                "b_percent_change"
            ].map(_format_percent),
            plot_data[
                "absolute_divergence"
            ].map(
                lambda value: (
                    f"{value:,.1f} pp"
                )
            ),
        ]
    )

    b_customdata = np.column_stack(
        [
            plot_data["zone"],
            plot_data["borough"],
            plot_data[
                "geo_policy_group"
            ],
            plot_data[
                "b_pre_daily_average"
            ].map(_format_number),
            plot_data[
                "b_post_daily_average"
            ].map(_format_number),
            plot_data[
                "b_absolute_change"
            ].map(_format_number),
            plot_data[
                "b_percent_change"
            ].map(_format_percent),
            plot_data[
                "a_percent_change"
            ].map(_format_percent),
            plot_data[
                "absolute_divergence"
            ].map(
                lambda value: (
                    f"{value:,.1f} pp"
                )
            ),
        ]
    )

    a_text = (
        plot_data[
            "a_percent_change"
        ].map(_format_percent)
        if show_endpoint_labels
        else None
    )

    b_text = (
        plot_data[
            "b_percent_change"
        ].map(_format_percent)
        if show_endpoint_labels
        else None
    )

    fig.add_trace(
        go.Scatter(
            x=plot_data[
                "a_percent_change"
            ],
            y=plot_data["zone"],
            mode=(
                "markers+text"
                if show_endpoint_labels
                else "markers"
            ),
            name=metric_a_label,
            marker={
                "size": 15,
                "color": METRIC_A_COLOR,
                "line": {
                    "color": "white",
                    "width": 1,
                },
            },
            text=a_text,
            textposition="middle right",
            textfont={
                "size": 10,
                "color": METRIC_A_COLOR,
            },
            cliponaxis=False,
            customdata=a_customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Borough: %{customdata[1]}<br>"
                "Policy geography: %{customdata[2]}<br>"
                f"{metric_a_label}: "
                "%{customdata[3]} → %{customdata[4]}<br>"
                "Daily-average change: %{customdata[5]}<br>"
                "Percent change: %{customdata[6]}<br>"
                f"{metric_b_label} change: "
                "%{customdata[7]}<br>"
                "Divergence: %{customdata[8]}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_data[
                "b_percent_change"
            ],
            y=plot_data["zone"],
            mode=(
                "markers+text"
                if show_endpoint_labels
                else "markers"
            ),
            name=metric_b_label,
            marker={
                "size": 15,
                "color": METRIC_B_COLOR,
                "line": {
                    "color": "white",
                    "width": 1,
                },
            },
            text=b_text,
            textposition="middle left",
            textfont={
                "size": 10,
                "color": METRIC_B_COLOR,
            },
            cliponaxis=False,
            customdata=b_customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Borough: %{customdata[1]}<br>"
                "Policy geography: %{customdata[2]}<br>"
                f"{metric_b_label}: "
                "%{customdata[3]} → %{customdata[4]}<br>"
                "Daily-average change: %{customdata[5]}<br>"
                "Percent change: %{customdata[6]}<br>"
                f"{metric_a_label} change: "
                "%{customdata[7]}<br>"
                "Divergence: %{customdata[8]}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_vline(
        x=0,
        line={
            "color": ZERO_LINE_COLOR,
            "width": 1.5,
        },
    )

    fig.update_xaxes(
        title_text=(
            "Change from pre-CP to post-CP"
        ),
        ticksuffix="%",
        tickformat=".0f",
        range=_axis_range(
            plot_data
        ),
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
    )

    fig.update_yaxes(
        title_text="",
        automargin=True,
    )

    fig = _apply_chart_branding(
        fig
    )

    fig.update_layout(
        height=height,
        margin={
            "l": 12,
            "r": (
                110
                if show_endpoint_labels
                else 30
            ),
            "t": 84,
            "b": 80,
        },
        legend={
            "title": {
                "text": "",
            },
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.09,
            "yanchor": "bottom",
        },
        hovermode="closest",
    )

    return fig


# ---------------------------------------------------------------------
# Dynamic summaries
# ---------------------------------------------------------------------
def _build_hero_takeaway(
    data: pd.DataFrame,
) -> str:
    """Summarize the default pre-CP versus post-CP Taxi–FHVHV comparison."""

    if data.empty:
        return (
            "No eligible opposite-direction Taxi Zones were available."
        )

    total = len(data)

    taxi_up_fhvhv_down = int(
        data["direction_relationship"]
        .eq("Metric A up · Metric B down")
        .sum()
    )

    manhattan_count = int(
        data["borough"]
        .eq("Manhattan")
        .sum()
    )

    taxi_median_change = float(
        data["a_percent_change"]
        .abs()
        .median()
    )

    fhvhv_median_change = float(
        data["b_percent_change"]
        .abs()
        .median()
    )

    return (
        "Comparing average daily activity before and after congestion pricing, "
        f"**{taxi_up_fhvhv_down} of {total} opposite-direction zones** pair "
        "rising Taxi demand with declining FHVHV demand. The size of the "
        f"change is much larger for Taxi: the median absolute change is "
        f"**{taxi_median_change:.1f}%** for Taxi Trips versus "
        f"**{fhvhv_median_change:.1f}%** for FHVHV Trips. "
        f"**{manhattan_count} of these {total} zones are in Manhattan.**"
    )


def _build_explorer_takeaway(
    data: pd.DataFrame,
    *,
    metric_a_label: str,
    metric_b_label: str,
) -> str:
    """Summarize the selected pre-CP versus post-CP divergence comparison."""

    if data.empty:
        return (
            "No eligible opposite-direction zones remain under the "
            "selected filters."
        )

    total = len(data)

    a_up_b_down = int(
        data["direction_relationship"]
        .eq("Metric A up · Metric B down")
        .sum()
    )

    a_down_b_up = int(
        data["direction_relationship"]
        .eq("Metric A down · Metric B up")
        .sum()
    )

    magnitude_a = float(
        data["a_percent_change"]
        .abs()
        .median()
    )

    magnitude_b = float(
        data["b_percent_change"]
        .abs()
        .median()
    )

    if a_up_b_down > a_down_b_up:
        direction_sentence = (
            f"**{a_up_b_down} of {total} zones** pair rising "
            f"{metric_a_label} with declining {metric_b_label}."
        )
    elif a_down_b_up > a_up_b_down:
        direction_sentence = (
            f"**{a_down_b_up} of {total} zones** pair declining "
            f"{metric_a_label} with rising {metric_b_label}."
        )
    else:
        direction_sentence = (
            f"The **{total} qualifying zones** are evenly split between "
            "the two opposite-direction patterns."
        )

    magnitude_sentence = (
        "Across these zones, the median absolute pre-to-post change is "
        f"**{magnitude_a:.1f}%** for {metric_a_label} and "
        f"**{magnitude_b:.1f}%** for {metric_b_label}."
    )

    borough_counts = (
        data["borough"]
        .dropna()
        .value_counts()
    )

    geography_sentence = ""

    if not borough_counts.empty:
        leading_borough = str(
            borough_counts.index[0]
        )
        leading_count = int(
            borough_counts.iloc[0]
        )

        if leading_count / total >= 0.60:
            geography_sentence = (
                f" **{leading_count} of {total}** are in "
                f"**{leading_borough}**."
            )

    top = (
        data.sort_values(
            "absolute_divergence",
            ascending=False,
        )
        .iloc[0]
    )

    example_sentence = (
        f" The widest separation is **{top['zone']}**: "
        f"{metric_a_label} **{top['a_percent_change']:+.1f}%** versus "
        f"{metric_b_label} **{top['b_percent_change']:+.1f}%**."
    )

    return (
        "Comparing average daily activity before and after congestion pricing, "
        f"{direction_sentence} "
        f"{magnitude_sentence}"
        f"{geography_sentence}"
        f"{example_sentence}"
    )


def _build_detail_table(
    data: pd.DataFrame,
    *,
    metric_a_label: str,
    metric_b_label: str,
) -> pd.DataFrame:
    display = (
        data.sort_values(
            "absolute_divergence",
            ascending=False,
        )
        .copy()
        .reset_index(drop=True)
    )

    display.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(display) + 1,
        ),
    )

    display = display[
        [
            "Rank",
            "zone",
            "borough",
            "geo_policy_group",
            "a_pre_daily_average",
            "a_post_daily_average",
            "a_absolute_change",
            "a_percent_change",
            "b_pre_daily_average",
            "b_post_daily_average",
            "b_absolute_change",
            "b_percent_change",
            "absolute_divergence",
        ]
    ].rename(
        columns={
            "zone": "Taxi Zone",
            "borough": "Borough",
            "geo_policy_group": (
                "Policy geography"
            ),
            "a_pre_daily_average": (
                f"{metric_a_label} · Pre"
            ),
            "a_post_daily_average": (
                f"{metric_a_label} · Post"
            ),
            "a_absolute_change": (
                f"{metric_a_label} · Daily-average change"
            ),
            "a_percent_change": (
                f"{metric_a_label} · Percent change"
            ),
            "b_pre_daily_average": (
                f"{metric_b_label} · Pre"
            ),
            "b_post_daily_average": (
                f"{metric_b_label} · Post"
            ),
            "b_absolute_change": (
                f"{metric_b_label} · Daily-average change"
            ),
            "b_percent_change": (
                f"{metric_b_label} · Percent change"
            ),
            "absolute_divergence": (
                "Divergence"
            ),
        }
    )

    return display


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------
st.caption(
    "MODE DIVERGENCES"
)

st.title(
    "Where do modes disagree?"
)

st.write(
    "Citywide trends can make two mobility measures look as though they moved together "
    "even when some neighborhoods went in opposite directions. This page isolates those "
    "local disagreements, measures how large they were, and tests whether the same "
    "opposite-direction pattern recurred across different parts of the week."
)


# ---------------------------------------------------------------------
# Frozen hero
# ---------------------------------------------------------------------
st.header(
    "Where did Taxi and FHVHV demand diverge most?"
)

st.write(
    "The fixed opening view ranks Taxi Zones where **Taxi Trips and FHVHV Trips moved "
    "in opposite directions**. Each connector spans the two percentage changes; a longer "
    "connector means a larger percentage-point gap between the modes."
)

with st.spinner(
    "Preparing the Taxi–FHVHV divergence ranking..."
):
    hero_source = (
        _load_metric_pair_source(
            TAXI_METRIC,
            FHVHV_METRIC,
            ALL_TEMPORAL_BUCKETS_LABEL,
        )
    )

if hero_source.empty:
    st.error(
        "The Taxi Zone pre/post summary returned no rows."
    )
    st.stop()

try:
    hero_data = (
        _prepare_metric_pair_divergence(
            hero_source,
            metric_a=TAXI_METRIC,
            metric_b=FHVHV_METRIC,
        )
    )
except (
    KeyError,
    TypeError,
    ValueError,
) as error:
    st.error(str(error))
    st.stop()

if hero_data.empty:
    st.warning(
        "No eligible Taxi–FHVHV divergences were available."
    )
    st.stop()

hero_top = (
    hero_data.head(
        HERO_TOP_N
    )
    .copy()
)

hero_card1, hero_card2, hero_card3 = (
    st.columns(3)
)

hero_card1.metric(
    "Eligible disagreements",
    f"{len(hero_data):,}",
)

hero_card2.metric(
    "Taxi up · FHVHV down",
    (
        f"{int(hero_data['direction_relationship'].eq('Metric A up · Metric B down').sum()):,}"
    ),
)

hero_card3.metric(
    "In Manhattan",
    (
        f"{int(hero_data['borough'].eq('Manhattan').sum())} "
        f"of {len(hero_data)}"
    ),
)

st.caption(
    "Ranked by the percentage-point gap between Taxi and FHVHV change."
)

hero_fig = build_dumbbell_chart(
    hero_top,
    metric_a_label=(
        _metric_label(
            TAXI_METRIC
        )
    ),
    metric_b_label=(
        _metric_label(
            FHVHV_METRIC
        )
    ),
    show_endpoint_labels=True,
    height=760,
)

st.plotly_chart(
    hero_fig,
    width="stretch",
    config={
        "displayModeBar": False,
        "responsive": True,
    },
    key="raw06_static_hero",
)

render_chart_insight(_build_hero_takeaway(hero_data))

st.caption(
    "Includes only zones with meaningful pre-period activity for both "
    "measures. Hover for pre-CP and post-CP daily averages and absolute "
    "changes."
)


# ---------------------------------------------------------------------
# Interactive explorer
# ---------------------------------------------------------------------
with exploration_section(
    key="raw06_exploration_area",
    title="Explore mode divergences",
    description=(
        "Choose two mobility measures and a geographic frame, then compare "
        "the largest opposite-direction changes with patterns that recur "
        "across the time-of-week profile."
    ),
):
    if "raw06_saved_view" not in st.session_state:
        st.session_state[
            "raw06_saved_view"
        ] = "Taxi vs FHVHV demand"

    saved_view = st.selectbox(
        "Start with a saved configuration",
        options=SAVED_VIEW_OPTIONS,
        key="raw06_saved_view",
        help=(
            "Saved configurations provide curated starting points for the "
            "largest-divergence view. Changing a shared control switches "
            "the selection to Custom."
        ),
    )

    if (
        saved_view != "Custom"
        and st.session_state.get(
            "_raw06_applied_saved_view"
        )
        != saved_view
    ):
        _apply_saved_view(
            saved_view
        )

        st.session_state[
            "_raw06_applied_saved_view"
        ] = saved_view

        st.rerun()

    if saved_view == "Custom":
        st.caption(
            "Custom view · adjust any measure, geography, time, or threshold."
        )
    else:
        st.caption(
            SAVED_VIEWS[
                saved_view
            ]["description"]
        )

    st.markdown(
        "##### Shared measures"
    )

    measure_col1, measure_col2 = st.columns(2)

    with measure_col1:
        metric_a = st.selectbox(
            "Metric A",
            options=CORE_METRICS,
            format_func=_metric_label,
            key="raw06_metric_a",
            on_change=_mark_saved_view_custom,
        )

    metric_b_options = [
        metric
        for metric in CORE_METRICS
        if metric != metric_a
    ]

    with measure_col2:
        if (
            st.session_state.get(
                "raw06_metric_b"
            )
            not in metric_b_options
        ):
            st.session_state[
                "raw06_metric_b"
            ] = (
                FHVHV_METRIC
                if FHVHV_METRIC in metric_b_options
                else metric_b_options[0]
            )

        metric_b = st.selectbox(
            "Metric B",
            options=metric_b_options,
            format_func=_metric_label,
            key="raw06_metric_b",
            on_change=_mark_saved_view_custom,
        )

    metric_a_label = _metric_label(
        metric_a
    )
    metric_b_label = _metric_label(
        metric_b
    )

    with st.spinner(
        "Preparing geography and divergence ranges..."
    ):
        overall_source = _load_metric_pair_source(
            metric_a,
            metric_b,
            ALL_TEMPORAL_BUCKETS_LABEL,
        )

        overall_evidence = _prepare_metric_pair_evidence(
            overall_source,
            metric_a=metric_a,
            metric_b=metric_b,
        )

    st.markdown(
        "##### Shared geography and threshold"
    )

    geography_col1, geography_col2, threshold_col = st.columns(
        [
            1,
            1,
            1.4,
        ]
    )

    with geography_col1:
        geography_filter = st.selectbox(
            "Geography filter",
            options=GEOGRAPHY_FILTER_OPTIONS,
            format_func=_format_geography_term,
            key="raw06_geography_filter",
            on_change=_mark_saved_view_custom,
        )

    geography_value: object | None = None

    if geography_filter == "Borough":
        borough_options = _ordered_values(
            overall_evidence["borough"],
            BOROUGH_ORDER,
        )

        if borough_options:
            if (
                st.session_state.get(
                    "raw06_borough_value"
                )
                not in borough_options
            ):
                st.session_state[
                    "raw06_borough_value"
                ] = borough_options[0]

            with geography_col2:
                geography_value = st.selectbox(
                    "Borough",
                    options=borough_options,
                    key="raw06_borough_value",
                    on_change=_mark_saved_view_custom,
                )
        else:
            with geography_col2:
                st.caption(
                    "No Borough values available"
                )

    elif geography_filter == "Geo-policy group":
        policy_options = _ordered_values(
            overall_evidence[
                "geo_policy_group"
            ],
            GEO_POLICY_ORDER,
        )

        if policy_options:
            if (
                st.session_state.get(
                    "raw06_policy_value"
                )
                not in policy_options
            ):
                st.session_state[
                    "raw06_policy_value"
                ] = policy_options[0]

            with geography_col2:
                geography_value = st.selectbox(
                    "Policy geography",
                    options=policy_options,
                    key="raw06_policy_value",
                    on_change=_mark_saved_view_custom,
                )
        else:
            with geography_col2:
                st.caption(
                    "No policy-geography values available"
                )

    elif geography_filter == "Mobility regime cluster":
        cluster_options = MOBILITY_REGIME_CLUSTER_ORDER

        if cluster_options:
            if (
                st.session_state.get(
                    "raw06_cluster_value"
                )
                not in cluster_options
            ):
                st.session_state[
                    "raw06_cluster_value"
                ] = cluster_options[0]

            with geography_col2:
                geography_value = st.selectbox(
                    "Mobility environment",
                    options=cluster_options,
                    format_func=format_mobility_regime_cluster_label,
                    key="raw06_cluster_value",
                    on_change=_mark_saved_view_custom,
                )
        else:
            with geography_col2:
                st.caption(
                    "No mobility environments available"
                )
    else:
        with geography_col2:
            st.caption(
                "All eligible Taxi Zones"
            )

    maximum_divergence = (
        float(
            np.ceil(
                overall_evidence[
                    "absolute_divergence"
                ].max()
                / 5
            )
            * 5
        )
        if not overall_evidence.empty
        else 100.0
    )

    with threshold_col:
        minimum_divergence = st.slider(
            "Minimum divergence",
            min_value=0.0,
            max_value=max(
                maximum_divergence,
                5.0,
            ),
            step=5.0,
            format="%.0f pp",
            help=(
                "Require at least this many percentage points between the "
                "two pre-to-post changes."
            ),
            key="raw06_minimum_divergence",
            on_change=_mark_saved_view_custom,
        )

    geography_context = (
        "All Taxi Zones"
        if geography_filter == "All Taxi Zones"
        else (
            f"Mobility environment: {format_mobility_regime_cluster_label(geography_value)}"
            if geography_filter == "Mobility regime cluster"
            else f"{_format_geography_term(geography_filter)}: {geography_value}"
        )
    )


    def _build_selected_zone_takeaway(
        zone_evidence: pd.DataFrame,
        *,
        zone_name: str,
        metric_a_label: str,
        metric_b_label: str,
        minimum_divergence: float,
    ) -> str:
        """Summarize the strongest time-bucket disagreement for one zone."""
        if zone_evidence.empty:
            return "No eligible time-bucket evidence is available for this zone."

        opposite = zone_evidence[
            zone_evidence["opposite_direction"]
            & zone_evidence["absolute_divergence"].ge(minimum_divergence)
        ].copy()
        source = opposite if not opposite.empty else zone_evidence
        strongest = source.loc[source["absolute_divergence"].idxmax()]
        relationship = _direction_label(
            str(strongest["direction_relationship"]),
            metric_a_label=metric_a_label,
            metric_b_label=metric_b_label,
        )
        opposite_count = int(opposite["temporal_bucket"].nunique())

        return (
            f"**{zone_name}** shows opposite movement in **{opposite_count} of "
            f"{zone_evidence['temporal_bucket'].nunique()} eligible buckets**. The "
            f"largest displayed gap occurs during **{strongest['temporal_bucket_label']}**: "
            f"**{relationship}**, separated by "
            f"**{float(strongest['absolute_divergence']):.1f} percentage points**."
        )
    largest_tab, recurrence_tab = st.tabs(
        [
            "Largest divergences",
            "Recurring patterns",
        ]
    )

    with largest_tab:
        st.markdown(
            "Rank the zones with the largest opposite-direction movement for "
            "one selected temporal bucket."
        )

        ranking_col1, ranking_col2, ranking_col3 = st.columns(3)

        with ranking_col1:
            temporal_bucket = st.selectbox(
                "Time bucket",
                options=TEMPORAL_BUCKET_OPTIONS,
                format_func=lambda value: (
                    TEMPORAL_BUCKET_LABELS[
                        value
                    ]
                ),
                key="raw06_temporal_bucket",
                on_change=_mark_saved_view_custom,
            )

        with ranking_col2:
            top_n = st.selectbox(
                "Zones to show",
                options=TOP_N_OPTIONS,
                key="raw06_top_n",
                on_change=_mark_saved_view_custom,
            )

        with ranking_col3:
            direction_filter = st.selectbox(
                "Direction pattern",
                options=DIRECTION_OPTIONS,
                format_func=lambda value: (
                    _direction_label(
                        value,
                        metric_a_label=metric_a_label,
                        metric_b_label=metric_b_label,
                    )
                ),
                key="raw06_direction_filter",
                on_change=_mark_saved_view_custom,
            )

        with st.spinner(
            "Updating the divergence ranking..."
        ):
            explorer_source = _load_metric_pair_source(
                metric_a,
                metric_b,
                temporal_bucket,
            )

            explorer_data = _prepare_metric_pair_divergence(
                explorer_source,
                metric_a=metric_a,
                metric_b=metric_b,
            )

        filtered_data = _filter_divergence_data(
            explorer_data,
            geography_filter=geography_filter,
            geography_value=geography_value,
            direction_filter=direction_filter,
            minimum_divergence=minimum_divergence,
        )

        if filtered_data.empty:
            st.info(
                "No eligible opposite-direction zones remain under the selected "
                "filters."
            )
        else:
            displayed_data = (
                filtered_data.head(
                    top_n
                )
                .copy()
            )

            explorer_card1, explorer_card2, explorer_card3, explorer_card4 = (
                st.columns(4)
            )

            explorer_card1.metric(
                "Eligible disagreements",
                f"{len(filtered_data):,}",
            )

            explorer_card2.metric(
                f"{metric_a_label} up · {metric_b_label} down",
                f"{int(filtered_data['direction_relationship'].eq('Metric A up · Metric B down').sum()):,}",
            )

            explorer_card3.metric(
                f"{metric_a_label} down · {metric_b_label} up",
                f"{int(filtered_data['direction_relationship'].eq('Metric A down · Metric B up').sum()):,}",
            )

            explorer_card4.metric(
                "Largest divergence",
                f"{filtered_data.iloc[0]['absolute_divergence']:.1f} pp",
            )

            st.caption(
                f"Metric A: {metric_a_label} · "
                f"Metric B: {metric_b_label} · "
                f"{TEMPORAL_BUCKET_LABELS[temporal_bucket]} · "
                f"{geography_context} · "
                f"Minimum gap: {minimum_divergence:.0f} pp"
            )

            if "subway_ridership" in {
                metric_a,
                metric_b,
            }:
                st.caption(
                    "Subway Ridership does not cover Staten Island, so Staten "
                    "Island cannot appear when Subway Ridership is selected."
                )

            explorer_takeaway = _build_explorer_takeaway(
                filtered_data,
                metric_a_label=metric_a_label,
                metric_b_label=metric_b_label,
            )

            explorer_fig = build_dumbbell_chart(
                displayed_data,
                metric_a_label=metric_a_label,
                metric_b_label=metric_b_label,
                show_endpoint_labels=(
                    top_n <= 15
                ),
                height=max(
                    560,
                    44 * len(
                        displayed_data
                    )
                    + 170,
                ),
            )

            st.plotly_chart(
                explorer_fig,
                width="stretch",
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
                key=(
                    f"raw06_largest_"
                    f"{metric_a}_{metric_b}_"
                    f"{temporal_bucket}_"
                    f"{geography_filter}_"
                    f"{geography_value}_"
                    f"{direction_filter}_"
                    f"{minimum_divergence}_"
                    f"{top_n}"
                ),
            )
            render_chart_insight(explorer_takeaway)

            st.caption(
                "Longer connectors indicate a larger percentage-point gap. "
                "Only zones with meaningful pre-period activity for both selected "
                "measures are eligible."
            )

            with st.expander(
                "Inspect the ranked values",
                expanded=False,
            ):
                detail = _build_detail_table(
                    displayed_data,
                    metric_a_label=metric_a_label,
                    metric_b_label=metric_b_label,
                )

                detail_column_config = {}

                for column in detail.columns:
                    if column.endswith(" · Pre") or column.endswith(" · Post"):
                        detail_column_config[column] = (
                            st.column_config.NumberColumn(
                                column,
                                format="%,.1f",
                            )
                        )
                    elif column.endswith(" · Daily-average change"):
                        detail_column_config[column] = (
                            st.column_config.NumberColumn(
                                column,
                                format="%+,.1f",
                            )
                        )
                    elif column.endswith(" · Percent change"):
                        detail_column_config[column] = (
                            st.column_config.NumberColumn(
                                column,
                                format="%+.1f%%",
                            )
                        )
                    elif column == "Divergence":
                        detail_column_config[column] = (
                            st.column_config.NumberColumn(
                                column,
                                format="%.1f pp",
                            )
                        )

                st.dataframe(
                    detail,
                    width="stretch",
                    hide_index=True,
                    column_config=detail_column_config,
                )

    with recurrence_tab:
        st.markdown(
            "Identify zones where opposite movement repeats across multiple "
            "weekday and weekend temporal buckets. This is recurring divergence "
            "evidence, not proof that one mode substituted for another."
        )

        recurrence_control1, recurrence_control2 = st.columns(2)

        with recurrence_control1:
            recurrence_top_n = st.selectbox(
                "Zones to show",
                options=TOP_N_OPTIONS,
                index=1,
                key="raw06_recurrence_top_n",
            )

        with recurrence_control2:
            minimum_recurring_buckets = st.slider(
                "Minimum recurring buckets",
                min_value=1,
                max_value=len(RECURRENCE_BUCKETS),
                value=2,
                step=1,
                help=(
                    "Require opposite-direction movement in at least this many "
                    "eligible time-of-week buckets."
                ),
                key="raw06_minimum_recurring_buckets",
            )

        with st.spinner(
            "Calculating recurrence across temporal buckets..."
        ):
            recurrence_summary, bucket_evidence = _build_recurrence_evidence(
                metric_a,
                metric_b,
                minimum_divergence,
            )

        filtered_recurrence = _filter_recurrence_summary(
            recurrence_summary,
            geography_filter=geography_filter,
            geography_value=geography_value,
            minimum_recurring_buckets=minimum_recurring_buckets,
        )

        if filtered_recurrence.empty:
            st.info(
                "No zones met the selected recurrence and divergence requirements."
            )
        else:
            displayed_recurrence = filtered_recurrence.head(
                recurrence_top_n
            ).copy()

            recurrence_cards = _build_recurrence_summary_cards(
                filtered_recurrence
            )

            _render_recurrence_cards(
                recurrence_cards
            )

            st.caption(
                f"Metric A: {metric_a_label} · "
                f"Metric B: {metric_b_label} · "
                f"{geography_context} · "
                f"Minimum gap for each counted recurring bucket: "
                f"{minimum_divergence:.0f} pp"
            )

            recurrence_takeaway = _build_recurrence_takeaway(
                filtered_recurrence,
                metric_a_label=metric_a_label,
                metric_b_label=metric_b_label,
            )

            recurrence_fig = build_recurrence_dot_plot(
                displayed_recurrence,
                metric_a_label=metric_a_label,
                metric_b_label=metric_b_label,
                height=max(
                    560,
                    42 * len(
                        displayed_recurrence
                    )
                    + 155,
                ),
            )

            st.plotly_chart(
                recurrence_fig,
                width="stretch",
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
                key=(
                    f"raw06_recurrence_"
                    f"{metric_a}_{metric_b}_"
                    f"{geography_filter}_"
                    f"{geography_value}_"
                    f"{minimum_divergence}_"
                    f"{minimum_recurring_buckets}_"
                    f"{recurrence_top_n}"
                ),
            )
            render_chart_insight(recurrence_takeaway)

            st.caption(
                "Horizontal position is the share of eligible temporal buckets "
                "showing opposite movement. Dot size represents median inverse "
                "divergence, and color represents the number of inverse buckets."
            )

            selected_zone_id = st.selectbox(
                "Inspect a recurring zone",
                options=displayed_recurrence[
                    "taxi_zone_id"
                ].tolist(),
                format_func=lambda zone_id: str(
                    displayed_recurrence.loc[
                        displayed_recurrence["taxi_zone_id"].eq(zone_id),
                        "zone",
                    ].iloc[0]
                ),
                key="raw06_selected_recurrence_zone",
            )

            selected_zone_name = str(
                displayed_recurrence.loc[
                    displayed_recurrence["taxi_zone_id"].eq(
                        selected_zone_id
                    ),
                    "zone",
                ].iloc[0]
            )

            selected_zone_evidence = bucket_evidence[
                bucket_evidence["taxi_zone_id"].eq(
                    selected_zone_id
                )
            ].copy()

            st.markdown(
                f"##### {selected_zone_name}: evidence by time bucket"
            )

            selected_zone_fig = build_selected_zone_evidence_chart(
                selected_zone_evidence,
                metric_a_label=metric_a_label,
                metric_b_label=metric_b_label,
            )

            st.plotly_chart(
                selected_zone_fig,
                width="stretch",
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
                key=(
                    f"raw06_selected_zone_"
                    f"{selected_zone_id}_"
                    f"{metric_a}_{metric_b}"
                ),
            )

            render_chart_insight(
                _build_selected_zone_takeaway(
                    selected_zone_evidence,
                    zone_name=selected_zone_name,
                    metric_a_label=metric_a_label,
                    metric_b_label=metric_b_label,
                    minimum_divergence=minimum_divergence,
                )
            )

            st.caption(
                "Opposite signs indicate an inverse pattern in that temporal "
                "bucket. A bucket counts toward recurrence only when its gap also "
                "meets the shared minimum-divergence threshold; same-sign bars show "
                "that the two measures moved together."
            )

            with st.expander(
                "Inspect recurrence evidence",
                expanded=False,
            ):
                recurrence_detail = _build_recurrence_detail_table(
                    filtered_recurrence,
                    metric_a_label=metric_a_label,
                    metric_b_label=metric_b_label,
                )

                st.dataframe(
                    recurrence_detail,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Inverse bucket share": (
                            st.column_config.NumberColumn(
                                format="%.1f%%",
                            )
                        ),
                        "Median divergence": (
                            st.column_config.NumberColumn(
                                format="%.1f pp",
                            )
                        ),
                        "Maximum divergence": (
                            st.column_config.NumberColumn(
                                format="%.1f pp",
                            )
                        ),
                    },
                )


st.markdown("### What this page establishes")
st.markdown(
    "Opposite-direction movement is not just the inverse of a citywide relationship. "
    "It identifies specific places where two measures separated, and the recurrence "
    "view distinguishes a one-off disagreement from a pattern that appears across "
    "multiple weekday or weekend time buckets."
)

with st.expander(
    "How this page works",
    expanded=False,
):
    st.markdown(
        """
        **1. Require a genuine opposite-direction pattern.** A Taxi Zone enters a
        divergence ranking only when both selected measures have complete pre-CP and
        post-CP values, both pass the project's metric-specific baseline requirement,
        and one measure increased while the other decreased.

        **2. Measure the size of the disagreement.** Divergence is the absolute
        percentage-point gap between the two percentage changes. The sign of each
        endpoint preserves which measure increased and which decreased; the connector
        length shows how far apart they were.

        **3. Check whether the pattern repeats.** The recurrence analysis applies the
        same eligibility and opposite-direction rules separately to the ten ordered
        weekday and weekend temporal buckets.

        **4. Require a meaningful gap for recurrence.** A time bucket counts only when
        its opposite-direction gap also meets the shared minimum-divergence threshold.
        Recurrence then summarizes how often that qualifying pattern repeats and how
        large those disagreements were.

        **5. Keep divergence descriptive.** Opposite-direction changes can surface
        possible substitution-like patterns worth investigating, but they do not
        establish that one mode displaced another or that congestion pricing caused
        the divergence.
        """
    )

st.caption(
    "Evidence scope: observed NYC mobility before and after the January 5, 2025 "
    "congestion-pricing launch. Divergence identifies opposite-direction pre/post "
    "changes among eligible measures and Taxi Zones; it is descriptive evidence, "
    "not a causal or substitution estimate."
)
