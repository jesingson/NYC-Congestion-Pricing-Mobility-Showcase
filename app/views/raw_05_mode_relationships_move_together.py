from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data_access.loaders import CORE_METRICS
from app.data_access.mode_relationships import (
    build_metric_pair_data,
    build_pre_post_pair_comparison,
    build_relationship_long_data,
    calculate_pair_correlations,
)
from app.data_access.spatial_aggregations import (
    ALL_TEMPORAL_BUCKETS_LABEL,
)
from app.utils.project_branding import (
    BRAND_COLORS,
    apply_branding,
    exploration_section,
    inject_app_css,
    render_chart_insight,
)
from app.data_access.mobility_environments import (
    attach_mobility_regime_cluster_context,
    format_mobility_regime_cluster_label,
    get_mobility_regime_cluster_options,
)


inject_app_css()

TAXI_METRIC = "taxi_trip_count"
FHVHV_METRIC = "fhvhv_trip_count"
SUBWAY_METRIC = "subway_ridership"

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

PERIOD_OPTIONS = [
    "Both",
    "Pre-CP",
    "Post-CP",
]

DISPLAY_MODE_OPTIONS = [
    "Taxi Zones",
    "Aggregated geographies",
]

TAXI_ZONE_FILTER_OPTIONS = [
    "All Taxi Zones",
    "Borough",
    "Geo-policy group",
    "Mobility regime cluster",
]

TAXI_ZONE_COLOR_OPTIONS = [
    "Geo-policy group",
    "Borough",
    "Mobility regime cluster",
]

AGGREGATION_OPTIONS = [
    "Borough",
    "Geo-policy group",
    "Mobility regime cluster",
]

MOBILITY_REGIME_CLUSTER_ORDER = [
    format_mobility_regime_cluster_label(label)
    for label in get_mobility_regime_cluster_options()
]

# Two additional categorical shades are midpoint blends of the canonical
# palette. They preserve five-way differentiation without introducing unrelated
# purple or olive accents into a page otherwise built from the project brand.
BRAND_TEAL_MID = "#42999A"
BRAND_PEACH_MID = "#F0B9A5"

MOBILITY_REGIME_CLUSTER_COLORS = {
    MOBILITY_REGIME_CLUSTER_ORDER[0]: BRAND_COLORS["dark_teal"],
    MOBILITY_REGIME_CLUSTER_ORDER[1]: BRAND_COLORS["terracotta"],
    MOBILITY_REGIME_CLUSTER_ORDER[2]: BRAND_COLORS["seafoam"],
    MOBILITY_REGIME_CLUSTER_ORDER[3]: BRAND_TEAL_MID,
    MOBILITY_REGIME_CLUSTER_ORDER[4]: BRAND_PEACH_MID,
}

GEO_POLICY_ORDER = [
    "CBD",
    "Gateway",
    "Adjacent",
    "Non-CBD",
    "Unknown",
]

BOROUGH_ORDER = [
    "Manhattan",
    "Brooklyn",
    "Queens",
    "Bronx",
    "Staten Island",
    "Unknown",
]

GEO_POLICY_COLORS = {
    "CBD": BRAND_COLORS["dark_teal"],
    "Gateway": BRAND_COLORS["terracotta"],
    "Adjacent": BRAND_COLORS["seafoam"],
    "Non-CBD": BRAND_TEAL_MID,
    "Unknown": "#8A8A8A",
}

BOROUGH_COLORS = {
    "Manhattan": BRAND_COLORS["dark_teal"],
    "Brooklyn": BRAND_COLORS["terracotta"],
    "Queens": BRAND_TEAL_MID,
    "Bronx": BRAND_PEACH_MID,
    "Staten Island": BRAND_COLORS["seafoam"],
    "Unknown": "#8A8A8A",
}


SAVED_VIEWS = {
    "Multimodal demand landscape": {
        "description": (
            "Taxi and FHVHV demand across Taxi Zones, with Subway "
            "Ridership represented by bubble area."
        ),
        "x_metric": TAXI_METRIC,
        "y_metric": FHVHV_METRIC,
        "bubble_metric": SUBWAY_METRIC,
        "display_mode": "Taxi Zones",
        "filter_scope": "All Taxi Zones",
        "taxi_zone_color": "Geo-policy group",
        "aggregate_by": "Geo-policy group",
        "temporal_bucket": ALL_TEMPORAL_BUCKETS_LABEL,
        "period_view": "Both",
    },
    "Surface-mode demand": {
        "description": (
            "A direct comparison of Taxi and FHVHV demand without a "
            "third measure narrowing the geographic sample."
        ),
        "x_metric": TAXI_METRIC,
        "y_metric": FHVHV_METRIC,
        "bubble_metric": None,
        "display_mode": "Taxi Zones",
        "filter_scope": "All Taxi Zones",
        "taxi_zone_color": "Borough",
        "aggregate_by": "Geo-policy group",
        "temporal_bucket": ALL_TEMPORAL_BUCKETS_LABEL,
        "period_view": "Both",
    },
    "Taxi and subway activity": {
        "description": (
            "Compare Taxi Trips with Subway Ridership across Taxi Zones, "
            "colored by Borough."
        ),
        "x_metric": TAXI_METRIC,
        "y_metric": SUBWAY_METRIC,
        "bubble_metric": None,
        "display_mode": "Taxi Zones",
        "filter_scope": "All Taxi Zones",
        "taxi_zone_color": "Borough",
        "aggregate_by": "Geo-policy group",
        "temporal_bucket": ALL_TEMPORAL_BUCKETS_LABEL,
        "period_view": "Both",
    },
    "Surface speeds at the PM peak": {
        "description": (
            "Compare Taxi and FHVHV average speeds during the weekday "
            "PM peak, with Taxi Trips represented by bubble area."
        ),
        "x_metric": "taxi_avg_trip_speed",
        "y_metric": "fhvhv_avg_trip_speed",
        "bubble_metric": TAXI_METRIC,
        "display_mode": "Taxi Zones",
        "filter_scope": "All Taxi Zones",
        "taxi_zone_color": "Geo-policy group",
        "aggregate_by": "Geo-policy group",
        "temporal_bucket": "weekday_pm_peak",
        "period_view": "Both",
    },
    "Policy-geography summary": {
        "description": (
            "Recreate the policy-level multimodal story using aggregated "
            "geo-policy groups."
        ),
        "x_metric": TAXI_METRIC,
        "y_metric": FHVHV_METRIC,
        "bubble_metric": SUBWAY_METRIC,
        "display_mode": "Aggregated geographies",
        "filter_scope": "All Taxi Zones",
        "taxi_zone_color": "Geo-policy group",
        "aggregate_by": "Geo-policy group",
        "temporal_bucket": ALL_TEMPORAL_BUCKETS_LABEL,
        "period_view": "Both",
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

    st.session_state["raw05_x_metric"] = (
        configuration["x_metric"]
    )

    st.session_state["raw05_y_metric"] = (
        configuration["y_metric"]
    )

    st.session_state["raw05_bubble_metric"] = (
        configuration["bubble_metric"]
    )

    st.session_state["raw05_display_mode"] = (
        configuration["display_mode"]
    )

    st.session_state[
        "raw05_taxi_zone_filter_scope"
    ] = configuration["filter_scope"]

    st.session_state["raw05_taxi_zone_color"] = (
        configuration["taxi_zone_color"]
    )

    st.session_state["raw05_aggregate_by"] = (
        configuration["aggregate_by"]
    )

    st.session_state["raw05_temporal_bucket"] = (
        configuration["temporal_bucket"]
    )

    st.session_state["raw05_period_view"] = (
        configuration["period_view"]
    )

    st.session_state.pop(
        "raw05_taxi_zone_filter_value",
        None,
    )


def _mark_saved_view_custom() -> None:
    st.session_state["raw05_saved_view"] = (
        "Custom"
    )


# ---------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------
def _normalize_geo_policy(
    value: object,
) -> str:
    if pd.isna(value):
        return "Unknown"

    text = str(value).strip()

    normalized = (
        text.lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    if normalized == "cbd":
        return "CBD"

    if normalized in {
        "gateway",
        "gateway to cbd",
    }:
        return "Gateway"

    if normalized in {
        "adjacent",
        "adjacent to cbd",
    }:
        return "Adjacent"

    if normalized in {
        "non cbd",
        "noncbd",
    }:
        return "Non-CBD"

    if normalized in {
        "",
        "unknown",
        "none",
        "nan",
    }:
        return "Unknown"

    return text


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


def _format_correlation(
    value: object,
) -> str:
    if pd.isna(value):
        return "Unavailable"

    return f"{float(value):+.3f}"


def _metric_display_label(
    metric: str | None,
) -> str:
    if metric is None:
        return "None · Fixed marker size"

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


def _safe_percent_change(
    pre_value: object,
    post_value: object,
) -> float:
    if (
        pd.isna(pre_value)
        or pd.isna(post_value)
        or float(pre_value) == 0
    ):
        return np.nan

    return (
        (
            float(post_value)
            - float(pre_value)
        )
        / float(pre_value)
        * 100
    )


def _safe_log_range(
    values: pd.Series,
    *,
    padding: float = 0.10,
) -> list[float] | None:
    positive = pd.to_numeric(
        values,
        errors="coerce",
    )

    positive = positive[
        positive.notna()
        & positive.gt(0)
    ]

    if positive.empty:
        return None

    minimum = float(positive.min())
    maximum = float(positive.max())

    if minimum == maximum:
        minimum *= 0.8
        maximum *= 1.2

    log_minimum = math.log10(minimum)
    log_maximum = math.log10(maximum)

    span = max(
        log_maximum - log_minimum,
        0.25,
    )

    return [
        log_minimum - span * padding,
        log_maximum + span * padding,
    ]


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

        clean_annotations.append(annotation)

    fig.update_layout(
        annotations=clean_annotations,
    )

    return fig


def _metric_label_lookup(
    relationship_data: pd.DataFrame,
) -> dict[str, str]:
    if relationship_data.empty:
        return {}

    return (
        relationship_data[
            [
                "metric",
                "metric_label",
            ]
        ]
        .drop_duplicates()
        .set_index("metric")["metric_label"]
        .to_dict()
    )


def _get_period_correlation(
    correlations: pd.DataFrame,
    *,
    period: str,
) -> float:
    if correlations.empty:
        return np.nan

    match = correlations[
        correlations["period"].eq(period)
    ]

    if match.empty:
        return np.nan

    return float(
        match.iloc[0]["pearson_correlation"]
    )


# ---------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------
def _get_color_map(
    *,
    geography_level: str,
    taxi_zone_color: str,
) -> dict[str, str]:
    if geography_level == "Mobility regime cluster":
        return MOBILITY_REGIME_CLUSTER_COLORS

    if geography_level == "Borough":
        return BOROUGH_COLORS

    if (
        geography_level == "Taxi Zone"
        and taxi_zone_color == "Borough"
    ):
        return BOROUGH_COLORS

    if (
        geography_level == "Taxi Zone"
        and taxi_zone_color == "Mobility regime cluster"
    ):
        return MOBILITY_REGIME_CLUSTER_COLORS

    return GEO_POLICY_COLORS


def _get_group_order(
    data: pd.DataFrame,
    *,
    geography_level: str,
    taxi_zone_color: str,
) -> list[str]:
    represented = set(
        data["color_group"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    use_borough_order = (
        geography_level == "Borough"
        or (
            geography_level == "Taxi Zone"
            and taxi_zone_color == "Borough"
        )
    )

    use_cluster_order = (
        geography_level == "Mobility regime cluster"
        or (
            geography_level == "Taxi Zone"
            and taxi_zone_color == "Mobility regime cluster"
        )
    )

    preferred = (
        BOROUGH_ORDER
        if use_borough_order
        else MOBILITY_REGIME_CLUSTER_ORDER
        if use_cluster_order
        else GEO_POLICY_ORDER
    )

    ordered = [
        value
        for value in preferred
        if value in represented
    ]

    return ordered + sorted(
        represented.difference(ordered)
    )


def _add_color_groups(
    data: pd.DataFrame,
    *,
    geography_level: str,
    taxi_zone_color: str,
) -> pd.DataFrame:
    result = data.copy()

    result["geo_policy"] = (
        result["cbd_spatial_category"]
        .map(_normalize_geo_policy)
    )

    if geography_level == "Borough":
        result["color_group"] = (
            result["geography_name"]
            .fillna("Unknown")
            .astype(str)
        )

    elif geography_level == "Mobility regime cluster":
        result["color_group"] = (
            result["geography_name"]
            .fillna("Cluster Unknown")
            .astype(str)
        )

    elif (
        geography_level == "Taxi Zone"
        and taxi_zone_color == "Borough"
    ):
        result["color_group"] = (
            result["borough"]
            .fillna("Unknown")
            .astype(str)
        )

    elif (
        geography_level == "Taxi Zone"
        and taxi_zone_color == "Mobility regime cluster"
    ):
        result["color_group"] = (
            result["mobility_regime_cluster_label"]
            .map(format_mobility_regime_cluster_label)
            .fillna("Cluster Unknown")
            .astype(str)
        )

    else:
        result["color_group"] = (
            result["geo_policy"]
        )

    return result


# ---------------------------------------------------------------------
# Bubble-size helpers
# ---------------------------------------------------------------------
def _calculate_sizeref(
    values: pd.Series,
    *,
    maximum_marker_size: float,
) -> float:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    numeric = numeric[
        numeric.notna()
        & numeric.gt(0)
    ]

    if numeric.empty:
        return 1.0

    return (
        2.0
        * float(numeric.max())
        / maximum_marker_size**2
    )


def _attach_bubble_metric(
    pair_data: pd.DataFrame,
    relationship_data: pd.DataFrame,
    *,
    bubble_metric: str | None,
) -> pd.DataFrame:
    result = pair_data.copy()

    if bubble_metric is None:
        result["bubble_metric"] = "Fixed marker size"
        result["bubble_metric_label"] = "Fixed marker size"
        result["bubble_value"] = 1.0
        return result

    key_columns = [
        "geography_level",
        "geography_id",
        "geography_name",
        "temporal_bucket",
        "period",
    ]

    bubble_data = (
        relationship_data[
            relationship_data["metric"].eq(
                bubble_metric
            )
        ][
            key_columns
            + [
                "metric_label",
                "metric_value",
            ]
        ]
        .copy()
        .rename(
            columns={
                "metric_label": "bubble_metric_label",
                "metric_value": "bubble_value",
            }
        )
    )

    result = result.merge(
        bubble_data,
        on=key_columns,
        how="inner",
        validate="one_to_one",
    )

    result["bubble_metric"] = bubble_metric

    return result


def _build_bubble_comparison(
    pair_comparison: pd.DataFrame,
    pair_data: pd.DataFrame,
    *,
    bubble_metric: str | None,
) -> pd.DataFrame:
    result = pair_comparison.copy()

    if bubble_metric is None:
        result["bubble_metric"] = "Fixed marker size"
        result["bubble_metric_label"] = "Fixed marker size"
        result["bubble_value_pre_cp"] = 1.0
        result["bubble_value_post_cp"] = 1.0
        result["bubble_percent_change"] = np.nan
        return result

    key_columns = [
        "geography_level",
        "geography_id",
        "geography_name",
        "temporal_bucket",
    ]

    bubble_rows = (
        pair_data[
            [
                *key_columns,
                "period",
                "bubble_metric",
                "bubble_metric_label",
                "bubble_value",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    pre = (
        bubble_rows[
            bubble_rows["period"].eq("Pre-CP")
        ][
            key_columns
            + [
                "bubble_metric",
                "bubble_metric_label",
                "bubble_value",
            ]
        ]
        .rename(
            columns={
                "bubble_value": "bubble_value_pre_cp",
            }
        )
    )

    post = (
        bubble_rows[
            bubble_rows["period"].eq("Post-CP")
        ][
            key_columns
            + [
                "bubble_value",
            ]
        ]
        .rename(
            columns={
                "bubble_value": "bubble_value_post_cp",
            }
        )
    )

    bubble_comparison = pre.merge(
        post,
        on=key_columns,
        how="inner",
        validate="one_to_one",
    )

    bubble_comparison["bubble_percent_change"] = (
        bubble_comparison.apply(
            lambda row: _safe_percent_change(
                row["bubble_value_pre_cp"],
                row["bubble_value_post_cp"],
            ),
            axis=1,
        )
    )

    return result.merge(
        bubble_comparison,
        on=key_columns,
        how="left",
        validate="one_to_one",
    )


# ---------------------------------------------------------------------
# Static hero data
# ---------------------------------------------------------------------
def _build_three_metric_data(
    relationship_data: pd.DataFrame,
) -> pd.DataFrame:
    key_columns = [
        "geography_level",
        "geography_id",
        "geography_name",
        "temporal_bucket",
        "period",
    ]

    metadata_columns = [
        "borough",
        "cbd_spatial_category",
    ]

    source = relationship_data[
        relationship_data["metric"].isin(
            [
                TAXI_METRIC,
                FHVHV_METRIC,
                SUBWAY_METRIC,
            ]
        )
    ].copy()

    if source.empty:
        return pd.DataFrame()

    metadata = (
        source[
            key_columns
            + metadata_columns
        ]
        .drop_duplicates(
            subset=key_columns
        )
        .copy()
    )

    metric_frames: list[pd.DataFrame] = []

    for metric in [
        TAXI_METRIC,
        FHVHV_METRIC,
        SUBWAY_METRIC,
    ]:
        metric_frame = (
            source[
                source["metric"].eq(metric)
            ][
                key_columns
                + [
                    "metric_value",
                ]
            ]
            .copy()
            .rename(
                columns={
                    "metric_value": metric,
                }
            )
        )

        metric_frames.append(metric_frame)

    wide = metric_frames[0]

    for metric_frame in metric_frames[1:]:
        wide = wide.merge(
            metric_frame,
            on=key_columns,
            how="inner",
            validate="one_to_one",
        )

    wide = wide.merge(
        metadata,
        on=key_columns,
        how="left",
        validate="one_to_one",
    )

    wide = wide[
        wide[TAXI_METRIC].gt(0)
        & wide[FHVHV_METRIC].gt(0)
        & wide[SUBWAY_METRIC].gt(0)
    ].copy()

    return wide.reset_index(drop=True)


def _build_three_metric_comparison(
    three_metric_data: pd.DataFrame,
) -> pd.DataFrame:
    if three_metric_data.empty:
        return pd.DataFrame()

    key_columns = [
        "geography_level",
        "geography_id",
        "geography_name",
        "temporal_bucket",
    ]

    metadata = (
        three_metric_data[
            key_columns
            + [
                "borough",
                "cbd_spatial_category",
            ]
        ]
        .drop_duplicates(
            subset=key_columns
        )
        .copy()
    )

    value_columns = [
        TAXI_METRIC,
        FHVHV_METRIC,
        SUBWAY_METRIC,
    ]

    pre = (
        three_metric_data[
            three_metric_data["period"].eq(
                "Pre-CP"
            )
        ][
            key_columns + value_columns
        ]
        .copy()
        .rename(
            columns={
                column: f"{column}_pre_cp"
                for column in value_columns
            }
        )
    )

    post = (
        three_metric_data[
            three_metric_data["period"].eq(
                "Post-CP"
            )
        ][
            key_columns + value_columns
        ]
        .copy()
        .rename(
            columns={
                column: f"{column}_post_cp"
                for column in value_columns
            }
        )
    )

    wide = pre.merge(
        post,
        on=key_columns,
        how="inner",
        validate="one_to_one",
    )

    wide = wide.merge(
        metadata,
        on=key_columns,
        how="left",
        validate="one_to_one",
    )

    wide["taxi_percent_change"] = wide.apply(
        lambda row: _safe_percent_change(
            row[f"{TAXI_METRIC}_pre_cp"],
            row[f"{TAXI_METRIC}_post_cp"],
        ),
        axis=1,
    )

    wide["fhvhv_percent_change"] = wide.apply(
        lambda row: _safe_percent_change(
            row[f"{FHVHV_METRIC}_pre_cp"],
            row[f"{FHVHV_METRIC}_post_cp"],
        ),
        axis=1,
    )

    wide["subway_percent_change"] = wide.apply(
        lambda row: _safe_percent_change(
            row[f"{SUBWAY_METRIC}_pre_cp"],
            row[f"{SUBWAY_METRIC}_post_cp"],
        ),
        axis=1,
    )

    return (
        wide.sort_values(
            "geography_name"
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Static hero chart
# ---------------------------------------------------------------------
def build_static_hero(
    three_metric_data: pd.DataFrame,
    comparison: pd.DataFrame,
) -> go.Figure:
    plot_data = three_metric_data.copy()

    plot_data["geo_policy"] = (
        plot_data["cbd_spatial_category"]
        .map(_normalize_geo_policy)
    )

    plot_data["color_group"] = (
        plot_data["geo_policy"]
    )

    plot_data["display_taxi"] = (
        plot_data[TAXI_METRIC]
        .map(_format_number)
    )

    plot_data["display_fhvhv"] = (
        plot_data[FHVHV_METRIC]
        .map(_format_number)
    )

    plot_data["display_subway"] = (
        plot_data[SUBWAY_METRIC]
        .map(_format_number)
    )

    comparison_data = comparison.copy()

    comparison_data["geo_policy"] = (
        comparison_data[
            "cbd_spatial_category"
        ]
        .map(_normalize_geo_policy)
    )

    fig = go.Figure()

    for _, row in comparison_data.iterrows():
        required = [
            row[f"{TAXI_METRIC}_pre_cp"],
            row[f"{TAXI_METRIC}_post_cp"],
            row[f"{FHVHV_METRIC}_pre_cp"],
            row[f"{FHVHV_METRIC}_post_cp"],
        ]

        if any(
            pd.isna(value)
            for value in required
        ):
            continue

        color = GEO_POLICY_COLORS.get(
            row["geo_policy"],
            "#8A8A8A",
        )

        fig.add_trace(
            go.Scatter(
                x=[
                    row[f"{TAXI_METRIC}_pre_cp"],
                    row[f"{TAXI_METRIC}_post_cp"],
                ],
                y=[
                    row[f"{FHVHV_METRIC}_pre_cp"],
                    row[f"{FHVHV_METRIC}_post_cp"],
                ],
                mode="lines",
                line={
                    "color": color,
                    "width": 3,
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

    sizeref = _calculate_sizeref(
        plot_data[SUBWAY_METRIC],
        maximum_marker_size=95,
    )

    for geo_policy in GEO_POLICY_ORDER:
        category_data = plot_data[
            plot_data["color_group"].eq(
                geo_policy
            )
        ].copy()

        if category_data.empty:
            continue

        color = GEO_POLICY_COLORS.get(
            geo_policy,
            "#8A8A8A",
        )

        for period in [
            "Pre-CP",
            "Post-CP",
        ]:
            period_data = category_data[
                category_data["period"].eq(
                    period
                )
            ].copy()

            if period_data.empty:
                continue

            period_data["label"] = np.where(
                period == "Post-CP",
                period_data["geography_name"],
                "",
            )

            customdata = np.column_stack(
                [
                    period_data["geography_name"],
                    period_data["period"],
                    period_data["display_taxi"],
                    period_data["display_fhvhv"],
                    period_data["display_subway"],
                ]
            )

            fig.add_trace(
                go.Scatter(
                    x=period_data[TAXI_METRIC],
                    y=period_data[FHVHV_METRIC],
                    mode="markers+text",
                    marker={
                        "size": period_data[
                            SUBWAY_METRIC
                        ],
                        "sizemode": "area",
                        "sizeref": sizeref,
                        "sizemin": 16,
                        "color": color,
                        "symbol": (
                            "circle-open"
                            if period == "Pre-CP"
                            else "circle"
                        ),
                        "opacity": (
                            0.55
                            if period == "Pre-CP"
                            else 0.88
                        ),
                        "line": {
                            "color": color,
                            "width": 2,
                        },
                    },
                    text=period_data["label"],
                    textposition="top center",
                    textfont={
                        "size": 12,
                    },
                    customdata=customdata,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Period: %{customdata[1]}<br>"
                        "Taxi trips: %{customdata[2]}<br>"
                        "FHVHV trips: %{customdata[3]}<br>"
                        "Subway ridership: %{customdata[4]}<br>"
                        "Bubble area represents subway ridership"
                        "<extra></extra>"
                    ),
                    name=geo_policy,
                    legendgroup=geo_policy,
                    showlegend=(
                        period == "Post-CP"
                    ),
                    cliponaxis=False,
                )
            )

    fig.update_xaxes(
        title_text=(
            "Taxi trips · daily average · log scale"
        ),
        type="log",
        range=_safe_log_range(
            plot_data[TAXI_METRIC]
        ),
        showgrid=True,
        gridcolor="rgba(0,109,119,0.10)",
        zeroline=False,
    )

    fig.update_yaxes(
        title_text=(
            "FHVHV trips · daily average · log scale"
        ),
        type="log",
        range=_safe_log_range(
            plot_data[FHVHV_METRIC]
        ),
        showgrid=True,
        gridcolor="rgba(0,109,119,0.10)",
        zeroline=False,
    )

    fig = _apply_chart_branding(fig)

    fig.update_layout(
        height=690,
        margin={
            "l": 10,
            "r": 45,
            "t": 20,
            "b": 125,
        },
        legend={
            "title": {
                "text": "",
            },
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.16,
            "yanchor": "top",
        },
        hovermode="closest",
    )

    return fig


# ---------------------------------------------------------------------
# Explorer filtering
# ---------------------------------------------------------------------
def _filter_taxi_zone_pair_data(
    pair_data: pd.DataFrame,
    *,
    filter_scope: str,
    filter_value: str | None,
) -> pd.DataFrame:
    if filter_scope == "All Taxi Zones":
        return pair_data.copy()

    if filter_scope == "Borough":
        return pair_data[
            pair_data["borough"]
            .fillna("")
            .astype(str)
            .eq(str(filter_value))
        ].copy()

    if filter_scope == "Geo-policy group":
        normalized = (
            pair_data["cbd_spatial_category"]
            .map(_normalize_geo_policy)
        )

        return pair_data[
            normalized.eq(
                str(filter_value)
            )
        ].copy()

    if filter_scope == "Mobility regime cluster":
        return pair_data[
            pair_data["mobility_regime_cluster_label"]
            .astype(str)
            .eq(str(filter_value))
        ].copy()

    return pair_data.copy()


def _available_filter_values(
    pair_data: pd.DataFrame,
    *,
    filter_scope: str,
) -> list[str]:
    if filter_scope == "Borough":
        represented = set(
            pair_data["borough"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        ordered = [
            borough
            for borough in BOROUGH_ORDER
            if borough in represented
        ]

        return ordered + sorted(
            represented.difference(ordered)
        )

    if filter_scope == "Geo-policy group":
        normalized = (
            pair_data["cbd_spatial_category"]
            .map(_normalize_geo_policy)
        )

        represented = set(
            normalized
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        ordered = [
            category
            for category in GEO_POLICY_ORDER
            if category in represented
        ]

        return ordered + sorted(
            represented.difference(ordered)
        )

    if filter_scope == "Mobility regime cluster":
        represented = set(
            pd.to_numeric(
                pair_data["mobility_regime_cluster_label"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .tolist()
        )

        ordered = [
            cluster
            for cluster in get_mobility_regime_cluster_options()
            if cluster in represented
        ]

        return ordered + sorted(
            represented.difference(ordered)
        )

    return []


# ---------------------------------------------------------------------
# Interactive relationship chart
# ---------------------------------------------------------------------
def build_interactive_relationship_chart(
    pair_data: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    geography_level: str,
    period_view: str,
    taxi_zone_color: str,
    bubble_metric: str | None,
) -> go.Figure:
    plot_data = _add_color_groups(
        pair_data,
        geography_level=geography_level,
        taxi_zone_color=taxi_zone_color,
    )

    comparison_data = _add_color_groups(
        comparison,
        geography_level=geography_level,
        taxi_zone_color=taxi_zone_color,
    )

    color_map = _get_color_map(
        geography_level=geography_level,
        taxi_zone_color=taxi_zone_color,
    )

    fig = go.Figure()

    if period_view == "Both":
        for _, row in comparison_data.iterrows():
            required = [
                row["x_value_pre_cp"],
                row["x_value_post_cp"],
                row["y_value_pre_cp"],
                row["y_value_post_cp"],
            ]

            if any(
                pd.isna(value)
                for value in required
            ):
                continue

            color = color_map.get(
                str(row["color_group"]),
                "#8A8A8A",
            )

            fig.add_trace(
                go.Scatter(
                    x=[
                        row["x_value_pre_cp"],
                        row["x_value_post_cp"],
                    ],
                    y=[
                        row["y_value_pre_cp"],
                        row["y_value_post_cp"],
                    ],
                    mode="lines",
                    line={
                        "color": (
                            "rgba(70,70,70,0.20)"
                            if geography_level
                            == "Taxi Zone"
                            else color
                        ),
                        "width": (
                            1
                            if geography_level
                            == "Taxi Zone"
                            else 2.5
                        ),
                    },
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    periods_to_show = (
        [
            "Pre-CP",
            "Post-CP",
        ]
        if period_view == "Both"
        else [period_view]
    )

    use_bubble_size = (
        bubble_metric is not None
        and plot_data["bubble_value"]
        .notna()
        .any()
    )

    maximum_marker_size = (
        58
        if geography_level == "Taxi Zone"
        else 88
    )

    sizeref = (
        _calculate_sizeref(
            plot_data["bubble_value"],
            maximum_marker_size=maximum_marker_size,
        )
        if use_bubble_size
        else 1.0
    )

    fixed_marker_size = (
        18
        if geography_level == "Taxi Zone"
        else 34
    )

    for color_group in _get_group_order(
        plot_data,
        geography_level=geography_level,
        taxi_zone_color=taxi_zone_color,
    ):
        category_data = plot_data[
            plot_data["color_group"].eq(
                color_group
            )
        ].copy()

        color = color_map.get(
            color_group,
            "#8A8A8A",
        )

        for period in periods_to_show:
            period_data = category_data[
                category_data["period"].eq(
                    period
                )
            ].copy()

            if period_data.empty:
                continue

            direct_labels = (
                geography_level
                in {
                    "Borough",
                    "Geo-policy group",
                }
            )

            period_data["label"] = np.where(
                direct_labels
                and (
                    period == "Post-CP"
                    or period_view != "Both"
                ),
                period_data["geography_name"],
                "",
            )

            period_data["display_x"] = (
                period_data["x_value"]
                .map(_format_number)
            )

            period_data["display_y"] = (
                period_data["y_value"]
                .map(_format_number)
            )

            period_data["display_bubble"] = (
                period_data["bubble_value"]
                .map(_format_number)
            )

            customdata = np.column_stack(
                [
                    period_data["geography_name"],
                    period_data["borough"],
                    period_data["geo_policy"],
                    period_data["period"],
                    period_data["x_metric_label"],
                    period_data["display_x"],
                    period_data["y_metric_label"],
                    period_data["display_y"],
                    period_data["bubble_metric_label"],
                    period_data["display_bubble"],
                    period_data["color_group"],
                ]
            )

            if use_bubble_size:
                marker = {
                    "size": period_data["bubble_value"],
                    "sizemode": "area",
                    "sizeref": sizeref,
                    "sizemin": 7,
                    "color": color,
                    "symbol": (
                        "circle-open"
                        if period == "Pre-CP"
                        else "circle"
                    ),
                    "opacity": (
                        0.48
                        if period == "Pre-CP"
                        else 0.82
                    ),
                    "line": {
                        "color": color,
                        "width": 1.5,
                    },
                }
            else:
                marker = {
                    "size": fixed_marker_size,
                    "color": color,
                    "symbol": (
                        "circle-open"
                        if period == "Pre-CP"
                        else "circle"
                    ),
                    "opacity": (
                        0.52
                        if period == "Pre-CP"
                        else 0.88
                    ),
                    "line": {
                        "color": color,
                        "width": 1.5,
                    },
                }

            fig.add_trace(
                go.Scatter(
                    x=period_data["x_value"],
                    y=period_data["y_value"],
                    mode="markers+text",
                    marker=marker,
                    text=period_data["label"],
                    textposition="top center",
                    textfont={
                        "size": 10,
                    },
                    customdata=customdata,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Borough: %{customdata[1]}<br>"
                        "Policy geography: %{customdata[2]}<br>"
                        "Period: %{customdata[3]}<br>"
                        "%{customdata[4]}: %{customdata[5]}<br>"
                        "%{customdata[6]}: %{customdata[7]}<br>"
                        "%{customdata[8]}: %{customdata[9]}<br>"
                        "Color group: %{customdata[10]}"
                        "<extra></extra>"
                    ),
                    name=f"{color_group} · {period}",
                    legendgroup=color_group,
                    showlegend=True,
                    cliponaxis=False,
                )
            )

    x_label = str(
        plot_data["x_metric_label"].iloc[0]
    )

    y_label = str(
        plot_data["y_metric_label"].iloc[0]
    )

    fig.update_xaxes(
        title_text=(
            f"{x_label} · daily average · log scale"
        ),
        type="log",
        range=_safe_log_range(
            plot_data["x_value"]
        ),
        showgrid=True,
        gridcolor="rgba(0,109,119,0.10)",
        zeroline=False,
    )

    fig.update_yaxes(
        title_text=(
            f"{y_label} · daily average · log scale"
        ),
        type="log",
        range=_safe_log_range(
            plot_data["y_value"]
        ),
        showgrid=True,
        gridcolor="rgba(0,109,119,0.10)",
        zeroline=False,
    )

    fig = _apply_chart_branding(fig)

    fig.update_layout(
        height=720,
        margin={
            "l": 10,
            "r": 45,
            "t": 20,
            "b": 130,
        },
        legend={
            "title": {
                "text": "",
            },
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.16,
            "yanchor": "top",
        },
        hovermode="closest",
    )

    return fig




# ---------------------------------------------------------------------
# Temporal-bucket relationship heatmap
# ---------------------------------------------------------------------
HEATMAP_BUCKETS = [
    bucket
    for bucket in TEMPORAL_BUCKET_OPTIONS
    if bucket != ALL_TEMPORAL_BUCKETS_LABEL
]


def _build_temporal_relationship_matrix(
    *,
    x_metric: str,
    y_metric: str,
    geography_level: str,
    display_mode: str,
    filter_scope: str,
    filter_value: str | None,
) -> pd.DataFrame:
    """Calculate pre/post cross-sectional correlations for every time bucket."""
    rows: list[dict[str, object]] = []

    for temporal_bucket in HEATMAP_BUCKETS:
        relationship_data = build_relationship_long_data(
            metrics=[x_metric, y_metric],
            temporal_bucket=temporal_bucket,
            aggregation_level=geography_level,
            apply_reliability_thresholds=False,
        )

        pair_data = build_metric_pair_data(
            relationship_data,
            x_metric=x_metric,
            y_metric=y_metric,
        )

        if (
            display_mode == "Taxi Zones"
            and filter_scope != "All Taxi Zones"
        ):
            pair_data = _filter_taxi_zone_pair_data(
                pair_data,
                filter_scope=filter_scope,
                filter_value=filter_value,
            )

        correlations = calculate_pair_correlations(
            pair_data
        )

        pre_row = correlations[
            correlations["period"].eq("Pre-CP")
        ]
        post_row = correlations[
            correlations["period"].eq("Post-CP")
        ]

        pre_correlation = (
            float(pre_row.iloc[0]["pearson_correlation"])
            if not pre_row.empty
            else np.nan
        )
        post_correlation = (
            float(post_row.iloc[0]["pearson_correlation"])
            if not post_row.empty
            else np.nan
        )

        pre_count = (
            int(pre_row.iloc[0]["observation_count"])
            if (
                not pre_row.empty
                and "observation_count" in pre_row.columns
                and pd.notna(pre_row.iloc[0]["observation_count"])
            )
            else int(
                pair_data[
                    pair_data["period"].eq("Pre-CP")
                ][["x_value", "y_value"]]
                .dropna()
                .shape[0]
            )
        )

        post_count = (
            int(post_row.iloc[0]["observation_count"])
            if (
                not post_row.empty
                and "observation_count" in post_row.columns
                and pd.notna(post_row.iloc[0]["observation_count"])
            )
            else int(
                pair_data[
                    pair_data["period"].eq("Post-CP")
                ][["x_value", "y_value"]]
                .dropna()
                .shape[0]
            )
        )

        change = (
            post_correlation - pre_correlation
            if pd.notna(pre_correlation)
            and pd.notna(post_correlation)
            else np.nan
        )

        rows.append(
            {
                "temporal_bucket": temporal_bucket,
                "temporal_bucket_label": TEMPORAL_BUCKET_LABELS[
                    temporal_bucket
                ],
                "pre_correlation": pre_correlation,
                "post_correlation": post_correlation,
                "correlation_change": change,
                "pre_observation_count": pre_count,
                "post_observation_count": post_count,
            }
        )

    return pd.DataFrame(rows)


def build_temporal_relationship_heatmap(
    matrix_df: pd.DataFrame,
) -> go.Figure:
    """Build the compact pre/post/change correlation matrix."""
    if matrix_df.empty:
        return _apply_chart_branding(go.Figure())

    columns = [
        "Pre-CP",
        "Post-CP",
        "Post − pre",
    ]

    z_values = matrix_df[
        [
            "pre_correlation",
            "post_correlation",
            "correlation_change",
        ]
    ].to_numpy(dtype=float)

    text_values = np.empty(
        z_values.shape,
        dtype=object,
    )

    for row_index in range(z_values.shape[0]):
        for column_index in range(z_values.shape[1]):
            value = z_values[row_index, column_index]
            text_values[row_index, column_index] = (
                f"{value:+.2f}"
                if np.isfinite(value)
                else "—"
            )

    customdata = np.empty(
        z_values.shape + (2,),
        dtype=object,
    )

    for row_index, row in matrix_df.reset_index(drop=True).iterrows():
        customdata[row_index, 0] = [
            row["pre_observation_count"],
            "Pre-CP",
        ]
        customdata[row_index, 1] = [
            row["post_observation_count"],
            "Post-CP",
        ]
        customdata[row_index, 2] = [
            min(
                row["pre_observation_count"],
                row["post_observation_count"],
            ),
            "Post minus pre",
        ]

    finite_values = z_values[np.isfinite(z_values)]
    color_bound = (
        max(1.0, float(np.abs(finite_values).max()))
        if finite_values.size
        else 1.0
    )

    fig = go.Figure(
        go.Heatmap(
            z=z_values,
            x=columns,
            y=matrix_df["temporal_bucket_label"],
            zmin=-color_bound,
            zmax=color_bound,
            zmid=0,
            colorscale=[
                [0.0, BRAND_COLORS["terracotta"]],
                [0.5, BRAND_COLORS["ice"]],
                [1.0, BRAND_COLORS["dark_teal"]],
            ],
            text=text_values,
            texttemplate="%{text}",
            textfont={"size": 13},
            customdata=customdata,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "%{customdata[1]}: %{z:+.3f}<br>"
                "Paired geographies: %{customdata[0]:,}"
                "<extra></extra>"
            ),
            colorbar={
                "title": {
                    "text": "Correlation<br>or change",
                },
                "tickformat": "+.1f",
            },
            hoverongaps=False,
        )
    )

    fig.update_xaxes(
        title_text="",
        side="top",
        showgrid=False,
    )
    fig.update_yaxes(
        title_text="",
        autorange="reversed",
        showgrid=False,
        automargin=True,
    )

    fig = _apply_chart_branding(fig)

    fig.update_layout(
        height=610,
        margin={
            "l": 10,
            "r": 45,
            "t": 45,
            "b": 35,
        },
    )

    return fig


def _build_heatmap_summary(
    matrix_df: pd.DataFrame,
) -> dict[str, object]:
    """Summarize the strongest temporal-bucket relationship patterns."""
    valid_post = matrix_df[
        matrix_df["post_correlation"].notna()
    ].copy()

    valid_change = matrix_df[
        matrix_df["correlation_change"].notna()
    ].copy()

    sign_flips = matrix_df[
        matrix_df[
            [
                "pre_correlation",
                "post_correlation",
            ]
        ]
        .notna()
        .all(axis=1)
        & (
            np.sign(matrix_df["pre_correlation"])
            != np.sign(matrix_df["post_correlation"])
        )
    ].copy()

    strongest_post = (
        valid_post.loc[
            valid_post["post_correlation"].abs().idxmax()
        ]
        if not valid_post.empty
        else None
    )

    largest_shift = (
        valid_change.loc[
            valid_change["correlation_change"].abs().idxmax()
        ]
        if not valid_change.empty
        else None
    )

    return {
        "strongest_post": strongest_post,
        "largest_shift": largest_shift,
        "sign_flip_count": int(len(sign_flips)),
        "valid_bucket_count": int(len(valid_change)),
    }


def _render_heatmap_summary_cards(
    summary: dict[str, object],
) -> None:
    strongest_post = summary["strongest_post"]
    largest_shift = summary["largest_shift"]

    card1, card2, card3 = st.columns(3)

    card1.metric(
        "Strongest post-CP relationship",
        (
            f"{strongest_post['post_correlation']:+.3f}"
            if strongest_post is not None
            else "Unavailable"
        ),
        (
            str(strongest_post["temporal_bucket_label"])
            if strongest_post is not None
            else None
        ),
    )

    card2.metric(
        "Largest pre/post shift",
        (
            f"{largest_shift['correlation_change']:+.3f}"
            if largest_shift is not None
            else "Unavailable"
        ),
        (
            str(largest_shift["temporal_bucket_label"])
            if largest_shift is not None
            else None
        ),
    )

    card3.metric(
        "Buckets that changed sign",
        f"{summary['sign_flip_count']} of {summary['valid_bucket_count']}",
    )


def _build_heatmap_insight(
    matrix_df: pd.DataFrame,
    *,
    x_label: str,
    y_label: str,
    geography_context: str,
) -> str:
    """Build a dynamic takeaway for the temporal relationship matrix."""
    valid = matrix_df[
        matrix_df[
            [
                "pre_correlation",
                "post_correlation",
                "correlation_change",
            ]
        ]
        .notna()
        .all(axis=1)
    ].copy()

    if valid.empty:
        return (
            "There were too few paired geographies to compare the selected "
            "relationship across temporal buckets."
        )

    strongest_positive = valid.loc[
        valid["post_correlation"].idxmax()
    ]
    strongest_inverse = valid.loc[
        valid["post_correlation"].idxmin()
    ]
    largest_shift = valid.loc[
        valid["correlation_change"].abs().idxmax()
    ]

    sign_flip_count = int(
        (
            np.sign(valid["pre_correlation"])
            != np.sign(valid["post_correlation"])
        ).sum()
    )

    post_spread = (
        float(valid["post_correlation"].max())
        - float(valid["post_correlation"].min())
    )

    if post_spread >= 0.50 or sign_flip_count >= 2:
        consistency_sentence = (
            "The relationship is strongly time-dependent, so a single overall "
            "correlation would hide meaningful differences across the week."
        )
    elif post_spread >= 0.25 or sign_flip_count == 1:
        consistency_sentence = (
            "The relationship varies by time of week, although most buckets "
            "retain a broadly similar direction."
        )
    else:
        consistency_sentence = (
            "The relationship is comparatively consistent across temporal "
            "buckets."
        )

    return (
        f"For **{x_label}** and **{y_label}** across **{geography_context}**, "
        f"the strongest positive post-CP relationship appeared during "
        f"**{strongest_positive['temporal_bucket_label']}** "
        f"(**{strongest_positive['post_correlation']:+.3f}**), while the most "
        f"inverse post-CP relationship appeared during "
        f"**{strongest_inverse['temporal_bucket_label']}** "
        f"(**{strongest_inverse['post_correlation']:+.3f}**). The largest "
        f"pre/post shift occurred during "
        f"**{largest_shift['temporal_bucket_label']}** "
        f"(**{largest_shift['correlation_change']:+.3f}**). "
        f"{consistency_sentence}"
    )


# ---------------------------------------------------------------------
# Dynamic summaries
# ---------------------------------------------------------------------
def _build_hero_summary(
    comparison: pd.DataFrame,
) -> dict[str, object]:
    if comparison.empty:
        return {
            "groups": 0,
            "taxi_up": 0,
            "fhvhv_up": 0,
            "subway_up": 0,
            "takeaway": (
                "No complete policy-geography comparison was available."
            ),
        }

    taxi_up = int(
        comparison["taxi_percent_change"]
        .gt(0)
        .sum()
    )

    fhvhv_up = int(
        comparison["fhvhv_percent_change"]
        .gt(0)
        .sum()
    )

    subway_up = int(
        comparison["subway_percent_change"]
        .gt(0)
        .sum()
    )

    groups = int(
        comparison["geography_id"]
        .nunique()
    )

    shared_growth = comparison[
        comparison["taxi_percent_change"].gt(0)
        & comparison["fhvhv_percent_change"].gt(0)
        & comparison["subway_percent_change"].gt(0)
    ]

    if len(shared_growth) == groups and groups:
        takeaway = (
            "All displayed policy geographies recorded increases in "
            "Taxi Trips, FHVHV Trips, and Subway Ridership. The chart "
            "therefore points to broad shared growth rather than a simple "
            "shift from one mode to another."
        )
    elif not shared_growth.empty:
        names = ", ".join(
            shared_growth["geography_name"]
            .astype(str)
            .tolist()
        )

        takeaway = (
            f"Shared growth across all three measures appears in "
            f"**{len(shared_growth)} of {groups} policy geographies**"
            f" ({names}). Other groups diverged on at least one mode, "
            "showing that the post-CP pattern was not uniform."
        )
    else:
        takeaway = (
            "No policy geography increased across all three measures. "
            "The post-CP pattern is therefore better understood as "
            "mode-specific divergence than broad shared growth."
        )

    return {
        "groups": groups,
        "taxi_up": taxi_up,
        "fhvhv_up": fhvhv_up,
        "subway_up": subway_up,
        "takeaway": takeaway,
    }


def _build_explorer_insight(
    comparison: pd.DataFrame,
    correlations: pd.DataFrame,
    *,
    x_label: str,
    y_label: str,
    geography_label: str,
    bubble_label: str,
    bubble_metric: str | None,
) -> str:
    if comparison.empty:
        return (
            "No complete pre/post comparison was available "
            "for the selected relationship."
        )

    valid = comparison[
        comparison[
            [
                "x_percent_change",
                "y_percent_change",
            ]
        ]
        .notna()
        .all(axis=1)
    ].copy()

    if valid.empty:
        return (
            "No complete pre/post movement pairs were available "
            "for the selected relationship."
        )

    moved_together_count = int(
        valid["moved_together"]
        .fillna(False)
        .sum()
    )

    total_count = len(valid)

    together_share = (
        moved_together_count
        / total_count
        if total_count
        else np.nan
    )

    pre_correlation = _get_period_correlation(
        correlations,
        period="Pre-CP",
    )

    post_correlation = _get_period_correlation(
        correlations,
        period="Post-CP",
    )

    if (
        pd.notna(pre_correlation)
        and pd.notna(post_correlation)
    ):
        if np.sign(pre_correlation) != np.sign(post_correlation):
            correlation_phrase = "changed direction"
        elif abs(post_correlation) > abs(pre_correlation):
            correlation_phrase = "became stronger in the same direction"
        elif abs(post_correlation) < abs(pre_correlation):
            correlation_phrase = "became weaker in the same direction"
        else:
            correlation_phrase = "kept the same strength"

        correlation_sentence = (
            f"The Pearson relationship {correlation_phrase}, "
            f"moving from **{pre_correlation:+.3f}** pre-CP to "
            f"**{post_correlation:+.3f}** post-CP."
        )
    else:
        correlation_sentence = (
            "There were too few observations to calculate "
            "a stable cross-sectional correlation."
        )

    bubble_sentence = ""

    if (
        bubble_metric is not None
        and "bubble_percent_change" in valid.columns
    ):
        bubble_valid = valid[
            valid["bubble_percent_change"].notna()
        ]

        if not bubble_valid.empty:
            bubble_increased = int(
                bubble_valid[
                    "bubble_percent_change"
                ]
                .gt(0)
                .sum()
            )

            bubble_share = (
                bubble_increased
                / len(bubble_valid)
            )

            bubble_sentence = (
                f" **{bubble_label} increased in "
                f"{bubble_share:.1%}** of complete geographies."
            )

    return (
        f"Across **{total_count} {geography_label}**, "
        f"**{together_share:.1%}** moved in the same direction for "
        f"**{x_label}** and **{y_label}**. "
        f"{correlation_sentence}"
        f"{bubble_sentence}"
    )


def _build_detail_table(
    comparison: pd.DataFrame,
    *,
    x_label: str,
    y_label: str,
    bubble_label: str,
    bubble_metric: str | None,
) -> pd.DataFrame:
    columns = [
        "geography_name",
        "borough",
        "cbd_spatial_category",
        "x_value_pre_cp",
        "x_value_post_cp",
        "x_percent_change",
        "y_value_pre_cp",
        "y_value_post_cp",
        "y_percent_change",
        "movement_relationship",
    ]

    if bubble_metric is not None:
        columns.extend(
            [
                "bubble_value_pre_cp",
                "bubble_value_post_cp",
                "bubble_percent_change",
            ]
        )

    display = comparison[
        columns
    ].copy()

    display["cbd_spatial_category"] = (
        display["cbd_spatial_category"]
        .map(_normalize_geo_policy)
    )

    movement_components = [
        display["x_percent_change"].abs(),
        display["y_percent_change"].abs(),
    ]

    if bubble_metric is not None:
        movement_components.append(
            display[
                "bubble_percent_change"
            ].abs()
        )

    display["_combined_movement"] = (
        pd.concat(
            movement_components,
            axis=1,
        )
        .sum(
            axis=1,
            min_count=1,
        )
    )

    display = (
        display.sort_values(
            [
                "_combined_movement",
                "geography_name",
            ],
            ascending=[
                False,
                True,
            ],
            na_position="last",
        )
        .drop(
            columns="_combined_movement"
        )
    )

    rename_map = {
        "geography_name": "Geography",
        "borough": "Borough",
        "cbd_spatial_category": "Policy geography",
        "x_value_pre_cp": f"{x_label} · Pre-CP",
        "x_value_post_cp": f"{x_label} · Post-CP",
        "x_percent_change": f"{x_label} · Change",
        "y_value_pre_cp": f"{y_label} · Pre-CP",
        "y_value_post_cp": f"{y_label} · Post-CP",
        "y_percent_change": f"{y_label} · Change",
        "movement_relationship": "X/Y movement relationship",
    }

    if bubble_metric is not None:
        rename_map.update(
            {
                "bubble_value_pre_cp": (
                    f"{bubble_label} · Pre-CP"
                ),
                "bubble_value_post_cp": (
                    f"{bubble_label} · Post-CP"
                ),
                "bubble_percent_change": (
                    f"{bubble_label} · Change"
                ),
            }
        )

    return display.rename(
        columns=rename_map
    )


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------
st.caption("MODE RELATIONSHIPS")
st.title("Do modes move together?")

st.write(
    "Two mobility modes can both increase citywide without changing in the same places "
    "or at the same times. This page looks beyond each mode on its own to ask whether "
    "mobility measures moved together across NYC—and whether those relationships looked "
    "different before and after congestion pricing began."
)


# ---------------------------------------------------------------------
# Static hero
# ---------------------------------------------------------------------
st.header(
    "How demand modes moved across policy geographies"
)

st.write(
    "The opening view follows three demand measures at once. **Taxi Trips** set the "
    "horizontal position, **FHVHV Trips** set the vertical position, and bubble area "
    "represents **Subway Ridership**. Each policy geography moves from an open pre-CP "
    "marker to a filled post-CP marker, so both direction and relative movement are visible."
)

with st.spinner(
    "Preparing the multimodal hero..."
):
    hero_data = build_relationship_long_data(
        metrics=[
            TAXI_METRIC,
            FHVHV_METRIC,
            SUBWAY_METRIC,
        ],
        temporal_bucket=(
            ALL_TEMPORAL_BUCKETS_LABEL
        ),
        aggregation_level="Geo-policy group",
        apply_reliability_thresholds=False,
    )

    hero_three_metric_data = (
        _build_three_metric_data(
            hero_data
        )
    )

    hero_comparison = (
        _build_three_metric_comparison(
            hero_three_metric_data
        )
    )

if (
    hero_three_metric_data.empty
    or hero_comparison.empty
):
    st.warning(
        "The multimodal policy-geography hero could not be produced."
    )
else:
    hero_summary = _build_hero_summary(
        hero_comparison
    )

    hero_card1, hero_card2, hero_card3, hero_card4 = (
        st.columns(4)
    )

    hero_card1.metric(
        "Policy geographies",
        f"{hero_summary['groups']:,}",
    )

    hero_card2.metric(
        "Taxi Trips increased",
        (
            f"{hero_summary['taxi_up']} of "
            f"{hero_summary['groups']}"
        ),
    )

    hero_card3.metric(
        "FHVHV Trips increased",
        (
            f"{hero_summary['fhvhv_up']} of "
            f"{hero_summary['groups']}"
        ),
    )

    hero_card4.metric(
        "Subway Ridership increased",
        (
            f"{hero_summary['subway_up']} of "
            f"{hero_summary['groups']}"
        ),
    )

    hero_fig = build_static_hero(
        hero_three_metric_data,
        hero_comparison,
    )

    st.plotly_chart(
        hero_fig,
        width="stretch",
        config={
            "displayModeBar": False,
            "responsive": True,
        },
        key="raw05_static_hero",
    )

    render_chart_insight(hero_summary["takeaway"])

    st.caption(
        "Bubble area—not radius—is proportional to Subway Ridership. "
        "Open markers show pre-CP values; filled markers show post-CP "
        "values. Axes use logarithmic scales."
    )

    st.caption(
        "Subway Ridership does not cover Staten Island. When Subway "
        "Ridership is required as an axis or bubble measure, Staten "
        "Island observations are therefore unavailable."
    )


# ---------------------------------------------------------------------
# Interactive explorer
# ---------------------------------------------------------------------
with exploration_section(
    key="raw05_exploration_area",
    title="Explore multimodal relationships",
    description=(
        "Choose two mobility measures and a geographic frame, then compare "
        "relationships across places or across the ordered time-of-week buckets."
    ),
):
    if "raw05_saved_view" not in st.session_state:
        st.session_state[
            "raw05_saved_view"
        ] = "Multimodal demand landscape"

    saved_view = st.selectbox(
        "Start with a saved configuration",
        options=SAVED_VIEW_OPTIONS,
        help=(
            "Saved configurations provide curated starting points for the "
            "across-geographies view. Changing any shared control switches "
            "the selection to Custom."
        ),
        key="raw05_saved_view",
    )

    if (
        saved_view != "Custom"
        and st.session_state.get(
            "_raw05_applied_saved_view"
        )
        != saved_view
    ):
        _apply_saved_view(
            saved_view
        )

        st.session_state[
            "_raw05_applied_saved_view"
        ] = saved_view

        st.rerun()

    if saved_view == "Custom":
        st.caption(
            "Custom view · adjust any measure or geography control."
        )
    else:
        st.caption(
            SAVED_VIEWS[saved_view][
                "description"
            ]
        )

    st.markdown("**Shared measures**")

    measure_col1, measure_col2 = st.columns(2)

    with measure_col1:
        x_metric = st.selectbox(
            "X measure",
            options=CORE_METRICS,
            format_func=_metric_display_label,
            key="raw05_x_metric",
            on_change=_mark_saved_view_custom,
        )

    with measure_col2:
        y_options = [
            metric
            for metric in CORE_METRICS
            if metric != x_metric
        ]

        if (
            st.session_state.get(
                "raw05_y_metric"
            )
            not in y_options
        ):
            st.session_state[
                "raw05_y_metric"
            ] = (
                FHVHV_METRIC
                if FHVHV_METRIC in y_options
                else y_options[0]
            )

        y_metric = st.selectbox(
            "Y measure",
            options=y_options,
            format_func=_metric_display_label,
            key="raw05_y_metric",
            on_change=_mark_saved_view_custom,
        )

    st.markdown("**Shared geography**")

    geography_col1, geography_col2 = st.columns(2)

    with geography_col1:
        display_mode = st.selectbox(
            "Geography display",
            options=DISPLAY_MODE_OPTIONS,
            help=(
                "Taxi Zones preserves local observations. Aggregated "
                "geographies combines the underlying zones into Borough "
                "or policy-geography summaries."
            ),
            key="raw05_display_mode",
            on_change=_mark_saved_view_custom,
        )

    if display_mode == "Taxi Zones":
        geography_level = "Taxi Zone"

        with geography_col2:
            filter_scope = st.selectbox(
                "Filter Taxi Zones",
                options=TAXI_ZONE_FILTER_OPTIONS,
                format_func=_format_geography_term,
                key="raw05_taxi_zone_filter_scope",
                on_change=_mark_saved_view_custom,
            )

        aggregate_by = None
    else:
        filter_scope = "All Taxi Zones"
        taxi_zone_color = "Geo-policy group"

        with geography_col2:
            aggregate_by = st.selectbox(
                "Aggregate geographies by",
                options=AGGREGATION_OPTIONS,
                format_func=_format_geography_term,
                key="raw05_aggregate_by",
                on_change=_mark_saved_view_custom,
            )

        geography_level = aggregate_by

    with st.spinner(
        "Preparing the shared geography sample..."
    ):
        shared_relationship_data = build_relationship_long_data(
            metrics=[x_metric, y_metric],
            temporal_bucket=ALL_TEMPORAL_BUCKETS_LABEL,
            aggregation_level=geography_level,
            apply_reliability_thresholds=False,
        )

        shared_pair_data = build_metric_pair_data(
            shared_relationship_data,
            x_metric=x_metric,
            y_metric=y_metric,
        )

        if display_mode == "Taxi Zones":
            shared_pair_data = attach_mobility_regime_cluster_context(
                shared_pair_data,
                assignment_period="post_cp",
            )

    filter_value: str | None = None

    if (
        display_mode == "Taxi Zones"
        and filter_scope != "All Taxi Zones"
    ):
        filter_values = _available_filter_values(
            shared_pair_data,
            filter_scope=filter_scope,
        )

        if filter_values:
            filter_label = (
                "Borough"
                if filter_scope == "Borough"
                else "Policy geography"
                if filter_scope == "Geo-policy group"
                else "Mobility environment"
            )

            filter_value = st.selectbox(
                filter_label,
                options=filter_values,
                index=0,
                key="raw05_taxi_zone_filter_value",
                on_change=_mark_saved_view_custom,
                format_func=(
                    (lambda value: value)
                    if filter_scope in {
                        "Borough",
                        "Geo-policy group",
                    }
                    else format_mobility_regime_cluster_label
                ),
            )
        else:
            st.warning(
                "No geographic values are available for this combination "
                "of measures."
            )

    metric_labels = _metric_label_lookup(
        shared_relationship_data
    )

    x_label = metric_labels.get(
        x_metric,
        _metric_display_label(x_metric),
    )

    y_label = metric_labels.get(
        y_metric,
        _metric_display_label(y_metric),
    )

    if display_mode == "Taxi Zones":
        geography_context = (
            "all eligible Taxi Zones"
            if filter_scope == "All Taxi Zones"
            else (
                f"{_format_geography_term(filter_scope)}: {filter_value}"
                if filter_scope != "Mobility regime cluster"
                else f"Mobility environment: {format_mobility_regime_cluster_label(filter_value)}"
            )
        )
        geography_label = "Taxi Zone observations"
    else:
        geography_context = (
            f"geographies aggregated by {_format_geography_term(aggregate_by).lower()}"
        )
        geography_label = f"{_format_geography_term(aggregate_by).lower()} groups"

    across_tab, temporal_tab = st.tabs(
        [
            "Across geographies",
            "By time of week",
        ]
    )

    with across_tab:
        st.markdown(
            "Compare where the selected modes sit across the city and how each "
            "geography moved from the pre-CP to post-CP period."
        )

        st.markdown("**Chart-specific measures**")

        bubble_metric = st.selectbox(
            "Bubble-area measure",
            options=[
                None,
                *[
                    metric
                    for metric in CORE_METRICS
                    if metric not in {
                        x_metric,
                        y_metric,
                    }
                ],
            ],
            format_func=_metric_display_label,
            help=(
                "Bubble area adds a third quantitative measure. "
                "Select None to use fixed-size markers."
            ),
            key="raw05_bubble_metric",
            on_change=_mark_saved_view_custom,
        )

        chart_control1, chart_control2, chart_control3 = st.columns(3)

        with chart_control1:
            if display_mode == "Taxi Zones":
                taxi_zone_color = st.selectbox(
                    "Color Taxi Zones by",
                    options=TAXI_ZONE_COLOR_OPTIONS,
                    format_func=_format_geography_term,
                    key="raw05_taxi_zone_color",
                    on_change=_mark_saved_view_custom,
                )
            else:
                taxi_zone_color = "Geo-policy group"
                st.markdown(
                    f"**Color grouping**  \n{aggregate_by}"
                )

        with chart_control2:
            temporal_bucket = st.selectbox(
                "Time bucket",
                options=TEMPORAL_BUCKET_OPTIONS,
                format_func=lambda value: (
                    TEMPORAL_BUCKET_LABELS[value]
                ),
                key="raw05_temporal_bucket",
                on_change=_mark_saved_view_custom,
            )

        with chart_control3:
            period_view = st.selectbox(
                "Period view",
                options=PERIOD_OPTIONS,
                key="raw05_period_view",
                on_change=_mark_saved_view_custom,
            )

        selected_metrics = [
            x_metric,
            y_metric,
        ]

        if (
            bubble_metric is not None
            and bubble_metric not in selected_metrics
        ):
            selected_metrics.append(
                bubble_metric
            )

        with st.spinner(
            "Updating the geographic relationship view..."
        ):
            explorer_data = build_relationship_long_data(
                metrics=selected_metrics,
                temporal_bucket=temporal_bucket,
                aggregation_level=geography_level,
                apply_reliability_thresholds=False,
            )

            xy_pair_data = build_metric_pair_data(
                explorer_data,
                x_metric=x_metric,
                y_metric=y_metric,
            )

            xy_geography_count = int(
                xy_pair_data["geography_id"]
                .nunique()
            )

            pair_data = _attach_bubble_metric(
                xy_pair_data,
                explorer_data,
                bubble_metric=bubble_metric,
            )

            if display_mode == "Taxi Zones":
                pair_data = attach_mobility_regime_cluster_context(
                    pair_data,
                    assignment_period="post_cp",
                )

        if (
            display_mode == "Taxi Zones"
            and filter_scope != "All Taxi Zones"
        ):
            pair_data = _filter_taxi_zone_pair_data(
                pair_data,
                filter_scope=filter_scope,
                filter_value=filter_value,
            )

        pair_data = pair_data[
            pair_data["x_value"].gt(0)
            & pair_data["y_value"].gt(0)
        ].copy()

        if bubble_metric is not None:
            pair_data = pair_data[
                pair_data["bubble_value"].gt(0)
            ].copy()

        comparison = build_pre_post_pair_comparison(
            pair_data
        )

        comparison = _build_bubble_comparison(
            comparison,
            pair_data,
            bubble_metric=bubble_metric,
        )

        correlations = calculate_pair_correlations(
            pair_data
        )

        if comparison.empty or pair_data.empty:
            st.info(
                "No complete positive observations were available for this "
                "combination of measures and filters."
            )
        else:
            bubble_label = (
                metric_labels.get(
                    bubble_metric,
                    _metric_display_label(
                        bubble_metric
                    ),
                )
                if bubble_metric is not None
                else "Fixed marker size"
            )

            displayed_geography_count = int(
                comparison["geography_id"]
                .nunique()
            )

            pre_correlation = _get_period_correlation(
                correlations,
                period="Pre-CP",
            )

            post_correlation = _get_period_correlation(
                correlations,
                period="Post-CP",
            )

            complete_comparison = comparison.loc[
                comparison[
                    [
                        "x_percent_change",
                        "y_percent_change",
                    ]
                ]
                .notna()
                .all(axis=1)
            ].copy()

            moved_together_share = (
                complete_comparison["moved_together"]
                .fillna(False)
                .mean()
                if not complete_comparison.empty
                else np.nan
            )

            card1, card2, card3, card4 = st.columns(4)

            card1.metric(
                (
                    "Taxi Zones shown"
                    if display_mode == "Taxi Zones"
                    else "Geographic groups"
                ),
                f"{displayed_geography_count:,}",
            )

            card2.metric(
                "Pre-CP correlation",
                _format_correlation(
                    pre_correlation
                ),
            )

            card3.metric(
                "Post-CP correlation",
                _format_correlation(
                    post_correlation
                ),
            )

            card4.metric(
                "Moved together",
                (
                    f"{moved_together_share:.1%}"
                    if pd.notna(
                        moved_together_share
                    )
                    else "Unavailable"
                ),
            )

            color_label = _format_geography_term(
                taxi_zone_color
                if display_mode == "Taxi Zones"
                else aggregate_by
            )

            st.caption(
                f"X: {x_label} · "
                f"Y: {y_label} · "
                f"Bubble area: {bubble_label} · "
                f"{geography_context} · "
                f"Color: {color_label} · "
                f"{TEMPORAL_BUCKET_LABELS[temporal_bucket]} · "
                f"{period_view}"
            )

            if (
                bubble_metric is not None
                and displayed_geography_count
                < xy_geography_count
            ):
                excluded_count = (
                    xy_geography_count
                    - displayed_geography_count
                )

                st.caption(
                    f"Adding {bubble_label} reduced the complete geographic "
                    f"sample by {excluded_count:,} because bubble sizing requires "
                    "a positive value for all three selected measures."
                )

            selected_metric_set = {
                x_metric,
                y_metric,
                bubble_metric,
            }

            if SUBWAY_METRIC in selected_metric_set:
                st.caption(
                    "Subway Ridership does not cover Staten Island, so Staten "
                    "Island cannot appear when this measure is required."
                )

            explorer_takeaway = _build_explorer_insight(
                comparison,
                correlations,
                x_label=x_label,
                y_label=y_label,
                geography_label=geography_label,
                bubble_label=bubble_label,
                bubble_metric=bubble_metric,
            )

            relationship_fig = (
                build_interactive_relationship_chart(
                    pair_data,
                    comparison,
                    geography_level=geography_level,
                    period_view=period_view,
                    taxi_zone_color=taxi_zone_color,
                    bubble_metric=bubble_metric,
                )
            )

            st.plotly_chart(
                relationship_fig,
                width="stretch",
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
                key=(
                    f"raw05_geographic_"
                    f"{x_metric}_{y_metric}_"
                    f"{bubble_metric}_"
                    f"{display_mode}_"
                    f"{geography_level}_"
                    f"{filter_scope}_"
                    f"{filter_value}_"
                    f"{taxi_zone_color}_"
                    f"{temporal_bucket}_"
                    f"{period_view}"
                ),
            )
            render_chart_insight(explorer_takeaway)

            if bubble_metric is not None:
                st.caption(
                    f"Bubble area—not radius—is proportional to {bubble_label}. "
                    "Open markers show pre-CP values; filled markers show post-CP "
                    "values. Logarithmic axes require positive X and Y values."
                )
            else:
                st.caption(
                    "Marker size is fixed. Open markers show pre-CP values; "
                    "filled markers show post-CP values. Logarithmic axes require "
                    "positive X and Y values."
                )

            with st.expander(
                "Inspect the paired pre- and post-CP values",
                expanded=False,
            ):
                st.caption(
                    "Rows are sorted by the largest combined absolute percentage "
                    "movement across the displayed measures."
                )

                detail = _build_detail_table(
                    comparison,
                    x_label=x_label,
                    y_label=y_label,
                    bubble_label=bubble_label,
                    bubble_metric=bubble_metric,
                )

                column_config: dict[
                    str,
                    st.column_config.Column
                ] = {
                    f"{x_label} · Pre-CP": (
                        st.column_config.NumberColumn(
                            format="%,.2f",
                        )
                    ),
                    f"{x_label} · Post-CP": (
                        st.column_config.NumberColumn(
                            format="%,.2f",
                        )
                    ),
                    f"{x_label} · Change": (
                        st.column_config.NumberColumn(
                            format="%+,.1f%%",
                        )
                    ),
                    f"{y_label} · Pre-CP": (
                        st.column_config.NumberColumn(
                            format="%,.2f",
                        )
                    ),
                    f"{y_label} · Post-CP": (
                        st.column_config.NumberColumn(
                            format="%,.2f",
                        )
                    ),
                    f"{y_label} · Change": (
                        st.column_config.NumberColumn(
                            format="%+,.1f%%",
                        )
                    ),
                }

                if bubble_metric is not None:
                    column_config.update(
                        {
                            f"{bubble_label} · Pre-CP": (
                                st.column_config.NumberColumn(
                                    format="%,.2f",
                                )
                            ),
                            f"{bubble_label} · Post-CP": (
                                st.column_config.NumberColumn(
                                    format="%,.2f",
                                )
                            ),
                            f"{bubble_label} · Change": (
                                st.column_config.NumberColumn(
                                    format="%+,.1f%%",
                                )
                            ),
                        }
                    )

                st.dataframe(
                    detail,
                    width="stretch",
                    hide_index=True,
                    column_config=column_config,
                )

    with temporal_tab:
        st.markdown(
            "Compare the same mode pair across every ordered temporal bucket. "
            "Each row reports the Pearson relationship before congestion pricing, "
            "after congestion pricing, and the post-minus-pre change."
        )

        with st.spinner(
            "Calculating relationships across temporal buckets..."
        ):
            temporal_matrix = _build_temporal_relationship_matrix(
                x_metric=x_metric,
                y_metric=y_metric,
                geography_level=geography_level,
                display_mode=display_mode,
                filter_scope=filter_scope,
                filter_value=filter_value,
            )

        heatmap_summary = _build_heatmap_summary(
            temporal_matrix
        )

        _render_heatmap_summary_cards(
            heatmap_summary
        )

        st.caption(
            f"{x_label} versus {y_label} · {geography_context} · "
            "Pearson correlations across displayed geographies"
        )

        heatmap_takeaway = _build_heatmap_insight(
            temporal_matrix,
            x_label=x_label,
            y_label=y_label,
            geography_context=geography_context,
        )

        temporal_fig = build_temporal_relationship_heatmap(
            temporal_matrix
        )

        st.plotly_chart(
            temporal_fig,
            width="stretch",
            config={
                "displayModeBar": False,
                "responsive": True,
            },
            key=(
                f"raw05_temporal_heatmap_"
                f"{x_metric}_{y_metric}_"
                f"{display_mode}_{geography_level}_"
                f"{filter_scope}_{filter_value}"
            ),
        )
        render_chart_insight(heatmap_takeaway)

        st.caption(
            "For the Pre-CP and Post-CP columns, teal indicates a positive "
            "relationship and terracotta an inverse relationship. In the Post − pre "
            "column, teal means the correlation coefficient moved upward and "
            "terracotta means it moved downward; that is not always the same as "
            "strengthening or weakening. Hover over a cell to see the number of "
            "paired geographies supporting that estimate."
        )

        if SUBWAY_METRIC in {
            x_metric,
            y_metric,
        }:
            st.caption(
                "Subway Ridership does not cover Staten Island, so Staten Island "
                "cannot contribute when Subway Ridership is selected."
            )

        with st.expander(
            "Inspect temporal-bucket correlations",
            expanded=False,
        ):
            heatmap_table = temporal_matrix[
                [
                    "temporal_bucket_label",
                    "pre_correlation",
                    "post_correlation",
                    "correlation_change",
                    "pre_observation_count",
                    "post_observation_count",
                ]
            ].copy()

            heatmap_table = heatmap_table.rename(
                columns={
                    "temporal_bucket_label": "Temporal bucket",
                    "pre_correlation": "Pre-CP correlation",
                    "post_correlation": "Post-CP correlation",
                    "correlation_change": "Post minus pre",
                    "pre_observation_count": "Pre-CP paired geographies",
                    "post_observation_count": "Post-CP paired geographies",
                }
            )

            st.dataframe(
                heatmap_table,
                width="stretch",
                hide_index=True,
                column_config={
                    "Pre-CP correlation": st.column_config.NumberColumn(
                        format="%+.3f",
                    ),
                    "Post-CP correlation": st.column_config.NumberColumn(
                        format="%+.3f",
                    ),
                    "Post minus pre": st.column_config.NumberColumn(
                        format="%+.3f",
                    ),
                    "Pre-CP paired geographies": st.column_config.NumberColumn(
                        format="%d",
                    ),
                    "Post-CP paired geographies": st.column_config.NumberColumn(
                        format="%d",
                    ),
                },
            )

st.markdown("### What this page establishes")
st.markdown(
    "Mode relationships add information that separate pre/post averages cannot. Two "
    "measures may rise together overall yet have a weak local relationship, or their "
    "association may change across geography and time of day. The page therefore treats "
    "co-movement as a pattern to measure rather than assuming that citywide changes imply "
    "the same neighborhood-level behavior."
)

with st.expander("How this page works", expanded=False):
    st.markdown(
        """
        **1. Compare modes on the same geographic units.** Relationship views pair
        mobility measures within the selected geography so each point represents a
        like-for-like place comparison.

        **2. Use the hero to show three demand measures at once.** Taxi Trips determine
        horizontal position, FHVHV Trips determine vertical position, and Subway
        Ridership determines bubble area. Open markers are pre-CP and filled markers
        are post-CP.

        **3. Separate movement from association.** A geography moving upward or to the
        right shows a change in its own mobility level. Correlation asks a different
        question: whether places with higher values on one measure also tend to have
        higher values on the other.

        **4. Compare relationships before and after launch.** The temporal heatmap shows
        Pearson correlations for the pre-CP and post-CP periods and their numerical
        difference. A higher post-minus-pre coefficient is not automatically the same
        thing as a stronger relationship; the sign and starting value matter.

        **5. Respect coverage.** Relationship estimates use only geographies with both
        selected measures available. Subway Ridership does not cover Staten Island, so
        Staten Island cannot contribute when Subway Ridership is selected.

        **6. Keep correlation descriptive.** Co-movement can identify places and times
        where two measures behave similarly or differently. It does not show that one
        mode caused the other to change.
        """
    )

st.caption(
    "Evidence scope: observed NYC mobility before and after the January 5, 2025 "
    "congestion-pricing launch. Relationships and correlations describe co-movement "
    "among the selected measures and geographies; they do not establish substitution "
    "between modes or a causal effect of congestion pricing."
)

