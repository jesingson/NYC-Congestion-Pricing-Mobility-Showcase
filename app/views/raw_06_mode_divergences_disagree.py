from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data_access.loaders import CORE_METRICS
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
    inject_app_css,
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

    return f"{float(value):+,.1f}%"


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
    return get_zone_pre_post_metric_summary(
        metrics=[
            metric_a,
            metric_b,
        ],
        temporal_bucket=temporal_bucket,
    ).copy()


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
                "Geo-policy group: %{customdata[2]}<br>"
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
                "Geo-policy group: %{customdata[2]}<br>"
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
    total = len(data)

    if total == 0:
        return (
            "No eligible opposite-direction Taxi Zones were available."
        )

    manhattan_count = int(
        data["borough"]
        .eq("Manhattan")
        .sum()
    )

    taxi_up_count = int(
        data[
            "direction_relationship"
        ]
        .eq(
            "Metric A up · Metric B down"
        )
        .sum()
    )

    return (
        f"Opposite Taxi–FHVHV movement was overwhelmingly concentrated "
        f"in Manhattan: **{manhattan_count} of {total} eligible zones**. "
        f"In **{taxi_up_count} zones**, Taxi Trips increased while "
        "FHVHV Trips declined."
    )


def _build_explorer_takeaway(
    data: pd.DataFrame,
    *,
    metric_a_label: str,
    metric_b_label: str,
) -> str:
    if data.empty:
        return (
            "No eligible opposite-direction zones remain under the "
            "selected filters."
        )

    top = data.iloc[0]

    a_up_b_down = int(
        data[
            "direction_relationship"
        ]
        .eq(
            "Metric A up · Metric B down"
        )
        .sum()
    )

    a_down_b_up = int(
        data[
            "direction_relationship"
        ]
        .eq(
            "Metric A down · Metric B up"
        )
        .sum()
    )

    return (
        f"Among **{len(data)} eligible opposite-direction zones**, "
        f"**{a_up_b_down}** show {metric_a_label} increasing while "
        f"{metric_b_label} declines, and **{a_down_b_up}** show the "
        f"reverse. The largest current gap is **{top['zone']}** at "
        f"**{top['absolute_divergence']:.1f} percentage points**."
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
                "Geo-policy group"
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
    "Find Taxi Zones where two mobility measures moved in opposite "
    "directions after congestion pricing, then compare the size and "
    "direction of those changes."
)


# ---------------------------------------------------------------------
# Frozen hero
# ---------------------------------------------------------------------
st.header(
    "Taxi and FHVHV demand diverged most in Manhattan"
)

st.write(
    "The distance between each pair of endpoints is the percentage-point "
    "gap between Taxi and FHVHV change. Longer connectors indicate greater "
    "disagreement."
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
    use_container_width=True,
    config={
        "displayModeBar": False,
        "responsive": True,
    },
    key="raw06_static_hero",
)

st.caption(
    "Includes only zones with meaningful pre-period activity for both "
    "measures. Hover for pre-CP and post-CP daily averages and absolute "
    "changes."
)

st.markdown(
    "#### What stands out"
)

st.info(
    _build_hero_takeaway(
        hero_data
    )
)


# ---------------------------------------------------------------------
# Interactive explorer
# ---------------------------------------------------------------------
st.divider()

st.header(
    "Explore mode divergences"
)

st.write(
    "Choose any two core measures, narrow the geography or direction, "
    "and set the minimum disagreement required to enter the ranking."
)

if "raw06_saved_view" not in st.session_state:
    st.session_state[
        "raw06_saved_view"
    ] = "Taxi vs FHVHV demand"

saved_view = st.selectbox(
    "Start with a saved configuration",
    options=SAVED_VIEW_OPTIONS,
    index=0,
    key="raw06_saved_view",
    help=(
        "Saved configurations provide curated starting points. "
        "Changing any control switches the selection to Custom."
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
    "##### Measures"
)

measure_col1, measure_col2 = (
    st.columns(2)
)

with measure_col1:
    metric_a = st.selectbox(
        "Metric A",
        options=CORE_METRICS,
        index=(
            CORE_METRICS.index(
                TAXI_METRIC
            )
            if TAXI_METRIC
            in CORE_METRICS
            else 0
        ),
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
            if FHVHV_METRIC
            in metric_b_options
            else metric_b_options[0]
        )

    metric_b = st.selectbox(
        "Metric B",
        options=metric_b_options,
        index=(
            metric_b_options.index(
                FHVHV_METRIC
            )
            if FHVHV_METRIC
            in metric_b_options
            else 0
        ),
        format_func=_metric_label,
        key="raw06_metric_b",
        on_change=_mark_saved_view_custom,
    )

st.markdown(
    "##### Time and ranking"
)

ranking_col1, ranking_col2, ranking_col3 = (
    st.columns(3)
)

with ranking_col1:
    temporal_bucket = st.selectbox(
        "Time bucket",
        options=TEMPORAL_BUCKET_OPTIONS,
        index=0,
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
        index=1,
        key="raw06_top_n",
        on_change=_mark_saved_view_custom,
    )

with ranking_col3:
    direction_filter = st.selectbox(
        "Direction pattern",
        options=DIRECTION_OPTIONS,
        index=0,
        format_func=lambda value: (
            _direction_label(
                value,
                metric_a_label=_metric_label(
                    metric_a
                ),
                metric_b_label=_metric_label(
                    metric_b
                ),
            )
        ),
        key="raw06_direction_filter",
        on_change=_mark_saved_view_custom,
    )

with st.spinner(
    "Updating the divergence explorer..."
):
    explorer_source = (
        _load_metric_pair_source(
            metric_a,
            metric_b,
            temporal_bucket,
        )
    )

try:
    explorer_data = (
        _prepare_metric_pair_divergence(
            explorer_source,
            metric_a=metric_a,
            metric_b=metric_b,
        )
    )
except (
    KeyError,
    TypeError,
    ValueError,
) as error:
    st.error(str(error))
    st.stop()

st.markdown(
    "##### Geography and threshold"
)

geography_col1, geography_col2, threshold_col = (
    st.columns(
        [
            1,
            1,
            1.4,
        ]
    )
)

with geography_col1:
    geography_filter = st.selectbox(
        "Geography filter",
        options=GEOGRAPHY_FILTER_OPTIONS,
        index=0,
        key="raw06_geography_filter",
        on_change=_mark_saved_view_custom,
    )

geography_value: str | None = None

if geography_filter == "Borough":
    borough_options = _ordered_values(
        explorer_data["borough"],
        BOROUGH_ORDER,
    )

    if not borough_options:
        with geography_col2:
            st.caption(
                "No Borough values available"
            )
    else:
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
                index=0,
                key="raw06_borough_value",
                on_change=_mark_saved_view_custom,
            )

elif (
    geography_filter
    == "Geo-policy group"
):
    policy_options = _ordered_values(
        explorer_data[
            "geo_policy_group"
        ],
        GEO_POLICY_ORDER,
    )

    if not policy_options:
        with geography_col2:
            st.caption(
                "No geo-policy values available"
            )
    else:
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
                "Geo-policy group",
                options=policy_options,
                index=0,
                key="raw06_policy_value",
                on_change=_mark_saved_view_custom,
            )

else:
    with geography_col2:
        st.caption(
            "All eligible Taxi Zones"
        )

maximum_divergence = (
    float(
        np.ceil(
            explorer_data[
                "absolute_divergence"
            ].max()
            / 5
        )
        * 5
    )
    if not explorer_data.empty
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
        value=0.0,
        step=5.0,
        format="%.0f pp",
        help=(
            "Require at least this many percentage points between the "
            "two pre-to-post changes."
        ),
        key="raw06_minimum_divergence",
        on_change=_mark_saved_view_custom,
    )

filtered_data = _filter_divergence_data(
    explorer_data,
    geography_filter=geography_filter,
    geography_value=geography_value,
    direction_filter=direction_filter,
    minimum_divergence=minimum_divergence,
)

metric_a_label = _metric_label(
    metric_a
)

metric_b_label = _metric_label(
    metric_b
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
        (
            f"{metric_a_label} up · "
            f"{metric_b_label} down"
        ),
        (
            f"{int(filtered_data['direction_relationship'].eq('Metric A up · Metric B down').sum()):,}"
        ),
    )

    explorer_card3.metric(
        (
            f"{metric_a_label} down · "
            f"{metric_b_label} up"
        ),
        (
            f"{int(filtered_data['direction_relationship'].eq('Metric A down · Metric B up').sum()):,}"
        ),
    )

    explorer_card4.metric(
        "Largest divergence",
        (
            f"{filtered_data.iloc[0]['absolute_divergence']:.1f} pp"
        ),
    )

    geography_context = (
        "All Taxi Zones"
        if geography_filter
        == "All Taxi Zones"
        else (
            f"{geography_filter}: "
            f"{geography_value}"
        )
    )

    st.caption(
        f"Metric A: {metric_a_label} · "
        f"Metric B: {metric_b_label} · "
        f"{TEMPORAL_BUCKET_LABELS[temporal_bucket]} · "
        f"{geography_context} · "
        f"Minimum gap: {minimum_divergence:.0f} pp"
    )


    if (
        "subway_ridership"
        in {
            metric_a,
            metric_b,
        }
    ):
        st.caption(
            "Subway Ridership does not cover Staten Island, so Staten "
            "Island cannot appear when Subway Ridership is selected."
        )

    st.markdown(
        "#### What stands out in this view"
    )

    st.info(
        _build_explorer_takeaway(
            filtered_data,
            metric_a_label=metric_a_label,
            metric_b_label=metric_b_label,
        )
    )

    explorer_fig = (
        build_dumbbell_chart(
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
    )

    st.plotly_chart(
        explorer_fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
        key=(
            f"raw06_explorer_"
            f"{metric_a}_{metric_b}_"
            f"{temporal_bucket}_"
            f"{geography_filter}_"
            f"{geography_value}_"
            f"{direction_filter}_"
            f"{minimum_divergence}_"
            f"{top_n}"
        ),
    )

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

        st.dataframe(
            detail,
            use_container_width=True,
            hide_index=True,
            column_config={
                f"{metric_a_label} · Pre": (
                    st.column_config.NumberColumn(
                        format="%,.2f",
                    )
                ),
                f"{metric_a_label} · Post": (
                    st.column_config.NumberColumn(
                        format="%,.2f",
                    )
                ),
                f"{metric_a_label} · Daily-average change": (
                    st.column_config.NumberColumn(
                        format="%+,.2f",
                    )
                ),
                f"{metric_a_label} · Percent change": (
                    st.column_config.NumberColumn(
                        format="%+,.1f%%",
                    )
                ),
                f"{metric_b_label} · Pre": (
                    st.column_config.NumberColumn(
                        format="%,.2f",
                    )
                ),
                f"{metric_b_label} · Post": (
                    st.column_config.NumberColumn(
                        format="%,.2f",
                    )
                ),
                f"{metric_b_label} · Daily-average change": (
                    st.column_config.NumberColumn(
                        format="%+,.2f",
                    )
                ),
                f"{metric_b_label} · Percent change": (
                    st.column_config.NumberColumn(
                        format="%+,.1f%%",
                    )
                ),
                "Divergence": (
                    st.column_config.NumberColumn(
                        format="%,.1f pp",
                    )
                ),
            },
        )

with st.expander(
    "How this ranking works",
    expanded=False,
):
    st.markdown(
        """
A zone enters the ranking only when:

- both selected measures have complete pre-CP and post-CP values;
- both pass the project's metric-specific baseline required for stable percentage change;
- one measure increased while the other decreased.

The ranking is the absolute percentage-point gap between the two changes.
It identifies divergence only and does not establish substitution or causality.
        """
    )