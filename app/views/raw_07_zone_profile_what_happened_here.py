from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data_access.loaders import (
    BASE_METRICS,
    CONGESTION_PRICING_START_DATE,
    METRIC_LABELS,
    TEMPORAL_BUCKET_ORDER,
)
from app.data_access.spatial_aggregations import (
    ALL_TEMPORAL_BUCKETS_LABEL,
)
from app.data_access.zone_profiles import (
    COMPARISON_LEVELS,
    format_geo_policy_label,
    get_comparison_label,
    get_recommended_zone_catalog,
    get_zone_and_baseline_daily_series,
    get_zone_metadata,
    get_zone_pairwise_divergences,
    get_zone_pre_post_profile,
    get_zone_rank_context,
    get_zone_temporal_profile,
    summarize_daily_comparison,
)
from app.data_access.anomalies import (
    get_zone_metric_driver_anomaly_events,
)

from app.utils.project_branding import (
    BRAND_COLORS,
    apply_branding,
    exploration_section,
    inject_app_css,
    render_chart_insight,
)


inject_app_css()

DEFAULT_ZONE_NAME = "Alphabet City"
DEFAULT_METRIC = "taxi_trip_count"
HERO_ZONE_NAME = "Alphabet City"
HERO_METRICS = [
    "taxi_trip_count",
    "taxi_avg_trip_speed",
    "fhvhv_trip_count",
    "fhvhv_avg_trip_speed",
    "subway_ridership",
    "avg_bus_speed",
]

TEMPORAL_BUCKET_OPTIONS = [
    ALL_TEMPORAL_BUCKETS_LABEL,
    *TEMPORAL_BUCKET_ORDER,
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

SMOOTHING_OPTIONS = {
    "Raw daily": None,
    "7-day rolling average": 7,
    "14-day rolling average": 14,
    "28-day rolling average": 28,
}


SAVED_PROFILES = {
    "Custom": None,
    "Alphabet City · Taxi trips · Citywide": {
        "zone": "Alphabet City",
        "metric": "taxi_trip_count",
        "comparison": "Citywide",
        "temporal_bucket": ALL_TEMPORAL_BUCKETS_LABEL,
        "trend_view": "Relative change",
        "smoothing": "14-day rolling average",
    },
    "Alphabet City · Taxi trips · Manhattan": {
        "zone": "Alphabet City",
        "metric": "taxi_trip_count",
        "comparison": "Borough",
        "temporal_bucket": ALL_TEMPORAL_BUCKETS_LABEL,
        "trend_view": "Relative change",
        "smoothing": "14-day rolling average",
    },
    "Midtown Center · Subway ridership · Manhattan": {
        "zone": "Midtown Center",
        "metric": "subway_ridership",
        "comparison": "Borough",
        "temporal_bucket": ALL_TEMPORAL_BUCKETS_LABEL,
        "trend_view": "Relative change",
        "smoothing": "14-day rolling average",
    },
    "JFK Airport · Taxi trips · Queens": {
        "zone": "JFK Airport",
        "metric": "taxi_trip_count",
        "comparison": "Borough",
        "temporal_bucket": ALL_TEMPORAL_BUCKETS_LABEL,
        "trend_view": "Actual values",
        "smoothing": "14-day rolling average",
    },
    "Fort Greene · Taxi trips · Brooklyn": {
        "zone": "Fort Greene",
        "metric": "taxi_trip_count",
        "comparison": "Borough",
        "temporal_bucket": ALL_TEMPORAL_BUCKETS_LABEL,
        "trend_view": "Relative change",
        "smoothing": "14-day rolling average",
    },
    "Jamaica Bay · Bus speed · Queens": {
        "zone": "Jamaica Bay",
        "metric": "avg_bus_speed",
        "comparison": "Borough",
        "temporal_bucket": ALL_TEMPORAL_BUCKETS_LABEL,
        "trend_view": "Actual values",
        "smoothing": "14-day rolling average",
    },
    "Breezy Point · Bus speed · Queens": {
        "zone": "Breezy Point/Fort Tilden/Riis Beach",
        "metric": "avg_bus_speed",
        "comparison": "Borough",
        "temporal_bucket": ALL_TEMPORAL_BUCKETS_LABEL,
        "trend_view": "Actual values",
        "smoothing": "14-day rolling average",
    },
}


def _format_percent(
    value: object,
) -> str:
    if pd.isna(value):
        return "Unavailable"
    return f"{float(value):+,.1f}%"


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


def _format_geo_policy(
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
        str(value).strip().title(),
    )


def _zone_selector_label(
    row: pd.Series,
) -> str:
    prefix = "★ " if bool(
        row["is_recommended"]
    ) else ""

    return (
        f"{prefix}{row['zone']} · "
        f"{row['borough']} · "
        f"Zone {row['taxi_zone_id']}"
    )


def _add_smoothed_columns(
    daily: pd.DataFrame,
    *,
    window: int | None,
) -> pd.DataFrame:
    result = daily.copy().sort_values("date")

    if window is None:
        result["zone_display"] = result["zone_value"]
        result["baseline_display"] = result["baseline_value"]
        result["zone_index_display"] = result["zone_index"]
        result["baseline_index_display"] = result["baseline_index"]
        return result

    for source, target in [
        ("zone_value", "zone_display"),
        ("baseline_value", "baseline_display"),
        ("zone_index", "zone_index_display"),
        ("baseline_index", "baseline_index_display"),
    ]:
        result[target] = (
            result[source]
            .rolling(
                window=window,
                min_periods=max(
                    2,
                    window // 3,
                ),
            )
            .mean()
        )

    return result


def _apply_chart_branding(
    fig: go.Figure,
) -> go.Figure:
    fig.update_layout(
        title={"text": " "}
    )

    fig = apply_branding(fig)

    fig.update_layout(
        title={"text": ""},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.18,
            "yanchor": "top",
        },
        margin={
            "l": 60,
            "r": 35,
            "t": 35,
            "b": 115,
        },
    )

    return fig


def _build_trend_chart(
    daily: pd.DataFrame,
    *,
    zone_name: str,
    baseline_label: str,
    metric_label: str,
    scale_mode: str,
    show_comparison: bool,
    anomaly_events: pd.DataFrame | None = None,
) -> go.Figure:
    fig = go.Figure()

    if scale_mode == "Relative change":
        zone_column = "zone_index_display"
        baseline_column = "baseline_index_display"
        y_axis_title = "Index value"
        hover_suffix = ""
        hover_format = ".1f"

        fig.add_hline(
            y=100,
            line_dash="dot",
            line_color="rgba(0,109,119,0.40)",
            annotation_text="Pre-CP average = 100",
            annotation_position="bottom right",
        )
    else:
        zone_column = "zone_display"
        baseline_column = "baseline_display"
        y_axis_title = metric_label
        hover_suffix = ""
        hover_format = ",.2f"

    if show_comparison:
        fig.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily[baseline_column],
                mode="lines",
                name=baseline_label,
                line={
                    "color": BRAND_COLORS["seafoam"],
                    "width": 3,
                },
                hovertemplate=(
                    f"<b>{baseline_label}</b><br>"
                    "Date: %{x|%b %d, %Y}<br>"
                    f"Value: %{{y:{hover_format}}}{hover_suffix}"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily[zone_column],
            mode="lines",
            name=zone_name,
            line={
                "color": BRAND_COLORS["dark_teal"],
                "width": 3.5,
            },
            hovertemplate=(
                f"<b>{zone_name}</b><br>"
                "Date: %{x|%b %d, %Y}<br>"
                f"Value: %{{y:{hover_format}}}{hover_suffix}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_vline(
        x=CONGESTION_PRICING_START_DATE,
        line_dash="dash",
        line_color=BRAND_COLORS["terracotta"],
        annotation_text="Congestion pricing starts",
        annotation_position="top left",
    )

    fig.update_xaxes(
        title_text="Date",
        showgrid=True,
        gridcolor="rgba(0,109,119,0.08)",
    )

    fig.update_yaxes(
        title_text=y_axis_title,
        showgrid=True,
        gridcolor="rgba(0,109,119,0.10)",
    )

    fig.update_layout(
        height=560,
    )

    if (
            anomaly_events is not None
            and not anomaly_events.empty
    ):
        anomaly_points = anomaly_events.merge(
            daily[
                [
                    "date",
                    zone_column,
                ]
            ],
            on="date",
            how="inner",
            validate="one_to_one",
        )

        if not anomaly_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=anomaly_points["date"],
                    y=anomaly_points[zone_column],
                    mode="markers",
                    name="Stress anomaly",
                    marker={
                        "size": 12,
                        "symbol": "circle-open",
                        "color": BRAND_COLORS["terracotta"],
                        "line": {
                            "color": BRAND_COLORS["terracotta"],
                            "width": 2,
                        },
                    },
                    customdata=anomaly_points[
                        [
                            "anomaly_event_count",
                        ]
                    ],
                    hovertemplate=(
                        "<b>Stress-anomaly date</b><br>"
                        "Date: %{x|%b %d, %Y}<br>"
                        "Stress-anomaly dayparts: %{customdata[0]:,}"
                        "<extra></extra>"
                    ),
                )
            )
    return _apply_chart_branding(fig)


def _build_zone_hero_chart(profile: pd.DataFrame) -> go.Figure:
    """Show the frozen hero zone's reliable multimodal Pre/Post changes."""
    plot_data = profile.loc[
        profile["metric"].isin(HERO_METRICS)
        & profile["zone_percent_change"].notna()
    ].copy()
    plot_data["eligible"] = plot_data["zone_eligible_for_percent_change"].fillna(False)
    plot_data["direction"] = np.select(
        [
            ~plot_data["eligible"],
            plot_data["zone_percent_change"].ge(0),
        ],
        ["Low baseline", "Increased"],
        default="Decreased",
    )
    plot_data = plot_data.sort_values("zone_percent_change")
    color_map = {
        "Increased": BRAND_COLORS["dark_teal"],
        "Decreased": BRAND_COLORS["terracotta"],
        "Low baseline": BRAND_COLORS["seafoam"],
    }

    fig = go.Figure()
    for direction in ["Decreased", "Increased", "Low baseline"]:
        subset = plot_data.loc[plot_data["direction"].eq(direction)]
        if subset.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=subset["zone_percent_change"],
                y=subset["metric_label"],
                orientation="h",
                name=direction,
                marker_color=color_map[direction],
                customdata=np.column_stack(
                    [
                        subset["zone_pre_daily_average"].map(_format_number),
                        subset["zone_post_daily_average"].map(_format_number),
                    ]
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Change: %{x:+.1f}%<br>"
                    "Pre-CP daily average: %{customdata[0]}<br>"
                    "Post-CP daily average: %{customdata[1]}"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_vline(x=0, line_color="rgba(50,50,50,0.55)", line_width=1.2)
    fig.update_xaxes(
        title_text="Change from Pre-CP to Post-CP",
        ticksuffix="%",
        showgrid=True,
        gridcolor="rgba(0,109,119,0.10)",
    )
    fig.update_yaxes(title_text="", automargin=True)
    fig.update_layout(
        height=390,
        barmode="overlay",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    return _apply_chart_branding(fig)


def _build_multimetric_profile_chart(
    profile: pd.DataFrame,
    *,
    show_comparison: bool,
) -> go.Figure:
    """Compare all ten base measures for the selected zone and geography."""
    plot_data = profile.copy()

    plot_data["eligible"] = (
        plot_data[
            "zone_eligible_for_percent_change"
        ]
        .fillna(False)
    )

    fig = go.Figure()

    if show_comparison:
        connector_rows = plot_data[
            plot_data[
                [
                    "zone_percent_change",
                    "baseline_percent_change",
                ]
            ]
            .notna()
            .all(axis=1)
        ]

        for _, row in connector_rows.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[
                        row["baseline_percent_change"],
                        row["zone_percent_change"],
                    ],
                    y=[
                        row["metric_label"],
                        row["metric_label"],
                    ],
                    mode="lines",
                    line={
                        "color": (
                            "rgba(80,80,80,0.30)"
                            if row["eligible"]
                            else "rgba(80,80,80,0.13)"
                        ),
                        "width": 3,
                    },
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        baseline_valid = plot_data[
            plot_data[
                "baseline_percent_change"
            ].notna()
        ]

        if not baseline_valid.empty:
            fig.add_trace(
                go.Scatter(
                    x=baseline_valid[
                        "baseline_percent_change"
                    ],
                    y=baseline_valid["metric_label"],
                    mode="markers",
                    name="Comparison average",
                    marker={
                        "size": 12,
                        "color": BRAND_COLORS["seafoam"],
                        "line": {
                            "color": "white",
                            "width": 1,
                        },
                    },
                    customdata=np.column_stack(
                        [
                            baseline_valid[
                                "baseline_pre_daily_average"
                            ].map(_format_number),
                            baseline_valid[
                                "baseline_post_daily_average"
                            ].map(_format_number),
                            baseline_valid[
                                "baseline_peer_zones"
                            ],
                        ]
                    ),
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Comparison change: %{x:+.1f}%<br>"
                        "Pre: %{customdata[0]}<br>"
                        "Post: %{customdata[1]}<br>"
                        "Comparison zones: %{customdata[2]}"
                        "<extra></extra>"
                    ),
                )
            )

    reliable = plot_data[
        plot_data["eligible"]
        & plot_data[
            "zone_percent_change"
        ].notna()
    ]

    unreliable = plot_data[
        ~plot_data["eligible"]
        & plot_data[
            "zone_percent_change"
        ].notna()
    ]

    if not reliable.empty:
        reliable_customdata = (
            np.column_stack(
                [
                    reliable[
                        "zone_pre_daily_average"
                    ].map(_format_number),
                    reliable[
                        "zone_post_daily_average"
                    ].map(_format_number),
                    reliable["change_gap"].map(
                        _format_percent
                    ),
                ]
            )
            if show_comparison
            else np.column_stack(
                [
                    reliable[
                        "zone_pre_daily_average"
                    ].map(_format_number),
                    reliable[
                        "zone_post_daily_average"
                    ].map(_format_number),
                    ["No comparison"] * len(reliable),
                ]
            )
        )

        fig.add_trace(
            go.Scatter(
                x=reliable["zone_percent_change"],
                y=reliable["metric_label"],
                mode="markers",
                name="Selected zone",
                marker={
                    "size": 14,
                    "color": BRAND_COLORS["dark_teal"],
                    "line": {
                        "color": "white",
                        "width": 1,
                    },
                },
                customdata=reliable_customdata,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Selected-zone change: %{x:+.1f}%<br>"
                    "Pre: %{customdata[0]}<br>"
                    "Post: %{customdata[1]}<br>"
                    "Comparison: %{customdata[2]}"
                    "<extra></extra>"
                ),
            )
        )

    if not unreliable.empty:
        fig.add_trace(
            go.Scatter(
                x=unreliable[
                    "zone_percent_change"
                ],
                y=unreliable["metric_label"],
                mode="markers",
                name="Selected zone · low baseline",
                marker={
                    "size": 14,
                    "color": BRAND_COLORS["dark_teal"],
                    "symbol": "circle-open",
                    "line": {
                        "color": BRAND_COLORS["dark_teal"],
                        "width": 2,
                    },
                },
                customdata=np.column_stack(
                    [
                        unreliable[
                            "zone_pre_daily_average"
                        ].map(_format_number),
                        unreliable[
                            "zone_post_daily_average"
                        ].map(_format_number),
                    ]
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Observed change: %{x:+.1f}%<br>"
                    "Pre: %{customdata[0]}<br>"
                    "Post: %{customdata[1]}<br>"
                    "Reliability: below threshold"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_vline(
        x=0,
        line_color="rgba(50,50,50,0.55)",
        line_width=1.2,
    )

    fig.update_xaxes(
        title_text="Change from pre-CP to post-CP",
        ticksuffix="%",
        showgrid=True,
        gridcolor="rgba(0,109,119,0.10)",
    )

    fig.update_yaxes(
        title_text="",
        autorange="reversed",
        automargin=True,
    )

    fig.update_layout(
        height=500,
    )

    return _apply_chart_branding(
        fig
    )


def _build_temporal_profile_chart(
    temporal_profile: pd.DataFrame,
    *,
    show_comparison: bool,
) -> go.Figure:
    """Compare the selected metric across all ordered temporal buckets."""
    plot_data = (
        temporal_profile
        .sort_values(
            "temporal_bucket_order",
            ascending=False,
        )
        .copy()
    )

    plot_data["bucket_label"] = (
        plot_data["temporal_bucket"]
        .map(
            lambda bucket: TEMPORAL_BUCKET_LABELS.get(
                bucket,
                bucket,
            )
        )
    )

    fig = go.Figure()

    if show_comparison:
        connector_rows = plot_data[
            plot_data[
                [
                    "zone_percent_change",
                    "baseline_percent_change",
                ]
            ]
            .notna()
            .all(axis=1)
        ]

        for _, row in connector_rows.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[
                        row["baseline_percent_change"],
                        row["zone_percent_change"],
                    ],
                    y=[
                        row["bucket_label"],
                        row["bucket_label"],
                    ],
                    mode="lines",
                    line={
                        "color": (
                            "rgba(80,80,80,0.32)"
                            if row[
                                "zone_eligible_for_percent_change"
                            ]
                            else "rgba(80,80,80,0.13)"
                        ),
                        "width": 3,
                    },
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        baseline_valid = plot_data[
            plot_data[
                "baseline_percent_change"
            ].notna()
        ]

        if not baseline_valid.empty:
            fig.add_trace(
                go.Scatter(
                    x=baseline_valid[
                        "baseline_percent_change"
                    ],
                    y=baseline_valid["bucket_label"],
                    mode="markers",
                    name="Comparison average",
                    marker={
                        "size": 11,
                        "color": BRAND_COLORS["seafoam"],
                        "line": {
                            "color": "white",
                            "width": 1,
                        },
                    },
                    customdata=baseline_valid[
                        "baseline_peer_zones"
                    ],
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Comparison change: %{x:+.1f}%<br>"
                        "Comparison zones: %{customdata}"
                        "<extra></extra>"
                    ),
                )
            )

    reliable = plot_data[
        plot_data[
            "zone_eligible_for_percent_change"
        ]
        & plot_data[
            "zone_percent_change"
        ].notna()
    ]

    unreliable = plot_data[
        ~plot_data[
            "zone_eligible_for_percent_change"
        ]
        & plot_data[
            "zone_percent_change"
        ].notna()
    ]

    if not reliable.empty:
        fig.add_trace(
            go.Scatter(
                x=reliable["zone_percent_change"],
                y=reliable["bucket_label"],
                mode="markers",
                name="Selected zone",
                marker={
                    "size": 13,
                    "color": BRAND_COLORS["dark_teal"],
                    "line": {
                        "color": "white",
                        "width": 1,
                    },
                },
                customdata=(
                    reliable["change_gap"].map(
                        _format_percent
                    )
                    if show_comparison
                    else pd.Series(
                        ["No comparison"] * len(reliable),
                        index=reliable.index,
                    )
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Selected-zone change: %{x:+.1f}%<br>"
                    "Comparison: %{customdata}"
                    "<extra></extra>"
                ),
            )
        )

    if not unreliable.empty:
        fig.add_trace(
            go.Scatter(
                x=unreliable[
                    "zone_percent_change"
                ],
                y=unreliable["bucket_label"],
                mode="markers",
                name="Selected zone · low baseline",
                marker={
                    "size": 13,
                    "color": BRAND_COLORS["dark_teal"],
                    "symbol": "circle-open",
                    "line": {
                        "color": BRAND_COLORS["dark_teal"],
                        "width": 2,
                    },
                },
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Observed change: %{x:+.1f}%<br>"
                    "Reliability: below threshold"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_vline(
        x=0,
        line_color="rgba(50,50,50,0.55)",
        line_width=1.2,
    )

    fig.update_xaxes(
        title_text="Change from pre-CP to post-CP",
        ticksuffix="%",
        showgrid=True,
        gridcolor="rgba(0,109,119,0.10)",
    )

    fig.update_yaxes(
        title_text="",
        automargin=True,
    )

    fig.update_layout(
        height=600,
    )

    return _apply_chart_branding(
        fig
    )


def _build_rank_chart(
    rank_context: pd.DataFrame,
) -> go.Figure:
    """Show the selected zone's citywide percentile across core metrics."""
    plot_data = (
        rank_context[
            rank_context["eligible"]
        ]
        .copy()
        .sort_values(
            "citywide_percentile",
            ascending=True,
        )
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=plot_data[
                "citywide_percentile"
            ],
            y=plot_data["metric_label"],
            orientation="h",
            marker={
                "color": BRAND_COLORS["dark_teal"],
            },
            text=plot_data[
                "citywide_percentile"
            ].map(
                lambda value: (
                    f"{value:.0f}th"
                )
            ),
            textposition="outside",
            cliponaxis=False,
            customdata=np.column_stack(
                [
                    plot_data["citywide_rank"],
                    plot_data[
                        "citywide_eligible_zones"
                    ],
                    plot_data["borough_rank"],
                    plot_data[
                        "borough_eligible_zones"
                    ],
                    plot_data[
                        "borough_percentile"
                    ],
                    plot_data[
                        "zone_percent_change"
                    ].map(_format_percent),
                ]
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Zone change: %{customdata[5]}<br>"
                "Citywide rank: %{customdata[0]} of %{customdata[1]}<br>"
                "Citywide percentile: %{x:.1f}<br>"
                "Borough rank: %{customdata[2]} of %{customdata[3]}<br>"
                "Borough percentile: %{customdata[4]:.1f}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    fig.update_xaxes(
        title_text="Citywide percentile",
        range=[0, 105],
        ticksuffix="th",
        showgrid=True,
        gridcolor="rgba(0,109,119,0.10)",
    )

    fig.update_yaxes(
        title_text="",
        automargin=True,
    )

    fig.update_layout(
        height=500,
        margin={
            "l": 60,
            "r": 75,
            "t": 35,
            "b": 75,
        },
    )

    return _apply_chart_branding(
        fig
    )


def _build_divergence_chart(
    divergences: pd.DataFrame,
) -> go.Figure:
    """Show the three strongest eligible opposite-direction metric pairs."""
    plot_data = (
        divergences.head(3)
        .copy()
        .sort_values(
            "absolute_divergence",
            ascending=True,
        )
    )

    plot_data["pair_label"] = (
        plot_data["metric_a_label"]
        + " vs "
        + plot_data["metric_b_label"]
    )

    fig = go.Figure()

    for _, row in plot_data.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[
                    row["metric_a_change"],
                    row["metric_b_change"],
                ],
                y=[
                    row["pair_label"],
                    row["pair_label"],
                ],
                mode="lines",
                line={
                    "color": "rgba(80,80,80,0.34)",
                    "width": 4,
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=plot_data[
                "metric_a_change"
            ],
            y=plot_data["pair_label"],
            mode="markers+text",
            name="First metric",
            showlegend=False,
            marker={
                "size": 14,
                "color": BRAND_COLORS["dark_teal"],
                "line": {
                    "color": "white",
                    "width": 1,
                },
            },
            text=plot_data[
                "metric_a_change"
            ].map(_format_percent),
            textposition="middle right",
            customdata=plot_data[
                "metric_a_label"
            ],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "%{customdata}: %{x:+.1f}%"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_data[
                "metric_b_change"
            ],
            y=plot_data["pair_label"],
            mode="markers+text",
            name="Second metric",
            showlegend=False,
            marker={
                "size": 14,
                "color": BRAND_COLORS["terracotta"],
                "line": {
                    "color": "white",
                    "width": 1,
                },
            },
            text=plot_data[
                "metric_b_change"
            ].map(_format_percent),
            textposition="middle left",
            customdata=plot_data[
                "metric_b_label"
            ],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "%{customdata}: %{x:+.1f}%"
                "<extra></extra>"
            ),
        )
    )

    fig.add_vline(
        x=0,
        line_color="rgba(50,50,50,0.55)",
        line_width=1.2,
    )

    fig.update_xaxes(
        title_text="Change from pre-CP to post-CP",
        ticksuffix="%",
        showgrid=True,
        gridcolor="rgba(0,109,119,0.10)",
    )

    fig.update_yaxes(
        title_text="",
        automargin=True,
    )

    fig.update_layout(
        height=380,
        showlegend=False,
    )

    return _apply_chart_branding(
        fig
    )


def _build_trend_takeaway(
    *,
    zone_name: str,
    metric_label: str,
    comparison_label: str,
    summary: dict[str, object],
    show_comparison: bool,
) -> str:
    zone_change = summary[
        "zone_percent_change"
    ]

    if pd.isna(zone_change):
        return (
            "This selection does not have enough observed pre- and "
            "post-period values to summarize."
        )

    if not show_comparison:
        direction = (
            "increased"
            if zone_change > 0
            else "decreased"
            if zone_change < 0
            else "was unchanged"
        )

        return (
            f"**{metric_label} {direction} by "
            f"{abs(float(zone_change)):.1f}% in {zone_name}** "
            "from the pre-CP to post-CP daily average."
        )

    comparison_change = summary[
        "baseline_percent_change"
    ]
    gap = summary["change_gap"]

    if (
        pd.isna(comparison_change)
        or pd.isna(gap)
    ):
        return (
            "The selected zone has a valid trend, but the geographic "
            "comparison does not have enough complete values."
        )

    if np.sign(zone_change) != np.sign(
        comparison_change
    ):
        relationship = (
            "moved in the opposite direction from"
        )
    elif abs(zone_change) > abs(
        comparison_change
    ):
        relationship = "changed more than"
    else:
        relationship = "changed less than"

    return (
        f"**{zone_name} changed {_format_percent(zone_change)} for "
        f"{metric_label}**, while the **{comparison_label} changed "
        f"{_format_percent(comparison_change)}**. The selected zone "
        f"{relationship} the geographic comparison, with a gap of "
        f"**{abs(gap):.1f} percentage points**."
    )


def _record_timing(
    timings: dict[str, float],
    label: str,
    start_time: float,
) -> None:
    timings[label] = (
        perf_counter()
        - start_time
    )


def _format_timing_report(
    timings: dict[str, float],
) -> str:
    ordered_labels = [
        "Daily series",
        "Daily summary + smoothing",
        "Ten-metric profile",
        "Temporal profile",
        "Rank context",
        "Pairwise divergences",
        "Trend chart",
        "Ten-metric chart",
        "Temporal chart",
        "Rank chart",
        "Divergence chart",
        "Total analytical preparation",
        "Total chart construction",
        "Total measured work",
    ]

    lines = [
        "Stage\tSeconds"
    ]

    for label in ordered_labels:
        if label not in timings:
            continue

        lines.append(
            f"{label}\t"
            f"{timings[label]:.3f}"
        )

    return "\n".join(
        lines
    )


page_timings: dict[str, float] = {}
page_profile_start = perf_counter()


st.caption(
    "ZONE PROFILE"
)

st.title(
    "What happened here?"
)

st.write(
    "Citywide and borough averages can conceal a neighborhood whose mobility profile "
    "changed in a very different way. This page brings the measures together for one "
    "Taxi Zone so we can see how its demand, speeds, timing, relative standing, and "
    "cross-mode disagreements fit into one local story."
)

catalog = get_recommended_zone_catalog()

if catalog.empty:
    st.error(
        "No Taxi Zones were available."
    )
    st.stop()

catalog["selector_label"] = (
    catalog.apply(
        _zone_selector_label,
        axis=1,
    )
)

zone_options = (
    catalog["taxi_zone_id"]
    .tolist()
)

default_matches = catalog[
    catalog["zone"].eq(
        DEFAULT_ZONE_NAME
    )
]

default_zone_id = (
    default_matches.iloc[0][
        "taxi_zone_id"
    ]
    if not default_matches.empty
    else zone_options[0]
)

hero_matches = catalog.loc[catalog["zone"].eq(HERO_ZONE_NAME)]
if not hero_matches.empty:
    hero_zone_id = int(hero_matches.iloc[0]["taxi_zone_id"])
    hero_profile = get_zone_pre_post_profile(
        taxi_zone_id=hero_zone_id,
        comparison_level="Citywide",
        temporal_bucket=ALL_TEMPORAL_BUCKETS_LABEL,
    )
    hero_core = hero_profile.loc[
        hero_profile["metric"].isin(HERO_METRICS)
        & hero_profile["zone_percent_change"].notna()
    ].copy()
    hero_reliable = hero_core.loc[
        hero_core["zone_eligible_for_percent_change"].fillna(False)
    ].copy()

    if not hero_reliable.empty:
        hero_increases = hero_reliable.loc[hero_reliable["zone_percent_change"].gt(0)]
        hero_decreases = hero_reliable.loc[hero_reliable["zone_percent_change"].lt(0)]
        hero_largest_increase = (
            hero_increases.loc[hero_increases["zone_percent_change"].idxmax()]
            if not hero_increases.empty
            else hero_reliable.loc[hero_reliable["zone_percent_change"].idxmax()]
        )
        hero_largest_decrease = (
            hero_decreases.loc[hero_decreases["zone_percent_change"].idxmin()]
            if not hero_decreases.empty
            else hero_reliable.loc[hero_reliable["zone_percent_change"].idxmin()]
        )

        increase_card_label = (
            f"Largest increase · {hero_largest_increase['metric_label']}"
            if not hero_increases.empty
            else f"Smallest decrease · {hero_largest_increase['metric_label']}"
        )
        decrease_card_label = (
            f"Largest decrease · {hero_largest_decrease['metric_label']}"
            if not hero_decreases.empty
            else f"Smallest increase · {hero_largest_decrease['metric_label']}"
        )

        st.header("How did Alphabet City's mobility profile change?")
        st.write(
            "The fixed opening profile compares five core mobility measures for Alphabet "
            "City using the same pre/post percentage-change scale. Bars to the right "
            "increased after launch; bars to the left decreased. The cards summarize the "
            "largest reliable movements without treating one measure as the whole story."
        )

        hero_card_1, hero_card_2, hero_card_3, hero_card_4 = st.columns(4)
        hero_card_1.metric(
            increase_card_label,
            _format_percent(hero_largest_increase["zone_percent_change"]),
        )
        hero_card_2.metric(
            decrease_card_label,
            _format_percent(hero_largest_decrease["zone_percent_change"]),
        )
        hero_card_3.metric(
            "Core metrics increasing",
            f"{int(hero_reliable['zone_percent_change'].gt(0).sum())} of {len(hero_reliable)}",
        )
        hero_card_4.metric(
            "Core metrics decreasing",
            f"{int(hero_reliable['zone_percent_change'].lt(0).sum())} of {len(hero_reliable)}",
        )

        st.subheader("How the five core mobility measures changed")
        st.plotly_chart(
            _build_zone_hero_chart(hero_profile),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
            key="raw07_frozen_alphabet_city_hero",
        )
        if not hero_increases.empty and not hero_decreases.empty:
            render_chart_insight(
                f"**{hero_largest_increase['metric_label']} increased "
                f"{abs(float(hero_largest_increase['zone_percent_change'])):.1f}%**, "
                f"while **{hero_largest_decrease['metric_label']} decreased "
                f"{abs(float(hero_largest_decrease['zone_percent_change'])):.1f}%**. "
                "Alphabet City therefore did not experience one uniform mobility shift: "
                "different modes tell different parts of its Post-CP story."
            )
        else:
            dominant_direction = "increased" if not hero_increases.empty else "decreased"
            render_chart_insight(
                f"All reliable displayed core metrics moved in the same direction: "
                f"they **{dominant_direction}**. The explorer below shows whether this "
                "pattern holds across other metrics, time windows, and comparison groups."
            )
        st.caption(
            "Dark-teal and terracotta bars meet the profile’s percentage-change "
            "eligibility rule. Seafoam bars, if present, have a low Pre-CP baseline "
            "and should be read cautiously."
        )
def _apply_saved_profile() -> None:
    """Apply the selected saved profile immediately."""
    saved_profile_name = st.session_state[
        "raw07_saved_profile"
    ]

    profile = SAVED_PROFILES.get(
        saved_profile_name
    )

    if profile is None:
        return

    zone_match = catalog[
        catalog["zone"].eq(
            profile["zone"]
        )
    ]

    if zone_match.empty:
        st.session_state[
            "raw07_saved_profile_error"
        ] = (
            f"The saved zone {profile['zone']} "
            "is not available."
        )
        return

    st.session_state.pop(
        "raw07_saved_profile_error",
        None,
    )

    st.session_state["raw07_zone"] = (
        zone_match.iloc[0][
            "taxi_zone_id"
        ]
    )
    st.session_state["raw07_metric"] = (
        profile["metric"]
    )
    st.session_state["raw07_comparison"] = (
        profile["comparison"]
    )
    st.session_state["raw07_bucket"] = (
        profile["temporal_bucket"]
    )
    st.session_state["raw07_scale"] = (
        profile["trend_view"]
    )
    st.session_state["raw07_smoothing"] = (
        profile["smoothing"]
    )


def _mark_profile_custom() -> None:
    """Mark the view custom when an individual control changes."""
    st.session_state[
        "raw07_saved_profile"
    ] = "Custom"


with exploration_section(
    key="raw07_exploration_area",
    title="Explore any Taxi Zone",
    description=(
        "Choose a saved profile or build your own. One selection coordinates "
        "the timeline, multimetric comparison, time-of-week pattern, rankings, "
        "and mode-divergence views below."
    ),
):
    if "raw07_saved_profile" not in st.session_state:
        st.session_state["raw07_saved_profile"] = "Custom"
    if "raw07_zone" not in st.session_state:
        st.session_state["raw07_zone"] = default_zone_id
    if "raw07_metric" not in st.session_state:
        st.session_state["raw07_metric"] = DEFAULT_METRIC
    if "raw07_comparison" not in st.session_state:
        st.session_state["raw07_comparison"] = (
            "Borough" if "Borough" in COMPARISON_LEVELS else COMPARISON_LEVELS[0]
        )
    if "raw07_bucket" not in st.session_state:
        st.session_state["raw07_bucket"] = ALL_TEMPORAL_BUCKETS_LABEL
    if "raw07_scale" not in st.session_state:
        st.session_state["raw07_scale"] = "Relative change"
    if "raw07_smoothing" not in st.session_state:
        st.session_state["raw07_smoothing"] = "14-day rolling average"
    if "raw07_show_anomalies" not in st.session_state:
        st.session_state["raw07_show_anomalies"] = False

    st.selectbox(
        "Start with a saved profile",
        options=list(
            SAVED_PROFILES
        ),
        key="raw07_saved_profile",
        on_change=_apply_saved_profile,
        help=(
            "Selecting a saved profile immediately updates all controls below. "
            "Changing an individual control returns the view to Custom."
        ),
    )

    saved_profile_error = st.session_state.get(
        "raw07_saved_profile_error"
    )

    if saved_profile_error:
        st.error(
            saved_profile_error
        )

    control1, control2 = st.columns(2)

    with control1:
        selected_zone_id = st.selectbox(
            "Taxi Zone",
            options=zone_options,
            format_func=lambda value: (
                catalog.loc[
                    catalog[
                        "taxi_zone_id"
                    ].eq(value),
                    "selector_label",
                ].iloc[0]
            ),
            key="raw07_zone",
            on_change=_mark_profile_custom,
        )

    with control2:
        selected_metric = st.selectbox(
            "Primary metric",
            options=BASE_METRICS,
            format_func=lambda metric: (
                METRIC_LABELS.get(
                    metric,
                    metric,
                )
            ),
            key="raw07_metric",
            on_change=_mark_profile_custom,
        )

    control3, control4 = st.columns(2)

    with control3:
        comparison_level = st.selectbox(
            "Geographic comparison",
            options=COMPARISON_LEVELS,
            key="raw07_comparison",
            on_change=_mark_profile_custom,
            help=(
                "Comparison averages exclude the selected Taxi Zone. "
                "These are geographic benchmarks, not statistically matched peers."
            ),
        )

    with control4:
        temporal_bucket = st.selectbox(
            "Temporal bucket",
            options=TEMPORAL_BUCKET_OPTIONS,
            format_func=lambda bucket: (
                TEMPORAL_BUCKET_LABELS.get(
                    bucket,
                    bucket,
                )
            ),
            key="raw07_bucket",
            on_change=_mark_profile_custom,
        )

    display1, display2, display3 = st.columns(3)

    with display1:
        scale_mode = st.selectbox(
            "Trend view",
            options=[
                "Relative change",
                "Actual values",
            ],
            key="raw07_scale",
            on_change=_mark_profile_custom,
        )

    with display2:
        smoothing_label = st.selectbox(
            "Trend smoothing",
            options=list(
                SMOOTHING_OPTIONS
            ),
            key="raw07_smoothing",
            on_change=_mark_profile_custom,
        )

    with display3:
        show_anomalies = st.checkbox(
            "Show stress anomalies",
            key="raw07_show_anomalies",
            help=(
                "Overlay selected stress anomalies where this metric "
                "was identified as one of the event drivers."
            ),
        )

    metadata = get_zone_metadata(
        selected_zone_id
    )

    metric_label = METRIC_LABELS.get(
        selected_metric,
        selected_metric,
    )

    comparison_label = get_comparison_label(
        taxi_zone_id=selected_zone_id,
        comparison_level=comparison_level,
    )

    show_comparison = (
        comparison_level
        != "No comparison"
    )

    with st.spinner(
        "Preparing the coordinated zone profile..."
    ):
        analytical_start = perf_counter()

        stage_start = perf_counter()
        daily = get_zone_and_baseline_daily_series(
            taxi_zone_id=selected_zone_id,
            metric=selected_metric,
            temporal_bucket=temporal_bucket,
            comparison_level=comparison_level,
        )
        anomaly_events = pd.DataFrame()

        if show_anomalies:
            anomaly_events = get_zone_metric_driver_anomaly_events(
                selected_zone_id,
                metric=selected_metric,
                temporal_bucket=temporal_bucket,
            )

            if not anomaly_events.empty:
                # Finalist anomalies are Taxi Zone × date × daypart events,
                # while this chart displays one selected metric per day.
                # Collapse multiple qualifying dayparts to one daily marker.
                anomaly_events = (
                    anomaly_events[
                        [
                            "date",
                            "comparison_event_id",
                        ]
                    ]
                    .groupby(
                        "date",
                        observed=True,
                        as_index=False,
                    )
                    .agg(
                        anomaly_event_count=(
                            "comparison_event_id",
                            "nunique",
                        ),
                    )
                )

        _record_timing(
            page_timings,
            "Daily series",
            stage_start,
        )

        stage_start = perf_counter()
        daily_summary = summarize_daily_comparison(
            daily
        )

        smoothed_daily = _add_smoothed_columns(
            daily,
            window=SMOOTHING_OPTIONS[
                smoothing_label
            ],
        )
        _record_timing(
            page_timings,
            "Daily summary + smoothing",
            stage_start,
        )

        stage_start = perf_counter()
        metric_profile = get_zone_pre_post_profile(
            taxi_zone_id=selected_zone_id,
            comparison_level=comparison_level,
            temporal_bucket=temporal_bucket,
        )
        _record_timing(
            page_timings,
            "Ten-metric profile",
            stage_start,
        )

        stage_start = perf_counter()
        temporal_profile = get_zone_temporal_profile(
            taxi_zone_id=selected_zone_id,
            metric=selected_metric,
            comparison_level=comparison_level,
        )
        _record_timing(
            page_timings,
            "Temporal profile",
            stage_start,
        )

        stage_start = perf_counter()
        rank_context = get_zone_rank_context(
            taxi_zone_id=selected_zone_id,
            temporal_bucket=temporal_bucket,
        )
        _record_timing(
            page_timings,
            "Rank context",
            stage_start,
        )

        stage_start = perf_counter()
        divergences = get_zone_pairwise_divergences(
            taxi_zone_id=selected_zone_id,
            temporal_bucket=temporal_bucket,
        )
        _record_timing(
            page_timings,
            "Pairwise divergences",
            stage_start,
        )

        _record_timing(
            page_timings,
            "Total analytical preparation",
            analytical_start,
        )

    st.divider()

    st.header(
        f"{metadata['zone']} · "
        f"{metadata['borough']} · "
        f"Zone {metadata['taxi_zone_id']}"
    )

    st.caption(
        f"Policy geography: "
        f"{format_geo_policy_label(metadata['cbd_spatial_category'])} · "
        f"{metric_label} · "
        f"{comparison_label} · "
        f"{TEMPORAL_BUCKET_LABELS.get(temporal_bucket, temporal_bucket)}"
    )

    card1, card2, card3, card4 = (
        st.columns(4)
    )

    card1.metric(
        "Pre-CP daily average",
        _format_number(
            daily_summary[
                "zone_pre_average"
            ]
        ),
    )

    card2.metric(
        "Post-CP daily average",
        _format_number(
            daily_summary[
                "zone_post_average"
            ]
        ),
    )

    card3.metric(
        "Zone change",
        _format_percent(
            daily_summary[
                "zone_percent_change"
            ]
        ),
    )

    comparison_metric_label = (
        "Difference vs comparison"
        if (
            not show_comparison
            or daily_summary[
                "baseline_observations"
            ]
            > 0
        )
        else "Comparison unavailable"
    )

    card4.metric(
        comparison_metric_label
        if show_comparison
        else "Observed days",
        (
            _format_percent(
                daily_summary[
                    "change_gap"
                ]
            )
            if show_comparison
            else (
                f"{daily_summary['zone_observations']:,}"
            )
        ),
    )

    st.markdown(
        "## What happened over time?"
    )

    if scale_mode == "Relative change":
        st.write(
            "Each displayed series is indexed to its own full pre-CP daily "
            "average. A value of 100 represents the typical pre-CP level."
        )
    else:
        st.write(
            "The chart uses the original metric units. Geographic comparisons "
            "show the average across all other Taxi Zones in the selected group."
        )

    chart_stage_start = perf_counter()
    trend_fig = _build_trend_chart(
        smoothed_daily,
        zone_name=str(
            metadata["zone"]
        ),
        baseline_label=comparison_label,
        metric_label=metric_label,
        scale_mode=scale_mode,
        show_comparison=show_comparison,
        anomaly_events=anomaly_events,
    )
    _record_timing(
        page_timings,
        "Trend chart",
        chart_stage_start,
    )

    st.plotly_chart(
        trend_fig,
        width="stretch",
        config={
            "displayModeBar": False,
            "responsive": True,
        },
        key=(
            f"raw07_trend_{selected_zone_id}_"
            f"{selected_metric}_{comparison_level}_"
            f"{temporal_bucket}_{scale_mode}_"
            f"{smoothing_label}"
        ),
    )

    render_chart_insight(
        _build_trend_takeaway(
            zone_name=str(
                metadata["zone"]
            ),
            metric_label=metric_label,
            comparison_label=comparison_label,
            summary=daily_summary,
            show_comparison=show_comparison,
        )
    )

    if show_comparison:
        st.caption(
            f"The geographic comparison excludes {metadata['zone']}. "
            "It is a geographic benchmark, not a statistically matched mobility-peer group."
        )

    st.divider()

    st.markdown(
        "## How did mobility measures change?"
    )

    st.write(
        "Compare the selected zone across all ten clean base measures. Filled "
        "markers show threshold-supported comparisons; open markers flag "
        "low-baseline changes that should be interpreted cautiously."
    )

    st.caption(
        "Low-baseline observations remain visible as open markers, but they are "
        "excluded from the highlighted largest-gap statement."
    )

    chart_stage_start = perf_counter()
    profile_fig = _build_multimetric_profile_chart(
        metric_profile,
        show_comparison=show_comparison,
    )
    _record_timing(
        page_timings,
        "Ten-metric chart",
        chart_stage_start,
    )

    st.plotly_chart(
        profile_fig,
        width="stretch",
        config={
            "displayModeBar": False,
            "responsive": True,
        },
        key=(
            f"raw07_profile_{selected_zone_id}_"
            f"{comparison_level}_{temporal_bucket}"
        ),
    )

    reliable_profile = metric_profile[
        metric_profile[
            "zone_eligible_for_percent_change"
        ]
    ].copy()

    if (
        show_comparison
        and not reliable_profile.empty
    ):
        reliable_profile = reliable_profile[
            reliable_profile[
                "change_gap"
            ].notna()
        ]

        if not reliable_profile.empty:
            strongest = reliable_profile.loc[
                reliable_profile[
                    "change_gap"
                ]
                .abs()
                .idxmax()
            ]

            render_chart_insight(
                f"The largest displayed zone-versus-comparison gap is "
                f"**{strongest['metric_label']}** at "
                f"**{strongest['change_gap']:+.1f} percentage points**."
            )
        else:
            render_chart_insight(
                "No comparison gap met the baseline and data-quality requirements "
                "needed to identify a reliable largest change in this view."
            )

    elif not reliable_profile.empty:
        strongest = reliable_profile.loc[
            reliable_profile[
                "zone_percent_change"
            ]
            .abs()
            .idxmax()
        ]

        render_chart_insight(
            f"The largest displayed selected-zone change is "
            f"**{strongest['metric_label']}** at "
            f"**{strongest['zone_percent_change']:+.1f}%**."
        )

    else:
        render_chart_insight(
            "No metric met the baseline and data-quality requirements needed "
            "to identify a reliable largest change in this view."
        )

    if selected_metric == "subway_ridership":
        st.caption(
            "Subway Ridership does not cover Staten Island and may be unavailable "
            "for Taxi Zones without mapped subway activity."
        )

    st.divider()

    st.markdown(
        "## When was the difference strongest?"
    )

    st.write(
        "The selected metric is compared across all ten ordered temporal buckets. "
        "Open selected-zone markers indicate low-baseline percentage changes."
    )

    st.caption(
        "Only periods meeting the minimum baseline and data-quality requirements "
        "are considered when the page identifies the largest displayed gap."
    )

    chart_stage_start = perf_counter()
    temporal_fig = _build_temporal_profile_chart(
        temporal_profile,
        show_comparison=show_comparison,
    )
    _record_timing(
        page_timings,
        "Temporal chart",
        chart_stage_start,
    )

    st.plotly_chart(
        temporal_fig,
        width="stretch",
        config={
            "displayModeBar": False,
            "responsive": True,
        },
        key=(
            f"raw07_temporal_{selected_zone_id}_"
            f"{selected_metric}_{comparison_level}"
        ),
    )

    reliable_temporal = temporal_profile[
        temporal_profile[
            "zone_eligible_for_percent_change"
        ]
    ].copy()

    if (
        show_comparison
        and not reliable_temporal.empty
    ):
        reliable_temporal = reliable_temporal[
            reliable_temporal[
                "change_gap"
            ].notna()
        ]

        if not reliable_temporal.empty:
            strongest_bucket = reliable_temporal.loc[
                reliable_temporal[
                    "change_gap"
                ]
                .abs()
                .idxmax()
            ]

            strongest_bucket_label = (
                TEMPORAL_BUCKET_LABELS.get(
                    strongest_bucket[
                        "temporal_bucket"
                    ],
                    strongest_bucket[
                        "temporal_bucket"
                    ],
                )
            )

            render_chart_insight(
                f"The largest displayed temporal gap occurred during "
                f"**{strongest_bucket_label}**, at "
                f"**{strongest_bucket['change_gap']:+.1f} "
                "percentage points**."
            )
        else:
            render_chart_insight(
                "No comparison-period gap met the baseline and data-quality "
                "requirements needed to identify a reliable temporal leader."
            )

    elif not reliable_temporal.empty:
        strongest_bucket = reliable_temporal.loc[
            reliable_temporal[
                "zone_percent_change"
            ]
            .abs()
            .idxmax()
        ]

        strongest_bucket_label = (
            TEMPORAL_BUCKET_LABELS.get(
                strongest_bucket[
                    "temporal_bucket"
                ],
                strongest_bucket[
                    "temporal_bucket"
                ],
            )
        )

        render_chart_insight(
            f"The largest displayed temporal change occurred during "
            f"**{strongest_bucket_label}**, at "
            f"**{strongest_bucket['zone_percent_change']:+.1f}%**."
        )

    else:
        render_chart_insight(
            "No temporal bucket met the baseline and data-quality requirements "
            "needed to identify a reliable largest change in this view."
        )

    st.divider()

    st.markdown(
        "## How unusual was this zone?"
    )

    st.write(
        "Percentiles compare the selected zone with all eligible Taxi Zones for "
        "the current temporal-bucket selection. Higher percentiles indicate larger "
        "pre-to-post increases."
    )

    eligible_rank_context = rank_context[
        rank_context["eligible"]
    ].copy()

    if eligible_rank_context.empty:
        st.info(
            "No rank context was available for this selection after applying "
            "the minimum data-quality requirements."
        )
    else:
        selected_rank = eligible_rank_context[
            eligible_rank_context[
                "metric"
            ].eq(selected_metric)
        ]

        if not selected_rank.empty:
            selected_rank_row = selected_rank.iloc[0]

            rank1, rank2, rank3, rank4 = (
                st.columns(4)
            )

            rank1.metric(
                "Citywide rank",
                (
                    f"{int(selected_rank_row['citywide_rank'])} "
                    f"of {int(selected_rank_row['citywide_eligible_zones'])}"
                ),
            )

            rank2.metric(
                "Citywide percentile",
                (
                    f"{selected_rank_row['citywide_percentile']:.1f}"
                ),
            )

            rank3.metric(
                f"{metadata['borough']} rank",
                (
                    f"{int(selected_rank_row['borough_rank'])} "
                    f"of {int(selected_rank_row['borough_eligible_zones'])}"
                ),
            )

            rank4.metric(
                f"{metadata['borough']} percentile",
                (
                    f"{selected_rank_row['borough_percentile']:.1f}"
                ),
            )

        chart_stage_start = perf_counter()
        rank_fig = _build_rank_chart(
            rank_context
        )
        _record_timing(
            page_timings,
            "Rank chart",
            chart_stage_start,
        )

        st.plotly_chart(
            rank_fig,
            width="stretch",
            config={
                "displayModeBar": False,
                "responsive": True,
            },
            key=(
                f"raw07_rank_{selected_zone_id}_"
                f"{temporal_bucket}"
            ),
        )

        percentile_leader = eligible_rank_context.loc[
            eligible_rank_context["citywide_percentile"].idxmax()
        ]
        percentile_laggard = eligible_rank_context.loc[
            eligible_rank_context["citywide_percentile"].idxmin()
        ]
        render_chart_insight(
            f"**{percentile_leader['metric_label']}** is this zone's highest "
            f"citywide standing at the **{float(percentile_leader['citywide_percentile']):.1f}th "
            f"percentile**; **{percentile_laggard['metric_label']}** is lowest at "
            f"the **{float(percentile_laggard['citywide_percentile']):.1f}th percentile**."
        )

    st.divider()

    st.markdown(
        "## Which modes disagreed most here?"
    )

    st.write(
        "The largest opposite-direction metric pairs reveal where one "
        "part of the zone's mobility profile increased while another declined."
    )

    if divergences.empty:
        st.info(
            "No opposite-direction metric pairs met the display requirements for "
            "this zone and temporal-bucket selection."
        )
    else:
        st.caption(
            "Endpoint labels show each metric's observed percentage change; "
            "the metric pair is named directly on the y-axis."
        )

        chart_stage_start = perf_counter()
        divergence_fig = _build_divergence_chart(
            divergences
        )
        _record_timing(
            page_timings,
            "Divergence chart",
            chart_stage_start,
        )

        st.plotly_chart(
            divergence_fig,
            width="stretch",
            config={
                "displayModeBar": False,
                "responsive": True,
            },
            key=(
                f"raw07_divergence_{selected_zone_id}_"
                f"{temporal_bucket}"
            ),
        )

        top_divergence = divergences.iloc[0]

        render_chart_insight(
            f"The strongest disagreement is "
            f"**{top_divergence['metric_a_label']} "
            f"({_format_percent(top_divergence['metric_a_change'])})** "
            f"versus **{top_divergence['metric_b_label']} "
            f"({_format_percent(top_divergence['metric_b_change'])})**, "
            f"a gap of **{top_divergence['absolute_divergence']:.1f} "
            "percentage points**."
        )

    page_timings[
        "Total chart construction"
    ] = sum(
        page_timings.get(
            label,
            0.0,
        )
        for label in [
            "Trend chart",
            "Ten-metric chart",
            "Temporal chart",
            "Rank chart",
            "Divergence chart",
        ]
    )

    page_timings[
        "Total measured work"
    ] = (
        page_timings.get(
            "Total analytical preparation",
            0.0,
        )
        + page_timings[
            "Total chart construction"
        ]
    )



st.markdown("### What this page establishes")
st.markdown(
    "A neighborhood's mobility story can be internally mixed. Demand, speed, timing, "
    "and relative standing do not have to move in the same direction, and the strongest "
    "local change may look very different from the borough or citywide pattern. A zone "
    "profile therefore adds context that is lost when each mobility measure or geography "
    "is viewed separately."
)

with st.expander(
    "How this page works",
    expanded=False,
):
    st.markdown(
        """
        **1. Put multiple mobility measures into one local profile.** The opening view
        compares five core measures for the same Taxi Zone so increases and decreases
        can be read together rather than as separate citywide stories.

        **2. Compare a zone with a contextual benchmark.** Actual-value comparisons use
        averages across the other Taxi Zones in the selected geography; the selected
        zone itself is excluded. Borough and citywide comparisons are contextual
        benchmarks, not matched peer groups.

        **3. Use a common relative-change scale when units differ.** Relative-change
        trends index each series to its own full pre-CP daily average. That makes the
        direction and scale of change comparable without implying that trips, riders,
        and speeds share the same units.

        **4. Protect percentage comparisons from weak baselines.** Percentage-change
        views apply the project's metric-specific minimum baseline and data-quality
        requirements. Open markers show observed low-baseline changes that should be
        interpreted cautiously and should not drive ranked claims.

        **5. Add relative standing without turning it into significance.** Borough and
        citywide percentiles show where the selected zone sits among eligible Taxi Zones.
        A high or low percentile is a descriptive rank, not a statistical significance
        test.

        **6. Treat mode disagreements as local descriptive evidence.** Opposite-direction
        metric pairs show where parts of the same zone's mobility profile separated.
        They do not establish substitution or explain why the measures changed.
        """
    )

st.caption(
    "Evidence scope: observed NYC mobility before and after the January 5, 2025 "
    "congestion-pricing launch. Zone profiles, comparison benchmarks, percentiles, and "
    "mode divergences describe local patterns; they do not establish that congestion "
    "pricing caused the observed changes."
)
