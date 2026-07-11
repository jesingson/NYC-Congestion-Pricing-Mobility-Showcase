from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data_access.aggregations import (
    CORE_METRICS,
    add_rolling_average,
    build_raw01_frozen_interpretation,
    build_selected_view_interpretation,
    format_summary_for_display,
    get_available_filter_values,
    get_daily_metric_trends,
    get_indexed_daily_trends,
    get_pre_post_metric_summary,
)
from app.data_access.loaders import (
    CONGESTION_PRICING_START_DATE,
    METRIC_LABELS,
    STUDY_END_DATE,
    STUDY_START_DATE,
    TEMPORAL_BUCKET_ORDER,
)
from app.utils.project_branding import (
    BRAND_COLORS,
    apply_branding,
    inject_app_css,
)


HERO_DEFAULT_METRICS = [
    "taxi_trip_count",
    "subway_ridership",
    "fhvhv_trip_count",
]


INTERESTING_VIEWS = {
    "Citywide taxi baseline": {
        "metric": "taxi_trip_count",
        "geography_scope": "Citywide",
        "borough": None,
        "cbd_spatial_category": None,
        "temporal_bucket": "All temporal buckets",
        "smoothing": "14-day rolling average",
        "interpretation": (
            "A baseline view of the citywide taxi activity pattern. Start here to understand the broad "
            "post-CP pattern before narrowing to specific geographies or time buckets."
        ),
    },
    "Gateway taxi surge": {
        "metric": "taxi_trip_count",
        "geography_scope": "CBD spatial category",
        "borough": None,
        "cbd_spatial_category": "gateway_to_cbd",
        "temporal_bucket": "All temporal buckets",
        "smoothing": "14-day rolling average",
        "interpretation": (
            "One of the clearest saved views: gateway-to-CBD zones show a much larger post-CP taxi "
            "activity difference than the citywide average."
        ),
    },
    "Citywide subway lift": {
        "metric": "subway_ridership",
        "geography_scope": "Citywide",
        "borough": None,
        "cbd_spatial_category": None,
        "temporal_bucket": "All temporal buckets",
        "smoothing": "28-day rolling average",
        "interpretation": (
            "A transit-demand counterpart to the taxi view. The 28-day smoothing makes the ridership "
            "pattern easier to see through daily noise."
        ),
    },
    "Weekday PM taxi activity": {
        "metric": "taxi_trip_count",
        "geography_scope": "Citywide",
        "borough": None,
        "cbd_spatial_category": None,
        "temporal_bucket": "weekday_pm_peak",
        "smoothing": "14-day rolling average",
        "interpretation": (
            "A commute-relevant view that tests whether the taxi activity pattern also appears during "
            "weekday PM peak periods."
        ),
    },
}


def _key_from_label(label: str) -> str:
    """Create a stable Streamlit widget-key suffix from a saved-view label."""
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _apply_bottom_legend(fig: go.Figure, *, bottom_margin: int = 105) -> go.Figure:
    """Move Plotly legend below chart so it does not collide with the title."""
    fig.update_layout(
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18,
            "xanchor": "center",
            "x": 0.5,
            "font": {"color": BRAND_COLORS["dark_teal"]},
        },
        margin={"l": 50, "r": 30, "t": 70, "b": bottom_margin},
    )
    return fig


def _add_full_period_trend_line(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    metric_col: str,
    color: str,
) -> None:
    """Add a simple visual linear fit across the selected window."""
    fit_df = df.loc[df["date"].notna() & df[metric_col].notna()].copy()

    if len(fit_df) < 10:
        return

    fit_df = fit_df.sort_values("date")
    x_numeric = (fit_df["date"] - fit_df["date"].min()).dt.days.to_numpy()
    y_values = fit_df[metric_col].to_numpy()

    if np.nanstd(y_values) == 0:
        return

    slope, intercept = np.polyfit(x_numeric, y_values, deg=1)
    fitted_values = intercept + slope * x_numeric

    fig.add_trace(
        go.Scatter(
            x=fit_df["date"],
            y=fitted_values,
            mode="lines",
            name="Full-period fitted line",
            line={"color": color, "width": 3, "dash": "dash"},
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Date: %{x|%b %d, %Y}<br>"
                "Fitted value: %{y:,.2f}<br>"
                "<extra></extra>"
            ),
        )
    )


def _add_period_trend_line(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    metric_col: str,
    label: str,
    period: str,
    color: str,
) -> None:
    """Add a simple visual linear fit for one side of the CP boundary."""
    period_df = df.loc[
        (df["pre_post_cp"] == period)
        & df["date"].notna()
        & df[metric_col].notna()
    ].copy()

    if len(period_df) < 10:
        return

    period_df = period_df.sort_values("date")
    x_numeric = (period_df["date"] - period_df["date"].min()).dt.days.to_numpy()
    y_values = period_df[metric_col].to_numpy()

    if np.nanstd(y_values) == 0:
        return

    slope, intercept = np.polyfit(x_numeric, y_values, deg=1)
    fitted_values = intercept + slope * x_numeric

    fig.add_trace(
        go.Scatter(
            x=period_df["date"],
            y=fitted_values,
            mode="lines",
            name=label,
            line={"color": color, "width": 3, "dash": "dash"},
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Date: %{x|%b %d, %Y}<br>"
                "Fitted value: %{y:,.2f}<br>"
                "<extra></extra>"
            ),
        )
    )


def _display_summary_table(summary_df: pd.DataFrame) -> None:
    """Display a formatted pre/post summary table."""
    st.dataframe(
        format_summary_for_display(summary_df),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Pre-CP daily avg": st.column_config.NumberColumn(
                "Pre-CP daily avg",
                format="localized",
            ),
            "Post-CP daily avg": st.column_config.NumberColumn(
                "Post-CP daily avg",
                format="localized",
            ),
            "Abs. change": st.column_config.NumberColumn(
                "Abs. change",
                format="localized",
            ),
            "% change": st.column_config.NumberColumn(
                "% change",
                format="%.2f%%",
            ),
            "Pre observed days": st.column_config.NumberColumn(
                "Pre observed days",
                format="localized",
            ),
            "Post observed days": st.column_config.NumberColumn(
                "Post observed days",
                format="localized",
            ),
        },
    )




RAW01_DEFAULT_SAVED_VIEW = "Citywide taxi baseline"
RAW01_SAVED_VIEW_OPTIONS = ["None"] + list(INTERESTING_VIEWS.keys())
RAW01_GEO_OPTIONS = ["Citywide", "Borough", "CBD spatial category"]
RAW01_SMOOTHING_OPTIONS = [
    "None",
    "7-day rolling average",
    "14-day rolling average",
    "28-day rolling average",
]


def _set_raw01_controls_from_preset(preset_name: str) -> None:
    """Apply a saved view to the interactive controls."""
    preset = INTERESTING_VIEWS[preset_name]

    st.session_state["raw01_metric"] = preset["metric"]
    st.session_state["raw01_geography_scope"] = preset["geography_scope"]
    st.session_state["raw01_temporal_bucket"] = preset["temporal_bucket"]
    st.session_state["raw01_smoothing"] = preset["smoothing"]
    st.session_state["raw01_start_date"] = STUDY_START_DATE.date()
    st.session_state["raw01_end_date"] = STUDY_END_DATE.date()

    if preset.get("borough") is not None:
        st.session_state["raw01_borough"] = preset["borough"]

    if preset.get("cbd_spatial_category") is not None:
        st.session_state["raw01_cbd_spatial_category"] = preset["cbd_spatial_category"]


def _initialize_raw01_controls() -> None:
    """Initialize saved-view state once per session."""
    if "raw01_saved_view" not in st.session_state:
        st.session_state["raw01_saved_view"] = RAW01_DEFAULT_SAVED_VIEW
        _set_raw01_controls_from_preset(RAW01_DEFAULT_SAVED_VIEW)


def _apply_raw01_saved_view() -> None:
    """Callback: load preset values when the saved-view selector changes."""
    selected = st.session_state.get("raw01_saved_view", "None")

    if selected != "None":
        _set_raw01_controls_from_preset(selected)


def _mark_raw01_custom() -> None:
    """Callback: mark saved view as custom after a manual control change."""
    st.session_state["raw01_saved_view"] = "None"

inject_app_css()

st.caption("Raw Data Explorer")
st.title("Temporal Explorer: Did mobility change?")

st.markdown(
    """
    This page looks at whether NYC's overall mobility pattern changed around the congestion-pricing
    launch. The top view is fixed and interpretation-first. The lower section lets you explore the same
    trend logic by changing the metric, geography, time window, and temporal bucket.
    """
)

st.divider()

# =============================================================================
# Curated answer view
# =============================================================================

st.subheader("What changed at a glance")

summary_df = get_pre_post_metric_summary(metrics=CORE_METRICS)
trend_df = get_indexed_daily_trends(metrics=CORE_METRICS, smoothing_window=14)

st.markdown(build_raw01_frozen_interpretation(summary_df))

hero_metrics = st.multiselect(
    "Show metrics in hero chart",
    options=CORE_METRICS,
    default=HERO_DEFAULT_METRICS,
    format_func=lambda metric_name: METRIC_LABELS.get(metric_name, metric_name),
    help="Demand metrics are shown by default to keep the hero readable. Add speed metrics when useful.",
)

if not hero_metrics:
    st.warning("Select at least one metric to show in the hero chart.")
    hero_metrics = HERO_DEFAULT_METRICS

fig = go.Figure()

for metric_name in hero_metrics:
    fig.add_trace(
        go.Scatter(
            x=trend_df["date"],
            y=trend_df[f"{metric_name}_index"],
            mode="lines",
            name=METRIC_LABELS.get(metric_name, metric_name),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Date: %{x|%b %d, %Y}<br>"
                "Index: %{y:.1f}<br>"
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

fig.add_hline(
    y=100,
    line_dash="dot",
    line_color="rgba(0, 109, 119, 0.45)",
    annotation_text="Pre-CP average = 100",
    annotation_position="bottom right",
)

fig.update_layout(
    title="Citywide demand indexed to pre-CP average",
    xaxis_title="Date",
    yaxis_title="Index value",
    hovermode="x unified",
    height=560,
)

fig = apply_branding(fig)
fig = _apply_bottom_legend(fig, bottom_margin=115)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Indexed values compare each selected metric against its own pre-CP average. "
    "Demand metrics are shown by default; speed metrics can be added with the selector."
)

taxi_change = summary_df.loc[
    summary_df["metric"] == "taxi_trip_count", "percent_change"
].iloc[0]

subway_change = summary_df.loc[
    summary_df["metric"] == "subway_ridership", "percent_change"
].iloc[0]

fhvhv_speed_change = summary_df.loc[
    summary_df["metric"] == "fhvhv_avg_trip_speed", "percent_change"
].iloc[0]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Taxi trips",
        value=f"{taxi_change:.1f}%",
        delta="Post-CP vs pre-CP daily average",
    )

with col2:
    st.metric(
        label="Subway ridership",
        value=f"{subway_change:.1f}%",
        delta="Post-CP vs pre-CP daily average",
    )

with col3:
    st.metric(
        label="FHVHV average speed",
        value=f"{fhvhv_speed_change:.1f}%",
        delta="Post-CP vs pre-CP daily average",
    )

with st.expander("Show pre/post summary across core metrics", expanded=False):
    _display_summary_table(summary_df)

with st.expander("How to read this view", expanded=False):
    st.markdown(
        """
        - The chart is **descriptive**, not causal. It shows how observed mobility changed before
          versus after congestion pricing began.
        - The post-CP period is shorter than the pre-CP period, so the summary table compares
          **daily averages**, not total period counts.
        - Count metrics are aggregated by summing observed activity.
        - Speed metrics use weighted averages where possible: Taxi speed is weighted by Taxi trips,
          FHVHV speed is weighted by FHVHV trips, and Bus speed is weighted by Bus trip count.
        - The **Gateway + adjacent** option combines gateway-to-CBD zones with zones adjacent to the CBD,
          matching the broader geography framing used in the report.
        - Trend guides in the Explore section are optional visual aids. A full-period fitted line summarizes
          the selected window with one simple direction; separate pre/post lines can reveal differences
          across the CP boundary, but they can also overstate discontinuity in noisy series.
        - Later forecasting and counterfactual pages will handle the stronger question of what mobility
          might have looked like without congestion pricing.
        """
    )

st.divider()

# =============================================================================
# Explore view
# =============================================================================

st.subheader("Explore the pattern yourself")

st.markdown(
    """
    Start with a saved view, or adjust the controls to test whether the same pattern holds for a
    different metric, geography, temporal bucket, or date window.
    """
)

_initialize_raw01_controls()

st.selectbox(
    "Start with a saved view",
    options=RAW01_SAVED_VIEW_OPTIONS,
    key="raw01_saved_view",
    on_change=_apply_raw01_saved_view,
)

active_saved_view = st.session_state.get("raw01_saved_view", "None")

if active_saved_view == "None":
    st.info("Custom view: the controls below no longer match a saved view.")
else:
    st.info(INTERESTING_VIEWS[active_saved_view]["interpretation"])

trend_guide = st.selectbox(
    "Trend guide",
    options=[
        "None",
        "Full-period trend line",
        "Separate pre/post trend lines",
    ],
    index=0,
    help=(
        "Optional visual guide for noisy series. Full-period shows one simple trend across the "
        "selected window; pre/post shows separate fits on each side of the CP start date."
    ),
)

filter_values = get_available_filter_values()

if "raw01_borough" not in st.session_state and filter_values["boroughs"]:
    st.session_state["raw01_borough"] = filter_values["boroughs"][0]

if (
    "raw01_cbd_spatial_category" not in st.session_state
    and filter_values["cbd_spatial_categories"]
):
    st.session_state["raw01_cbd_spatial_category"] = filter_values["cbd_spatial_categories"][0]

metric = st.selectbox(
    "Metric",
    options=CORE_METRICS,
    format_func=lambda metric_name: METRIC_LABELS.get(metric_name, metric_name),
    key="raw01_metric",
    on_change=_mark_raw01_custom,
)

control_col1, control_col2, control_col3 = st.columns(3)

with control_col1:
    geography_scope = st.selectbox(
        "Geography scope",
        options=RAW01_GEO_OPTIONS,
        key="raw01_geography_scope",
        on_change=_mark_raw01_custom,
    )

with control_col2:
    temporal_bucket_options = ["All temporal buckets"] + TEMPORAL_BUCKET_ORDER

    temporal_bucket = st.selectbox(
        "Temporal bucket",
        options=temporal_bucket_options,
        key="raw01_temporal_bucket",
        on_change=_mark_raw01_custom,
    )

with control_col3:
    smoothing_window_label = st.selectbox(
        "Smoothing",
        options=RAW01_SMOOTHING_OPTIONS,
        key="raw01_smoothing",
        on_change=_mark_raw01_custom,
    )

borough = None
cbd_spatial_category = None

if geography_scope == "Borough":
    borough = st.selectbox(
        "Borough",
        options=filter_values["boroughs"],
        key="raw01_borough",
        on_change=_mark_raw01_custom,
    )

elif geography_scope == "CBD spatial category":
    cbd_spatial_category = st.selectbox(
        "CBD spatial category",
        options=filter_values["cbd_spatial_categories"],
        key="raw01_cbd_spatial_category",
        on_change=_mark_raw01_custom,
    )

date_col1, date_col2 = st.columns(2)

with date_col1:
    date_start = st.date_input(
        "Start date",
        min_value=STUDY_START_DATE.date(),
        max_value=STUDY_END_DATE.date(),
        key="raw01_start_date",
        on_change=_mark_raw01_custom,
    )

with date_col2:
    date_end = st.date_input(
        "End date",
        min_value=STUDY_START_DATE.date(),
        max_value=STUDY_END_DATE.date(),
        key="raw01_end_date",
        on_change=_mark_raw01_custom,
    )

if date_start > date_end:
    st.warning("Start date must be before or equal to end date.")
    st.stop()

smoothing_lookup = {
    "None": None,
    "7-day rolling average": 7,
    "14-day rolling average": 14,
    "28-day rolling average": 28,
}

smoothing_window = smoothing_lookup[smoothing_window_label]

date_range = (
    pd.Timestamp(date_start),
    pd.Timestamp(date_end),
)

selected_daily_df = get_daily_metric_trends(
    metrics=[metric],
    temporal_bucket=temporal_bucket,
    borough=borough,
    cbd_spatial_category=cbd_spatial_category,
    date_range=date_range,
)

selected_daily_df = add_rolling_average(
    selected_daily_df,
    metrics=[metric],
    window=smoothing_window,
)

selected_summary_df = get_pre_post_metric_summary(
    metrics=[metric],
    temporal_bucket=temporal_bucket,
    borough=borough,
    cbd_spatial_category=cbd_spatial_category,
    date_range=date_range,
)

selected_summary_display_df = format_summary_for_display(selected_summary_df)

selected_metric_label = METRIC_LABELS.get(metric, metric)
display_col = f"{metric}_display"

selected_fig = go.Figure()

if smoothing_window is not None:
    selected_fig.add_trace(
        go.Scatter(
            x=selected_daily_df["date"],
            y=selected_daily_df[metric],
            mode="lines",
            name="Daily value",
            line={"color": "rgba(0, 109, 119, 0.25)"},
            hovertemplate=(
                "Date: %{x|%b %d, %Y}<br>"
                f"{selected_metric_label}: "
                + "%{y:,.2f}<br>"
                "<extra></extra>"
            ),
        )
    )

selected_fig.add_trace(
    go.Scatter(
        x=selected_daily_df["date"],
        y=selected_daily_df[display_col],
        mode="lines",
        name=smoothing_window_label if smoothing_window is not None else "Daily value",
        line={"color": BRAND_COLORS["dark_teal"], "width": 3},
        hovertemplate=(
            "Date: %{x|%b %d, %Y}<br>"
            f"{selected_metric_label}: "
            + "%{y:,.2f}<br>"
            "<extra></extra>"
        ),
    )
)

if date_range[0] <= CONGESTION_PRICING_START_DATE <= date_range[1]:
    selected_fig.add_vline(
        x=CONGESTION_PRICING_START_DATE,
        line_dash="dash",
        line_color=BRAND_COLORS["terracotta"],
        annotation_text="CP starts",
        annotation_position="top left",
    )

if trend_guide == "Full-period trend line":
    _add_full_period_trend_line(
        selected_fig,
        selected_daily_df,
        metric_col=metric,
        color=BRAND_COLORS["terracotta"],
    )

elif trend_guide == "Separate pre/post trend lines":
    _add_period_trend_line(
        selected_fig,
        selected_daily_df,
        metric_col=metric,
        label="Pre-CP fitted line",
        period="pre_cp",
        color=BRAND_COLORS["terracotta"],
    )
    _add_period_trend_line(
        selected_fig,
        selected_daily_df,
        metric_col=metric,
        label="Post-CP fitted line",
        period="post_cp",
        color=BRAND_COLORS["dark_teal"],
    )

selected_fig.update_layout(
    title=f"{selected_metric_label} over time",
    xaxis_title="Date",
    yaxis_title=selected_metric_label,
    hovermode="x unified",
    height=500,
)

selected_fig = apply_branding(selected_fig)
selected_fig = _apply_bottom_legend(selected_fig, bottom_margin=105)

st.plotly_chart(selected_fig, use_container_width=True)

st.markdown(
    f"""
    <div class="soft-callout">
        <strong>What to notice:</strong><br>
        {build_selected_view_interpretation(
            selected_summary_df,
            metric=metric,
            geography_scope=geography_scope,
            temporal_bucket=temporal_bucket,
            borough=borough,
            cbd_spatial_category=cbd_spatial_category,
        )}
    </div>
    """,
    unsafe_allow_html=True,
)

selected_change = selected_summary_df["percent_change"].iloc[0]
selected_pre_avg = selected_summary_df["pre_daily_average"].iloc[0]
selected_post_avg = selected_summary_df["post_daily_average"].iloc[0]

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric(
        label="Pre-CP daily average",
        value=f"{selected_pre_avg:,.2f}",
    )

with metric_col2:
    st.metric(
        label="Post-CP daily average",
        value=f"{selected_post_avg:,.2f}",
    )

with metric_col3:
    st.metric(
        label="Post vs pre difference",
        value=f"{selected_change:,.2f}%",
    )

with st.expander("Show selected pre/post summary", expanded=False):
    _display_summary_table(selected_summary_df)
