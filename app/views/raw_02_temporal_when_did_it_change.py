from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data_access.aggregations import (
    CORE_METRICS,
    filter_temporal_bucket_summary_for_metric,
    format_cbd_spatial_category_label,
    format_metric_value,
    format_signed_percent,
    format_temporal_bucket_summary_for_display,
    get_available_filter_values,
    get_temporal_bucket_metric_summary,
)
from app.data_access.loaders import METRIC_LABELS
from app.data_access.mobility_environments import (
    format_mobility_regime_cluster_label,
    get_mobility_regime_cluster_options,
)
from app.utils.project_branding import (
    BRAND_COLORS,
    apply_branding,
    inject_app_css,
    render_chart_insight,
)


DEMAND_METRICS = [
    "taxi_trip_count",
    "subway_ridership",
    "fhvhv_trip_count",
]

MODE_COLORS = {
    "taxi_trip_count": BRAND_COLORS["terracotta"],
    "subway_ridership": BRAND_COLORS["dark_teal"],
    "fhvhv_trip_count": BRAND_COLORS["seafoam"],
}

INTERESTING_VIEWS = {
    "Multimodal demand shifts": {
        "metric": "taxi_trip_count",
        "geography_scope": "Citywide",
        "borough": None,
        "cbd_spatial_category": None,
        "week_part": "All buckets",
        "sort_mode": "High to low",
        "value_mode": "Percent change",
        "interpretation": (
            "The largest demand shifts are concentrated in Taxi activity, especially overnight, "
            "evening, and weekend buckets."
        ),
    },
    "Taxi late-night pattern": {
        "metric": "taxi_trip_count",
        "geography_scope": "Citywide",
        "borough": None,
        "cbd_spatial_category": None,
        "week_part": "All buckets",
        "sort_mode": "Temporal order",
        "value_mode": "Percent change",
        "interpretation": (
            "Shows the full ordered Taxi pattern. Taxi trips rose in every temporal bucket, with the "
            "largest proportional gains in overnight, evening, and weekend periods."
        ),
    },
    "Gateway + adjacent taxi timing": {
        "metric": "taxi_trip_count",
        "geography_scope": "CBD spatial category",
        "borough": None,
        "cbd_spatial_category": "Gateway + adjacent",
        "week_part": "All buckets",
        "sort_mode": "Temporal order",
        "value_mode": "Percent change",
        "interpretation": (
            "Compares the Taxi timing pattern across gateway-to-CBD zones and the zones immediately "
            "adjacent to the CBD."
        ),
    },
    "Subway steady lift": {
        "metric": "subway_ridership",
        "geography_scope": "Citywide",
        "borough": None,
        "cbd_spatial_category": None,
        "week_part": "All buckets",
        "sort_mode": "Temporal order",
        "value_mode": "Percent change",
        "interpretation": (
            "Shows that Subway ridership increased broadly across the temporal structure, but with a "
            "steadier pattern than Taxi."
        ),
    },
    "FHVHV mixed timing": {
        "metric": "fhvhv_trip_count",
        "geography_scope": "Citywide",
        "borough": None,
        "cbd_spatial_category": None,
        "week_part": "All buckets",
        "sort_mode": "Temporal order",
        "value_mode": "Percent change",
        "interpretation": (
            "Shows the more uneven FHVHV pattern: stronger gains in AM/midday/PM buckets, with weaker "
            "or negative movement in some weekend late-day buckets."
        ),
    },
}


def _key_from_label(label: str) -> str:
    """Create a stable Streamlit widget-key suffix from a saved-view label."""
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _format_percent(value: float) -> str:
    """Format a percent value for chart text."""
    if pd.isna(value):
        return ""
    return f"{value:+.2f}%"


def _format_value(value: float) -> str:
    """Format a numeric chart label."""
    if pd.isna(value):
        return ""
    return f"{value:+,.2f}"


def _build_multimodal_demand_chart(demand_df: pd.DataFrame, *, top_n: int = 20) -> go.Figure:
    """Build ranked horizontal bar chart for demand metrics by temporal bucket."""
    chart_df = demand_df.sort_values("percent_change", ascending=False).head(top_n).copy()

    chart_df["mode_bucket_label"] = (
        chart_df["metric_label"] + " — " + chart_df["temporal_bucket_label"]
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=chart_df["percent_change"],
            y=chart_df["mode_bucket_label"],
            orientation="h",
            name="Post-CP % change",
            text=chart_df["percent_change"].map(_format_percent),
            textposition="outside",
            marker={
                "color": chart_df["metric"].map(MODE_COLORS),
                "line": {
                    "color": chart_df["metric"].map(MODE_COLORS),
                    "width": 1,
                },
            },
            customdata=chart_df[
                [
                    "metric_label",
                    "temporal_bucket_label",
                    "pre_daily_average",
                    "post_daily_average",
                    "absolute_change",
                ]
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Bucket: %{customdata[1]}<br>"
                "Percent change: %{x:.2f}%<br>"
                "Pre-CP daily avg: %{customdata[2]:,.2f}<br>"
                "Post-CP daily avg: %{customdata[3]:,.2f}<br>"
                "Abs. change: %{customdata[4]:,.2f}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.add_vline(
        x=0,
        line_dash="dot",
        line_color="rgba(0, 109, 119, 0.55)",
    )

    max_x = chart_df["percent_change"].max()
    if pd.notna(max_x):
        fig.update_xaxes(range=[0, max_x * 1.18])

    fig.update_yaxes(autorange="reversed")

    fig.update_layout(
        title=f"Top {top_n} post-CP demand shifts by mode and temporal bucket",
        xaxis_title="Post-CP vs pre-CP daily average",
        yaxis_title="Mode and temporal bucket",
        height=680,
        showlegend=False,
        margin={"l": 210, "r": 50, "t": 70, "b": 70},
    )

    return apply_branding(fig)


def _sort_selected_bucket_df(
    bucket_df: pd.DataFrame,
    *,
    sort_mode: str,
    value_mode: str,
) -> pd.DataFrame:
    """Sort selected temporal buckets using the metric currently displayed."""
    result = bucket_df.copy()
    value_col = "percent_change" if value_mode == "Percent change" else "absolute_change"

    if sort_mode == "Temporal order":
        result = result.sort_values("temporal_bucket_order")
    elif sort_mode == "High to low":
        result = result.sort_values(value_col, ascending=False)
    elif sort_mode == "Low to high":
        result = result.sort_values(value_col, ascending=True)
    elif sort_mode == "Largest absolute shift":
        result = result.assign(
            _selected_change_magnitude=result[value_col].abs()
        ).sort_values("_selected_change_magnitude", ascending=False)
    else:
        raise ValueError(
            "sort_mode must be one of: 'Temporal order', 'High to low', "
            "'Low to high', 'Largest absolute shift'"
        )

    return result.drop(columns=["_selected_change_magnitude"], errors="ignore").reset_index(drop=True)


def _build_selected_bucket_chart(
    bucket_df: pd.DataFrame,
    *,
    metric: str,
    metric_label: str,
    value_mode: str,
) -> go.Figure:
    """Build the interactive selected-metric temporal-bucket chart."""
    chart_df = bucket_df.copy()

    if value_mode == "Percent change":
        x_col = "percent_change"
        x_title = "Post-CP vs pre-CP daily average"
        text_values = chart_df[x_col].map(_format_percent)
        hover_value_label = "Percent change"
        hover_value_format = "%{x:.2f}%"
    elif value_mode == "Daily-average change":
        x_col = "absolute_change"
        x_title = "Change in daily average"
        text_values = chart_df[x_col].map(_format_value)
        hover_value_label = "Daily-average change"
        hover_value_format = "%{x:,.2f}"
    else:
        raise ValueError("value_mode must be 'Percent change' or 'Daily-average change'")

    positive_color = BRAND_COLORS["dark_teal"]
    negative_color = BRAND_COLORS["terracotta"]
    neutral_color = BRAND_COLORS["seafoam"]

    bar_colors = chart_df[x_col].map(
        lambda value: (
            positive_color
            if pd.notna(value) and value > 0
            else negative_color
            if pd.notna(value) and value < 0
            else neutral_color
        )
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=chart_df[x_col],
            y=chart_df["temporal_bucket_label"],
            orientation="h",
            name=value_mode,
            text=text_values,
            textposition="outside",
            marker={
                "color": bar_colors,
                "line": {"color": bar_colors, "width": 1},
            },
            customdata=chart_df[
                [
                    "temporal_bucket_label",
                    "pre_daily_average",
                    "post_daily_average",
                    "absolute_change",
                    "percent_change",
                ]
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"{hover_value_label}: {hover_value_format}<br>"
                "Pre-CP daily avg: %{customdata[1]:,.2f}<br>"
                "Post-CP daily avg: %{customdata[2]:,.2f}<br>"
                "Daily-average change: %{customdata[3]:,.2f}<br>"
                "Percent change: %{customdata[4]:.2f}%<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.add_vline(
        x=0,
        line_dash="dot",
        line_color="rgba(0, 109, 119, 0.55)",
    )

    min_x = chart_df[x_col].min()
    max_x = chart_df[x_col].max()

    if pd.notna(min_x) and pd.notna(max_x):
        span = max(abs(min_x), abs(max_x))
        if span > 0:
            fig.update_xaxes(range=[min_x - span * 0.18, max_x + span * 0.18])

    fig.update_yaxes(autorange="reversed")

    fig.update_layout(
        title=f"{metric_label}: {value_mode.lower()} by temporal bucket",
        xaxis_title=x_title,
        yaxis_title="Temporal bucket",
        height=560,
        showlegend=False,
        margin={"l": 150, "r": 50, "t": 70, "b": 70},
    )

    return apply_branding(fig)

def _geography_text(
    *,
    geography_scope: str,
    borough: str | None,
    cbd_spatial_category: str | None,
    mobility_regime_cluster_label: int | None = None,
) -> str:
    """Build a readable geography phrase."""
    if geography_scope == "Citywide":
        return "citywide"
    if geography_scope == "Borough" and borough:
        return f"in {borough}"
    if geography_scope == "CBD spatial category" and cbd_spatial_category:
        return f"for {format_cbd_spatial_category_label(cbd_spatial_category)} zones"
    if (
        geography_scope == "Mobility regime cluster"
        and mobility_regime_cluster_label is not None
    ):
        return (
            f"for {format_mobility_regime_cluster_label(mobility_regime_cluster_label)} zones"
        )
    return "for the selected geography"


def _build_selected_view_interpretation(
    bucket_df: pd.DataFrame,
    *,
    metric_label: str,
    geography_scope: str,
    week_part: str,
    borough: str | None = None,
    cbd_spatial_category: str | None = None,
    mobility_regime_cluster_label: int | None = None,
) -> str:
    """Build sign-aware interpretation for the selected temporal-bucket pattern."""
    if bucket_df.empty:
        return "This selection does not have enough data to summarize."

    valid_df = bucket_df.dropna(subset=["percent_change"])
    if valid_df.empty:
        return "This selection does not have enough observed pre/post data to summarize."

    positive_count = int((valid_df["percent_change"] > 0).sum())
    negative_count = int((valid_df["percent_change"] < 0).sum())
    total_count = len(valid_df)

    top_row = valid_df.sort_values("percent_change", ascending=False).iloc[0]
    bottom_row = valid_df.sort_values("percent_change", ascending=True).iloc[0]

    geography = _geography_text(
        geography_scope=geography_scope,
        borough=borough,
        cbd_spatial_category=cbd_spatial_category,
        mobility_regime_cluster_label=mobility_regime_cluster_label,
    )

    week_text = (
        "across all temporal buckets"
        if week_part == "All buckets"
        else f"across {week_part.lower().replace(' only', '')} buckets"
    )

    if positive_count == total_count:
        return (
            f"For this selection, <strong>{metric_label} was higher post-CP in all "
            f"{total_count} buckets</strong> {geography} {week_text}. "
            f"The largest proportional increase was {top_row['temporal_bucket_label']} "
            f"({format_signed_percent(top_row['percent_change'])}), moving from "
            f"{format_metric_value(top_row['pre_daily_average'])} to "
            f"{format_metric_value(top_row['post_daily_average'])} per day. "
            f"The smallest increase was {bottom_row['temporal_bucket_label']} "
            f"({format_signed_percent(bottom_row['percent_change'])})."
        )

    if negative_count == total_count:
        return (
            f"For this selection, <strong>{metric_label} was lower post-CP in all "
            f"{total_count} buckets</strong> {geography} {week_text}. "
            f"The smallest decline was {top_row['temporal_bucket_label']} "
            f"({format_signed_percent(top_row['percent_change'])}), moving from "
            f"{format_metric_value(top_row['pre_daily_average'])} to "
            f"{format_metric_value(top_row['post_daily_average'])} per day. "
            f"The largest decline was {bottom_row['temporal_bucket_label']} "
            f"({format_signed_percent(bottom_row['percent_change'])})."
        )

    return (
        f"For this selection, <strong>{metric_label} was higher post-CP in "
        f"{positive_count} of {total_count} buckets</strong> {geography} {week_text}. "
        f"The largest increase was {top_row['temporal_bucket_label']} "
        f"({format_signed_percent(top_row['percent_change'])}), moving from "
        f"{format_metric_value(top_row['pre_daily_average'])} to "
        f"{format_metric_value(top_row['post_daily_average'])} per day. "
        f"The largest decline was {bottom_row['temporal_bucket_label']} "
        f"({format_signed_percent(bottom_row['percent_change'])})."
    )


def _build_hero_takeaway(demand_summary: pd.DataFrame) -> str:
    """Summarize the leading demand shift directly from the hero data."""
    valid = demand_summary.dropna(subset=["percent_change"]).copy()
    if valid.empty:
        return "The hero view does not contain enough reliable pre/post data to summarize."

    leader = valid.sort_values("percent_change", ascending=False).iloc[0]
    top_five = valid.nlargest(5, "percent_change")
    taxi_top_five = int(top_five["metric"].eq("taxi_trip_count").sum())
    positive_buckets = (
        valid.groupby("metric", observed=True)["percent_change"]
        .apply(lambda values: int((values > 0).sum()))
        .to_dict()
    )

    return (
        f"**{METRIC_LABELS.get(leader['metric'], leader['metric'])} during "
        f"{leader['temporal_bucket_label']}** had the largest displayed demand "
        f"shift at **{float(leader['percent_change']):+.1f}%**. Taxi accounts for "
        f"**{taxi_top_five} of the five largest shifts**; Subway is higher in "
        f"**{positive_buckets.get('subway_ridership', 0)} of 10 buckets**, while "
        f"FHVHV is higher in **{positive_buckets.get('fhvhv_trip_count', 0)} of 10**."
    )


def _selected_metric_card_labels(bucket_df: pd.DataFrame) -> dict[str, str]:
    """Return card labels that match all-positive, all-negative, or mixed selections."""
    valid_df = bucket_df.dropna(subset=["percent_change"])
    positive_count = int((valid_df["percent_change"] > 0).sum())
    negative_count = int((valid_df["percent_change"] < 0).sum())
    total_count = len(valid_df)

    if total_count and positive_count == total_count:
        return {
            "count_label": "Buckets higher post-CP",
            "count_value": f"{positive_count} of {total_count}",
            "top_label": "Largest increase",
            "bottom_label": "Smallest increase",
        }

    if total_count and negative_count == total_count:
        return {
            "count_label": "Buckets lower post-CP",
            "count_value": f"{negative_count} of {total_count}",
            "top_label": "Smallest decline",
            "bottom_label": "Largest decline",
        }

    return {
        "count_label": "Buckets higher post-CP",
        "count_value": f"{positive_count} of {total_count}",
        "top_label": "Largest increase",
        "bottom_label": "Largest decline",
    }


def _display_summary_table(summary_df: pd.DataFrame) -> None:
    """Display the temporal-bucket summary with controlled precision."""
    display_table = format_temporal_bucket_summary_for_display(
        summary_df
    )

    st.dataframe(
        display_table,
        width="stretch",
        hide_index=True,
        column_config={
            "Pre-CP daily avg": st.column_config.NumberColumn(
                "Pre-CP daily avg",
                format="%,.1f",
            ),
            "Post-CP daily avg": st.column_config.NumberColumn(
                "Post-CP daily avg",
                format="%,.1f",
            ),
            "Abs. change": st.column_config.NumberColumn(
                "Abs. change",
                format="%+,.1f",
            ),
            "% change": st.column_config.NumberColumn(
                "% change",
                format="%+.1f%%",
            ),
            "Pre observed days": st.column_config.NumberColumn(
                "Pre observed days",
                format="%d",
            ),
            "Post observed days": st.column_config.NumberColumn(
                "Post observed days",
                format="%d",
            ),
        },
    )


def _build_mode_count_summary(demand_df: pd.DataFrame) -> dict[str, int]:
    """Count how many temporal buckets increased by mode."""
    result = {}

    for metric in DEMAND_METRICS:
        metric_df = demand_df[demand_df["metric"] == metric]
        result[metric] = int((metric_df["percent_change"] > 0).sum())

    return result




RAW02_DEFAULT_SAVED_VIEW = "Multimodal demand shifts"
RAW02_SAVED_VIEW_OPTIONS = ["None"] + list(INTERESTING_VIEWS.keys())
RAW02_GEO_OPTIONS = [
    "Citywide",
    "Borough",
    "CBD spatial category",
    "Mobility regime cluster",
]


def _format_raw02_geography_scope(value: str) -> str:
    """Keep internal geography keys stable while using app-facing terminology."""
    if value == "Mobility regime cluster":
        return "Mobility environment"
    return value
RAW02_WEEK_PART_OPTIONS = ["All buckets", "Weekday only", "Weekend only"]
RAW02_SORT_MODE_OPTIONS = [
    "Temporal order",
    "High to low",
    "Low to high",
    "Largest absolute shift",
]
RAW02_VALUE_MODE_OPTIONS = ["Percent change", "Daily-average change"]


def _set_raw02_controls_from_preset(preset_name: str) -> None:
    """Apply a saved view to the interactive controls."""
    preset = INTERESTING_VIEWS[preset_name]

    st.session_state["raw02_metric"] = preset["metric"]
    st.session_state["raw02_geography_scope"] = preset["geography_scope"]
    st.session_state["raw02_week_part"] = preset["week_part"]
    st.session_state["raw02_sort_mode"] = preset["sort_mode"]
    st.session_state["raw02_value_mode"] = preset["value_mode"]

    if preset.get("borough") is not None:
        st.session_state["raw02_borough"] = preset["borough"]

    if preset.get("cbd_spatial_category") is not None:
        st.session_state["raw02_cbd_spatial_category"] = preset["cbd_spatial_category"]


def _initialize_raw02_controls() -> None:
    """Initialize saved-view state once per session."""
    if "raw02_saved_view" not in st.session_state:
        st.session_state["raw02_saved_view"] = RAW02_DEFAULT_SAVED_VIEW
        _set_raw02_controls_from_preset(RAW02_DEFAULT_SAVED_VIEW)


def _apply_raw02_saved_view() -> None:
    """Callback: load preset values when the saved-view selector changes."""
    selected = st.session_state.get("raw02_saved_view", "None")

    if selected != "None":
        _set_raw02_controls_from_preset(selected)


def _mark_raw02_custom() -> None:
    """Callback: mark saved view as custom after a manual control change."""
    st.session_state["raw02_saved_view"] = "None"

inject_app_css()

st.caption("TIME-OF-DAY PATTERNS")
st.title("When did mobility change most?")

st.write(
    "Compare pre/post differences across weekday, weekend, and time-of-day buckets to see "
    "when each mobility mode changed most."
)

st.divider()

# =============================================================================
# Curated answer view
# =============================================================================

st.header("When were the largest demand shifts?")

st.markdown(
    """
    The largest temporal-bucket demand shifts were concentrated in **Taxi trips**, especially in
    late-night, evening, and weekend windows. Subway ridership also increased across the full bucket
    structure, but more evenly. FHVHV trips showed a more mixed pattern, with stronger gains in
    AM/midday/PM periods and weaker late-night or weekend-evening movement.
    """
)

summary_df = get_temporal_bucket_metric_summary(metrics=CORE_METRICS)
demand_summary_df = summary_df[summary_df["metric"].isin(DEMAND_METRICS)].copy()

fig = _build_multimodal_demand_chart(demand_summary_df, top_n=20)
st.plotly_chart(fig, width="stretch")
render_chart_insight(_build_hero_takeaway(demand_summary_df))

st.caption(
    "Bars compare post-CP daily averages against pre-CP daily averages within each temporal bucket. "
    "The hero chart ranks demand metrics only: Taxi trips, Subway ridership, and FHVHV trips."
)

mode_bucket_counts = _build_mode_count_summary(demand_summary_df)

taxi_top_row = demand_summary_df[
    demand_summary_df["metric"] == "taxi_trip_count"
].sort_values("percent_change", ascending=False).iloc[0]

subway_top_row = demand_summary_df[
    demand_summary_df["metric"] == "subway_ridership"
].sort_values("percent_change", ascending=False).iloc[0]

fhvhv_top_row = demand_summary_df[
    demand_summary_df["metric"] == "fhvhv_trip_count"
].sort_values("percent_change", ascending=False).iloc[0]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Taxi buckets higher post-CP",
        value=f"{mode_bucket_counts['taxi_trip_count']} of 10",
        delta=f"Top: {taxi_top_row['temporal_bucket_label']} ({taxi_top_row['percent_change']:.1f}%)",
    )

with col2:
    st.metric(
        label="Subway buckets higher post-CP",
        value=f"{mode_bucket_counts['subway_ridership']} of 10",
        delta=f"Top: {subway_top_row['temporal_bucket_label']} ({subway_top_row['percent_change']:.1f}%)",
    )

with col3:
    st.metric(
        label="FHVHV buckets higher post-CP",
        value=f"{mode_bucket_counts['fhvhv_trip_count']} of 10",
        delta=f"Top: {fhvhv_top_row['temporal_bucket_label']} ({fhvhv_top_row['percent_change']:.1f}%)",
    )

with st.expander("Show supporting data tables", expanded=False):
    st.markdown("**Top ranked demand shifts**")
    top_demand_display_df = demand_summary_df.sort_values(
        "percent_change",
        ascending=False,
    ).head(20)
    _display_summary_table(top_demand_display_df)

    st.markdown("**Taxi trips by temporal bucket**")
    taxi_bucket_df = filter_temporal_bucket_summary_for_metric(
        summary_df,
        metric="taxi_trip_count",
        week_part="All buckets",
        sort_mode="Temporal order",
    )
    _display_summary_table(taxi_bucket_df)

with st.expander("How to read this view", expanded=False):
    st.markdown(
        """
        - Each row compares the pre-CP and post-CP **daily average** for one temporal bucket.
        - The hero chart uses demand/activity metrics only: Taxi trips, Subway ridership, and FHVHV trips.
        - Speed metrics are excluded from the hero because they answer a different question: service performance,
          not demand timing.
        - The **Gateway + adjacent** option combines gateway-to-CBD zones with zones immediately adjacent
          to the CBD so their timing patterns can be evaluated together.
        - The temporal buckets combine weekday/weekend with time-of-day windows.
        - Count metrics are aggregated by summing observed activity within each date and temporal bucket.
        - This page is descriptive. It shows when observed differences are largest, not why they happened.
        """
    )

st.divider()

# =============================================================================
# Explore view
# =============================================================================

st.header("Explore time-of-day patterns")

st.markdown(
    """
    Start with a saved view, or adjust the controls to test whether the timing pattern changes by
    mode, geography, weekday/weekend grouping, or ranking method.
    """
)

_initialize_raw02_controls()

st.selectbox(
    "Start with a saved view",
    options=RAW02_SAVED_VIEW_OPTIONS,
    key="raw02_saved_view",
    on_change=_apply_raw02_saved_view,
)

active_saved_view = st.session_state.get("raw02_saved_view", "None")

if active_saved_view == "None":
    st.info("Custom view: the controls below no longer match a saved view.")
else:
    st.info(INTERESTING_VIEWS[active_saved_view]["interpretation"])

filter_values = get_available_filter_values()

if "raw02_borough" not in st.session_state and filter_values["boroughs"]:
    st.session_state["raw02_borough"] = filter_values["boroughs"][0]

if (
    "raw02_cbd_spatial_category" not in st.session_state
    and filter_values["cbd_spatial_categories"]
):
    st.session_state["raw02_cbd_spatial_category"] = filter_values["cbd_spatial_categories"][0]

control_col1, control_col2, control_col3 = st.columns(3)

with control_col1:
    metric = st.selectbox(
        "Metric",
        options=CORE_METRICS,
        format_func=lambda metric_name: METRIC_LABELS.get(metric_name, metric_name),
        key="raw02_metric",
        on_change=_mark_raw02_custom,
    )

with control_col2:
    geography_scope = st.selectbox(
        "Geography scope",
        options=RAW02_GEO_OPTIONS,
        format_func=_format_raw02_geography_scope,
        key="raw02_geography_scope",
        on_change=_mark_raw02_custom,
    )

with control_col3:
    week_part = st.selectbox(
        "Weekday/weekend filter",
        options=RAW02_WEEK_PART_OPTIONS,
        key="raw02_week_part",
        on_change=_mark_raw02_custom,
    )

borough = None
cbd_spatial_category = None
mobility_regime_cluster_label = None

if geography_scope == "Borough":
    borough = st.selectbox(
        "Borough",
        options=filter_values["boroughs"],
        key="raw02_borough",
        on_change=_mark_raw02_custom,
    )

elif geography_scope == "CBD spatial category":
    cbd_spatial_category = st.selectbox(
        "CBD spatial category",
        options=filter_values["cbd_spatial_categories"],
        key="raw02_cbd_spatial_category",
        on_change=_mark_raw02_custom,
    )
elif geography_scope == "Mobility regime cluster":
    mobility_regime_cluster_label = st.selectbox(
        "Mobility environment",
        options=get_mobility_regime_cluster_options(),
        format_func=format_mobility_regime_cluster_label,
        key="raw02_mobility_regime_cluster",
        on_change=_mark_raw02_custom,
    )

control_col4, control_col5 = st.columns(2)

with control_col4:
    sort_mode = st.selectbox(
        "Sort order",
        options=RAW02_SORT_MODE_OPTIONS,
        key="raw02_sort_mode",
        help=(
            "High to low and low to high sort by the currently displayed value mode. "
            "Largest absolute shift sorts by magnitude regardless of sign."
        ),
        on_change=_mark_raw02_custom,
    )

with control_col5:
    value_mode = st.selectbox(
        "Value mode",
        options=RAW02_VALUE_MODE_OPTIONS,
        key="raw02_value_mode",
        on_change=_mark_raw02_custom,
    )

selected_summary_df = get_temporal_bucket_metric_summary(
    metrics=CORE_METRICS,
    borough=borough,
    cbd_spatial_category=cbd_spatial_category,
    mobility_regime_cluster_label=mobility_regime_cluster_label,
)

selected_bucket_df = filter_temporal_bucket_summary_for_metric(
    selected_summary_df,
    metric=metric,
    week_part=week_part,
    sort_mode="Temporal order",
)

selected_bucket_df = _sort_selected_bucket_df(
    selected_bucket_df,
    sort_mode=sort_mode,
    value_mode=value_mode,
)

selected_metric_label = METRIC_LABELS.get(metric, metric)

selected_fig = _build_selected_bucket_chart(
    selected_bucket_df,
    metric=metric,
    metric_label=selected_metric_label,
    value_mode=value_mode,
)

st.plotly_chart(selected_fig, width="stretch")

render_chart_insight(
    _build_selected_view_interpretation(
        selected_bucket_df,
        metric_label=selected_metric_label,
        geography_scope=geography_scope,
        week_part=week_part,
        borough=borough,
        cbd_spatial_category=cbd_spatial_category,
        mobility_regime_cluster_label=mobility_regime_cluster_label,
    )
)

valid_selected_df = selected_bucket_df.dropna(subset=["percent_change"])
if valid_selected_df.empty:
    st.warning("This selection does not have enough observed pre/post data to summarize.")
else:
    top_selected_row = valid_selected_df.sort_values(
        "percent_change",
        ascending=False,
    ).iloc[0]
    bottom_selected_row = valid_selected_df.sort_values(
        "percent_change",
        ascending=True,
    ).iloc[0]
    card_labels = _selected_metric_card_labels(valid_selected_df)

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric(
            label=card_labels["count_label"],
            value=card_labels["count_value"],
        )

    with metric_col2:
        st.metric(
            label=card_labels["top_label"],
            value=f"{top_selected_row['percent_change']:.1f}%",
            delta=top_selected_row["temporal_bucket_label"],
        )

    with metric_col3:
        st.metric(
            label=card_labels["bottom_label"],
            value=f"{bottom_selected_row['percent_change']:.1f}%",
            delta=bottom_selected_row["temporal_bucket_label"],
        )

with st.expander("Show selected bucket data", expanded=False):
    _display_summary_table(selected_bucket_df)
