from __future__ import annotations

import pandas as pd
import streamlit as st

from app.data_access.aggregations import TEMPORAL_BUCKET_LABELS
from app.data_access.loaders import CORE_METRICS, METRIC_LABELS
from app.data_access.spatial_aggregations import (
    ALL_TEMPORAL_BUCKETS_LABEL,
    format_zone_summary_for_display,
    get_zone_pre_post_metric_summary,
    get_zone_rankings,
)
from app.data_access.spatial_visuals import (
    DEMAND_METRICS,
    MAP_CONFIG,
    MEANINGFUL_BASELINE_THRESHOLDS,
    add_reliability_flags,
    build_agreement_map,
    build_continuous_zone_map,
    build_demand_agreement_table,
    build_explorer_interpretation,
    build_hero_interpretation,
    get_agreement_category_counts,
    get_continuous_map_kpis,
    get_kpi_labels,
)
from app.utils.project_branding import inject_app_css


inject_app_css()

st.caption("SPATIAL EXPLORER")
st.title("Where did mobility change?")
st.write(
    "Mobility shifted differently across the city. Start with a curated "
    "spatial story, then explore every core metric and time window."
)

STORY_CONFIGS = {
    "Taxi volume": {
        "metric": "taxi_trip_count",
        "value_column": "absolute_change",
        "title": "Added daily Taxi trips after congestion pricing",
        "apply_threshold": False,
    },
    "Taxi growth": {
        "metric": "taxi_trip_count",
        "value_column": "percent_change",
        "title": "Thresholded post-CP growth in Taxi trips",
        "apply_threshold": True,
    },
    "FHVHV shift": {
        "metric": "fhvhv_trip_count",
        "value_column": "percent_change",
        "title": "Thresholded post-CP change in FHVHV trips",
        "apply_threshold": True,
    },
    "Subway growth": {
        "metric": "subway_ridership",
        "value_column": "percent_change",
        "title": "Thresholded post-CP change in Subway ridership",
        "apply_threshold": True,
    },
    "All modes": {
        "metric": None,
        "value_column": None,
        "title": "How did demand modes move together?",
        "apply_threshold": True,
    },
}

hero_story = st.segmented_control(
    "Choose a spatial story",
    options=list(STORY_CONFIGS),
    default="Taxi volume",
    key="raw03_hero_story",
) or "Taxi volume"

hero_summary = get_zone_pre_post_metric_summary(
    metrics=DEMAND_METRICS,
)
hero_config = STORY_CONFIGS[hero_story]

if hero_story == "All modes":
    agreement = build_demand_agreement_table(hero_summary)
    fig = build_agreement_map(
        agreement,
        title=hero_config["title"],
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config=MAP_CONFIG,
    )

    counts = get_agreement_category_counts(agreement)
    lookup = dict(
        zip(
            counts["agreement_category"].astype(str),
            counts["zone_count"],
        )
    )

    eligible = int(
        agreement[DEMAND_METRICS].notna().all(axis=1).sum()
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Zones with all 3 modes eligible", f"{eligible:,}")
    c2.metric(
        "All modes increased",
        f"{int(lookup.get('All modes increased', 0)):,}",
    )
    c3.metric(
        "Divergent demand pattern",
        f"{int(lookup.get('Divergent demand pattern', 0)):,}",
    )
else:
    fig = build_continuous_zone_map(
        hero_summary,
        metric=hero_config["metric"],
        value_column=hero_config["value_column"],
        title=hero_config["title"],
        apply_threshold=hero_config["apply_threshold"],
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config=MAP_CONFIG,
    )

    kpis = get_continuous_map_kpis(
        hero_summary,
        metric=hero_config["metric"],
        value_column=hero_config["value_column"],
        apply_threshold=hero_config["apply_threshold"],
    )

    labels = get_kpi_labels(hero_config["value_column"])
    suffix = (
        "%"
        if hero_config["value_column"] == "percent_change"
        else ""
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(labels[0], f"{kpis['zones_increasing']:,}")
    c2.metric(
        labels[1],
        f"{kpis['median_change']:+,.2f}{suffix}",
    )
    c3.metric(
        labels[2],
        f"{kpis['largest_value']:+,.2f}{suffix}",
        help=kpis["largest_zone"],
    )

hero_insight = build_hero_interpretation(
    hero_summary,
    story=hero_story,
)
st.markdown(
    f'<div class="insight-callout">{hero_insight}</div>',
    unsafe_allow_html=True,
)

with st.expander("How to read this view", expanded=False):
    st.markdown(
        """
        Maps compare average daily mobility before and after congestion
        pricing. Count metrics are summed by Taxi Zone and day; speed metrics
        use weighted averages where supporting activity volume is available.

        Percentage-change maps apply baseline reliability thresholds. Speed
        thresholds use supporting trip or bus volume—not the speed value itself.
        Low-support zones remain neutral, and change maps use robust
        95th-percentile clipping so isolated extremes do not wash out the broader
        citywide pattern.

        **Divergent demand pattern** identifies zones where Taxi trips and Subway
        ridership increased while FHVHV trips declined. It describes co-movement;
        it does not establish why the modes changed differently.
        """
    )

if hero_story != "All modes":
    st.divider()
    st.subheader("Did the demand modes move together?")
    st.write(
        "**Divergent demand pattern** means Taxi and Subway increased while "
        "FHVHV declined."
    )

    agreement = build_demand_agreement_table(hero_summary)
    fig = build_agreement_map(
        agreement,
        title="Multimodal demand pattern by Taxi Zone",
        height=620,
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config=MAP_CONFIG,
    )

st.divider()
st.header("Explore the spatial pattern")

SAVED_VIEWS = {
    "None": None,
    "Taxi added volume": {
        "metric": "taxi_trip_count",
        "value_mode": "Daily-average change",
        "temporal_bucket": ALL_TEMPORAL_BUCKETS_LABEL,
        "apply_threshold": False,
    },
    "Taxi proportional growth": {
        "metric": "taxi_trip_count",
        "value_mode": "Percent change",
        "temporal_bucket": ALL_TEMPORAL_BUCKETS_LABEL,
        "apply_threshold": True,
    },
    "FHVHV outward shift": {
        "metric": "fhvhv_trip_count",
        "value_mode": "Percent change",
        "temporal_bucket": ALL_TEMPORAL_BUCKETS_LABEL,
        "apply_threshold": True,
    },
    "Subway growth corridors": {
        "metric": "subway_ridership",
        "value_mode": "Percent change",
        "temporal_bucket": ALL_TEMPORAL_BUCKETS_LABEL,
        "apply_threshold": True,
    },
    "Bus-speed change": {
        "metric": "avg_bus_speed",
        "value_mode": "Percent change",
        "temporal_bucket": ALL_TEMPORAL_BUCKETS_LABEL,
        "apply_threshold": True,
    },
    "FHVHV speed change": {
        "metric": "fhvhv_avg_trip_speed",
        "value_mode": "Percent change",
        "temporal_bucket": ALL_TEMPORAL_BUCKETS_LABEL,
        "apply_threshold": True,
    },
}

METRIC_OPTIONS = {
    METRIC_LABELS.get(metric, metric): metric
    for metric in CORE_METRICS
}
METRIC_LABEL_BY_VALUE = {
    value: label
    for label, value in METRIC_OPTIONS.items()
}

VALUE_COLUMNS = {
    "Pre-CP daily average": "pre_daily_average",
    "Post-CP daily average": "post_daily_average",
    "Daily-average change": "absolute_change",
    "Percent change": "percent_change",
}

TEMPORAL_OPTIONS = [
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


def _apply_saved_view() -> None:
    name = st.session_state.raw03_saved_view
    preset = SAVED_VIEWS[name]

    if preset is None:
        return

    st.session_state.raw03_metric = preset["metric"]
    st.session_state.raw03_value_mode = preset["value_mode"]
    st.session_state.raw03_temporal_bucket = preset["temporal_bucket"]
    st.session_state.raw03_apply_threshold = preset["apply_threshold"]


def _mark_custom_view() -> None:
    st.session_state.raw03_saved_view = "None"


if "raw03_saved_view" not in st.session_state:
    st.session_state.raw03_saved_view = "Taxi added volume"
if "raw03_metric" not in st.session_state:
    st.session_state.raw03_metric = "taxi_trip_count"
if "raw03_value_mode" not in st.session_state:
    st.session_state.raw03_value_mode = "Daily-average change"
if "raw03_temporal_bucket" not in st.session_state:
    st.session_state.raw03_temporal_bucket = ALL_TEMPORAL_BUCKETS_LABEL
if "raw03_apply_threshold" not in st.session_state:
    st.session_state.raw03_apply_threshold = True

st.selectbox(
    "Saved view",
    options=list(SAVED_VIEWS),
    key="raw03_saved_view",
    on_change=_apply_saved_view,
)

control1, control2, control3 = st.columns([1.25, 1.15, 1.35])

with control1:
    st.selectbox(
        "Metric",
        options=list(METRIC_OPTIONS.values()),
        format_func=lambda value: METRIC_LABEL_BY_VALUE[value],
        key="raw03_metric",
        on_change=_mark_custom_view,
    )

with control2:
    st.selectbox(
        "Map value",
        options=list(VALUE_COLUMNS),
        key="raw03_value_mode",
        on_change=_mark_custom_view,
    )

with control3:
    st.selectbox(
        "Temporal bucket",
        options=TEMPORAL_OPTIONS,
        format_func=lambda value: (
            value
            if value == ALL_TEMPORAL_BUCKETS_LABEL
            else TEMPORAL_BUCKET_LABELS.get(
                value,
                value.replace("_", " ").title(),
            )
        ),
        key="raw03_temporal_bucket",
        on_change=_mark_custom_view,
    )

if st.session_state.raw03_value_mode == "Percent change":
    st.toggle(
        "Apply reliability threshold",
        key="raw03_apply_threshold",
        on_change=_mark_custom_view,
        help=(
            "Demand thresholds use the metric baseline. Speed thresholds use "
            "supporting trip or bus volume."
        ),
    )
    threshold_active = st.session_state.raw03_apply_threshold
else:
    threshold_active = False
    st.caption(
        "Reliability thresholds apply only to percentage-change maps."
    )

selected_metric = st.session_state.raw03_metric
selected_value_mode = st.session_state.raw03_value_mode
selected_bucket = st.session_state.raw03_temporal_bucket
value_column = VALUE_COLUMNS[selected_value_mode]

explore_summary = get_zone_pre_post_metric_summary(
    metrics=[selected_metric],
    temporal_bucket=selected_bucket,
)

title_parts = [
    METRIC_LABELS.get(selected_metric, selected_metric),
    selected_value_mode.lower(),
]
if selected_bucket != ALL_TEMPORAL_BUCKETS_LABEL:
    title_parts.append(
        TEMPORAL_BUCKET_LABELS.get(
            selected_bucket,
            selected_bucket.replace("_", " ").title(),
        )
    )

fig = build_continuous_zone_map(
    explore_summary,
    metric=selected_metric,
    value_column=value_column,
    title=" · ".join(title_parts),
    apply_threshold=threshold_active,
)
st.plotly_chart(
    fig,
    use_container_width=True,
    config=MAP_CONFIG,
)

kpis = get_continuous_map_kpis(
    explore_summary,
    metric=selected_metric,
    value_column=value_column,
    apply_threshold=threshold_active,
)

labels = get_kpi_labels(value_column)
suffix = "%" if value_column == "percent_change" else ""

first_value = (
    kpis["zones_shown"]
    if value_column in {"pre_daily_average", "post_daily_average"}
    else kpis["zones_increasing"]
)

c1, c2, c3 = st.columns(3)
c1.metric(labels[0], f"{first_value:,}")
c2.metric(
    labels[1],
    (
        f"{kpis['median_change']:+,.2f}{suffix}"
        if pd.notna(kpis["median_change"])
        else "Unknown"
    ),
)
c3.metric(
    labels[2],
    (
        f"{kpis['largest_value']:+,.2f}{suffix}"
        if pd.notna(kpis["largest_value"])
        else "Unknown"
    ),
    help=kpis["largest_zone"],
)

explorer_insight = build_explorer_interpretation(
    explore_summary,
    metric=selected_metric,
    value_column=value_column,
    temporal_bucket=selected_bucket,
    apply_threshold=threshold_active,
)
st.markdown(
    f'<div class="insight-callout">{explorer_insight}</div>',
    unsafe_allow_html=True,
)

with st.expander("Explore zone rankings", expanded=False):
    sort_mode = st.selectbox(
        "Sort",
        [
            "High to low",
            "Low to high",
            "Largest absolute shift",
        ],
        key="raw03_ranking_sort",
    )

    ranking_source = explore_summary.copy()

    if value_column == "percent_change" and threshold_active:
        ranking_source = add_reliability_flags(ranking_source)
        ranking_source = ranking_source[
            ranking_source["eligible_for_percent_change"]
        ]

    rankings = get_zone_rankings(
        ranking_source,
        metric=selected_metric,
        value_column=value_column,
        sort_mode=sort_mode,
        limit=25,
    )

    st.dataframe(
        format_zone_summary_for_display(rankings),
        use_container_width=True,
        hide_index=True,
    )

with st.expander("Reliability details", expanded=False):
    if selected_value_mode == "Percent change":
        threshold = MEANINGFUL_BASELINE_THRESHOLDS[selected_metric]
        support_metric = explore_summary["support_metric"].iloc[0]
        support_label = METRIC_LABELS.get(
            support_metric,
            support_metric.replace("_", " "),
        )
        st.write(
            f"The threshold is a pre-CP daily average of at least "
            f"**{threshold:,.0f}** for **{support_label}**."
        )
    else:
        st.write(
            "Thresholds are not used for level or absolute-change maps."
        )