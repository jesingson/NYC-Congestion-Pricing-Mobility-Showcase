from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data_access.loaders import CORE_METRICS
from app.data_access.mode_relationships import (
    NOTEBOOK_ROLLING_MIN_MATCHED_DAYS as DEFAULT_MIN_MATCHED_DAYS,
    NOTEBOOK_ROLLING_STEP_DAYS as DEFAULT_STEP_DAYS,
    NOTEBOOK_ROLLING_WINDOW_DAYS as DEFAULT_WINDOW_DAYS,
    build_bucket_rolling_relationship_data,
    build_citywide_rolling_relationship_data,
    build_notebook_rolling_relationship_data as build_default_rolling_relationship_data,
)
from app.data_access.spatial_aggregations import (
    ALL_TEMPORAL_BUCKETS_LABEL,
)
from app.utils.project_branding import (
    BRAND_COLORS,
    apply_branding,
    inject_app_css,
    render_chart_insight,
)


inject_app_css()

CP_START_DATE = pd.Timestamp("2025-01-05")

METRIC_LABELS = {
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

WINDOW_OPTIONS = [60, 90, 120]

HERO_PAIR_COLORS = {
    ("taxi_trip_count", "fhvhv_trip_count"): BRAND_COLORS["dark_teal"],
    ("taxi_trip_count", "subway_ridership"): BRAND_COLORS["terracotta"],
}

SAVED_VIEWS = {
    "Taxi and FHVHV alignment": {
        "description": (
            "Track how closely Taxi and FHVHV demand moved together over time."
        ),
        "metric_a": "taxi_trip_count",
        "metric_b": "fhvhv_trip_count",
        "temporal_bucket": ALL_TEMPORAL_BUCKETS_LABEL,
        "rolling_window_days": 90,
    },
    "Taxi and subway alignment": {
        "description": (
            "Track how the relationship between Taxi demand and Subway "
            "Ridership evolved."
        ),
        "metric_a": "taxi_trip_count",
        "metric_b": "subway_ridership",
        "temporal_bucket": ALL_TEMPORAL_BUCKETS_LABEL,
        "rolling_window_days": 90,
    },
    "Weekend overnight taxi and subway": {
        "description": (
            "Explore the volatile weekend overnight relationship between "
            "Taxi Trips and Subway Ridership."
        ),
        "metric_a": "taxi_trip_count",
        "metric_b": "subway_ridership",
        "temporal_bucket": "weekend_overnight",
        "rolling_window_days": 90,
    },
    "Weekend AM taxi and subway": {
        "description": (
            "Explore the weakening weekend morning relationship between "
            "Taxi Trips and Subway Ridership."
        ),
        "metric_a": "taxi_trip_count",
        "metric_b": "subway_ridership",
        "temporal_bucket": "weekend_am_peak",
        "rolling_window_days": 90,
    },
    "Weekend PM taxi and FHVHV": {
        "description": (
            "Explore the strengthening weekend PM relationship between "
            "Taxi and FHVHV demand."
        ),
        "metric_a": "taxi_trip_count",
        "metric_b": "fhvhv_trip_count",
        "temporal_bucket": "weekend_pm_peak",
        "rolling_window_days": 90,
    },
    "Weekend evening taxi and FHVHV": {
        "description": (
            "Explore the consistently strong weekend evening relationship "
            "between Taxi and FHVHV demand."
        ),
        "metric_a": "taxi_trip_count",
        "metric_b": "fhvhv_trip_count",
        "temporal_bucket": "weekend_evening",
        "rolling_window_days": 90,
    },
    "Surface-mode speeds": {
        "description": (
            "Track the highly stable relationship between Taxi and FHVHV "
            "average speeds."
        ),
        "metric_a": "taxi_avg_trip_speed",
        "metric_b": "fhvhv_avg_trip_speed",
        "temporal_bucket": ALL_TEMPORAL_BUCKETS_LABEL,
        "rolling_window_days": 90,
    },
}

SAVED_VIEW_OPTIONS = [*SAVED_VIEWS.keys(), "Custom"]


def _apply_saved_view(saved_view_name: str) -> None:
    if saved_view_name == "Custom":
        return

    configuration = SAVED_VIEWS[saved_view_name]

    st.session_state["raw08_metric_a"] = configuration["metric_a"]
    st.session_state["raw08_metric_b"] = configuration["metric_b"]
    st.session_state["raw08_temporal_bucket"] = configuration["temporal_bucket"]
    st.session_state["raw08_rolling_window_days"] = configuration[
        "rolling_window_days"
    ]


def _mark_saved_view_custom() -> None:
    st.session_state["raw08_saved_view"] = "Custom"


def _metric_label(metric: str) -> str:
    return METRIC_LABELS.get(
        metric,
        metric.replace("_", " ").title(),
    )


def _format_correlation(value: object) -> str:
    if pd.isna(value):
        return "Unavailable"

    return f"{float(value):+.3f}"


def _apply_chart_branding(fig: go.Figure) -> go.Figure:
    fig.update_layout(title={"text": " ", "x": 0, "y": 1})
    fig = apply_branding(fig)

    fig.update_layout(
        title={
            "text": "",
            "x": 0,
            "y": 1,
            "pad": {"t": 0, "b": 0, "l": 0, "r": 0},
        }
    )

    clean_annotations = []

    for annotation in fig.layout.annotations or []:
        text = str(getattr(annotation, "text", "")).strip()

        if text.lower() in {"", "undefined", "none", "nan"}:
            continue

        clean_annotations.append(annotation)

    fig.update_layout(annotations=clean_annotations)

    return fig


def build_rolling_relationship_chart(
    rolling_data: pd.DataFrame,
    *,
    height: int,
) -> go.Figure:
    fig = go.Figure()

    if rolling_data.empty:
        return _apply_chart_branding(fig)

    pair_order = rolling_data["pair_label"].drop_duplicates().tolist()

    fallback_colors = [
        BRAND_COLORS["dark_teal"],
        BRAND_COLORS["terracotta"],
        "#5B5F97",
        BRAND_COLORS["seafoam"],
        "#6B8E23",
    ]

    for index, pair_label in enumerate(pair_order):
        pair_df = (
            rolling_data[
                rolling_data["pair_label"].eq(pair_label)
            ]
            .sort_values("window_midpoint")
            .copy()
        )

        customdata = np.column_stack(
            [
                pair_df["window_start"].dt.strftime("%Y-%m-%d"),
                pair_df["window_end"].dt.strftime("%Y-%m-%d"),
                pair_df["matched_days"],
                pair_df["required_matched_days"],
                pair_df["coverage_share"],
            ]
        )

        pair_key = (
            str(pair_df.iloc[0]["metric_x"]),
            str(pair_df.iloc[0]["metric_y"]),
        )

        color = HERO_PAIR_COLORS.get(
            pair_key,
            fallback_colors[index % len(fallback_colors)],
        )

        fig.add_trace(
            go.Scatter(
                x=pair_df["window_midpoint"],
                y=pair_df["rolling_correlation"],
                mode="lines+markers",
                name=pair_label,
                line={"color": color, "width": 3},
                marker={
                    "size": 7,
                    "symbol": "circle-open",
                    "color": color,
                    "line": {"color": color, "width": 1.5},
                },
                customdata=customdata,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Midpoint: %{x|%Y-%m-%d}<br>"
                    "Window: %{customdata[0]} to %{customdata[1]}<br>"
                    "Rolling correlation: %{y:.3f}<br>"
                    "Matched days: %{customdata[2]:.0f}<br>"
                    "Required days: %{customdata[3]:.0f}<br>"
                    "Coverage: %{customdata[4]:.1%}"
                    "<extra></extra>"
                ),
                connectgaps=False,
            )
        )

    fig.add_hline(
        y=0,
        line={"color": "rgba(40,40,40,0.58)", "width": 1.5},
    )

    fig.add_vline(
        x=CP_START_DATE,
        line={
            "color": BRAND_COLORS["terracotta"],
            "width": 2,
            "dash": "dash",
        },
        annotation_text="Congestion pricing begins",
        annotation_position="top left",
    )

    fig.update_xaxes(
        title_text="Window midpoint",
        showgrid=True,
        gridcolor="rgba(0,109,119,0.10)",
        zeroline=False,
    )

    fig.update_yaxes(
        title_text="Rolling Spearman correlation",
        range=[-1.05, 1.05],
        tickformat=".1f",
        showgrid=True,
        gridcolor="rgba(0,109,119,0.10)",
        zeroline=False,
    )

    fig = _apply_chart_branding(fig)

    fig.update_layout(
        height=height,
        margin={"l": 15, "r": 30, "t": 35, "b": 95},
        legend={
            "title": {"text": ""},
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.16,
            "yanchor": "top",
        },
        hovermode="x unified",
    )

    return fig


def _summarize_pair(pair_data: pd.DataFrame) -> dict[str, object]:
    valid = (
        pair_data[
            pair_data["rolling_correlation"].notna()
        ]
        .sort_values("window_midpoint")
        .copy()
    )

    if valid.empty:
        return {
            "latest": np.nan,
            "minimum": np.nan,
            "maximum": np.nan,
            "pre_mean": np.nan,
            "post_mean": np.nan,
            "shift": np.nan,
            "sign_changes": 0,
            "valid_windows": 0,
        }

    pre = valid[valid["window_midpoint"].lt(CP_START_DATE)]
    post = valid[valid["window_midpoint"].ge(CP_START_DATE)]

    signs = np.sign(valid["rolling_correlation"])

    pre_mean = (
        float(pre["rolling_correlation"].mean())
        if not pre.empty
        else np.nan
    )

    post_mean = (
        float(post["rolling_correlation"].mean())
        if not post.empty
        else np.nan
    )

    return {
        "latest": float(valid.iloc[-1]["rolling_correlation"]),
        "minimum": float(valid["rolling_correlation"].min()),
        "maximum": float(valid["rolling_correlation"].max()),
        "pre_mean": pre_mean,
        "post_mean": post_mean,
        "shift": (
            post_mean - pre_mean
            if pd.notna(pre_mean) and pd.notna(post_mean)
            else np.nan
        ),
        "sign_changes": int(
            signs.ne(signs.shift()).iloc[1:].sum()
        ),
        "valid_windows": int(len(valid)),
    }



def _get_pair_summary(
    rolling_data: pd.DataFrame,
    metric_x: str,
    metric_y: str,
) -> dict[str, object] | None:
    pair_data = rolling_data[
        rolling_data["metric_x"].eq(metric_x)
        & rolling_data["metric_y"].eq(metric_y)
    ]

    if pair_data.empty:
        pair_data = rolling_data[
            rolling_data["metric_x"].eq(metric_y)
            & rolling_data["metric_y"].eq(metric_x)
        ]

    if pair_data.empty:
        return None

    return _summarize_pair(pair_data)

def _build_hero_takeaway(hero_data: pd.DataFrame) -> str:
    taxi_fhvhv = _get_pair_summary(
        hero_data,
        "taxi_trip_count",
        "fhvhv_trip_count",
    )

    taxi_subway = _get_pair_summary(
        hero_data,
        "taxi_trip_count",
        "subway_ridership",
    )

    if taxi_fhvhv is None or taxi_subway is None:
        return (
            "The two headline relationships could not both be summarized."
        )

    return (
        "The two relationships moved in opposite directions. Taxi and FHVHV "
        "demand became more synchronized, with their average rolling "
        f"relationship increasing from **{taxi_fhvhv['pre_mean']:+.3f}** "
        f"before congestion pricing to **{taxi_fhvhv['post_mean']:+.3f}** "
        "afterward. Taxi demand and Subway Ridership remained positively "
        "related, but their average alignment weakened from "
        f"**{taxi_subway['pre_mean']:+.3f}** to "
        f"**{taxi_subway['post_mean']:+.3f}**."
    )


def _build_explorer_takeaway(
    pair_data: pd.DataFrame,
    *,
    metric_a_label: str,
    metric_b_label: str,
    temporal_bucket_label: str,
) -> str:
    summary = _summarize_pair(pair_data)

    if summary["valid_windows"] == 0:
        return (
            "No eligible rolling windows were available for the selected "
            "relationship."
        )

    shift = summary["shift"]

    if pd.isna(shift):
        shift_sentence = (
            "There were not enough windows on both sides of congestion "
            "pricing to compare period averages."
        )
    elif shift >= 0.10:
        shift_sentence = (
            f"The average relationship strengthened by "
            f"**{shift:+.3f}** after congestion pricing."
        )
    elif shift <= -0.10:
        shift_sentence = (
            f"The average relationship weakened by "
            f"**{shift:+.3f}** after congestion pricing."
        )
    else:
        shift_sentence = (
            f"The average relationship changed only modestly after "
            f"congestion pricing (**{shift:+.3f}**)."
        )

    if summary["sign_changes"] > 0:
        stability_sentence = (
            f"The relationship crossed zero **{summary['sign_changes']} "
            "times**, indicating a volatile pattern rather than one stable "
            "direction."
        )
    else:
        stability_sentence = (
            "The relationship never changed sign, so its direction remained "
            "stable even as its strength varied."
        )

    return (
        f"For **{metric_a_label}** and **{metric_b_label}** during "
        f"**{temporal_bucket_label}**, the latest rolling correlation is "
        f"**{summary['latest']:+.3f}**. {shift_sentence} "
        f"{stability_sentence} The observed range was "
        f"**{summary['minimum']:+.3f}** to "
        f"**{summary['maximum']:+.3f}**."
    )



def _classify_relationship(
    pair_data: pd.DataFrame,
) -> tuple[str, str]:
    summary = _summarize_pair(pair_data)

    if summary["valid_windows"] == 0:
        return (
            "Unavailable",
            "No eligible rolling windows were available.",
        )

    if summary["sign_changes"] >= 2:
        return (
            "Volatile",
            (
                f"The relationship crossed zero "
                f"{summary['sign_changes']} times."
            ),
        )

    shift = summary["shift"]

    if pd.isna(shift):
        return (
            "Insufficient comparison",
            "There are not enough windows on both sides of congestion pricing.",
        )

    if shift >= 0.10:
        return (
            "Strengthening",
            f"Post-CP average increased by {shift:+.3f}.",
        )

    if shift <= -0.10:
        return (
            "Weakening",
            f"Post-CP average decreased by {shift:+.3f}.",
        )

    return (
        "Stable",
        f"Post-CP average changed by only {shift:+.3f}.",
    )

def _build_detail_table(
    rolling_data: pd.DataFrame,
) -> pd.DataFrame:
    display = rolling_data[
        [
            "pair_label",
            "window_start",
            "window_end",
            "window_midpoint",
            "rolling_correlation",
            "matched_days",
            "required_matched_days",
            "eligible_date_count",
            "coverage_share",
            "eligible_window",
        ]
    ].copy()

    return display.rename(
        columns={
            "pair_label": "Metric pair",
            "window_start": "Window start",
            "window_end": "Window end",
            "window_midpoint": "Window midpoint",
            "rolling_correlation": "Rolling correlation",
            "matched_days": "Matched days",
            "required_matched_days": "Required days",
            "eligible_date_count": "Eligible dates",
            "coverage_share": "Coverage",
            "eligible_window": "Eligible window",
        }
    )


st.caption("ROLLING MODE RELATIONSHIPS")
st.title("How did relationships between modes evolve?")

st.write(
    "Track whether selected mobility measures became more aligned, less "
    "aligned, or more volatile over time."
)

st.header("Taxi aligned more with FHVHV and less with subway")

st.write(
    "After congestion pricing, Taxi and FHVHV demand moved more closely "
    "together, while the relationship between Taxi demand and Subway "
    "Ridership weakened."
)

with st.spinner("Preparing the rolling relationship overview..."):
    hero_data = build_default_rolling_relationship_data()

if hero_data.empty:
    st.error("The rolling relationship overview returned no rows.")
    st.stop()

taxi_fhvhv_summary = _get_pair_summary(
    hero_data,
    "taxi_trip_count",
    "fhvhv_trip_count",
)

taxi_subway_summary = _get_pair_summary(
    hero_data,
    "taxi_trip_count",
    "subway_ridership",
)

if taxi_fhvhv_summary is None or taxi_subway_summary is None:
    available_pairs = (
        hero_data[
            [
                "metric_x",
                "metric_y",
                "pair_label",
            ]
        ]
        .drop_duplicates()
        .to_dict("records")
    )

    st.error(
        "The expected headline metric pairs were not found in the rolling "
        f"relationship data. Available pairs: {available_pairs}"
    )
    st.stop()

hero_card1, hero_card2, hero_card3, hero_card4 = st.columns(4)

hero_card1.metric(
    "Taxi–FHVHV latest",
    _format_correlation(taxi_fhvhv_summary["latest"]),
    f"{taxi_fhvhv_summary['shift']:+.3f} post minus pre",
)

hero_card2.metric(
    "Taxi–Subway latest",
    _format_correlation(taxi_subway_summary["latest"]),
    f"{taxi_subway_summary['shift']:+.3f} post minus pre",
)

hero_card3.metric(
    "Rolling window",
    f"{DEFAULT_WINDOW_DAYS} days",
)

hero_card4.metric(
    "Update step",
    f"{DEFAULT_STEP_DAYS} days",
)

hero_status_col1, hero_status_col2 = st.columns(2)

hero_status_col1.info(
    f"**Taxi–FHVHV: Strengthening**  \n"
    f"Post-CP average changed by "
    f"{taxi_fhvhv_summary['shift']:+.3f}."
)

hero_status_col2.info(
    f"**Taxi–Subway: Weakening**  \n"
    f"Post-CP average changed by "
    f"{taxi_subway_summary['shift']:+.3f}."
)

hero_fig = build_rolling_relationship_chart(
    hero_data,
    height=670,
)

st.plotly_chart(
    hero_fig,
    width="stretch",
    config={"displayModeBar": False, "responsive": True},
    key="raw08_static_hero",
)

render_chart_insight(_build_hero_takeaway(hero_data))

st.caption(
    f"Each overall window requires at least {DEFAULT_MIN_MATCHED_DAYS} "
    f"matched days within the {DEFAULT_WINDOW_DAYS}-day period."
)

st.divider()
st.header("Explore rolling relationships")

st.write(
    "Choose two measures, select a time-of-week context, and compare how "
    "their citywide relationship changed over time."
)

if "raw08_saved_view" not in st.session_state:
    st.session_state["raw08_saved_view"] = "Taxi and FHVHV alignment"

saved_view = st.selectbox(
    "Start with a saved configuration",
    options=SAVED_VIEW_OPTIONS,
    index=0,
    key="raw08_saved_view",
    help=(
        "Saved configurations provide curated starting points. Changing "
        "any control switches the selection to Custom."
    ),
)

if (
    saved_view != "Custom"
    and st.session_state.get("_raw08_applied_saved_view") != saved_view
):
    _apply_saved_view(saved_view)
    st.session_state["_raw08_applied_saved_view"] = saved_view
    st.rerun()

if saved_view == "Custom":
    st.session_state["_raw08_applied_saved_view"] = "Custom"

if saved_view == "Custom":
    st.caption(
        "Custom view · adjust the relationship, time context, or window."
    )
else:
    st.caption(SAVED_VIEWS[saved_view]["description"])

measure_col1, measure_col2 = st.columns(2)

with measure_col1:
    metric_a = st.selectbox(
        "Metric A",
        options=CORE_METRICS,
        index=(
            CORE_METRICS.index("taxi_trip_count")
            if "taxi_trip_count" in CORE_METRICS
            else 0
        ),
        format_func=_metric_label,
        key="raw08_metric_a",
        on_change=_mark_saved_view_custom,
    )

metric_b_options = [
    metric
    for metric in CORE_METRICS
    if metric != metric_a
]

if not metric_b_options:
    st.error(
        "At least two core metrics are required to build a relationship view."
    )
    st.stop()

with measure_col2:
    if st.session_state.get("raw08_metric_b") not in metric_b_options:
        st.session_state["raw08_metric_b"] = (
            "fhvhv_trip_count"
            if "fhvhv_trip_count" in metric_b_options
            else metric_b_options[0]
        )

    metric_b = st.selectbox(
        "Metric B",
        options=metric_b_options,
        index=0,
        format_func=_metric_label,
        key="raw08_metric_b",
        on_change=_mark_saved_view_custom,
    )

time_col1, time_col2 = st.columns(2)

with time_col1:
    temporal_bucket = st.selectbox(
        "Time context",
        options=TEMPORAL_BUCKET_OPTIONS,
        index=0,
        format_func=lambda value: TEMPORAL_BUCKET_LABELS[value],
        key="raw08_temporal_bucket",
        on_change=_mark_saved_view_custom,
    )

with time_col2:
    rolling_window_days = st.selectbox(
        "Rolling window",
        options=WINDOW_OPTIONS,
        index=1,
        format_func=lambda value: f"{value} days",
        key="raw08_rolling_window_days",
        on_change=_mark_saved_view_custom,
    )

minimum_matched_days = int(round(rolling_window_days * 0.50))

with st.spinner("Updating the rolling relationship..."):
    if temporal_bucket == ALL_TEMPORAL_BUCKETS_LABEL:
        explorer_data = build_citywide_rolling_relationship_data(
            pair_definitions=[(metric_a, metric_b)],
            temporal_bucket=temporal_bucket,
            rolling_window_days=rolling_window_days,
            rolling_step_days=14,
            minimum_matched_days=minimum_matched_days,
            minimum_coverage_share=None,
            correlation_method="spearman",
        )
    else:
        explorer_data = build_bucket_rolling_relationship_data(
            pair_definitions=[(metric_a, metric_b)],
            temporal_bucket=temporal_bucket,
            rolling_window_days=rolling_window_days,
            rolling_step_days=14,
            minimum_coverage_share=0.50,
            correlation_method="spearman",
        )

if explorer_data.empty:
    st.info(
        "No rolling relationship data were available for this selection."
    )
else:
    metric_a_label = _metric_label(metric_a)
    metric_b_label = _metric_label(metric_b)
    temporal_bucket_label = TEMPORAL_BUCKET_LABELS[temporal_bucket]

    explorer_summary = _summarize_pair(explorer_data)

    explorer_card1, explorer_card2, explorer_card3, explorer_card4 = (
        st.columns(4)
    )

    explorer_card1.metric(
        "Latest correlation",
        _format_correlation(explorer_summary["latest"]),
    )

    explorer_card2.metric(
        "Pre-CP average",
        _format_correlation(explorer_summary["pre_mean"]),
    )

    explorer_card3.metric(
        "Post-CP average",
        _format_correlation(explorer_summary["post_mean"]),
        (
            f"{explorer_summary['shift']:+.3f}"
            if pd.notna(explorer_summary["shift"])
            else None
        ),
    )

    explorer_card4.metric(
        "Sign changes",
        f"{explorer_summary['sign_changes']:,}",
    )

    relationship_status, relationship_status_detail = (
        _classify_relationship(explorer_data)
    )

    st.info(
        f"**{relationship_status}**  \n"
        f"{relationship_status_detail}"
    )

    st.caption(
        f"{metric_a_label} vs {metric_b_label} · "
        f"{temporal_bucket_label} · "
        f"{rolling_window_days}-day rolling window · "
        "14-day update step · Spearman correlation"
    )

    if temporal_bucket == ALL_TEMPORAL_BUCKETS_LABEL:
        st.caption(
            f"Overall windows require at least "
            f"{minimum_matched_days} matched days."
        )
    else:
        st.caption(
            "Time-bucket windows require coverage on at least half of the "
            "dates eligible for the selected bucket."
        )

    explorer_takeaway = _build_explorer_takeaway(
        explorer_data,
        metric_a_label=metric_a_label,
        metric_b_label=metric_b_label,
        temporal_bucket_label=temporal_bucket_label,
    )

    explorer_fig = build_rolling_relationship_chart(
        explorer_data,
        height=640,
    )

    st.plotly_chart(
        explorer_fig,
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
        key=(
            f"raw08_explorer_{metric_a}_{metric_b}_"
            f"{temporal_bucket}_{rolling_window_days}"
        ),
    )
    render_chart_insight(explorer_takeaway)

    with st.expander(
        "View rolling-window details",
        expanded=False,
    ):
        detail = _build_detail_table(explorer_data)

        st.dataframe(
            detail,
            width="stretch",
            hide_index=True,
            column_config={
                "Rolling correlation": st.column_config.NumberColumn(
                    format="%+.3f",
                ),
                "Coverage": st.column_config.NumberColumn(
                    format="%.1f%%",
                ),
                "Matched days": st.column_config.NumberColumn(
                    format="%d",
                ),
                "Required days": st.column_config.NumberColumn(
                    format="%d",
                ),
                "Eligible dates": st.column_config.NumberColumn(
                    format="%d",
                ),
            },
        )

with st.expander(
    "How to read this page",
    expanded=False,
):
    st.markdown(
        """
A rolling correlation summarizes how two citywide daily series moved together
inside a trailing calendar window.

- Values near **+1** indicate that the two measures tended to rise and fall
  together.
- Values near **−1** indicate that one tended to rise when the other fell.
- Values near **0** indicate little stable monotonic relationship.
- Overall views require observations for at least half of the calendar window.
- Weekday and weekend views require observations for at least half of the
  dates eligible for the selected time bucket.
- Unsupported windows appear as gaps rather than being filled.

These relationships are descriptive. They do not establish causality or
substitution between modes.
        """
    )
