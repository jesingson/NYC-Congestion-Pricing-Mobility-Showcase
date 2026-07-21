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

ADJUSTMENT_BASELINE_DAYS = 90

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


def _build_monthly_adjustment_path(
    daily_df: pd.DataFrame,
    *,
    metrics: list[str],
    baseline_days: int = ADJUSTMENT_BASELINE_DAYS,
) -> pd.DataFrame:
    """Index monthly post-CP values to the immediate pre-CP daily baseline."""
    if daily_df.empty:
        return pd.DataFrame()

    working = daily_df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.loc[working["date"].notna()].sort_values("date")

    baseline_start = CONGESTION_PRICING_START_DATE - pd.Timedelta(days=baseline_days)
    baseline_mask = (
        working["date"].ge(baseline_start)
        & working["date"].lt(CONGESTION_PRICING_START_DATE)
    )
    post_mask = working["date"].ge(CONGESTION_PRICING_START_DATE)

    rows: list[dict[str, object]] = []

    for metric_name in metrics:
        if metric_name not in working.columns:
            continue

        baseline_values = working.loc[baseline_mask, metric_name].dropna()
        if baseline_values.empty:
            continue

        baseline_value = float(baseline_values.mean())
        if not np.isfinite(baseline_value) or baseline_value == 0:
            continue

        rows.append(
            {
                "metric": metric_name,
                "metric_label": METRIC_LABELS.get(metric_name, metric_name),
                "period_start": baseline_start.normalize(),
                "period_label": "Pre-CP reference",
                "period_order": 0,
                "period_value": baseline_value,
                "baseline_value": baseline_value,
                "index_value": 100.0,
                "observed_days": int(baseline_values.count()),
            }
        )

        metric_post = working.loc[post_mask, ["date", metric_name]].dropna().copy()
        if metric_post.empty:
            continue

        metric_post["period_start"] = (
            metric_post["date"].dt.to_period("M").dt.to_timestamp()
        )

        monthly = (
            metric_post.groupby("period_start", as_index=False)
            .agg(
                period_value=(metric_name, "mean"),
                observed_days=(metric_name, "count"),
            )
            .sort_values("period_start")
        )

        for period_order, row in enumerate(monthly.itertuples(index=False), start=1):
            period_value = float(row.period_value)
            rows.append(
                {
                    "metric": metric_name,
                    "metric_label": METRIC_LABELS.get(metric_name, metric_name),
                    "period_start": row.period_start,
                    "period_label": pd.Timestamp(row.period_start).strftime("%b %Y"),
                    "period_order": period_order,
                    "period_value": period_value,
                    "baseline_value": baseline_value,
                    "index_value": period_value / baseline_value * 100.0,
                    "observed_days": int(row.observed_days),
                }
            )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    return result.sort_values(["metric", "period_order"]).reset_index(drop=True)


def _build_adjustment_figure(
    adjustment_df: pd.DataFrame,
    *,
    title: str,
    height: int = 520,
) -> go.Figure:
    """Build a policy-relative monthly adjustment-path chart."""
    fig = go.Figure()

    if adjustment_df.empty:
        fig.update_layout(title=title, height=height)
        return apply_branding(fig)

    period_order = (
        adjustment_df[["period_order", "period_label"]]
        .drop_duplicates()
        .sort_values("period_order")
    )
    category_order = period_order["period_label"].tolist()

    for metric_name, metric_df in adjustment_df.groupby("metric", sort=False):
        metric_df = metric_df.sort_values("period_order")
        fig.add_trace(
            go.Scatter(
                x=metric_df["period_label"],
                y=metric_df["index_value"],
                mode="lines+markers",
                name=METRIC_LABELS.get(metric_name, metric_name),
                customdata=metric_df[["period_value", "observed_days"]],
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Period: %{x}<br>"
                    "Index: %{y:.1f}<br>"
                    "Period daily avg: %{customdata[0]:,.2f}<br>"
                    "Observed days: %{customdata[1]:,}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_hline(
        y=100,
        line_dash="dot",
        line_color="rgba(0, 109, 119, 0.50)",
        annotation_text=f"Final {ADJUSTMENT_BASELINE_DAYS} pre-CP days = 100",
        annotation_position="bottom right",
    )

    fig.update_xaxes(
        categoryorder="array",
        categoryarray=category_order,
        tickangle=-35,
    )
    fig.update_layout(
        title=title,
        xaxis_title="Policy-relative period",
        yaxis_title="Index value",
        hovermode="x unified",
        height=height,
    )

    fig = apply_branding(fig)
    return _apply_bottom_legend(fig, bottom_margin=145)


def _build_hero_metric_insight(
    summary_df: pd.DataFrame,
    *,
    metrics: list[str],
) -> str:
    """Summarize the strongest pre/post movements among the selected hero metrics."""
    selected = summary_df.loc[
        summary_df["metric"].isin(metrics)
        & summary_df["percent_change"].notna()
    ].copy()

    if selected.empty:
        return "The selected metrics do not have enough pre/post coverage for a concise comparison."

    selected["metric_label"] = selected["metric"].map(
        lambda value: METRIC_LABELS.get(value, value)
    )
    selected = selected.sort_values("percent_change")

    weakest = selected.iloc[0]
    strongest = selected.iloc[-1]

    if len(selected) == 1:
        return (
            f"<strong>{strongest['metric_label']}</strong> changed "
            f"<strong>{strongest['percent_change']:+.1f}%</strong> in the full-period "
            "post-CP versus pre-CP comparison."
        )

    same_direction = (selected["percent_change"] > 0).all() or (
        selected["percent_change"] < 0
    ).all()
    direction_note = (
        "The selected metrics moved in the same broad direction, but by different magnitudes."
        if same_direction
        else "The selected metrics did not move uniformly, pointing to a mixed multimodal response."
    )

    return (
        f"Among the selected series, <strong>{strongest['metric_label']}</strong> had the "
        f"largest full-period change at <strong>{strongest['percent_change']:+.1f}%</strong>, "
        f"while <strong>{weakest['metric_label']}</strong> had the smallest at "
        f"<strong>{weakest['percent_change']:+.1f}%</strong>. {direction_note}"
    )


def _classify_adjustment_pattern(metric_df: pd.DataFrame) -> tuple[str, float, float]:
    """Return a concise pattern label plus early and later index levels."""
    post_df = metric_df.loc[
        metric_df["period_order"].gt(0) & metric_df["index_value"].notna()
    ].sort_values("period_order")

    if post_df.empty:
        return "insufficient post-CP data", np.nan, np.nan

    early = float(post_df.head(min(2, len(post_df)))["index_value"].mean())
    later = float(post_df.tail(min(3, len(post_df)))["index_value"].mean())
    early_change = early - 100.0
    later_change = later - 100.0
    threshold = 3.0

    if abs(early_change) < threshold and abs(later_change) < threshold:
        label = "stayed near baseline"
    elif abs(early_change) >= threshold and abs(later_change) < abs(early_change) * 0.55:
        label = "faded toward baseline"
    elif abs(early_change) < threshold and abs(later_change) >= threshold:
        label = "emerged later"
    elif np.sign(early_change) != np.sign(later_change) and abs(later_change) >= threshold:
        label = "changed direction"
    elif abs(later_change) >= threshold:
        label = "persisted above baseline" if later_change > 0 else "persisted below baseline"
    else:
        label = "followed a mixed path"

    return label, early, later


def _build_adjustment_overview(
    adjustment_df: pd.DataFrame,
    *,
    metrics: list[str],
) -> str:
    """Summarize how the selected adjustment paths differ in persistence."""
    summaries: list[dict[str, object]] = []

    for metric in metrics:
        metric_df = adjustment_df.loc[adjustment_df["metric"] == metric]
        label, early, later = _classify_adjustment_pattern(metric_df)
        if not np.isfinite(later):
            continue
        summaries.append(
            {
                "metric": metric,
                "metric_label": METRIC_LABELS.get(metric, metric),
                "pattern": label,
                "early": early,
                "later": later,
                "later_change": later - 100.0,
            }
        )

    if not summaries:
        return "The selected metrics do not have enough post-CP coverage to compare persistence."

    strongest = max(summaries, key=lambda row: abs(float(row["later_change"])))
    pattern_text = "; ".join(
        f"<strong>{row['metric_label']}</strong> {row['pattern']}"
        for row in summaries
    )

    return (
        f"{pattern_text}. The largest later-period departure among these selections was "
        f"<strong>{strongest['metric_label']}</strong> at "
        f"<strong>{float(strongest['later']):.1f}</strong> on the index "
        f"({float(strongest['later_change']):+.1f} relative to baseline)."
    )


def _build_adjustment_interpretation(
    adjustment_df: pd.DataFrame,
    *,
    metric: str,
) -> str:
    """Describe whether the selected post-CP path persisted, faded, or developed later."""
    metric_df = adjustment_df.loc[
        (adjustment_df["metric"] == metric)
        & adjustment_df["period_order"].gt(0)
        & adjustment_df["index_value"].notna()
    ].sort_values("period_order")

    metric_label = METRIC_LABELS.get(metric, metric)
    if metric_df.empty:
        return "This selection does not have enough post-CP data to summarize an adjustment path."

    pattern_label, early, later = _classify_adjustment_pattern(metric_df)
    pattern_lookup = {
        "stayed near baseline": "stayed close to its immediate pre-CP baseline",
        "faded toward baseline": "showed an early shift that later moved substantially back toward baseline",
        "emerged later": "showed limited initial movement but a clearer shift later in the post-CP period",
        "changed direction": "changed direction between the early and later post-CP periods",
        "persisted above baseline": "remained meaningfully above its immediate pre-CP baseline in later months",
        "persisted below baseline": "remained meaningfully below its immediate pre-CP baseline in later months",
        "followed a mixed path": "followed a mixed path without a clearly persistent later shift",
    }
    pattern = pattern_lookup.get(pattern_label, pattern_label)

    return (
        f"<strong>{metric_label}</strong> {pattern}. "
        f"The first two post-CP months averaged <strong>{early:.1f}</strong> on the index, "
        f"while the latest three available months averaged <strong>{later:.1f}</strong>."
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
adjustment_daily_df = get_daily_metric_trends(metrics=CORE_METRICS)

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

st.markdown(
    f"""
    <div class="soft-callout">
        <strong>So what?</strong><br>
        {_build_hero_metric_insight(summary_df, metrics=hero_metrics)}
    </div>
    """,
    unsafe_allow_html=True,
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

st.markdown("### Did the change persist?")
st.markdown(
    "A full pre/post average can hide whether a response appeared immediately, "
    "strengthened gradually, or faded. This view compares each selected metric with "
    f"its average during the final {ADJUSTMENT_BASELINE_DAYS} pre-CP days."
)

hero_adjustment_df = _build_monthly_adjustment_path(
    adjustment_daily_df,
    metrics=hero_metrics,
)

if hero_adjustment_df.empty:
    st.warning("The monthly post-CP adjustment path could not be calculated.")
else:
    hero_adjustment_fig = _build_adjustment_figure(
        hero_adjustment_df,
        title="Monthly mobility adjustment after congestion pricing",
        height=540,
    )
    st.plotly_chart(
        hero_adjustment_fig,
        use_container_width=True,
        key="raw01_curated_adjustment_path",
    )
    st.caption(
        f"Each line begins at 100 for the final {ADJUSTMENT_BASELINE_DAYS} days before "
        "congestion pricing. Monthly points are averages of the available daily values; "
        "the January 2025 point begins on the January 5 launch date."
    )
    st.markdown(
        f"""
        <div class="soft-callout">
            <strong>So what?</strong><br>
            {_build_adjustment_overview(hero_adjustment_df, metrics=hero_metrics)}
        </div>
        """,
        unsafe_allow_html=True,
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
    Start with a saved view, then use the tabs below to inspect either the detailed daily pattern
    or the persistence of the post-CP adjustment. Shared controls apply to both tabs; only the
    controls specific to each chart appear inside that tab.
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

control_col1, control_col2 = st.columns(2)

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


daily_tab, adjustment_tab = st.tabs([
    "Daily trend",
    "Post-CP persistence",
])

with daily_tab:
    st.markdown(
        "Explore the full observed timeline, including smoothing, custom dates, "
        "and optional trend guides."
    )
    c1, c2 = st.columns(2)
    with c1:
        smoothing_window_label = st.selectbox(
            "Smoothing", RAW01_SMOOTHING_OPTIONS,
            key="raw01_smoothing", on_change=_mark_raw01_custom,
        )
    with c2:
        trend_guide = st.selectbox(
            "Trend guide",
            ["None", "Full-period trend line", "Separate pre/post trend lines"],
            key="raw01_trend_guide",
        )
    d1, d2 = st.columns(2)
    with d1:
        date_start = st.date_input(
            "Start date", min_value=STUDY_START_DATE.date(),
            max_value=STUDY_END_DATE.date(), key="raw01_start_date",
            on_change=_mark_raw01_custom,
        )
    with d2:
        date_end = st.date_input(
            "End date", min_value=STUDY_START_DATE.date(),
            max_value=STUDY_END_DATE.date(), key="raw01_end_date",
            on_change=_mark_raw01_custom,
        )
    if date_start > date_end:
        st.warning("Start date must be before or equal to end date.")
    else:
        smoothing_window = {
            "None": None, "7-day rolling average": 7,
            "14-day rolling average": 14, "28-day rolling average": 28,
        }[smoothing_window_label]
        date_range = (pd.Timestamp(date_start), pd.Timestamp(date_end))
        daily_df = get_daily_metric_trends(
            metrics=[metric], temporal_bucket=temporal_bucket,
            borough=borough, cbd_spatial_category=cbd_spatial_category,
            date_range=date_range,
        )
        daily_df = add_rolling_average(daily_df, metrics=[metric], window=smoothing_window)
        summary = get_pre_post_metric_summary(
            metrics=[metric], temporal_bucket=temporal_bucket,
            borough=borough, cbd_spatial_category=cbd_spatial_category,
            date_range=date_range,
        )
        label = METRIC_LABELS.get(metric, metric)
        display_col = f"{metric}_display"
        chart = go.Figure()
        if smoothing_window is not None:
            chart.add_trace(go.Scatter(
                x=daily_df["date"], y=daily_df[metric], mode="lines",
                name="Daily value", line={"color": "rgba(0,109,119,0.25)"},
            ))
        chart.add_trace(go.Scatter(
            x=daily_df["date"], y=daily_df[display_col], mode="lines",
            name=smoothing_window_label if smoothing_window else "Daily value",
            line={"color": BRAND_COLORS["dark_teal"], "width": 3},
        ))
        if date_range[0] <= CONGESTION_PRICING_START_DATE <= date_range[1]:
            chart.add_vline(
                x=CONGESTION_PRICING_START_DATE, line_dash="dash",
                line_color=BRAND_COLORS["terracotta"],
                annotation_text="CP starts", annotation_position="top left",
            )
        if trend_guide == "Full-period trend line":
            _add_full_period_trend_line(chart, daily_df, metric_col=metric, color=BRAND_COLORS["terracotta"])
        elif trend_guide == "Separate pre/post trend lines":
            _add_period_trend_line(chart, daily_df, metric_col=metric, label="Pre-CP fitted line", period="pre_cp", color=BRAND_COLORS["terracotta"])
            _add_period_trend_line(chart, daily_df, metric_col=metric, label="Post-CP fitted line", period="post_cp", color=BRAND_COLORS["dark_teal"])
        chart.update_layout(title=f"{label} over time", xaxis_title="Date", yaxis_title=label, hovermode="x unified", height=500)
        chart = _apply_bottom_legend(apply_branding(chart), bottom_margin=105)
        st.plotly_chart(chart, use_container_width=True, key="raw01_daily_timeline_chart")
        insight = build_selected_view_interpretation(
            summary, metric=metric, geography_scope=geography_scope,
            temporal_bucket=temporal_bucket, borough=borough,
            cbd_spatial_category=cbd_spatial_category,
        )
        st.markdown('<div class="soft-callout"><strong>What to notice:</strong><br>'+insight+'</div>', unsafe_allow_html=True)
        m1,m2,m3=st.columns(3)
        m1.metric("Pre-CP daily average", f"{summary['pre_daily_average'].iloc[0]:,.2f}")
        m2.metric("Post-CP daily average", f"{summary['post_daily_average'].iloc[0]:,.2f}")
        m3.metric("Post vs pre difference", f"{summary['percent_change'].iloc[0]:,.2f}%")
        with st.expander("Show selected pre/post summary", expanded=False):
            _display_summary_table(summary)

with adjustment_tab:
    st.markdown(
        "Compare monthly post-CP movement with the immediate pre-CP baseline "
        "to see whether changes persisted, faded, or emerged gradually."
    )
    st.caption(
        f"This view uses calendar months and a fixed {ADJUSTMENT_BASELINE_DAYS}-day "
        "immediate pre-CP baseline. Smoothing and arbitrary date windows do not apply."
    )
    daily_df = get_daily_metric_trends(
        metrics=[metric], temporal_bucket=temporal_bucket,
        borough=borough, cbd_spatial_category=cbd_spatial_category,
        date_range=(STUDY_START_DATE, STUDY_END_DATE),
    )
    adjustment_df = _build_monthly_adjustment_path(daily_df, metrics=[metric])
    label = METRIC_LABELS.get(metric, metric)
    chart = _build_adjustment_figure(
        adjustment_df, title=f"{label}: monthly post-CP adjustment path", height=520,
    )
    st.plotly_chart(chart, use_container_width=True, key="raw01_adjustment_path_chart")
    insight = _build_adjustment_interpretation(adjustment_df, metric=metric)
    st.markdown('<div class="soft-callout"><strong>What to notice:</strong><br>'+insight+'</div>', unsafe_allow_html=True)
    path = adjustment_df.loc[adjustment_df["metric"].eq(metric)].sort_values("period_order")
    if path.empty:
        baseline = first_idx = latest_idx = np.nan
    else:
        baseline = path.iloc[0]["baseline_value"]
        post = path.loc[path["period_order"].gt(0)]
        first_idx = post.iloc[0]["index_value"] if not post.empty else np.nan
        latest_idx = post.tail(min(3, len(post)))["index_value"].mean() if not post.empty else np.nan
    m1,m2,m3=st.columns(3)
    m1.metric(f"Final {ADJUSTMENT_BASELINE_DAYS}-day pre-CP average", f"{baseline:,.2f}" if pd.notna(baseline) else "—")
    m2.metric("First post-CP month", f"{first_idx:.1f}" if pd.notna(first_idx) else "—", delta=f"{first_idx-100:+.1f} vs baseline" if pd.notna(first_idx) else None)
    m3.metric("Latest 3-month average", f"{latest_idx:.1f}" if pd.notna(latest_idx) else "—", delta=f"{latest_idx-100:+.1f} vs baseline" if pd.notna(latest_idx) else None)
