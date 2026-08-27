"""Reusable aggregation helpers for the NYC mobility showcase app.

These functions centralize the app's metric logic so page files can stay focused
on layout and interpretation.

Key policy:
- Count / demand metrics aggregate by sum.
- Speed metrics aggregate by weighted average where a reliable weight exists.
- Pre/post summaries use daily averages, not full-period totals, because the
  pre- and post-congestion-pricing windows have different lengths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from app.data_access.loaders import (
    BASE_METRICS,
    CONGESTION_PRICING_START_DATE,
    CORE_METRICS,
    METRIC_LABELS,
    load_analysis_panel,
)
from app.data_access.mobility_environments import (
    attach_mobility_regime_cluster_context,
    format_mobility_regime_cluster_label,
)


COUNT_METRICS = [
    "taxi_trip_count",
    "fhvhv_trip_count",
    "subway_ridership",
    "subway_transfers",
    "bus_trip_count",
]

SPEED_METRICS = [
    "taxi_avg_trip_speed",
    "fhvhv_avg_trip_speed",
    "avg_bus_speed",
]

DURATION_METRICS = [
    "taxi_avg_trip_duration",
    "fhvhv_avg_trip_duration",
]

WEIGHTED_MEAN_METRICS = [
    *SPEED_METRICS,
    *DURATION_METRICS,
]

# Hidden support columns. These can be used for aggregation weights without
# exposing them as primary app metrics.
WEIGHT_COLUMNS = {
    "taxi_avg_trip_speed": "taxi_trip_count",
    "taxi_avg_trip_duration": "taxi_trip_count",
    "fhvhv_avg_trip_speed": "fhvhv_trip_count",
    "fhvhv_avg_trip_duration": "fhvhv_trip_count",
    "avg_bus_speed": "bus_trip_count",
}



BASE_FILTER_COLUMNS = [
    "date",
    "pre_post_cp",
    "temporal_bucket",
    "borough",
    "cbd_spatial_category",
]

OPTIONAL_FILTER_COLUMNS = [
    "taxi_zone_id",
    "zone",
]

# App-facing CBD geography groups. These intentionally support both the raw
# snake_case values used in the parquet and title-case variants in case earlier
# notebooks produced friendlier labels.
CBD_SPATIAL_CATEGORY_GROUPS = {
    "All geography groups": None,
    "CBD": ["cbd", "CBD"],
    "Gateway + adjacent": [
        "gateway_to_cbd",
        "adjacent_to_cbd",
        "gateway",
        "adjacent",
        "Gateway",
        "Adjacent",
    ],
    "Gateway only": ["gateway_to_cbd", "gateway", "Gateway"],
    "Adjacent only": ["adjacent_to_cbd", "adjacent", "Adjacent"],
    "Non-CBD": ["non_cbd", "non-cbd", "Non-CBD", "Non CBD"],
}

CBD_SPATIAL_CATEGORY_LABELS = {
    "cbd": "CBD",
    "CBD": "CBD",
    "gateway_to_cbd": "Gateway only",
    "gateway": "Gateway only",
    "Gateway": "Gateway only",
    "adjacent_to_cbd": "Adjacent only",
    "adjacent": "Adjacent only",
    "Adjacent": "Adjacent only",
    "non_cbd": "Non-CBD",
    "non-cbd": "Non-CBD",
    "Non-CBD": "Non-CBD",
    "Non CBD": "Non-CBD",
}


@dataclass(frozen=True)
class MetricAggregationSpec:
    """Aggregation metadata for one app metric."""

    metric: str
    display_label: str
    aggregation_type: str
    weight_column: str | None = None


def get_metric_spec(metric: str) -> MetricAggregationSpec:
    """Return aggregation metadata for a metric."""
    if metric in COUNT_METRICS:
        return MetricAggregationSpec(
            metric=metric,
            display_label=METRIC_LABELS.get(
                metric,
                metric,
            ),
            aggregation_type="sum",
            weight_column=None,
        )

    if metric in WEIGHTED_MEAN_METRICS:
        return MetricAggregationSpec(
            metric=metric,
            display_label=METRIC_LABELS.get(
                metric,
                metric,
            ),
            aggregation_type="weighted_mean",
            weight_column=WEIGHT_COLUMNS.get(
                metric
            ),
        )

    raise ValueError(
        f"Unsupported metric: {metric}. "
        f"Expected one of: {BASE_METRICS}"
    )


def get_required_columns(metrics: list[str]) -> list[str]:
    """Return the panel columns needed to aggregate selected metrics."""
    columns = set(BASE_FILTER_COLUMNS + OPTIONAL_FILTER_COLUMNS)

    for metric in metrics:
        columns.add(metric)

        spec = get_metric_spec(metric)
        if spec.weight_column:
            columns.add(spec.weight_column)

    return sorted(columns)


def get_cbd_spatial_category_members(
    cbd_spatial_category: str | None,
) -> list[str] | None:
    """Return raw cbd_spatial_category values represented by an app-facing option."""
    if cbd_spatial_category is None or cbd_spatial_category == "All geography groups":
        return None

    if cbd_spatial_category in CBD_SPATIAL_CATEGORY_GROUPS:
        return CBD_SPATIAL_CATEGORY_GROUPS[cbd_spatial_category]

    # Backward compatibility: old saved views may still pass raw values.
    return [cbd_spatial_category]


def format_cbd_spatial_category_label(cbd_spatial_category: str | None) -> str:
    """Return a readable label for a CBD geography option or raw category."""
    if cbd_spatial_category is None or cbd_spatial_category == "All geography groups":
        return "all geography groups"

    if cbd_spatial_category in CBD_SPATIAL_CATEGORY_GROUPS:
        return cbd_spatial_category

    return CBD_SPATIAL_CATEGORY_LABELS.get(
        cbd_spatial_category,
        cbd_spatial_category.replace("_", " "),
    )


def _safe_weighted_average(
    values: pd.Series,
    weights: pd.Series | None,
) -> float:
    """Calculate a weighted average with a simple mean fallback.

    Fallback behavior matters because some speed metrics may have missing or
    zero weights after filtering.
    """
    valid_values = values.notna()

    if weights is None:
        return float(values[valid_values].mean()) if valid_values.any() else np.nan

    valid_weights = weights.notna() & (weights > 0)
    valid = valid_values & valid_weights

    if valid.any():
        return float(np.average(values[valid], weights=weights[valid]))

    return float(values[valid_values].mean()) if valid_values.any() else np.nan


def _aggregate_one_group(group: pd.DataFrame, metric: str) -> float:
    """Aggregate one metric inside one grouped dataframe slice."""
    spec = get_metric_spec(metric)

    if spec.aggregation_type == "sum":
        return float(group[metric].sum(skipna=True))

    if spec.aggregation_type == "weighted_mean":
        weight_column = spec.weight_column

        if weight_column and weight_column in group.columns:
            return _safe_weighted_average(group[metric], group[weight_column])

        return _safe_weighted_average(group[metric], None)

    raise ValueError(f"Unsupported aggregation type: {spec.aggregation_type}")


def apply_common_filters(
    df: pd.DataFrame,
    *,
    temporal_bucket: str | None = None,
    borough: str | None = None,
    cbd_spatial_category: str | None = None,
    mobility_regime_cluster_label: int | None = None,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Apply shared filters used by raw-data explorer pages."""
    filtered = df.copy()

    if "date" in filtered.columns:
        filtered["date"] = pd.to_datetime(filtered["date"])

    if mobility_regime_cluster_label is not None:
        if "pre_post_cp" not in filtered.columns:
            raise ValueError(
                "mobility_regime_cluster_label requires pre_post_cp in the source dataframe"
            )

        pre_rows = filtered[
            filtered["pre_post_cp"].astype(str).eq("pre_cp")
        ].copy()
        post_rows = filtered[
            filtered["pre_post_cp"].astype(str).eq("post_cp")
        ].copy()

        attached_frames: list[pd.DataFrame] = []

        if not pre_rows.empty:
            attached_frames.append(
                attach_mobility_regime_cluster_context(
                    pre_rows,
                    assignment_period="pre_cp",
                )
            )

        if not post_rows.empty:
            attached_frames.append(
                attach_mobility_regime_cluster_context(
                    post_rows,
                    assignment_period="post_cp",
                )
            )

        filtered = (
            pd.concat(attached_frames, ignore_index=True)
            if attached_frames
            else filtered.iloc[0:0].copy()
        )

        if "mobility_regime_cluster_label" not in filtered.columns:
            filtered["mobility_regime_cluster_label"] = pd.Series(
                dtype="Int64"
            )
            filtered["mobility_regime_cluster_name"] = pd.Series(
                dtype="string"
            )

        filtered = filtered[
            filtered["mobility_regime_cluster_label"]
            .astype("Int64")
            .eq(int(mobility_regime_cluster_label))
        ].copy()

    if temporal_bucket and temporal_bucket != "All temporal buckets":
        filtered = filtered[filtered["temporal_bucket"].astype(str) == temporal_bucket]

    if borough and borough != "Citywide":
        filtered = filtered[filtered["borough"] == borough]

    category_members = get_cbd_spatial_category_members(cbd_spatial_category)
    if category_members is not None:
        filtered = filtered[
            filtered["cbd_spatial_category"].astype(str).isin(category_members)
        ]

    if date_range is not None:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered["date"] >= pd.to_datetime(start_date))
            & (filtered["date"] <= pd.to_datetime(end_date))
        ]

    return filtered


def aggregate_metrics(
    df: pd.DataFrame,
    *,
    metrics: list[str],
    group_cols: list[str],
) -> pd.DataFrame:
    """Aggregate selected metrics using their correct app aggregation rules."""
    missing_metrics = [metric for metric in metrics if metric not in df.columns]
    if missing_metrics:
        raise ValueError(f"Missing metric columns: {missing_metrics}")

    missing_group_cols = [col for col in group_cols if col not in df.columns]
    if missing_group_cols:
        raise ValueError(f"Missing group columns: {missing_group_cols}")

    rows = []

    for group_key, group in df.groupby(group_cols, observed=True, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        row = dict(zip(group_cols, group_key, strict=True))

        for metric in metrics:
            row[metric] = _aggregate_one_group(group, metric)

        row["row_count"] = len(group)
        rows.append(row)

    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Aggregating daily mobility trends...")
def get_daily_metric_trends(
    metrics: list[str] | None = None,
    *,
    temporal_bucket: str | None = None,
    borough: str | None = None,
    cbd_spatial_category: str | None = None,
    mobility_regime_cluster_label: int | None = None,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Return daily trend data for selected metrics.

    The result is one row per date, with one column per metric.
    """
    metrics = metrics or CORE_METRICS
    columns = get_required_columns(metrics)

    df = load_analysis_panel(columns=columns)
    df = apply_common_filters(
        df,
        temporal_bucket=temporal_bucket,
        borough=borough,
        cbd_spatial_category=cbd_spatial_category,
        mobility_regime_cluster_label=mobility_regime_cluster_label,
        date_range=date_range,
    )

    daily = aggregate_metrics(
        df,
        metrics=metrics,
        group_cols=["date"],
    )

    daily = daily.sort_values("date").reset_index(drop=True)
    daily["pre_post_cp"] = np.where(
        daily["date"] >= CONGESTION_PRICING_START_DATE,
        "post_cp",
        "pre_cp",
    )

    return daily


def add_rolling_average(
    df: pd.DataFrame,
    *,
    metrics: list[str],
    window: int | None = 14,
) -> pd.DataFrame:
    """Add rolling-average columns for selected metrics."""
    result = df.sort_values("date").copy()

    if window is None or window <= 1:
        for metric in metrics:
            result[f"{metric}_display"] = result[metric]
        return result

    for metric in metrics:
        result[f"{metric}_display"] = (
            result[metric]
            .rolling(window=window, min_periods=max(2, window // 3))
            .mean()
        )

    return result


def add_indexed_to_pre_cp_average(
    df: pd.DataFrame,
    *,
    metrics: list[str],
    value_suffix: str = "_display",
    index_suffix: str = "_index",
) -> pd.DataFrame:
    """Index each metric so its pre-CP average equals 100."""
    result = df.copy()
    pre_mask = result["date"] < CONGESTION_PRICING_START_DATE

    for metric in metrics:
        value_col = f"{metric}{value_suffix}"
        index_col = f"{metric}{index_suffix}"

        if value_col not in result.columns:
            value_col = metric

        baseline = result.loc[pre_mask, value_col].mean(skipna=True)

        if pd.isna(baseline) or baseline == 0:
            result[index_col] = np.nan
        else:
            result[index_col] = (result[value_col] / baseline) * 100

    return result


@st.cache_data(show_spinner="Building indexed hero trend...")
def get_indexed_daily_trends(
    metrics: list[str] | None = None,
    *,
    smoothing_window: int | None = 14,
    temporal_bucket: str | None = None,
    borough: str | None = None,
    cbd_spatial_category: str | None = None,
    mobility_regime_cluster_label: int | None = None,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Return daily trends indexed to each metric's pre-CP average."""
    metrics = metrics or CORE_METRICS

    daily = get_daily_metric_trends(
        metrics=metrics,
        temporal_bucket=temporal_bucket,
        borough=borough,
        cbd_spatial_category=cbd_spatial_category,
        mobility_regime_cluster_label=mobility_regime_cluster_label,
        date_range=date_range,
    )

    daily = add_rolling_average(
        daily,
        metrics=metrics,
        window=smoothing_window,
    )

    daily = add_indexed_to_pre_cp_average(
        daily,
        metrics=metrics,
    )

    return daily


@st.cache_data(show_spinner="Calculating pre/post summaries...")
def get_pre_post_metric_summary(
    metrics: list[str] | None = None,
    *,
    temporal_bucket: str | None = None,
    borough: str | None = None,
    cbd_spatial_category: str | None = None,
    mobility_regime_cluster_label: int | None = None,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Return descriptive pre/post summaries for selected metrics.

    Summaries are based on daily aggregated values. This avoids comparing full
    pre-period totals against a shorter post-period.
    """
    metrics = metrics or CORE_METRICS

    daily = get_daily_metric_trends(
        metrics=metrics,
        temporal_bucket=temporal_bucket,
        borough=borough,
        cbd_spatial_category=cbd_spatial_category,
        mobility_regime_cluster_label=mobility_regime_cluster_label,
        date_range=date_range,
    )

    rows = []

    for metric in metrics:
        pre_values = daily.loc[daily["pre_post_cp"] == "pre_cp", metric].dropna()
        post_values = daily.loc[daily["pre_post_cp"] == "post_cp", metric].dropna()

        pre_avg = pre_values.mean() if len(pre_values) else np.nan
        post_avg = post_values.mean() if len(post_values) else np.nan
        abs_change = (
            post_avg - pre_avg
            if pd.notna(pre_avg) and pd.notna(post_avg)
            else np.nan
        )

        if pd.notna(pre_avg) and pre_avg != 0 and pd.notna(post_avg):
            pct_change = (abs_change / pre_avg) * 100
        else:
            pct_change = np.nan

        rows.append(
            {
                "metric": metric,
                "metric_label": METRIC_LABELS.get(metric, metric),
                "aggregation": get_metric_spec(metric).aggregation_type,
                "pre_daily_average": pre_avg,
                "post_daily_average": post_avg,
                "absolute_change": abs_change,
                "percent_change": pct_change,
                "pre_observed_days": int(pre_values.index.nunique()),
                "post_observed_days": int(post_values.index.nunique()),
                "direction": _change_direction(abs_change),
            }
        )

    return pd.DataFrame(rows)


def _change_direction(abs_change: float) -> str:
    """Return a simple direction label for summary tables."""
    if pd.isna(abs_change):
        return "Unknown"
    if abs_change > 0:
        return "Higher post-CP"
    if abs_change < 0:
        return "Lower post-CP"
    return "No change"


def get_available_filter_values() -> dict[str, list[str]]:
    """Return available values for common app filters."""
    columns = [
        "borough",
        "cbd_spatial_category",
        "temporal_bucket",
    ]

    df = load_analysis_panel(columns=columns)
    raw_categories = set(
        df["cbd_spatial_category"].dropna().astype(str).unique().tolist()
    )

    app_categories = []
    for label, raw_values in CBD_SPATIAL_CATEGORY_GROUPS.items():
        if label == "All geography groups" or raw_values is None:
            continue
        if any(raw_value in raw_categories for raw_value in raw_values):
            app_categories.append(label)

    covered_raw_values = {
        raw_value
        for raw_values in CBD_SPATIAL_CATEGORY_GROUPS.values()
        if raw_values is not None
        for raw_value in raw_values
    }

    # Keep unexpected raw categories available rather than silently hiding them.
    for raw_value in sorted(raw_categories):
        if raw_value not in covered_raw_values:
            app_categories.append(raw_value)

    return {
        "boroughs": sorted(df["borough"].dropna().unique().tolist()),
        "cbd_spatial_categories": app_categories,
        "temporal_buckets": sorted(
            df["temporal_bucket"].dropna().astype(str).unique().tolist()
        ),
    }


def format_summary_for_display(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Format pre/post summary columns for Streamlit display."""
    display = summary_df.copy()

    numeric_cols = [
        "pre_daily_average",
        "post_daily_average",
        "absolute_change",
        "percent_change",
    ]

    for col in numeric_cols:
        if col in display.columns:
            display[col] = display[col].round(2)

    display = display.rename(
        columns={
            "metric_label": "Metric",
            "aggregation": "Aggregation",
            "pre_daily_average": "Pre-CP daily avg",
            "post_daily_average": "Post-CP daily avg",
            "absolute_change": "Abs. change",
            "percent_change": "% change",
            "pre_observed_days": "Pre observed days",
            "post_observed_days": "Post observed days",
            "direction": "Direction",
        }
    )

    keep_cols = [
        "Metric",
        "Pre-CP daily avg",
        "Post-CP daily avg",
        "Abs. change",
        "% change",
        "Direction",
        "Pre observed days",
        "Post observed days",
    ]

    return display[keep_cols]


def format_signed_percent(value: float, *, digits: int = 1) -> str:
    """Format a percent change with sign and percent symbol."""
    if pd.isna(value):
        return "unknown"
    return f"{value:+,.{digits}f}%"


def format_metric_value(value: float, *, digits: int = 2) -> str:
    """Format a numeric metric value with commas."""
    if pd.isna(value):
        return "unknown"
    return f"{value:,.{digits}f}"


def build_raw01_frozen_interpretation(summary_df: pd.DataFrame) -> str:
    """Build the top-page interpretation for raw_01 from actual summary values."""
    taxi_pct = summary_df.loc[
        summary_df["metric"] == "taxi_trip_count", "percent_change"
    ].iloc[0]
    subway_pct = summary_df.loc[
        summary_df["metric"] == "subway_ridership", "percent_change"
    ].iloc[0]
    fhvhv_pct = summary_df.loc[
        summary_df["metric"] == "fhvhv_trip_count", "percent_change"
    ].iloc[0]

    taxi_speed_pct = summary_df.loc[
        summary_df["metric"] == "taxi_avg_trip_speed", "percent_change"
    ].iloc[0]
    bus_speed_pct = summary_df.loc[
        summary_df["metric"] == "avg_bus_speed", "percent_change"
    ].iloc[0]
    fhvhv_speed_pct = summary_df.loc[
        summary_df["metric"] == "fhvhv_avg_trip_speed", "percent_change"
    ].iloc[0]

    return (
        f"The clearest post-CP shift is in **mobility activity**, not speed. "
        f"Taxi trips rose the most ({format_signed_percent(taxi_pct)}), while subway ridership "
        f"({format_signed_percent(subway_pct)}) and FHVHV trips ({format_signed_percent(fhvhv_pct)}) "
        f"also increased. Speed changes were much smaller: Taxi speed "
        f"({format_signed_percent(taxi_speed_pct)}) and Bus speed ({format_signed_percent(bus_speed_pct)}) "
        f"edged up, while FHVHV speed ({format_signed_percent(fhvhv_speed_pct)}) slipped."
    )


def build_selected_view_interpretation(
    summary_df: pd.DataFrame,
    *,
    metric: str,
    geography_scope: str,
    temporal_bucket: str,
    borough: str | None = None,
    cbd_spatial_category: str | None = None,
    mobility_regime_cluster_label: int | None = None,
) -> str:
    """Build a plain-English interpretation for the current interactive selection."""
    row = summary_df.loc[summary_df["metric"] == metric].iloc[0]

    label = row["metric_label"]
    pct_change = row["percent_change"]
    pre_avg = row["pre_daily_average"]
    post_avg = row["post_daily_average"]

    if geography_scope == "Citywide":
        geography_text = "citywide"
    elif geography_scope == "Borough" and borough:
        geography_text = f"in {borough}"
    elif geography_scope == "CBD spatial category" and cbd_spatial_category:
        geography_text = f"for {format_cbd_spatial_category_label(cbd_spatial_category)} zones"
    elif (
        geography_scope == "Mobility regime cluster"
        and mobility_regime_cluster_label is not None
    ):
        geography_text = (
            f"for {format_mobility_regime_cluster_label(mobility_regime_cluster_label)} zones"
        )
    else:
        geography_text = "for the selected geography"

    bucket_text = (
        "across all temporal buckets"
        if temporal_bucket == "All temporal buckets"
        else f"during {temporal_bucket.replace('_', ' ')}"
    )

    if pd.isna(pct_change):
        return (
            f"{label} {geography_text} {bucket_text} does not have enough observed data "
            f"to summarize the pre/post change."
        )

    if pct_change >= 10:
        strength = "a clear increase"
    elif pct_change >= 2:
        strength = "a modest increase"
    elif pct_change > -2:
        strength = "little net change"
    elif pct_change > -10:
        strength = "a modest decrease"
    else:
        strength = "a clear decrease"

    return (
        f"For this selection, <strong>{label} shows {strength}</strong> after congestion pricing: "
        f"{format_signed_percent(pct_change)} post-CP, moving from "
        f"{format_metric_value(pre_avg)} to {format_metric_value(post_avg)} per day on average "
        f"{geography_text} {bucket_text}."
    )


# =============================================================================
# Temporal-bucket summaries for View 2
# =============================================================================

WEEKDAY_TEMPORAL_BUCKETS = [
    "weekday_overnight",
    "weekday_am_peak",
    "weekday_midday",
    "weekday_pm_peak",
    "weekday_evening",
]

WEEKEND_TEMPORAL_BUCKETS = [
    "weekend_overnight",
    "weekend_am_peak",
    "weekend_midday",
    "weekend_pm_peak",
    "weekend_evening",
]

TEMPORAL_BUCKET_LABELS = {
    "weekday_overnight": "Weekday overnight",
    "weekday_am_peak": "Weekday AM peak",
    "weekday_midday": "Weekday midday",
    "weekday_pm_peak": "Weekday PM peak",
    "weekday_evening": "Weekday evening",
    "weekend_overnight": "Weekend overnight",
    "weekend_am_peak": "Weekend AM peak",
    "weekend_midday": "Weekend midday",
    "weekend_pm_peak": "Weekend PM peak",
    "weekend_evening": "Weekend evening",
}

TEMPORAL_BUCKET_WEEK_PART = {
    "weekday_overnight": "Weekday",
    "weekday_am_peak": "Weekday",
    "weekday_midday": "Weekday",
    "weekday_pm_peak": "Weekday",
    "weekday_evening": "Weekday",
    "weekend_overnight": "Weekend",
    "weekend_am_peak": "Weekend",
    "weekend_midday": "Weekend",
    "weekend_pm_peak": "Weekend",
    "weekend_evening": "Weekend",
}

TEMPORAL_BUCKET_DAYPART = {
    "weekday_overnight": "Overnight",
    "weekday_am_peak": "AM peak",
    "weekday_midday": "Midday",
    "weekday_pm_peak": "PM peak",
    "weekday_evening": "Evening",
    "weekend_overnight": "Overnight",
    "weekend_am_peak": "AM peak",
    "weekend_midday": "Midday",
    "weekend_pm_peak": "PM peak",
    "weekend_evening": "Evening",
}


def _filter_temporal_bucket_summary(
    df: pd.DataFrame,
    *,
    week_part: str | None = None,
) -> pd.DataFrame:
    """Filter temporal-bucket summary rows by weekday/weekend grouping."""
    if week_part is None or week_part == "All buckets":
        return df

    if week_part == "Weekday only":
        return df[df["temporal_bucket"].isin(WEEKDAY_TEMPORAL_BUCKETS)]

    if week_part == "Weekend only":
        return df[df["temporal_bucket"].isin(WEEKEND_TEMPORAL_BUCKETS)]

    raise ValueError(
        "week_part must be one of: None, 'All buckets', 'Weekday only', 'Weekend only'"
    )


def _add_temporal_bucket_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add display labels and ordering metadata for temporal buckets."""
    result = df.copy()

    bucket_order = {
        bucket: idx
        for idx, bucket in enumerate(
            WEEKDAY_TEMPORAL_BUCKETS + WEEKEND_TEMPORAL_BUCKETS,
            start=1,
        )
    }

    result["temporal_bucket"] = result["temporal_bucket"].astype(str)
    result["temporal_bucket_label"] = result["temporal_bucket"].map(
        TEMPORAL_BUCKET_LABELS
    )
    result["temporal_bucket_order"] = result["temporal_bucket"].map(bucket_order)
    result["week_part"] = result["temporal_bucket"].map(TEMPORAL_BUCKET_WEEK_PART)
    result["daypart_label"] = result["temporal_bucket"].map(TEMPORAL_BUCKET_DAYPART)

    return result


@st.cache_data(show_spinner="Calculating temporal-bucket summaries...")
def get_temporal_bucket_metric_summary(
    metrics: list[str] | None = None,
    *,
    borough: str | None = None,
    cbd_spatial_category: str | None = None,
    mobility_regime_cluster_label: int | None = None,
    week_part: str | None = None,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Return pre/post summaries by temporal bucket for selected metrics.

    Output grain:
        metric × temporal_bucket

    Aggregation policy:
        1. Aggregate to Date × Temporal Bucket using the app metric rules.
        2. Compare pre-CP and post-CP daily averages within each temporal bucket.

    This avoids comparing total pre-period volume against a shorter post-period.
    """
    metrics = metrics or CORE_METRICS
    columns = get_required_columns(metrics)

    df = load_analysis_panel(columns=columns)
    df = apply_common_filters(
        df,
        temporal_bucket=None,
        borough=borough,
        cbd_spatial_category=cbd_spatial_category,
        mobility_regime_cluster_label=mobility_regime_cluster_label,
        date_range=date_range,
    )

    df = _filter_temporal_bucket_summary(df, week_part=week_part)

    daily_bucket = aggregate_metrics(
        df,
        metrics=metrics,
        group_cols=["date", "temporal_bucket"],
    )

    daily_bucket["pre_post_cp"] = np.where(
        daily_bucket["date"] >= CONGESTION_PRICING_START_DATE,
        "post_cp",
        "pre_cp",
    )

    daily_bucket = _add_temporal_bucket_metadata(daily_bucket)

    rows = []

    for metric in metrics:
        metric_label = METRIC_LABELS.get(metric, metric)
        metric_spec = get_metric_spec(metric)

        for temporal_bucket, bucket_df in daily_bucket.groupby(
            "temporal_bucket",
            observed=True,
            dropna=False,
        ):
            pre_values = bucket_df.loc[
                bucket_df["pre_post_cp"] == "pre_cp", metric
            ].dropna()
            post_values = bucket_df.loc[
                bucket_df["pre_post_cp"] == "post_cp", metric
            ].dropna()

            pre_avg = pre_values.mean() if len(pre_values) else np.nan
            post_avg = post_values.mean() if len(post_values) else np.nan

            if pd.notna(pre_avg) and pd.notna(post_avg):
                abs_change = post_avg - pre_avg
            else:
                abs_change = np.nan

            if pd.notna(pre_avg) and pre_avg != 0 and pd.notna(post_avg):
                pct_change = (abs_change / pre_avg) * 100
            else:
                pct_change = np.nan

            rows.append(
                {
                    "metric": metric,
                    "metric_label": metric_label,
                    "aggregation": metric_spec.aggregation_type,
                    "temporal_bucket": str(temporal_bucket),
                    "pre_daily_average": pre_avg,
                    "post_daily_average": post_avg,
                    "absolute_change": abs_change,
                    "percent_change": pct_change,
                    "pre_observed_days": int(pre_values.index.nunique()),
                    "post_observed_days": int(post_values.index.nunique()),
                    "direction": _change_direction(abs_change),
                }
            )

    summary = pd.DataFrame(rows)

    if summary.empty:
        return summary

    summary = _add_temporal_bucket_metadata(summary)

    summary = summary.sort_values(
        ["metric", "temporal_bucket_order"],
        ascending=[True, True],
    ).reset_index(drop=True)

    return summary


def filter_temporal_bucket_summary_for_metric(
    summary_df: pd.DataFrame,
    *,
    metric: str,
    week_part: str | None = None,
    sort_mode: str = "Temporal order",
) -> pd.DataFrame:
    """Filter and sort a temporal-bucket summary for one selected metric."""
    result = summary_df[summary_df["metric"] == metric].copy()

    result = _filter_temporal_bucket_summary(result, week_part=week_part)

    if sort_mode == "Temporal order":
        result = result.sort_values("temporal_bucket_order")
    elif sort_mode == "Largest increase":
        result = result.sort_values("percent_change", ascending=False)
    elif sort_mode == "Largest decrease":
        result = result.sort_values("percent_change", ascending=True)
    elif sort_mode == "Largest absolute difference":
        result = result.assign(
            abs_percent_change=result["percent_change"].abs()
        ).sort_values("abs_percent_change", ascending=False)
    else:
        raise ValueError(
            "sort_mode must be one of: 'Temporal order', 'Largest increase', "
            "'Largest decrease', 'Largest absolute difference'"
        )

    return result.reset_index(drop=True)


def format_temporal_bucket_summary_for_display(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Format temporal-bucket summaries for Streamlit display."""
    display = summary_df.copy()

    display = display.rename(
        columns={
            "temporal_bucket_label": "Temporal bucket",
            "week_part": "Week part",
            "daypart_label": "Daypart",
            "metric_label": "Metric",
            "pre_daily_average": "Pre-CP daily avg",
            "post_daily_average": "Post-CP daily avg",
            "absolute_change": "Abs. change",
            "percent_change": "% change",
            "direction": "Direction",
            "pre_observed_days": "Pre observed days",
            "post_observed_days": "Post observed days",
        }
    )

    keep_cols = [
        "Temporal bucket",
        "Week part",
        "Daypart",
        "Metric",
        "Pre-CP daily avg",
        "Post-CP daily avg",
        "Abs. change",
        "% change",
        "Direction",
        "Pre observed days",
        "Post observed days",
    ]

    return display[keep_cols]
