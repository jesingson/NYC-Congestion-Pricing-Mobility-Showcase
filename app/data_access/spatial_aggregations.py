"""Precomputed spatial summaries for Raw View 03.

Normal app behavior
-------------------
The Streamlit page reads a compact precomputed parquet at:

    data/processed/app_tables/spatial_zone_pre_post_summary.parquet

The table is rebuilt only when:
- it does not exist, or
- REBUILD_SPATIAL_APP_TABLES is set to a truthy value.

The standalone build script is the preferred way to refresh the asset:

    python scripts/build_spatial_app_tables.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from app.data_access.aggregations import (
    COUNT_METRICS,
    SPEED_METRICS,
    WEIGHT_COLUMNS,
    get_metric_spec,
    get_required_columns,
)
from app.data_access.loaders import (
    CONGESTION_PRICING_START_DATE,
    CORE_METRICS,
    METRIC_LABELS,
    load_analysis_panel,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_TABLE_DIR = REPO_ROOT / "data" / "processed" / "app_tables"
SPATIAL_SUMMARY_PATH = (
    APP_TABLE_DIR / "spatial_zone_pre_post_summary.parquet"
)

ZONE_ID_COLUMN = "taxi_zone_id"
ZONE_NAME_COLUMN = "zone"

ALL_TEMPORAL_BUCKETS_LABEL = "All temporal buckets"

ZONE_METADATA_COLUMNS = [
    ZONE_ID_COLUMN,
    ZONE_NAME_COLUMN,
    "borough",
    "cbd_spatial_category",
]


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _safe_percent_change(
    pre_values: pd.Series,
    post_values: pd.Series,
) -> pd.Series:
    valid = pre_values.notna() & post_values.notna() & pre_values.ne(0)
    result = pd.Series(np.nan, index=pre_values.index, dtype="float64")
    result.loc[valid] = (
        (post_values.loc[valid] - pre_values.loc[valid])
        / pre_values.loc[valid]
        * 100
    )
    return result


def _direction(values: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [values.isna(), values.gt(0), values.lt(0)],
            ["Unknown", "Increase", "Decrease"],
            default="No change",
        ),
        index=values.index,
    )


def _mode_or_first(values: pd.Series):
    non_null = values.dropna()
    if non_null.empty:
        return np.nan

    modes = non_null.mode()
    return modes.iloc[0] if not modes.empty else non_null.iloc[0]


def _support_columns(metrics: list[str]) -> list[str]:
    return sorted(
        {
            support
            for metric in metrics
            for support in [WEIGHT_COLUMNS.get(metric)]
            if support is not None
        }
    )


def _build_zone_metadata(df: pd.DataFrame) -> pd.DataFrame:
    available = [
        column
        for column in ZONE_METADATA_COLUMNS
        if column in df.columns
    ]

    aggregation_map = {
        column: _mode_or_first
        for column in available
        if column != ZONE_ID_COLUMN
    }

    return (
        df[available]
        .groupby(
            ZONE_ID_COLUMN,
            observed=True,
            dropna=False,
            sort=False,
        )
        .agg(aggregation_map)
        .reset_index()
    )


def _aggregate_zone_daily(
    df: pd.DataFrame,
    *,
    metrics: list[str],
    include_temporal_bucket: bool,
) -> pd.DataFrame:
    """Vectorized Taxi Zone daily aggregation."""
    group_cols = [ZONE_ID_COLUMN, "date"]
    if include_temporal_bucket:
        group_cols.append("temporal_bucket")

    count_metrics = [
        metric for metric in metrics
        if metric in COUNT_METRICS
    ]
    speed_metrics = [
        metric for metric in metrics
        if metric in SPEED_METRICS
    ]
    support_columns = _support_columns(metrics)

    summed_columns = sorted(set(count_metrics + support_columns))
    pieces: list[pd.DataFrame] = []

    if summed_columns:
        summed = (
            df.groupby(
                group_cols,
                observed=True,
                dropna=False,
                sort=False,
            )[summed_columns]
            .sum(min_count=1)
        )
        pieces.append(summed)

    for metric in speed_metrics:
        support_column = WEIGHT_COLUMNS.get(metric)

        if support_column and support_column in df.columns:
            valid = (
                df[metric].notna()
                & df[support_column].notna()
                & df[support_column].gt(0)
            )

            temp = df[group_cols].copy()
            temp["_weighted_value"] = np.where(
                valid,
                df[metric] * df[support_column],
                0.0,
            )
            temp["_weight"] = np.where(
                valid,
                df[support_column],
                0.0,
            )

            grouped = (
                temp.groupby(
                    group_cols,
                    observed=True,
                    dropna=False,
                    sort=False,
                )[["_weighted_value", "_weight"]]
                .sum()
            )

            weighted = (
                grouped["_weighted_value"]
                / grouped["_weight"].replace(0, np.nan)
            )

            fallback = (
                df.groupby(
                    group_cols,
                    observed=True,
                    dropna=False,
                    sort=False,
                )[metric]
                .mean()
            )

            pieces.append(
                weighted.fillna(fallback).rename(metric).to_frame()
            )
        else:
            pieces.append(
                df.groupby(
                    group_cols,
                    observed=True,
                    dropna=False,
                    sort=False,
                )[metric]
                .mean()
                .to_frame()
            )

    if not pieces:
        return pd.DataFrame()

    daily = pd.concat(pieces, axis=1)
    daily = daily.loc[:, ~daily.columns.duplicated()].reset_index()

    return daily


def _summarize_one_metric(
    daily: pd.DataFrame,
    *,
    metric: str,
    temporal_bucket_label: str,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    support_metric = WEIGHT_COLUMNS.get(metric) or metric

    columns = [ZONE_ID_COLUMN, "date", metric]
    if support_metric != metric and support_metric in daily.columns:
        columns.append(support_metric)

    metric_daily = daily[columns].copy()
    metric_daily["pre_post_cp"] = np.where(
        pd.to_datetime(metric_daily["date"])
        >= CONGESTION_PRICING_START_DATE,
        "post_cp",
        "pre_cp",
    )

    aggregation = {
        "metric_daily_average": (metric, "mean"),
        "metric_observed_days": (metric, "count"),
    }

    if support_metric in metric_daily.columns:
        aggregation["support_daily_average"] = (
            support_metric,
            "mean",
        )

    grouped = (
        metric_daily.groupby(
            [ZONE_ID_COLUMN, "pre_post_cp"],
            observed=True,
            dropna=False,
            sort=False,
        )
        .agg(**aggregation)
        .reset_index()
    )

    def pivot_value(value_column: str, rename_map: dict[str, str]):
        result = grouped.pivot(
            index=ZONE_ID_COLUMN,
            columns="pre_post_cp",
            values=value_column,
        ).reset_index()
        result.columns.name = None
        return result.rename(columns=rename_map)

    metric_values = pivot_value(
        "metric_daily_average",
        {
            "pre_cp": "pre_daily_average",
            "post_cp": "post_daily_average",
        },
    )
    observed_days = pivot_value(
        "metric_observed_days",
        {
            "pre_cp": "pre_observed_days",
            "post_cp": "post_observed_days",
        },
    )

    summary = metric_values.merge(
        observed_days,
        on=ZONE_ID_COLUMN,
        how="outer",
        validate="one_to_one",
    )

    if "support_daily_average" in grouped.columns:
        support_values = pivot_value(
            "support_daily_average",
            {
                "pre_cp": "pre_support_daily_average",
                "post_cp": "post_support_daily_average",
            },
        )
        summary = summary.merge(
            support_values,
            on=ZONE_ID_COLUMN,
            how="left",
            validate="one_to_one",
        )
    else:
        summary["pre_support_daily_average"] = (
            summary["pre_daily_average"]
        )
        summary["post_support_daily_average"] = (
            summary["post_daily_average"]
        )

    summary = summary.merge(
        metadata,
        on=ZONE_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    for column in [
        "pre_daily_average",
        "post_daily_average",
        "pre_support_daily_average",
        "post_support_daily_average",
        "pre_observed_days",
        "post_observed_days",
    ]:
        if column not in summary.columns:
            summary[column] = np.nan

    summary["metric"] = metric
    summary["metric_label"] = METRIC_LABELS.get(metric, metric)
    summary["aggregation"] = get_metric_spec(metric).aggregation_type
    summary["support_metric"] = support_metric
    summary["temporal_bucket"] = temporal_bucket_label

    summary["absolute_change"] = (
        summary["post_daily_average"]
        - summary["pre_daily_average"]
    )
    summary["percent_change"] = _safe_percent_change(
        summary["pre_daily_average"],
        summary["post_daily_average"],
    )

    summary["pre_observed_days"] = (
        summary["pre_observed_days"].fillna(0).astype(int)
    )
    summary["post_observed_days"] = (
        summary["post_observed_days"].fillna(0).astype(int)
    )
    summary["direction"] = _direction(summary["absolute_change"])
    summary["has_both_periods"] = (
        summary["pre_observed_days"].gt(0)
        & summary["post_observed_days"].gt(0)
    )

    return summary


def build_spatial_zone_pre_post_summary(
    *,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Build all-zone summaries for all buckets plus the all-bucket rollup."""
    metrics = metrics or CORE_METRICS
    columns = get_required_columns(metrics)

    df = load_analysis_panel(columns=columns)
    df["date"] = pd.to_datetime(df["date"])

    metadata = _build_zone_metadata(df)
    summaries: list[pd.DataFrame] = []

    # All temporal buckets combined.
    daily_all = _aggregate_zone_daily(
        df,
        metrics=metrics,
        include_temporal_bucket=False,
    )

    for metric in metrics:
        summaries.append(
            _summarize_one_metric(
                daily_all,
                metric=metric,
                temporal_bucket_label=ALL_TEMPORAL_BUCKETS_LABEL,
                metadata=metadata,
            )
        )

    # Individual temporal buckets.
    daily_bucket = _aggregate_zone_daily(
        df,
        metrics=metrics,
        include_temporal_bucket=True,
    )

    for temporal_bucket, bucket_df in daily_bucket.groupby(
        "temporal_bucket",
        observed=True,
        dropna=False,
        sort=False,
    ):
        bucket_df = bucket_df.drop(columns=["temporal_bucket"])

        for metric in metrics:
            summaries.append(
                _summarize_one_metric(
                    bucket_df,
                    metric=metric,
                    temporal_bucket_label=str(temporal_bucket),
                    metadata=metadata,
                )
            )

    summary = pd.concat(summaries, ignore_index=True)

    ordered_columns = [
        ZONE_ID_COLUMN,
        ZONE_NAME_COLUMN,
        "borough",
        "cbd_spatial_category",
        "temporal_bucket",
        "metric",
        "metric_label",
        "aggregation",
        "support_metric",
        "pre_support_daily_average",
        "post_support_daily_average",
        "pre_daily_average",
        "post_daily_average",
        "absolute_change",
        "percent_change",
        "pre_observed_days",
        "post_observed_days",
        "direction",
        "has_both_periods",
    ]

    return (
        summary[ordered_columns]
        .sort_values(
            ["temporal_bucket", "metric", ZONE_ID_COLUMN]
        )
        .reset_index(drop=True)
    )


def write_spatial_app_table(
    summary: pd.DataFrame,
    *,
    output_path: Path = SPATIAL_SUMMARY_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(output_path, index=False)
    return output_path


def ensure_spatial_app_table(
    *,
    force_rebuild: bool | None = None,
) -> Path:
    """Build the compact summary only when missing or explicitly requested."""
    if force_rebuild is None:
        force_rebuild = _env_flag(
            "REBUILD_SPATIAL_APP_TABLES",
            default=False,
        )

    if SPATIAL_SUMMARY_PATH.exists() and not force_rebuild:
        return SPATIAL_SUMMARY_PATH

    summary = build_spatial_zone_pre_post_summary()
    return write_spatial_app_table(summary)


@st.cache_data(show_spinner="Loading spatial app table...")
def load_spatial_zone_pre_post_summary() -> pd.DataFrame:
    path = ensure_spatial_app_table()
    return pd.read_parquet(path)


def get_zone_pre_post_metric_summary(
    metrics: list[str] | None = None,
    *,
    temporal_bucket: str | None = None,
) -> pd.DataFrame:
    """Read and filter the compact precomputed spatial summary."""
    metrics = metrics or CORE_METRICS
    bucket = temporal_bucket or ALL_TEMPORAL_BUCKETS_LABEL

    summary = load_spatial_zone_pre_post_summary()

    return summary[
        summary["metric"].isin(metrics)
        & summary["temporal_bucket"].eq(bucket)
    ].copy().reset_index(drop=True)


def get_zone_rankings(
    summary_df: pd.DataFrame,
    *,
    metric: str,
    value_column: str = "percent_change",
    sort_mode: str = "High to low",
    limit: int | None = 15,
) -> pd.DataFrame:
    result = summary_df[
        summary_df["metric"].eq(metric)
        & summary_df["has_both_periods"]
        & summary_df[value_column].notna()
    ].copy()

    if sort_mode == "High to low":
        result = result.sort_values(value_column, ascending=False)
    elif sort_mode == "Low to high":
        result = result.sort_values(value_column, ascending=True)
    elif sort_mode == "Largest absolute shift":
        result = (
            result.assign(_absolute_sort=result[value_column].abs())
            .sort_values("_absolute_sort", ascending=False)
            .drop(columns="_absolute_sort")
        )
    else:
        raise ValueError(f"Unsupported sort mode: {sort_mode}")

    if limit is not None:
        result = result.head(limit)

    return result.reset_index(drop=True)


def format_zone_summary_for_display(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    display = summary_df.copy()

    for column in [
        "pre_daily_average",
        "post_daily_average",
        "absolute_change",
        "percent_change",
        "pre_support_daily_average",
    ]:
        if column in display.columns:
            display[column] = display[column].round(2)

    display = display.rename(
        columns={
            "zone": "Taxi Zone",
            "borough": "Borough",
            "metric_label": "Metric",
            "pre_daily_average": "Pre-CP daily avg",
            "post_daily_average": "Post-CP daily avg",
            "absolute_change": "Daily-average change",
            "percent_change": "% change",
            "pre_support_daily_average": "Pre-CP support avg",
            "direction": "Direction",
            "pre_observed_days": "Pre days",
            "post_observed_days": "Post days",
        }
    )

    keep = [
        column
        for column in [
            "Taxi Zone",
            "Borough",
            "Metric",
            "Pre-CP daily avg",
            "Post-CP daily avg",
            "Daily-average change",
            "% change",
            "Pre-CP support avg",
            "Direction",
            "Pre days",
            "Post days",
        ]
        if column in display.columns
    ]

    return display[keep]