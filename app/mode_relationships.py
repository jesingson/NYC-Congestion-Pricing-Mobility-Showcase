from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from app.data_access.loaders import CORE_METRICS
from app.data_access.spatial_aggregations import (
    ALL_TEMPORAL_BUCKETS_LABEL,
    get_zone_pre_post_metric_summary,
)
from app.data_access.spatial_visuals import add_reliability_flags


COUNT_METRICS = {
    "taxi_trip_count",
    "fhvhv_trip_count",
    "subway_ridership",
}

SPEED_METRICS = {
    "taxi_avg_trip_speed",
    "fhvhv_avg_trip_speed",
    "avg_bus_speed",
}

SUPPORTED_AGGREGATION_LEVELS = (
    "Taxi Zone",
    "Borough",
    "Citywide",
)

RELATIONSHIP_GRAIN_COLUMNS = [
    "geography_level",
    "geography_id",
    "geography_name",
    "borough",
    "cbd_spatial_category",
    "temporal_bucket",
    "period",
    "metric",
]


def _safe_weighted_average(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    valid = (
        values.notna()
        & weights.notna()
        & weights.gt(0)
    )

    if valid.any():
        return float(
            np.average(
                values.loc[valid].astype(float),
                weights=weights.loc[valid].astype(float),
            )
        )

    fallback = values.dropna()

    if fallback.empty:
        return np.nan

    return float(fallback.mean())


def _safe_percent_change(
    pre_value: float,
    post_value: float,
) -> float:
    if (
        pd.isna(pre_value)
        or pd.isna(post_value)
        or pre_value == 0
    ):
        return np.nan

    return float(
        (post_value - pre_value)
        / pre_value
        * 100
    )


def _first_valid_value(
    values: pd.Series,
    *,
    fallback: object = np.nan,
) -> object:
    valid = values.dropna()

    if valid.empty:
        return fallback

    return valid.iloc[0]


def _normalize_zone_summary(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    if summary.empty:
        return summary

    result = summary.copy()

    optional_defaults: dict[str, object] = {
        "taxi_zone_id": np.nan,
        "zone": "Unknown",
        "borough": "Unknown",
        "cbd_spatial_category": "Unknown",
        "metric_label": result["metric"]
        if "metric" in result.columns
        else "Unknown",
        "aggregation": "Unknown",
        "support_metric": "Unknown",
        "pre_support_daily_average": np.nan,
        "post_support_daily_average": np.nan,
        "pre_observed_days": 0,
        "post_observed_days": 0,
        "has_both_periods": False,
    }

    for column, default in optional_defaults.items():
        if column not in result.columns:
            result[column] = default

    result["taxi_zone_id"] = pd.to_numeric(
        result["taxi_zone_id"],
        errors="coerce",
    )

    for column in [
        "pre_daily_average",
        "post_daily_average",
        "absolute_change",
        "percent_change",
        "pre_support_daily_average",
        "post_support_daily_average",
        "pre_observed_days",
        "post_observed_days",
    ]:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    result = add_reliability_flags(result)

    return result


def load_zone_relationship_summary(
    *,
    metrics: Iterable[str] | None = None,
    temporal_bucket: str = ALL_TEMPORAL_BUCKETS_LABEL,
    apply_reliability_thresholds: bool = True,
) -> pd.DataFrame:
    """
    Return the canonical Taxi Zone-level pre/post metric summary.

    The result remains wide by period so it can be reused for:
    - validation
    - relationship pairing
    - higher-level geographic aggregation
    """
    selected_metrics = list(metrics or CORE_METRICS)

    summary = get_zone_pre_post_metric_summary(
        metrics=selected_metrics,
        temporal_bucket=temporal_bucket,
    )

    if summary.empty:
        return summary

    result = _normalize_zone_summary(summary)

    result = result[
        result["metric"].isin(selected_metrics)
        & result["has_both_periods"]
    ].copy()

    if apply_reliability_thresholds:
        result = result[
            result["eligible_for_percent_change"]
        ].copy()

    result["temporal_bucket"] = temporal_bucket

    return (
        result.sort_values(
            [
                "metric",
                "taxi_zone_id",
            ]
        )
        .reset_index(drop=True)
    )


def reshape_relationship_periods(
    zone_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert a wide pre/post summary into one row per zone × metric × period.
    """
    if zone_summary.empty:
        return pd.DataFrame()

    identifier_columns = [
        column
        for column in [
            "taxi_zone_id",
            "zone",
            "borough",
            "cbd_spatial_category",
            "temporal_bucket",
            "metric",
            "metric_label",
            "aggregation",
            "support_metric",
            "eligible_for_percent_change",
        ]
        if column in zone_summary.columns
    ]

    frames: list[pd.DataFrame] = []

    for period in ["pre", "post"]:
        period_frame = zone_summary[
            identifier_columns
        ].copy()

        period_frame["period"] = (
            "Pre-CP"
            if period == "pre"
            else "Post-CP"
        )

        period_frame["metric_value"] = zone_summary[
            f"{period}_daily_average"
        ]

        period_frame["support_value"] = zone_summary[
            f"{period}_support_daily_average"
        ]

        period_frame["observed_days"] = zone_summary[
            f"{period}_observed_days"
        ]

        frames.append(period_frame)

    return (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "metric",
                "taxi_zone_id",
                "period",
            ]
        )
        .reset_index(drop=True)
    )


def _aggregate_metric_group(
    group: pd.DataFrame,
) -> pd.Series:
    metric = str(group["metric"].iloc[0])

    if metric in COUNT_METRICS:
        metric_value = group["metric_value"].sum(
            min_count=1
        )
    elif metric in SPEED_METRICS:
        metric_value = _safe_weighted_average(
            group["metric_value"],
            group["support_value"],
        )
    else:
        metric_value = group["metric_value"].mean()

    support_value = group["support_value"].sum(
        min_count=1
    )

    observed_days = group["observed_days"].max()

    return pd.Series(
        {
            "metric_value": metric_value,
            "support_value": support_value,
            "observed_days": observed_days,
            "zone_count": int(
                group["taxi_zone_id"].nunique()
            ),
        }
    )


def aggregate_relationship_geography(
    period_data: pd.DataFrame,
    *,
    aggregation_level: str,
) -> pd.DataFrame:
    """
    Aggregate period-level relationship data to the requested geography.

    Count metrics are summed.
    Speed metrics are support-weighted.
    """
    if aggregation_level not in SUPPORTED_AGGREGATION_LEVELS:
        raise ValueError(
            "aggregation_level must be one of "
            f"{SUPPORTED_AGGREGATION_LEVELS}; "
            f"received {aggregation_level!r}."
        )

    if period_data.empty:
        return pd.DataFrame()

    source = period_data.copy()

    if aggregation_level == "Taxi Zone":
        result = source.copy()

        result["geography_level"] = "Taxi Zone"
        result["geography_id"] = (
            result["taxi_zone_id"]
            .astype("Int64")
            .astype(str)
        )
        result["geography_name"] = result["zone"]
        result["zone_count"] = 1

        return (
            result[
                [
                    "geography_level",
                    "geography_id",
                    "geography_name",
                    "taxi_zone_id",
                    "zone",
                    "borough",
                    "cbd_spatial_category",
                    "temporal_bucket",
                    "period",
                    "metric",
                    "metric_label",
                    "aggregation",
                    "support_metric",
                    "metric_value",
                    "support_value",
                    "observed_days",
                    "zone_count",
                ]
            ]
            .sort_values(
                [
                    "metric",
                    "period",
                    "taxi_zone_id",
                ]
            )
            .reset_index(drop=True)
        )

    if aggregation_level == "Borough":
        group_columns = [
            "borough",
            "temporal_bucket",
            "period",
            "metric",
            "metric_label",
            "aggregation",
            "support_metric",
        ]

        result = (
            source.groupby(
                group_columns,
                observed=True,
                dropna=False,
            )
            .apply(
                _aggregate_metric_group,
                include_groups=False,
            )
            .reset_index()
        )

        result["geography_level"] = "Borough"
        result["geography_id"] = result["borough"].astype(str)
        result["geography_name"] = result["borough"]
        result["taxi_zone_id"] = pd.NA
        result["zone"] = pd.NA
        result["cbd_spatial_category"] = "Mixed"

    else:
        group_columns = [
            "temporal_bucket",
            "period",
            "metric",
            "metric_label",
            "aggregation",
            "support_metric",
        ]

        result = (
            source.groupby(
                group_columns,
                observed=True,
                dropna=False,
            )
            .apply(
                _aggregate_metric_group,
                include_groups=False,
            )
            .reset_index()
        )

        result["geography_level"] = "Citywide"
        result["geography_id"] = "NYC"
        result["geography_name"] = "New York City"
        result["taxi_zone_id"] = pd.NA
        result["zone"] = pd.NA
        result["borough"] = "All boroughs"
        result["cbd_spatial_category"] = "Mixed"

    return (
        result[
            [
                "geography_level",
                "geography_id",
                "geography_name",
                "taxi_zone_id",
                "zone",
                "borough",
                "cbd_spatial_category",
                "temporal_bucket",
                "period",
                "metric",
                "metric_label",
                "aggregation",
                "support_metric",
                "metric_value",
                "support_value",
                "observed_days",
                "zone_count",
            ]
        ]
        .sort_values(
            [
                "metric",
                "period",
                "geography_name",
            ]
        )
        .reset_index(drop=True)
    )


def build_relationship_long_data(
    *,
    metrics: Iterable[str] | None = None,
    temporal_bucket: str = ALL_TEMPORAL_BUCKETS_LABEL,
    aggregation_level: str = "Taxi Zone",
    apply_reliability_thresholds: bool = True,
) -> pd.DataFrame:
    """
    Build canonical relationship data at one geography × period × metric row.
    """
    zone_summary = load_zone_relationship_summary(
        metrics=metrics,
        temporal_bucket=temporal_bucket,
        apply_reliability_thresholds=apply_reliability_thresholds,
    )

    period_data = reshape_relationship_periods(
        zone_summary
    )

    return aggregate_relationship_geography(
        period_data,
        aggregation_level=aggregation_level,
    )


def build_metric_pair_data(
    relationship_data: pd.DataFrame,
    *,
    x_metric: str,
    y_metric: str,
) -> pd.DataFrame:
    """
    Pair two metrics at a shared geography × temporal bucket × period grain.
    """
    if x_metric == y_metric:
        raise ValueError(
            "x_metric and y_metric must be different."
        )

    if relationship_data.empty:
        return pd.DataFrame()

    key_columns = [
        "geography_level",
        "geography_id",
        "geography_name",
        "taxi_zone_id",
        "zone",
        "borough",
        "cbd_spatial_category",
        "temporal_bucket",
        "period",
    ]

    x_columns = key_columns + [
        "metric_label",
        "metric_value",
        "support_value",
        "observed_days",
        "zone_count",
    ]

    y_columns = key_columns + [
        "metric_label",
        "metric_value",
        "support_value",
        "observed_days",
        "zone_count",
    ]

    x_data = relationship_data[
        relationship_data["metric"].eq(x_metric)
    ][x_columns].copy()

    y_data = relationship_data[
        relationship_data["metric"].eq(y_metric)
    ][y_columns].copy()

    x_data = x_data.rename(
        columns={
            "metric_label": "x_metric_label",
            "metric_value": "x_value",
            "support_value": "x_support_value",
            "observed_days": "x_observed_days",
            "zone_count": "x_zone_count",
        }
    )

    y_data = y_data.rename(
        columns={
            "metric_label": "y_metric_label",
            "metric_value": "y_value",
            "support_value": "y_support_value",
            "observed_days": "y_observed_days",
            "zone_count": "y_zone_count",
        }
    )

    pair = x_data.merge(
        y_data,
        on=key_columns,
        how="inner",
        validate="one_to_one",
    )

    pair["pair_support_value"] = pair[
        [
            "x_support_value",
            "y_support_value",
        ]
    ].min(axis=1)

    pair["pair_observed_days"] = pair[
        [
            "x_observed_days",
            "y_observed_days",
        ]
    ].min(axis=1)

    pair["zone_count"] = pair[
        [
            "x_zone_count",
            "y_zone_count",
        ]
    ].min(axis=1)

    return (
        pair.sort_values(
            [
                "period",
                "geography_name",
            ]
        )
        .reset_index(drop=True)
    )


def build_pre_post_pair_comparison(
    pair_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return one row per geography with pre/post values for both metrics.
    """
    if pair_data.empty:
        return pd.DataFrame()

    identifiers = [
        "geography_level",
        "geography_id",
        "geography_name",
        "taxi_zone_id",
        "zone",
        "borough",
        "cbd_spatial_category",
        "temporal_bucket",
        "x_metric_label",
        "y_metric_label",
    ]

    value_columns = [
        "x_value",
        "y_value",
        "x_support_value",
        "y_support_value",
        "pair_support_value",
        "pair_observed_days",
        "zone_count",
    ]

    wide = pair_data.pivot_table(
        index=identifiers,
        columns="period",
        values=value_columns,
        aggfunc="first",
        observed=True,
    )

    wide.columns = [
        f"{value}_{str(period).lower().replace('-', '_')}"
        for value, period in wide.columns
    ]

    wide = wide.reset_index()

    expected_columns = {
        "x_value_pre_cp": np.nan,
        "x_value_post_cp": np.nan,
        "y_value_pre_cp": np.nan,
        "y_value_post_cp": np.nan,
    }

    for column, default in expected_columns.items():
        if column not in wide.columns:
            wide[column] = default

    wide["x_absolute_change"] = (
        wide["x_value_post_cp"]
        - wide["x_value_pre_cp"]
    )

    wide["y_absolute_change"] = (
        wide["y_value_post_cp"]
        - wide["y_value_pre_cp"]
    )

    wide["x_percent_change"] = wide.apply(
        lambda row: _safe_percent_change(
            row["x_value_pre_cp"],
            row["x_value_post_cp"],
        ),
        axis=1,
    )

    wide["y_percent_change"] = wide.apply(
        lambda row: _safe_percent_change(
            row["y_value_pre_cp"],
            row["y_value_post_cp"],
        ),
        axis=1,
    )

    x_direction = np.sign(
        wide["x_absolute_change"]
    )

    y_direction = np.sign(
        wide["y_absolute_change"]
    )

    wide["movement_relationship"] = np.select(
        [
            x_direction.gt(0) & y_direction.gt(0),
            x_direction.lt(0) & y_direction.lt(0),
            x_direction.gt(0) & y_direction.lt(0),
            x_direction.lt(0) & y_direction.gt(0),
        ],
        [
            "Both increased",
            "Both decreased",
            "X increased · Y decreased",
            "X decreased · Y increased",
        ],
        default="Mixed or unchanged",
    )

    wide["moved_together"] = (
        x_direction.eq(y_direction)
        & x_direction.ne(0)
    )

    return (
        wide.sort_values(
            "geography_name"
        )
        .reset_index(drop=True)
    )


def calculate_pair_correlations(
    pair_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate Pearson and Spearman correlations by period.

    Correlations require at least three complete observations.
    """
    if pair_data.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []

    for period, period_df in pair_data.groupby(
        "period",
        observed=True,
        sort=False,
    ):
        valid = period_df[
            [
                "x_value",
                "y_value",
            ]
        ].dropna()

        observation_count = len(valid)

        if observation_count >= 3:
            pearson = valid["x_value"].corr(
                valid["y_value"],
                method="pearson",
            )

            spearman = valid["x_value"].corr(
                valid["y_value"],
                method="spearman",
            )
        else:
            pearson = np.nan
            spearman = np.nan

        rows.append(
            {
                "period": period,
                "observation_count": observation_count,
                "pearson_correlation": pearson,
                "spearman_correlation": spearman,
            }
        )

    return pd.DataFrame(rows)


def validate_relationship_data(
    relationship_data: pd.DataFrame,
) -> dict[str, object]:
    """
    Return QA checks for a relationship dataset.
    """
    if relationship_data.empty:
        return {
            "row_count": 0,
            "duplicate_grain_rows": 0,
            "missing_metric_values": 0,
            "missing_support_values": 0,
            "metrics": [],
            "periods": [],
            "geography_levels": [],
            "geography_units": 0,
            "is_valid": False,
        }

    duplicate_grain_rows = int(
        relationship_data.duplicated(
            RELATIONSHIP_GRAIN_COLUMNS
        ).sum()
    )

    missing_metric_values = int(
        relationship_data["metric_value"]
        .isna()
        .sum()
    )

    missing_support_values = int(
        relationship_data["support_value"]
        .isna()
        .sum()
    )

    return {
        "row_count": int(len(relationship_data)),
        "duplicate_grain_rows": duplicate_grain_rows,
        "missing_metric_values": missing_metric_values,
        "missing_support_values": missing_support_values,
        "metrics": sorted(
            relationship_data["metric"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "periods": sorted(
            relationship_data["period"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "geography_levels": sorted(
            relationship_data["geography_level"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "geography_units": int(
            relationship_data["geography_id"]
            .nunique()
        ),
        "is_valid": (
            duplicate_grain_rows == 0
            and missing_metric_values == 0
        ),
    }