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
    "Geo-policy group",
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

        grouped_rows: list[dict[str, object]] = []

        for group_values, group_df in source.groupby(
            group_columns,
            observed=True,
            dropna=False,
            sort=False,
        ):
            aggregation_result = _aggregate_metric_group(
                group_df
            )

            row = dict(
                zip(
                    group_columns,
                    group_values,
                )
            )

            row.update(
                aggregation_result.to_dict()
            )

            grouped_rows.append(row)

        result = pd.DataFrame(grouped_rows)

        result["geography_level"] = "Borough"
        result["geography_id"] = (
            result["borough"].astype(str)
        )
        result["geography_name"] = result["borough"]
        result["taxi_zone_id"] = pd.NA
        result["zone"] = pd.NA
        result["cbd_spatial_category"] = "Mixed"

    elif aggregation_level == "Geo-policy group":
        group_columns = [
            "cbd_spatial_category",
            "temporal_bucket",
            "period",
            "metric",
            "metric_label",
            "aggregation",
            "support_metric",
        ]

        grouped_rows = []

        for group_values, group_df in source.groupby(
            group_columns,
            observed=True,
            dropna=False,
            sort=False,
        ):
            aggregation_result = _aggregate_metric_group(
                group_df
            )

            row = dict(
                zip(
                    group_columns,
                    group_values,
                )
            )

            row.update(
                aggregation_result.to_dict()
            )

            grouped_rows.append(row)

        result = pd.DataFrame(grouped_rows)

        result["geography_level"] = "Geo-policy group"
        result["geography_id"] = (
            result["cbd_spatial_category"]
            .astype(str)
        )
        result["geography_name"] = (
            result["cbd_spatial_category"]
            .astype(str)
        )
        result["taxi_zone_id"] = pd.NA
        result["zone"] = pd.NA
        result["borough"] = "Multiple boroughs"

    else:
        group_columns = [
            "temporal_bucket",
            "period",
            "metric",
            "metric_label",
            "aggregation",
            "support_metric",
        ]

        grouped_rows = []

        for group_values, group_df in source.groupby(
            group_columns,
            observed=True,
            dropna=False,
            sort=False,
        ):
            aggregation_result = _aggregate_metric_group(
                group_df
            )

            row = dict(
                zip(
                    group_columns,
                    group_values,
                )
            )

            row.update(
                aggregation_result.to_dict()
            )

            grouped_rows.append(row)

        result = pd.DataFrame(grouped_rows)

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
    Return one real row per geography with pre/post values for both metrics.

    Uses an explicit Pre-CP/Post-CP merge rather than pivot_table so no
    synthetic geography combinations or duplicate rows are created.
    """
    if pair_data.empty:
        return pd.DataFrame()

    key_columns = [
        "geography_level",
        "geography_id",
        "geography_name",
        "temporal_bucket",
        "x_metric_label",
        "y_metric_label",
    ]

    descriptive_columns = [
        "taxi_zone_id",
        "zone",
        "borough",
        "cbd_spatial_category",
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

    metadata = (
        pair_data[
            key_columns + descriptive_columns
        ]
        .drop_duplicates(subset=key_columns)
        .copy()
    )

    pre = (
        pair_data[
            pair_data["period"].eq("Pre-CP")
        ][key_columns + value_columns]
        .copy()
        .rename(
            columns={
                column: f"{column}_pre_cp"
                for column in value_columns
            }
        )
    )

    post = (
        pair_data[
            pair_data["period"].eq("Post-CP")
        ][key_columns + value_columns]
        .copy()
        .rename(
            columns={
                column: f"{column}_post_cp"
                for column in value_columns
            }
        )
    )

    wide = pre.merge(
        post,
        on=key_columns,
        how="inner",
        validate="one_to_one",
    )

    wide = wide.merge(
        metadata,
        on=key_columns,
        how="left",
        validate="one_to_one",
    )

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
        wide.sort_values("geography_name")
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

# ---------------------------------------------------------------------
# Rolling relationship strength through time
# ---------------------------------------------------------------------
NOTEBOOK_ROLLING_PAIR_DEFINITIONS = (
    ("taxi_trip_count", "subway_ridership"),
    ("taxi_trip_count", "fhvhv_trip_count"),
)

NOTEBOOK_ROLLING_WINDOW_DAYS = 90
NOTEBOOK_ROLLING_STEP_DAYS = 14
NOTEBOOK_ROLLING_MIN_MATCHED_DAYS = 45
NOTEBOOK_ROLLING_METHOD = "spearman"

SUPPORTED_ROLLING_METHODS = (
    "pearson",
    "spearman",
)


def calculate_rolling_relationship_strength(
    daily_wide_data: pd.DataFrame,
    *,
    pair_definitions: Iterable[tuple[str, str]],
    rolling_window_days: int = NOTEBOOK_ROLLING_WINDOW_DAYS,
    rolling_step_days: int = NOTEBOOK_ROLLING_STEP_DAYS,
    minimum_matched_days: int = NOTEBOOK_ROLLING_MIN_MATCHED_DAYS,
    minimum_coverage_share: float | None = None,
    correlation_method: str = NOTEBOOK_ROLLING_METHOD,
    metric_labels: dict[str, str] | None = None,
    geography_level: str = "Citywide",
    geography_id: str = "NYC",
    geography_name: str = "New York City",
    temporal_bucket: str = ALL_TEMPORAL_BUCKETS_LABEL,
) -> pd.DataFrame:
    """
    Calculate rolling relationships across dates for one or more metric pairs.

    This reproduces the Notebook 1.4.2 implementation used in the cell
    "Track relationship strength through time for two high-value pairings":

    - trailing calendar windows;
    - 90-day windows by default;
    - a new window every 14 days;
    - Spearman correlation by default;
    - at least 45 matched days;
    - one output row per pair × rolling window;
    - the plotted date is the window midpoint.

    The input must already be aggregated to the intended geography and contain
    one row per date. Unsupported windows remain in the output with a missing
    correlation so charts can display honest gaps.
    """
    if rolling_window_days < 2:
        raise ValueError(
            "rolling_window_days must be at least 2."
        )

    if rolling_step_days < 1:
        raise ValueError(
            "rolling_step_days must be at least 1."
        )

    if minimum_matched_days < 3:
        raise ValueError(
            "minimum_matched_days must be at least 3."
        )

    if minimum_matched_days > rolling_window_days:
        raise ValueError(
            "minimum_matched_days cannot exceed rolling_window_days."
        )

    if minimum_coverage_share is not None:
        if not 0 < minimum_coverage_share <= 1:
            raise ValueError(
                "minimum_coverage_share must be greater than 0 and "
                "less than or equal to 1."
            )

    normalized_method = correlation_method.strip().lower()

    if normalized_method not in SUPPORTED_ROLLING_METHODS:
        raise ValueError(
            "correlation_method must be one of "
            f"{SUPPORTED_ROLLING_METHODS}; "
            f"received {correlation_method!r}."
        )

    if daily_wide_data.empty:
        return pd.DataFrame()

    if "date" not in daily_wide_data.columns:
        raise KeyError(
            "daily_wide_data must contain a 'date' column."
        )

    pairs = list(pair_definitions)

    if not pairs:
        return pd.DataFrame()

    for metric_x, metric_y in pairs:
        if metric_x == metric_y:
            raise ValueError(
                "Each rolling relationship pair must contain two "
                "different metrics."
            )

        missing_metrics = [
            metric
            for metric in (metric_x, metric_y)
            if metric not in daily_wide_data.columns
        ]

        if missing_metrics:
            raise KeyError(
                "daily_wide_data is missing required metric columns: "
                + ", ".join(missing_metrics)
            )

    labels = metric_labels or {}

    source = daily_wide_data.copy()
    source["date"] = pd.to_datetime(
        source["date"],
        errors="coerce",
    )

    source = (
        source[
            source["date"].notna()
        ]
        .sort_values("date")
        .drop_duplicates(
            subset="date",
            keep="last",
        )
        .reset_index(drop=True)
    )

    records: list[dict[str, object]] = []

    for metric_x, metric_y in pairs:
        pair_label = (
            f"{labels.get(metric_x, metric_x)} vs "
            f"{labels.get(metric_y, metric_y)}"
        )

        pair_df = (
            source[
                [
                    "date",
                    metric_x,
                    metric_y,
                ]
            ]
            .sort_values("date")
            .reset_index(drop=True)
        )

        if pair_df.empty:
            continue

        window_start = pair_df["date"].min()
        window_end_limit = pair_df["date"].max()

        while (
            window_start
            + pd.Timedelta(
                days=rolling_window_days - 1
            )
            <= window_end_limit
        ):
            window_end = (
                window_start
                + pd.Timedelta(
                    days=rolling_window_days - 1
                )
            )

            eligible_window_df = pair_df.loc[
                pair_df["date"].between(
                    window_start,
                    window_end,
                )
            ].copy()

            eligible_date_count = int(
                eligible_window_df["date"].nunique()
            )

            window_df = eligible_window_df[
                [
                    metric_x,
                    metric_y,
                ]
            ].dropna()

            matched_days = int(len(window_df))
            x_unique = int(
                window_df[metric_x].nunique()
            )
            y_unique = int(
                window_df[metric_y].nunique()
            )

            if minimum_coverage_share is None:
                required_matched_days = minimum_matched_days
                support_mode = "fixed_days"
            else:
                required_matched_days = max(
                    3,
                    int(
                        np.ceil(
                            eligible_date_count
                            * minimum_coverage_share
                        )
                    ),
                )
                support_mode = "eligible_date_share"

            eligible_window = (
                matched_days >= required_matched_days
                and x_unique > 1
                and y_unique > 1
            )

            correlation = (
                window_df[metric_x].corr(
                    window_df[metric_y],
                    method=normalized_method,
                )
                if eligible_window
                else np.nan
            )

            midpoint = (
                window_start
                + pd.Timedelta(
                    days=rolling_window_days // 2
                )
            )

            records.append(
                {
                    "pair_label": pair_label,
                    "metric_x": metric_x,
                    "metric_y": metric_y,
                    "metric_x_label": labels.get(
                        metric_x,
                        metric_x,
                    ),
                    "metric_y_label": labels.get(
                        metric_y,
                        metric_y,
                    ),
                    "window_start": window_start,
                    "window_end": window_end,
                    "window_midpoint": midpoint,
                    "rolling_correlation": (
                        float(correlation)
                        if pd.notna(correlation)
                        else np.nan
                    ),
                    "rolling_spearman_correlation": (
                        float(correlation)
                        if (
                            normalized_method == "spearman"
                            and pd.notna(correlation)
                        )
                        else np.nan
                    ),
                    "rolling_pearson_correlation": (
                        float(correlation)
                        if (
                            normalized_method == "pearson"
                            and pd.notna(correlation)
                        )
                        else np.nan
                    ),
                    "matched_days": matched_days,
                    "eligible_date_count": eligible_date_count,
                    "required_matched_days": required_matched_days,
                    "coverage_share": (
                        matched_days / eligible_date_count
                        if eligible_date_count
                        else np.nan
                    ),
                    "support_mode": support_mode,
                    "minimum_coverage_share": minimum_coverage_share,
                    "x_unique_values": x_unique,
                    "y_unique_values": y_unique,
                    "eligible_window": eligible_window,
                    "rolling_window_days": rolling_window_days,
                    "rolling_step_days": rolling_step_days,
                    "minimum_matched_days": minimum_matched_days,
                    "correlation_method": normalized_method,
                    "geography_level": geography_level,
                    "geography_id": geography_id,
                    "geography_name": geography_name,
                    "temporal_bucket": temporal_bucket,
                }
            )

            window_start = (
                window_start
                + pd.Timedelta(
                    days=rolling_step_days
                )
            )

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)

    return (
        result.sort_values(
            [
                "pair_label",
                "window_midpoint",
            ]
        )
        .reset_index(drop=True)
    )


def build_citywide_rolling_relationship_data(
    *,
    pair_definitions: Iterable[
        tuple[str, str]
    ] = NOTEBOOK_ROLLING_PAIR_DEFINITIONS,
    temporal_bucket: str = ALL_TEMPORAL_BUCKETS_LABEL,
    rolling_window_days: int = NOTEBOOK_ROLLING_WINDOW_DAYS,
    rolling_step_days: int = NOTEBOOK_ROLLING_STEP_DAYS,
    minimum_matched_days: int = NOTEBOOK_ROLLING_MIN_MATCHED_DAYS,
    minimum_coverage_share: float | None = None,
    correlation_method: str = NOTEBOOK_ROLLING_METHOD,
) -> pd.DataFrame:
    """
    Build app-ready citywide rolling relationship data.

    Daily citywide metrics are obtained through the app's canonical aggregation
    layer, which preserves the established semantics:

    - demand metrics are summed;
    - Taxi speed is weighted by Taxi Trips;
    - FHVHV speed is weighted by FHVHV Trips;
    - Bus speed is weighted by Bus Trip Count.

    The local import prevents a module-import cycle while keeping the rolling
    relationship calculation in the relationship data-access module.
    """
    pairs = list(pair_definitions)

    selected_metrics = list(
        dict.fromkeys(
            metric
            for pair in pairs
            for metric in pair
        )
    )

    if not selected_metrics:
        return pd.DataFrame()

    from app.data_access.aggregations import (
        get_daily_metric_trends,
    )
    from app.data_access.loaders import (
        METRIC_LABELS,
    )

    daily_data = get_daily_metric_trends(
        metrics=selected_metrics,
        temporal_bucket=temporal_bucket,
    )

    return calculate_rolling_relationship_strength(
        daily_data,
        pair_definitions=pairs,
        rolling_window_days=rolling_window_days,
        rolling_step_days=rolling_step_days,
        minimum_matched_days=minimum_matched_days,
        minimum_coverage_share=minimum_coverage_share,
        correlation_method=correlation_method,
        metric_labels=METRIC_LABELS,
        geography_level="Citywide",
        geography_id="NYC",
        geography_name="New York City",
        temporal_bucket=temporal_bucket,
    )



def build_bucket_rolling_relationship_data(
    *,
    pair_definitions: Iterable[tuple[str, str]],
    temporal_bucket: str,
    rolling_window_days: int = NOTEBOOK_ROLLING_WINDOW_DAYS,
    rolling_step_days: int = NOTEBOOK_ROLLING_STEP_DAYS,
    minimum_coverage_share: float = 0.50,
    correlation_method: str = NOTEBOOK_ROLLING_METHOD,
) -> pd.DataFrame:
    """
    Build rolling relationships for an ordered temporal bucket.

    Unlike the overall notebook view, bucket-specific windows use a
    proportional support rule. A rolling point is eligible when both metrics
    are present for at least `minimum_coverage_share` of the dates represented
    by that bucket inside the calendar window.

    This keeps weekday and weekend views comparable without imposing an
    impossible fixed 45-day requirement on weekend-only series.
    """
    return build_citywide_rolling_relationship_data(
        pair_definitions=pair_definitions,
        temporal_bucket=temporal_bucket,
        rolling_window_days=rolling_window_days,
        rolling_step_days=rolling_step_days,
        minimum_matched_days=3,
        minimum_coverage_share=minimum_coverage_share,
        correlation_method=correlation_method,
    )

def build_notebook_rolling_relationship_data() -> pd.DataFrame:
    """
    Reproduce the fixed Notebook 1.4.2 rolling relationship dataset.

    Pairings:
    - Taxi Trips vs Subway Ridership
    - Taxi Trips vs FHVHV Trips

    Definition:
    - citywide daily aggregates;
    - all temporal buckets;
    - 90-day calendar windows;
    - 14-day steps;
    - Spearman correlation;
    - minimum 45 matched days.
    """
    return build_citywide_rolling_relationship_data(
        pair_definitions=NOTEBOOK_ROLLING_PAIR_DEFINITIONS,
        temporal_bucket=ALL_TEMPORAL_BUCKETS_LABEL,
        rolling_window_days=NOTEBOOK_ROLLING_WINDOW_DAYS,
        rolling_step_days=NOTEBOOK_ROLLING_STEP_DAYS,
        minimum_matched_days=NOTEBOOK_ROLLING_MIN_MATCHED_DAYS,
        correlation_method=NOTEBOOK_ROLLING_METHOD,
    )


def summarize_rolling_relationship_validation(
    rolling_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return compact QA statistics for each rolling relationship pair.

    This table supports numerical comparison with Notebook 1.4.2 before
    Page 8 is built.
    """
    if rolling_data.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []

    for pair_label, pair_df in rolling_data.groupby(
        "pair_label",
        observed=True,
        sort=False,
    ):
        ordered = pair_df.sort_values(
            "window_midpoint"
        )

        valid = ordered[
            ordered["rolling_correlation"].notna()
        ].copy()

        if valid.empty:
            first_valid_date = pd.NaT
            last_valid_date = pd.NaT
            latest_correlation = np.nan
            minimum_correlation = np.nan
            maximum_correlation = np.nan
            median_correlation = np.nan
            valid_window_count = 0
            sign_change_count = 0
        else:
            first_valid_date = valid[
                "window_midpoint"
            ].min()
            last_valid_date = valid[
                "window_midpoint"
            ].max()
            latest_correlation = float(
                valid.iloc[-1][
                    "rolling_correlation"
                ]
            )
            minimum_correlation = float(
                valid[
                    "rolling_correlation"
                ].min()
            )
            maximum_correlation = float(
                valid[
                    "rolling_correlation"
                ].max()
            )
            median_correlation = float(
                valid[
                    "rolling_correlation"
                ].median()
            )
            valid_window_count = int(
                len(valid)
            )

            signs = np.sign(
                valid[
                    "rolling_correlation"
                ]
            )

            sign_change_count = int(
                signs.ne(
                    signs.shift()
                )
                .iloc[1:]
                .sum()
            )

        rows.append(
            {
                "pair_label": pair_label,
                "first_valid_midpoint": first_valid_date,
                "last_valid_midpoint": last_valid_date,
                "total_window_count": int(
                    len(ordered)
                ),
                "valid_window_count": valid_window_count,
                "unsupported_window_count": int(
                    ordered[
                        "rolling_correlation"
                    ]
                    .isna()
                    .sum()
                ),
                "minimum_correlation": minimum_correlation,
                "maximum_correlation": maximum_correlation,
                "median_correlation": median_correlation,
                "latest_correlation": latest_correlation,
                "sign_change_count": sign_change_count,
                "minimum_matched_days_observed": int(
                    ordered[
                        "matched_days"
                    ].min()
                ),
                "maximum_matched_days_observed": int(
                    ordered[
                        "matched_days"
                    ].max()
                ),
                "rolling_window_days": int(
                    ordered.iloc[0][
                        "rolling_window_days"
                    ]
                ),
                "rolling_step_days": int(
                    ordered.iloc[0][
                        "rolling_step_days"
                    ]
                ),
                "minimum_matched_days_required": int(
                    ordered.iloc[0][
                        "minimum_matched_days"
                    ]
                ),
                "correlation_method": str(
                    ordered.iloc[0][
                        "correlation_method"
                    ]
                ),
            }
        )

    return pd.DataFrame(rows)


def validate_rolling_relationship_data(
    rolling_data: pd.DataFrame,
) -> dict[str, object]:
    """
    Return structural QA checks for an app-ready rolling relationship table.
    """
    required_columns = {
        "pair_label",
        "metric_x",
        "metric_y",
        "window_start",
        "window_end",
        "window_midpoint",
        "rolling_correlation",
        "matched_days",
        "eligible_date_count",
        "required_matched_days",
        "coverage_share",
        "support_mode",
        "eligible_window",
        "rolling_window_days",
        "rolling_step_days",
        "minimum_matched_days",
        "correlation_method",
        "geography_level",
        "geography_id",
        "temporal_bucket",
    }

    missing_columns = sorted(
        required_columns.difference(
            rolling_data.columns
        )
    )

    if rolling_data.empty:
        return {
            "row_count": 0,
            "pair_count": 0,
            "valid_window_count": 0,
            "unsupported_window_count": 0,
            "duplicate_window_rows": 0,
            "missing_columns": missing_columns,
            "invalid_supported_rows": 0,
            "is_valid": False,
        }

    duplicate_window_rows = int(
        rolling_data.duplicated(
            [
                "pair_label",
                "geography_level",
                "geography_id",
                "temporal_bucket",
                "window_start",
                "window_end",
            ]
        ).sum()
    )

    invalid_supported_rows = int(
        (
            rolling_data["eligible_window"]
            & rolling_data[
                "rolling_correlation"
            ].isna()
        ).sum()
    )

    correlation_out_of_range = int(
        (
            rolling_data[
                "rolling_correlation"
            ]
            .dropna()
            .abs()
            .gt(1)
        ).sum()
    )

    return {
        "row_count": int(
            len(rolling_data)
        ),
        "pair_count": int(
            rolling_data[
                "pair_label"
            ].nunique()
        ),
        "valid_window_count": int(
            rolling_data[
                "rolling_correlation"
            ]
            .notna()
            .sum()
        ),
        "unsupported_window_count": int(
            rolling_data[
                "rolling_correlation"
            ]
            .isna()
            .sum()
        ),
        "duplicate_window_rows": duplicate_window_rows,
        "missing_columns": missing_columns,
        "invalid_supported_rows": invalid_supported_rows,
        "correlation_out_of_range": correlation_out_of_range,
        "is_valid": (
            not missing_columns
            and duplicate_window_rows == 0
            and invalid_supported_rows == 0
            and correlation_out_of_range == 0
        ),
    }
