from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_TABLE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "1.3.1.final_tables"
)

MOBILITY_PANEL_PATH = (
    FINAL_TABLE_DIR
    / "analysis_ready_mobility_panel.parquet"
)

WEATHER_PANEL_PATH = (
    FINAL_TABLE_DIR
    / "weather_taxi_zone_context.parquet"
)

ALL_TEMPORAL_BUCKETS_LABEL = "All temporal buckets"

MOBILITY_METRICS = (
    "taxi_trip_count",
    "taxi_avg_trip_speed",
    "fhvhv_trip_count",
    "fhvhv_avg_trip_speed",
    "subway_ridership",
    "avg_bus_speed",
)

WEATHER_METRICS = (
    "temperature",
    "precipitation",
    "visibility",
    "wind_speed",
    "wind_gust",
    "relative_humidity",
    "ceiling_height",
    "pressure_3hr_change",
)

MOBILITY_LABELS = {
    "taxi_trip_count": "Taxi Trips",
    "taxi_avg_trip_speed": "Taxi Average Speed",
    "fhvhv_trip_count": "FHVHV Trips",
    "fhvhv_avg_trip_speed": "FHVHV Average Speed",
    "subway_ridership": "Subway Ridership",
    "avg_bus_speed": "Bus Average Speed",
}

WEATHER_LABELS = {
    "temperature": "Temperature",
    "precipitation": "Precipitation",
    "visibility": "Visibility",
    "wind_speed": "Wind Speed",
    "wind_gust": "Wind Gust",
    "relative_humidity": "Relative Humidity",
    "ceiling_height": "Ceiling Height",
    "pressure_3hr_change": "3-Hour Pressure Change",
}

TEMPORAL_BUCKETS = (
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
)

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

COUNT_METRICS = {
    "taxi_trip_count",
    "fhvhv_trip_count",
    "subway_ridership",
}

SPEED_WEIGHT_COLUMNS = {
    "taxi_avg_trip_speed": "taxi_trip_count",
    "fhvhv_avg_trip_speed": "fhvhv_trip_count",
    "avg_bus_speed": "bus_trip_count",
}


@lru_cache(maxsize=1)
def load_zone_lookup() -> pd.DataFrame:
    """
    Return one row per available Taxi Zone for production controls.
    """
    _validate_files()

    candidate_columns = [
        "taxi_zone_id",
        "zone",
        "borough",
    ]

    data = pd.read_parquet(
        MOBILITY_PANEL_PATH,
        columns=candidate_columns,
    )

    return (
        data.dropna(subset=["taxi_zone_id", "zone"])
        .drop_duplicates(
            subset=["taxi_zone_id"]
        )
        .sort_values(
            ["borough", "zone", "taxi_zone_id"]
        )
        .reset_index(drop=True)
    )


@lru_cache(maxsize=1)
def load_relationship_date_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Return the overlapping date range available in both source tables.
    """
    _validate_files()

    mobility_dates = pd.read_parquet(
        MOBILITY_PANEL_PATH,
        columns=["date"],
    )
    weather_dates = pd.read_parquet(
        WEATHER_PANEL_PATH,
        columns=["date"],
    )

    mobility_dates["date"] = pd.to_datetime(
        mobility_dates["date"]
    )
    weather_dates["date"] = pd.to_datetime(
        weather_dates["date"]
    )

    minimum_date = max(
        mobility_dates["date"].min(),
        weather_dates["date"].min(),
    )
    maximum_date = min(
        mobility_dates["date"].max(),
        weather_dates["date"].max(),
    )

    return minimum_date, maximum_date


def _validate_files() -> None:
    missing = [
        str(path)
        for path in (
            MOBILITY_PANEL_PATH,
            WEATHER_PANEL_PATH,
        )
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required Page 9 source files were not found: "
            + ", ".join(missing)
        )


def _validate_metric(
    metric: str,
    allowed_metrics: Iterable[str],
    metric_type: str,
) -> None:
    if metric not in allowed_metrics:
        raise ValueError(
            f"Unsupported {metric_type} metric: {metric}"
        )


def _filter_source(
    data: pd.DataFrame,
    *,
    temporal_bucket: str,
    taxi_zone_id: int | None,
    start_date: str | pd.Timestamp | None,
    end_date: str | pd.Timestamp | None,
) -> pd.DataFrame:
    result = data.copy()
    result["date"] = pd.to_datetime(result["date"])

    if temporal_bucket != ALL_TEMPORAL_BUCKETS_LABEL:
        result = result[
            result["temporal_bucket"].eq(
                temporal_bucket
            )
        ]

    if taxi_zone_id is not None:
        result = result[
            result["taxi_zone_id"].eq(
                int(taxi_zone_id)
            )
        ]

    if start_date is not None:
        result = result[
            result["date"].ge(
                pd.Timestamp(start_date)
            )
        ]

    if end_date is not None:
        result = result[
            result["date"].le(
                pd.Timestamp(end_date)
            )
        ]

    return result


def _safe_weighted_average(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    valid = (
        values.notna()
        & weights.notna()
        & weights.gt(0)
    )

    if not valid.any():
        return np.nan

    return float(
        np.average(
            values.loc[valid],
            weights=weights.loc[valid],
        )
    )


def _aggregate_mobility_daily(
    mobility: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for date, group in mobility.groupby(
        "date",
        observed=True,
        sort=True,
    ):
        row: dict[str, object] = {
            "date": date,
            "pre_post_cp": (
                "pre_cp"
                if date < pd.Timestamp("2025-01-05")
                else "post_cp"
            ),
        }

        for metric in COUNT_METRICS:
            row[metric] = float(
                group[metric]
                .fillna(0)
                .sum()
            )

        for metric, weight_column in SPEED_WEIGHT_COLUMNS.items():
            row[metric] = _safe_weighted_average(
                group[metric],
                group[weight_column],
            )

        rows.append(row)

    return pd.DataFrame(rows)


def _aggregate_weather_daily(
    weather: pd.DataFrame,
) -> pd.DataFrame:
    return (
        weather.groupby(
            "date",
            observed=True,
            as_index=False,
        )[list(WEATHER_METRICS)]
        .mean()
        .sort_values("date")
        .reset_index(drop=True)
    )


def build_daily_weather_mobility_panel(
    *,
    temporal_bucket: str = ALL_TEMPORAL_BUCKETS_LABEL,
    taxi_zone_id: int | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Build one daily analytical panel containing all supported mobility and
    weather measures.

    When taxi_zone_id is None, mobility is aggregated citywide and weather is
    averaged across Taxi Zones. When taxi_zone_id is supplied, both sources are
    restricted to that zone.
    """
    _validate_files()

    if temporal_bucket not in TEMPORAL_BUCKETS:
        raise ValueError(
            f"Unsupported temporal bucket: {temporal_bucket}"
        )

    mobility_columns = {
        "date",
        "taxi_zone_id",
        "temporal_bucket",
        *MOBILITY_METRICS,
        *SPEED_WEIGHT_COLUMNS.values(),
    }

    weather_columns = {
        "date",
        "taxi_zone_id",
        "temporal_bucket",
        *WEATHER_METRICS,
    }

    mobility_filters: list[tuple[str, str, object]] = []
    weather_filters: list[tuple[str, str, object]] = []

    if temporal_bucket != ALL_TEMPORAL_BUCKETS_LABEL:
        mobility_filters.append(
            ("temporal_bucket", "==", temporal_bucket)
        )
        weather_filters.append(
            ("temporal_bucket", "==", temporal_bucket)
        )

    if taxi_zone_id is not None:
        mobility_filters.append(
            ("taxi_zone_id", "==", int(taxi_zone_id))
        )
        weather_filters.append(
            ("taxi_zone_id", "==", int(taxi_zone_id))
        )

    if start_date is not None:
        start_value = pd.Timestamp(start_date)
        mobility_filters.append(
            ("date", ">=", start_value)
        )
        weather_filters.append(
            ("date", ">=", start_value)
        )

    if end_date is not None:
        end_value = pd.Timestamp(end_date)
        mobility_filters.append(
            ("date", "<=", end_value)
        )
        weather_filters.append(
            ("date", "<=", end_value)
        )

    mobility = pd.read_parquet(
        MOBILITY_PANEL_PATH,
        columns=sorted(mobility_columns),
        filters=mobility_filters or None,
    )

    weather = pd.read_parquet(
        WEATHER_PANEL_PATH,
        columns=sorted(weather_columns),
        filters=weather_filters or None,
    )

    mobility = _filter_source(
        mobility,
        temporal_bucket=temporal_bucket,
        taxi_zone_id=taxi_zone_id,
        start_date=start_date,
        end_date=end_date,
    )

    weather = _filter_source(
        weather,
        temporal_bucket=temporal_bucket,
        taxi_zone_id=taxi_zone_id,
        start_date=start_date,
        end_date=end_date,
    )

    mobility_daily = _aggregate_mobility_daily(
        mobility
    )
    weather_daily = _aggregate_weather_daily(
        weather
    )

    result = mobility_daily.merge(
        weather_daily,
        on="date",
        how="inner",
        validate="one_to_one",
    )

    result["geography_level"] = (
        "Citywide"
        if taxi_zone_id is None
        else "Taxi Zone"
    )

    result["taxi_zone_id"] = (
        pd.NA
        if taxi_zone_id is None
        else int(taxi_zone_id)
    )

    result["temporal_bucket"] = (
        "Overall"
        if temporal_bucket
        == ALL_TEMPORAL_BUCKETS_LABEL
        else temporal_bucket
    )

    ordered_columns = [
        "date",
        "pre_post_cp",
        "geography_level",
        "taxi_zone_id",
        "temporal_bucket",
        *MOBILITY_METRICS,
        *WEATHER_METRICS,
    ]

    return (
        result[ordered_columns]
        .sort_values("date")
        .reset_index(drop=True)
    )


def build_weather_relationship_pair_data(
    *,
    mobility_metric: str,
    weather_metric: str,
    temporal_bucket: str = ALL_TEMPORAL_BUCKETS_LABEL,
    taxi_zone_id: int | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    _validate_metric(
        mobility_metric,
        MOBILITY_METRICS,
        "mobility",
    )

    _validate_metric(
        weather_metric,
        WEATHER_METRICS,
        "weather",
    )

    panel = build_daily_weather_mobility_panel(
        temporal_bucket=temporal_bucket,
        taxi_zone_id=taxi_zone_id,
        start_date=start_date,
        end_date=end_date,
    )

    if panel.empty:
        return panel

    result = panel[
        [
            "date",
            "pre_post_cp",
            "geography_level",
            "taxi_zone_id",
            "temporal_bucket",
            mobility_metric,
            weather_metric,
        ]
    ].copy()

    result = result.rename(
        columns={
            mobility_metric: "mobility_value",
            weather_metric: "weather_value",
        }
    )

    result["mobility_metric"] = mobility_metric
    result["weather_metric"] = weather_metric
    result["mobility_label"] = MOBILITY_LABELS[mobility_metric]
    result["weather_label"] = WEATHER_LABELS[weather_metric]
    result["pair_label"] = (
        result["weather_label"]
        + " vs "
        + result["mobility_label"]
    )

    return result


def calculate_relationship_statistics(
    pair_data: pd.DataFrame,
    *,
    minimum_observations: int = 30,
) -> pd.DataFrame:
    required_columns = {
        "date",
        "pre_post_cp",
        "mobility_value",
        "weather_value",
        "mobility_metric",
        "weather_metric",
        "mobility_label",
        "weather_label",
        "pair_label",
    }

    missing = sorted(
        required_columns.difference(
            pair_data.columns
        )
    )

    if missing:
        raise ValueError(
            "Relationship data is missing columns: "
            + ", ".join(missing)
        )

    periods = [
        ("all", pair_data),
        (
            "pre_cp",
            pair_data[
                pair_data["pre_post_cp"].eq("pre_cp")
            ],
        ),
        (
            "post_cp",
            pair_data[
                pair_data["pre_post_cp"].eq("post_cp")
            ],
        ),
    ]

    rows: list[dict[str, object]] = []

    for period, period_data in periods:
        valid = period_data[
            [
                "date",
                "mobility_value",
                "weather_value",
            ]
        ].dropna()

        observation_count = int(len(valid))
        unique_mobility = int(
            valid["mobility_value"].nunique()
        )
        unique_weather = int(
            valid["weather_value"].nunique()
        )

        supported = (
            observation_count >= minimum_observations
            and unique_mobility > 1
            and unique_weather > 1
        )

        if supported:
            pearson = float(
                valid["weather_value"].corr(
                    valid["mobility_value"],
                    method="pearson",
                )
            )

            spearman = float(
                valid["weather_value"].corr(
                    valid["mobility_value"],
                    method="spearman",
                )
            )

            slope = float(
                np.polyfit(
                    valid["weather_value"],
                    valid["mobility_value"],
                    deg=1,
                )[0]
            )
        else:
            pearson = np.nan
            spearman = np.nan
            slope = np.nan

        rows.append(
            {
                "period": period,
                "observation_count": observation_count,
                "supported": supported,
                "pearson_correlation": pearson,
                "spearman_correlation": spearman,
                "linear_slope": slope,
                "minimum_date": (
                    valid["date"].min()
                    if not valid.empty
                    else pd.NaT
                ),
                "maximum_date": (
                    valid["date"].max()
                    if not valid.empty
                    else pd.NaT
                ),
            }
        )

    result = pd.DataFrame(rows)

    metadata_columns = [
        "geography_level",
        "taxi_zone_id",
        "temporal_bucket",
        "mobility_metric",
        "weather_metric",
        "mobility_label",
        "weather_label",
        "pair_label",
    ]

    for column in metadata_columns:
        if column in pair_data.columns and not pair_data.empty:
            result[column] = pair_data.iloc[0][column]

    return result


def scan_weather_relationships(
    *,
    temporal_bucket: str = ALL_TEMPORAL_BUCKETS_LABEL,
    taxi_zone_id: int | None = None,
    minimum_observations: int = 30,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    panel = build_daily_weather_mobility_panel(
        temporal_bucket=temporal_bucket,
        taxi_zone_id=taxi_zone_id,
        start_date=start_date,
        end_date=end_date,
    )

    if panel.empty:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []

    for mobility_metric in MOBILITY_METRICS:
        for weather_metric in WEATHER_METRICS:
            pair_data = panel[
                [
                    "date",
                    "pre_post_cp",
                    "geography_level",
                    "taxi_zone_id",
                    "temporal_bucket",
                    mobility_metric,
                    weather_metric,
                ]
            ].copy()

            pair_data = pair_data.rename(
                columns={
                    mobility_metric: "mobility_value",
                    weather_metric: "weather_value",
                }
            )

            pair_data["mobility_metric"] = mobility_metric
            pair_data["weather_metric"] = weather_metric
            pair_data["mobility_label"] = MOBILITY_LABELS[mobility_metric]
            pair_data["weather_label"] = WEATHER_LABELS[weather_metric]
            pair_data["pair_label"] = (
                pair_data["weather_label"]
                + " vs "
                + pair_data["mobility_label"]
            )

            rows.append(
                calculate_relationship_statistics(
                    pair_data,
                    minimum_observations=minimum_observations,
                )
            )

    result = pd.concat(
        rows,
        ignore_index=True,
    )

    wide = result.pivot_table(
        index=[
            "geography_level",
            "taxi_zone_id",
            "temporal_bucket",
            "mobility_metric",
            "weather_metric",
            "mobility_label",
            "weather_label",
            "pair_label",
        ],
        columns="period",
        values=[
            "observation_count",
            "pearson_correlation",
            "spearman_correlation",
            "supported",
        ],
        aggfunc="first",
        dropna=False,
    )

    wide.columns = [
        f"{measure}_{period}"
        for measure, period in wide.columns
    ]

    wide = wide.reset_index().rename_axis(
        None,
        axis=1,
    )

    for period in ("all", "pre_cp", "post_cp"):
        column = f"spearman_correlation_{period}"
        wide[f"absolute_spearman_{period}"] = (
            wide[column].abs()
        )

    wide["spearman_post_minus_pre"] = (
        wide["spearman_correlation_post_cp"]
        - wide["spearman_correlation_pre_cp"]
    )

    wide["absolute_spearman_shift"] = (
        wide["spearman_post_minus_pre"].abs()
    )

    return wide.sort_values(
        [
            "absolute_spearman_all",
            "absolute_spearman_shift",
        ],
        ascending=[False, False],
    ).reset_index(drop=True)


def scan_all_temporal_buckets(
    *,
    taxi_zone_id: int | None = None,
    minimum_observations: int = 30,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for temporal_bucket in TEMPORAL_BUCKETS:
        frame = scan_weather_relationships(
            temporal_bucket=temporal_bucket,
            taxi_zone_id=taxi_zone_id,
            minimum_observations=minimum_observations,
        )

        if not frame.empty:
            frame["temporal_bucket_label"] = (
                TEMPORAL_BUCKET_LABELS[
                    temporal_bucket
                ]
            )
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


def validate_weather_relationship_panel(
    panel: pd.DataFrame,
) -> dict[str, object]:
    expected_columns = {
        "date",
        "pre_post_cp",
        *MOBILITY_METRICS,
        *WEATHER_METRICS,
    }

    missing_columns = sorted(
        expected_columns.difference(
            panel.columns
        )
    )

    duplicate_dates = (
        int(
            panel.duplicated(
                subset=["date"]
            ).sum()
        )
        if "date" in panel.columns
        else 0
    )

    return {
        "row_count": int(len(panel)),
        "minimum_date": (
            panel["date"].min()
            if not panel.empty
            and "date" in panel.columns
            else pd.NaT
        ),
        "maximum_date": (
            panel["date"].max()
            if not panel.empty
            and "date" in panel.columns
            else pd.NaT
        ),
        "duplicate_dates": duplicate_dates,
        "missing_columns": missing_columns,
        "is_valid": (
            not panel.empty
            and not missing_columns
            and duplicate_dates == 0
        ),
    }
