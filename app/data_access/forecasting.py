"""Shared data-access contracts for Showcase forecasting pages.

This module owns the frozen Chapter 4 forecasting handoff:
- canonical artifact locations,
- common forecasting constants,
- schema validation,
- lightweight dtype normalization,
- reusable final-holdout / reader-facing filters,
- and canonical temporal-bucket decoding.

Page-specific analysis and visualization stay in Raw 18–20.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from app.data_access.mobility_environments import (
    load_canonical_cluster_assignments,
)


# ---------------------------------------------------------------------
# Frozen forecasting contract
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FORECAST_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "4.7.1.final_tables"
)

FORECAST_RECORD_SURFACE_PATH = (
    FORECAST_DIR
    / "showcase_forecast_record_surface.parquet"
)

FORECAST_HISTORY_SURFACE_PATH = (
    FORECAST_DIR
    / "showcase_forecast_history_surface.parquet"
)

FORECAST_JOB_SUMMARY_PATH = (
    FORECAST_DIR
    / "showcase_forecast_job_summary.parquet"
)

FORECAST_ZONE_SUMMARY_PATH = (
    FORECAST_DIR
    / "showcase_forecast_zone_summary.parquet"
)


FORECAST_TEMPORAL_ZONE_SUMMARY_PATH = (
    FORECAST_DIR
    / "showcase_forecast_temporal_zone_summary.parquet"
)

FORECAST_FAILURE_CASES_PATH = (
    FORECAST_DIR
    / "forecast_failure_case_studies.parquet"
)

FORECAST_FAILURE_CONTEXT_PATH = (
    FORECAST_DIR
    / "forecast_failure_case_context.parquet"
)

FORECAST_FAILURE_MECHANISM_PATH = (
    FORECAST_DIR
    / "forecast_failure_mechanism_summary.parquet"
)


FINAL_HOLDOUT_START_DATE = pd.Timestamp("2026-01-05")
FINAL_HOLDOUT_END_DATE = pd.Timestamp("2026-03-31")

FORECAST_HORIZONS = (1, 2, 5)

FORECAST_METRICS = (
    "taxi_trip_count",
    "taxi_avg_trip_speed",
    "fhvhv_trip_count",
    "fhvhv_avg_trip_speed",
    "subway_ridership",
)

FORECAST_METRIC_LABELS = {
    "taxi_trip_count": "Taxi trips",
    "taxi_avg_trip_speed": "Taxi average speed",
    "fhvhv_trip_count": "FHVHV trips",
    "fhvhv_avg_trip_speed": "FHVHV average speed",
    "subway_ridership": "Subway ridership",
}

FORECAST_METRIC_UNITS = {
    "taxi_trip_count": "trips",
    "taxi_avg_trip_speed": "mph",
    "fhvhv_trip_count": "trips",
    "fhvhv_avg_trip_speed": "mph",
    "subway_ridership": "riders",
}

DAYPART_LABELS = {
    "overnight": "Overnight",
    "am_peak": "AM peak",
    "midday": "Midday",
    "pm_peak": "PM peak",
    "evening": "Evening",
}


POLICY_GEOGRAPHY_MAP = {
    "cbd": "CBD",
    "adjacent_to_cbd": "Gateway + adjacent",
    "gateway_to_cbd": "Gateway + adjacent",
    "non_cbd": "Outside",
}


# ---------------------------------------------------------------------
# Errors + validation
# ---------------------------------------------------------------------

class ForecastDataContractError(RuntimeError):
    """Raised when a frozen forecasting artifact no longer matches its contract."""


def require_forecast_file(
    path: Path,
    *,
    label: str,
) -> None:
    """Fail loudly when one authoritative forecasting artifact is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required forecasting artifact not found for {label}: {path}"
        )


def require_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    *,
    label: str,
) -> None:
    """Protect downstream pages from silent forecasting schema drift."""
    missing = sorted(
        set(required)
        - set(frame.columns)
    )

    if missing:
        raise ForecastDataContractError(
            f"{label} is missing required columns: {missing}"
        )


# ---------------------------------------------------------------------
# Shared normalization
# ---------------------------------------------------------------------

def _normalize_common_forecast_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize columns whose meaning is shared across forecasting artifacts."""
    result = frame.copy()

    if "target_date" in result.columns:
        result["target_date"] = pd.to_datetime(
            result["target_date"],
            errors="coerce",
        )

    if "horizon" in result.columns:
        result["horizon"] = pd.to_numeric(
            result["horizon"],
            errors="coerce",
        ).astype("Int64")

    if "taxi_zone_id" in result.columns:
        result["taxi_zone_id"] = pd.to_numeric(
            result["taxi_zone_id"],
            errors="coerce",
        ).astype("Int64")

    boolean_columns = [
        "reader_facing_zone",
        "severe_error",
        "champion_better_than_benchmark",
        "low_zero_demand",
        "unstable_conditions",
        "shared_shock",
    ]

    for column in boolean_columns:
        if column in result.columns:
            result[column] = (
                result[column]
                .fillna(False)
                .astype(bool)
            )

    return result


def _as_list(
    values: Iterable[object] | object | None,
) -> list[object] | None:
    """Normalize optional scalar-or-sequence filters."""
    if values is None:
        return None

    if isinstance(values, (str, bytes)):
        return [values]

    try:
        return list(values)
    except TypeError:
        return [values]


# ---------------------------------------------------------------------
# Canonical temporal decoding
# ---------------------------------------------------------------------

def forecast_day_type(
    bucket: pd.Series,
) -> pd.Series:
    """Translate canonical temporal buckets into Weekdays / Weekends."""
    values = (
        bucket
        .astype("string")
        .str.lower()
    )

    return pd.Series(
        np.where(
            values.str.startswith(
                "weekend",
                na=False,
            ),
            "Weekends",
            "Weekdays",
        ),
        index=bucket.index,
        dtype="string",
    )


def forecast_daypart(
    bucket: pd.Series,
) -> pd.Series:
    """Translate canonical temporal buckets into reader-facing dayparts."""
    values = (
        bucket
        .astype("string")
        .str.lower()
    )

    result = pd.Series(
        "Unknown",
        index=bucket.index,
        dtype="string",
    )

    for token, label in DAYPART_LABELS.items():
        result.loc[
            values.str.contains(
                token,
                regex=False,
                na=False,
            )
        ] = label

    return result


# ---------------------------------------------------------------------
# Generic parquet reader
# ---------------------------------------------------------------------

def _read_forecast_parquet(
    path: Path,
    *,
    label: str,
    columns: Sequence[str] | None = None,
    filters: list[tuple[str, str, object]] | None = None,
) -> pd.DataFrame:
    """Read a frozen forecasting artifact with predicate-pushdown fallback."""
    require_forecast_file(
        path,
        label=label,
    )

    try:
        return pd.read_parquet(
            path,
            columns=list(columns) if columns is not None else None,
            filters=filters,
        )
    except Exception:
        # WHY: preserve the same contract when a local parquet engine cannot
        # push a supported filter down to the file.
        return pd.read_parquet(
            path,
            columns=list(columns) if columns is not None else None,
        )


# ---------------------------------------------------------------------
# Forecast record surface
# ---------------------------------------------------------------------

def load_forecast_records(
    *,
    columns: Sequence[str] | None = None,
    required_columns: Sequence[str] | None = None,
    metrics: Iterable[str] | str | None = None,
    horizons: Iterable[int] | int | None = None,
    taxi_zone_ids: Iterable[int] | int | None = None,
    boroughs: Iterable[str] | str | None = None,
    zones: Iterable[str] | str | None = None,
    reader_facing_only: bool = False,
    final_holdout_only: bool = False,
) -> pd.DataFrame:
    """Load the authoritative row-level forecasting surface."""
    metric_values = _as_list(metrics)
    horizon_values = _as_list(horizons)
    zone_id_values = _as_list(taxi_zone_ids)
    borough_values = _as_list(boroughs)
    zone_values = _as_list(zones)

    parquet_filters: list[tuple[str, str, object]] = []

    if reader_facing_only:
        parquet_filters.append(
            ("reader_facing_zone", "==", True)
        )

    if metric_values and len(metric_values) == 1:
        parquet_filters.append(
            ("metric", "==", metric_values[0])
        )

    if horizon_values and len(horizon_values) == 1:
        parquet_filters.append(
            ("horizon", "==", int(horizon_values[0]))
        )

    if zone_id_values and len(zone_id_values) == 1:
        parquet_filters.append(
            ("taxi_zone_id", "==", int(zone_id_values[0]))
        )

    if borough_values and len(borough_values) == 1:
        parquet_filters.append(
            ("borough", "==", borough_values[0])
        )

    if zone_values and len(zone_values) == 1:
        parquet_filters.append(
            ("zone", "==", zone_values[0])
        )

    frame = _read_forecast_parquet(
        FORECAST_RECORD_SURFACE_PATH,
        label="forecast record surface",
        columns=columns,
        filters=parquet_filters or None,
    )

    frame = _normalize_common_forecast_columns(
        frame
    )

    if required_columns is not None:
        require_columns(
            frame,
            required_columns,
            label="showcase_forecast_record_surface",
        )

    if reader_facing_only:
        if "reader_facing_zone" not in frame.columns:
            raise ForecastDataContractError(
                "reader_facing_only=True requires reader_facing_zone "
                "in the requested columns."
            )

        frame = frame.loc[
            frame["reader_facing_zone"]
        ].copy()

    if metric_values is not None:
        frame = frame.loc[
            frame["metric"].isin(
                metric_values
            )
        ].copy()

    if horizon_values is not None:
        frame = frame.loc[
            frame["horizon"].isin(
                [int(value) for value in horizon_values]
            )
        ].copy()

    if zone_id_values is not None:
        frame = frame.loc[
            frame["taxi_zone_id"].isin(
                [int(value) for value in zone_id_values]
            )
        ].copy()

    if borough_values is not None:
        frame = frame.loc[
            frame["borough"].isin(
                borough_values
            )
        ].copy()

    if zone_values is not None:
        frame = frame.loc[
            frame["zone"].isin(
                zone_values
            )
        ].copy()

    if final_holdout_only:
        if "target_date" not in frame.columns:
            raise ForecastDataContractError(
                "final_holdout_only=True requires target_date "
                "in the requested columns."
            )

        frame = frame.loc[
            frame["target_date"].between(
                FINAL_HOLDOUT_START_DATE,
                FINAL_HOLDOUT_END_DATE,
                inclusive="both",
            )
        ].copy()

    return frame.reset_index(drop=True)


# ---------------------------------------------------------------------
# Forecast history surface
# ---------------------------------------------------------------------

def load_forecast_history(
    *,
    columns: Sequence[str] | None = None,
    required_columns: Sequence[str] | None = None,
    metrics: Iterable[str] | str | None = None,
    horizons: Iterable[int] | int | None = None,
    taxi_zone_ids: Iterable[int] | int | None = None,
    reader_facing_only: bool = False,
) -> pd.DataFrame:
    """
    Load the frozen longitudinal forecasting-history surface.

    WHY:
    Raw 18 needs narrow history slices for its interactive explorer and
    personalized Quilts. Predicate filters are pushed down when possible and
    then reapplied in Pandas so behavior is consistent across parquet engines.
    """
    metric_values = _as_list(metrics)
    horizon_values = _as_list(horizons)
    zone_id_values = _as_list(taxi_zone_ids)

    parquet_filters: list[tuple[str, str, object]] = []

    if reader_facing_only:
        parquet_filters.append(
            ("reader_facing_zone", "==", True)
        )

    if metric_values:
        parquet_filters.append(
            (
                "metric",
                "==" if len(metric_values) == 1 else "in",
                metric_values[0] if len(metric_values) == 1 else metric_values,
            )
        )

    if horizon_values:
        integer_horizons = [
            int(value)
            for value in horizon_values
        ]
        parquet_filters.append(
            (
                "horizon",
                "==" if len(integer_horizons) == 1 else "in",
                (
                    integer_horizons[0]
                    if len(integer_horizons) == 1
                    else integer_horizons
                ),
            )
        )

    if zone_id_values:
        integer_zone_ids = [
            int(value)
            for value in zone_id_values
        ]
        parquet_filters.append(
            (
                "taxi_zone_id",
                "==" if len(integer_zone_ids) == 1 else "in",
                (
                    integer_zone_ids[0]
                    if len(integer_zone_ids) == 1
                    else integer_zone_ids
                ),
            )
        )

    frame = _read_forecast_parquet(
        FORECAST_HISTORY_SURFACE_PATH,
        label="forecast history surface",
        columns=columns,
        filters=parquet_filters or None,
    )

    frame = _normalize_common_forecast_columns(
        frame
    )

    if required_columns is not None:
        require_columns(
            frame,
            required_columns,
            label="showcase_forecast_history_surface",
        )

    if reader_facing_only:
        if "reader_facing_zone" not in frame.columns:
            raise ForecastDataContractError(
                "reader_facing_only=True requires reader_facing_zone "
                "in the requested columns."
            )
        frame = frame.loc[
            frame["reader_facing_zone"]
        ].copy()

    if metric_values is not None:
        frame = frame.loc[
            frame["metric"].isin(
                metric_values
            )
        ].copy()

    if horizon_values is not None:
        frame = frame.loc[
            frame["horizon"].isin(
                [int(value) for value in horizon_values]
            )
        ].copy()

    if zone_id_values is not None:
        frame = frame.loc[
            frame["taxi_zone_id"].isin(
                [int(value) for value in zone_id_values]
            )
        ].copy()

    return frame.reset_index(drop=True)


# ---------------------------------------------------------------------
# Forecast summaries
# ---------------------------------------------------------------------

def load_forecast_job_summary(
    *,
    columns: Sequence[str] | None = None,
    required_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load the frozen Metric × Horizon summary."""
    frame = _read_forecast_parquet(
        FORECAST_JOB_SUMMARY_PATH,
        label="forecast job summary",
        columns=columns,
    )

    frame = _normalize_common_forecast_columns(
        frame
    )

    if required_columns is not None:
        require_columns(
            frame,
            required_columns,
            label="showcase_forecast_job_summary",
        )

    return frame.reset_index(drop=True)


def load_forecast_zone_summary(
    *,
    columns: Sequence[str] | None = None,
    required_columns: Sequence[str] | None = None,
    metrics: Iterable[str] | str | None = None,
    horizons: Iterable[int] | int | None = None,
    reader_facing_only: bool = False,
) -> pd.DataFrame:
    """Load the frozen Taxi-Zone forecasting summary."""
    metric_values = _as_list(metrics)
    horizon_values = _as_list(horizons)

    parquet_filters: list[tuple[str, str, object]] = []

    if reader_facing_only:
        parquet_filters.append(
            ("reader_facing_zone", "==", True)
        )

    if metric_values and len(metric_values) == 1:
        parquet_filters.append(
            ("metric", "==", metric_values[0])
        )

    if horizon_values and len(horizon_values) == 1:
        parquet_filters.append(
            ("horizon", "==", int(horizon_values[0]))
        )

    frame = _read_forecast_parquet(
        FORECAST_ZONE_SUMMARY_PATH,
        label="forecast zone summary",
        columns=columns,
        filters=parquet_filters or None,
    )

    frame = _normalize_common_forecast_columns(
        frame
    )

    if required_columns is not None:
        require_columns(
            frame,
            required_columns,
            label="showcase_forecast_zone_summary",
        )

    if reader_facing_only:
        if "reader_facing_zone" not in frame.columns:
            raise ForecastDataContractError(
                "reader_facing_only=True requires reader_facing_zone."
            )
        frame = frame.loc[
            frame["reader_facing_zone"]
        ].copy()

    if metric_values is not None:
        frame = frame.loc[
            frame["metric"].isin(
                metric_values
            )
        ].copy()

    if horizon_values is not None:
        frame = frame.loc[
            frame["horizon"].isin(
                [int(value) for value in horizon_values]
            )
        ].copy()

    return frame.reset_index(drop=True)


def load_forecast_temporal_zone_summary(
    *,
    columns: Sequence[str] | None = None,
    required_columns: Sequence[str] | None = None,
    metrics: Iterable[str] | str | None = None,
    horizons: Iterable[int] | int | None = None,
) -> pd.DataFrame:
    """
    Load Raw 19's pre-aggregated final-holdout temporal-zone evidence.

    The artifact stores additive sufficient statistics at:
    Metric × Horizon × Taxi Zone × Day type × Daypart.

    Raw 19 can therefore rebuild any supported day-type/daypart slice exactly
    without scanning and regrouping the full row-level forecast surface.
    """
    metric_values = _as_list(metrics)
    horizon_values = _as_list(horizons)

    parquet_filters: list[tuple[str, str, object]] = []

    if metric_values and len(metric_values) == 1:
        parquet_filters.append(
            ("metric", "==", metric_values[0])
        )

    if horizon_values and len(horizon_values) == 1:
        parquet_filters.append(
            ("horizon", "==", int(horizon_values[0]))
        )

    frame = _read_forecast_parquet(
        FORECAST_TEMPORAL_ZONE_SUMMARY_PATH,
        label="forecast temporal-zone summary",
        columns=columns,
        filters=parquet_filters or None,
    )

    frame = _normalize_common_forecast_columns(
        frame
    )

    if required_columns is not None:
        require_columns(
            frame,
            required_columns,
            label="showcase_forecast_temporal_zone_summary",
        )

    if metric_values is not None:
        frame = frame.loc[
            frame["metric"].isin(
                metric_values
            )
        ].copy()

    if horizon_values is not None:
        frame = frame.loc[
            frame["horizon"].isin(
                [int(value) for value in horizon_values]
            )
        ].copy()

    return frame.reset_index(drop=True)


# ---------------------------------------------------------------------
# Shared forecasting geography context
# ---------------------------------------------------------------------

def load_forecast_geography_context() -> pd.DataFrame:
    """
    Return one Post-CP Taxi-Zone geography lookup for forecasting explorers.

    Raw 18 and Raw 19 both use this same policy-geography and mobility-
    environment assignment, so the grouping contract belongs here.
    """
    assignments = load_canonical_cluster_assignments().copy()

    required = [
        "taxi_zone_id",
        "zone",
        "borough",
        "cbd_spatial_category",
        "pre_post_cp",
        "canonical_cluster_name",
    ]
    require_columns(
        assignments,
        required,
        label="canonical mobility-environment assignments",
    )

    context = assignments.loc[
        assignments["pre_post_cp"]
        .astype(str)
        .eq("post_cp"),
        [
            "taxi_zone_id",
            "zone",
            "borough",
            "cbd_spatial_category",
            "canonical_cluster_name",
        ],
    ].drop_duplicates("taxi_zone_id")

    context["taxi_zone_id"] = pd.to_numeric(
        context["taxi_zone_id"],
        errors="coerce",
    ).astype("Int64")

    context["policy_geography"] = (
        context["cbd_spatial_category"]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace(POLICY_GEOGRAPHY_MAP)
        .fillna("Unknown")
    )

    context = context.rename(
        columns={
            "canonical_cluster_name": "mobility_environment",
        }
    )

    context["mobility_environment"] = (
        context["mobility_environment"]
        .astype("string")
        .fillna("Unknown")
    )

    return context[
        [
            "taxi_zone_id",
            "zone",
            "borough",
            "cbd_spatial_category",
            "policy_geography",
            "mobility_environment",
        ]
    ].reset_index(drop=True)


# ---------------------------------------------------------------------
# Forecast failure / diagnostic artifacts
# ---------------------------------------------------------------------

def load_forecast_failure_cases(
    *,
    columns: Sequence[str] | None = None,
    required_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load the frozen forecasting failure-case registry."""
    frame = _read_forecast_parquet(
        FORECAST_FAILURE_CASES_PATH,
        label="forecast failure case studies",
        columns=columns,
    )

    frame = _normalize_common_forecast_columns(
        frame
    )

    if required_columns is not None:
        require_columns(
            frame,
            required_columns,
            label="forecast_failure_case_studies",
        )

    return frame.reset_index(drop=True)


def load_forecast_failure_context(
    *,
    columns: Sequence[str] | None = None,
    required_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load the frozen context rows behind forecasting failure cases."""
    frame = _read_forecast_parquet(
        FORECAST_FAILURE_CONTEXT_PATH,
        label="forecast failure case context",
        columns=columns,
    )

    frame = _normalize_common_forecast_columns(
        frame
    )

    if required_columns is not None:
        require_columns(
            frame,
            required_columns,
            label="forecast_failure_case_context",
        )

    return frame.reset_index(drop=True)


def load_forecast_failure_mechanisms(
    *,
    columns: Sequence[str] | None = None,
    required_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load the frozen forecasting diagnostic-mechanism summary."""
    frame = _read_forecast_parquet(
        FORECAST_FAILURE_MECHANISM_PATH,
        label="forecast failure mechanism summary",
        columns=columns,
    )

    if required_columns is not None:
        require_columns(
            frame,
            required_columns,
            label="forecast_failure_mechanism_summary",
        )

    return frame.reset_index(drop=True)
