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

APP_TABLES_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "app_tables"
)

FORECAST_RECORD_RUNTIME_PATH = (
    APP_TABLES_DIR
    / "forecast_holdout_runtime.parquet"
)

FORECAST_HISTORY_RUNTIME_PATH = (
    APP_TABLES_DIR
    / "forecast_explorer_runtime.parquet"
)

FORECAST_HOLDOUT_SLICE_SUMMARY_PATH = (
    APP_TABLES_DIR
    / "forecast_holdout_slice_summary.parquet"
)

FORECAST_HOLDOUT_SCATTER_SAMPLE_PATH = (
    APP_TABLES_DIR
    / "forecast_holdout_scatter_sample.parquet"
)

FORECAST_HOLDOUT_CURATED_CASES_PATH = (
    APP_TABLES_DIR
    / "forecast_holdout_curated_cases.parquet"
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

def _forecast_zone_lookup() -> pd.DataFrame:
    """Return one canonical reader-facing geography row per Taxi Zone."""
    zones = pd.read_parquet(
        FORECAST_ZONE_SUMMARY_PATH,
        columns=[
            "taxi_zone_id",
            "zone",
            "borough",
            "reader_facing_zone",
        ],
    )

    zones["taxi_zone_id"] = pd.to_numeric(
        zones["taxi_zone_id"],
        errors="coerce",
    ).astype("Int64")

    return (
        zones[
            [
                "taxi_zone_id",
                "zone",
                "borough",
                "reader_facing_zone",
            ]
        ]
        .drop_duplicates("taxi_zone_id")
        .reset_index(drop=True)
    )


def _forecast_job_lookup() -> pd.DataFrame:
    """Return the selected model family for each metric × horizon."""
    jobs = pd.read_parquet(
        FORECAST_JOB_SUMMARY_PATH,
        columns=[
            "metric",
            "horizon",
            "champion_family",
        ],
    )

    jobs["horizon"] = pd.to_numeric(
        jobs["horizon"],
        errors="coerce",
    ).astype("Int64")

    return (
        jobs[
            [
                "metric",
                "horizon",
                "champion_family",
            ]
        ]
        .drop_duplicates(
            [
                "metric",
                "horizon",
            ]
        )
        .reset_index(drop=True)
    )


def _enrich_forecast_runtime(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Restore lightweight descriptive fields omitted from the compact runtime.

    WHY:
    Zone names, boroughs, reader-facing status, and champion family are
    dimension attributes. Repeating them 1.4 million times in the runtime
    Parquet wastes space, so they are joined from the tiny frozen summaries.
    """
    frame = frame.copy()

    frame = frame.merge(
        _forecast_zone_lookup(),
        on="taxi_zone_id",
        how="left",
        validate="many_to_one",
    )

    frame = frame.merge(
        _forecast_job_lookup(),
        on=[
            "metric",
            "horizon",
        ],
        how="left",
        validate="many_to_one",
    )

    # These quantities are exact functions of the retained native-unit values.
    frame["absolute_error"] = (
        frame["actual"]
        - frame["champion_prediction"]
    ).abs()

    frame["benchmark_absolute_error"] = (
        frame["actual"]
        - frame["benchmark_prediction"]
    ).abs()

    frame["champion_better_than_benchmark"] = (
        frame["absolute_error"]
        < frame["benchmark_absolute_error"]
    )

    return frame

def load_forecast_records(
    *,
    columns: Sequence[str] | None = None,
    required_columns: Sequence[str] | None = None,
    metrics: Iterable[str] | str | None = None,
    horizons: Iterable[int] | int | None = None,
    taxi_zone_ids: Iterable[int] | int | None = None,
    temporal_buckets: Iterable[str] | str | None = None,
    boroughs: Iterable[str] | str | None = None,
    zones: Iterable[str] | str | None = None,
    reader_facing_only: bool = False,
    final_holdout_only: bool = False,
) -> pd.DataFrame:
    """
    Load exact final-holdout forecasting records from the compact runtime.

    Metric, horizon, Taxi Zone, and temporal-bucket filters are pushed into
    the Parquet read whenever possible so interactive pages do not have to
    materialize the entire holdout surface for a narrow slice.
    """
    metric_values = _as_list(metrics)
    horizon_values = _as_list(horizons)
    zone_id_values = _as_list(taxi_zone_ids)
    bucket_values = _as_list(temporal_buckets)
    borough_values = _as_list(boroughs)
    zone_values = _as_list(zones)

    parquet_filters: list[tuple[str, str, object]] = []

    def add_runtime_filter(column: str, values: list[object] | None) -> None:
        """Push scalar or multi-value filters into the compact Parquet read."""
        if not values:
            return

        if len(values) == 1:
            parquet_filters.append((column, "==", values[0]))
        else:
            parquet_filters.append((column, "in", list(values)))

    add_runtime_filter("metric", metric_values)
    add_runtime_filter(
        "horizon",
        [int(value) for value in horizon_values]
        if horizon_values is not None
        else None,
    )
    add_runtime_filter(
        "taxi_zone_id",
        [int(value) for value in zone_id_values]
        if zone_id_values is not None
        else None,
    )
    add_runtime_filter("target_temporal_bucket", bucket_values)

    frame = _read_forecast_parquet(
        FORECAST_RECORD_RUNTIME_PATH,
        label="forecast holdout runtime",
        filters=parquet_filters or None,
    )

    frame = _normalize_common_forecast_columns(frame)
    frame = _enrich_forecast_runtime(frame)

    # WHY: retain explicit Pandas filtering after the Parquet read. It keeps the
    # contract exact even if the local Parquet engine falls back without pushdown.
    if metric_values is not None:
        frame = frame.loc[frame["metric"].isin(metric_values)]

    if horizon_values is not None:
        frame = frame.loc[
            frame["horizon"].isin([int(value) for value in horizon_values])
        ]

    if zone_id_values is not None:
        frame = frame.loc[
            frame["taxi_zone_id"].isin([int(value) for value in zone_id_values])
        ]

    if bucket_values is not None:
        frame = frame.loc[
            frame["target_temporal_bucket"].isin(bucket_values)
        ]

    if borough_values is not None:
        frame = frame.loc[frame["borough"].isin(borough_values)]

    if zone_values is not None:
        frame = frame.loc[frame["zone"].isin(zone_values)]

    if reader_facing_only:
        frame = frame.loc[
            frame["reader_facing_zone"].fillna(False).astype(bool)
        ]

    if final_holdout_only:
        frame = frame.loc[
            frame["target_date"].between(
                FINAL_HOLDOUT_START_DATE,
                FINAL_HOLDOUT_END_DATE,
                inclusive="both",
            )
        ]

    if required_columns is not None:
        require_columns(
            frame,
            required_columns,
            label="forecast_holdout_runtime",
        )

    if columns is not None:
        require_columns(
            frame,
            columns,
            label="forecast_holdout_runtime",
        )
        frame = frame[list(columns)]

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
    Load longitudinal forecasting evidence from the compact wide runtime.

    Storage is one row per target observation. This loader restores the old
    metric × horizon long-form API so downstream analytical behavior remains
    unchanged.
    """
    metric_values = _as_list(metrics)
    horizon_values = _as_list(horizons)
    zone_id_values = _as_list(taxi_zone_ids)

    frame = pd.read_parquet(
        FORECAST_HISTORY_RUNTIME_PATH
    )

    frame["target_date"] = pd.to_datetime(
        frame["target_date"],
        errors="coerce",
    )

    frame["taxi_zone_id"] = pd.to_numeric(
        frame["taxi_zone_id"],
        errors="coerce",
    ).astype("Int64")

    if metric_values is not None:
        frame = frame.loc[
            frame["metric"].isin(
                metric_values
            )
        ]

    if zone_id_values is not None:
        frame = frame.loc[
            frame["taxi_zone_id"].isin(
                [
                    int(value)
                    for value in zone_id_values
                ]
            )
        ]

    requested_horizons = (
        [
            int(value)
            for value in horizon_values
        ]
        if horizon_values is not None
        else list(FORECAST_HORIZONS)
    )

    id_columns = [
        "metric",
        "taxi_zone_id",
        "zone",
        "borough",
        "target_date",
        "target_temporal_bucket",
        "target_observation_sequence_id",
        "evaluation_period",
        "actual",
    ]

    long_frames = []

    for horizon in requested_horizons:
        forecast_column = (
            f"forecast_h{horizon}"
        )
        benchmark_column = (
            f"benchmark_h{horizon}"
        )

        require_columns(
            frame,
            [
                forecast_column,
                benchmark_column,
            ],
            label="forecast_explorer_runtime",
        )

        horizon_frame = frame[
            id_columns
            + [
                forecast_column,
                benchmark_column,
            ]
        ].copy()

        horizon_frame["horizon"] = (
            horizon
        )

        horizon_frame[
            "champion_prediction"
        ] = horizon_frame[
            forecast_column
        ]

        horizon_frame[
            "benchmark_prediction"
        ] = horizon_frame[
            benchmark_column
        ]

        horizon_frame.drop(
            columns=[
                forecast_column,
                benchmark_column,
            ],
            inplace=True,
        )

        long_frames.append(
            horizon_frame
        )

    result = pd.concat(
        long_frames,
        ignore_index=True,
    )

    result["horizon"] = pd.to_numeric(
        result["horizon"],
        errors="coerce",
    ).astype("Int64")

    # Every row in this runtime was already restricted to the canonical
    # reader-facing population by the builder.
    result["reader_facing_zone"] = True

    if reader_facing_only:
        result = result.loc[
            result["reader_facing_zone"]
        ]

    if required_columns is not None:
        require_columns(
            result,
            required_columns,
            label="forecast_explorer_runtime",
        )

    if columns is not None:
        require_columns(
            result,
            columns,
            label="forecast_explorer_runtime",
        )

        result = result[
            list(columns)
        ]

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------
# Forecast summaries
# ---------------------------------------------------------------------

def load_forecast_holdout_slice_summary(
    *,
    metrics: Iterable[str] | str | None = None,
    horizons: Iterable[int] | int | None = None,
) -> pd.DataFrame:
    """Load Raw 20's exact filterable final-holdout summary."""
    metric_values = _as_list(metrics)
    horizon_values = _as_list(horizons)

    parquet_filters: list[tuple[str, str, object]] = []

    if metric_values and len(metric_values) == 1:
        parquet_filters.append(("metric", "==", metric_values[0]))

    if horizon_values and len(horizon_values) == 1:
        parquet_filters.append(("horizon", "==", int(horizon_values[0])))

    frame = _read_forecast_parquet(
        FORECAST_HOLDOUT_SLICE_SUMMARY_PATH,
        label="forecast holdout slice summary",
        filters=parquet_filters or None,
    )

    frame = _normalize_common_forecast_columns(frame)

    if metric_values is not None:
        frame = frame.loc[frame["metric"].isin(metric_values)]

    if horizon_values is not None:
        frame = frame.loc[
            frame["horizon"].isin([int(value) for value in horizon_values])
        ]

    return frame.reset_index(drop=True)

def load_forecast_holdout_scatter_sample() -> pd.DataFrame:
    """Load Raw 20's deterministic tail-preserving Win-Miss sample."""
    frame = _read_forecast_parquet(
        FORECAST_HOLDOUT_SCATTER_SAMPLE_PATH,
        label="forecast holdout scatter sample",
    )

    return _normalize_common_forecast_columns(frame).reset_index(drop=True)

def load_forecast_holdout_curated_cases() -> pd.DataFrame:
    """Load Raw 20's frozen clean-win, hard-helpful, and clear-miss cases."""
    frame = _read_forecast_parquet(
        FORECAST_HOLDOUT_CURATED_CASES_PATH,
        label="forecast holdout curated cases",
    )

    frame = _normalize_common_forecast_columns(frame)

    required = [
        "curated_group",
        "curated_rank",
        "metric",
        "horizon",
        "taxi_zone_id",
        "zone",
        "borough",
        "target_date",
        "target_temporal_bucket",
        "actual",
        "champion_prediction",
        "benchmark_prediction",
        "severe_error",
        "system_row_error_index",
        "benchmark_advantage_index",
    ]

    require_columns(
        frame,
        required,
        label="forecast_holdout_curated_cases",
    )

    return (
        frame.sort_values(["curated_group", "curated_rank"])
        .reset_index(drop=True)
    )

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
