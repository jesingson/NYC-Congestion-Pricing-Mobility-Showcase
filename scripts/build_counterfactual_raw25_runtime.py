"""
Rebuild the compact Raw 25 serving artifacts with slice-specific
Pre-CP Reference-system calibration.

WHY THIS EXISTS
---------------
Raw 25 asks how large the observed-versus-no-CP mobility gap is relative
to the forecasting error normally seen before congestion pricing.

The old Raw 25 runtime used one systemwide MAE per Metric × Horizon.
That meant a Brooklyn, CBD, Taxi Zone, weekday, or PM-peak result could
be divided by a denominator calculated at a different aggregation level.

The corrected runtime uses:

    counterfactual_raw25_reference_mae.parquet

whose MAE is already calculated at the matching:

    Geography
    × Day type
    × Daypart
    × Metric
    × Horizon

The post-CP numerator is rebuilt at that same aggregation level:

- counts / ridership: SUM observed and no-CP values first;
- average speeds: activity-weight observed and no-CP worlds separately;
- only then calculate the post-CP gap;
- divide that aggregate gap by the matching Pre-CP Reference MAE.

The canonical Chapter 5 `counterfactual_gap_mae_units` field is never
modified. Raw 23 and Raw 24 therefore retain their existing contract.

INPUTS
------
data/processed/5.3.1.final_tables/
    counterfactual_gap_surface.parquet
    counterfactual_horizon_stability.parquet
    counterfactual_raw25_reference_mae.parquet
    counterfactual_raw25_reference_mae_qa.parquet

OUTPUTS
-------
data/processed/5.3.1.final_tables/
    counterfactual_raw25_job_histogram.parquet
    counterfactual_raw25_job_summary.parquet
    counterfactual_raw25_slice_scout.parquet
    counterfactual_raw25_qa.parquet

RUN
---
python scripts/build_counterfactual_raw25_runtime.py
"""

from __future__ import annotations

import gc
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FINAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "5.3.1.final_tables"
)

GAP_SURFACE_PATH = (
    FINAL_DIR
    / "counterfactual_gap_surface.parquet"
)

CONTEXT_PATH = (
    FINAL_DIR
    / "counterfactual_horizon_stability.parquet"
)

REFERENCE_MAE_PATH = (
    FINAL_DIR
    / "counterfactual_raw25_reference_mae.parquet"
)

REFERENCE_MAE_QA_PATH = (
    FINAL_DIR
    / "counterfactual_raw25_reference_mae_qa.parquet"
)

JOB_HISTOGRAM_PATH = (
    FINAL_DIR
    / "counterfactual_raw25_job_histogram.parquet"
)

JOB_SUMMARY_PATH = (
    FINAL_DIR
    / "counterfactual_raw25_job_summary.parquet"
)

SLICE_SCOUT_PATH = (
    FINAL_DIR
    / "counterfactual_raw25_slice_scout.parquet"
)

QA_PATH = (
    FINAL_DIR
    / "counterfactual_raw25_qa.parquet"
)


# ---------------------------------------------------------------------
# Frozen analytical contract
# ---------------------------------------------------------------------

CP_START = pd.Timestamp("2025-01-05")
CP_END = pd.Timestamp("2026-03-31")

METRICS = [
    "taxi_trip_count",
    "taxi_avg_trip_speed",
    "fhvhv_trip_count",
    "fhvhv_avg_trip_speed",
    "subway_ridership",
]

HORIZONS = [
    1,
    2,
    5,
]

COUNT_METRICS = {
    "taxi_trip_count",
    "fhvhv_trip_count",
    "subway_ridership",
}

SPEED_METRICS = {
    "taxi_avg_trip_speed",
    "fhvhv_avg_trip_speed",
}

GEOGRAPHY_TYPES = [
    "Systemwide",
    "Borough",
    "Policy geography",
    "Mobility environment",
    "Taxi Zone",
]

DAY_TYPES = [
    "All days",
    "Weekdays",
    "Weekends",
]

DAYPARTS = [
    "All dayparts",
    "Overnight",
    "AM peak",
    "Midday",
    "PM peak",
    "Evening",
]

# Raw 25 should not expose an extremely sparse post-CP distribution even when
# the corresponding Pre-CP calibration slice itself has sufficient support.
MIN_POST_CP_PERIODS = 10

# Histogram is descriptive only. Values beyond 8× MAE are collected into one
# overflow bin rather than stretching the display ruler indefinitely.
HISTOGRAM_BIN_WIDTH = 0.25
HISTOGRAM_OVERFLOW_AT = 8.0

ZONE_ALIASES = {
    57: 56,
    105: 103,
}

UNKNOWN_ZONE_IDS = {
    264,
    265,
}


# ---------------------------------------------------------------------
# Source schemas
# ---------------------------------------------------------------------

GAP_SOURCE_COLUMNS = [
    "taxi_zone_id",
    "metric",
    "horizon",
    "target_date",
    "target_temporal_bucket",
    "system_prediction",
    "target_observed_value",
    "primary_gap_eligible",
    "observed_activity_weight",
    "no_cp_activity_weight",
]

CONTEXT_COLUMNS = [
    "taxi_zone_id",
    "zone",
    "borough",
    "cbd_spatial_category",
    "pre_cp_mobility_environment",
]

REFERENCE_REQUIRED_COLUMNS = {
    "geography_type",
    "geography_id",
    "geography_label",
    "day_type",
    "daypart",
    "metric",
    "horizon",
    "pre_cp_reference_mae",
    "validation_periods",
    "validation_rows",
    "support_ok",
}

REFERENCE_KEY = [
    "geography_type",
    "geography_id",
    "day_type",
    "daypart",
    "metric",
    "horizon",
]

SLICE_KEY = [
    "geography_type",
    "geography_id",
    "day_type",
    "daypart",
    "metric",
    "horizon",
]


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------


def require_file(
    path: Path,
    label: str,
) -> None:
    """Fail loudly when one frozen runtime input is unavailable."""
    if not path.exists():
        raise FileNotFoundError(
            f"{label} was not found:\n  {path}"
        )


def require_parquet_columns(
    path: Path,
    required_columns: set[str] | list[str],
    label: str,
) -> None:
    """Validate a Parquet schema without loading the full artifact."""
    observed = set(
        pq.ParquetFile(path).schema.names
    )

    missing = sorted(
        set(required_columns) - observed
    )

    if missing:
        raise KeyError(
            f"{label} is missing required columns: {missing}"
        )


def canonicalize_zone_ids(
    series: pd.Series,
) -> pd.Series:
    """Map historical split IDs onto the reader-facing physical zone."""
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).astype("Int64")

    return values.replace(
        ZONE_ALIASES
    )


def atomic_stage_parquet(
    frame: pd.DataFrame,
    target_path: Path,
) -> Path:
    """
    Write one staged Parquet artifact.

    WHY:
    Existing Raw 25 artifacts remain untouched until every new artifact has
    been built and the complete runtime passes QA.
    """
    target_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = target_path.with_name(
        f".{target_path.stem}.{uuid.uuid4().hex[:8]}.tmp.parquet"
    )

    frame.to_parquet(
        temp_path,
        index=False,
        compression="zstd",
    )

    return temp_path


def publish_staged_artifacts(
    staged: dict[Path, Path],
) -> None:
    """Atomically replace the existing Raw 25 serving files."""
    for target_path, temp_path in staged.items():
        os.replace(
            temp_path,
            target_path,
        )


def cleanup_staged_artifacts(
    staged: dict[Path, Path],
) -> None:
    """Remove temporary files after a failed publication attempt."""
    for temp_path in staged.values():
        if temp_path.exists():
            temp_path.unlink()


def pct_ge(
    series: pd.Series,
    threshold: float,
) -> float:
    """Return the percentage of finite values at or above one threshold."""
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = values.loc[
        np.isfinite(values)
    ]

    if values.empty:
        return np.nan

    return float(
        values.ge(threshold).mean()
        * 100.0
    )


# ---------------------------------------------------------------------
# Input QA
# ---------------------------------------------------------------------


def validate_reference_qa() -> pd.DataFrame:
    """Require the newly built Reference-MAE lookup to have passed its QA."""
    require_file(
        REFERENCE_MAE_QA_PATH,
        "Raw 25 Reference-MAE QA",
    )

    qa = pd.read_parquet(
        REFERENCE_MAE_QA_PATH
    )

    required = {
        "check_id",
        "status",
        "details",
    }

    missing = sorted(
        required - set(qa.columns)
    )

    if missing:
        raise KeyError(
            "Reference-MAE QA is missing columns: "
            f"{missing}"
        )

    failures = qa.loc[
        qa["status"]
        .astype(str)
        .str.upper()
        .ne("PASS")
    ]

    if not failures.empty:
        raise RuntimeError(
            "Reference-MAE preprocessing did not pass QA:\n\n"
            + failures.to_string(index=False)
        )

    return qa


def load_reference_lookup() -> pd.DataFrame:
    """Load and normalize the slice-specific Pre-CP Reference MAE contract."""
    require_file(
        REFERENCE_MAE_PATH,
        "Raw 25 Reference-MAE lookup",
    )

    require_parquet_columns(
        REFERENCE_MAE_PATH,
        REFERENCE_REQUIRED_COLUMNS,
        "Raw 25 Reference-MAE lookup",
    )

    reference = pd.read_parquet(
        REFERENCE_MAE_PATH
    ).copy()

    reference["geography_type"] = (
        reference["geography_type"]
        .astype("string")
    )

    reference["geography_id"] = (
        reference["geography_id"]
        .astype("string")
    )

    reference["geography_label"] = (
        reference["geography_label"]
        .astype("string")
    )

    reference["day_type"] = (
        reference["day_type"]
        .astype("string")
    )

    reference["daypart"] = (
        reference["daypart"]
        .astype("string")
    )

    reference["metric"] = (
        reference["metric"]
        .astype("string")
    )

    reference["horizon"] = pd.to_numeric(
        reference["horizon"],
        errors="raise",
    ).astype(int)

    reference["pre_cp_reference_mae"] = pd.to_numeric(
        reference["pre_cp_reference_mae"],
        errors="coerce",
    )

    reference["validation_periods"] = pd.to_numeric(
        reference["validation_periods"],
        errors="raise",
    ).astype(int)

    reference["validation_rows"] = pd.to_numeric(
        reference["validation_rows"],
        errors="raise",
    ).astype(int)

    reference["support_ok"] = (
        reference["support_ok"]
        .fillna(False)
        .astype(bool)
    )

    duplicate_count = int(
        reference.duplicated(
            REFERENCE_KEY
        ).sum()
    )

    if duplicate_count:
        raise RuntimeError(
            "Reference-MAE lookup has "
            f"{duplicate_count:,} duplicate lookup keys."
        )

    expected_jobs = {
        (metric, horizon)
        for metric in METRICS
        for horizon in HORIZONS
    }

    observed_jobs = set(
        reference[
            [
                "metric",
                "horizon",
            ]
        ]
        .drop_duplicates()
        .itertuples(
            index=False,
            name=None,
        )
    )

    if observed_jobs != expected_jobs:
        raise RuntimeError(
            "Reference-MAE lookup does not preserve the expected "
            "15 Metric × Horizon jobs."
        )

    return reference


# ---------------------------------------------------------------------
# Static physical-zone context
# ---------------------------------------------------------------------


def load_zone_context() -> pd.DataFrame:
    """
    Reuse the same physical-zone context used to build the Reference MAE.

    WHY:
    The numerator and denominator must agree on the 57→56 and 105→103
    physical-zone normalization and on the geography attached to those zones.
    """
    require_file(
        CONTEXT_PATH,
        "Counterfactual horizon-stability context",
    )

    require_parquet_columns(
        CONTEXT_PATH,
        CONTEXT_COLUMNS,
        "Counterfactual horizon-stability context",
    )

    context = (
        pd.read_parquet(
            CONTEXT_PATH,
            columns=CONTEXT_COLUMNS,
        )
        .drop_duplicates()
        .copy()
    )

    context["taxi_zone_id"] = pd.to_numeric(
        context["taxi_zone_id"],
        errors="coerce",
    ).astype("Int64")

    context["canonical_taxi_zone_id"] = canonicalize_zone_ids(
        context["taxi_zone_id"]
    )

    # Prefer metadata already attached to the canonical physical-zone ID when
    # both the canonical ID and a historical alias are present.
    context["_canonical_row"] = (
        context["taxi_zone_id"]
        .eq(
            context["canonical_taxi_zone_id"]
        )
    )

    context = (
        context.sort_values(
            [
                "canonical_taxi_zone_id",
                "_canonical_row",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop_duplicates(
            subset=[
                "canonical_taxi_zone_id",
            ],
            keep="first",
        )
        [
            [
                "canonical_taxi_zone_id",
                "zone",
                "borough",
                "cbd_spatial_category",
                "pre_cp_mobility_environment",
            ]
        ]
        .reset_index(drop=True)
    )

    if context[
        "canonical_taxi_zone_id"
    ].duplicated().any():
        raise RuntimeError(
            "Canonical Taxi Zone context is not unique."
        )

    return context


# ---------------------------------------------------------------------
# Post-CP source loading
# ---------------------------------------------------------------------


def daypart_from_bucket(
    series: pd.Series,
) -> pd.Series:
    """Translate temporal-bucket IDs into Raw 25's five daypart labels."""
    bucket = (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )

    result = pd.Series(
        pd.NA,
        index=series.index,
        dtype="string",
    )

    mapping = {
        "overnight": "Overnight",
        "am_peak": "AM peak",
        "midday": "Midday",
        "pm_peak": "PM peak",
        "evening": "Evening",
    }

    for suffix, label in mapping.items():
        mask = bucket.str.endswith(
            suffix,
            na=False,
        )

        result.loc[mask] = label

    if result.isna().any():
        unexpected = sorted(
            bucket.loc[
                result.isna()
            ]
            .dropna()
            .unique()
            .tolist()
        )

        raise RuntimeError(
            "Unexpected target_temporal_bucket values: "
            f"{unexpected}"
        )

    return result


def load_metric_source(
    metric: str,
    zone_context: pd.DataFrame,
) -> pd.DataFrame:
    """Load one metric at a time to keep the runtime build memory-friendly."""
    try:
        frame = pd.read_parquet(
            GAP_SURFACE_PATH,
            columns=GAP_SOURCE_COLUMNS,
            filters=[
                (
                    "metric",
                    "==",
                    metric,
                ),
                (
                    "primary_gap_eligible",
                    "==",
                    True,
                ),
            ],
        )

        predicate_message = "used"

    except Exception as exc:
        print(
            "    Predicate pushdown unavailable; "
            f"using projected-column fallback ({type(exc).__name__})."
        )

        frame = pd.read_parquet(
            GAP_SURFACE_PATH,
            columns=GAP_SOURCE_COLUMNS,
        )

        predicate_message = "fallback"

    frame["metric"] = (
        frame["metric"]
        .astype("string")
    )

    frame["horizon"] = pd.to_numeric(
        frame["horizon"],
        errors="raise",
    ).astype(int)

    frame["target_date"] = pd.to_datetime(
        frame["target_date"],
        errors="raise",
    ).dt.normalize()

    frame["primary_gap_eligible"] = (
        frame["primary_gap_eligible"]
        .fillna(False)
        .astype(bool)
    )

    frame = frame.loc[
        frame["metric"].eq(metric)
        & frame["horizon"].isin(HORIZONS)
        & frame["primary_gap_eligible"]
        & frame["target_date"].between(
            CP_START,
            CP_END,
            inclusive="both",
        )
    ].copy()

    frame["taxi_zone_id"] = pd.to_numeric(
        frame["taxi_zone_id"],
        errors="coerce",
    ).astype("Int64")

    frame["canonical_taxi_zone_id"] = canonicalize_zone_ids(
        frame["taxi_zone_id"]
    )

    numeric_columns = [
        "system_prediction",
        "target_observed_value",
        "observed_activity_weight",
        "no_cp_activity_weight",
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame["day_type"] = np.where(
        frame["target_date"]
        .dt.dayofweek
        .lt(5),
        "Weekdays",
        "Weekends",
    )

    frame["daypart"] = daypart_from_bucket(
        frame["target_temporal_bucket"]
    )

    # Attach geography from the same canonical context used by the Pre-CP
    # calibration builder. Systemwide calculations still retain every row,
    # including Unknown 264/265 where present.
    frame = frame.merge(
        zone_context,
        on="canonical_taxi_zone_id",
        how="left",
        validate="many_to_one",
    )

    print(
        f"    rows={len(frame):,}; predicate_pushdown={predicate_message}",
        flush=True,
    )

    return frame


# ---------------------------------------------------------------------
# Geography projection
# ---------------------------------------------------------------------


def project_geography(
    frame: pd.DataFrame,
    geography_type: str,
) -> pd.DataFrame:
    """Project one post-CP job onto a Raw 25 reader-facing geography level."""
    scoped = frame.copy()

    if geography_type == "Systemwide":
        scoped["geography_id"] = "systemwide"
        return scoped

    # Unknown / nonphysical zones remain part of Systemwide totals but are not
    # reader-facing members of the other geography systems.
    scoped = scoped.loc[
        ~scoped[
            "canonical_taxi_zone_id"
        ].isin(UNKNOWN_ZONE_IDS)
    ].copy()

    if geography_type == "Borough":
        source_column = "borough"

    elif geography_type == "Policy geography":
        source_column = "cbd_spatial_category"

    elif geography_type == "Mobility environment":
        source_column = "pre_cp_mobility_environment"

    elif geography_type == "Taxi Zone":
        scoped = scoped.loc[
            scoped["canonical_taxi_zone_id"].notna()
            & scoped["zone"].notna()
        ].copy()

        scoped["geography_id"] = (
            scoped["canonical_taxi_zone_id"]
            .astype(int)
            .astype(str)
        )

        return scoped

    else:
        raise ValueError(
            f"Unsupported geography type: {geography_type}"
        )

    scoped = scoped.loc[
        scoped[source_column].notna()
    ].copy()

    scoped["geography_id"] = (
        scoped[source_column]
        .astype("string")
    )

    return scoped


# ---------------------------------------------------------------------
# Aggregate the post-CP numerator at the matching geography level
# ---------------------------------------------------------------------


def aggregate_count_periods(
    scoped: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sum additive observed and no-CP quantities before calculating the gap.

    One output row is one:
        Geography × Date × Temporal Bucket
    post-CP target period.
    """
    observed = pd.to_numeric(
        scoped["target_observed_value"],
        errors="coerce",
    )

    no_cp = pd.to_numeric(
        scoped["system_prediction"],
        errors="coerce",
    )

    eligible = (
        np.isfinite(observed)
        & np.isfinite(no_cp)
    )

    working = scoped.loc[
        eligible
    ].copy()

    working["_observed"] = (
        observed.loc[
            eligible
        ].to_numpy()
    )

    working["_no_cp"] = (
        no_cp.loc[
            eligible
        ].to_numpy()
    )

    group_columns = [
        "geography_id",
        "target_date",
        "target_temporal_bucket",
        "day_type",
        "daypart",
    ]

    periods = (
        working.groupby(
            group_columns,
            observed=True,
            dropna=False,
            sort=False,
        )
        .agg(
            observed_level=(
                "_observed",
                "sum",
            ),
            no_cp_level=(
                "_no_cp",
                "sum",
            ),
            source_rows=(
                "_observed",
                "size",
            ),
        )
        .reset_index()
    )

    periods["counterfactual_gap_native"] = (
        periods["no_cp_level"]
        - periods["observed_level"]
    )

    return periods


def aggregate_speed_periods(
    scoped: pd.DataFrame,
) -> pd.DataFrame:
    """
    Activity-weight observed and no-CP speed worlds separately.

    WHY:
    The observed world and synthetic no-CP world can have different trip-count
    activity. Using one world's weights for the other would distort the gap.
    """
    observed_speed = pd.to_numeric(
        scoped["target_observed_value"],
        errors="coerce",
    )

    no_cp_speed = pd.to_numeric(
        scoped["system_prediction"],
        errors="coerce",
    )

    observed_weight = pd.to_numeric(
        scoped["observed_activity_weight"],
        errors="coerce",
    )

    no_cp_weight = pd.to_numeric(
        scoped["no_cp_activity_weight"],
        errors="coerce",
    )

    eligible = (
        np.isfinite(observed_speed)
        & np.isfinite(no_cp_speed)
        & np.isfinite(observed_weight)
        & np.isfinite(no_cp_weight)
        & observed_weight.ge(0)
        & no_cp_weight.ge(0)
    )

    working = scoped.loc[
        eligible
    ].copy()

    working["_observed_weight"] = (
        observed_weight.loc[
            eligible
        ].to_numpy()
    )

    working["_no_cp_weight"] = (
        no_cp_weight.loc[
            eligible
        ].to_numpy()
    )

    working["_observed_numerator"] = (
        observed_speed.loc[
            eligible
        ].to_numpy()
        * working["_observed_weight"].to_numpy()
    )

    working["_no_cp_numerator"] = (
        no_cp_speed.loc[
            eligible
        ].to_numpy()
        * working["_no_cp_weight"].to_numpy()
    )

    group_columns = [
        "geography_id",
        "target_date",
        "target_temporal_bucket",
        "day_type",
        "daypart",
    ]

    periods = (
        working.groupby(
            group_columns,
            observed=True,
            dropna=False,
            sort=False,
        )
        .agg(
            observed_numerator=(
                "_observed_numerator",
                "sum",
            ),
            observed_denominator=(
                "_observed_weight",
                "sum",
            ),
            no_cp_numerator=(
                "_no_cp_numerator",
                "sum",
            ),
            no_cp_denominator=(
                "_no_cp_weight",
                "sum",
            ),
            source_rows=(
                "_observed_numerator",
                "size",
            ),
        )
        .reset_index()
    )

    periods = periods.loc[
        periods[
            "observed_denominator"
        ].gt(0)
        & periods[
            "no_cp_denominator"
        ].gt(0)
    ].copy()

    periods["observed_level"] = (
        periods["observed_numerator"]
        / periods["observed_denominator"]
    )

    periods["no_cp_level"] = (
        periods["no_cp_numerator"]
        / periods["no_cp_denominator"]
    )

    periods["counterfactual_gap_native"] = (
        periods["no_cp_level"]
        - periods["observed_level"]
    )

    keep = [
        "geography_id",
        "target_date",
        "target_temporal_bucket",
        "day_type",
        "daypart",
        "observed_level",
        "no_cp_level",
        "counterfactual_gap_native",
        "source_rows",
    ]

    return periods[
        keep
    ]


def aggregate_post_cp_periods(
    frame: pd.DataFrame,
    *,
    geography_type: str,
    metric: str,
) -> pd.DataFrame:
    """Apply the correct metric-specific post-CP aggregation contract."""
    scoped = project_geography(
        frame,
        geography_type,
    )

    if scoped.empty:
        return pd.DataFrame()

    if metric in COUNT_METRICS:
        periods = aggregate_count_periods(
            scoped
        )

    elif metric in SPEED_METRICS:
        periods = aggregate_speed_periods(
            scoped
        )

    else:
        raise ValueError(
            f"Unsupported metric: {metric}"
        )

    periods["geography_type"] = (
        geography_type
    )

    return periods


# ---------------------------------------------------------------------
# Expand each post-CP target period into Raw 25's four applicable
# day-type × daypart scopes.
# ---------------------------------------------------------------------


def expand_time_scopes(
    periods: pd.DataFrame,
) -> pd.DataFrame:
    """
    Let one target period contribute to each applicable Raw 25 scope.

    Example:
        Tuesday PM peak

    contributes to:
        All days × All dayparts
        Weekdays × All dayparts
        All days × PM peak
        Weekdays × PM peak

    This does NOT average dayparts together. It simply determines which target
    periods enter each slice's post-CP gap distribution.
    """
    base_columns = [
        "geography_type",
        "geography_id",
        "target_date",
        "target_temporal_bucket",
        "day_type",
        "daypart",
        "counterfactual_gap_native",
        "source_rows",
    ]

    base = periods[
        base_columns
    ].copy()

    all_all = base.copy()
    all_all["day_type"] = "All days"
    all_all["daypart"] = "All dayparts"

    actual_day_all_part = base.copy()
    actual_day_all_part["daypart"] = (
        "All dayparts"
    )

    all_day_actual_part = base.copy()
    all_day_actual_part["day_type"] = (
        "All days"
    )

    actual_actual = base

    return pd.concat(
        [
            all_all,
            actual_day_all_part,
            all_day_actual_part,
            actual_actual,
        ],
        ignore_index=True,
    )


# ---------------------------------------------------------------------
# Join the matching Reference MAE and summarize ×MAE magnitude
# ---------------------------------------------------------------------


def summarize_calibrated_slices(
    periods: pd.DataFrame,
    *,
    reference_job: pd.DataFrame,
    metric: str,
    horizon: int,
    geography_type: str,
    diagnostics: dict[str, int],
) -> pd.DataFrame:
    """
    Build Raw 25 percentiles and threshold shares using the matching denominator.
    """
    expanded = expand_time_scopes(
        periods
    )

    reference_columns = [
        "geography_id",
        "geography_label",
        "day_type",
        "daypart",
        "pre_cp_reference_mae",
        "validation_periods",
        "validation_rows",
        "support_ok",
    ]

    reference_scope = (
        reference_job[
            reference_columns
        ]
        .rename(
            columns={
                "validation_periods":
                    "reference_validation_periods",
                "validation_rows":
                    "reference_validation_rows",
                "support_ok":
                    "reference_support_ok",
            }
        )
        .copy()
    )

    merged = expanded.merge(
        reference_scope,
        on=[
            "geography_id",
            "day_type",
            "daypart",
        ],
        how="left",
        validate="many_to_one",
        indicator="_reference_join",
    )

    diagnostics[
        "expanded_post_cp_period_rows"
    ] += len(merged)

    diagnostics[
        "unmatched_reference_period_rows"
    ] += int(
        merged["_reference_join"]
        .ne("both")
        .sum()
    )

    mae = pd.to_numeric(
        merged["pre_cp_reference_mae"],
        errors="coerce",
    )

    gap = pd.to_numeric(
        merged["counterfactual_gap_native"],
        errors="coerce",
    )

    usable = (
        merged["_reference_join"].eq("both")
        & np.isfinite(mae)
        & mae.gt(0)
        & np.isfinite(gap)
    )

    merged = merged.loc[
        usable
    ].copy()

    if merged.empty:
        return pd.DataFrame()

    merged[
        "counterfactual_gap_reference_mae_units"
    ] = (
        merged["counterfactual_gap_native"]
        / merged["pre_cp_reference_mae"]
    )

    merged["_abs_reference_mae_units"] = (
        merged[
            "counterfactual_gap_reference_mae_units"
        ].abs()
    )

    group_columns = [
        "geography_type",
        "geography_id",
        "geography_label",
        "day_type",
        "daypart",
        "pre_cp_reference_mae",
        "reference_validation_periods",
        "reference_validation_rows",
        "reference_support_ok",
    ]

    grouped = (
        merged.groupby(
            group_columns,
            observed=True,
            dropna=False,
            sort=False,
        )
        .agg(
            rows=(
                "_abs_reference_mae_units",
                "size",
            ),
            source_rows=(
                "source_rows",
                "sum",
            ),
            p50_abs_mae_units=(
                "_abs_reference_mae_units",
                lambda values: values.quantile(0.50),
            ),
            p75_abs_mae_units=(
                "_abs_reference_mae_units",
                lambda values: values.quantile(0.75),
            ),
            p90_abs_mae_units=(
                "_abs_reference_mae_units",
                lambda values: values.quantile(0.90),
            ),
            p95_abs_mae_units=(
                "_abs_reference_mae_units",
                lambda values: values.quantile(0.95),
            ),
            p99_abs_mae_units=(
                "_abs_reference_mae_units",
                lambda values: values.quantile(0.99),
            ),
            share_abs_ge_0_5_mae_pct=(
                "_abs_reference_mae_units",
                lambda values: pct_ge(
                    values,
                    0.5,
                ),
            ),
            share_abs_ge_1_0_mae_pct=(
                "_abs_reference_mae_units",
                lambda values: pct_ge(
                    values,
                    1.0,
                ),
            ),
            share_abs_ge_2_0_mae_pct=(
                "_abs_reference_mae_units",
                lambda values: pct_ge(
                    values,
                    2.0,
                ),
            ),
            share_abs_ge_3_0_mae_pct=(
                "_abs_reference_mae_units",
                lambda values: pct_ge(
                    values,
                    3.0,
                ),
            ),
            overflow_ge_8_mae_pct=(
                "_abs_reference_mae_units",
                lambda values: pct_ge(
                    values,
                    HISTOGRAM_OVERFLOW_AT,
                ),
            ),
        )
        .reset_index()
    )

    grouped["metric"] = metric
    grouped["horizon"] = int(
        horizon
    )

    grouped["rows"] = pd.to_numeric(
        grouped["rows"],
        errors="raise",
    ).astype(int)

    grouped["source_rows"] = pd.to_numeric(
        grouped["source_rows"],
        errors="raise",
    ).astype(int)

    grouped["reference_support_ok"] = (
        grouped["reference_support_ok"]
        .fillna(False)
        .astype(bool)
    )

    # A reader-facing slice needs support on BOTH sides:
    # enough Pre-CP Reference validation periods and enough post-CP periods.
    grouped["support_ok"] = (
        grouped["reference_support_ok"]
        & grouped["rows"].ge(
            MIN_POST_CP_PERIODS
        )
    )

    return grouped


# ---------------------------------------------------------------------
# Systemwide histogram
# ---------------------------------------------------------------------


def build_histogram(
    absolute_xmae: pd.Series,
    *,
    metric: str,
    horizon: int,
) -> pd.DataFrame:
    """Build one compact absolute-gap histogram for the systemwide hero job."""
    values = pd.to_numeric(
        absolute_xmae,
        errors="coerce",
    )

    values = values.loc[
        np.isfinite(values)
        & values.ge(0)
    ]

    if values.empty:
        raise RuntimeError(
            f"No finite Systemwide ×MAE values for {metric} · h={horizon}."
        )

    edges = np.arange(
        0.0,
        HISTOGRAM_OVERFLOW_AT
        + HISTOGRAM_BIN_WIDTH,
        HISTOGRAM_BIN_WIDTH,
    )

    finite_values = values.loc[
        values.lt(
            HISTOGRAM_OVERFLOW_AT
        )
    ]

    counts, _ = np.histogram(
        finite_values,
        bins=edges,
    )

    total = len(values)

    rows = []

    for left, right, count in zip(
        edges[:-1],
        edges[1:],
        counts,
    ):
        rows.append(
            {
                "metric": metric,
                "horizon": int(horizon),
                "bin_mid": (
                    float(left)
                    + float(right)
                ) / 2.0,
                "density_pct": (
                    float(count)
                    / total
                    * 100.0
                ),
                "overflow": False,
            }
        )

    overflow_count = int(
        values.ge(
            HISTOGRAM_OVERFLOW_AT
        ).sum()
    )

    rows.append(
        {
            "metric": metric,
            "horizon": int(horizon),
            "bin_mid": HISTOGRAM_OVERFLOW_AT,
            "density_pct": (
                overflow_count
                / total
                * 100.0
            ),
            "overflow": True,
        }
    )

    return pd.DataFrame(
        rows
    )


def systemwide_absolute_xmae(
    periods: pd.DataFrame,
    *,
    reference_job: pd.DataFrame,
    metric: str,
    horizon: int,
) -> pd.Series:
    """Return the corrected full-period Systemwide |gap| / Reference-MAE values."""
    match = reference_job.loc[
        reference_job["geography_type"]
        .eq("Systemwide")
        & reference_job["geography_id"]
        .astype(str)
        .eq("systemwide")
        & reference_job["day_type"]
        .eq("All days")
        & reference_job["daypart"]
        .eq("All dayparts")
    ]

    if len(match) != 1:
        raise RuntimeError(
            "Expected exactly one Systemwide · All days · All dayparts "
            f"Reference-MAE row for {metric} · h={horizon}; "
            f"found {len(match)}."
        )

    mae = float(
        match.iloc[0][
            "pre_cp_reference_mae"
        ]
    )

    if not np.isfinite(mae) or mae <= 0:
        raise RuntimeError(
            "Systemwide Pre-CP Reference MAE must be positive for "
            f"{metric} · h={horizon}."
        )

    values = (
        pd.to_numeric(
            periods[
                "counterfactual_gap_native"
            ],
            errors="coerce",
        )
        .abs()
        / mae
    )

    return values.loc[
        np.isfinite(values)
    ].reset_index(
        drop=True
    )


# ---------------------------------------------------------------------
# Main runtime build
# ---------------------------------------------------------------------


def build_runtime(
    reference: pd.DataFrame,
    zone_context: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, int],
]:
    """Build slice scout, systemwide summary, and histogram artifacts."""
    slice_parts: list[pd.DataFrame] = []
    histogram_parts: list[pd.DataFrame] = []

    diagnostics = {
        "expanded_post_cp_period_rows": 0,
        "unmatched_reference_period_rows": 0,
        "raw_source_rows": 0,
    }

    for metric in METRICS:
        print()
        print(
            f"Metric: {metric}",
            flush=True,
        )

        metric_frame = load_metric_source(
            metric,
            zone_context,
        )

        diagnostics[
            "raw_source_rows"
        ] += len(metric_frame)

        for horizon in HORIZONS:
            job_source = metric_frame.loc[
                metric_frame["horizon"]
                .eq(horizon)
            ].copy()

            if job_source.empty:
                raise RuntimeError(
                    f"No post-CP source rows for {metric} · h={horizon}."
                )

            reference_job = reference.loc[
                reference["metric"].eq(metric)
                & reference["horizon"].eq(horizon)
            ].copy()

            print(
                f"  h={horizon}: source_rows={len(job_source):,}",
                flush=True,
            )

            for geography_type in GEOGRAPHY_TYPES:
                periods = aggregate_post_cp_periods(
                    job_source,
                    geography_type=geography_type,
                    metric=metric,
                )

                if periods.empty:
                    continue

                # The histogram uses the exact same Systemwide native gaps and
                # exact same Systemwide Reference denominator as the slice scout.
                if geography_type == "Systemwide":
                    x_values = systemwide_absolute_xmae(
                        periods,
                        reference_job=reference_job,
                        metric=metric,
                        horizon=horizon,
                    )

                    histogram_parts.append(
                        build_histogram(
                            x_values,
                            metric=metric,
                            horizon=horizon,
                        )
                    )

                slice_summary = summarize_calibrated_slices(
                    periods,
                    reference_job=reference_job.loc[
                        reference_job[
                            "geography_type"
                        ].eq(
                            geography_type
                        )
                    ].copy(),
                    metric=metric,
                    horizon=horizon,
                    geography_type=geography_type,
                    diagnostics=diagnostics,
                )

                if not slice_summary.empty:
                    slice_parts.append(
                        slice_summary
                    )

                del periods
                del slice_summary

            del job_source
            gc.collect()

        del metric_frame
        gc.collect()

    if not slice_parts:
        raise RuntimeError(
            "No Raw 25 slice summaries were produced."
        )

    if not histogram_parts:
        raise RuntimeError(
            "No Raw 25 systemwide histograms were produced."
        )

    slice_scout = pd.concat(
        slice_parts,
        ignore_index=True,
    )

    histogram = pd.concat(
        histogram_parts,
        ignore_index=True,
    )

    # -----------------------------------------------------------------
    # Freeze the hero as the matching Systemwide/full-period slice.
    # -----------------------------------------------------------------

    job_summary = slice_scout.loc[
        slice_scout[
            "geography_type"
        ].eq("Systemwide")
        & slice_scout[
            "geography_id"
        ].astype(str).eq(
            "systemwide"
        )
        & slice_scout[
            "day_type"
        ].eq("All days")
        & slice_scout[
            "daypart"
        ].eq("All dayparts")
    ].copy()

    if len(job_summary) != 15:
        raise RuntimeError(
            "Expected 15 Systemwide Raw 25 hero jobs; "
            f"found {len(job_summary)}."
        )

    # Preserve the existing summary aliases used by the current visual grammar.
    # Their values are now based on the corrected Reference-system denominator.
    job_summary[
        "median_absolute_mae_units"
    ] = (
        job_summary[
            "p50_abs_mae_units"
        ]
    )

    job_summary[
        "p90_absolute_mae_units"
    ] = (
        job_summary[
            "p90_abs_mae_units"
        ]
    )

    job_summary[
        "share_abs_ge_1_mae_pct"
    ] = (
        job_summary[
            "share_abs_ge_1_0_mae_pct"
        ]
    )

    job_summary[
        "share_abs_ge_2_mae_pct"
    ] = (
        job_summary[
            "share_abs_ge_2_0_mae_pct"
        ]
    )

    job_summary = job_summary[
        [
            "metric",
            "horizon",
            "pre_cp_reference_mae",
            "reference_validation_periods",
            "reference_validation_rows",
            "rows",
            "median_absolute_mae_units",
            "p90_absolute_mae_units",
            "share_abs_ge_1_mae_pct",
            "share_abs_ge_2_mae_pct",
            "p50_abs_mae_units",
            "p75_abs_mae_units",
            "p90_abs_mae_units",
            "p95_abs_mae_units",
            "p99_abs_mae_units",
            "share_abs_ge_0_5_mae_pct",
            "share_abs_ge_1_0_mae_pct",
            "share_abs_ge_2_0_mae_pct",
            "share_abs_ge_3_0_mae_pct",
            "overflow_ge_8_mae_pct",
            "support_ok",
        ]
    ].copy()

    return (
        histogram,
        job_summary,
        slice_scout,
        diagnostics,
    )


# ---------------------------------------------------------------------
# Sort and normalize outputs
# ---------------------------------------------------------------------


def finalize_outputs(
    histogram: pd.DataFrame,
    job_summary: pd.DataFrame,
    slice_scout: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Apply deterministic output ordering and stable dtypes."""
    metric_rank = {
        metric: index
        for index, metric in enumerate(
            METRICS
        )
    }

    geography_rank = {
        geography: index
        for index, geography in enumerate(
            GEOGRAPHY_TYPES
        )
    }

    day_type_rank = {
        value: index
        for index, value in enumerate(
            DAY_TYPES
        )
    }

    daypart_rank = {
        value: index
        for index, value in enumerate(
            DAYPARTS
        )
    }

    for frame in [
        histogram,
        job_summary,
        slice_scout,
    ]:
        frame["horizon"] = pd.to_numeric(
            frame["horizon"],
            errors="raise",
        ).astype(int)

    histogram["_metric_rank"] = (
        histogram["metric"]
        .map(metric_rank)
    )

    histogram = (
        histogram.sort_values(
            [
                "_metric_rank",
                "horizon",
                "overflow",
                "bin_mid",
            ],
            kind="stable",
        )
        .drop(
            columns=[
                "_metric_rank",
            ]
        )
        .reset_index(drop=True)
    )

    job_summary["_metric_rank"] = (
        job_summary["metric"]
        .map(metric_rank)
    )

    job_summary = (
        job_summary.sort_values(
            [
                "_metric_rank",
                "horizon",
            ],
            kind="stable",
        )
        .drop(
            columns=[
                "_metric_rank",
            ]
        )
        .reset_index(drop=True)
    )

    slice_scout["_metric_rank"] = (
        slice_scout["metric"]
        .map(metric_rank)
    )

    slice_scout["_geography_rank"] = (
        slice_scout[
            "geography_type"
        ].map(
            geography_rank
        )
    )

    slice_scout["_day_type_rank"] = (
        slice_scout[
            "day_type"
        ].map(
            day_type_rank
        )
    )

    slice_scout["_daypart_rank"] = (
        slice_scout[
            "daypart"
        ].map(
            daypart_rank
        )
    )

    slice_scout = (
        slice_scout.sort_values(
            [
                "_geography_rank",
                "geography_label",
                "_day_type_rank",
                "_daypart_rank",
                "_metric_rank",
                "horizon",
            ],
            kind="stable",
            na_position="last",
        )
        .drop(
            columns=[
                "_metric_rank",
                "_geography_rank",
                "_day_type_rank",
                "_daypart_rank",
            ]
        )
        .reset_index(drop=True)
    )

    return (
        histogram,
        job_summary,
        slice_scout,
    )


# ---------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------


def build_qa(
    *,
    reference: pd.DataFrame,
    histogram: pd.DataFrame,
    job_summary: pd.DataFrame,
    slice_scout: pd.DataFrame,
    diagnostics: dict[str, int],
) -> pd.DataFrame:
    """Validate the corrected Raw 25 runtime before publication."""
    qa_rows = []

    def add_check(
        check_id: str,
        passed: bool,
        details: str,
    ) -> None:
        qa_rows.append(
            {
                "check_id": check_id,
                "status": (
                    "PASS"
                    if bool(passed)
                    else "FAIL"
                ),
                "details": details,
            }
        )

    expected_jobs = {
        (metric, horizon)
        for metric in METRICS
        for horizon in HORIZONS
    }

    # -------------------------------------------------------------
    # Reference contract
    # -------------------------------------------------------------

    reference_jobs = set(
        reference[
            [
                "metric",
                "horizon",
            ]
        ]
        .drop_duplicates()
        .itertuples(
            index=False,
            name=None,
        )
    )

    add_check(
        "reference_lookup_preserves_15_jobs",
        reference_jobs == expected_jobs,
        (
            f"observed_jobs={len(reference_jobs)}; "
            "expected_jobs=15"
        ),
    )

    supported_reference = reference.loc[
        reference["support_ok"]
    ].copy()

    supported_mae = pd.to_numeric(
        supported_reference[
            "pre_cp_reference_mae"
        ],
        errors="coerce",
    )

    supported_mae_valid = (
        np.isfinite(supported_mae)
        & supported_mae.gt(0)
    )

    add_check(
        "supported_reference_mae_is_positive",
        bool(
            supported_mae_valid.all()
        ),
        (
            f"valid={int(supported_mae_valid.sum()):,}; "
            f"supported={len(supported_reference):,}"
        ),
    )

    # -------------------------------------------------------------
    # Slice scout
    # -------------------------------------------------------------

    duplicate_slices = int(
        slice_scout.duplicated(
            SLICE_KEY
        ).sum()
    )

    add_check(
        "slice_scout_key_unique",
        duplicate_slices == 0,
        f"duplicate_rows={duplicate_slices:,}",
    )

    slice_metrics = set(
        slice_scout[
            "metric"
        ]
        .astype(str)
        .unique()
    )

    add_check(
        "slice_scout_has_all_metrics",
        slice_metrics == set(METRICS),
        f"metrics={sorted(slice_metrics)}",
    )

    slice_horizons = set(
        pd.to_numeric(
            slice_scout[
                "horizon"
            ],
            errors="raise",
        )
        .astype(int)
        .unique()
        .tolist()
    )

    add_check(
        "slice_scout_has_horizons_1_2_5",
        slice_horizons == set(HORIZONS),
        f"horizons={sorted(slice_horizons)}",
    )

    slice_geographies = set(
        slice_scout[
            "geography_type"
        ]
        .astype(str)
        .unique()
    )

    add_check(
        "slice_scout_has_all_geography_levels",
        slice_geographies == set(
            GEOGRAPHY_TYPES
        ),
        (
            "geography_types="
            f"{sorted(slice_geographies)}"
        ),
    )

    systemwide_grid = slice_scout.loc[
        slice_scout[
            "geography_type"
        ].eq("Systemwide")
        & slice_scout[
            "geography_id"
        ].astype(str).eq(
            "systemwide"
        )
    ].copy()

    expected_systemwide = {
        (
            day_type,
            daypart,
            metric,
            horizon,
        )
        for day_type in DAY_TYPES
        for daypart in DAYPARTS
        for metric in METRICS
        for horizon in HORIZONS
    }

    observed_systemwide = set(
        systemwide_grid[
            [
                "day_type",
                "daypart",
                "metric",
                "horizon",
            ]
        ]
        .itertuples(
            index=False,
            name=None,
        )
    )

    add_check(
        "systemwide_control_grid_complete",
        (
            observed_systemwide
            == expected_systemwide
        ),
        (
            f"observed={len(observed_systemwide)}; "
            f"expected={len(expected_systemwide)}"
        ),
    )

    statistic_columns = [
        "p50_abs_mae_units",
        "p75_abs_mae_units",
        "p90_abs_mae_units",
        "p95_abs_mae_units",
        "p99_abs_mae_units",
        "share_abs_ge_0_5_mae_pct",
        "share_abs_ge_1_0_mae_pct",
        "share_abs_ge_2_0_mae_pct",
        "share_abs_ge_3_0_mae_pct",
    ]

    finite_statistics = np.ones(
        len(slice_scout),
        dtype=bool,
    )

    for column in statistic_columns:
        values = pd.to_numeric(
            slice_scout[column],
            errors="coerce",
        )

        finite_statistics &= (
            np.isfinite(values)
        )

    add_check(
        "slice_statistics_are_finite",
        bool(
            finite_statistics.all()
        ),
        (
            f"valid={int(finite_statistics.sum()):,}; "
            f"rows={len(slice_scout):,}"
        ),
    )

    support_expected = (
        slice_scout[
            "reference_support_ok"
        ]
        .fillna(False)
        .astype(bool)
        & pd.to_numeric(
            slice_scout["rows"],
            errors="raise",
        ).ge(
            MIN_POST_CP_PERIODS
        )
    )

    support_matches = (
        slice_scout[
            "support_ok"
        ]
        .fillna(False)
        .astype(bool)
        .eq(
            support_expected
        )
        .all()
    )

    add_check(
        "slice_support_rule_is_consistent",
        bool(support_matches),
        (
            "support_ok = reference_support_ok "
            f"AND post_cp_periods >= {MIN_POST_CP_PERIODS}"
        ),
    )

    taxi_zone_ids = set(
        slice_scout.loc[
            slice_scout[
                "geography_type"
            ].eq("Taxi Zone"),
            "geography_id",
        ]
        .astype(str)
        .unique()
    )

    forbidden_zone_ids = {
        "57",
        "105",
        "264",
        "265",
    }

    exposed_forbidden = sorted(
        taxi_zone_ids
        & forbidden_zone_ids
    )

    add_check(
        "taxi_zone_output_uses_physical_zones",
        not exposed_forbidden,
        f"forbidden_ids={exposed_forbidden}",
    )

    # -------------------------------------------------------------
    # Hero summary
    # -------------------------------------------------------------

    summary_jobs = set(
        job_summary[
            [
                "metric",
                "horizon",
            ]
        ]
        .itertuples(
            index=False,
            name=None,
        )
    )

    add_check(
        "hero_summary_has_15_jobs",
        (
            len(job_summary) == 15
            and summary_jobs == expected_jobs
        ),
        (
            f"rows={len(job_summary)}; "
            f"jobs={len(summary_jobs)}"
        ),
    )

    hero_mae = pd.to_numeric(
        job_summary[
            "pre_cp_reference_mae"
        ],
        errors="coerce",
    )

    add_check(
        "hero_reference_mae_is_positive",
        bool(
            (
                np.isfinite(hero_mae)
                & hero_mae.gt(0)
            ).all()
        ),
        (
            f"valid={int((np.isfinite(hero_mae) & hero_mae.gt(0)).sum())}; "
            f"rows={len(job_summary)}"
        ),
    )

    add_check(
        "hero_jobs_are_supported",
        bool(
            job_summary[
                "support_ok"
            ]
            .fillna(False)
            .astype(bool)
            .all()
        ),
        (
            "supported="
            f"{int(job_summary['support_ok'].sum())}/"
            f"{len(job_summary)}"
        ),
    )

    # -------------------------------------------------------------
    # Histogram
    # -------------------------------------------------------------

    histogram_jobs = set(
        histogram[
            [
                "metric",
                "horizon",
            ]
        ]
        .drop_duplicates()
        .itertuples(
            index=False,
            name=None,
        )
    )

    add_check(
        "histogram_has_15_jobs",
        histogram_jobs == expected_jobs,
        f"jobs={len(histogram_jobs)}",
    )

    density_totals = (
        histogram.groupby(
            [
                "metric",
                "horizon",
            ],
            observed=True,
        )[
            "density_pct"
        ]
        .sum()
    )

    density_ok = np.allclose(
        density_totals.to_numpy(),
        100.0,
        atol=1e-6,
    )

    add_check(
        "histogram_density_sums_to_100",
        bool(density_ok),
        (
            f"min={density_totals.min():.6f}; "
            f"max={density_totals.max():.6f}"
        ),
    )

    # -------------------------------------------------------------
    # Build diagnostics
    # -------------------------------------------------------------

    add_check(
        "post_cp_source_rows_loaded",
        diagnostics[
            "raw_source_rows"
        ] > 0,
        (
            "raw_source_rows="
            f"{diagnostics['raw_source_rows']:,}"
        ),
    )

    # Narrow geography/time slices can legitimately exist post-CP while lacking
    # a usable Pre-CP denominator. They are not reader-facing supported slices.
    # Systemwide completeness above is the strict structural requirement.
    add_check(
        "calibrated_post_cp_periods_produced",
        len(slice_scout) > 0,
        (
            f"slice_rows={len(slice_scout):,}; "
            "expanded_period_rows="
            f"{diagnostics['expanded_post_cp_period_rows']:,}; "
            "unmatched_reference_period_rows="
            f"{diagnostics['unmatched_reference_period_rows']:,}"
        ),
    )

    return pd.DataFrame(
        qa_rows
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    """Build, validate, and publish the corrected Raw 25 runtime."""
    print("=" * 78)
    print("RAW 25 SLICE-SPECIFIC REFERENCE CALIBRATION RUNTIME BUILD")
    print("=" * 78)

    require_file(
        GAP_SURFACE_PATH,
        "Counterfactual gap surface",
    )

    require_parquet_columns(
        GAP_SURFACE_PATH,
        GAP_SOURCE_COLUMNS,
        "Counterfactual gap surface",
    )

    print(
        f"Gap source:\n  {GAP_SURFACE_PATH}"
    )

    print(
        f"Reference MAE:\n  {REFERENCE_MAE_PATH}"
    )

    print(
        f"Zone context:\n  {CONTEXT_PATH}"
    )

    print()
    print(
        "Checking Reference-MAE QA...",
        flush=True,
    )

    reference_qa = (
        validate_reference_qa()
    )

    print(
        f"  PASS — {len(reference_qa):,} Reference-MAE QA checks"
    )

    reference = (
        load_reference_lookup()
    )

    zone_context = (
        load_zone_context()
    )

    print()
    print(
        "Reference lookup:"
    )
    print(
        f"  rows: {len(reference):,}"
    )
    print(
        "  supported slices: "
        f"{int(reference['support_ok'].sum()):,} / "
        f"{len(reference):,}"
    )
    print(
        f"  canonical context zones: {len(zone_context):,}"
    )

    print()
    print(
        "Rebuilding post-CP Raw 25 calibration...",
        flush=True,
    )

    (
        histogram,
        job_summary,
        slice_scout,
        diagnostics,
    ) = build_runtime(
        reference,
        zone_context,
    )

    (
        histogram,
        job_summary,
        slice_scout,
    ) = finalize_outputs(
        histogram,
        job_summary,
        slice_scout,
    )

    qa = build_qa(
        reference=reference,
        histogram=histogram,
        job_summary=job_summary,
        slice_scout=slice_scout,
        diagnostics=diagnostics,
    )

    print()
    print("RAW 25 RUNTIME QA")
    print(
        qa.to_string(
            index=False
        )
    )

    failures = qa.loc[
        qa["status"].ne("PASS")
    ]

    if not failures.empty:
        print()
        print(
            "BUILD NOT PUBLISHED — existing Raw 25 serving files "
            "were left untouched."
        )

        raise RuntimeError(
            "Raw 25 runtime failed QA:\n\n"
            + failures.to_string(
                index=False
            )
        )

    # -----------------------------------------------------------------
    # Stage all four outputs before replacing any existing serving file.
    # -----------------------------------------------------------------

    staged: dict[Path, Path] = {}

    try:
        staged[
            JOB_HISTOGRAM_PATH
        ] = atomic_stage_parquet(
            histogram,
            JOB_HISTOGRAM_PATH,
        )

        staged[
            JOB_SUMMARY_PATH
        ] = atomic_stage_parquet(
            job_summary,
            JOB_SUMMARY_PATH,
        )

        staged[
            SLICE_SCOUT_PATH
        ] = atomic_stage_parquet(
            slice_scout,
            SLICE_SCOUT_PATH,
        )

        staged[
            QA_PATH
        ] = atomic_stage_parquet(
            qa,
            QA_PATH,
        )

        publish_staged_artifacts(
            staged
        )

    except Exception:
        cleanup_staged_artifacts(
            staged
        )

        raise

    supported_slices = int(
        slice_scout[
            "support_ok"
        ].sum()
    )

    print()
    print("Runtime outputs:")
    print(
        f"  job histogram: {len(histogram):,} rows"
    )
    print(
        f"  job summary: {len(job_summary):,} rows"
    )
    print(
        f"  slice scout: {len(slice_scout):,} rows"
    )
    print(
        "  supported slices: "
        f"{supported_slices:,} / {len(slice_scout):,}"
    )

    print()
    print("Hero denominator:")
    hero_denominator = (
        job_summary[
            [
                "metric",
                "horizon",
                "pre_cp_reference_mae",
                "reference_validation_periods",
                "rows",
                "p50_abs_mae_units",
                "p90_abs_mae_units",
                "share_abs_ge_1_0_mae_pct",
                "share_abs_ge_2_0_mae_pct",
            ]
        ]
    )

    print(
        hero_denominator.to_string(
            index=False
        )
    )

    print()
    print("QA: PASS")

    print()
    print("Published:")
    print(
        f"  {JOB_HISTOGRAM_PATH}"
    )
    print(
        f"  {JOB_SUMMARY_PATH}"
    )
    print(
        f"  {SLICE_SCOUT_PATH}"
    )
    print(
        f"  {QA_PATH}"
    )


if __name__ == "__main__":
    main()