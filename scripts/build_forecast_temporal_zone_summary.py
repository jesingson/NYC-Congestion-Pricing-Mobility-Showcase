"""Build Raw 19's compact temporal-zone forecasting serving artifact.

Run from the repository root:

    python scripts/build_forecast_temporal_zone_summary.py

Input:
    data/processed/4.7.1.final_tables/
        showcase_forecast_record_surface.parquet

Output:
    data/processed/4.7.1.final_tables/
        showcase_forecast_temporal_zone_summary.parquet

The output preserves the exact additive sufficient statistics Raw 19 needs to
reconstruct Relative MAE, native-unit MAE, and improvement versus the Last-week
baseline for arbitrary Day type / Daypart selections.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Repository imports
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    # WHY: running a script from /scripts puts that directory on sys.path,
    # not necessarily the repository root that contains the app package.
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from app.data_access.forecasting import (  # noqa: E402
    FINAL_HOLDOUT_END_DATE,
    FINAL_HOLDOUT_START_DATE,
    FORECAST_HORIZONS,
    FORECAST_METRICS,
    FORECAST_RECORD_SURFACE_PATH,
    FORECAST_TEMPORAL_ZONE_SUMMARY_PATH,
    forecast_daypart,
    forecast_day_type,
)


# ---------------------------------------------------------------------
# Build contract
# ---------------------------------------------------------------------

SOURCE_COLUMNS = [
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
    "absolute_error",
    "benchmark_absolute_error",
    "reader_facing_zone",
]

GROUP_KEYS = [
    "metric",
    "horizon",
    "taxi_zone_id",
    "zone",
    "borough",
    "day_type",
    "daypart",
]

OUTPUT_COLUMNS = [
    *GROUP_KEYS,
    "forecast_rows",
    "observed_abs_sum",
    "absolute_error_sum",
    "benchmark_absolute_error_sum",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _require_columns(
    frame: pd.DataFrame,
    required: list[str],
    *,
    label: str,
) -> None:
    """Fail before building if an upstream schema is incomplete."""
    missing = sorted(
        set(required)
        - set(frame.columns)
    )

    if missing:
        raise RuntimeError(
            f"{label} is missing required columns: {missing}"
        )


def _normalize_source(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the exact final-holdout source contract used by Raw 19."""
    result = frame.copy()

    result["target_date"] = pd.to_datetime(
        result["target_date"],
        errors="coerce",
    )

    result["horizon"] = pd.to_numeric(
        result["horizon"],
        errors="coerce",
    ).astype("Int64")

    result["taxi_zone_id"] = pd.to_numeric(
        result["taxi_zone_id"],
        errors="coerce",
    ).astype("Int64")

    result["reader_facing_zone"] = (
        result["reader_facing_zone"]
        .fillna(False)
        .astype(bool)
    )

    result = result.loc[
        result["reader_facing_zone"]
        & result["metric"].isin(FORECAST_METRICS)
        & result["horizon"].isin(FORECAST_HORIZONS)
        & result["target_date"].between(
            FINAL_HOLDOUT_START_DATE,
            FINAL_HOLDOUT_END_DATE,
            inclusive="both",
        )
        & result["actual"].notna()
        & result["champion_prediction"].notna()
        & result["benchmark_prediction"].notna()
    ].copy()

    result["day_type"] = forecast_day_type(
        result["target_temporal_bucket"]
    )

    result["daypart"] = forecast_daypart(
        result["target_temporal_bucket"]
    )

    result["observed_abs"] = (
        pd.to_numeric(
            result["actual"],
            errors="coerce",
        )
        .abs()
    )

    return result


def build_temporal_zone_summary(
    source: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse final-holdout rows to exact additive temporal-zone statistics."""
    summary = (
        source.groupby(
            GROUP_KEYS,
            observed=True,
            as_index=False,
        )
        .agg(
            forecast_rows=(
                "actual",
                "size",
            ),
            observed_abs_sum=(
                "observed_abs",
                "sum",
            ),
            absolute_error_sum=(
                "absolute_error",
                "sum",
            ),
            benchmark_absolute_error_sum=(
                "benchmark_absolute_error",
                "sum",
            ),
        )
    )

    summary = summary[
        OUTPUT_COLUMNS
    ].sort_values(
        GROUP_KEYS,
        kind="stable",
    ).reset_index(
        drop=True
    )

    return summary


def validate_summary(
    source: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Validate uniqueness and exact additive reconciliation before writing."""
    if summary.empty:
        raise RuntimeError(
            "Temporal-zone summary is empty."
        )

    duplicates = int(
        summary.duplicated(
            GROUP_KEYS
        ).sum()
    )

    if duplicates:
        raise RuntimeError(
            "Temporal-zone summary is not unique at its canonical grain: "
            f"{duplicates:,} duplicate rows."
        )

    if summary["forecast_rows"].isna().any():
        raise RuntimeError(
            "Temporal-zone summary contains missing forecast_rows."
        )

    source_rows = int(
        len(source)
    )
    summary_rows = int(
        summary["forecast_rows"].sum()
    )

    if source_rows != summary_rows:
        raise RuntimeError(
            "Forecast-row reconciliation failed: "
            f"source={source_rows:,}, summary={summary_rows:,}."
        )

    additive_columns = {
        "observed_abs_sum": "observed_abs",
        "absolute_error_sum": "absolute_error",
        "benchmark_absolute_error_sum": "benchmark_absolute_error",
    }

    for summary_column, source_column in additive_columns.items():
        source_total = float(
            pd.to_numeric(
                source[source_column],
                errors="coerce",
            ).sum()
        )

        summary_total = float(
            pd.to_numeric(
                summary[summary_column],
                errors="coerce",
            ).sum()
        )

        if not np.isclose(
            source_total,
            summary_total,
            rtol=1e-10,
            atol=1e-8,
        ):
            raise RuntimeError(
                f"{summary_column} reconciliation failed: "
                f"source={source_total:.12g}, "
                f"summary={summary_total:.12g}."
            )

    expected_day_types = {
        "Weekdays",
        "Weekends",
    }
    observed_day_types = set(
        summary["day_type"]
        .dropna()
        .astype(str)
        .unique()
    )

    if not observed_day_types.issubset(
        expected_day_types
    ):
        raise RuntimeError(
            "Unexpected day_type values: "
            f"{sorted(observed_day_types - expected_day_types)}"
        )

    expected_dayparts = {
        "Overnight",
        "AM peak",
        "Midday",
        "PM peak",
        "Evening",
    }
    observed_dayparts = set(
        summary["daypart"]
        .dropna()
        .astype(str)
        .unique()
    )

    if not observed_dayparts.issubset(
        expected_dayparts
    ):
        raise RuntimeError(
            "Unexpected daypart values: "
            f"{sorted(observed_dayparts - expected_dayparts)}"
        )


# ---------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------

def main() -> None:
    """Build, validate, and write the compact Raw 19 serving artifact."""
    if not FORECAST_RECORD_SURFACE_PATH.exists():
        raise FileNotFoundError(
            "Forecast record surface not found: "
            f"{FORECAST_RECORD_SURFACE_PATH}"
        )

    print(
        "Reading Raw 19 final-holdout source..."
    )

    source = pd.read_parquet(
        FORECAST_RECORD_SURFACE_PATH,
        columns=SOURCE_COLUMNS,
    )

    _require_columns(
        source,
        SOURCE_COLUMNS,
        label="showcase_forecast_record_surface",
    )

    source = _normalize_source(
        source
    )

    print(
        f"Eligible source rows: {len(source):,}"
    )

    summary = build_temporal_zone_summary(
        source
    )

    validate_summary(
        source,
        summary,
    )

    FORECAST_TEMPORAL_ZONE_SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_parquet(
        FORECAST_TEMPORAL_ZONE_SUMMARY_PATH,
        index=False,
    )

    file_mb = (
        FORECAST_TEMPORAL_ZONE_SUMMARY_PATH.stat().st_size
        / (1024 ** 2)
    )

    print(
        f"Summary rows: {len(summary):,}"
    )
    print(
        "Canonical grain: "
        "Metric × Horizon × Taxi Zone × Day type × Daypart"
    )
    print(
        f"Output: {FORECAST_TEMPORAL_ZONE_SUMMARY_PATH}"
    )
    print(
        f"Output size: {file_mb:.2f} MB"
    )
    print(
        "Validation: PASS"
    )


if __name__ == "__main__":
    main()
