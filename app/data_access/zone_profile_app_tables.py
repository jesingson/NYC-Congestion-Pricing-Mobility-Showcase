"""Precomputed app tables for the Raw 07 Taxi Zone profile.

Normal app behavior
-------------------
The Streamlit page reads two purpose-built parquet datasets:

    data/processed/app_tables/zone_profile_daily_metrics/
    data/processed/app_tables/zone_profile_comparison_daily_totals/

The standalone build script is the preferred way to create or refresh them:

    python scripts/build_zone_profile_app_tables.py

The selected-zone daily dataset is partitioned by temporal bucket. The
comparison dataset is partitioned by comparison level, temporal bucket, and
metric so runtime reads touch only a small subset of rows.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from time import perf_counter

from app.data_access.aggregations import (
    aggregate_metrics,
    get_required_columns,
)
from app.data_access.loaders import (
    CORE_METRICS,
    load_analysis_panel,
)
from app.data_access.spatial_aggregations import (
    ALL_TEMPORAL_BUCKETS_LABEL,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_TABLE_DIR = REPO_ROOT / "data" / "processed" / "app_tables"

ZONE_PROFILE_DAILY_DIR = (
    APP_TABLE_DIR
    / "zone_profile_daily_metrics"
)

ZONE_PROFILE_COMPARISON_DIR = (
    APP_TABLE_DIR
    / "zone_profile_comparison_daily_totals"
)

ALL_TAXI_ZONES_GROUP = "All Taxi Zones"

ZONE_ID_COLUMN = "taxi_zone_id"

ZONE_METADATA_COLUMNS = [
    ZONE_ID_COLUMN,
    "zone",
    "borough",
    "cbd_spatial_category",
]

ZONE_DAILY_GRAIN = [
    ZONE_ID_COLUMN,
    "date",
    "temporal_bucket",
]

COMPARISON_GRAIN = [
    "comparison_level",
    "comparison_group",
    "date",
    "temporal_bucket",
    "metric",
]


def _mode_or_first(
    values: pd.Series,
) -> object:
    non_null = values.dropna()

    if non_null.empty:
        return np.nan

    modes = non_null.mode()

    if not modes.empty:
        return modes.iloc[0]

    return non_null.iloc[0]


def _build_zone_metadata(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    return (
        panel[
            ZONE_METADATA_COLUMNS
        ]
        .groupby(
            ZONE_ID_COLUMN,
            observed=True,
            dropna=False,
            sort=False,
        )
        .agg(
            zone=("zone", _mode_or_first),
            borough=("borough", _mode_or_first),
            cbd_spatial_category=(
                "cbd_spatial_category",
                _mode_or_first,
            ),
        )
        .reset_index()
    )


def _build_zone_daily_metrics(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Build Taxi Zone × date × temporal bucket plus an Overall rollup."""
    total_start = perf_counter()

    print(
        "Building stable Taxi Zone metadata...",
        flush=True,
    )
    stage_start = perf_counter()

    metadata = _build_zone_metadata(
        panel
    )

    print(
        f"Built metadata in "
        f"{perf_counter() - stage_start:,.1f} seconds.",
        flush=True,
    )

    print(
        "Preparing existing Taxi Zone × date × temporal-bucket rows...",
        flush=True,
    )
    stage_start = perf_counter()

    bucket_columns = [
        ZONE_ID_COLUMN,
        "date",
        "temporal_bucket",
        *CORE_METRICS,
    ]

    bucket_daily = panel[
        bucket_columns
    ].copy()

    duplicate_count = int(
        bucket_daily.duplicated(
            [
                ZONE_ID_COLUMN,
                "date",
                "temporal_bucket",
            ]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            "The source analysis panel contains "
            f"{duplicate_count:,} duplicate "
            "Taxi Zone × date × temporal-bucket rows."
        )

    print(
        f"Prepared bucket-level rows: "
        f"{len(bucket_daily):,} rows in "
        f"{perf_counter() - stage_start:,.1f} seconds.",
        flush=True,
    )

    print(
        "Aggregating Taxi Zone × date Overall rollup...",
        flush=True,
    )
    stage_start = perf_counter()

    overall_daily = aggregate_metrics(
        panel,
        metrics=CORE_METRICS,
        group_cols=[
            ZONE_ID_COLUMN,
            "date",
        ],
    )

    print(
        f"Built Overall daily metrics: "
        f"{len(overall_daily):,} rows in "
        f"{perf_counter() - stage_start:,.1f} seconds.",
        flush=True,
    )

    overall_daily[
        "temporal_bucket"
    ] = ALL_TEMPORAL_BUCKETS_LABEL

    print(
        "Combining bucket and Overall rows...",
        flush=True,
    )
    stage_start = perf_counter()

    zone_daily = pd.concat(
        [
            bucket_daily,
            overall_daily,
        ],
        ignore_index=True,
    )

    zone_daily = zone_daily.merge(
        metadata,
        on=ZONE_ID_COLUMN,
        how="left",
        validate="many_to_one",
    )

    zone_daily["date"] = pd.to_datetime(
        zone_daily["date"]
    )

    ordered_columns = [
        ZONE_ID_COLUMN,
        "date",
        "temporal_bucket",
        "zone",
        "borough",
        "cbd_spatial_category",
        *CORE_METRICS,
    ]

    zone_daily = (
        zone_daily[
            ordered_columns
        ]
        .sort_values(
            ZONE_DAILY_GRAIN
        )
        .reset_index(drop=True)
    )

    duplicate_count = int(
        zone_daily.duplicated(
            ZONE_DAILY_GRAIN
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            "Zone daily app table contains "
            f"{duplicate_count:,} duplicate grain rows."
        )

    print(
        f"Zone-daily build completed in "
        f"{perf_counter() - total_start:,.1f} seconds.",
        flush=True,
    )

    return zone_daily


def _summarize_comparison_group(
    metric_frame: pd.DataFrame,
    *,
    metric: str,
    comparison_level: str,
    comparison_group_column: str | None,
) -> pd.DataFrame:
    """Return daily metric sums and valid-zone counts for one comparison type."""
    working = metric_frame.copy()

    if comparison_group_column is None:
        working[
            "comparison_group"
        ] = ALL_TAXI_ZONES_GROUP
    else:
        working[
            "comparison_group"
        ] = (
            working[
                comparison_group_column
            ]
            .fillna("Unknown")
            .astype(str)
        )

    grouped = (
        working.groupby(
            [
                "comparison_group",
                "date",
                "temporal_bucket",
            ],
            observed=True,
            dropna=False,
            sort=False,
        )
        .agg(
            metric_sum=(
                metric,
                lambda values: values.sum(
                    min_count=1
                ),
            ),
            valid_zone_count=(
                metric,
                "count",
            ),
        )
        .reset_index()
    )

    grouped[
        "comparison_level"
    ] = comparison_level

    grouped["metric"] = metric

    return grouped[
        [
            "comparison_level",
            "comparison_group",
            "date",
            "temporal_bucket",
            "metric",
            "metric_sum",
            "valid_zone_count",
        ]
    ]


def _build_comparison_daily_totals(
    zone_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Build additive summaries used to exclude a zone algebraically."""
    total_start = perf_counter()
    summaries: list[pd.DataFrame] = []

    id_columns = [
        ZONE_ID_COLUMN,
        "date",
        "temporal_bucket",
        "borough",
        "cbd_spatial_category",
    ]

    for metric_number, metric in enumerate(
        CORE_METRICS,
        start=1,
    ):
        metric_start = perf_counter()

        print(
            f"Building comparison totals for "
            f"{metric_number}/{len(CORE_METRICS)}: {metric}",
            flush=True,
        )

        metric_frame = zone_daily[
            [
                *id_columns,
                metric,
            ]
        ]

        summaries.append(
            _summarize_comparison_group(
                metric_frame,
                metric=metric,
                comparison_level="Citywide",
                comparison_group_column=None,
            )
        )

        summaries.append(
            _summarize_comparison_group(
                metric_frame,
                metric=metric,
                comparison_level="Borough",
                comparison_group_column="borough",
            )
        )

        summaries.append(
            _summarize_comparison_group(
                metric_frame,
                metric=metric,
                comparison_level="Geo-policy group",
                comparison_group_column=(
                    "cbd_spatial_category"
                ),
            )
        )

        print(
            f"Completed {metric} in "
            f"{perf_counter() - metric_start:,.1f} seconds.",
            flush=True,
        )

    print(
        "Combining comparison summaries...",
        flush=True,
    )

    comparison_totals = (
        pd.concat(
            summaries,
            ignore_index=True,
        )
        .sort_values(
            COMPARISON_GRAIN
        )
        .reset_index(drop=True)
    )

    duplicate_count = int(
        comparison_totals.duplicated(
            COMPARISON_GRAIN
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            "Comparison totals contain "
            f"{duplicate_count:,} duplicate grain rows."
        )

    print(
        f"Comparison totals completed in "
        f"{perf_counter() - total_start:,.1f} seconds.",
        flush=True,
    )

    return comparison_totals


def _write_partitioned_dataset(
    frame: pd.DataFrame,
    *,
    path: Path,
    partition_columns: list[str],
) -> None:
    if path.exists():
        shutil.rmtree(
            path
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_parquet(
        path,
        engine="pyarrow",
        compression="zstd",
        index=False,
        partition_cols=partition_columns,
    )


def build_zone_profile_app_tables(
    *,
    verbose: bool = True,
) -> dict[str, object]:
    """Build and write both Raw 07 app-ready parquet datasets."""
    required_columns = sorted(
        set(
            get_required_columns(
                CORE_METRICS
            )
            + ZONE_METADATA_COLUMNS
            + [
                "date",
                "temporal_bucket",
            ]
        )
    )

    if verbose:
        print(
            "Loading analysis panel columns..."
        )

    panel = load_analysis_panel(
        columns=required_columns
    )

    if verbose:
        print(
            f"Loaded {len(panel):,} source rows."
        )

    zone_daily = _build_zone_daily_metrics(
        panel
    )

    if verbose:
        print(
            "Built zone daily metrics: "
            f"{len(zone_daily):,} rows."
        )

    comparison_totals = (
        _build_comparison_daily_totals(
            zone_daily
        )
    )

    if verbose:
        print(
            "Built comparison totals: "
            f"{len(comparison_totals):,} rows."
        )

    print(
        "Writing partitioned zone-daily dataset...",
        flush=True,
    )

    _write_partitioned_dataset(
        zone_daily,
        path=ZONE_PROFILE_DAILY_DIR,
        partition_columns=[
            "temporal_bucket",
        ],
    )

    print(
        "Finished writing zone-daily dataset.",
        flush=True,
    )

    print(
        "Writing comparison-total dataset...",
        flush=True,
    )

    _write_partitioned_dataset(
        comparison_totals,
        path=ZONE_PROFILE_COMPARISON_DIR,
        partition_columns=[
            "metric",
        ],
    )

    print(
        "Finished writing comparison-total dataset.",
        flush=True,
    )

    results = {
        "zone_daily_path": (
            ZONE_PROFILE_DAILY_DIR
        ),
        "comparison_path": (
            ZONE_PROFILE_COMPARISON_DIR
        ),
        "zone_daily_rows": len(
            zone_daily
        ),
        "comparison_rows": len(
            comparison_totals
        ),
        "zone_count": int(
            zone_daily[
                ZONE_ID_COLUMN
            ].nunique()
        ),
        "date_min": zone_daily[
            "date"
        ].min(),
        "date_max": zone_daily[
            "date"
        ].max(),
        "temporal_bucket_count": int(
            zone_daily[
                "temporal_bucket"
            ].nunique()
        ),
    }

    if verbose:
        print(
            "Zone-profile app tables written successfully."
        )

        for key, value in results.items():
            print(
                f"{key}: {value}"
            )

    return results


def ensure_zone_profile_app_tables() -> None:
    missing = [
        path
        for path in [
            ZONE_PROFILE_DAILY_DIR,
            ZONE_PROFILE_COMPARISON_DIR,
        ]
        if not path.exists()
    ]

    if missing:
        missing_text = "\n".join(
            f"- {path}"
            for path in missing
        )

        raise FileNotFoundError(
            "Raw 07 app tables are missing:\n"
            f"{missing_text}\n\n"
            "Build them from the repository root with:\n"
            "python scripts/build_zone_profile_app_tables.py"
        )


@st.cache_data(show_spinner=False)
def load_zone_daily_metric(
    *,
    taxi_zone_id: object,
    metric: str,
    temporal_bucket: str,
) -> pd.DataFrame:
    """Read only one selected zone, metric, and temporal-bucket partition."""
    ensure_zone_profile_app_tables()

    frame = pd.read_parquet(
        ZONE_PROFILE_DAILY_DIR,
        engine="pyarrow",
        columns=[
            ZONE_ID_COLUMN,
            "date",
            metric,
        ],
        filters=[
            (
                "temporal_bucket",
                "==",
                temporal_bucket,
            ),
            (
                ZONE_ID_COLUMN,
                "==",
                taxi_zone_id,
            ),
        ],
    )

    frame["date"] = pd.to_datetime(
        frame["date"]
    )

    return (
        frame.rename(
            columns={
                metric: "zone_value",
            }
        )
        .sort_values("date")
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False)
def load_comparison_daily_totals(
    *,
    comparison_level: str,
    comparison_group: str,
    metric: str,
    temporal_bucket: str,
) -> pd.DataFrame:
    """Read one compact comparison partition and group."""
    ensure_zone_profile_app_tables()

    frame = pd.read_parquet(
        ZONE_PROFILE_COMPARISON_DIR,
        engine="pyarrow",
        columns=[
            "comparison_group",
            "date",
            "metric_sum",
            "valid_zone_count",
        ],
        filters=[
            (
                "comparison_level",
                "==",
                comparison_level,
            ),
            (
                "temporal_bucket",
                "==",
                temporal_bucket,
            ),
            (
                "metric",
                "==",
                metric,
            ),
            (
                "comparison_group",
                "==",
                comparison_group,
            ),
        ],
    )

    frame["date"] = pd.to_datetime(
        frame["date"]
    )

    return (
        frame.sort_values("date")
        .reset_index(drop=True)
    )