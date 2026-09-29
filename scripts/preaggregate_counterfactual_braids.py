"""
Preaggregate counterfactual timelines for the Streamlit Showcase.

This is a one-time local preprocessing step against data that is ALREADY in git.
It does not rerun Deepnote, fit models, or regenerate counterfactual forecasts.

Input
-----
data/processed/5.3.1.final_tables/
    counterfactual_horizon_stability.parquet

Output
------
data/processed/5.3.1.final_tables/
    counterfactual_braid_summary.parquet
    counterfactual_braid_summary_qa.parquet
    counterfactual_braid_temporal_explorer.parquet
    counterfactual_braid_temporal_explorer_qa.parquet

The source horizon-stability surface is already aligned so h=1, h=2, and h=5
refer to the same future target state. This script keeps only exact common
three-horizon support, then creates compact weekly/monthly/time-of-week summaries
for Systemwide, Borough, Policy geography, Mobility environment, and Taxi Zone.

Counts/ridership are summed. Average speeds use the same world-specific
activity-weighted aggregation contract as notebook 5.2.1.
"""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Frozen Chapter 5 contracts
# ---------------------------------------------------------------------

CP_START = pd.Timestamp("2025-01-05")
CP_END = pd.Timestamp("2026-03-31")

HORIZONS = [1, 2, 5]

COUNT_METRICS = {
    "taxi_trip_count",
    "fhvhv_trip_count",
    "subway_ridership",
}

SPEED_METRICS = {
    "taxi_avg_trip_speed",
    "fhvhv_avg_trip_speed",
}

METRICS = [
    "taxi_trip_count",
    "taxi_avg_trip_speed",
    "fhvhv_trip_count",
    "fhvhv_avg_trip_speed",
    "subway_ridership",
]

SOURCE_NAME = "counterfactual_horizon_stability.parquet"

OUTPUT_NAME = "counterfactual_braid_summary.parquet"
QA_NAME = "counterfactual_braid_summary_qa.parquet"

TEMPORAL_EXPLORER_NAME = "counterfactual_braid_temporal_explorer.parquet"
TEMPORAL_EXPLORER_QA_NAME = "counterfactual_braid_temporal_explorer_qa.parquet"

REQUIRED_COLUMNS = [
    "taxi_zone_id",
    "zone",
    "borough",
    "metric",
    "target_observation_sequence_id",
    "target_date",
    "target_month",
    "target_temporal_bucket",
    "primary_cross_regime",
    "cbd_spatial_category",
    "pre_cp_mobility_environment",
    "target_observed_value",
    "target_observed_available",
    "observed_activity_weight",
    "no_cp_h1",
    "no_cp_h2",
    "no_cp_h5",
    "no_cp_activity_weight_h1",
    "no_cp_activity_weight_h2",
    "no_cp_activity_weight_h5",
    "all_horizons_available",
]

READ_COLUMNS = REQUIRED_COLUMNS


# ---------------------------------------------------------------------
# Paths and atomic output
# ---------------------------------------------------------------------


def find_repo_root(explicit_root: str | None) -> Path:
    """Locate the Showcase repo without hard-coding a developer-specific path."""
    if explicit_root:
        root = Path(explicit_root).expanduser().resolve()
        expected = root / "data" / "processed" / "5.3.1.final_tables"

        if not expected.exists():
            raise FileNotFoundError(
                f"--repo-root does not contain `{expected.relative_to(root)}`."
            )

        return root

    starts = [
        Path.cwd().resolve(),
        Path(__file__).resolve().parent,
    ]

    for start in starts:
        for candidate in [start, *start.parents]:
            final_dir = (
                candidate
                / "data"
                / "processed"
                / "5.3.1.final_tables"
            )

            if final_dir.exists():
                return candidate

    raise FileNotFoundError(
        "Could not locate the Showcase repo. Run this script from inside the "
        "repo or pass `--repo-root`."
    )


def resolve_source(final_dir: Path) -> Path:
    """Return the canonical aligned horizon-stability surface."""
    source_path = final_dir / SOURCE_NAME

    if not source_path.exists():
        raise FileNotFoundError(
            "Could not find the aligned horizon-stability surface:\n"
            f"  {source_path}"
        )

    return source_path


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Never leave a half-written parquet looking like a valid artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_name(
        f".{path.stem}.{uuid.uuid4().hex[:8]}.tmp.parquet"
    )

    try:
        frame.to_parquet(
            temp_path,
            index=False,
            compression="zstd",
        )
        os.replace(temp_path, path)

    finally:
        if temp_path.exists():
            temp_path.unlink()


# ---------------------------------------------------------------------
# Source validation
# ---------------------------------------------------------------------


def require_columns(frame: pd.DataFrame) -> None:
    """Stop immediately if the frozen aligned surface changes schema."""
    missing = sorted(
        set(REQUIRED_COLUMNS)
        - set(frame.columns)
    )

    if missing:
        raise KeyError(
            "Aligned horizon-stability surface is missing required columns: "
            + ", ".join(missing)
        )


def load_common_support(source_path: Path) -> pd.DataFrame:
    """
    Load only the fields needed by the Showcase and retain exact h=1/2/5 support.

    `all_horizons_available` is produced upstream by 5.2.1 after aligning
    h=1, h=2, and h=5 to the same forecast sequence, Taxi Zone, metric, and
    target observation. That makes this the correct source for a Horizon Braid.
    """
    print(f"Reading: {source_path}")
    frame = pd.read_parquet(
        source_path,
        columns=READ_COLUMNS,
    )

    require_columns(frame)

    frame = frame.copy()

    frame["target_date"] = pd.to_datetime(
        frame["target_date"],
        errors="raise",
    ).dt.normalize()

    frame["target_month"] = pd.to_datetime(
        frame["target_month"],
        errors="raise",
    ).dt.to_period("M").dt.to_timestamp()

    frame["taxi_zone_id"] = pd.to_numeric(
        frame["taxi_zone_id"],
        errors="raise",
    ).astype("Int64")

    frame["target_observed_value"] = pd.to_numeric(
        frame["target_observed_value"],
        errors="coerce",
    )

    for horizon in HORIZONS:
        frame[f"no_cp_h{horizon}"] = pd.to_numeric(
            frame[f"no_cp_h{horizon}"],
            errors="coerce",
        )

    eligible = (
        frame["primary_cross_regime"].fillna(False).astype(bool)
        & frame["target_observed_available"].fillna(False).astype(bool)
        & frame["all_horizons_available"].fillna(False).astype(bool)
        & frame["metric"].isin(METRICS)
        & frame["target_date"].between(
            CP_START,
            CP_END,
            inclusive="both",
        )
        & np.isfinite(frame["target_observed_value"])
    )

    for horizon in HORIZONS:
        eligible &= np.isfinite(
            frame[f"no_cp_h{horizon}"]
        )

    frame = frame.loc[eligible].copy()

    if frame.empty:
        raise RuntimeError(
            "No exact common-support post-CP rows remain after filtering."
        )

    duplicated = frame.duplicated(
        [
            "taxi_zone_id",
            "metric",
            "target_observation_sequence_id",
        ]
    )

    if duplicated.any():
        raise RuntimeError(
            "Common-support horizon surface contains duplicate aligned target "
            f"states: {int(duplicated.sum()):,} duplicates."
        )

    # Seven-day policy-relative weeks start exactly on Jan 5, 2025.
    day_offset = (
        frame["target_date"]
        - CP_START
    ).dt.days

    frame["week_start"] = (
        CP_START
        + pd.to_timedelta(
            (day_offset // 7) * 7,
            unit="D",
        )
    )

    print(
        "Common-support rows: "
        f"{len(frame):,} "
        f"({frame['metric'].nunique()} metrics, "
        f"{frame['taxi_zone_id'].nunique()} Taxi Zones)"
    )

    return frame


# ---------------------------------------------------------------------
# Geography lenses
# ---------------------------------------------------------------------


def geography_views(frame: pd.DataFrame):
    """
    Yield the same rows under each Showcase geography lens.

    Raw 21 can use Systemwide / Policy geography / Borough small multiples.
    Mobility environment and Taxi Zone cost little after this one-time pass and
    give the later explorer useful granularity without reading the large source.
    """
    yield (
        "Systemwide",
        frame.assign(
            geography_type="Systemwide",
            geography_value="NYC supported system",
            geography_taxi_zone_id=pd.NA,
        ),
    )

    geography_specs = [
        ("Borough", "borough"),
        ("Policy geography", "cbd_spatial_category"),
        (
            "Mobility environment",
            "pre_cp_mobility_environment",
        ),
    ]

    for geography_type, source_column in geography_specs:
        scoped = frame.loc[
            frame[source_column].notna()
        ].copy()

        scoped["geography_type"] = geography_type
        scoped["geography_value"] = (
            scoped[source_column]
            .astype(str)
        )
        scoped["geography_taxi_zone_id"] = pd.NA

        yield geography_type, scoped

    zone_frame = frame.loc[
        frame["taxi_zone_id"].notna()
        & frame["zone"].notna()
    ].copy()

    zone_frame["geography_type"] = "Taxi Zone"
    zone_frame["geography_value"] = (
        zone_frame["zone"].astype(str)
    )
    zone_frame["geography_taxi_zone_id"] = (
        zone_frame["taxi_zone_id"]
        .astype("Int64")
    )

    yield "Taxi Zone", zone_frame


# ---------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------


def _period_complete(
    grain: str,
    period_start: pd.Timestamp | pd.NaT,
) -> bool:
    """Flag edge periods that do not span their full natural display interval."""
    if pd.isna(period_start):
        return True

    period_start = pd.Timestamp(period_start)

    if grain == "week":
        return (
            period_start >= CP_START
            and period_start + pd.Timedelta(days=6) <= CP_END
        )

    if grain == "month":
        natural_start = period_start.to_period("M").start_time
        natural_end = period_start.to_period("M").end_time.normalize()

        return (
            natural_start >= CP_START
            and natural_end <= CP_END
        )

    return True


def aggregate_count_metric(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Sum trips/riders over the exact same common-support rows."""
    named_aggs = {
        "support_rows": (
            "target_observed_value",
            "size",
        ),
        "observed_level": (
            "target_observed_value",
            "sum",
        ),
    }

    for horizon in HORIZONS:
        named_aggs[f"no_cp_level_h{horizon}"] = (
            f"no_cp_h{horizon}",
            "sum",
        )

    return (
        frame.groupby(
            group_columns,
            observed=True,
            dropna=False,
            sort=False,
        )
        .agg(**named_aggs)
        .reset_index()
    )


def aggregate_speed_metric(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """
    Compute activity-weighted average speed for observed and each no-CP world.

    The common row population is shared across horizons. Each world keeps its
    own trip-activity weight because observed and no-CP represent different
    mobility worlds.
    """
    weight_columns = [
        "observed_activity_weight",
        "no_cp_activity_weight_h1",
        "no_cp_activity_weight_h2",
        "no_cp_activity_weight_h5",
    ]

    scoped = frame.copy()

    for column in weight_columns:
        scoped[column] = pd.to_numeric(
            scoped[column],
            errors="coerce",
        )

    valid_weights = scoped[
        weight_columns
    ].notna().all(axis=1)

    if (
        scoped.loc[
            valid_weights,
            weight_columns,
        ] < 0
    ).any(axis=None):
        raise RuntimeError(
            "Negative activity weights found in the aligned speed surface."
        )

    scoped = scoped.loc[
        valid_weights
    ].copy()

    if scoped.empty:
        return pd.DataFrame()

    scoped["observed_numerator"] = (
        scoped["target_observed_value"]
        * scoped["observed_activity_weight"]
    )

    for horizon in HORIZONS:
        scoped[f"no_cp_numerator_h{horizon}"] = (
            scoped[f"no_cp_h{horizon}"]
            * scoped[
                f"no_cp_activity_weight_h{horizon}"
            ]
        )

    named_aggs = {
        "support_rows": (
            "target_observed_value",
            "size",
        ),
        "observed_numerator": (
            "observed_numerator",
            "sum",
        ),
        "observed_denominator": (
            "observed_activity_weight",
            "sum",
        ),
    }

    for horizon in HORIZONS:
        named_aggs[
            f"no_cp_numerator_h{horizon}"
        ] = (
            f"no_cp_numerator_h{horizon}",
            "sum",
        )
        named_aggs[
            f"no_cp_denominator_h{horizon}"
        ] = (
            f"no_cp_activity_weight_h{horizon}",
            "sum",
        )

    summary = (
        scoped.groupby(
            group_columns,
            observed=True,
            dropna=False,
            sort=False,
        )
        .agg(**named_aggs)
        .reset_index()
    )

    summary["observed_level"] = (
        summary["observed_numerator"]
        / summary[
            "observed_denominator"
        ].where(
            summary["observed_denominator"] > 0
        )
    )

    for horizon in HORIZONS:
        summary[f"no_cp_level_h{horizon}"] = (
            summary[f"no_cp_numerator_h{horizon}"]
            / summary[
                f"no_cp_denominator_h{horizon}"
            ].where(
                summary[
                    f"no_cp_denominator_h{horizon}"
                ] > 0
            )
        )

    drop_columns = [
        "observed_numerator",
        "observed_denominator",
    ]

    for horizon in HORIZONS:
        drop_columns.extend(
            [
                f"no_cp_numerator_h{horizon}",
                f"no_cp_denominator_h{horizon}",
            ]
        )

    return summary.drop(
        columns=drop_columns
    )


def wide_to_long(
    summary: pd.DataFrame,
    *,
    summary_grain: str,
) -> pd.DataFrame:
    """Convert one observed level + three no-CP levels into app-friendly long form."""
    parts = []

    for horizon in HORIZONS:
        part = summary[
            [
                column
                for column in summary.columns
                if not column.startswith("no_cp_level_h")
            ]
        ].copy()

        part["horizon"] = horizon
        part["no_cp_level"] = summary[
            f"no_cp_level_h{horizon}"
        ].to_numpy()

        parts.append(part)

    result = pd.concat(
        parts,
        ignore_index=True,
    )

    result["summary_grain"] = summary_grain
    result["counterfactual_gap"] = (
        result["no_cp_level"]
        - result["observed_level"]
    )

    result["counterfactual_gap_pct"] = np.where(
        result["observed_level"].ne(0),
        100
        * result["counterfactual_gap"]
        / result["observed_level"],
        np.nan,
    )

    return result


def aggregate_one_grain(
    frame: pd.DataFrame,
    *,
    summary_grain: str,
    time_columns: list[str],
) -> pd.DataFrame:
    """Aggregate every geography × metric for one display grain."""
    output_parts = []

    for geography_name, geography_frame in geography_views(frame):
        print(
            f"  {summary_grain:15s} | "
            f"{geography_name}"
        )

        base_group_columns = [
            "geography_type",
            "geography_value",
            "geography_taxi_zone_id",
            *time_columns,
        ]

        for metric in METRICS:
            metric_frame = geography_frame.loc[
                geography_frame[
                    "metric"
                ].eq(metric)
            ].copy()

            if metric_frame.empty:
                continue

            group_columns = [
                *base_group_columns,
                "metric",
            ]

            if metric in COUNT_METRICS:
                summary = aggregate_count_metric(
                    metric_frame,
                    group_columns,
                )
            else:
                summary = aggregate_speed_metric(
                    metric_frame,
                    group_columns,
                )

            if summary.empty:
                continue

            output_parts.append(
                wide_to_long(
                    summary,
                    summary_grain=summary_grain,
                )
            )

    if not output_parts:
        raise RuntimeError(
            f"No rows were produced for `{summary_grain}`."
        )

    return pd.concat(
        output_parts,
        ignore_index=True,
    )


# ---------------------------------------------------------------------
# Build compact Showcase artifact
# ---------------------------------------------------------------------


def build_braid_summary(
    source: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build three complementary display grains.

    week:
        Signature Horizon Braid. Weeks are policy-relative seven-day blocks
        beginning Sunday Jan 5, 2025.

    month:
        Slower editorial trend and QA context.

    temporal_bucket:
        Full-period weekday/weekend × daypart glyphs.
    """
    print("Aggregating compact Showcase views...")

    weekly = aggregate_one_grain(
        source,
        summary_grain="week",
        time_columns=["week_start"],
    )

    monthly = aggregate_one_grain(
        source,
        summary_grain="month",
        time_columns=["target_month"],
    )

    temporal_bucket = aggregate_one_grain(
        source,
        summary_grain="temporal_bucket",
        time_columns=[
            "target_temporal_bucket"
        ],
    )

    # Give every grain the same simple temporal columns.
    weekly["period_start"] = weekly[
        "week_start"
    ]
    monthly["period_start"] = monthly[
        "target_month"
    ]
    temporal_bucket["period_start"] = pd.NaT

    weekly["period_complete"] = weekly[
        "period_start"
    ].map(
        lambda value: _period_complete(
            "week",
            value,
        )
    )

    monthly["period_complete"] = monthly[
        "period_start"
    ].map(
        lambda value: _period_complete(
            "month",
            value,
        )
    )

    temporal_bucket["period_complete"] = True

    for frame in [
        weekly,
        monthly,
        temporal_bucket,
    ]:
        if "week_start" not in frame.columns:
            frame["week_start"] = pd.NaT

        if "target_month" not in frame.columns:
            frame["target_month"] = pd.NaT

        if (
            "target_temporal_bucket"
            not in frame.columns
        ):
            frame[
                "target_temporal_bucket"
            ] = pd.NA

    result = pd.concat(
        [
            weekly,
            monthly,
            temporal_bucket,
        ],
        ignore_index=True,
    )

    result["geography_taxi_zone_id"] = (
        pd.to_numeric(
            result[
                "geography_taxi_zone_id"
            ],
            errors="coerce",
        ).astype("Int64")
    )

    column_order = [
        "summary_grain",
        "period_start",
        "period_complete",
        "week_start",
        "target_month",
        "target_temporal_bucket",
        "geography_type",
        "geography_value",
        "geography_taxi_zone_id",
        "metric",
        "horizon",
        "support_rows",
        "observed_level",
        "no_cp_level",
        "counterfactual_gap",
        "counterfactual_gap_pct",
    ]

    result = (
        result[
            column_order
        ]
        .sort_values(
            [
                "summary_grain",
                "geography_type",
                "geography_value",
                "metric",
                "period_start",
                "target_temporal_bucket",
                "horizon",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    return result


# ---------------------------------------------------------------------
# Build compact zone × week × temporal-bucket explorer artifact
# ---------------------------------------------------------------------


def build_temporal_explorer(source: pd.DataFrame) -> pd.DataFrame:
    """
    Preserve the minimum zone-level ingredients needed for Raw 21 filtering.

    WHY this stays wide:
    h=1/h=2/h=5 share the same observed target row. Keeping the three no-CP
    worlds on one row avoids tripling the zone/week/daypart metadata and lets
    Streamlit aggregate selected temporal buckets without reopening the large
    Chapter 5 source. Speed rows retain world-specific activity weights so
    borough/environment/systemwide views can be rebuilt exactly.
    """
    output_parts = []

    group_columns = [
        "week_start",
        "target_temporal_bucket",
        "taxi_zone_id",
        "zone",
        "borough",
        "cbd_spatial_category",
        "pre_cp_mobility_environment",
        "metric",
    ]

    print("Aggregating zone × week × temporal-bucket explorer rows...")

    for metric in METRICS:
        scoped = source.loc[
            source["metric"].eq(metric)
        ].copy()

        if scoped.empty:
            continue

        if metric in COUNT_METRICS:
            named_aggs = {
                "support_rows": ("target_observed_value", "size"),
                "observed_level": ("target_observed_value", "sum"),
            }

            for horizon in HORIZONS:
                named_aggs[f"no_cp_h{horizon}"] = (
                    f"no_cp_h{horizon}",
                    "sum",
                )

            summary = (
                scoped.groupby(
                    group_columns,
                    observed=True,
                    dropna=False,
                    sort=False,
                )
                .agg(**named_aggs)
                .reset_index()
            )

            summary["observed_weight"] = np.nan
            for horizon in HORIZONS:
                summary[f"no_cp_weight_h{horizon}"] = np.nan

        else:
            weight_columns = [
                "observed_activity_weight",
                *[
                    f"no_cp_activity_weight_h{horizon}"
                    for horizon in HORIZONS
                ],
            ]

            for column in weight_columns:
                scoped[column] = pd.to_numeric(
                    scoped[column],
                    errors="coerce",
                )

            valid_weights = scoped[weight_columns].notna().all(axis=1)

            if (
                scoped.loc[valid_weights, weight_columns] < 0
            ).any(axis=None):
                raise RuntimeError(
                    f"{metric}: negative activity weights found in temporal explorer source."
                )

            scoped = scoped.loc[valid_weights].copy()

            if scoped.empty:
                continue

            scoped["observed_numerator"] = (
                scoped["target_observed_value"]
                * scoped["observed_activity_weight"]
            )

            for horizon in HORIZONS:
                scoped[f"no_cp_numerator_h{horizon}"] = (
                    scoped[f"no_cp_h{horizon}"]
                    * scoped[f"no_cp_activity_weight_h{horizon}"]
                )

            named_aggs = {
                "support_rows": ("target_observed_value", "size"),
                "observed_numerator": ("observed_numerator", "sum"),
                "observed_weight": ("observed_activity_weight", "sum"),
            }

            for horizon in HORIZONS:
                named_aggs[f"no_cp_numerator_h{horizon}"] = (
                    f"no_cp_numerator_h{horizon}",
                    "sum",
                )
                named_aggs[f"no_cp_weight_h{horizon}"] = (
                    f"no_cp_activity_weight_h{horizon}",
                    "sum",
                )

            summary = (
                scoped.groupby(
                    group_columns,
                    observed=True,
                    dropna=False,
                    sort=False,
                )
                .agg(**named_aggs)
                .reset_index()
            )

            summary["observed_level"] = (
                summary["observed_numerator"]
                / summary["observed_weight"].where(
                    summary["observed_weight"].gt(0)
                )
            )

            for horizon in HORIZONS:
                summary[f"no_cp_h{horizon}"] = (
                    summary[f"no_cp_numerator_h{horizon}"]
                    / summary[f"no_cp_weight_h{horizon}"].where(
                        summary[f"no_cp_weight_h{horizon}"].gt(0)
                    )
                )

            summary = summary.drop(
                columns=[
                    "observed_numerator",
                    *[
                        f"no_cp_numerator_h{horizon}"
                        for horizon in HORIZONS
                    ],
                ]
            )

        output_parts.append(summary)

    if not output_parts:
        raise RuntimeError("No temporal explorer rows were produced.")

    result = pd.concat(output_parts, ignore_index=True)
    result["taxi_zone_id"] = pd.to_numeric(
        result["taxi_zone_id"],
        errors="raise",
    ).astype("Int64")
    result["period_complete"] = result["week_start"].map(
        lambda value: _period_complete("week", value)
    )

    column_order = [
        "week_start",
        "period_complete",
        "target_temporal_bucket",
        "taxi_zone_id",
        "zone",
        "borough",
        "cbd_spatial_category",
        "pre_cp_mobility_environment",
        "metric",
        "support_rows",
        "observed_level",
        "observed_weight",
        "no_cp_h1",
        "no_cp_weight_h1",
        "no_cp_h2",
        "no_cp_weight_h2",
        "no_cp_h5",
        "no_cp_weight_h5",
    ]

    return (
        result[column_order]
        .sort_values(
            [
                "metric",
                "taxi_zone_id",
                "week_start",
                "target_temporal_bucket",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def collapse_temporal_explorer_systemwide(
    explorer: pd.DataFrame,
) -> pd.DataFrame:
    """Rebuild all-day systemwide weekly values for exact QA against the braid."""
    output_parts = []

    for metric in METRICS:
        scoped = explorer.loc[
            explorer["metric"].eq(metric)
        ].copy()

        if scoped.empty:
            continue

        if metric in COUNT_METRICS:
            named_aggs = {
                "observed_level": ("observed_level", "sum"),
            }
            for horizon in HORIZONS:
                named_aggs[f"no_cp_h{horizon}"] = (
                    f"no_cp_h{horizon}",
                    "sum",
                )

            weekly = (
                scoped.groupby(
                    "week_start",
                    observed=True,
                    sort=False,
                )
                .agg(**named_aggs)
                .reset_index()
            )

        else:
            scoped["observed_numerator"] = (
                scoped["observed_level"]
                * scoped["observed_weight"]
            )
            for horizon in HORIZONS:
                scoped[f"no_cp_numerator_h{horizon}"] = (
                    scoped[f"no_cp_h{horizon}"]
                    * scoped[f"no_cp_weight_h{horizon}"]
                )

            named_aggs = {
                "observed_numerator": ("observed_numerator", "sum"),
                "observed_weight": ("observed_weight", "sum"),
            }
            for horizon in HORIZONS:
                named_aggs[f"no_cp_numerator_h{horizon}"] = (
                    f"no_cp_numerator_h{horizon}",
                    "sum",
                )
                named_aggs[f"no_cp_weight_h{horizon}"] = (
                    f"no_cp_weight_h{horizon}",
                    "sum",
                )

            weekly = (
                scoped.groupby(
                    "week_start",
                    observed=True,
                    sort=False,
                )
                .agg(**named_aggs)
                .reset_index()
            )

            weekly["observed_level"] = (
                weekly["observed_numerator"]
                / weekly["observed_weight"].where(
                    weekly["observed_weight"].gt(0)
                )
            )
            for horizon in HORIZONS:
                weekly[f"no_cp_h{horizon}"] = (
                    weekly[f"no_cp_numerator_h{horizon}"]
                    / weekly[f"no_cp_weight_h{horizon}"].where(
                        weekly[f"no_cp_weight_h{horizon}"].gt(0)
                    )
                )

            weekly = weekly[[
                "week_start",
                "observed_level",
                *[f"no_cp_h{horizon}" for horizon in HORIZONS],
            ]]

        weekly["metric"] = metric
        output_parts.append(weekly)

    return pd.concat(output_parts, ignore_index=True)


def build_temporal_explorer_qa(
    source: pd.DataFrame,
    explorer: pd.DataFrame,
    braid_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Prove that the new drill-down artifact reconstructs the existing braid."""
    checks = []

    def add(check_id: str, passed: bool, details: str) -> None:
        checks.append(
            {
                "check_id": check_id,
                "status": "PASS" if passed else "FAIL",
                "details": details,
            }
        )

    duplicate_key = [
        "week_start",
        "target_temporal_bucket",
        "taxi_zone_id",
        "metric",
    ]
    duplicate_count = int(explorer.duplicated(duplicate_key).sum())
    add(
        "unique_zone_week_bucket_rows",
        duplicate_count == 0,
        f"{duplicate_count:,} duplicate explorer keys.",
    )

    source_buckets = set(
        source["target_temporal_bucket"].dropna().astype(str).unique()
    )
    explorer_buckets = set(
        explorer["target_temporal_bucket"].dropna().astype(str).unique()
    )
    add(
        "temporal_bucket_coverage",
        explorer_buckets == source_buckets,
        f"Explorer buckets: {sorted(explorer_buckets)}.",
    )

    add(
        "metric_coverage",
        set(explorer["metric"].unique()) == set(METRICS),
        "All five counterfactual metrics are present.",
    )

    add(
        "taxi_zone_coverage",
        explorer["taxi_zone_id"].nunique() == source["taxi_zone_id"].nunique(),
        (
            f"Explorer zones={explorer['taxi_zone_id'].nunique():,}; "
            f"source zones={source['taxi_zone_id'].nunique():,}."
        ),
    )

    rebuilt = collapse_temporal_explorer_systemwide(explorer)
    expected = braid_summary.loc[
        braid_summary["summary_grain"].eq("week")
        & braid_summary["geography_type"].eq("Systemwide"),
        [
            "week_start",
            "metric",
            "horizon",
            "observed_level",
            "no_cp_level",
        ],
    ].copy()

    comparison_pass = True
    max_abs_difference = 0.0

    for metric in METRICS:
        for horizon in HORIZONS:
            left = rebuilt.loc[
                rebuilt["metric"].eq(metric),
                [
                    "week_start",
                    "observed_level",
                    f"no_cp_h{horizon}",
                ],
            ].rename(
                columns={
                    "observed_level": "observed_rebuilt",
                    f"no_cp_h{horizon}": "no_cp_rebuilt",
                }
            )

            right = expected.loc[
                expected["metric"].eq(metric)
                & expected["horizon"].eq(horizon),
                [
                    "week_start",
                    "observed_level",
                    "no_cp_level",
                ],
            ].rename(
                columns={
                    "observed_level": "observed_expected",
                    "no_cp_level": "no_cp_expected",
                }
            )

            merged = left.merge(
                right,
                on="week_start",
                how="outer",
                validate="one_to_one",
                indicator=True,
            )

            if not merged["_merge"].eq("both").all():
                comparison_pass = False
                continue

            observed_diff = (
                merged["observed_rebuilt"]
                - merged["observed_expected"]
            ).abs()
            no_cp_diff = (
                merged["no_cp_rebuilt"]
                - merged["no_cp_expected"]
            ).abs()

            finite_diff = pd.concat([observed_diff, no_cp_diff]).dropna()
            if not finite_diff.empty:
                max_abs_difference = max(
                    max_abs_difference,
                    float(finite_diff.max()),
                )

            comparison_pass &= bool(
                np.allclose(
                    merged["observed_rebuilt"],
                    merged["observed_expected"],
                    rtol=1e-9,
                    atol=1e-9,
                    equal_nan=True,
                )
                and np.allclose(
                    merged["no_cp_rebuilt"],
                    merged["no_cp_expected"],
                    rtol=1e-9,
                    atol=1e-9,
                    equal_nan=True,
                )
            )

    add(
        "rebuilds_existing_systemwide_weekly_braid",
        comparison_pass,
        f"Maximum absolute reconstruction difference: {max_abs_difference:.12g}.",
    )

    return pd.DataFrame(checks)


# ---------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------


def build_qa(
    source: pd.DataFrame,
    result: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the exact comparison contract before Raw 21 consumes the output."""
    checks = []

    def add(
        check_id: str,
        passed: bool,
        details: str,
    ) -> None:
        checks.append(
            {
                "check_id": check_id,
                "status": (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
                "details": details,
            }
        )

    add(
        "source_common_support_only",
        bool(
            source[
                "all_horizons_available"
            ].astype(bool).all()
        ),
        (
            f"{len(source):,} source rows all have "
            "h=1/h=2/h=5 available."
        ),
    )

    expected_horizons = set(HORIZONS)
    actual_horizons = set(
        pd.to_numeric(
            result["horizon"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
    )

    add(
        "horizon_coverage",
        actual_horizons == expected_horizons,
        (
            f"Output horizons: "
            f"{sorted(actual_horizons)}."
        ),
    )

    add(
        "metric_coverage",
        set(
            result["metric"].unique()
        ) == set(METRICS),
        (
            "Output metrics: "
            + ", ".join(
                sorted(
                    result[
                        "metric"
                    ].unique()
                )
            )
        ),
    )

    duplicate_key = [
        "summary_grain",
        "period_start",
        "target_temporal_bucket",
        "geography_type",
        "geography_value",
        "geography_taxi_zone_id",
        "metric",
        "horizon",
    ]

    duplicate_count = int(
        result.duplicated(
            duplicate_key
        ).sum()
    )

    add(
        "unique_output_rows",
        duplicate_count == 0,
        f"{duplicate_count:,} duplicate output keys.",
    )

    gap_identity = np.isclose(
        result["counterfactual_gap"],
        (
            result["no_cp_level"]
            - result["observed_level"]
        ),
        rtol=0,
        atol=1e-10,
        equal_nan=True,
    )

    add(
        "gap_identity",
        bool(gap_identity.all()),
        (
            "Every output row preserves "
            "gap = No-CP − observed."
        ),
    )

    # Because every geography/time group starts from one exact common-support
    # row population, observed mobility must be identical across horizons.
    observed_key = [
        "summary_grain",
        "period_start",
        "target_temporal_bucket",
        "geography_type",
        "geography_value",
        "geography_taxi_zone_id",
        "metric",
    ]

    observed_check = (
        result.groupby(
            observed_key,
            observed=True,
            dropna=False,
        )["observed_level"]
        .agg(["min", "max", "count"])
        .reset_index()
    )

    observed_difference = (
        observed_check["max"]
        - observed_check["min"]
    )

    observed_alignment_pass = (
        observed_check["count"].eq(3).all()
        and np.allclose(
            observed_difference.fillna(0),
            0,
            rtol=1e-10,
            atol=1e-10,
        )
    )

    add(
        "one_observed_path_per_group",
        bool(observed_alignment_pass),
        (
            "Every display group has three horizons "
            "and one identical observed aggregate."
        ),
    )

    system_weekly = result.loc[
        result[
            "summary_grain"
        ].eq("week")
        & result[
            "geography_type"
        ].eq("Systemwide")
    ]

    add(
        "systemwide_weekly_available",
        not system_weekly.empty,
        (
            f"{len(system_weekly):,} systemwide weekly rows "
            "available for Horizon Braid scouting."
        ),
    )

    january = result.loc[
        result[
            "summary_grain"
        ].eq("month")
        & result[
            "target_month"
        ].eq(pd.Timestamp("2025-01-01"))
    ]

    add(
        "january_marked_partial",
        (
            not january.empty
            and january[
                "period_complete"
            ].eq(False).all()
        ),
        (
            "January 2025 is explicitly marked partial "
            "because the counterfactual begins Jan 5."
        ),
    )

    return pd.DataFrame(checks)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preaggregate the existing aligned counterfactual horizon surface "
            "into compact Showcase-ready timeline artifacts."
        )
    )

    parser.add_argument(
        "--repo-root",
        default=None,
        help=(
            "Optional path to the Showcase repo. Usually unnecessary when "
            "running from inside the repo."
        ),
    )

    args = parser.parse_args()
    repo_root = find_repo_root(args.repo_root)

    final_dir = (
        repo_root
        / "data"
        / "processed"
        / "5.3.1.final_tables"
    )
    source_path = resolve_source(final_dir)

    output_path = final_dir / OUTPUT_NAME
    qa_path = final_dir / QA_NAME
    temporal_path = final_dir / TEMPORAL_EXPLORER_NAME
    temporal_qa_path = final_dir / TEMPORAL_EXPLORER_QA_NAME

    source = load_common_support(source_path)

    result = build_braid_summary(source)
    qa = build_qa(source, result)

    temporal_explorer = build_temporal_explorer(source)
    temporal_qa = build_temporal_explorer_qa(
        source,
        temporal_explorer,
        result,
    )

    print()
    print("BRAID SUMMARY QA")
    print(qa.to_string(index=False))
    print()
    print("TEMPORAL EXPLORER QA")
    print(temporal_qa.to_string(index=False))
    print()

    failed = pd.concat(
        [
            qa.loc[qa["status"].ne("PASS")],
            temporal_qa.loc[temporal_qa["status"].ne("PASS")],
        ],
        ignore_index=True,
    )

    if not failed.empty:
        raise RuntimeError(
            "Counterfactual braid preprocessing failed QA. "
            "No output files were written."
        )

    atomic_write_parquet(result, output_path)
    atomic_write_parquet(qa, qa_path)
    atomic_write_parquet(temporal_explorer, temporal_path)
    atomic_write_parquet(temporal_qa, temporal_qa_path)

    source_size_mb = source_path.stat().st_size / (1024 ** 2)
    output_size_mb = output_path.stat().st_size / (1024 ** 2)
    temporal_size_mb = temporal_path.stat().st_size / (1024 ** 2)

    print(f"Wrote {len(result):,} compact braid-summary rows:")
    print(f"  {output_path}")
    print(f"  {qa_path}")
    print()
    print(f"Wrote {len(temporal_explorer):,} temporal explorer rows:")
    print(f"  {temporal_path}")
    print(f"  {temporal_qa_path}")
    print()
    print(f"Source size: {source_size_mb:,.1f} MB")
    print(f"Braid summary size: {output_size_mb:,.2f} MB")
    print(f"Temporal explorer size: {temporal_size_mb:,.2f} MB")
    print()
    print(
        "Raw 21 can use the original compact braid for its hero/small multiples "
        "and the zone-level temporal explorer for geography + day-type + daypart filtering."
    )


if __name__ == "__main__":
    main()
