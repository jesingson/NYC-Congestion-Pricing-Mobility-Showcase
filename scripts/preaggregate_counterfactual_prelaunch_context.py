"""Build compact prelaunch observed context for Raw 21 Horizon Braids.

RUN FROM REPO ROOT
------------------
python scripts/build_counterfactual_prelaunch_context.py


WHY THIS VERSION EXISTS
-----------------------
The aligned counterfactual horizon surface begins at the post-CP application
window, so it cannot supply observed history before Jan 5, 2025.

The prelaunch run-up therefore comes from the canonical 1.3.1 mobility panel
that already lives in the Showcase repo. This script does NOT rerun Deepnote,
forecasting, or counterfactual modeling.

INPUTS
------
data/processed/1.3.1.final_tables/
    analysis_ready_mobility_panel.parquet

data/processed/5.3.1.final_tables/
    counterfactual_braid_summary.parquet

OUTPUTS
-------
data/processed/5.3.1.final_tables/
    counterfactual_prelaunch_context.parquet
    counterfactual_prelaunch_context_qa.parquet
    counterfactual_prelaunch_temporal_explorer.parquet
    counterfactual_prelaunch_temporal_explorer_qa.parquet

DEFAULT WINDOW
--------------
52 policy-relative weeks ending Jan 4, 2025.

AGGREGATION CONTRACT
--------------------
Trips / riders:
    sum observed values.

Average speeds:
    activity-weighted average using the matching trip-count metric:
      Taxi speed  -> taxi_trip_count
      FHVHV speed -> fhvhv_trip_count

The script restricts each metric to the exact Taxi Zone × temporal-bucket
sequence population supported by all three postlaunch horizons. This keeps the
prelaunch run-up on the same broad comparison population as the braid.
"""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Frozen project contracts
# ---------------------------------------------------------------------

BUILD_VERSION = "2026-09-26-v7-temporal-explorer"

CP_START = pd.Timestamp("2025-01-05").normalize()
HISTORY_WEEKS_DEFAULT = 52

MOBILITY_PANEL_NAME = "analysis_ready_mobility_panel.parquet"
HORIZON_STABILITY_NAME = "counterfactual_horizon_stability.parquet"
CANONICAL_CLUSTER_ASSIGNMENTS_NAME = (
    "canonical_cluster_assignments-20260803-193800.parquet"
)

OUTPUT_NAME = "counterfactual_prelaunch_context.parquet"
QA_NAME = "counterfactual_prelaunch_context_qa.parquet"

TEMPORAL_EXPLORER_NAME = "counterfactual_prelaunch_temporal_explorer.parquet"
TEMPORAL_EXPLORER_QA_NAME = (
    "counterfactual_prelaunch_temporal_explorer_qa.parquet"
)

METRICS = [
    "taxi_trip_count",
    "taxi_avg_trip_speed",
    "fhvhv_trip_count",
    "fhvhv_avg_trip_speed",
    "subway_ridership",
]

COUNT_METRICS = {
    "taxi_trip_count",
    "fhvhv_trip_count",
    "subway_ridership",
}

SPEED_WEIGHT_COLUMNS = {
    "taxi_avg_trip_speed": "taxi_trip_count",
    "fhvhv_avg_trip_speed": "fhvhv_trip_count",
}

PANEL_COLUMNS = [
    "taxi_zone_id",
    "zone",
    "borough",
    "cbd_spatial_category",
    "date",
    "temporal_bucket",
    "taxi_trip_count",
    "taxi_avg_trip_speed",
    "fhvhv_trip_count",
    "fhvhv_avg_trip_speed",
    "subway_ridership",
]

HORIZON_SUPPORT_COLUMNS = [
    "taxi_zone_id",
    "metric",
    "target_temporal_bucket",
    "primary_cross_regime",
    "target_observed_available",
    "all_horizons_available",
]


# ---------------------------------------------------------------------
# Paths and atomic output
# ---------------------------------------------------------------------


def find_repo_root(explicit_root: str | None) -> Path:
    """Locate the Showcase repo without hard-coding a developer-specific path."""
    if explicit_root:
        root = Path(explicit_root).expanduser().resolve()

        if not (
            root
            / "data"
            / "processed"
            / "1.3.1.final_tables"
        ).exists():
            raise FileNotFoundError(
                "--repo-root does not contain data/processed/1.3.1.final_tables."
            )

        return root

    starts = [
        Path.cwd().resolve(),
        Path(__file__).resolve().parent,
    ]

    for start in starts:
        for candidate in [start, *start.parents]:
            if (
                candidate
                / "data"
                / "processed"
                / "1.3.1.final_tables"
            ).exists():
                return candidate

    raise FileNotFoundError(
        "Could not locate the Showcase repo. Run this script from inside the "
        "repo or pass --repo-root."
    )


def atomic_write_parquet(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    """Write atomically so Streamlit never sees a half-written artifact."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_name(
        f".{path.stem}.{uuid.uuid4().hex[:8]}.tmp.parquet"
    )

    try:
        frame.to_parquet(
            temp_path,
            index=False,
            compression="zstd",
        )
        os.replace(
            temp_path,
            path,
        )

    finally:
        if temp_path.exists():
            temp_path.unlink()


# ---------------------------------------------------------------------
# Source validation
# ---------------------------------------------------------------------


def require_columns(
    frame: pd.DataFrame,
    required: list[str],
    label: str,
) -> None:
    """Fail immediately when an expected local contract changes."""
    missing = sorted(
        set(required)
        - set(frame.columns)
    )

    if missing:
        raise KeyError(
            f"{label} is missing required columns: "
            + ", ".join(missing)
        )


def load_supported_sequence_contract(
    horizon_path: Path,
) -> dict[str, pd.DataFrame]:
    """
    Resolve the exact broad sequence population used by the postlaunch braid.

    The counterfactual source is supported at Taxi Zone × temporal bucket ×
    metric. Using that same sequence population for prelaunch history avoids a
    visible level jump caused merely by aggregating a broader set of dayparts or
    places before Jan 5.
    """
    horizon = pd.read_parquet(
        horizon_path,
        columns=HORIZON_SUPPORT_COLUMNS,
    )

    require_columns(
        horizon,
        HORIZON_SUPPORT_COLUMNS,
        "Counterfactual horizon-stability surface",
    )

    horizon = horizon.copy()

    horizon["taxi_zone_id"] = pd.to_numeric(
        horizon["taxi_zone_id"],
        errors="raise",
    ).astype("Int64")

    eligible = (
        horizon["primary_cross_regime"].fillna(False).astype(bool)
        & horizon["target_observed_available"].fillna(False).astype(bool)
        & horizon["all_horizons_available"].fillna(False).astype(bool)
        & horizon["metric"].isin(METRICS)
        & horizon["taxi_zone_id"].notna()
        & horizon["target_temporal_bucket"].notna()
    )

    horizon = horizon.loc[
        eligible,
        [
            "taxi_zone_id",
            "metric",
            "target_temporal_bucket",
        ],
    ].drop_duplicates()

    if horizon.empty:
        raise RuntimeError(
            "No common-support sequence keys were found in the canonical "
            "counterfactual horizon surface."
        )

    support: dict[str, pd.DataFrame] = {}

    print("Common-support Taxi Zone × temporal-bucket sequences by metric:")

    for metric in METRICS:
        metric_support = (
            horizon.loc[
                horizon["metric"].eq(metric),
                [
                    "taxi_zone_id",
                    "target_temporal_bucket",
                ],
            ]
            .drop_duplicates()
            .rename(
                columns={
                    "target_temporal_bucket": "temporal_bucket",
                }
            )
            .reset_index(drop=True)
        )

        if metric_support.empty:
            raise RuntimeError(
                f"No common-support sequence keys found for {metric}."
            )

        support[metric] = metric_support

        print(
            f"  {metric:<24} "
            f"{len(metric_support):>5,} sequences · "
            f"{metric_support['taxi_zone_id'].nunique():>3,} zones"
        )

    return support

def load_prelaunch_panel(
    panel_path: Path,
    *,
    history_weeks: int,
) -> pd.DataFrame:
    """Load only the requested prelaunch window and columns from 1.3.1."""
    history_start = (
        CP_START
        - pd.Timedelta(
            days=7 * history_weeks
        )
    )

    history_end = (
        CP_START
        - pd.Timedelta(days=1)
    )

    print()
    print(
        "Reading canonical observed mobility panel "
        f"from {history_start.date()} through {history_end.date()}..."
    )

    # Predicate pushdown keeps this a local preprocessing step rather than
    # materializing the full 1.56M-row study panel.
    frame = pd.read_parquet(
        panel_path,
        columns=PANEL_COLUMNS,
        filters=[
            (
                "date",
                ">=",
                history_start,
            ),
            (
                "date",
                "<=",
                history_end,
            ),
        ],
    )

    require_columns(
        frame,
        PANEL_COLUMNS,
        "Canonical 1.3.1 mobility panel",
    )

    frame = frame.copy()

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="raise",
    ).dt.normalize()

    frame["taxi_zone_id"] = pd.to_numeric(
        frame["taxi_zone_id"],
        errors="raise",
    ).astype("Int64")

    if frame.empty:
        raise RuntimeError(
            "No rows were read from the canonical 1.3.1 prelaunch window."
        )

    day_offset = (
        frame["date"]
        - CP_START
    ).dt.days

    # Policy-relative weeks align perfectly with the postlaunch braid:
    # ... Dec 22, Dec 29, Jan 5, Jan 12 ...
    frame["week_start"] = (
        CP_START
        + pd.to_timedelta(
            (day_offset // 7) * 7,
            unit="D",
        )
    )

    print(
        f"Loaded {len(frame):,} prelaunch Taxi Zone × date × temporal-bucket rows."
    )

    return frame


# ---------------------------------------------------------------------
# Frozen Pre-CP mobility-environment context
# ---------------------------------------------------------------------


def load_pre_cp_mobility_environment(
    assignments_path: Path,
) -> pd.DataFrame:
    """Return one frozen Pre-CP mobility-environment label per Taxi Zone."""
    required = [
        "taxi_zone_id",
        "pre_post_cp",
        "canonical_cluster_name",
    ]

    assignments = pd.read_parquet(
        assignments_path,
        columns=required,
    )

    require_columns(
        assignments,
        required,
        "Canonical mobility-environment assignments",
    )

    assignments = assignments.copy()
    assignments["taxi_zone_id"] = pd.to_numeric(
        assignments["taxi_zone_id"],
        errors="raise",
    ).astype("Int64")

    pre_cp = (
        assignments.loc[
            assignments["pre_post_cp"].astype(str).eq("pre_cp"),
            [
                "taxi_zone_id",
                "canonical_cluster_name",
            ],
        ]
        .drop_duplicates()
        .rename(
            columns={
                "canonical_cluster_name": "pre_cp_mobility_environment",
            }
        )
    )

    if pre_cp.empty:
        raise RuntimeError(
            "Canonical cluster assignments contain no Pre-CP mobility environments."
        )

    if pre_cp["taxi_zone_id"].duplicated().any():
        duplicate_ids = (
            pre_cp.loc[
                pre_cp["taxi_zone_id"].duplicated(keep=False),
                "taxi_zone_id",
            ]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        raise RuntimeError(
            "Pre-CP mobility-environment assignments are not one-to-one by "
            f"Taxi Zone: {duplicate_ids[:10]}"
        )

    return pre_cp.reset_index(drop=True)


def attach_pre_cp_mobility_environment(
    panel: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the same frozen Pre-CP environment used by the postlaunch braid."""
    result = panel.merge(
        assignments,
        on="taxi_zone_id",
        how="left",
        validate="many_to_one",
    )

    known = int(result["pre_cp_mobility_environment"].notna().sum())
    print(
        "Attached Pre-CP mobility environments to "
        f"{known:,}/{len(result):,} prelaunch rows."
    )

    return result


# ---------------------------------------------------------------------
# Geography helpers
# ---------------------------------------------------------------------


def geography_views(
    frame: pd.DataFrame,
):
    """
    Yield only the geography lenses Raw 21 currently uses.

    We intentionally do not build Taxi-Zone or mobility-environment run-ups here.
    Those are not needed for the Raw 21 hero and keeping the artifact narrow helps
    Cloud Run startup/runtime performance.
    """
    yield (
        "Systemwide",
        frame.assign(
            geography_type="Systemwide",
            geography_value="NYC supported system",
        ),
    )

    borough = frame.loc[
        frame["borough"].notna()
    ].copy()

    borough["geography_type"] = "Borough"
    borough["geography_value"] = (
        borough["borough"]
        .astype(str)
    )

    yield (
        "Borough",
        borough,
    )

    policy = frame.loc[
        frame[
            "cbd_spatial_category"
        ].notna()
    ].copy()

    policy["geography_type"] = (
        "Policy geography"
    )

    # Preserve the exact 1.3.1 labels because the postlaunch compact braid uses
    # the same frozen source field.
    policy["geography_value"] = (
        policy[
            "cbd_spatial_category"
        ]
        .astype(str)
    )

    yield (
        "Policy geography",
        policy,
    )


# ---------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------


def metric_source_columns(
    metric: str,
) -> list[str]:
    """Return the minimum value/weight columns needed for one metric."""
    columns = [
        "taxi_zone_id",
        "zone",
        "borough",
        "cbd_spatial_category",
        "date",
        "temporal_bucket",
        "week_start",
        metric,
    ]

    weight_column = (
        SPEED_WEIGHT_COLUMNS.get(
            metric
        )
    )

    if (
        weight_column is not None
        and weight_column not in columns
    ):
        columns.append(
            weight_column
        )

    return columns


def aggregate_count_metric(
    frame: pd.DataFrame,
    *,
    metric: str,
    group_columns: list[str],
) -> pd.DataFrame:
    """Sum trips/riders after dropping only rows where that metric is absent."""
    scoped = frame.loc[
        frame[metric].notna()
    ].copy()

    if scoped.empty:
        return pd.DataFrame()

    scoped[metric] = pd.to_numeric(
        scoped[metric],
        errors="coerce",
    )

    scoped = scoped.loc[
        scoped[metric].notna()
    ].copy()

    return (
        scoped.groupby(
            group_columns,
            observed=True,
            dropna=False,
            sort=False,
        )
        .agg(
            support_rows=(
                metric,
                "size",
            ),
            observed_level=(
                metric,
                "sum",
            ),
        )
        .reset_index()
    )


def aggregate_speed_metric(
    frame: pd.DataFrame,
    *,
    metric: str,
    group_columns: list[str],
) -> pd.DataFrame:
    """Activity-weighted observed speed using the matching observed trip count."""
    weight_column = (
        SPEED_WEIGHT_COLUMNS[
            metric
        ]
    )

    scoped = frame.loc[
        frame[metric].notna()
        & frame[
            weight_column
        ].notna()
    ].copy()

    if scoped.empty:
        return pd.DataFrame()

    scoped[metric] = pd.to_numeric(
        scoped[metric],
        errors="coerce",
    )

    scoped[
        weight_column
    ] = pd.to_numeric(
        scoped[
            weight_column
        ],
        errors="coerce",
    )

    scoped = scoped.loc[
        scoped[metric].notna()
        & scoped[
            weight_column
        ].notna()
        & scoped[
            weight_column
        ].gt(0)
    ].copy()

    if scoped.empty:
        return pd.DataFrame()

    scoped[
        "weighted_speed_numerator"
    ] = (
        scoped[metric]
        * scoped[
            weight_column
        ]
    )

    summary = (
        scoped.groupby(
            group_columns,
            observed=True,
            dropna=False,
            sort=False,
        )
        .agg(
            support_rows=(
                metric,
                "size",
            ),
            numerator=(
                "weighted_speed_numerator",
                "sum",
            ),
            denominator=(
                weight_column,
                "sum",
            ),
        )
        .reset_index()
    )

    summary[
        "observed_level"
    ] = (
        summary[
            "numerator"
        ]
        / summary[
            "denominator"
        ].where(
            summary[
                "denominator"
            ].gt(0)
        )
    )

    return summary.drop(
        columns=[
            "numerator",
            "denominator",
        ]
    )


# ---------------------------------------------------------------------
# Build compact prelaunch artifact
# ---------------------------------------------------------------------


def build_prelaunch_context(
    panel: pd.DataFrame,
    supported_sequences: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Aggregate weekly prelaunch context for systemwide, borough, and policy views."""
    output_parts = []

    print()
    print(
        "Aggregating compact prelaunch context..."
    )

    for metric in METRICS:
        # Build the slice inline rather than through a helper so the two
        # support keys used below can never be accidentally dropped.
        metric_columns = [
            "taxi_zone_id",
            "zone",
            "borough",
            "cbd_spatial_category",
            "date",
            "temporal_bucket",
            "week_start",
            metric,
        ]

        weight_column = SPEED_WEIGHT_COLUMNS.get(metric)

        if (
            weight_column is not None
            and weight_column not in metric_columns
        ):
            metric_columns.append(
                weight_column
            )

        metric_frame = panel[
            metric_columns
        ].copy()

        # Fail here with a precise message rather than allowing pandas.merge()
        # to surface an opaque KeyError.
        required_support_keys = {
            "taxi_zone_id",
            "temporal_bucket",
        }

        missing_support_keys = sorted(
            required_support_keys
            - set(metric_frame.columns)
        )

        if missing_support_keys:
            raise KeyError(
                f"{metric}: metric slice lost support keys before merge: "
                + ", ".join(missing_support_keys)
            )

        print(
            f"    support-key check: "
            f"{', '.join(sorted(required_support_keys))} present"
        )

        # WHY:
        # Postlaunch common support is defined at Taxi Zone × temporal bucket.
        # Use explicit MultiIndex membership rather than a dataframe merge.
        # This makes the support filter transparent and removes the join path
        # that was previously surfacing the temporal_bucket KeyError.
        support_keys = supported_sequences[metric][
            [
                "taxi_zone_id",
                "temporal_bucket",
            ]
        ].copy()

        support_keys["taxi_zone_id"] = pd.to_numeric(
            support_keys["taxi_zone_id"],
            errors="raise",
        ).astype("Int64")

        support_keys["temporal_bucket"] = (
            support_keys["temporal_bucket"]
            .astype("string")
        )

        metric_frame["taxi_zone_id"] = pd.to_numeric(
            metric_frame["taxi_zone_id"],
            errors="raise",
        ).astype("Int64")

        metric_frame["temporal_bucket"] = (
            metric_frame["temporal_bucket"]
            .astype("string")
        )

        support_index = pd.MultiIndex.from_frame(
            support_keys[
                [
                    "taxi_zone_id",
                    "temporal_bucket",
                ]
            ]
        )

        metric_index = pd.MultiIndex.from_frame(
            metric_frame[
                [
                    "taxi_zone_id",
                    "temporal_bucket",
                ]
            ]
        )

        support_mask = metric_index.isin(
            support_index
        )

        metric_frame = metric_frame.loc[
            support_mask
        ].copy()

        print(
            f"    retained "
            f"{len(metric_frame):,} prelaunch rows after exact "
            f"Taxi Zone × temporal-bucket support filter"
        )

        if metric_frame.empty:
            raise RuntimeError(
                f"No prelaunch rows remain for {metric} after applying the "
                "postlaunch common-support Taxi Zone × temporal-bucket population."
            )

        for (
            geography_name,
            geography_frame,
        ) in geography_views(
            metric_frame
        ):
            print(
                f"  {metric:<24} | "
                f"{geography_name}"
            )

            group_columns = [
                "geography_type",
                "geography_value",
                "week_start",
            ]

            if metric in COUNT_METRICS:
                summary = (
                    aggregate_count_metric(
                        geography_frame,
                        metric=metric,
                        group_columns=(
                            group_columns
                        ),
                    )
                )
            else:
                summary = (
                    aggregate_speed_metric(
                        geography_frame,
                        metric=metric,
                        group_columns=(
                            group_columns
                        ),
                    )
                )

            if summary.empty:
                continue

            summary["metric"] = metric
            summary["summary_grain"] = "week"
            summary["period_start"] = (
                summary[
                    "week_start"
                ]
            )
            summary["period_complete"] = True

            output_parts.append(
                summary
            )

    if not output_parts:
        raise RuntimeError(
            "No prelaunch context rows were produced."
        )

    result = pd.concat(
        output_parts,
        ignore_index=True,
    )

    column_order = [
        "summary_grain",
        "period_start",
        "period_complete",
        "week_start",
        "geography_type",
        "geography_value",
        "metric",
        "support_rows",
        "observed_level",
    ]

    result = (
        result[
            column_order
        ]
        .sort_values(
            [
                "geography_type",
                "geography_value",
                "metric",
                "week_start",
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ---------------------------------------------------------------------
# Build zone × week × temporal-bucket explorer history
# ---------------------------------------------------------------------


def build_prelaunch_temporal_explorer(
    panel: pd.DataFrame,
    supported_sequences: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Preserve the minimum observed history needed for Raw 21 drill-downs.

    The artifact stays at Taxi Zone × policy-relative week × temporal bucket.
    Raw 21 can therefore combine weekday/weekend and daypart selections while
    preserving exact count sums and activity-weighted speed aggregation.
    """
    output_parts = []

    print()
    print("Aggregating prelaunch zone × week × temporal-bucket explorer rows...")

    for metric in METRICS:
        metric_columns = [
            "taxi_zone_id",
            "zone",
            "borough",
            "cbd_spatial_category",
            "pre_cp_mobility_environment",
            "date",
            "temporal_bucket",
            "week_start",
            metric,
        ]

        weight_column = SPEED_WEIGHT_COLUMNS.get(metric)
        if weight_column is not None and weight_column not in metric_columns:
            metric_columns.append(weight_column)

        metric_frame = panel[metric_columns].copy()
        metric_frame["taxi_zone_id"] = pd.to_numeric(
            metric_frame["taxi_zone_id"],
            errors="raise",
        ).astype("Int64")
        metric_frame["temporal_bucket"] = (
            metric_frame["temporal_bucket"].astype("string")
        )

        support_keys = supported_sequences[metric][
            ["taxi_zone_id", "temporal_bucket"]
        ].copy()
        support_keys["taxi_zone_id"] = pd.to_numeric(
            support_keys["taxi_zone_id"],
            errors="raise",
        ).astype("Int64")
        support_keys["temporal_bucket"] = (
            support_keys["temporal_bucket"].astype("string")
        )

        support_index = pd.MultiIndex.from_frame(
            support_keys[["taxi_zone_id", "temporal_bucket"]]
        )
        metric_index = pd.MultiIndex.from_frame(
            metric_frame[["taxi_zone_id", "temporal_bucket"]]
        )
        metric_frame = metric_frame.loc[
            metric_index.isin(support_index)
        ].copy()

        if metric_frame.empty:
            raise RuntimeError(
                f"{metric}: no prelaunch temporal-explorer rows remain after "
                "the common-support filter."
            )

        group_columns = [
            "week_start",
            "temporal_bucket",
            "taxi_zone_id",
            "zone",
            "borough",
            "cbd_spatial_category",
            "pre_cp_mobility_environment",
        ]

        if metric in COUNT_METRICS:
            summary = aggregate_count_metric(
                metric_frame,
                metric=metric,
                group_columns=group_columns,
            )
            summary["observed_weight"] = np.nan

        else:
            scoped = metric_frame.loc[
                metric_frame[metric].notna()
                & metric_frame[weight_column].notna()
            ].copy()
            scoped[metric] = pd.to_numeric(
                scoped[metric],
                errors="coerce",
            )
            scoped[weight_column] = pd.to_numeric(
                scoped[weight_column],
                errors="coerce",
            )
            scoped = scoped.loc[
                scoped[metric].notna()
                & scoped[weight_column].notna()
                & scoped[weight_column].gt(0)
            ].copy()

            if scoped.empty:
                continue

            scoped["weighted_speed_numerator"] = (
                scoped[metric] * scoped[weight_column]
            )

            summary = (
                scoped.groupby(
                    group_columns,
                    observed=True,
                    dropna=False,
                    sort=False,
                )
                .agg(
                    support_rows=(metric, "size"),
                    numerator=("weighted_speed_numerator", "sum"),
                    observed_weight=(weight_column, "sum"),
                )
                .reset_index()
            )
            summary["observed_level"] = (
                summary["numerator"]
                / summary["observed_weight"].where(
                    summary["observed_weight"].gt(0)
                )
            )
            summary = summary.drop(columns="numerator")

        summary["metric"] = metric
        output_parts.append(summary)

    if not output_parts:
        raise RuntimeError("No prelaunch temporal explorer rows were produced.")

    result = pd.concat(output_parts, ignore_index=True)
    result["taxi_zone_id"] = pd.to_numeric(
        result["taxi_zone_id"],
        errors="raise",
    ).astype("Int64")

    column_order = [
        "week_start",
        "temporal_bucket",
        "taxi_zone_id",
        "zone",
        "borough",
        "cbd_spatial_category",
        "pre_cp_mobility_environment",
        "metric",
        "support_rows",
        "observed_level",
        "observed_weight",
    ]

    return (
        result[column_order]
        .sort_values(
            [
                "metric",
                "taxi_zone_id",
                "week_start",
                "temporal_bucket",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def collapse_prelaunch_temporal_systemwide(
    explorer: pd.DataFrame,
) -> pd.DataFrame:
    """Rebuild all-day systemwide weekly history for exact QA."""
    output_parts = []

    for metric in METRICS:
        scoped = explorer.loc[
            explorer["metric"].eq(metric)
        ].copy()

        if metric in COUNT_METRICS:
            weekly = (
                scoped.groupby(
                    "week_start",
                    observed=True,
                    sort=False,
                )["observed_level"]
                .sum()
                .reset_index()
            )
        else:
            scoped["numerator"] = (
                scoped["observed_level"]
                * scoped["observed_weight"]
            )
            weekly = (
                scoped.groupby(
                    "week_start",
                    observed=True,
                    sort=False,
                )
                .agg(
                    numerator=("numerator", "sum"),
                    observed_weight=("observed_weight", "sum"),
                )
                .reset_index()
            )
            weekly["observed_level"] = (
                weekly["numerator"]
                / weekly["observed_weight"].where(
                    weekly["observed_weight"].gt(0)
                )
            )
            weekly = weekly[["week_start", "observed_level"]]

        weekly["metric"] = metric
        output_parts.append(weekly)

    return pd.concat(output_parts, ignore_index=True)


def build_prelaunch_temporal_qa(
    explorer: pd.DataFrame,
    compact_context: pd.DataFrame,
    *,
    history_weeks: int,
) -> pd.DataFrame:
    """Validate temporal drill-down history against the existing compact run-up."""
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
        "temporal_bucket",
        "taxi_zone_id",
        "metric",
    ]
    duplicate_count = int(explorer.duplicated(duplicate_key).sum())
    add(
        "unique_zone_week_bucket_rows",
        duplicate_count == 0,
        f"{duplicate_count:,} duplicate explorer keys.",
    )

    add(
        "metric_coverage",
        set(explorer["metric"].unique()) == set(METRICS),
        "All five counterfactual metrics are present.",
    )

    bucket_values = sorted(
        explorer["temporal_bucket"].dropna().astype(str).unique().tolist()
    )
    add(
        "temporal_bucket_coverage",
        len(bucket_values) == 10,
        f"Temporal buckets ({len(bucket_values)}): {bucket_values}.",
    )

    environment_count = explorer[
        "pre_cp_mobility_environment"
    ].dropna().nunique()
    add(
        "mobility_environment_context_available",
        environment_count >= 5,
        f"Pre-CP mobility environments available: {environment_count}.",
    )

    rebuilt = collapse_prelaunch_temporal_systemwide(explorer)
    expected = compact_context.loc[
        compact_context["geography_type"].eq("Systemwide"),
        ["week_start", "metric", "observed_level"],
    ].copy()

    merged = rebuilt.merge(
        expected,
        on=["week_start", "metric"],
        how="outer",
        validate="one_to_one",
        suffixes=("_rebuilt", "_expected"),
        indicator=True,
    )

    aligned = merged["_merge"].eq("both").all()
    values_match = aligned and np.allclose(
        merged["observed_level_rebuilt"],
        merged["observed_level_expected"],
        rtol=1e-9,
        atol=1e-9,
        equal_nan=True,
    )
    max_abs_difference = (
        float(
            (
                merged["observed_level_rebuilt"]
                - merged["observed_level_expected"]
            ).abs().max()
        )
        if aligned and not merged.empty
        else np.nan
    )
    add(
        "rebuilds_existing_systemwide_prelaunch",
        bool(values_match),
        f"Maximum absolute reconstruction difference: {max_abs_difference:.12g}.",
    )

    week_counts = (
        rebuilt.groupby("metric", observed=True)["week_start"]
        .nunique()
    )
    add(
        "systemwide_history_length",
        bool(week_counts.eq(history_weeks).all()),
        ", ".join(
            f"{metric}={int(count)}"
            for metric, count in week_counts.items()
        ),
    )

    return pd.DataFrame(checks)


# ---------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------


def build_qa(
    result: pd.DataFrame,
    *,
    history_weeks: int,
) -> pd.DataFrame:
    """Validate the app-facing history before Raw 21 consumes it."""
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
        "metric_coverage",
        set(
            result[
                "metric"
            ].unique()
        ) == set(
            METRICS
        ),
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

    add(
        "geography_lenses",
        set(
            result[
                "geography_type"
            ].unique()
        ) == {
            "Systemwide",
            "Borough",
            "Policy geography",
        },
        (
            "Output geography lenses: "
            + ", ".join(
                sorted(
                    result[
                        "geography_type"
                    ].unique()
                )
            )
        ),
    )

    duplicate_count = int(
        result.duplicated(
            [
                "geography_type",
                "geography_value",
                "metric",
                "week_start",
            ]
        ).sum()
    )

    add(
        "unique_output_rows",
        duplicate_count == 0,
        (
            f"{duplicate_count:,} duplicate output keys."
        ),
    )

    systemwide = result.loc[
        result[
            "geography_type"
        ].eq(
            "Systemwide"
        )
    ].copy()

    week_counts = (
        systemwide.groupby(
            "metric",
            observed=True,
            dropna=False,
        )[
            "week_start"
        ]
        .nunique()
        .reset_index(
            name="week_count"
        )
    )

    add(
        "systemwide_history_length",
        bool(
            week_counts[
                "week_count"
            ].eq(
                history_weeks
            ).all()
        ),
        (
            "Systemwide weekly counts: "
            + ", ".join(
                (
                    f"{row.metric}="
                    f"{int(row.week_count)}"
                )
                for row
                in week_counts.itertuples(
                    index=False
                )
            )
        ),
    )

    expected_first_week = (
        CP_START
        - pd.Timedelta(
            days=7 * history_weeks
        )
    )

    expected_last_week = (
        CP_START
        - pd.Timedelta(
            days=7
        )
    )

    actual_first_week = (
        systemwide[
            "week_start"
        ].min()
    )

    actual_last_week = (
        systemwide[
            "week_start"
        ].max()
    )

    add(
        "history_window_bounds",
        (
            actual_first_week
            == expected_first_week
            and actual_last_week
            == expected_last_week
        ),
        (
            f"Expected {expected_first_week.date()} through "
            f"{expected_last_week.date()}; found "
            f"{actual_first_week.date()} through "
            f"{actual_last_week.date()}."
        ),
    )

    null_level_count = int(
        result[
            "observed_level"
        ].isna().sum()
    )

    add(
        "observed_levels_available",
        null_level_count == 0,
        (
            f"{null_level_count:,} output rows have null observed levels."
        ),
    )

    return pd.DataFrame(
        checks
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    print(
        f"COUNTERFACTUAL PRELAUNCH CONTEXT BUILD · {BUILD_VERSION}"
    )
    print()

    parser = argparse.ArgumentParser(
        description=(
            "Build compact pre-Jan-5 observed history for Raw 21 Horizon Braids."
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
    parser.add_argument(
        "--history-weeks",
        type=int,
        default=HISTORY_WEEKS_DEFAULT,
        help=(
            "Number of policy-relative weeks of observed history to keep "
            "(default: 52)."
        ),
    )

    args = parser.parse_args()

    if args.history_weeks <= 0:
        raise ValueError("--history-weeks must be positive.")

    repo_root = find_repo_root(args.repo_root)

    panel_path = (
        repo_root
        / "data"
        / "processed"
        / "1.3.1.final_tables"
        / MOBILITY_PANEL_NAME
    )
    assignments_path = (
        repo_root
        / "data"
        / "processed"
        / "3.2.2.final_tables"
        / CANONICAL_CLUSTER_ASSIGNMENTS_NAME
    )
    final_dir = (
        repo_root
        / "data"
        / "processed"
        / "5.3.1.final_tables"
    )
    horizon_path = final_dir / HORIZON_STABILITY_NAME

    output_path = final_dir / OUTPUT_NAME
    qa_path = final_dir / QA_NAME
    temporal_path = final_dir / TEMPORAL_EXPLORER_NAME
    temporal_qa_path = final_dir / TEMPORAL_EXPLORER_QA_NAME

    for path in [panel_path, assignments_path, horizon_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found:\n  {path}")

    supported_sequences = load_supported_sequence_contract(horizon_path)
    panel = load_prelaunch_panel(
        panel_path,
        history_weeks=args.history_weeks,
    )
    environment_assignments = load_pre_cp_mobility_environment(
        assignments_path
    )
    panel = attach_pre_cp_mobility_environment(
        panel,
        environment_assignments,
    )

    result = build_prelaunch_context(
        panel,
        supported_sequences,
    )
    qa = build_qa(
        result,
        history_weeks=args.history_weeks,
    )

    temporal_explorer = build_prelaunch_temporal_explorer(
        panel,
        supported_sequences,
    )
    temporal_qa = build_prelaunch_temporal_qa(
        temporal_explorer,
        result,
        history_weeks=args.history_weeks,
    )

    print()
    print("PRELAUNCH CONTEXT QA")
    print(qa.to_string(index=False))
    print()
    print("PRELAUNCH TEMPORAL EXPLORER QA")
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
            "Prelaunch preprocessing failed QA. No output files were written."
        )

    atomic_write_parquet(result, output_path)
    atomic_write_parquet(qa, qa_path)
    atomic_write_parquet(temporal_explorer, temporal_path)
    atomic_write_parquet(temporal_qa, temporal_qa_path)

    compact_size_mb = output_path.stat().st_size / (1024 ** 2)
    temporal_size_mb = temporal_path.stat().st_size / (1024 ** 2)

    print(f"Wrote {len(result):,} compact prelaunch rows:")
    print(f"  {output_path}")
    print(f"  {qa_path}")
    print()
    print(f"Wrote {len(temporal_explorer):,} prelaunch temporal explorer rows:")
    print(f"  {temporal_path}")
    print(f"  {temporal_qa_path}")
    print()
    print(f"Prelaunch context size: {compact_size_mb:,.2f} MB")
    print(f"Prelaunch temporal explorer size: {temporal_size_mb:,.2f} MB")
    print()
    print(
        "Raw 21 can now reconstruct the 52-week observed run-up for "
        "Systemwide, Borough, Policy geography, Mobility environment, or Taxi Zone "
        "and filter it by weekday/weekend and daypart without loading 1.3.1 at runtime."
    )


if __name__ == "__main__":
    main()
