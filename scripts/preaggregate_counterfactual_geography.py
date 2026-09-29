"""
Build the compact production data contract for Raw 22 — Counterfactual Geography.

WHY THIS EXISTS
---------------
Raw 22 needs to answer spatial questions quickly:

- What was observed in each Taxi Zone?
- What did the no-CP forecast estimate?
- What is the native and percentage gap?
- Which zones contributed most to the selected systemwide divergence?
- How does local percentage intensity differ from system contribution?

The authoritative Chapter 5 counterfactual surface is too large to scan live in
Streamlit. This script compresses it to:

    canonical Taxi Zone
    × policy-relative week
    × temporal bucket
    × metric

while preserving horizon-specific h=1 / h=2 / h=5 support.

Counts/ridership retain additive totals.

Average speeds retain sufficient statistics:
    weighted numerator + activity denominator

for the observed and no-CP worlds separately. This allows Raw 22 to reconstruct
correct weighted averages after any Period × Day type × Daypart filter.

Canonical Taxi Zone handling:
- source IDs 56 and 57 are recombined into physical Corona (canonical ID 56);
- nonphysical Unknown IDs 264/265 remain in the analytical artifact so system
  totals can reconcile, but map_eligible=False keeps them off geographic maps
  and reader-facing Taxi Zone rankings.

RUN FROM REPO ROOT
------------------
python .\\scripts\\preaggregate_counterfactual_geography.py
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path.cwd().resolve()

FINAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "5.3.1.final_tables"
)

if not FINAL_DIR.exists():
    raise FileNotFoundError(
        "Run this script from the Showcase repository root."
    )

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from app.data_access.spatial_visuals import get_zone_geojson  # noqa: E402


GAP_SURFACE_PATH = (
    FINAL_DIR
    / "counterfactual_gap_surface.parquet"
)

SPATIAL_SUMMARY_PATH = (
    FINAL_DIR
    / "counterfactual_spatial_summary.parquet"
)

OUTPUT_PATH = (
    FINAL_DIR
    / "counterfactual_geography_temporal_explorer.parquet"
)

QA_PATH = (
    FINAL_DIR
    / "counterfactual_geography_temporal_explorer_qa.parquet"
)


# ---------------------------------------------------------------------
# Frozen analytical contract
# ---------------------------------------------------------------------

CP_START = pd.Timestamp(
    "2025-01-05"
)

CP_END = pd.Timestamp(
    "2026-03-31"
)

HORIZONS = [
    1,
    2,
    5,
]

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

SPEED_METRICS = {
    "taxi_avg_trip_speed",
    "fhvhv_avg_trip_speed",
}

UNKNOWN_ZONE_IDS = {
    264,
    265,
}

SOURCE_COLUMNS = [
    "taxi_zone_id",
    "canonical_location_id",
    "zone",
    "borough",
    "cbd_spatial_category",
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

OUTPUT_KEY = [
    "canonical_taxi_zone_id",
    "week_start",
    "target_temporal_bucket",
    "metric",
]

SPATIAL_COLUMNS = [
    "summary_grain",
    "metric",
    "horizon",
    "taxi_zone_id",
    "support_rows",
    "observed_level",
    "no_cp_level",
    "counterfactual_gap",
    "counterfactual_gap_pct",
]


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    """Fail closed when an upstream contract changes."""
    missing = sorted(
        set(columns)
        - set(frame.columns)
    )

    if missing:
        raise KeyError(
            f"{label} is missing required columns: "
            + ", ".join(missing)
        )


def add_week_start(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach the same policy-relative Sunday weeks used elsewhere in Chapter 5.

    Jan. 5, 2025 is week zero. The final week beginning Mar. 29, 2026 is
    intentionally partial because the study ends Mar. 31.
    """
    result = frame.copy()

    result["target_date"] = pd.to_datetime(
        result["target_date"],
        errors="raise",
    ).dt.normalize()

    days_since_launch = (
        result["target_date"]
        - CP_START
    ).dt.days

    result["week_start"] = (
        CP_START
        + pd.to_timedelta(
            (days_since_launch // 7) * 7,
            unit="D",
        )
    )

    return result


def geometry_zone_ids() -> set[int]:
    """Return physical Taxi Zone IDs represented by the Showcase GeoJSON."""
    result: set[int] = set()

    for feature in get_zone_geojson().get(
        "features",
        [],
    ):
        properties = (
            feature.get(
                "properties",
                {},
            )
            or {}
        )

        value = properties.get(
            "taxi_zone_id"
        )

        if value is None:
            continue

        try:
            result.add(
                int(value)
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

    return result


def compare_numeric(
    left: pd.Series,
    right: pd.Series,
    *,
    atol: float = 1e-6,
) -> tuple[bool, float]:
    """Return equality status and maximum finite absolute difference."""
    left_values = pd.to_numeric(
        left,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    right_values = pd.to_numeric(
        right,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    close = np.isclose(
        left_values,
        right_values,
        rtol=1e-9,
        atol=atol,
        equal_nan=True,
    )

    finite = (
        np.isfinite(left_values)
        & np.isfinite(right_values)
    )

    if finite.any():
        max_difference = float(
            np.max(
                np.abs(
                    left_values[finite]
                    - right_values[finite]
                )
            )
        )
    else:
        max_difference = 0.0

    return bool(close.all()), max_difference


# ---------------------------------------------------------------------
# Metric × horizon aggregation
# ---------------------------------------------------------------------
def valid_speed_weight_rows(
    frame: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    """
    Keep only rows that can participate in a world-specific weighted speed.

    A counterfactual gap can be available even when its companion trip-count
    weight is unavailable. That row remains valid in the Chapter 5 gap surface,
    but it cannot enter an activity-weighted speed aggregate.

    Zero weights are valid and contribute zero activity. Missing weights are
    excluded. Negative weights remain a hard failure.
    """
    scoped = frame.copy()

    weight_columns = [
        "observed_activity_weight",
        "no_cp_activity_weight",
    ]

    for column in weight_columns:
        scoped[column] = pd.to_numeric(
            scoped[column],
            errors="coerce",
        )

    complete_weights = (
        scoped[
            weight_columns
        ]
        .notna()
        .all(axis=1)
    )

    if (
        scoped.loc[
            complete_weights,
            weight_columns,
        ] < 0
    ).any(axis=None):
        raise RuntimeError(
            f"{label}: negative activity weights found."
        )

    dropped_rows = int(
        (~complete_weights).sum()
    )

    if dropped_rows:
        print(
            f"    {label}: excluding {dropped_rows:,} rows "
            "without complete activity weights",
            flush=True,
        )

    scoped = scoped.loc[
        complete_weights
    ].copy()

    if scoped.empty:
        raise RuntimeError(
            f"{label}: no rows remain after applying the speed-weight contract."
        )

    return scoped

def aggregate_count_slice(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate one additive target while preserving canonical geography."""
    return (
        frame.groupby(
            [
                "canonical_location_id",
                "week_start",
                "target_temporal_bucket",
            ],
            observed=True,
            dropna=False,
            sort=False,
        )
        .agg(
            support_rows=(
                "target_observed_value",
                "size",
            ),
            observed_level=(
                "target_observed_value",
                "sum",
            ),
            no_cp_level=(
                "system_prediction",
                "sum",
            ),
        )
        .reset_index()
    )


def aggregate_speed_slice(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate one speed target using each world's own activity weights.

    Rows without a complete observed/no-CP weight pair cannot contribute to a
    weighted speed and are excluded here. Zero activity remains valid.

    The retained numerators and denominators are sufficient for Raw 22 to
    rebuild the correct weighted average after any temporal filter.
    """
    scoped = valid_speed_weight_rows(
        frame,
        label="speed aggregation",
    )

    scoped[
        "observed_numerator"
    ] = (
        scoped["target_observed_value"]
        * scoped["observed_activity_weight"]
    )

    scoped[
        "no_cp_numerator"
    ] = (
        scoped["system_prediction"]
        * scoped["no_cp_activity_weight"]
    )

    summary = (
        scoped.groupby(
            [
                "canonical_location_id",
                "week_start",
                "target_temporal_bucket",
            ],
            observed=True,
            dropna=False,
            sort=False,
        )
        .agg(
            support_rows=(
                "target_observed_value",
                "size",
            ),
            observed_numerator=(
                "observed_numerator",
                "sum",
            ),
            observed_weight=(
                "observed_activity_weight",
                "sum",
            ),
            no_cp_numerator=(
                "no_cp_numerator",
                "sum",
            ),
            no_cp_weight=(
                "no_cp_activity_weight",
                "sum",
            ),
        )
        .reset_index()
    )

    summary[
        "observed_level"
    ] = (
        summary["observed_numerator"]
        / summary[
            "observed_weight"
        ].where(
            summary[
                "observed_weight"
            ].gt(0)
        )
    )

    summary[
        "no_cp_level"
    ] = (
        summary["no_cp_numerator"]
        / summary[
            "no_cp_weight"
        ].where(
            summary[
                "no_cp_weight"
            ].gt(0)
        )
    )

    return summary


def source_zone_summary(
    frame: pd.DataFrame,
    *,
    metric: str,
    horizon: int,
) -> pd.DataFrame:
    """
    Rebuild the original source-ID Taxi Zone summary for upstream QA.

    This happens before canonicalizing 56/57, so it should reproduce the
    authoritative 5.2.1 Taxi Zone summary exactly.

    Speed rows use the same complete-weight rule as the production aggregate.
    """
    if metric in COUNT_METRICS:
        result = (
            frame.groupby(
                "taxi_zone_id",
                observed=True,
                dropna=False,
            )
            .agg(
                support_rows=(
                    "target_observed_value",
                    "size",
                ),
                observed_level=(
                    "target_observed_value",
                    "sum",
                ),
                no_cp_level=(
                    "system_prediction",
                    "sum",
                ),
            )
            .reset_index()
        )

    else:
        scoped = valid_speed_weight_rows(
            frame,
            label=f"{metric} h={horizon} source-summary QA",
        )

        scoped[
            "observed_numerator"
        ] = (
            scoped["target_observed_value"]
            * scoped["observed_activity_weight"]
        )

        scoped[
            "no_cp_numerator"
        ] = (
            scoped["system_prediction"]
            * scoped["no_cp_activity_weight"]
        )

        result = (
            scoped.groupby(
                "taxi_zone_id",
                observed=True,
                dropna=False,
            )
            .agg(
                support_rows=(
                    "target_observed_value",
                    "size",
                ),
                observed_numerator=(
                    "observed_numerator",
                    "sum",
                ),
                observed_weight=(
                    "observed_activity_weight",
                    "sum",
                ),
                no_cp_numerator=(
                    "no_cp_numerator",
                    "sum",
                ),
                no_cp_weight=(
                    "no_cp_activity_weight",
                    "sum",
                ),
            )
            .reset_index()
        )

        result[
            "observed_level"
        ] = (
            result["observed_numerator"]
            / result[
                "observed_weight"
            ].where(
                result[
                    "observed_weight"
                ].gt(0)
            )
        )

        result[
            "no_cp_level"
        ] = (
            result["no_cp_numerator"]
            / result[
                "no_cp_weight"
            ].where(
                result[
                    "no_cp_weight"
                ].gt(0)
            )
        )

    result["metric"] = metric
    result["horizon"] = horizon

    result[
        "counterfactual_gap"
    ] = (
        result["no_cp_level"]
        - result["observed_level"]
    )

    result[
        "counterfactual_gap_pct"
    ] = np.where(
        result[
            "observed_level"
        ].ne(0),
        (
            100.0
            * result[
                "counterfactual_gap"
            ]
            / result[
                "observed_level"
            ]
        ),
        np.nan,
    )

    return result


# ---------------------------------------------------------------------
# Build compact horizon-specific wide surface
# ---------------------------------------------------------------------

print(
    "Building Raw 22 counterfactual geography preaggregate...",
    flush=True,
)

geometry_ids = geometry_zone_ids()

context_parts: list[pd.DataFrame] = []
metric_parts: list[pd.DataFrame] = []
source_zone_parts: list[pd.DataFrame] = []


for metric in METRICS:
    print(
        f"\nLoading {metric}...",
        flush=True,
    )

    metric_frame = pd.read_parquet(
        GAP_SURFACE_PATH,
        columns=SOURCE_COLUMNS,
        filters=[
            (
                "metric",
                "==",
                metric,
            )
        ],
    )

    require_columns(
        metric_frame,
        SOURCE_COLUMNS,
        f"{metric} gap surface",
    )

    metric_frame = metric_frame.loc[
        metric_frame[
            "primary_gap_eligible"
        ].astype(bool)
    ].copy()

    metric_frame["horizon"] = pd.to_numeric(
        metric_frame["horizon"],
        errors="raise",
    ).astype(int)

    metric_frame["taxi_zone_id"] = pd.to_numeric(
        metric_frame["taxi_zone_id"],
        errors="raise",
    ).astype("Int64")

    metric_frame[
        "canonical_location_id"
    ] = pd.to_numeric(
        metric_frame[
            "canonical_location_id"
        ],
        errors="raise",
    ).astype("Int64")

    metric_frame = add_week_start(
        metric_frame
    )

    metric_frame = metric_frame.loc[
        metric_frame[
            "target_date"
        ].between(
            CP_START,
            CP_END,
            inclusive="both",
        )
        & metric_frame[
            "horizon"
        ].isin(
            HORIZONS
        )
    ].copy()

    if metric_frame.empty:
        raise RuntimeError(
            f"{metric}: no eligible post-CP rows remain."
        )

    # Canonical identity must be stable before source aliases are combined.
    context = (
        metric_frame[
            [
                "canonical_location_id",
                "zone",
                "borough",
                "cbd_spatial_category",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    context_parts.append(
        context
    )

    horizon_parts: list[pd.DataFrame] = []

    for horizon in HORIZONS:
        horizon_frame = metric_frame.loc[
            metric_frame[
                "horizon"
            ].eq(
                horizon
            )
        ].copy()

        if horizon_frame.empty:
            raise RuntimeError(
                f"{metric} h={horizon}: no eligible rows."
            )

        print(
            f"  h={horizon}: "
            f"{len(horizon_frame):,} eligible rows",
            flush=True,
        )

        # Rebuild the original source-ID spatial summary so QA can prove that
        # this preprocessor starts from exactly the same Chapter 5 contract.
        source_zone_parts.append(
            source_zone_summary(
                horizon_frame,
                metric=metric,
                horizon=horizon,
            )
        )

        if metric in COUNT_METRICS:
            summary = aggregate_count_slice(
                horizon_frame
            )

            summary[
                "observed_weight"
            ] = np.nan

            summary[
                "no_cp_weight"
            ] = np.nan

        else:
            summary = aggregate_speed_slice(
                horizon_frame
            )

        summary["metric"] = metric

        summary = summary.rename(
            columns={
                "support_rows":
                    f"support_rows_h{horizon}",
                "observed_level":
                    f"observed_level_h{horizon}",
                "no_cp_level":
                    f"no_cp_level_h{horizon}",
                "observed_weight":
                    f"observed_weight_h{horizon}",
                "no_cp_weight":
                    f"no_cp_weight_h{horizon}",
            }
        )

        keep_columns = [
            "canonical_location_id",
            "week_start",
            "target_temporal_bucket",
            "metric",
            f"support_rows_h{horizon}",
            f"observed_level_h{horizon}",
            f"no_cp_level_h{horizon}",
            f"observed_weight_h{horizon}",
            f"no_cp_weight_h{horizon}",
        ]

        horizon_parts.append(
            summary[
                keep_columns
            ].copy()
        )

    # Outer merge is intentional: each horizon preserves its own valid support.
    metric_wide = horizon_parts[0]

    for horizon_part in horizon_parts[1:]:
        metric_wide = metric_wide.merge(
            horizon_part,
            on=[
                "canonical_location_id",
                "week_start",
                "target_temporal_bucket",
                "metric",
            ],
            how="outer",
            validate="one_to_one",
        )

    metric_parts.append(
        metric_wide
    )

    del metric_frame
    gc.collect()


# ---------------------------------------------------------------------
# Resolve one stable canonical Taxi Zone identity
# ---------------------------------------------------------------------

context_all = (
    pd.concat(
        context_parts,
        ignore_index=True,
    )
    .drop_duplicates()
    .reset_index(drop=True)
)

context_conflicts = (
    context_all.groupby(
        "canonical_location_id",
        observed=True,
        dropna=False,
    )
    .size()
)

conflicting_ids = (
    context_conflicts.loc[
        context_conflicts.gt(1)
    ]
    .index
    .tolist()
)

if conflicting_ids:
    conflict_rows = context_all.loc[
        context_all[
            "canonical_location_id"
        ].isin(
            conflicting_ids
        )
    ]

    raise RuntimeError(
        "Canonical Taxi Zone context is not stable:\n"
        + conflict_rows.to_string(
            index=False
        )
    )

context_all = context_all.rename(
    columns={
        "canonical_location_id":
            "canonical_taxi_zone_id",
    }
)

context_all[
    "map_eligible"
] = (
    context_all[
        "canonical_taxi_zone_id"
    ]
    .map(
        lambda value: (
            int(value) in geometry_ids
            if pd.notna(value)
            else False
        )
    )
)

context_all[
    "reader_facing_zone"
] = (
    context_all[
        "map_eligible"
    ]
    & ~context_all[
        "canonical_taxi_zone_id"
    ].isin(
        UNKNOWN_ZONE_IDS
    )
    & context_all[
        "zone"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
    .ne(
        "unknown"
    )
)


# ---------------------------------------------------------------------
# Assemble final compact artifact
# ---------------------------------------------------------------------

result = pd.concat(
    metric_parts,
    ignore_index=True,
)

result = result.rename(
    columns={
        "canonical_location_id":
            "canonical_taxi_zone_id",
    }
)

result = result.merge(
    context_all,
    on="canonical_taxi_zone_id",
    how="left",
    validate="many_to_one",
)

if result[
    "zone"
].isna().any():
    raise RuntimeError(
        "Some compact geography rows did not resolve to canonical zone context."
    )

result[
    "period_complete"
] = (
    result["week_start"]
    + pd.Timedelta(
        days=6
    )
).le(
    CP_END
)

result[
    "aggregation_type"
] = np.where(
    result[
        "metric"
    ].isin(
        COUNT_METRICS
    ),
    "sum",
    "weighted_mean",
)

result = result[
    [
        "week_start",
        "period_complete",
        "target_temporal_bucket",
        "canonical_taxi_zone_id",
        "zone",
        "borough",
        "cbd_spatial_category",
        "map_eligible",
        "reader_facing_zone",
        "metric",
        "aggregation_type",
        "support_rows_h1",
        "observed_level_h1",
        "no_cp_level_h1",
        "observed_weight_h1",
        "no_cp_weight_h1",
        "support_rows_h2",
        "observed_level_h2",
        "no_cp_level_h2",
        "observed_weight_h2",
        "no_cp_weight_h2",
        "support_rows_h5",
        "observed_level_h5",
        "no_cp_level_h5",
        "observed_weight_h5",
        "no_cp_weight_h5",
    ]
].sort_values(
    [
        "metric",
        "canonical_taxi_zone_id",
        "week_start",
        "target_temporal_bucket",
    ],
    na_position="last",
).reset_index(
    drop=True
)


# ---------------------------------------------------------------------
# QA 1 — Exact output grain
# ---------------------------------------------------------------------

qa_rows: list[dict[str, object]] = []


def add_qa(
    check_id: str,
    passed: bool,
    details: str,
) -> None:
    """Append one compact QA result."""
    qa_rows.append(
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


duplicate_count = int(
    result.duplicated(
        OUTPUT_KEY
    ).sum()
)

add_qa(
    "unique_canonical_zone_week_bucket_metric",
    duplicate_count == 0,
    f"duplicate keys={duplicate_count:,}",
)

add_qa(
    "metric_coverage",
    set(
        result[
            "metric"
        ].unique()
    ) == set(
        METRICS
    ),
    (
        "metrics="
        + ", ".join(
            sorted(
                result[
                    "metric"
                ].unique()
            )
        )
    ),
)

actual_buckets = set(
    result[
        "target_temporal_bucket"
    ]
    .dropna()
    .astype(str)
    .unique()
)

add_qa(
    "temporal_bucket_coverage",
    len(
        actual_buckets
    ) == 10,
    f"buckets={len(actual_buckets)}",
)


# ---------------------------------------------------------------------
# QA 2 — Original source-ID rebuild matches authoritative 5.2.1 summary
# ---------------------------------------------------------------------

rebuilt_source_zones = pd.concat(
    source_zone_parts,
    ignore_index=True,
)

authoritative = pd.read_parquet(
    SPATIAL_SUMMARY_PATH,
    columns=SPATIAL_COLUMNS,
)

authoritative = authoritative.loc[
    authoritative[
        "summary_grain"
    ].astype(str).eq(
        "taxi_zone"
    )
].copy()

authoritative[
    "taxi_zone_id"
] = pd.to_numeric(
    authoritative[
        "taxi_zone_id"
    ],
    errors="raise",
).astype("Int64")

comparison = rebuilt_source_zones.merge(
    authoritative[
        [
            "metric",
            "horizon",
            "taxi_zone_id",
            "support_rows",
            "observed_level",
            "no_cp_level",
            "counterfactual_gap",
            "counterfactual_gap_pct",
        ]
    ],
    on=[
        "metric",
        "horizon",
        "taxi_zone_id",
    ],
    how="outer",
    suffixes=(
        "_rebuilt",
        "_authoritative",
    ),
    indicator=True,
    validate="one_to_one",
)

keys_match = bool(
    comparison[
        "_merge"
    ].eq(
        "both"
    ).all()
)

add_qa(
    "authoritative_spatial_keys_reproduced",
    keys_match,
    (
        "rebuilt="
        f"{len(rebuilt_source_zones):,}; "
        "authoritative="
        f"{len(authoritative):,}; "
        "unmatched="
        f"{int(comparison['_merge'].ne('both').sum()):,}"
    ),
)

for column in [
    "support_rows",
    "observed_level",
    "no_cp_level",
    "counterfactual_gap",
    "counterfactual_gap_pct",
]:
    passed, max_diff = compare_numeric(
        comparison[
            f"{column}_rebuilt"
        ],
        comparison[
            f"{column}_authoritative"
        ],
    )

    add_qa(
        f"authoritative_{column}_reproduced",
        passed,
        f"max abs difference={max_diff:.12g}",
    )


# ---------------------------------------------------------------------
# QA 3 — Canonical geography behaves as intended
# ---------------------------------------------------------------------

corona_rows = result.loc[
    result[
        "canonical_taxi_zone_id"
    ].eq(
        56
    )
]

add_qa(
    "corona_canonical_zone_present",
    not corona_rows.empty,
    (
        f"canonical 56 rows={len(corona_rows):,}; "
        "source aliases 56/57 are aggregated before export"
    ),
)

unknown_rows = result.loc[
    result[
        "canonical_taxi_zone_id"
    ].isin(
        UNKNOWN_ZONE_IDS
    )
]

unknown_map_eligible = int(
    unknown_rows[
        "map_eligible"
    ].fillna(
        False
    ).sum()
)

add_qa(
    "unknown_zones_not_map_eligible",
    unknown_map_eligible == 0,
    (
        f"Unknown rows={len(unknown_rows):,}; "
        f"map-eligible Unknown rows={unknown_map_eligible:,}"
    ),
)

mapped_physical_rows = result.loc[
    result[
        "reader_facing_zone"
    ]
]

unmapped_reader_ids = sorted(
    set(
        mapped_physical_rows[
            "canonical_taxi_zone_id"
        ]
        .dropna()
        .astype(int)
        .unique()
    )
    - geometry_ids
)

add_qa(
    "reader_facing_geometry_coverage",
    len(
        unmapped_reader_ids
    ) == 0,
    f"unmapped canonical IDs={unmapped_reader_ids}",
)


# ---------------------------------------------------------------------
# QA 4 — Horizon-specific values exist independently
# ---------------------------------------------------------------------

for horizon in HORIZONS:
    support_column = (
        f"support_rows_h{horizon}"
    )

    supported_rows = int(
        pd.to_numeric(
            result[
                support_column
            ],
            errors="coerce",
        )
        .fillna(0)
        .gt(0)
        .sum()
    )

    add_qa(
        f"h{horizon}_support_present",
        supported_rows > 0,
        f"compact rows with support={supported_rows:,}",
    )


# ---------------------------------------------------------------------
# QA 5 — Every displayed speed slice has valid aggregate weights
# ---------------------------------------------------------------------

for metric in sorted(
    SPEED_METRICS
):
    metric_result = result.loc[
        result[
            "metric"
        ].eq(
            metric
        )
    ]

    for horizon in HORIZONS:
        support = pd.to_numeric(
            metric_result[
                f"support_rows_h{horizon}"
            ],
            errors="coerce",
        ).fillna(
            0
        )

        observed_weight = pd.to_numeric(
            metric_result[
                f"observed_weight_h{horizon}"
            ],
            errors="coerce",
        )

        no_cp_weight = pd.to_numeric(
            metric_result[
                f"no_cp_weight_h{horizon}"
            ],
            errors="coerce",
        )

        observed_level = pd.to_numeric(
            metric_result[
                f"observed_level_h{horizon}"
            ],
            errors="coerce",
        )

        no_cp_level = pd.to_numeric(
            metric_result[
                f"no_cp_level_h{horizon}"
            ],
            errors="coerce",
        )

        supported = support.gt(
            0
        )

        invalid_observed = int(
            (
                supported
                & observed_level.notna()
                & ~observed_weight.gt(0)
            ).sum()
        )

        invalid_no_cp = int(
            (
                supported
                & no_cp_level.notna()
                & ~no_cp_weight.gt(0)
            ).sum()
        )

        add_qa(
            f"{metric}_h{horizon}_speed_denominators",
            (
                invalid_observed == 0
                and invalid_no_cp == 0
            ),
            (
                "finite displayed levels without positive denominator: "
                f"observed={invalid_observed:,}, "
                f"no_cp={invalid_no_cp:,}"
            ),
        )


# ---------------------------------------------------------------------
# QA 6 — Full-period system totals survive canonicalization exactly
# ---------------------------------------------------------------------

def aggregate_output_job(
    frame: pd.DataFrame,
    *,
    metric: str,
    horizon: int,
) -> tuple[float, float]:
    """Reconstruct one systemwide full-period job from the compact output."""
    scoped = frame.loc[
        frame[
            "metric"
        ].eq(
            metric
        )
        & pd.to_numeric(
            frame[
                f"support_rows_h{horizon}"
            ],
            errors="coerce",
        )
        .fillna(0)
        .gt(0)
    ].copy()

    if metric in COUNT_METRICS:
        observed = float(
            pd.to_numeric(
                scoped[
                    f"observed_level_h{horizon}"
                ],
                errors="coerce",
            ).sum()
        )

        no_cp = float(
            pd.to_numeric(
                scoped[
                    f"no_cp_level_h{horizon}"
                ],
                errors="coerce",
            ).sum()
        )

        return observed, no_cp

    observed_weights = pd.to_numeric(
        scoped[
            f"observed_weight_h{horizon}"
        ],
        errors="coerce",
    )

    no_cp_weights = pd.to_numeric(
        scoped[
            f"no_cp_weight_h{horizon}"
        ],
        errors="coerce",
    )

    observed_levels = pd.to_numeric(
        scoped[
            f"observed_level_h{horizon}"
        ],
        errors="coerce",
    )

    no_cp_levels = pd.to_numeric(
        scoped[
            f"no_cp_level_h{horizon}"
        ],
        errors="coerce",
    )

    observed_denominator = float(
        observed_weights.sum()
    )

    no_cp_denominator = float(
        no_cp_weights.sum()
    )

    observed = float(
        (
            observed_levels
            * observed_weights
        ).sum()
        / observed_denominator
    )

    no_cp = float(
        (
            no_cp_levels
            * no_cp_weights
        ).sum()
        / no_cp_denominator
    )

    return observed, no_cp


# The authoritative source-ID rebuild preserves the same total mass before
# canonical aliases are merged. Compare that source system against the compact
# canonical system for every Metric × Horizon job.
system_qa_rows = []

for metric in METRICS:
    for horizon in HORIZONS:
        source_job = rebuilt_source_zones.loc[
            rebuilt_source_zones[
                "metric"
            ].eq(
                metric
            )
            & rebuilt_source_zones[
                "horizon"
            ].eq(
                horizon
            )
        ].copy()

        if metric in COUNT_METRICS:
            source_observed = float(
                source_job[
                    "observed_level"
                ].sum()
            )

            source_no_cp = float(
                source_job[
                    "no_cp_level"
                ].sum()
            )


        else:

            # Re-read the native speed rows because the source-zone QA table

            # stores final weighted levels, not the underlying sufficient

            # statistics needed to reconstruct the systemwide weighted mean.

            raw_job = pd.read_parquet(

                GAP_SURFACE_PATH,

                columns=[

                    "metric",

                    "horizon",

                    "primary_gap_eligible",

                    "target_observed_value",

                    "system_prediction",

                    "observed_activity_weight",

                    "no_cp_activity_weight",

                ],

                filters=[

                    (

                        "metric",

                        "==",

                        metric,

                    ),

                    (

                        "horizon",

                        "==",

                        horizon,

                    ),

                ],

            )

            raw_job = raw_job.loc[

                raw_job[

                    "primary_gap_eligible"

                ].astype(bool)

            ].copy()

            raw_job = valid_speed_weight_rows(

                raw_job,

                label=f"{metric} h={horizon} system reconciliation",

            )

            observed_weights = pd.to_numeric(

                raw_job[

                    "observed_activity_weight"

                ],

                errors="coerce",

            )

            no_cp_weights = pd.to_numeric(

                raw_job[

                    "no_cp_activity_weight"

                ],

                errors="coerce",

            )

            observed_values = pd.to_numeric(

                raw_job[

                    "target_observed_value"

                ],

                errors="coerce",

            )

            no_cp_values = pd.to_numeric(

                raw_job[

                    "system_prediction"

                ],

                errors="coerce",

            )

            source_observed_denominator = float(

                observed_weights.sum()

            )

            source_no_cp_denominator = float(

                no_cp_weights.sum()

            )

            if source_observed_denominator <= 0:
                raise RuntimeError(

                    f"{metric} h={horizon}: "

                    "system observed speed denominator is not positive."

                )

            if source_no_cp_denominator <= 0:
                raise RuntimeError(

                    f"{metric} h={horizon}: "

                    "system no-CP speed denominator is not positive."

                )

            source_observed = float(

                (

                        observed_values

                        * observed_weights

                ).sum()

                / source_observed_denominator

            )

            source_no_cp = float(

                (

                        no_cp_values

                        * no_cp_weights

                ).sum()

                / source_no_cp_denominator

            )

            del raw_job

            gc.collect()

        compact_observed, compact_no_cp = (
            aggregate_output_job(
                result,
                metric=metric,
                horizon=horizon,
            )
        )

        observed_diff = abs(
            compact_observed
            - source_observed
        )

        no_cp_diff = abs(
            compact_no_cp
            - source_no_cp
        )

        system_qa_rows.append(
            {
                "metric": metric,
                "horizon": horizon,
                "observed_abs_diff":
                    observed_diff,
                "no_cp_abs_diff":
                    no_cp_diff,
            }
        )

system_qa = pd.DataFrame(
    system_qa_rows
)

max_system_observed_diff = float(
    system_qa[
        "observed_abs_diff"
    ].max()
)

max_system_no_cp_diff = float(
    system_qa[
        "no_cp_abs_diff"
    ].max()
)

add_qa(
    "canonicalization_preserves_system_observed",
    bool(
        np.allclose(
            system_qa[
                "observed_abs_diff"
            ],
            0,
            rtol=1e-9,
            atol=1e-6,
        )
    ),
    (
        "max abs difference="
        f"{max_system_observed_diff:.12g}"
    ),
)

add_qa(
    "canonicalization_preserves_system_no_cp",
    bool(
        np.allclose(
            system_qa[
                "no_cp_abs_diff"
            ],
            0,
            rtol=1e-9,
            atol=1e-6,
        )
    ),
    (
        "max abs difference="
        f"{max_system_no_cp_diff:.12g}"
    ),
)


# ---------------------------------------------------------------------
# Write only after QA is known
# ---------------------------------------------------------------------

qa = pd.DataFrame(
    qa_rows
)

failed = qa.loc[
    qa[
        "status"
    ].ne(
        "PASS"
    )
]

print()
print(
    "=" * 88
)
print(
    "RAW 22 PREAGGREGATE QA"
)
print(
    "=" * 88
)
print(
    qa.to_string(
        index=False
    )
)

if not failed.empty:
    raise RuntimeError(
        "Raw 22 preaggregate QA failed. "
        "Nothing was written."
    )


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result.to_parquet(
    OUTPUT_PATH,
    index=False,
    compression="zstd",
)

qa.to_parquet(
    QA_PATH,
    index=False,
    compression="zstd",
)


# ---------------------------------------------------------------------
# Final handoff summary
# ---------------------------------------------------------------------

output_size_mb = (
    OUTPUT_PATH.stat().st_size
    / 1024**2
)

qa_size_mb = (
    QA_PATH.stat().st_size
    / 1024**2
)

print()
print(
    "=" * 88
)
print(
    "RAW 22 PRODUCTION HANDOFF"
)
print(
    "=" * 88
)

print(
    f"Output rows: {len(result):,}"
)

print(
    "Canonical Taxi Zones: "
    f"{result['canonical_taxi_zone_id'].nunique():,}"
)

print(
    "Reader-facing physical Taxi Zones: "
    f"{result.loc[result['reader_facing_zone'], 'canonical_taxi_zone_id'].nunique():,}"
)

print(
    "Weeks: "
    f"{result['week_start'].nunique():,} · "
    f"{result['week_start'].min():%Y-%m-%d} to "
    f"{result['week_start'].max():%Y-%m-%d}"
)

print(
    "Temporal buckets: "
    f"{result['target_temporal_bucket'].nunique():,}"
)

print(
    "Metrics: "
    f"{result['metric'].nunique():,}"
)

print(
    f"Artifact: {OUTPUT_PATH}"
)

print(
    f"Artifact size: {output_size_mb:.2f} MB"
)

print(
    f"QA: {QA_PATH}"
)

print(
    f"QA size: {qa_size_mb:.2f} MB"
)

print()
print(
    "Full-period system reconciliation:"
)

print(
    system_qa.to_string(
        index=False
    )
)

print()
print(
    "All QA checks PASS."
)

print(
    "Raw 22 now has the sufficient statistics needed for maps, "
    "local-intensity comparisons, exact weighted speed aggregation, "
    "and dynamic Taxi Zone contribution rankings."
)