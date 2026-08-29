from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.data_access.anomalies import (
    ANOMALY_EVENT_UNIVERSE_PATH,
    ANOMALY_ZONE_FREQUENCY_PATH,
)
from app.data_access.loaders import (
    CONGESTION_PRICING_START_DATE,
    load_analysis_panel,
)
from app.data_access.mobility_environments import (
    load_mobility_regime_cluster_assignments,
)


TOLERANCE = 1e-10


def _policy_geography(values: pd.Series) -> pd.Series:
    normalized = (
        values.astype("string")
        .str.strip()
        .str.lower()
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )

    result = pd.Series(
        "Unknown",
        index=values.index,
        dtype="string",
    )

    result.loc[
        normalized.eq("cbd").fillna(False)
    ] = "CBD"

    result.loc[
        normalized.isin(
            {
                "gateway_to_cbd",
                "gateway",
                "adjacent_to_cbd",
                "adjacent",
            }
        ).fillna(False)
    ] = "Gateway + adjacent"

    result.loc[
        normalized.isin(
            {
                "non_cbd",
                "noncbd",
            }
        ).fillna(False)
    ] = "Outside"

    return result


def _print_group_counts(
    frame: pd.DataFrame,
    column: str,
    label: str,
) -> None:
    print()
    print(label)
    counts = (
        frame[column]
        .fillna("Unknown")
        .astype(str)
        .value_counts(dropna=False)
    )

    for value, count in counts.items():
        print(
            f"  {value}: {int(count):,}"
        )


print("=" * 88)
print("PAGE 13 DENOMINATOR DIAGNOSTIC")
print("=" * 88)

# ---------------------------------------------------------------------
# 1. Load only the row-grain fields needed from the 3.3.6 universe.
#    Do NOT trust its enrichment-only geography/environment columns for
#    non-selected rows; those are what Phase 1 accidentally used.
# ---------------------------------------------------------------------
universe = pd.read_parquet(
    ANOMALY_EVENT_UNIVERSE_PATH,
    columns=[
        "taxi_zone_id",
        "date",
        "daypart",
        "temporal_bucket",
        "comparison_group_support_review_flag",
        "selected_finalist_flag",
    ],
)

universe["date"] = pd.to_datetime(
    universe["date"],
    errors="coerce",
)

duplicate_grain = int(
    universe.duplicated(
        [
            "taxi_zone_id",
            "date",
            "daypart",
        ]
    ).sum()
)

eligible = universe[
    universe[
        "comparison_group_support_review_flag"
    ]
    .fillna(False)
    .astype(bool)
].copy()

eligible["pre_post_cp"] = np.where(
    eligible["date"]
    < pd.Timestamp(
        CONGESTION_PRICING_START_DATE
    ),
    "pre_cp",
    "post_cp",
)

print()
print("1. BASE ELIGIBILITY")
print(
    f"Full universe rows: {len(universe):,}"
)
print(
    f"Duplicate Taxi Zone × date × daypart rows: {duplicate_grain:,}"
)
print(
    f"Support-eligible rows: {len(eligible):,}"
)
print(
    "Selected finalist rows inside eligible universe: "
    f"{int(eligible['selected_finalist_flag'].fillna(False).astype(bool).sum()):,}"
)

_print_group_counts(
    eligible,
    "pre_post_cp",
    "Derived policy-period coverage",
)

# ---------------------------------------------------------------------
# 2. Attach stable policy geography from the canonical 1.3.1 backbone.
# ---------------------------------------------------------------------
zone_geo_source = load_analysis_panel(
    columns=[
        "taxi_zone_id",
        "cbd_spatial_category",
    ]
)

zone_geo = (
    zone_geo_source[
        [
            "taxi_zone_id",
            "cbd_spatial_category",
        ]
    ]
    .drop_duplicates()
)

geo_duplicate_zone_count = int(
    zone_geo.duplicated(
        "taxi_zone_id",
        keep=False,
    ).sum()
)

if geo_duplicate_zone_count:
    conflict_counts = (
        zone_geo.groupby(
            "taxi_zone_id",
            observed=True,
            dropna=False,
        )[
            "cbd_spatial_category"
        ]
        .nunique(dropna=False)
    )

    conflicting_zones = int(
        conflict_counts.gt(1).sum()
    )

    if conflicting_zones:
        raise ValueError(
            "Policy geography is not stable by Taxi Zone: "
            f"{conflicting_zones:,} conflicting zones."
        )

    zone_geo = (
        zone_geo.drop_duplicates(
            "taxi_zone_id"
        )
    )

eligible = eligible.merge(
    zone_geo,
    on="taxi_zone_id",
    how="left",
    validate="many_to_one",
)

eligible["policy_geography"] = (
    _policy_geography(
        eligible[
            "cbd_spatial_category"
        ]
    )
)

print()
print("2. POLICY GEOGRAPHY")
print(
    "Taxi Zones in stable geography lookup: "
    f"{zone_geo['taxi_zone_id'].nunique():,}"
)

_print_group_counts(
    eligible,
    "policy_geography",
    "Eligible-row geography coverage",
)

# ---------------------------------------------------------------------
# 3. Attach period-specific mobility environment using the canonical
#    Taxi Zone × pre/post assignment table.
# ---------------------------------------------------------------------
assignments = (
    load_mobility_regime_cluster_assignments()[
        [
            "taxi_zone_id",
            "pre_post_cp",
            "canonical_cluster_name",
        ]
    ]
    .drop_duplicates()
)

assignment_duplicates = int(
    assignments.duplicated(
        [
            "taxi_zone_id",
            "pre_post_cp",
        ],
        keep=False,
    ).sum()
)

if assignment_duplicates:
    raise ValueError(
        "Mobility-environment assignments are not unique at "
        "Taxi Zone × pre/post period."
    )

eligible = eligible.merge(
    assignments.rename(
        columns={
            "canonical_cluster_name": (
                "mobility_environment"
            )
        }
    ),
    on=[
        "taxi_zone_id",
        "pre_post_cp",
    ],
    how="left",
    validate="many_to_one",
)

eligible[
    "mobility_environment"
] = (
    eligible[
        "mobility_environment"
    ]
    .astype("string")
    .fillna("Unknown")
)

print()
print("3. MOBILITY ENVIRONMENT")
print(
    "Canonical Taxi Zone × period assignments: "
    f"{len(assignments):,}"
)

_print_group_counts(
    eligible,
    "mobility_environment",
    "Eligible-row environment coverage",
)

# ---------------------------------------------------------------------
# 4. Gold-standard check: reconstruct the canonical All-3 pre/post
#    zone event shares and compare them with the final 3.3.6 frequency
#    handoff. This validates the denominator definition, not just the joins.
# ---------------------------------------------------------------------
zone_period = (
    eligible.groupby(
        [
            "taxi_zone_id",
            "pre_post_cp",
        ],
        observed=True,
        dropna=False,
    )
    .agg(
        eligible_rows=(
            "selected_finalist_flag",
            "size",
        ),
        selected_events=(
            "selected_finalist_flag",
            lambda values: int(
                values.fillna(False)
                .astype(bool)
                .sum()
            ),
        ),
    )
    .reset_index()
)

zone_period[
    "reconstructed_event_share"
] = (
    zone_period[
        "selected_events"
    ]
    / zone_period[
        "eligible_rows"
    ]
)

reconstructed = (
    zone_period.pivot(
        index="taxi_zone_id",
        columns="pre_post_cp",
        values=(
            "reconstructed_event_share"
        ),
    )
    .reset_index()
    .rename(
        columns={
            "pre_cp": (
                "reconstructed_pre_share"
            ),
            "post_cp": (
                "reconstructed_post_share"
            ),
        }
    )
)

frequency = pd.read_parquet(
    ANOMALY_ZONE_FREQUENCY_PATH
)

all3 = frequency[
    frequency[
        "candidate_surface_label"
    ]
    .astype(str)
    .str.contains(
        "All 3",
        case=False,
        na=False,
    )
].copy()

print()
print("4. CANONICAL ZONE-FREQUENCY CROSS-CHECK")
print(
    "Candidate surfaces in final frequency handoff: "
    + ", ".join(
        sorted(
            frequency[
                "candidate_surface_label"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    )
)

if all3.empty:
    raise ValueError(
        "Could not find the All-3 candidate surface in "
        "finalist_prepost_zone_frequency.parquet."
    )

comparison = all3.merge(
    reconstructed,
    on="taxi_zone_id",
    how="inner",
    validate="one_to_one",
)

comparison[
    "pre_abs_diff"
] = (
    comparison[
        "pre_cp_event_share_of_eligible_rows"
    ]
    - comparison[
        "reconstructed_pre_share"
    ]
).abs()

comparison[
    "post_abs_diff"
] = (
    comparison[
        "post_cp_event_share_of_eligible_rows"
    ]
    - comparison[
        "reconstructed_post_share"
    ]
).abs()

print(
    f"All-3 frequency rows: {len(all3):,}"
)
print(
    f"Matched reconstructed zones: {len(comparison):,}"
)
print(
    "Maximum pre-CP absolute share difference: "
    f"{comparison['pre_abs_diff'].max():.12f}"
)
print(
    "Maximum post-CP absolute share difference: "
    f"{comparison['post_abs_diff'].max():.12f}"
)
print(
    "Zones outside tolerance "
    f"({TOLERANCE:g}): "
    f"{int((comparison['pre_abs_diff'].gt(TOLERANCE) | comparison['post_abs_diff'].gt(TOLERANCE)).sum()):,}"
)

# ---------------------------------------------------------------------
# 5. Final pass/fail summary.
# ---------------------------------------------------------------------
unknown_period = int(
    eligible[
        "pre_post_cp"
    ].isna().sum()
)

unknown_geo = int(
    eligible[
        "policy_geography"
    ].eq("Unknown").sum()
)

unknown_environment = int(
    eligible[
        "mobility_environment"
    ].eq("Unknown").sum()
)

frequency_match = bool(
    (
        comparison[
            "pre_abs_diff"
        ].le(TOLERANCE)
        | comparison[
            "pre_abs_diff"
        ].isna()
    ).all()
    and (
        comparison[
            "post_abs_diff"
        ].le(TOLERANCE)
        | comparison[
            "post_abs_diff"
        ].isna()
    ).all()
)

print()
print("5. SUMMARY")
print(
    f"Unknown policy-period eligible rows: {unknown_period:,}"
)
print(
    f"Unknown policy-geography eligible rows: {unknown_geo:,}"
)
print(
    f"Unknown mobility-environment eligible rows: {unknown_environment:,}"
)
print(
    "Canonical All-3 zone-frequency reconstruction: "
    + (
        "PASS"
        if frequency_match
        else "FAIL"
    )
)

if (
    unknown_period == 0
    and unknown_geo == 0
    and unknown_environment == 0
    and frequency_match
):
    print()
    print(
        "RESULT: denominator backbone is fully validated for Page 13 incidence."
    )
else:
    print()
    print(
        "RESULT: inspect the non-zero unknown counts and/or frequency mismatch "
        "before exposing incidence in Page 13."
    )
