from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from app.data_access.anomalies import (
    ANOMALY_EVENT_UNIVERSE_PATH,
    ANOMALY_ZONE_FREQUENCY_PATH,
)
from app.data_access.loaders import (
    CONGESTION_PRICING_START_DATE,
    load_analysis_panel,
)


TOLERANCE = 1e-10

SUSPICIOUS_NAME_TOKENS = (
    "eligib",
    "support",
    "review",
    "retain",
    "valid",
    "shared",
    "finalist",
    "candidate",
    "comparison",
)


def _period_from_date(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(
        values,
        errors="coerce",
    )

    return pd.Series(
        np.where(
            dates
            < pd.Timestamp(
                CONGESTION_PRICING_START_DATE
            ),
            "pre_cp",
            "post_cp",
        ),
        index=values.index,
        dtype="string",
    )


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
                "outside",
            }
        ).fillna(False)
    ] = "Outside"

    return result


def _zone_period_counts(
    frame: pd.DataFrame,
    *,
    mask: pd.Series | np.ndarray,
    selected_mask: pd.Series | np.ndarray,
) -> pd.DataFrame:
    subset = frame.loc[
        np.asarray(mask, dtype=bool),
        [
            "taxi_zone_id",
            "period",
        ],
    ].copy()

    subset["selected_in_mask"] = (
        np.asarray(
            selected_mask,
            dtype=bool,
        )[
            np.asarray(
                mask,
                dtype=bool,
            )
        ]
    )

    result = (
        subset.groupby(
            [
                "taxi_zone_id",
                "period",
            ],
            observed=True,
            dropna=False,
        )
        .agg(
            denominator_rows=(
                "selected_in_mask",
                "size",
            ),
            selected_rows_in_mask=(
                "selected_in_mask",
                "sum",
            ),
        )
        .reset_index()
    )

    result[
        "selected_rows_in_mask"
    ] = (
        result[
            "selected_rows_in_mask"
        ]
        .astype(int)
    )

    return result


def _pivot_zone_period(
    frame: pd.DataFrame,
    value_column: str,
    prefix: str,
) -> pd.DataFrame:
    wide = (
        frame.pivot(
            index="taxi_zone_id",
            columns="period",
            values=value_column,
        )
        .reset_index()
        .rename_axis(
            None,
            axis=1,
        )
    )

    rename_map = {
        "pre_cp": f"{prefix}_pre",
        "post_cp": f"{prefix}_post",
    }

    return wide.rename(
        columns=rename_map
    )


def _score_candidate(
    *,
    candidate_name: str,
    denominator_counts: pd.DataFrame,
    all_selected_counts: pd.DataFrame,
    saved: pd.DataFrame,
) -> dict[str, object]:
    den_wide = _pivot_zone_period(
        denominator_counts,
        "denominator_rows",
        "den",
    )

    selected_in_mask_wide = _pivot_zone_period(
        denominator_counts,
        "selected_rows_in_mask",
        "selected_mask",
    )

    merged = (
        saved.merge(
            den_wide,
            on="taxi_zone_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            selected_in_mask_wide,
            on="taxi_zone_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            all_selected_counts,
            on="taxi_zone_id",
            how="left",
            validate="one_to_one",
        )
    )

    for col in [
        "den_pre",
        "den_post",
        "selected_mask_pre",
        "selected_mask_post",
        "selected_all_pre",
        "selected_all_post",
    ]:
        if col not in merged.columns:
            merged[col] = 0

        merged[col] = (
            merged[col]
            .fillna(0)
            .astype(float)
        )

    merged[
        "recon_mask_pre"
    ] = np.where(
        merged["den_pre"] > 0,
        merged["selected_mask_pre"]
        / merged["den_pre"],
        np.nan,
    )

    merged[
        "recon_mask_post"
    ] = np.where(
        merged["den_post"] > 0,
        merged["selected_mask_post"]
        / merged["den_post"],
        np.nan,
    )

    merged[
        "recon_all_pre"
    ] = np.where(
        merged["den_pre"] > 0,
        merged["selected_all_pre"]
        / merged["den_pre"],
        np.nan,
    )

    merged[
        "recon_all_post"
    ] = np.where(
        merged["den_post"] > 0,
        merged["selected_all_post"]
        / merged["den_post"],
        np.nan,
    )

    saved_pre = (
        merged[
            "pre_cp_event_share_of_eligible_rows"
        ]
    )

    saved_post = (
        merged[
            "post_cp_event_share_of_eligible_rows"
        ]
    )

    mask_pre_diff = (
        merged["recon_mask_pre"]
        - saved_pre
    ).abs()

    mask_post_diff = (
        merged["recon_mask_post"]
        - saved_post
    ).abs()

    all_pre_diff = (
        merged["recon_all_pre"]
        - saved_pre
    ).abs()

    all_post_diff = (
        merged["recon_all_post"]
        - saved_post
    ).abs()

    mask_bad = (
        mask_pre_diff.gt(TOLERANCE)
        | mask_post_diff.gt(TOLERANCE)
    )

    all_bad = (
        all_pre_diff.gt(TOLERANCE)
        | all_post_diff.gt(TOLERANCE)
    )

    total_denominator = int(
        denominator_counts[
            "denominator_rows"
        ].sum()
    )

    total_selected_in_mask = int(
        denominator_counts[
            "selected_rows_in_mask"
        ].sum()
    )

    return {
        "candidate": candidate_name,
        "denominator_rows": (
            total_denominator
        ),
        "selected_rows_in_mask": (
            total_selected_in_mask
        ),
        "mask_max_abs_diff": float(
            max(
                mask_pre_diff.max(
                    skipna=True
                ),
                mask_post_diff.max(
                    skipna=True
                ),
            )
        ),
        "mask_bad_zones": int(
            mask_bad.fillna(True).sum()
        ),
        "all_selected_max_abs_diff": float(
            max(
                all_pre_diff.max(
                    skipna=True
                ),
                all_post_diff.max(
                    skipna=True
                ),
            )
        ),
        "all_selected_bad_zones": int(
            all_bad.fillna(True).sum()
        ),
    }


print("=" * 96)
print("PAGE 13 — FREQUENCY / DENOMINATOR CONTRACT DIAGNOSTIC")
print("=" * 96)

# ---------------------------------------------------------------------
# 1. Inspect the full-universe schema for plausible eligibility fields.
# ---------------------------------------------------------------------
parquet_file = pq.ParquetFile(
    ANOMALY_EVENT_UNIVERSE_PATH
)

schema_names = (
    parquet_file.schema_arrow.names
)

candidate_columns = [
    name
    for name in schema_names
    if any(
        token in name.lower()
        for token in SUSPICIOUS_NAME_TOKENS
    )
]

required_columns = [
    "taxi_zone_id",
    "date",
    "selected_finalist_flag",
]

read_columns = list(
    dict.fromkeys(
        required_columns
        + candidate_columns
    )
)

print()
print("1. PLAUSIBLE ELIGIBILITY / SUPPORT FIELDS")
for column in candidate_columns:
    print(f"  {column}")

universe = pd.read_parquet(
    ANOMALY_EVENT_UNIVERSE_PATH,
    columns=read_columns,
)

universe["period"] = (
    _period_from_date(
        universe["date"]
    )
)

selected_mask = (
    universe[
        "selected_finalist_flag"
    ]
    .fillna(False)
    .astype(bool)
)

print()
print(
    f"Full universe rows: {len(universe):,}"
)
print(
    "Canonical selected-finalist rows: "
    f"{int(selected_mask.sum()):,}"
)

# ---------------------------------------------------------------------
# 2. Print low-cardinality values for candidate columns.
# ---------------------------------------------------------------------
print()
print("2. LOW-CARDINALITY CANDIDATE VALUES")

boolean_like_columns: list[str] = []

for column in candidate_columns:
    series = universe[column]

    unique_values = (
        series.dropna()
        .drop_duplicates()
    )

    unique_count = int(
        unique_values.shape[0]
    )

    if unique_count <= 12:
        counts = (
            series.value_counts(
                dropna=False
            )
        )

        print()
        print(
            f"{column} "
            f"(dtype={series.dtype}, unique={unique_count})"
        )

        for value, count in counts.items():
            print(
                f"  {repr(value)}: "
                f"{int(count):,}"
            )

    non_null = (
        series.dropna()
    )

    if (
        pd.api.types.is_bool_dtype(
            non_null.dtype
        )
        or (
            not non_null.empty
            and set(
                non_null.astype(str)
                .str.lower()
                .unique()
            ).issubset(
                {
                    "true",
                    "false",
                    "0",
                    "1",
                }
            )
        )
    ):
        boolean_like_columns.append(
            column
        )

# ---------------------------------------------------------------------
# 3. Load saved All-3 frequency surface.
# ---------------------------------------------------------------------
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

if all3.empty:
    raise ValueError(
        "Could not identify the All-3 candidate surface."
    )

if all3.duplicated(
    "taxi_zone_id"
).any():
    raise ValueError(
        "All-3 frequency surface is not unique by Taxi Zone."
    )

print()
print("3. SAVED ALL-3 FREQUENCY SURFACE")
print(
    f"Rows: {len(all3):,}"
)

# ---------------------------------------------------------------------
# 4. Build canonical selected-event counts by zone × period.
# ---------------------------------------------------------------------
selected_counts_long = (
    universe.loc[
        selected_mask,
        [
            "taxi_zone_id",
            "period",
        ],
    ]
    .groupby(
        [
            "taxi_zone_id",
            "period",
        ],
        observed=True,
        dropna=False,
    )
    .size()
    .rename(
        "selected_all"
    )
    .reset_index()
)

selected_all_wide = (
    _pivot_zone_period(
        selected_counts_long,
        "selected_all",
        "selected_all",
    )
)

# ---------------------------------------------------------------------
# 5. Reverse-engineer the implied eligible counts from saved shares.
#    Do this twice:
#    A) all canonical selected events as numerator;
#    B) comparison_group_support_review_flag-selected events if available.
# ---------------------------------------------------------------------
print()
print("4. IMPLIED DENOMINATORS FROM SAVED SHARES")

implied = all3.merge(
    selected_all_wide,
    on="taxi_zone_id",
    how="left",
    validate="one_to_one",
)

for col in [
    "selected_all_pre",
    "selected_all_post",
]:
    if col not in implied.columns:
        implied[col] = 0

    implied[col] = (
        implied[col]
        .fillna(0)
        .astype(float)
    )

for period, share_col, numerator_col in [
    (
        "pre",
        "pre_cp_event_share_of_eligible_rows",
        "selected_all_pre",
    ),
    (
        "post",
        "post_cp_event_share_of_eligible_rows",
        "selected_all_post",
    ),
]:
    implied_col = (
        f"implied_den_{period}"
    )

    implied[implied_col] = np.where(
        implied[share_col] > 0,
        implied[numerator_col]
        / implied[share_col],
        np.nan,
    )

    integer_distance = (
        implied[implied_col]
        - implied[implied_col].round()
    ).abs()

    valid = (
        implied[implied_col]
        .notna()
    )

    print()
    print(
        f"{period.upper()} using ALL selected finalists as numerator"
    )
    print(
        "  zones with non-zero saved share: "
        f"{int(valid.sum()):,}"
    )
    print(
        "  max distance from integer implied denominator: "
        f"{integer_distance[valid].max():.12f}"
    )
    print(
        "  zones > 1e-6 from integer: "
        f"{int(integer_distance[valid].gt(1e-6).sum()):,}"
    )

# ---------------------------------------------------------------------
# 6. Score obvious candidate row universes.
# ---------------------------------------------------------------------
print()
print("5. CANDIDATE DENOMINATOR RECONSTRUCTION SCORES")

candidate_masks: dict[
    str,
    np.ndarray,
] = {
    "ALL_ROWS": np.ones(
        len(universe),
        dtype=bool,
    )
}

for column in boolean_like_columns:
    values = universe[column]

    if pd.api.types.is_bool_dtype(
        values.dtype
    ):
        mask = (
            values.fillna(False)
            .to_numpy(dtype=bool)
        )
    else:
        lowered = (
            values.astype("string")
            .str.lower()
        )
        mask = (
            lowered.isin(
                [
                    "true",
                    "1",
                ]
            )
            .fillna(False)
            .to_numpy(dtype=bool)
        )

    candidate_masks[
        f"{column}=True"
    ] = mask

scores: list[
    dict[str, object]
] = []

for candidate_name, mask in (
    candidate_masks.items()
):
    counts = _zone_period_counts(
        universe,
        mask=mask,
        selected_mask=(
            selected_mask
        ),
    )

    scores.append(
        _score_candidate(
            candidate_name=(
                candidate_name
            ),
            denominator_counts=(
                counts
            ),
            all_selected_counts=(
                selected_all_wide
            ),
            saved=all3,
        )
    )

scores_frame = (
    pd.DataFrame(
        scores
    )
    .sort_values(
        [
            "mask_bad_zones",
            "mask_max_abs_diff",
            "all_selected_bad_zones",
            "all_selected_max_abs_diff",
        ],
        ascending=True,
    )
    .reset_index(
        drop=True
    )
)

for rank, row in (
    scores_frame.iterrows()
):
    print(
        (
            f"  {rank + 1}. {row['candidate']} | "
            f"den={int(row['denominator_rows']):,} | "
            f"selected-in-mask={int(row['selected_rows_in_mask']):,} | "
            f"numerator=selected-in-mask: "
            f"bad_zones={int(row['mask_bad_zones']):,}, "
            f"max_diff={float(row['mask_max_abs_diff']):.12f} | "
            f"numerator=ALL-selected: "
            f"bad_zones={int(row['all_selected_bad_zones']):,}, "
            f"max_diff={float(row['all_selected_max_abs_diff']):.12f}"
        )
    )

# ---------------------------------------------------------------------
# 7. Identify the policy-geography unknowns from the canonical backbone.
# ---------------------------------------------------------------------
print()
print("6. POLICY-GEOGRAPHY UNKNOWN ZONE(S)")

zone_geo_source = load_analysis_panel(
    columns=[
        "taxi_zone_id",
        "zone",
        "borough",
        "cbd_spatial_category",
    ]
)

zone_geo = (
    zone_geo_source[
        [
            "taxi_zone_id",
            "zone",
            "borough",
            "cbd_spatial_category",
        ]
    ]
    .drop_duplicates()
)

if (
    zone_geo.groupby(
        "taxi_zone_id",
        observed=True,
        dropna=False,
    )[
        "cbd_spatial_category"
    ]
    .nunique(
        dropna=False
    )
    .gt(1)
    .any()
):
    raise ValueError(
        "cbd_spatial_category is not stable by Taxi Zone."
    )

zone_geo = (
    zone_geo.drop_duplicates(
        "taxi_zone_id"
    )
)

zone_geo[
    "policy_geography"
] = _policy_geography(
    zone_geo[
        "cbd_spatial_category"
    ]
)

unknown_geo = (
    zone_geo[
        zone_geo[
            "policy_geography"
        ].eq("Unknown")
    ]
    .sort_values(
        "taxi_zone_id"
    )
)

if unknown_geo.empty:
    print("  None")
else:
    for _, row in (
        unknown_geo.iterrows()
    ):
        print(
            (
                f"  taxi_zone_id={row['taxi_zone_id']} | "
                f"zone={row['zone']} | "
                f"borough={row['borough']} | "
                f"cbd_spatial_category={repr(row['cbd_spatial_category'])}"
            )
        )

# ---------------------------------------------------------------------
# 8. Final concise guidance.
# ---------------------------------------------------------------------
best = (
    scores_frame.iloc[0]
    if not scores_frame.empty
    else None
)

print()
print("7. NEXT-STEP SUMMARY")

if best is None:
    print(
        "  No candidate masks could be scored."
    )
else:
    print(
        "  Best single-field candidate: "
        f"{best['candidate']}"
    )
    print(
        "  Saved-share mismatched zones "
        "(selected-in-mask numerator): "
        f"{int(best['mask_bad_zones']):,}"
    )
    print(
        "  Saved-share mismatched zones "
        "(all-selected numerator): "
        f"{int(best['all_selected_bad_zones']):,}"
    )

print()
print(
    "Copy this entire terminal output back into ChatGPT. "
    "Do not change Page 13 incidence logic yet."
)
