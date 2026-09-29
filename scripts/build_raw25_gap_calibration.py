"""
Raw 25 Phase 2.5 scout.

Run once from the Showcase repo root:

    python scripts/build_raw25_gap_calibration.py

This replaces the earlier Phase-2 preprocessor. It does not finalize the
interactive explorer. Instead it profiles the calibration evidence so we can
choose the hero and explorer from the actual distribution shape.

Authoritative inputs:
- counterfactual_gap_surface.parquet
- counterfactual_pre_cp_calibration.parquet

Outputs:
- counterfactual_raw25_job_histogram.parquet
- counterfactual_raw25_job_summary.parquet
- counterfactual_raw25_slice_scout.parquet
- counterfactual_raw25_qa.parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    "taxi_trip_count",
    "taxi_avg_trip_speed",
    "fhvhv_trip_count",
    "fhvhv_avg_trip_speed",
    "subway_ridership",
]
HORIZONS = [1, 2, 5]

SOURCE_COLUMNS = [
    "taxi_zone_id",
    "canonical_location_id",
    "zone",
    "borough",
    "cbd_spatial_category",
    "pre_cp_mobility_environment",
    "metric",
    "horizon",
    "target_temporal_bucket",
    "counterfactual_gap",
    "counterfactual_gap_mae_units",
    "primary_gap_eligible",
    "pre_cp_validation_system_mae",
]

THRESHOLDS = [0.5, 1.0, 2.0, 3.0]
QUANTILES = {
    "p50_abs_mae_units": 0.50,
    "p75_abs_mae_units": 0.75,
    "p90_abs_mae_units": 0.90,
    "p95_abs_mae_units": 0.95,
    "p99_abs_mae_units": 0.99,
}

FINITE_MAX = 8.0
BIN_WIDTH = 0.20
MIN_SLICE_ROWS = 100


def repo_root() -> Path:
    """Find the Showcase root without hard-coding a local user path."""
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (
            candidate
            / "data/processed"
            / "5.3.1.final_tables"
        ).exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate data/processed/5.3.1.final_tables."
    )


def require_columns(
    frame: pd.DataFrame,
    required: list[str] | set[str],
    label: str,
) -> None:
    """Fail loudly when an upstream contract changes."""
    missing = sorted(
        set(required)
        - set(frame.columns)
    )

    if missing:
        raise KeyError(
            f"{label} missing columns: "
            + ", ".join(missing)
        )


def summarize(
    values: pd.Series,
) -> dict[str, float]:
    """Return the calibration landmarks needed for scouting."""
    clean = pd.to_numeric(
        values,
        errors="coerce",
    ).abs()

    clean = clean.loc[
        clean.notna()
        & np.isfinite(clean)
    ]

    if clean.empty:
        return {}

    result = {
        "rows": int(len(clean)),
        "mean_abs_mae_units": float(clean.mean()),
        "max_abs_mae_units": float(clean.max()),
        "overflow_ge_8_mae_pct": float(
            100.0 * clean.gt(FINITE_MAX).mean()
        ),
    }

    for name, q in QUANTILES.items():
        result[name] = float(
            clean.quantile(q)
        )

    for threshold in THRESHOLDS:
        token = str(threshold).replace(".", "_")
        result[
            f"share_abs_ge_{token}_mae_pct"
        ] = float(
            100.0
            * clean.ge(threshold).mean()
        )

    return result


def add_time_labels(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Translate the frozen temporal bucket into reader controls."""
    result = frame.copy()

    bucket = result[
        "target_temporal_bucket"
    ].astype(str)

    result["day_type"] = np.where(
        bucket.str.startswith("weekday_"),
        "Weekdays",
        "Weekends",
    )

    result["daypart"] = (
        bucket.str.replace(
            r"^(weekday|weekend)_",
            "",
            regex=True,
        )
        .map(
            {
                "overnight": "Overnight",
                "am_peak": "AM peak",
                "midday": "Midday",
                "pm_peak": "PM peak",
                "evening": "Evening",
            }
        )
    )

    return result


def append_slice_groups(
    output: list[dict[str, object]],
    source: pd.DataFrame,
    *,
    geography_type: str,
    id_col: str | None,
    label_col: str | None,
) -> None:
    """
    Scout one geography lens at all reader-facing temporal combinations.

    This is intentionally a compact summary, not a new analytical result.
    """
    working = source.copy()

    if geography_type == "Systemwide":
        working["_geo_id"] = "systemwide"
        working["_geo_label"] = (
            "NYC supported system"
        )
    else:
        working["_geo_id"] = (
            working[id_col]
            .astype("string")
        )
        working["_geo_label"] = (
            working[label_col]
            .astype("string")
        )

    temporal_variants = [
        # fixed day type, fixed daypart, extra grouping columns
        ("All days", "All dayparts", []),
        (None, "All dayparts", ["day_type"]),
        ("All days", None, ["daypart"]),
        (None, None, ["day_type", "daypart"]),
    ]

    base = [
        "metric",
        "horizon",
        "_geo_id",
        "_geo_label",
    ]

    for fixed_day_type, fixed_daypart, extras in temporal_variants:
        group_cols = [
            *base,
            *extras,
        ]

        for keys, part in working.groupby(
            group_cols,
            observed=True,
            dropna=False,
            sort=False,
        ):
            if not isinstance(keys, tuple):
                keys = (keys,)

            key_map = dict(
                zip(group_cols, keys)
            )

            stats = summarize(
                part[
                    "counterfactual_gap_mae_units"
                ]
            )

            if not stats:
                continue

            output.append(
                {
                    "metric": str(
                        key_map["metric"]
                    ),
                    "horizon": int(
                        key_map["horizon"]
                    ),
                    "geography_type": geography_type,
                    "geography_id": str(
                        key_map["_geo_id"]
                    ),
                    "geography_label": str(
                        key_map["_geo_label"]
                    ),
                    "day_type": (
                        fixed_day_type
                        if fixed_day_type is not None
                        else str(
                            key_map["day_type"]
                        )
                    ),
                    "daypart": (
                        fixed_daypart
                        if fixed_daypart is not None
                        else str(
                            key_map["daypart"]
                        )
                    ),
                    **stats,
                }
            )


root = repo_root()
final_dir = (
    root
    / "data/processed"
    / "5.3.1.final_tables"
)

gap_path = (
    final_dir
    / "counterfactual_gap_surface.parquet"
)
calibration_path = (
    final_dir
    / "counterfactual_pre_cp_calibration.parquet"
)

print("Reading frozen 5.3.1 calibration summary...")
calibration = pd.read_parquet(
    calibration_path
)

print("Reading row-level gap fields needed for the Raw 25 scout...")
gap = pd.read_parquet(
    gap_path,
    columns=SOURCE_COLUMNS,
)

require_columns(
    gap,
    SOURCE_COLUMNS,
    "counterfactual_gap_surface",
)

gap["horizon"] = pd.to_numeric(
    gap["horizon"],
    errors="raise",
).astype(int)

gap["taxi_zone_id"] = pd.to_numeric(
    gap["taxi_zone_id"],
    errors="coerce",
).astype("Int64")

gap["canonical_location_id"] = pd.to_numeric(
    gap["canonical_location_id"],
    errors="coerce",
).astype("Int64")

gap["counterfactual_gap"] = pd.to_numeric(
    gap["counterfactual_gap"],
    errors="coerce",
)

gap["counterfactual_gap_mae_units"] = pd.to_numeric(
    gap["counterfactual_gap_mae_units"],
    errors="coerce",
)

gap["pre_cp_validation_system_mae"] = pd.to_numeric(
    gap["pre_cp_validation_system_mae"],
    errors="coerce",
)

gap = gap.loc[
    gap["metric"].isin(METRICS)
    & gap["horizon"].isin(HORIZONS)
    & gap["primary_gap_eligible"]
    .fillna(False)
    .astype(bool)
    & gap["counterfactual_gap_mae_units"].notna()
    & np.isfinite(
        gap["counterfactual_gap_mae_units"]
    )
].copy()

gap = add_time_labels(gap)

print(
    f"Eligible calibration rows: {len(gap):,}"
)


# ---------------------------------------------------------------------
# Job-level scout
# ---------------------------------------------------------------------

job_rows = []

for (
    metric,
    horizon,
), part in gap.groupby(
    ["metric", "horizon"],
    observed=True,
    sort=False,
):
    job_rows.append(
        {
            "metric": metric,
            "horizon": int(horizon),
            **summarize(
                part[
                    "counterfactual_gap_mae_units"
                ]
            ),
        }
    )

job_summary = pd.DataFrame(job_rows)

# Preserve the native Pre-CP forecast-error ruler used to construct ×MAE.
mae_contract = (
    gap.groupby(
        ["metric", "horizon"],
        observed=True,
        sort=False,
    )
    .agg(
        pre_cp_validation_system_mae=(
            "pre_cp_validation_system_mae",
            "first",
        ),
        unique_pre_cp_validation_system_mae=(
            "pre_cp_validation_system_mae",
            "nunique",
        ),
    )
    .reset_index()
)

invalid_mae_contract = mae_contract.loc[
    mae_contract["unique_pre_cp_validation_system_mae"].ne(1)
    | mae_contract["pre_cp_validation_system_mae"].isna()
    | mae_contract["pre_cp_validation_system_mae"].le(0)
]

if not invalid_mae_contract.empty:
    raise RuntimeError(
        "Expected exactly one positive Pre-CP validation system MAE "
        "per Metric × Horizon:\n"
        + invalid_mae_contract.to_string(index=False)
    )

mae_contract = mae_contract.drop(
    columns=["unique_pre_cp_validation_system_mae"]
)

job_summary = job_summary.merge(
    mae_contract,
    on=["metric", "horizon"],
    how="left",
    validate="one_to_one",
)

# Verify that this native MAE is exactly the denominator behind the frozen ×MAE field.
denominator_check = gap.loc[
    gap["counterfactual_gap"].notna()
    & gap["counterfactual_gap_mae_units"].notna()
    & gap["pre_cp_validation_system_mae"].notna()
    & gap["pre_cp_validation_system_mae"].gt(0),
    [
        "counterfactual_gap",
        "counterfactual_gap_mae_units",
        "pre_cp_validation_system_mae",
    ],
].copy()

if denominator_check.empty:
    raise RuntimeError(
        "No eligible rows were available to validate the Pre-CP MAE denominator."
    )

reconstructed_mae_units = (
    denominator_check["counterfactual_gap"]
    / denominator_check["pre_cp_validation_system_mae"]
)

max_mae_denominator_error = float(
    (
        reconstructed_mae_units
        - denominator_check["counterfactual_gap_mae_units"]
    )
    .abs()
    .max()
)

if not np.isclose(
    max_mae_denominator_error,
    0.0,
    atol=1e-10,
    rtol=0.0,
):
    raise RuntimeError(
        "Pre-CP validation system MAE does not reproduce "
        "counterfactual_gap_mae_units. "
        f"Maximum absolute difference: {max_mae_denominator_error:.12g}"
    )

# Preserve the frozen 5.3.1 summary alongside the richer scout.
job_summary = job_summary.merge(
    calibration,
    on=["metric", "horizon"],
    how="left",
    validate="one_to_one",
    suffixes=("", "_frozen"),
)


# ---------------------------------------------------------------------
# Job histogram for visual prototypes
# ---------------------------------------------------------------------

edges = np.arange(
    0.0,
    FINITE_MAX + BIN_WIDTH,
    BIN_WIDTH,
)

hist_rows = []

for (
    metric,
    horizon,
), part in gap.groupby(
    ["metric", "horizon"],
    observed=True,
    sort=False,
):
    values = (
        part[
            "counterfactual_gap_mae_units"
        ]
        .abs()
        .to_numpy(dtype=float)
    )

    total = len(values)

    counts, current_edges = np.histogram(
        values[
            values <= FINITE_MAX
        ],
        bins=edges,
    )

    for idx, count in enumerate(counts):
        left = float(
            current_edges[idx]
        )
        right = float(
            current_edges[idx + 1]
        )

        hist_rows.append(
            {
                "metric": metric,
                "horizon": int(horizon),
                "bin_left": left,
                "bin_right": right,
                "bin_mid": (
                    left + right
                ) / 2.0,
                "row_count": int(count),
                "density_pct": (
                    100.0 * count / total
                ),
                "overflow": False,
            }
        )

    overflow = int(
        np.sum(
            values > FINITE_MAX
        )
    )

    hist_rows.append(
        {
            "metric": metric,
            "horizon": int(horizon),
            "bin_left": FINITE_MAX,
            "bin_right": np.nan,
            "bin_mid": FINITE_MAX + BIN_WIDTH / 2.0,
            "row_count": overflow,
            "density_pct": (
                100.0 * overflow / total
            ),
            "overflow": True,
        }
    )

histogram = pd.DataFrame(hist_rows)


# ---------------------------------------------------------------------
# Prospective explorer support scout
# ---------------------------------------------------------------------

slice_rows = []

append_slice_groups(
    slice_rows,
    gap,
    geography_type="Systemwide",
    id_col=None,
    label_col=None,
)

append_slice_groups(
    slice_rows,
    gap,
    geography_type="Borough",
    id_col="borough",
    label_col="borough",
)

append_slice_groups(
    slice_rows,
    gap,
    geography_type="Policy geography",
    id_col="cbd_spatial_category",
    label_col="cbd_spatial_category",
)

append_slice_groups(
    slice_rows,
    gap,
    geography_type="Mobility environment",
    id_col="pre_cp_mobility_environment",
    label_col="pre_cp_mobility_environment",
)

# Reader-facing Taxi Zone uses the frozen physical canonical location.
zone_labels = (
    gap[
        [
            "canonical_location_id",
            "taxi_zone_id",
            "zone",
        ]
    ]
    .dropna(
        subset=[
            "canonical_location_id",
            "zone",
        ]
    )
    .sort_values(
        [
            "canonical_location_id",
            "taxi_zone_id",
        ]
    )
    .drop_duplicates(
        "canonical_location_id"
    )
    .set_index(
        "canonical_location_id"
    )["zone"]
)

taxi_source = gap.copy()
taxi_source["_physical_zone_label"] = (
    taxi_source[
        "canonical_location_id"
    ].map(zone_labels)
)

append_slice_groups(
    slice_rows,
    taxi_source,
    geography_type="Taxi Zone",
    id_col="canonical_location_id",
    label_col="_physical_zone_label",
)

slice_scout = pd.DataFrame(slice_rows)
slice_scout["support_ok"] = (
    slice_scout["rows"]
    .ge(MIN_SLICE_ROWS)
)


# ---------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------

qa_rows = []


def qa(
    check_id: str,
    passed: bool,
    details: str,
) -> None:
    """Append one QA row."""
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


qa(
    "job_summary_has_15_jobs",
    len(job_summary) == 15,
    f"rows={len(job_summary)}",
)

qa(
    "job_summary_unique",
    not job_summary.duplicated(
        ["metric", "horizon"]
    ).any(),
    "Metric × Horizon is unique.",
)

qa(
    "all_five_geography_lenses_present",
    set(
        slice_scout[
            "geography_type"
        ].unique()
    )
    == {
        "Systemwide",
        "Borough",
        "Policy geography",
        "Mobility environment",
        "Taxi Zone",
    },
    str(
        sorted(
            slice_scout[
                "geography_type"
            ].unique()
        )
    ),
)

qa(
    "one_pre_cp_validation_system_mae_per_job",
    (
        len(mae_contract) == 15
        and not mae_contract.duplicated(
            ["metric", "horizon"]
        ).any()
        and mae_contract["pre_cp_validation_system_mae"].notna().all()
        and mae_contract["pre_cp_validation_system_mae"].gt(0).all()
    ),
    (
        f"jobs={len(mae_contract)}; "
        "expected one positive MAE for each of 15 Metric × Horizon jobs."
    ),
)

qa(
    "pre_cp_mae_reproduces_mae_units",
    bool(
        np.isclose(
            max_mae_denominator_error,
            0.0,
            atol=1e-10,
            rtol=0.0,
        )
    ),
    (
        "max_abs_difference="
        f"{max_mae_denominator_error:.12g}"
    ),
)

# The richer scout must reproduce the frozen 5.3.1 calibration summary.
comparisons = [
    (
        "p50_abs_mae_units",
        "median_absolute_mae_units",
        "median",
    ),
    (
        "p90_abs_mae_units",
        "p90_absolute_mae_units",
        "p90",
    ),
    (
        "share_abs_ge_1_0_mae_pct",
        "share_abs_ge_1_mae_pct",
        "share_ge_1",
    ),
    (
        "share_abs_ge_2_0_mae_pct",
        "share_abs_ge_2_mae_pct",
        "share_ge_2",
    ),
]

for scout_col, frozen_col, label in comparisons:
    difference = (
        pd.to_numeric(
            job_summary[scout_col],
            errors="coerce",
        )
        - pd.to_numeric(
            job_summary[frozen_col],
            errors="coerce",
        )
    ).abs()

    qa(
        f"{label}_matches_frozen_5_3_1",
        bool(
            difference.le(1e-9).all()
        ),
        (
            "max_abs_difference="
            f"{difference.max():.12f}"
        ),
    )

qa_frame = pd.DataFrame(qa_rows)

failed = qa_frame.loc[
    qa_frame["status"].eq("FAIL")
]

if not failed.empty:
    raise RuntimeError(
        "Raw 25 Phase-2.5 QA failed:\n"
        + failed.to_string(index=False)
    )


# ---------------------------------------------------------------------
# Persist + print the dump we actually need for the next decision
# ---------------------------------------------------------------------

histogram_path = (
    final_dir
    / "counterfactual_raw25_job_histogram.parquet"
)
summary_path = (
    final_dir
    / "counterfactual_raw25_job_summary.parquet"
)
slice_path = (
    final_dir
    / "counterfactual_raw25_slice_scout.parquet"
)
qa_path = (
    final_dir
    / "counterfactual_raw25_qa.parquet"
)

histogram.to_parquet(
    histogram_path,
    index=False,
    compression="zstd",
)
job_summary.to_parquet(
    summary_path,
    index=False,
    compression="zstd",
)
slice_scout.to_parquet(
    slice_path,
    index=False,
    compression="zstd",
)
qa_frame.to_parquet(
    qa_path,
    index=False,
    compression="zstd",
)

print()
print("Raw 25 Phase-2.5 scout complete.")
print(
    f"  histogram:   {len(histogram):,} rows -> {histogram_path}"
)
print(
    f"  job summary: {len(job_summary):,} rows -> {summary_path}"
)
print(
    f"  slice scout: {len(slice_scout):,} rows -> {slice_path}"
)
print(
    f"  QA:          {len(qa_frame):,} rows -> {qa_path}"
)

print()
print("JOB-LEVEL CALIBRATION DUMP")
print(
    job_summary[
        [
            "metric",
            "horizon",
            "pre_cp_validation_system_mae",
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
        ]
    ]
    .sort_values(
        ["metric", "horizon"]
    )
    .to_string(index=False)
)

print()
print("PROSPECTIVE EXPLORER SUPPORT DUMP")
support = (
    slice_scout.groupby(
        [
            "geography_type",
            "day_type",
            "daypart",
        ],
        observed=True,
        dropna=False,
    )
    .agg(
        slices=("rows", "size"),
        min_rows=("rows", "min"),
        median_rows=("rows", "median"),
        supported_pct=(
            "support_ok",
            lambda values: (
                100.0 * values.mean()
            ),
        ),
    )
    .reset_index()
)

print(
    support.to_string(index=False)
)

print()
print("QA")
print(
    qa_frame.to_string(index=False)
)
