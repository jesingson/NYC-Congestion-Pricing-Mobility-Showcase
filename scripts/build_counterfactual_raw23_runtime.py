"""
Build the compact exact-row runtime artifact used by Showcase Raw 23.

This is a deployment/storage optimization only. It preserves the exact primary
Chapter 5 rows and values Raw 23 already consumes; no aggregation, rounding, or
quantization is applied.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "5.3.1.final_tables"
    / "counterfactual_gap_surface.parquet"
)

PROFILE_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "5.3.1.final_tables"
    / "counterfactual_mobility_profile_summary.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "app_tables"
)

OUTPUT_PATH = OUTPUT_DIR / "counterfactual_raw23_runtime.parquet"
QA_PATH = OUTPUT_DIR / "counterfactual_raw23_runtime_qa.parquet"

CP_START = pd.Timestamp("2025-01-05")
CP_END = pd.Timestamp("2026-03-31")

HORIZONS = [1, 2, 5]

METRIC_ORDER = [
    "taxi_trip_count",
    "taxi_avg_trip_speed",
    "fhvhv_trip_count",
    "fhvhv_avg_trip_speed",
    "subway_ridership",
]

SOURCE_COLUMNS = [
    "taxi_zone_id",
    "canonical_location_id",
    "zone",
    "borough",
    "cbd_spatial_category",
    "pre_cp_mobility_environment",
    "metric",
    "horizon",
    "target_date",
    "target_temporal_bucket",
    "counterfactual_gap_mae_units",
    "primary_gap_eligible",
]

RUNTIME_COLUMNS = [
    column
    for column in SOURCE_COLUMNS
    if column != "primary_gap_eligible"
]


def _require_file(path: Path, label: str) -> None:
    """Fail loudly when a required frozen input is absent."""
    if not path.exists():
        raise FileNotFoundError(
            f"{label} was not found:\n  {path}"
        )


def _mb(path: Path) -> float:
    """Return file size in MiB."""
    return path.stat().st_size / (1024 ** 2)


def _read_source() -> pd.DataFrame:
    """
    Read only Raw 23's required columns.

    WHY:
    Predicate pushdown avoids scanning irrelevant rows where the local Parquet
    engine supports it. The fallback still projects only the twelve required
    source columns and applies the identical filters in Pandas.
    """
    filters = [
        ("primary_gap_eligible", "==", True),
        ("target_date", ">=", CP_START),
        ("target_date", "<=", CP_END),
    ]

    try:
        frame = pd.read_parquet(
            SOURCE_PATH,
            columns=SOURCE_COLUMNS,
            filters=filters,
        )
        print("  Predicate pushdown: used")
    except Exception as exc:
        print(
            "  Predicate pushdown unavailable; "
            f"using projected-column fallback ({type(exc).__name__})."
        )
        frame = pd.read_parquet(
            SOURCE_PATH,
            columns=SOURCE_COLUMNS,
        )

    frame["target_date"] = pd.to_datetime(
        frame["target_date"],
        errors="raise",
    ).dt.normalize()

    frame["horizon"] = pd.to_numeric(
        frame["horizon"],
        errors="raise",
    ).astype(int)

    frame["primary_gap_eligible"] = (
        frame["primary_gap_eligible"]
        .fillna(False)
        .astype(bool)
    )

    frame = frame.loc[
        frame["primary_gap_eligible"]
        & frame["metric"].isin(METRIC_ORDER)
        & frame["horizon"].isin(HORIZONS)
        & frame["target_date"].between(
            CP_START,
            CP_END,
            inclusive="both",
        )
    ].copy()

    return frame


def _profile_contract_qa(runtime: pd.DataFrame) -> tuple[int, int, int, float, float]:
    """
    Re-run the existing Raw 23 profile contract against the compact runtime.

    Coverage intentionally differs between the exact-row runtime and the frozen
    profile summary. Equality is required only where profile keys overlap,
    matching the existing Showcase validation semantics.
    """
    authoritative = pd.read_parquet(PROFILE_SUMMARY_PATH)

    required = {
        "profile_id",
        "profile_label",
        "profile_context",
        "metric",
        "horizon",
        "profile_rows",
        "mean_gap_mae_units",
    }
    missing = sorted(required - set(authoritative.columns))
    if missing:
        raise KeyError(
            "counterfactual_mobility_profile_summary is missing: "
            + ", ".join(missing)
        )

    qa_source = runtime.loc[
        runtime["counterfactual_gap_mae_units"].notna()
    ].copy()

    rebuilt = (
        qa_source.groupby(
            [
                "taxi_zone_id",
                "zone",
                "borough",
                "metric",
                "horizon",
            ],
            observed=True,
            dropna=False,
        )
        .agg(
            profile_rows=(
                "counterfactual_gap_mae_units",
                "count",
            ),
            mean_gap_mae_units=(
                "counterfactual_gap_mae_units",
                "mean",
            ),
        )
        .reset_index()
    )

    rebuilt["taxi_zone_id"] = (
        rebuilt["taxi_zone_id"]
        .astype("Int64")
        .astype("string")
    )

    authoritative = authoritative.copy()
    authoritative["profile_id"] = (
        authoritative["profile_id"]
        .astype("string")
    )
    authoritative["horizon"] = pd.to_numeric(
        authoritative["horizon"],
        errors="raise",
    ).astype(int)

    comparison = rebuilt.merge(
        authoritative[
            [
                "profile_id",
                "profile_label",
                "profile_context",
                "metric",
                "horizon",
                "profile_rows",
                "mean_gap_mae_units",
            ]
        ],
        left_on=[
            "taxi_zone_id",
            "zone",
            "borough",
            "metric",
            "horizon",
        ],
        right_on=[
            "profile_id",
            "profile_label",
            "profile_context",
            "metric",
            "horizon",
        ],
        how="outer",
        suffixes=(
            "_rebuilt",
            "_authoritative",
        ),
        indicator=True,
    )

    matched = comparison.loc[
        comparison["_merge"].eq("both")
    ].copy()

    row_diff = (
        pd.to_numeric(
            matched["profile_rows_rebuilt"],
            errors="coerce",
        )
        - pd.to_numeric(
            matched["profile_rows_authoritative"],
            errors="coerce",
        )
    ).abs()

    value_diff = (
        pd.to_numeric(
            matched["mean_gap_mae_units_rebuilt"],
            errors="coerce",
        )
        - pd.to_numeric(
            matched["mean_gap_mae_units_authoritative"],
            errors="coerce",
        )
    ).abs()

    max_row_diff = (
        float(row_diff.max())
        if row_diff.notna().any()
        else 0.0
    )
    max_value_diff = (
        float(value_diff.max())
        if value_diff.notna().any()
        else 0.0
    )

    passed = (
        len(matched) > 0
        and row_diff.fillna(0).eq(0).all()
        and value_diff.fillna(0).le(1e-10).all()
    )

    if not passed:
        raise RuntimeError(
            "Raw 23 profile contract QA failed: "
            f"matched={len(matched):,}; "
            f"max row diff={max_row_diff:.12g}; "
            f"max MAE diff={max_value_diff:.12g}."
        )

    return (
        len(matched),
        int(comparison["_merge"].eq("left_only").sum()),
        int(comparison["_merge"].eq("right_only").sum()),
        max_row_diff,
        max_value_diff,
    )


def main() -> None:
    """Build, verify, and report the compact Raw 23 runtime artifact."""
    started = perf_counter()

    print("RAW 23 COUNTERFACTUAL RUNTIME BUILD")
    print()

    _require_file(
        SOURCE_PATH,
        "Chapter 5 counterfactual gap surface",
    )
    _require_file(
        PROFILE_SUMMARY_PATH,
        "Chapter 5 counterfactual profile summary",
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("[1/4] Read exact Raw 23 source subset")
    stage = perf_counter()
    print(f"  Source size: {_mb(SOURCE_PATH):,.2f} MiB")

    source = _read_source()

    print(f"  Eligible rows: {len(source):,}")
    print(
        "  Date range: "
        f"{source['target_date'].min().date()} → "
        f"{source['target_date'].max().date()}"
    )
    print(
        "  Metrics: "
        + ", ".join(
            sorted(
                source["metric"]
                .dropna()
                .astype(str)
                .unique()
            )
        )
    )
    print(
        "  Horizons: "
        + ", ".join(
            str(value)
            for value in sorted(
                source["horizon"]
                .dropna()
                .unique()
            )
        )
    )
    print(f"  Stage: {perf_counter() - stage:,.1f}s")
    print()

    print("[2/4] Write compact exact-row runtime")
    stage = perf_counter()

    runtime = source[
        RUNTIME_COLUMNS
    ].copy()

    # WHY: ZSTD materially shrinks deployment storage without changing values.
    runtime.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Output size: {_mb(OUTPUT_PATH):,.2f} MiB")
    print(f"  Columns: {len(runtime.columns)}")
    print(f"  Stage: {perf_counter() - stage:,.1f}s")
    print()

    print("[3/4] Read-back and exact-value QA")
    stage = perf_counter()

    rebuilt = pd.read_parquet(
        OUTPUT_PATH,
        columns=RUNTIME_COLUMNS,
    )

    if len(rebuilt) != len(runtime):
        raise RuntimeError(
            "Raw 23 runtime row-count mismatch after write: "
            f"{len(runtime):,} → {len(rebuilt):,}."
        )

    numeric_source = pd.to_numeric(
        runtime["counterfactual_gap_mae_units"],
        errors="coerce",
    ).to_numpy(dtype=float)

    numeric_rebuilt = pd.to_numeric(
        rebuilt["counterfactual_gap_mae_units"],
        errors="coerce",
    ).to_numpy(dtype=float)

    max_numeric_delta = float(
        np.nanmax(
            np.abs(
                numeric_source
                - numeric_rebuilt
            )
        )
    ) if len(runtime) else 0.0

    if max_numeric_delta != 0.0:
        raise RuntimeError(
            "Raw 23 runtime changed counterfactual_gap_mae_units: "
            f"max delta={max_numeric_delta:.12g}."
        )

    print(f"  Read-back rows: {len(rebuilt):,}")
    print(f"  Max numeric delta: {max_numeric_delta:.12g}")
    print(f"  Stage: {perf_counter() - stage:,.1f}s")
    print()

    print("[4/4] Existing Raw 23 profile-contract QA")
    stage = perf_counter()

    (
        matched,
        rebuilt_only,
        authoritative_only,
        max_row_diff,
        max_value_diff,
    ) = _profile_contract_qa(rebuilt)

    print(f"  Matched profiles: {matched:,}")
    print(f"  Runtime-only profiles: {rebuilt_only:,}")
    print(f"  Authoritative-only profiles: {authoritative_only:,}")
    print(f"  Max profile row delta: {max_row_diff:.12g}")
    print(f"  Max profile MAE delta: {max_value_diff:.12g}")
    print(f"  Stage: {perf_counter() - stage:,.1f}s")
    print()

    qa = pd.DataFrame(
        [
            {
                "check_id": "runtime_rows_preserved",
                "status": "PASS",
                "details": f"{len(rebuilt):,} rows",
            },
            {
                "check_id": "runtime_values_preserved",
                "status": "PASS",
                "details": f"max delta={max_numeric_delta:.12g}",
            },
            {
                "check_id": "profile_contract",
                "status": "PASS",
                "details": (
                    f"matched={matched:,}; "
                    f"runtime_only={rebuilt_only:,}; "
                    f"authoritative_only={authoritative_only:,}; "
                    f"max_mae_delta={max_value_diff:.12g}"
                ),
            },
        ]
    )

    qa.to_parquet(
        QA_PATH,
        index=False,
        compression="zstd",
    )

    elapsed = perf_counter() - started
    reduction_pct = (
        100.0
        * (1.0 - OUTPUT_PATH.stat().st_size / SOURCE_PATH.stat().st_size)
    )

    print("BUILD COMPLETE")
    print(
        f"  {OUTPUT_PATH.name}: "
        f"{len(rebuilt):,} rows · "
        f"{_mb(OUTPUT_PATH):,.2f} MiB"
    )
    print(f"  Source reduction: {reduction_pct:,.1f}%")
    print(f"  QA: {QA_PATH.name}")
    print(f"  Total runtime: {elapsed:,.1f}s")

    if _mb(OUTPUT_PATH) >= 100.0:
        print()
        print(
            "WARNING: The exact-row runtime is still at or above 100 MiB. "
            "Do not add it to Git yet; a second-stage sufficient-statistics "
            "compaction will be needed."
        )


if __name__ == "__main__":
    main()
