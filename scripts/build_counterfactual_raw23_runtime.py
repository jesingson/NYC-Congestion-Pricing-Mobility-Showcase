"""
Build purpose-built serving artifacts for Showcase Raw 23.

Raw 23 is an interactive reader-facing page. It should not repeatedly scan and
group the exact-row counterfactual surface inside Streamlit.

This builder moves all stable profile work offline:

1. counterfactual_raw23_profiles.parquet
   One multimodal profile per:
       grouping × geography × horizon × period × day type × daypart

2. counterfactual_raw23_children.parquet
   Taxi-Zone child profiles per:
       parent grouping × parent geography × horizon × period × day type × daypart

3. counterfactual_raw23_weekly.parquet
   Weekly standardized gaps at the smallest temporal-bucket grain needed by
   the storyline. Streamlit only filters one geography and combines a handful
   of bucket rows.

4. counterfactual_raw23_runtime_qa.parquet
   Build and contract checks.

The existing exact-row Raw 23 runtime may remain in the repository for
backward compatibility and forensic QA, but normal Raw 23 rendering no longer
depends on it.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

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

PROFILE_OUTPUT_PATH = OUTPUT_DIR / "counterfactual_raw23_profiles.parquet"
CHILD_OUTPUT_PATH = OUTPUT_DIR / "counterfactual_raw23_children.parquet"
WEEKLY_OUTPUT_PATH = OUTPUT_DIR / "counterfactual_raw23_weekly.parquet"
QA_PATH = OUTPUT_DIR / "counterfactual_raw23_runtime_qa.parquet"


# ---------------------------------------------------------------------
# Frozen Raw 23 contracts
# ---------------------------------------------------------------------

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

GROUPINGS = {
    "Policy geography": {
        "group_id": "cbd_spatial_category",
        "group_label": "cbd_spatial_category",
    },
    "Mobility environment": {
        "group_id": "pre_cp_mobility_environment",
        "group_label": "pre_cp_mobility_environment",
    },
    "Borough": {
        "group_id": "borough",
        "group_label": "borough",
    },
    "Taxi Zone": {
        "group_id": "taxi_zone_id",
        "group_label": "zone",
    },
}

PARENT_GROUPINGS = [
    "Policy geography",
    "Mobility environment",
    "Borough",
]

POLICY_ORDER = [
    "cbd",
    "adjacent_to_cbd",
    "gateway_to_cbd",
    "non_cbd",
]

DAYPART_NAMES = [
    "overnight",
    "am_peak",
    "midday",
    "pm_peak",
    "evening",
]

TEMPORAL_BUCKETS = [
    f"{day_type}_{daypart}"
    for day_type in ["weekday", "weekend"]
    for daypart in DAYPART_NAMES
]

DAY_TYPE_BUCKETS = {
    "All days": TEMPORAL_BUCKETS,
    "Weekdays": [
        bucket
        for bucket in TEMPORAL_BUCKETS
        if bucket.startswith("weekday_")
    ],
    "Weekends": [
        bucket
        for bucket in TEMPORAL_BUCKETS
        if bucket.startswith("weekend_")
    ],
}

DAYPART_BUCKETS = {
    "All dayparts": TEMPORAL_BUCKETS,
    "Overnight": ["weekday_overnight", "weekend_overnight"],
    "AM peak": ["weekday_am_peak", "weekend_am_peak"],
    "Midday": ["weekday_midday", "weekend_midday"],
    "PM peak": ["weekday_pm_peak", "weekend_pm_peak"],
    "Evening": ["weekday_evening", "weekend_evening"],
}

PERIODS = {
    "Full post-CP period": (
        pd.Timestamp("2025-01-05"),
        pd.Timestamp("2026-03-31"),
    ),
    "First 4 weeks": (
        pd.Timestamp("2025-01-05"),
        pd.Timestamp("2025-02-01"),
    ),
    "First 3 months": (
        pd.Timestamp("2025-01-05"),
        pd.Timestamp("2025-04-04"),
    ),
    "2025": (
        pd.Timestamp("2025-01-05"),
        pd.Timestamp("2025-12-31"),
    ),
    "2026 through Mar 31": (
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-03-31"),
    ),
}

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


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def _require_file(path: Path, label: str) -> None:
    """Fail loudly when a required frozen input is absent."""
    if not path.exists():
        raise FileNotFoundError(f"{label} was not found:\n  {path}")


def _mb(path: Path) -> float:
    """Return file size in MiB."""
    return path.stat().st_size / (1024 ** 2)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write one deterministic compressed serving artifact."""
    frame.to_parquet(
        path,
        index=False,
        compression="zstd",
    )


def _read_source() -> pd.DataFrame:
    """Read only primary-gap-eligible Raw 23 rows and required columns."""
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

    mask = (
        frame["primary_gap_eligible"]
        & frame["metric"].isin(METRIC_ORDER)
        & frame["horizon"].isin(HORIZONS)
        & frame["target_date"].between(
            CP_START,
            CP_END,
            inclusive="both",
        )
    )

    return frame.loc[mask].copy()


def _canonicalize_taxi_zone_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Match Raw 23's existing reader-facing Taxi-Zone canonicalization.

    WHY: Raw 23 currently canonicalizes Taxi-Zone profile views after any parent
    geography filter has been applied. We preserve that order exactly.
    """
    result = frame.copy()

    canonical = pd.to_numeric(
        result["canonical_location_id"],
        errors="coerce",
    ).astype("Int64")

    result["taxi_zone_id"] = canonical

    # Preserve the existing Raw 23 special-case label contract.
    result.loc[
        canonical.eq(56),
        "zone",
    ] = "Corona"

    return result


def _rms_profile(values: pd.Series) -> float:
    """Return overall standardized distance from zero."""
    clean = pd.to_numeric(values, errors="coerce").dropna()

    if clean.empty:
        return np.nan

    return float(np.sqrt(np.mean(np.square(clean))))


def _sign_pattern(row: pd.Series) -> str:
    """Encode metric signs in Raw 23's fixed five-measure order."""
    result = []

    for metric in METRIC_ORDER:
        value = row.get(metric, np.nan)

        if pd.isna(value):
            result.append("?")
        elif value > 0:
            result.append("+")
        elif value < 0:
            result.append("−")
        else:
            result.append("0")

    return "".join(result)


def _opposite_sign_pairs(row: pd.Series) -> int:
    """Count directional disagreements among available non-zero measures."""
    values = [
        float(row[metric])
        for metric in METRIC_ORDER
        if pd.notna(row.get(metric))
        and float(row[metric]) != 0
    ]

    count = 0

    for left_index in range(len(values)):
        for right_index in range(left_index + 1, len(values)):
            if np.sign(values[left_index]) != np.sign(values[right_index]):
                count += 1

    return count


def _finalize_profile_table(
    long_summary: pd.DataFrame,
    *,
    index_columns: list[str],
) -> pd.DataFrame:
    """Convert one long metric summary to Raw 23's wide profile shape."""
    if long_summary.empty:
        return pd.DataFrame()

    wide = (
        long_summary.pivot(
            index=index_columns,
            columns="metric",
            values="mean_gap_mae_units",
        )
        .reset_index()
    )

    support = (
        long_summary.groupby(
            index_columns,
            observed=True,
            dropna=False,
        )["support_rows"]
        .sum()
        .reset_index()
    )

    wide = wide.merge(
        support,
        on=index_columns,
        how="left",
        validate="one_to_one",
    )

    for metric in METRIC_ORDER:
        if metric not in wide.columns:
            wide[metric] = np.nan

    wide["multimodal_rms"] = wide[METRIC_ORDER].apply(
        _rms_profile,
        axis=1,
    )
    wide["max_abs_mae"] = wide[METRIC_ORDER].abs().max(axis=1)
    wide["sign_pattern"] = wide.apply(_sign_pattern, axis=1)
    wide["opposite_sign_pairs"] = wide.apply(
        _opposite_sign_pairs,
        axis=1,
    )

    return wide


# ---------------------------------------------------------------------
# Profile builders
# ---------------------------------------------------------------------

def _build_group_profiles(
    scoped: pd.DataFrame,
    *,
    grouping_name: str,
    horizon: int,
    period: str,
    day_type: str,
    daypart: str,
) -> pd.DataFrame:
    """Build one preselected Raw 23 profile slice."""
    working = scoped.loc[
        scoped["horizon"].eq(int(horizon))
    ]

    if grouping_name == "Taxi Zone":
        working = _canonicalize_taxi_zone_rows(working)

        group_columns = [
            "taxi_zone_id",
            "zone",
            "borough",
        ]
        id_column = "taxi_zone_id"
        label_column = "zone"
    else:
        group_id = GROUPINGS[grouping_name]["group_id"]
        group_label = GROUPINGS[grouping_name]["group_label"]

        group_columns = list(
            dict.fromkeys([group_id, group_label])
        )
        id_column = group_id
        label_column = group_label

    valid = working[group_columns].notna().all(axis=1)
    working = working.loc[valid]

    if working.empty:
        return pd.DataFrame()

    long_summary = (
        working.groupby(
            [*group_columns, "metric"],
            observed=True,
            dropna=False,
        )
        .agg(
            mean_gap_mae_units=(
                "counterfactual_gap_mae_units",
                "mean",
            ),
            support_rows=(
                "counterfactual_gap_mae_units",
                "count",
            ),
        )
        .reset_index()
    )

    wide = _finalize_profile_table(
        long_summary,
        index_columns=group_columns,
    )

    wide["grouping"] = grouping_name
    wide["group_value"] = wide[id_column].astype("string")
    wide["group_label"] = wide[label_column].astype("string")
    wide["horizon"] = int(horizon)
    wide["period"] = period
    wide["day_type"] = day_type
    wide["daypart"] = daypart

    if "borough" not in wide.columns:
        wide["borough"] = pd.NA

    keep = [
        "grouping",
        "group_value",
        "group_label",
        "borough",
        "horizon",
        "period",
        "day_type",
        "daypart",
        *METRIC_ORDER,
        "support_rows",
        "multimodal_rms",
        "max_abs_mae",
        "sign_pattern",
        "opposite_sign_pairs",
    ]

    return wide[keep]


def _build_child_profiles(
    scoped: pd.DataFrame,
    *,
    parent_grouping: str,
    horizon: int,
    period: str,
    day_type: str,
    daypart: str,
) -> pd.DataFrame:
    """
    Build all Taxi-Zone children for one parent grouping in one pass.

    WHY: parent membership must be applied before Taxi-Zone canonicalization,
    matching the existing Raw 23 semantics.
    """
    parent_column = GROUPINGS[parent_grouping]["group_id"]

    working = scoped.loc[
        scoped["horizon"].eq(int(horizon))
        & scoped[parent_column].notna()
    ].copy()

    if working.empty:
        return pd.DataFrame()

    working["parent_value"] = working[parent_column].astype("string")
    working = _canonicalize_taxi_zone_rows(working)

    group_columns = [
        "parent_value",
        "taxi_zone_id",
        "zone",
        "borough",
    ]

    valid = working[
        ["taxi_zone_id", "zone"]
    ].notna().all(axis=1)

    working = working.loc[valid]

    if working.empty:
        return pd.DataFrame()

    long_summary = (
        working.groupby(
            [*group_columns, "metric"],
            observed=True,
            dropna=False,
        )
        .agg(
            mean_gap_mae_units=(
                "counterfactual_gap_mae_units",
                "mean",
            ),
            support_rows=(
                "counterfactual_gap_mae_units",
                "count",
            ),
        )
        .reset_index()
    )

    wide = _finalize_profile_table(
        long_summary,
        index_columns=group_columns,
    )

    wide["parent_grouping"] = parent_grouping
    wide["taxi_zone_id"] = wide["taxi_zone_id"].astype("Int64")
    wide["horizon"] = int(horizon)
    wide["period"] = period
    wide["day_type"] = day_type
    wide["daypart"] = daypart

    keep = [
        "parent_grouping",
        "parent_value",
        "taxi_zone_id",
        "zone",
        "borough",
        "horizon",
        "period",
        "day_type",
        "daypart",
        *METRIC_ORDER,
        "support_rows",
        "multimodal_rms",
        "max_abs_mae",
        "sign_pattern",
        "opposite_sign_pairs",
    ]

    return wide[keep]


def build_profile_runtimes(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build every stable Raw 23 profile and child-profile control state."""
    profile_parts = []
    child_parts = []

    total_slices = (
        len(PERIODS)
        * len(DAY_TYPE_BUCKETS)
        * len(DAYPART_BUCKETS)
    )

    slice_number = 0

    for period, (start_date, end_date) in PERIODS.items():
        period_source = source.loc[
            source["target_date"].between(
                start_date,
                end_date,
                inclusive="both",
            )
        ]

        for day_type, day_type_buckets in DAY_TYPE_BUCKETS.items():
            for daypart, daypart_buckets in DAYPART_BUCKETS.items():
                slice_number += 1

                valid_buckets = (
                    set(day_type_buckets)
                    & set(daypart_buckets)
                )

                scoped = period_source.loc[
                    period_source[
                        "target_temporal_bucket"
                    ].isin(valid_buckets)
                ]

                print(
                    f"    Slice {slice_number:>2}/{total_slices}: "
                    f"{period} · {day_type} · {daypart}"
                )

                for horizon in HORIZONS:
                    for grouping_name in GROUPINGS:
                        profiles = _build_group_profiles(
                            scoped,
                            grouping_name=grouping_name,
                            horizon=horizon,
                            period=period,
                            day_type=day_type,
                            daypart=daypart,
                        )

                        if not profiles.empty:
                            profile_parts.append(profiles)

                    for parent_grouping in PARENT_GROUPINGS:
                        children = _build_child_profiles(
                            scoped,
                            parent_grouping=parent_grouping,
                            horizon=horizon,
                            period=period,
                            day_type=day_type,
                            daypart=daypart,
                        )

                        if not children.empty:
                            child_parts.append(children)

    profiles = pd.concat(
        profile_parts,
        ignore_index=True,
    )

    children = pd.concat(
        child_parts,
        ignore_index=True,
    )

    return profiles, children


# ---------------------------------------------------------------------
# Weekly storyline builder
# ---------------------------------------------------------------------

def build_weekly_runtime(source: pd.DataFrame) -> pd.DataFrame:
    """
    Build one compact weekly × temporal-bucket serving surface.

    Streamlit will only need to filter one geography and combine at most ten
    temporal buckets. No exact-row counterfactual scan remains in the page.
    """
    working = source.copy()

    days_since_launch = (
        working["target_date"]
        - CP_START
    ).dt.days

    working["week_start"] = (
        CP_START
        + pd.to_timedelta(
            (days_since_launch // 7) * 7,
            unit="D",
        )
    )

    parts = []

    for grouping_name, grouping in GROUPINGS.items():
        group_id = grouping["group_id"]
        group_label = grouping["group_label"]

        # IMPORTANT: preserve current storyline semantics. Raw 23's existing
        # weekly Taxi-Zone path filters the source taxi_zone_id directly rather
        # than canonicalizing 56/57 before the weekly aggregation.
        group_columns = list(
            dict.fromkeys([group_id, group_label])
        )

        valid = working[group_columns].notna().all(axis=1)
        scoped = working.loc[valid]

        weekly = (
            scoped.groupby(
                [
                    *group_columns,
                    "week_start",
                    "target_temporal_bucket",
                    "metric",
                    "horizon",
                ],
                observed=True,
                dropna=False,
            )
            .agg(
                gap_sum=(
                    "counterfactual_gap_mae_units",
                    "sum",
                ),
                support_rows=(
                    "counterfactual_gap_mae_units",
                    "count",
                ),
            )
            .reset_index()
        )

        weekly["grouping"] = grouping_name
        weekly["group_value"] = weekly[group_id].astype("string")
        weekly["group_label"] = weekly[group_label].astype("string")

        parts.append(
            weekly[
                [
                    "grouping",
                    "group_value",
                    "group_label",
                    "week_start",
                    "target_temporal_bucket",
                    "metric",
                    "horizon",
                    "gap_sum",
                    "support_rows",
                ]
            ]
        )

    return pd.concat(
        parts,
        ignore_index=True,
    )


# ---------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------

def _check_unique(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    """Fail if a serving artifact contains duplicate analytical keys."""
    duplicates = int(
        frame.duplicated(columns, keep=False).sum()
    )

    if duplicates:
        raise RuntimeError(
            f"{label} has {duplicates:,} duplicated rows "
            f"for key {columns}."
        )


def _authoritative_profile_contract(source: pd.DataFrame) -> tuple[int, float]:
    """
    Preserve the existing Chapter 5 Raw 23 source/profile reconciliation.

    This is independent from the new serving-table design.
    """
    authoritative = pd.read_parquet(PROFILE_SUMMARY_PATH)

    qa_source = source.loc[
        source["counterfactual_gap_mae_units"].notna()
    ]

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
        how="inner",
        suffixes=("_rebuilt", "_authoritative"),
    )

    if comparison.empty:
        raise RuntimeError(
            "Raw 23 authoritative profile QA found no overlapping profiles."
        )

    row_diff = (
        comparison["profile_rows_rebuilt"]
        - comparison["profile_rows_authoritative"]
    ).abs()

    value_diff = (
        comparison["mean_gap_mae_units_rebuilt"]
        - comparison["mean_gap_mae_units_authoritative"]
    ).abs()

    max_delta = float(value_diff.max())

    if not row_diff.eq(0).all() or max_delta > 1e-10:
        raise RuntimeError(
            "Raw 23 authoritative profile QA failed: "
            f"max MAE delta={max_delta:.12g}."
        )

    return len(comparison), max_delta


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """Build and verify the complete Raw 23 serving package."""
    started = perf_counter()

    print("=" * 72)
    print("RAW 23 PURPOSE-BUILT RUNTIME BUILD")
    print("=" * 72)
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

    print("[1/5] Read primary Raw 23 source")
    stage = perf_counter()

    source = _read_source()

    print(f"      Source size: {_mb(SOURCE_PATH):,.2f} MiB")
    print(f"      Eligible rows: {len(source):,}")
    print(
        "      Date range: "
        f"{source['target_date'].min().date()} → "
        f"{source['target_date'].max().date()}"
    )
    print(f"      Stage runtime: {perf_counter() - stage:,.1f}s")
    print()

    print("[2/5] Build explorer-ready profile runtimes")
    stage = perf_counter()

    profiles, children = build_profile_runtimes(source)

    print(f"      Profiles: {len(profiles):,} rows")
    print(f"      Child profiles: {len(children):,} rows")
    print(f"      Stage runtime: {perf_counter() - stage:,.1f}s")
    print()

    print("[3/5] Build weekly storyline runtime")
    stage = perf_counter()

    weekly = build_weekly_runtime(source)

    print(f"      Weekly cells: {len(weekly):,}")
    print(f"      Stage runtime: {perf_counter() - stage:,.1f}s")
    print()

    print("[4/5] Validate analytical contracts")
    stage = perf_counter()

    _check_unique(
        profiles,
        [
            "grouping",
            "group_value",
            "horizon",
            "period",
            "day_type",
            "daypart",
        ],
        "Raw 23 profile runtime",
    )

    _check_unique(
        children,
        [
            "parent_grouping",
            "parent_value",
            "taxi_zone_id",
            "horizon",
            "period",
            "day_type",
            "daypart",
        ],
        "Raw 23 child runtime",
    )

    _check_unique(
        weekly,
        [
            "grouping",
            "group_value",
            "week_start",
            "target_temporal_bucket",
            "metric",
            "horizon",
        ],
        "Raw 23 weekly runtime",
    )

    matched_profiles, max_contract_delta = (
        _authoritative_profile_contract(source)
    )

    print(f"      Authoritative matched profiles: {matched_profiles:,}")
    print(f"      Max authoritative MAE delta: {max_contract_delta:.12g}")
    print("      Duplicate analytical keys: 0")
    print(f"      Stage runtime: {perf_counter() - stage:,.1f}s")
    print()

    print("[5/5] Write serving artifacts")
    stage = perf_counter()

    _write_parquet(profiles, PROFILE_OUTPUT_PATH)
    _write_parquet(children, CHILD_OUTPUT_PATH)
    _write_parquet(weekly, WEEKLY_OUTPUT_PATH)

    qa = pd.DataFrame(
        [
            {
                "check_id": "profile_rows",
                "status": "PASS",
                "details": f"{len(profiles):,}",
            },
            {
                "check_id": "child_profile_rows",
                "status": "PASS",
                "details": f"{len(children):,}",
            },
            {
                "check_id": "weekly_rows",
                "status": "PASS",
                "details": f"{len(weekly):,}",
            },
            {
                "check_id": "authoritative_profile_contract",
                "status": "PASS",
                "details": (
                    f"matched={matched_profiles:,}; "
                    f"max_delta={max_contract_delta:.12g}"
                ),
            },
            {
                "check_id": "profile_duplicate_keys",
                "status": "PASS",
                "details": "0",
            },
            {
                "check_id": "child_duplicate_keys",
                "status": "PASS",
                "details": "0",
            },
            {
                "check_id": "weekly_duplicate_keys",
                "status": "PASS",
                "details": "0",
            },
        ]
    )

    _write_parquet(qa, QA_PATH)

    print(
        f"      {PROFILE_OUTPUT_PATH.name}: "
        f"{_mb(PROFILE_OUTPUT_PATH):,.2f} MiB"
    )
    print(
        f"      {CHILD_OUTPUT_PATH.name}: "
        f"{_mb(CHILD_OUTPUT_PATH):,.2f} MiB"
    )
    print(
        f"      {WEEKLY_OUTPUT_PATH.name}: "
        f"{_mb(WEEKLY_OUTPUT_PATH):,.2f} MiB"
    )
    print(
        f"      {QA_PATH.name}: "
        f"{_mb(QA_PATH):,.3f} MiB"
    )
    print(f"      Stage runtime: {perf_counter() - stage:,.1f}s")
    print()

    print("=" * 72)
    print(
        "BUILD COMPLETE · "
        f"{perf_counter() - started:,.1f}s"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()