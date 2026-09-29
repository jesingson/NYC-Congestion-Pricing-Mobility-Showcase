"""
Build compact Raw 24 Taxi-Zone robustness scouting artifacts.

Run once from repo root:
    python scripts/build_raw24_zone_scout.py

This is preprocessing, not Streamlit runtime work. It consumes compact,
already-exported Chapter 5 weekly ingredients and writes three small artifacts:

    counterfactual_raw24_zone_diagnostics.parquet
    counterfactual_raw24_zone_weekly.parquet
    counterfactual_raw24_zone_scout_qa.parquet

The diagnostics are transparent scouting aids for selecting illustrative
strong/concerning Taxi-Zone trajectories. They do not alter the frozen 5.3.1
Stable / Mixed / Sensitive conclusion registry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


METRIC_ORDER = [
    "taxi_trip_count",
    "taxi_avg_trip_speed",
    "fhvhv_trip_count",
    "fhvhv_avg_trip_speed",
    "subway_ridership",
]

HORIZON_ORDER = [1, 2, 5]

COUNT_METRICS = {
    "taxi_trip_count",
    "fhvhv_trip_count",
    "subway_ridership",
}

CANONICAL_ZONE_ALIASES = {
    57: 56,
    105: 103,
}

UNKNOWN_ZONE_IDS = {
    264,
    265,
}

POST_REQUIRED = {
    "week_start",
    "period_complete",
    "target_temporal_bucket",
    "taxi_zone_id",
    "zone",
    "borough",
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
}

PRE_REQUIRED = {
    "week_start",
    "temporal_bucket",
    "taxi_zone_id",
    "zone",
    "borough",
    "metric",
    "support_rows",
    "observed_level",
    "observed_weight",
}


def find_repo_root() -> Path:
    """Locate the Showcase repo from the current working directory."""
    for candidate in [
        Path.cwd(),
        *Path.cwd().parents,
    ]:
        final_dir = (
            candidate
            / "data"
            / "processed"
            / "5.3.1.final_tables"
        )
        if final_dir.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate data/processed/5.3.1.final_tables."
    )


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    label: str,
) -> None:
    """Fail loudly if a compact upstream contract changes."""
    missing = sorted(
        required - set(frame.columns)
    )
    if missing:
        raise KeyError(
            f"{label} is missing columns: "
            + ", ".join(missing)
        )


def validate_qa(
    frame: pd.DataFrame,
    *,
    label: str,
) -> None:
    """Require every upstream compact-artifact QA row to pass."""
    if "status" not in frame.columns:
        raise KeyError(
            f"{label} has no status column."
        )

    failed = frame.loc[
        ~frame["status"]
        .astype(str)
        .str.upper()
        .eq("PASS")
    ]

    if not failed.empty:
        raise RuntimeError(
            f"{label} contains failed QA rows:\n"
            + failed.to_string(index=False)
        )


def weighted_value(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    """Activity-weighted mean with explicit positive-weight support."""
    value_array = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(dtype=float)
    weight_array = pd.to_numeric(
        weights,
        errors="coerce",
    ).to_numpy(dtype=float)

    valid = (
        np.isfinite(value_array)
        & np.isfinite(weight_array)
        & (weight_array > 0)
    )

    if not valid.any():
        return np.nan

    return float(
        np.average(
            value_array[valid],
            weights=weight_array[valid],
        )
    )


def choose_zone_label(
    group: pd.DataFrame,
) -> str:
    """Prefer the canonical source zone's label."""
    canonical_id = int(
        group["canonical_taxi_zone_id"].iloc[0]
    )

    canonical_rows = group.loc[
        group["taxi_zone_id"].eq(canonical_id)
        & group["zone"].notna()
    ]

    if not canonical_rows.empty:
        return str(
            canonical_rows["zone"].iloc[0]
        )

    names = (
        group["zone"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return (
        names[0]
        if names
        else f"Taxi Zone {canonical_id}"
    )


def safe_relative_pct(
    numerator: float,
    denominator: float,
) -> float:
    """Absolute percentage ratio with near-zero protection."""
    if (
        not np.isfinite(numerator)
        or not np.isfinite(denominator)
        or abs(denominator) < 1e-12
    ):
        return np.nan

    return float(
        100.0 * abs(numerator / denominator)
    )


def percentile_high(
    series: pd.Series,
) -> pd.Series:
    """Larger raw values receive larger percentiles."""
    return series.rank(
        pct=True,
        method="average",
    )


def percentile_low(
    series: pd.Series,
) -> pd.Series:
    """Smaller raw values receive larger percentiles."""
    return 1.0 - series.rank(
        pct=True,
        method="average",
    )


root = find_repo_root()
final_dir = (
    root
    / "data"
    / "processed"
    / "5.3.1.final_tables"
)

post_path = (
    final_dir
    / "counterfactual_braid_temporal_explorer.parquet"
)
post_qa_path = (
    final_dir
    / "counterfactual_braid_temporal_explorer_qa.parquet"
)
pre_path = (
    final_dir
    / "counterfactual_prelaunch_temporal_explorer.parquet"
)
pre_qa_path = (
    final_dir
    / "counterfactual_prelaunch_temporal_explorer_qa.parquet"
)

for path in [
    post_path,
    post_qa_path,
    pre_path,
    pre_qa_path,
]:
    if not path.exists():
        raise FileNotFoundError(path)

print("Reading compact Raw 21 weekly ingredients...")

post = pd.read_parquet(post_path)
pre = pd.read_parquet(pre_path)
post_qa = pd.read_parquet(post_qa_path)
pre_qa = pd.read_parquet(pre_qa_path)

require_columns(
    post,
    POST_REQUIRED,
    label="counterfactual_braid_temporal_explorer",
)
require_columns(
    pre,
    PRE_REQUIRED,
    label="counterfactual_prelaunch_temporal_explorer",
)
validate_qa(
    post_qa,
    label="counterfactual_braid_temporal_explorer_qa",
)
validate_qa(
    pre_qa,
    label="counterfactual_prelaunch_temporal_explorer_qa",
)

for frame in [post, pre]:
    frame["week_start"] = pd.to_datetime(
        frame["week_start"],
        errors="raise",
    )
    frame["taxi_zone_id"] = pd.to_numeric(
        frame["taxi_zone_id"],
        errors="coerce",
    ).astype("Int64")
    frame["canonical_taxi_zone_id"] = (
        frame["taxi_zone_id"]
        .replace(CANONICAL_ZONE_ALIASES)
        .astype("Int64")
    )

post = post.loc[
    ~post["canonical_taxi_zone_id"]
    .isin(UNKNOWN_ZONE_IDS)
    & post["period_complete"]
    .fillna(False)
    .astype(bool)
].copy()

pre = pre.loc[
    ~pre["canonical_taxi_zone_id"]
    .isin(UNKNOWN_ZONE_IDS)
].copy()


def aggregate_weekly(
    frame: pd.DataFrame,
    *,
    period: str,
) -> pd.DataFrame:
    """Aggregate temporal buckets to physical Taxi Zone × metric × week."""
    records: list[dict[str, object]] = []

    keys = [
        "canonical_taxi_zone_id",
        "metric",
        "week_start",
    ]

    groups = frame.groupby(
        keys,
        observed=True,
        sort=False,
    )

    total = groups.ngroups
    print(
        f"Aggregating {period}: {total:,} zone-metric-week groups..."
    )

    for i, (
        (zone_id, metric, week_start),
        group,
    ) in enumerate(
        groups,
        start=1,
    ):
        if i % 10000 == 0:
            print(
                f"  {period}: {i:,}/{total:,} groups"
            )

        label = choose_zone_label(group)
        borough_values = (
            group["borough"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        borough = (
            borough_values[0]
            if borough_values
            else ""
        )

        if metric in COUNT_METRICS:
            observed = pd.to_numeric(
                group["observed_level"],
                errors="coerce",
            ).sum(min_count=1)
        else:
            observed = weighted_value(
                group["observed_level"],
                group["observed_weight"],
            )

        row = {
            "taxi_zone_id": int(zone_id),
            "zone": label,
            "borough": borough,
            "metric": metric,
            "week_start": pd.Timestamp(week_start),
            "period": period,
            "observed_level": float(observed),
            "no_cp_h1": np.nan,
            "no_cp_h2": np.nan,
            "no_cp_h5": np.nan,
        }

        if period == "post_cp":
            for horizon_value in HORIZON_ORDER:
                if metric in COUNT_METRICS:
                    no_cp = pd.to_numeric(
                        group[f"no_cp_h{horizon_value}"],
                        errors="coerce",
                    ).sum(min_count=1)
                else:
                    no_cp = weighted_value(
                        group[f"no_cp_h{horizon_value}"],
                        group[f"no_cp_weight_h{horizon_value}"],
                    )

                row[f"no_cp_h{horizon_value}"] = float(no_cp)

        records.append(row)

    return pd.DataFrame(records)


pre_weekly = aggregate_weekly(
    pre,
    period="pre_cp",
)
post_weekly = aggregate_weekly(
    post,
    period="post_cp",
)

weekly = pd.concat(
    [
        pre_weekly,
        post_weekly,
    ],
    ignore_index=True,
).sort_values(
    [
        "metric",
        "taxi_zone_id",
        "week_start",
    ]
).reset_index(drop=True)

print(
    f"Weekly artifact: {len(weekly):,} rows"
)

records: list[dict[str, object]] = []

for (
    metric,
    zone_id,
    zone,
    borough,
), post_group in post_weekly.groupby(
    [
        "metric",
        "taxi_zone_id",
        "zone",
        "borough",
    ],
    observed=True,
    sort=False,
):
    pre_group = pre_weekly.loc[
        pre_weekly["metric"].eq(metric)
        & pre_weekly["taxi_zone_id"].eq(zone_id)
    ].sort_values("week_start")

    post_group = post_group.sort_values("week_start")

    if len(pre_group) < 26 or len(post_group) < 40:
        continue

    pre_values = pd.to_numeric(
        pre_group["observed_level"],
        errors="coerce",
    ).dropna()

    observed = pd.to_numeric(
        post_group["observed_level"],
        errors="coerce",
    )
    h1 = pd.to_numeric(
        post_group["no_cp_h1"],
        errors="coerce",
    )
    h2 = pd.to_numeric(
        post_group["no_cp_h2"],
        errors="coerce",
    )
    h5 = pd.to_numeric(
        post_group["no_cp_h5"],
        errors="coerce",
    )

    valid = (
        observed.notna()
        & h1.notna()
        & h2.notna()
        & h5.notna()
    )

    if len(pre_values) < 26 or valid.sum() < 40:
        continue

    observed = observed.loc[valid]
    h1 = h1.loc[valid]
    h2 = h2.loc[valid]
    h5 = h5.loc[valid]

    gap_h1 = h1 - observed
    gap_h2 = h2 - observed
    gap_h5 = h5 - observed

    median_gap = float(gap_h1.median())
    baseline_sign = (
        0
        if np.isclose(median_gap, 0.0, atol=1e-12)
        else int(np.sign(median_gap))
    )

    weekly_sign = np.sign(
        gap_h1.to_numpy(dtype=float)
    )

    direction_persistence = (
        np.nan
        if baseline_sign == 0
        else float(
            100.0
            * np.mean(
                weekly_sign == baseline_sign
            )
        )
    )

    signs = np.column_stack(
        [
            np.sign(gap_h1.to_numpy(dtype=float)),
            np.sign(gap_h2.to_numpy(dtype=float)),
            np.sign(gap_h5.to_numpy(dtype=float)),
        ]
    )

    all_nonzero = np.all(
        signs != 0,
        axis=1,
    )
    same_sign = (
        (signs[:, 0] == signs[:, 1])
        & (signs[:, 0] == signs[:, 2])
        & all_nonzero
    )
    horizon_agreement = float(
        100.0 * np.mean(same_sign)
    )

    last_pre = float(pre_values.iloc[-1])
    first_h1 = float(h1.iloc[0])
    launch_jump_pct = safe_relative_pct(
        first_h1 - last_pre,
        last_pre,
    )

    p05 = float(
        pre_values.quantile(0.05)
    )
    p95 = float(
        pre_values.quantile(0.95)
    )
    pre_range_share = float(
        100.0
        * h1.between(
            p05,
            p95,
        ).mean()
    )

    pre_median_abs = float(
        pre_values.abs().median()
    )
    median_abs_gap = float(
        gap_h1.abs().median()
    )
    separation_pct = safe_relative_pct(
        median_abs_gap,
        pre_median_abs,
    )
    h1_h5_spread_pct = safe_relative_pct(
        float(
            (h1 - h5).abs().median()
        ),
        pre_median_abs,
    )

    records.append(
        {
            "metric": metric,
            "taxi_zone_id": int(zone_id),
            "zone": str(zone),
            "borough": str(borough),
            "pre_weeks": int(len(pre_values)),
            "post_weeks": int(valid.sum()),
            "launch_jump_pct": launch_jump_pct,
            "direction_persistence_pct": direction_persistence,
            "horizon_agreement_pct": horizon_agreement,
            "pre_range_share_pct": pre_range_share,
            "median_abs_gap_relative_to_pre_pct": separation_pct,
            "median_h1_h5_spread_relative_to_pre_pct": h1_h5_spread_pct,
            "median_h1_gap_native": median_gap,
        }
    )

diagnostics = pd.DataFrame(records)

if diagnostics.empty:
    raise RuntimeError(
        "No eligible zone diagnostics were produced."
    )

ranked_parts: list[pd.DataFrame] = []

for metric, part in diagnostics.groupby(
    "metric",
    observed=True,
    sort=False,
):
    part = part.copy()

    part["_smooth"] = percentile_low(
        part["launch_jump_pct"]
    )
    part["_persistence"] = percentile_high(
        part["direction_persistence_pct"]
    )
    part["_horizon"] = percentile_high(
        part["horizon_agreement_pct"]
    )
    part["_plausibility"] = percentile_high(
        part["pre_range_share_pct"]
    )
    part["_separation"] = percentile_high(
        part["median_abs_gap_relative_to_pre_pct"]
    )

    part["strong_scout_index"] = (
        part[
            [
                "_smooth",
                "_persistence",
                "_horizon",
                "_plausibility",
                "_separation",
            ]
        ]
        .mean(axis=1)
    )

    concern_components = pd.DataFrame(
        {
            "launch_jump": (
                1.0
                - part["_smooth"]
            ),
            "direction_instability": (
                1.0
                - part["_persistence"]
            ),
            "horizon_disagreement": (
                1.0
                - part["_horizon"]
            ),
            "historical_range_departure": (
                1.0
                - part["_plausibility"]
            ),
        },
        index=part.index,
    )

    # WHY:
    # Some valid Taxi Zones can have an undefined percentage diagnostic when
    # the comparison denominator is effectively zero. Adding the components
    # directly propagates that one NaN into the entire concern index. Average
    # the available diagnostics instead, matching the strong-example index's
    # skip-missing behavior.
    part["concern_scout_index"] = (
        concern_components.mean(
            axis=1,
            skipna=True,
        )
    )

    part["concern_component_count"] = (
        concern_components.notna().sum(
            axis=1
        )
    )

    ranked_parts.append(
        part.drop(
            columns=[
                "_smooth",
                "_persistence",
                "_horizon",
                "_plausibility",
                "_separation",
            ]
        )
    )

diagnostics = pd.concat(
    ranked_parts,
    ignore_index=True,
)

qa_rows: list[dict[str, object]] = []


def add_qa(
    check_id: str,
    passed: bool,
    details: str,
) -> None:
    qa_rows.append(
        {
            "check_id": check_id,
            "status": "PASS" if passed else "FAIL",
            "details": details,
        }
    )


add_qa(
    "all_five_metrics_present",
    set(diagnostics["metric"]) == set(METRIC_ORDER),
    f"metrics={sorted(diagnostics['metric'].unique())}",
)

add_qa(
    "no_unknown_reader_zones",
    not diagnostics["taxi_zone_id"]
    .isin(UNKNOWN_ZONE_IDS)
    .any(),
    "Unknown 264/265 are absent.",
)

duplicate_corona = (
    diagnostics.loc[
        diagnostics["taxi_zone_id"].eq(56)
    ]
    .groupby(
        "metric",
        observed=True,
    )
    .size()
    .gt(1)
    .any()
)

add_qa(
    "canonical_corona_once_per_metric",
    not bool(duplicate_corona),
    "Source 56/57 are combined before diagnostics.",
)

add_qa(
    "minimum_history_support",
    diagnostics["pre_weeks"].ge(26).all()
    and diagnostics["post_weeks"].ge(40).all(),
    (
        f"min_pre={int(diagnostics['pre_weeks'].min())}; "
        f"min_post={int(diagnostics['post_weeks'].min())}"
    ),
)

bounded = all(
    diagnostics[column]
    .dropna()
    .between(0, 100)
    .all()
    for column in [
        "direction_persistence_pct",
        "horizon_agreement_pct",
        "pre_range_share_pct",
    ]
)

add_qa(
    "percentage_diagnostics_bounded",
    bounded,
    "direction persistence / horizon agreement / pre-range share",
)

strong_index = pd.to_numeric(
    diagnostics["strong_scout_index"],
    errors="coerce",
)
concern_index = pd.to_numeric(
    diagnostics["concern_scout_index"],
    errors="coerce",
)

add_qa(
    "scout_indices_complete",
    strong_index.notna().all()
    and concern_index.notna().all(),
    (
        f"strong_missing={int(strong_index.isna().sum())}; "
        f"concern_missing={int(concern_index.isna().sum())}"
    ),
)

add_qa(
    "concern_has_enough_components",
    diagnostics[
        "concern_component_count"
    ].ge(3).all(),
    (
        "min_available_components="
        f"{int(diagnostics['concern_component_count'].min())}"
    ),
)

add_qa(
    "scout_indices_bounded",
    strong_index.dropna().between(
        0,
        1,
        inclusive="both",
    ).all()
    and concern_index.dropna().between(
        0,
        1,
        inclusive="both",
    ).all(),
    (
        "strong_range="
        f"[{strong_index.min():.6f}, {strong_index.max():.6f}]; "
        "concern_range="
        f"[{concern_index.min():.6f}, {concern_index.max():.6f}]"
    ),
)

qa = pd.DataFrame(qa_rows)

failed = qa.loc[
    qa["status"].eq("FAIL")
]

if not failed.empty:
    raise RuntimeError(
        "Raw 24 zone-scout QA failed:\n"
        + failed.to_string(index=False)
    )

diagnostics_path = (
    final_dir
    / "counterfactual_raw24_zone_diagnostics.parquet"
)
weekly_path = (
    final_dir
    / "counterfactual_raw24_zone_weekly.parquet"
)
qa_path = (
    final_dir
    / "counterfactual_raw24_zone_scout_qa.parquet"
)

diagnostics.to_parquet(
    diagnostics_path,
    index=False,
    compression="zstd",
)
weekly.to_parquet(
    weekly_path,
    index=False,
    compression="zstd",
)
qa.to_parquet(
    qa_path,
    index=False,
    compression="zstd",
)

print()
print("Raw 24 Taxi-Zone scout complete.")
print(
    f"  diagnostics: {len(diagnostics):,} rows -> {diagnostics_path}"
)
print(
    f"  weekly:      {len(weekly):,} rows -> {weekly_path}"
)
print(
    f"  QA:          {len(qa):,} rows -> {qa_path}"
)
print()
print(qa.to_string(index=False))
