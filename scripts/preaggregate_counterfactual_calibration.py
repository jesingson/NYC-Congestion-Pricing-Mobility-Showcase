"""
Build the slice-specific Pre-CP Reference MAE lookup for Showcase Raw 25.

WHY THIS EXISTS
---------------
Raw 25 asks whether an observed post-congestion-pricing gap is large relative
to the forecasting error seen before congestion pricing.

That comparison must use the SAME Chapter 5 Pre-CP Reference system that
produced the no-CP forecast, and the error benchmark must match the geography
and time slice selected by the reader.

This script therefore recalculates Pre-CP Reference MAE at:

    Geography
    × Day type
    × Daypart
    × Metric
    × Horizon

Counts and ridership are aggregated by SUM before error is calculated.

Average speeds are aggregated separately in the observed and Reference worlds
using their matching trip-count activity weights before error is calculated.

The script never averages lower-level MAEs upward.

INPUTS
------
data/processed/5.3.1.final_tables/
    counterfactual_pre_cp_reference_validation_surface.parquet
    counterfactual_horizon_stability.parquet

OUTPUTS
-------
data/processed/5.3.1.final_tables/
    counterfactual_raw25_reference_mae.parquet
    counterfactual_raw25_reference_mae_qa.parquet

RUN FROM REPO ROOT
------------------
python scripts/preaggregate_counterfactual_calibration.py
"""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ---------------------------------------------------------------------
# Frozen Raw 25 calibration contract
# ---------------------------------------------------------------------

VALIDATION_SOURCE_NAME = (
    "counterfactual_pre_cp_reference_validation_surface.parquet"
)

CONTEXT_SOURCE_NAME = (
    "counterfactual_horizon_stability.parquet"
)

OUTPUT_NAME = (
    "counterfactual_raw25_reference_mae.parquet"
)

QA_NAME = (
    "counterfactual_raw25_reference_mae_qa.parquet"
)


METRICS = [
    "taxi_trip_count",
    "taxi_avg_trip_speed",
    "fhvhv_trip_count",
    "fhvhv_avg_trip_speed",
    "subway_ridership",
]

HORIZONS = [
    1,
    2,
    5,
]

COUNT_METRICS = {
    "taxi_trip_count",
    "fhvhv_trip_count",
    "subway_ridership",
}

SPEED_WEIGHT_METRIC = {
    "taxi_avg_trip_speed": "taxi_trip_count",
    "fhvhv_avg_trip_speed": "fhvhv_trip_count",
}

GEOGRAPHY_TYPES = [
    "Systemwide",
    "Borough",
    "Policy geography",
    "Mobility environment",
    "Taxi Zone",
]

DAY_TYPES = [
    "All days",
    "Weekdays",
    "Weekends",
]

DAYPARTS = [
    "All dayparts",
    "Overnight",
    "AM peak",
    "Midday",
    "PM peak",
    "Evening",
]

DAYPART_LABELS = {
    "overnight": "Overnight",
    "am_peak": "AM peak",
    "midday": "Midday",
    "pm_peak": "PM peak",
    "evening": "Evening",
}

# Narrow reader slices are allowed to exist below this threshold, but Raw 25
# should treat them as insufficiently supported rather than hiding them.
MIN_VALIDATION_PERIODS = 10

# Reader-facing Taxi Zones use the project's canonical physical-zone contract.
ZONE_ALIASES = {
    57: 56,
    105: 103,
}

UNKNOWN_ZONE_IDS = {
    264,
    265,
}


VALIDATION_REQUIRED_COLUMNS = {
    "validation_contract_id",
    "forecast_sequence_id",
    "taxi_zone_id",
    "metric",
    "horizon",
    "origin_observation_sequence_id",
    "target_observation_sequence_id",
    "target_date",
    "target_daypart",
    "target_temporal_bucket",
    "target_observed_value",
    "target_observed_available",
    "reference_prediction",
    "reference_prediction_available",
    "reference_head_type",
    "reference_family",
    "reference_head_id",
    "benchmark_fallback_used",
}

VALIDATION_READ_COLUMNS = [
    "validation_contract_id",
    "taxi_zone_id",
    "metric",
    "horizon",
    "target_observation_sequence_id",
    "target_date",
    "target_daypart",
    "target_temporal_bucket",
    "target_observed_value",
    "target_observed_available",
    "reference_prediction",
    "reference_prediction_available",
]

CONTEXT_COLUMNS = [
    "taxi_zone_id",
    "zone",
    "borough",
    "cbd_spatial_category",
    "pre_cp_mobility_environment",
]

OUTPUT_COLUMNS = [
    "geography_type",
    "geography_id",
    "geography_label",
    "day_type",
    "daypart",
    "metric",
    "horizon",
    "pre_cp_reference_mae",
    "validation_periods",
    "validation_rows",
    "support_ok",
    "validation_start_date",
    "validation_end_date",
    "validation_contract_id",
]

LOOKUP_KEY = [
    "geography_type",
    "geography_id",
    "day_type",
    "daypart",
    "metric",
    "horizon",
]


# ---------------------------------------------------------------------
# Paths and safe writes
# ---------------------------------------------------------------------


def find_repo_root(explicit_root: str | None) -> Path:
    """Locate the Showcase repository without a developer-specific path."""
    if explicit_root:
        root = Path(explicit_root).expanduser().resolve()
        expected = (
            root
            / "data"
            / "processed"
            / "5.3.1.final_tables"
        )

        if not expected.exists():
            raise FileNotFoundError(
                "--repo-root does not contain "
                "`data/processed/5.3.1.final_tables`."
            )

        return root

    starts = [
        Path.cwd().resolve(),
        Path(__file__).resolve().parent,
    ]

    for start in starts:
        for candidate in [
            start,
            *start.parents,
        ]:
            expected = (
                candidate
                / "data"
                / "processed"
                / "5.3.1.final_tables"
            )

            if expected.exists():
                return candidate

    raise FileNotFoundError(
        "Could not locate the Showcase repository. "
        "Run from inside the repo or pass --repo-root."
    )


def atomic_write_parquet(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    """Write a Parquet file without leaving a partial file behind."""
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
    path: Path,
    required_columns: set[str],
    label: str,
) -> None:
    """Stop immediately when a frozen upstream schema has changed."""
    observed_columns = set(
        pq.ParquetFile(path).schema.names
    )

    missing = sorted(
        required_columns - observed_columns
    )

    if missing:
        raise KeyError(
            f"{label} is missing required columns: {missing}"
        )


def canonicalize_zone_ids(
    series: pd.Series,
) -> pd.Series:
    """Map historical zone aliases onto reader-facing physical Taxi Zones."""
    numeric = pd.to_numeric(
        series,
        errors="raise",
    ).astype(int)

    return numeric.replace(
        ZONE_ALIASES
    )


# ---------------------------------------------------------------------
# Load the frozen validation surface
# ---------------------------------------------------------------------


def load_validation_surface(
    validation_path: Path,
) -> pd.DataFrame:
    """
    Load only the fields needed to rebuild slice-specific Reference error.

    Missing observed values or predictions are valid source conditions. Their
    availability flags determine whether a row contributes to calibration.
    """
    require_columns(
        validation_path,
        VALIDATION_REQUIRED_COLUMNS,
        "Pre-CP Reference validation surface",
    )

    frame = pd.read_parquet(
        validation_path,
        columns=VALIDATION_READ_COLUMNS,
    ).copy()

    frame["taxi_zone_id"] = pd.to_numeric(
        frame["taxi_zone_id"],
        errors="raise",
    ).astype(int)

    # Preserve the original zone ID for cross-metric activity-weight joins.
    # Canonicalization happens separately for reader-facing geography.
    frame["canonical_taxi_zone_id"] = canonicalize_zone_ids(
        frame["taxi_zone_id"]
    )

    frame["horizon"] = pd.to_numeric(
        frame["horizon"],
        errors="raise",
    ).astype(int)

    frame["target_observation_sequence_id"] = pd.to_numeric(
        frame["target_observation_sequence_id"],
        errors="raise",
    ).astype(int)

    frame["target_date"] = pd.to_datetime(
        frame["target_date"],
        errors="raise",
    ).dt.normalize()

    frame["target_observed_value"] = pd.to_numeric(
        frame["target_observed_value"],
        errors="coerce",
    )

    frame["reference_prediction"] = pd.to_numeric(
        frame["reference_prediction"],
        errors="coerce",
    )

    frame["target_observed_available"] = (
        frame["target_observed_available"]
        .fillna(False)
        .astype(bool)
    )

    frame["reference_prediction_available"] = (
        frame["reference_prediction_available"]
        .fillna(False)
        .astype(bool)
    )

    frame["metric"] = (
        frame["metric"]
        .astype("string")
    )

    frame["validation_contract_id"] = (
        frame["validation_contract_id"]
        .astype("string")
    )

    # Derive the reader-facing weekday/weekend control directly from the
    # target date rather than relying on display strings stored upstream.
    frame["_day_type"] = np.where(
        frame["target_date"].dt.dayofweek.lt(5),
        "Weekdays",
        "Weekends",
    )

    daypart_key = (
        frame["target_daypart"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    unexpected_dayparts = sorted(
        set(daypart_key.dropna().unique())
        - set(DAYPART_LABELS)
    )

    if unexpected_dayparts:
        raise RuntimeError(
            "Unexpected validation dayparts: "
            f"{unexpected_dayparts}"
        )

    frame["_daypart"] = daypart_key.map(
        DAYPART_LABELS
    )

    return frame


# ---------------------------------------------------------------------
# Static Taxi Zone context
# ---------------------------------------------------------------------


def load_zone_context(
    context_path: Path,
) -> pd.DataFrame:
    """
    Build one static reader-facing context row per canonical physical zone.

    The horizon-stability file is used only for location metadata. No post-CP
    observed or counterfactual values enter the calibration calculation.
    """
    require_columns(
        context_path,
        set(CONTEXT_COLUMNS),
        "Counterfactual horizon-stability context",
    )

    context = (
        pd.read_parquet(
            context_path,
            columns=CONTEXT_COLUMNS,
        )
        .drop_duplicates()
        .copy()
    )

    context["taxi_zone_id"] = pd.to_numeric(
        context["taxi_zone_id"],
        errors="raise",
    ).astype(int)

    context["canonical_taxi_zone_id"] = canonicalize_zone_ids(
        context["taxi_zone_id"]
    )

    # When both the canonical ID and a historical alias exist, prefer the
    # canonical row's descriptive metadata.
    context["_preferred_canonical_row"] = (
        context["taxi_zone_id"]
        .eq(
            context["canonical_taxi_zone_id"]
        )
    )

    context = (
        context.sort_values(
            [
                "canonical_taxi_zone_id",
                "_preferred_canonical_row",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop_duplicates(
            subset=["canonical_taxi_zone_id"],
            keep="first",
        )
        [
            [
                "canonical_taxi_zone_id",
                "zone",
                "borough",
                "cbd_spatial_category",
                "pre_cp_mobility_environment",
            ]
        ]
        .reset_index(drop=True)
    )

    if context["canonical_taxi_zone_id"].duplicated().any():
        raise RuntimeError(
            "Canonical Taxi Zone context is not unique."
        )

    return context


# ---------------------------------------------------------------------
# Cross-metric activity weights for speed aggregation
# ---------------------------------------------------------------------


def build_activity_weight_table(
    validation: pd.DataFrame,
    count_metric: str,
) -> pd.DataFrame:
    """
    Preserve observed and Reference trip counts used to weight speed.

    The join remains on the original Taxi Zone ID so historical split zones do
    not create a many-to-many join before their physical-zone aggregation.
    """
    join_key = [
        "taxi_zone_id",
        "horizon",
        "target_observation_sequence_id",
    ]

    weights = validation.loc[
        validation["metric"].eq(count_metric),
        [
            *join_key,
            "target_observed_value",
            "target_observed_available",
            "reference_prediction",
            "reference_prediction_available",
        ],
    ].copy()

    if weights.duplicated(join_key).any():
        duplicate_count = int(
            weights.duplicated(
                join_key,
                keep=False,
            ).sum()
        )

        raise RuntimeError(
            f"{count_metric}: activity-weight source contains "
            f"{duplicate_count:,} duplicate join rows."
        )

    weights = weights.rename(
        columns={
            "target_observed_value":
                "observed_activity_weight",
            "target_observed_available":
                "observed_activity_weight_available",
            "reference_prediction":
                "reference_activity_weight",
            "reference_prediction_available":
                "reference_activity_weight_available",
        }
    )

    return weights


# ---------------------------------------------------------------------
# Geography projection
# ---------------------------------------------------------------------


def attach_geography(
    frame: pd.DataFrame,
    geography_type: str,
    zone_context: pd.DataFrame,
) -> pd.DataFrame:
    """Attach one reader-facing geography definition to validation rows."""
    scoped = frame.copy()

    if geography_type == "Systemwide":
        scoped["geography_type"] = "Systemwide"
        scoped["geography_id"] = "systemwide"
        scoped["geography_label"] = "Systemwide"

        return scoped

    # Unknown / nonphysical zones remain in Systemwide totals only.
    scoped = scoped.loc[
        ~scoped["canonical_taxi_zone_id"].isin(
            UNKNOWN_ZONE_IDS
        )
    ].copy()

    scoped = scoped.merge(
        zone_context,
        on="canonical_taxi_zone_id",
        how="left",
        validate="many_to_one",
    )

    if geography_type == "Taxi Zone":
        scoped = scoped.loc[
            scoped["zone"].notna()
        ].copy()

        scoped["geography_type"] = "Taxi Zone"

        scoped["geography_id"] = (
            scoped["canonical_taxi_zone_id"]
            .astype(int)
            .astype(str)
        )

        scoped["geography_label"] = (
            scoped["zone"]
            .astype(str)
        )

        return scoped

    context_field = {
        "Borough": "borough",
        "Policy geography": "cbd_spatial_category",
        "Mobility environment":
            "pre_cp_mobility_environment",
    }[geography_type]

    scoped = scoped.loc[
        scoped[context_field].notna()
    ].copy()

    scoped["geography_type"] = geography_type

    scoped["geography_id"] = (
        scoped[context_field]
        .astype(str)
    )

    scoped["geography_label"] = (
        scoped[context_field]
        .astype(str)
    )

    return scoped


# ---------------------------------------------------------------------
# Build target-period errors
# ---------------------------------------------------------------------


def build_count_period_errors(
    scoped: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate additive metrics first, then calculate Reference error.

    One output row is one geography × validation target observation.
    """
    observed = pd.to_numeric(
        scoped["target_observed_value"],
        errors="coerce",
    )

    reference = pd.to_numeric(
        scoped["reference_prediction"],
        errors="coerce",
    )

    eligible = (
        scoped["target_observed_available"]
        & scoped["reference_prediction_available"]
        & np.isfinite(observed)
        & np.isfinite(reference)
    )

    scoped = scoped.loc[
        eligible
    ].copy()

    scoped["_observed"] = observed.loc[
        eligible
    ].to_numpy()

    scoped["_reference"] = reference.loc[
        eligible
    ].to_numpy()

    group_columns = [
        "geography_type",
        "geography_id",
        "geography_label",
        "metric",
        "horizon",
        "target_observation_sequence_id",
        "target_date",
        "_day_type",
        "_daypart",
    ]

    periods = (
        scoped.groupby(
            group_columns,
            observed=True,
            dropna=False,
            sort=False,
        )
        .agg(
            observed_level=(
                "_observed",
                "sum",
            ),
            reference_level=(
                "_reference",
                "sum",
            ),
            validation_rows=(
                "_reference",
                "size",
            ),
        )
        .reset_index()
    )

    periods["error"] = (
        periods["reference_level"]
        - periods["observed_level"]
    )

    periods["absolute_error"] = (
        periods["error"].abs()
    )

    return periods


def build_speed_period_errors(
    scoped: pd.DataFrame,
    weight_table: pd.DataFrame,
    speed_metric: str,
    diagnostics: dict[str, int],
) -> pd.DataFrame:
    """
    Aggregate average speeds using world-specific activity weights.

    Observed speed uses observed trip counts.
    Reference speed uses Reference-system trip counts.
    """
    join_key = [
        "taxi_zone_id",
        "horizon",
        "target_observation_sequence_id",
    ]

    scoped = scoped.merge(
        weight_table,
        on=join_key,
        how="left",
        validate="many_to_one",
        indicator="_weight_join",
    )

    missing_weight_rows = int(
        scoped["_weight_join"]
        .ne("both")
        .sum()
    )

    diagnostics["speed_source_rows"] += len(scoped)
    diagnostics["speed_weight_join_missing_rows"] += (
        missing_weight_rows
    )

    observed_speed = pd.to_numeric(
        scoped["target_observed_value"],
        errors="coerce",
    )

    reference_speed = pd.to_numeric(
        scoped["reference_prediction"],
        errors="coerce",
    )

    observed_weight = pd.to_numeric(
        scoped["observed_activity_weight"],
        errors="coerce",
    )

    reference_weight = pd.to_numeric(
        scoped["reference_activity_weight"],
        errors="coerce",
    )

    negative_weights = (
        (
            scoped[
                "observed_activity_weight_available"
            ]
            .fillna(False)
            .astype(bool)
            & observed_weight.lt(0)
        )
        |
        (
            scoped[
                "reference_activity_weight_available"
            ]
            .fillna(False)
            .astype(bool)
            & reference_weight.lt(0)
        )
    )

    diagnostics["negative_activity_weight_rows"] += int(
        negative_weights.sum()
    )

    eligible = (
        scoped["target_observed_available"]
        & scoped["reference_prediction_available"]
        & scoped[
            "observed_activity_weight_available"
        ].fillna(False).astype(bool)
        & scoped[
            "reference_activity_weight_available"
        ].fillna(False).astype(bool)
        & np.isfinite(observed_speed)
        & np.isfinite(reference_speed)
        & np.isfinite(observed_weight)
        & np.isfinite(reference_weight)
        & observed_weight.ge(0)
        & reference_weight.ge(0)
    )

    scoped = scoped.loc[
        eligible
    ].copy()

    scoped["_observed_weight"] = (
        observed_weight.loc[
            eligible
        ].to_numpy()
    )

    scoped["_reference_weight"] = (
        reference_weight.loc[
            eligible
        ].to_numpy()
    )

    scoped["_observed_weighted_speed"] = (
        observed_speed.loc[
            eligible
        ].to_numpy()
        * scoped["_observed_weight"].to_numpy()
    )

    scoped["_reference_weighted_speed"] = (
        reference_speed.loc[
            eligible
        ].to_numpy()
        * scoped["_reference_weight"].to_numpy()
    )

    group_columns = [
        "geography_type",
        "geography_id",
        "geography_label",
        "metric",
        "horizon",
        "target_observation_sequence_id",
        "target_date",
        "_day_type",
        "_daypart",
    ]

    periods = (
        scoped.groupby(
            group_columns,
            observed=True,
            dropna=False,
            sort=False,
        )
        .agg(
            observed_numerator=(
                "_observed_weighted_speed",
                "sum",
            ),
            observed_denominator=(
                "_observed_weight",
                "sum",
            ),
            reference_numerator=(
                "_reference_weighted_speed",
                "sum",
            ),
            reference_denominator=(
                "_reference_weight",
                "sum",
            ),
            validation_rows=(
                "_reference_weighted_speed",
                "size",
            ),
        )
        .reset_index()
    )

    # A zero-activity world has no defined average speed for that target.
    periods = periods.loc[
        periods["observed_denominator"].gt(0)
        & periods["reference_denominator"].gt(0)
    ].copy()

    periods["observed_level"] = (
        periods["observed_numerator"]
        / periods["observed_denominator"]
    )

    periods["reference_level"] = (
        periods["reference_numerator"]
        / periods["reference_denominator"]
    )

    periods["error"] = (
        periods["reference_level"]
        - periods["observed_level"]
    )

    periods["absolute_error"] = (
        periods["error"].abs()
    )

    return periods


# ---------------------------------------------------------------------
# Convert target-period errors into Raw 25 lookup rows
# ---------------------------------------------------------------------


def summarize_period_errors(
    periods: pd.DataFrame,
    validation_contract_id: str,
) -> pd.DataFrame:
    """
    Calculate MAE across validation target periods for all Raw 25 time scopes.

    "All dayparts" means all eligible target observations enter the MAE.
    It does not average the five daypart MAEs together.
    """
    base_group = [
        "geography_type",
        "geography_id",
        "geography_label",
        "metric",
        "horizon",
    ]

    def summarize(
        group_columns: list[str],
    ) -> pd.DataFrame:
        return (
            periods.groupby(
                group_columns,
                observed=True,
                dropna=False,
                sort=False,
            )
            .agg(
                pre_cp_reference_mae=(
                    "absolute_error",
                    "mean",
                ),
                validation_periods=(
                    "absolute_error",
                    "size",
                ),
                validation_rows=(
                    "validation_rows",
                    "sum",
                ),
                validation_start_date=(
                    "target_date",
                    "min",
                ),
                validation_end_date=(
                    "target_date",
                    "max",
                ),
            )
            .reset_index()
        )

    # All days × all dayparts.
    all_all = summarize(
        base_group
    )

    all_all["day_type"] = "All days"
    all_all["daypart"] = "All dayparts"

    # Weekdays / weekends × all dayparts.
    day_type_only = summarize(
        [
            *base_group,
            "_day_type",
        ]
    ).rename(
        columns={
            "_day_type": "day_type",
        }
    )

    day_type_only["daypart"] = "All dayparts"

    # All days × individual dayparts.
    daypart_only = summarize(
        [
            *base_group,
            "_daypart",
        ]
    ).rename(
        columns={
            "_daypart": "daypart",
        }
    )

    daypart_only["day_type"] = "All days"

    # Weekdays / weekends × individual dayparts.
    day_type_daypart = summarize(
        [
            *base_group,
            "_day_type",
            "_daypart",
        ]
    ).rename(
        columns={
            "_day_type": "day_type",
            "_daypart": "daypart",
        }
    )

    result = pd.concat(
        [
            all_all,
            day_type_only,
            daypart_only,
            day_type_daypart,
        ],
        ignore_index=True,
    )

    result["validation_periods"] = pd.to_numeric(
        result["validation_periods"],
        errors="raise",
    ).astype(int)

    result["validation_rows"] = pd.to_numeric(
        result["validation_rows"],
        errors="raise",
    ).astype(int)

    result["horizon"] = pd.to_numeric(
        result["horizon"],
        errors="raise",
    ).astype(int)

    mae_is_usable = (
            np.isfinite(
                result["pre_cp_reference_mae"]
            )
            & result["pre_cp_reference_mae"].gt(0)
    )

    result["support_ok"] = (
            result["validation_periods"]
            .ge(MIN_VALIDATION_PERIODS)
            & mae_is_usable
    )

    result["validation_contract_id"] = (
        validation_contract_id
    )

    return result[
        OUTPUT_COLUMNS
    ]


# ---------------------------------------------------------------------
# Main calibration build
# ---------------------------------------------------------------------


def build_reference_mae_lookup(
    validation: pd.DataFrame,
    zone_context: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build every reader-facing Raw 25 calibration slice."""
    contract_ids = (
        validation["validation_contract_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if len(contract_ids) != 1:
        raise RuntimeError(
            "Expected exactly one validation contract ID; "
            f"found {contract_ids}."
        )

    validation_contract_id = contract_ids[0]

    weight_tables = {
        count_metric: build_activity_weight_table(
            validation,
            count_metric,
        )
        for count_metric in {
            *SPEED_WEIGHT_METRIC.values(),
        }
    }

    diagnostics = {
        "speed_source_rows": 0,
        "speed_weight_join_missing_rows": 0,
        "negative_activity_weight_rows": 0,
    }

    output_parts = []

    for geography_type in GEOGRAPHY_TYPES:
        print(
            f"  Geography: {geography_type}",
            flush=True,
        )

        for metric in METRICS:
            for horizon in HORIZONS:
                scoped = validation.loc[
                    validation["metric"].eq(metric)
                    & validation["horizon"].eq(horizon)
                ].copy()

                if scoped.empty:
                    raise RuntimeError(
                        f"No validation rows for {metric} h={horizon}."
                    )

                scoped = attach_geography(
                    scoped,
                    geography_type,
                    zone_context,
                )

                if scoped.empty:
                    continue

                if metric in COUNT_METRICS:
                    periods = build_count_period_errors(
                        scoped
                    )

                else:
                    periods = build_speed_period_errors(
                        scoped,
                        weight_tables[
                            SPEED_WEIGHT_METRIC[metric]
                        ],
                        metric,
                        diagnostics,
                    )

                if periods.empty:
                    continue

                output_parts.append(
                    summarize_period_errors(
                        periods,
                        validation_contract_id,
                    )
                )

    if not output_parts:
        raise RuntimeError(
            "No Raw 25 Reference-MAE rows were produced."
        )

    result = pd.concat(
        output_parts,
        ignore_index=True,
    )

    result["geography_type"] = (
        result["geography_type"].astype("string")
    )

    result["geography_id"] = (
        result["geography_id"].astype("string")
    )

    result["geography_label"] = (
        result["geography_label"].astype("string")
    )

    result["day_type"] = (
        result["day_type"].astype("string")
    )

    result["daypart"] = (
        result["daypart"].astype("string")
    )

    result["metric"] = (
        result["metric"].astype("string")
    )

    result["validation_contract_id"] = (
        result["validation_contract_id"]
        .astype("string")
    )

    geography_order = {
        value: index
        for index, value in enumerate(
            GEOGRAPHY_TYPES
        )
    }

    day_type_order = {
        value: index
        for index, value in enumerate(
            DAY_TYPES
        )
    }

    daypart_order = {
        value: index
        for index, value in enumerate(
            DAYPARTS
        )
    }

    metric_order = {
        value: index
        for index, value in enumerate(
            METRICS
        )
    }

    result["_geography_order"] = (
        result["geography_type"]
        .map(geography_order)
    )

    result["_day_type_order"] = (
        result["day_type"]
        .map(day_type_order)
    )

    result["_daypart_order"] = (
        result["daypart"]
        .map(daypart_order)
    )

    result["_metric_order"] = (
        result["metric"]
        .map(metric_order)
    )

    result = (
        result.sort_values(
            [
                "_geography_order",
                "geography_label",
                "_day_type_order",
                "_daypart_order",
                "_metric_order",
                "horizon",
            ],
            na_position="last",
        )
        .drop(
            columns=[
                "_geography_order",
                "_day_type_order",
                "_daypart_order",
                "_metric_order",
            ]
        )
        .reset_index(drop=True)
    )

    return (
        result,
        diagnostics,
    )


# ---------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------


def build_qa(
    validation: pd.DataFrame,
    zone_context: pd.DataFrame,
    result: pd.DataFrame,
    diagnostics: dict[str, int],
) -> pd.DataFrame:
    """Build structural QA for the Raw 25 calibration lookup."""
    qa_rows = []

    def add_check(
        check_id: str,
        passed: bool,
        details: str,
    ) -> None:
        qa_rows.append(
            {
                "check_id": check_id,
                "status": (
                    "PASS"
                    if bool(passed)
                    else "FAIL"
                ),
                "details": details,
            }
        )

    expected_jobs = {
        (metric, horizon)
        for metric in METRICS
        for horizon in HORIZONS
    }

    source_jobs = set(
        validation[
            [
                "metric",
                "horizon",
            ]
        ]
        .drop_duplicates()
        .itertuples(
            index=False,
            name=None,
        )
    )

    add_check(
        "source_preserves_15_reference_jobs",
        source_jobs == expected_jobs,
        f"observed_jobs={len(source_jobs)}; expected_jobs=15",
    )

    add_check(
        "context_unique_by_canonical_zone",
        not zone_context[
            "canonical_taxi_zone_id"
        ].duplicated().any(),
        f"context_rows={len(zone_context):,}",
    )

    known_source_zone_ids = set(
        validation.loc[
            ~validation[
                "canonical_taxi_zone_id"
            ].isin(UNKNOWN_ZONE_IDS),
            "canonical_taxi_zone_id",
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    context_zone_ids = set(
        zone_context[
            "canonical_taxi_zone_id"
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    missing_context_ids = sorted(
        known_source_zone_ids
        - context_zone_ids
    )

    add_check(
        "known_validation_zones_have_context",
        not missing_context_ids,
        (
            "missing_context_ids="
            f"{missing_context_ids[:20]}"
        ),
    )

    duplicate_lookup_rows = int(
        result.duplicated(
            LOOKUP_KEY
        ).sum()
    )

    add_check(
        "lookup_key_unique",
        duplicate_lookup_rows == 0,
        f"duplicate_rows={duplicate_lookup_rows:,}",
    )

    observed_metrics = set(
        result["metric"]
        .dropna()
        .astype(str)
        .unique()
    )

    add_check(
        "all_five_metrics_present",
        observed_metrics == set(METRICS),
        f"metrics={sorted(observed_metrics)}",
    )

    observed_horizons = set(
        pd.to_numeric(
            result["horizon"],
            errors="raise",
        )
        .astype(int)
        .unique()
        .tolist()
    )

    add_check(
        "horizons_are_1_2_5",
        observed_horizons == set(HORIZONS),
        f"horizons={sorted(observed_horizons)}",
    )

    observed_geography_types = set(
        result["geography_type"]
        .astype(str)
        .unique()
    )

    add_check(
        "all_geography_levels_present",
        observed_geography_types == set(
            GEOGRAPHY_TYPES
        ),
        (
            "geography_types="
            f"{sorted(observed_geography_types)}"
        ),
    )

    observed_day_types = set(
        result["day_type"]
        .astype(str)
        .unique()
    )

    add_check(
        "all_day_type_controls_present",
        observed_day_types == set(DAY_TYPES),
        f"day_types={sorted(observed_day_types)}",
    )

    observed_dayparts = set(
        result["daypart"]
        .astype(str)
        .unique()
    )

    add_check(
        "all_daypart_controls_present",
        observed_dayparts == set(DAYPARTS),
        f"dayparts={sorted(observed_dayparts)}",
    )

    systemwide = result.loc[
        result["geography_type"]
        .eq("Systemwide")
    ].copy()

    expected_systemwide_keys = {
        (
            day_type,
            daypart,
            metric,
            horizon,
        )
        for day_type in DAY_TYPES
        for daypart in DAYPARTS
        for metric in METRICS
        for horizon in HORIZONS
    }

    observed_systemwide_keys = set(
        systemwide[
            [
                "day_type",
                "daypart",
                "metric",
                "horizon",
            ]
        ]
        .itertuples(
            index=False,
            name=None,
        )
    )

    add_check(
        "systemwide_control_grid_complete",
        (
            observed_systemwide_keys
            == expected_systemwide_keys
        ),
        (
            f"observed={len(observed_systemwide_keys)}; "
            f"expected={len(expected_systemwide_keys)}"
        ),
    )

    mae_values = pd.to_numeric(
        result["pre_cp_reference_mae"],
        errors="coerce",
    )

    valid_mae = (
        np.isfinite(mae_values)
        & mae_values.ge(0)
    )

    add_check(
        "mae_values_are_finite_and_nonnegative",
        bool(valid_mae.all()),
        (
            f"valid={int(valid_mae.sum()):,}; "
            f"rows={len(result):,}"
        ),
    )

    expected_support = (
            result["validation_periods"]
            .ge(MIN_VALIDATION_PERIODS)
            & np.isfinite(
        result["pre_cp_reference_mae"]
    )
            & result["pre_cp_reference_mae"].gt(0)
    )

    support_consistent = (
        result["support_ok"]
        .astype(bool)
        .eq(expected_support)
        .all()
    )

    add_check(
        "support_flag_matches_period_threshold",
        bool(support_consistent),
        (
            f"minimum_periods={MIN_VALIDATION_PERIODS}; "
            "MAE must be finite and > 0"
        ),
    )

    add_check(
        "speed_weights_join_completely",
        (
            diagnostics[
                "speed_weight_join_missing_rows"
            ]
            == 0
        ),
        (
            f"speed_rows="
            f"{diagnostics['speed_source_rows']:,}; "
            "missing_weight_joins="
            f"{diagnostics['speed_weight_join_missing_rows']:,}"
        ),
    )

    add_check(
        "activity_weights_are_nonnegative",
        (
            diagnostics[
                "negative_activity_weight_rows"
            ]
            == 0
        ),
        (
            "negative_weight_rows="
            f"{diagnostics['negative_activity_weight_rows']:,}"
        ),
    )

    taxi_zone_output = result.loc[
        result["geography_type"]
        .eq("Taxi Zone")
    ].copy()

    taxi_zone_ids = set(
        taxi_zone_output[
            "geography_id"
        ]
        .astype(str)
        .unique()
    )

    forbidden_zone_ids = {
        "57",
        "105",
        "264",
        "265",
    }

    exposed_forbidden = sorted(
        taxi_zone_ids
        & forbidden_zone_ids
    )

    add_check(
        "taxi_zone_output_uses_physical_zones",
        not exposed_forbidden,
        f"forbidden_ids={exposed_forbidden}",
    )

    source_canonical_ids = set(
        validation[
            "canonical_taxi_zone_id"
        ]
        .dropna()
        .astype(int)
        .unique()
    )

    expected_recombined_ids = {
        str(zone_id)
        for zone_id in [
            56,
            103,
        ]
        if zone_id in source_canonical_ids
    }

    missing_recombined_ids = sorted(
        expected_recombined_ids
        - taxi_zone_ids
    )

    add_check(
        "recombined_physical_zones_remain_available",
        not missing_recombined_ids,
        (
            "missing_recombined_ids="
            f"{missing_recombined_ids}"
        ),
    )

    return pd.DataFrame(
        qa_rows
    )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------


def main() -> None:
    """Build and validate the Raw 25 Reference-MAE lookup."""
    parser = argparse.ArgumentParser(
        description=(
            "Build slice-specific Pre-CP Reference MAE "
            "for Showcase Raw 25."
        )
    )

    parser.add_argument(
        "--repo-root",
        default=None,
        help=(
            "Optional Showcase repository root. "
            "Usually unnecessary when run inside the repo."
        ),
    )

    args = parser.parse_args()

    repo_root = find_repo_root(
        args.repo_root
    )

    final_dir = (
        repo_root
        / "data"
        / "processed"
        / "5.3.1.final_tables"
    )

    validation_path = (
        final_dir
        / VALIDATION_SOURCE_NAME
    )

    context_path = (
        final_dir
        / CONTEXT_SOURCE_NAME
    )

    output_path = (
        final_dir
        / OUTPUT_NAME
    )

    qa_path = (
        final_dir
        / QA_NAME
    )

    print("=" * 72)
    print("RAW 25 REFERENCE MAE BUILD")
    print("=" * 72)

    for path in [
        validation_path,
        context_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found:\n  {path}"
            )

    print(
        f"Validation source:\n  {validation_path}",
        flush=True,
    )

    print(
        f"Zone context:\n  {context_path}",
        flush=True,
    )

    validation = load_validation_surface(
        validation_path
    )

    zone_context = load_zone_context(
        context_path
    )

    source_jobs = (
        validation[
            [
                "metric",
                "horizon",
            ]
        ]
        .drop_duplicates()
    )

    print()
    print(
        f"Source validation rows: {len(validation):,}"
    )

    print(
        f"Reference jobs: {len(source_jobs):,}"
    )

    print(
        "Validation window: "
        f"{validation['target_date'].min().date()} "
        "→ "
        f"{validation['target_date'].max().date()}"
    )

    print()
    print(
        "Building matching geography × time calibration..."
    )

    result, diagnostics = (
        build_reference_mae_lookup(
            validation,
            zone_context,
        )
    )

    qa = build_qa(
        validation,
        zone_context,
        result,
        diagnostics,
    )

    print()
    print("RAW 25 REFERENCE MAE QA")
    print(
        qa.to_string(
            index=False
        )
    )

    failed = qa.loc[
        qa["status"].ne("PASS")
    ]

    if not failed.empty:
        raise RuntimeError(
            "Raw 25 Reference-MAE preprocessing failed QA.\n\n"
            + failed.to_string(
                index=False
            )
        )

    atomic_write_parquet(
        result,
        output_path,
    )

    atomic_write_parquet(
        qa,
        qa_path,
    )

    supported_slices = int(
        result["support_ok"].sum()
    )

    print()
    print("Calibration lookup:")
    print(
        f"  rows: {len(result):,}"
    )
    print(
        "  geography levels: "
        f"{result['geography_type'].nunique():,}"
    )
    print(
        "  metrics: "
        f"{result['metric'].nunique():,}"
    )
    print(
        "  horizons: "
        + ", ".join(
            str(value)
            for value in sorted(
                result["horizon"]
                .astype(int)
                .unique()
            )
        )
    )
    print(
        "  supported slices: "
        f"{supported_slices:,} / {len(result):,}"
    )

    print()
    print("QA: PASS")

    print()
    print("Wrote:")
    print(
        f"  {output_path}"
    )
    print(
        f"  {qa_path}"
    )


if __name__ == "__main__":
    main()