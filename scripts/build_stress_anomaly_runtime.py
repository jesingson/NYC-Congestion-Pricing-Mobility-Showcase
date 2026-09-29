from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------
# Make the Showcase package importable when this script runs directly.
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.data_access.anomalies import (  # noqa: E402
    ANOMALY_EVENT_UNIVERSE_PATH,
    ANOMALY_METRIC_DIAGNOSTICS_PATH,
    ANOMALY_ZONE_FREQUENCY_PATH,
    EVENT_ID_COLUMN,
    load_selected_anomaly_events,
)
from app.data_access.loaders import (  # noqa: E402
    CONGESTION_PRICING_START_DATE,
    load_analysis_panel,
)
from app.data_access.mobility_environments import (  # noqa: E402
    load_mobility_regime_cluster_assignments,
)


# =============================================================================
# Paths and analytical contracts
# =============================================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "app_tables"
    / "stress_anomaly_runtime"
)

EVENT_OUTPUT_PATH = OUTPUT_DIR / "stress_anomaly_events.parquet"
OBSERVATION_UNIVERSE_OUTPUT_PATH = (
    OUTPUT_DIR / "stress_anomaly_observation_universe.parquet"
)

RAW14_DENOMINATOR_OUTPUT_PATH = (
    OUTPUT_DIR / "raw14_observation_denominator.parquet"
)

EVIDENCE_OUTPUT_PATH = OUTPUT_DIR / "stress_anomaly_metric_evidence.parquet"
QA_OUTPUT_PATH = OUTPUT_DIR / "stress_anomaly_runtime_qa.parquet"

CP_START_DATE = pd.Timestamp(CONGESTION_PRICING_START_DATE)

SELECTED_FLAG = "selected_finalist_flag"

TOLERANCE = 1e-10
DIRECTIONAL_SCORE_THRESHOLD = 1.28
PARQUET_ROW_GROUP_SIZE = 25_000

MODE_ORDER = ("Taxi", "FHVHV", "Subway", "Bus")

METRIC_TO_MODE = {
    "taxi_trip_count": "Taxi",
    "taxi_avg_trip_speed": "Taxi",
    "taxi_avg_trip_duration": "Taxi",
    "fhvhv_trip_count": "FHVHV",
    "fhvhv_avg_trip_speed": "FHVHV",
    "fhvhv_avg_trip_duration": "FHVHV",
    "subway_ridership": "Subway",
    "subway_transfers": "Subway",
    "bus_trip_count": "Bus",
    "avg_bus_speed": "Bus",
}

METRIC_LABELS = {
    "taxi_trip_count": "Taxi trips",
    "taxi_avg_trip_speed": "Taxi average speed",
    "taxi_avg_trip_duration": "Taxi average duration",
    "fhvhv_trip_count": "FHVHV trips",
    "fhvhv_avg_trip_speed": "FHVHV average speed",
    "fhvhv_avg_trip_duration": "FHVHV average duration",
    "subway_ridership": "Subway ridership",
    "subway_transfers": "Subway transfers",
    "bus_trip_count": "Bus trips",
    "avg_bus_speed": "Bus average speed",
}

DEMAND_METRICS = {
    "taxi_trip_count",
    "fhvhv_trip_count",
    "subway_ridership",
    "subway_transfers",
    "bus_trip_count",
}

CONGESTION_METRICS = {
    "taxi_avg_trip_speed",
    "taxi_avg_trip_duration",
    "fhvhv_avg_trip_speed",
    "fhvhv_avg_trip_duration",
    "avg_bus_speed",
}

STARTED = time.perf_counter()


# =============================================================================
# Progress / storage helpers
# =============================================================================

def progress(message: str) -> None:
    """Print one timestamped progress message."""
    elapsed = time.perf_counter() - STARTED
    print(f"[{elapsed:7.1f}s] {message}", flush=True)


def size_mb(path: Path) -> float:
    """Return file size in MiB."""
    if not path.exists():
        return 0.0

    return path.stat().st_size / (1024 ** 2)


def write_parquet(
    frame: pd.DataFrame,
    path: Path,
    *,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
) -> None:
    """
    Write a ZSTD-compressed Parquet file with moderate row groups.

    WHY: Raw 15 will request individual event IDs from the evidence table.
    Sorting by event ID plus modest row groups gives Parquet predicate
    pushdown a useful chance to avoid reading the entire file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        path.unlink()

    table = pa.Table.from_pandas(
        frame,
        preserve_index=False,
    )

    pq.write_table(
        table,
        path,
        compression="zstd",
        row_group_size=row_group_size,
    )


# =============================================================================
# Canonical normalization helpers
# =============================================================================

def policy_geography(values: pd.Series) -> pd.Series:
    """Normalize the canonical Taxi Zone policy-geography field."""
    normalized = (
        values.astype("string")
        .fillna("")
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


def normalize_period(values: pd.Series) -> pd.Series:
    """Normalize pre/post labels to the reader-facing contract."""
    normalized = (
        values.astype("string")
        .fillna("")
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

    result.loc[normalized.eq("pre_cp")] = "Pre-CP"
    result.loc[normalized.eq("post_cp")] = "Post-CP"

    return result


def canonical_daypart(temporal_bucket: pd.Series) -> pd.Series:
    """Derive the reader-facing daypart from the temporal-bucket key."""
    return (
        temporal_bucket.astype("string")
        .str.replace(r"^(weekday|weekend)_", "", regex=True)
        .str.replace("_", " ", regex=False)
        .str.title()
        .replace(
            {
                "Am Peak": "AM Peak",
                "Pm Peak": "PM Peak",
            }
        )
    )


def driver_tokens(value: object) -> tuple[str, ...]:
    """Extract unique recognized metric names from serialized driver text."""
    if value is None or pd.isna(value):
        return tuple()

    tokens = re.findall(
        r"[A-Za-z][A-Za-z0-9_]*",
        str(value),
    )

    return tuple(
        dict.fromkeys(
            token.lower()
            for token in tokens
            if token.lower() in METRIC_TO_MODE
        )
    )


def metric_signature(
    metrics: tuple[str, ...],
    *,
    family: str,
) -> tuple[str, ...]:
    """Collapse metric drivers to ordered modality participation."""
    if family == "Demand":
        relevant = DEMAND_METRICS
    elif family == "Congestion":
        relevant = CONGESTION_METRICS
    elif family == "All":
        relevant = set(METRIC_TO_MODE)
    else:
        raise ValueError(f"Unsupported signature family: {family}")

    present_modes = {
        METRIC_TO_MODE[metric]
        for metric in metrics
        if metric in relevant
    }

    return tuple(
        mode
        for mode in MODE_ORDER
        if mode in present_modes
    )


def signature_text(signature: tuple[str, ...]) -> str:
    """Store one modality signature as a compact stable string."""
    return "|".join(signature)


def signature_label(signature: tuple[str, ...]) -> str:
    """Return the reader-facing modality-combination label."""
    return " + ".join(signature) if signature else "(none)"


# =============================================================================
# Shared canonical lookups
# =============================================================================

def build_zone_geography() -> pd.DataFrame:
    """Build one stable Taxi Zone -> borough/policy-geography lookup."""
    progress("[1/9] Building canonical Taxi Zone geography lookup...")

    source = load_analysis_panel(
        columns=[
            "taxi_zone_id",
            "zone",
            "borough",
            "cbd_spatial_category",
        ]
    )

    zone_geo = (
        source[
            [
                "taxi_zone_id",
                "zone",
                "borough",
                "cbd_spatial_category",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    conflict_counts = (
        zone_geo.groupby(
            "taxi_zone_id",
            observed=True,
            dropna=False,
        )["cbd_spatial_category"]
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
        zone_geo.drop_duplicates("taxi_zone_id")
        .reset_index(drop=True)
    )

    zone_geo["geography_group"] = policy_geography(
        zone_geo["cbd_spatial_category"]
    )

    progress(
        f"      {zone_geo['taxi_zone_id'].nunique():,} Taxi Zones."
    )

    return zone_geo


def build_environment_assignments() -> pd.DataFrame:
    """Build the canonical Taxi Zone × policy-period environment lookup."""
    progress("[2/9] Loading mobility-environment assignments...")

    assignments = (
        load_mobility_regime_cluster_assignments()[
            [
                "taxi_zone_id",
                "pre_post_cp",
                "cluster_label",
                "canonical_cluster_name",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    assignments["period_group"] = normalize_period(
        assignments["pre_post_cp"]
    )

    duplicates = int(
        assignments.duplicated(
            [
                "taxi_zone_id",
                "period_group",
            ]
        ).sum()
    )

    if duplicates:
        raise ValueError(
            "Mobility-environment assignments are not unique at "
            f"Taxi Zone × policy period: {duplicates:,} duplicates."
        )

    assignments = assignments.rename(
        columns={
            "cluster_label": "mobility_regime_cluster_label",
            "canonical_cluster_name": "environment_group",
        }
    )

    progress(
        f"      {len(assignments):,} Taxi Zone × period assignments."
    )

    return assignments[
        [
            "taxi_zone_id",
            "period_group",
            "mobility_regime_cluster_label",
            "environment_group",
        ]
    ]


# =============================================================================
# Full observation universe
# =============================================================================

def build_observation_universe_runtime(
    zone_geo: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    Build the complete observation universe used by shared anomaly incidence.

    WHY:
    The final 3.3.6 frequency surface was generated from every Taxi Zone × date
    × daypart observation, not only rows passing
    comparison_group_support_review_flag.

    The source row count is captured dynamically and returned for QA so the
    runtime must preserve the complete source grain without hard-coding today's
    observed count of 1,559,590 rows as a permanent contract.
    """
    progress("[3/9] Building full observation-universe runtime...")

    required_columns = [
        "taxi_zone_id",
        "date",
        "daypart",
        "temporal_bucket",
        SELECTED_FLAG,
        "has_congestion_oriented",
        "has_positive_demand_shock",
    ]

    observation_universe = pd.read_parquet(
        ANOMALY_EVENT_UNIVERSE_PATH,
        columns=required_columns,
    ).copy()

    source_universe_rows = len(observation_universe)

    observation_universe["date"] = pd.to_datetime(
        observation_universe["date"],
        errors="coerce",
    )

    observation_universe["period_group"] = np.where(
        observation_universe["date"].lt(CP_START_DATE),
        "Pre-CP",
        "Post-CP",
    )

    observation_universe["day_type"] = np.where(
        observation_universe["temporal_bucket"]
        .astype("string")
        .str.startswith("weekend"),
        "Weekend",
        "Weekday",
    )

    observation_universe["daypart"] = canonical_daypart(
        observation_universe["temporal_bucket"]
    )

    observation_universe = observation_universe.merge(
        zone_geo[
            [
                "taxi_zone_id",
                "zone",
                "borough",
                "geography_group",
            ]
        ],
        on="taxi_zone_id",
        how="left",
        validate="many_to_one",
    )

    observation_universe = observation_universe.merge(
        assignments,
        on=[
            "taxi_zone_id",
            "period_group",
        ],
        how="left",
        validate="many_to_one",
    )

    for column in [
        "zone",
        "borough",
        "geography_group",
        "environment_group",
    ]:
        observation_universe[column] = (
            observation_universe[column]
            .astype("string")
            .fillna("Unknown")
        )

    observation_universe[SELECTED_FLAG] = (
        observation_universe[SELECTED_FLAG]
        .fillna(False)
        .astype(bool)
    )

    for flag in [
        "has_congestion_oriented",
        "has_positive_demand_shock",
    ]:
        observation_universe[flag] = observation_universe[flag].fillna(False).astype(bool)

    observation_universe["mobility_regime_cluster_label"] = pd.to_numeric(
        observation_universe["mobility_regime_cluster_label"],
        errors="coerce",
    ).astype("Int64")

    duplicate_grain = int(
        observation_universe.duplicated(
            [
                "taxi_zone_id",
                "date",
                "daypart",
            ]
        ).sum()
    )

    if duplicate_grain:
        raise ValueError(
            "Observation-universe runtime is not unique at "
            "Taxi Zone × date × daypart: "
            f"{duplicate_grain:,} duplicate rows."
        )

    runtime_rows = len(observation_universe)

    if runtime_rows != source_universe_rows:
        raise ValueError(
            "Observation-universe row-count parity failed: "
            f"source={source_universe_rows:,}; "
            f"runtime={runtime_rows:,}."
        )

    progress(
        f"      {runtime_rows:,} observations retained "
        f"from {source_universe_rows:,} source rows."
    )

    output = observation_universe[
        [
            "taxi_zone_id",
            "date",
            "daypart",
            "day_type",
            "temporal_bucket",
            "period_group",
            "zone",
            "borough",
            "geography_group",
            "environment_group",
            SELECTED_FLAG,
            "mobility_regime_cluster_label",
            "has_congestion_oriented",
            "has_positive_demand_shock",
        ]
    ].reset_index(drop=True)

    return output, source_universe_rows

# =============================================================================
# Raw 14 compact denominator
# =============================================================================

def build_raw14_denominator_runtime(
    observation_universe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Collapse Raw 14's canonical observation universe to additive denominator cells.

    WHY:
    Raw 14 needs eligible-observation counts across date, temporal bucket,
    policy period, borough, policy geography, and mobility environment. It does
    not need all 1.56M individual Taxi Zone × date × daypart rows at runtime.
    """
    progress("[4/9] Building compact Raw 14 denominator runtime...")

    group_columns = [
        "date",
        "temporal_bucket",
        "period_group",
        "borough",
        "geography_group",
        "environment_group",
    ]

    missing = sorted(
        set(group_columns).difference(
            observation_universe.columns
        )
    )

    if missing:
        raise ValueError(
            "Observation universe is missing Raw 14 denominator fields: "
            + ", ".join(missing)
        )

    denominator = (
        observation_universe.groupby(
            group_columns,
            observed=True,
            dropna=False,
        )
        .size()
        .rename("eligible_observations")
        .reset_index()
    )

    denominator["eligible_observations"] = (
        denominator["eligible_observations"]
        .astype("int32")
    )

    reconstructed_rows = int(
        denominator["eligible_observations"].sum()
    )

    source_rows = int(
        len(observation_universe)
    )

    if reconstructed_rows != source_rows:
        raise ValueError(
            "Raw 14 denominator aggregate does not reconstruct the "
            "canonical observation universe. "
            f"Expected {source_rows:,}; got {reconstructed_rows:,}."
        )

    duplicate_cells = int(
        denominator.duplicated(
            group_columns
        ).sum()
    )

    if duplicate_cells:
        raise ValueError(
            "Raw 14 denominator runtime contains duplicate aggregate cells: "
            f"{duplicate_cells:,}."
        )

    progress(
        f"      {source_rows:,} observations -> "
        f"{len(denominator):,} denominator cells."
    )

    return denominator

# =============================================================================
# Selected event runtime
# =============================================================================

def build_event_runtime(
    zone_geo: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Build one enriched row per selected stress-anomaly event."""
    progress("[5/9] Building selected-event runtime...")

    events = load_selected_anomaly_events().copy()

    required = {
        EVENT_ID_COLUMN,
        "taxi_zone_id",
        "date",
        "temporal_bucket",
        "has_congestion_oriented",
        "has_positive_demand_shock",
        "stress_metric_driver_list",
    }

    missing = sorted(
        required.difference(events.columns)
    )

    if missing:
        raise ValueError(
            "Selected anomaly events are missing required fields: "
            + ", ".join(missing)
        )

    events[EVENT_ID_COLUMN] = (
        events[EVENT_ID_COLUMN]
        .astype(str)
    )

    events["date"] = pd.to_datetime(
        events["date"],
        errors="coerce",
    )

    events["period_group"] = np.where(
        events["date"].lt(CP_START_DATE),
        "Pre-CP",
        "Post-CP",
    )

    events["day_type"] = np.where(
        events["temporal_bucket"]
        .astype("string")
        .str.startswith("weekend"),
        "Weekend",
        "Weekday",
    )

    events["daypart"] = canonical_daypart(
        events["temporal_bucket"]
    )

    # Use canonical zone context rather than relying on enrichment fields that
    # were assembled at different upstream stages.
    events = events.drop(
        columns=[
            column
            for column in [
                "zone",
                "borough",
                "geography_group",
                "environment_group",
                "mobility_regime_cluster_label",
            ]
            if column in events.columns
        ],
        errors="ignore",
    )

    events = events.merge(
        zone_geo[
            [
                "taxi_zone_id",
                "zone",
                "borough",
                "geography_group",
            ]
        ],
        on="taxi_zone_id",
        how="left",
        validate="many_to_one",
    )

    events = events.merge(
        assignments,
        on=[
            "taxi_zone_id",
            "period_group",
        ],
        how="left",
        validate="many_to_one",
    )

    for column in [
        "zone",
        "borough",
        "geography_group",
        "environment_group",
    ]:
        events[column] = (
            events[column]
            .astype("string")
            .fillna("Unknown")
        )

    congestion = (
        events["has_congestion_oriented"]
        .fillna(False)
        .astype(bool)
    )

    demand = (
        events["has_positive_demand_shock"]
        .fillna(False)
        .astype(bool)
    )

    events["stress_family"] = np.select(
        [
            congestion & demand,
            congestion,
            demand,
        ],
        [
            "Both",
            "Congestion",
            "Demand",
        ],
        default="Other stress",
    )

    events["driver_metrics_tuple"] = (
        events["stress_metric_driver_list"]
        .map(driver_tokens)
    )

    for family_name, suffix in [
        ("All", "all"),
        ("Demand", "demand"),
        ("Congestion", "congestion"),
    ]:
        signatures = events[
            "driver_metrics_tuple"
        ].map(
            lambda metrics: metric_signature(
                metrics,
                family=family_name,
            )
        )

        events[f"signature_{suffix}"] = signatures.map(
            signature_text
        )

        events[f"signature_{suffix}_label"] = signatures.map(
            signature_label
        )

        for mode in MODE_ORDER:
            key = mode.lower()

            events[
                f"{key}_{suffix}_driver"
            ] = signatures.map(
                lambda values, mode=mode: mode in values
            )

    events["driver_metric_count"] = (
        events["driver_metrics_tuple"]
        .map(len)
        .astype("int16")
    )

    events["driver_mode_count"] = (
        events["signature_all"]
        .map(
            lambda value: (
                0
                if not value
                else len(str(value).split("|"))
            )
        )
        .astype("int8")
    )

    events["driver_mode_label"] = (
        events["signature_all_label"]
    )

    events["zone_label"] = (
        events["zone"].astype(str)
        + " · "
        + events["borough"].astype(str)
        + " · ID "
        + events["taxi_zone_id"].astype(str)
    )

    duplicate_events = int(
        events[EVENT_ID_COLUMN]
        .duplicated()
        .sum()
    )

    if duplicate_events:
        raise ValueError(
            "Selected-event runtime contains duplicate comparison_event_id "
            f"values: {duplicate_events:,}."
        )

    output_columns = [
        EVENT_ID_COLUMN,
        "taxi_zone_id",
        "date",
        "temporal_bucket",
        "daypart",
        "day_type",
        "period_group",
        "zone",
        "borough",
        "geography_group",
        "environment_group",
        "zone_label",
        "has_congestion_oriented",
        "has_positive_demand_shock",
        "stress_family",
        "stress_metric_driver_list",
        "driver_metric_count",
        "driver_mode_count",
        "driver_mode_label",
        "signature_all",
        "signature_all_label",
        "signature_demand",
        "signature_demand_label",
        "signature_congestion",
        "signature_congestion_label",
    ]

    for suffix in [
        "all",
        "demand",
        "congestion",
    ]:
        for mode in MODE_ORDER:
            output_columns.append(
                f"{mode.lower()}_{suffix}_driver"
            )

    output = (
        events[output_columns]
        .sort_values(
            [
                "date",
                "taxi_zone_id",
                "temporal_bucket",
            ]
        )
        .reset_index(drop=True)
    )

    progress(
        f"      {len(output):,} selected anomaly events."
    )

    return output


# =============================================================================
# Metric-evidence runtime
# =============================================================================

def counter_stress_mask(
    frame: pd.DataFrame,
) -> pd.Series:
    """Identify strong metric deviations running opposite the stress direction."""
    metric = frame["metric"].astype(str)

    zscore = pd.to_numeric(
        frame["residual_zscore"],
        errors="coerce",
    )

    strong = zscore.abs().ge(
        DIRECTIONAL_SCORE_THRESHOLD
    )

    counter_direction = (
        (
            metric.isin(DEMAND_METRICS)
            & zscore.lt(0)
        )
        | (
            metric.str.contains(
                "speed",
                na=False,
            )
            & zscore.gt(0)
        )
        | (
            metric.str.contains(
                "duration",
                na=False,
            )
            & zscore.lt(0)
        )
    )

    return strong & counter_direction


def build_metric_evidence_runtime(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the Raw 15 event-evidence contract.

    All event/metric reconciliation happens here once. Raw 15 should only
    retrieve the ten evidence rows belonging to the currently selected event.
    """
    progress("[6/9] Building compact metric-evidence runtime...")

    diagnostics = pd.read_parquet(
        ANOMALY_METRIC_DIAGNOSTICS_PATH
    ).copy()

    required = {
        EVENT_ID_COLUMN,
        "taxi_zone_id",
        "date",
        "daypart",
        "temporal_bucket",
        "metric",
        "observed_value",
        "expected_value",
        "residual_value",
        "residual_zscore",
        "residual_center",
        "residual_scale",
        "support_status",
        "stress_driver_flag",
        "directional_label",
    }

    missing = sorted(
        required.difference(diagnostics.columns)
    )

    if missing:
        raise ValueError(
            "Metric diagnostics are missing required fields: "
            + ", ".join(missing)
        )

    diagnostics[EVENT_ID_COLUMN] = (
        diagnostics[EVENT_ID_COLUMN]
        .astype(str)
    )

    diagnostics["date"] = pd.to_datetime(
        diagnostics["date"],
        errors="coerce",
    )

    diagnostics["metric"] = (
        diagnostics["metric"]
        .astype(str)
    )

    diagnostics["stress_driver_flag"] = (
        diagnostics["stress_driver_flag"]
        .fillna(False)
        .astype(bool)
    )

    diagnostics["metric_label"] = (
        diagnostics["metric"]
        .map(METRIC_LABELS)
        .fillna(diagnostics["metric"])
    )

    diagnostics["mode"] = (
        diagnostics["metric"]
        .map(METRIC_TO_MODE)
    )

    diagnostics["stress_signal"] = np.where(
        diagnostics["metric"].isin(DEMAND_METRICS),
        "Demand pressure",
        "Congestion pressure",
    )

    diagnostics["is_defining_driver"] = (
        diagnostics["stress_driver_flag"]
    )

    diagnostics["is_counter_stress"] = (
        counter_stress_mask(
            diagnostics
        )
    )

    diagnostics["support_label"] = (
        diagnostics["support_status"]
        .astype("string")
        .str.title()
    )

    diagnostics["difference"] = (
        diagnostics["observed_value"]
        - diagnostics["expected_value"]
    )

    duplicate_grain = int(
        diagnostics.duplicated(
            [
                EVENT_ID_COLUMN,
                "metric",
            ]
        ).sum()
    )

    if duplicate_grain:
        raise ValueError(
            "Metric evidence is not unique at event × metric: "
            f"{duplicate_grain:,} duplicate rows."
        )

    # -----------------------------------------------------------------
    # Move Raw 15's expensive event-list ↔ diagnostic reconciliation QA
    # out of Streamlit and into this one-time build.
    # -----------------------------------------------------------------
    event_driver_lookup = (
        events[
            [
                EVENT_ID_COLUMN,
                "stress_metric_driver_list",
            ]
        ]
        .copy()
    )

    event_driver_lookup["listed_metric"] = (
        event_driver_lookup[
            "stress_metric_driver_list"
        ].map(driver_tokens)
    )

    listed = (
        event_driver_lookup[
            [
                EVENT_ID_COLUMN,
                "listed_metric",
            ]
        ]
        .explode("listed_metric")
        .dropna(
            subset=["listed_metric"]
        )
        .rename(
            columns={
                "listed_metric": "metric",
            }
        )
    )

    flagged = diagnostics.loc[
        diagnostics["stress_driver_flag"],
        [
            EVENT_ID_COLUMN,
            "metric",
        ],
    ].copy()

    listed_keys = set(
        map(
            tuple,
            listed[
                [
                    EVENT_ID_COLUMN,
                    "metric",
                ]
            ].to_numpy(),
        )
    )

    flagged_keys = set(
        map(
            tuple,
            flagged[
                [
                    EVENT_ID_COLUMN,
                    "metric",
                ]
            ].to_numpy(),
        )
    )

    if listed_keys != flagged_keys:
        only_listed = len(
            listed_keys - flagged_keys
        )

        only_flagged = len(
            flagged_keys - listed_keys
        )

        raise ValueError(
            "Event-level driver lists do not reconcile to metric-level "
            "stress_driver_flag rows. "
            f"Only event list: {only_listed:,}; "
            f"only diagnostics: {only_flagged:,}."
        )

    metric_counts = (
        diagnostics.groupby(
            EVENT_ID_COLUMN,
            observed=True,
        )["metric"]
        .nunique()
    )

    distinct_metric_count = int(
        diagnostics["metric"]
        .nunique()
    )

    if (
        int(metric_counts.min())
        != distinct_metric_count
        or int(metric_counts.max())
        != distinct_metric_count
    ):
        raise ValueError(
            "Selected events do not have complete metric evidence. "
            f"Expected {distinct_metric_count} metrics per event; "
            f"observed {int(metric_counts.min())}–"
            f"{int(metric_counts.max())}."
        )

    output_columns = [
        EVENT_ID_COLUMN,
        "taxi_zone_id",
        "date",
        "daypart",
        "temporal_bucket",
        "metric",
        "metric_label",
        "mode",
        "stress_signal",
        "observed_value",
        "expected_value",
        "difference",
        "residual_value",
        "residual_zscore",
        "residual_center",
        "residual_scale",
        "support_status",
        "support_label",
        "stress_driver_flag",
        "is_defining_driver",
        "is_counter_stress",
        "directional_label",
    ]

    output = (
        diagnostics[output_columns]
        .sort_values(
            [
                EVENT_ID_COLUMN,
                "metric",
            ]
        )
        .reset_index(drop=True)
    )

    progress(
        f"      {len(output):,} event × metric evidence rows "
        f"({distinct_metric_count} metrics/event)."
    )

    return output


# =============================================================================
# Canonical Page 13 denominator QA
# =============================================================================

def validate_frequency_contract(
    observation_universe: pd.DataFrame,
) -> tuple[float, int]:
    """
    Reproduce the validated All-3 pre/post zone-frequency handoff.

    WHY:
    The saved 3.3.6 frequency surface uses all observations as its denominator.
    This is the non-negotiable semantic QA gate for the shared runtime.
    """
    progress(
        "[7/9] Validating full observation universe against "
        "Page 13 All-3 frequency..."
    )

    zone_period = (
        observation_universe.groupby(
            [
                "taxi_zone_id",
                "period_group",
            ],
            observed=True,
            dropna=False,
        )
        .agg(
            observation_rows=(
                SELECTED_FLAG,
                "size",
            ),
            selected_events=(
                SELECTED_FLAG,
                "sum",
            ),
        )
        .reset_index()
    )

    zone_period["reconstructed_event_share"] = (
        zone_period["selected_events"]
        / zone_period["observation_rows"]
    )

    reconstructed = (
        zone_period.pivot(
            index="taxi_zone_id",
            columns="period_group",
            values="reconstructed_event_share",
        )
        .reset_index()
        .rename_axis(None, axis=1)
        .rename(
            columns={
                "Pre-CP": "reconstructed_pre_share",
                "Post-CP": "reconstructed_post_share",
            }
        )
    )

    frequency = pd.read_parquet(
        ANOMALY_ZONE_FREQUENCY_PATH
    )

    all3 = frequency.loc[
        frequency["candidate_surface_label"]
        .astype(str)
        .str.contains(
            "All 3",
            case=False,
            na=False,
        )
    ].copy()

    if all3.empty:
        raise ValueError(
            "Could not identify the canonical All-3 candidate surface."
        )

    comparison = all3.merge(
        reconstructed,
        on="taxi_zone_id",
        how="inner",
        validate="one_to_one",
    )

    comparison["pre_abs_diff"] = (
        comparison["pre_cp_event_share_of_eligible_rows"]
        - comparison["reconstructed_pre_share"]
    ).abs()

    comparison["post_abs_diff"] = (
        comparison["post_cp_event_share_of_eligible_rows"]
        - comparison["reconstructed_post_share"]
    ).abs()

    max_diff = float(
        max(
            comparison["pre_abs_diff"].max(skipna=True),
            comparison["post_abs_diff"].max(skipna=True),
        )
    )

    bad_zones = int(
        (
            comparison["pre_abs_diff"].gt(TOLERANCE)
            | comparison["post_abs_diff"].gt(TOLERANCE)
        )
        .fillna(True)
        .sum()
    )

    if bad_zones or max_diff > TOLERANCE:
        raise ValueError(
            "Shared observation-universe runtime does not reproduce the "
            "canonical Page 13 All-3 frequency surface. "
            f"Bad zones: {bad_zones:,}; "
            f"max diff: {max_diff:.12f}."
        )

    progress(
        f"      PASS · {len(comparison):,} matched zones · "
        f"bad zones {bad_zones:,} · "
        f"max abs diff {max_diff:.12f}."
    )

    return max_diff, bad_zones

# =============================================================================
# Cross-runtime QA and output
# =============================================================================

def build_qa(
    events: pd.DataFrame,
    observation_universe: pd.DataFrame,
    raw14_denominator: pd.DataFrame,
    evidence: pd.DataFrame,
    *,
    source_universe_rows: int,
    frequency_max_diff: float,
    frequency_bad_zones: int,
) -> pd.DataFrame:
    """Create one compact deployment QA artifact."""
    runtime_universe_rows = len(observation_universe)
    raw14_denominator_rows = len(raw14_denominator)
    raw14_reconstructed_rows = int(
        raw14_denominator["eligible_observations"].sum()
    )
    raw14_row_count_match = (
        raw14_reconstructed_rows == runtime_universe_rows
    )

    if not raw14_row_count_match:
        raise ValueError(
            "Raw 14 denominator QA failed: "
            f"runtime universe={runtime_universe_rows:,}; "
            f"reconstructed={raw14_reconstructed_rows:,}."
        )

    selected_rows_in_universe = int(
        observation_universe[SELECTED_FLAG].sum()
    )

    event_ids = set(
        events[EVENT_ID_COLUMN].astype(str)
    )

    evidence_ids = set(
        evidence[EVENT_ID_COLUMN].astype(str)
    )

    row_count_match = (
        runtime_universe_rows
        == source_universe_rows
    )

    if not row_count_match:
        raise ValueError(
            "Observation-universe row-count QA failed: "
            f"source={source_universe_rows:,}; "
            f"runtime={runtime_universe_rows:,}."
        )

    if frequency_bad_zones != 0:
        raise ValueError(
            "Frequency-contract QA failed: "
            f"{frequency_bad_zones:,} mismatched zones."
        )

    if frequency_max_diff > TOLERANCE:
        raise ValueError(
            "Frequency-contract QA exceeded tolerance: "
            f"max_diff={frequency_max_diff:.12f}; "
            f"tolerance={TOLERANCE:g}."
        )

    qa = pd.DataFrame(
        [
            {
                "check": "source_observation_universe_rows",
                "value": source_universe_rows,
            },
            {
                "check": "runtime_observation_universe_rows",
                "value": runtime_universe_rows,
            },
            {
                "check": "observation_universe_row_count_match",
                "value": row_count_match,
            },
            {
                "check": "raw14_denominator_rows",
                "value": raw14_denominator_rows,
            },
            {
                "check": "raw14_denominator_observation_sum",
                "value": raw14_reconstructed_rows,
            },
            {
                "check": "raw14_denominator_row_count_match",
                "value": raw14_row_count_match,
            },
            {
                "check": "selected_event_rows",
                "value": len(events),
            },
            {
                "check": "distinct_selected_event_ids",
                "value": events[EVENT_ID_COLUMN].nunique(),
            },
            {
                "check": "selected_rows_inside_observation_universe",
                "value": selected_rows_in_universe,
            },
            {
                "check": "metric_evidence_rows",
                "value": len(evidence),
            },
            {
                "check": "distinct_metric_evidence_event_ids",
                "value": evidence[EVENT_ID_COLUMN].nunique(),
            },
            {
                "check": "distinct_metrics",
                "value": evidence["metric"].nunique(),
            },
            {
                "check": "events_missing_metric_evidence",
                "value": len(event_ids - evidence_ids),
            },
            {
                "check": "metric_evidence_without_event",
                "value": len(evidence_ids - event_ids),
            },
            {
                "check": "unknown_event_geography_rows",
                "value": int(
                    events["geography_group"]
                    .eq("Unknown")
                    .sum()
                ),
            },
            {
                "check": "unknown_event_environment_rows",
                "value": int(
                    events["environment_group"]
                    .eq("Unknown")
                    .sum()
                ),
            },
            {
                "check": "unknown_observation_geography_rows",
                "value": int(
                    observation_universe["geography_group"]
                    .eq("Unknown")
                    .sum()
                ),
            },
            {
                "check": "unknown_observation_environment_rows",
                "value": int(
                    observation_universe["environment_group"]
                    .eq("Unknown")
                    .sum()
                ),
            },
            {
                "check": "page13_frequency_max_abs_diff",
                "value": frequency_max_diff,
            },
            {
                "check": "page13_frequency_bad_zones",
                "value": frequency_bad_zones,
            },
            {
                "check": "observation_rows_missing_environment_id",
                "value": int(
                    observation_universe["mobility_regime_cluster_label"].isna().sum()
                ),
            },
            {
                "check": "selected_congestion_flag_rows",
                "value": int(
                    (
                            observation_universe[SELECTED_FLAG]
                            & observation_universe["has_congestion_oriented"]
                    ).sum()
                ),
            },
            {
                "check": "selected_demand_flag_rows",
                "value": int(
                    (
                            observation_universe[SELECTED_FLAG]
                            & observation_universe["has_positive_demand_shock"]
                    ).sum()
                ),
            },
        ]
    )

    # WHY: QA values intentionally mix counts, booleans, and floating-point
    # diagnostics. Store them as strings so the compact audit artifact has one
    # stable Parquet type instead of an ambiguous pandas object column.
    qa["value"] = qa["value"].map(str)

    return qa


def main() -> None:
    print("=" * 82)
    print("SHARED STRESS-ANOMALY RUNTIME BUILD · RAW 14 / 15 / 17")
    print("=" * 82, flush=True)

    required_paths = [
        ANOMALY_EVENT_UNIVERSE_PATH,
        ANOMALY_METRIC_DIAGNOSTICS_PATH,
        ANOMALY_ZONE_FREQUENCY_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: {path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    zone_geo = build_zone_geography()
    assignments = build_environment_assignments()

    (
        observation_universe,
        source_universe_rows,
    ) = build_observation_universe_runtime(
        zone_geo,
        assignments,
    )

    raw14_denominator = build_raw14_denominator_runtime(
        observation_universe
    )

    events = build_event_runtime(
        zone_geo,
        assignments,
    )

    evidence = build_metric_evidence_runtime(
        events
    )

    frequency_max_diff, frequency_bad_zones = (
        validate_frequency_contract(
            observation_universe
        )
    )

    progress("[8/9] Writing compact shared runtime artifacts...")

    write_parquet(
        events,
        EVENT_OUTPUT_PATH,
    )

    write_parquet(
        observation_universe,
        OBSERVATION_UNIVERSE_OUTPUT_PATH,
        row_group_size=50_000,
    )

    write_parquet(
        raw14_denominator,
        RAW14_DENOMINATOR_OUTPUT_PATH,
        row_group_size=25_000,
    )

    write_parquet(
        evidence,
        EVIDENCE_OUTPUT_PATH,
    )

    qa = build_qa(
        events,
        observation_universe,
        raw14_denominator,
        evidence,
        source_universe_rows=source_universe_rows,
        frequency_max_diff=frequency_max_diff,
        frequency_bad_zones=frequency_bad_zones,
    )

    write_parquet(
        qa,
        QA_OUTPUT_PATH,
        row_group_size=1_000,
    )

    progress("[9/9] Final artifact summary")

    print()

    print(
        f"Events       : {len(events):,} rows · "
        f"{size_mb(EVENT_OUTPUT_PATH):.2f} MiB"
    )

    print(
        f"Observations : {len(observation_universe):,} rows · "
        f"{size_mb(OBSERVATION_UNIVERSE_OUTPUT_PATH):.2f} MiB"
    )

    print(
        f"Raw 14 denom : {len(raw14_denominator):,} rows · "
        f"{size_mb(RAW14_DENOMINATOR_OUTPUT_PATH):.2f} MiB · "
        f"{int(raw14_denominator['eligible_observations'].sum()):,} observations"
    )

    print(
        f"Evidence     : {len(evidence):,} rows · "
        f"{size_mb(EVIDENCE_OUTPUT_PATH):.2f} MiB"
    )

    print(
        f"QA           : {len(qa):,} rows · "
        f"{size_mb(QA_OUTPUT_PATH):.3f} MiB"
    )

    print()
    print(qa.to_string(index=False))

    print()
    print("=" * 82)
    print(
        "BUILD COMPLETE · "
        f"{time.perf_counter() - STARTED:.1f}s"
    )
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 82)


if __name__ == "__main__":
    main()
