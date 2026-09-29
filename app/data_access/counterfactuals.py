"""
Counterfactual data-access helpers for Showcase Raw 21–25.

The Streamlit pages stay focused on controls, chart construction, and reader
interpretation. This module owns the reusable counterfactual data contracts:
- loading validated compact Raw 21 and Raw 22 artifacts;
- loading the compact Raw 23 counterfactual runtime surface;
- multimodal profile construction and Taxi-Zone peer/child lookups;
- canonical physical Taxi-Zone handling for reader-facing zone views;
- frozen h=1 Chapter 4 vs synthetic-world feature-reliance comparison;
- cross-artifact calculation QA.

Purpose-built compact artifacts remain valid storage optimizations. Keeping
their loading and validation here lets page code use one stable access layer
without coupling the UI to physical Parquet paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from app.data_access.loaders import (
    APP_ROOT,
    load_taxi_zone_connectivity,
)


FINAL_DIR = (
    APP_ROOT
    / "data"
    / "processed"
    / "5.3.1.final_tables"
)

APP_TABLE_DIR = (
    APP_ROOT
    / "data"
    / "processed"
    / "app_tables"
)

RAW23_RUNTIME_PATH = (
    APP_TABLE_DIR
    / "counterfactual_raw23_runtime.parquet"
)

PROFILE_SUMMARY_PATH = (
    FINAL_DIR
    / "counterfactual_mobility_profile_summary.parquet"
)

# Raw 21 compact temporal / braid contracts.
BRAID_SUMMARY_PATH = FINAL_DIR / "counterfactual_braid_summary.parquet"
BRAID_QA_PATH = FINAL_DIR / "counterfactual_braid_summary_qa.parquet"
PRELAUNCH_CONTEXT_PATH = FINAL_DIR / "counterfactual_prelaunch_context.parquet"
PRELAUNCH_QA_PATH = FINAL_DIR / "counterfactual_prelaunch_context_qa.parquet"
BRAID_TEMPORAL_EXPLORER_PATH = (
    FINAL_DIR / "counterfactual_braid_temporal_explorer.parquet"
)
BRAID_TEMPORAL_EXPLORER_QA_PATH = (
    FINAL_DIR / "counterfactual_braid_temporal_explorer_qa.parquet"
)
PRELAUNCH_TEMPORAL_EXPLORER_PATH = (
    FINAL_DIR / "counterfactual_prelaunch_temporal_explorer.parquet"
)
PRELAUNCH_TEMPORAL_EXPLORER_QA_PATH = (
    FINAL_DIR / "counterfactual_prelaunch_temporal_explorer_qa.parquet"
)
CONCLUSION_REGISTRY_PATH = FINAL_DIR / "counterfactual_conclusion_registry.parquet"

# Raw 22 compact spatial contract.
GEOGRAPHY_TEMPORAL_EXPLORER_PATH = (
    FINAL_DIR / "counterfactual_geography_temporal_explorer.parquet"
)
GEOGRAPHY_TEMPORAL_EXPLORER_QA_PATH = (
    FINAL_DIR / "counterfactual_geography_temporal_explorer_qa.parquet"
)


# Raw 24 frozen robustness package.
ROBUSTNESS_MATRIX_PATH = (
    FINAL_DIR / "counterfactual_robustness_matrix.parquet"
)
CONCLUSION_REGISTRY_PATH = (
    FINAL_DIR / "counterfactual_conclusion_registry.parquet"
)
POPULATION_ROBUSTNESS_PATH = (
    FINAL_DIR / "counterfactual_population_robustness.parquet"
)
SPEED_WEIGHTING_ROBUSTNESS_PATH = (
    FINAL_DIR / "counterfactual_speed_weighting_robustness.parquet"
)
TEMPORAL_ROBUSTNESS_PATH = (
    FINAL_DIR / "counterfactual_temporal_robustness.parquet"
)
GEOGRAPHY_ROBUSTNESS_PATH = (
    FINAL_DIR / "counterfactual_geography_robustness.parquet"
)
HORIZON_ROBUSTNESS_PATH = (
    FINAL_DIR / "counterfactual_horizon_robustness.parquet"
)
PRE_CP_CALIBRATION_PATH = (
    FINAL_DIR / "counterfactual_pre_cp_calibration.parquet"
)
VALIDATION_QA_PATH = (
    FINAL_DIR / "counterfactual_validation_qa.parquet"
)
VALIDATION_HANDOFF_MANIFEST_PATH = (
    FINAL_DIR / "counterfactual_validation_handoff_manifest.parquet"
)

# Raw 24 purpose-built Taxi-Zone serving artifacts.
RAW24_ZONE_DIAGNOSTICS_PATH = (
    FINAL_DIR / "counterfactual_raw24_zone_diagnostics.parquet"
)
RAW24_ZONE_WEEKLY_PATH = (
    FINAL_DIR / "counterfactual_raw24_zone_weekly.parquet"
)
RAW24_ZONE_SCOUT_QA_PATH = (
    FINAL_DIR / "counterfactual_raw24_zone_scout_qa.parquet"
)


# Raw 25 compact calibration serving artifacts.
RAW25_JOB_HISTOGRAM_PATH = (
    FINAL_DIR / "counterfactual_raw25_job_histogram.parquet"
)
RAW25_JOB_SUMMARY_PATH = (
    FINAL_DIR / "counterfactual_raw25_job_summary.parquet"
)
RAW25_QA_PATH = (
    FINAL_DIR / "counterfactual_raw25_qa.parquet"
)
RAW25_SLICE_SCOUT_PATH = (
    FINAL_DIR / "counterfactual_raw25_slice_scout.parquet"
)

# Reader-facing physical Taxi Zone normalization used by Raw 24.
COUNTERFACTUAL_CANONICAL_ZONE_ALIASES = {
    57: 56,
    105: 103,
}
COUNTERFACTUAL_UNKNOWN_ZONE_IDS = {
    264,
    265,
}

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

METRIC_LABELS = {
    "taxi_trip_count": "Taxi trips",
    "taxi_avg_trip_speed": "Taxi speed",
    "fhvhv_trip_count": "FHVHV trips",
    "fhvhv_avg_trip_speed": "FHVHV speed",
    "subway_ridership": "Subway ridership",
}

METRIC_MODALITY = {
    "taxi_trip_count": "taxi",
    "taxi_avg_trip_speed": "taxi",
    "fhvhv_trip_count": "fhvhv",
    "fhvhv_avg_trip_speed": "fhvhv",
    "subway_ridership": "subway",
}

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

POLICY_LABELS = {
    "cbd": "CBD",
    "adjacent_to_cbd": "Adjacent to CBD",
    "gateway_to_cbd": "Gateway to CBD",
    "non_cbd": "Non-CBD",
}

POLICY_ORDER = [
    "cbd",
    "adjacent_to_cbd",
    "gateway_to_cbd",
    "non_cbd",
]

DAYPARTS = [
    "overnight",
    "am_peak",
    "midday",
    "pm_peak",
    "evening",
]

TEMPORAL_BUCKETS = [
    f"{day_type}_{daypart}"
    for day_type in ["weekday", "weekend"]
    for daypart in DAYPARTS
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
    "Overnight": [
        "weekday_overnight",
        "weekend_overnight",
    ],
    "AM peak": [
        "weekday_am_peak",
        "weekend_am_peak",
    ],
    "Midday": [
        "weekday_midday",
        "weekend_midday",
    ],
    "PM peak": [
        "weekday_pm_peak",
        "weekend_pm_peak",
    ],
    "Evening": [
        "weekday_evening",
        "weekend_evening",
    ],
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

RAW23_RUNTIME_COLUMNS = [
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
]

PROFILE_REQUIRED_COLUMNS = [
    "profile_id",
    "profile_type",
    "profile_label",
    "profile_context",
    "metric",
    "horizon",
    "profile_rows",
    "mean_gap_mae_units",
]


RAW21_BRAID_REQUIRED_COLUMNS = {
    "summary_grain",
    "period_start",
    "period_complete",
    "week_start",
    "geography_type",
    "geography_value",
    "metric",
    "horizon",
    "support_rows",
    "observed_level",
    "no_cp_level",
    "counterfactual_gap_pct",
}

RAW21_PRELAUNCH_REQUIRED_COLUMNS = {
    "summary_grain",
    "period_start",
    "period_complete",
    "week_start",
    "geography_type",
    "geography_value",
    "metric",
    "support_rows",
    "observed_level",
}

RAW21_TEMPORAL_REQUIRED_COLUMNS = {
    "week_start",
    "period_complete",
    "target_temporal_bucket",
    "taxi_zone_id",
    "zone",
    "borough",
    "cbd_spatial_category",
    "pre_cp_mobility_environment",
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

RAW21_PRELAUNCH_TEMPORAL_REQUIRED_COLUMNS = {
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
}

RAW21_CONCLUSION_REQUIRED_COLUMNS = {
    "metric",
    "horizon",
    "primary_gap_pct",
}

RAW24_ROBUSTNESS_REQUIRED_COLUMNS = {
    "matrix": {
        "Target",
        "metric",
        "horizon",
        "baseline_abs_gap_pct",
        "baseline_gap_direction",
        "baseline_result_summary",
        "evidence_status",
        "caution_note",
        "supporting_view",
        "absolute_gap_pct_shift_pp",
        "monthly_direction_agreement_pct",
        "monthly_sign_flips",
        "geography_mean_direction_agreement_pct",
        "geography_min_direction_agreement_pct",
        "aggregate_horizons_same_direction",
        "aggregate_horizon_spread_pp",
        "row_direction_agreement_pct",
        "median_absolute_mae_units",
        "share_abs_ge_1_mae_pct",
        "share_abs_ge_2_mae_pct",
        "showcase_include",
    },
    "registry": {
        "Target",
        "metric",
        "horizon",
        "primary_gap_pct",
        "baseline_gap_direction",
        "baseline_result_summary",
        "evidence_status",
        "caution_note",
        "supporting_view",
        "showcase_include",
    },
    "population": {
        "Target",
        "metric",
        "horizon",
        "primary_gap_pct",
        "native_gap_pct",
        "gap_pct_shift_pp",
        "absolute_gap_pct_shift_pp",
        "same_direction",
    },
    "speed_weighting": {
        "Target",
        "metric",
        "horizon",
        "world_specific_gap_pct",
        "common_observed_gap_pct",
        "gap_pct_shift_pp",
        "same_direction",
    },
    "temporal": {
        "Target",
        "metric",
        "horizon",
        "month_count",
        "monthly_direction_agreement_pct",
        "monthly_gap_pct_min",
        "monthly_gap_pct_max",
        "monthly_sign_flips",
    },
    "geography": {
        "Target",
        "metric",
        "horizon",
        "geography_dimension",
        "group_count",
        "direction_agreement_pct",
        "gap_pct_min",
        "gap_pct_max",
    },
    "horizon": {
        "Target",
        "metric",
        "h1_gap_pct",
        "h2_gap_pct",
        "h5_gap_pct",
        "aggregate_horizons_same_direction",
        "aggregate_horizon_spread_pp",
        "median_row_horizon_spread_pct",
        "row_direction_agreement_pct",
    },
    "calibration": {
        "Target",
        "metric",
        "horizon",
        "median_signed_mae_units",
        "median_absolute_mae_units",
        "p90_absolute_mae_units",
        "share_abs_ge_1_mae_pct",
        "share_abs_ge_2_mae_pct",
        "rows",
    },
}

RAW24_ZONE_DIAGNOSTIC_REQUIRED_COLUMNS = {
    "metric",
    "taxi_zone_id",
    "zone",
    "borough",
    "pre_weeks",
    "post_weeks",
    "launch_jump_pct",
    "direction_persistence_pct",
    "horizon_agreement_pct",
    "pre_range_share_pct",
    "median_abs_gap_relative_to_pre_pct",
    "median_h1_h5_spread_relative_to_pre_pct",
    "median_h1_gap_native",
    "strong_scout_index",
    "concern_scout_index",
}

RAW24_ZONE_WEEKLY_REQUIRED_COLUMNS = {
    "metric",
    "taxi_zone_id",
    "zone",
    "borough",
    "week_start",
    "period",
    "observed_level",
    "no_cp_h1",
    "no_cp_h2",
    "no_cp_h5",
}


RAW25_JOB_HISTOGRAM_REQUIRED_COLUMNS = {
    "metric",
    "horizon",
    "bin_mid",
    "density_pct",
    "overflow",
}

RAW25_JOB_SUMMARY_REQUIRED_COLUMNS = {
    "metric",
    "horizon",
    "pre_cp_validation_system_mae",
    "median_absolute_mae_units",
    "p90_absolute_mae_units",
    "share_abs_ge_1_mae_pct",
    "share_abs_ge_2_mae_pct",
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
}

RAW25_SLICE_SCOUT_REQUIRED_COLUMNS = {
    "metric",
    "horizon",
    "geography_type",
    "geography_id",
    "geography_label",
    "day_type",
    "daypart",
    "rows",
    "p50_abs_mae_units",
    "p75_abs_mae_units",
    "p90_abs_mae_units",
    "p95_abs_mae_units",
    "share_abs_ge_0_5_mae_pct",
    "share_abs_ge_1_0_mae_pct",
    "share_abs_ge_2_0_mae_pct",
    "share_abs_ge_3_0_mae_pct",
    "support_ok",
}

RAW25_QA_REQUIRED_COLUMNS = {
    "check_id",
    "status",
    "details",
}


RAW22_GEOGRAPHY_REQUIRED_COLUMNS = {
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
}


def _require_columns(
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
            f"{label} is missing required columns: "
            + ", ".join(missing)
        )



def _require_files(paths: list[Path], label: str) -> None:
    """Fail loudly when one or more frozen counterfactual artifacts are absent."""
    missing = [
        path
        for path in paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"{label} is missing required files:\n"
            + "\n".join(
                f"  - {path}"
                for path in missing
            )
        )


def _validate_qa_table(
    qa: pd.DataFrame,
    *,
    label: str,
) -> None:
    """Fail closed when a compact-artifact QA table reports a non-PASS check."""
    status_column = (
        "status"
        if "status" in qa.columns
        else "Status"
        if "Status" in qa.columns
        else None
    )

    if status_column is None:
        raise KeyError(
            f"{label} is missing a status column."
        )

    failed = qa.loc[
        qa[
            status_column
        ]
        .astype(str)
        .str.upper()
        .ne("PASS")
    ]

    if not failed.empty:
        raise RuntimeError(
            f"{label} contains non-PASS checks."
        )


@st.cache_data(show_spinner=False)
def load_counterfactual_braid_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load and validate the compact always-on Raw 21 artifacts.

    WHY: Raw 21 should consume a stable data-access contract rather than know
    physical Parquet paths or repeat schema / QA checks in the page.
    """
    paths = [
        BRAID_SUMMARY_PATH,
        BRAID_QA_PATH,
        PRELAUNCH_CONTEXT_PATH,
        PRELAUNCH_QA_PATH,
        BRAID_TEMPORAL_EXPLORER_QA_PATH,
        PRELAUNCH_TEMPORAL_EXPLORER_QA_PATH,
        CONCLUSION_REGISTRY_PATH,
    ]
    _require_files(
        paths,
        "Raw 21 counterfactual contract",
    )

    braid = pd.read_parquet(
        BRAID_SUMMARY_PATH
    )
    braid_qa = pd.read_parquet(
        BRAID_QA_PATH
    )
    prelaunch = pd.read_parquet(
        PRELAUNCH_CONTEXT_PATH
    )
    prelaunch_qa = pd.read_parquet(
        PRELAUNCH_QA_PATH
    )
    temporal_qa = pd.read_parquet(
        BRAID_TEMPORAL_EXPLORER_QA_PATH
    )
    prelaunch_temporal_qa = pd.read_parquet(
        PRELAUNCH_TEMPORAL_EXPLORER_QA_PATH
    )
    conclusions = pd.read_parquet(
        CONCLUSION_REGISTRY_PATH
    )

    _require_columns(
        braid,
        RAW21_BRAID_REQUIRED_COLUMNS,
        "counterfactual_braid_summary",
    )
    _require_columns(
        prelaunch,
        RAW21_PRELAUNCH_REQUIRED_COLUMNS,
        "counterfactual_prelaunch_context",
    )
    _require_columns(
        conclusions,
        RAW21_CONCLUSION_REQUIRED_COLUMNS,
        "counterfactual_conclusion_registry",
    )

    for label, qa in [
        ("counterfactual_braid_summary_qa", braid_qa),
        ("counterfactual_prelaunch_context_qa", prelaunch_qa),
        ("counterfactual_braid_temporal_explorer_qa", temporal_qa),
        (
            "counterfactual_prelaunch_temporal_explorer_qa",
            prelaunch_temporal_qa,
        ),
    ]:
        _validate_qa_table(
            qa,
            label=label,
        )

    for frame in [
        braid,
        prelaunch,
    ]:
        for column in [
            "period_start",
            "week_start",
        ]:
            frame[
                column
            ] = pd.to_datetime(
                frame[
                    column
                ],
                errors="coerce",
            )

    braid[
        "horizon"
    ] = pd.to_numeric(
        braid[
            "horizon"
        ],
        errors="raise",
    ).astype(int)

    conclusions[
        "horizon"
    ] = pd.to_numeric(
        conclusions[
            "horizon"
        ],
        errors="raise",
    ).astype(int)

    return (
        braid,
        braid_qa,
        prelaunch,
        prelaunch_qa,
        temporal_qa,
        prelaunch_temporal_qa,
        conclusions,
    )


@st.cache_data(show_spinner=False)
def load_counterfactual_temporal_explorer_metric(
    metric: str,
) -> pd.DataFrame:
    """Load one post-launch Raw 21 temporal-explorer metric."""
    _require_files(
        [
            BRAID_TEMPORAL_EXPLORER_PATH,
        ],
        "Raw 21 post-launch temporal explorer",
    )

    frame = pd.read_parquet(
        BRAID_TEMPORAL_EXPLORER_PATH,
        filters=[
            (
                "metric",
                "==",
                metric,
            )
        ],
    )

    _require_columns(
        frame,
        RAW21_TEMPORAL_REQUIRED_COLUMNS,
        "counterfactual_braid_temporal_explorer",
    )

    frame = frame.copy()
    frame[
        "week_start"
    ] = pd.to_datetime(
        frame[
            "week_start"
        ],
        errors="raise",
    )
    frame[
        "taxi_zone_id"
    ] = pd.to_numeric(
        frame[
            "taxi_zone_id"
        ],
        errors="raise",
    ).astype("Int64")

    return frame


@st.cache_data(show_spinner=False)
def load_counterfactual_prelaunch_temporal_metric(
    metric: str,
) -> pd.DataFrame:
    """Load one prelaunch Raw 21 temporal-explorer metric."""
    _require_files(
        [
            PRELAUNCH_TEMPORAL_EXPLORER_PATH,
        ],
        "Raw 21 prelaunch temporal explorer",
    )

    frame = pd.read_parquet(
        PRELAUNCH_TEMPORAL_EXPLORER_PATH,
        filters=[
            (
                "metric",
                "==",
                metric,
            )
        ],
    )

    _require_columns(
        frame,
        RAW21_PRELAUNCH_TEMPORAL_REQUIRED_COLUMNS,
        "counterfactual_prelaunch_temporal_explorer",
    )

    frame = frame.copy()
    frame[
        "week_start"
    ] = pd.to_datetime(
        frame[
            "week_start"
        ],
        errors="raise",
    )
    frame[
        "taxi_zone_id"
    ] = pd.to_numeric(
        frame[
            "taxi_zone_id"
        ],
        errors="raise",
    ).astype("Int64")

    return frame


@st.cache_data(show_spinner=False)
def load_counterfactual_geography_explorer() -> pd.DataFrame:
    """
    Load and validate the compact Raw 22 canonical Taxi-Zone explorer.

    WHY: Raw 22 can keep its spatial visualization logic while delegating
    storage, schema, and QA ownership to the shared counterfactual access layer.
    """
    _require_files(
        [
            GEOGRAPHY_TEMPORAL_EXPLORER_PATH,
            GEOGRAPHY_TEMPORAL_EXPLORER_QA_PATH,
        ],
        "Raw 22 counterfactual geography contract",
    )

    frame = pd.read_parquet(
        GEOGRAPHY_TEMPORAL_EXPLORER_PATH
    )
    qa = pd.read_parquet(
        GEOGRAPHY_TEMPORAL_EXPLORER_QA_PATH
    )

    _require_columns(
        frame,
        RAW22_GEOGRAPHY_REQUIRED_COLUMNS,
        "counterfactual_geography_temporal_explorer",
    )
    _validate_qa_table(
        qa,
        label="counterfactual_geography_temporal_explorer_qa",
    )

    frame = frame.copy()
    frame[
        "week_start"
    ] = pd.to_datetime(
        frame[
            "week_start"
        ],
        errors="raise",
    ).dt.normalize()
    frame[
        "canonical_taxi_zone_id"
    ] = pd.to_numeric(
        frame[
            "canonical_taxi_zone_id"
        ],
        errors="raise",
    ).astype("Int64")

    return frame


@st.cache_data(show_spinner=False)
def load_counterfactual_surface() -> pd.DataFrame:
    """
    Load Raw 23's compact exact-row runtime surface.

    WHY:
    Raw 23 only needs the frozen primary population and eleven columns from the
    much wider Chapter 5 gap surface. The deployment artifact stores exactly
    that subset without rounding or aggregation, so all existing Raw 23 slicing
    and profile calculations retain their original semantics.
    """
    _require_files(
        [RAW23_RUNTIME_PATH],
        "Raw 23 counterfactual runtime contract",
    )

    frame = pd.read_parquet(
        RAW23_RUNTIME_PATH,
        columns=RAW23_RUNTIME_COLUMNS,
    )

    _require_columns(
        frame,
        RAW23_RUNTIME_COLUMNS,
        "counterfactual_raw23_runtime",
    )

    frame["target_date"] = pd.to_datetime(
        frame["target_date"],
        errors="raise",
    ).dt.normalize()

    frame["horizon"] = pd.to_numeric(
        frame["horizon"],
        errors="raise",
    ).astype(int)

    frame["taxi_zone_id"] = pd.to_numeric(
        frame["taxi_zone_id"],
        errors="coerce",
    ).astype("Int64")

    frame["canonical_location_id"] = pd.to_numeric(
        frame["canonical_location_id"],
        errors="coerce",
    ).astype("Int64")

    frame["counterfactual_gap_mae_units"] = pd.to_numeric(
        frame["counterfactual_gap_mae_units"],
        errors="coerce",
    )

    return frame.loc[
        frame["metric"].isin(METRIC_ORDER)
        & frame["horizon"].isin(HORIZONS)
        & frame["target_date"].between(
            CP_START,
            CP_END,
            inclusive="both",
        )
    ].copy()


@st.cache_data(show_spinner=False)
def load_primary_counterfactual_surface() -> pd.DataFrame:
    """
    Return the frozen reader-facing Chapter 5 primary population.

    The compact Raw 23 runtime is already filtered to primary-gap-eligible rows
    during its offline build, so this remains a compatibility wrapper for the
    existing Raw 23 API.
    """
    return load_counterfactual_surface().copy()


@st.cache_data(show_spinner=False)
def load_authoritative_profile_summary() -> pd.DataFrame:
    """Load the compact Chapter 5 Taxi-Zone profile summary."""
    frame = pd.read_parquet(
        PROFILE_SUMMARY_PATH
    )

    _require_columns(
        frame,
        PROFILE_REQUIRED_COLUMNS,
        "counterfactual_mobility_profile_summary",
    )

    return frame


def validate_counterfactual_contract() -> str:
    """
    Validate the profile calculation against the frozen compact artifact.

    Coverage differs between the two frozen Chapter 5 artifacts, so equality is
    required wherever profile keys overlap.
    """
    source = load_counterfactual_surface()
    authoritative = load_authoritative_profile_summary()

    qa_source = source.loc[
        source[
            "counterfactual_gap_mae_units"
        ].notna()
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

    passed = (
        len(matched) > 0
        and row_diff.fillna(0).eq(0).all()
        and value_diff.fillna(0).le(
            1e-10
        ).all()
    )

    if not passed:
        raise RuntimeError(
            "Raw 23 counterfactual profile QA failed: "
            f"matched={len(matched):,}; "
            f"max row diff="
            f"{float(row_diff.max()) if row_diff.notna().any() else 0:.12g}; "
            f"max MAE diff="
            f"{float(value_diff.max()) if value_diff.notna().any() else 0:.12g}."
        )

    return (
        f"matched={len(matched):,}; "
        f"rebuilt_only="
        f"{int(comparison['_merge'].eq('left_only').sum()):,}; "
        f"authoritative_only="
        f"{int(comparison['_merge'].eq('right_only').sum()):,}"
    )


def _subset_time(
    frame: pd.DataFrame,
    *,
    period: str,
    day_type: str,
    daypart: str,
) -> pd.DataFrame:
    """Apply one Raw 23 temporal slice."""
    start_date, end_date = PERIODS[
        period
    ]

    valid_buckets = (
        set(DAY_TYPE_BUCKETS[day_type])
        & set(DAYPART_BUCKETS[daypart])
    )

    return frame.loc[
        frame["target_date"].between(
            start_date,
            end_date,
            inclusive="both",
        )
        & frame[
            "target_temporal_bucket"
        ].isin(
            valid_buckets
        )
    ].copy()


def _rms_profile(
    values: pd.Series,
) -> float:
    """Return overall standardized distance from zero."""
    clean = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if clean.empty:
        return np.nan

    return float(
        np.sqrt(
            np.mean(
                np.square(clean)
            )
        )
    )


def _sign_pattern(
    row: pd.Series,
) -> str:
    """Encode signs in the fixed five-metric order."""
    result = []

    for metric in METRIC_ORDER:
        value = row.get(
            metric,
            np.nan,
        )

        if pd.isna(value):
            result.append("?")
        elif value > 0:
            result.append("+")
        elif value < 0:
            result.append("−")
        else:
            result.append("0")

    return "".join(result)


def _opposite_sign_pairs(
    row: pd.Series,
) -> int:
    """Count directional disagreements among available measures."""
    values = [
        float(row[metric])
        for metric in METRIC_ORDER
        if pd.notna(row.get(metric))
        and float(row[metric]) != 0
    ]

    count = 0

    for left_index in range(
        len(values)
    ):
        for right_index in range(
            left_index + 1,
            len(values),
        ):
            if np.sign(
                values[left_index]
            ) != np.sign(
                values[right_index]
            ):
                count += 1

    return count



def _canonicalize_physical_taxi_zone_rows(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert source Taxi Zone IDs to canonical physical geography for zone views.

    WHY: source IDs 56 and 57 both represent the physical Corona geography.
    Reader-facing Taxi Zone views must aggregate them as one physical place,
    while higher-level mobility-environment grouping must keep each source
    row's frozen Pre-CP environment assignment. Call this helper only after any
    parent geography/environment filter has already been applied.
    """
    result = frame.copy()

    canonical = pd.to_numeric(
        result[
            "canonical_location_id"
        ],
        errors="coerce",
    ).astype("Int64")

    result[
        "taxi_zone_id"
    ] = canonical

    # Preserve a deterministic physical label. For canonical 56, the physical
    # reader-facing place is Corona regardless of which source member supplied
    # the row. Other canonical IDs retain their existing zone label.
    result.loc[
        canonical.eq(56),
        "zone",
    ] = "Corona"

    return result


@st.cache_data(show_spinner=False)
def get_counterfactual_profiles(
    *,
    grouping_name: str,
    horizon: int,
    period: str = "Full post-CP period",
    day_type: str = "All days",
    daypart: str = "All dayparts",
) -> pd.DataFrame:
    """
    Return one multimodal standardized profile per requested geography.

    The statistic matches the Chapter 5 mobility-profile definition: mean
    row-level counterfactual gap in Pre-CP validation-MAE units.
    """
    if grouping_name not in GROUPINGS:
        raise ValueError(
            f"Unsupported grouping: {grouping_name}"
        )

    source = load_primary_counterfactual_surface()

    grouping = GROUPINGS[
        grouping_name
    ]

    group_id = grouping[
        "group_id"
    ]
    group_label = grouping[
        "group_label"
    ]

    group_columns = list(
        dict.fromkeys(
            [
                group_id,
                group_label,
            ]
        )
    )

    scoped = source.loc[
        source["horizon"].eq(
            int(horizon)
        )
    ].copy()

    if grouping_name == "Taxi Zone":
        scoped = _canonicalize_physical_taxi_zone_rows(
            scoped
        )

    scoped = _subset_time(
        scoped,
        period=period,
        day_type=day_type,
        daypart=daypart,
    )

    mask = pd.Series(
        True,
        index=scoped.index,
    )

    for column in group_columns:
        mask &= scoped[
            column
        ].notna()

    scoped = scoped.loc[
        mask
    ].copy()

    if scoped.empty:
        return pd.DataFrame()

    long_summary = (
        scoped.groupby(
            [
                *group_columns,
                "metric",
            ],
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

    wide = (
        long_summary.pivot(
            index=group_columns,
            columns="metric",
            values="mean_gap_mae_units",
        )
        .reset_index()
    )

    support = (
        long_summary.groupby(
            group_columns,
            observed=True,
            dropna=False,
        )["support_rows"]
        .sum()
        .reset_index()
    )

    wide = wide.merge(
        support,
        on=group_columns,
        how="left",
        validate="one_to_one",
    )

    for metric in METRIC_ORDER:
        if metric not in wide.columns:
            wide[metric] = np.nan

    wide["multimodal_rms"] = wide[
        METRIC_ORDER
    ].apply(
        _rms_profile,
        axis=1,
    )

    wide["max_abs_mae"] = wide[
        METRIC_ORDER
    ].abs().max(
        axis=1
    )

    wide["sign_pattern"] = wide.apply(
        _sign_pattern,
        axis=1,
    )

    wide["opposite_sign_pairs"] = wide.apply(
        _opposite_sign_pairs,
        axis=1,
    )

    wide["grouping"] = grouping_name

    if grouping_name == "Policy geography":
        wide["_policy_order"] = pd.Categorical(
            wide[
                "cbd_spatial_category"
            ],
            categories=POLICY_ORDER,
            ordered=True,
        )

        wide = (
            wide.sort_values(
                "_policy_order"
            )
            .drop(
                columns="_policy_order"
            )
            .reset_index(
                drop=True
            )
        )

    return wide


def get_profile_label(
    grouping_name: str,
    value: object,
) -> str:
    """Return a reader-facing geography label."""
    if grouping_name == "Policy geography":
        return POLICY_LABELS.get(
            str(value),
            str(value)
            .replace("_", " ")
            .title(),
        )

    return str(value)


def get_profile_label_column(
    grouping_name: str,
) -> str:
    """Return the display-label field for one grouping lens."""
    return GROUPINGS[
        grouping_name
    ]["group_label"]


def get_complete_profiles(
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Keep profiles where all five measures are available."""
    if profiles.empty:
        return profiles.copy()

    return profiles.loc[
        profiles[
            METRIC_ORDER
        ]
        .notna()
        .all(axis=1)
    ].copy()


def get_strongest_profiles(
    profiles: pd.DataFrame,
    *,
    count: int,
) -> pd.DataFrame:
    """Return complete profiles with the largest overall divergence."""
    return (
        get_complete_profiles(
            profiles
        )
        .sort_values(
            [
                "multimodal_rms",
                "max_abs_mae",
            ],
            ascending=False,
        )
        .head(
            count
        )
        .reset_index(
            drop=True
        )
    )


def get_same_level_peers(
    *,
    grouping_name: str,
    selected_value: object,
    horizon: int,
    period: str,
    day_type: str,
    daypart: str,
) -> pd.DataFrame:
    """
    Return peers at the same analytical level.

    Non-Taxi-Zone levels return all available groups at that level.
    Taxi Zone peers return transportation-network neighbors.
    """
    profiles = get_counterfactual_profiles(
        grouping_name=grouping_name,
        horizon=horizon,
        period=period,
        day_type=day_type,
        daypart=daypart,
    )

    if grouping_name != "Taxi Zone":
        return profiles

    if profiles.empty:
        return profiles

    canonical_id = int(
        float(
            selected_value
        )
    )

    source = load_primary_counterfactual_surface()

    connectivity = load_taxi_zone_connectivity()

    _require_columns(
        connectivity,
        {
            "location_id",
            "connected_location_id",
        },
        "taxi_zone_connectivity",
    )

    neighbor_canonical_ids = set(
        pd.to_numeric(
            connectivity.loc[
                pd.to_numeric(
                    connectivity[
                        "location_id"
                    ],
                    errors="coerce",
                ).eq(
                    canonical_id
                ),
                "connected_location_id",
            ],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .tolist()
    )

    if not neighbor_canonical_ids:
        return profiles.iloc[
            0:0
        ].copy()

    return (
        profiles.loc[
            profiles[
                "taxi_zone_id"
            ]
            .astype("Int64")
            .isin(
                neighbor_canonical_ids
            )
        ]
        .sort_values(
            "multimodal_rms",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def get_child_taxi_zone_profiles(
    *,
    parent_grouping: str,
    selected_value: object,
    horizon: int,
    period: str,
    day_type: str,
    daypart: str,
) -> pd.DataFrame:
    """
    Return Taxi-Zone profiles contained by the selected parent geography.

    Mobility-environment membership intentionally remains tied to the source
    Taxi Zone assignment carried by Chapter 5.
    """
    if parent_grouping == "Taxi Zone":
        raise ValueError(
            "Taxi Zone does not have child Taxi Zones."
        )

    source = load_primary_counterfactual_surface()

    parent_column = GROUPINGS[
        parent_grouping
    ]["group_id"]

    scoped = source.loc[
        source[
            parent_column
        ]
        .astype(str)
        .eq(
            str(selected_value)
        )
    ].copy()

    if scoped.empty:
        return pd.DataFrame()

    scoped = scoped.loc[
        scoped[
            "horizon"
        ].eq(
            int(horizon)
        )
    ].copy()

    scoped = _subset_time(
        scoped,
        period=period,
        day_type=day_type,
        daypart=daypart,
    )

    if scoped.empty:
        return pd.DataFrame()

    scoped = _canonicalize_physical_taxi_zone_rows(
        scoped
    )

    long_summary = (
        scoped.groupby(
            [
                "taxi_zone_id",
                "zone",
                "metric",
            ],
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

    wide = (
        long_summary.pivot(
            index=[
                "taxi_zone_id",
                "zone",
            ],
            columns="metric",
            values="mean_gap_mae_units",
        )
        .reset_index()
    )

    support = (
        long_summary.groupby(
            [
                "taxi_zone_id",
                "zone",
            ],
            observed=True,
            dropna=False,
        )["support_rows"]
        .sum()
        .reset_index()
    )

    wide = wide.merge(
        support,
        on=[
            "taxi_zone_id",
            "zone",
        ],
        how="left",
        validate="one_to_one",
    )

    for metric in METRIC_ORDER:
        if metric not in wide.columns:
            wide[
                metric
            ] = np.nan

    wide["multimodal_rms"] = wide[
        METRIC_ORDER
    ].apply(
        _rms_profile,
        axis=1,
    )

    wide["max_abs_mae"] = wide[
        METRIC_ORDER
    ].abs().max(
        axis=1
    )

    wide["sign_pattern"] = wide.apply(
        _sign_pattern,
        axis=1,
    )

    wide["opposite_sign_pairs"] = wide.apply(
        _opposite_sign_pairs,
        axis=1,
    )

    wide["grouping"] = "Taxi Zone"

    return (
        wide.sort_values(
            [
                "multimodal_rms",
                "max_abs_mae",
            ],
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )




@st.cache_data(show_spinner=False)
def get_h1_feature_reliance_comparison() -> pd.DataFrame:
    """
    Return the frozen h=1 feature-reliance comparison from 5.2.1 §2O/2P.

    Each Target × Environment pair is normalized to 100%. The environments are:
    - Chapter 4 forecaster
    - Synthetic-world forecaster

    These values describe the model's internal information mix, not causal
    importance or policy effect size.
    """
    rows = [
        # FHVHV average speed
        ("fhvhv_avg_trip_speed", "Tree", "Chapter 4 forecaster", "Calendar timing", 32.920),
        ("fhvhv_avg_trip_speed", "Tree", "Chapter 4 forecaster", "Mobility history", 44.747),
        ("fhvhv_avg_trip_speed", "Tree", "Chapter 4 forecaster", "Multimodal context", 2.186),
        ("fhvhv_avg_trip_speed", "Tree", "Chapter 4 forecaster", "Policy context", 0.061),
        ("fhvhv_avg_trip_speed", "Tree", "Chapter 4 forecaster", "Spatial / network", 5.785),
        ("fhvhv_avg_trip_speed", "Tree", "Chapter 4 forecaster", "Weather", 3.083),
        ("fhvhv_avg_trip_speed", "Tree", "Chapter 4 forecaster", "Weekly seasonality", 1.277),
        ("fhvhv_avg_trip_speed", "Tree", "Chapter 4 forecaster", "Zone identity", 9.941),
        ("fhvhv_avg_trip_speed", "Tree", "Synthetic-world forecaster", "Calendar timing", 72.222),
        ("fhvhv_avg_trip_speed", "Tree", "Synthetic-world forecaster", "Mobility history", 5.656),
        ("fhvhv_avg_trip_speed", "Tree", "Synthetic-world forecaster", "Multimodal context", 1.659),
        ("fhvhv_avg_trip_speed", "Tree", "Synthetic-world forecaster", "Policy context", 0.000),
        ("fhvhv_avg_trip_speed", "Tree", "Synthetic-world forecaster", "Spatial / network", 1.436),
        ("fhvhv_avg_trip_speed", "Tree", "Synthetic-world forecaster", "Weather", 7.056),
        ("fhvhv_avg_trip_speed", "Tree", "Synthetic-world forecaster", "Weekly seasonality", 5.955),
        ("fhvhv_avg_trip_speed", "Tree", "Synthetic-world forecaster", "Zone identity", 6.016),

        # FHVHV trips
        ("fhvhv_trip_count", "Neural", "Chapter 4 forecaster", "Calendar timing", 1.239),
        ("fhvhv_trip_count", "Neural", "Chapter 4 forecaster", "Mobility history", 94.620),
        ("fhvhv_trip_count", "Neural", "Chapter 4 forecaster", "Multimodal context", 1.894),
        ("fhvhv_trip_count", "Neural", "Chapter 4 forecaster", "Spatial / network", 0.273),
        ("fhvhv_trip_count", "Neural", "Chapter 4 forecaster", "Weather", 0.000),
        ("fhvhv_trip_count", "Neural", "Chapter 4 forecaster", "Weekly seasonality", 1.491),
        ("fhvhv_trip_count", "Neural", "Chapter 4 forecaster", "Zone identity", 0.482),
        ("fhvhv_trip_count", "Neural", "Synthetic-world forecaster", "Calendar timing", 0.000),
        ("fhvhv_trip_count", "Neural", "Synthetic-world forecaster", "Mobility history", 40.042),
        ("fhvhv_trip_count", "Neural", "Synthetic-world forecaster", "Multimodal context", 7.948),
        ("fhvhv_trip_count", "Neural", "Synthetic-world forecaster", "Policy context", 0.000),
        ("fhvhv_trip_count", "Neural", "Synthetic-world forecaster", "Spatial / network", 13.177),
        ("fhvhv_trip_count", "Neural", "Synthetic-world forecaster", "Weather", 2.090),
        ("fhvhv_trip_count", "Neural", "Synthetic-world forecaster", "Weekly seasonality", 26.680),
        ("fhvhv_trip_count", "Neural", "Synthetic-world forecaster", "Zone identity", 10.063),

        # Subway ridership
        ("subway_ridership", "Neural", "Chapter 4 forecaster", "Calendar timing", 5.376),
        ("subway_ridership", "Neural", "Chapter 4 forecaster", "Mobility history", 85.820),
        ("subway_ridership", "Neural", "Chapter 4 forecaster", "Multimodal context", 4.253),
        ("subway_ridership", "Neural", "Chapter 4 forecaster", "Spatial / network", 2.126),
        ("subway_ridership", "Neural", "Chapter 4 forecaster", "Weather", 0.012),
        ("subway_ridership", "Neural", "Chapter 4 forecaster", "Weekly seasonality", 1.604),
        ("subway_ridership", "Neural", "Chapter 4 forecaster", "Zone identity", 0.809),
        ("subway_ridership", "Neural", "Synthetic-world forecaster", "Calendar timing", 0.000),
        ("subway_ridership", "Neural", "Synthetic-world forecaster", "Mobility history", 39.176),
        ("subway_ridership", "Neural", "Synthetic-world forecaster", "Multimodal context", 5.072),
        ("subway_ridership", "Neural", "Synthetic-world forecaster", "Policy context", 0.000),
        ("subway_ridership", "Neural", "Synthetic-world forecaster", "Spatial / network", 8.393),
        ("subway_ridership", "Neural", "Synthetic-world forecaster", "Weather", 1.251),
        ("subway_ridership", "Neural", "Synthetic-world forecaster", "Weekly seasonality", 36.047),
        ("subway_ridership", "Neural", "Synthetic-world forecaster", "Zone identity", 10.062),

        # Taxi average speed
        ("taxi_avg_trip_speed", "Neural", "Chapter 4 forecaster", "Calendar timing", 16.973),
        ("taxi_avg_trip_speed", "Neural", "Chapter 4 forecaster", "Mobility history", 61.986),
        ("taxi_avg_trip_speed", "Neural", "Chapter 4 forecaster", "Multimodal context", 9.190),
        ("taxi_avg_trip_speed", "Neural", "Chapter 4 forecaster", "Spatial / network", 3.731),
        ("taxi_avg_trip_speed", "Neural", "Chapter 4 forecaster", "Weather", 0.000),
        ("taxi_avg_trip_speed", "Neural", "Chapter 4 forecaster", "Weekly seasonality", 0.896),
        ("taxi_avg_trip_speed", "Neural", "Chapter 4 forecaster", "Zone identity", 7.222),
        ("taxi_avg_trip_speed", "Neural", "Synthetic-world forecaster", "Calendar timing", 0.000),
        ("taxi_avg_trip_speed", "Neural", "Synthetic-world forecaster", "Mobility history", 41.931),
        ("taxi_avg_trip_speed", "Neural", "Synthetic-world forecaster", "Multimodal context", 7.148),
        ("taxi_avg_trip_speed", "Neural", "Synthetic-world forecaster", "Policy context", 0.000),
        ("taxi_avg_trip_speed", "Neural", "Synthetic-world forecaster", "Spatial / network", 23.449),
        ("taxi_avg_trip_speed", "Neural", "Synthetic-world forecaster", "Weather", 1.367),
        ("taxi_avg_trip_speed", "Neural", "Synthetic-world forecaster", "Weekly seasonality", 3.106),
        ("taxi_avg_trip_speed", "Neural", "Synthetic-world forecaster", "Zone identity", 23.000),

        # Taxi trips
        ("taxi_trip_count", "Neural", "Chapter 4 forecaster", "Calendar timing", 3.794),
        ("taxi_trip_count", "Neural", "Chapter 4 forecaster", "Mobility history", 81.319),
        ("taxi_trip_count", "Neural", "Chapter 4 forecaster", "Multimodal context", 6.767),
        ("taxi_trip_count", "Neural", "Chapter 4 forecaster", "Spatial / network", 5.850),
        ("taxi_trip_count", "Neural", "Chapter 4 forecaster", "Weather", 0.000),
        ("taxi_trip_count", "Neural", "Chapter 4 forecaster", "Weekly seasonality", 0.729),
        ("taxi_trip_count", "Neural", "Chapter 4 forecaster", "Zone identity", 1.541),
        ("taxi_trip_count", "Neural", "Synthetic-world forecaster", "Calendar timing", 0.000),
        ("taxi_trip_count", "Neural", "Synthetic-world forecaster", "Mobility history", 56.745),
        ("taxi_trip_count", "Neural", "Synthetic-world forecaster", "Multimodal context", 5.744),
        ("taxi_trip_count", "Neural", "Synthetic-world forecaster", "Policy context", 0.000),
        ("taxi_trip_count", "Neural", "Synthetic-world forecaster", "Spatial / network", 9.605),
        ("taxi_trip_count", "Neural", "Synthetic-world forecaster", "Weather", 2.173),
        ("taxi_trip_count", "Neural", "Synthetic-world forecaster", "Weekly seasonality", 16.693),
        ("taxi_trip_count", "Neural", "Synthetic-world forecaster", "Zone identity", 9.040),
    ]

    frame = pd.DataFrame(
        rows,
        columns=[
            "metric",
            "family",
            "environment",
            "information_family",
            "reliance_share_pct",
        ],
    )

    frame["target_label"] = frame["metric"].map(METRIC_LABELS)
    frame["horizon"] = 1

    return frame


@st.cache_data(show_spinner=False)
def get_h1_synthetic_feature_reliance() -> pd.DataFrame:
    """
    Return the frozen 5.2.1 h=1 synthetic-world feature-reliance summary.

    The values are the normalized within-target reliance shares from the
    replay-validated production interpretation exercise. Neural writers use
    grouped prediction-displacement permutation reliance; the Tree writer uses
    grouped absolute SHAP share. These percentages describe the mix of
    information used within each model, not causal effects.
    """
    rows = [
        # FHVHV average speed — Tree writer.
        ("fhvhv_avg_trip_speed", "tree", "Origin calendar", 55.37),
        ("fhvhv_avg_trip_speed", "tree", "Target calendar", 8.94),
        ("fhvhv_avg_trip_speed", "tree", "Annual calendar cycle", 7.91),
        ("fhvhv_avg_trip_speed", "tree", "Weather", 7.06),
        ("fhvhv_avg_trip_speed", "tree", "Series identity", 6.02),
        ("fhvhv_avg_trip_speed", "tree", "Weekly seasonal state", 5.95),
        ("fhvhv_avg_trip_speed", "tree", "Recent sequence history", 3.68),
        ("fhvhv_avg_trip_speed", "tree", "Multimodal origin", 1.66),
        ("fhvhv_avg_trip_speed", "tree", "Comparable Daypart history", 1.66),
        ("fhvhv_avg_trip_speed", "tree", "Static geography", 0.76),
        ("fhvhv_avg_trip_speed", "tree", "Transportation connected", 0.68),
        ("fhvhv_avg_trip_speed", "tree", "Current mobility state", 0.32),
        ("fhvhv_avg_trip_speed", "tree", "Policy-regime context", 0.00),

        # FHVHV trips — Neural writer.
        ("fhvhv_trip_count", "neural", "Recent mobility history", 40.04),
        ("fhvhv_trip_count", "neural", "Weekly STL seasonal state", 26.68),
        ("fhvhv_trip_count", "neural", "Spatial / network context", 13.18),
        ("fhvhv_trip_count", "neural", "Taxi Zone identity", 10.06),
        ("fhvhv_trip_count", "neural", "Multimodal / connected context", 7.95),
        ("fhvhv_trip_count", "neural", "Weather", 2.09),
        ("fhvhv_trip_count", "neural", "Annual / calendar timing", 0.00),
        ("fhvhv_trip_count", "neural", "Congestion-pricing regime", 0.00),

        # Subway ridership — Neural writer.
        ("subway_ridership", "neural", "Recent mobility history", 39.18),
        ("subway_ridership", "neural", "Weekly STL seasonal state", 36.05),
        ("subway_ridership", "neural", "Taxi Zone identity", 10.06),
        ("subway_ridership", "neural", "Spatial / network context", 8.39),
        ("subway_ridership", "neural", "Multimodal / connected context", 5.07),
        ("subway_ridership", "neural", "Weather", 1.25),
        ("subway_ridership", "neural", "Annual / calendar timing", 0.00),
        ("subway_ridership", "neural", "Congestion-pricing regime", 0.00),

        # Taxi average speed — Neural writer.
        ("taxi_avg_trip_speed", "neural", "Recent mobility history", 41.93),
        ("taxi_avg_trip_speed", "neural", "Spatial / network context", 23.45),
        ("taxi_avg_trip_speed", "neural", "Taxi Zone identity", 23.00),
        ("taxi_avg_trip_speed", "neural", "Multimodal / connected context", 7.15),
        ("taxi_avg_trip_speed", "neural", "Weekly STL seasonal state", 3.11),
        ("taxi_avg_trip_speed", "neural", "Weather", 1.37),
        ("taxi_avg_trip_speed", "neural", "Annual / calendar timing", 0.00),
        ("taxi_avg_trip_speed", "neural", "Congestion-pricing regime", 0.00),

        # Taxi trips — Neural writer.
        ("taxi_trip_count", "neural", "Recent mobility history", 56.74),
        ("taxi_trip_count", "neural", "Weekly STL seasonal state", 16.69),
        ("taxi_trip_count", "neural", "Spatial / network context", 9.60),
        ("taxi_trip_count", "neural", "Taxi Zone identity", 9.04),
        ("taxi_trip_count", "neural", "Multimodal / connected context", 5.74),
        ("taxi_trip_count", "neural", "Weather", 2.17),
        ("taxi_trip_count", "neural", "Annual / calendar timing", 0.00),
        ("taxi_trip_count", "neural", "Congestion-pricing regime", 0.00),
    ]

    frame = pd.DataFrame(
        rows,
        columns=[
            "metric",
            "family",
            "information_source",
            "reliance_share_pct",
        ],
    )

    frame["interpretation_method"] = np.where(
        frame["family"].eq("tree"),
        "Grouped SHAP share",
        "Grouped permutation reliance",
    )

    return frame


def get_same_borough_taxi_zone_profiles(
    *,
    selected_zone_id: int,
    horizon: int,
    period: str,
    day_type: str,
    daypart: str,
) -> pd.DataFrame:
    """Return other Taxi Zones in the selected zone's borough."""
    source = load_primary_counterfactual_surface()

    selected = source.loc[
        source[
            "taxi_zone_id"
        ].eq(
            int(selected_zone_id)
        )
    ]

    if selected.empty:
        return pd.DataFrame()

    borough = (
        selected[
            "borough"
        ]
        .dropna()
        .astype(str)
        .iloc[0]
    )

    children = get_child_taxi_zone_profiles(
        parent_grouping="Borough",
        selected_value=borough,
        horizon=horizon,
        period=period,
        day_type=day_type,
        daypart=daypart,
    )

    return (
        children.loc[
            ~children[
                "taxi_zone_id"
            ].eq(
                int(selected_zone_id)
            )
        ]
        .reset_index(
            drop=True
        )
    )

@st.cache_data(show_spinner=False)
def load_counterfactual_robustness_inputs() -> dict[str, pd.DataFrame]:
    """
    Load and validate Raw 24's frozen 5.3.1 robustness package.

    WHY:
    Raw 24 should own the analytical interpretation and visualization, while
    this module owns physical files and schema contracts.
    """
    paths = {
        "matrix": ROBUSTNESS_MATRIX_PATH,
        "registry": CONCLUSION_REGISTRY_PATH,
        "population": POPULATION_ROBUSTNESS_PATH,
        "speed_weighting": SPEED_WEIGHTING_ROBUSTNESS_PATH,
        "temporal": TEMPORAL_ROBUSTNESS_PATH,
        "geography": GEOGRAPHY_ROBUSTNESS_PATH,
        "horizon": HORIZON_ROBUSTNESS_PATH,
        "calibration": PRE_CP_CALIBRATION_PATH,
        "qa": VALIDATION_QA_PATH,
        "manifest": VALIDATION_HANDOFF_MANIFEST_PATH,
    }

    _require_files(
        list(paths.values()),
        "Raw 24 robustness contract",
    )

    frames = {
        name: pd.read_parquet(path)
        for name, path in paths.items()
    }

    for name, required in RAW24_ROBUSTNESS_REQUIRED_COLUMNS.items():
        _require_columns(
            frames[name],
            required,
            f"counterfactual_{name}_robustness"
            if name not in {"matrix", "registry", "calibration"}
            else {
                "matrix": "counterfactual_robustness_matrix",
                "registry": "counterfactual_conclusion_registry",
                "calibration": "counterfactual_pre_cp_calibration",
            }[name],
        )

    return frames


@st.cache_data(show_spinner=False)
def load_counterfactual_raw24_zone_weekly_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load the Raw 21 weekly ingredients Raw 24 uses for detailed local slicing.

    The page still decides how to aggregate a reader-selected geography and
    daypart. This loader owns storage, QA, typing, and canonical zone IDs.
    """
    _require_files(
        [
            BRAID_TEMPORAL_EXPLORER_PATH,
            BRAID_TEMPORAL_EXPLORER_QA_PATH,
            PRELAUNCH_TEMPORAL_EXPLORER_PATH,
            PRELAUNCH_TEMPORAL_EXPLORER_QA_PATH,
        ],
        "Raw 24 weekly local-consistency contract",
    )

    post = pd.read_parquet(
        BRAID_TEMPORAL_EXPLORER_PATH
    )
    post_qa = pd.read_parquet(
        BRAID_TEMPORAL_EXPLORER_QA_PATH
    )
    pre = pd.read_parquet(
        PRELAUNCH_TEMPORAL_EXPLORER_PATH
    )
    pre_qa = pd.read_parquet(
        PRELAUNCH_TEMPORAL_EXPLORER_QA_PATH
    )

    _require_columns(
        post,
        RAW21_TEMPORAL_REQUIRED_COLUMNS,
        "counterfactual_braid_temporal_explorer",
    )
    _require_columns(
        pre,
        RAW21_PRELAUNCH_TEMPORAL_REQUIRED_COLUMNS,
        "counterfactual_prelaunch_temporal_explorer",
    )

    _validate_qa_table(
        post_qa,
        label="counterfactual_braid_temporal_explorer_qa",
    )
    _validate_qa_table(
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
            .replace(
                COUNTERFACTUAL_CANONICAL_ZONE_ALIASES
            )
            .astype("Int64")
        )

    post = post.loc[
        ~post["canonical_taxi_zone_id"].isin(
            COUNTERFACTUAL_UNKNOWN_ZONE_IDS
        )
    ].copy()

    pre = pre.loc[
        ~pre["canonical_taxi_zone_id"].isin(
            COUNTERFACTUAL_UNKNOWN_ZONE_IDS
        )
    ].copy()

    return post, pre


@st.cache_data(show_spinner=False)
def load_counterfactual_raw24_zone_scout() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
] | None:
    """
    Load Raw 24's optional compact Taxi-Zone scout package.

    Missing scout artifacts return None because the candidate-card layer is an
    optional convenience. Present artifacts are validated strictly.
    """
    paths = [
        RAW24_ZONE_DIAGNOSTICS_PATH,
        RAW24_ZONE_WEEKLY_PATH,
        RAW24_ZONE_SCOUT_QA_PATH,
    ]

    if not all(path.exists() for path in paths):
        return None

    diagnostics = pd.read_parquet(
        RAW24_ZONE_DIAGNOSTICS_PATH
    )
    weekly = pd.read_parquet(
        RAW24_ZONE_WEEKLY_PATH
    )
    scout_qa = pd.read_parquet(
        RAW24_ZONE_SCOUT_QA_PATH
    )

    _require_columns(
        diagnostics,
        RAW24_ZONE_DIAGNOSTIC_REQUIRED_COLUMNS,
        "counterfactual_raw24_zone_diagnostics",
    )
    _require_columns(
        weekly,
        RAW24_ZONE_WEEKLY_REQUIRED_COLUMNS,
        "counterfactual_raw24_zone_weekly",
    )
    _validate_qa_table(
        scout_qa,
        label="counterfactual_raw24_zone_scout_qa",
    )

    diagnostics = diagnostics.copy()
    weekly = weekly.copy()

    diagnostics["taxi_zone_id"] = pd.to_numeric(
        diagnostics["taxi_zone_id"],
        errors="raise",
    ).astype("Int64")
    weekly["taxi_zone_id"] = pd.to_numeric(
        weekly["taxi_zone_id"],
        errors="raise",
    ).astype("Int64")
    weekly["week_start"] = pd.to_datetime(
        weekly["week_start"],
        errors="raise",
    )

    return diagnostics, weekly, scout_qa

def _raw25_artifact_fingerprint() -> tuple[tuple[str, int, int], ...]:
    """Return a cache key that changes whenever a Raw 25 serving artifact changes."""
    paths = [
        RAW25_JOB_HISTOGRAM_PATH,
        RAW25_JOB_SUMMARY_PATH,
        RAW25_QA_PATH,
        RAW25_SLICE_SCOUT_PATH,
    ]
    _require_files(
        paths,
        "Raw 25 calibration contract",
    )
    return tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in paths
    )


@st.cache_data(show_spinner=False)
def _load_counterfactual_raw25_inputs_cached(
    artifact_fingerprint: tuple[tuple[str, int, int], ...],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load and validate the compact serving artifacts used by Raw 25.

    WHY:
    Raw 25 owns reader-facing calibration logic and visualization. This shared
    layer owns physical files, schemas, QA, and basic type normalization.
    """
    paths = [
        RAW25_JOB_HISTOGRAM_PATH,
        RAW25_JOB_SUMMARY_PATH,
        RAW25_QA_PATH,
        RAW25_SLICE_SCOUT_PATH,
    ]

    _require_files(
        paths,
        "Raw 25 calibration contract",
    )

    histogram = pd.read_parquet(
        RAW25_JOB_HISTOGRAM_PATH
    )
    summary = pd.read_parquet(
        RAW25_JOB_SUMMARY_PATH
    )
    qa = pd.read_parquet(
        RAW25_QA_PATH
    )
    slice_scout = pd.read_parquet(
        RAW25_SLICE_SCOUT_PATH
    )

    _require_columns(
        histogram,
        RAW25_JOB_HISTOGRAM_REQUIRED_COLUMNS,
        "counterfactual_raw25_job_histogram",
    )
    _require_columns(
        summary,
        RAW25_JOB_SUMMARY_REQUIRED_COLUMNS,
        "counterfactual_raw25_job_summary",
    )
    _require_columns(
        slice_scout,
        RAW25_SLICE_SCOUT_REQUIRED_COLUMNS,
        "counterfactual_raw25_slice_scout",
    )
    _require_columns(
        qa,
        RAW25_QA_REQUIRED_COLUMNS,
        "counterfactual_raw25_qa",
    )

    _validate_qa_table(
        qa,
        label="counterfactual_raw25_qa",
    )

    histogram = histogram.copy()
    summary = summary.copy()
    slice_scout = slice_scout.copy()

    histogram["horizon"] = pd.to_numeric(
        histogram["horizon"],
        errors="raise",
    ).astype(int)

    summary["horizon"] = pd.to_numeric(
        summary["horizon"],
        errors="raise",
    ).astype(int)

    slice_scout["horizon"] = pd.to_numeric(
        slice_scout["horizon"],
        errors="raise",
    ).astype(int)

    slice_scout["rows"] = pd.to_numeric(
        slice_scout["rows"],
        errors="raise",
    ).astype(int)

    slice_scout["support_ok"] = (
        slice_scout["support_ok"]
        .fillna(False)
        .astype(bool)
    )

    return (
        histogram,
        summary,
        slice_scout,
    )

def load_counterfactual_raw25_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load Raw 25 inputs and invalidate cached frames when source files change."""
    return _load_counterfactual_raw25_inputs_cached(
        _raw25_artifact_fingerprint()
    )
