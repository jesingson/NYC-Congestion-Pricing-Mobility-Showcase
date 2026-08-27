"""Data-access helpers for the final 3.3.6 anomaly package.

Canonical Showcase anomaly contract
-----------------------------------
The app consumes three final anomaly assets:

    data/processed/3.3.6.final_tables/
        selected_finalist_full_row_universe_enriched.parquet
        selected_finalist_metric_diagnostics.parquet
        finalist_prepost_zone_frequency.parquet

The complete event universe contains one row per Taxi Zone × date × daypart.
`selected_finalist_flag` identifies the canonical finalist anomaly events.

Metric diagnostics contain ten metric rows for every selected finalist event
and support event-level explanation through observed, expected, residual, and
standardized-residual values.

The zone-frequency table is a compact pre/post summary and contains multiple
candidate surfaces. It is primarily intended for the first-class anomaly
diagnostic page rather than for the existing Raw 07/10/11 overlays.
"""

from __future__ import annotations

from pathlib import Path

import re
import pandas as pd
import streamlit as st

from app.data_access.loaders import APP_ROOT


ANOMALY_DATA_DIR = (
    APP_ROOT
    / "data"
    / "processed"
    / "3.3.6.final_tables"
)

ANOMALY_EVENT_UNIVERSE_PATH = (
    ANOMALY_DATA_DIR
    / "selected_finalist_full_row_universe_enriched.parquet"
)

ANOMALY_METRIC_DIAGNOSTICS_PATH = (
    ANOMALY_DATA_DIR
    / "selected_finalist_metric_diagnostics.parquet"
)

ANOMALY_ZONE_FREQUENCY_PATH = (
    ANOMALY_DATA_DIR
    / "finalist_prepost_zone_frequency.parquet"
)

EVENT_ID_COLUMN = "comparison_event_id"
SELECTED_FINALIST_FLAG = "selected_finalist_flag"


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------
def _require_file(path: Path) -> None:
    """Raise a clear error when an expected final anomaly asset is missing."""
    if not path.exists():
        raise FileNotFoundError(
            "Missing expected anomaly data file:\n"
            f"{path}\n\n"
            "Expected local data layout:\n"
            "data/processed/3.3.6.final_tables/"
        )


def _normalize_event_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize duplicated merge-suffix fields from the 3.3.6 export.

    The enriched 3.3.6 universe contains several fields twice because they
    originated from separate working/finalist layers before the final merge.
    The app exposes one canonical field for each concept and prefers the
    finalist-side `_y` value when available, falling back to `_x`.
    """
    result = frame.copy()

    alias_pairs = {
        "positive_direction_profile": (
            "positive_direction_profile_y",
            "positive_direction_profile_x",
        ),
        "framework_anomaly_count": (
            "framework_anomaly_count_y",
            "framework_anomaly_count_x",
        ),
        "framework_support_pattern": (
            "framework_support_pattern_y",
            "framework_support_pattern_x",
        ),
        "dbscan_event_flag": (
            "dbscan_event_flag_y",
            "dbscan_event_flag_x",
        ),
        "gmm_event_flag": (
            "gmm_event_flag_y",
            "gmm_event_flag_x",
        ),
        "if_event_flag": (
            "if_event_flag_y",
            "if_event_flag_x",
        ),
        "event_metric_driver_list": (
            "event_metric_driver_list_y",
            "event_metric_driver_list_x",
        ),
        "event_modality_driver_list": (
            "event_modality_driver_list_y",
            "event_modality_driver_list_x",
        ),
    }

    for canonical_column, source_columns in alias_pairs.items():
        primary, fallback = source_columns

        if primary in result.columns and fallback in result.columns:
            result[canonical_column] = (
                result[primary]
                .combine_first(result[fallback])
            )
        elif primary in result.columns:
            result[canonical_column] = result[primary]
        elif fallback in result.columns:
            result[canonical_column] = result[fallback]

    # App-facing aliases used elsewhere in the Showcase.
    if "canonical_cluster_label" in result.columns:
        result["mobility_regime_cluster_label"] = (
            pd.to_numeric(
                result["canonical_cluster_label"],
                errors="coerce",
            )
            .astype("Int64")
        )

    if "canonical_cluster_name" in result.columns:
        result["mobility_regime_cluster_name"] = (
            result["canonical_cluster_name"]
        )
        result["mobility_environment"] = (
            result["canonical_cluster_name"]
        )

    if "date" in result.columns:
        result["date"] = pd.to_datetime(
            result["date"],
            errors="coerce",
        )

    return result


# ---------------------------------------------------------------------
# Canonical event universe
# ---------------------------------------------------------------------
@st.cache_data(show_spinner="Loading final anomaly event universe...")
def load_anomaly_event_universe() -> pd.DataFrame:
    """Load the complete 3.3.6 event universe."""
    _require_file(
        ANOMALY_EVENT_UNIVERSE_PATH
    )

    frame = pd.read_parquet(
        ANOMALY_EVENT_UNIVERSE_PATH
    )

    required_columns = {
        EVENT_ID_COLUMN,
        "taxi_zone_id",
        "date",
        "daypart",
        "temporal_bucket",
        "zone",
        "borough",
        SELECTED_FINALIST_FLAG,
    }

    missing = sorted(
        required_columns.difference(
            frame.columns
        )
    )

    if missing:
        raise ValueError(
            "The final anomaly event universe is missing required columns: "
            + ", ".join(missing)
        )

    return _normalize_event_columns(
        frame
    )


@st.cache_data(show_spinner=False)
def load_selected_anomaly_events() -> pd.DataFrame:
    """Return only events retained in the canonical finalist surface."""
    universe = load_anomaly_event_universe()

    selected = universe[
        universe[
            SELECTED_FINALIST_FLAG
        ]
        .fillna(False)
        .astype(bool)
    ].copy()

    return (
        selected.sort_values(
            [
                "date",
                "taxi_zone_id",
                "daypart_order",
            ]
        )
        .reset_index(drop=True)
    )


def get_zone_anomaly_events(
    taxi_zone_id: int | float | str,
    *,
    temporal_bucket: str | None = None,
) -> pd.DataFrame:
    """Return selected finalist events for one Taxi Zone."""
    events = load_selected_anomaly_events()

    result = events[
        events["taxi_zone_id"]
        .astype(str)
        .eq(str(taxi_zone_id))
    ].copy()

    if (
        temporal_bucket
        and temporal_bucket
        != "All temporal buckets"
    ):
        result = result[
            result["temporal_bucket"]
            .astype(str)
            .eq(str(temporal_bucket))
        ].copy()

    return (
        result.sort_values(
            [
                "date",
                "daypart_order",
            ]
        )
        .reset_index(drop=True)
    )

def _metric_is_event_driver(
    driver_list: object,
    metric: str,
) -> bool:
    """Return True when one metric appears in an event's driver list.

    The 3.3.6 handoff stores event_metric_driver_list as a string. Use an
    exact token match rather than substring matching so similarly named
    metrics cannot accidentally match one another.
    """
    if pd.isna(driver_list):
        return False

    driver_text = str(driver_list).strip()

    if not driver_text:
        return False

    pattern = (
        rf"(?<![A-Za-z0-9_])"
        rf"{re.escape(metric)}"
        rf"(?![A-Za-z0-9_])"
    )

    return re.search(
        pattern,
        driver_text,
    ) is not None

def get_metric_driver_anomaly_events(
    *,
    metric: str,
    temporal_bucket: str | None = None,
) -> pd.DataFrame:
    """Return selected anomaly events where the chosen metric was a driver."""
    events = load_selected_anomaly_events()

    if events.empty:
        return events

    if "event_metric_driver_list" not in events.columns:
        raise KeyError(
            "Normalized anomaly events are missing "
            "'event_metric_driver_list'."
        )

    driver_mask = events[
        "event_metric_driver_list"
    ].map(
        lambda value: _metric_is_event_driver(
            value,
            metric,
        )
    )

    result = events[
        driver_mask
    ].copy()

    if (
        temporal_bucket
        and temporal_bucket != "All temporal buckets"
    ):
        result = result[
            result["temporal_bucket"]
            .astype(str)
            .eq(str(temporal_bucket))
        ].copy()

    return (
        result.sort_values(
            [
                "date",
                "taxi_zone_id",
                "daypart_order",
            ]
        )
        .reset_index(drop=True)
    )

def get_zone_metric_driver_anomaly_events(
    taxi_zone_id: int | float | str,
    *,
    metric: str,
    temporal_bucket: str | None = None,
) -> pd.DataFrame:
    """Return selected metric-driver anomalies for one Taxi Zone."""
    events = get_metric_driver_anomaly_events(
        metric=metric,
        temporal_bucket=temporal_bucket,
    )

    return (
        events[
            events["taxi_zone_id"]
            .astype(str)
            .eq(str(taxi_zone_id))
        ]
        .copy()
        .sort_values(
            [
                "date",
                "daypart_order",
            ]
        )
        .reset_index(drop=True)
    )

# ---------------------------------------------------------------------
# Metric-level anomaly diagnostics
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_selected_metric_diagnostics() -> pd.DataFrame:
    """Load all metric diagnostics for selected finalist events."""
    _require_file(
        ANOMALY_METRIC_DIAGNOSTICS_PATH
    )

    frame = pd.read_parquet(
        ANOMALY_METRIC_DIAGNOSTICS_PATH
    )

    required_columns = {
        EVENT_ID_COLUMN,
        "taxi_zone_id",
        "date",
        "daypart",
        "temporal_bucket",
        "metric",
        "support_status",
        "observed_value",
        "expected_value",
        "residual_value",
        "residual_zscore",
    }

    missing = sorted(
        required_columns.difference(
            frame.columns
        )
    )

    if missing:
        raise ValueError(
            "The finalist metric-diagnostic table is missing required columns: "
            + ", ".join(missing)
        )

    frame = frame.copy()
    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    return (
        frame.sort_values(
            [
                "date",
                "taxi_zone_id",
                "daypart_order",
                "metric",
            ]
        )
        .reset_index(drop=True)
    )


def get_event_metric_diagnostics(
    comparison_event_id: str,
) -> pd.DataFrame:
    """Return all metric diagnostics for one selected finalist event."""
    diagnostics = load_selected_metric_diagnostics()

    return (
        diagnostics[
            diagnostics[
                EVENT_ID_COLUMN
            ].eq(str(comparison_event_id))
        ]
        .copy()
        .sort_values("metric")
        .reset_index(drop=True)
    )


def get_zone_metric_diagnostics(
    taxi_zone_id: int | float | str,
    *,
    metric: str | None = None,
    temporal_bucket: str | None = None,
) -> pd.DataFrame:
    """Return selected-event metric diagnostics for one Taxi Zone."""
    diagnostics = load_selected_metric_diagnostics()

    result = diagnostics[
        diagnostics["taxi_zone_id"]
        .astype(str)
        .eq(str(taxi_zone_id))
    ].copy()

    if metric is not None:
        result = result[
            result["metric"].eq(metric)
        ].copy()

    if (
        temporal_bucket
        and temporal_bucket
        != "All temporal buckets"
    ):
        result = result[
            result["temporal_bucket"]
            .astype(str)
            .eq(str(temporal_bucket))
        ].copy()

    return (
        result.sort_values(
            [
                "date",
                "daypart_order",
                "metric",
            ]
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Compact zone-frequency handoff
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_finalist_zone_frequency() -> pd.DataFrame:
    """Load the compact candidate-surface pre/post zone-frequency table."""
    _require_file(
        ANOMALY_ZONE_FREQUENCY_PATH
    )

    frame = pd.read_parquet(
        ANOMALY_ZONE_FREQUENCY_PATH
    )

    required_columns = {
        "candidate_surface_label",
        "taxi_zone_id",
        "zone",
        "borough",
        "post_cp_event_share_of_eligible_rows",
        "pre_cp_event_share_of_eligible_rows",
        "post_minus_pre_event_share_delta",
    }

    missing = sorted(
        required_columns.difference(
            frame.columns
        )
    )

    if missing:
        raise ValueError(
            "The finalist zone-frequency table is missing required columns: "
            + ", ".join(missing)
        )

    return frame


# ---------------------------------------------------------------------
# Cross-file contract QA
# ---------------------------------------------------------------------
def validate_anomaly_data_contract() -> dict[str, object]:
    """Validate that the three 3.3.6 Showcase assets agree.

    This is intended for migration QA and diagnostics rather than every page
    render. It deliberately fails loudly when finalist IDs or metric coverage
    disagree across the two primary anomaly tables.
    """
    selected_events = load_selected_anomaly_events()
    diagnostics = load_selected_metric_diagnostics()
    frequency = load_finalist_zone_frequency()

    duplicate_event_ids = int(
        selected_events.duplicated(
            EVENT_ID_COLUMN
        ).sum()
    )

    duplicate_metric_grain = int(
        diagnostics.duplicated(
            [
                EVENT_ID_COLUMN,
                "metric",
            ]
        ).sum()
    )

    selected_ids = set(
        selected_events[
            EVENT_ID_COLUMN
        ]
        .dropna()
        .astype(str)
    )

    diagnostic_ids = set(
        diagnostics[
            EVENT_ID_COLUMN
        ]
        .dropna()
        .astype(str)
    )

    missing_in_diagnostics = (
        selected_ids
        - diagnostic_ids
    )

    extra_in_diagnostics = (
        diagnostic_ids
        - selected_ids
    )

    metric_counts = (
        diagnostics.groupby(
            EVENT_ID_COLUMN,
            observed=True,
            dropna=False,
        )["metric"]
        .nunique()
    )

    source_metric_count = int(
        diagnostics["metric"]
        .nunique()
    )

    if duplicate_event_ids:
        raise ValueError(
            "Selected finalist events contain duplicate comparison_event_id "
            f"values: {duplicate_event_ids:,}"
        )

    if duplicate_metric_grain:
        raise ValueError(
            "Metric diagnostics contain duplicate event × metric rows: "
            f"{duplicate_metric_grain:,}"
        )

    if missing_in_diagnostics:
        raise ValueError(
            "Selected finalist events are missing metric diagnostics: "
            f"{len(missing_in_diagnostics):,} event IDs"
        )

    if extra_in_diagnostics:
        raise ValueError(
            "Metric diagnostics contain events not marked selected_finalist: "
            f"{len(extra_in_diagnostics):,} event IDs"
        )

    if (
        not metric_counts.empty
        and (
            int(metric_counts.min())
            != source_metric_count
            or int(metric_counts.max())
            != source_metric_count
        )
    ):
        raise ValueError(
            "Selected finalist events do not all have complete metric coverage. "
            f"Expected {source_metric_count} metrics per event; observed "
            f"{int(metric_counts.min())} to {int(metric_counts.max())}."
        )

    return {
        "selected_event_rows": int(
            len(selected_events)
        ),
        "selected_event_ids": int(
            len(selected_ids)
        ),
        "metric_diagnostic_rows": int(
            len(diagnostics)
        ),
        "diagnostic_event_ids": int(
            len(diagnostic_ids)
        ),
        "distinct_metrics": source_metric_count,
        "min_metrics_per_event": (
            int(metric_counts.min())
            if not metric_counts.empty
            else 0
        ),
        "median_metrics_per_event": (
            float(metric_counts.median())
            if not metric_counts.empty
            else 0.0
        ),
        "max_metrics_per_event": (
            int(metric_counts.max())
            if not metric_counts.empty
            else 0
        ),
        "duplicate_event_ids": duplicate_event_ids,
        "duplicate_event_metric_rows": duplicate_metric_grain,
        "zone_frequency_rows": int(
            len(frequency)
        ),
        "zone_frequency_surfaces": sorted(
            frequency[
                "candidate_surface_label"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "is_valid": True,
    }