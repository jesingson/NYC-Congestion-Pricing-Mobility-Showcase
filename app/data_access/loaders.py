"""Data loading utilities for the NYC Congestion Pricing Mobility Showcase app.

The app uses flattened local exports from the SIADS 696 pipeline.
Notebook-origin path:
    pipeline_data/1.3.1.final_tables/

App-local path:
    data/processed/1.3.1.final_tables/
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


APP_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = APP_ROOT / "data" / "processed" / "1.3.1.final_tables"

ANALYSIS_PANEL_PATH = DATA_DIR / "analysis_ready_mobility_panel.parquet"
AUDIT_PANEL_PATH = DATA_DIR / "audit_only_mobility_panel.parquet"
BRIDGE_TUNNEL_PANEL_PATH = DATA_DIR / "bridge_tunnel_mobility_panel.parquet"
TAXI_ZONES_PATH = DATA_DIR / "nyc_taxi_zones_harmonized.parquet"
TAXI_ZONE_CONNECTIVITY_PATH = DATA_DIR / "taxi_zone_connectivity.parquet"
WEATHER_CONTEXT_PATH = DATA_DIR / "weather_taxi_zone_context.parquet"

STUDY_START_DATE = pd.Timestamp("2023-01-01")
STUDY_END_DATE = pd.Timestamp("2026-03-31")
CONGESTION_PRICING_START_DATE = pd.Timestamp("2025-01-05")

# Six trusted headline metrics used by the broad exploratory pages.
CORE_METRICS = [
    "taxi_trip_count",
    "taxi_avg_trip_speed",
    "fhvhv_trip_count",
    "fhvhv_avg_trip_speed",
    "subway_ridership",
    "avg_bus_speed",
]

# Full set of clean base measures available to investigative views such as
# the Taxi Zone Profile. Keeping this separate prevents Pages 01–06 from
# expanding automatically when deeper diagnostic metrics are added.
BASE_METRICS = [
    "taxi_trip_count",
    "taxi_avg_trip_speed",
    "taxi_avg_trip_duration",
    "fhvhv_trip_count",
    "fhvhv_avg_trip_speed",
    "fhvhv_avg_trip_duration",
    "subway_ridership",
    "subway_transfers",
    "bus_trip_count",
    "avg_bus_speed",
]

METRIC_LABELS = {
    "taxi_trip_count": "Taxi trips",
    "taxi_avg_trip_speed": "Taxi average speed",
    "taxi_avg_trip_duration": "Taxi average trip duration",
    "fhvhv_trip_count": "FHVHV trips",
    "fhvhv_avg_trip_speed": "FHVHV average speed",
    "fhvhv_avg_trip_duration": "FHVHV average trip duration",
    "subway_ridership": "Subway ridership",
    "subway_transfers": "Subway transfers",
    "bus_trip_count": "Bus trips",
    "avg_bus_speed": "Average bus speed",
}

COUNT_METRICS = [
    "taxi_trip_count",
    "fhvhv_trip_count",
    "subway_ridership",
    "subway_transfers",
    "bus_trip_count",
]

SPEED_METRICS = [
    "taxi_avg_trip_speed",
    "fhvhv_avg_trip_speed",
    "avg_bus_speed",
]

DURATION_METRICS = [
    "taxi_avg_trip_duration",
    "fhvhv_avg_trip_duration",
]

DAYPART_ORDER = [
    "overnight",
    "am_peak",
    "midday",
    "pm_peak",
    "evening",
]

TEMPORAL_BUCKET_ORDER = [
    "weekday_overnight",
    "weekday_am_peak",
    "weekday_midday",
    "weekday_pm_peak",
    "weekday_evening",
    "weekend_overnight",
    "weekend_am_peak",
    "weekend_midday",
    "weekend_pm_peak",
    "weekend_evening",
]

ANALYSIS_BASE_COLUMNS = [
    "taxi_zone_id",
    "zone",
    "borough",
    "canonical_location_id",
    "centroid_x",
    "centroid_y",
    "cbd_spatial_category",
    "is_cbd_zone",
    "is_cbd_gateway_zone",
    "is_cbd_adjacent_zone",
    "distance_to_cbd_miles",
    "distance_to_gateway_miles",
    "neighbor_count",
    "date",
    "year",
    "month",
    "day_of_week",
    "daypart",
    "daypart_order",
    "temporal_bucket",
    "temporal_bucket_order",
    "observation_sequence_id",
    "pre_post_cp",
]

ANALYSIS_DEFAULT_COLUMNS = ANALYSIS_BASE_COLUMNS + CORE_METRICS


def _require_file(path: Path) -> None:
    """Raise a clear error if an expected local data file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing expected data file:\n{path}\n\n"
            "Expected local data layout:\n"
            "data/processed/1.3.1.final_tables/\n\n"
            "Make sure the 1.3.1 final parquet exports were copied into the app repo "
            "with the flattened folder structure."
        )


def _read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a parquet file after validating that it exists."""
    _require_file(path)
    return pd.read_parquet(path, columns=columns)


def _standardize_mobility_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Apply shared app-side type cleanup for the mobility panel."""
    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    if "daypart" in df.columns:
        df["daypart"] = pd.Categorical(
            df["daypart"],
            categories=DAYPART_ORDER,
            ordered=True,
        )

    if "temporal_bucket" in df.columns:
        df["temporal_bucket"] = pd.Categorical(
            df["temporal_bucket"],
            categories=TEMPORAL_BUCKET_ORDER,
            ordered=True,
        )

    return df


@st.cache_data(show_spinner="Loading analysis panel...")
def load_analysis_panel(columns: list[str] | None = None) -> pd.DataFrame:
    """Load the 1.3.1 analysis-ready mobility panel.

    Default behavior loads the app backbone columns plus the six trusted core metrics.
    Pass columns=None only if you intentionally want the full analysis panel.
    """
    read_columns = ANALYSIS_DEFAULT_COLUMNS if columns is None else columns
    df = _read_parquet(ANALYSIS_PANEL_PATH, columns=read_columns)
    return _standardize_mobility_panel(df)


@st.cache_data(show_spinner="Loading audit panel...")
def load_audit_panel(columns: list[str] | None = None) -> pd.DataFrame:
    """Load the wider audit-only panel.

    Use sparingly. Most app pages should use load_analysis_panel().
    """
    df = _read_parquet(AUDIT_PANEL_PATH, columns=columns)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df


@st.cache_data(show_spinner="Loading Taxi Zone geography...")
def load_taxi_zones() -> pd.DataFrame:
    """Load Taxi Zone spatial context carried forward by 1.3.1."""
    return _read_parquet(TAXI_ZONES_PATH)


@st.cache_data(show_spinner="Loading Taxi Zone connectivity...")
def load_taxi_zone_connectivity() -> pd.DataFrame:
    """Load Taxi Zone adjacency / connectivity context."""
    return _read_parquet(TAXI_ZONE_CONNECTIVITY_PATH)


@st.cache_data(show_spinner="Loading Bridge/Tunnel context...")
def load_bridge_tunnel_panel(columns: list[str] | None = None) -> pd.DataFrame:
    """Load Bridge/Tunnel mobility context.

    This is contextual only, not part of the core app metric backbone.
    """
    df = _read_parquet(BRIDGE_TUNNEL_PANEL_PATH, columns=columns)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df


@st.cache_data(show_spinner="Loading weather context...")
def load_weather_context(columns: list[str] | None = None) -> pd.DataFrame:
    """Load weather context.

    This is contextual only, not part of the core app metric backbone.
    """
    df = _read_parquet(WEATHER_CONTEXT_PATH, columns=columns)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df


@st.cache_data(show_spinner=False)
def get_data_inventory() -> pd.DataFrame:
    """Return a compact inventory of expected local app data files."""
    specs = [
        ("analysis_ready_mobility_panel", ANALYSIS_PANEL_PATH, "Primary app backbone"),
        ("audit_only_mobility_panel", AUDIT_PANEL_PATH, "Wider QA/audit panel"),
        ("bridge_tunnel_mobility_panel", BRIDGE_TUNNEL_PANEL_PATH, "Contextual layer"),
        ("nyc_taxi_zones_harmonized", TAXI_ZONES_PATH, "Taxi Zone spatial context"),
        ("taxi_zone_connectivity", TAXI_ZONE_CONNECTIVITY_PATH, "Zone adjacency/connectivity"),
        ("weather_taxi_zone_context", WEATHER_CONTEXT_PATH, "Contextual layer"),
    ]

    rows = []
    for name, path, role in specs:
        rows.append(
            {
                "dataset": name,
                "role": role,
                "path": str(path.relative_to(APP_ROOT)),
                "exists": path.exists(),
                "size_mb": round(path.stat().st_size / 1_000_000, 2) if path.exists() else None,
            }
        )

    return pd.DataFrame(rows)


def get_metric_label(metric: str) -> str:
    """Return a display label for a metric."""
    return METRIC_LABELS.get(metric, metric)