"""Build the compact runtime surface used by Raw 17.

This script preserves Raw 17's existing analytical joins, including the
raw Taxi Zone -> canonical location bridge, but performs them once at build
time rather than every time the Streamlit page starts.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from pathlib import Path
import sys


# ---------------------------------------------------------------------
# Make the Showcase package importable when this script is executed
# directly from scripts/.
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data_access.anomalies import (
    load_stress_anomaly_events_runtime,
)
from app.data_access.loaders import CONGESTION_PRICING_START_DATE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "data" / "processed" / "1.3.1.final_tables"
APP_TABLE_DIR = PROJECT_ROOT / "data" / "processed" / "app_tables"

HANDOFF_PATH = PROJECT_ROOT / "data/processed/3.4.2.final_tables/weather_stress_analytical_handoff.parquet"
MOBILITY_PANEL_PATH = SOURCE_DIR / "analysis_ready_mobility_panel.parquet"
WEATHER_PANEL_PATH = SOURCE_DIR / "weather_taxi_zone_context.parquet"
OUTPUT_PATH = APP_TABLE_DIR / "weather_stress_runtime.parquet"
QA_PATH = APP_TABLE_DIR / "weather_stress_runtime_qa.parquet"
CP_START_DATE = pd.Timestamp(CONGESTION_PRICING_START_DATE)

WEATHER_METRICS = [
    "precipitation", "visibility", "wind_speed", "wind_gust",
    "temperature", "pressure_3hr_change",
]
STARTED = time.perf_counter()


def progress(message: str) -> None:
    print(f"[{time.perf_counter() - STARTED:7.1f}s] {message}", flush=True)


def size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2) if path.exists() else 0.0


def main() -> None:
    print("=" * 76)
    print("Raw 17 · Build compact weather/stress runtime surface")
    print("=" * 76, flush=True)

    for path in [HANDOFF_PATH, MOBILITY_PANEL_PATH, WEATHER_PANEL_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required build input not found: {path}")

    APP_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    progress("[1/6] Reading 3.4.2 weather/stress handoff...")
    frame = pd.read_parquet(HANDOFF_PATH)
    source_rows = len(frame)
    required = {
        "comparison_event_id", "taxi_zone_id", "date", "temporal_bucket",
        "selected_finalist_flag", "weather_available_flag",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Weather/stress handoff is missing required columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["selected_finalist_flag"] = frame["selected_finalist_flag"].fillna(False).astype(bool)
    frame["all_stress_anomaly_flag"] = frame["selected_finalist_flag"]
    frame = frame.loc[frame["weather_available_flag"].fillna(False).astype(bool)].copy()
    weather_eligible_rows = len(frame)
    progress(f"      {source_rows:,} source rows -> {weather_eligible_rows:,} weather-eligible rows.")

    progress("[2/6] Building authoritative Taxi Zone -> canonical bridge...")
    zone_bridge = pd.read_parquet(
        MOBILITY_PANEL_PATH,
        columns=["taxi_zone_id", "canonical_location_id", "zone", "borough"],
    ).drop_duplicates("taxi_zone_id")
    zone_bridge["taxi_zone_id"] = pd.to_numeric(zone_bridge["taxi_zone_id"], errors="coerce").astype("Int64")
    zone_bridge["canonical_location_id"] = pd.to_numeric(zone_bridge["canonical_location_id"], errors="coerce").astype("Int64")
    frame["taxi_zone_id"] = pd.to_numeric(frame["taxi_zone_id"], errors="coerce").astype("Int64")

    existing_zone = "zone" in frame.columns
    existing_borough = "borough" in frame.columns
    existing_canonical = "canonical_location_id" in frame.columns
    frame = frame.merge(
        zone_bridge, on="taxi_zone_id", how="left", suffixes=("", "_bridge"), validate="many_to_one"
    )
    if existing_zone:
        frame["zone"] = frame["zone"].fillna(frame["zone_bridge"])
        frame = frame.drop(columns="zone_bridge")
    if existing_borough:
        frame["borough"] = frame["borough"].fillna(frame["borough_bridge"])
        frame = frame.drop(columns="borough_bridge")
    if existing_canonical:
        frame["canonical_location_id"] = frame["canonical_location_id"].fillna(frame["canonical_location_id_bridge"])
        frame = frame.drop(columns="canonical_location_id_bridge")
    progress(f"      Bridge rows: {len(zone_bridge):,}; runtime grain preserved at {len(frame):,} rows.")

    progress("[3/6] Joining weather on canonical location × date × temporal bucket...")
    weather = pd.read_parquet(
        WEATHER_PANEL_PATH,
        columns=["taxi_zone_id", "date", "temporal_bucket", *WEATHER_METRICS],
    ).rename(columns={"taxi_zone_id": "canonical_location_id"})
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
    weather["canonical_location_id"] = pd.to_numeric(weather["canonical_location_id"], errors="coerce").astype("Int64")
    before_join = len(frame)
    frame = frame.merge(
        weather,
        on=["canonical_location_id", "date", "temporal_bucket"],
        how="left",
        validate="many_to_one",
    )
    if len(frame) != before_join:
        raise ValueError("Canonical weather join changed the event-context grain.")
    progress(f"      Weather joined; {len(frame):,} rows retained.")

    progress("[4/6] Adding shared anomaly-family and driver details...")

    selected_events = (
        load_stress_anomaly_events_runtime()
        .copy()
    )

    detail_columns = [
        "has_congestion_oriented",
        "has_positive_demand_shock",
        "stress_family",
        "stress_metric_driver_list",
        "signature_all",
        "signature_demand",
        "signature_congestion",
        "signature_all_label",
        "signature_demand_label",
        "signature_congestion_label",
    ]

    missing_details = sorted(
        set(detail_columns).difference(
            selected_events.columns
        )
    )

    if missing_details:
        raise ValueError(
            "Shared stress-anomaly runtime is missing Raw 17 fields: "
            + ", ".join(missing_details)
        )

    event_details = (
        selected_events[
            [
                "comparison_event_id",
                *detail_columns,
            ]
        ]
        .drop_duplicates(
            "comparison_event_id"
        )
    )

    if event_details["comparison_event_id"].duplicated().any():
        raise ValueError(
            "Shared anomaly runtime is not unique by comparison_event_id."
        )

    before_join = len(frame)

    frame = frame.merge(
        event_details,
        on="comparison_event_id",
        how="left",
        suffixes=(
            "",
            "_event",
        ),
        validate="one_to_one",
    )

    if len(frame) != before_join:
        raise ValueError(
            "Anomaly-detail join changed the weather runtime grain."
        )

    # WHY: keep any authoritative handoff fields already present, but fill missing
    # anomaly attribution from the validated shared event runtime.
    for column in detail_columns:
        event_column = f"{column}_event"

        if event_column not in frame.columns:
            continue

        if column in frame.columns:
            frame[column] = (
                frame[column]
                .combine_first(
                    frame[event_column]
                )
            )
        else:
            frame[column] = frame[event_column]

        frame = frame.drop(
            columns=event_column
        )

    progress(
        f"      Added {len(detail_columns)} shared anomaly-detail fields."
    )

    progress("[5/6] Deriving reader-facing runtime fields and QA...")
    frame["calendar_month"] = frame["date"].dt.month
    frame["policy_period"] = np.where(frame["date"].lt(CP_START_DATE), "Pre-CP", "Post-CP")
    frame["weekday_weekend"] = np.where(frame["date"].dt.dayofweek.ge(5), "Weekend", "Weekday")
    frame["zone"] = frame["zone"].fillna("Unknown")
    frame["borough"] = frame["borough"].fillna("Unknown")
    frame["policy_geography_label"] = frame.get(
        "policy_geography_label", pd.Series(index=frame.index, dtype="object")
    ).fillna("Unknown")
    frame["canonical_cluster_name"] = frame.get(
        "canonical_cluster_name", pd.Series(index=frame.index, dtype="object")
    ).fillna("Unassigned")

    selected_mask = (
        frame["selected_finalist_flag"]
        .fillna(False)
        .astype(bool)
    )

    missing_selected_stress_family = int(
        frame.loc[
            selected_mask,
            "stress_family",
        ].isna().sum()
    )

    missing_selected_signature = int(
        frame.loc[
            selected_mask,
            "signature_all",
        ].isna().sum()
    )

    if missing_selected_stress_family:
        raise ValueError(
            f"{missing_selected_stress_family:,} selected weather-context rows "
            "are missing shared stress-family attribution."
        )

    if missing_selected_signature:
        raise ValueError(
            f"{missing_selected_signature:,} selected weather-context rows "
            "are missing shared modality signatures."
        )

    duplicate_events = int(frame["comparison_event_id"].duplicated().sum())
    qa = pd.DataFrame([
        {"check": "source_handoff_rows", "value": source_rows},
        {"check": "weather_eligible_rows", "value": weather_eligible_rows},
        {"check": "runtime_rows", "value": len(frame)},
        {"check": "duplicate_comparison_event_ids", "value": duplicate_events},
        {"check": "minimum_date", "value": str(frame["date"].min().date())},
        {"check": "maximum_date", "value": str(frame["date"].max().date())},
        {"check": "unique_taxi_zones", "value": int(frame["taxi_zone_id"].nunique())},
        {
            "check": "selected_rows_missing_stress_family",
            "value": missing_selected_stress_family,
        },
        {
            "check": "selected_rows_missing_signature_all",
            "value": missing_selected_signature,
        },
    ])

    # WHY: QA values intentionally mix counts and date strings. Store them as
    # strings so Arrow sees one stable Parquet type instead of an object column
    # containing incompatible Python types.
    qa["value"] = qa["value"].map(str)
    if duplicate_events:
        raise ValueError(f"Runtime surface has {duplicate_events:,} duplicate comparison_event_id values.")

    progress("[6/6] Writing compact runtime artifacts...")
    frame.to_parquet(OUTPUT_PATH, index=False, compression="zstd")
    qa.to_parquet(QA_PATH, index=False, compression="zstd")
    progress(f"      Runtime surface: {len(frame):,} rows · {size_mb(OUTPUT_PATH):.2f} MB")
    progress(f"      QA table: {len(qa):,} rows · {size_mb(QA_PATH):.2f} MB")
    print("=" * 76)
    print(f"DONE · total runtime {time.perf_counter() - STARTED:.1f}s")
    print(f"Output: {OUTPUT_PATH}")
    print(f"QA:     {QA_PATH}")
    print("=" * 76)


if __name__ == "__main__":
    main()
