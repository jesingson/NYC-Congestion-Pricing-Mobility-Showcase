"""Raw 17 — Weather and Stress Episodes with clarified temporal-bucket analysis."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from app.data_access.anomalies import load_selected_anomaly_events
from app.data_access.loaders import CONGESTION_PRICING_START_DATE
from app.data_access.weather_relationships import MOBILITY_PANEL_PATH, WEATHER_PANEL_PATH
from app.utils.project_branding import (
    BRAND_COLORS,
    apply_branding,
    exploration_section,
    inject_app_css,
    render_chart_insight,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HANDOFF_PATH = (
    PROJECT_ROOT
    / "data/processed/3.4.2.final_tables/weather_stress_analytical_handoff.parquet"
)
MIN_ZONE_EXPOSED_CONTEXTS = 25
MIN_ZONE_EXPOSED_DATES = 10
MIN_DAILY_CONDITION_CONTEXTS = 25
HERO_LABEL = "Unusually cold"
CP_START_DATE = pd.Timestamp(CONGESTION_PRICING_START_DATE)
STRESS_TYPE_OPTIONS = [
    "All stress anomalies",
    "Congestion-related",
    "Demand-related",
    "Combined congestion + demand",
]
MODALITY_OPTIONS = ["All modes", "Taxi", "FHVHV", "Subway", "Bus"]
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
MODALITIES_BY_STRESS_TYPE = {
    "All stress anomalies": MODALITY_OPTIONS,
    "Congestion-related": ["All modes", "Taxi", "FHVHV", "Bus"],
    "Demand-related": ["All modes", "Taxi", "FHVHV", "Subway", "Bus"],
    "Combined congestion + demand": MODALITY_OPTIONS,
}
STRESS_TYPES_BY_MODALITY = {
    "All modes": STRESS_TYPE_OPTIONS,
    "Taxi": STRESS_TYPE_OPTIONS,
    "FHVHV": STRESS_TYPE_OPTIONS,
    "Subway": [
        "All stress anomalies",
        "Demand-related",
        "Combined congestion + demand",
    ],
    "Bus": STRESS_TYPE_OPTIONS,
}
TIME_SCOPE_OPTIONS = ["Full period", "Pre-CP", "Post-CP", "Custom dates"]

CONDITIONS = {
    "Measurable precipitation": {
        "flag": "measurable_precipitation_flag",
        "metric": "precipitation",
        "definition": "Any available observation with precipitation greater than zero.",
    },
    "Heavier precipitation": {
        "flag": "heavy_precipitation_flag",
        "metric": "precipitation",
        "definition": "The highest 10% of positive precipitation observations.",
    },
    "Low visibility": {
        "flag": "low_visibility_flag",
        "metric": "visibility",
        "definition": "The lowest 10% of available visibility observations.",
    },
    "High sustained wind": {
        "flag": "high_sustained_wind_flag",
        "metric": "wind_speed",
        "definition": "The highest 10% of available sustained-wind observations.",
    },
    "Strong wind gust": {
        "flag": "strong_wind_gust_flag",
        "metric": "wind_gust",
        "definition": "The highest 10% of available wind-gust observations.",
    },
    "Unusually cold": {
        "flag": "unusually_cold_flag",
        "metric": "temperature",
        "definition": "The lowest 5% of available temperature observations.",
    },
    "Unusually hot": {
        "flag": "unusually_hot_flag",
        "metric": "temperature",
        "definition": "The highest 5% of available temperature observations.",
    },
    "Large pressure change": {
        "flag": "large_pressure_change_flag",
        "metric": "pressure_3hr_change",
        "definition": "The highest 10% of absolute three-hour pressure changes.",
    },
}
WEATHER_METRIC_LABELS = {
    "precipitation": "precipitation",
    "visibility": "visibility",
    "wind_speed": "sustained-wind",
    "wind_gust": "wind-gust",
    "temperature": "temperature",
    "pressure_3hr_change": "pressure-change",
}


@st.cache_data(show_spinner="Loading weather and stress-anomaly observations...")
def load_weather_stress_surface() -> pd.DataFrame:
    """Enrich the 3.4.2 handoff with weather eligibility and readable geography."""
    # Read the complete compact handoff so approved cluster, policy-geography,
    # and stress-family fields remain available to the investigation controls.
    frame = pd.read_parquet(HANDOFF_PATH)
    required_columns = {
        "comparison_event_id", "taxi_zone_id", "date", "temporal_bucket",
        "selected_finalist_flag", "weather_available_flag",
    }
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            "The weather/stress handoff is missing required columns: "
            f"{sorted(missing_columns)}"
        )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["selected_finalist_flag"] = (
        frame["selected_finalist_flag"].fillna(False).astype(bool)
    )
    # Preserve an immutable copy of the complete stress-anomaly outcome. Every
    # interactive stress-family view is derived from this field so changing a
    # filter can never overwrite the source outcome used by a later rerun.
    frame["all_stress_anomaly_flag"] = frame["selected_finalist_flag"]
    frame = frame.loc[frame["weather_available_flag"].fillna(False).astype(bool)].copy()

    # The 1.3.1 mobility panel preserves the authoritative raw-to-canonical
    # Taxi Zone bridge as well as reader-facing zone and borough names.
    zone_bridge = pd.read_parquet(
        MOBILITY_PANEL_PATH,
        columns=["taxi_zone_id", "canonical_location_id", "zone", "borough"],
    ).drop_duplicates("taxi_zone_id")
    zone_bridge["taxi_zone_id"] = pd.to_numeric(
        zone_bridge["taxi_zone_id"], errors="coerce"
    ).astype("Int64")
    zone_bridge["canonical_location_id"] = pd.to_numeric(
        zone_bridge["canonical_location_id"], errors="coerce"
    ).astype("Int64")

    frame["taxi_zone_id"] = pd.to_numeric(
        frame["taxi_zone_id"], errors="coerce"
    ).astype("Int64")
    existing_zone = "zone" in frame.columns
    existing_borough = "borough" in frame.columns
    existing_canonical_id = "canonical_location_id" in frame.columns
    frame = frame.merge(
        zone_bridge,
        on="taxi_zone_id",
        how="left",
        suffixes=("", "_bridge"),
        validate="many_to_one",
    )
    if existing_zone:
        frame["zone"] = frame["zone"].fillna(frame["zone_bridge"])
        frame = frame.drop(columns="zone_bridge")
    if existing_borough:
        frame["borough"] = frame["borough"].fillna(frame["borough_bridge"])
        frame = frame.drop(columns="borough_bridge")
    if existing_canonical_id:
        frame["canonical_location_id"] = frame["canonical_location_id"].fillna(
            frame["canonical_location_id_bridge"]
        )
        frame = frame.drop(columns="canonical_location_id_bridge")

    # Actual weather measures distinguish a condition that is absent from a
    # condition that cannot be evaluated because its source measure is missing.
    weather_metrics = sorted({spec["metric"] for spec in CONDITIONS.values()})
    weather = pd.read_parquet(
        WEATHER_PANEL_PATH,
        columns=["taxi_zone_id", "date", "temporal_bucket", *weather_metrics],
    ).rename(columns={"taxi_zone_id": "canonical_location_id"})
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
    weather["canonical_location_id"] = pd.to_numeric(
        weather["canonical_location_id"], errors="coerce"
    ).astype("Int64")

    row_count = len(frame)
    frame = frame.merge(
        weather,
        on=["canonical_location_id", "date", "temporal_bucket"],
        how="left",
        validate="many_to_one",
    )
    if len(frame) != row_count:
        raise ValueError("The canonical weather join changed the event-context grain.")

    # Pull the established Showcase anomaly-family and modality contracts from
    # the canonical selected-event export. This avoids reconstructing them from
    # weather-handoff fields that may be absent or encoded differently.
    selected_events = load_selected_anomaly_events()
    anomaly_detail_columns = [
        column
        for column in [
            "has_congestion_oriented",
            "has_positive_demand_shock",
            "stress_family_exclusive",
            "event_modality_driver_list",
            "event_metric_driver_list",
        ]
        if column in selected_events.columns
    ]
    if anomaly_detail_columns:
        event_details = selected_events[
            ["comparison_event_id", *anomaly_detail_columns]
        ].drop_duplicates("comparison_event_id")
        frame = frame.merge(
            event_details,
            on="comparison_event_id",
            how="left",
            suffixes=("", "_event"),
            validate="one_to_one",
        )
        for column in anomaly_detail_columns:
            event_column = f"{column}_event"
            if event_column in frame.columns:
                frame[column] = frame[event_column].combine_first(frame[column])
                frame = frame.drop(columns=event_column)

    frame["calendar_month"] = frame["date"].dt.month
    frame["policy_period"] = np.where(
        frame["date"].lt(CP_START_DATE),
        "Pre-CP",
        "Post-CP",
    )
    frame["weekday_weekend"] = np.where(
        frame["date"].dt.dayofweek.ge(5), "Weekend", "Weekday"
    )
    frame["zone"] = frame["zone"].fillna("Unknown")
    frame["borough"] = frame["borough"].fillna("Unknown")
    frame["policy_geography_label"] = frame.get(
        "policy_geography_label",
        pd.Series(index=frame.index, dtype="object"),
    ).fillna("Unknown")
    frame["canonical_cluster_name"] = frame.get(
        "canonical_cluster_name",
        pd.Series(index=frame.index, dtype="object"),
    ).fillna("Unassigned")
    return frame


def stress_type_flag(frame: pd.DataFrame, stress_type: str) -> pd.Series:
    """Identify the requested stress-anomaly family without removing denominator rows."""
    selected = frame["all_stress_anomaly_flag"].fillna(False).astype(bool)
    if stress_type == "All stress anomalies":
        return selected

    # Use the same authoritative event-family flags as the existing stress-
    # anomaly Showcase pages. Older handoffs fall back to the exclusive label,
    # followed only as a last resort by the attribution counts.
    has_family_flags = {
        "has_congestion_oriented",
        "has_positive_demand_shock",
    }.issubset(frame.columns)
    if has_family_flags:
        congestion_driver = frame["has_congestion_oriented"].fillna(False).astype(bool)
        demand_driver = frame["has_positive_demand_shock"].fillna(False).astype(bool)
    elif "stress_family_exclusive" in frame.columns:
        exclusive_family = frame["stress_family_exclusive"].fillna("").astype(str)
        congestion_driver = exclusive_family.isin(["Congestion-only", "Both"])
        demand_driver = exclusive_family.isin(["Demand-only", "Both"])
    elif {
        "stress_congestion_metric_count",
        "stress_positive_demand_metric_count",
    }.issubset(frame.columns):
        congestion_driver = pd.to_numeric(
            frame["stress_congestion_metric_count"], errors="coerce"
        ).fillna(0).gt(0)
        demand_driver = pd.to_numeric(
            frame["stress_positive_demand_metric_count"], errors="coerce"
        ).fillna(0).gt(0)
    else:
        family_column = next(
            (
                column
                for column in ["stress_family", "positive_direction_profile"]
                if column in frame.columns
            ),
            None,
        )
        if family_column is None:
            return pd.Series(False, index=frame.index)
        stress_family = frame[family_column].fillna("").astype(str).str.lower()
        congestion_driver = stress_family.str.contains("congestion")
        demand_driver = stress_family.str.contains("demand")

    if stress_type == "Congestion-related":
        family_match = congestion_driver
    elif stress_type == "Demand-related":
        family_match = demand_driver
    else:
        family_match = congestion_driver & demand_driver
    return selected & family_match


def _recognized_metric_drivers(value: object) -> tuple[str, ...]:
    """Extract recognized mobility metrics from list-like or serialized driver text."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return tuple()

    if isinstance(value, (list, tuple, set)):
        text = " ".join(str(item) for item in value)
    else:
        text = str(value)

    return tuple(
        metric
        for metric in METRIC_TO_MODE
        if re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(metric)}(?![A-Za-z0-9_])",
            text,
        )
    )


def modality_flag(
    frame: pd.DataFrame,
    modality: str,
    stress_type: str,
) -> pd.Series:
    """Identify selected-family stress evidence involving one mobility mode."""
    selected = frame["all_stress_anomaly_flag"].fillna(False).astype(bool)
    if modality == "All modes":
        return selected

    # Prefer metric-level driver evidence so "Demand + Bus" means Bus demand
    # evidence, while "Congestion + Bus" means Bus speed evidence.
    metric_column = next(
        (
            column
            for column in [
                "event_metric_driver_list",
                "stress_metric_driver_list",
            ]
            if column in frame.columns
        ),
        None,
    )

    if metric_column is not None:
        metric_sets = frame[metric_column].map(_recognized_metric_drivers)

        if stress_type == "Demand-related":
            relevant_metrics = DEMAND_METRICS
        elif stress_type == "Congestion-related":
            relevant_metrics = CONGESTION_METRICS
        else:
            # All-stress and compound views preserve every recognized driver.
            relevant_metrics = set(METRIC_TO_MODE)

        mode_match = metric_sets.map(
            lambda metrics: any(
                metric in relevant_metrics
                and METRIC_TO_MODE[metric] == modality
                for metric in metrics
            )
        )
        return selected & mode_match

    # Backward-compatible fallback for handoffs that only retain mode lists.
    driver_column = next(
        (
            column
            for column in [
                "event_modality_driver_list",
                "stress_modality_driver_list",
            ]
            if column in frame.columns
        ),
        None,
    )
    if driver_column is None:
        return pd.Series(False, index=frame.index)

    normalized_drivers = (
        frame[driver_column]
        .fillna("")
        .astype(str)
        .str.split(",")
        .apply(lambda values: {value.strip() for value in values if value.strip()})
    )
    return selected & normalized_drivers.apply(lambda values: modality in values)


def apply_analysis_scope(
    frame: pd.DataFrame,
    geography_dimension: str,
    geography_segment: str,
    stress_type: str,
    modality: str,
    temporal_bucket: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Apply one geography segmentation and preserve the full denominator."""
    scoped = frame.copy()
    geography_column = {
        "Borough": "borough",
        "Policy geography": "policy_geography_label",
        "Mobility environment": "canonical_cluster_name",
    }.get(geography_dimension)
    if geography_column is not None and geography_segment != "All segments":
        scoped = scoped.loc[scoped[geography_column].eq(geography_segment)]

    if temporal_bucket != "All temporal buckets":
        scoped = scoped.loc[scoped["temporal_bucket"].eq(temporal_bucket)]

    scoped = scoped.loc[
        scoped["date"].between(start_date, end_date, inclusive="both")
    ].copy()

    family_outcome = stress_type_flag(scoped, stress_type)
    mode_outcome = modality_flag(scoped, modality, stress_type)
    scoped["selected_finalist_flag"] = family_outcome & mode_outcome
    return scoped


def format_temporal_bucket(value: str) -> str:
    """Convert the stored temporal-bucket key into a reader-facing label."""
    if value == "All temporal buckets":
        return value
    replacements = {"am": "AM", "pm": "PM"}
    return " ".join(
        replacements.get(token, token.title())
        for token in str(value).split("_")
    )


def analysis_scope_label(
    geography_dimension: str,
    geography_segment: str,
    stress_type: str,
    modality: str,
    temporal_bucket: str,
    time_scope: str,
) -> str:
    """Build a plain-language label for the active investigation scope."""
    selected_values = []
    if geography_dimension != "All NYC" and geography_segment != "All segments":
        selected_values.append(geography_segment)
    if stress_type != "All stress anomalies":
        selected_values.append(stress_type)
    if modality != "All modes":
        selected_values.append(modality)
    if temporal_bucket != "All temporal buckets":
        selected_values.append(format_temporal_bucket(temporal_bucket))
    if time_scope != "Full period":
        selected_values.append(time_scope)
    return " · ".join(selected_values) if selected_values else "All NYC mobility observations"


def condition_comparison(
    frame: pd.DataFrame,
    condition_label: str,
    policy_geography_fixed: bool = False,
) -> dict[str, float]:
    """Return overall and like-for-like anomaly incidence for one condition."""
    spec = CONDITIONS[condition_label]
    eligible = frame[spec["metric"]].notna()
    condition_frame = frame.loc[eligible].copy()
    condition_frame["condition_present"] = (
        condition_frame[spec["flag"]].fillna(False).astype(bool)
    )

    present = condition_frame["condition_present"]
    present_contexts = int(present.sum())
    present_dates = int(condition_frame.loc[present, "date"].nunique())
    if present_contexts == 0 or not (~present).any():
        return {
            "eligible_contexts": len(condition_frame),
            "present_contexts": present_contexts,
            "present_share": np.nan,
            "present_dates": present_dates,
            "overall_present": np.nan,
            "overall_absent": np.nan,
            "overall_difference": np.nan,
            "adjusted_present": np.nan,
            "adjusted_absent": np.nan,
            "adjusted_difference": np.nan,
            "contributing_strata": 0,
        }

    present_rate = 100 * condition_frame.loc[
        present, "selected_finalist_flag"
    ].mean()
    absent_rate = 100 * condition_frame.loc[
        ~present, "selected_finalist_flag"
    ].mean()

    strata = [
        "temporal_bucket",
        "calendar_month",
        "policy_period",
    ]
    if not policy_geography_fixed:
        strata.append("policy_geography_label")
    grouped = (
        condition_frame.groupby(strata + ["condition_present"], dropna=False)
        .agg(
            contexts=("comparison_event_id", "size"),
            stress_anomalies=("selected_finalist_flag", "sum"),
        )
        .reset_index()
    )
    grouped["incidence"] = 100 * grouped["stress_anomalies"] / grouped["contexts"]
    paired = grouped.loc[grouped["condition_present"]].merge(
        grouped.loc[~grouped["condition_present"]],
        on=strata,
        how="inner",
        suffixes=("_present", "_absent"),
    )
    if paired.empty:
        adjusted_present = np.nan
        adjusted_absent = np.nan
    else:
        paired["weight"] = paired["contexts_present"] + paired["contexts_absent"]
        adjusted_present = np.average(
            paired["incidence_present"], weights=paired["weight"]
        )
        adjusted_absent = np.average(
            paired["incidence_absent"], weights=paired["weight"]
        )

    return {
        "eligible_contexts": len(condition_frame),
        "present_contexts": present_contexts,
        "present_share": 100 * present.mean(),
        "present_dates": present_dates,
        "overall_present": present_rate,
        "overall_absent": absent_rate,
        "overall_difference": present_rate - absent_rate,
        "adjusted_present": adjusted_present,
        "adjusted_absent": adjusted_absent,
        "adjusted_difference": adjusted_present - adjusted_absent,
        "contributing_strata": len(paired),
    }


def daily_condition_summary(frame: pd.DataFrame, condition_label: str) -> pd.DataFrame:
    """Build a date series using only contexts with an available source measure."""
    spec = CONDITIONS[condition_label]
    daily_frame = frame.loc[frame[spec["metric"]].notna()].copy()
    daily_frame["condition_present"] = daily_frame[spec["flag"]].fillna(False).astype(bool)
    daily_frame["present_anomaly"] = (
        daily_frame["condition_present"] & daily_frame["selected_finalist_flag"]
    )
    daily = (
        daily_frame.groupby("date")
        .agg(
            eligible_contexts=("comparison_event_id", "size"),
            present_contexts=("condition_present", "sum"),
            present_anomalies=("present_anomaly", "sum"),
        )
        .reset_index()
    )
    daily["Condition coverage %"] = (
        100 * daily["present_contexts"] / daily["eligible_contexts"]
    )
    daily["Anomaly incidence when present %"] = np.where(
        daily["present_contexts"].gt(0),
        100 * daily["present_anomalies"] / daily["present_contexts"],
        np.nan,
    )
    return daily


def zone_condition_summary(
    frame: pd.DataFrame,
    condition_label: str,
    episode_date: object | None = None,
) -> pd.DataFrame:
    """Rank zones for the full study or a selected weather-active date."""
    spec = CONDITIONS[condition_label]
    zone_frame = frame.loc[frame[spec["metric"]].notna()].copy()
    if episode_date is not None:
        zone_frame = zone_frame.loc[
            zone_frame["date"].dt.date.eq(episode_date)
        ].copy()
    zone_frame["condition_present"] = zone_frame[spec["flag"]].fillna(False).astype(bool)
    zone_frame["present_anomaly"] = (
        zone_frame["condition_present"] & zone_frame["selected_finalist_flag"]
    )
    zone_frame["absent_anomaly"] = (
        ~zone_frame["condition_present"] & zone_frame["selected_finalist_flag"]
    )
    zone_frame["present_date"] = zone_frame["date"].where(
        zone_frame["condition_present"]
    )
    zones = (
        zone_frame.groupby(["taxi_zone_id", "zone", "borough"], dropna=False)
        .agg(
            present_contexts=("condition_present", "sum"),
            present_anomalies=("present_anomaly", "sum"),
            absent_contexts=("condition_present", lambda values: (~values).sum()),
            absent_anomalies=("absent_anomaly", "sum"),
            present_dates=("present_date", "nunique"),
        )
        .reset_index()
    )
    zones["Condition-present anomaly incidence %"] = (
        100 * zones["present_anomalies"] / zones["present_contexts"]
    )
    zones["Condition-absent anomaly incidence %"] = (
        100 * zones["absent_anomalies"] / zones["absent_contexts"]
    )
    zones["Difference pp"] = (
        zones["Condition-present anomaly incidence %"]
        - zones["Condition-absent anomaly incidence %"]
    )
    zones = zones.loc[
        zones["present_contexts"].gt(0)
        & zones["absent_contexts"].gt(0)
    ].copy()
    if episode_date is None:
        zones = zones.loc[
            zones["present_contexts"].ge(MIN_ZONE_EXPOSED_CONTEXTS)
            & zones["present_dates"].ge(MIN_ZONE_EXPOSED_DATES)
        ].copy()
    return zones


def normalize_modality_drivers(value: object) -> list[str]:
    """Normalize comma-separated or list-like modality-driver fields."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    values = value if isinstance(value, (list, tuple, set)) else str(value).split(",")
    return [str(item).strip() for item in values if str(item).strip()]


@st.cache_data(show_spinner=False)
def modality_composition(condition_label: str) -> pd.DataFrame:
    """Compare identified modality drivers when a condition is present or absent."""
    frame = load_weather_stress_surface()
    spec = CONDITIONS[condition_label]
    condition_events = frame.loc[
        frame[spec["metric"]].notna() & frame["selected_finalist_flag"],
        ["comparison_event_id", spec["flag"]],
    ].copy()
    condition_events["Condition group"] = np.where(
        condition_events[spec["flag"]].fillna(False), "Present", "Absent"
    )

    selected_events = load_selected_anomaly_events()
    driver_column = next(
        (
            column for column in [
                "event_modality_driver_list", "stress_modality_driver_list"
            ]
            if column in selected_events.columns
        ),
        None,
    )
    if driver_column is None:
        return pd.DataFrame()

    drivers = selected_events[["comparison_event_id", driver_column]].merge(
        condition_events[["comparison_event_id", "Condition group"]],
        on="comparison_event_id",
        how="inner",
        validate="one_to_one",
    )
    drivers["Modality"] = drivers[driver_column].apply(normalize_modality_drivers)
    drivers = drivers.explode("Modality")
    drivers = drivers.loc[drivers["Modality"].isin(["Taxi", "FHVHV", "Subway", "Bus"])]
    summary = (
        drivers.groupby(["Condition group", "Modality"])
        .size()
        .reset_index(name="Driver records")
    )
    summary["Composition share %"] = (
        100
        * summary["Driver records"]
        / summary.groupby("Condition group")["Driver records"].transform("sum")
    )
    return summary


def scoped_driver_composition(
    frame: pd.DataFrame,
    condition_label: str,
    stress_type: str,
    episode_date: object | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize only drivers compatible with the selected anomaly family."""
    spec = CONDITIONS[condition_label]
    condition_events = frame.loc[
        frame[spec["metric"]].notna() & frame["selected_finalist_flag"],
        ["comparison_event_id", "date", spec["flag"]],
    ].copy()
    if episode_date is not None:
        condition_events = condition_events.loc[
            condition_events["date"].dt.date.eq(episode_date)
        ].copy()
    condition_events["Condition group"] = np.where(
        condition_events[spec["flag"]].fillna(False),
        "Present",
        "Absent",
    )

    selected_events = load_selected_anomaly_events()
    metric_column = next(
        (
            column
            for column in [
                "event_metric_driver_list",
                "stress_metric_driver_list",
            ]
            if column in selected_events.columns
        ),
        None,
    )
    if condition_events.empty or metric_column is None:
        return pd.DataFrame(), pd.DataFrame()

    drivers = selected_events[["comparison_event_id", metric_column]].merge(
        condition_events[["comparison_event_id", "Condition group"]],
        on="comparison_event_id",
        how="inner",
        validate="one_to_one",
    )

    metric_drivers = drivers[["Condition group", metric_column]].copy()
    metric_drivers["Metric driver"] = metric_drivers[metric_column].apply(
        normalize_modality_drivers
    )
    metric_drivers = metric_drivers.explode("Metric driver").dropna(
        subset=["Metric driver"]
    )
    metric_drivers = metric_drivers.loc[
        metric_drivers["Metric driver"].isin(METRIC_TO_MODE)
    ].copy()

    if stress_type == "Congestion-related":
        metric_drivers = metric_drivers.loc[
            metric_drivers["Metric driver"].isin(CONGESTION_METRICS)
        ].copy()
    elif stress_type == "Demand-related":
        metric_drivers = metric_drivers.loc[
            metric_drivers["Metric driver"].isin(DEMAND_METRICS)
        ].copy()

    if metric_drivers.empty:
        return pd.DataFrame(), pd.DataFrame()

    metric_summary = (
        metric_drivers.groupby(["Condition group", "Metric driver"])
        .size()
        .reset_index(name="Driver records")
    )
    metric_summary["Composition share %"] = (
        100
        * metric_summary["Driver records"]
        / metric_summary.groupby("Condition group")["Driver records"].transform("sum")
    )

    metric_drivers["Modality"] = metric_drivers["Metric driver"].map(METRIC_TO_MODE)
    modality_summary = (
        metric_drivers.groupby(["Condition group", "Modality"])
        .size()
        .reset_index(name="Driver records")
    )
    modality_summary["Composition share %"] = (
        100
        * modality_summary["Driver records"]
        / modality_summary.groupby("Condition group")["Driver records"].transform("sum")
    )
    return modality_summary, metric_summary


def comparison_chart(summary: dict[str, float], title: str) -> go.Figure:
    """Show incidence in comparable contexts with visible values."""
    labels = ["Weather condition present", "Similar observations without it"]
    values = [summary["adjusted_present"], summary["adjusted_absent"]]
    figure = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=[BRAND_COLORS["dark_teal"], BRAND_COLORS["seafoam"]],
        text=[f"{value:.1f}%" for value in values],
        textposition="outside",
        cliponaxis=False,
        customdata=[[summary["contributing_strata"]]] * 2,
        hovertemplate=(
            "<b>%{x}</b><br>Stress-anomaly incidence: %{y:.1f}%<br>"
            "Comparison groups: %{customdata[0]:,.0f}<extra></extra>"
        ),
    ))
    figure.update_layout(
        title=title, xaxis_title="", yaxis_title="Stress-anomaly incidence (%)",
        showlegend=False, height=390, margin=dict(t=70, r=30, b=50, l=60),
    )
    figure.update_yaxes(ticksuffix="%", rangemode="tozero")
    return apply_branding(figure)


def stability_message(summary: dict[str, float]) -> str:
    """Explain what changes after comparing similar contexts."""
    overall = summary["overall_difference"]
    adjusted = summary["adjusted_difference"]
    if np.sign(overall) != np.sign(adjusted):
        return (
            "The direction reverses after comparing like-for-like observations, so "
            "the unadjusted pattern is not dependable."
        )
    if abs(adjusted) < 0.5 * abs(overall):
        return (
            "The gap becomes much smaller after the like-for-like comparison, so "
            "timing and geography explain much of the initial pattern."
        )
    return (
        "The gap remains similar after the like-for-like comparison, so it is not "
        "explained by when or where the condition occurred."
    )


inject_app_css()

for required_path in [HANDOFF_PATH, MOBILITY_PANEL_PATH, WEATHER_PANEL_PATH]:
    if not required_path.exists():
        st.error(f"Required weather/stress input not found: {required_path}")
        st.stop()

weather_stress_df = load_weather_stress_surface()

# ---------------------------------------------------------------------
# Fixed answer view
# ---------------------------------------------------------------------

st.caption("WEATHER RELATIONSHIPS")
st.title("Do recurring weather conditions coincide with more stress anomalies?")
st.write(
    "Weather and mobility stress can occur at the same time without weather necessarily "
    "being the reason for the anomaly. This page compares stress-anomaly incidence when "
    "a recurring weather condition is present with otherwise similar observations where "
    "it is absent, helping separate simple co-occurrence from differences that persist "
    "after broad timing and geography are held more comparable."
)

hero = condition_comparison(weather_stress_df, HERO_LABEL)

st.header("Does the cold-weather difference persist in like-for-like contexts?")
st.write(
    "The fixed opening comparison uses **unusually cold** observations and pairs them "
    "with observations from the same calendar month, policy period, temporal bucket, "
    "and policy-geography group. The bars show the percentage of eligible observations "
    "that were retained as stress anomalies in each group."
)
st.caption(CONDITIONS[HERO_LABEL]["definition"])

st.plotly_chart(
    comparison_chart(hero, "Stress anomalies in similar observations with and without cold weather"),
    use_container_width=True,
)
hero_direction = "more" if hero["adjusted_difference"] >= 0 else "less"
render_chart_insight(
    f"Stress anomalies were **{abs(hero['adjusted_difference']):.1f} percentage "
    f"points {hero_direction} common** during unusually cold weather "
    f"(**{hero['adjusted_present']:.1f}%**) than in otherwise similar observations "
    f"without it (**{hero['adjusted_absent']:.1f}%**)."
)

hero_cards = st.columns(4)
hero_cards[0].metric("During cold weather", f"{hero['adjusted_present']:.1f}%")
hero_cards[1].metric("Without cold weather", f"{hero['adjusted_absent']:.1f}%")
hero_cards[2].metric("Difference", f"{hero['adjusted_difference']:+.1f} pp")
hero_cards[3].metric("Cold-weather dates", f"{hero['present_dates']:,}")

# ---------------------------------------------------------------------
# Condition comparison
# ---------------------------------------------------------------------

with exploration_section(
    key="raw17_exploration_area",
    title="Investigate a weather pattern",
    description=(
        "Choose a weather condition, stress family, mobility mode, time-of-week "
        "bucket, geography, and study period. Every result below uses the same "
        "active scope."
    ),
):
    filter_row_one = st.columns(4)
    selected_label = filter_row_one[0].selectbox(
        "Weather pattern to compare", list(CONDITIONS),
        index=list(CONDITIONS).index("Heavier precipitation"),
    )

    # Both controls read the other control's current session-state value before
    # rendering. This prevents invalid combinations regardless of which selector
    # the user changes first.
    st.session_state.setdefault("raw17_stress_type", "All stress anomalies")
    st.session_state.setdefault("raw17_modality", "All modes")
    current_modality = st.session_state["raw17_modality"]
    allowed_stress_types = STRESS_TYPES_BY_MODALITY.get(
        current_modality, STRESS_TYPE_OPTIONS
    )
    if st.session_state["raw17_stress_type"] not in allowed_stress_types:
        st.session_state["raw17_stress_type"] = "All stress anomalies"
    selected_stress_type = filter_row_one[1].selectbox(
        "Stress-anomaly type",
        allowed_stress_types,
        key="raw17_stress_type",
    )
    allowed_modalities = MODALITIES_BY_STRESS_TYPE[selected_stress_type]
    if st.session_state["raw17_modality"] not in allowed_modalities:
        st.session_state["raw17_modality"] = "All modes"
    selected_modality = filter_row_one[2].selectbox(
        "Mobility mode",
        allowed_modalities,
        key="raw17_modality",
    )
    temporal_bucket_options = [
        "All temporal buckets",
        *sorted(weather_stress_df["temporal_bucket"].dropna().astype(str).unique()),
    ]
    selected_temporal_bucket = filter_row_one[3].selectbox(
        "Temporal bucket",
        temporal_bucket_options,
        format_func=format_temporal_bucket,
    )
    st.caption(
        "The selected weather pattern defines the during-condition group. Observations "
        "without that pattern remain in the comparison group."
    )

    filter_row_two = st.columns(3)
    selected_geography_dimension = filter_row_two[0].selectbox(
        "Geography lens",
        ["All NYC", "Borough", "Policy geography", "Mobility environment"],
    )

    geography_column = {
        "Borough": "borough",
        "Policy geography": "policy_geography_label",
        "Mobility environment": "canonical_cluster_name",
    }.get(selected_geography_dimension)
    if geography_column is None:
        selected_geography_segment = "All segments"
    else:
        excluded_values = {"Unknown", "Unassigned"}
        geography_segments = sorted(
            value
            for value in weather_stress_df[geography_column].dropna().unique()
            if value not in excluded_values
        )
        selected_geography_segment = filter_row_two[1].selectbox(
            f"{selected_geography_dimension} segment",
            ["All segments", *geography_segments],
        )
    if geography_column is None:
        filter_row_two[1].selectbox(
            "Geography segment", ["All NYC"], disabled=True
        )

    selected_time_scope = filter_row_two[2].selectbox(
        "Time period", TIME_SCOPE_OPTIONS
    )
    study_start = weather_stress_df["date"].min().normalize()
    study_end = weather_stress_df["date"].max().normalize()
    if selected_time_scope == "Full period":
        selected_start_date = study_start
        selected_end_date = study_end
    elif selected_time_scope == "Pre-CP":
        selected_start_date = study_start
        selected_end_date = min(study_end, CP_START_DATE - pd.Timedelta(days=1))
    elif selected_time_scope == "Post-CP":
        selected_start_date = max(study_start, CP_START_DATE)
        selected_end_date = study_end
    else:
        custom_date_columns = st.columns(2)
        selected_start_date = pd.Timestamp(custom_date_columns[0].date_input(
            "Start date", value=study_start.date(),
            min_value=study_start.date(), max_value=study_end.date(),
        ))
        selected_end_date = pd.Timestamp(custom_date_columns[1].date_input(
            "End date", value=study_end.date(),
            min_value=study_start.date(), max_value=study_end.date(),
        ))
        if selected_start_date > selected_end_date:
            st.error("Start date must be on or before end date.")
            st.stop()

    scoped_weather_stress_df = apply_analysis_scope(
        weather_stress_df,
        selected_geography_dimension,
        selected_geography_segment,
        selected_stress_type,
        selected_modality,
        selected_temporal_bucket,
        selected_start_date,
        selected_end_date,
    )
    active_scope = analysis_scope_label(
        selected_geography_dimension,
        selected_geography_segment,
        selected_stress_type,
        selected_modality,
        selected_temporal_bucket,
        selected_time_scope,
    )
    selected = condition_comparison(
        scoped_weather_stress_df,
        selected_label,
        policy_geography_fixed=(
            selected_geography_dimension == "Policy geography"
            and selected_geography_segment != "All segments"
        ),
    )
    st.caption(
        f"**Active scope:** {active_scope}. "
        f"**Condition definition:** {CONDITIONS[selected_label]['definition']}"
    )

    filtered_anomaly_count = int(
        scoped_weather_stress_df["selected_finalist_flag"].sum()
    )
    st.caption(
        f"This scope contains **{filtered_anomaly_count:,} stress anomalies** among "
        f"**{selected['eligible_contexts']:,} evaluated zone–temporal-bucket observations**. The weather "
        f"condition was present in **{selected['present_contexts']:,} observations** across "
        f"**{selected['present_dates']:,} dates**."
    )

    if not np.isfinite(selected["adjusted_difference"]):
        st.warning(
            "This scope does not contain enough comparable zone–temporal-bucket observations with and without "
            "the selected condition. Broaden a filter or choose another condition."
        )
        st.stop()

    selected_cards = st.columns(3)
    selected_cards[0].metric("During condition", f"{selected['adjusted_present']:.1f}%")
    selected_cards[1].metric("Without condition", f"{selected['adjusted_absent']:.1f}%")
    selected_cards[2].metric("Difference", f"{selected['adjusted_difference']:+.1f} pp")

    st.plotly_chart(
        comparison_chart(selected, f"{selected_label}: comparison within the active scope"),
        use_container_width=True,
    )
    selected_direction = "more" if selected["adjusted_difference"] >= 0 else "less"
    render_chart_insight(
        f"Within **{active_scope}**, {selected_stress_type.lower()} were "
        f"**{abs(selected['adjusted_difference']):.1f} percentage points "
        f"{selected_direction} common** during **{selected_label.lower()}**. "
        f"{stability_message(selected)}"
    )

    geography_is_filtered = (
        selected_geography_dimension != "All NYC"
        and selected_geography_segment != "All segments"
    )
    if geography_is_filtered:
        citywide_reference_df = apply_analysis_scope(
            weather_stress_df,
            "All NYC",
            "All segments",
            selected_stress_type,
            selected_modality,
            selected_temporal_bucket,
            selected_start_date,
            selected_end_date,
        )
        citywide_reference = condition_comparison(
            citywide_reference_df,
            selected_label,
        )
        scope_comparison_figure = go.Figure(go.Bar(
            x=[active_scope, "All NYC"],
            y=[
                selected["adjusted_difference"],
                citywide_reference["adjusted_difference"],
            ],
            marker_color=[
                BRAND_COLORS["dark_teal"],
                BRAND_COLORS["seafoam"],
            ],
            text=[
                f"{selected['adjusted_difference']:+.1f} pp",
                f"{citywide_reference['adjusted_difference']:+.1f} pp",
            ],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{x}</b><br>Weather-associated difference: %{y:+.1f} pp"
                "<extra></extra>"
            ),
        ))
        scope_comparison_figure.update_layout(
            title="How does the focused result compare with the citywide pattern?",
            xaxis_title="",
            yaxis_title="Weather-associated difference (percentage points)",
            showlegend=False,
            height=390,
            margin=dict(t=75, r=30, b=75, l=70),
        )
        scope_comparison_figure.add_hline(
            y=0,
            line_dash="dash",
            line_color="#66747A",
        )
        st.plotly_chart(
            apply_branding(scope_comparison_figure),
            use_container_width=True,
        )
        scope_gap = (
            selected["adjusted_difference"]
            - citywide_reference["adjusted_difference"]
        )
        render_chart_insight(
            f"The weather-associated difference in **{selected_geography_segment}** "
            f"was **{selected['adjusted_difference']:+.1f} pp**, compared with "
            f"**{citywide_reference['adjusted_difference']:+.1f} pp** citywide—a "
            f"**{scope_gap:+.1f} pp** gap."
        )

    temporal_tab, spatial_tab, modes_tab = st.tabs([
        "When did it happen?",
        "Where did rates differ?",
        "Which modes were involved?",
    ])

    # ---------------------------------------------------------------------
    # Temporal concentration
    # ---------------------------------------------------------------------

    temporal_tab.subheader("When and how widely did the condition occur?")
    temporal_tab.write(
        "For each date, the upper panel measures the share of evaluated Taxi Zone × "
        "temporal-bucket observations that met the selected weather definition. Its denominator "
        "includes every observation with the underlying weather measure available. On any one "
        "date, only the five weekday or five weekend buckets applicable to that date are present. "
        "The lower panel "
        "measures how many of those observations contained a stress anomaly "
        f"matching the active filters and shows only dates with at least "
        f"{MIN_DAILY_CONDITION_CONTEXTS} weather-present observations. Dot size reflects "
        "the number of observations behind the daily anomaly rate."
    )

    selected_daily = daily_condition_summary(scoped_weather_stress_df, selected_label)
    available_dates = selected_daily.loc[selected_daily["present_contexts"].gt(0)].copy()
    available_dates = available_dates.sort_values("date").reset_index(drop=True)
    supported_incidence_dates = available_dates.loc[
        available_dates["present_contexts"].ge(MIN_DAILY_CONDITION_CONTEXTS)
    ].copy()
    median_supported_daily_incidence = (
        supported_incidence_dates["Anomaly incidence when present %"].median()
        if not supported_incidence_dates.empty
        else np.nan
    )

    timeline = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
        subplot_titles=(
            "Share of zone–temporal-bucket observations with the selected weather pattern",
            "Stress-anomaly rate when the weather condition was present",
        ),
    )
    weather_metric_label = WEATHER_METRIC_LABELS.get(
        CONDITIONS[selected_label]["metric"],
        CONDITIONS[selected_label]["metric"].replace("_", " "),
    )
    timeline.add_trace(go.Bar(
        x=available_dates["date"], y=available_dates["Condition coverage %"],
        marker_color=BRAND_COLORS["dark_teal"],
        name="Share meeting weather definition",
        customdata=available_dates[["present_contexts", "eligible_contexts"]],
        hovertemplate=(
            f"<b>%{{x|%b %d, %Y}}</b><br>{selected_label}: "
            "%{customdata[0]:,.0f} zone–temporal-bucket observations"
            f"<br>All observations with {weather_metric_label} data: "
            "%{customdata[1]:,.0f}"
            "<br>Share with selected weather pattern: %{y:.1f}%"
            "<extra></extra>"
        ),
    ), row=1, col=1)
    timeline.add_trace(go.Scatter(
        x=supported_incidence_dates["date"],
        y=supported_incidence_dates["Anomaly incidence when present %"],
        mode="markers",
        marker={
            "size": np.clip(
                5 + np.sqrt(supported_incidence_dates["present_contexts"]), 7, 18
            ),
            "color": BRAND_COLORS["terracotta"],
            "opacity": 0.75,
        },
        name="Anomaly incidence",
        customdata=supported_incidence_dates[["present_anomalies", "present_contexts"]],
        hovertemplate=(
            "<b>%{x|%b %d, %Y}</b><br>Stress anomalies: %{customdata[0]:,.0f}"
            "<br>Observations with selected weather pattern: %{customdata[1]:,.0f}"
            "<br>Stress-anomaly rate: %{y:.1f}%<extra></extra>"
        ),
    ), row=2, col=1)
    if np.isfinite(median_supported_daily_incidence):
        timeline.add_hline(
            y=median_supported_daily_incidence,
            line_dash="dot",
            line_color=BRAND_COLORS["terracotta"],
            annotation_text=(
                f"Median anomaly rate on dates with this weather: "
                f"{median_supported_daily_incidence:.1f}%"
            ),
            row=2,
            col=1,
        )
    timeline.update_layout(
        title=f"{selected_label} across the active time period", height=600,
        hovermode="closest", showlegend=False,
        margin=dict(t=90, r=30, b=50, l=65),
    )
    timeline.update_yaxes(ticksuffix="%", rangemode="tozero", row=1, col=1)
    timeline.update_yaxes(ticksuffix="%", rangemode="tozero", row=2, col=1)
    temporal_tab.plotly_chart(apply_branding(timeline), use_container_width=True)
    if not available_dates.empty:
        most_widespread_date = available_dates.sort_values(
            ["Condition coverage %", "present_contexts"], ascending=False
        ).iloc[0]
        temporal_insight = (
            f"The condition covered the largest share of the active mobility surface on "
            f"**{most_widespread_date['date']:%b %d, %Y}** "
            f"(**{most_widespread_date['Condition coverage %']:.1f}%** of evaluated observations)."
        )
        if not supported_incidence_dates.empty:
            highest_incidence_date = supported_incidence_dates.sort_values(
                ["Anomaly incidence when present %", "present_contexts"],
                ascending=False,
            ).iloc[0]
            temporal_insight += (
                f" Across dates with at least {MIN_DAILY_CONDITION_CONTEXTS} observations "
                f"meeting the weather definition, the median anomaly rate was "
                f"**{median_supported_daily_incidence:.1f}%**. The highest rate was on "
                f"**{highest_incidence_date['date']:%b %d, %Y}** "
                f"(**{int(highest_incidence_date['present_anomalies']):,} of "
                f"{int(highest_incidence_date['present_contexts']):,} observations; "
                f"{highest_incidence_date['Anomaly incidence when present %']:.1f}%**)."
            )
        with temporal_tab:
            render_chart_insight(temporal_insight)

    # ---------------------------------------------------------------------
    # Spatial concentration
    # ---------------------------------------------------------------------

    spatial_tab.subheader("How did stress-anomaly rates vary by Taxi Zone?")
    spatial_tab.write(
        "For each Taxi Zone, compare its stress-anomaly rate when the selected weather "
        "condition was present with its own rate when the condition was absent. A zone is "
        f"shown only when it has at least {MIN_ZONE_EXPOSED_CONTEXTS} zone–temporal-bucket observations "
        f"meeting the weather definition across at least {MIN_ZONE_EXPOSED_DATES} distinct dates. This minimum "
        "evidence threshold prevents a few observations from dominating the ranking; it "
        "does not represent statistical significance."
    )

    zone_summary_df = zone_condition_summary(
        scoped_weather_stress_df,
        selected_label,
    )
    zone_ranking = zone_summary_df.sort_values(
        "Difference pp", ascending=False
    ).head(10)

    if zone_ranking.empty:
        spatial_tab.info(
            "No Taxi Zones meet the minimum evidence thresholds in this scope. Broaden the "
            "geography or time period, or choose another condition."
        )
    else:
        present_hover_data = np.column_stack([
            zone_ranking["borough"].fillna("Unknown").astype(str),
            zone_ranking["present_anomalies"].map(lambda value: f"{value:,.0f}"),
            zone_ranking["present_contexts"].map(lambda value: f"{value:,.0f}"),
            zone_ranking["present_dates"].map(lambda value: f"{value:,.0f}"),
            zone_ranking["Difference pp"].map(lambda value: f"{value:+.3f} pp"),
        ])
        absent_hover_data = np.column_stack([
            zone_ranking["borough"].fillna("Unknown").astype(str),
            zone_ranking["absent_anomalies"].map(lambda value: f"{value:,.0f}"),
            zone_ranking["absent_contexts"].map(lambda value: f"{value:,.0f}"),
            zone_ranking["Difference pp"].map(lambda value: f"{value:+.3f} pp"),
        ])
        connector_x = []
        connector_y = []
        for _, zone_row in zone_ranking.iterrows():
            connector_x.extend([
                zone_row["Condition-absent anomaly incidence %"],
                zone_row["Condition-present anomaly incidence %"],
                None,
            ])
            connector_y.extend([zone_row["zone"], zone_row["zone"], None])

        zone_figure = go.Figure()
        zone_figure.add_trace(go.Scatter(
            x=connector_x,
            y=connector_y,
            mode="lines",
            line={"color": "#AAB8BC", "width": 2},
            hoverinfo="skip",
            showlegend=False,
        ))
        zone_figure.add_trace(go.Scatter(
            x=zone_ranking["Condition-absent anomaly incidence %"],
            y=zone_ranking["zone"],
            mode="markers",
            name="Without condition",
            marker={"size": 11, "color": BRAND_COLORS["seafoam"]},
            customdata=absent_hover_data,
            hovertemplate=(
                "<b>%{y}</b><br>Borough: %{customdata[0]}<br>"
                "Without condition: %{x:.3f}%<br>Stress anomalies: %{customdata[1]}"
                "<br>Observations without condition: %{customdata[2]}<br>Difference: %{customdata[3]}"
                "<extra></extra>"
            ),
        ))
        zone_figure.add_trace(go.Scatter(
            x=zone_ranking["Condition-present anomaly incidence %"],
            y=zone_ranking["zone"],
            mode="markers+text",
            name="During condition",
            marker={"size": 12, "color": BRAND_COLORS["dark_teal"]},
            text=[f"{value:+.1f} pp" for value in zone_ranking["Difference pp"]],
            textposition="middle right",
            customdata=present_hover_data,
            hovertemplate=(
                "<b>%{y}</b><br>Borough: %{customdata[0]}<br>"
                "During condition: %{x:.3f}%<br>Stress anomalies: %{customdata[1]}"
                "<br>Observations meeting weather definition: %{customdata[2]}"
                "<br>Condition-present dates: %{customdata[3]}"
                "<br>Difference: %{customdata[4]}<extra></extra>"
            ),
        ))
        zone_figure.update_layout(
            title={
                "text": f"Taxi Zone anomaly rates within {active_scope}",
                "x": 0.01,
                "y": 0.98,
                "xanchor": "left",
                "yanchor": "top",
            },
            xaxis_title="Stress-anomaly rate (%)",
            yaxis_title="", height=650, margin=dict(t=120, r=115, b=85, l=240),
            legend={
                "orientation": "h", "x": 0.01, "y": 0.90,
                "xanchor": "left", "yanchor": "bottom", "title_text": "",
            },
        )
        zone_figure.update_yaxes(
            autorange="reversed", automargin=True, ticks="outside", ticklen=6
        )
        zone_figure.update_xaxes(
            automargin=True, title_standoff=20, ticksuffix="%", rangemode="tozero"
        )
        spatial_tab.plotly_chart(apply_branding(zone_figure), use_container_width=True)

        leading_zone = zone_ranking.iloc[0]
        positive_zone_count = int(zone_summary_df["Difference pp"].gt(0).sum())
        eligible_zone_count = len(zone_summary_df)
        if positive_zone_count:
            with spatial_tab:
                render_chart_insight(
                    f"Stress-anomaly rates were higher during the condition in "
                    f"**{positive_zone_count} of {eligible_zone_count}** adequately "
                    f"observed Taxi Zones. **{leading_zone['zone']}** had the largest "
                    f"increase at **{leading_zone['Difference pp']:+.1f} percentage "
                    "points**."
                )
        else:
            with spatial_tab:
                render_chart_insight(
                    "Stress-anomaly rates were not higher during the condition in any "
                    f"of the **{eligible_zone_count}** adequately observed Taxi Zones. "
                    f"**{leading_zone['zone']}** was closest to the no-change line at "
                    f"**{leading_zone['Difference pp']:+.1f} percentage points**."
                )

        zone_table = zone_ranking.rename(columns={
            "zone": "Taxi Zone",
            "borough": "Borough",
            "present_contexts": "Observations during condition",
            "present_dates": "Dates with condition",
            "present_anomalies": "Anomalies during condition",
            "absent_contexts": "Observations without condition",
            "absent_anomalies": "Anomalies without condition",
            "Condition-present anomaly incidence %": "Anomaly rate during condition %",
            "Condition-absent anomaly incidence %": "Anomaly rate without condition %",
        })[[
            "Taxi Zone", "Borough", "Observations during condition", "Dates with condition",
            "Anomalies during condition", "Anomaly rate during condition %",
            "Observations without condition", "Anomalies without condition",
            "Anomaly rate without condition %", "Difference pp",
        ]].copy()
        rate_columns = [
            "Anomaly rate during condition %",
            "Anomaly rate without condition %",
            "Difference pp",
        ]
        zone_table[rate_columns] = zone_table[rate_columns].round(3)
        with spatial_tab.expander("See Taxi Zone details", expanded=False):
            st.dataframe(
                zone_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    column: st.column_config.NumberColumn(format="%.3f")
                    for column in rate_columns
                },
            )

    # ---------------------------------------------------------------------
    # Mobility-system composition
    # ---------------------------------------------------------------------

    modes_tab.subheader("Which mobility modes were most often involved?")
    modes_tab.write(
        "Compare the share of anomaly-related mobility measures associated with each "
        "mode during the selected weather condition and when it was absent."
    )

    modality_df, metric_driver_df = scoped_driver_composition(
        scoped_weather_stress_df,
        selected_label,
        selected_stress_type,
    )
    if not metric_driver_df.empty:
        metric_driver_df["Metric driver"] = metric_driver_df["Metric driver"].map(
            lambda metric: METRIC_LABELS.get(metric, str(metric).replace("_", " ").title())
        )
    if modality_df.empty:
        modes_tab.info("Modality-driver detail is unavailable for this active scope.")
    else:
        if selected_stress_type == "Congestion-related":
            modalities = ["Taxi", "FHVHV", "Bus"]
        elif selected_stress_type == "Demand-related":
            modalities = ["Taxi", "FHVHV", "Subway"]
        else:
            modalities = ["Taxi", "FHVHV", "Subway", "Bus"]
        modality_figure = go.Figure()
        for group_label, color in [
            ("Present", BRAND_COLORS["dark_teal"]),
            ("Absent", BRAND_COLORS["seafoam"]),
        ]:
            group = (
                modality_df.loc[modality_df["Condition group"].eq(group_label)]
                .set_index("Modality")
                .reindex(modalities)
                .fillna(0)
                .reset_index()
            )
            modality_figure.add_trace(go.Bar(
                x=group["Modality"], y=group["Composition share %"],
                name=("During condition" if group_label == "Present" else "Without condition"),
                marker_color=color,
                text=[f"{value:.1f}%" for value in group["Composition share %"]],
                textposition="outside", cliponaxis=False,
                customdata=group[["Driver records"]],
                hovertemplate=(
                    "<b>%{x}</b><br>Weather: " + group_label
                    + "<br>Share of identified drivers: %{y:.1f}%"
                    + "<br>Driver records: %{customdata[0]:,.0f}<extra></extra>"
                ),
            ))
        modality_figure.update_layout(
            title={
                "text": f"Modes involved during {selected_label.lower()}",
                "x": 0.01,
                "y": 0.98,
                "xanchor": "left",
                "yanchor": "top",
            },
            xaxis_title="", yaxis_title="Share of identified modality drivers (%)",
            barmode="group", height=480, margin=dict(t=45, r=30, b=55, l=70),
            legend={
                "orientation": "h",
                "x": 0.01,
                "y": 0.86,
                "xanchor": "left",
                "yanchor": "bottom",
                "title_text": "",
            },
        )
        modality_figure.update_yaxes(
            ticksuffix="%", rangemode="tozero", domain=[0, 0.78]
        )
        modes_tab.plotly_chart(apply_branding(modality_figure), use_container_width=True)

        driver_pivot = (
            modality_df.pivot(
                index="Modality",
                columns="Condition group",
                values="Composition share %",
            )
            .reindex(columns=["Present", "Absent"], fill_value=0)
            .fillna(0)
        )
        driver_pivot["Shift pp"] = (
            driver_pivot["Present"] - driver_pivot["Absent"]
        )
        leading_driver = driver_pivot["Present"].idxmax()
        largest_shift_driver = driver_pivot["Shift pp"].abs().idxmax()
        largest_shift = driver_pivot.loc[largest_shift_driver, "Shift pp"]
        shift_direction = "increased" if largest_shift >= 0 else "decreased"
        with modes_tab:
            render_chart_insight(
                f"**{leading_driver}** accounted for the largest share of involved "
                f"mobility measures during {selected_label.lower()} "
                f"(**{driver_pivot.loc[leading_driver, 'Present']:.1f}%**). "
                f"**{largest_shift_driver}** changed the most, with its share "
                f"{shift_direction} by **{abs(largest_shift):.1f} percentage points**."
            )

    metric_detail = modes_tab.expander("See the underlying mobility measures", expanded=False)
    metric_detail.write(
        "Drill into the metric drivers behind the anomalies in the active scope. The "
        "chart shows the five most frequently identified metrics and compares their "
        "share during the condition with their share when the condition was absent."
    )

    if metric_driver_df.empty:
        metric_detail.info("Metric-driver detail is unavailable for this active scope.")
    else:
        leading_metrics = (
            metric_driver_df.groupby("Metric driver")["Driver records"]
            .sum()
            .nlargest(5)
            .index
            .tolist()
        )
        metric_driver_figure = go.Figure()
        for group_label, color in [
            ("Present", BRAND_COLORS["dark_teal"]),
            ("Absent", BRAND_COLORS["seafoam"]),
        ]:
            group = (
                metric_driver_df.loc[
                    metric_driver_df["Condition group"].eq(group_label)
                ]
                .set_index("Metric driver")
                .reindex(leading_metrics)
                .fillna(0)
                .reset_index()
            )
            metric_driver_figure.add_trace(go.Bar(
                x=group["Metric driver"],
                y=group["Composition share %"],
                name=(
                    "During condition"
                    if group_label == "Present"
                    else "Without condition"
                ),
                marker_color=color,
                text=[f"{value:.1f}%" for value in group["Composition share %"]],
                textposition="outside",
                cliponaxis=False,
                customdata=group[["Driver records"]],
                hovertemplate=(
                    "<b>%{x}</b><br>Share of identified metric drivers: %{y:.1f}%<br>"
                    "Driver records: %{customdata[0]:,.0f}<extra></extra>"
                ),
            ))
        metric_driver_figure.update_layout(
            title={
                "text": f"Leading metric drivers during {selected_label.lower()}",
                "x": 0.01,
                "y": 0.98,
                "xanchor": "left",
                "yanchor": "top",
            },
            xaxis_title="",
            yaxis_title="Share of identified metric drivers (%)",
            barmode="group",
            height=550,
            margin=dict(t=45, r=30, b=130, l=70),
            legend={
                "orientation": "h",
                "x": 0.01,
                "y": 0.86,
                "xanchor": "left",
                "yanchor": "bottom",
                "title_text": "",
            },
        )
        metric_driver_figure.update_xaxes(tickangle=-30)
        metric_driver_figure.update_yaxes(
            ticksuffix="%", rangemode="tozero", domain=[0, 0.78]
        )
        metric_detail.plotly_chart(
            apply_branding(metric_driver_figure),
            use_container_width=True,
        )

        metric_pivot = (
            metric_driver_df.pivot(
                index="Metric driver",
                columns="Condition group",
                values="Composition share %",
            )
            .reindex(columns=["Present", "Absent"], fill_value=0)
            .fillna(0)
        )
        metric_pivot["Shift pp"] = (
            metric_pivot["Present"] - metric_pivot["Absent"]
        )
        leading_metric = metric_pivot["Present"].idxmax()
        largest_metric_shift = metric_pivot["Shift pp"].abs().idxmax()
        metric_shift_value = metric_pivot.loc[largest_metric_shift, "Shift pp"]
        metric_shift_direction = "increased" if metric_shift_value >= 0 else "decreased"
        with metric_detail:
            render_chart_insight(
                f"**{leading_metric}** was the most frequently identified metric "
                f"driver during {selected_label.lower()}, representing "
                f"**{metric_pivot.loc[leading_metric, 'Present']:.1f}%** of "
                f"metric-driver records. **{largest_metric_shift}** changed the most, "
                f"and its share {metric_shift_direction} by "
                f"**{abs(metric_shift_value):.1f} percentage points**."
            )


st.markdown("### What this page establishes")
st.markdown(
    "Some weather conditions coincide with different stress-anomaly rates even after "
    "broad calendar, policy-period, time-of-week, and geography context is made more "
    "comparable; other apparent gaps shrink or reverse under that comparison. Weather "
    "is therefore useful context for understanding when mobility stress appears, but "
    "the relationship is conditional rather than a simple citywide weather effect."
)

with st.expander("How this page works", expanded=False):
    st.markdown(
        """
        **1. Use one comparable observation grain.** Each observation is one Taxi Zone
        during one time-of-week bucket on one date.

        **2. Define weather exposure only where the source measure exists.** **Condition
        present** means the underlying weather measure is available and meets the
        displayed definition. **Condition absent** means the measure is available but
        does not meet that definition.

        **3. Compare broadly similar contexts.** The like-for-like calculation pairs
        condition-present and condition-absent observations from the same calendar
        month, policy period, and temporal bucket and—unless policy geography is already
        fixed by a filter—the same policy-geography group.

        **4. Keep the denominator tied to the selected weather condition.** Stress
        incidence is calculated only among observations eligible for that condition,
        preventing missing weather measurements from being treated as condition-absent.

        **5. Use the explorer to test whether the relationship depends on context.**
        Weather condition, stress family, mobility mode, temporal bucket, geography, and
        study period can be changed while preserving the same comparison logic.

        **6. Treat station-based weather as broad spatial context.** Taxi Zone weather
        is assigned from a small station network, so it should not be interpreted as
        block-level measurement.
        """
    )

st.caption(
    "Evidence scope: descriptive co-occurrence between recurring weather conditions and "
    "selected mobility stress anomalies after broad timing and geography adjustment. "
    "The comparison reduces some obvious contextual differences but does not establish "
    "that weather caused an anomaly."
)
