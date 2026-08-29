from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st
import altair as alt

from app.data_access.anomalies import (
    ANOMALY_EVENT_UNIVERSE_PATH,
    ANOMALY_METRIC_DIAGNOSTICS_PATH,
    EVENT_ID_COLUMN,
    SELECTED_FINALIST_FLAG,
    load_metric_history,
    load_selected_anomaly_events,
)
from app.utils.project_branding import inject_app_css


PAGE_CAPTION = "STRESS ANOMALY EVENT PROFILER"
PAGE_TITLE = "Why was this mobility event unusual?"
REPORT_TITLE = "RAW 15 — PHASE 1 STRESS-ANOMALY EVENT PROFILER SCOUTING REPORT"

FROZEN_HERO_EVENT_ID = "34 | 2025-03-02 | overnight"

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

MODE_ORDER = ("Taxi", "FHVHV", "Subway", "Bus")
DIRECTIONAL_SCORE_THRESHOLD = 1.28
STRESS_ALIGNED_DIRECTIONAL_LABELS = {
    "congestion_oriented",
    "positive_demand_shock",
}

# Metric-specific modeled-support thresholds produced by notebook 3.3.1.
# The compact history retains enough information to explain a fallback series
# against this frozen production contract without loading another app table.
MODELED_SUPPORT_THRESHOLDS = {
    "avg_bus_speed": (1186, 1186, 0.134),
    "bus_trip_count": (1186, 1186, 3379.661),
    "fhvhv_avg_trip_duration": (1186, 1186, 6710.367),
    "fhvhv_avg_trip_speed": (1186, 1186, 1.723),
    "fhvhv_trip_count": (1186, 1186, 1837.883),
    "subway_ridership": (1184, 1186, 57996.262),
    "subway_transfers": (1184, 1186, 110.634),
    "taxi_avg_trip_duration": (627, 1186, 304926.411),
    "taxi_avg_trip_speed": (627, 1186, 12.439),
    "taxi_trip_count": (1186, 1186, 2.800),
}


def _tokens(value: object) -> tuple[str, ...]:
    if pd.isna(value):
        return tuple()

    return tuple(
        dict.fromkeys(
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*", str(value))
        )
    )


def _driver_metrics(value: object) -> tuple[str, ...]:
    return tuple(token for token in _tokens(value) if token in METRIC_TO_MODE)


def _driver_modes(value: object) -> tuple[str, ...]:
    modes = {METRIC_TO_MODE[metric] for metric in _driver_metrics(value)}
    return tuple(mode for mode in MODE_ORDER if mode in modes)


def _stress_family(frame: pd.DataFrame) -> pd.Series:
    demand = frame.get(
        "has_positive_demand_shock",
        pd.Series(False, index=frame.index),
    ).fillna(False).astype(bool)
    congestion = frame.get(
        "has_congestion_oriented",
        pd.Series(False, index=frame.index),
    ).fillna(False).astype(bool)

    return pd.Series(
        np.select(
            [demand & congestion, congestion, demand],
            ["Both", "Congestion", "Demand"],
            default="Other stress",
        ),
        index=frame.index,
        dtype="string",
    )


def _coherent_driver_direction(metric: str, zscore: float) -> bool:
    if pd.isna(zscore):
        return False
    if abs(float(zscore)) < DIRECTIONAL_SCORE_THRESHOLD:
        return False
    if metric in DEMAND_METRICS:
        return zscore > 0
    if "speed" in metric:
        return zscore < 0
    if "duration" in metric:
        return zscore > 0
    return False


def _stress_alignment_mask(frame: pd.DataFrame) -> pd.Series:
    """Return the upstream 3.3.5 positive-direction rule at metric grain."""
    metric = frame["metric"].astype(str)
    zscore = pd.to_numeric(frame["residual_zscore"], errors="coerce")
    strong_enough = zscore.abs().ge(DIRECTIONAL_SCORE_THRESHOLD)
    stress_direction = (
        (metric.isin(DEMAND_METRICS) & zscore.gt(0))
        | (metric.str.contains("speed", na=False) & zscore.lt(0))
        | (metric.str.contains("duration", na=False) & zscore.gt(0))
    )
    return strong_enough & stress_direction


def _counter_stress_mask(frame: pd.DataFrame) -> pd.Series:
    metric = frame["metric"].astype(str)
    zscore = pd.to_numeric(frame["residual_zscore"], errors="coerce")
    strong_enough = zscore.abs().ge(DIRECTIONAL_SCORE_THRESHOLD)
    counter_direction = (
        (metric.isin(DEMAND_METRICS) & zscore.lt(0))
        | (metric.str.contains("speed", na=False) & zscore.gt(0))
        | (metric.str.contains("duration", na=False) & zscore.lt(0))
    )
    return strong_enough & counter_direction


def _annotate_evidence(
    evidence: pd.DataFrame,
    driver_metrics: set[str],
) -> pd.DataFrame:
    """Separate authoritative stress drivers from retained metric context."""
    annotated = evidence.copy()
    if "stress_driver_flag" not in annotated.columns:
        raise ValueError(
            "Raw 15 requires stress_driver_flag in the metric-diagnostics export."
        )

    annotated["stress_driver_flag"] = (
        annotated["stress_driver_flag"].fillna(False).astype(bool)
    )
    annotated["is_listed_anomaly_driver"] = annotated["stress_driver_flag"]
    annotated["is_stress_aligned"] = annotated["stress_driver_flag"]
    annotated["is_counter_stress"] = _counter_stress_mask(annotated)
    annotated["is_defining_driver"] = annotated["stress_driver_flag"]
    annotated["is_counter_stress_driver"] = False

    flagged_metrics = set(
        annotated.loc[annotated["stress_driver_flag"], "metric"].astype(str)
    )
    if flagged_metrics != set(driver_metrics):
        raise ValueError(
            "Event-level stress_metric_driver_list does not match the "
            "metric-level stress_driver_flag contract."
        )
    return annotated


def _attach_stress_aligned_drivers(
    events: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Attach authoritative stress drivers and validate both export grains."""
    enriched = events.copy()
    required_event_columns = {EVENT_ID_COLUMN, "stress_metric_driver_list"}
    required_diagnostic_columns = {
        EVENT_ID_COLUMN,
        "metric",
        "stress_driver_flag",
        "directional_label",
    }
    missing_event_columns = sorted(required_event_columns.difference(enriched.columns))
    missing_diagnostic_columns = sorted(
        required_diagnostic_columns.difference(diagnostics.columns)
    )
    if missing_event_columns or missing_diagnostic_columns:
        raise ValueError(
            "Raw 15 requires the corrected 3.3.6 stress-attribution exports. "
            f"Missing event fields: {missing_event_columns or 'none'}; "
            f"missing diagnostic fields: {missing_diagnostic_columns or 'none'}."
        )

    listed = enriched[[EVENT_ID_COLUMN, "stress_metric_driver_list"]].copy()
    listed["metric"] = listed["stress_metric_driver_list"].map(_driver_metrics)
    listed = listed.explode("metric").dropna(subset=["metric"])

    flagged = diagnostics.loc[
        diagnostics["stress_driver_flag"].fillna(False).astype(bool),
        [EVENT_ID_COLUMN, "metric", "directional_label"],
    ].copy()
    flagged["metric"] = flagged["metric"].astype(str)
    invalid_direction_count = int(
        (~flagged["directional_label"].isin(STRESS_ALIGNED_DIRECTIONAL_LABELS)).sum()
    )
    if invalid_direction_count:
        raise ValueError(
            f"Metric diagnostics contain {invalid_direction_count:,} flagged "
            "counter-directional stress-driver rows."
        )

    listed_keys = set(map(tuple, listed[[EVENT_ID_COLUMN, "metric"]].to_numpy()))
    flagged_keys = set(map(tuple, flagged[[EVENT_ID_COLUMN, "metric"]].to_numpy()))
    if listed_keys != flagged_keys:
        raise ValueError(
            "Event-level stress_metric_driver_list does not reconcile to the "
            "metric-level stress_driver_flag rows."
        )

    metric_map = (
        listed.groupby(EVENT_ID_COLUMN, observed=True)["metric"]
        .agg(lambda values: tuple(dict.fromkeys(values)))
        .to_dict()
    )
    enriched["driver_metrics"] = enriched[EVENT_ID_COLUMN].map(metric_map).map(
        lambda value: value if isinstance(value, tuple) else tuple()
    )
    enriched["driver_modes"] = enriched["driver_metrics"].map(
        lambda metrics: tuple(
            mode
            for mode in MODE_ORDER
            if mode in {METRIC_TO_MODE[metric] for metric in metrics}
        )
    )
    enriched["driver_mode_label"] = enriched["driver_modes"].map(
        lambda values: " + ".join(values)
    )
    return enriched


def _csv(frame: pd.DataFrame, *, decimals: int = 3) -> str:
    if frame.empty:
        return "No rows"

    formatted = frame.copy()
    float_columns = formatted.select_dtypes(include=["float", "float32", "float64"]).columns
    formatted[float_columns] = formatted[float_columns].round(decimals)
    return formatted.to_csv(index=False).strip()


def _section(title: str, body: str) -> str:
    return f"\n## {title}\n{body.strip()}\n"


def _parquet_columns(path: Path) -> list[str]:
    return list(pq.ParquetFile(path).schema.names)


@st.cache_data(show_spinner="Scouting candidate stress-anomaly events...")
def _build_scouting_report() -> str:
    events = load_selected_anomaly_events().copy()
    diagnostics = pd.read_parquet(ANOMALY_METRIC_DIAGNOSTICS_PATH).copy()

    events[EVENT_ID_COLUMN] = events[EVENT_ID_COLUMN].astype(str)
    diagnostics[EVENT_ID_COLUMN] = diagnostics[EVENT_ID_COLUMN].astype(str)
    events["date"] = pd.to_datetime(events["date"], errors="coerce")
    diagnostics["date"] = pd.to_datetime(diagnostics["date"], errors="coerce")

    required_event_columns = {
        EVENT_ID_COLUMN,
        "taxi_zone_id",
        "date",
        "temporal_bucket",
        "zone",
        "borough",
        "stress_metric_driver_list",
    }
    required_diagnostic_columns = {
        EVENT_ID_COLUMN,
        "metric",
        "observed_value",
        "expected_value",
        "residual_value",
        "residual_zscore",
        "support_status",
        "stress_driver_flag",
        "directional_label",
    }

    missing_events = sorted(required_event_columns.difference(events.columns))
    missing_diagnostics = sorted(required_diagnostic_columns.difference(diagnostics.columns))
    if missing_events or missing_diagnostics:
        raise ValueError(
            "Raw 15 scouting cannot run because required fields are missing. "
            f"Event fields: {missing_events or 'none'}; "
            f"diagnostic fields: {missing_diagnostics or 'none'}."
        )

    events = _attach_stress_aligned_drivers(events, diagnostics)
    events["driver_metric_count"] = events["driver_metrics"].map(len)
    events["driver_mode_count"] = events["driver_modes"].map(len)
    events["driver_metric_label"] = events["driver_metrics"].map(
        lambda values: " + ".join(METRIC_LABELS.get(value, value) for value in values)
    )
    events["driver_mode_label"] = events["driver_modes"].map(
        lambda values: " + ".join(values)
    )
    events["stress_family"] = _stress_family(events)

    event_context_columns = [
        EVENT_ID_COLUMN,
        "driver_metrics",
        "driver_modes",
    ]
    diagnostics = diagnostics.merge(
        events[event_context_columns],
        on=EVENT_ID_COLUMN,
        how="inner",
        validate="many_to_one",
    )
    diagnostics["metric"] = diagnostics["metric"].astype(str)
    diagnostics["stress_driver_flag"] = (
        diagnostics["stress_driver_flag"].fillna(False).astype(bool)
    )
    diagnostics["is_listed_anomaly_driver"] = diagnostics["stress_driver_flag"]
    diagnostics["is_stress_aligned"] = diagnostics["stress_driver_flag"]
    diagnostics["is_counter_stress"] = _counter_stress_mask(diagnostics)
    diagnostics["is_defining_driver"] = diagnostics["stress_driver_flag"]
    diagnostics["absolute_zscore"] = diagnostics["residual_zscore"].abs()
    diagnostics["direction_is_coherent"] = (
        diagnostics["is_defining_driver"]
        & diagnostics["directional_label"].isin(STRESS_ALIGNED_DIRECTIONAL_LABELS)
    )
    diagnostics["complete_explanation_row"] = diagnostics[
        ["observed_value", "expected_value", "residual_value", "residual_zscore"]
    ].notna().all(axis=1)

    diagnostic_event_summary = (
        diagnostics.groupby(EVENT_ID_COLUMN, observed=True)
        .agg(
            diagnostic_rows=("metric", "size"),
            distinct_diagnostic_metrics=("metric", "nunique"),
            complete_diagnostic_rows=("complete_explanation_row", "sum"),
            max_absolute_zscore=("absolute_zscore", "max"),
            median_absolute_zscore=("absolute_zscore", "median"),
        )
        .reset_index()
    )

    driver_summary = (
        diagnostics[diagnostics["is_defining_driver"]]
        .groupby(EVENT_ID_COLUMN, observed=True)
        .agg(
            matched_driver_rows=("metric", "size"),
            complete_driver_rows=("complete_explanation_row", "sum"),
            coherent_driver_rows=("direction_is_coherent", "sum"),
            strongest_driver_zscore=("absolute_zscore", "max"),
            median_driver_zscore=("absolute_zscore", "median"),
        )
        .reset_index()
    )

    candidates = events.merge(
        diagnostic_event_summary,
        on=EVENT_ID_COLUMN,
        how="left",
        validate="one_to_one",
    ).merge(
        driver_summary,
        on=EVENT_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    numeric_fill_columns = [
        "diagnostic_rows",
        "distinct_diagnostic_metrics",
        "complete_diagnostic_rows",
        "matched_driver_rows",
        "complete_driver_rows",
        "coherent_driver_rows",
    ]
    candidates[numeric_fill_columns] = candidates[numeric_fill_columns].fillna(0)
    candidates["driver_coverage_share"] = np.where(
        candidates["driver_metric_count"] > 0,
        candidates["matched_driver_rows"] / candidates["driver_metric_count"],
        np.nan,
    )
    candidates["driver_completeness_share"] = np.where(
        candidates["matched_driver_rows"] > 0,
        candidates["complete_driver_rows"] / candidates["matched_driver_rows"],
        np.nan,
    )
    candidates["coherent_driver_share"] = np.where(
        candidates["matched_driver_rows"] > 0,
        candidates["coherent_driver_rows"] / candidates["matched_driver_rows"],
        np.nan,
    )

    strength_column = (
        "event_median_directional_strength"
        if "event_median_directional_strength" in candidates.columns
        else "strongest_driver_zscore"
    )
    candidates["ranking_strength"] = pd.to_numeric(
        candidates[strength_column],
        errors="coerce",
    ).abs()
    candidates["post_cp_bonus"] = (
        candidates.get("pre_post_cp", "").astype(str).str.lower().eq("post_cp")
        if "pre_post_cp" in candidates.columns
        else candidates["date"].ge(pd.Timestamp("2025-01-05"))
    ).astype(float)
    candidates["hero_score"] = (
        candidates["ranking_strength"].rank(pct=True).fillna(0) * 0.35
        + candidates["driver_mode_count"].clip(upper=3) / 3 * 0.15
        + candidates["driver_completeness_share"].fillna(0) * 0.20
        + candidates["driver_coverage_share"].fillna(0).clip(upper=1) * 0.15
        + candidates["coherent_driver_share"].fillna(0) * 0.10
        + candidates["post_cp_bonus"] * 0.05
    )

    candidate_columns = [
        EVENT_ID_COLUMN,
        "date",
        "zone",
        "borough",
        "temporal_bucket",
        "stress_family",
        "driver_mode_label",
        "driver_metric_label",
        strength_column,
        "strongest_driver_zscore",
        "driver_coverage_share",
        "driver_completeness_share",
        "coherent_driver_share",
        "hero_score",
    ]
    top_candidates = (
        candidates.sort_values(
            ["hero_score", "ranking_strength", "driver_mode_count"],
            ascending=[False, False, False],
        )[candidate_columns]
        .head(30)
        .reset_index(drop=True)
    )

    top_event_ids = top_candidates[EVENT_ID_COLUMN].head(10).tolist()
    top_evidence = diagnostics[diagnostics[EVENT_ID_COLUMN].isin(top_event_ids)].copy()
    top_evidence["metric_label"] = top_evidence["metric"].map(METRIC_LABELS).fillna(
        top_evidence["metric"]
    )
    top_evidence = top_evidence[
        [
            EVENT_ID_COLUMN,
            "metric_label",
            "observed_value",
            "expected_value",
            "residual_value",
            "residual_zscore",
            "support_status",
            "is_defining_driver",
            "direction_is_coherent",
        ]
    ].sort_values([EVENT_ID_COLUMN, "is_defining_driver", "residual_zscore"], ascending=[True, False, True])

    event_id_duplicates = int(events[EVENT_ID_COLUMN].duplicated().sum())
    diagnostic_duplicates = int(
        diagnostics.duplicated([EVENT_ID_COLUMN, "metric"]).sum()
    )
    diagnostic_id_count = int(diagnostics[EVENT_ID_COLUMN].nunique())
    event_id_count = int(events[EVENT_ID_COLUMN].nunique())
    missing_diagnostic_event_count = len(
        set(events[EVENT_ID_COLUMN]) - set(diagnostics[EVENT_ID_COLUMN])
    )

    family_summary = (
        events["stress_family"]
        .value_counts(dropna=False)
        .rename_axis("stress_family")
        .rename("stress_anomaly_events")
        .reset_index()
    )
    family_summary["share"] = family_summary["stress_anomaly_events"] / len(events)

    mode_breadth = (
        events["driver_mode_count"]
        .value_counts()
        .sort_index()
        .rename_axis("defining_modes")
        .rename("stress_anomaly_events")
        .reset_index()
    )
    mode_breadth["share"] = mode_breadth["stress_anomaly_events"] / len(events)

    zone_summary = (
        events.groupby(["taxi_zone_id", "zone", "borough"], observed=True)
        .size()
        .rename("stress_anomaly_events")
        .reset_index()
        .sort_values("stress_anomaly_events", ascending=False)
        .head(15)
    )
    month_summary = (
        events.assign(month=events["date"].dt.to_period("M").astype(str))
        .groupby("month", observed=True)
        .size()
        .rename("stress_anomaly_events")
        .reset_index()
        .sort_values("stress_anomaly_events", ascending=False)
        .head(15)
    )

    universe_columns = _parquet_columns(ANOMALY_EVENT_UNIVERSE_PATH)
    diagnostic_columns = _parquet_columns(ANOMALY_METRIC_DIAGNOSTICS_PATH)
    context_keywords = (
        "observed",
        "expected",
        "residual",
        "trend",
        "season",
        "baseline",
        "zscore",
    )
    universe_context_columns = [
        column
        for column in universe_columns
        if any(keyword in column.lower() for keyword in context_keywords)
    ]
    diagnostic_context_columns = [
        column
        for column in diagnostic_columns
        if any(keyword in column.lower() for keyword in context_keywords)
    ]

    qa = pd.DataFrame(
        [
            ("Selected stress-anomaly rows", len(events)),
            ("Distinct selected event IDs", event_id_count),
            ("Duplicate selected event IDs", event_id_duplicates),
            ("Metric-diagnostic rows", len(diagnostics)),
            ("Distinct diagnostic event IDs", diagnostic_id_count),
            ("Duplicate event × metric diagnostics", diagnostic_duplicates),
            ("Selected events missing diagnostics", missing_diagnostic_event_count),
            ("Distinct diagnostic metrics", diagnostics["metric"].nunique()),
            ("First selected event date", events["date"].min().date()),
            ("Last selected event date", events["date"].max().date()),
            ("Taxi Zones represented", events["taxi_zone_id"].nunique()),
            ("Temporal buckets represented", events["temporal_bucket"].nunique()),
        ],
        columns=["check", "result"],
    )

    coverage = pd.DataFrame(
        [
            (
                "Events with every listed driver matched to a diagnostic row",
                int(candidates["driver_coverage_share"].eq(1).sum()),
                float(candidates["driver_coverage_share"].eq(1).mean()),
            ),
            (
                "Events with complete observed/expected/residual/z-score driver evidence",
                int(candidates["driver_completeness_share"].eq(1).sum()),
                float(candidates["driver_completeness_share"].eq(1).mean()),
            ),
            (
                "Events whose defining drivers all follow the expected stress direction",
                int(candidates["coherent_driver_share"].eq(1).sum()),
                float(candidates["coherent_driver_share"].eq(1).mean()),
            ),
        ],
        columns=["coverage_check", "events", "share"],
    )

    finder_summary = pd.DataFrame(
        [
            ("Distinct Taxi Zones", events["taxi_zone_id"].nunique()),
            ("Distinct event dates", events["date"].nunique()),
            ("Median events per Taxi Zone", events.groupby("taxi_zone_id").size().median()),
            ("Median events per active date", events.groupby("date").size().median()),
            ("Maximum events on one date", events.groupby("date").size().max()),
            ("Events with 2+ defining modes", int(events["driver_mode_count"].ge(2).sum())),
            ("Events with 3+ defining modes", int(events["driver_mode_count"].ge(3).sum())),
        ],
        columns=["finder_check", "result"],
    )

    timeline_assessment = pd.DataFrame(
        [
            (
                "Selected metric diagnostics",
                "Selected stress anomalies only",
                ", ".join(diagnostic_context_columns) or "None",
                "Explains the selected event but cannot by itself reconstruct a continuous surrounding timeline.",
            ),
            (
                "Full event universe",
                "All Taxi Zone × date × daypart observations",
                ", ".join(universe_context_columns) or "None",
                (
                    "Potential direct timeline source; inspect the listed fields."
                    if universe_context_columns
                    else "No obvious expected/residual fields found; observed values may need to come from the analysis panel and expected values may require an app table."
                ),
            ),
        ],
        columns=["source", "row_universe", "relevant_columns", "phase_4_implication"],
    )

    report = REPORT_TITLE + "\n"
    report += _section("CONTRACT QA", _csv(qa, decimals=3))
    report += _section("STRESS-FAMILY MIX", _csv(family_summary, decimals=3))
    report += _section("MODALITY BREADTH", _csv(mode_breadth, decimals=3))
    report += _section("EXPLANATION COVERAGE", _csv(coverage, decimals=3))
    report += _section("TOP 30 FROZEN-HERO CANDIDATES", _csv(top_candidates, decimals=3))
    report += _section("TOP-10 CANDIDATE METRIC EVIDENCE", _csv(top_evidence, decimals=3))
    report += _section("EVENT-FINDER FEASIBILITY", _csv(finder_summary, decimals=3))
    report += _section("TOP TAXI ZONES BY EVENT COUNT", _csv(zone_summary, decimals=1))
    report += _section("TOP MONTHS BY EVENT COUNT", _csv(month_summary, decimals=1))
    report += _section("TEMPORAL-CONTEXT FEASIBILITY", _csv(timeline_assessment, decimals=3))
    report += _section(
        "PHASE-1 DECISION NOTES",
        """
1. Select the frozen hero from the candidate table only after reviewing its metric-evidence rows; the score is a scouting aid, not the final editorial decision.
2. Prefer a clear, coherent event with complete observed-versus-expected evidence over the single largest z-score.
3. The production page should call the object a stress-anomaly event: one Taxi Zone × date × daypart observation with reconciled metric evidence.
4. A map can serve as an alternate first-stage finder because events cover many Taxi Zones; after selecting a zone, a short sortable event table should select the canonical comparison_event_id.
5. A calendar or timeline finder is warranted only if event density remains readable after the user first narrows geography or stress family.
6. Phase 4 must not fabricate expected values. If the full universe lacks continuous expected-value fields, build a compact event-context app table from the authoritative upstream model outputs.
        """,
    )
    return report.strip()


@st.cache_data(show_spinner="Loading the featured stress-anomaly event...")
def _load_frozen_hero() -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    events = load_selected_anomaly_events().copy()
    diagnostics = pd.read_parquet(ANOMALY_METRIC_DIAGNOSTICS_PATH).copy()

    events[EVENT_ID_COLUMN] = events[EVENT_ID_COLUMN].astype(str)
    diagnostics[EVENT_ID_COLUMN] = diagnostics[EVENT_ID_COLUMN].astype(str)
    matches = events.loc[events[EVENT_ID_COLUMN].eq(FROZEN_HERO_EVENT_ID)].copy()
    if matches.empty:
        raise ValueError(
            f"The featured event {FROZEN_HERO_EVENT_ID!r} is not present in the "
            "selected stress-anomaly export."
        )

    event = matches.iloc[0]
    events["date"] = pd.to_datetime(events["date"], errors="coerce")
    events["stress_family"] = _stress_family(events)
    events = _attach_stress_aligned_drivers(events, diagnostics)
    context_start = pd.Timestamp(event["date"]) - pd.DateOffset(months=6)
    context_end = pd.Timestamp(event["date"]) + pd.DateOffset(months=6)
    zone_context = events.loc[
        events["taxi_zone_id"].eq(event["taxi_zone_id"])
        & events["date"].between(context_start, context_end)
    ].copy()
    zone_context["is_featured"] = zone_context[EVENT_ID_COLUMN].eq(
        FROZEN_HERO_EVENT_ID
    )
    driver_metrics = set(
        events.loc[
            events[EVENT_ID_COLUMN].eq(FROZEN_HERO_EVENT_ID), "driver_metrics"
        ].iloc[0]
    )
    evidence = diagnostics.loc[
        diagnostics[EVENT_ID_COLUMN].eq(FROZEN_HERO_EVENT_ID)
    ].copy()
    if evidence.empty:
        raise ValueError("The featured event has no defining metric diagnostics.")

    evidence["metric_label"] = evidence["metric"].map(METRIC_LABELS).fillna(
        evidence["metric"]
    )
    evidence = _annotate_evidence(evidence, driver_metrics)
    evidence["mode"] = evidence["metric"].map(METRIC_TO_MODE)
    evidence["stress_signal"] = np.where(
        evidence["metric"].isin(DEMAND_METRICS),
        "Demand pressure",
        "Congestion pressure",
    )
    evidence["observed_display"] = evidence.apply(
        lambda row: f"{row['observed_value']:,.0f}"
        if row["metric"] in DEMAND_METRICS
        else f"{row['observed_value']:,.2f}",
        axis=1,
    )
    evidence["expected_display"] = evidence.apply(
        lambda row: f"{row['expected_value']:,.0f}"
        if row["metric"] in DEMAND_METRICS
        else f"{row['expected_value']:,.2f}",
        axis=1,
    )
    evidence["support_label"] = evidence["support_status"].str.title()
    return (
        event,
        evidence.sort_values(
            ["is_defining_driver", "metric_label"], ascending=[False, True]
        ),
        zone_context,
    )


def _evidence_table(evidence: pd.DataFrame) -> pd.DataFrame:
    table = evidence.copy()
    table["difference"] = table["observed_value"] - table["expected_value"]
    positive = table["residual_zscore"].gt(0)
    unavailable = (
        table["support_status"].eq("unavailable")
        | table["observed_value"].isna()
        | table["expected_value"].isna()
        | table["residual_zscore"].isna()
    )
    table["why_it_matters"] = np.select(
        [
            unavailable,
            ~unavailable & table["metric"].isin(DEMAND_METRICS) & positive,
            ~unavailable & table["metric"].isin(DEMAND_METRICS) & ~positive,
            ~unavailable & table["metric"].str.contains("speed", na=False) & positive,
            ~unavailable & table["metric"].str.contains("speed", na=False) & ~positive,
            ~unavailable & table["metric"].str.contains("duration", na=False) & positive,
            ~unavailable & table["metric"].str.contains("duration", na=False) & ~positive,
        ],
        [
            "No comparable baseline — unavailable",
            "Higher activity than expected — stress aligned",
            "Lower activity than expected — counter-stress",
            "Faster movement than expected — counter-stress",
            "Slower movement than expected — stress aligned",
            "Longer travel time than expected — stress aligned",
            "Shorter travel time than expected — counter-stress",
        ],
        default="Departure from expectation",
    )
    table["statistical_distance"] = table["residual_zscore"].map(
        lambda value: "Not available"
        if pd.isna(value)
        else (
            f"{abs(value):,.1f} residual-scale units "
            f"{'above' if value > 0 else 'below'}"
        )
    )
    return table.rename(
        columns={
            "metric_label": "Metric",
            "observed_value": "Observed",
            "expected_value": "Expected",
            "difference": "Difference",
            "residual_scale": "Residual scale",
            "statistical_distance": "Statistical distance",
            "why_it_matters": "Why it matters",
            "support_label": "Support",
        }
    )[
        [
            "Metric",
            "Observed",
            "Expected",
            "Difference",
            "Residual scale",
            "Statistical distance",
            "Why it matters",
            "Support",
        ]
    ]


def _event_context_chart(
    context: pd.DataFrame,
    *,
    interactive: bool = False,
) -> alt.Chart:
    daypart_order = ["Overnight", "AM Peak", "Midday", "PM Peak", "Evening"]
    context = context.copy()
    context["event_key"] = context[EVENT_ID_COLUMN].astype(str)
    context["day_type"] = np.where(
        context["temporal_bucket"].str.startswith("weekend"), "Weekend", "Weekday"
    )
    context["daypart"] = (
        context["temporal_bucket"]
        .str.replace(r"^(weekday|weekend)_", "", regex=True)
        .str.replace("_", " ", regex=False)
        .str.title()
        .replace({"Am Peak": "AM Peak", "Pm Peak": "PM Peak"})
    )
    tooltip = [
        alt.Tooltip("date:T", title="Date", format="%b %d, %Y"),
        alt.Tooltip("daypart:N", title="Daypart"),
        alt.Tooltip("day_type:N", title="Day type"),
        alt.Tooltip("stress_family:N", title="Stress family"),
        alt.Tooltip("driver_mode_label:N", title="Defining modes"),
    ]
    base = alt.Chart(context).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y")),
        y=alt.Y(
            "daypart:N",
            title=None,
            sort=daypart_order,
            axis=alt.Axis(labelLimit=180),
        ),
    )
    events = base.mark_circle(size=62, opacity=0.68).encode(
        color=alt.Color(
            "stress_family:N",
            title=None,
            scale=alt.Scale(
                domain=["Congestion", "Demand", "Both"],
                range=["#007681", "#E79573", "#83C5BE"],
            ),
            legend=alt.Legend(orient="top", direction="horizontal"),
        ),
        tooltip=tooltip,
    )
    selected = base.transform_filter(
        alt.datum.is_featured == True  # noqa: E712
    ).mark_point(
        size=260,
        filled=False,
        stroke="#003F46",
        strokeWidth=3,
    ).encode(tooltip=tooltip)
    layered = (events + selected).properties(height=300)
    if not interactive:
        return layered

    event_pick = alt.selection_point(
        name="event_pick",
        fields=["event_key"],
        on="click",
        clear=False,
        empty=False,
    )
    selectable_events = events.encode(
        opacity=alt.condition(event_pick, alt.value(1.0), alt.value(0.62))
    )
    return (selectable_events + selected).add_params(event_pick).properties(height=300)


@st.cache_data(show_spinner="Loading stress-anomaly events...")
def _load_profiler_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    events = load_selected_anomaly_events().copy()
    diagnostics = pd.read_parquet(ANOMALY_METRIC_DIAGNOSTICS_PATH).copy()
    events[EVENT_ID_COLUMN] = events[EVENT_ID_COLUMN].astype(str)
    diagnostics[EVENT_ID_COLUMN] = diagnostics[EVENT_ID_COLUMN].astype(str)
    events["date"] = pd.to_datetime(events["date"], errors="coerce")
    events["stress_family"] = _stress_family(events)
    events["day_type"] = np.where(
        events["temporal_bucket"].str.startswith("weekend"), "Weekend", "Weekday"
    )
    events["daypart"] = (
        events["temporal_bucket"]
        .str.replace(r"^(weekday|weekend)_", "", regex=True)
        .str.replace("_", " ", regex=False)
        .str.title()
        .replace({"Am Peak": "AM Peak", "Pm Peak": "PM Peak"})
    )
    events = _attach_stress_aligned_drivers(events, diagnostics)
    events["zone_label"] = (
        events["zone"].fillna("Unknown zone").astype(str)
        + " · "
        + events["borough"].fillna("Unknown borough").astype(str)
        + " · ID "
        + events["taxi_zone_id"].astype(str)
    )
    return events, diagnostics


def _event_evidence(
    event: pd.Series,
    diagnostics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_id = str(event[EVENT_ID_COLUMN])
    drivers = set(event["driver_metrics"])
    evidence = diagnostics.loc[diagnostics[EVENT_ID_COLUMN].eq(event_id)].copy()
    evidence["metric_label"] = evidence["metric"].map(METRIC_LABELS).fillna(
        evidence["metric"]
    )
    evidence = _annotate_evidence(evidence, drivers)
    evidence["support_label"] = evidence["support_status"].str.title()
    return (
        evidence.loc[evidence["is_defining_driver"]].copy(),
        evidence.loc[~evidence["is_defining_driver"]].copy(),
    )


@st.cache_data(show_spinner=False)
def _load_series_stress_driver_events(
    taxi_zone_id: int | float | str,
    daypart: str,
    metric: str,
) -> pd.DataFrame:
    """Load selected stress events attributed to one Zone × daypart × metric."""
    available_columns = set(_parquet_columns(ANOMALY_METRIC_DIAGNOSTICS_PATH))
    required_columns = {
        EVENT_ID_COLUMN,
        "taxi_zone_id",
        "date",
        "daypart",
        "metric",
        "stress_driver_flag",
        "support_status",
    }
    missing = sorted(required_columns.difference(available_columns))
    if missing:
        raise ValueError(
            "The metric-diagnostics export cannot supply timeline markers. "
            "Missing columns: " + ", ".join(missing)
        )

    optional_columns = [
        column
        for column in ["temporal_bucket", "directional_label"]
        if column in available_columns
    ]
    normalized_zone_id = int(float(taxi_zone_id))
    normalized_daypart = str(daypart).strip().lower().replace(" ", "_")
    drivers = pd.read_parquet(
        ANOMALY_METRIC_DIAGNOSTICS_PATH,
        columns=[*required_columns, *optional_columns],
        filters=[
            ("taxi_zone_id", "==", normalized_zone_id),
            ("daypart", "==", normalized_daypart),
            ("metric", "==", str(metric)),
            ("stress_driver_flag", "==", True),
        ],
    ).copy()
    drivers["date"] = pd.to_datetime(drivers["date"], errors="coerce")
    return (
        drivers.dropna(subset=["date"])
        .drop_duplicates(["taxi_zone_id", "date", "daypart", "metric"])
        .sort_values("date")
        .reset_index(drop=True)
    )


def _metric_history_chart(
    history: pd.DataFrame,
    *,
    selected_date: pd.Timestamp,
    metric_label: str,
    stress_driver_events: pd.DataFrame,
    show_all_stress_events: bool,
) -> alt.Chart:
    """Plot observed history, expected baseline, and a robust-scale band."""
    chart_data = history.copy()
    chart_data["reference_center"] = (
        chart_data["expected_value"] + chart_data["residual_center"]
    )
    chart_data["reference_lower"] = (
        chart_data["reference_center"] - 2 * chart_data["residual_scale"]
    )
    chart_data["reference_upper"] = (
        chart_data["reference_center"] + 2 * chart_data["residual_scale"]
    )
    selected_date = pd.Timestamp(selected_date).normalize()
    context_start = selected_date - pd.DateOffset(months=6)
    context_end = selected_date + pd.DateOffset(months=6)
    chart_data = chart_data.loc[
        chart_data["date"].between(context_start, context_end)
    ].copy()

    tooltip = [
        alt.Tooltip("date:T", title="Date", format="%b %d, %Y"),
        alt.Tooltip("observed_value:Q", title="Observed", format=",.3f"),
        alt.Tooltip("expected_value:Q", title="Expected", format=",.3f"),
        alt.Tooltip(
            "residual_zscore:Q",
            title="Reconstructed residual score",
            format=".3f",
        ),
        alt.Tooltip("support_status:N", title="Support"),
        alt.Tooltip("scale_method:N", title="Scale method"),
    ]
    shared_x = alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y"))

    legend_domain = [
        "Observed",
        "Expected",
        "±2 residual-scale band",
    ]
    legend_range = [
        "#006D77",
        "#E29578",
        "#83C5BE",
    ]
    series_color = alt.Color(
        "series:N",
        title=None,
        scale=alt.Scale(domain=legend_domain, range=legend_range),
        legend=None,
    )

    band = alt.Chart(chart_data).mark_area(
        opacity=0.28,
    ).encode(
        x=shared_x,
        y=alt.Y(
            "reference_lower:Q",
            title=metric_label,
            scale=alt.Scale(zero=False),
        ),
        y2="reference_upper:Q",
        color=series_color,
        tooltip=tooltip,
    ).transform_calculate(series="'±2 residual-scale band'")

    line_data = pd.concat(
        [
            chart_data.assign(series="Observed", plot_value=chart_data["observed_value"]),
            chart_data.assign(series="Expected", plot_value=chart_data["expected_value"]),
        ],
        ignore_index=True,
    )
    lines = alt.Chart(line_data).mark_line(
        strokeWidth=2.2,
    ).encode(
        x=shared_x,
        y=alt.Y("plot_value:Q", scale=alt.Scale(zero=False)),
        color=series_color,
        strokeDash=alt.StrokeDash(
            "series:N",
            scale=alt.Scale(
                domain=["Observed", "Expected"],
                range=[[1, 0], [6, 4]],
            ),
            legend=None,
        ),
        tooltip=tooltip,
    )
    marker_dates = stress_driver_events.copy()
    marker_dates["date"] = pd.to_datetime(marker_dates["date"], errors="coerce")
    if not show_all_stress_events:
        marker_dates = marker_dates.loc[
            marker_dates["date"].dt.normalize().eq(selected_date)
        ].copy()
    marker_columns = ["date", "support_status"]
    if "temporal_bucket" in marker_dates.columns:
        marker_columns.append("temporal_bucket")
    marker_data = chart_data.merge(
        marker_dates[marker_columns].drop_duplicates("date"),
        on="date",
        how="inner",
        suffixes=("", "_driver"),
        validate="one_to_one",
    )
    marker_data["marker_role"] = np.where(
        marker_data["date"].dt.normalize().eq(selected_date),
        "Selected stress event",
        "Other stress-driver event",
    )
    marker_data["marker_size"] = np.where(
        marker_data["marker_role"].eq("Selected stress event"), 230, 105
    )
    marker_data["marker_width"] = np.where(
        marker_data["marker_role"].eq("Selected stress event"), 3.0, 1.7
    )
    marker_tooltip = [
        *tooltip,
        alt.Tooltip("marker_role:N", title="Timeline marker"),
    ]
    markers = alt.Chart(marker_data).mark_point(filled=False).encode(
        x=shared_x,
        y=alt.Y("observed_value:Q", scale=alt.Scale(zero=False)),
        color=alt.Color(
            "marker_role:N",
            title=None,
            scale=alt.Scale(
                domain=["Selected stress event", "Other stress-driver event"],
                range=["#003F46", "#4F9DA6"],
            ),
            legend=None,
        ),
        size=alt.Size("marker_size:Q", scale=None, legend=None),
        strokeWidth=alt.StrokeWidth(
            "marker_width:Q", scale=None, legend=None
        ),
        tooltip=marker_tooltip,
    )
    policy_date = pd.Timestamp("2025-01-05")
    layers: list[alt.Chart] = [band, lines, markers]
    if context_start <= policy_date <= context_end:
        policy_rule = alt.Chart(pd.DataFrame({"date": [policy_date]})).mark_rule(
            color="#006D77",
            strokeDash=[5, 5],
            opacity=0.72,
        ).encode(x="date:T")
        layers.append(policy_rule)
    return (
        alt.layer(*layers)
        .resolve_scale(color="independent")
        .properties(height=330)
    )


def _render_metric_history(
    event: pd.Series,
    defining_evidence: pd.DataFrame,
    *,
    key_prefix: str,
) -> None:
    """Render a lazy-loaded defining-metric history for one selected event."""
    if defining_evidence.empty:
        st.warning("This event has no defining metric evidence to chart.")
        return

    metric_options = defining_evidence["metric"].astype(str).tolist()
    selected_metric = st.selectbox(
        "Defining metric to inspect",
        metric_options,
        format_func=lambda metric: METRIC_LABELS.get(metric, metric),
        key=f"{key_prefix}_metric",
    )
    selected_diagnostic = defining_evidence.loc[
        defining_evidence["metric"].eq(selected_metric)
    ].iloc[0]

    show_all_stress_events = st.toggle(
        "Show all stress-driver events in this time series",
        value=False,
        key=f"{key_prefix}_show_all_stress_events",
        help=(
            "Adds a hollow marker wherever this metric supplied stress-aligned "
            "evidence for another selected event in the same Taxi Zone and daypart."
        ),
    )

    with st.spinner("Loading this metric's temporal context..."):
        history = load_metric_history(
            selected_metric,
            event["taxi_zone_id"],
            event["daypart"],
        )
        stress_driver_events = _load_series_stress_driver_events(
            event["taxi_zone_id"],
            event["daypart"],
            selected_metric,
        )
    if history.empty:
        st.warning(
            "No compact metric history is available for this Taxi Zone and "
            "daypart."
        )
        return

    metric_label = METRIC_LABELS.get(selected_metric, selected_metric)
    st.markdown(
        """
        <div style="display:flex;flex-wrap:wrap;gap:10px 20px;align-items:center;"
             aria-label="Metric-history chart legend">
          <span><span style="display:inline-block;width:24px;border-top:3px solid #006D77;
                margin-right:6px;vertical-align:middle;"></span>Observed</span>
          <span><span style="display:inline-block;width:24px;border-top:3px dashed #E29578;
                margin-right:6px;vertical-align:middle;"></span>Expected</span>
          <span><span style="display:inline-block;width:24px;height:10px;background:#83C5BE;
                opacity:.38;margin-right:6px;vertical-align:middle;"></span>±2 residual-scale band</span>
          <span><span style="display:inline-block;width:11px;height:11px;border:3px solid #003F46;
                border-radius:50%;margin-right:6px;vertical-align:middle;"></span>Selected stress event</span>
          <span><span style="display:inline-block;width:9px;height:9px;border:2px solid #4F9DA6;
                border-radius:50%;margin-right:6px;vertical-align:middle;"></span>Other stress-driver event</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.altair_chart(
        _metric_history_chart(
            history,
            selected_date=pd.Timestamp(event["date"]),
            metric_label=metric_label,
            stress_driver_events=stress_driver_events,
            show_all_stress_events=show_all_stress_events,
        ),
        width="stretch",
    )
    exact_zscore = selected_diagnostic.get("residual_zscore")
    direction = "above" if pd.notna(exact_zscore) and exact_zscore > 0 else "below"
    zscore_text = (
        f"**{abs(exact_zscore):,.1f} residual-scale units {direction}** the "
        "model expectation"
        if pd.notna(exact_zscore)
        else "outside the available expected context"
    )
    support_status = str(
        selected_diagnostic.get("support_status", "unknown")
    ).lower()
    support_explanation = {
        "modeled": (
            "The expected line comes from the preferred same-daypart seasonal model."
        ),
        "fallback": (
            "The expected line is a simpler running average of earlier observations "
            "in this daypart because the series did not support the preferred "
            "seasonal model."
        ),
        "unavailable": (
            "No usable expectation could be estimated for this series."
        ),
    }.get(support_status, "The baseline method is not identified.")
    if support_status == "fallback" and selected_metric in MODELED_SUPPORT_THRESHOLDS:
        non_null_cutoff, span_cutoff, variance_floor = MODELED_SUPPORT_THRESHOLDS[
            selected_metric
        ]
        observed = pd.to_numeric(history["observed_value"], errors="coerce")
        non_null_count = int(observed.notna().sum())
        active_span_days = int(
            (history["date"].max() - history["date"].min()).days + 1
        )
        observed_variance = float(observed.var(ddof=0))
        failed_requirements = []
        if non_null_count < non_null_cutoff:
            failed_requirements.append(
                f"{non_null_count:,} observed dates versus {non_null_cutoff:,} required"
            )
        if active_span_days < span_cutoff:
            failed_requirements.append(
                f"a {active_span_days:,}-day span versus {span_cutoff:,} required"
            )
        if not np.isfinite(observed_variance) or observed_variance <= variance_floor:
            failed_requirements.append(
                f"variance {observed_variance:,.3f} versus more than "
                f"{variance_floor:,.3f} required"
            )
        if failed_requirements:
            support_explanation = (
                "This series used the fallback because it missed the modeled-support "
                "requirement for " + "; ".join(failed_requirements) + "."
            )
    st.info(
        f"For this event, **{metric_label}** was observed at "
        f"**{selected_diagnostic['observed_value']:,.3f}** versus "
        f"**{selected_diagnostic['expected_value']:,.3f}** expected—"
        f"{zscore_text}. The shaded region is a **±2 robust residual-scale "
        "reference band**, not a confidence interval or the anomaly-selection "
        f"threshold. This selected metric uses **{support_status.title()}** "
        f"support. {support_explanation}"
    )


inject_app_css()

st.caption(PAGE_CAPTION)
st.title(PAGE_TITLE)
st.write(
    "Open one stress-anomaly event and trace the evidence behind its flag: the "
    "observed values, modeled expectations, defining metrics, and support status."
)

try:
    hero_event, hero_evidence, hero_zone_context = _load_frozen_hero()
except Exception as error:
    st.error(f"The featured event could not be loaded: {error}")
    st.stop()

hero_date = pd.to_datetime(hero_event["date"], errors="coerce")
hero_family = str(_stress_family(pd.DataFrame([hero_event])).iloc[0])
defining_evidence = hero_evidence.loc[hero_evidence["is_defining_driver"]].copy()
context_evidence = hero_evidence.loc[~hero_evidence["is_defining_driver"]].copy()
hero_modes = tuple(
    mode
    for mode in MODE_ORDER
    if mode in set(defining_evidence["metric"].map(METRIC_TO_MODE).dropna())
)
modeled_drivers = int(defining_evidence["support_status"].eq("modeled").sum())

st.header("Stress anomalies around the featured event")
st.write(
    "Each point is one Brooklyn Navy Yard stress-anomaly event—one date and "
    "daypart—during the six months before and after the featured case. The outlined "
    "point is the event examined below."
)
st.altair_chart(_event_context_chart(hero_zone_context), width="stretch")
featured_day_events = int(hero_zone_context["date"].eq(hero_date).sum())
st.info(
    f"Brooklyn Navy Yard recorded **{len(hero_zone_context):,} stress-anomaly "
    f"events** in this 12-month window. **{featured_day_events:,}** occurred on "
    f"**{hero_date:%B %d, %Y}**; the selected weekend-overnight event is outlined."
)

st.header("A mixed-stress event at Brooklyn Navy Yard")
st.caption(
    f"{hero_date:%B %d, %Y} · Weekend overnight · "
    f"{hero_event.get('zone', 'Unknown zone')}, {hero_event.get('borough', 'Unknown borough')}"
)

card_1, card_2, card_3, card_4 = st.columns(4)
card_1.metric("Stress family", hero_family)
card_2.metric("Defining modes", len(hero_modes), help=" + ".join(hero_modes))
card_3.metric("Defining metrics", len(defining_evidence))
card_4.metric(
    "Modeled evidence",
    f"{modeled_drivers} of {len(defining_evidence)}",
    help="Defining metrics evaluated with modeled rather than fallback support.",
)

st.subheader("Why this observation was flagged")
st.write(
    "For this one Brooklyn Navy Yard overnight observation, for-hire demand was "
    "far above expectation while bus movement was slower than expected. The event "
    "therefore contains both demand and congestion stress."
)
st.dataframe(
    _evidence_table(defining_evidence).style.format(
        {
            "Observed": "{:,.2f}",
            "Expected": "{:,.2f}",
            "Difference": "{:+,.2f}",
            "Residual scale": "{:,.3f}",
        }
    ),
    hide_index=True,
    width="stretch",
)

evidence_lookup = defining_evidence.set_index("metric")
bus = evidence_lookup.loc["avg_bus_speed"]
fhv = evidence_lookup.loc["fhvhv_trip_count"]
taxi = evidence_lookup.loc["taxi_trip_count"]
st.info(
    "This event was flagged as **Both** because two kinds of "
    "stress appeared in the same Taxi Zone × date × time-window observation. "
    f"FHVHV trips reached **{fhv['observed_value']:,.0f}** versus "
    f"**{fhv['expected_value']:,.0f}** expected, and Taxi trips reached "
    f"**{taxi['observed_value']:,.0f}** versus **{taxi['expected_value']:,.1f}**. "
    f"At the same time, average bus speed fell to **{bus['observed_value']:.2f} mph** "
    f"versus **{bus['expected_value']:.2f} mph** expected. These co-occurring "
    "deviations—not the neighborhood’s overall history—define this event."
)

st.subheader("See the selected metric in temporal context")
_render_metric_history(
    hero_event,
    defining_evidence,
    key_prefix="raw15_hero",
)

with st.expander("See counter-stress and other contextual measurements"):
    st.caption(
        "These metrics describe the same observation but did not supply the "
        "stress-aligned evidence used to select it. Some point toward faster "
        "movement or lower demand; unavailable rows have no usable observation or "
        "baseline and therefore have no direction."
    )
    st.dataframe(
        _evidence_table(context_evidence).style.format(
            {
                "Observed": "{:,.2f}",
                "Expected": "{:,.2f}",
                "Difference": "{:+,.2f}",
                "Residual scale": "{:,.3f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )

with st.expander("How to read this event case file"):
    st.markdown(
        "- The first table contains only **stress-aligned defining metrics**: higher "
        "demand, slower speeds, or longer trip durations that also clear the upstream "
        "1.28 residual-scale threshold.\n"
        "- Observed and expected values remain in their original units; the page "
        "does not ask readers to interpret residuals alone.\n"
        "- Demand pressure means unusually high trip or ridership activity. "
        "Congestion pressure means slower speeds or longer durations.\n"
        "- The three featured drivers have **modeled** support. Metrics with fallback "
        "support are labeled explicitly. **Modeled** support uses the preferred "
        "same-daypart seasonal decomposition. **Fallback** means observations exist "
        "but the series was too sparse, too short-lived, or too flat for that model; "
        "its expected value is the running mean of prior observations in the same "
        "daypart. **Unavailable** means there were no observations from which to "
        "estimate either direction or expectation.\n"
        "- A continuous surrounding trend and seasonal decomposition are not yet "
        "available from the event-only diagnostic export."
    )

st.caption(
    "A stress-anomaly event represents one Taxi Zone on one date during one "
    "daypart. The Zone Profile page summarizes how a place behaves across time; "
    "this page explains the evidence for one selected event and places it within "
    "the history of its defining metric."
)

st.divider()
st.header("Find and inspect another event")
st.write(
    "Start with a Taxi Zone, then narrow its stress-anomaly events by date, "
    "daypart, day type, or stress family. Selecting an event updates both the "
    "context view and its observed-versus-expected evidence."
)

all_events, all_diagnostics = _load_profiler_data()
zone_options = sorted(all_events["zone_label"].dropna().unique().tolist())
default_zone_label = next(
    (label for label in zone_options if label.startswith("Brooklyn Navy Yard ·")),
    zone_options[0],
)

control_1, control_2 = st.columns([1, 1.35])
with control_1:
    selected_zone_label = st.selectbox(
        "Taxi Zone",
        zone_options,
        index=zone_options.index(default_zone_label),
        key="raw15_zone",
    )

zone_events = all_events.loc[all_events["zone_label"].eq(selected_zone_label)].copy()
zone_min_date = zone_events["date"].min().date()
zone_max_date = zone_events["date"].max().date()
if selected_zone_label == default_zone_label:
    default_start = max(zone_min_date, (pd.Timestamp("2025-03-02") - pd.DateOffset(months=6)).date())
    default_end = min(zone_max_date, (pd.Timestamp("2025-03-02") + pd.DateOffset(months=6)).date())
else:
    default_end = zone_max_date
    default_start = max(zone_min_date, (pd.Timestamp(default_end) - pd.DateOffset(months=12)).date())

with control_2:
    selected_dates = st.date_input(
        "Event date range",
        value=(default_start, default_end),
        min_value=zone_min_date,
        max_value=zone_max_date,
        key=f"raw15_dates_{selected_zone_label}",
    )

filter_1, filter_2, filter_3 = st.columns(3)
daypart_order = ["Overnight", "AM Peak", "Midday", "PM Peak", "Evening"]
with filter_1:
    selected_dayparts = st.multiselect(
        "Daypart",
        daypart_order,
        default=daypart_order,
        key="raw15_dayparts",
    )
with filter_2:
    selected_day_types = st.multiselect(
        "Day type",
        ["Weekday", "Weekend"],
        default=["Weekday", "Weekend"],
        key="raw15_day_types",
    )
with filter_3:
    selected_families = st.multiselect(
        "Stress family",
        ["Congestion", "Demand", "Both"],
        default=["Congestion", "Demand", "Both"],
        key="raw15_families",
    )

if isinstance(selected_dates, (tuple, list)):
    if len(selected_dates) == 2:
        selected_start, selected_end = map(pd.Timestamp, selected_dates)
    elif len(selected_dates) == 1:
        # Streamlit briefly returns a one-item tuple after the first click while
        # the user is still completing a date range. Treat it as a one-day range
        # so the intermediate rerun remains valid.
        selected_start = selected_end = pd.Timestamp(selected_dates[0])
    else:
        selected_start = selected_end = pd.Timestamp(default_end)
else:
    selected_start = selected_end = pd.Timestamp(selected_dates)

filtered_events = zone_events.loc[
    zone_events["date"].between(selected_start, selected_end)
    & zone_events["daypart"].isin(selected_dayparts)
    & zone_events["day_type"].isin(selected_day_types)
    & zone_events["stress_family"].isin(selected_families)
].copy()

if filtered_events.empty:
    st.warning("No stress-anomaly events match the current controls.")
else:
    filtered_events = filtered_events.sort_values(
        ["date", "temporal_bucket"], ascending=[False, True]
    )
    filtered_events["event_label"] = filtered_events.apply(
        lambda row: (
            f"{row['date']:%b %d, %Y} · {row['daypart']} · {row['day_type']} · "
            f"{row['stress_family']} · {row['driver_mode_label']}"
        ),
        axis=1,
    )
    label_to_id = dict(
        zip(filtered_events["event_label"], filtered_events[EVENT_ID_COLUMN])
    )
    default_event_id = (
        FROZEN_HERO_EVENT_ID
        if FROZEN_HERO_EVENT_ID in set(filtered_events[EVENT_ID_COLUMN])
        else str(filtered_events.iloc[0][EVENT_ID_COLUMN])
    )
    valid_event_ids = set(filtered_events[EVENT_ID_COLUMN].astype(str))
    if str(st.session_state.get("raw15_selected_event_id", "")) not in valid_event_ids:
        st.session_state["raw15_selected_event_id"] = default_event_id
    selected_event_id = str(st.session_state["raw15_selected_event_id"])

    chart_events = filtered_events.copy()
    chart_events["is_featured"] = chart_events[EVENT_ID_COLUMN].eq(selected_event_id)
    st.caption("Select any point to open that stress-anomaly event.")
    selection_event = st.altair_chart(
        _event_context_chart(chart_events, interactive=True),
        width="stretch",
        key="raw15_event_raster",
        on_select="rerun",
        selection_mode="event_pick",
    )
    selection_payload = getattr(selection_event, "selection", {})
    selected_points = (
        selection_payload.get("event_pick", [])
        if hasattr(selection_payload, "get")
        else getattr(selection_payload, "event_pick", [])
    )
    if selected_points:
        clicked_event_id = str(selected_points[0].get("event_key", ""))
        if clicked_event_id in valid_event_ids and clicked_event_id != selected_event_id:
            st.session_state["raw15_selected_event_id"] = clicked_event_id
            st.rerun()

    selected_event_id = str(st.session_state["raw15_selected_event_id"])
    selected_event = filtered_events.loc[
        filtered_events[EVENT_ID_COLUMN].eq(selected_event_id)
    ].iloc[0]
    st.info(
        f"The current controls contain **{len(filtered_events):,} stress-anomaly "
        f"events** in **{selected_event['zone']}**. The outlined point is the "
        "event selected for diagnosis below."
    )

    with st.expander("Search for an event instead"):
        current_label = filtered_events.loc[
            filtered_events[EVENT_ID_COLUMN].eq(selected_event_id), "event_label"
        ].iloc[0]
        fallback_label = st.selectbox(
            "Search by date, daypart, stress family, or defining modes",
            list(label_to_id),
            index=list(label_to_id).index(current_label),
            key="raw15_event_fallback",
        )
        if st.button("Open this event", key="raw15_open_fallback"):
            st.session_state["raw15_selected_event_id"] = str(
                label_to_id[fallback_label]
            )
            st.rerun()

    selected_defining, selected_context = _event_evidence(
        selected_event, all_diagnostics
    )
    st.subheader(
        f"{selected_event['zone']} · {selected_event['date']:%B %d, %Y} · "
        f"{selected_event['daypart']}"
    )
    detail_1, detail_2, detail_3, detail_4 = st.columns(4)
    detail_1.metric("Day type", selected_event["day_type"])
    detail_2.metric("Stress family", selected_event["stress_family"])
    detail_3.metric("Defining modes", len(selected_event["driver_modes"]))
    detail_4.metric("Defining metrics", len(selected_defining))

    st.dataframe(
        _evidence_table(selected_defining).style.format(
            {
                "Observed": "{:,.2f}",
                "Expected": "{:,.2f}",
                "Difference": "{:+,.2f}",
                "Residual scale": "{:,.3f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    st.subheader("See the selected metric in temporal context")
    selected_key = re.sub(
        r"[^A-Za-z0-9_]+", "_", str(selected_event_id)
    ).strip("_")
    _render_metric_history(
        selected_event,
        selected_defining,
        key_prefix=f"raw15_event_{selected_key}",
    )

    with st.expander("See counter-stress and other contextual measurements"):
        st.caption(
            "These rows were not used as stress-aligned reasons for selecting this "
            "event. Some may be unusual in the opposite direction."
        )
        st.dataframe(
            _evidence_table(selected_context).style.format(
                {
                    "Observed": "{:,.2f}",
                    "Expected": "{:,.2f}",
                    "Difference": "{:+,.2f}",
                    "Residual scale": "{:,.3f}",
                }
            ),
            hide_index=True,
            width="stretch",
        )
