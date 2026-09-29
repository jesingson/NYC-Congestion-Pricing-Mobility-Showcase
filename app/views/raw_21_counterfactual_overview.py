
"""Raw 21 — Counterfactual Overview.

Page question
-------------
What might mobility have looked like without congestion pricing?

The page teaches the counterfactual visually with a fixed Bronx Taxi-trip
Horizon Braid: observed pre-CP history approaches Jan. 5, 2025, the estimated
no-CP paths continue from that history, and observed post-launch mobility moves
onto a different trajectory. The page then keeps the same visual grammar in
geographic small multiples, summarizes all five mobility measures in a compact
5 × 3 matrix, and ends with a drill-down explorer.

The explorer supports Systemwide, Borough, Policy geography, Mobility
environment, and Taxi Zone views plus weekday/weekend, daypart, end-date,
forecast-horizon, horizon-spread, and post-CP trend-guide controls. Runtime stays
lightweight by reading only compact Chapter 5 Showcase artifacts.
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.data_access.counterfactuals import (
    get_h1_feature_reliance_comparison,
    load_counterfactual_braid_inputs,
    load_counterfactual_prelaunch_temporal_metric,
    load_counterfactual_temporal_explorer_metric,
)

from app.utils.project_branding import (
    BRAND_COLORS,
    apply_branding,
    exploration_section,
    inject_app_css,
    render_chart_insight,
)

# ---------------------------------------------------------------------
# Page contract
# ---------------------------------------------------------------------

PAGE_CAPTION = "NO-CP COUNTERFACTUAL MOBILITY"
PAGE_TITLE = "What might mobility have looked like without congestion pricing?"
PAGE_SUBTITLE = (
    "The observed record tells us what happened after congestion pricing began; it "
    "cannot tell us what the same period would have looked like without the policy. "
    "This page uses the frozen forecasting system to estimate that unseen no-CP path "
    "and compares it with the mobility that was actually observed."
)

PLOT_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
}

CP_START = pd.Timestamp("2025-01-05")
CP_END = pd.Timestamp("2026-03-31")
HORIZONS = [1, 2, 5]
PRIMARY_HORIZON = 1

HERO_METRIC = "taxi_trip_count"
HERO_GEOGRAPHY_TYPE = "Borough"
HERO_GEOGRAPHY_VALUE = "Bronx"

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

DAY_TYPE_OPTIONS = ["All days", "Weekdays", "Weekends"]
DAYPART_OPTIONS = [
    "All dayparts",
    "Overnight",
    "AM peak",
    "Midday",
    "PM peak",
    "Evening",
]

DAYPART_TO_SUFFIX = {
    "Overnight": "overnight",
    "AM peak": "am_peak",
    "Midday": "midday",
    "PM peak": "pm_peak",
    "Evening": "evening",
}

GEOGRAPHY_LENSES = [
    "Systemwide",
    "Borough",
    "Policy geography",
    "Mobility environment",
    "Taxi Zone",
]

COUNTERFACTUAL_PATH_OPTIONS = [
    "All horizons",
    "h=1 only",
]

SAVED_VIEWS = {
    "Bronx Taxi trips": {
        "metric": "taxi_trip_count",
        "geography_type": "Borough",
        "geography_value": "Bronx",
        "day_type": "All days",
        "daypart": "All dayparts",
        "counterfactual_paths": "All horizons",
        "description": (
            "Start with a clear example: observed Bronx Taxi trips rise well above the "
            "estimated no-CP paths after launch."
        ),
    },
    "Brooklyn Taxi trips": {
        "metric": "taxi_trip_count",
        "geography_type": "Borough",
        "geography_value": "Brooklyn",
        "day_type": "All days",
        "daypart": "All dayparts",
        "counterfactual_paths": "All horizons",
        "description": (
            "Compare Brooklyn Taxi trips with the three estimated no-CP paths and see "
            "how the gap develops over time."
        ),
    },
    "Citywide Taxi trips": {
        "metric": "taxi_trip_count",
        "geography_type": "Systemwide",
        "geography_value": None,
        "day_type": "All days",
        "daypart": "All dayparts",
        "counterfactual_paths": "All horizons",
        "description": (
            "Compare observed Taxi trips with all three estimated no-CP paths across "
            "the supported NYC system."
        ),
    },
    "Citywide Subway ridership": {
        "metric": "subway_ridership",
        "geography_type": "Systemwide",
        "geography_value": None,
        "day_type": "All days",
        "daypart": "All dayparts",
        "counterfactual_paths": "All horizons",
        "description": (
            "Compare observed Subway ridership with the three estimated no-CP paths "
            "across the supported NYC system."
        ),
    },
    "Citywide FHVHV speed": {
        "metric": "fhvhv_avg_trip_speed",
        "geography_type": "Systemwide",
        "geography_value": None,
        "day_type": "All days",
        "daypart": "All dayparts",
        "counterfactual_paths": "All horizons",
        "description": (
            "Compare citywide FHVHV speed with the estimated no-CP paths and see how "
            "their direction differs from the Taxi-trip examples."
        ),
    },
    "Gateway Taxi trips": {
        "metric": "taxi_trip_count",
        "geography_type": "Policy geography",
        "geography_value": "gateway_to_cbd",
        "day_type": "All days",
        "daypart": "All dayparts",
        "counterfactual_paths": "All horizons",
        "description": (
            "Focus the Taxi-trip comparison on zones that act as gateways into the "
            "congestion-pricing area."
        ),
    },
    "Manhattan Subway ridership": {
        "metric": "subway_ridership",
        "geography_type": "Borough",
        "geography_value": "Manhattan",
        "day_type": "All days",
        "daypart": "All dayparts",
        "counterfactual_paths": "All horizons",
        "description": (
            "Narrow the Subway comparison to Manhattan while keeping the same "
            "observed-versus-no-CP view."
        ),
    },
    "Custom": None,
}

SAVED_VIEW_OPTIONS = list(SAVED_VIEWS)

METRIC_ORDER = [
    "taxi_trip_count",
    "taxi_avg_trip_speed",
    "fhvhv_trip_count",
    "fhvhv_avg_trip_speed",
    "subway_ridership",
]

METRIC_LABELS = {
    "taxi_trip_count": "Taxi trips",
    "taxi_avg_trip_speed": "Taxi average speed",
    "fhvhv_trip_count": "FHVHV trips",
    "fhvhv_avg_trip_speed": "FHVHV average speed",
    "subway_ridership": "Subway ridership",
}

COUNT_METRICS = {
    "taxi_trip_count",
    "fhvhv_trip_count",
    "subway_ridership",
}

HORIZON_COLORS = {
    1: BRAND_COLORS["dark_teal"],
    2: BRAND_COLORS["seafoam"],
    5: BRAND_COLORS["terracotta"],
}

HORIZON_DASHES = {
    1: "solid",
    2: "dash",
    5: "dot",
}

HORIZON_WIDTHS = {
    1: 3.2,
    2: 2.1,
    5: 2.1,
}

OBSERVED_COLOR = "#003F46"
HISTORY_COLOR = "#5A6B73"
LAUNCH_LINE_COLOR = "rgba(0, 109, 119, 0.35)"
ENVELOPE_COLOR = "rgba(226, 149, 120, 0.16)"
ZERO_LINE_COLOR = "rgba(51, 92, 103, 0.45)"

EXPECTED_BRAID_COLUMNS = {
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

EXPECTED_PRELAUNCH_COLUMNS = {
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

EXPECTED_TEMPORAL_EXPLORER_COLUMNS = {
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

EXPECTED_PRELAUNCH_TEMPORAL_COLUMNS = {
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

EXPECTED_CONCLUSION_COLUMNS = {
    "metric",
    "horizon",
    "primary_gap_pct",
}

# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

# WHY: storage paths, compact-artifact QA, and schema validation belong to the
# shared counterfactual access layer. Keep the rest of this page focused on
# reader controls, aggregation semantics, and visualization.
load_inputs = load_counterfactual_braid_inputs
load_temporal_explorer_metric = load_counterfactual_temporal_explorer_metric
load_prelaunch_temporal_explorer_metric = (
    load_counterfactual_prelaunch_temporal_metric
)


# ---------------------------------------------------------------------
# Labels / helpers
# ---------------------------------------------------------------------



def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def format_geography_value(value: object) -> str:
    text = str(value)

    replacements = {
        "gateway_to_cbd": "Gateway to CBD",
        "adjacent_to_cbd": "Adjacent to CBD",
        "non_cbd": "Outside CBD",
        "cbd": "CBD",
    }

    normalized = text.strip().lower().replace("-", "_").replace(" ", "_")

    if normalized in replacements:
        return replacements[normalized]

    return text.replace("_", " ").strip().title()


def systemwide_value(braid: pd.DataFrame) -> str:
    values = (
        braid.loc[
            braid["geography_type"].eq("Systemwide"),
            "geography_value",
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    if len(values) != 1:
        raise ValueError(
            "Expected exactly one Systemwide geography_value, found "
            f"{values}."
        )

    return values[0]


def level_axis_title(metric: str) -> str:
    if metric == "taxi_trip_count":
        return "Weekly Taxi trips"
    if metric == "fhvhv_trip_count":
        return "Weekly FHVHV trips"
    if metric == "subway_ridership":
        return "Weekly Subway riders"
    if metric == "taxi_avg_trip_speed":
        return "Taxi average speed (mph)"
    if metric == "fhvhv_avg_trip_speed":
        return "FHVHV average speed (mph)"
    return metric_label(metric)


def qa_status_summary(frame: pd.DataFrame) -> tuple[int, int]:
    status_column = "status" if "status" in frame.columns else "Status"
    passed = int(frame[status_column].astype(str).str.upper().eq("PASS").sum())
    failed = int(frame[status_column].astype(str).str.upper().ne("PASS").sum())
    return passed, failed


# ---------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------


def selected_temporal_buckets(
    *,
    day_type: str,
    daypart: str,
) -> list[str]:
    """Translate the two reader-facing time controls into frozen bucket IDs."""
    if day_type not in DAY_TYPE_OPTIONS:
        raise ValueError(f"Unsupported day type: {day_type}")
    if daypart not in DAYPART_OPTIONS:
        raise ValueError(f"Unsupported daypart: {daypart}")

    prefix = {
        "All days": None,
        "Weekdays": "weekday_",
        "Weekends": "weekend_",
    }[day_type]
    suffix = DAYPART_TO_SUFFIX.get(daypart)

    return [
        bucket
        for bucket in TEMPORAL_BUCKET_ORDER
        if (prefix is None or bucket.startswith(prefix))
        and (suffix is None or bucket.endswith(suffix))
    ]


def _scope_explorer_geography(
    frame: pd.DataFrame,
    *,
    geography_type: str,
    geography_value: object,
) -> pd.DataFrame:
    """Apply one of the five Raw 21 geography lenses to zone-level explorer rows."""
    if geography_type == "Systemwide":
        return frame.copy()

    column = {
        "Borough": "borough",
        "Policy geography": "cbd_spatial_category",
        "Mobility environment": "pre_cp_mobility_environment",
        "Taxi Zone": "taxi_zone_id",
    }.get(geography_type)

    if column is None:
        raise ValueError(f"Unsupported geography lens: {geography_type}")

    if geography_type == "Taxi Zone":
        value = int(geography_value)
        mask = pd.to_numeric(
            frame[column], errors="coerce"
        ).eq(value)
    else:
        mask = frame[column].astype(str).eq(str(geography_value))

    return frame.loc[mask].copy()


def taxi_zone_label_lookup(frame: pd.DataFrame) -> dict[int, str]:
    """Return stable reader-facing labels for supported Taxi Zones."""
    zones = (
        frame[["taxi_zone_id", "zone", "borough"]]
        .dropna(subset=["taxi_zone_id", "zone"])
        .drop_duplicates("taxi_zone_id")
        .copy()
    )
    zones = zones.loc[
        zones["zone"].astype(str).str.strip().str.lower().ne("unknown")
    ].copy()

    lookup = {}
    for row in zones.itertuples(index=False):
        zone_id = int(row.taxi_zone_id)
        zone = str(row.zone)
        borough = "" if pd.isna(row.borough) else str(row.borough)
        lookup[zone_id] = (
            f"{zone} · {borough}" if borough and borough != "Unknown" else zone
        )

    return lookup


def explorer_geography_options(
    frame: pd.DataFrame,
    geography_type: str,
) -> list[object]:
    """Return only geography values actually supported for the selected metric."""
    if geography_type == "Systemwide":
        return ["NYC supported system"]

    if geography_type == "Taxi Zone":
        lookup = taxi_zone_label_lookup(frame)
        return sorted(lookup, key=lambda zone_id: lookup[zone_id].lower())

    column = {
        "Borough": "borough",
        "Policy geography": "cbd_spatial_category",
        "Mobility environment": "pre_cp_mobility_environment",
    }[geography_type]

    values = (
        frame[column]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    values = [value for value in values if value.strip().lower() != "unknown"]

    if geography_type == "Borough":
        preferred = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
        ordered = [value for value in preferred if value in values]
        extras = sorted(value for value in values if value not in ordered)
        return [*ordered, *extras]

    return sorted(values, key=lambda value: format_geography_value(value).lower())


def explorer_geography_label(
    *,
    geography_type: str,
    geography_value: object,
    zone_lookup: dict[int, str] | None = None,
) -> str:
    """Format one selected geography for chart titles and saved-story copy."""
    if geography_type == "Systemwide":
        return "NYC supported system"
    if geography_type == "Taxi Zone":
        zone_id = int(geography_value)
        return (zone_lookup or {}).get(zone_id, f"Taxi Zone {zone_id}")
    return format_geography_value(geography_value)


def get_postlaunch_weekly(
    braid: pd.DataFrame,
    *,
    metric: str,
    geography_type: str,
    geography_value: str,
) -> pd.DataFrame:
    """Read the existing all-day weekly braid used by hero and small multiples."""
    result = braid.loc[
        braid["summary_grain"].eq("week")
        & braid["metric"].eq(metric)
        & braid["geography_type"].eq(geography_type)
        & braid["geography_value"].astype(str).eq(str(geography_value))
    ].copy()

    return result.sort_values(["week_start", "horizon"]).reset_index(drop=True)


def get_prelaunch_weekly(
    prelaunch: pd.DataFrame,
    *,
    metric: str,
    geography_type: str,
    geography_value: str,
) -> pd.DataFrame:
    """Read the existing all-day prelaunch history used by hero and small multiples."""
    result = prelaunch.loc[
        prelaunch["summary_grain"].eq("week")
        & prelaunch["metric"].eq(metric)
        & prelaunch["geography_type"].eq(geography_type)
        & prelaunch["geography_value"].astype(str).eq(str(geography_value))
    ].copy()

    return result.sort_values("week_start").reset_index(drop=True)


def aggregate_postlaunch_explorer(
    frame: pd.DataFrame,
    *,
    metric: str,
    geography_type: str,
    geography_value: object,
    day_type: str,
    daypart: str,
    view_through: pd.Timestamp,
) -> pd.DataFrame:
    """Build the selected weekly observed + no-CP braid from zone-level ingredients."""
    buckets = selected_temporal_buckets(
        day_type=day_type,
        daypart=daypart,
    )
    # Weekly explorer rows represent policy-relative weeks. Keep only weeks that
    # have fully elapsed by the chosen end date; the final study week is naturally
    # shorter because the counterfactual ends Mar. 31, 2026.
    period_end = (
        frame["week_start"] + pd.Timedelta(days=6)
    ).where(
        frame["week_start"] + pd.Timedelta(days=6) <= CP_END,
        CP_END,
    )

    scoped = frame.loc[
        frame["target_temporal_bucket"].astype(str).isin(buckets)
        & period_end.le(pd.Timestamp(view_through))
    ].copy()
    scoped = _scope_explorer_geography(
        scoped,
        geography_type=geography_type,
        geography_value=geography_value,
    )

    if scoped.empty:
        return pd.DataFrame()

    if metric in COUNT_METRICS:
        named_aggs = {
            "period_complete": ("period_complete", "all"),
            "support_rows": ("support_rows", "sum"),
            "observed_level": ("observed_level", "sum"),
        }
        for horizon in HORIZONS:
            named_aggs[f"no_cp_h{horizon}"] = (
                f"no_cp_h{horizon}",
                "sum",
            )

        weekly = (
            scoped.groupby(
                "week_start",
                observed=True,
                sort=False,
            )
            .agg(**named_aggs)
            .reset_index()
        )

    else:
        scoped["observed_numerator"] = (
            pd.to_numeric(scoped["observed_level"], errors="coerce")
            * pd.to_numeric(scoped["observed_weight"], errors="coerce")
        )

        for horizon in HORIZONS:
            scoped[f"no_cp_numerator_h{horizon}"] = (
                pd.to_numeric(scoped[f"no_cp_h{horizon}"], errors="coerce")
                * pd.to_numeric(
                    scoped[f"no_cp_weight_h{horizon}"],
                    errors="coerce",
                )
            )

        named_aggs = {
            "period_complete": ("period_complete", "all"),
            "support_rows": ("support_rows", "sum"),
            "observed_numerator": ("observed_numerator", "sum"),
            "observed_weight": ("observed_weight", "sum"),
        }
        for horizon in HORIZONS:
            named_aggs[f"no_cp_numerator_h{horizon}"] = (
                f"no_cp_numerator_h{horizon}",
                "sum",
            )
            named_aggs[f"no_cp_weight_h{horizon}"] = (
                f"no_cp_weight_h{horizon}",
                "sum",
            )

        weekly = (
            scoped.groupby(
                "week_start",
                observed=True,
                sort=False,
            )
            .agg(**named_aggs)
            .reset_index()
        )
        weekly["observed_level"] = (
            weekly["observed_numerator"]
            / weekly["observed_weight"].where(weekly["observed_weight"].gt(0))
        )

        for horizon in HORIZONS:
            weekly[f"no_cp_h{horizon}"] = (
                weekly[f"no_cp_numerator_h{horizon}"]
                / weekly[f"no_cp_weight_h{horizon}"].where(
                    weekly[f"no_cp_weight_h{horizon}"].gt(0)
                )
            )

        keep = [
            "week_start",
            "period_complete",
            "support_rows",
            "observed_level",
            *[f"no_cp_h{horizon}" for horizon in HORIZONS],
        ]
        weekly = weekly[keep]

    weekly["no_cp_low"] = weekly[
        [f"no_cp_h{horizon}" for horizon in HORIZONS]
    ].min(axis=1)
    weekly["no_cp_high"] = weekly[
        [f"no_cp_h{horizon}" for horizon in HORIZONS]
    ].max(axis=1)

    return weekly.sort_values("week_start").reset_index(drop=True)


def aggregate_prelaunch_explorer(
    frame: pd.DataFrame,
    *,
    metric: str,
    geography_type: str,
    geography_value: object,
    day_type: str,
    daypart: str,
) -> pd.DataFrame:
    """Rebuild the matching 52-week observed history for the selected explorer view."""
    buckets = selected_temporal_buckets(
        day_type=day_type,
        daypart=daypart,
    )
    scoped = frame.loc[
        frame["temporal_bucket"].astype(str).isin(buckets)
    ].copy()
    scoped = _scope_explorer_geography(
        scoped,
        geography_type=geography_type,
        geography_value=geography_value,
    )

    if scoped.empty:
        return pd.DataFrame()

    if metric in COUNT_METRICS:
        weekly = (
            scoped.groupby(
                "week_start",
                observed=True,
                sort=False,
            )
            .agg(
                support_rows=("support_rows", "sum"),
                observed_level=("observed_level", "sum"),
            )
            .reset_index()
        )
    else:
        scoped["numerator"] = (
            pd.to_numeric(scoped["observed_level"], errors="coerce")
            * pd.to_numeric(scoped["observed_weight"], errors="coerce")
        )
        weekly = (
            scoped.groupby(
                "week_start",
                observed=True,
                sort=False,
            )
            .agg(
                support_rows=("support_rows", "sum"),
                numerator=("numerator", "sum"),
                observed_weight=("observed_weight", "sum"),
            )
            .reset_index()
        )
        weekly["observed_level"] = (
            weekly["numerator"]
            / weekly["observed_weight"].where(weekly["observed_weight"].gt(0))
        )
        weekly = weekly[["week_start", "support_rows", "observed_level"]]

    weekly["period_complete"] = True
    return weekly.sort_values("week_start").reset_index(drop=True)


def _weekly_wide(weekly: pd.DataFrame) -> pd.DataFrame:
    """Return one row per week with one observed and three no-CP levels."""
    if weekly.empty:
        return pd.DataFrame()

    wide_columns = {f"no_cp_h{horizon}" for horizon in HORIZONS}
    if wide_columns.issubset(weekly.columns):
        result = weekly.copy()
        if "no_cp_low" not in result.columns:
            result["no_cp_low"] = result[list(wide_columns)].min(axis=1)
        if "no_cp_high" not in result.columns:
            result["no_cp_high"] = result[list(wide_columns)].max(axis=1)
        return result.sort_values("week_start").reset_index(drop=True)

    observed_check = (
        weekly.groupby("week_start", observed=True, dropna=False)["observed_level"]
        .agg(["min", "max", "count"])
        .reset_index()
    )

    if not (
        observed_check["count"].eq(3).all()
        and np.allclose(
            observed_check["min"],
            observed_check["max"],
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )
    ):
        raise ValueError(
            "Weekly common-support artifact does not preserve one observed "
            "aggregate across h=1/h=2/h=5."
        )

    observed = (
        weekly.loc[
            weekly["horizon"].eq(1),
            ["week_start", "period_complete", "support_rows", "observed_level"],
        ]
        .drop_duplicates("week_start")
        .set_index("week_start")
    )

    no_cp = (
        weekly.pivot(index="week_start", columns="horizon", values="no_cp_level")
        .reindex(columns=HORIZONS)
        .rename(columns={horizon: f"no_cp_h{horizon}" for horizon in HORIZONS})
    )

    result = observed.join(no_cp, how="inner").reset_index()
    result["no_cp_low"] = result[
        [f"no_cp_h{horizon}" for horizon in HORIZONS]
    ].min(axis=1)
    result["no_cp_high"] = result[
        [f"no_cp_h{horizon}" for horizon in HORIZONS]
    ].max(axis=1)

    return result.sort_values("week_start").reset_index(drop=True)


# ---------------------------------------------------------------------
# Visuals
# ---------------------------------------------------------------------


def build_counterfactual_braid(
    *,
    metric: str,
    geography_label: str,
    prelaunch_weekly: pd.DataFrame,
    postlaunch_weekly: pd.DataFrame,
    compact: bool = False,
    horizons_to_show: list[int] | None = None,
    show_horizon_spread: bool = True,
    show_trend_guides: bool = False,
    hero_mode: bool = False,
) -> go.Figure:
    """Observed history + observed post-launch mobility + selected no-CP paths."""
    postlaunch = _weekly_wide(postlaunch_weekly)
    if postlaunch.empty:
        return go.Figure()

    horizons_to_show = horizons_to_show or list(HORIZONS)
    horizons_to_show = [
        horizon for horizon in HORIZONS if horizon in set(horizons_to_show)
    ]
    if not horizons_to_show:
        raise ValueError("At least one counterfactual horizon must be shown.")

    figure = go.Figure()

    # -------------------------------------------------------------
    # Prelaunch observed history.
    # -------------------------------------------------------------
    anchor_x = None
    anchor_y = None

    if not prelaunch_weekly.empty:
        prelaunch_weekly = (
            prelaunch_weekly
            .sort_values("week_start")
            .reset_index(drop=True)
        )

        anchor_x = pd.Timestamp(prelaunch_weekly.iloc[-1]["week_start"])
        anchor_y = float(prelaunch_weekly.iloc[-1]["observed_level"])

        figure.add_trace(
            go.Scatter(
                x=prelaunch_weekly["week_start"],
                y=prelaunch_weekly["observed_level"],
                customdata=(
                    pd.to_numeric(
                        prelaunch_weekly["observed_level"],
                        errors="coerce",
                    ).map(lambda value: f"{value:,.3f}")
                ),
                mode="lines",
                name="Observed history",
                line={
                    "color": HISTORY_COLOR,
                    "width": 2.2 if compact else 2.6,
                },
                hovertemplate=(
                    "<b>Observed history</b><br>"
                    "%{x|%b %d, %Y}<br>"
                    "%{customdata}"
                    "<extra></extra>"
                ),
                showlegend=not compact,
            )
        )

    # -------------------------------------------------------------
    # Bridge the final pre-CP weekly point to the first real post-CP point.
    # No synthetic observation is inserted; these are visual connectors only.
    # -------------------------------------------------------------
    if anchor_x is not None and anchor_y is not None:
        first_post = postlaunch.iloc[0]
        first_x = pd.Timestamp(first_post["week_start"])

        figure.add_trace(
            go.Scatter(
                x=[anchor_x, first_x],
                y=[anchor_y, float(first_post["observed_level"])],
                mode="lines",
                line={
                    "color": OBSERVED_COLOR,
                    "width": 3.0 if compact else 3.6,
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

        for horizon in horizons_to_show:
            figure.add_trace(
                go.Scatter(
                    x=[anchor_x, first_x],
                    y=[anchor_y, float(first_post[f"no_cp_h{horizon}"])],
                    mode="lines",
                    line={
                        "color": HORIZON_COLORS[horizon],
                        "width": (
                            HORIZON_WIDTHS[horizon]
                            if not compact
                            else max(1.6, HORIZON_WIDTHS[horizon] - 0.6)
                        ),
                        "dash": HORIZON_DASHES[horizon],
                    },
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # -------------------------------------------------------------
    # Hero annotations deliberately point to an established post-launch fork,
    # rather than implying that the first weekly point alone proves a break.
    # -------------------------------------------------------------
    if hero_mode and not compact and len(postlaunch) >= 2:
        annotation_index = min(10, len(postlaunch) - 1)
        annotation_row = postlaunch.iloc[annotation_index]
        annotation_x = pd.Timestamp(annotation_row["week_start"])

        figure.add_annotation(
            x=annotation_x,
            y=float(annotation_row["no_cp_h1"]),
            text="Estimated no-CP path<br>continues near the pre-CP pattern",
            showarrow=True,
            arrowhead=2,
            arrowsize=0.8,
            arrowwidth=1,
            arrowcolor=BRAND_COLORS["dark_teal"],
            ax=92,
            ay=40,
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="rgba(0,109,119,0.18)",
            font={"size": 10, "color": BRAND_COLORS["dark_teal"]},
        )

        figure.add_annotation(
            x=annotation_x,
            y=float(annotation_row["observed_level"]),
            text="Observed Taxi trips<br>move onto a higher trajectory",
            showarrow=True,
            arrowhead=2,
            arrowsize=0.8,
            arrowwidth=1,
            arrowcolor=OBSERVED_COLOR,
            ax=92,
            ay=-40,
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="rgba(0,63,70,0.16)",
            font={"size": 10, "color": OBSERVED_COLOR},
        )

    # -------------------------------------------------------------
    # Horizon spread: range across h=1/h=2/h=5, not statistical uncertainty.
    # -------------------------------------------------------------
    if show_horizon_spread and len(horizons_to_show) > 1:
        visible_columns = [
            f"no_cp_h{horizon}" for horizon in horizons_to_show
        ]
        visible_high = postlaunch[visible_columns].max(axis=1)
        visible_low = postlaunch[visible_columns].min(axis=1)

        figure.add_trace(
            go.Scatter(
                x=postlaunch["week_start"],
                y=visible_high,
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=postlaunch["week_start"],
                y=visible_low,
                mode="lines",
                fill="tonexty",
                fillcolor=ENVELOPE_COLOR,
                line={"width": 0},
                name="Forecast-horizon spread",
                hoverinfo="skip",
                showlegend=not compact,
            )
        )

    # -------------------------------------------------------------
    # Selected no-CP paths, with h=1 emphasized whenever it is visible.
    # -------------------------------------------------------------
    for horizon in horizons_to_show:
        figure.add_trace(
            go.Scatter(
                x=postlaunch["week_start"],
                y=postlaunch[f"no_cp_h{horizon}"],
                customdata=(
                    pd.to_numeric(
                        postlaunch[f"no_cp_h{horizon}"],
                        errors="coerce",
                    ).map(lambda value: f"{value:,.3f}")
                ),
                mode="lines",
                name=f"No-CP h={horizon}",
                line={
                    "color": HORIZON_COLORS[horizon],
                    "width": (
                        HORIZON_WIDTHS[horizon]
                        if not compact
                        else max(1.6, HORIZON_WIDTHS[horizon] - 0.6)
                    ),
                    "dash": HORIZON_DASHES[horizon],
                },
                hovertemplate=(
                    f"<b>No-CP h={horizon}</b><br>"
                    "%{x|%b %d, %Y}<br>"
                    "%{customdata}"
                    "<extra></extra>"
                ),
                showlegend=not compact,
            )
        )

    # -------------------------------------------------------------
    # Observed post-launch path.
    # -------------------------------------------------------------
    figure.add_trace(
        go.Scatter(
            x=postlaunch["week_start"],
            y=postlaunch["observed_level"],
            customdata=(
                pd.to_numeric(
                    postlaunch["observed_level"],
                    errors="coerce",
                ).map(lambda value: f"{value:,.3f}")
            ),
            mode="lines",
            name="Observed after launch",
            line={
                "color": OBSERVED_COLOR,
                "width": 3.0 if compact else 3.6,
            },
            hovertemplate=(
                "<b>Observed after launch</b><br>"
                "%{x|%b %d, %Y}<br>"
                "%{customdata}"
                "<extra></extra>"
            ),
            showlegend=not compact,
        )
    )

    # -------------------------------------------------------------
    # Optional post-CP visual trend guides for the interactive explorer.
    # These summarize the visible weeks only; they are not additional forecasts.
    # -------------------------------------------------------------
    if show_trend_guides and not compact and len(postlaunch) >= 2:
        elapsed_days = (
            postlaunch["week_start"] - postlaunch["week_start"].min()
        ).dt.days.to_numpy(dtype=float)

        def add_trend(column: str, *, name: str, color: str) -> None:
            values = pd.to_numeric(
                postlaunch[column], errors="coerce"
            ).to_numpy(dtype=float)
            valid = np.isfinite(elapsed_days) & np.isfinite(values)
            if valid.sum() < 2:
                return

            slope, intercept = np.polyfit(
                elapsed_days[valid], values[valid], 1
            )
            fitted = intercept + slope * elapsed_days

            figure.add_trace(
                go.Scatter(
                    x=postlaunch["week_start"],
                    y=fitted,
                    mode="lines",
                    name=name,
                    line={
                        "color": color,
                        "width": 2,
                        "dash": "longdash",
                    },
                    opacity=0.72,
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

        add_trend(
            "observed_level",
            name="Observed post-CP trend",
            color=OBSERVED_COLOR,
        )
        add_trend(
            "no_cp_h1",
            name="No-CP h=1 trend",
            color=HORIZON_COLORS[1],
        )

    figure.add_vline(
        x=CP_START,
        line_width=1.5,
        line_dash="dash",
        line_color=LAUNCH_LINE_COLOR,
    )

    figure.add_annotation(
        x=CP_START,
        y=1.05,
        xref="x",
        yref="paper",
        text="Jan 5, 2025<br>Congestion pricing launch",
        showarrow=False,
        font={"size": 11, "color": BRAND_COLORS["dark_teal"]},
        align="left",
        bgcolor="rgba(255,255,255,0.75)",
    )

    figure = apply_branding(figure)

    figure.update_layout(
        title={"text": f"{metric_label(metric)} · {geography_label}"},
        height=320 if compact else 540,
        margin={
            "l": 60,
            "r": 18,
            "t": 88 if not compact else 64,
            "b": 60 if not compact else 42,
        },
        hovermode="x unified",
        legend=(
            {}
            if compact
            else {
                "orientation": "h",
                "yanchor": "top",
                "y": -0.14,
                "xanchor": "center",
                "x": 0.5,
            }
        ),
    )

    figure.update_xaxes(showgrid=True, title=None)
    figure.update_yaxes(
        title=None if compact else level_axis_title(metric),
        rangemode="tozero" if metric in COUNT_METRICS else "normal",
    )

    return figure


def build_results_matrix(conclusions: pd.DataFrame) -> go.Figure:
    """Compact 5 × 3 full-period result matrix."""
    scoped = conclusions.loc[
        conclusions["metric"].isin(METRIC_ORDER) & conclusions["horizon"].isin(HORIZONS)
    ].copy()

    pivot = (
        scoped.pivot(index="metric", columns="horizon", values="primary_gap_pct")
        .reindex(index=METRIC_ORDER, columns=HORIZONS)
    )

    values = pivot.to_numpy(dtype=float)
    bound = max(1.0, float(np.nanmax(np.abs(values))))

    text = np.vectorize(
        lambda value: f"{value:+.1f}%" if np.isfinite(value) else "—"
    )(values)
    hover_values = np.vectorize(
        lambda value: f"{value:+.3f}%" if np.isfinite(value) else "—"
    )(values)

    figure = go.Figure(
        go.Heatmap(
            z=values,
            customdata=hover_values,
            x=["h=1", "h=2", "h=5"],
            y=[metric_label(metric) for metric in METRIC_ORDER],
            zmin=-bound,
            zmax=bound,
            zmid=0,
            colorscale=[
                [0.00, BRAND_COLORS["dark_teal"]],
                [0.43, BRAND_COLORS["seafoam"]],
                [0.50, "white"],
                [0.57, BRAND_COLORS["pale_peach"]],
                [1.00, BRAND_COLORS["terracotta"]],
            ],
            text=text,
            texttemplate="<b>%{text}</b>",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "%{x}<br>"
                "No-CP − observed: %{customdata}"
                "<extra></extra>"
            ),
            colorbar={
                "title": "No-CP −<br>observed (%)",
                "thickness": 14,
            },
        )
    )

    figure = apply_branding(figure)
    figure.update_layout(
        title={"text": "Observed mobility versus the estimated no-CP world"},
        height=400,
        margin={"l": 120, "r": 30, "t": 70, "b": 35},
    )
    figure.update_yaxes(autorange="reversed")
    return figure


def _available_geographies(frame: pd.DataFrame, geography_type: str) -> list[str]:
    values = (
        frame.loc[
            frame["geography_type"].eq(geography_type),
            "geography_value",
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    if geography_type == "Borough":
        preferred = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
        ordered = [value for value in preferred if value in values]
        extras = [value for value in values if value not in ordered and value != "Unknown"]
        return [*ordered, *sorted(extras)]

    return sorted(values, key=format_geography_value)


def build_small_multiple_grid(
    *,
    metric: str,
    geography_type: str,
    prelaunch: pd.DataFrame,
    postlaunch: pd.DataFrame,
) -> list[tuple[str, go.Figure, dict]]:
    """Return braid figures plus a transparent trajectory screen for each panel."""
    cards = []

    for geography_value in _available_geographies(postlaunch, geography_type):
        if str(geography_value).lower() == "unknown":
            continue

        prelaunch_weekly = get_prelaunch_weekly(
            prelaunch,
            metric=metric,
            geography_type=geography_type,
            geography_value=geography_value,
        )

        postlaunch_weekly = get_postlaunch_weekly(
            postlaunch,
            metric=metric,
            geography_type=geography_type,
            geography_value=geography_value,
        )

        if postlaunch_weekly.empty:
            continue

        figure = build_counterfactual_braid(
            metric=metric,
            geography_label=format_geography_value(geography_value),
            prelaunch_weekly=prelaunch_weekly,
            postlaunch_weekly=postlaunch_weekly,
            compact=True,
        )

        check = counterfactual_pattern_check(
            prelaunch_weekly,
            _weekly_wide(postlaunch_weekly),
        )

        cards.append(
            (
                format_geography_value(geography_value),
                figure,
                check,
            )
        )

    return cards



# ---------------------------------------------------------------------
# Feature-reliance comparison
# ---------------------------------------------------------------------

RELIANCE_FAMILY_ORDER = [
    "Mobility history",
    "Weekly seasonality",
    "Calendar timing",
    "Spatial / network",
    "Zone identity",
    "Multimodal context",
    "Weather",
    "Policy context",
]

RELIANCE_FAMILY_COLORS = {
    "Mobility history": "#005C64",
    "Weekly seasonality": "#83C5BE",
    "Calendar timing": "#CB6D51",
    "Spatial / network": "#5E8CA1",
    "Zone identity": "#7D6EA8",
    "Multimodal context": "#8A9B0F",
    "Weather": "#B5C99A",
    "Policy context": "#6C757D",
}


RELIANCE_FAMILY_EXAMPLES = {
    "Mobility history": (
        "recent target lags, recent sequence history, comparable-daypart history"
    ),
    "Weekly seasonality": (
        "weekly STL seasonal state and recurring weekly pattern"
    ),
    "Calendar timing": (
        "daypart, day of week, origin/target calendar, annual calendar cycle"
    ),
    "Spatial / network": (
        "static geography, connected-zone context, transportation-network context"
    ),
    "Zone identity": (
        "Taxi Zone identity / series identity"
    ),
    "Multimodal context": (
        "other-mode mobility and connected multimodal conditions"
    ),
    "Weather": (
        "temperature, precipitation, wind, and related weather context"
    ),
    "Policy context": (
        "congestion-pricing regime indicator / policy-regime context"
    ),
}


def build_feature_reliance_comparison(
    reliance: pd.DataFrame,
    *,
    metric: str,
) -> go.Figure:
    """
    Compare the h=1 information mix in ordinary vs synthetic-world forecasting.

    WHY: both rows are 100% compositions, so readers can see which information
    families gain or lose share when the production writer moves into the
    recursive no-CP world.
    """
    scoped = reliance.loc[
        reliance["metric"].eq(metric)
    ].copy()

    if scoped.empty:
        return go.Figure()

    wide = (
        scoped.pivot_table(
            index="environment",
            columns="information_family",
            values="reliance_share_pct",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(
            index=[
                "Chapter 4 forecaster",
                "Synthetic-world forecaster",
            ],
            columns=RELIANCE_FAMILY_ORDER,
            fill_value=0.0,
        )
    )

    figure = go.Figure()

    for family in RELIANCE_FAMILY_ORDER:
        values = pd.to_numeric(
            wide[family],
            errors="coerce",
        ).fillna(0.0)

        hover_values = values.map(
            lambda value: f"{float(value):.3f}%"
        )

        examples = RELIANCE_FAMILY_EXAMPLES[
            family
        ]

        customdata = np.column_stack(
            [
                hover_values.to_numpy(),
                np.repeat(
                    examples,
                    len(values),
                ),
            ]
        )

        figure.add_trace(
            go.Bar(
                x=values,
                y=[
                    "Chapter 4 forecast",
                    "Synthetic no-CP forecast",
                ],
                orientation="h",
                name=family,
                marker={
                    "color": RELIANCE_FAMILY_COLORS[
                        family
                    ]
                },
                customdata=customdata,
                hovertemplate=(
                    f"<b>{family}</b><br>"
                    "%{y}<br>"
                    "Share of reliance: %{customdata[0]}<br>"
                    "Examples: %{customdata[1]}"
                    "<extra></extra>"
                ),
            )
        )

    figure = apply_branding(
        figure
    )

    family = (
        scoped[
            "family"
        ]
        .dropna()
        .astype(str)
        .iloc[0]
    )

    figure.update_layout(
        title={
            "text": (
                f"{metric_label(metric)} · h=1 {family} writer"
            )
        },
        barmode="stack",
        height=330,
        margin={
            "l": 175,
            "r": 20,
            "t": 70,
            "b": 95,
        },
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.22,
            "yanchor": "top",
            "title": {
                "text": "",
            },
        },
        hovermode="closest",
    )

    figure.update_xaxes(
        title_text="Share of measured feature reliance",
        title_standoff=6,
        range=[
            0,
            100,
        ],
        ticksuffix="%",
        dtick=20,
    )
    figure.update_yaxes(
        title_text=None,
        autorange="reversed",
    )

    return figure


def feature_reliance_takeaway(
    reliance: pd.DataFrame,
    *,
    metric: str,
) -> str:
    """Explain the largest composition shift for the selected metric."""
    scoped = reliance.loc[
        reliance["metric"].eq(metric)
    ].copy()

    wide = (
        scoped.pivot_table(
            index="information_family",
            columns="environment",
            values="reliance_share_pct",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(
            RELIANCE_FAMILY_ORDER
        )
        .fillna(0.0)
    )

    if wide.empty:
        return (
            "Feature-reliance comparison is unavailable for this metric."
        )

    wide[
        "change_pp"
    ] = (
        wide[
            "Synthetic-world forecaster"
        ]
        - wide[
            "Chapter 4 forecaster"
        ]
    )

    gain_family = wide[
        "change_pp"
    ].idxmax()
    loss_family = wide[
        "change_pp"
    ].idxmin()

    gain = float(
        wide.loc[
            gain_family,
            "change_pp",
        ]
    )
    loss = float(
        wide.loc[
            loss_family,
            "change_pp",
        ]
    )

    return (
        f"When **{metric_label(metric)}** moves into the recursive no-CP world, "
        f"**{gain_family}** gains the most share of measured reliance "
        f"({gain:+.1f} percentage points), while **{loss_family}** loses the most "
        f"({loss:+.1f} points). Each row still sums to 100%, so this describes a "
        "change in information mix rather than a causal effect."
    )


# ---------------------------------------------------------------------
# Reader-facing summaries and explorer state
# ---------------------------------------------------------------------


def full_period_gap(
    conclusions: pd.DataFrame,
    *,
    metric: str,
    horizon: int,
) -> float:
    """Return the one frozen full-period gap for a metric × horizon."""
    match = conclusions.loc[
        conclusions["metric"].eq(metric)
        & conclusions["horizon"].eq(horizon),
        "primary_gap_pct",
    ]

    if len(match) != 1:
        raise ValueError(
            "Expected exactly one full-period conclusion for "
            f"{metric} h={horizon}; found {len(match)}."
        )

    return float(match.iloc[0])


def median_weekly_gap_pct(
    weekly_wide: pd.DataFrame,
    *,
    horizon: int,
) -> float:
    """Return the median weekly no-CP minus observed gap for one visible path."""
    if weekly_wide.empty:
        return np.nan

    observed = pd.to_numeric(
        weekly_wide["observed_level"], errors="coerce"
    )
    no_cp = pd.to_numeric(
        weekly_wide[f"no_cp_h{horizon}"], errors="coerce"
    )

    gap_pct = np.where(
        observed.ne(0),
        100.0 * (no_cp - observed) / observed,
        np.nan,
    )
    finite = pd.Series(gap_pct).replace([np.inf, -np.inf], np.nan).dropna()
    return float(finite.median()) if not finite.empty else np.nan


def weekly_gap_pct_series(
    weekly_wide: pd.DataFrame,
    *,
    horizon: int = PRIMARY_HORIZON,
) -> pd.Series:
    """Return the weekly no-CP minus observed percentage gap."""
    if weekly_wide.empty:
        return pd.Series(dtype=float)

    observed = pd.to_numeric(
        weekly_wide["observed_level"], errors="coerce"
    )
    no_cp = pd.to_numeric(
        weekly_wide[f"no_cp_h{horizon}"], errors="coerce"
    )

    gap = pd.Series(
        np.where(
            observed.ne(0),
            100.0 * (no_cp - observed) / observed,
            np.nan,
        ),
        index=weekly_wide.index,
        dtype=float,
    )
    return gap.replace([np.inf, -np.inf], np.nan)


def visible_gap_summary(weekly_wide: pd.DataFrame) -> dict:
    """Summarize h=1 gap magnitude, endpoint, and directional persistence."""
    gap = weekly_gap_pct_series(weekly_wide, horizon=PRIMARY_HORIZON).dropna()

    if gap.empty:
        return {
            "median_gap_pct": np.nan,
            "latest_gap_pct": np.nan,
            "direction_persistence_pct": np.nan,
        }

    median_gap = float(gap.median())
    latest_gap = float(gap.iloc[-1])
    median_sign = np.sign(median_gap)

    if median_sign == 0:
        persistence = float(100.0 * gap.eq(0).mean())
    else:
        persistence = float(100.0 * np.sign(gap).eq(median_sign).mean())

    return {
        "median_gap_pct": median_gap,
        "latest_gap_pct": latest_gap,
        "direction_persistence_pct": persistence,
    }


def median_horizon_gap_spread_pp(
    weekly_wide: pd.DataFrame,
    *,
    horizons: list[int],
) -> float:
    """Return median weekly disagreement across horizon gap percentages."""
    if weekly_wide.empty or len(horizons) < 2:
        return np.nan

    parts = []
    for horizon in horizons:
        parts.append(
            weekly_gap_pct_series(
                weekly_wide,
                horizon=horizon,
            ).rename(horizon)
        )

    gap_frame = pd.concat(parts, axis=1)
    spread = gap_frame.max(axis=1) - gap_frame.min(axis=1)
    spread = spread.replace([np.inf, -np.inf], np.nan).dropna()
    return float(spread.median()) if not spread.empty else np.nan


def counterfactual_pattern_check(
    prelaunch_weekly: pd.DataFrame,
    postlaunch_weekly_wide: pd.DataFrame,
) -> dict:
    """
    Screen the h=1 path for obvious discontinuity or scale drift.

    WHY: after Jan. 5 the true no-CP world is unobserved, so this is not an
    accuracy test. It asks a narrower question: does the estimated h=1 path
    behave like a broadly plausible continuation of its own prelaunch history?
    The deliberately wide thresholds are a reader-facing warning system, not a
    statistical acceptance test.
    """
    pre = (
        prelaunch_weekly
        .sort_values("week_start")
        .loc[:, "observed_level"]
        .pipe(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if postlaunch_weekly_wide.empty or "no_cp_h1" not in postlaunch_weekly_wide:
        return {
            "status": "Not enough data",
            "launch_jump_units": np.nan,
            "outside_history_pct": np.nan,
            "history_lower": np.nan,
            "history_upper": np.nan,
        }

    no_cp = (
        pd.to_numeric(
            postlaunch_weekly_wide["no_cp_h1"],
            errors="coerce",
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(pre) < 8 or len(no_cp) < 2:
        return {
            "status": "Not enough data",
            "launch_jump_units": np.nan,
            "outside_history_pct": np.nan,
            "history_lower": np.nan,
            "history_upper": np.nan,
        }

    # Compare the first month of h=1 with the final month before launch so a
    # single noisy week cannot trigger the continuity warning by itself.
    recent_pre_level = float(pre.tail(min(4, len(pre))).median())
    first_no_cp_level = float(no_cp.head(min(4, len(no_cp))).median())

    typical_weekly_move = float(pre.diff().abs().dropna().median())
    launch_difference = abs(first_no_cp_level - recent_pre_level)

    if typical_weekly_move > 0:
        launch_jump_units = launch_difference / typical_weekly_move
    else:
        launch_jump_units = 0.0 if launch_difference == 0 else np.inf

    # Use a deliberately broad prelaunch envelope. The 5th/95th percentile
    # range is padded by 1.5 IQRs so ordinary seasonality is not flagged merely
    # for reaching the edge of the prior year's observed range.
    q05, q25, q75, q95 = pre.quantile([0.05, 0.25, 0.75, 0.95]).tolist()
    iqr = float(q75 - q25)
    padding = max(1.5 * iqr, 2.0 * typical_weekly_move, 0.0)
    history_lower = max(0.0, float(q05 - padding))
    history_upper = float(q95 + padding)

    outside_history = no_cp.lt(history_lower) | no_cp.gt(history_upper)
    outside_history_pct = float(100.0 * outside_history.mean())

    if launch_jump_units >= 10.0 or outside_history_pct >= 50.0:
        status = "Strong caution"
    elif launch_jump_units >= 5.0 or outside_history_pct >= 25.0:
        status = "Caution"
    else:
        status = "Within pattern"

    return {
        "status": status,
        "launch_jump_units": float(launch_jump_units),
        "outside_history_pct": outside_history_pct,
        "history_lower": history_lower,
        "history_upper": history_upper,
        "recent_pre_level": recent_pre_level,
        "first_no_cp_level": first_no_cp_level,
        "typical_weekly_move": typical_weekly_move,
    }


def pattern_check_summary(check: dict) -> str:
    """Return one short explanation for the counterfactual trajectory screen."""
    status = check.get("status", "Not enough data")
    launch_units = check.get("launch_jump_units", np.nan)
    outside_pct = check.get("outside_history_pct", np.nan)

    if status == "Not enough data":
        return "Not enough history is available to run the trajectory check."

    if status == "Within pattern":
        return (
            "The h=1 path stays within the page's broad continuity checks against "
            "the selected geography's own prelaunch history."
        )

    reasons = []
    if np.isfinite(launch_units) and launch_units >= 5.0:
        reasons.append(
            f"the launch transition is {launch_units:.1f}× a typical prelaunch "
            "week-to-week move"
        )
    if np.isfinite(outside_pct) and outside_pct >= 25.0:
        reasons.append(
            f"{outside_pct:.0f}% of visible h=1 weeks fall outside the broad "
            "prelaunch envelope"
        )

    if not reasons:
        return "The h=1 path triggered a broad continuity warning."

    return "The h=1 path is flagged because " + " and ".join(reasons) + "."


def counterfactual_flag_tooltip(check: dict) -> str:
    """Return a compact two-line explanation for the trajectory flag."""
    status = check.get("status", "Not enough data")
    launch_units = check.get("launch_jump_units", np.nan)
    outside_pct = check.get("outside_history_pct", np.nan)

    if status == "Not enough data":
        return (
            "Not enough pre-CP history\n"
            "to evaluate this trajectory."
        )

    if status == "Within pattern":
        return (
            "No flag\n"
            "The path stays within its pre-CP pattern."
        )

    launch_flag = np.isfinite(launch_units) and launch_units >= 5.0
    range_flag = np.isfinite(outside_pct) and outside_pct >= 25.0

    if launch_flag and range_flag:
        return (
            "Flagged: unusual departure from this area's pre-CP pattern.\n"
            f"Launch jump: {launch_units:.3f}× typical weekly movement · "
            f"{outside_pct:.3f}% of h=1 weeks outside the historical range."
        )

    if launch_flag:
        return (
            "Flagged: unusually large launch jump.\n"
            f"{launch_units:.3f}× a typical pre-CP weekly move."
        )

    if range_flag:
        return (
            "Flagged: unusually far outside the pre-CP pattern.\n"
            f"{outside_pct:.3f}% of h=1 weeks beyond the historical range."
        )

    return (
        "Flagged\n"
        "The path departs unusually from this area's pre-CP pattern."
    )


def counterfactual_flag_display(check: dict) -> dict:
    """Map the internal screen result to a softer reader-facing flag."""
    status = check.get("status", "Not enough data")

    if status == "Strong caution":
        return {
            "label": "Flagged",
            "color": "#B66A50",
            "background": "#FBEDE7",
        }
    if status == "Caution":
        return {
            "label": "Flagged",
            "color": BRAND_COLORS["terracotta"],
            "background": BRAND_COLORS["pale_peach"],
        }
    if status == "Within pattern":
        return {
            "label": "No flag",
            "color": BRAND_COLORS["dark_teal"],
            "background": BRAND_COLORS["ice"],
        }

    return {
        "label": "Not checked",
        "color": "#7A878C",
        "background": "#F2F4F5",
    }


def render_counterfactual_flag(
    check: dict,
    *,
    compact: bool = False,
) -> None:
    """Render one low-emphasis colored flag with hover/focus explanation."""
    display = counterfactual_flag_display(check)
    tooltip_text = counterfactual_flag_tooltip(check)
    tooltip = html.escape(tooltip_text, quote=True).replace("\n", "&#10;")
    aria_tooltip = html.escape(tooltip_text.replace("\n", " "), quote=True)
    label = html.escape(display["label"])
    # Small-multiple flags belong visually to the chart above them. Pull the
    # compact treatment upward and center it so it cannot be mistaken for a
    # label on the next panel. The interactive-card treatment stays left-aligned.
    margin = "-0.75rem 0 0.15rem" if compact else "0.15rem 0 0"
    justify = "center" if compact else "flex-start"
    size = "0.82rem" if compact else "0.92rem"

    st.markdown(
        f"""
        <div style="margin:{margin}; display:flex; justify-content:{justify};">
          <span
            title="{tooltip}"
            tabindex="0"
            aria-label="{aria_tooltip}"
            style="
              display:inline-flex; align-items:center; gap:0.35rem;
              padding:0.16rem 0.48rem; border-radius:999px;
              background:{display['background']};
              border:1px solid {display['color']}55;
              color:{display['color']}; font-size:{size};
              font-weight:600; cursor:help; line-height:1.25;
            "
          >
            <span aria-hidden="true" style="font-size:1.05em;">⚑</span>
            <span>{label}</span>
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero_takeaway(braid: pd.DataFrame) -> str:
    """Explain the Bronx Taxi counterfactual in direct mobility terms."""
    weekly = get_postlaunch_weekly(
        braid,
        metric=HERO_METRIC,
        geography_type=HERO_GEOGRAPHY_TYPE,
        geography_value=HERO_GEOGRAPHY_VALUE,
    )
    wide = _weekly_wide(weekly)

    horizon_gaps = {
        horizon: median_weekly_gap_pct(wide, horizon=horizon)
        for horizon in HORIZONS
    }
    finite_gaps = [
        gap for gap in horizon_gaps.values()
        if np.isfinite(gap)
    ]

    if not finite_gaps:
        return (
            "The colored paths estimate how Bronx Taxi activity might have evolved "
            "during the post-launch period if congestion pricing had not begun. "
            "A typical weekly difference could not be calculated for this view."
        )

    typical_gap = float(np.median(finite_gaps))

    if typical_gap < 0:
        relationship = (
            "The model estimates that **Bronx Taxi activity would have been "
            f"about {abs(typical_gap):.0f}% lower if congestion pricing had not begun**. "
            "Instead, the Taxi activity actually observed after Jan. 5, 2025 stayed "
            "well above the estimated no-CP paths for most of the period."
        )
    elif typical_gap > 0:
        relationship = (
            "The model estimates that **Bronx Taxi activity would have been "
            f"about {abs(typical_gap):.0f}% higher if congestion pricing had not begun**. "
            "The Taxi activity actually observed after Jan. 5, 2025 stayed below "
            "the estimated no-CP paths for most of the period."
        )
    else:
        relationship = (
            "The model estimates very little typical difference between Bronx Taxi "
            "activity with and without congestion pricing over this period."
        )

    horizon_range = max(finite_gaps) - min(finite_gaps)

    if len(finite_gaps) > 1 and horizon_range < 5.0:
        relationship += (
            " The 1-, 2-, and 5-step forecast horizons tell nearly the same story."
        )

    return relationship


def geography_takeaway(
    braid: pd.DataFrame,
    *,
    metric: str,
    geography_type: str,
) -> str:
    """Translate geographic counterfactual gaps into direct mobility language."""
    weekly = braid.loc[
        braid["summary_grain"].eq("week")
        & braid["metric"].eq(metric)
        & braid["horizon"].eq(PRIMARY_HORIZON)
        & braid["geography_type"].eq(geography_type)
    ].copy()

    weekly["counterfactual_gap_pct"] = pd.to_numeric(
        weekly["counterfactual_gap_pct"],
        errors="coerce",
    )
    weekly = weekly.loc[
        np.isfinite(weekly["counterfactual_gap_pct"])
        & weekly["geography_value"].astype(str).str.lower().ne("unknown")
    ].copy()

    medians = (
        weekly.groupby(
            "geography_value",
            observed=True,
        )["counterfactual_gap_pct"]
        .median()
        .dropna()
    )

    if medians.empty:
        return (
            "These panels compare what was actually observed after Jan. 5, 2025 "
            "with the model's estimate of what the same period might have looked "
            "like if congestion pricing had not begun."
        )

    lower = medians.loc[medians.lt(0)]
    higher = medians.loc[medians.gt(0)]

    metric_name = metric_label(metric)
    geography_name = (
        "boroughs"
        if geography_type == "Borough"
        else "policy geographies"
    )

    # WHY: translate the mathematical sign into the mobility relationship so
    # readers never have to decode "no-CP minus observed" themselves.
    if len(lower) == len(medians):
        direction_sentence = (
            f"Across **all {len(medians)} {geography_name}**, the model estimates "
            f"that **{metric_name} would have been lower without congestion pricing** "
            "than the levels actually observed after launch."
        )
    elif len(higher) == len(medians):
        direction_sentence = (
            f"Across **all {len(medians)} {geography_name}**, the model estimates "
            f"that **{metric_name} would have been higher without congestion pricing** "
            "than the levels actually observed after launch."
        )
    else:
        direction_sentence = (
            f"The estimated no-CP comparison for **{metric_name}** is not uniform "
            f"across {geography_name}: **{len(lower)}** show a lower estimated "
            f"no-CP level than observed, while **{len(higher)}** show a higher one."
        )

    strongest_value = medians.abs().idxmax()
    strongest_gap = float(medians.loc[strongest_value])
    strongest_label = format_geography_value(strongest_value)

    if strongest_gap < 0:
        strongest_sentence = (
            f" The largest typical separation is in **{strongest_label}**, where "
            f"the no-CP estimate is about **{abs(strongest_gap):.1f}% lower** than "
            "the mobility actually observed."
        )
    else:
        strongest_sentence = (
            f" The largest typical separation is in **{strongest_label}**, where "
            f"the no-CP estimate is about **{abs(strongest_gap):.1f}% higher** than "
            "the mobility actually observed."
        )

    return (
        "The colored paths estimate mobility during the post-launch period "
        "**if congestion pricing had not begun**. "
        + direction_sentence
        + strongest_sentence
    )


def matrix_takeaway(conclusions: pd.DataFrame) -> str:
    """Summarize the directional pattern across the 15 headline results."""
    scoped = conclusions.loc[
        conclusions["metric"].isin(METRIC_ORDER)
        & conclusions["horizon"].isin(HORIZONS),
        ["metric", "horizon", "primary_gap_pct"],
    ].copy()

    direction_by_metric = (
        scoped.assign(
            direction=np.sign(
                pd.to_numeric(
                    scoped["primary_gap_pct"],
                    errors="coerce",
                )
            )
        )
        .groupby("metric", observed=True)["direction"]
        .nunique()
    )

    consistent_count = int(direction_by_metric.eq(1).sum())

    return (
        f"All **{consistent_count} mobility measures** keep the same overall gap "
        "direction across h=1, h=2, and h=5. In this sign convention, negative "
        "means the estimated no-CP level is below observed mobility; positive means "
        "it is above observed mobility."
    )


def selected_view_takeaway(
    weekly_wide: pd.DataFrame,
    *,
    prelaunch_weekly: pd.DataFrame,
    metric: str,
    geography_label: str,
    horizons_to_show: list[int],
    day_type: str,
    daypart: str,
    view_through: pd.Timestamp,
) -> str:
    """Explain the visible counterfactual comparison without sign decoding."""
    gap_summary = visible_gap_summary(weekly_wide)
    median_gap = gap_summary["median_gap_pct"]
    latest_gap = gap_summary["latest_gap_pct"]
    persistence = gap_summary["direction_persistence_pct"]

    pattern_check = counterfactual_pattern_check(
        prelaunch_weekly,
        weekly_wide,
    )

    time_scope = (
        f"{day_type.lower()} · {daypart.lower()}"
        if daypart != "All dayparts"
        else day_type.lower()
    )

    metric_name = metric_label(metric)

    parts = [
        f"For **{geography_label} · {metric_name}** ({time_scope}), the colored "
        "path estimates what mobility might have looked like during the same "
        "post-launch period **if congestion pricing had not begun**. "
    ]

    if np.isfinite(median_gap):
        if median_gap < 0:
            parts.append(
                f"The model estimates that {metric_name} would typically have been "
                f"**{abs(median_gap):.1f}% lower without congestion pricing** than "
                "the level actually observed."
            )
        elif median_gap > 0:
            parts.append(
                f"The model estimates that {metric_name} would typically have been "
                f"**{abs(median_gap):.1f}% higher without congestion pricing** than "
                "the level actually observed."
            )
        else:
            parts.append(
                f"The model estimates very little typical difference in {metric_name} "
                "between the no-CP path and what was actually observed."
            )
    else:
        parts.append("A typical h=1 difference is unavailable for this view.")

    if np.isfinite(persistence):
        parts.append(
            f" That relationship holds in **{persistence:.0f}% of visible weeks** "
            f"through **{pd.Timestamp(view_through):%b %d, %Y}**."
        )

    if np.isfinite(latest_gap):
        if latest_gap < 0:
            parts.append(
                f" In the latest visible week, the no-CP estimate is "
                f"**{abs(latest_gap):.1f}% lower** than observed."
            )
        elif latest_gap > 0:
            parts.append(
                f" In the latest visible week, the no-CP estimate is "
                f"**{abs(latest_gap):.1f}% higher** than observed."
            )

    horizon_spread = median_horizon_gap_spread_pp(
        weekly_wide,
        horizons=horizons_to_show,
    )
    if np.isfinite(horizon_spread):
        parts.append(
            " The visible forecast horizons typically differ by only "
            f"**{horizon_spread:.1f} percentage points**."
        )

    if pattern_check["status"] != "Within pattern":
        parts.append(
            " **Counterfactual check: "
            f"{pattern_check['status']}.** "
            + pattern_check_summary(pattern_check)
        )

    return "".join(parts)


def _mark_explorer_custom() -> None:
    """Mark the explorer as custom after the reader changes any control."""
    st.session_state["raw21_saved_view"] = "Custom"


def _apply_saved_view() -> None:
    """Restore all controls encoded by the selected saved story."""
    name = st.session_state.get(
        "raw21_saved_view",
        "Bronx Taxi trips",
    )
    configuration = SAVED_VIEWS.get(name)

    if configuration is None:
        return

    st.session_state["raw21_explore_metric"] = configuration["metric"]
    st.session_state["raw21_explore_geography_type"] = configuration[
        "geography_type"
    ]
    st.session_state["raw21_explore_geography_value"] = configuration[
        "geography_value"
    ]
    st.session_state["raw21_explore_day_type"] = configuration["day_type"]
    st.session_state["raw21_explore_daypart"] = configuration["daypart"]
    st.session_state["raw21_counterfactual_paths"] = configuration[
        "counterfactual_paths"
    ]
    st.session_state["raw21_view_through"] = CP_END.date()
    st.session_state["raw21_show_horizon_spread"] = True
    st.session_state["raw21_show_trend_guides"] = False


def _initialize_explorer_state() -> None:
    """Give Raw 21 the same saved-story-first behavior as other Showcase pages."""
    defaults = SAVED_VIEWS["Bronx Taxi trips"]

    default_values = {
        "raw21_saved_view": "Bronx Taxi trips",
        "raw21_explore_metric": defaults["metric"],
        "raw21_explore_geography_type": defaults["geography_type"],
        "raw21_explore_geography_value": defaults["geography_value"],
        "raw21_explore_day_type": defaults["day_type"],
        "raw21_explore_daypart": defaults["daypart"],
        "raw21_view_through": CP_END.date(),
        "raw21_counterfactual_paths": defaults["counterfactual_paths"],
        "raw21_show_horizon_spread": True,
        "raw21_show_trend_guides": False,
    }

    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------

inject_app_css()

st.caption(PAGE_CAPTION)
st.title(PAGE_TITLE)
st.markdown(
    f'<p class="app-subtitle">{PAGE_SUBTITLE}</p>',
    unsafe_allow_html=True,
)

(
    braid,
    braid_qa,
    prelaunch,
    prelaunch_qa,
    temporal_qa,
    prelaunch_temporal_qa,
    conclusions,
) = load_inputs()

feature_reliance = get_h1_feature_reliance_comparison()

qa_frames = [
    braid_qa,
    prelaunch_qa,
    temporal_qa,
    prelaunch_temporal_qa,
]
qa_failures = sum(qa_status_summary(frame)[1] for frame in qa_frames)

if qa_failures:
    st.error(
        "One or more compact Counterfactual Overview inputs failed QA. "
        "Review the QA details at the bottom of the page before using this view."
    )
    st.stop()

# ------------------------------------------------------------------
# 1. Fixed editorial hero
# ------------------------------------------------------------------

st.divider()
st.subheader("1. How do the observed and estimated no-CP paths separate?")
st.write(
    "The fixed opening example uses **Bronx Taxi trips** to teach the comparison. "
    "The gray line is the observed year before launch; the dark line is what was "
    "actually observed after Jan. 5, 2025; and the colored lines are the forecasting "
    "system's estimated **no-CP** paths. **h=1, h=2, and h=5 are forecast horizons, "
    "not different policy scenarios.** Their spread shows horizon disagreement, not "
    "statistical uncertainty."
)

st.caption(
    "Hero focus: Taxi trips · Bronx · all days and dayparts · weekly view · "
    "Jan. 5, 2025 launch"
)

hero_prelaunch = get_prelaunch_weekly(
    prelaunch,
    metric=HERO_METRIC,
    geography_type=HERO_GEOGRAPHY_TYPE,
    geography_value=HERO_GEOGRAPHY_VALUE,
)
hero_postlaunch = get_postlaunch_weekly(
    braid,
    metric=HERO_METRIC,
    geography_type=HERO_GEOGRAPHY_TYPE,
    geography_value=HERO_GEOGRAPHY_VALUE,
)

hero_figure = build_counterfactual_braid(
    metric=HERO_METRIC,
    geography_label=HERO_GEOGRAPHY_VALUE,
    prelaunch_weekly=hero_prelaunch,
    postlaunch_weekly=hero_postlaunch,
    compact=False,
    horizons_to_show=HORIZONS,
    show_horizon_spread=True,
    show_trend_guides=False,
    hero_mode=True,
)

st.plotly_chart(
    hero_figure,
    width="stretch",
    config=PLOT_CONFIG,
    key="raw21_hero_bronx_taxi_trips",
)

hero_wide = _weekly_wide(hero_postlaunch)
hero_metric_columns = st.columns(3)
for column, horizon in zip(hero_metric_columns, HORIZONS):
    gap = median_weekly_gap_pct(
        hero_wide,
        horizon=horizon,
    )
    column.metric(
        f"h={horizon} median weekly gap",
        f"{gap:+.1f}%" if np.isfinite(gap) else "—",
        help=(
            "Median weekly no-CP minus observed percentage gap for Bronx Taxi trips. "
            "Negative means the estimated no-CP Taxi level is below what was observed."
        ),
    )

render_chart_insight(hero_takeaway(braid))

with st.expander("How to read the Horizon Braid", expanded=False):
    st.markdown(
        """
- **Observed history** shows what actually happened before congestion pricing.
- **Observed after launch** continues the actual weekly mobility record after Jan. 5, 2025.
- **No-CP h=1, h=2, and h=5** are forecast-based estimates of mobility if congestion pricing had not begun. They are different forecast horizons, not different policy scenarios.
- The **forecast-horizon spread** is the range between those three no-CP estimates. It is **not a statistical confidence interval**.
- **Gap = no-CP − observed.** For Taxi trips, a negative gap means the estimated no-CP level is below the number of trips that was actually observed.
- The short connectors from the last pre-CP week to the first post-CP week are visual bridges. They do not invent extra forecast observations.
        """
    )

# ------------------------------------------------------------------
# 2. Geographic small multiples
# ------------------------------------------------------------------

st.divider()
st.subheader("2. Does the pattern look the same across NYC?")
st.write(
    "Switch the mobility measure and compare the same weekly braid across policy "
    "geographies and boroughs. Look for places where the observed and estimated "
    "no-CP paths separate early, late, strongly, or barely at all."
)

small_multiple_metric = st.selectbox(
    "Metric",
    options=METRIC_ORDER,
    index=0,
    format_func=metric_label,
    key="raw21_small_multiple_metric",
)

policy_tab, borough_tab = st.tabs(
    ["Policy geography", "Boroughs"]
)


# ---------------------------------------------------------------------
# Policy-geography small multiples
# ---------------------------------------------------------------------

with policy_tab:
    policy_cards = build_small_multiple_grid(
        metric=small_multiple_metric,
        geography_type="Policy geography",
        prelaunch=prelaunch,
        postlaunch=braid,
    )

    policy_columns = st.columns(2)

    for index, (_, figure, check) in enumerate(policy_cards):
        with policy_columns[index % 2]:
            st.plotly_chart(
                figure,
                width="stretch",
                config=PLOT_CONFIG,
                key=f"raw21_policy_{small_multiple_metric}_{index}",
            )

            if check["status"] in {"Caution", "Strong caution"}:
                render_counterfactual_flag(
                    check,
                    compact=True,
                )

    # WHY: interpret the policy-geography evidence directly rather than
    # forcing readers to translate the no-CP-minus-observed sign themselves.
    render_chart_insight(
        geography_takeaway(
            braid,
            metric=small_multiple_metric,
            geography_type="Policy geography",
        )
    )


# ---------------------------------------------------------------------
# Borough small multiples
# ---------------------------------------------------------------------

with borough_tab:
    borough_cards = build_small_multiple_grid(
        metric=small_multiple_metric,
        geography_type="Borough",
        prelaunch=prelaunch,
        postlaunch=braid,
    )

    borough_columns = st.columns(2)

    for index, (_, figure, check) in enumerate(borough_cards):
        with borough_columns[index % 2]:
            st.plotly_chart(
                figure,
                width="stretch",
                config=PLOT_CONFIG,
                key=f"raw21_borough_{small_multiple_metric}_{index}",
            )

            if check["status"] in {"Caution", "Strong caution"}:
                render_counterfactual_flag(
                    check,
                    compact=True,
                )

    # WHY: keep the borough interpretation specific to the borough evidence
    # instead of combining it with the policy-geography result.
    render_chart_insight(
        geography_takeaway(
            braid,
            metric=small_multiple_metric,
            geography_type="Borough",
        )
    )


# ---------------------------------------------------------------------
# Small-multiple reading guide
# ---------------------------------------------------------------------

with st.expander(
    "How to read the geographic small multiples",
    expanded=False,
):
    st.markdown(
        """
Each panel compares **observed post-launch mobility** with the model's estimate
of what mobility might have looked like during the same period **if congestion
pricing had not begun**.

The dark observed line shows what NYC actually experienced. The colored
counterfactual paths show the estimated no-CP alternatives at different
forecast horizons.

A **Flagged** badge does not mean the result is wrong. It means the estimated
no-CP trajectory deserves extra caution because its post-launch behavior is a
less natural continuation of that geography's pre-launch pattern.
        """
    )

# ------------------------------------------------------------------
# 3. Five-measure summary
# ------------------------------------------------------------------

st.divider()
st.subheader("3. How do all five mobility measures compare?")
st.write(
    "The weekly braids show when the observed and estimated no-CP paths separate. "
    "This matrix compresses the full post-launch period into one gap for each "
    "mobility measure and forecast horizon."
)

matrix = build_results_matrix(conclusions)
st.plotly_chart(
    matrix,
    width="stretch",
    config=PLOT_CONFIG,
    key="raw21_results_matrix",
)

render_chart_insight(matrix_takeaway(conclusions))

with st.expander("How to read the 5 × 3 matrix", expanded=False):
    st.markdown(
        """
- Each cell is the **full-period no-CP minus observed percentage gap** for one mobility measure and forecast horizon.
- **Negative** means the estimated no-CP level is below observed mobility.
- **Positive** means the estimated no-CP level is above observed mobility.
- The sign has to be interpreted in mobility terms: a negative trip-count gap and a positive speed gap do not mean the same thing operationally.
        """
    )

# ------------------------------------------------------------------
# 4. Reader-controlled exploration
# ------------------------------------------------------------------

_initialize_explorer_state()

with exploration_section(
    key="raw21_counterfactual_overview_exploration_area",
    title="Explore the counterfactual yourself",
    description=(
        "Start with a saved story or build your own view. Choose a mobility measure and "
        "place, then narrow the comparison by day type, daypart, and how far after "
        "launch you want to follow it."
    ),
):
    st.selectbox(
        "Start with a story",
        SAVED_VIEW_OPTIONS,
        key="raw21_saved_view",
        on_change=_apply_saved_view,
    )

    selected_story = st.session_state.get(
        "raw21_saved_view",
        "Bronx Taxi trips",
    )
    selected_story_config = SAVED_VIEWS.get(selected_story)

    if selected_story_config is None:
        st.info(
            "Custom view: the controls below no longer match a saved story."
        )
    else:
        st.info(selected_story_config["description"])

    control_row = st.columns([1.05, 1.05, 1.5])

    with control_row[0]:
        explore_metric = st.selectbox(
            "Mobility measure",
            METRIC_ORDER,
            format_func=metric_label,
            key="raw21_explore_metric",
            on_change=_mark_explorer_custom,
        )

    post_temporal = load_temporal_explorer_metric(explore_metric)
    pre_temporal = load_prelaunch_temporal_explorer_metric(explore_metric)

    with control_row[1]:
        explore_geography_type = st.selectbox(
            "Geography level",
            GEOGRAPHY_LENSES,
            key="raw21_explore_geography_type",
            on_change=_mark_explorer_custom,
        )

    geography_value_options = explorer_geography_options(
        post_temporal,
        explore_geography_type,
    )

    if not geography_value_options:
        st.warning(
            "No supported geography values are available for this mobility measure."
        )
        st.stop()

    stored_geography = st.session_state.get(
        "raw21_explore_geography_value"
    )
    if stored_geography not in geography_value_options:
        st.session_state["raw21_explore_geography_value"] = (
            geography_value_options[0]
        )

    zone_lookup = taxi_zone_label_lookup(post_temporal)

    with control_row[2]:
        explore_geography_value = st.selectbox(
            "Geography",
            geography_value_options,
            format_func=lambda value: explorer_geography_label(
                geography_type=explore_geography_type,
                geography_value=value,
                zone_lookup=zone_lookup,
            ),
            key="raw21_explore_geography_value",
            on_change=_mark_explorer_custom,
        )

    time_row = st.columns([1, 1, 1.15])

    with time_row[0]:
        explore_day_type = st.selectbox(
            "Day type",
            DAY_TYPE_OPTIONS,
            key="raw21_explore_day_type",
            on_change=_mark_explorer_custom,
        )

    with time_row[1]:
        explore_daypart = st.selectbox(
            "Daypart",
            DAYPART_OPTIONS,
            key="raw21_explore_daypart",
            on_change=_mark_explorer_custom,
        )

    with time_row[2]:
        view_through_date = st.date_input(
            "View through",
            min_value=(CP_START + pd.Timedelta(days=6)).date(),
            max_value=CP_END.date(),
            key="raw21_view_through",
            on_change=_mark_explorer_custom,
            help=(
                "The counterfactual begins Jan. 5, 2025. Because the chart is weekly, "
                "it shows policy-relative weeks completed by this date. The final "
                "study week ends Mar. 31, 2026."
            ),
        )

    display_row = st.columns([1.15, 1, 1])

    with display_row[0]:
        counterfactual_paths = st.selectbox(
            "Counterfactual paths",
            COUNTERFACTUAL_PATH_OPTIONS,
            key="raw21_counterfactual_paths",
            on_change=_mark_explorer_custom,
            help=(
                "All horizons shows h=1, h=2, and h=5. h=1 only focuses on the "
                "primary recursive no-CP path."
            ),
        )

    with display_row[1]:
        spread_requested = st.checkbox(
            "Show forecast-horizon spread",
            key="raw21_show_horizon_spread",
            on_change=_mark_explorer_custom,
            disabled=counterfactual_paths == "h=1 only",
            help=(
                "Shades the range between the visible no-CP forecast horizons. "
                "It shows horizon disagreement, not statistical uncertainty."
            ),
        )

    with display_row[2]:
        show_trend_guides = st.checkbox(
            "Show post-CP trend guides",
            key="raw21_show_trend_guides",
            on_change=_mark_explorer_custom,
            help=(
                "Adds simple fitted lines to the visible observed and h=1 no-CP paths. "
                "They summarize direction; they are not additional forecasts."
            ),
        )

    horizons_to_show = (
        [1]
        if counterfactual_paths == "h=1 only"
        else list(HORIZONS)
    )
    show_horizon_spread = (
        bool(spread_requested)
        and counterfactual_paths == "All horizons"
    )

    view_through = pd.Timestamp(view_through_date)
    days_after_launch = int((view_through - CP_START).days)
    st.caption(
        f"View ends {days_after_launch:,} days after Jan. 5, 2025. "
        "The chart keeps the full 52-week prelaunch history for context."
    )

    explore_prelaunch = aggregate_prelaunch_explorer(
        pre_temporal,
        metric=explore_metric,
        geography_type=explore_geography_type,
        geography_value=explore_geography_value,
        day_type=explore_day_type,
        daypart=explore_daypart,
    )
    explore_postlaunch = aggregate_postlaunch_explorer(
        post_temporal,
        metric=explore_metric,
        geography_type=explore_geography_type,
        geography_value=explore_geography_value,
        day_type=explore_day_type,
        daypart=explore_daypart,
        view_through=view_through,
    )

    geography_label = explorer_geography_label(
        geography_type=explore_geography_type,
        geography_value=explore_geography_value,
        zone_lookup=zone_lookup,
    )

    if explore_postlaunch.empty:
        st.warning(
            "No completed post-launch weekly rows match this combination of metric, "
            "geography, day type, daypart, and end date."
        )
    else:
        explore_figure = build_counterfactual_braid(
            metric=explore_metric,
            geography_label=geography_label,
            prelaunch_weekly=explore_prelaunch,
            postlaunch_weekly=explore_postlaunch,
            compact=False,
            horizons_to_show=horizons_to_show,
            show_horizon_spread=show_horizon_spread,
            show_trend_guides=show_trend_guides,
            hero_mode=False,
        )

        st.plotly_chart(
            explore_figure,
            width="stretch",
            config=PLOT_CONFIG,
            key="raw21_explore_chart",
        )

        explore_wide = _weekly_wide(explore_postlaunch)
        gap_summary = visible_gap_summary(explore_wide)
        pattern_check = counterfactual_pattern_check(
            explore_prelaunch,
            explore_wide,
        )

        metric_cards = st.columns(4)
        metric_cards[0].metric(
            "Typical h=1 gap",
            (
                f"{gap_summary['median_gap_pct']:+.1f}%"
                if np.isfinite(gap_summary["median_gap_pct"])
                else "—"
            ),
            help=(
                "Median weekly no-CP minus observed percentage gap across the visible "
                "post-launch period."
            ),
        )
        metric_cards[1].metric(
            "Latest h=1 gap",
            (
                f"{gap_summary['latest_gap_pct']:+.1f}%"
                if np.isfinite(gap_summary["latest_gap_pct"])
                else "—"
            ),
            help=(
                "No-CP minus observed percentage gap in the latest visible completed week."
            ),
        )
        metric_cards[2].metric(
            "Direction persistence",
            (
                f"{gap_summary['direction_persistence_pct']:.0f}%"
                if np.isfinite(gap_summary["direction_persistence_pct"])
                else "—"
            ),
            help=(
                "Share of visible weeks where the h=1 gap points in the same direction "
                "as its median gap."
            ),
        )
        with metric_cards[3]:
            st.caption("Trajectory check")
            render_counterfactual_flag(pattern_check, compact=False)

        render_chart_insight(
            selected_view_takeaway(
                explore_wide,
                prelaunch_weekly=explore_prelaunch,
                metric=explore_metric,
                geography_label=geography_label,
                horizons_to_show=horizons_to_show,
                day_type=explore_day_type,
                daypart=explore_daypart,
                view_through=view_through,
            )
        )

        with st.expander("About the trajectory flag", expanded=False):
            st.markdown(
                "The flag highlights counterfactual paths that depart unusually strongly "
                "from the selected geography's prelaunch pattern."
            )

            launch_units = pattern_check.get("launch_jump_units", np.nan)
            outside_pct = pattern_check.get("outside_history_pct", np.nan)
            history_lower = pattern_check.get("history_lower", np.nan)
            history_upper = pattern_check.get("history_upper", np.nan)

            detail_lines = []
            if np.isfinite(launch_units):
                detail_lines.append(
                    "**Launch jump:** the first four h=1 weeks are "
                    f"{launch_units:.3f}× a typical prelaunch weekly move from "
                    "the final prelaunch level."
                )
            if np.isfinite(outside_pct):
                detail_lines.append(
                    "**Historical range:** "
                    f"{outside_pct:.3f}% of visible h=1 weeks fall outside the broad "
                    "prelaunch range"
                    + (
                        f" ({history_lower:,.3f} to {history_upper:,.3f})."
                        if np.isfinite(history_lower) and np.isfinite(history_upper)
                        else "."
                    )
                )

            if detail_lines:
                st.markdown("\n\n".join(detail_lines))

            st.caption(
                "Flag thresholds: 5× launch jump or 25% of visible h=1 weeks outside "
                "the historical range; stronger color begins at 10× or 50%."
            )

        if show_trend_guides:
            st.caption(
                "Trend guides are simple linear fits across the visible post-launch "
                "weekly points. They summarize direction; they do not add a new model "
                "or imply a causal effect."
            )


        st.markdown("#### What information did this model rely on?")
        st.write(
            "The two bars use the same 100% scale. The first shows the information "
            "mix measured for the ordinary Chapter 4 forecast; the second shows the "
            "same h=1 production writer operating inside the recursive synthetic "
            "no-CP world."
        )

        reliance_figure = build_feature_reliance_comparison(
            feature_reliance,
            metric=explore_metric,
        )

        st.plotly_chart(
            reliance_figure,
            width="stretch",
            config=PLOT_CONFIG,
            key=f"raw21_feature_reliance_{explore_metric}",
        )

        render_chart_insight(
            feature_reliance_takeaway(
                feature_reliance,
                metric=explore_metric,
            )
        )

        with st.expander(
            "How to read the feature-reliance comparison",
            expanded=False,
        ):
            st.markdown(
                """
- Both bars sum to **100%**. A larger segment means that information family
  accounts for a larger share of the model's measured reliance mix.
- The comparison follows the **mobility measure selected above**; there is no
  separate metric selector.
- The four Neural writers use grouped permutation reliance. The FHVHV-speed
  Tree writer uses grouped SHAP share, so read the bars as within-model
  composition rather than as directly comparable causal quantities.
- The synthetic-world bar explains how the accepted h=1 writer uses information
  while operating inside the recursively generated no-CP world. It does **not**
  identify what caused the observed post-launch mobility difference.
                """
            )

# ------------------------------------------------------------------
# Closing synthesis
# ------------------------------------------------------------------

st.divider()

st.markdown("### What this page establishes")
st.markdown(
    "The observed post-launch record and the estimated no-CP path are not the same "
    "mobility story, and the size and direction of that gap vary by measure, forecast "
    "horizon, geography, and time. The counterfactual therefore provides a more demanding "
    "reference than a simple before/after comparison: it asks how observed mobility "
    "differed from the trajectory the frozen forecasting system estimated in a synthetic "
    "world where congestion pricing did not begin."
)

with st.expander("How this page works", expanded=False):
    st.markdown(
        """
        **1. Build the no-CP comparison from a frozen forecasting system.** The no-CP
        paths are model-based estimates of an unseen world, not observations from a
        second NYC. The accepted production writers operate recursively inside a
        synthetic post-launch history that does not contain the observed post-CP target
        values they are trying to replace.

        **2. Keep forecast horizon separate from policy scenario.** **h=1, h=2, and h=5**
        are different forecast horizons for the same no-congestion-pricing scenario.
        **h=1** is the primary recursive path; h=2 and h=5 provide additional horizon
        views.

        **3. Define the counterfactual gap consistently.** **Gap = no-CP − observed.**
        A positive gap means the model estimated a higher level without congestion
        pricing; a negative gap means it estimated a lower level. The operational meaning
        depends on whether the measure is demand or speed.

        **4. Do not read horizon spread as uncertainty.** The visible band is simply the
        range across h=1, h=2, and h=5. It is **not a statistical confidence interval**.

        **5. Preserve the Showcase's aggregation rules.** Day type and daypart use the
        same ten frozen temporal buckets. Counts are summed; speed measures remain
        activity-weighted. Mobility-environment views use the frozen **Pre-CP**
        assignment rather than regrouping the synthetic no-CP world after launch.

        **6. Treat trajectory flags and trend guides as diagnostics.** Trajectory flags
        compare h=1 with the selected geography's own prelaunch pattern. Optional trend
        guides are simple fitted lines over the visible weekly paths; neither adds a new
        forecast or establishes a causal effect.

        **7. Interpret feature reliance within the forecasting model.** The reliance
        comparison shows which information families the accepted h=1 writer uses in the
        ordinary forecast and in the recursive synthetic world. It does not identify
        what caused the observed post-launch mobility difference.
        """
    )

st.caption(
    "Evidence scope: model-estimated no-congestion-pricing mobility compared with the "
    "observed post-launch record. These counterfactual gaps support structured comparison "
    "with an estimated no-CP path, but they are not direct observations of an alternative "
    "world and do not by themselves prove that congestion pricing caused the full gap. "
    "Taxi Zone views are local forecast comparisons, not neighborhood-level causal estimates."
)
