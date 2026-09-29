"""Raw 22 — Counterfactual Geography.

This page turns the frozen Chapter 5 no-CP counterfactual into a spatial story.
It compares Observed, estimated No-CP, and their gap across canonical Taxi Zones,
then separates locally dramatic differences from zones that account for more of
the selected systemwide divergence.

The runtime consumes the compact Raw 22 preaggregate only. It does not recreate
counterfactual simulation or scan the multi-million-row Chapter 5 surface.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data_access.spatial_visuals import get_zone_geojson
from app.data_access.counterfactuals import (
    load_counterfactual_geography_explorer,
)

from app.utils.project_branding import (
    BRAND_COLORS,
    BRAND_DIVERGING_SEQUENCE,
    apply_branding,
    exploration_section,
    inject_app_css,
    render_chart_insight,
)


# ---------------------------------------------------------------------
# Page contract
# ---------------------------------------------------------------------

CP_START = pd.Timestamp("2025-01-05")
CP_END = pd.Timestamp("2026-03-31")
HORIZONS = [1, 2, 5]
PRIMARY_HORIZON = 1

HERO_METRIC = "taxi_trip_count"
HERO_HORIZON = 1
HERO_PERIOD = "Full post-CP period"
HERO_DAY_TYPE = "All days"
HERO_DAYPART = "All dayparts"

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

SPEED_METRICS = {
    "taxi_avg_trip_speed",
    "fhvhv_avg_trip_speed",
}

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

DAYPART_SUFFIX = {
    "Overnight": "overnight",
    "AM peak": "am_peak",
    "Midday": "midday",
    "PM peak": "pm_peak",
    "Evening": "evening",
}

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

PERIOD_OPTIONS = [
    "Full post-CP period",
    "First 4 weeks",
    "First 13 weeks (~3 months)",
    "First 26 weeks (~6 months)",
    "Final 13 weeks (~3 months)",
]

POLICY_LABELS = {
    "cbd": "CBD",
    "adjacent_to_cbd": "Adjacent to CBD",
    "gateway_to_cbd": "Gateway to CBD",
    "non_cbd": "Outside CBD / adjacent / gateway",
}

PLOT_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
}

MAP_CENTER = {
    "lat": 40.7128,
    "lon": -74.0060,
}

SEQUENTIAL_SCALE = [
    [0.0, BRAND_COLORS["ice"]],
    [0.50, BRAND_COLORS["seafoam"]],
    [1.0, BRAND_COLORS["dark_teal"]],
]

DIRECTION_COLORS = {
    "No-CP above observed": BRAND_COLORS["terracotta"],
    "No-CP below observed": BRAND_COLORS["dark_teal"],
    "No difference": BRAND_COLORS["seafoam"],
}

EXPECTED_COLUMNS = {
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

SAVED_STORIES = {
    "Taxi trips · full period": {
        "metric": "taxi_trip_count",
        "horizon": 1,
        "period": "Full post-CP period",
        "day_type": "All days",
        "daypart": "All dayparts",
    },
    "Taxi speed · weekday PM": {
        "metric": "taxi_avg_trip_speed",
        "horizon": 1,
        "period": "Full post-CP period",
        "day_type": "Weekdays",
        "daypart": "PM peak",
    },
    "Subway · PM peak": {
        "metric": "subway_ridership",
        "horizon": 1,
        "period": "Full post-CP period",
        "day_type": "All days",
        "daypart": "PM peak",
    },
    "FHVHV trips · full period": {
        "metric": "fhvhv_trip_count",
        "horizon": 1,
        "period": "Full post-CP period",
        "day_type": "All days",
        "daypart": "All dayparts",
    },
    "Custom view": None,
}


# ---------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------

# WHY: the shared counterfactual access layer owns compact-file paths, schema
# checks, and creation-time QA. Raw 22 owns only spatial aggregation,
# interaction, and visualization.
load_inputs = load_counterfactual_geography_explorer


# ---------------------------------------------------------------------
# Reader-facing helpers
# ---------------------------------------------------------------------



def metric_label(metric: str) -> str:
    """Return the stable Showcase label for one mobility measure."""
    return METRIC_LABELS.get(metric, metric)


def native_unit(metric: str) -> str:
    """Return the natural unit used by a selected metric."""
    if metric in {"taxi_trip_count", "fhvhv_trip_count"}:
        return "trips"
    if metric == "subway_ridership":
        return "riders"
    if metric in SPEED_METRICS:
        return "mph"
    return "units"


def level_label(metric: str) -> str:
    """Return a short level label for color bars and cards."""
    if metric == "taxi_trip_count":
        return "Taxi trips"
    if metric == "fhvhv_trip_count":
        return "FHVHV trips"
    if metric == "subway_ridership":
        return "Subway riders"
    return "Average speed (mph)"


def format_level(
    value: float,
    metric: str,
    *,
    signed: bool = False,
) -> str:
    """Format a native-unit value without exceeding three decimals."""
    if not np.isfinite(value):
        return "Not available"

    sign = "+" if signed and value > 0 else ""
    if metric in COUNT_METRICS:
        return f"{sign}{value:,.0f}"
    return f"{sign}{value:,.3f}"


def format_hover_number(
    value: float,
    *,
    signed: bool = False,
    percent: bool = False,
) -> str:
    """Preformat every Plotly hover number to exactly three decimals."""
    if not np.isfinite(value):
        return "Not available"

    sign = "+" if signed and value > 0 else ""
    suffix = "%" if percent else ""
    return f"{sign}{value:,.3f}{suffix}"


def policy_label(value: object) -> str:
    """Translate the frozen policy-geography code into reader-facing text."""
    if pd.isna(value):
        return "Not available"
    text = str(value)
    return POLICY_LABELS.get(text, text.replace("_", " ").title())


def direction_label(value: float) -> str:
    """Translate the frozen No-CP − observed sign into plain language."""
    if not np.isfinite(value):
        return "Gap unavailable"
    if value > 0:
        return "No-CP above observed"
    if value < 0:
        return "No-CP below observed"
    return "No difference"


def selected_temporal_buckets(
    day_type: str,
    daypart: str,
) -> list[str]:
    """Resolve reader-facing Day type × Daypart into frozen bucket IDs."""
    prefix = {
        "All days": None,
        "Weekdays": "weekday_",
        "Weekends": "weekend_",
    }[day_type]
    suffix = DAYPART_SUFFIX.get(daypart)

    return [
        bucket
        for bucket in TEMPORAL_BUCKET_ORDER
        if (prefix is None or bucket.startswith(prefix))
        and (suffix is None or bucket.endswith(suffix))
    ]


def period_mask(
    frame: pd.DataFrame,
    period: str,
) -> pd.Series:
    """Return policy-week-aligned period filters supported by the compact table."""
    week = frame["week_start"]
    unique_weeks = sorted(pd.Timestamp(value) for value in week.dropna().unique())
    if not unique_weeks:
        return pd.Series(False, index=frame.index)

    first = unique_weeks[0]
    last = unique_weeks[-1]

    if period == "Full post-CP period":
        return week.notna()
    if period == "First 4 weeks":
        return week.between(first, first + pd.Timedelta(weeks=3))
    if period == "First 13 weeks (~3 months)":
        return week.between(first, first + pd.Timedelta(weeks=12))
    if period == "First 26 weeks (~6 months)":
        return week.between(first, first + pd.Timedelta(weeks=25))
    if period == "Final 13 weeks (~3 months)":
        return week.between(last - pd.Timedelta(weeks=12), last)

    raise ValueError(f"Unsupported Raw 22 period: {period}")


def period_caption(
    frame: pd.DataFrame,
    period: str,
) -> str:
    """Describe the actual policy-week dates represented by one period choice."""
    mask = period_mask(frame, period)
    weeks = frame.loc[mask, "week_start"].dropna()
    if weeks.empty:
        return period

    start = pd.Timestamp(weeks.min())
    final_week = pd.Timestamp(weeks.max())
    end = min(final_week + pd.Timedelta(days=6), CP_END)
    return f"{start:%b %d, %Y}–{end:%b %d, %Y}"


# ---------------------------------------------------------------------
# Spatial aggregation
# ---------------------------------------------------------------------


def aggregate_zone_slice(
    frame: pd.DataFrame,
    *,
    metric: str,
    horizon: int,
    period: str,
    day_type: str,
    daypart: str,
) -> pd.DataFrame:
    """
    Rebuild one selected spatial slice at canonical Taxi Zone grain.

    Counts/ridership sum. Average speeds are rebuilt from world-specific
    weighted numerators and denominators so any temporal filter preserves the
    frozen Chapter 5 weighting contract.
    """
    buckets = selected_temporal_buckets(day_type, daypart)
    support_column = f"support_rows_h{horizon}"
    observed_column = f"observed_level_h{horizon}"
    no_cp_column = f"no_cp_level_h{horizon}"
    observed_weight_column = f"observed_weight_h{horizon}"
    no_cp_weight_column = f"no_cp_weight_h{horizon}"

    scoped = frame.loc[
        frame["metric"].eq(metric)
        & period_mask(frame, period)
        & frame["target_temporal_bucket"].isin(buckets)
        & pd.to_numeric(frame[support_column], errors="coerce").fillna(0).gt(0)
    ].copy()

    if scoped.empty:
        return pd.DataFrame()

    zone_keys = [
        "canonical_taxi_zone_id",
        "zone",
        "borough",
        "cbd_spatial_category",
        "map_eligible",
        "reader_facing_zone",
    ]

    if metric in COUNT_METRICS:
        zones = (
            scoped.groupby(
                zone_keys,
                observed=True,
                dropna=False,
                as_index=False,
            )
            .agg(
                support_rows=(support_column, "sum"),
                observed_level=(observed_column, "sum"),
                no_cp_level=(no_cp_column, "sum"),
            )
        )
        zones["observed_weight"] = np.nan
        zones["no_cp_weight"] = np.nan
        zones["observed_numerator"] = zones["observed_level"]
        zones["no_cp_numerator"] = zones["no_cp_level"]

    else:
        observed_weight = pd.to_numeric(
            scoped[observed_weight_column],
            errors="coerce",
        ).fillna(0.0)
        no_cp_weight = pd.to_numeric(
            scoped[no_cp_weight_column],
            errors="coerce",
        ).fillna(0.0)
        observed_level = pd.to_numeric(
            scoped[observed_column],
            errors="coerce",
        )
        no_cp_level = pd.to_numeric(
            scoped[no_cp_column],
            errors="coerce",
        )

        # A zero activity weight contributes a zero numerator even when the
        # corresponding average speed is undefined for that tiny slice.
        scoped["_observed_numerator"] = np.where(
            observed_weight.gt(0) & observed_level.notna(),
            observed_level * observed_weight,
            0.0,
        )
        scoped["_no_cp_numerator"] = np.where(
            no_cp_weight.gt(0) & no_cp_level.notna(),
            no_cp_level * no_cp_weight,
            0.0,
        )
        scoped["_observed_weight"] = observed_weight
        scoped["_no_cp_weight"] = no_cp_weight

        zones = (
            scoped.groupby(
                zone_keys,
                observed=True,
                dropna=False,
                as_index=False,
            )
            .agg(
                support_rows=(support_column, "sum"),
                observed_numerator=("_observed_numerator", "sum"),
                observed_weight=("_observed_weight", "sum"),
                no_cp_numerator=("_no_cp_numerator", "sum"),
                no_cp_weight=("_no_cp_weight", "sum"),
            )
        )

        zones["observed_level"] = (
            zones["observed_numerator"]
            / zones["observed_weight"].where(zones["observed_weight"].gt(0))
        )
        zones["no_cp_level"] = (
            zones["no_cp_numerator"]
            / zones["no_cp_weight"].where(zones["no_cp_weight"].gt(0))
        )

    zones["counterfactual_gap"] = (
        zones["no_cp_level"] - zones["observed_level"]
    )
    zones["counterfactual_gap_pct"] = np.where(
        zones["observed_level"].ne(0),
        100.0 * zones["counterfactual_gap"] / zones["observed_level"],
        np.nan,
    )
    zones["local_gap_intensity_pct"] = zones["counterfactual_gap_pct"].abs()
    zones["gap_direction"] = zones["counterfactual_gap"].map(direction_label)

    if metric in COUNT_METRICS:
        zones["system_contribution_native"] = zones["counterfactual_gap"]
    else:
        observed_weight_total = float(zones["observed_weight"].sum())
        no_cp_weight_total = float(zones["no_cp_weight"].sum())

        if observed_weight_total <= 0 or no_cp_weight_total <= 0:
            zones["system_contribution_native"] = np.nan
        else:
            zones["system_contribution_native"] = (
                zones["no_cp_numerator"] / no_cp_weight_total
                - zones["observed_numerator"] / observed_weight_total
            )

    zones["system_contribution_direction"] = zones[
        "system_contribution_native"
    ].map(direction_label)

    absolute_contribution = pd.to_numeric(
        zones["system_contribution_native"],
        errors="coerce",
    ).abs()
    total_absolute_contribution = float(absolute_contribution.sum())

    zones["system_contribution_share_pct"] = np.where(
        total_absolute_contribution > 0,
        100.0 * absolute_contribution / total_absolute_contribution,
        np.nan,
    )

    physical_rank = (
        zones.loc[
            zones["reader_facing_zone"].fillna(False)
            & zones["system_contribution_share_pct"].notna(),
            ["canonical_taxi_zone_id", "system_contribution_share_pct"],
        ]
        .sort_values(
            ["system_contribution_share_pct", "canonical_taxi_zone_id"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )
    physical_rank["system_contribution_rank"] = (
        np.arange(len(physical_rank)) + 1
    )

    zones = zones.merge(
        physical_rank[
            ["canonical_taxi_zone_id", "system_contribution_rank"]
        ],
        on="canonical_taxi_zone_id",
        how="left",
        validate="one_to_one",
    )

    return zones.sort_values("canonical_taxi_zone_id").reset_index(drop=True)


def system_summary(
    zones: pd.DataFrame,
    metric: str,
) -> dict[str, float]:
    """Return systemwide levels and gap from all analytical zones."""
    if zones.empty:
        return {
            "observed": np.nan,
            "no_cp": np.nan,
            "gap": np.nan,
            "gap_pct": np.nan,
        }

    if metric in COUNT_METRICS:
        observed = float(pd.to_numeric(zones["observed_level"], errors="coerce").sum())
        no_cp = float(pd.to_numeric(zones["no_cp_level"], errors="coerce").sum())
    else:
        observed_weight = float(
            pd.to_numeric(zones["observed_weight"], errors="coerce").sum()
        )
        no_cp_weight = float(
            pd.to_numeric(zones["no_cp_weight"], errors="coerce").sum()
        )
        observed = (
            float(pd.to_numeric(zones["observed_numerator"], errors="coerce").sum())
            / observed_weight
            if observed_weight > 0
            else np.nan
        )
        no_cp = (
            float(pd.to_numeric(zones["no_cp_numerator"], errors="coerce").sum())
            / no_cp_weight
            if no_cp_weight > 0
            else np.nan
        )

    gap = no_cp - observed if np.isfinite(observed) and np.isfinite(no_cp) else np.nan
    gap_pct = (
        100.0 * gap / observed
        if np.isfinite(gap) and observed != 0
        else np.nan
    )

    return {
        "observed": observed,
        "no_cp": no_cp,
        "gap": gap,
        "gap_pct": gap_pct,
    }


def physical_comparison_zones(zones: pd.DataFrame) -> pd.DataFrame:
    """Return mapped physical zones with both Observed and No-CP levels."""
    if zones.empty:
        return zones.copy()

    observed = pd.to_numeric(zones["observed_level"], errors="coerce")
    no_cp = pd.to_numeric(zones["no_cp_level"], errors="coerce")

    return zones.loc[
        zones["reader_facing_zone"].fillna(False)
        & np.isfinite(observed)
        & np.isfinite(no_cp)
    ].copy()


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------


def _robust_shared_level_bounds(
    zones: pd.DataFrame,
    metric: str,
) -> tuple[float, float]:
    """Return one robust shared scale for Observed and No-CP maps."""
    combined = pd.concat(
        [zones["observed_level"], zones["no_cp_level"]],
        ignore_index=True,
    )
    clean = pd.to_numeric(combined, errors="coerce").dropna()

    if clean.empty:
        return 0.0, 1.0

    if metric in COUNT_METRICS:
        low = 0.0
        high = float(clean.quantile(0.95))
    else:
        low = float(clean.quantile(0.05))
        high = float(clean.quantile(0.95))

    if not np.isfinite(low):
        low = float(clean.min())
    if not np.isfinite(high):
        high = float(clean.max())
    if high <= low:
        high = low + 1.0

    return low, high


def _robust_gap_bound(values: pd.Series) -> float:
    """Return a symmetric robust bound for the signed gap map."""
    clean = pd.to_numeric(values, errors="coerce").dropna().abs()
    if clean.empty:
        return 1.0

    bound = float(clean.quantile(0.95))
    if not np.isfinite(bound) or bound <= 0:
        bound = float(clean.max()) if len(clean) else 1.0
    return max(bound, 1e-9)


def _map_hover_table(
    zones: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Attach deterministic three-decimal strings for all map hovers."""
    result = zones.copy()
    result["hover_observed"] = result["observed_level"].map(format_hover_number)
    result["hover_no_cp"] = result["no_cp_level"].map(format_hover_number)
    result["hover_gap"] = result["counterfactual_gap"].map(
        lambda value: format_hover_number(value, signed=True)
    )
    result["hover_gap_pct"] = result["counterfactual_gap_pct"].map(
        lambda value: format_hover_number(value, signed=True, percent=True)
    )
    result["hover_contribution"] = result["system_contribution_share_pct"].map(
        lambda value: format_hover_number(value, percent=True)
    )
    result["hover_unit"] = native_unit(metric)
    return result


def build_zone_map(
    zones: pd.DataFrame,
    *,
    metric: str,
    value_column: str,
    title: str,
    selected_zone_id: int | None,
    shared_bounds: tuple[float, float],
    show_colorbar: bool,
    map_center: dict[str, float] | None = None,
    map_zoom: float = 8.8,
    height: int = 430,
) -> go.Figure:
    """Build one member of the linked Observed / No-CP / Gap triptych."""
    plot = _map_hover_table(zones, metric)

    if value_column == "counterfactual_gap":
        bound = _robust_gap_bound(plot[value_column])
        zmin = -bound
        zmax = bound
        zmid = 0.0
        colorscale = BRAND_DIVERGING_SEQUENCE

        unit = native_unit(metric)
        colorbar_title = (
            f"Gap<br>({unit})"
        )
    else:
        zmin, zmax = shared_bounds
        zmid = None
        colorscale = SEQUENTIAL_SCALE

        if metric in SPEED_METRICS:
            colorbar_title = (
                "Average<br>speed<br>(mph)"
            )
        else:
            colorbar_title = level_label(
                metric
            )

    trace_kwargs = {
        "geojson": get_zone_geojson(),
        "locations": plot["canonical_taxi_zone_id"].astype(str),
        "featureidkey": "properties.taxi_zone_id",
        "z": pd.to_numeric(plot[value_column], errors="coerce"),
        "zmin": zmin,
        "zmax": zmax,
        "colorscale": colorscale,
        "marker": {"line": {"width": 0.35, "color": "white"}},
        "customdata": plot[
            [
                "canonical_taxi_zone_id",
                "zone",
                "borough",
                "hover_observed",
                "hover_no_cp",
                "hover_gap",
                "hover_gap_pct",
                "hover_contribution",
            ]
        ].to_numpy(),
        "hovertemplate": (
            "<b>%{customdata[1]}</b> · %{customdata[2]}<br>"
            "Taxi Zone: %{customdata[0]}<br><br>"
            "Observed: %{customdata[3]}<br>"
            "Estimated no-CP: %{customdata[4]}<br>"
            "Gap (No-CP − observed): %{customdata[5]}<br>"
            "Local gap: %{customdata[6]}<br>"
            "Share of system divergence: %{customdata[7]}"
            "<extra></extra>"
        ),
        "showscale": show_colorbar,
    }

    if zmid is not None:
        trace_kwargs["zmid"] = zmid

    if show_colorbar:
        trace_kwargs["colorbar"] = {
            "title": {
                "text": colorbar_title,
                "font": {
                    "size": 11,
                },
                "side": "top",
            },
            "tickfont": {
                "size": 10,
            },
            "x": 1.01,
            "xpad": 2,
            "len": 0.66,
            "thickness": 8,
            "outlinewidth": 0,
        }

    figure = go.Figure(go.Choroplethmap(**trace_kwargs))

    if selected_zone_id is not None and selected_zone_id in set(
        plot["canonical_taxi_zone_id"].dropna().astype(int)
    ):
        figure.add_trace(
            go.Choroplethmap(
                geojson=get_zone_geojson(),
                locations=[str(selected_zone_id)],
                featureidkey="properties.taxi_zone_id",
                z=[0],
                zmin=0,
                zmax=1,
                colorscale=[
                    [0.0, "rgba(255,255,255,0.02)"],
                    [1.0, "rgba(255,255,255,0.02)"],
                ],
                showscale=False,
                marker={
                    "line": {
                        "width": 3.0,
                        "color": "#003F46",
                    }
                },
                customdata=[[selected_zone_id]],
                hoverinfo="skip",
            )
        )

    figure = apply_branding(figure)
    figure.update_layout(
        title={"text": title, "font": {"size": 16}},
        map={
            "style": "carto-positron",
            "center": map_center or MAP_CENTER,
            "zoom": map_zoom,
        },
        height=height,
        margin={
            "l": 0,
            "r": 46 if show_colorbar else 0,
            "t": 48,
            "b": 0,
        },
        hovermode="closest",
        showlegend=False,
    )
    return figure


def build_intensity_contribution_plane(
    zones: pd.DataFrame,
    *,
    metric: str,
    title: str,
    selected_zone_id: int | None,
) -> go.Figure:
    """Plot local percentage intensity against share of system divergence."""
    plot = zones.loc[
        zones["local_gap_intensity_pct"].notna()
        & zones["system_contribution_share_pct"].notna()
        & zones["local_gap_intensity_pct"].ge(0)
        & zones["system_contribution_share_pct"].ge(0)
    ].copy()

    if plot.empty:
        return go.Figure()

    # A tiny floor lets true zero-intensity zones remain on the logarithmic x-axis.
    positive_intensity = plot.loc[
        plot["local_gap_intensity_pct"].gt(0),
        "local_gap_intensity_pct",
    ]
    floor = (
        max(float(positive_intensity.min()) / 2.0, 0.001)
        if not positive_intensity.empty
        else 0.001
    )
    plot["plot_intensity"] = plot["local_gap_intensity_pct"].clip(lower=floor)

    x_median = float(plot["plot_intensity"].median())
    y_median = float(plot["system_contribution_share_pct"].median())

    plot["hover_observed"] = plot["observed_level"].map(format_hover_number)
    plot["hover_no_cp"] = plot["no_cp_level"].map(format_hover_number)
    plot["hover_gap"] = plot["counterfactual_gap"].map(
        lambda value: format_hover_number(value, signed=True)
    )
    plot["hover_gap_pct"] = plot["counterfactual_gap_pct"].map(
        lambda value: format_hover_number(value, signed=True, percent=True)
    )
    plot["hover_intensity"] = plot["local_gap_intensity_pct"].map(
        lambda value: format_hover_number(value, percent=True)
    )
    plot["hover_contribution"] = plot["system_contribution_share_pct"].map(
        lambda value: format_hover_number(value, percent=True)
    )

    figure = go.Figure()

    direction_order = [
        "No-CP below observed",
        "No-CP above observed",
        "No difference",
    ]

    for direction in direction_order:
        part = plot.loc[plot["gap_direction"].eq(direction)]
        if part.empty:
            continue

        figure.add_trace(
            go.Scatter(
                x=part["plot_intensity"],
                y=part["system_contribution_share_pct"],
                mode="markers",
                name=direction,
                marker={
                    "size": 9,
                    "opacity": 0.72,
                    "color": DIRECTION_COLORS[direction],
                    "line": {"width": 0.5, "color": "white"},
                },
                customdata=part[
                    [
                        "canonical_taxi_zone_id",
                        "zone",
                        "borough",
                        "hover_observed",
                        "hover_no_cp",
                        "hover_gap",
                        "hover_gap_pct",
                        "hover_intensity",
                        "hover_contribution",
                    ]
                ].to_numpy(),
                hovertemplate=(
                    "<b>%{customdata[1]}</b> · %{customdata[2]}<br>"
                    "Taxi Zone: %{customdata[0]}<br><br>"
                    "Observed: %{customdata[3]}<br>"
                    "Estimated no-CP: %{customdata[4]}<br>"
                    "Gap (No-CP − observed): %{customdata[5]}<br>"
                    "Local gap: %{customdata[6]}<br>"
                    "Local intensity: %{customdata[7]}<br>"
                    "Share of system divergence: %{customdata[8]}"
                    "<extra></extra>"
                ),
            )
        )

    if selected_zone_id is not None:
        selected = plot.loc[
            plot["canonical_taxi_zone_id"].eq(int(selected_zone_id))
        ]
        if not selected.empty:
            figure.add_trace(
                go.Scatter(
                    x=selected["plot_intensity"],
                    y=selected["system_contribution_share_pct"],
                    mode="markers",
                    name="Selected Taxi Zone",
                    marker={
                        "size": 18,
                        "color": "rgba(255,255,255,0.02)",
                        "line": {
                            "width": 3.0,
                            "color": "#003F46",
                        },
                    },
                    customdata=selected[
                        [
                            "canonical_taxi_zone_id",
                            "zone",
                            "borough",
                        ]
                    ].to_numpy(),
                    hovertemplate=(
                        "<b>%{customdata[1]}</b> · %{customdata[2]}<br>"
                        "Taxi Zone: %{customdata[0]}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

    figure.add_vline(
        x=x_median,
        line_dash="dot",
        line_width=1,
        line_color="rgba(0, 109, 119, 0.35)",
    )
    figure.add_hline(
        y=y_median,
        line_dash="dot",
        line_width=1,
        line_color="rgba(0, 109, 119, 0.35)",
    )

    quadrant_annotations = [
        (0.02, 0.96, "Modest local · larger system"),
        (0.98, 0.96, "Large local · larger system"),
        (0.02, 0.04, "Modest local · smaller system"),
        (0.98, 0.04, "Large local · smaller system"),
    ]
    for x, y, text in quadrant_annotations:
        figure.add_annotation(
            x=x,
            y=y,
            xref="paper",
            yref="paper",
            text=text,
            showarrow=False,
            xanchor="left" if x < 0.5 else "right",
            yanchor="top" if y > 0.5 else "bottom",
            font={"size": 10, "color": "#335C67"},
            bgcolor="rgba(255,255,255,0.72)",
            borderpad=3,
        )

    figure = apply_branding(figure)
    figure.update_layout(
        title={"text": title, "font": {"size": 17}},
        height=520,
        margin={"l": 65, "r": 20, "t": 65, "b": 70},
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.02,
            "yanchor": "bottom",
            "title": {"text": ""},
        },
    )
    figure.update_xaxes(
        type="log",
        title_text="Local gap intensity (%) · log scale",
        gridcolor="rgba(0, 109, 119, 0.14)",
    )
    figure.update_yaxes(
        title_text="Share of absolute system divergence (%)",
        rangemode="tozero",
        ticksuffix="%",
        gridcolor="rgba(0, 109, 119, 0.14)",
    )
    return figure


def _zone_id_from_plotly_event(event: object) -> int | None:
    """Return the selected Taxi Zone from either a map or scatter event."""
    try:
        points = event.selection.points
    except (AttributeError, TypeError):
        return None

    if not points:
        return None

    point = points[0]
    if not isinstance(point, dict):
        return None

    customdata = point.get("customdata")
    if customdata is not None and len(customdata) > 0:
        try:
            return int(customdata[0])
        except (TypeError, ValueError):
            pass

    location = point.get("location")
    if location is not None:
        try:
            return int(location)
        except (TypeError, ValueError):
            pass

    return None


# ---------------------------------------------------------------------
# Narrative, rankings, and selected-zone profile
# ---------------------------------------------------------------------


def _rank_correlation(zones: pd.DataFrame) -> float:
    """Return Spearman-like rank correlation between local and system magnitude."""
    usable = zones[
        ["local_gap_intensity_pct", "system_contribution_share_pct"]
    ].replace([np.inf, -np.inf], np.nan).dropna()
    if len(usable) < 3:
        return np.nan

    return float(
        usable["local_gap_intensity_pct"].rank().corr(
            usable["system_contribution_share_pct"].rank()
        )
    )


def spatial_takeaway(
    zones: pd.DataFrame,
    *,
    metric: str,
) -> str:
    """Explain the observed-vs-no-CP geography without requiring sign decoding."""
    summary = system_summary(zones, metric)

    physical = zones.loc[
        zones["reader_facing_zone"].fillna(False)
    ].copy()

    contributors = physical.loc[
        physical["system_contribution_share_pct"].notna()
    ].sort_values(
        ["system_contribution_share_pct", "canonical_taxi_zone_id"],
        ascending=[False, True],
    )

    local = physical.loc[
        physical["local_gap_intensity_pct"].notna()
    ].sort_values(
        ["local_gap_intensity_pct", "canonical_taxi_zone_id"],
        ascending=[False, True],
    )

    if contributors.empty:
        return "No supported physical Taxi Zones are available for this view."

    leader = contributors.iloc[0]

    top10_share = float(
        contributors.head(10)["system_contribution_share_pct"].sum()
    )

    # WHY: describe what NYC actually experienced first, then compare it
    # with the model's estimated world without congestion pricing.
    if np.isfinite(summary["gap_pct"]):
        gap_pct = float(summary["gap_pct"])

        if gap_pct < 0:
            system_sentence = (
                f"Observed **{metric_label(metric)}** was about "
                f"**{abs(gap_pct):.1f}% higher** than the model estimates it "
                "would have been if congestion pricing had not begun. "
            )
        elif gap_pct > 0:
            system_sentence = (
                f"Observed **{metric_label(metric)}** was about "
                f"**{abs(gap_pct):.1f}% lower** than the model estimates it "
                "would have been if congestion pricing had not begun. "
            )
        else:
            system_sentence = (
                f"Observed **{metric_label(metric)}** was essentially equal to "
                "the model's estimated no-congestion-pricing level. "
            )
    else:
        system_sentence = (
            "The systemwide observed-versus-no-CP percentage difference is "
            "unavailable for this selection. "
        )

    contribution_sentence = (
        f"**{leader['zone']}** contributes the largest share of the total "
        f"geographic difference at **{float(leader['system_contribution_share_pct']):.1f}%**, "
        f"while the top 10 zones together account for **{top10_share:.1f}%**. "
    )

    if local.empty:
        return system_sentence + contribution_sentence

    local_leader = local.iloc[0]

    if int(local_leader["canonical_taxi_zone_id"]) == int(
        leader["canonical_taxi_zone_id"]
    ):
        contrast_sentence = (
            "The same zone also has the largest percentage difference locally, "
            "so the strongest local contrast and the largest contribution to the "
            "citywide difference point to the same place."
        )
    else:
        local_gap = float(local_leader["counterfactual_gap_pct"])

        if local_gap < 0:
            local_direction = (
                f"observed {metric_label(metric)} was "
                f"{abs(local_gap):.1f}% higher than its estimated no-CP level"
            )
        elif local_gap > 0:
            local_direction = (
                f"observed {metric_label(metric)} was "
                f"{abs(local_gap):.1f}% lower than its estimated no-CP level"
            )
        else:
            local_direction = (
                f"observed {metric_label(metric)} was essentially equal to its "
                "estimated no-CP level"
            )

        contrast_sentence = (
            f"**{local_leader['zone']}** has the largest percentage difference "
            f"locally: {local_direction}. But it accounts for only "
            f"**{float(local_leader['system_contribution_share_pct']):.3f}%** "
            "of the total geographic difference. A place can therefore look "
            "dramatic locally without contributing much to the citywide result."
        )

    return system_sentence + contribution_sentence + contrast_sentence


def selected_zone_default(zones: pd.DataFrame) -> int:
    """Choose the largest physical system contributor as a stable default."""
    physical = zones.loc[
        zones["reader_facing_zone"].fillna(False)
        & zones["system_contribution_share_pct"].notna()
    ].copy()
    if physical.empty:
        raise RuntimeError("No reader-facing Taxi Zones are available.")

    row = physical.sort_values(
        ["system_contribution_share_pct", "canonical_taxi_zone_id"],
        ascending=[False, True],
    ).iloc[0]
    return int(row["canonical_taxi_zone_id"])


def zone_option_data(
    zones: pd.DataFrame,
) -> tuple[list[int], dict[int, str]]:
    """Return stable dropdown IDs and Zone · Borough labels."""
    physical = physical_comparison_zones(zones)
    options = (
        physical[
            ["canonical_taxi_zone_id", "zone", "borough"]
        ]
        .drop_duplicates("canonical_taxi_zone_id")
        .sort_values(["zone", "borough", "canonical_taxi_zone_id"])
    )

    ids = options["canonical_taxi_zone_id"].astype(int).tolist()
    labels = {
        int(row.canonical_taxi_zone_id): f"{row.zone} · {row.borough}"
        for row in options.itertuples(index=False)
    }
    return ids, labels


def _prepare_zone_state(
    zones: pd.DataFrame,
    *,
    selector_key: str,
    pending_key: str,
    reset_key: str | None = None,
) -> tuple[list[int], dict[int, str]]:
    """Keep selection coherent when filters change or a chart point is clicked."""
    zone_ids, labels = zone_option_data(zones)
    if not zone_ids:
        return zone_ids, labels

    supported_defaults = zones.loc[
        zones["canonical_taxi_zone_id"].isin(zone_ids)
    ].copy()

    # A story or filter change starts the reader at the most consequential
    # physical Taxi Zone for that newly selected slice. Manual clicks made
    # afterward remain sticky until the next filter change.
    if reset_key is not None and st.session_state.pop(reset_key, False):
        st.session_state[selector_key] = selected_zone_default(
            supported_defaults
        )

    pending = st.session_state.pop(pending_key, None)
    if pending in zone_ids:
        st.session_state[selector_key] = int(pending)

    if st.session_state.get(selector_key) not in zone_ids:
        st.session_state[selector_key] = selected_zone_default(
            supported_defaults
        )

    return zone_ids, labels


def selected_zone_quadrant(
    zones: pd.DataFrame,
    selected: pd.Series,
) -> str:
    """Describe the selected point relative to current median guide lines."""
    plane = zones.loc[
        zones["local_gap_intensity_pct"].notna()
        & zones["system_contribution_share_pct"].notna()
    ]
    if plane.empty:
        return "Plane position unavailable"

    x_value = float(selected.get("local_gap_intensity_pct", np.nan))
    y_value = float(selected.get("system_contribution_share_pct", np.nan))
    if not np.isfinite(x_value) or not np.isfinite(y_value):
        return "Plane position unavailable"

    x_high = x_value >= float(plane["local_gap_intensity_pct"].median())
    y_high = y_value >= float(plane["system_contribution_share_pct"].median())

    if x_high and y_high:
        return "Large local gap · larger system contribution"
    if x_high and not y_high:
        return "Large local gap · smaller system contribution"
    if not x_high and y_high:
        return "Modest local gap · larger system contribution"
    return "Modest local gap · smaller system contribution"


def render_selected_zone_profile(
    zones: pd.DataFrame,
    *,
    metric: str,
    selected_zone_id: int,
) -> None:
    """Render a compact single-metric profile for the currently selected zone."""
    match = zones.loc[
        zones["canonical_taxi_zone_id"].eq(int(selected_zone_id))
    ]
    if match.empty:
        st.info("The selected Taxi Zone is unavailable for this view.")
        return

    row = match.iloc[0]
    contribution_rank = row.get("system_contribution_rank", np.nan)
    physical_count = int(
        zones.loc[zones["reader_facing_zone"].fillna(False), "canonical_taxi_zone_id"]
        .nunique()
    )

    rank_suffix = (
        " · #1 system contributor"
        if pd.notna(contribution_rank) and int(contribution_rank) == 1
        else ""
    )
    st.markdown(f"### {row['zone']}{rank_suffix}")
    st.caption(
        f"{row['borough']} · Taxi Zone {int(row['canonical_taxi_zone_id'])} · "
        f"{policy_label(row['cbd_spatial_category'])}"
    )

    c1, c2 = st.columns(2)
    c1.metric(
        "Observed",
        format_level(float(row["observed_level"]), metric),
    )
    c2.metric(
        "Estimated no-CP",
        format_level(float(row["no_cp_level"]), metric),
    )

    c3, c4 = st.columns(2)
    c3.metric(
        "Gap · No-CP − observed",
        format_level(float(row["counterfactual_gap"]), metric, signed=True),
    )
    c4.metric(
        "Local gap",
        (
            f"{float(row['counterfactual_gap_pct']):+.3f}%"
            if np.isfinite(float(row["counterfactual_gap_pct"]))
            else "Not available"
        ),
    )

    share = float(row.get("system_contribution_share_pct", np.nan))
    if np.isfinite(share):
        st.metric(
            "Share of system divergence",
            f"{share:.3f}%",
        )

    if pd.notna(contribution_rank):
        st.caption(
            f"Contribution rank: **#{int(contribution_rank)} of {physical_count}** "
            "physical Taxi Zones in this view."
        )

    st.caption(selected_zone_quadrant(zones, row))


def build_ranking_table(
    zones: pd.DataFrame,
    *,
    metric: str,
    ranking_mode: str,
    limit: int,
) -> pd.DataFrame:
    """Return the requested physical-zone ranking for the current filters."""
    physical = zones.loc[zones["reader_facing_zone"].fillna(False)].copy()

    if ranking_mode == "System divergence contribution":
        physical = physical.loc[
            physical["system_contribution_share_pct"].notna()
        ].sort_values(
            ["system_contribution_share_pct", "canonical_taxi_zone_id"],
            ascending=[False, True],
        )
    elif ranking_mode == "Local gap intensity":
        physical = physical.loc[
            physical["local_gap_intensity_pct"].notna()
        ].sort_values(
            ["local_gap_intensity_pct", "canonical_taxi_zone_id"],
            ascending=[False, True],
        )
    else:
        raise ValueError(f"Unsupported ranking mode: {ranking_mode}")

    physical = physical.head(limit).reset_index(drop=True)
    physical.insert(0, "Rank", np.arange(len(physical)) + 1)

    result = pd.DataFrame(
        {
            "Rank": physical["Rank"],
            "Taxi Zone": physical["zone"].astype(str),
            "Borough": physical["borough"].astype(str),
            "System divergence share": physical["system_contribution_share_pct"].map(
                lambda value: (
                    f"{float(value):.3f}%" if np.isfinite(value) else "—"
                )
            ),
            "Contribution to system gap": physical["system_contribution_native"].map(
                lambda value: (
                    format_hover_number(float(value), signed=True)
                    if np.isfinite(value)
                    else "—"
                )
            ),
            "Local gap": physical["counterfactual_gap_pct"].map(
                lambda value: (
                    f"{float(value):+.3f}%" if np.isfinite(value) else "—"
                )
            ),
            "Observed": physical["observed_level"].map(
                lambda value: format_hover_number(float(value))
            ),
            "Estimated no-CP": physical["no_cp_level"].map(
                lambda value: format_hover_number(float(value))
            ),
        }
    )

    return result


# ---------------------------------------------------------------------
# Linked spatial bundle
# ---------------------------------------------------------------------


def render_linked_spatial_bundle(
    zones: pd.DataFrame,
    *,
    metric: str,
    horizon: int,
    scope_label: str,
    key_prefix: str,
    show_ranking: bool,
    reset_key: str | None = None,
) -> None:
    """Render the maps, plane, selected-zone profile, and optional ranking."""
    comparison = physical_comparison_zones(zones)
    if comparison.empty:
        st.info("No physical Taxi Zones have both Observed and No-CP values here.")
        return

    selector_key = f"{key_prefix}_zone_selector"
    pending_key = f"{key_prefix}_pending_zone_id"
    zone_ids, zone_labels = _prepare_zone_state(
        zones,
        selector_key=selector_key,
        pending_key=pending_key,
        reset_key=reset_key,
    )

    if not zone_ids:
        st.info("No reader-facing Taxi Zones are available for this view.")
        return

    selected_zone_id = int(
        st.selectbox(
            "Selected Taxi Zone",
            options=zone_ids,
            key=selector_key,
            format_func=lambda value: zone_labels.get(int(value), str(value)),
        )
    )

    selected_match = zones.loc[
        zones["canonical_taxi_zone_id"].eq(selected_zone_id)
    ]
    if not selected_match.empty:
        selected_rank = selected_match.iloc[0].get(
            "system_contribution_rank", np.nan
        )
        if pd.notna(selected_rank) and int(selected_rank) == 1:
            st.caption(
                "Start here: this view opens on the #1 physical Taxi Zone "
                "contributor to system divergence. Click any map zone or point "
                "below to inspect another place."
            )
        else:
            st.caption(
                "Click a Taxi Zone on any map or point in the plane to carry "
                "the same selection across the linked views."
            )

    shared_bounds = _robust_shared_level_bounds(comparison, metric)

    observed_map = build_zone_map(
        comparison,
        metric=metric,
        value_column="observed_level",
        title="Observed",
        selected_zone_id=selected_zone_id,
        shared_bounds=shared_bounds,
        show_colorbar=False,
    )
    no_cp_map = build_zone_map(
        comparison,
        metric=metric,
        value_column="no_cp_level",
        title="Estimated no-CP · shared scale",
        selected_zone_id=selected_zone_id,
        shared_bounds=shared_bounds,
        show_colorbar=True,
    )
    gap_map = build_zone_map(
        comparison,
        metric=metric,
        value_column="counterfactual_gap",
        title="Gap · No-CP − observed",
        selected_zone_id=selected_zone_id,
        shared_bounds=shared_bounds,
        show_colorbar=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        observed_event = st.plotly_chart(
            observed_map,
            width="stretch",
            config=PLOT_CONFIG,
            key=f"{key_prefix}_observed_map",
            on_select="rerun",
            selection_mode="points",
        )
    with c2:
        no_cp_event = st.plotly_chart(
            no_cp_map,
            width="stretch",
            config=PLOT_CONFIG,
            key=f"{key_prefix}_no_cp_map",
            on_select="rerun",
            selection_mode="points",
        )
    with c3:
        gap_event = st.plotly_chart(
            gap_map,
            width="stretch",
            config=PLOT_CONFIG,
            key=f"{key_prefix}_gap_map",
            on_select="rerun",
            selection_mode="points",
        )

    st.caption(
        f"{scope_label}. Observed and no-CP use one shared robust scale; the Gap "
        "map uses a separate zero-centered robust scale. Hover shows underlying "
        "values rounded to three decimals."
    )

    render_chart_insight(
        spatial_takeaway(
            zones,
            metric=metric,
        )
    )

    st.markdown("### Local intensity × system contribution")
    st.write(
        "The horizontal axis asks how dramatic the gap is inside each Taxi Zone. "
        "The vertical axis asks how much that zone contributes to the selected "
        "systemwide divergence. Median guide lines are orientation aids, not model "
        "thresholds."
    )

    left, right = st.columns([1.75, 1.0])
    with left:
        plane = build_intensity_contribution_plane(
            comparison,
            metric=metric,
            title=f"Local vs system importance · {metric_label(metric)} · h={horizon}",
            selected_zone_id=selected_zone_id,
        )
        plane_event = st.plotly_chart(
            plane,
            width="stretch",
            config=PLOT_CONFIG,
            key=f"{key_prefix}_plane",
            on_select="rerun",
            selection_mode="points",
        )
        st.caption(
            "Local intensity uses the absolute percentage gap and a log x-axis so "
            "small-denominator extremes do not flatten the rest of the city. Dot "
            "color preserves the signed No-CP − observed direction."
        )

    with right:
        render_selected_zone_profile(
            zones,
            metric=metric,
            selected_zone_id=selected_zone_id,
        )

    clicked_candidates = [
        _zone_id_from_plotly_event(observed_event),
        _zone_id_from_plotly_event(no_cp_event),
        _zone_id_from_plotly_event(gap_event),
        _zone_id_from_plotly_event(plane_event),
    ]
    clicked_id = next(
        (
            value
            for value in reversed(clicked_candidates)
            if value in zone_ids and value != selected_zone_id
        ),
        None,
    )

    if clicked_id is not None:
        st.session_state[pending_key] = int(clicked_id)
        st.rerun()

    if show_ranking:
        st.markdown("### Top Taxi Zone contributors to system divergence")
        st.write(
            "The default ranking answers: which physical Taxi Zones account for "
            "the largest share of the selected absolute system divergence? Switch "
            "to local gap intensity to see why the biggest percentages are not "
            "always the biggest system contributors."
        )

        r1, r2 = st.columns([1.6, 0.8])
        with r1:
            ranking_mode = st.segmented_control(
                "Rank by",
                options=[
                    "System divergence contribution",
                    "Local gap intensity",
                ],
                default="System divergence contribution",
                key=f"{key_prefix}_ranking_mode",
            ) or "System divergence contribution"
        with r2:
            ranking_limit = int(
                st.selectbox(
                    "Rows shown",
                    options=[10, 20, 50],
                    index=0,
                    key=f"{key_prefix}_ranking_limit",
                )
            )

        ranking = build_ranking_table(
            zones,
            metric=metric,
            ranking_mode=ranking_mode,
            limit=ranking_limit,
        )

        st.dataframe(
            ranking,
            hide_index=True,
            width="stretch",
            height=min(620, 38 * (len(ranking) + 1)),
        )

        if ranking_mode == "System divergence contribution":
            st.caption(
                "System divergence share uses each zone's absolute contribution to "
                "the selected system gap. For average speed, contribution comes "
                "from the exact world-specific weighted-average decomposition—not "
                "from raw zone speed-gap magnitude."
            )
        else:
            st.caption(
                "Local intensity is the absolute percentage gap inside the Taxi "
                "Zone. Large values can occur in low-activity places, so this "
                "ordering should be read alongside system contribution."
            )


# ---------------------------------------------------------------------
# Explorer state
# ---------------------------------------------------------------------


def _initialize_explorer_state() -> None:
    """Set the default saved story once without overwriting reader changes."""
    defaults = {
        # Start the explorer on a different story from the Taxi-trip hero so
        # the interactive section immediately demonstrates a new spatial use case.
        "raw22_story": "Taxi speed · weekday PM",
        "raw22_metric": "taxi_avg_trip_speed",
        "raw22_horizon": 1,
        "raw22_period": "Full post-CP period",
        "raw22_day_type": "Weekdays",
        "raw22_daypart": "PM peak",
        "raw22_last_story": None,
        "raw22_explorer_reset_to_top": True,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _mark_story_custom() -> None:
    """A direct filter change leaves the story and resets to its top contributor."""
    st.session_state["raw22_story"] = "Custom view"
    st.session_state["raw22_explorer_reset_to_top"] = True


def _apply_story_if_changed() -> None:
    """Apply one saved story before its downstream widgets are instantiated."""
    story = st.session_state.get("raw22_story", "Custom view")
    previous = st.session_state.get("raw22_last_story")

    if story == previous:
        return

    config = SAVED_STORIES.get(story)
    if config is not None:
        st.session_state["raw22_metric"] = config["metric"]
        st.session_state["raw22_horizon"] = config["horizon"]
        st.session_state["raw22_period"] = config["period"]
        st.session_state["raw22_day_type"] = config["day_type"]
        st.session_state["raw22_daypart"] = config["daypart"]

    st.session_state["raw22_last_story"] = story
    st.session_state["raw22_explorer_reset_to_top"] = True


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------

inject_app_css()

st.caption("COUNTERFACTUAL GEOGRAPHY")
st.title("Where did observed mobility differ from the estimated no-CP path?")
st.write(
    "Compare observed mobility with the path estimated by our frozen no-CP "
    "forecast, then separate two questions that are easy to confuse: where the "
    "percentage gap looks dramatic, and which Taxi Zones actually account for "
    "more of the systemwide divergence."
)

try:
    geography_df = load_inputs()
except Exception as exc:
    st.error(
        "Raw 22 could not load its validated spatial contract. "
        f"Details: {exc}"
    )
    st.stop()


# ---------------------------------------------------------------------
# Editorial hero
# ---------------------------------------------------------------------

st.header("Where did observed and estimated no-CP Taxi trips differ?")

hero_zones = aggregate_zone_slice(
    geography_df,
    metric=HERO_METRIC,
    horizon=HERO_HORIZON,
    period=HERO_PERIOD,
    day_type=HERO_DAY_TYPE,
    daypart=HERO_DAYPART,
)

hero_summary = system_summary(
    hero_zones,
    HERO_METRIC,
)
hero_physical = physical_comparison_zones(
    hero_zones
)
hero_ranked = hero_physical.loc[
    hero_physical["system_contribution_share_pct"].notna()
].sort_values(
    ["system_contribution_share_pct", "canonical_taxi_zone_id"],
    ascending=[False, True],
)
hero_local_ranked = hero_physical.loc[
    hero_physical["local_gap_intensity_pct"].notna()
].sort_values(
    ["local_gap_intensity_pct", "canonical_taxi_zone_id"],
    ascending=[False, True],
)

st.markdown(
    "The fixed opening view compares **Taxi trips · h=1 · full post-CP period · "
    "all days and dayparts**. The first two maps use the same scale for observed "
    "and estimated no-CP levels; the third maps **Gap = no-CP − observed**. The "
    "view is zoomed toward Manhattan so dense Taxi Zone differences remain legible."
)

st.markdown("#### Read the triptych in three steps")
read_1, read_2, read_3 = st.columns(3)
with read_1:
    st.markdown("**1 · Compare the first two maps**")
    st.caption(
        "Observed and estimated no-CP use the same color scale, so the shading "
        "is directly comparable."
    )
with read_2:
    st.markdown("**2 · Read the signed Gap**")
    st.caption(
        "Gap = no-CP − observed. Teal means the no-CP estimate is lower; "
        "terracotta means it is higher."
    )
with read_3:
    st.markdown("**3 · Then ask what matters systemwide**")
    st.caption(
        "A dramatic local percentage can come from a tiny denominator. The "
        "interactive section below separates local intensity from system contribution."
    )

hero_shared_bounds = _robust_shared_level_bounds(
    hero_physical,
    HERO_METRIC,
)
hero_map_center = {
    "lat": 40.7580,
    "lon": -73.9855,
}

hero_observed = build_zone_map(
    hero_physical,
    metric=HERO_METRIC,
    value_column="observed_level",
    title="Observed Taxi trips",
    selected_zone_id=None,
    shared_bounds=hero_shared_bounds,
    show_colorbar=False,
    map_center=hero_map_center,
    map_zoom=10.15,
    height=450,
)
hero_no_cp = build_zone_map(
    hero_physical,
    metric=HERO_METRIC,
    value_column="no_cp_level",
    title="Estimated no-CP · same scale",
    selected_zone_id=None,
    shared_bounds=hero_shared_bounds,
    show_colorbar=True,
    map_center=hero_map_center,
    map_zoom=10.15,
    height=450,
)
hero_gap = build_zone_map(
    hero_physical,
    metric=HERO_METRIC,
    value_column="counterfactual_gap",
    title="Gap · no-CP − observed",
    selected_zone_id=None,
    shared_bounds=hero_shared_bounds,
    show_colorbar=True,
    map_center=hero_map_center,
    map_zoom=10.15,
    height=450,
)

hero_c1, hero_c2, hero_c3 = st.columns(3)
with hero_c1:
    st.plotly_chart(
        hero_observed,
        width="stretch",
        config=PLOT_CONFIG,
        key="raw22_hero_observed",
    )
with hero_c2:
    st.plotly_chart(
        hero_no_cp,
        width="stretch",
        config=PLOT_CONFIG,
        key="raw22_hero_no_cp",
    )
with hero_c3:
    st.plotly_chart(
        hero_gap,
        width="stretch",
        config=PLOT_CONFIG,
        key="raw22_hero_gap",
    )

st.caption(
    f"{metric_label(HERO_METRIC)} · h={HERO_HORIZON} · "
    f"{period_caption(geography_df, HERO_PERIOD)} · all days · all dayparts. "
    "The first two maps share one robust scale; the Gap map is zero-centered. "
    "Hover shows the underlying values to three decimals."
)

render_chart_insight(
    spatial_takeaway(
        hero_zones,
        metric=HERO_METRIC,
    )
)

st.markdown("### Four Taxi Zones that explain the page")
st.write(
    "Three cards show the largest contributors to the Taxi-trip system divergence. "
    "The fourth deliberately shows the biggest local percentage gap, which can tell "
    "a very different story."
)

spotlight_rows: list[tuple[str, pd.Series]] = []
for position, (_, row) in enumerate(hero_ranked.head(3).iterrows(), start=1):
    spotlight_rows.append((f"System contributor #{position}", row))

if not hero_local_ranked.empty:
    local_row = hero_local_ranked.iloc[0]
    spotlight_rows.append(("Largest local percentage gap", local_row))

spotlight_columns = st.columns(max(len(spotlight_rows), 1))
for column, (role, row) in zip(spotlight_columns, spotlight_rows):
    with column:
        with st.container(border=True):
            st.caption(role.upper())
            st.markdown(f"**{row['zone']}**")
            st.caption(f"{row['borough']}")
            st.metric(
                "Local gap",
                (
                    f"{float(row['counterfactual_gap_pct']):+.3f}%"
                    if np.isfinite(float(row['counterfactual_gap_pct']))
                    else "Not available"
                ),
            )
            st.metric(
                "System divergence share",
                (
                    f"{float(row['system_contribution_share_pct']):.3f}%"
                    if np.isfinite(float(row['system_contribution_share_pct']))
                    else "Not available"
                ),
            )
            st.caption(
                "Native gap: "
                + format_level(
                    float(row["counterfactual_gap"]),
                    HERO_METRIC,
                    signed=True,
                )
                + f" {native_unit(HERO_METRIC)}"
            )

# ---------------------------------------------------------------------
# Reader-controlled explorer
# ---------------------------------------------------------------------

_initialize_explorer_state()

with exploration_section(
    key="raw22_exploration_area",
    title="Explore another spatial counterfactual",
    description=(
        "Start with a guided story or build your own view. Whenever the story "
        "or filters change, the selected Taxi Zone resets to the #1 physical "
        "contributor to system divergence for that exact slice. You can then "
        "click any other zone to inspect it."
    ),
):
    control_1, control_2, control_3 = st.columns([1.45, 1.25, 0.70])

    with control_1:
        st.selectbox(
            "Start with a story",
            options=list(SAVED_STORIES),
            key="raw22_story",
        )

    _apply_story_if_changed()

    with control_2:
        selected_metric = st.selectbox(
            "Mobility measure",
            options=METRIC_ORDER,
            key="raw22_metric",
            format_func=metric_label,
            on_change=_mark_story_custom,
        )

    with control_3:
        selected_horizon = int(
            st.selectbox(
                "Horizon",
                options=HORIZONS,
                key="raw22_horizon",
                format_func=lambda value: f"h={value}",
                on_change=_mark_story_custom,
            )
        )

    control_4, control_5, control_6 = st.columns([1.45, 1.0, 1.0])

    with control_4:
        selected_period = st.selectbox(
            "Period",
            options=PERIOD_OPTIONS,
            key="raw22_period",
            on_change=_mark_story_custom,
        )

    with control_5:
        selected_day_type = st.selectbox(
            "Day type",
            options=DAY_TYPES,
            key="raw22_day_type",
            on_change=_mark_story_custom,
        )

    with control_6:
        selected_daypart = st.selectbox(
            "Daypart",
            options=DAYPARTS,
            key="raw22_daypart",
            on_change=_mark_story_custom,
        )

    explorer_zones = aggregate_zone_slice(
        geography_df,
        metric=selected_metric,
        horizon=selected_horizon,
        period=selected_period,
        day_type=selected_day_type,
        daypart=selected_daypart,
    )

    if explorer_zones.empty:
        st.info("No supported rows are available for this filter combination.")
    else:
        current_summary = system_summary(
            explorer_zones,
            selected_metric,
        )
        physical_ranked = explorer_zones.loc[
            explorer_zones["reader_facing_zone"].fillna(False)
            & explorer_zones["system_contribution_share_pct"].notna()
        ].sort_values("system_contribution_share_pct", ascending=False)

        top10_share = (
            float(physical_ranked.head(10)["system_contribution_share_pct"].sum())
            if not physical_ranked.empty
            else np.nan
        )
        physical_count = int(
            explorer_zones.loc[
                explorer_zones["reader_facing_zone"].fillna(False),
                "canonical_taxi_zone_id",
            ].nunique()
        )

        k1, k2, k3 = st.columns(3)
        k1.metric(
            "Systemwide gap",
            (
                f"{current_summary['gap_pct']:+.3f}%"
                if np.isfinite(current_summary["gap_pct"])
                else "Not available"
            ),
            help="Estimated no-CP minus observed, divided by observed.",
        )
        k2.metric(
            "Top 10 divergence share",
            f"{top10_share:.3f}%" if np.isfinite(top10_share) else "Not available",
            help=(
                "Share of absolute system divergence carried by the ten largest "
                "physical Taxi Zone contributors in this selection."
            ),
        )
        k3.metric(
            "Physical Taxi Zones",
            f"{physical_count:,}",
            help="Mapped physical Taxi Zones represented in this selected slice.",
        )

        scope_parts = [
            metric_label(selected_metric),
            f"h={selected_horizon}",
            period_caption(geography_df, selected_period),
        ]
        if selected_day_type != "All days":
            scope_parts.append(selected_day_type)
        if selected_daypart != "All dayparts":
            scope_parts.append(selected_daypart)
        scope_label = " · ".join(scope_parts)

        st.markdown("#### Follow the same question through four linked views")
        st.caption(
            "Maps show **where** the gap sits. The plane separates **local "
            "intensity** from **system contribution**. The selected-zone card "
            "explains one place. The ranking shows which physical Taxi Zones "
            "carry the most of the selected system divergence."
        )

        render_linked_spatial_bundle(
            explorer_zones,
            metric=selected_metric,
            horizon=selected_horizon,
            scope_label=scope_label,
            key_prefix="raw22_explorer",
            show_ranking=True,
            reset_key="raw22_explorer_reset_to_top",
        )

# ---------------------------------------------------------------------
# Closing synthesis
# ---------------------------------------------------------------------
st.divider()

st.markdown("### What this page establishes")
st.markdown(
    "Counterfactual divergence was spatially uneven, and the places with the most "
    "dramatic local percentage gaps were not necessarily the places carrying the most "
    "of the systemwide difference. Separating **local intensity** from **system "
    "contribution** prevents small-denominator extremes from being mistaken for the "
    "geographies that matter most to the aggregate counterfactual story."
)

with st.expander("How this page works", expanded=False):
    st.markdown(
        """
        **1. Compare observed and estimated no-CP levels at the same Taxi Zone grain.**
        The core quantity is **Gap = estimated no-CP − observed**. Positive and negative
        values indicate which path is higher; the practical meaning depends on whether
        the selected measure is demand or speed.

        **2. Preserve the correct aggregation for each mobility measure.** Counts and
        ridership sum across supported observations. Average speeds are rebuilt with
        each world's own activity weights rather than averaging zone speeds equally.

        **3. Separate a large local percentage from a large system contribution.**
        **Local gap intensity** is the percentage difference within one Taxi Zone and
        can become very large when observed activity is small. **System divergence
        share** asks how much of the selected absolute system difference is associated
        with each physical Taxi Zone.

        **4. Use robust map limits without hiding the underlying values.** Color ranges
        are capped so isolated extremes do not flatten the rest of the spatial pattern;
        hover retains the uncapped observed, no-CP, and gap values.

        **5. Keep time-window comparisons policy-week aligned.** Full-period and
        first/final-week selections regroup the same post-launch counterfactual surface
        rather than redefining the underlying no-CP estimate.
        """
    )

st.caption(
    "Evidence scope: model-estimated no-congestion-pricing mobility compared with "
    "observed post-launch mobility at Taxi Zone grain. Spatial gaps and contribution "
    "shares describe where the model-based counterfactual differs from observation; "
    "they are not a direct causal decomposition of every change after congestion "
    "pricing began."
)
