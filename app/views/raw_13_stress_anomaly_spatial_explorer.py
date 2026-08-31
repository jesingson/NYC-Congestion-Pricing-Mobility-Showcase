from __future__ import annotations

from datetime import date
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data_access.anomalies import (
    ANOMALY_EVENT_UNIVERSE_PATH,
    EVENT_ID_COLUMN,
    SELECTED_FINALIST_FLAG,
    load_selected_anomaly_events,
)
from app.data_access.loaders import load_analysis_panel
from app.data_access.aggregations import apply_common_filters
from app.data_access.mobility_environments import (
    format_mobility_regime_cluster_label,
    get_mobility_regime_cluster_options,
)
from app.data_access.spatial_visuals import get_zone_geojson
from app.utils.project_branding import (
    BRAND_COLORS,
    apply_branding,
    inject_app_css,
)


inject_app_css()
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        font-size: clamp(1.30rem, 1.75vw, 1.85rem);
        line-height: 1.15;
    }
    .raw13-chart-insight {
        background: #EDF6F9;
        border-left: 4px solid #006D77;
        border-radius: 0.45rem;
        color: #163F45;
        margin: 0.35rem 0 1.15rem 0;
        padding: 0.80rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------
CP_START_DATE = pd.Timestamp("2025-01-05")
CONGESTION_FLAG = "has_congestion_oriented"
DEMAND_FLAG = "has_positive_demand_shock"
FAMILY_COLUMN = "stress_family_exclusive"
HOTSPOT_QUANTILE = 0.75
ROBUST_COLOR_QUANTILE = 0.95

FAMILY_ORDER = [
    "Congestion-only",
    "Demand-only",
    "Both",
]

EVENT_COLUMNS = [
    EVENT_ID_COLUMN,
    "taxi_zone_id",
    "date",
    "temporal_bucket",
    SELECTED_FINALIST_FLAG,
    CONGESTION_FLAG,
    DEMAND_FLAG,
]

POLICY_GEOGRAPHY_MAP = {
    "cbd": "CBD",
    "gateway_to_cbd": "Gateway + adjacent",
    "adjacent_to_cbd": "Gateway + adjacent",
    "non_cbd": "Outside",
}

CHANGE_COLORSCALE = [
    [0.00, BRAND_COLORS["terracotta"]],
    [0.35, BRAND_COLORS["pale_peach"]],
    [0.50, BRAND_COLORS["ice"]],
    [0.65, BRAND_COLORS["seafoam"]],
    [1.00, BRAND_COLORS["dark_teal"]],
]

INCIDENCE_COLORSCALE = [
    [0.00, "#FFFFFF"],
    [0.20, BRAND_COLORS["ice"]],
    [0.55, BRAND_COLORS["seafoam"]],
    [1.00, BRAND_COLORS["dark_teal"]],
]

MAP_CENTER = {
    "lat": 40.7128,
    "lon": -74.0060,
}

EVENT_PROFILER_PAGE = Path(__file__).with_name(
    "raw_15_stress_anomaly_event_profiler.py"
)

ALL_TEMPORAL_BUCKETS = "All temporal buckets"
ALL_ZONES = "All zones"
GEOGRAPHY_SCHEMES = [
    ALL_ZONES,
    "Borough",
    "Policy geography",
    "Mobility environment",
]

TIME_VIEW_OPTIONS = [
    "Policy-period change",
    "Pre-CP",
    "Post-CP",
    "Full period",
    "Custom dates",
]

SAVED_VIEWS = {
    "Custom": None,
    "All-stress policy shift": {
        "family": "All stress anomalies",
        "time_view": "Policy-period change",
        "temporal_bucket": ALL_TEMPORAL_BUCKETS,
        "geography_scheme": ALL_ZONES,
        "geography_value": None,
    },
    "Demand-stress gains": {
        "family": "Demand-only",
        "time_view": "Policy-period change",
        "temporal_bucket": ALL_TEMPORAL_BUCKETS,
        "geography_scheme": ALL_ZONES,
        "geography_value": None,
    },
    "Congestion-stress retreats": {
        "family": "Congestion-only",
        "time_view": "Policy-period change",
        "temporal_bucket": ALL_TEMPORAL_BUCKETS,
        "geography_scheme": ALL_ZONES,
        "geography_value": None,
    },
    "Weekend overnight footprint": {
        "family": "All stress anomalies",
        "time_view": "Full period",
        "temporal_bucket": "weekend_overnight",
        "geography_scheme": ALL_ZONES,
        "geography_value": None,
    },
}


# ---------------------------------------------------------------------
# Data loading and exact incidence summaries
# ---------------------------------------------------------------------
@st.cache_data(show_spinner="Loading the stress-anomaly spatial universe...")
def _load_spatial_universe() -> pd.DataFrame:
    """Load the exact 3.3.6 numerator and denominator universe."""
    frame = pd.read_parquet(
        ANOMALY_EVENT_UNIVERSE_PATH,
        columns=EVENT_COLUMNS,
    )

    missing = sorted(set(EVENT_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(
            "The 3.3.6 event universe is missing Page 13 columns: "
            + ", ".join(missing)
        )

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError("The 3.3.6 event universe contains unparseable dates.")

    frame["taxi_zone_id"] = pd.to_numeric(
        frame["taxi_zone_id"], errors="coerce"
    ).astype("Int64")
    if frame["taxi_zone_id"].isna().any():
        raise ValueError("The event universe contains missing Taxi Zone IDs.")

    for flag in [SELECTED_FINALIST_FLAG, CONGESTION_FLAG, DEMAND_FLAG]:
        frame[flag] = frame[flag].fillna(False).astype(bool)

    frame["policy_period"] = np.where(
        frame["date"].lt(CP_START_DATE),
        "Pre-CP",
        "Post-CP",
    )

    selected = frame[SELECTED_FINALIST_FLAG]
    congestion = frame[CONGESTION_FLAG]
    demand = frame[DEMAND_FLAG]

    frame[FAMILY_COLUMN] = "Not selected"
    frame.loc[selected & congestion & ~demand, FAMILY_COLUMN] = "Congestion-only"
    frame.loc[selected & ~congestion & demand, FAMILY_COLUMN] = "Demand-only"
    frame.loc[selected & congestion & demand, FAMILY_COLUMN] = "Both"
    frame.loc[selected & ~congestion & ~demand, FAMILY_COLUMN] = "Unclassified"

    return frame


@st.cache_data(show_spinner="Loading Taxi Zone reference metadata...")
def _load_zone_reference() -> pd.DataFrame:
    """Return one complete, canonical metadata row per analysis Taxi Zone."""
    panel = load_analysis_panel(
        columns=[
            "taxi_zone_id",
            "zone",
            "borough",
            "cbd_spatial_category",
        ]
    ).copy()

    panel["taxi_zone_id"] = pd.to_numeric(
        panel["taxi_zone_id"], errors="coerce"
    ).astype("Int64")

    reference = (
        panel.dropna(subset=["taxi_zone_id"])
        .sort_values("taxi_zone_id")
        .drop_duplicates("taxi_zone_id", keep="first")
        .reset_index(drop=True)
    )
    reference["policy_geography"] = (
        reference["cbd_spatial_category"]
        .astype("string")
        .str.lower()
        .map(POLICY_GEOGRAPHY_MAP)
        .fillna("Unknown")
    )

    return reference


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Calculate events per 1,000 eligible observations."""
    denominator = pd.to_numeric(denominator, errors="coerce")
    numerator = pd.to_numeric(numerator, errors="coerce")
    return numerator.div(denominator.where(denominator.gt(0))).mul(1_000)


def _summarize_zone_period(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate exact zone-period totals and mutually exclusive family rates."""
    group_columns = ["taxi_zone_id", "policy_period"]

    denominator = (
        frame.groupby(group_columns, observed=True)[EVENT_ID_COLUMN]
        .nunique()
        .rename("eligible_observations")
        .reset_index()
    )

    selected = frame.loc[frame[SELECTED_FINALIST_FLAG]].copy()
    numerator = (
        selected.groupby(group_columns, observed=True)[EVENT_ID_COLUMN]
        .nunique()
        .rename("stress_anomalies")
        .reset_index()
    )

    family_counts = (
        selected.groupby(
            [*group_columns, FAMILY_COLUMN],
            observed=True,
        )[EVENT_ID_COLUMN]
        .nunique()
        .unstack(FAMILY_COLUMN, fill_value=0)
        .reindex(columns=FAMILY_ORDER, fill_value=0)
        .reset_index()
    )

    summary = (
        denominator.merge(numerator, on=group_columns, how="left")
        .merge(family_counts, on=group_columns, how="left")
    )

    count_columns = ["stress_anomalies", *FAMILY_ORDER]
    summary[count_columns] = summary[count_columns].fillna(0).astype("int64")
    summary["incidence_per_1k"] = _safe_rate(
        summary["stress_anomalies"], summary["eligible_observations"]
    )

    for family in FAMILY_ORDER:
        summary[f"{family}_per_1k"] = _safe_rate(
            summary[family], summary["eligible_observations"]
        )

    family_rates = summary[
        [f"{family}_per_1k" for family in FAMILY_ORDER]
    ].copy()
    family_rates.columns = FAMILY_ORDER
    summary["leading_family"] = family_rates.idxmax(axis=1)

    return summary


@st.cache_data(show_spinner="Calculating the frozen spatial hero...")
def _build_hero_table() -> pd.DataFrame:
    """Return one row per Taxi Zone with Pre/Post incidence and context."""
    frame = _load_spatial_universe()
    reference = _load_zone_reference()
    long = _summarize_zone_period(frame)

    # Build each side independently rather than pivoting mixed numeric and
    # text values together. This preserves numeric dtypes across pandas
    # versions so ranking and quantile operations remain reliable.
    period_frames: dict[str, pd.DataFrame] = {}
    for period_label, prefix in [("Pre-CP", "pre_cp"), ("Post-CP", "post_cp")]:
        period_frame = (
            long.loc[long["policy_period"].eq(period_label)]
            .drop(columns="policy_period")
            .copy()
        )
        period_frame = period_frame.rename(
            columns={
                column: f"{prefix}_{column}"
                for column in period_frame.columns
                if column != "taxi_zone_id"
            }
        )
        period_frames[prefix] = period_frame

    wide = period_frames["pre_cp"].merge(
        period_frames["post_cp"],
        on="taxi_zone_id",
        how="outer",
        validate="one_to_one",
    )

    wide = wide.merge(
        reference[
            ["taxi_zone_id", "zone", "borough", "policy_geography"]
        ],
        on="taxi_zone_id",
        how="left",
        validate="one_to_one",
    )

    wide["post_minus_pre_incidence_delta"] = (
        wide["post_cp_incidence_per_1k"]
        - wide["pre_cp_incidence_per_1k"]
    )

    for family in FAMILY_ORDER:
        wide[f"{family}_post_minus_pre_delta"] = (
            wide[f"post_cp_{family}_per_1k"]
            - wide[f"pre_cp_{family}_per_1k"]
        )

    pre_threshold = float(
        wide["pre_cp_incidence_per_1k"].quantile(HOTSPOT_QUANTILE)
    )
    post_threshold = float(
        wide["post_cp_incidence_per_1k"].quantile(HOTSPOT_QUANTILE)
    )
    pre_hot = wide["pre_cp_incidence_per_1k"].ge(pre_threshold)
    post_hot = wide["post_cp_incidence_per_1k"].ge(post_threshold)

    wide["hotspot_status"] = np.select(
        [pre_hot & post_hot, ~pre_hot & post_hot, pre_hot & ~post_hot],
        ["Persistent hotspot", "Emerging hotspot", "Receding hotspot"],
        default="Below hotspot threshold in both periods",
    )

    return wide.sort_values("taxi_zone_id").reset_index(drop=True)


# ---------------------------------------------------------------------
# Map construction
# ---------------------------------------------------------------------
def _geometry_ids(geojson: dict) -> set[int]:
    """Return unique Taxi Zone IDs represented in the canonical GeoJSON."""
    result: set[int] = set()

    for feature in geojson.get("features", []):
        properties = feature.get("properties") or {}
        value = properties.get("taxi_zone_id")
        if value is None:
            continue
        try:
            result.add(int(value))
        except (TypeError, ValueError):
            continue

    return result


def _robust_positive_cap(values: pd.Series) -> float:
    """Return a stable upper bound for the shared Pre/Post color scale."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return 1.0
    return max(float(numeric.quantile(ROBUST_COLOR_QUANTILE)), 1.0)


def _robust_symmetric_bound(values: pd.Series) -> float:
    """Return a symmetric robust bound for the change color scale."""
    numeric = pd.to_numeric(values, errors="coerce").dropna().abs()
    if numeric.empty:
        return 1.0
    return max(float(numeric.quantile(ROBUST_COLOR_QUANTILE)), 1.0)


def _build_map(
    map_frame: pd.DataFrame,
    geojson: dict,
    *,
    view: str,
    shared_incidence_cap: float,
    change_bound: float,
) -> go.Figure:
    """Build one frozen hero map with view-specific values and tooltip copy."""
    plot_frame = map_frame.copy()

    if view == "Change":
        value_column = "post_minus_pre_incidence_delta"
        zmin = -change_bound
        zmax = change_bound
        colorscale = CHANGE_COLORSCALE
        colorbar_title = "Post − Pre<br>per 1,000"
        plot_frame["_pre_display"] = plot_frame[
            "pre_cp_incidence_per_1k"
        ].map(lambda value: f"{value:.1f}")
        plot_frame["_post_display"] = plot_frame[
            "post_cp_incidence_per_1k"
        ].map(lambda value: f"{value:.1f}")
        plot_frame["_change_display"] = plot_frame[
            "post_minus_pre_incidence_delta"
        ].map(lambda value: f"{value:+.1f}")
        custom_columns = [
            "zone",
            "borough",
            "policy_geography",
            "_pre_display",
            "_post_display",
            "_change_display",
            "hotspot_status",
            "pre_cp_leading_family",
            "post_cp_leading_family",
        ]
        hovertemplate = (
            "<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
            "%{customdata[2]}<br><br>"
            "Pre-CP: %{customdata[3]} per 1,000<br>"
            "Post-CP: %{customdata[4]} per 1,000<br>"
            "<b>Change: %{customdata[5]} per 1,000</b><br><br>"
            "%{customdata[6]}<br>"
            "Leading family: %{customdata[7]} → %{customdata[8]}"
            "<extra></extra>"
        )
    else:
        period_prefix = "pre_cp" if view == "Pre-CP" else "post_cp"
        value_column = f"{period_prefix}_incidence_per_1k"
        zmin = 0.0
        zmax = shared_incidence_cap
        colorscale = INCIDENCE_COLORSCALE
        colorbar_title = "Stress anomalies<br>per 1,000"
        plot_frame["_incidence_display"] = plot_frame[value_column].map(
            lambda value: f"{value:.1f}"
        )
        custom_columns = [
            "zone",
            "borough",
            "policy_geography",
            "_incidence_display",
            f"{period_prefix}_leading_family",
            "hotspot_status",
        ]
        hovertemplate = (
            "<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
            "%{customdata[2]}<br><br>"
            f"{view} incidence: %{{customdata[3]}} per 1,000<br>"
            "Leading family: %{customdata[4]}<br>"
            "%{customdata[5]}"
            "<extra></extra>"
        )

    fig = go.Figure(
        go.Choroplethmap(
            geojson=geojson,
            locations=plot_frame["taxi_zone_id"],
            featureidkey="properties.taxi_zone_id",
            z=plot_frame[value_column],
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale,
            marker={"line": {"width": 0.45, "color": "white"}},
            customdata=plot_frame[custom_columns].to_numpy(),
            hovertemplate=hovertemplate,
            colorbar={
                "title": {"text": colorbar_title},
                "x": 1.01,
                "xanchor": "left",
                "y": 0.5,
                "yanchor": "middle",
                "len": 0.66,
                "thickness": 13,
                "outlinewidth": 0,
            },
            showscale=True,
        )
    )

    apply_branding(fig)
    fig.update_layout(
        # Shared branding contains title formatting without title text. Keep
        # Plotly's internal title explicitly blank to prevent an "undefined"
        # subtitle in current Streamlit/Plotly frontends.
        title={"text": ""},
        map={"style": "carto-positron", "center": MAP_CENTER, "zoom": 9.35},
        height=650,
        margin={"l": 0, "r": 105, "t": 10, "b": 5},
        hovermode="closest",
        uirevision=f"raw13-hero-{view}",
    )

    return fig


# ---------------------------------------------------------------------
# Frozen insight calculations
# ---------------------------------------------------------------------
def _hero_insights(hero: pd.DataFrame) -> dict[str, object]:
    """Calculate the evidence used by the frozen cards and narrative."""
    largest_increase = hero.nlargest(1, "post_minus_pre_incidence_delta").iloc[0]
    largest_decrease = hero.nsmallest(1, "post_minus_pre_incidence_delta").iloc[0]

    pre_top10 = set(hero.nlargest(10, "pre_cp_incidence_per_1k")["taxi_zone_id"])
    post_top10 = set(hero.nlargest(10, "post_cp_incidence_per_1k")["taxi_zone_id"])
    top10_overlap = len(pre_top10.intersection(post_top10))
    rank_correlation = hero["pre_cp_incidence_per_1k"].corr(
        hero["post_cp_incidence_per_1k"], method="spearman"
    )

    status_counts = hero["hotspot_status"].value_counts()
    increasing = hero.loc[hero["post_minus_pre_incidence_delta"].gt(0)]
    decreasing = hero.loc[hero["post_minus_pre_incidence_delta"].lt(0)]
    family_delta_columns = {
        family: f"{family}_post_minus_pre_delta" for family in FAMILY_ORDER
    }
    increase_family_totals = pd.Series(
        {
            family: increasing[column].sum()
            for family, column in family_delta_columns.items()
        }
    )
    decrease_family_totals = pd.Series(
        {
            family: decreasing[column].sum()
            for family, column in family_delta_columns.items()
        }
    )

    return {
        "largest_increase": largest_increase,
        "largest_decrease": largest_decrease,
        "top10_overlap": top10_overlap,
        "rank_correlation": float(rank_correlation),
        "emerging_hotspots": int(status_counts.get("Emerging hotspot", 0)),
        "receding_hotspots": int(status_counts.get("Receding hotspot", 0)),
        "persistent_hotspots": int(status_counts.get("Persistent hotspot", 0)),
        "leading_increase_family": str(increase_family_totals.idxmax()),
        "leading_decrease_family": str(decrease_family_totals.idxmin()),
    }


def _custom_scope_insights(
    table: pd.DataFrame,
    *,
    time_view: str,
) -> dict[str, object]:
    """Calculate compact, visitor-facing evidence for the active explorer scope."""
    valid = table.loc[table["value"].notna()].copy()
    if valid.empty:
        return {"zones": 0, "narrative": "No comparable zone values are available."}

    values = valid["value"].astype(float)
    result: dict[str, object] = {
        "zones": len(valid),
        "median": float(values.median()),
    }

    if time_view == "Policy-period change":
        increase = valid.loc[valid["value"].idxmax()]
        decrease = valid.loc[valid["value"].idxmin()]
        increased_count = int(values.gt(0).sum())
        decreased_count = int(values.lt(0).sum())
        unchanged_count = int(values.eq(0).sum())
        result.update(
            {
                "increase": increase,
                "decrease": decrease,
                "increased_count": increased_count,
                "decreased_count": decreased_count,
                "unchanged_count": unchanged_count,
                "increased_share": increased_count / len(valid) * 100,
            }
        )
    else:
        peak = valid.loc[valid["value"].idxmax()]
        upper_quartile = float(values.quantile(0.75))
        result.update(
            {
                "peak": peak,
                "upper_quartile": upper_quartile,
                "upper_quartile_count": int(values.ge(upper_quartile).sum()),
                "stress_anomalies": int(valid["stress_anomalies"].sum()),
            }
        )

    return result


def _scope_narrative(
    insights: dict[str, object],
    *,
    time_view: str,
    family: str,
    geography_label: str,
) -> str:
    """Spell out the active spatial pattern in plain language."""
    if not insights.get("zones"):
        return "No comparable zone values are available for this scope."

    family_label = family.lower()
    if time_view == "Policy-period change":
        increase = insights["increase"]
        decrease = insights["decrease"]
        direction = (
            "increased in most zones"
            if insights["increased_count"] > insights["decreased_count"]
            else "decreased in most zones"
            if insights["decreased_count"] > insights["increased_count"]
            else "split evenly between increases and decreases"
        )
        increase_phrase = (
            "largest rise" if increase["value"] > 0 else "smallest decline"
        )
        decrease_phrase = (
            "largest decline" if decrease["value"] < 0 else "smallest rise"
        )
        return (
            f"Across {geography_label.lower()}, {family_label} {direction}: "
            f"{insights['increased_count']} of {insights['zones']} zones increased "
            f"and {insights['decreased_count']} decreased. "
            f"{increase['zone']} recorded the {increase_phrase} "
            f"({increase['value']:+.1f} per 1,000), while {decrease['zone']} "
            f"recorded the {decrease_phrase} ({decrease['value']:+.1f}). The median "
            f"zone changed by {insights['median']:+.1f} per 1,000. These are "
            "descriptive shifts around the policy date, not causal estimates."
        )

    peak = insights["peak"]
    return (
        f"Within {geography_label.lower()}, {peak['zone']} had the highest "
        f"{family_label} incidence at {peak['value']:.1f} per 1,000 eligible "
        f"observations. The median zone was {insights['median']:.1f}; the upper "
        f"quarter began at {insights['upper_quartile']:.1f}. In total, "
        f"{insights['stress_anomalies']:,} matching stress anomalies were observed "
        f"across {insights['zones']} zones in this scope."
    )


def _render_chart_takeaway(text: str) -> None:
    """Render the page's single, consistent chart-insight treatment."""
    st.markdown(
        f'<div class="raw13-chart-insight"><strong>Takeaway.</strong> '
        f"{escape(text)}</div>",
        unsafe_allow_html=True,
    )


def _hero_period_narrative(hero: pd.DataFrame, period: str) -> str:
    """Summarize one frozen Pre-CP or Post-CP incidence map."""
    column = (
        "pre_cp_incidence_per_1k"
        if period == "Pre-CP"
        else "post_cp_incidence_per_1k"
    )
    values = hero[column].dropna().astype(float)
    peak = hero.loc[hero[column].idxmax()]
    threshold = float(values.quantile(HOTSPOT_QUANTILE))
    hotspot_count = int(values.ge(threshold).sum())
    return (
        f"{peak['zone']} had the highest {period} incidence at "
        f"{peak[column]:.1f} stress anomalies per 1,000 eligible observations. "
        f"The median zone was {values.median():.1f}, and {hotspot_count} zones "
        f"fell in the period's upper quarter at or above {threshold:.1f}."
    )


def _ranking_narrative(
    frame: pd.DataFrame,
    *,
    time_view: str,
    rank_by: str,
) -> str:
    """Summarize the bars currently visible in the zone ranking."""
    if rank_by == "Lowest change":
        displayed = frame.nsmallest(15, "value").sort_values("value")
    elif rank_by == "Largest absolute change":
        displayed = (
            frame.assign(_absolute_value=frame["value"].abs())
            .nlargest(15, "_absolute_value")
            .sort_values("_absolute_value", ascending=False)
        )
    else:
        displayed = frame.nlargest(15, "value").sort_values("value", ascending=False)

    leader = displayed.iloc[0]
    runner_up = displayed.iloc[1] if len(displayed) > 1 else None
    shown = len(displayed)

    if time_view != "Policy-period change":
        runner_copy = (
            f" {runner_up['zone']} followed at {runner_up['value']:.1f}."
            if runner_up is not None
            else ""
        )
        return (
            f"{leader['zone']} led the {shown} displayed zones at "
            f"{leader['value']:.1f} stress anomalies per 1,000.{runner_copy} "
            f"The displayed-zone median was {displayed['value'].median():.1f}."
        )

    positive_count = int(displayed["value"].gt(0).sum())
    negative_count = int(displayed["value"].lt(0).sum())
    runner_copy = (
        f" {runner_up['zone']} was next at {runner_up['value']:+.1f}."
        if runner_up is not None
        else ""
    )
    ranking_phrase = {
        "Highest change": "highest change",
        "Lowest change": "lowest change",
        "Largest absolute change": "largest absolute change",
    }.get(rank_by, "leading change")
    return (
        f"{leader['zone']} had the {ranking_phrase} in this view at "
        f"{leader['value']:+.1f} per 1,000.{runner_copy} Among the "
        f"{shown} displayed zones, {positive_count} increased and {negative_count} "
        "decreased."
    )


def _timeline_narrative(timeline: pd.DataFrame) -> str:
    """Summarize peaks and the broad movement in the displayed monthly lines."""
    valid = timeline.loc[timeline["incidence_per_1k"].notna()].copy()
    peak = valid.loc[valid["incidence_per_1k"].idxmax()]
    peak_month = pd.Timestamp(peak["month"]).strftime("%B %Y")
    monthly_mean = (
        valid.groupby("month", observed=True)["incidence_per_1k"]
        .mean()
        .sort_index()
    )
    pre_values = valid.loc[
        valid["month"].lt(CP_START_DATE), "incidence_per_1k"
    ]
    post_values = valid.loc[
        valid["month"].ge(CP_START_DATE), "incidence_per_1k"
    ]
    if not pre_values.empty and not post_values.empty:
        pre_mean = float(pre_values.mean())
        post_mean = float(post_values.mean())
        movement = (
            f"Across the displayed zone-months, average incidence moved from "
            f"{pre_mean:.1f} Pre-CP to {post_mean:.1f} Post-CP "
            f"({post_mean - pre_mean:+.1f} per 1,000)."
        )
    else:
        first_value = float(monthly_mean.iloc[0])
        last_value = float(monthly_mean.iloc[-1])
        movement = (
            f"The displayed-zone monthly average moved from {first_value:.1f} "
            f"at the start of the view to {last_value:.1f} at the end "
            f"({last_value - first_value:+.1f} per 1,000)."
        )
    return (
        f"{peak['zone']} reached the highest monthly incidence in {peak_month} "
        f"at {peak['incidence_per_1k']:.1f} per 1,000. {movement}"
    )


# ---------------------------------------------------------------------
# Custom spatial explorer
# ---------------------------------------------------------------------
def _format_temporal_bucket(value: str) -> str:
    """Convert an internal temporal-bucket value to visitor-facing copy."""
    if value == ALL_TEMPORAL_BUCKETS:
        return value

    return (
        value.replace("_", " ")
        .title()
        .replace("Am ", "AM ")
        .replace("Pm ", "PM ")
    )


def _selected_family_mask(frame: pd.DataFrame, family: str) -> pd.Series:
    """Return the selected-event numerator mask for one exclusive family."""
    selected = frame[SELECTED_FINALIST_FLAG]
    if family == "All stress anomalies":
        return selected
    return selected & frame[FAMILY_COLUMN].eq(family)


def _single_scope_zone_summary(
    frame: pd.DataFrame,
    family: str,
) -> pd.DataFrame:
    """Return one incidence row per zone for a single time scope."""
    denominator = (
        frame.groupby("taxi_zone_id", observed=True)[EVENT_ID_COLUMN]
        .nunique()
        .rename("eligible_observations")
        .reset_index()
    )
    numerator = (
        frame.loc[_selected_family_mask(frame, family)]
        .groupby("taxi_zone_id", observed=True)[EVENT_ID_COLUMN]
        .nunique()
        .rename("stress_anomalies")
        .reset_index()
    )
    summary = denominator.merge(numerator, on="taxi_zone_id", how="left")
    summary["stress_anomalies"] = summary["stress_anomalies"].fillna(0).astype(int)
    summary["incidence_per_1k"] = _safe_rate(
        summary["stress_anomalies"], summary["eligible_observations"]
    )
    return summary


@st.cache_data(show_spinner="Updating the spatial explorer...")
def _build_custom_table(
    family: str,
    time_view: str,
    temporal_bucket: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Build the custom map table from the exact event universe."""
    frame = _load_spatial_universe()
    reference = _load_zone_reference()

    if temporal_bucket != ALL_TEMPORAL_BUCKETS:
        frame = frame.loc[frame["temporal_bucket"].eq(temporal_bucket)].copy()

    if time_view == "Policy-period change":
        period_tables: dict[str, pd.DataFrame] = {}
        for period_label, prefix in [("Pre-CP", "pre"), ("Post-CP", "post")]:
            period_frame = frame.loc[frame["policy_period"].eq(period_label)]
            summary = _single_scope_zone_summary(period_frame, family).rename(
                columns={
                    "eligible_observations": f"{prefix}_eligible_observations",
                    "stress_anomalies": f"{prefix}_stress_anomalies",
                    "incidence_per_1k": f"{prefix}_incidence_per_1k",
                }
            )
            period_tables[prefix] = summary

        result = period_tables["pre"].merge(
            period_tables["post"],
            on="taxi_zone_id",
            how="outer",
            validate="one_to_one",
        )
        result["value"] = (
            result["post_incidence_per_1k"]
            - result["pre_incidence_per_1k"]
        )
    else:
        if time_view == "Pre-CP":
            frame = frame.loc[frame["date"].lt(CP_START_DATE)].copy()
        elif time_view == "Post-CP":
            frame = frame.loc[frame["date"].ge(CP_START_DATE)].copy()
        elif time_view == "Custom dates":
            start_timestamp = pd.Timestamp(start_date)
            end_timestamp = pd.Timestamp(end_date)
            frame = frame.loc[
                frame["date"].between(start_timestamp, end_timestamp, inclusive="both")
            ].copy()

        result = _single_scope_zone_summary(frame, family)
        result["value"] = result["incidence_per_1k"]

    result = result.merge(
        reference[["taxi_zone_id", "zone", "borough", "policy_geography"]],
        on="taxi_zone_id",
        how="left",
        validate="one_to_one",
    )
    return result


def _build_explorer_map(
    frame: pd.DataFrame,
    geojson: dict,
    *,
    time_view: str,
    selected_zone_ids: tuple[int, ...],
) -> go.Figure:
    """Build the custom explorer choropleth with preformatted tooltip values."""
    plot_frame = frame.copy()

    if time_view == "Policy-period change":
        bound = _robust_symmetric_bound(plot_frame["value"])
        zmin, zmax = -bound, bound
        colorscale = CHANGE_COLORSCALE
        colorbar_title = "Post − Pre<br>per 1,000"
        plot_frame["_pre_display"] = plot_frame["pre_incidence_per_1k"].map(
            lambda value: f"{value:.1f}"
        )
        plot_frame["_post_display"] = plot_frame["post_incidence_per_1k"].map(
            lambda value: f"{value:.1f}"
        )
        plot_frame["_value_display"] = plot_frame["value"].map(
            lambda value: f"{value:+.1f}"
        )
        custom_columns = [
            "taxi_zone_id",
            "zone",
            "borough",
            "policy_geography",
            "_pre_display",
            "_post_display",
            "_value_display",
        ]
        hovertemplate = (
            "<b>%{customdata[1]}</b> · %{customdata[2]}<br>"
            "%{customdata[3]}<br><br>"
            "Pre-CP: %{customdata[4]} per 1,000<br>"
            "Post-CP: %{customdata[5]} per 1,000<br>"
            "<b>Change: %{customdata[6]} per 1,000</b>"
            "<extra></extra>"
        )
    else:
        cap = _robust_positive_cap(plot_frame["value"])
        zmin, zmax = 0.0, cap
        colorscale = INCIDENCE_COLORSCALE
        colorbar_title = "Stress anomalies<br>per 1,000"
        plot_frame["_value_display"] = plot_frame["value"].map(
            lambda value: f"{value:.1f}"
        )
        custom_columns = [
            "taxi_zone_id",
            "zone",
            "borough",
            "policy_geography",
            "_value_display",
        ]
        hovertemplate = (
            "<b>%{customdata[1]}</b> · %{customdata[2]}<br>"
            "%{customdata[3]}<br><br>"
            "Incidence: %{customdata[4]} per 1,000"
            "<extra></extra>"
        )

    fig = go.Figure(
        go.Choroplethmap(
            geojson=geojson,
            locations=plot_frame["taxi_zone_id"],
            featureidkey="properties.taxi_zone_id",
            z=plot_frame["value"],
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale,
            marker={"line": {"width": 0.45, "color": "white"}},
            customdata=plot_frame[custom_columns].to_numpy(),
            hovertemplate=hovertemplate,
            colorbar={
                "title": {"text": colorbar_title},
                "x": 1.01,
                "xanchor": "left",
                "len": 0.66,
                "thickness": 13,
                "outlinewidth": 0,
            },
        )
    )

    selected_frame = plot_frame.loc[
        plot_frame["taxi_zone_id"].astype(int).isin(selected_zone_ids)
    ].copy()
    if not selected_frame.empty:
        # A transparent overlay provides deterministic two-way selection
        # feedback independent of Plotly's transient client selection state.
        fig.add_trace(
            go.Choroplethmap(
                geojson=geojson,
                locations=selected_frame["taxi_zone_id"],
                featureidkey="properties.taxi_zone_id",
                z=np.zeros(len(selected_frame)),
                zmin=0,
                zmax=1,
                colorscale=[
                    [0.0, "rgba(0,0,0,0)"],
                    [1.0, "rgba(0,0,0,0)"],
                ],
                marker={
                    "line": {
                        "width": 3.0,
                        "color": "#003F46",
                    }
                },
                customdata=selected_frame[["taxi_zone_id"]].to_numpy(),
                hoverinfo="skip",
                showscale=False,
                name="Selected Taxi Zones",
                showlegend=False,
            )
        )
    apply_branding(fig)
    fig.update_layout(
        title={"text": ""},
        map={"style": "carto-positron", "center": MAP_CENTER, "zoom": 9.35},
        height=650,
        margin={"l": 0, "r": 105, "t": 10, "b": 5},
        hovermode="closest",
        uirevision="raw13-custom-map",
    )
    return fig


def _clear_zone_selection() -> None:
    """Clear the authoritative zone selection and reset Plotly client state."""
    st.session_state["raw13_selected_zones"] = []
    st.session_state["raw13_map_revision"] = (
        int(st.session_state.get("raw13_map_revision", 0)) + 1
    )


def _zone_multiselect_changed() -> None:
    """Redraw the map after a dropdown edit so both controls stay in sync."""
    st.session_state["raw13_map_revision"] = (
        int(st.session_state.get("raw13_map_revision", 0)) + 1
    )


def _build_ranking_chart(
    frame: pd.DataFrame,
    *,
    time_view: str,
    rank_by: str,
) -> tuple[go.Figure, pd.DataFrame]:
    """Return the Top-15 ranking chart and its display table."""
    if rank_by in {"Largest decrease", "Lowest change"}:
        ranked = frame.nsmallest(15, "value").sort_values("value", ascending=True)
    elif rank_by == "Largest absolute change":
        ranked = (
            frame.assign(_absolute_value=frame["value"].abs())
            .nlargest(15, "_absolute_value")
            .sort_values("value", ascending=True)
        )
    else:
        ranked = frame.nlargest(15, "value").sort_values("value", ascending=True)

    colors = (
        np.where(
            ranked["value"].ge(0),
            BRAND_COLORS["dark_teal"],
            BRAND_COLORS["terracotta"],
        )
        if time_view == "Policy-period change"
        else BRAND_COLORS["dark_teal"]
    )
    value_suffix = " change per 1,000" if time_view == "Policy-period change" else " per 1,000"

    fig = go.Figure(
        go.Bar(
            x=ranked["value"],
            y=ranked["zone"],
            orientation="h",
            marker_color=colors,
            customdata=ranked[["borough", "policy_geography"]].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b> · %{customdata[0]}<br>"
                "%{customdata[1]}<br>"
                f"%{{x:.1f}}{value_suffix}"
                "<extra></extra>"
            ),
        )
    )
    apply_branding(fig)
    fig.update_layout(
        title={"text": ""},
        height=510,
        margin={"l": 190, "r": 25, "t": 10, "b": 55},
        showlegend=False,
    )
    fig.update_xaxes(
        title_text=(
            "Post-CP minus Pre-CP incidence per 1,000"
            if time_view == "Policy-period change"
            else "Stress anomalies per 1,000 eligible observations"
        ),
        zeroline=True,
        zerolinecolor="#335C67",
    )
    fig.update_yaxes(title_text="")

    table_columns = ["taxi_zone_id", "zone", "borough", "policy_geography"]
    if time_view == "Policy-period change":
        table_columns.extend(
            ["pre_incidence_per_1k", "post_incidence_per_1k", "value"]
        )
    else:
        table_columns.extend(["eligible_observations", "stress_anomalies", "value"])

    display_table = ranked[table_columns].copy()
    display_table = display_table.rename(
        columns={
            "pre_incidence_per_1k": "Pre-CP per 1,000",
            "post_incidence_per_1k": "Post-CP per 1,000",
            "eligible_observations": "Eligible observations",
            "stress_anomalies": "Stress anomalies",
            "value": (
                "Change per 1,000"
                if time_view == "Policy-period change"
                else "Incidence per 1,000"
            ),
        }
    )
    numeric_rate_columns = [
        column for column in display_table.columns if "per 1,000" in column
    ]
    display_table[numeric_rate_columns] = display_table[numeric_rate_columns].round(1)
    return fig, display_table


def _apply_saved_view() -> None:
    """Apply one saved spatial view immediately."""
    view_name = st.session_state["raw13_saved_view"]
    config = SAVED_VIEWS.get(view_name)
    if config is None:
        return

    st.session_state["raw13_family"] = config["family"]
    st.session_state["raw13_time_view"] = config["time_view"]
    st.session_state["raw13_bucket"] = config["temporal_bucket"]
    st.session_state["raw13_geography_scheme"] = config["geography_scheme"]
    st.session_state["raw13_geography_value"] = config["geography_value"]


def _mark_saved_view_custom() -> None:
    """Mark the explorer custom after an individual control changes."""
    st.session_state["raw13_saved_view"] = "Custom"


def _change_geography_scheme() -> None:
    """Reset the competing geographic segment when its scheme changes."""
    st.session_state["raw13_saved_view"] = "Custom"
    st.session_state.pop("raw13_geography_value", None)


@st.cache_data(show_spinner=False)
def _get_mobility_environment_zone_ids(
    cluster_label: int,
    time_view: str,
    start_date: date,
    end_date: date,
) -> tuple[int, ...]:
    """Return zones assigned to one period-aware mobility environment."""
    panel = load_analysis_panel(
        columns=["taxi_zone_id", "date", "pre_post_cp"]
    ).copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")

    if time_view == "Pre-CP":
        panel = panel.loc[panel["date"].lt(CP_START_DATE)].copy()
    elif time_view == "Post-CP":
        panel = panel.loc[panel["date"].ge(CP_START_DATE)].copy()
    elif time_view == "Custom dates":
        panel = panel.loc[
            panel["date"].between(
                pd.Timestamp(start_date),
                pd.Timestamp(end_date),
                inclusive="both",
            )
        ].copy()

    panel = apply_common_filters(
        panel,
        mobility_regime_cluster_label=int(cluster_label),
    )
    return tuple(
        sorted(panel["taxi_zone_id"].dropna().astype(int).unique().tolist())
    )


def _apply_time_scope(
    frame: pd.DataFrame,
    *,
    time_view: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Apply one non-comparative time scope to event-level rows."""
    if time_view == "Pre-CP":
        return frame.loc[frame["date"].lt(CP_START_DATE)].copy()
    if time_view == "Post-CP":
        return frame.loc[frame["date"].ge(CP_START_DATE)].copy()
    if time_view == "Custom dates":
        return frame.loc[
            frame["date"].between(
                pd.Timestamp(start_date),
                pd.Timestamp(end_date),
                inclusive="both",
            )
        ].copy()
    return frame.copy()


@st.cache_data(show_spinner=False)
def _build_zone_monthly_timeline(
    zone_ids: tuple[int, ...],
    family: str,
    time_view: str,
    temporal_bucket: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Build monthly incidence for one to five selected Taxi Zones."""
    frame = _load_spatial_universe()
    reference = _load_zone_reference()
    frame = frame.loc[frame["taxi_zone_id"].astype(int).isin(zone_ids)].copy()

    if temporal_bucket != ALL_TEMPORAL_BUCKETS:
        frame = frame.loc[frame["temporal_bucket"].eq(temporal_bucket)].copy()

    # A policy-change map still receives the full timeline so the policy date
    # remains visible instead of collapsing the drill-down to two aggregates.
    if time_view != "Policy-period change":
        frame = _apply_time_scope(
            frame,
            time_view=time_view,
            start_date=start_date,
            end_date=end_date,
        )

    frame["month"] = frame["date"].dt.to_period("M").dt.to_timestamp()
    denominator = (
        frame.groupby(["taxi_zone_id", "month"], observed=True)[EVENT_ID_COLUMN]
        .nunique()
        .rename("eligible_observations")
        .reset_index()
    )

    selected = frame.loc[_selected_family_mask(frame, family)]
    numerator = (
        selected.groupby(["taxi_zone_id", "month"], observed=True)[EVENT_ID_COLUMN]
        .nunique()
        .rename("stress_anomalies")
        .reset_index()
    )
    result = denominator.merge(
        numerator,
        on=["taxi_zone_id", "month"],
        how="left",
    )
    result["stress_anomalies"] = result["stress_anomalies"].fillna(0).astype(int)
    result["incidence_per_1k"] = _safe_rate(
        result["stress_anomalies"], result["eligible_observations"]
    )
    return result.merge(
        reference[["taxi_zone_id", "zone"]],
        on="taxi_zone_id",
        how="left",
        validate="many_to_one",
    )


def _build_zone_timeline_chart(
    timeline: pd.DataFrame,
    *,
    selected_zone_count: int,
) -> go.Figure:
    """Build a monthly line comparison for the selected Taxi Zones."""
    fig = go.Figure()
    zone_names = timeline["zone"].dropna().astype(str).unique().tolist()
    color_sequence = [
        BRAND_COLORS["dark_teal"],
        BRAND_COLORS["terracotta"],
        BRAND_COLORS["seafoam"],
        "#5B5F97",
        "#335C67",
    ]

    for index, zone_name in enumerate(zone_names):
        zone_timeline = timeline.loc[timeline["zone"].eq(zone_name)].copy()
        zone_timeline["_incidence_display"] = zone_timeline[
            "incidence_per_1k"
        ].map(lambda value: f"{value:.1f}")
        fig.add_trace(
            go.Scatter(
                x=zone_timeline["month"],
                y=zone_timeline["incidence_per_1k"],
                mode="lines+markers" if selected_zone_count > 1 else "lines",
                name=zone_name,
                line={"width": 2.5, "color": color_sequence[index % len(color_sequence)]},
                marker={
                    "size": 7,
                    "symbol": "circle-open",
                    "color": color_sequence[index % len(color_sequence)],
                    "line": {
                        "color": color_sequence[index % len(color_sequence)],
                        "width": 1.5,
                    },
                },
                customdata=zone_timeline[["_incidence_display"]].to_numpy(),
                hovertemplate=(
                    f"<b>{zone_name}</b><br>"
                    "Month: %{x|%B %Y}<br>"
                    "Incidence: %{customdata[0]} per 1,000"
                    "<extra></extra>"
                ),
            )
        )

    apply_branding(fig)
    fig.update_layout(
        title={"text": ""},
        height=430,
        margin={"l": 65, "r": 25, "t": 35, "b": 60},
        hovermode="x unified" if selected_zone_count > 1 else "closest",
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.02,
            "yanchor": "bottom",
        },
    )
    fig.update_xaxes(title_text="Month", showgrid=False, dtick="M3")
    fig.update_yaxes(
        title_text="Stress anomalies per 1,000 eligible observations",
        rangemode="tozero",
    )
    if timeline["month"].min() <= CP_START_DATE <= timeline["month"].max():
        fig.add_vline(
            x=CP_START_DATE,
            line_width=2,
            line_dash="dash",
            line_color="#335C67",
        )
    return fig


@st.cache_data(show_spinner=False)
def _load_zone_event_rows(
    zone_ids: tuple[int, ...],
    family: str,
    time_view: str,
    temporal_bucket: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Return selected stress-anomaly events for the zone drill-down."""
    events = load_selected_anomaly_events().copy()
    events["taxi_zone_id"] = pd.to_numeric(
        events["taxi_zone_id"], errors="coerce"
    ).astype("Int64")
    events["date"] = pd.to_datetime(events["date"], errors="coerce")
    events = events.loc[events["taxi_zone_id"].astype(int).isin(zone_ids)].copy()

    if temporal_bucket != ALL_TEMPORAL_BUCKETS:
        events = events.loc[events["temporal_bucket"].eq(temporal_bucket)].copy()
    if time_view != "Policy-period change":
        events = _apply_time_scope(
            events,
            time_view=time_view,
            start_date=start_date,
            end_date=end_date,
        )

    congestion = events[CONGESTION_FLAG].fillna(False).astype(bool)
    demand = events[DEMAND_FLAG].fillna(False).astype(bool)
    events[FAMILY_COLUMN] = np.select(
        [congestion & ~demand, ~congestion & demand, congestion & demand],
        FAMILY_ORDER,
        default="Unclassified",
    )
    if family != "All stress anomalies":
        events = events.loc[events[FAMILY_COLUMN].eq(family)].copy()

    return events.sort_values(
        ["date", "taxi_zone_id", "daypart_order"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def _event_display_table(events: pd.DataFrame) -> pd.DataFrame:
    """Return a compact visitor-facing event table using available fields."""
    preferred_columns = [
        EVENT_ID_COLUMN,
        "date",
        "zone",
        "borough",
        "daypart",
        "temporal_bucket",
        FAMILY_COLUMN,
        "event_modality_driver_list",
        "event_metric_driver_list",
    ]
    available = [column for column in preferred_columns if column in events.columns]
    result = events[available].copy()
    if "date" in result.columns:
        result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    return result.rename(
        columns={
            EVENT_ID_COLUMN: "Event ID",
            "date": "Date",
            "zone": "Taxi Zone",
            "borough": "Borough",
            "daypart": "Daypart",
            "temporal_bucket": "Temporal bucket",
            FAMILY_COLUMN: "Stress family",
            "event_modality_driver_list": "Modality drivers",
            "event_metric_driver_list": "Metric drivers",
        }
    )


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------
st.caption("STRESS ANOMALY SPATIAL EXPLORER")
st.title("Where were stress anomalies concentrated—and how did that geography change?")
st.write(
    "Compare Taxi Zone stress-anomaly incidence before and after congestion "
    "pricing. Incidence uses the full eligible observation universe so zones "
    "and periods remain directly comparable."
)

try:
    hero_table = _build_hero_table()
    zone_geojson = get_zone_geojson()
except Exception:
    st.error(
        "The spatial view could not be prepared. Confirm that the final anomaly "
        "tables and Taxi Zone geometry are available, then reload the page."
    )
    st.stop()

analysis_zone_ids = set(hero_table["taxi_zone_id"].dropna().astype(int))
geometry_zone_ids = _geometry_ids(zone_geojson)
mapped_zone_ids = analysis_zone_ids.intersection(geometry_zone_ids)

map_table = hero_table.loc[
    hero_table["taxi_zone_id"].astype(int).isin(mapped_zone_ids)
].copy()

pooled_incidence = pd.concat(
    [map_table["pre_cp_incidence_per_1k"], map_table["post_cp_incidence_per_1k"]],
    ignore_index=True,
)
shared_incidence_cap = _robust_positive_cap(pooled_incidence)
change_bound = _robust_symmetric_bound(map_table["post_minus_pre_incidence_delta"])
insights = _hero_insights(hero_table)
map_insights = _hero_insights(map_table)

increase_row = insights["largest_increase"]
decrease_row = insights["largest_decrease"]

card1, card2, card3, card4 = st.columns(4)
card1.metric(
    "Largest increase",
    str(increase_row["zone"]),
    delta=f"{increase_row['post_minus_pre_incidence_delta']:+.1f} per 1,000",
    help="Largest absolute Post-CP minus Pre-CP incidence increase.",
)
card2.metric(
    "Largest decrease",
    str(decrease_row["zone"]),
    delta=f"{decrease_row['post_minus_pre_incidence_delta']:+.1f} per 1,000",
    help="Largest absolute Post-CP minus Pre-CP incidence decrease.",
)
card3.metric(
    "Hotspot turnover",
    f"{insights['emerging_hotspots']} emerging",
    delta=f"{insights['receding_hotspots']} receding",
    delta_color="off",
    help=(
        "Emerging zones crossed above the Post-CP 75th-percentile threshold; "
        "receding zones fell below it."
    ),
)
card4.metric(
    "Top-10 continuity",
    f"{insights['top10_overlap']} of 10 retained",
    help="Zones appearing in both the Pre-CP and Post-CP incidence Top 10.",
)

st.markdown("### Stress-anomaly geography changed sharply")
change_tab, pre_tab, post_tab = st.tabs(["Change", "Pre-CP", "Post-CP"])

with change_tab:
    st.caption("Post-CP minus Pre-CP stress anomalies per 1,000 eligible observations")
    st.plotly_chart(
        _build_map(
            map_table,
            zone_geojson,
            view="Change",
            shared_incidence_cap=shared_incidence_cap,
            change_bound=change_bound,
        ),
        width="stretch",
        config={"displayModeBar": False},
    )
    st.caption(
        f"The diverging scale is capped symmetrically at ±{change_bound:.1f} "
        "per 1,000 (the 95th percentile of absolute zone changes). Exact "
        "values remain available in the tooltip."
    )
    _render_chart_takeaway(
        f"The spatial ranking changed substantially: the Pre/Post rank "
        f"correlation was {map_insights['rank_correlation']:.2f}, and only "
        f"{map_insights['top10_overlap']} of the Pre-CP Top 10 remained in the "
        f"Post-CP Top 10. {map_insights['emerging_hotspots']} hotspots emerged, "
        f"{map_insights['receding_hotspots']} receded, and "
        f"{map_insights['persistent_hotspots']} remained elevated in both periods."
    )

with pre_tab:
    st.caption("Pre-CP stress anomalies per 1,000 eligible observations")
    st.plotly_chart(
        _build_map(
            map_table,
            zone_geojson,
            view="Pre-CP",
            shared_incidence_cap=shared_incidence_cap,
            change_bound=change_bound,
        ),
        width="stretch",
        config={"displayModeBar": False},
    )
    st.caption(
        f"Uses the shared 0–{shared_incidence_cap:.1f} per 1,000 scale. "
        "Tooltips retain exact values above the color cap."
    )
    _render_chart_takeaway(_hero_period_narrative(map_table, "Pre-CP"))

with post_tab:
    st.caption("Post-CP stress anomalies per 1,000 eligible observations")
    st.plotly_chart(
        _build_map(
            map_table,
            zone_geojson,
            view="Post-CP",
            shared_incidence_cap=shared_incidence_cap,
            change_bound=change_bound,
        ),
        width="stretch",
        config={"displayModeBar": False},
    )
    st.caption(
        f"Uses the shared 0–{shared_incidence_cap:.1f} per 1,000 scale. "
        "Tooltips retain exact values above the color cap."
    )
    _render_chart_takeaway(_hero_period_narrative(map_table, "Post-CP"))

with st.expander("How to read this view", expanded=False):
    st.markdown(
        """
        - **Change** subtracts each zone's Pre-CP incidence from its Post-CP incidence. Teal indicates an increase; terracotta indicates a decrease.
        - **Pre-CP and Post-CP** use one shared sequential color scale, so the same color represents the same incidence in both tabs.
        - **Incidence** is distinct selected stress anomalies divided by all distinct eligible observations in the matching zone and period, multiplied by 1,000.
        - **Emerging, receding, and persistent hotspots** use the period-specific 75th percentile of Taxi Zone incidence. They are descriptive categories, not causal estimates.
        - The maps describe spatial change around the policy date; they do not establish that congestion pricing caused a zone's change.
        """
    )

st.divider()
st.markdown("## Explore spatial stress patterns")
st.write(
    "Change the stress family, time scope, temporal bucket, or geographic "
    "segmentation to locate concentrations beyond the frozen policy-period hero."
)

reference = _load_zone_reference()
borough_options = [
    *[
        value
        for value in sorted(reference["borough"].dropna().astype(str).unique())
        if value.lower() not in {"unknown", "ewr"}
    ],
]
policy_geography_options = [
    *sorted(reference["policy_geography"].dropna().astype(str).unique()),
]
mobility_environment_options = [
    *get_mobility_regime_cluster_options(),
]
temporal_bucket_options = [
    ALL_TEMPORAL_BUCKETS,
    *sorted(
        _load_spatial_universe()["temporal_bucket"]
        .dropna()
        .astype(str)
        .unique()
    ),
]

default_saved_view = "All-stress policy shift"
default_config = SAVED_VIEWS[default_saved_view]
st.session_state.setdefault("raw13_saved_view", default_saved_view)
st.session_state.setdefault("raw13_family", default_config["family"])
st.session_state.setdefault("raw13_time_view", default_config["time_view"])
st.session_state.setdefault("raw13_bucket", default_config["temporal_bucket"])
st.session_state.setdefault(
    "raw13_geography_scheme", default_config["geography_scheme"]
)
st.session_state.setdefault(
    "raw13_geography_value", default_config["geography_value"]
)
st.session_state.setdefault("raw13_start_date", date(2023, 1, 1))
st.session_state.setdefault("raw13_end_date", date(2026, 3, 31))

st.selectbox(
    "Saved view",
    options=list(SAVED_VIEWS),
    key="raw13_saved_view",
    on_change=_apply_saved_view,
    help="Saved views apply immediately; changing another control marks the view Custom.",
)

control1, control2, control3 = st.columns(3)
with control1:
    selected_family = st.selectbox(
        "Stress family",
        options=["All stress anomalies", *FAMILY_ORDER],
        key="raw13_family",
        on_change=_mark_saved_view_custom,
    )
with control2:
    selected_time_view = st.selectbox(
        "Time scope",
        options=TIME_VIEW_OPTIONS,
        key="raw13_time_view",
        on_change=_mark_saved_view_custom,
    )
with control3:
    selected_bucket = st.selectbox(
        "Temporal bucket",
        options=temporal_bucket_options,
        format_func=_format_temporal_bucket,
        key="raw13_bucket",
        on_change=_mark_saved_view_custom,
    )
scope1, scope2 = st.columns(2)
with scope1:
    selected_geography_scheme = st.selectbox(
        "Geographic segmentation",
        options=GEOGRAPHY_SCHEMES,
        key="raw13_geography_scheme",
        on_change=_change_geography_scheme,
        help=(
            "Borough, policy geography, and mobility environment are competing "
            "ways to segment Taxi Zones; only one can be active at a time."
        ),
    )
with scope2:
    if selected_geography_scheme == "Borough":
        geography_value_options = borough_options
        geography_value_format = str
    elif selected_geography_scheme == "Policy geography":
        geography_value_options = policy_geography_options
        geography_value_format = str
    elif selected_geography_scheme == "Mobility environment":
        geography_value_options = mobility_environment_options
        geography_value_format = format_mobility_regime_cluster_label
    else:
        geography_value_options = []
        geography_value_format = str

    if geography_value_options:
        if st.session_state.get("raw13_geography_value") not in geography_value_options:
            st.session_state["raw13_geography_value"] = geography_value_options[0]
        selected_geography_value = st.selectbox(
            "Segment",
            options=geography_value_options,
            format_func=geography_value_format,
            key="raw13_geography_value",
            on_change=_mark_saved_view_custom,
            help=(
                "Mobility-environment membership is period-aware; a policy-period "
                "comparison includes zones assigned in either period."
                if selected_geography_scheme == "Mobility environment"
                else None
            ),
        )
    else:
        selected_geography_value = None
        st.caption("All Taxi Zones are included.")

if selected_time_view == "Custom dates":
    date1, date2 = st.columns(2)
    with date1:
        selected_start_date = st.date_input(
            "Start date",
            min_value=date(2023, 1, 1),
            max_value=date(2026, 3, 31),
            key="raw13_start_date",
            on_change=_mark_saved_view_custom,
        )
    with date2:
        selected_end_date = st.date_input(
            "End date",
            min_value=date(2023, 1, 1),
            max_value=date(2026, 3, 31),
            key="raw13_end_date",
            on_change=_mark_saved_view_custom,
        )
else:
    selected_start_date = st.session_state["raw13_start_date"]
    selected_end_date = st.session_state["raw13_end_date"]

if selected_start_date > selected_end_date:
    st.warning("Start date must be on or before end date.")
    st.stop()

custom_table = _build_custom_table(
    selected_family,
    selected_time_view,
    selected_bucket,
    selected_start_date,
    selected_end_date,
)

if selected_geography_scheme == "Borough":
    custom_table = custom_table.loc[
        custom_table["borough"].eq(selected_geography_value)
    ].copy()
elif selected_geography_scheme == "Policy geography":
    custom_table = custom_table.loc[
        custom_table["policy_geography"].eq(selected_geography_value)
    ].copy()
elif selected_geography_scheme == "Mobility environment":
    mobility_environment_zone_ids = set(
        _get_mobility_environment_zone_ids(
            int(selected_geography_value),
            selected_time_view,
            selected_start_date,
            selected_end_date,
        )
    )
    custom_table = custom_table.loc[
        custom_table["taxi_zone_id"].astype(int).isin(
            mobility_environment_zone_ids
        )
    ].copy()

custom_map_table = custom_table.loc[
    custom_table["taxi_zone_id"].astype(int).isin(geometry_zone_ids)
].copy()

if custom_table.empty:
    st.info("No eligible observations were available for this combination of controls.")
    st.stop()

scope_label = (
    "Post-CP minus Pre-CP incidence"
    if selected_time_view == "Policy-period change"
    else f"{selected_time_view} incidence"
)
geography_scope_label = (
    ALL_ZONES
    if selected_geography_scheme == ALL_ZONES
    else (
        format_mobility_regime_cluster_label(selected_geography_value)
        if selected_geography_scheme == "Mobility environment"
        else f"{selected_geography_scheme}: {selected_geography_value}"
    )
)
st.markdown(f"### {scope_label} · {selected_family}")
st.caption(
    f"{_format_temporal_bucket(selected_bucket)} · {geography_scope_label} · "
    f"{len(custom_table):,} zones in scope"
)

scope_insights = _custom_scope_insights(
    custom_table,
    time_view=selected_time_view,
)
if not scope_insights["zones"]:
    st.info("No comparable incidence values were available for this scope.")
    st.stop()
insight1, insight2, insight3, insight4 = st.columns(4)
if selected_time_view == "Policy-period change":
    scope_increase = scope_insights["increase"]
    scope_decrease = scope_insights["decrease"]
    increase_card_label = (
        "Largest increase" if scope_increase["value"] > 0 else "Smallest decrease"
    )
    decrease_card_label = (
        "Largest decrease" if scope_decrease["value"] < 0 else "Smallest increase"
    )
    insight1.metric(
        increase_card_label,
        str(scope_increase["zone"]),
        delta=f"{scope_increase['value']:+.1f} per 1,000",
    )
    insight2.metric(
        decrease_card_label,
        str(scope_decrease["zone"]),
        delta=f"{scope_decrease['value']:+.1f} per 1,000",
    )
    insight3.metric(
        "Zones increasing",
        f"{scope_insights['increased_count']} of {scope_insights['zones']}",
        delta=f"{scope_insights['increased_share']:.1f}% of scope",
        delta_color="off",
    )
    insight4.metric(
        "Median zone change",
        f"{scope_insights['median']:+.1f}",
        delta="per 1,000",
        delta_color="off",
    )
else:
    scope_peak = scope_insights["peak"]
    insight1.metric(
        "Highest incidence",
        str(scope_peak["zone"]),
        delta=f"{scope_peak['value']:.1f} per 1,000",
        delta_color="off",
    )
    insight2.metric(
        "Median zone incidence",
        f"{scope_insights['median']:.1f}",
        delta="per 1,000",
        delta_color="off",
    )
    insight3.metric(
        "Matching stress anomalies",
        f"{scope_insights['stress_anomalies']:,}",
        delta=f"across {scope_insights['zones']} zones",
        delta_color="off",
    )
    insight4.metric(
        "Upper-quartile threshold",
        f"{scope_insights['upper_quartile']:.1f}",
        delta="per 1,000",
        delta_color="off",
    )

st.session_state.setdefault("raw13_selected_zones", [])
st.session_state.setdefault("raw13_map_revision", 0)
available_zone_ids = custom_table["taxi_zone_id"].dropna().astype(int).tolist()
available_zone_id_set = set(available_zone_ids)
st.session_state["raw13_selected_zones"] = [
    int(value)
    for value in st.session_state["raw13_selected_zones"]
    if int(value) in available_zone_id_set
]

zone_label_lookup = {
    int(row.taxi_zone_id): f"{row.zone} · {row.borough}"
    for row in custom_table[["taxi_zone_id", "zone", "borough"]]
    .drop_duplicates("taxi_zone_id")
    .itertuples(index=False)
}
current_zone_ids = list(st.session_state["raw13_selected_zones"])
selection_summary, clear_selection = st.columns([5, 1])
with selection_summary:
    if current_zone_ids:
        selected_labels = [
            zone_label_lookup.get(zone_id, f"Zone {zone_id}")
            for zone_id in current_zone_ids
        ]
        visible_labels = selected_labels[:8]
        remaining_label = (
            f" · +{len(selected_labels) - 8} more"
            if len(selected_labels) > 8
            else ""
        )
        st.caption(
            f"Selected ({len(current_zone_ids)}): "
            + " · ".join(visible_labels)
            + remaining_label
        )
    else:
        st.caption("Selected (0): click a map polygon or use the selector below.")
with clear_selection:
    st.button(
        "Clear selections",
        on_click=_clear_zone_selection,
        disabled=not current_zone_ids,
        width="stretch",
    )

if custom_map_table.empty:
    st.info(
        "The matching analysis zones do not have an exact drawable polygon. "
        "Use the Taxi Zone selector below to inspect them."
    )
    map_selection = None
else:
    map_selection = st.plotly_chart(
        _build_explorer_map(
            custom_map_table,
            zone_geojson,
            time_view=selected_time_view,
            selected_zone_ids=tuple(current_zone_ids),
        ),
        width="stretch",
        config={"displayModeBar": False},
        key=f"raw13_custom_map_{st.session_state['raw13_map_revision']}",
        on_select="rerun",
        selection_mode="points",
    )
    _render_chart_takeaway(
        _scope_narrative(
            _custom_scope_insights(
                custom_map_table,
                time_view=selected_time_view,
            ),
            time_view=selected_time_view,
            family=selected_family,
            geography_label=geography_scope_label,
        )
    )

try:
    selected_points = map_selection.selection.points
except (AttributeError, TypeError):
    selected_points = []

clicked_zone_ids: list[int] = []
for point in selected_points:
    customdata = point.get("customdata") if isinstance(point, dict) else None
    if customdata is not None and len(customdata) > 0:
        try:
            clicked_zone_ids.append(int(customdata[0]))
        except (TypeError, ValueError):
            continue

if clicked_zone_ids:
    updated_ids = list(current_zone_ids)
    for clicked_zone_id in dict.fromkeys(clicked_zone_ids):
        if clicked_zone_id not in available_zone_id_set:
            continue
        if clicked_zone_id in updated_ids:
            updated_ids.remove(clicked_zone_id)
        else:
            updated_ids.append(clicked_zone_id)
    st.session_state["raw13_selected_zones"] = updated_ids
    st.session_state["raw13_map_revision"] += 1
    st.rerun()

selected_zone_ids = st.multiselect(
    "Taxi Zones to inspect",
    options=available_zone_ids,
    format_func=lambda value: zone_label_lookup.get(int(value), f"Zone {value}"),
    key="raw13_selected_zones",
    on_change=_zone_multiselect_changed,
    help=(
        "Click map polygons or select any number of Taxi Zones. One zone opens "
        "a focused profile; multiple zones produce a comparison. The monthly "
        "line chart displays the first five selections."
    ),
)

ranking_source = custom_table
ranking_scope_suffix = ""
if selected_zone_ids:
    selected_zone_id_set = {int(value) for value in selected_zone_ids}
    ranking_source = custom_table.loc[
        custom_table["taxi_zone_id"].astype(int).isin(selected_zone_id_set)
    ].copy()
    ranking_scope_suffix = " · selected zones"

if selected_time_view == "Policy-period change":
    rank_options = [
        "Highest change",
        "Lowest change",
        "Largest absolute change",
    ]
else:
    rank_options = ["Highest incidence"]

ranking_heading, ranking_control = st.columns([4, 1.4], vertical_alignment="bottom")
with ranking_control:
    rank_by = (
        st.selectbox("Rank zones by", options=rank_options)
        if len(rank_options) > 1
        else rank_options[0]
    )
with ranking_heading:
    st.markdown(f"### {rank_by}{ranking_scope_suffix}")

ranking_figure, ranking_table = _build_ranking_chart(
    ranking_source,
    time_view=selected_time_view,
    rank_by=rank_by,
)
if selected_zone_ids:
    st.caption(
        f"Ranking {len(ranking_source):,} selected zones; clear the selection "
        "to restore the full in-scope ranking."
    )
st.plotly_chart(
    ranking_figure,
    width="stretch",
    config={"displayModeBar": False},
)
_render_chart_takeaway(
    _ranking_narrative(
        ranking_source,
        time_view=selected_time_view,
        rank_by=rank_by,
    )
)

with st.expander("View ranked zone data", expanded=False):
    st.dataframe(ranking_table, width="stretch", hide_index=True)


st.divider()
st.markdown("## Investigate selected Taxi Zones")

if not selected_zone_ids:
    st.info(
        "Click a polygon on the custom map or use the Taxi Zone multiselect "
        "to open the zone-level stress-anomaly drill-down."
    )
else:
    selected_zone_ids_tuple = tuple(int(value) for value in selected_zone_ids)
    selected_comparison = custom_table.loc[
        custom_table["taxi_zone_id"].astype(int).isin(selected_zone_ids_tuple)
    ].copy()

    comparison_columns = [
        "taxi_zone_id",
        "zone",
        "borough",
        "policy_geography",
    ]
    if selected_time_view == "Policy-period change":
        comparison_columns.extend(
            ["pre_incidence_per_1k", "post_incidence_per_1k", "value"]
        )
    else:
        comparison_columns.extend(
            ["eligible_observations", "stress_anomalies", "value"]
        )

    comparison_display = selected_comparison[comparison_columns].copy().rename(
        columns={
            "pre_incidence_per_1k": "Pre-CP per 1,000",
            "post_incidence_per_1k": "Post-CP per 1,000",
            "eligible_observations": "Eligible observations",
            "stress_anomalies": "Stress anomalies",
            "value": (
                "Change per 1,000"
                if selected_time_view == "Policy-period change"
                else "Incidence per 1,000"
            ),
        }
    )
    rate_columns = [
        column for column in comparison_display.columns if "per 1,000" in column
    ]
    comparison_display[rate_columns] = comparison_display[rate_columns].round(1)

    if len(selected_zone_ids_tuple) == 1:
        selected_zone_name = comparison_display.iloc[0]["zone"]
        st.markdown(f"### {selected_zone_name} stress profile")
    else:
        st.markdown(f"### Comparing {len(selected_zone_ids_tuple)} Taxi Zones")

    st.dataframe(comparison_display, width="stretch", hide_index=True)

    if len(selected_zone_ids_tuple) == 1:
        selected_row = selected_comparison.iloc[0]
        if selected_time_view == "Policy-period change":
            selected_narrative = (
                f"{selected_row['zone']} moved from "
                f"{selected_row['pre_incidence_per_1k']:.1f} Pre-CP to "
                f"{selected_row['post_incidence_per_1k']:.1f} Post-CP stress "
                f"anomalies per 1,000—a change of {selected_row['value']:+.1f}."
            )
        else:
            selected_narrative = (
                f"{selected_row['zone']} recorded {int(selected_row['stress_anomalies']):,} "
                f"matching stress anomalies across "
                f"{int(selected_row['eligible_observations']):,} eligible observations, "
                f"or {selected_row['value']:.1f} per 1,000."
            )
    else:
        selected_insights = _custom_scope_insights(
            selected_comparison,
            time_view=selected_time_view,
        )
        selected_narrative = _scope_narrative(
            selected_insights,
            time_view=selected_time_view,
            family=selected_family,
            geography_label="the selected zones",
        )

    _render_chart_takeaway(selected_narrative)

    timeline_zone_ids = selected_zone_ids_tuple[:5]
    monthly_timeline = _build_zone_monthly_timeline(
        timeline_zone_ids,
        selected_family,
        selected_time_view,
        selected_bucket,
        selected_start_date,
        selected_end_date,
    )
    if not monthly_timeline.empty:
        st.markdown("#### Monthly incidence")
        if len(selected_zone_ids_tuple) > 5:
            timeline_zone_names = [
                zone_label_lookup.get(zone_id, f"Zone {zone_id}").split(" · ")[0]
                for zone_id in timeline_zone_ids
            ]
            st.caption(
                "Showing the first five selected zones: "
                + ", ".join(timeline_zone_names)
                + ". All selected zones remain included in the tables below."
            )
        st.plotly_chart(
            _build_zone_timeline_chart(
                monthly_timeline,
                selected_zone_count=len(timeline_zone_ids),
            ),
            width="stretch",
            config={"displayModeBar": False},
        )
        _render_chart_takeaway(_timeline_narrative(monthly_timeline))

    if len(selected_zone_ids_tuple) == 1:
        all_family_zone_events = _load_zone_event_rows(
            selected_zone_ids_tuple,
            "All stress anomalies",
            selected_time_view,
            selected_bucket,
            selected_start_date,
            selected_end_date,
        )
        if not all_family_zone_events.empty:
            family_summary = (
                all_family_zone_events.groupby(FAMILY_COLUMN, observed=True)
                .size()
                .rename("Stress anomalies")
                .reset_index()
                .rename(columns={FAMILY_COLUMN: "Stress family"})
            )
            family_summary["Share"] = (
                family_summary["Stress anomalies"]
                .div(family_summary["Stress anomalies"].sum())
                .mul(100)
                .round(1)
                .map(lambda value: f"{value:.1f}%")
            )
            bucket_summary = (
                all_family_zone_events.groupby("temporal_bucket", observed=True)
                .size()
                .rename("Stress anomalies")
                .reset_index()
                .sort_values("Stress anomalies", ascending=False)
                .head(10)
                .rename(columns={"temporal_bucket": "Temporal bucket"})
            )
            bucket_summary["Temporal bucket"] = bucket_summary[
                "Temporal bucket"
            ].map(_format_temporal_bucket)

            mix_column, bucket_column = st.columns(2)
            with mix_column:
                st.markdown("#### Stress-family mix")
                st.dataframe(family_summary, width="stretch", hide_index=True)
            with bucket_column:
                st.markdown("#### Leading temporal buckets")
                st.dataframe(bucket_summary, width="stretch", hide_index=True)

    zone_events = _load_zone_event_rows(
        selected_zone_ids_tuple,
        selected_family,
        selected_time_view,
        selected_bucket,
        selected_start_date,
        selected_end_date,
    )
    st.markdown("#### Stress-anomaly events")
    st.caption(
        f"{len(zone_events):,} selected events match the current spatial-explorer scope."
    )

    if zone_events.empty:
        st.info("No stress-anomaly events matched the selected zones and controls.")
    else:
        display_events = _event_display_table(zone_events.head(500))
        st.dataframe(display_events, width="stretch", hide_index=True, height=360)
        if len(zone_events) > 500:
            st.caption("Showing the 500 most recent matching events.")

        event_label_lookup: dict[str, str] = {}
        for row in zone_events.head(2_000).itertuples(index=False):
            event_id = str(getattr(row, EVENT_ID_COLUMN))
            event_date = pd.Timestamp(getattr(row, "date")).strftime("%b %d, %Y")
            zone_name = str(getattr(row, "zone", f"Zone {getattr(row, 'taxi_zone_id')}"))
            daypart = str(getattr(row, "daypart", getattr(row, "temporal_bucket", "")))
            family_value = str(getattr(row, FAMILY_COLUMN))
            event_label_lookup[event_id] = (
                f"{event_date} · {zone_name} · {daypart} · {family_value}"
            )

        event_options = list(event_label_lookup)
        selected_event_id = st.selectbox(
            "Event to diagnose",
            options=event_options,
            format_func=lambda value: event_label_lookup[value],
            help="Choose one Taxi Zone × date × daypart event for Page 15.",
        )
        st.session_state["raw15_selected_event_id"] = selected_event_id

        if EVENT_PROFILER_PAGE.exists():
            st.page_link(
                "views/raw_15_stress_anomaly_event_profiler.py",
                label="Open this event in Stress Anomaly Event Profiler",
                icon=":material/troubleshoot:",
            )
        else:
            st.button(
                "Open this event in Stress Anomaly Event Profiler",
                disabled=True,
                help=(
                    "The selected event ID has been saved, but Page 15 "
                    "could not be found at the expected path."
                ),
            )
            st.caption(
                "The selected event is ready to pass to Stress Anomaly "
                "Event Profiler once the page is available."
            )
