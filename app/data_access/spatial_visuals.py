"""Spatial visual and interpretation helpers for Raw View 03."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from shapely import wkb

from app.data_access.loaders import METRIC_LABELS
from app.utils.project_branding import BRAND_COLORS, apply_branding


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_TABLE_DIR = REPO_ROOT / "data" / "processed" / "app_tables"
SIMPLIFIED_GEOJSON_PATH = (
    APP_TABLE_DIR / "taxi_zones_simplified.geojson"
)
SOURCE_GEOMETRY_PATH = (
    REPO_ROOT
    / "data"
    / "processed"
    / "1.3.1.final_tables"
    / "nyc_taxi_zones_harmonized.parquet"
)

DEMAND_METRICS = [
    "taxi_trip_count",
    "subway_ridership",
    "fhvhv_trip_count",
]

MEANINGFUL_BASELINE_THRESHOLDS = {
    "taxi_trip_count": 25.0,
    "taxi_avg_trip_speed": 100.0,
    "fhvhv_trip_count": 100.0,
    "fhvhv_avg_trip_speed": 500.0,
    "subway_ridership": 500.0,
    "avg_bus_speed": 25.0,
}

SIMPLIFIED_AGREEMENT_ORDER = [
    "All modes increased",
    "Divergent demand pattern",
    "Other eligible pattern",
    "Insufficient coverage",
]

SIMPLIFIED_AGREEMENT_COLORS = {
    "All modes increased": BRAND_COLORS["dark_teal"],
    "Divergent demand pattern": BRAND_COLORS["terracotta"],
    "Other eligible pattern": BRAND_COLORS["seafoam"],
    "Insufficient coverage": "#E5E7E9",
}

CONTINUOUS_COLOR_SCALE = [
    [0.00, BRAND_COLORS["terracotta"]],
    [0.35, BRAND_COLORS["pale_peach"]],
    [0.50, BRAND_COLORS["ice"]],
    [0.65, BRAND_COLORS["seafoam"]],
    [1.00, BRAND_COLORS["dark_teal"]],
]

MAP_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToRemove": [
        "select2d",
        "lasso2d",
        "toImage",
    ],
}


def _decode_geometry(value):
    if isinstance(value, (bytes, bytearray, memoryview)):
        return wkb.loads(bytes(value))
    return value


def _format_number(
    value: object,
    *,
    signed: bool = False,
    percent: bool = False,
) -> str:
    if pd.isna(value):
        return "Unavailable"

    numeric = float(value)
    sign = "+" if signed else ""
    suffix = "%" if percent else ""
    return f"{numeric:{sign},.2f}{suffix}"


def _format_boolean(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    return "Yes" if bool(value) else "No"


def _bucket_text(temporal_bucket: str) -> str:
    if temporal_bucket == "All temporal buckets":
        return "across all temporal buckets"
    return f"during {temporal_bucket.replace('_', ' ')}"


def build_simplified_zone_geojson(
    *,
    output_path: Path = SIMPLIFIED_GEOJSON_PATH,
    simplify_tolerance_feet: float = 75.0,
) -> Path:
    raw = pd.read_parquet(SOURCE_GEOMETRY_PATH)

    join_key = (
        "canonical_location_id"
        if "canonical_location_id" in raw.columns
        else "location_id"
        if "location_id" in raw.columns
        else "taxi_zone_id"
    )

    geometry_df = raw.loc[
        raw[join_key].notna() & raw["geometry"].notna(),
        [
            column
            for column in [join_key, "zone", "borough", "geometry"]
            if column in raw.columns
        ],
    ].copy()

    geometry_df[join_key] = geometry_df[join_key].astype(int)
    geometry_df["geometry"] = geometry_df["geometry"].apply(
        _decode_geometry
    )

    gdf = gpd.GeoDataFrame(
        geometry_df,
        geometry="geometry",
        crs="EPSG:4326",
    ).drop_duplicates(subset=[join_key])

    projected = gdf.to_crs("EPSG:2263")
    projected["geometry"] = projected.geometry.simplify(
        simplify_tolerance_feet,
        preserve_topology=True,
    )
    gdf = projected.to_crs("EPSG:4326")
    gdf = gdf.rename(columns={join_key: "taxi_zone_id"})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(gdf.to_json(), encoding="utf-8")

    return output_path


@st.cache_data(show_spinner=False)
def get_zone_geojson() -> dict:
    if not SIMPLIFIED_GEOJSON_PATH.exists():
        build_simplified_zone_geojson()

    return json.loads(
        SIMPLIFIED_GEOJSON_PATH.read_text(encoding="utf-8")
    )


def add_reliability_flags(summary_df: pd.DataFrame) -> pd.DataFrame:
    result = summary_df.copy()
    result["reliability_threshold"] = result["metric"].map(
        MEANINGFUL_BASELINE_THRESHOLDS
    )
    result["eligible_for_percent_change"] = (
        result["has_both_periods"]
        & result["percent_change"].notna()
        & result["pre_support_daily_average"].notna()
        & result["reliability_threshold"].notna()
        & result["pre_support_daily_average"].ge(
            result["reliability_threshold"]
        )
    )
    result["thresholded_percent_change"] = result["percent_change"].where(
        result["eligible_for_percent_change"]
    )
    return result


def calculate_robust_symmetric_bound(
    values: pd.Series,
    *,
    percentile: float = 0.95,
    minimum: float = 1.0,
) -> float:
    clean = values.dropna().abs()
    if clean.empty:
        return minimum

    bound = float(clean.quantile(percentile))
    return max(bound, minimum) if np.isfinite(bound) else minimum


def build_demand_agreement_table(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    reliable = add_reliability_flags(summary_df)
    reliable = reliable[reliable["metric"].isin(DEMAND_METRICS)]

    index_columns = [
        "taxi_zone_id",
        "zone",
        "borough",
        "cbd_spatial_category",
    ]

    wide = reliable.pivot_table(
        index=index_columns,
        columns="metric",
        values="thresholded_percent_change",
        aggfunc="first",
    ).reset_index()

    for metric in DEMAND_METRICS:
        if metric not in wide.columns:
            wide[metric] = np.nan

    complete = wide[DEMAND_METRICS].notna().all(axis=1)
    all_up = complete & wide[DEMAND_METRICS].gt(0).all(axis=1)
    divergent = (
        complete
        & wide["taxi_trip_count"].gt(0)
        & wide["subway_ridership"].gt(0)
        & wide["fhvhv_trip_count"].lt(0)
    )

    wide["agreement_category"] = np.select(
        [~complete, all_up, divergent],
        [
            "Insufficient coverage",
            "All modes increased",
            "Divergent demand pattern",
        ],
        default="Other eligible pattern",
    )

    wide["agreement_category"] = pd.Categorical(
        wide["agreement_category"],
        categories=SIMPLIFIED_AGREEMENT_ORDER,
        ordered=True,
    )

    return wide


def get_agreement_category_counts(
    agreement_df: pd.DataFrame,
) -> pd.DataFrame:
    counts = (
        agreement_df.groupby(
            "agreement_category",
            observed=False,
        )
        .size()
        .rename("zone_count")
        .reset_index()
    )

    total = counts["zone_count"].sum()
    counts["zone_share"] = np.where(
        total > 0,
        counts["zone_count"] / total,
        np.nan,
    )

    return counts[counts["zone_count"] > 0].reset_index(drop=True)


def build_agreement_map(
    agreement_df: pd.DataFrame,
    *,
    title: str,
    height: int = 650,
) -> go.Figure:
    plot_df = agreement_df.copy()

    plot_df["hover_taxi"] = plot_df["taxi_trip_count"].map(
        lambda value: _format_number(
            value,
            signed=True,
            percent=True,
        )
    )
    plot_df["hover_subway"] = plot_df["subway_ridership"].map(
        lambda value: _format_number(
            value,
            signed=True,
            percent=True,
        )
    )
    plot_df["hover_fhvhv"] = plot_df["fhvhv_trip_count"].map(
        lambda value: _format_number(
            value,
            signed=True,
            percent=True,
        )
    )

    fig = px.choropleth_mapbox(
        plot_df,
        geojson=get_zone_geojson(),
        locations="taxi_zone_id",
        featureidkey="properties.taxi_zone_id",
        color="agreement_category",
        category_orders={
            "agreement_category": SIMPLIFIED_AGREEMENT_ORDER,
        },
        color_discrete_map=SIMPLIFIED_AGREEMENT_COLORS,
        custom_data=[
            "zone",
            "borough",
            "agreement_category",
            "hover_taxi",
            "hover_subway",
            "hover_fhvhv",
        ],
        mapbox_style="carto-positron",
        center={"lat": 40.7128, "lon": -74.0060},
        zoom=9.0,
        opacity=0.84,
        title=title,
    )

    fig = apply_branding(fig)

    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Borough: %{customdata[1]}<br>"
            "Pattern: %{customdata[2]}<br>"
            "Taxi trips: %{customdata[3]}<br>"
            "Subway ridership: %{customdata[4]}<br>"
            "FHVHV trips: %{customdata[5]}"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        height=height,
        margin={"l": 0, "r": 0, "t": 72, "b": 150},
        legend={
            "title": {"text": ""},
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.09,
            "yanchor": "top",
            "entrywidth": 175,
            "entrywidthmode": "pixels",
        },
    )

    return fig


def build_continuous_zone_map(
    summary_df: pd.DataFrame,
    *,
    metric: str,
    value_column: str,
    title: str,
    apply_threshold: bool,
    height: int = 665,
) -> go.Figure:
    reliable = add_reliability_flags(summary_df)
    metric_df = reliable[reliable["metric"].eq(metric)].copy()

    if value_column == "percent_change" and apply_threshold:
        metric_df["map_value"] = metric_df[
            "thresholded_percent_change"
        ]
    else:
        metric_df["map_value"] = metric_df[value_column]

    metric_df["hover_pre"] = metric_df["pre_daily_average"].map(
        _format_number
    )
    metric_df["hover_post"] = metric_df["post_daily_average"].map(
        _format_number
    )
    metric_df["hover_absolute"] = metric_df["absolute_change"].map(
        lambda value: _format_number(value, signed=True)
    )
    metric_df["hover_percent"] = metric_df["percent_change"].map(
        lambda value: _format_number(
            value,
            signed=True,
            percent=True,
        )
    )
    metric_df["hover_support"] = metric_df[
        "pre_support_daily_average"
    ].map(_format_number)
    metric_df["hover_eligible"] = metric_df[
        "eligible_for_percent_change"
    ].map(_format_boolean)

    change_mode = value_column in {
        "absolute_change",
        "percent_change",
    }

    if change_mode:
        bound = calculate_robust_symmetric_bound(
            metric_df["map_value"]
        )
        range_color = (-bound, bound)
        color_scale = CONTINUOUS_COLOR_SCALE
    else:
        range_color = None
        color_scale = [
            [0.0, BRAND_COLORS["ice"]],
            [0.45, BRAND_COLORS["seafoam"]],
            [1.0, BRAND_COLORS["dark_teal"]],
        ]

    fig = px.choropleth_mapbox(
        metric_df,
        geojson=get_zone_geojson(),
        locations="taxi_zone_id",
        featureidkey="properties.taxi_zone_id",
        color="map_value",
        color_continuous_scale=color_scale,
        range_color=range_color,
        custom_data=[
            "zone",
            "borough",
            "hover_pre",
            "hover_post",
            "hover_absolute",
            "hover_percent",
            "hover_support",
            "hover_eligible",
        ],
        mapbox_style="carto-positron",
        center={"lat": 40.7128, "lon": -74.0060},
        zoom=9.0,
        opacity=0.84,
        title=title,
    )

    colorbar_title = {
        "pre_daily_average": "Pre-CP daily avg.",
        "post_daily_average": "Post-CP daily avg.",
        "absolute_change": "Daily avg. change",
        "percent_change": "% change",
    }[value_column]

    fig = apply_branding(fig)

    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Borough: %{customdata[1]}<br>"
            "Pre-CP daily average: %{customdata[2]}<br>"
            "Post-CP daily average: %{customdata[3]}<br>"
            "Daily-average change: %{customdata[4]}<br>"
            "Percent change: %{customdata[5]}<br>"
            "Pre-CP support average: %{customdata[6]}<br>"
            "Threshold eligible: %{customdata[7]}"
            "<extra></extra>"
        )
    )

    fig.update_coloraxes(
        colorbar={
            "title": {"text": colorbar_title},
            "ticksuffix": "%" if value_column == "percent_change" else "",
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.12,
            "yanchor": "top",
            "len": 0.58,
            "thickness": 14,
        }
    )
    fig.update_layout(
        height=height,
        margin={"l": 0, "r": 0, "t": 72, "b": 125},
    )

    return fig


def get_continuous_map_kpis(
    summary_df: pd.DataFrame,
    *,
    metric: str,
    value_column: str,
    apply_threshold: bool,
) -> dict[str, object]:
    reliable = add_reliability_flags(summary_df)
    metric_df = reliable[reliable["metric"].eq(metric)].copy()

    if value_column == "percent_change" and apply_threshold:
        metric_df = metric_df[
            metric_df["eligible_for_percent_change"]
        ]

    metric_df = metric_df[metric_df[value_column].notna()]

    if metric_df.empty:
        return {
            "zones_shown": 0,
            "zones_increasing": 0,
            "zones_decreasing": 0,
            "median_change": np.nan,
            "largest_zone": "Unknown",
            "largest_value": np.nan,
        }

    largest = metric_df.nlargest(1, value_column).iloc[0]

    return {
        "zones_shown": int(metric_df["taxi_zone_id"].nunique()),
        "zones_increasing": int((metric_df[value_column] > 0).sum()),
        "zones_decreasing": int((metric_df[value_column] < 0).sum()),
        "median_change": float(metric_df[value_column].median()),
        "largest_zone": largest["zone"],
        "largest_value": float(largest[value_column]),
    }


def build_hero_interpretation(
    summary_df: pd.DataFrame,
    *,
    story: str,
) -> str:
    """Return a curated, data-backed interpretation for the hero story."""
    if story == "All modes":
        agreement = build_demand_agreement_table(summary_df)
        counts = get_agreement_category_counts(agreement)
        lookup = dict(
            zip(
                counts["agreement_category"].astype(str),
                counts["zone_count"],
            )
        )
        eligible = int(
            agreement[DEMAND_METRICS].notna().all(axis=1).sum()
        )
        all_up = int(lookup.get("All modes increased", 0))
        divergent = int(
            lookup.get("Divergent demand pattern", 0)
        )

        return (
            f"Two multimodal patterns dominated among the <strong>{eligible:,}</strong> "
            f"fully eligible zones. <strong>{all_up:,}</strong> increased across all three "
            f"demand modes, while <strong>{divergent:,}</strong> followed the divergent "
            f"pattern: Taxi and Subway increased as FHVHV declined."
        )

    config = {
        "Taxi volume": (
            "taxi_trip_count",
            "absolute_change",
            False,
        ),
        "Taxi growth": (
            "taxi_trip_count",
            "percent_change",
            True,
        ),
        "FHVHV shift": (
            "fhvhv_trip_count",
            "percent_change",
            True,
        ),
        "Subway growth": (
            "subway_ridership",
            "percent_change",
            True,
        ),
    }

    metric, value_column, apply_threshold = config[story]
    kpis = get_continuous_map_kpis(
        summary_df,
        metric=metric,
        value_column=value_column,
        apply_threshold=apply_threshold,
    )

    largest = _format_number(
        kpis["largest_value"],
        signed=True,
        percent=value_column == "percent_change",
    )
    median = _format_number(
        kpis["median_change"],
        signed=True,
        percent=value_column == "percent_change",
    )

    if story == "Taxi volume":
        return (
            f"The largest added Taxi volumes clustered in Manhattan. "
            f"<strong>{kpis['largest_zone']}</strong> recorded the biggest daily-average "
            f"increase ({largest}), while {kpis['zones_increasing']:,} zones "
            f"increased overall."
        )

    if story == "Taxi growth":
        return (
            f"Threshold-eligible Taxi growth extended well beyond the Manhattan "
            f"core. The median eligible-zone change was <strong>{median}</strong>, with "
            f"<strong>{kpis['largest_zone']}</strong> recording the strongest proportional gain."
        )

    if story == "FHVHV shift":
        return (
            f"FHVHV activity split geographically: many outer-borough zones "
            f"increased while a substantial group of Manhattan core zones "
            f"declined. The median eligible-zone change was <strong>{median}</strong>."
        )

    return (
        f"Subway ridership increased across most reliably observed zones. "
        f"{kpis['zones_increasing']:,} of {kpis['zones_shown']:,} eligible zones "
        f"rose, with a median change of <strong>{median}</strong>."
    )


def build_explorer_interpretation(
    summary_df: pd.DataFrame,
    *,
    metric: str,
    value_column: str,
    temporal_bucket: str,
    apply_threshold: bool,
) -> str:
    """Return a dynamic descriptive insight for the explorer."""
    kpis = get_continuous_map_kpis(
        summary_df,
        metric=metric,
        value_column=value_column,
        apply_threshold=apply_threshold,
    )

    label = METRIC_LABELS.get(metric, metric)
    bucket_text = _bucket_text(temporal_bucket)

    if kpis["zones_shown"] == 0:
        return (
            f"{label} does not have enough observed data to summarize "
            f"{bucket_text}."
        )

    median = _format_number(
        kpis["median_change"],
        signed=value_column in {"absolute_change", "percent_change"},
        percent=value_column == "percent_change",
    )
    largest = _format_number(
        kpis["largest_value"],
        signed=value_column in {"absolute_change", "percent_change"},
        percent=value_column == "percent_change",
    )

    if value_column in {"pre_daily_average", "post_daily_average"}:
        period = (
            "pre-CP"
            if value_column == "pre_daily_average"
            else "post-CP"
        )
        return (
            f"{label} had a median Taxi Zone value of <strong>{median}</strong> in the "
            f"{period} period {bucket_text}. <strong>{kpis['largest_zone']}</strong> recorded "
            f"the highest zone value ({largest})."
        )

    increasing = kpis["zones_increasing"]
    decreasing = kpis["zones_decreasing"]
    shown = kpis["zones_shown"]

    if increasing == shown:
        direction_text = "Every mapped zone increased"
    elif decreasing == shown:
        direction_text = "Every mapped zone decreased"
    elif increasing > decreasing:
        direction_text = (
            f"Most mapped zones increased ({increasing:,} of {shown:,})"
        )
    elif decreasing > increasing:
        direction_text = (
            f"Most mapped zones decreased ({decreasing:,} of {shown:,})"
        )
    else:
        direction_text = (
            f"Mapped zones were evenly split between increases and decreases"
        )

    return (
        f"<strong>{direction_text}</strong> for {label.lower()} {bucket_text}. "
        f"The median zone change was <strong>{median}</strong>, led by "
        f"<strong>{kpis['largest_zone']}</strong> ({largest})."
    )


def get_kpi_labels(value_column: str) -> tuple[str, str, str]:
    """Return map-mode-appropriate KPI labels."""
    if value_column in {"pre_daily_average", "post_daily_average"}:
        return (
            "Zones shown",
            "Median zone value",
            "Highest zone value",
        )

    return (
        "Zones increasing",
        "Median zone change",
        "Largest increase",
    )