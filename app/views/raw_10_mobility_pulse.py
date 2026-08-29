from __future__ import annotations

import json
from datetime import date
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from shapely.geometry import shape

from app.data_access.anomalies import (
    get_metric_driver_anomaly_events,
)
from app.data_access.loaders import (
    BASE_METRICS,
    CONGESTION_PRICING_START_DATE,
    METRIC_LABELS,
    TEMPORAL_BUCKET_ORDER,
)
from app.data_access.spatial_aggregations import (
    ALL_TEMPORAL_BUCKETS_LABEL,
)
from app.data_access.spatial_visuals import (
    BRAND_COLORS,
    MAP_CONFIG,
    apply_branding,
    calculate_robust_symmetric_bound,
    get_zone_geojson,
)
from app.data_access.zone_profile_app_tables import (
    ZONE_PROFILE_DAILY_DIR,
)
from app.data_access.zone_profiles import (
    get_zone_catalog,
)
from app.utils.project_branding import inject_app_css, render_chart_insight


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRESET_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "app_tables"
    / "mobility_pulse_presets"
)

PRESET_MANIFEST_PATH = (
    PRESET_DIR
    / "manifest.json"
)


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

PULSE_METRICS = list(BASE_METRICS)

DEFAULT_METRIC = "taxi_trip_count"

# Saved MP4s use every available observation date.
#
# Custom Plotly animations use sparse frame updates and remain bounded
# so long windows stay responsive in the browser.
CUSTOM_MAX_FRAMES = 240
DEFAULT_CUSTOM_WINDOW_DAYS = 120

DATA_WINDOW_START = pd.Timestamp(
    "2023-01-01"
).date()

DATA_WINDOW_END = pd.Timestamp(
    "2026-03-31"
).date()



# ---------------------------------------------------------------------
# Saved-story editorial copy
# ---------------------------------------------------------------------
#
# These are intentionally not generic descriptions of the controls.
# Each saved animation has a distinct visitor-facing analytical story.

PRESET_STORY_COPY = {
    "full_history_subway_citywide": {
        "tab_label": "Full history",
        "heading": "The city’s transit rhythm changed without disappearing",
        "what_to_watch": (
            "Follow Subway ridership across the full study period. "
            "The seasonal rhythm remains visible, but individual neighborhoods "
            "move above and below their own pre-CP baselines at different times. "
            "Watch for shifts that persist across weeks rather than isolated daily movement."
        ),
    },

    "cp_launch_taxi_citywide": {
        "tab_label": "CP launch",
        "heading": "Taxi activity changed unevenly around the policy launch",
        "what_to_watch": (
            "Watch weekend-evening Taxi activity around January 2025. "
            "Some parts of the city move sharply away from their earlier baseline "
            "while others remain much closer to it. The key pattern is the spatial "
            "difference, not a single citywide rise or fall."
        ),
    },

    "post_cp_bronx_taxi_overnight": {
        "tab_label": "Bronx overnight",
        "heading": "Overnight Taxi activity shifted sharply across the Bronx",
        "what_to_watch": (
            "This view follows weekend-overnight Taxi activity during the first "
            "six months after congestion pricing began. Look for whether changes "
            "appear across several neighboring zones together or remain concentrated "
            "in a handful of places."
        ),
    },

    "early_2026_brooklyn_fhvhv": {
        "tab_label": "Brooklyn FHVHV",
        "heading": "Brooklyn’s for-hire activity moved in different directions",
        "what_to_watch": (
            "Follow weekend-evening high-volume for-hire activity across Brooklyn. "
            "Some neighborhoods move farther from their pre-CP norms than others, "
            "and the stress-anomaly markers help identify dates when those changes were "
            "especially unusual for the displayed metric."
        ),
    },

    "early_2026_queens_bus": {
        "tab_label": "Queens bus",
        "heading": "Weekend bus activity changed in patches across Queens",
        "what_to_watch": (
            "Track weekend-midday Bus activity across Queens. Rather than expecting "
            "the borough to move as one unit, look for groups of zones that repeatedly "
            "sit above or below their own earlier activity levels."
        ),
    },

    "early_2026_manhattan_subway_transfers": {
        "tab_label": "Manhattan transit",
        "heading": "Late-night transfer activity reveals a concentrated Manhattan pattern",
        "what_to_watch": (
            "This view isolates weekend-overnight Subway transfers within Manhattan. "
            "Watch whether unusual activity appears broadly across the borough or "
            "clusters in a smaller set of zones, and whether those patterns persist "
            "from one observation to the next."
        ),
    },

    "early_2026_citywide_subway_transfers": {
        "tab_label": "Citywide transit",
        "heading": "The citywide view puts Manhattan’s late-night pattern in context",
        "what_to_watch": (
            "Now widen the same weekend-overnight Subway-transfer lens to the full city. "
            "Compare this with the Manhattan story to see whether the strongest changes "
            "remain geographically concentrated or appear across multiple boroughs."
        ),
    },
}


# ---------------------------------------------------------------------
# Metric-specific reading cues
# ---------------------------------------------------------------------

PULSE_METRIC_NOTES = {
    "taxi_trip_count": (
        "Look for sustained rises or drops in Taxi activity rather than "
        "isolated spikes."
    ),
    "taxi_avg_trip_speed": (
        "Watch for speed changes that persist across several dates rather "
        "than reacting to one-day movement."
    ),
    "taxi_avg_trip_duration": (
        "Longer Taxi trip durations can reflect slower movement or changing "
        "trip mix; focus on sustained shifts."
    ),
    "fhvhv_trip_count": (
        "Watch for changes in high-volume for-hire activity that hold across "
        "several dates rather than one-off bumps."
    ),
    "fhvhv_avg_trip_speed": (
        "Look for persistent changes in FHVHV travel speed, especially when "
        "several neighboring zones move together."
    ),
    "fhvhv_avg_trip_duration": (
        "Focus on durable changes in FHVHV trip duration rather than isolated "
        "daily jumps."
    ),
    "subway_ridership": (
        "Focus on longer runs of higher or lower Subway ridership, since "
        "day-to-day noise is common."
    ),
    "subway_transfers": (
        "Watch for sustained changes in transfer activity, which can reveal "
        "shifts in how riders move through the network."
    ),
    "bus_trip_count": (
        "Look for persistent movement in Bus activity, especially where "
        "neighboring zones change together."
    ),
    "avg_bus_speed": (
        "Pay attention to repeated Bus speed changes across several dates "
        "rather than isolated daily movement."
    ),
}


# ---------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------

def _format_number(value: object) -> str:
    """Format a compact numeric value for KPI cards."""
    if pd.isna(value):
        return "Unavailable"

    numeric = float(value)

    if abs(numeric) >= 1_000_000:
        return f"{numeric / 1_000_000:,.2f}M"

    if abs(numeric) >= 1_000:
        return f"{numeric / 1_000:,.2f}K"

    return f"{numeric:,.2f}"


def _format_percent(value: object) -> str:
    """Format a signed percentage."""
    if pd.isna(value):
        return "Unavailable"

    return f"{float(value):+,.2f}%"


def _format_temporal_bucket(temporal_bucket: str) -> str:
    """Create a compact app-facing temporal-bucket label."""
    if temporal_bucket == ALL_TEMPORAL_BUCKETS_LABEL:
        return "All temporal buckets"

    return (
        temporal_bucket
        .replace("_", " ")
        .title()
        .replace("Am ", "AM ")
        .replace("Pm ", "PM ")
    )


def _format_preset_period(preset: dict[str, object]) -> str:
    """Format a saved preset's date range."""
    start_date = pd.Timestamp(preset["start_date"])
    end_date = pd.Timestamp(preset["end_date"])

    return (
        f"{start_date:%b %Y}"
        f"–"
        f"{end_date:%b %Y}"
    )


def _metric_notice(metric: str) -> str:
    """Return a metric-specific custom-view reading cue."""
    return PULSE_METRIC_NOTES.get(
        metric,
        (
            "Look for sustained changes and be cautious about reading too "
            "much into small one-day moves."
        ),
    )


# ---------------------------------------------------------------------
# Saved animation loaders
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_preset_manifest() -> list[dict[str, object]]:
    """Load metadata for the pre-rendered Mobility Pulse videos."""
    if not PRESET_MANIFEST_PATH.exists():
        return []

    with PRESET_MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        manifest = json.load(file_handle)

    if not isinstance(manifest, list):
        return []

    return [
        row
        for row in manifest
        if isinstance(row, dict)
        and row.get("preset_id")
        and row.get("filename")
    ]


@st.cache_data(show_spinner=False)
def _load_preset_video_bytes(filename: str) -> bytes:
    """Read one saved MP4 for Streamlit playback."""
    video_path = PRESET_DIR / filename
    return video_path.read_bytes()


# ---------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_zone_centroids() -> pd.DataFrame:
    """Return one centroid per Taxi Zone."""
    geojson = get_zone_geojson()

    records: list[dict[str, object]] = []

    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        geom = feature.get("geometry")

        if not props or not geom:
            continue

        taxi_zone_id = props.get("taxi_zone_id")

        if taxi_zone_id is None:
            continue

        centroid = shape(geom).centroid

        records.append(
            {
                "taxi_zone_id": int(taxi_zone_id),
                "lon": float(centroid.x),
                "lat": float(centroid.y),
            }
        )

    return pd.DataFrame(records)


def _borough_focus_view(
    center_source: pd.DataFrame,
    *,
    selected_borough: str,
) -> tuple[dict[str, float], float]:
    """Return a borough-specific center and zoom."""
    if center_source.empty:
        return {"lat": 40.7128, "lon": -74.0060}, 9.8

    center = {
        "lat": float(center_source["lat"].mean()),
        "lon": float(center_source["lon"].mean()),
    }

    zoom = 10.3

    borough_offsets = {
        "Queens": {
            "lat": -0.05,
            "lon": 0.0,
            "zoom": 9.85,
        },
        "Brooklyn": {
            "lat": 0.01,
            "lon": 0.02,
            "zoom": 9.95,
        },
        "Bronx": {
            "lat": 0.02,
            "lon": 0.0,
            "zoom": 10.0,
        },
        "Manhattan": {
            "lat": 0.01,
            "lon": 0.0,
            "zoom": 10.7,
        },
        "Staten Island": {
            "lat": -0.01,
            "lon": 0.0,
            "zoom": 10.15,
        },
    }

    adjustments = borough_offsets.get(selected_borough)

    if adjustments:
        center["lat"] += adjustments["lat"]
        center["lon"] += adjustments["lon"]
        zoom = adjustments["zoom"]

    return center, zoom


# ---------------------------------------------------------------------
# Mobility Pulse data
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_pulse_panel(
    *,
    metric: str,
    temporal_bucket: str,
) -> pd.DataFrame:
    """Load one metric panel and calculate each zone's pre-CP baseline."""
    columns = [
        "taxi_zone_id",
        "date",
        "zone",
        "borough",
        "cbd_spatial_category",
        metric,
    ]

    frame = pd.read_parquet(
        ZONE_PROFILE_DAILY_DIR,
        engine="pyarrow",
        columns=columns,
        filters=[
            (
                "temporal_bucket",
                "==",
                temporal_bucket,
            ),
        ],
    )

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame[metric] = pd.to_numeric(
        frame[metric],
        errors="coerce",
    )

    pre_reference = (
        frame.loc[
            frame["date"] < CONGESTION_PRICING_START_DATE,
            [
                "taxi_zone_id",
                metric,
            ],
        ]
        .groupby(
            "taxi_zone_id",
            observed=True,
            dropna=False,
        )[metric]
        .mean()
        .rename("pre_reference")
        .reset_index()
    )

    frame = frame.merge(
        pre_reference,
        on="taxi_zone_id",
        how="left",
        validate="many_to_one",
    )

    valid_reference = (
        frame["pre_reference"].notna()
        & frame["pre_reference"].gt(0)
    )

    frame["pulse_change"] = np.where(
        valid_reference,
        (
            frame[metric]
            - frame["pre_reference"]
        )
        / frame["pre_reference"]
        * 100,
        np.nan,
    )

    frame["pulse_index"] = np.where(
        valid_reference,
        frame[metric]
        / frame["pre_reference"]
        * 100,
        np.nan,
    )

    frame["date_label"] = (
        frame["date"]
        .dt
        .strftime("%Y-%m-%d")
    )

    return (
        frame.sort_values(
            [
                "date",
                "taxi_zone_id",
            ]
        )
        .reset_index(drop=True)
    )


def _sample_dates(
    dates: pd.Series,
    *,
    max_frames: int = CUSTOM_MAX_FRAMES,
) -> list[pd.Timestamp]:
    """Sample only the custom Plotly animation to a bounded frame count."""
    unique_dates = list(
        pd.to_datetime(
            dates
            .dropna()
            .drop_duplicates()
        )
        .sort_values()
    )

    if len(unique_dates) <= max_frames:
        return unique_dates

    step = ceil(
        len(unique_dates)
        / max_frames
    )

    sampled = unique_dates[::step]

    if sampled[-1] != unique_dates[-1]:
        sampled.append(unique_dates[-1])

    return sampled


def _clamp_date_window(
    date_window: object,
    *,
    min_date: date,
    max_date: date,
) -> tuple[date, date]:
    """Clamp a date widget result to the supported data window."""
    if (
        not isinstance(
            date_window,
            (tuple, list),
        )
        or len(date_window) != 2
    ):
        return min_date, max_date

    start_date = pd.to_datetime(
        date_window[0]
    ).date()

    end_date = pd.to_datetime(
        date_window[1]
    ).date()

    start_date = max(
        min_date,
        min(
            start_date,
            max_date,
        ),
    )

    end_date = max(
        min_date,
        min(
            end_date,
            max_date,
        ),
    )

    if end_date < start_date:
        return min_date, max_date

    return start_date, end_date


# ---------------------------------------------------------------------
# Anomaly helpers
# ---------------------------------------------------------------------

def _get_metric_anomalies(
    *,
    metric: str,
    temporal_bucket: str,
) -> pd.DataFrame:
    """Load selected-finalist events where the displayed metric was a driver.

    The app's All temporal buckets row is a daily aggregate. Anomaly events
    retain their underlying daypart, so None means do not constrain the
    anomaly source to one temporal bucket.
    """
    anomaly_bucket = (
        None
        if temporal_bucket == ALL_TEMPORAL_BUCKETS_LABEL
        else temporal_bucket
    )

    return get_metric_driver_anomaly_events(
        metric=metric,
        temporal_bucket=anomaly_bucket,
    )


def _collapse_anomalies_for_daily_display(
    anomalies: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse daypart events to one Taxi Zone × date map marker."""
    if anomalies.empty:
        return pd.DataFrame(
            columns=[
                "taxi_zone_id",
                "date",
                "zone",
                "borough",
                "anomaly_event_count",
            ]
        )

    return (
        anomalies.groupby(
            [
                "taxi_zone_id",
                "date",
                "zone",
                "borough",
            ],
            observed=True,
            dropna=False,
            as_index=False,
        )
        .agg(
            anomaly_event_count=(
                "comparison_event_id",
                "nunique",
            ),
        )
        .sort_values(
            [
                "date",
                "taxi_zone_id",
            ]
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Saved story renderer
# ---------------------------------------------------------------------

def _render_saved_story(
    preset: dict[str, object],
) -> None:
    """Render one curated saved Mobility Pulse story."""
    preset_id = str(
        preset["preset_id"]
    )

    story_copy = PRESET_STORY_COPY.get(
        preset_id,
        {},
    )

    heading = story_copy.get(
        "heading",
        preset.get(
            "title",
            "Mobility Pulse",
        ),
    )

    what_to_watch = story_copy.get(
        "what_to_watch",
        preset.get(
            "description",
            "",
        ),
    )

    metric = str(
        preset["metric"]
    )

    metric_label = METRIC_LABELS.get(
        metric,
        metric,
    )

    scope_label = (
        str(preset["borough"])
        if preset.get("borough")
        else "Citywide"
    )

    bucket_label = _format_temporal_bucket(
        str(
            preset["temporal_bucket"]
        )
    )

    period_label = _format_preset_period(
        preset
    )

    st.subheader(heading)

    st.caption(
        (
            f"{metric_label} · "
            f"{scope_label} · "
            f"{bucket_label} · "
            f"{period_label}"
        )
    )

    video_path = (
        PRESET_DIR
        / str(
            preset["filename"]
        )
    )

    if not video_path.exists():
        st.error(
            (
                "This saved animation is listed in the manifest, "
                "but its MP4 is missing."
            )
        )
        return

    st.video(
        _load_preset_video_bytes(
            str(
                preset["filename"]
            )
        ),
        format="video/mp4",
    )

    render_chart_insight(what_to_watch)

    st.caption(
        (
            "Map color shows change from each Taxi Zone's own pre-CP "
            "reference. Orange dots mark stress anomalies where the displayed "
            "metric was one of the event drivers."
        )
    )


# ---------------------------------------------------------------------
# Custom interactive animation
# ---------------------------------------------------------------------

def _build_animation_figure(
    *,
    panel: pd.DataFrame,
    anomalies: pd.DataFrame,
    metric_label: str,
    focus_title: str,
    center: dict[str, float],
    zoom: float,
    show_anomalies: bool,
) -> go.Figure:
    """Build a lightweight custom Plotly animation.

    Static GeoJSON, colorscale, marker styling, and hover templates live only
    on the base traces. Each animation frame updates only values that change
    by date. Panel and anomaly rows are grouped once up front rather than
    repeatedly filtered inside the frame loop.
    """
    zone_centroids = _load_zone_centroids()

    frame_dates = _sample_dates(
        panel["date"]
    )

    if not frame_dates:
        return go.Figure()

    bound = calculate_robust_symmetric_bound(
        panel["pulse_change"]
    )

    zmin = -bound
    zmax = bound

    zone_geojson = get_zone_geojson()

    colorscale = [
        [0.00, BRAND_COLORS["terracotta"]],
        [0.35, BRAND_COLORS["pale_peach"]],
        [0.50, BRAND_COLORS["ice"]],
        [0.65, BRAND_COLORS["seafoam"]],
        [1.00, BRAND_COLORS["dark_teal"]],
    ]

    def _date_annotation(
        date_value: pd.Timestamp,
    ) -> dict:
        """Return the in-map date badge used during animation playback."""
        return {
            "text": pd.Timestamp(
                date_value
            ).strftime(
                "%b %d, %Y"
            ),
            "x": 0.97,
            "y": 0.97,
            "xref": "paper",
            "yref": "paper",
            "xanchor": "right",
            "yanchor": "top",
            "showarrow": False,
            "font": {
                "size": 18,
                "color": BRAND_COLORS[
                    "dark_teal"
                ],
            },
            "bgcolor": (
                "rgba(255,255,255,0.88)"
            ),
            "borderpad": 6,
        }

    # Group once so the frame loop does not rescan the full dataframe for
    # every animation date.
    panel_groups = {
        pd.Timestamp(date_value): group.sort_values(
            "taxi_zone_id"
        ).reset_index(drop=True)
        for date_value, group in panel.groupby(
            "date",
            observed=True,
            sort=False,
        )
    }

    if show_anomalies and not anomalies.empty:
        anomaly_frame = anomalies.merge(
            zone_centroids,
            on="taxi_zone_id",
            how="left",
            validate="many_to_one",
        )

        anomaly_groups = {
            pd.Timestamp(date_value): group.sort_values(
                "taxi_zone_id"
            ).reset_index(drop=True)
            for date_value, group in anomaly_frame.groupby(
                "date",
                observed=True,
                sort=False,
            )
        }

    else:
        anomaly_groups = {}

    empty_anomalies = pd.DataFrame(
        columns=[
            "lat",
            "lon",
            "zone",
            "anomaly_event_count",
        ]
    )

    def _panel_customdata(
        date_panel: pd.DataFrame,
    ) -> np.ndarray:
        return np.stack(
            [
                date_panel["zone"],
                date_panel["borough"],
                date_panel["date_label"],
                date_panel["pulse_index"].round(2),
                date_panel["pulse_change"].round(2),
                date_panel["pre_reference"].round(2),
            ],
            axis=-1,
        )

    def _anomaly_customdata(
        date_anomalies: pd.DataFrame,
    ) -> np.ndarray:
        return np.column_stack(
            [
                date_anomalies.get(
                    "anomaly_event_count",
                    pd.Series(dtype=int),
                ),
            ]
        )

    first_date = pd.Timestamp(
        frame_dates[0]
    )

    first_panel = panel_groups.get(
        first_date,
        pd.DataFrame(),
    )

    first_anomalies = anomaly_groups.get(
        first_date,
        empty_anomalies,
    )

    base_trace = go.Choroplethmap(
        geojson=zone_geojson,
        locations=first_panel["taxi_zone_id"],
        featureidkey="properties.taxi_zone_id",
        z=first_panel["pulse_change"],
        zmin=zmin,
        zmax=zmax,
        colorscale=colorscale,
        marker={
            "line": {
                "width": 0.25,
                "color": "white",
            }
        },
        customdata=_panel_customdata(
            first_panel
        ),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Borough: %{customdata[1]}<br>"
            "Date: %{customdata[2]}<br>"
            f"{metric_label} index: "
            "%{customdata[3]}<br>"
            "Change vs pre-CP average: "
            "%{customdata[4]}%<br>"
            "Pre-CP reference: "
            "%{customdata[5]}"
            "<extra></extra>"
        ),
        colorbar={
            "title": {
                "text": (
                    "% change "
                    "vs pre-CP"
                )
            },
            "ticksuffix": "%",
            "x": 1.02,
            "xanchor": "left",
            "y": 0.5,
            "yanchor": "middle",
            "len": 0.62,
            "thickness": 12,
        },
        name="Mobility pulse",
        showscale=True,
    )

    anomaly_trace = go.Scattermap(
        lat=first_anomalies.get(
            "lat",
            pd.Series(dtype=float),
        ),
        lon=first_anomalies.get(
            "lon",
            pd.Series(dtype=float),
        ),
        mode="markers",
        name="Stress anomaly",
        marker={
            "size": 10,
            "color": BRAND_COLORS["terracotta"],
            "opacity": 0.9,
        },
        text=first_anomalies.get(
            "zone",
            pd.Series(dtype=str),
        ),
        customdata=_anomaly_customdata(
            first_anomalies
        ),
        hovertemplate=(
            "<b>Stress anomaly</b><br>"
            "Zone: %{text}<br>"
            "Stress-anomaly dayparts: "
            "%{customdata[0]:,}"
            "<extra></extra>"
        ),
        showlegend=True,
    )

    frames: list[go.Frame] = []

    # Sparse frame updates: do not repeat GeoJSON, colorscale, style,
    # featureidkey, hover templates, or legend metadata.
    for date_value in frame_dates:
        frame_date = pd.Timestamp(
            date_value
        )

        date_panel = panel_groups.get(
            frame_date,
            pd.DataFrame(),
        )

        date_anomalies = anomaly_groups.get(
            frame_date,
            empty_anomalies,
        )

        frames.append(
            go.Frame(
                name=frame_date.strftime(
                    "%Y-%m-%d"
                ),
                traces=[
                    0,
                    1,
                ],
                data=[
                    go.Choroplethmap(
                        locations=date_panel[
                            "taxi_zone_id"
                        ],
                        z=date_panel[
                            "pulse_change"
                        ],
                        customdata=_panel_customdata(
                            date_panel
                        ),
                    ),
                    go.Scattermap(
                        lat=date_anomalies.get(
                            "lat",
                            pd.Series(dtype=float),
                        ),
                        lon=date_anomalies.get(
                            "lon",
                            pd.Series(dtype=float),
                        ),
                        text=date_anomalies.get(
                            "zone",
                            pd.Series(dtype=str),
                        ),
                        customdata=_anomaly_customdata(
                            date_anomalies
                        ),
                    ),
                ],
                layout=go.Layout(
                    annotations=[
                        _date_annotation(
                            frame_date
                        )
                    ]
                ),
            )
        )

    fig = go.Figure(
        data=[
            base_trace,
            anomaly_trace,
        ],
        frames=frames,
    )

    slider_steps = [
        {
            "args": [
                [
                    pd.Timestamp(
                        date_value
                    ).strftime(
                        "%Y-%m-%d"
                    )
                ],
                {
                    "frame": {
                        "duration": 0,
                        "redraw": True,
                    },
                    "mode": "immediate",
                },
            ],
            "label": pd.Timestamp(
                date_value
            ).strftime(
                "%b %d"
            ),
            "method": "animate",
        }
        for date_value in frame_dates
    ]

    fig.update_layout(
        title={
            "text": focus_title
        },
        annotations=[
            _date_annotation(
                first_date
            )
        ],
        map={
            "style": "carto-positron",
            "center": center,
            "zoom": zoom,
        },
        margin={
            "l": 0,
            "r": 120,
            "t": 60,
            "b": 10,
        },
        height=700,
        hovermode="closest",
        uirevision="mobility-pulse-custom",
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.07,
            "yanchor": "top",
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.02,
                "y": 0.98,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {
                                    "duration": 450,
                                    "redraw": True,
                                },
                                "fromcurrent": True,
                                "transition": {
                                    "duration": 0
                                },
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {
                                    "duration": 0,
                                    "redraw": False,
                                },
                                "mode": "immediate",
                                "transition": {
                                    "duration": 0
                                },
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.08,
                "y": 0.02,
                "len": 0.85,
                "pad": {
                    "b": 10,
                    "t": 30,
                },
                "currentvalue": {
                    "prefix": "Date: ",
                    "font": {
                        "size": 14
                    },
                },
                "steps": slider_steps,
            }
        ],
    )

    return apply_branding(
        fig
    )


# ---------------------------------------------------------------------
# Custom supporting timeline
# ---------------------------------------------------------------------

def _build_timeline_figure(
    *,
    panel: pd.DataFrame,
    anomalies: pd.DataFrame,
    focus_label: str,
    metric_column: str,
    pre_cp_baseline: float,
    show_anomalies: bool,
) -> go.Figure:
    """Build the compact selected-geography timeline."""
    if panel.empty:
        return go.Figure()

    daily = (
        panel.groupby(
            "date",
            observed=True,
            dropna=False,
        )
        .agg(
            focus_value=(
                metric_column,
                "mean",
            ),
            zone_count=(
                "taxi_zone_id",
                "nunique",
            ),
        )
        .reset_index()
        .sort_values("date")
    )

    daily["focus_index"] = np.where(
        (
            pd.notna(pre_cp_baseline)
            and pre_cp_baseline != 0
        ),
        daily["focus_value"]
        / pre_cp_baseline
        * 100,
        np.nan,
    )

    fig = make_subplots(
        specs=[
            [
                {
                    "secondary_y": True
                }
            ]
        ]
    )

    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["focus_index"],
            mode="lines",
            line={
                "color": (
                    BRAND_COLORS[
                        "dark_teal"
                    ]
                ),
                "width": 3,
            },
            name=(
                f"{focus_label} average"
            ),
            hovertemplate=(
                "<b>%{x|%b %d, %Y}</b><br>"
                "Index: %{y:.1f}"
                "<extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    if (
        show_anomalies
        and not anomalies.empty
    ):
        anomaly_series = daily.merge(
            anomalies[
                ["date"]
            ].drop_duplicates(),
            on="date",
            how="inner",
        )

        if not anomaly_series.empty:
            fig.add_trace(
                go.Scatter(
                    x=anomaly_series[
                        "date"
                    ],
                    y=anomaly_series[
                        "focus_index"
                    ],
                    mode="markers",
                    name="Stress anomaly",
                    marker={
                        "size": 10,
                        "symbol": "circle-open",
                        "color": (
                            BRAND_COLORS[
                                "terracotta"
                            ]
                        ),
                        "line": {
                            "color": BRAND_COLORS["terracotta"],
                            "width": 2,
                        },
                    },
                    hovertemplate=(
                        "<b>Stress-anomaly activity</b><br>"
                        "Date: %{x|%b %d, %Y}"
                        "<extra></extra>"
                    ),
                ),
                secondary_y=False,
            )

        anomaly_counts = (
            anomalies.groupby(
                "date",
                observed=True,
                dropna=False,
            )
            .size()
            .rename(
                "anomaly_zone_count"
            )
            .reset_index()
        )

        fig.add_trace(
            go.Bar(
                x=anomaly_counts["date"],
                y=anomaly_counts[
                    "anomaly_zone_count"
                ],
                marker_color=(
                    "rgba(231,111,81,0.30)"
                ),
                name="Zones with stress anomalies",
                hovertemplate=(
                    "<b>%{x|%b %d, %Y}</b><br>"
                    "Zones with stress anomalies: %{y:,}"
                    "<extra></extra>"
                ),
            ),
            secondary_y=True,
        )

    fig.update_layout(
        height=380,
        margin={
            "l": 0,
            "r": 0,
            "t": 10,
            "b": 40,
        },
        title={
            "text": (
                f"{focus_label} over time"
            )
        },
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.18,
            "yanchor": "top",
        },
    )

    fig.update_xaxes(
        title_text="Date"
    )

    fig.update_yaxes(
        title_text="Index (pre-CP = 100)",
        secondary_y=False,
    )

    fig.update_yaxes(
        title_text="Zones with stress anomalies",
        secondary_y=True,
    )

    return apply_branding(fig)


# ---------------------------------------------------------------------
# Custom explorer renderer
# ---------------------------------------------------------------------

def _render_custom_view(
    *,
    selected_metric: str,
    temporal_bucket: str,
    show_anomalies: bool,
    date_window: object,
    map_focus: str,
    selected_borough: str,
    zone_catalog: pd.DataFrame,
) -> None:
    """Generate one explicitly requested custom Mobility Pulse."""
    (
        start_date_raw,
        end_date_raw,
    ) = _clamp_date_window(
        date_window,
        min_date=DATA_WINDOW_START,
        max_date=DATA_WINDOW_END,
    )

    start_date = pd.Timestamp(
        start_date_raw
    )

    end_date = pd.Timestamp(
        end_date_raw
    )

    with st.status(
        "Building your custom Mobility Pulse…",
        expanded=True,
    ) as status:

        st.write(
            "Loading the selected mobility metric and pre-CP reference."
        )
        panel = _load_pulse_panel(
            metric=selected_metric,
            temporal_bucket=(
                temporal_bucket
            ),
        )

        focus_scope_panel = (
            panel.copy()
        )

        if map_focus == "Borough":
            focus_scope_panel = (
                focus_scope_panel[
                    focus_scope_panel[
                        "borough"
                    ]
                    .astype(str)
                    .eq(
                        selected_borough
                    )
                ]
                .copy()
            )

        pre_cp_daily = (
            focus_scope_panel.loc[
                focus_scope_panel["date"]
                < CONGESTION_PRICING_START_DATE
            ]
            .groupby(
                "date",
                observed=True,
                dropna=False,
            )
            .agg(
                focus_value=(
                    selected_metric,
                    "mean",
                ),
            )
            .reset_index()
        )

        focus_pre_baseline = (
            pre_cp_daily["focus_value"].mean()
            if not pre_cp_daily.empty
            else np.nan
        )

        focus_panel = (
            focus_scope_panel[
                focus_scope_panel[
                    "date"
                ]
                .between(
                    start_date,
                    end_date,
                )
            ]
            .copy()
        )

        if focus_panel.empty:
            status.update(
                label=(
                    "No rows matched "
                    "the custom view."
                ),
                state="error",
                expanded=False,
            )

            st.warning(
                (
                    "No rows match the current metric, date window, "
                    "and geography."
                )
            )

            return

        if show_anomalies:
            st.write(
                "Loading metric-linked stress anomalies."
            )

            anomalies = (
                _get_metric_anomalies(
                    metric=selected_metric,
                    temporal_bucket=(
                        temporal_bucket
                    ),
                )
                .copy()
            )

            if (
                map_focus == "Borough"
                and not anomalies.empty
            ):
                anomalies = (
                    anomalies[
                        anomalies[
                            "borough"
                        ]
                        .astype(str)
                        .eq(
                            selected_borough
                        )
                    ]
                    .copy()
                )

            if not anomalies.empty:
                anomalies["date"] = (
                    pd.to_datetime(
                        anomalies["date"]
                    )
                )

                anomalies = (
                    anomalies[
                        anomalies["date"]
                        .between(
                            start_date,
                            end_date,
                        )
                    ]
                    .copy()
                )

                anomalies = (
                    _collapse_anomalies_for_daily_display(
                        anomalies
                    )
                )

        else:
            st.write(
                "Stress-anomaly overlay disabled for this custom view."
            )

            anomalies = pd.DataFrame(
                columns=[
                    "taxi_zone_id",
                    "date",
                    "zone",
                    "borough",
                    "anomaly_event_count",
                ]
            )

        st.write(
            (
                "Resolving the map focus and building up to "
                f"{CUSTOM_MAX_FRAMES:,} interactive animation frames."
            )
        )

        centroids = (
            _load_zone_centroids()
            .merge(
                zone_catalog[
                    [
                        "taxi_zone_id",
                        "borough",
                    ]
                ],
                on="taxi_zone_id",
                how="left",
                validate="many_to_one",
            )
        )

        if map_focus == "Borough":
            center_source = (
                centroids[
                    centroids[
                        "borough"
                    ]
                    .astype(str)
                    .eq(
                        selected_borough
                    )
                ]
            )

            (
                map_center,
                focus_zoom,
            ) = _borough_focus_view(
                center_source,
                selected_borough=(
                    selected_borough
                ),
            )

            focus_title = (
                f"{selected_borough} "
                "mobility pulse"
            )

        else:
            map_center = {
                "lat": 40.7128,
                "lon": -74.0060,
            }

            focus_zoom = 9.2
            focus_title = (
                "Citywide mobility pulse"
            )

        metric_label = (
            METRIC_LABELS.get(
                selected_metric,
                selected_metric,
            )
        )
        animation_fig = (
            _build_animation_figure(
                panel=focus_panel,
                anomalies=anomalies,
                metric_label=metric_label,
                focus_title=focus_title,
                center=map_center,
                zoom=focus_zoom,
                show_anomalies=(
                    show_anomalies
                ),
            )
        )

        st.write(
            "Building the supporting timeline."
        )
        timeline_fig = (
            _build_timeline_figure(
                panel=focus_panel,
                anomalies=anomalies,
                focus_label=(
                    selected_borough
                    if map_focus == "Borough"
                    else "Citywide"
                ),
                metric_column=(
                    selected_metric
                ),
                pre_cp_baseline=(
                    focus_pre_baseline
                ),
                show_anomalies=(
                    show_anomalies
                ),
            )
        )

        status.update(
            label=(
                "Custom Mobility Pulse ready."
            ),
            state="complete",
            expanded=False,
        )

    # -----------------------------------------------------------------
    # Supporting summary values
    # -----------------------------------------------------------------

    window_daily = (
        focus_panel.groupby(
            "date",
            observed=True,
            dropna=False,
        )
        .agg(
            focus_value=(
                selected_metric,
                "mean",
            ),
        )
        .reset_index()
        .sort_values("date")
    )

    focus_window_avg = (
        window_daily["focus_value"].mean()
        if not window_daily.empty
        else np.nan
    )

    focus_change = (
        (
            focus_window_avg
            - focus_pre_baseline
        )
        / focus_pre_baseline
        * 100
        if (
            pd.notna(focus_pre_baseline)
            and pd.notna(focus_window_avg)
            and focus_pre_baseline != 0
        )
        else np.nan
    )

    focus_anomaly_days = (
        int(
            anomalies["date"].nunique()
        )
        if not anomalies.empty
        else 0
    )

    zones_in_view = int(
        focus_panel[
            "taxi_zone_id"
        ].nunique()
    )

    if not window_daily.empty:
        latest_focus_row = (
            window_daily.iloc[-1]
        )

        latest_value = (
            latest_focus_row[
                "focus_value"
            ]
        )

        latest_date = (
            latest_focus_row[
                "date"
            ]
        )

    else:
        latest_value = np.nan
        latest_date = pd.NaT

    geography_label = (
        selected_borough
        if map_focus == "Borough"
        else "Citywide"
    )

    # -----------------------------------------------------------------
    # Custom output
    # -----------------------------------------------------------------

    left_col, right_col = (
        st.columns(
            [
                3,
                1,
            ]
        )
    )

    with left_col:
        st.plotly_chart(
            animation_fig,
            width="stretch",
            config=MAP_CONFIG,
            key=(
                "mobility_pulse_custom_map"
            ),
        )
        render_chart_insight(
            f"The map contains **{zones_in_view:,} Taxi Zones** in "
            f"**{geography_label}**. Their selected-window {metric_label.lower()} "
            f"average is **{_format_percent(focus_change)} versus the pre-CP "
            f"baseline**, with **{focus_anomaly_days:,} metric-linked "
            "stress-anomaly days** in the focus geography."
        )

        st.plotly_chart(
            timeline_fig,
            width="stretch",
            config={
                "displayModeBar": False,
                "responsive": True,
            },
            key=(
                "mobility_pulse_custom_timeline"
            ),
        )
        render_chart_insight(
            f"The latest displayed {metric_label.lower()} value is "
            f"**{_format_number(latest_value)} on "
            f"{latest_date.strftime('%b %d, %Y') if pd.notna(latest_date) else 'an unavailable date'}**; "
            f"the selected-window average is **{_format_number(focus_window_avg)}** "
            f"versus a pre-CP baseline of **{_format_number(focus_pre_baseline)}**."
        )

    with right_col:
        st.subheader(
            "Custom view summary"
        )

        st.caption(
            (
                "The selected window is compared with the same geography's "
                "pre-CP baseline."
            )
        )

        quick_card1, quick_card2 = (
            st.columns(2)
        )

        with quick_card1:
            st.caption("Focus")

            st.markdown(
                (
                    "<div style='font-size:1.05rem; "
                    "line-height:1.05; white-space:normal; "
                    "word-break:break-word;'>"
                    f"{focus_title}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

        quick_card2.metric(
            "Zones in view",
            f"{zones_in_view:,}",
        )

        quick_card3, quick_card4 = (
            st.columns(2)
        )

        quick_card3.metric(
            "Pre-CP baseline",
            _format_number(
                focus_pre_baseline
            ),
        )

        quick_card4.metric(
            "Window avg.",
            _format_number(
                focus_window_avg
            ),
        )

        quick_card5, quick_card6 = (
            st.columns(2)
        )

        quick_card5.metric(
            "Vs baseline",
            _format_percent(
                focus_change
            ),
        )

        quick_card6.metric(
            "Stress-anomaly days",
            f"{focus_anomaly_days:,}",
        )

        st.markdown(
            f"**{metric_label}**"
        )

        st.caption(
            (
                f"{geography_label} · "
                f"{zones_in_view:,} zones"
            )
        )

        latest_date_text = (
            latest_date.strftime(
                "%b %d, %Y"
            )
            if pd.notna(latest_date)
            else "Unavailable"
        )

        st.caption(
            (
                f"Latest window value on "
                f"{latest_date_text}: "
                f"{_format_number(latest_value)}"
            )
        )

        st.caption(
            (
                "Use Zone Profile for a deeper explanation of an individual "
                "Taxi Zone."
            )
        )

# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------

inject_app_css()

st.caption(
    "MOBILITY PULSE"
)

st.title(
    "How did mobility change across the city over time?"
)

st.write(
    (
        "Mobility did not move uniformly across New York. "
        "Watch how individual Taxi Zones departed from their own "
        "pre-congestion-pricing patterns, where those shifts persisted, "
        "and when unusual metric-linked events appeared."
    )
)

st.header("Featured mobility stories")


# ---------------------------------------------------------------------
# Curated saved stories
# ---------------------------------------------------------------------

preset_manifest = (
    _load_preset_manifest()
)

if not preset_manifest:
    st.warning(
        (
            "Saved Mobility Pulse animations are not available. "
            "Rebuild `data/processed/app_tables/mobility_pulse_presets/` "
            "before deployment."
        )
    )

else:
    available_presets = [
        preset
        for preset in preset_manifest
        if (
            PRESET_DIR
            / str(
                preset["filename"]
            )
        ).exists()
    ]

    if not available_presets:
        st.warning(
            (
                "The Mobility Pulse manifest exists, but none of its "
                "saved MP4 files could be found."
            )
        )

    else:
        tab_labels = [
            PRESET_STORY_COPY.get(
                str(
                    preset["preset_id"]
                ),
                {},
            ).get(
                "tab_label",
                str(
                    preset.get(
                        "menu_label",
                        preset["preset_id"],
                    )
                ),
            )
            for preset
            in available_presets
        ]

        story_tabs = st.tabs(
            tab_labels
        )

        for tab, preset in zip(
            story_tabs,
            available_presets,
        ):
            with tab:
                _render_saved_story(
                    preset
                )


# ---------------------------------------------------------------------
# Custom explorer
# ---------------------------------------------------------------------

st.divider()

st.header("Explore Mobility Pulse patterns")

st.write(
    (
        "Use the custom builder when you want a different metric, "
        "time bucket, geography, or date range than the curated stories."
    )
)

st.info(
    (
        "Custom interactive animations are generated only after you click "
        "**Build custom animation**. Long windows are sampled to keep the "
        "interactive view responsive; the curated stories use every available "
        "observation date."
    )
)

zone_catalog = (
    get_zone_catalog()
    .copy()
)

borough_options = sorted(
    [
        borough
        for borough
        in (
            zone_catalog[
                "borough"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        if borough != "Unknown"
    ]
)

default_borough_index = (
    borough_options.index(
        "Manhattan"
    )
    if "Manhattan"
    in borough_options
    else 0
)

default_custom_start = max(
    DATA_WINDOW_START,
    (
        pd.Timestamp(
            DATA_WINDOW_END
        )
        - pd.Timedelta(
            days=(
                DEFAULT_CUSTOM_WINDOW_DAYS
            )
        )
    ).date(),
)

with st.expander(
    "Custom animation controls",
    expanded=False,
):
    with st.form(
        "mobility_pulse_custom_form"
    ):
        (
            control1,
            control2,
            control3,
        ) = st.columns(3)

        with control1:
            custom_metric = (
                st.selectbox(
                    "Metric to animate",
                    options=PULSE_METRICS,
                    index=(
                        PULSE_METRICS.index(
                            DEFAULT_METRIC
                        )
                    ),
                    format_func=lambda metric: (
                        METRIC_LABELS.get(
                            metric,
                            metric,
                        )
                    ),
                    key=(
                        "mobility_pulse_custom_metric"
                    ),
                )
            )

        with control2:
            custom_temporal_bucket = (
                st.selectbox(
                    "Temporal bucket",
                    options=[
                        ALL_TEMPORAL_BUCKETS_LABEL,
                        *TEMPORAL_BUCKET_ORDER,
                    ],
                    index=0,
                    key=(
                        "mobility_pulse_custom_bucket"
                    ),
                )
            )

        with control3:
            custom_show_anomalies = (
                st.checkbox(
                    "Show stress anomalies",
                    value=True,
                    key=(
                        "mobility_pulse_custom_anomalies"
                    ),
                    help=(
                        "Show stress anomalies where the displayed metric "
                        "was identified as one of the event drivers."
                    ),
                )
            )

        date_col, geography_col = st.columns(
            [
                2,
                1,
            ]
        )

        with date_col:
            custom_date_window = (
                st.date_input(
                    "Animation window",
                    value=(
                        default_custom_start,
                        DATA_WINDOW_END,
                    ),
                    min_value=(
                        DATA_WINDOW_START
                    ),
                    max_value=(
                        DATA_WINDOW_END
                    ),
                    key=(
                        "mobility_pulse_custom_window"
                    ),
                    help=(
                        "Custom animations are capped at approximately "
                        f"{CUSTOM_MAX_FRAMES:,} frames. Longer windows are "
                        "sampled."
                    ),
                )
            )

        with geography_col:
            custom_geography = (
                st.selectbox(
                    "Geography",
                    options=[
                        "Citywide",
                        *borough_options,
                    ],
                    index=0,
                    key=(
                        "mobility_pulse_custom_geography"
                    ),
                    help=(
                        "Choose Citywide or focus the animation on one borough."
                    ),
                )
            )

        custom_map_focus = (
            "Citywide"
            if custom_geography == "Citywide"
            else "Borough"
        )

        custom_borough = (
            custom_geography
            if custom_map_focus == "Borough"
            else (
                borough_options[
                    default_borough_index
                ]
                if borough_options
                else ""
            )
        )

        st.caption(
            (
                "What to notice: "
                f"{_metric_notice(custom_metric)}"
            )
        )

        custom_submitted = (
            st.form_submit_button(
                "Build custom animation",
                type="primary",
            )
        )


if custom_submitted:
    _render_custom_view(
        selected_metric=(
            custom_metric
        ),
        temporal_bucket=(
            custom_temporal_bucket
        ),
        show_anomalies=(
            custom_show_anomalies
        ),
        date_window=(
            custom_date_window
        ),
        map_focus=(
            custom_map_focus
        ),
        selected_borough=(
            custom_borough
        ),
        zone_catalog=(
            zone_catalog
        ),
    )


# ---------------------------------------------------------------------
# Reading guide
# ---------------------------------------------------------------------

st.divider()

st.header("How to read this page")

st.markdown(
    """
    - **Map color:** teal means the displayed metric is above that Taxi Zone's own pre-CP reference; terracotta means it is below.
    - **Orange dot:** at least one stress anomaly occurred on that date where the displayed metric was identified as an event driver. The production stress-anomaly surface retains events identified by all three anomaly-detection methods.
    - **Curated stories:** pre-rendered and use every available observation date in the selected period.
    - **Custom animations:** interactive and generated only when requested. Long windows are sampled to keep playback responsive.
    """
)
