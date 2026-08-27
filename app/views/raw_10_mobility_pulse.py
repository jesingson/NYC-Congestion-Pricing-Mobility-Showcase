from __future__ import annotations

from datetime import date
from math import ceil

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from shapely.geometry import shape

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
from app.data_access.anomalies import (
    get_metric_driver_anomaly_events,
)
from app.data_access.zone_profiles import (
    get_zone_catalog,
)
from app.utils.project_branding import inject_app_css


PULSE_METRICS = list(BASE_METRICS)
DEFAULT_METRIC = "taxi_trip_count"
DEFAULT_WINDOW_DAYS = 120
DATA_WINDOW_START = pd.to_datetime("2023-01-01").date()
DATA_WINDOW_END = pd.to_datetime("2026-03-31").date()

MAP_FOCUS_OPTIONS = [
    "Citywide",
    "Borough",
]

PULSE_METRIC_NOTES = {
    "taxi_trip_count": "Look for sustained rises or drops in trip volume rather than isolated spikes.",
    "fhv_trip_count": "Watch for changes in for-hire activity that hold across several dates, not just one-off bumps.",
    "subway_ridership": "Focus on longer runs of higher or lower ridership, since day-to-day noise is common.",
    "bus_ridership": "Look for persistent movement in bus usage, especially when it lines up with other mode shifts.",
    "average_speed": "Pay attention to slower speeds that persist across multiple days or buckets.",
    "trip_duration": "Longer durations can signal slower movement, but small changes may still be normal variation.",
    "pm_peak_speed": "Watch for repeated PM slowdowns, since they often reflect recurring congestion conditions.",
    "am_peak_speed": "Focus on morning slowdowns that last beyond a few points in the series.",
    "daily_vehicle_hours": "Look for durable changes in total vehicle hours rather than short-lived swings.",
    "anomaly_count": "Treat spikes as disruption signals, not a new baseline pattern.",
}


def _format_number(value: object) -> str:
    if pd.isna(value):
        return "Unavailable"
    numeric = float(value)
    if abs(numeric) >= 1_000_000:
        return f"{numeric / 1_000_000:,.2f}M"
    if abs(numeric) >= 1_000:
        return f"{numeric / 1_000:,.2f}K"
    return f"{numeric:,.2f}"


def _format_percent(value: object) -> str:
    if pd.isna(value):
        return "Unavailable"
    return f"{float(value):+,.2f}%"


def _zone_label(row: pd.Series) -> str:
    return (
        f"{row['zone']} · {row['borough']} · "
        f"Zone {row['taxi_zone_id']}"
    )


@st.cache_data(show_spinner=False)
def _load_zone_centroids() -> pd.DataFrame:
    geojson = get_zone_geojson()
    records: list[dict[str, object]] = []

    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        geom = feature.get("geometry")
        if not props or not geom:
            continue

        centroid = shape(geom).centroid
        taxi_zone_id = props.get("taxi_zone_id")
        if taxi_zone_id is None:
            continue

        records.append(
            {
                "taxi_zone_id": int(taxi_zone_id),
                "lon": float(centroid.x),
                "lat": float(centroid.y),
            }
        )

    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def _load_pulse_panel(
    *,
    metric: str,
    temporal_bucket: str,
) -> pd.DataFrame:
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
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")

    pre_reference = (
        frame.loc[
            frame["date"] < CONGESTION_PRICING_START_DATE,
            [
                "taxi_zone_id",
                metric,
            ],
        ]
        .groupby("taxi_zone_id", observed=True, dropna=False)[metric]
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

    frame["pulse_change"] = np.where(
        frame["pre_reference"].gt(0),
        (frame[metric] - frame["pre_reference"])
        / frame["pre_reference"]
        * 100,
        np.nan,
    )
    frame["pulse_index"] = np.where(
        frame["pre_reference"].gt(0),
        frame[metric] / frame["pre_reference"] * 100,
        np.nan,
    )
    frame["date_label"] = frame["date"].dt.strftime("%Y-%m-%d")

    return frame.sort_values(
        [
            "date",
            "taxi_zone_id",
        ]
    ).reset_index(drop=True)


def _sample_dates(
    dates: pd.Series,
    *,
    max_frames: int = DEFAULT_WINDOW_DAYS,
) -> list[pd.Timestamp]:
    unique_dates = list(
        pd.to_datetime(dates.dropna().drop_duplicates()).sort_values()
    )

    if len(unique_dates) <= max_frames:
        return unique_dates

    step = ceil(len(unique_dates) / max_frames)
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
    """Clamp a date-range widget value to the supported data window."""
    if not isinstance(date_window, (tuple, list)) or len(date_window) != 2:
        return min_date, max_date

    start_date = pd.to_datetime(date_window[0]).date()
    end_date = pd.to_datetime(date_window[1]).date()

    start_date = max(min_date, min(start_date, max_date))
    end_date = max(min_date, min(end_date, max_date))

    if end_date < start_date:
        start_date, end_date = min_date, max_date

    return start_date, end_date


def _metric_notice(metric: str) -> str:
    return PULSE_METRIC_NOTES.get(
        metric,
        "Look for sustained changes and be cautious about reading too much into small one-day moves.",
    )


def _borough_focus_view(
    center_source: pd.DataFrame,
    *,
    selected_borough: str,
) -> tuple[dict[str, float], float]:
    """Return a borough-specific center and zoom that fit the borough more naturally."""
    if center_source.empty:
        return {"lat": 40.7128, "lon": -74.0060}, 9.8

    center = {
        "lat": float(center_source["lat"].mean()),
        "lon": float(center_source["lon"].mean()),
    }
    zoom = 10.3

    borough_offsets = {
        "Queens": {"lat": -0.05, "lon": 0.0, "zoom": 9.85},
        "Brooklyn": {"lat": 0.01, "lon": 0.02, "zoom": 9.95},
        "Bronx": {"lat": 0.02, "lon": 0.0, "zoom": 10.0},
        "Manhattan": {"lat": 0.01, "lon": 0.0, "zoom": 10.7},
        "Staten Island": {"lat": -0.01, "lon": 0.0, "zoom": 10.15},
    }

    adjustments = borough_offsets.get(selected_borough)
    if adjustments:
        center["lat"] += adjustments["lat"]
        center["lon"] += adjustments["lon"]
        zoom = adjustments["zoom"]

    return center, zoom


def _filter_focus_panel(
    panel: pd.DataFrame,
    *,
    focus_mode: str,
    focus_borough: str,
) -> pd.DataFrame:
    if focus_mode == "Borough":
        return panel[panel["borough"].astype(str).eq(focus_borough)].copy()

    return panel.copy()

def _collapse_anomalies_for_daily_display(
    anomalies: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse daypart anomaly events to one Taxi Zone × date map marker."""
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
    zone_centroids = _load_zone_centroids()
    frame_dates = _sample_dates(panel["date"])

    if not frame_dates:
        return go.Figure()

    bound = calculate_robust_symmetric_bound(panel["pulse_change"])
    zmin, zmax = -bound, bound

    first_date = frame_dates[0]
    first_panel = panel[panel["date"].eq(first_date)].copy()
    first_anomalies = anomalies[anomalies["date"].eq(first_date)].copy()

    if show_anomalies and not first_anomalies.empty:
        first_anomalies = first_anomalies.merge(
            zone_centroids,
            on="taxi_zone_id",
            how="left",
            validate="many_to_one",
        )
    else:
        first_anomalies = pd.DataFrame(columns=["lat", "lon"])

    base_trace = go.Choroplethmap(
        geojson=get_zone_geojson(),
        locations=first_panel["taxi_zone_id"],
        featureidkey="properties.taxi_zone_id",
        z=first_panel["pulse_change"],
        zmin=zmin,
        zmax=zmax,
        colorscale=[
            [0.00, BRAND_COLORS["terracotta"]],
            [0.35, BRAND_COLORS["pale_peach"]],
            [0.50, BRAND_COLORS["ice"]],
            [0.65, BRAND_COLORS["seafoam"]],
            [1.00, BRAND_COLORS["dark_teal"]],
        ],
        marker={"line": {"width": 0.25, "color": "white"}},
        customdata=np.stack(
            [
                first_panel["zone"],
                first_panel["borough"],
                first_panel["date_label"],
                first_panel["pulse_index"].round(2),
                first_panel["pulse_change"].round(2),
                first_panel["pre_reference"].round(2),
            ],
            axis=-1,
        ),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Borough: %{customdata[1]}<br>"
            "Date: %{customdata[2]}<br>"
            f"{metric_label} index: %{{customdata[3]}}<br>"
            "Change vs pre-CP average: %{customdata[4]}%<br>"
            "Pre-CP reference: %{customdata[5]}<extra></extra>"
        ),
        colorbar={
            "title": {"text": "% change vs pre-CP"},
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
        name="Anomaly",
        marker={
            "size": 10,
            "color": BRAND_COLORS["terracotta"],
            "opacity": 0.9,
        },
        text=first_anomalies.get(
            "zone",
            pd.Series(dtype=str),
        ),
        customdata=np.column_stack(
            [
                first_anomalies.get(
                    "anomaly_event_count",
                    pd.Series(dtype=int),
                ),
            ]
        ),
        hovertemplate=(
            "<b>Anomaly</b><br>"
            "Zone: %{text}<br>"
            "Anomalous daypart events: %{customdata[0]:,}"
            "<extra></extra>"
        ),
        showlegend=True,
    )

    frames: list[go.Frame] = []
    for date_value in frame_dates:
        date_panel = panel[panel["date"].eq(date_value)].copy()
        date_anomalies = anomalies[anomalies["date"].eq(date_value)].copy()

        if show_anomalies and not date_anomalies.empty:
            date_anomalies = date_anomalies.merge(
                zone_centroids,
                on="taxi_zone_id",
                how="left",
                validate="many_to_one",
            )
        else:
            date_anomalies = pd.DataFrame(columns=["lat", "lon"])

        frame_trace = go.Choroplethmap(
            geojson=get_zone_geojson(),
            locations=date_panel["taxi_zone_id"],
            featureidkey="properties.taxi_zone_id",
            z=date_panel["pulse_change"],
            zmin=zmin,
            zmax=zmax,
            colorscale=[
                [0.00, BRAND_COLORS["terracotta"]],
                [0.35, BRAND_COLORS["pale_peach"]],
                [0.50, BRAND_COLORS["ice"]],
                [0.65, BRAND_COLORS["seafoam"]],
                [1.00, BRAND_COLORS["dark_teal"]],
            ],
            marker={"line": {"width": 0.25, "color": "white"}},
            customdata=np.stack(
                [
                    date_panel["zone"],
                    date_panel["borough"],
                    date_panel["date_label"],
                    date_panel["pulse_index"].round(2),
                    date_panel["pulse_change"].round(2),
                    date_panel["pre_reference"].round(2),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Borough: %{customdata[1]}<br>"
                "Date: %{customdata[2]}<br>"
                f"{metric_label} index: %{{customdata[3]}}<br>"
                "Change vs pre-CP average: %{customdata[4]}%<br>"
                "Pre-CP reference: %{customdata[5]}<extra></extra>"
            ),
            showscale=True,
            name="Mobility pulse",
        )

        frame_anomaly_trace = go.Scattermap(
            lat=date_anomalies.get(
                "lat",
                pd.Series(dtype=float),
            ),
            lon=date_anomalies.get(
                "lon",
                pd.Series(dtype=float),
            ),
            mode="markers",
            name="Anomaly",
            marker={
                "size": 10,
                "color": BRAND_COLORS["terracotta"],
                "opacity": 0.9,
            },
            text=date_anomalies.get(
                "zone",
                pd.Series(dtype=str),
            ),
            customdata=np.column_stack(
                [
                    date_anomalies.get(
                        "anomaly_event_count",
                        pd.Series(dtype=int),
                    ),
                ]
            ),
            hovertemplate=(
                "<b>Anomaly</b><br>"
                "Zone: %{text}<br>"
                "Anomalous daypart events: %{customdata[0]:,}"
                "<extra></extra>"
            ),
            showlegend=True,
        )

        frames.append(
            go.Frame(
                name=date_value.strftime("%Y-%m-%d"),
                data=[frame_trace, frame_anomaly_trace],
            )
        )

    fig = go.Figure(data=[base_trace, anomaly_trace], frames=frames)

    slider_steps = [
        {
            "args": [[date_value.strftime("%Y-%m-%d")], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": date_value.strftime("%b %d"),
            "method": "animate",
        }
        for date_value in frame_dates
    ]

    fig.update_layout(
        title={"text": focus_title},
        map={
            "style": "carto-positron",
            "center": center,
            "zoom": zoom,
        },
        margin={"l": 0, "r": 120, "t": 60, "b": 10},
        height=700,
        hovermode="closest",
        uirevision="mobility-pulse",
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
                                "frame": {"duration": 450, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
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
                "pad": {"b": 10, "t": 30},
                "currentvalue": {"prefix": "Date: ", "font": {"size": 14}},
                "steps": slider_steps,
            }
        ],
    )

    return apply_branding(fig)


def _build_timeline_figure(
    *,
    panel: pd.DataFrame,
    anomalies: pd.DataFrame,
    focus_label: str,
    metric_column: str,
    metric_label: str,
    show_anomalies: bool,
) -> go.Figure:
    if panel.empty:
        return go.Figure()

    daily = (
        panel.groupby("date", observed=True, dropna=False)
        .agg(
            focus_value=(metric_column, "mean"),
            zone_count=("taxi_zone_id", "nunique"),
        )
        .reset_index()
        .sort_values("date")
    )

    pre_reference = daily.loc[
        daily["date"] < CONGESTION_PRICING_START_DATE,
        "focus_value",
    ].mean()

    daily["focus_index"] = np.where(
        pd.notna(pre_reference) and pre_reference != 0,
        daily["focus_value"] / pre_reference * 100,
        np.nan,
    )

    daily["focus_change"] = np.where(
        pd.notna(pre_reference) and pre_reference != 0,
        (daily["focus_value"] - pre_reference) / pre_reference * 100,
        np.nan,
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if not daily.empty:
        fig.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily["focus_index"],
                mode="lines",
                line={"color": BRAND_COLORS["dark_teal"], "width": 3},
                name=f"{focus_label} average",
                hovertemplate=(
                    "<b>%{x|%b %d, %Y}</b><br>"
                    "Index: %{y:.1f}<extra></extra>"
                ),
            ),
            secondary_y=False,
        )

    if show_anomalies and not anomalies.empty and not daily.empty:
        anomaly_series = daily.merge(
            anomalies[["date"]].drop_duplicates(),
            on="date",
            how="inner",
        )

        if not anomaly_series.empty:
            fig.add_trace(
                go.Scatter(
                    x=anomaly_series["date"],
                    y=anomaly_series["focus_index"],
                    mode="markers",
                    name="Anomaly",
                    marker={
                        "size": 10,
                        "color": BRAND_COLORS["terracotta"],
                        "line": {"color": "white", "width": 1},
                    },
                    hovertemplate=(
                        "<b>Anomaly activity</b><br>"
                        "Date: %{x|%b %d, %Y}"
                        "<extra></extra>"
                    ),
                ),
                secondary_y=False,
            )

    if show_anomalies and not anomalies.empty:
        anomaly_counts = (
            anomalies.groupby(
                "date",
                observed=True,
                dropna=False,
            )
            .size()
            .rename("anomaly_zone_count")
            .reset_index()
        )
        fig.add_trace(
            go.Bar(
                x=anomaly_counts["date"],
                y=anomaly_counts["anomaly_zone_count"],
                marker_color="rgba(231,111,81,0.30)",
                name="Zones with anomalies",
                hovertemplate=(
                    "<b>%{x|%b %d, %Y}</b><br>"
                    "Zones with anomalies: %{y:,}"
                    "<extra></extra>"
                ),
            ),
            secondary_y=True,
        )

    fig.update_layout(
        height=380,
        margin={"l": 0, "r": 0, "t": 10, "b": 40},
        title={"text": f"{focus_label} over time"},
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.18,
            "yanchor": "top",
        },
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Focus-group index", secondary_y=False)
    fig.update_yaxes(
        title_text="Zones with anomalies",
        secondary_y=True,
    )

    return apply_branding(fig)


inject_app_css()

st.caption("PAGE 10")
st.title("Mobility Pulse")
st.write(
    "Find where unusual mobility activity emerges, then use the compact borough or citywide view to jump into deeper diagnosis."
)

st.info(
    "Use one metric at a time for the animation. The map shows the selected metric as a change vs each zone's pre-CP reference."
)

zone_catalog = get_zone_catalog().copy()
zone_catalog["zone_label"] = zone_catalog.apply(_zone_label, axis=1)

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    selected_metric = st.selectbox(
        "Metric to animate",
        options=PULSE_METRICS,
        index=PULSE_METRICS.index(DEFAULT_METRIC),
        format_func=lambda metric: METRIC_LABELS.get(metric, metric),
        key="mobility_pulse_metric",
    )

with metric_col2:
    temporal_bucket = st.selectbox(
        "Temporal bucket",
        options=[ALL_TEMPORAL_BUCKETS_LABEL, *TEMPORAL_BUCKET_ORDER],
        index=0,
        key="mobility_pulse_bucket",
    )

with metric_col3:
    show_anomalies = st.checkbox(
        "Show anomalies",
        value=True,
        key="mobility_pulse_anomaly_icons",
        help=(
            "Show anomaly events where the selected metric "
            "was identified as one of the event drivers."
        ),
    )

st.caption(f"What to notice: {_metric_notice(selected_metric)}")

window_col1, window_col2, window_col3 = st.columns([2, 2, 1])

full_window_start = pd.to_datetime("2023-01-01").date()
full_window_end = pd.to_datetime("2026-03-31").date()
default_start = max(
    full_window_start,
    (pd.Timestamp(full_window_end) - pd.Timedelta(days=DEFAULT_WINDOW_DAYS)).date(),
)

date_window_key = "mobility_pulse_window"
if date_window_key in st.session_state:
    st.session_state[date_window_key] = _clamp_date_window(
        st.session_state[date_window_key],
        min_date=DATA_WINDOW_START,
        max_date=DATA_WINDOW_END,
    )

with window_col1:
    raw_date_window = st.date_input(
        "Animation window",
        value=(default_start, full_window_end),
        min_value=DATA_WINDOW_START,
        max_value=DATA_WINDOW_END,
        key=date_window_key,
        help="Long windows are sampled to keep the animation responsive.",
    )

date_window = _clamp_date_window(
    raw_date_window,
    min_date=DATA_WINDOW_START,
    max_date=DATA_WINDOW_END,
)
if raw_date_window != date_window:
    st.session_state[date_window_key] = date_window
    st.warning(
        "The requested animation window was outside the available data range, so it was clamped to January 1, 2023 through March 31, 2026."
    )

with window_col3:
    st.write("")
    st.write("")
    if st.button(
        "Reset range",
        key="mobility_pulse_window_reset",
        width="stretch",
    ):
        st.session_state[date_window_key] = (DATA_WINDOW_START, DATA_WINDOW_END)
        st.rerun()

with window_col2:
    map_focus = st.selectbox(
        "Map focus",
        options=MAP_FOCUS_OPTIONS,
        index=0,
        key="mobility_pulse_focus",
        help="Citywide keeps the full map visible; borough focus narrows the view to a single borough.",
    )

start_date = pd.to_datetime(date_window[0])
end_date = pd.to_datetime(date_window[1])

panel = _load_pulse_panel(
    metric=selected_metric,
    temporal_bucket=temporal_bucket,
)
focus_scope_panel = panel.copy()

anomalies = get_metric_driver_anomaly_events(
    metric=selected_metric,
    temporal_bucket=temporal_bucket,
)

borough_options = (
    ["All boroughs"]
    + sorted(
        [
            borough
            for borough in zone_catalog["borough"].dropna().astype(str).unique().tolist()
            if borough != "Unknown"
        ]
    )
)

selected_borough = "All boroughs"

if map_focus == "Borough":
    selected_borough = st.selectbox(
        "Borough focus",
        options=borough_options,
        index=0,
        key="mobility_pulse_borough",
    )
    if selected_borough != "All boroughs":
        focus_scope_panel = focus_scope_panel[
            focus_scope_panel["borough"].astype(str).eq(selected_borough)
        ].copy()
        if not anomalies.empty:
            anomalies = anomalies[
                anomalies["borough"].astype(str).eq(selected_borough)
            ].copy()

if map_focus == "Citywide":
    focus_zoom = 9.2
    focus_title = "Citywide mobility pulse"
elif map_focus == "Borough":
    focus_title = (
        f"{selected_borough} mobility pulse"
        if selected_borough != "All boroughs"
        else "Borough-focused mobility pulse"
    )

focus_panel = focus_scope_panel[
    focus_scope_panel["date"].between(start_date, end_date)
].copy()

if not anomalies.empty:
    anomalies = anomalies[
        anomalies["date"].between(start_date, end_date)
    ].copy()
    anomalies = _collapse_anomalies_for_daily_display(
        anomalies
    )

if focus_panel.empty:
    st.warning(
        "No rows match the current window and focus settings."
    )
    st.stop()

centroids = _load_zone_centroids().merge(
    zone_catalog[["taxi_zone_id", "borough"]],
    on="taxi_zone_id",
    how="left",
    validate="many_to_one",
)

if map_focus == "Borough" and selected_borough != "All boroughs":
    center_source = centroids[
        centroids["borough"].astype(str).eq(selected_borough)
    ]
else:
    center_source = centroids

if center_source.empty:
    map_center = {"lat": 40.7128, "lon": -74.0060}
    focus_zoom = 9.8 if map_focus == "Borough" else 9.2
else:
    map_center, focus_zoom = _borough_focus_view(
        center_source,
        selected_borough=selected_borough,
    )

focus_daily = (
    focus_scope_panel.groupby(
        "date",
        observed=True,
        dropna=False,
    )
    .agg(
        focus_value=(selected_metric, "mean"),
        zone_count=("taxi_zone_id", "nunique"),
    )
    .reset_index()
    .sort_values("date")
)

focus_pre = focus_daily.loc[
    focus_daily["date"] < CONGESTION_PRICING_START_DATE,
    "focus_value",
].mean()
focus_post = focus_daily.loc[
    focus_daily["date"] >= CONGESTION_PRICING_START_DATE,
    "focus_value",
].mean()
focus_change = (
    (focus_post - focus_pre) / focus_pre * 100
    if pd.notna(focus_pre)
    and pd.notna(focus_post)
    and focus_pre != 0
    else np.nan
)
focus_anomaly_days = (
    int(anomalies["date"].nunique())
    if not anomalies.empty
    else 0
)
zones_in_view = int(
    focus_panel["taxi_zone_id"].nunique()
)
if not focus_daily.empty:
    latest_focus_row = focus_daily.iloc[-1]
    latest_value = latest_focus_row["focus_value"]
    latest_date = latest_focus_row["date"]
else:
    latest_value = np.nan
    latest_date = pd.NaT

with st.spinner("Preparing the mobility pulse view..."):
    animation_fig = _build_animation_figure(
        panel=focus_panel,
        anomalies=anomalies,
        metric_label=METRIC_LABELS.get(
            selected_metric,
            selected_metric,
        ),
        focus_title=focus_title,
        center=map_center,
        zoom=focus_zoom,
        show_anomalies=show_anomalies,
    )
    timeline_fig = _build_timeline_figure(
        panel=focus_panel,
        anomalies=anomalies,
        focus_label=focus_title,
        metric_column=selected_metric,
        metric_label=METRIC_LABELS.get(
            selected_metric,
            selected_metric,
        ),
        show_anomalies=show_anomalies,
    )

left_col, right_col = st.columns([3, 1])

with left_col:
    st.plotly_chart(
        animation_fig,
        width="stretch",
        config=MAP_CONFIG,
        key="mobility_pulse_map",
    )

    st.plotly_chart(
        timeline_fig,
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
        key="mobility_pulse_timeline",
    )

with right_col:
    st.subheader("Focus quick view")
    st.caption("This panel stays compact and points to deeper diagnosis.")

    quick_card1, quick_card2 = st.columns(2)
    with quick_card1:
        st.caption("Focus")
        st.markdown(
            f"<div style='font-size:1.05rem; line-height:1.05; white-space:normal; word-break:break-word;'>{focus_title}</div>",
            unsafe_allow_html=True,
        )
    quick_card2.metric(
        "Zones in view",
        f"{zones_in_view:,}",
    )

    quick_card3, quick_card4 = st.columns(2)
    quick_card3.metric(
        "Pre-CP avg.",
        _format_number(focus_pre),
    )
    quick_card4.metric(
        "Post-CP avg.",
        _format_number(focus_post),
    )

    quick_card5, quick_card6 = st.columns(2)
    quick_card5.metric(
        "Change",
        _format_percent(focus_change),
    )
    quick_card6.metric(
        "Anomaly days",
        f"{focus_anomaly_days:,}",
    )

    st.markdown(
        f"**{METRIC_LABELS.get(selected_metric, selected_metric)}**"
    )
    st.caption(
        f"{map_focus} · {zones_in_view:,} zones in the focus group"
    )
    latest_date_text = (
        latest_date.strftime("%b %d, %Y")
        if pd.notna(latest_date)
        else "Unavailable"
    )
    st.caption(
        f"Latest window value on {latest_date_text}: "
        f"{_format_number(latest_value)}"
    )
    st.caption(
        "Use Zone Profile (Raw 07) for a deeper explanation of borough patterns."
    )

st.divider()

st.markdown("### How to read this page")
st.markdown(
    """
    - The map animates one metric at a time, normalized to each zone's pre-CP reference.
    - The animation window is sampled if it gets too long, so the page stays responsive.
    - The focus panel is intentionally compact and is meant to hand off to deeper diagnosis.
    """
)
