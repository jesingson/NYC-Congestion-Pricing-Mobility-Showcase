from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data_access.aggregations import (
    CBD_SPATIAL_CATEGORY_GROUPS,
    aggregate_metrics,
    apply_common_filters,
    format_metric_value,
    format_signed_percent,
    get_required_columns,
)
from app.data_access.loaders import (
    BASE_METRICS,
    CONGESTION_PRICING_START_DATE,
    METRIC_LABELS,
    STUDY_END_DATE,
    STUDY_START_DATE,
    load_analysis_panel,
)
from app.data_access.mobility_environments import (
    format_mobility_regime_cluster_label,
    get_mobility_regime_cluster_options,
)
from app.data_access.anomalies import (
    load_selected_anomaly_events,
)
from app.utils.project_branding import (
    BRAND_COLORS,
    BRAND_DIVERGING_SEQUENCE,
    apply_branding,
    inject_app_css,
)


inject_app_css()

PAGE_TITLE = "Cyclical Time"
DEFAULT_METRIC = "taxi_trip_count"
ANOMALY_RATE_METRIC = "finalist_anomaly_zone_rate"
SMOOTHING_WINDOW_DAYS = 14
ANNUAL_CYCLE_DAYS = 365

SEASON_COLORS = {
    "Winter": "#5B5F97",
    "Spring": BRAND_COLORS["seafoam"],
    "Summer": BRAND_COLORS["terracotta"],
    "Fall": BRAND_COLORS["dark_teal"],
}

SEASON_ORDER = [
    "Winter",
    "Spring",
    "Summer",
    "Fall",
]

MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

GEOGRAPHY_OPTIONS = [
    "Citywide",
    "Borough",
    "Geo-policy group",
    "Mobility regime cluster",
]

FEATURED_STORIES = {
    "Citywide · Subway transfers": {
        "headline": "Subway transfers show a stable annual lift",
        "summary": (
            "Subway transfers trace a strong annual cycle, and the post-CP loop sits above the "
            "pre-CP baseline instead of merely repeating the earlier pattern."
        ),
        "metric": "subway_transfers",
        "geography_scope": "Citywide",
        "borough": None,
        "geo_policy": None,
        "cluster": None,
    },
    "Citywide · Subway ridership": {
        "headline": "Subway ridership follows the same yearly cadence",
        "summary": (
            "Subway ridership follows the same seasonal cadence, but the loop is smoother and "
            "more stable than transfers, so the annual pattern reads as a broad backbone."
        ),
        "metric": "subway_ridership",
        "geography_scope": "Citywide",
        "borough": None,
        "geo_policy": None,
        "cluster": None,
    },
    "Cluster 2 · Long-Trip Fast-Mobility Zones · Subway ridership": {
        "headline": "Cluster 2 keeps the cycle but shifts its shape",
        "summary": (
            "Cluster 2 still cycles annually, but the spiral shape shifts enough to show that "
            "fast-mobility zones do not move like the citywide average."
        ),
        "metric": "subway_ridership",
        "geography_scope": "Mobility regime cluster",
        "borough": None,
        "geo_policy": None,
        "cluster": 2,
    },
    "Gateway only · Taxi trips": {
        "headline": "Gateway zones show the sharpest post-CP lift",
        "summary": (
            "Gateway zones keep the cyclical loop, but the post-CP segment sits noticeably above "
            "the earlier loop, making the policy shift the dominant read."
        ),
        "metric": "taxi_trip_count",
        "geography_scope": "Geo-policy group",
        "borough": None,
        "geo_policy": "Gateway only",
        "cluster": None,
    },
    "Cluster 4 · Staten Island Fast-Mobility · Taxi trips": {
        "headline": "Staten Island is the compact cluster contrast",
        "summary": (
            "A smaller but still legible mobility-regime example. It is useful when "
            "we want a cluster-specific story that feels distinct from the citywide view."
        ),
        "metric": "taxi_trip_count",
        "geography_scope": "Mobility regime cluster",
        "borough": None,
        "geo_policy": None,
        "cluster": 4,
    },
    "EWR · FHVHV average trip duration": {
        "headline": "EWR makes the annual cycle look dramatic, but specific",
        "summary": (
            "This is a very strong signal, though it is more diagnostic than general. "
            "It is useful as a special-case view, not the page default."
        ),
        "metric": "fhvhv_avg_trip_duration",
        "geography_scope": "Borough",
        "borough": "EWR",
        "geo_policy": None,
        "cluster": None,
    },
    "EWR · FHVHV average speed": {
        "headline": "EWR speed is another strong special-case cycle",
        "summary": (
            "Like the duration view, this gives a very strong annual pattern while "
            "remaining clearly airport-specific."
        ),
        "metric": "fhvhv_avg_trip_speed",
        "geography_scope": "Borough",
        "borough": "EWR",
        "geo_policy": None,
        "cluster": None,
    },
}

FEATURED_STORY_OPTIONS = [
    "Citywide - Subway transfers",
    "Citywide - Subway ridership",
    "Cluster 2 - Long-Trip Fast-Mobility Zones - Subway ridership",
    "Gateway only - Taxi trips",
]


def _metric_label(metric: str) -> str:
    if metric == ANOMALY_RATE_METRIC:
        return "Anomaly zone rate"

    return METRIC_LABELS.get(
        metric,
        metric.replace("_", " ").title(),
    )


def _format_value(value: object) -> str:
    if pd.isna(value):
        return "Unavailable"
    return format_metric_value(float(value), digits=2)


def _format_change(value: object) -> str:
    if pd.isna(value):
        return "Unavailable"
    return format_signed_percent(float(value), digits=2)


def _scope_summary(
    geography_scope: str,
    *,
    borough: str | None,
    geo_policy: str | None,
    cluster_label: int | None,
) -> str:
    if geography_scope == "Citywide":
        return "Citywide"
    if geography_scope == "Borough" and borough:
        return borough
    if geography_scope == "Geo-policy group" and geo_policy:
        return geo_policy
    if geography_scope == "Mobility regime cluster" and cluster_label is not None:
        return format_mobility_regime_cluster_label(cluster_label)
    return "Selected geography"


def _build_date_frame(
    daily: pd.DataFrame,
    *,
    value_column: str,
    fill_value: float | None = None,
) -> pd.DataFrame:
    frame = daily.copy()
    all_dates = pd.date_range(STUDY_START_DATE, STUDY_END_DATE, freq="D")

    if "date" not in frame.columns:
        frame["date"] = pd.Series(dtype="datetime64[ns]")

    if value_column not in frame.columns:
        frame[value_column] = np.nan

    frame = frame.set_index("date").reindex(all_dates).rename_axis("date").reset_index()

    if fill_value is not None:
        frame[value_column] = frame[value_column].fillna(fill_value)

    return frame


def _apply_smoothing(
    values: pd.Series,
    *,
    enabled: bool,
) -> pd.Series:
    if not enabled:
        return values

    return values.rolling(window=SMOOTHING_WINDOW_DAYS, min_periods=7).mean()


def _add_index_column(
    df: pd.DataFrame,
    *,
    source_column: str,
    index_column: str,
) -> pd.DataFrame:
    result = df.copy()
    pre_mask = result["date"] < CONGESTION_PRICING_START_DATE
    baseline = result.loc[pre_mask, source_column].mean(skipna=True)

    if pd.isna(baseline) or baseline == 0:
        result[index_column] = np.nan
        return result

    result[index_column] = (result[source_column] / baseline) * 100
    return result


def _season_bounds() -> list[tuple[str, int, int, str]]:
    return [
        ("Winter", 1, 59, SEASON_COLORS["Winter"]),
        ("Spring", 60, 151, SEASON_COLORS["Spring"]),
        ("Summer", 152, 243, SEASON_COLORS["Summer"]),
        ("Fall", 244, 334, SEASON_COLORS["Fall"]),
        ("Winter", 335, 365, SEASON_COLORS["Winter"]),
    ]


def _day_to_theta(day_of_year: int) -> float:
    return 2 * math.pi * ((day_of_year - 1) / ANNUAL_CYCLE_DAYS)


def _build_sector_polygon(
    *,
    start_day: int,
    end_day: int,
    outer_radius: float,
    inner_radius: float = 0.0,
    points: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(
        _day_to_theta(start_day),
        _day_to_theta(end_day),
        points,
    )
    outer_x = outer_radius * np.cos(theta)
    outer_y = outer_radius * np.sin(theta)
    inner_theta = theta[::-1]
    inner_x = inner_radius * np.cos(inner_theta)
    inner_y = inner_radius * np.sin(inner_theta)
    x = np.concatenate([[0.0], outer_x, inner_x, [0.0]])
    y = np.concatenate([[0.0], outer_y, inner_y, [0.0]])
    return x, y


def _add_season_wedges(fig: go.Figure, *, outer_radius: float) -> None:
    for season_name, start_day, end_day, color in _season_bounds():
        x, y = _build_sector_polygon(
            start_day=start_day,
            end_day=end_day,
            outer_radius=outer_radius,
            inner_radius=0.0,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line={"width": 0},
                fill="toself",
                fillcolor=color,
                opacity=0.09,
                hoverinfo="skip",
                showlegend=False,
            )
        )


def _add_season_labels(fig: go.Figure, *, outer_radius: float) -> None:
    label_radius = outer_radius * 0.92
    label_specs = [
        ("Winter", 20),
        ("Spring", 110),
        ("Summer", 200),
        ("Fall", 290),
    ]
    for season_name, day_of_year in label_specs:
        theta = _day_to_theta(day_of_year)
        x = label_radius * np.cos(theta)
        y = label_radius * np.sin(theta)
        fig.add_annotation(
            x=x,
            y=y,
            text=season_name,
            showarrow=False,
            font={"size": 11, "color": SEASON_COLORS[season_name]},
            bgcolor="rgba(255,255,255,0.55)",
            bordercolor="rgba(255,255,255,0)",
            borderpad=1,
            xanchor="center",
            yanchor="middle",
        )


@st.cache_data(show_spinner=False)
def _get_scope_options() -> dict[str, list[object]]:
    panel = load_analysis_panel(columns=["borough", "cbd_spatial_category", "taxi_zone_id"])
    boroughs = [
        value
        for value in sorted(panel["borough"].dropna().astype(str).unique().tolist())
        if value and value.lower() != "unknown"
    ]

    geo_policy = list(CBD_SPATIAL_CATEGORY_GROUPS.keys())

    cluster_values = get_mobility_regime_cluster_options()

    return {
        "boroughs": boroughs,
        "geo_policy": geo_policy,
        "clusters": cluster_values,
    }


def _build_metric_series(
    *,
    metric: str,
    geography_scope: str,
    borough: str | None,
    geo_policy_group: str | None,
    cluster_label: int | None,
    smoothing_enabled: bool,
) -> tuple[pd.DataFrame, int]:
    columns = get_required_columns([metric])
    panel = load_analysis_panel(columns=columns)

    filters: dict[str, object] = {}

    if geography_scope == "Borough" and borough:
        filters["borough"] = borough
    elif geography_scope == "Geo-policy group" and geo_policy_group:
        filters["cbd_spatial_category"] = geo_policy_group
    elif geography_scope == "Mobility regime cluster" and cluster_label is not None:
        filters["mobility_regime_cluster_label"] = cluster_label

    panel = apply_common_filters(panel, **filters)

    zones_in_scope = int(panel["taxi_zone_id"].nunique()) if "taxi_zone_id" in panel.columns else 0

    daily = aggregate_metrics(
        panel,
        metrics=[metric],
        group_cols=["date"],
    )

    daily = _build_date_frame(daily, value_column=metric)
    daily = daily.sort_values("date").reset_index(drop=True)

    daily[f"{metric}_smoothed"] = _apply_smoothing(
        daily[metric],
        enabled=smoothing_enabled,
    )
    daily = _add_index_column(
        daily,
        source_column=f"{metric}_smoothed",
        index_column="display_index",
    )

    return daily, zones_in_scope


def _build_anomaly_rate_series(
    *,
    geography_scope: str,
    borough: str | None,
    geo_policy_group: str | None,
    cluster_label: int | None,
    smoothing_enabled: bool,
) -> tuple[pd.DataFrame, int]:
    """Build the daily share of in-scope zones with a finalist anomaly.

    The canonical anomaly events are Taxi Zone × date × daypart. This page is
    daily, so multiple anomalous dayparts in the same zone on the same date
    collapse to one anomalous-zone observation.

    The denominator is the number of Taxi Zones represented in the selected
    geography on each date. Geography membership comes from the analysis
    panel so Mobility Environment filtering remains period-aware.
    """
    panel = load_analysis_panel(
        columns=[
            "taxi_zone_id",
            "date",
            "borough",
            "cbd_spatial_category",
            "pre_post_cp",
        ]
    )

    filters: dict[str, object] = {}

    if geography_scope == "Borough" and borough:
        filters["borough"] = borough

    elif geography_scope == "Geo-policy group" and geo_policy_group:
        filters["cbd_spatial_category"] = geo_policy_group

    elif (
        geography_scope == "Mobility regime cluster"
        and cluster_label is not None
    ):
        filters["mobility_regime_cluster_label"] = cluster_label

    panel = apply_common_filters(
        panel,
        **filters,
    )

    if panel.empty:
        empty = pd.DataFrame(
            {
                "date": pd.to_datetime([]),
                ANOMALY_RATE_METRIC: pd.Series(dtype=float),
            }
        )

        empty = _build_date_frame(
            empty,
            value_column=ANOMALY_RATE_METRIC,
        )

        empty[f"{ANOMALY_RATE_METRIC}_smoothed"] = np.nan
        empty["display_index"] = np.nan

        return empty, 0

    panel = panel.copy()
    panel["date"] = pd.to_datetime(
        panel["date"]
    )

    # The analysis panel contains temporal-bucket rows. Reduce it to the
    # daily geography membership needed for this page.
    scope_zone_dates = (
        panel[
            [
                "taxi_zone_id",
                "date",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    zones_in_scope = int(
        scope_zone_dates[
            "taxi_zone_id"
        ].nunique()
    )

    daily_scope = (
        scope_zone_dates.groupby(
            "date",
            observed=True,
            dropna=False,
        )
        .agg(
            scope_zone_count=(
                "taxi_zone_id",
                "nunique",
            ),
        )
        .reset_index()
    )

    anomaly_events = load_selected_anomaly_events()

    if anomaly_events.empty:
        daily_anomalies = pd.DataFrame(
            {
                "date": pd.to_datetime([]),
                "finalist_anomaly_zone_count": pd.Series(
                    dtype="int64"
                ),
            }
        )

    else:
        anomaly_zone_dates = (
            anomaly_events[
                [
                    "taxi_zone_id",
                    "date",
                ]
            ]
            .copy()
        )

        anomaly_zone_dates["date"] = pd.to_datetime(
            anomaly_zone_dates["date"]
        )

        # Collapse multiple anomalous dayparts in the same zone/date.
        anomaly_zone_dates = (
            anomaly_zone_dates
            .drop_duplicates(
                [
                    "taxi_zone_id",
                    "date",
                ]
            )
        )

        # The filtered analysis panel defines geography membership. This
        # avoids separately reimplementing Borough, geo-policy, or
        # period-aware Mobility Environment logic against the anomaly file.
        anomaly_zone_dates = (
            anomaly_zone_dates.merge(
                scope_zone_dates,
                on=[
                    "taxi_zone_id",
                    "date",
                ],
                how="inner",
                validate="one_to_one",
            )
        )

        daily_anomalies = (
            anomaly_zone_dates.groupby(
                "date",
                observed=True,
                dropna=False,
            )
            .agg(
                finalist_anomaly_zone_count=(
                    "taxi_zone_id",
                    "nunique",
                ),
            )
            .reset_index()
        )

    daily = daily_scope.merge(
        daily_anomalies,
        on="date",
        how="left",
        validate="one_to_one",
    )

    daily[
        "finalist_anomaly_zone_count"
    ] = (
        daily[
            "finalist_anomaly_zone_count"
        ]
        .fillna(0)
        .astype("int64")
    )

    daily[ANOMALY_RATE_METRIC] = np.where(
        daily["scope_zone_count"].gt(0),
        (
            daily["finalist_anomaly_zone_count"]
            / daily["scope_zone_count"]
            * 100
        ),
        np.nan,
    )

    daily = _build_date_frame(
        daily,
        value_column=ANOMALY_RATE_METRIC,
    )

    daily = daily.sort_values(
        "date"
    ).reset_index(drop=True)

    daily[
        f"{ANOMALY_RATE_METRIC}_smoothed"
    ] = _apply_smoothing(
        daily[ANOMALY_RATE_METRIC],
        enabled=smoothing_enabled,
    )

    daily = _add_index_column(
        daily,
        source_column=(
            f"{ANOMALY_RATE_METRIC}_smoothed"
        ),
        index_column="display_index",
    )

    return daily, zones_in_scope


def _cycle_coordinates(date_series: pd.Series) -> pd.DataFrame:
    dates = pd.to_datetime(date_series)
    frame = pd.DataFrame({"date": dates})
    frame = frame[~((frame["date"].dt.month == 2) & (frame["date"].dt.day == 29))].copy()

    year_start = frame["date"].dt.year.min()
    year_offset = frame["date"].dt.year - year_start

    cycle_day = frame["date"].dt.dayofyear
    leap_after_feb = frame["date"].dt.is_leap_year & (
        (frame["date"].dt.month > 2)
        | ((frame["date"].dt.month == 2) & (frame["date"].dt.day > 29))
    )
    cycle_day = cycle_day - leap_after_feb.astype(int)

    theta = 2 * math.pi * ((cycle_day - 1) / ANNUAL_CYCLE_DAYS)
    radius = year_offset + ((cycle_day - 1) / ANNUAL_CYCLE_DAYS)

    frame["theta"] = theta
    frame["radius"] = radius
    frame["x"] = radius * np.cos(theta)
    frame["y"] = radius * np.sin(theta)
    return frame


def _apply_plot_branding(fig: go.Figure) -> go.Figure:
    fig = apply_branding(fig)
    fig.update_layout(
        margin={"l": 30, "r": 20, "t": 60, "b": 40},
        legend={"orientation": "h", "y": -0.15, "x": 0},
    )
    return fig


def _segments_to_xy(segments: list[list[float]]) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for x0, y0, x1, y1 in segments:
        xs.extend([x0, x1, None])
        ys.extend([y0, y1, None])
    return xs, ys


def _month_name(month: int | float | None) -> str:
    if month is None or pd.isna(month):
        return "an unknown month"
    return MONTH_NAMES.get(int(month), "an unknown month")


def _format_percent_text(value: float | None, *, digits: int = 1, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    sign = "+" if signed else ""
    return f"{value:{sign}.{digits}f}%"


def _build_shape_metrics(daily: pd.DataFrame) -> dict[str, object]:
    frame = daily[["date", "display_index"]].dropna(subset=["display_index"]).copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date")

    if frame.empty:
        return {
            "seasonality_swing_pct": np.nan,
            "peak_month": np.nan,
            "trough_month": np.nan,
            "peak_month_name": "an unknown month",
            "trough_month_name": "an unknown month",
            "post_cp_shift_pct": np.nan,
            "latest_12m_change_pct": np.nan,
            "annual_mean": np.nan,
        }

    series = frame.set_index("date")["display_index"].astype(float)
    monthly = series.groupby(series.index.month).mean()
    annual_mean = float(series.mean()) if len(series) else np.nan

    seasonality_swing_pct = np.nan
    if pd.notna(annual_mean) and annual_mean != 0 and not monthly.empty:
        seasonality_swing_pct = ((monthly.max() - monthly.min()) / annual_mean) * 100

    peak_month = int(monthly.idxmax()) if not monthly.empty else np.nan
    trough_month = int(monthly.idxmin()) if not monthly.empty else np.nan

    pre_mask = frame["date"] < CONGESTION_PRICING_START_DATE
    post_mask = frame["date"] >= CONGESTION_PRICING_START_DATE
    pre_avg = frame.loc[pre_mask, "display_index"].mean(skipna=True)
    post_avg = frame.loc[post_mask, "display_index"].mean(skipna=True)
    post_cp_shift_pct = np.nan
    if pd.notna(pre_avg) and pre_avg != 0 and pd.notna(post_avg):
        post_cp_shift_pct = ((post_avg - pre_avg) / pre_avg) * 100

    latest_date = series.index.max()
    latest_window_start = latest_date - pd.Timedelta(days=365)
    prior_window_start = latest_window_start - pd.Timedelta(days=365)
    latest_12m = series.loc[series.index > latest_window_start]
    prior_12m = series.loc[(series.index > prior_window_start) & (series.index <= latest_window_start)]
    latest_12m_change_pct = np.nan
    if len(latest_12m) and len(prior_12m):
        prior_avg = prior_12m.mean()
        latest_avg = latest_12m.mean()
        if pd.notna(prior_avg) and prior_avg != 0 and pd.notna(latest_avg):
            latest_12m_change_pct = ((latest_avg - prior_avg) / prior_avg) * 100

    return {
        "seasonality_swing_pct": seasonality_swing_pct,
        "peak_month": peak_month,
        "trough_month": trough_month,
        "peak_month_name": _month_name(peak_month),
        "trough_month_name": _month_name(trough_month),
        "post_cp_shift_pct": post_cp_shift_pct,
        "latest_12m_change_pct": latest_12m_change_pct,
        "annual_mean": annual_mean,
    }


def _shape_insight_text(*, metric_label: str, scope_label: str, shape: dict[str, object]) -> str:
    seasonality_swing_pct = shape["seasonality_swing_pct"]
    peak_month_name = shape["peak_month_name"]
    trough_month_name = shape["trough_month_name"]
    post_cp_shift_pct = shape["post_cp_shift_pct"]
    latest_12m_change_pct = shape["latest_12m_change_pct"]

    if pd.isna(seasonality_swing_pct):
        seasonal_clause = "The annual pattern is not stable enough to summarize cleanly."
    elif seasonality_swing_pct < 8:
        seasonal_clause = f"The annual swing is fairly muted at {seasonality_swing_pct:.1f}% of the mean."
    elif seasonality_swing_pct < 15:
        seasonal_clause = f"The annual swing is visible but moderate at {seasonality_swing_pct:.1f}% of the mean."
    else:
        seasonal_clause = f"The annual swing is strong at {seasonality_swing_pct:.1f}% of the mean."

    if pd.isna(post_cp_shift_pct):
        shift_clause = "The post-CP level shift could not be estimated cleanly."
    elif pd.notna(seasonality_swing_pct) and abs(post_cp_shift_pct) >= seasonality_swing_pct:
        shift_clause = (
            f"The post-CP shift ({post_cp_shift_pct:+.1f}%) is larger than the seasonal swing, "
            "so the level change matters more than the loop itself."
        )
    else:
        shift_clause = (
            f"The post-CP shift ({post_cp_shift_pct:+.1f}%) is smaller than the seasonal swing, "
            "so the loop shape still carries the main story."
        )

    yoy_clause = ""
    if pd.notna(latest_12m_change_pct):
        yoy_clause = f" The most recent 12-month average is {latest_12m_change_pct:+.1f}% versus the prior 12 months."

    return (
        f"{metric_label} in {scope_label} peaks in {peak_month_name} and bottoms out in {trough_month_name}. "
        f"{seasonal_clause} {shift_clause}{yoy_clause}"
    )


@st.cache_data(show_spinner=False)
def _prepare_spiral_geometry(daily: pd.DataFrame) -> dict[str, object]:
    spiral = daily[["date", "display_index"]].dropna(subset=["display_index"]).copy()
    spiral = spiral[~((spiral["date"].dt.month == 2) & (spiral["date"].dt.day == 29))].copy()
    spiral = spiral.sort_values("date").reset_index(drop=True)

    if spiral.empty:
        return {
            "spiral": spiral,
            "xy": np.empty((0, 2), dtype=float),
            "radial_max": 1.0,
            "max_abs_delta": 1.0,
            "bar_scale": 0.23,
            "positive_segments": [],
            "negative_segments": [],
            "hover_x": [],
            "hover_y": [],
            "hover_customdata": np.empty((0, 4), dtype=object),
            "year_markers": pd.DataFrame(columns=["x", "y", "year"]),
            "cp_point": None,
            "lim": 1.5,
        }

    coords = _cycle_coordinates(spiral["date"])
    spiral = pd.concat([spiral, coords[["x", "y"]].reset_index(drop=True)], axis=1)

    xy = spiral[["x", "y"]].to_numpy(dtype=float)
    values = spiral["display_index"].to_numpy(dtype=float)
    deltas = values - 100.0

    radial_max = float(np.nanmax(np.sqrt((xy**2).sum(axis=1))))
    if not np.isfinite(radial_max) or radial_max <= 0:
        radial_max = 1.0

    max_abs_delta = float(np.nanpercentile(np.abs(deltas), 95))
    if not np.isfinite(max_abs_delta) or max_abs_delta <= 0:
        max_abs_delta = 1.0

    bar_scale = radial_max * 0.23

    def _unit_normal(prev_point: np.ndarray, point: np.ndarray, next_point: np.ndarray) -> np.ndarray:
        tangent = next_point - prev_point
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        length = float(np.linalg.norm(normal))
        if not np.isfinite(length) or length == 0:
            return np.array([0.0, 0.0], dtype=float)
        normal = normal / length
        if float(np.dot(normal, point)) < 0:
            normal = -normal
        return normal

    positive_segments: list[list[float]] = []
    negative_segments: list[list[float]] = []
    hover_x: list[float] = []
    hover_y: list[float] = []
    hover_customdata: list[list[object]] = []

    bar_min = 0.10
    bar_max = 0.92
    for idx in range(len(spiral)):
        point = xy[idx]
        prev_point = xy[idx - 1] if idx > 0 else xy[idx]
        next_point = xy[idx + 1] if idx < len(spiral) - 1 else xy[idx]
        normal = _unit_normal(prev_point, point, next_point)
        if not np.any(normal):
            continue

        bar_length = np.clip(deltas[idx] / max_abs_delta, -1.0, 1.0)
        magnitude = bar_min + (abs(bar_length) * (bar_max - bar_min))
        signed_length = np.sign(bar_length) * magnitude * bar_scale
        end_point = point + normal * signed_length

        segment = [float(point[0]), float(point[1]), float(end_point[0]), float(end_point[1])]
        if bar_length >= 0:
            positive_segments.append(segment)
        else:
            negative_segments.append(segment)

        hover_x.append(float(end_point[0]))
        hover_y.append(float(end_point[1]))
        hover_customdata.append(
            [
                spiral.iloc[idx]["date"].strftime("%Y-%m-%d"),
                f"{float(values[idx]):.3f}",
                f"{float(deltas[idx]):+.3f}",
                "Above baseline" if bar_length >= 0 else "Below baseline",
            ]
        )

    year_markers = spiral.loc[
        spiral["date"].dt.month.eq(1) & spiral["date"].dt.day.eq(1),
        ["date", "x", "y"],
    ].copy()
    year_markers["year"] = year_markers["date"].dt.year.astype(int)

    cp_rows = spiral.loc[
        spiral["date"].dt.normalize().eq(CONGESTION_PRICING_START_DATE.normalize()),
        ["date", "x", "y", "display_index"],
    ]
    if cp_rows.empty:
        cp_rows = spiral.iloc[(spiral["date"] - CONGESTION_PRICING_START_DATE).abs().argsort()[:1]][
            ["date", "x", "y", "display_index"]
        ]
    cp_point = cp_rows.iloc[0].to_dict() if not cp_rows.empty else None

    return {
        "spiral": spiral,
        "xy": xy,
        "radial_max": radial_max,
        "max_abs_delta": max_abs_delta,
        "bar_scale": bar_scale,
        "positive_segments": positive_segments,
        "negative_segments": negative_segments,
        "hover_x": hover_x,
        "hover_y": hover_y,
        "hover_customdata": np.asarray(hover_customdata, dtype=object),
        "year_markers": year_markers,
        "cp_point": cp_point,
        "lim": radial_max + (bar_scale * 1.25),
    }


def _build_spiral_hero_figure(
    daily: pd.DataFrame,
) -> go.Figure:
    geometry = _prepare_spiral_geometry(daily)
    spiral = geometry["spiral"]
    if spiral.empty:
        return _apply_plot_branding(go.Figure())

    fig = go.Figure()
    lim = float(geometry["lim"])

    _add_season_wedges(fig, outer_radius=lim * 1.01)
    _add_season_labels(fig, outer_radius=lim * 1.01)

    def _segments_to_xy(segments: list[list[float]]) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        for x0, y0, x1, y1 in segments:
            xs.extend([x0, x1, None])
            ys.extend([y0, y1, None])
        return xs, ys

    positive_x, positive_y = _segments_to_xy(geometry["positive_segments"])
    negative_x, negative_y = _segments_to_xy(geometry["negative_segments"])

    fig.add_trace(
        go.Scatter(
            x=spiral["x"],
            y=spiral["y"],
            mode="lines",
            line={"color": "rgba(0, 109, 119, 0.18)", "width": 5.8},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=spiral["x"],
            y=spiral["y"],
            mode="lines",
            line={"color": BRAND_COLORS["dark_teal"], "width": 2.1},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=positive_x,
            y=positive_y,
            mode="lines",
            line={"color": BRAND_COLORS["dark_teal"], "width": 2.1},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=negative_x,
            y=negative_y,
            mode="lines",
            line={"color": BRAND_COLORS["terracotta"], "width": 2.1},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=geometry["hover_x"],
            y=geometry["hover_y"],
            mode="markers",
            marker={"size": 10, "color": "rgba(0,0,0,0.01)"},
            customdata=geometry["hover_customdata"],
            hovertemplate=(
                "Date: %{customdata[0]}<br>"
                "Indexed value: %{customdata[1]}<br>"
                "Delta vs baseline: %{customdata[2]}<br>"
                "Position: %{customdata[3]}<extra></extra>"
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    year_markers = geometry["year_markers"]
    if not year_markers.empty:
        fig.add_trace(
            go.Scatter(
                x=year_markers["x"],
                y=year_markers["y"],
                mode="markers+text",
                text=year_markers["year"].astype(str).tolist(),
                textposition="middle right",
                marker={
                    "size": 8,
                    "color": "white",
                    "line": {"width": 1.2, "color": BRAND_COLORS["terracotta"]},
                },
                textfont={"size": 9, "color": BRAND_COLORS["terracotta"]},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    cp_point = geometry["cp_point"]
    if cp_point is not None:
        fig.add_trace(
            go.Scatter(
                x=[cp_point["x"]],
                y=[cp_point["y"]],
                mode="markers+text",
                text=["CP start"],
                textposition="top right",
                marker={
                    "size": 12,
                    "symbol": "star",
                    "color": BRAND_COLORS["terracotta"],
                    "line": {"width": 1, "color": "white"},
                },
                textfont={"size": 10, "color": BRAND_COLORS["terracotta"]},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_xaxes(visible=False, range=[-lim, lim], scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False, range=[-lim, lim], title="")
    fig.update_layout(
        title={"text": ""},
        paper_bgcolor="white",
        plot_bgcolor=BRAND_COLORS["ice"],
        height=700,
        margin={"l": 20, "r": 20, "t": 10, "b": 20},
        showlegend=False,
    )
    return _apply_plot_branding(fig)


def _build_spiral_figure(
    daily: pd.DataFrame,
    *,
    metric_label: str,
) -> go.Figure:
    geometry = _prepare_spiral_geometry(daily)
    spiral = geometry["spiral"]

    if spiral.empty:
        return _apply_plot_branding(go.Figure())

    fig = go.Figure()
    lim = float(geometry["lim"])
    _add_season_wedges(fig, outer_radius=lim * 1.01)
    _add_season_labels(fig, outer_radius=lim * 1.01)
    positive_x, positive_y = _segments_to_xy(geometry["positive_segments"])
    negative_x, negative_y = _segments_to_xy(geometry["negative_segments"])

    fig.add_trace(
        go.Scatter(
            x=spiral["x"],
            y=spiral["y"],
            mode="lines",
            line={"color": "rgba(0, 109, 119, 0.18)", "width": 5.8},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=spiral["x"],
            y=spiral["y"],
            mode="lines",
            line={"color": BRAND_COLORS["dark_teal"], "width": 2.1},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=positive_x,
            y=positive_y,
            mode="lines",
            line={"color": BRAND_COLORS["dark_teal"], "width": 2.1},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=negative_x,
            y=negative_y,
            mode="lines",
            line={"color": BRAND_COLORS["terracotta"], "width": 2.1},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=geometry["hover_x"],
            y=geometry["hover_y"],
            mode="markers",
            marker={"size": 10, "color": "rgba(0,0,0,0.01)"},
            customdata=geometry["hover_customdata"],
            hovertemplate=(
                "Date: %{customdata[0]}<br>"
                "Indexed value: %{customdata[1]}<br>"
                "Delta vs baseline: %{customdata[2]}<br>"
                "Position: %{customdata[3]}<extra></extra>"
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    year_markers = geometry["year_markers"]
    if not year_markers.empty:
        fig.add_trace(
            go.Scatter(
                x=year_markers["x"],
                y=year_markers["y"],
                mode="markers+text",
                text=year_markers["year"].astype(str).tolist(),
                textposition="middle right",
                marker={
                    "size": 8,
                    "color": "white",
                    "line": {"width": 1.2, "color": BRAND_COLORS["terracotta"]},
                },
                textfont={"size": 9, "color": BRAND_COLORS["terracotta"]},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    cp_point = geometry["cp_point"]
    if cp_point is not None:
        fig.add_trace(
            go.Scatter(
                x=[cp_point["x"]],
                y=[cp_point["y"]],
                mode="markers+text",
                text=["CP start"],
                textposition="top right",
                marker={
                    "size": 12,
                    "symbol": "star",
                    "color": BRAND_COLORS["terracotta"],
                    "line": {"width": 1, "color": "white"},
                },
                textfont={"size": 10, "color": BRAND_COLORS["terracotta"]},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_xaxes(
        visible=False,
        constrain="domain",
        scaleanchor="y",
        scaleratio=1,
    )
    fig.update_yaxes(
        visible=False,
        title="",
    )

    fig.update_layout(
        title={"text": ""},
        paper_bgcolor="white",
        plot_bgcolor=BRAND_COLORS["ice"],
        height=700,
        margin={"l": 20, "r": 20, "t": 10, "b": 30},
        showlegend=False,
    )

    return _apply_plot_branding(fig)


def _build_timeline_figure(
    daily: pd.DataFrame,
    *,
    metric_label: str,
    spiral_style: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["display_index"],
            mode="lines",
            line={"color": BRAND_COLORS["dark_teal"], "width": 2.4},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    if spiral_style == "Sparkline + value dots":
        fig.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily["display_index"],
                mode="markers",
                marker={
                    "size": 4,
                    "color": daily["display_index"],
                    "colorscale": [
                        [0.0, BRAND_DIVERGING_SEQUENCE[4]],
                        [0.25, BRAND_DIVERGING_SEQUENCE[3]],
                        [0.5, BRAND_DIVERGING_SEQUENCE[2]],
                        [0.75, BRAND_DIVERGING_SEQUENCE[1]],
                        [1.0, BRAND_DIVERGING_SEQUENCE[0]],
                    ],
                    "cmin": 85,
                    "cmax": 115,
                    "line": {"width": 0},
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_vline(
        x=CONGESTION_PRICING_START_DATE,
        line_width=1.8,
        line_dash="dash",
        line_color=BRAND_COLORS["terracotta"],
    )

    fig.add_annotation(
        x=CONGESTION_PRICING_START_DATE,
        y=1.02,
        xref="x",
        yref="paper",
        text="Congestion pricing starts",
        showarrow=False,
        font={"size": 12, "color": BRAND_COLORS["terracotta"]},
        xanchor="center",
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Indexed value",
        title={"text": ""},
        height=260,
        margin={"l": 50, "r": 30, "t": 0, "b": 48},
        showlegend=False,
    )

    return _apply_plot_branding(fig)


def _summary_cards(
    *,
    metric_label: str,
    scope_label: str,
    daily: pd.DataFrame,
    zones_in_scope: int,
) -> None:
    pre_mask = daily["date"] < CONGESTION_PRICING_START_DATE
    post_mask = daily["date"] >= CONGESTION_PRICING_START_DATE

    pre_avg = daily.loc[pre_mask, "display_index"].mean(skipna=True)
    post_avg = daily.loc[post_mask, "display_index"].mean(skipna=True)
    pct_change = np.nan
    if pd.notna(pre_avg) and pre_avg != 0 and pd.notna(post_avg):
        pct_change = ((post_avg - pre_avg) / pre_avg) * 100

    observed_days = int(daily["display_index"].notna().sum())
    latest_value = daily["display_index"].dropna().iloc[-1] if daily["display_index"].notna().any() else np.nan

    cards = st.columns(5)

    card_payload = [
        ("Focus", metric_label),
        ("Scope", scope_label),
        ("Pre-CP avg", _format_value(pre_avg)),
        ("Post-CP avg", _format_value(post_avg)),
        ("Change", _format_change(pct_change)),
    ]

    for column, (label, value) in zip(cards, card_payload, strict=True):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:0.85rem; opacity:0.8; margin-bottom:0.2rem;">{label}</div>
                    <div style="font-size:1.35rem; font-weight:700; line-height:1.15;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(
        f"{zones_in_scope} zones in scope - {observed_days} observed days - "
        f"latest indexed value {_format_value(latest_value)}"
    )


st.caption("PAGE 11")
st.title(f"{PAGE_TITLE}: Which mobility series repeat on a 365-day cycle, and does the rhythm shift after congestion pricing?")
st.write(
    "The spiral compresses three years into a recurring loop so the viewer can see whether each series follows a stable seasonal rhythm, a meaningful post-CP shift, or both."
)

hero_story_name = st.segmented_control(
    "Featured story",
    options=FEATURED_STORY_OPTIONS,
    default=FEATURED_STORY_OPTIONS[0],
    key="raw11_featured_story",
) or FEATURED_STORY_OPTIONS[0]

hero_profile = FEATURED_STORIES[hero_story_name.replace(" - ", " · ")]

hero_daily, hero_zones_in_scope = _build_metric_series(
    metric=hero_profile["metric"],
    geography_scope=hero_profile["geography_scope"],
    borough=hero_profile["borough"],
    geo_policy_group=hero_profile["geo_policy"],
    cluster_label=hero_profile["cluster"],
    smoothing_enabled=True,
)

hero_metric_label = _metric_label(hero_profile["metric"])
hero_scope_label = _scope_summary(
    hero_profile["geography_scope"],
    borough=hero_profile["borough"],
    geo_policy=hero_profile["geo_policy"],
    cluster_label=hero_profile["cluster"],
)
hero_shape = _build_shape_metrics(hero_daily)
hero_shape_headline = (
    f"{hero_metric_label} peaks in {hero_shape['peak_month_name']} and bottoms out in {hero_shape['trough_month_name']}"
)
hero_shape_insight = _shape_insight_text(
    metric_label=hero_metric_label,
    scope_label=hero_scope_label,
    shape=hero_shape,
)

hero_pre_mask = hero_daily["date"] < CONGESTION_PRICING_START_DATE
hero_post_mask = hero_daily["date"] >= CONGESTION_PRICING_START_DATE
hero_pre_avg = hero_daily.loc[hero_pre_mask, "display_index"].mean(skipna=True)
hero_post_avg = hero_daily.loc[hero_post_mask, "display_index"].mean(skipna=True)
hero_pct_change = np.nan
if pd.notna(hero_pre_avg) and hero_pre_avg != 0 and pd.notna(hero_post_avg):
    hero_pct_change = ((hero_post_avg - hero_pre_avg) / hero_pre_avg) * 100
hero_observed_days = int(hero_daily["display_index"].notna().sum())
hero_latest = (
    hero_daily["display_index"].dropna().iloc[-1]
    if hero_daily["display_index"].notna().any()
    else np.nan
)

hero_story_col, hero_stats_col = st.columns([1.4, 1.1], gap="large")
with hero_story_col:
    st.markdown(f"#### {hero_shape_headline}")
    st.write(hero_shape_insight)

with hero_stats_col:
    hero_stat_cols = st.columns(2)
    hero_stat_cols[0].metric("Pre-CP avg", _format_value(hero_pre_avg))
    hero_stat_cols[1].metric("Post-CP avg", _format_value(hero_post_avg))
    hero_stat_cols_2 = st.columns(2)
    hero_stat_cols_2[0].metric("Change", _format_change(hero_pct_change))
    hero_stat_cols_2[1].metric("Zones in view", f"{hero_zones_in_scope:,}")
    hero_stat_cols_3 = st.columns(2)
    hero_stat_cols_3[0].metric("Seasonality swing", _format_percent_text(hero_shape["seasonality_swing_pct"]))
    hero_stat_cols_3[1].metric("Peak month", hero_shape["peak_month_name"])
    st.caption(
        f"{hero_observed_days} observed days - latest indexed value {_format_value(hero_latest)}"
    )

st.markdown(
    f"""
    <div class="soft-callout" style="margin-top:1rem;">
        <strong>What to notice:</strong> the annual swing is {_format_percent_text(hero_shape['seasonality_swing_pct'])}
        of the mean. The post-CP shift is {_format_percent_text(hero_shape['post_cp_shift_pct'], signed=True)}
        and the most recent 12-month average is {_format_percent_text(hero_shape['latest_12m_change_pct'], signed=True)}
        versus the prior 12 months.
    </div>
    """,
    unsafe_allow_html=True,
)

hero_spiral_figure = _build_spiral_hero_figure(hero_daily)
st.markdown("#### Annual spiral")
st.caption(
    "Each loop is one year. The bars grow from the spiral itself, and the year-start markers keep the cycle anchored."
)
st.plotly_chart(hero_spiral_figure, width="stretch", key="raw11_hero_spiral")

st.markdown("### Explorer")
st.write(
    "Use the explorer below to change the metric and geography. The spiral-bar geometry stays fixed so the comparison is always made on the same visual language."
)

scope_options = _get_scope_options()

controls = st.columns([1.2, 1.2, 1.0, 1.0])

with controls[0]:
    metric_choice = st.selectbox(
        "Metric",
        options=[*BASE_METRICS, ANOMALY_RATE_METRIC],
        index=[*BASE_METRICS, ANOMALY_RATE_METRIC].index(DEFAULT_METRIC),
        format_func=_metric_label,
        key="raw11_metric",
    )

with controls[1]:
    geography_scope = st.selectbox(
        "Geography",
        options=GEOGRAPHY_OPTIONS,
        index=0,
        key="raw11_geography_scope",
    )

borough_choice: str | None = None
geo_policy_choice: str | None = None
cluster_choice: int | None = None

with controls[2]:
    if geography_scope == "Borough":
        if scope_options["boroughs"]:
            borough_choice = st.selectbox(
                "Borough",
                options=scope_options["boroughs"],
                index=0,
                key="raw11_borough",
            )
        else:
            st.info("No borough values are available in the current data.")
    elif geography_scope == "Geo-policy group":
        if scope_options["geo_policy"]:
            geo_policy_choice = st.selectbox(
                "Geo-policy group",
                options=scope_options["geo_policy"],
                index=0,
                key="raw11_geo_policy",
            )
        else:
            st.info("No geo-policy groups are available in the current data.")
    elif geography_scope == "Mobility regime cluster":
        if scope_options["clusters"]:
            cluster_choice = st.selectbox(
                "Cluster",
                options=scope_options["clusters"],
                index=0,
                format_func=format_mobility_regime_cluster_label,
                key="raw11_cluster",
            )
        else:
            st.info("No mobility regime clusters are available in the current data.")
    else:
        st.markdown(
            "<div style='padding-top:1.75rem; color:#6b7d7f;'>No sub-selection needed.</div>",
            unsafe_allow_html=True,
        )

with controls[3]:
    smoothing_enabled = st.checkbox(
        "Smooth daily values",
        value=True,
        key="raw11_smoothing",
    )

if metric_choice == ANOMALY_RATE_METRIC:
    daily, zones_in_scope = _build_anomaly_rate_series(
        geography_scope=geography_scope,
        borough=borough_choice,
        geo_policy_group=geo_policy_choice,
        cluster_label=cluster_choice,
        smoothing_enabled=smoothing_enabled,
    )
else:
    daily, zones_in_scope = _build_metric_series(
        metric=metric_choice,
        geography_scope=geography_scope,
        borough=borough_choice,
        geo_policy_group=geo_policy_choice,
        cluster_label=cluster_choice,
        smoothing_enabled=smoothing_enabled,
    )

metric_label = _metric_label(metric_choice)
scope_label = _scope_summary(
    geography_scope,
    borough=borough_choice,
    geo_policy=geo_policy_choice,
    cluster_label=cluster_choice,
)

st.divider()
_summary_cards(
    metric_label=metric_label,
    scope_label=scope_label,
    daily=daily,
    zones_in_scope=zones_in_scope,
)

if daily["display_index"].notna().sum() == 0:
    st.warning(
        "There is not enough data in the current selection to draw the cyclical view."
    )
    st.stop()

explorer_shape = _build_shape_metrics(daily)
explorer_shape_text = _shape_insight_text(
    metric_label=metric_label,
    scope_label=scope_label,
    shape=explorer_shape,
)

st.markdown(
    """
    <div class="soft-callout">
        <strong>What to notice:</strong> look for recurring calendar peaks, whether the annual swing is actually large or just decorative, and whether the latest 12-month average is moving faster than the long-run seasonal loop.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="soft-callout">
        <strong>Selection insight:</strong> {explorer_shape_text}
    </div>
    """,
    unsafe_allow_html=True,
)

shape_cols = st.columns(4)
shape_cols[0].metric("Seasonality swing", _format_percent_text(explorer_shape["seasonality_swing_pct"]))
shape_cols[1].metric("Post-CP shift", _format_percent_text(explorer_shape["post_cp_shift_pct"], signed=True))
shape_cols[2].metric("Peak month", explorer_shape["peak_month_name"])
shape_cols[3].metric("Latest 12m change", _format_percent_text(explorer_shape["latest_12m_change_pct"], signed=True))

spiral_fig = _build_spiral_figure(
    daily,
    metric_label=metric_label,
)
timeline_fig = _build_timeline_figure(
    daily,
    metric_label=metric_label,
    spiral_style="Sparkline + value dots",
)

st.markdown("#### Explorer spiral")
st.plotly_chart(spiral_fig, width="stretch", key="raw11_spiral")
st.markdown("#### Chronological sparkline")
st.plotly_chart(timeline_fig, width="stretch", key="raw11_timeline")

show_data_table = st.checkbox(
    "Show data table",
    value=False,
    key="raw11_show_data_table",
)
if show_data_table:
    table_columns = ["date"]
    if metric_choice == ANOMALY_RATE_METRIC:
        table_columns.extend(
            [
                "finalist_anomaly_zone_count",
                "scope_zone_count",
                ANOMALY_RATE_METRIC,
                f"{ANOMALY_RATE_METRIC}_smoothed",
                "display_index",
            ]
        )
    else:
        table_columns.extend([
            metric_choice,
            f"{metric_choice}_smoothed",
            "display_index",
        ])
    display_table = daily[table_columns].copy()
    display_table["date"] = pd.to_datetime(display_table["date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(display_table, width="stretch", hide_index=True, height=320)

with st.expander("How to read these numbers", expanded=False):
    st.markdown(
        """
        - **Seasonality swing** means the gap between the highest and lowest monthly averages, expressed as a share of the overall mean. Bigger numbers mean the series has a stronger recurring seasonal pattern.
        - **Indexing** means every series is normalized to its own pre-congestion-pricing daily average, so `100` is the baseline before CP started and values above or below 100 show relative movement from that baseline.
        - **Post-CP shift** compares the average after congestion pricing to the average before it, so it shows the longer-run level change rather than the seasonal loop itself.
        """
    )

st.markdown("### Useful questions")
st.write(
    "Do anomalies recur in the same calendar positions? Did the post-CP period alter an established seasonal rhythm? Are some modes more cyclical than others?"
)
