from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data_access.weather_relationships import (
    ALL_TEMPORAL_BUCKETS_LABEL,
    MOBILITY_LABELS,
    MOBILITY_METRICS,
    TEMPORAL_BUCKET_LABELS,
    TEMPORAL_BUCKETS,
    WEATHER_LABELS,
    WEATHER_METRICS,
    build_weather_relationship_pair_data,
    calculate_relationship_statistics,
    load_relationship_date_bounds,
    load_zone_lookup,
)


CP_START_DATE = pd.Timestamp("2025-01-05")

BRAND_COLORS = {
    "dark_teal": "#006D77",
    "seafoam": "#83C5BE",
    "ice": "#EDF6F9",
    "pale_peach": "#FFDDD2",
    "terracotta": "#E29578",
    "charcoal": "#243238",
    "gray": "#66747A",
}

PRE_COLOR = BRAND_COLORS["dark_teal"]
POST_COLOR = BRAND_COLORS["terracotta"]
WEATHER_COLOR = BRAND_COLORS["dark_teal"]
MOBILITY_COLOR = BRAND_COLORS["terracotta"]


@dataclass(frozen=True)
class SavedView:
    label: str
    mobility_metric: str
    weather_metric: str
    temporal_bucket: str
    geography: str
    description: str


SAVED_VIEWS = (
    SavedView(
        label="Weekday evening temperature and bus speed",
        mobility_metric="avg_bus_speed",
        weather_metric="temperature",
        temporal_bucket="weekday_evening",
        geography="Citywide",
        description=(
            "The strongest relationship in the citywide scan."
        ),
    ),
    SavedView(
        label="Weekday AM temperature and FHVHV demand",
        mobility_metric="fhvhv_trip_count",
        weather_metric="temperature",
        temporal_bucket="weekday_am_peak",
        geography="Citywide",
        description=(
            "A strong demand relationship during the morning peak."
        ),
    ),
    SavedView(
        label="Weekend evening temperature and bus speed",
        mobility_metric="avg_bus_speed",
        weather_metric="temperature",
        temporal_bucket="weekend_evening",
        geography="Citywide",
        description=(
            "A strong weekend version of the temperature–speed pattern."
        ),
    ),
    SavedView(
        label="Weekend PM rain and subway ridership",
        mobility_metric="subway_ridership",
        weather_metric="precipitation",
        temporal_bucket="weekend_pm_peak",
        geography="Citywide",
        description=(
            "A direct weather-and-transit demand comparison."
        ),
    ),
    SavedView(
        label="Weekend midday rain and FHVHV demand",
        mobility_metric="fhvhv_trip_count",
        weather_metric="precipitation",
        temporal_bucket="weekend_midday",
        geography="Citywide",
        description=(
            "A positive relationship between rain and for-hire demand."
        ),
    ),
    SavedView(
        label="Overall rain and subway ridership",
        mobility_metric="subway_ridership",
        weather_metric="precipitation",
        temporal_bucket=ALL_TEMPORAL_BUCKETS_LABEL,
        geography="Citywide",
        description=(
            "A broad, full-period view across the city."
        ),
    ),
)


def _relationship_strength(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"

    magnitude = abs(float(value))

    if magnitude >= 0.50:
        return "Strong"
    if magnitude >= 0.30:
        return "Moderate"
    if magnitude >= 0.15:
        return "Modest"
    return "Weak"


def _relationship_direction(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "unavailable"

    if value > 0.03:
        return "positive"
    if value < -0.03:
        return "negative"
    return "near-zero"


def _format_correlation(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:+.3f}"


def _format_shift(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:+.3f}"


def _extract_period_row(
    statistics: pd.DataFrame,
    period: str,
) -> pd.Series | None:
    rows = statistics[
        statistics["period"].eq(period)
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


def _safe_trend_line(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray] | None:
    valid = frame[
        ["weather_value", "mobility_value"]
    ].dropna()

    if (
        len(valid) < 2
        or valid["weather_value"].nunique() < 2
    ):
        return None

    x_values = np.linspace(
        valid["weather_value"].min(),
        valid["weather_value"].max(),
        100,
    )

    slope, intercept = np.polyfit(
        valid["weather_value"],
        valid["mobility_value"],
        deg=1,
    )

    y_values = slope * x_values + intercept
    return x_values, y_values


def _build_scatter_figure(
    pair_data: pd.DataFrame,
    *,
    weather_label: str,
    mobility_label: str,
    show_periods: bool,
) -> go.Figure:
    figure = go.Figure()

    if show_periods:
        period_settings = (
            ("pre_cp", "Before congestion pricing", PRE_COLOR),
            ("post_cp", "After congestion pricing", POST_COLOR),
        )
    else:
        period_settings = (
            ("all", "All observations", BRAND_COLORS["dark_teal"]),
        )

    for period, label, color in period_settings:
        if period == "all":
            frame = pair_data.copy()
        else:
            frame = pair_data[
                pair_data["pre_post_cp"].eq(period)
            ].copy()

        valid = frame[
            ["date", "weather_value", "mobility_value"]
        ].dropna()

        if valid.empty:
            continue

        figure.add_trace(
            go.Scatter(
                x=valid["weather_value"],
                y=valid["mobility_value"],
                mode="markers",
                name=label,
                marker={
                    "color": color,
                    "size": 7,
                    "opacity": 0.55,
                },
                customdata=np.stack(
                    [
                        valid["date"].dt.strftime("%b %d, %Y"),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    f"{weather_label}: %{{x:,.2f}}<br>"
                    f"{mobility_label}: %{{y:,.2f}}"
                    "<extra></extra>"
                ),
            )
        )

        trend_line = _safe_trend_line(frame)

        if trend_line is not None:
            x_values, y_values = trend_line
            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    name=f"{label} trend",
                    line={
                        "color": color,
                        "width": 3,
                    },
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    figure.update_layout(
        height=520,
        margin={
            "l": 20,
            "r": 20,
            "t": 25,
            "b": 20,
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        xaxis={
            "title": weather_label,
            "showgrid": True,
            "gridcolor": "rgba(102,116,122,0.15)",
            "zeroline": False,
        },
        yaxis={
            "title": mobility_label,
            "showgrid": True,
            "gridcolor": "rgba(102,116,122,0.15)",
            "zeroline": False,
        },
        hovermode="closest",
    )

    return figure


def _build_time_figure(
    pair_data: pd.DataFrame,
    *,
    weather_metric: str,
    weather_label: str,
    mobility_label: str,
) -> go.Figure:
    """
    Plot both measures in their original units.

    Mobility always uses the left axis. Weather always uses a separate right
    axis because its units and value range are not directly comparable with
    mobility measures.

    Precipitation and pressure change use bars because they are event-like or
    centered around zero. Other weather measures use lines.
    """
    frame = pair_data.sort_values("date").copy()

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["mobility_value"],
            mode="lines",
            name=mobility_label,
            line={
                "color": MOBILITY_COLOR,
                "width": 2,
            },
            yaxis="y",
            hovertemplate=(
                "<b>%{x|%b %d, %Y}</b><br>"
                f"{mobility_label}: %{{y:,.2f}}"
                "<extra></extra>"
            ),
        )
    )

    if weather_metric == "precipitation":
        figure.add_trace(
            go.Bar(
                x=frame["date"],
                y=frame["weather_value"],
                name=weather_label,
                marker={
                    "color": WEATHER_COLOR,
                },
                opacity=0.38,
                yaxis="y2",
                hovertemplate=(
                    "<b>%{x|%b %d, %Y}</b><br>"
                    f"{weather_label}: %{{y:,.3f}}"
                    "<extra></extra>"
                ),
            )
        )
    elif weather_metric == "pressure_3hr_change":
        figure.add_trace(
            go.Bar(
                x=frame["date"],
                y=frame["weather_value"],
                name=weather_label,
                marker={
                    "color": WEATHER_COLOR,
                },
                opacity=0.45,
                yaxis="y2",
                hovertemplate=(
                    "<b>%{x|%b %d, %Y}</b><br>"
                    f"{weather_label}: %{{y:,.2f}}"
                    "<extra></extra>"
                ),
            )
        )
    else:
        figure.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame["weather_value"],
                mode="lines",
                name=weather_label,
                line={
                    "color": WEATHER_COLOR,
                    "width": 1.8,
                },
                yaxis="y2",
                hovertemplate=(
                    "<b>%{x|%b %d, %Y}</b><br>"
                    f"{weather_label}: %{{y:,.2f}}"
                    "<extra></extra>"
                ),
            )
        )

    if (
        not frame.empty
        and frame["date"].min()
        <= CP_START_DATE
        <= frame["date"].max()
    ):
        figure.add_vline(
            x=CP_START_DATE,
            line_width=2,
            line_dash="dash",
            line_color=BRAND_COLORS["gray"],
            annotation_text="Congestion pricing",
            annotation_position="top left",
        )

    weather_axis = {
        "title": {
            "text": weather_label,
            "font": {
                "color": WEATHER_COLOR,
            },
        },
        "tickfont": {
            "color": WEATHER_COLOR,
        },
        "overlaying": "y",
        "side": "right",
        "showgrid": False,
        "zeroline": weather_metric == "pressure_3hr_change",
        "zerolinecolor": "rgba(102,116,122,0.45)",
    }

    if weather_metric == "precipitation":
        weather_axis["rangemode"] = "tozero"

    figure.update_layout(
        height=470,
        margin={
            "l": 20,
            "r": 35,
            "t": 25,
            "b": 20,
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="overlay",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        xaxis={
            "title": None,
            "showgrid": False,
        },
        yaxis={
            "title": {
                "text": mobility_label,
                "font": {
                    "color": MOBILITY_COLOR,
                },
            },
            "tickfont": {
                "color": MOBILITY_COLOR,
            },
            "showgrid": True,
            "gridcolor": "rgba(102,116,122,0.15)",
            "zeroline": False,
        },
        yaxis2=weather_axis,
        hovermode="x unified",
    )

    return figure


def _support_label(observation_count: int) -> str:
    if observation_count >= 365:
        return "Strong support"
    if observation_count >= 120:
        return "Good support"
    if observation_count >= 45:
        return "Limited support"
    return "Very limited support"


def _dynamic_interpretation(
    statistics: pd.DataFrame,
    *,
    weather_label: str,
    mobility_label: str,
    temporal_label: str,
    geography_label: str,
) -> tuple[str, str]:
    all_row = _extract_period_row(
        statistics,
        "all",
    )
    pre_row = _extract_period_row(
        statistics,
        "pre_cp",
    )
    post_row = _extract_period_row(
        statistics,
        "post_cp",
    )

    if all_row is None:
        return (
            "Relationship unavailable",
            "There are not enough matched observations to summarize this view.",
        )

    all_spearman = all_row["spearman_correlation"]
    all_pearson = all_row["pearson_correlation"]
    observation_count = int(all_row["observation_count"])

    strength = _relationship_strength(all_spearman)
    direction = _relationship_direction(all_spearman)
    support = _support_label(observation_count)

    if pd.isna(all_spearman):
        return (
            "Relationship unavailable",
            "There are not enough matched observations to estimate this relationship.",
        )

    if direction == "near-zero":
        headline = "Little consistent relationship"
    else:
        headline = f"{strength} {direction} relationship"

    agreement_note = ""
    if not pd.isna(all_pearson):
        if (
            np.sign(all_spearman) == np.sign(all_pearson)
            and abs(all_spearman - all_pearson) <= 0.10
        ):
            agreement_note = (
                " Pearson and Spearman tell a similar story."
            )
        elif np.sign(all_spearman) != np.sign(all_pearson):
            agreement_note = (
                " Pearson and Spearman disagree on direction, suggesting "
                "the pattern is not consistently linear."
            )
        else:
            agreement_note = (
                " Pearson and Spearman differ somewhat, suggesting the "
                "relationship may not be purely linear."
            )

    period_note = ""
    if pre_row is not None and post_row is not None:
        pre_value = pre_row["spearman_correlation"]
        post_value = post_row["spearman_correlation"]

        if not pd.isna(pre_value) and not pd.isna(post_value):
            shift = post_value - pre_value

            if np.sign(pre_value) != np.sign(post_value) and (
                abs(pre_value) >= 0.05 or abs(post_value) >= 0.05
            ):
                period_note = (
                    " The relationship changed direction after congestion "
                    f"pricing ({_format_correlation(pre_value)} before versus "
                    f"{_format_correlation(post_value)} after), so it should "
                    "be treated as unstable rather than persistent."
                )
            elif abs(shift) < 0.05:
                period_note = (
                    " The relationship was broadly similar before and after "
                    "congestion pricing."
                )
            elif abs(post_value) > abs(pre_value):
                period_note = (
                    " The relationship was stronger after congestion pricing "
                    f"({_format_correlation(pre_value)} before versus "
                    f"{_format_correlation(post_value)} after)."
                )
            else:
                period_note = (
                    " The relationship was weaker after congestion pricing "
                    f"({_format_correlation(pre_value)} before versus "
                    f"{_format_correlation(post_value)} after)."
                )

    body = (
        f"Across {geography_label.lower()} {temporal_label.lower()} observations, "
        f"{weather_label.lower()} and {mobility_label.lower()} had a "
        f"Spearman correlation of {_format_correlation(all_spearman)} across "
        f"{observation_count:,} matched observations. {support}."
        f"{agreement_note}{period_note}"
    )

    return headline, body


def _initialize_state(
    minimum_date: pd.Timestamp,
    maximum_date: pd.Timestamp,
) -> None:
    defaults = {
        "raw09_saved_view": SAVED_VIEWS[0].label,
        "raw09_geography": "Citywide",
        "raw09_zone_id": None,
        "raw09_mobility_metric": SAVED_VIEWS[0].mobility_metric,
        "raw09_weather_metric": SAVED_VIEWS[0].weather_metric,
        "raw09_temporal_bucket": SAVED_VIEWS[0].temporal_bucket,
        "raw09_start_date": minimum_date.date(),
        "raw09_end_date": maximum_date.date(),
        "raw09_show_periods": True,
        "raw09_applied_saved_view": SAVED_VIEWS[0].label,
    }

    for key, value in defaults.items():
        st.session_state.setdefault(
            key,
            value,
        )


def _apply_saved_view(
    saved_view: SavedView,
    minimum_date: pd.Timestamp,
    maximum_date: pd.Timestamp,
) -> None:
    st.session_state["raw09_geography"] = saved_view.geography
    st.session_state["raw09_zone_id"] = None
    st.session_state["raw09_mobility_metric"] = saved_view.mobility_metric
    st.session_state["raw09_weather_metric"] = saved_view.weather_metric
    st.session_state["raw09_temporal_bucket"] = saved_view.temporal_bucket
    st.session_state["raw09_start_date"] = minimum_date.date()
    st.session_state["raw09_end_date"] = maximum_date.date()
    st.session_state["raw09_show_periods"] = True
    st.session_state["raw09_applied_saved_view"] = saved_view.label


def _mark_custom() -> None:
    st.session_state["raw09_applied_saved_view"] = "Custom"


def _render_relationship_cards(
    statistics: pd.DataFrame,
) -> None:
    all_row = _extract_period_row(statistics, "all")
    pre_row = _extract_period_row(statistics, "pre_cp")
    post_row = _extract_period_row(statistics, "post_cp")

    all_value = (
        all_row["spearman_correlation"]
        if all_row is not None
        else np.nan
    )
    pearson_value = (
        all_row["pearson_correlation"]
        if all_row is not None
        else np.nan
    )
    observations = (
        int(all_row["observation_count"])
        if all_row is not None
        else 0
    )

    pre_value = (
        pre_row["spearman_correlation"]
        if pre_row is not None
        else np.nan
    )
    post_value = (
        post_row["spearman_correlation"]
        if post_row is not None
        else np.nan
    )
    shift = (
        post_value - pre_value
        if not pd.isna(pre_value)
        and not pd.isna(post_value)
        else np.nan
    )

    card_1, card_2, card_3, card_4 = st.columns(4)

    card_1.metric(
        "Spearman correlation",
        _format_correlation(all_value),
        help=(
            "Measures whether the two variables generally rise or fall "
            "together, including non-linear monotonic relationships."
        ),
    )

    card_2.metric(
        "Pearson correlation",
        _format_correlation(pearson_value),
        help=(
            "Measures the strength of the linear relationship."
        ),
    )

    card_3.metric(
        "Matched observations",
        f"{observations:,}",
    )

    card_4.metric(
        "Post-minus-pre shift",
        _format_shift(shift),
        help=(
            "Post-congestion-pricing Spearman correlation minus the "
            "pre-congestion-pricing correlation."
        ),
    )


minimum_date, maximum_date = load_relationship_date_bounds()
zone_lookup = load_zone_lookup()

_initialize_state(
    minimum_date,
    maximum_date,
)

st.caption("WEATHER RELATIONSHIPS")

st.title("How did weather relate to mobility?")

st.write(
    "Explore how weather conditions moved with transportation demand and "
    "speed across the city and within individual Taxi Zones."
)

st.subheader("Bus speeds were lower on warmer weekday evenings")

hero_pair = build_weather_relationship_pair_data(
    mobility_metric="avg_bus_speed",
    weather_metric="temperature",
    temporal_bucket="weekday_evening",
    taxi_zone_id=None,
    start_date=minimum_date,
    end_date=maximum_date,
)

hero_statistics = calculate_relationship_statistics(
    hero_pair,
    minimum_observations=30,
)

hero_all = _extract_period_row(
    hero_statistics,
    "all",
)
hero_pre = _extract_period_row(
    hero_statistics,
    "pre_cp",
)
hero_post = _extract_period_row(
    hero_statistics,
    "post_cp",
)

hero_spearman = (
    hero_all["spearman_correlation"]
    if hero_all is not None
    else np.nan
)
hero_pre_value = (
    hero_pre["spearman_correlation"]
    if hero_pre is not None
    else np.nan
)
hero_post_value = (
    hero_post["spearman_correlation"]
    if hero_post is not None
    else np.nan
)

st.write(
    "Warmer weekday evenings were associated with lower citywide bus speeds. "
    "The relationship remained negative before and after congestion pricing."
)

hero_card_1, hero_card_2, hero_card_3 = st.columns(3)

hero_card_1.metric(
    "Full-period Spearman",
    _format_correlation(hero_spearman),
)

hero_card_2.metric(
    "Before congestion pricing",
    _format_correlation(hero_pre_value),
)

hero_card_3.metric(
    "After congestion pricing",
    _format_correlation(hero_post_value),
)

hero_figure = _build_scatter_figure(
    hero_pair,
    weather_label="Temperature",
    mobility_label="Bus Average Speed",
    show_periods=True,
)

st.plotly_chart(
    hero_figure,
    width="stretch",
    key="raw09_hero_scatter",
    config={
        "displayModeBar": False,
    },
)

st.info(
    "This view shows association, not causation. Seasonal travel patterns, "
    "traffic, roadway conditions, and service changes may affect both measures."
)

st.divider()

st.header("Explore weather and mobility")

saved_view_labels = [
    saved_view.label
    for saved_view in SAVED_VIEWS
]


def _apply_selected_saved_view() -> None:
    selected_label = st.session_state[
        "raw09_saved_view"
    ]

    selected_view = next(
        view
        for view in SAVED_VIEWS
        if view.label == selected_label
    )

    _apply_saved_view(
        selected_view,
        minimum_date,
        maximum_date,
    )


selected_saved_view = st.selectbox(
    "Saved view",
    options=saved_view_labels,
    key="raw09_saved_view",
    on_change=_apply_selected_saved_view,
)

selected_saved_view_config = next(
    saved_view
    for saved_view in SAVED_VIEWS
    if saved_view.label == selected_saved_view
)

st.caption(
    selected_saved_view_config.description
)

control_row_1 = st.columns(
    [1.2, 1.8, 1.8]
)

with control_row_1[0]:
    geography = st.selectbox(
        "Geography",
        options=["Citywide", "Taxi Zone"],
        key="raw09_geography",
        on_change=_mark_custom,
    )

with control_row_1[1]:
    mobility_metric = st.selectbox(
        "Mobility metric",
        options=list(MOBILITY_METRICS),
        format_func=lambda value: MOBILITY_LABELS[value],
        key="raw09_mobility_metric",
        on_change=_mark_custom,
    )

with control_row_1[2]:
    weather_metric = st.selectbox(
        "Weather metric",
        options=list(WEATHER_METRICS),
        format_func=lambda value: WEATHER_LABELS[value],
        key="raw09_weather_metric",
        on_change=_mark_custom,
    )

selected_zone_id: int | None = None
selected_zone_label = "Citywide"

if geography == "Taxi Zone":
    zone_options = zone_lookup["taxi_zone_id"].astype(int).tolist()

    zone_label_map = {
        int(row.taxi_zone_id): (
            f"{row.zone} · {row.borough}"
        )
        for row in zone_lookup.itertuples()
    }

    if (
        st.session_state.get("raw09_zone_id")
        not in zone_options
    ):
        st.session_state["raw09_zone_id"] = zone_options[0]

    selected_zone_id = st.selectbox(
        "Taxi Zone",
        options=zone_options,
        format_func=lambda value: zone_label_map[value],
        key="raw09_zone_id",
        on_change=_mark_custom,
    )

    selected_zone_label = zone_label_map[
        selected_zone_id
    ]

control_row_2 = st.columns(
    [1.5, 1.2, 1.2, 1]
)

with control_row_2[0]:
    temporal_bucket = st.selectbox(
        "Time of week",
        options=list(TEMPORAL_BUCKETS),
        format_func=lambda value: TEMPORAL_BUCKET_LABELS[value],
        key="raw09_temporal_bucket",
        on_change=_mark_custom,
    )

with control_row_2[1]:
    start_date = st.date_input(
        "Start date",
        min_value=minimum_date.date(),
        max_value=maximum_date.date(),
        key="raw09_start_date",
        on_change=_mark_custom,
    )

with control_row_2[2]:
    end_date = st.date_input(
        "End date",
        min_value=minimum_date.date(),
        max_value=maximum_date.date(),
        key="raw09_end_date",
        on_change=_mark_custom,
    )

with control_row_2[3]:
    show_periods = st.toggle(
        "Compare pre/post",
        key="raw09_show_periods",
        on_change=_mark_custom,
    )

if start_date > end_date:
    st.error(
        "The start date must be on or before the end date."
    )
    st.stop()

pair_data = build_weather_relationship_pair_data(
    mobility_metric=mobility_metric,
    weather_metric=weather_metric,
    temporal_bucket=temporal_bucket,
    taxi_zone_id=selected_zone_id,
    start_date=start_date,
    end_date=end_date,
)

minimum_support = (
    20
    if temporal_bucket
    != ALL_TEMPORAL_BUCKETS_LABEL
    else 30
)

statistics = calculate_relationship_statistics(
    pair_data,
    minimum_observations=minimum_support,
)

all_row = _extract_period_row(
    statistics,
    "all",
)

if (
    pair_data.empty
    or all_row is None
    or not bool(all_row["supported"])
):
    st.warning(
        "There are not enough matched observations to estimate this "
        "relationship reliably. Broaden the date range, choose Citywide, "
        "or select another time-of-week view."
    )
    st.stop()

observation_count = int(
    all_row["observation_count"]
)

if observation_count < 45:
    st.warning(
        f"Only {observation_count:,} matched observations are available. "
        "Treat this relationship as exploratory."
    )
elif observation_count < 120:
    st.caption(
        f"Support note: {observation_count:,} matched observations are "
        "available, so this view should be interpreted with some caution."
    )

st.caption(
    f"Current view: {st.session_state['raw09_applied_saved_view']}"
)

_render_relationship_cards(statistics)

relationship_tab, time_tab = st.tabs(
    [
        "Relationship",
        "Over time",
    ]
)

weather_label = WEATHER_LABELS[
    weather_metric
]
mobility_label = MOBILITY_LABELS[
    mobility_metric
]
temporal_label = TEMPORAL_BUCKET_LABELS[
    temporal_bucket
]

with relationship_tab:
    relationship_figure = _build_scatter_figure(
        pair_data,
        weather_label=weather_label,
        mobility_label=mobility_label,
        show_periods=show_periods,
    )

    st.plotly_chart(
        relationship_figure,
        width="stretch",
        key="raw09_explorer_scatter",
        config={
            "displayModeBar": False,
        },
    )

    st.subheader("What stands out")

    insight_headline, insight_body = _dynamic_interpretation(
        statistics,
        weather_label=weather_label,
        mobility_label=mobility_label,
        temporal_label=temporal_label,
        geography_label=selected_zone_label,
    )

    st.info(
        f"**{insight_headline}**  \n"
        f"{insight_body}"
    )

with time_tab:
    st.write(
        "Mobility uses the left axis and weather the right. Compare timing "
        "and direction, not line or bar height."
    )

    time_figure = _build_time_figure(
        pair_data,
        weather_metric=weather_metric,
        weather_label=weather_label,
        mobility_label=mobility_label,
    )

    st.plotly_chart(
        time_figure,
        width="stretch",
        key="raw09_explorer_time_series",
        config={
            "displayModeBar": False,
        },
    )

with st.expander("View matched observations"):
    detail_table = pair_data[
        [
            "date",
            "pre_post_cp",
            "weather_value",
            "mobility_value",
        ]
    ].copy()

    detail_table = detail_table.rename(
        columns={
            "date": "Date",
            "pre_post_cp": "Period",
            "weather_value": weather_label,
            "mobility_value": mobility_label,
        }
    )

    detail_table["Period"] = (
        detail_table["Period"]
        .map(
            {
                "pre_cp": "Before congestion pricing",
                "post_cp": "After congestion pricing",
            }
        )
    )

    st.dataframe(
        detail_table.sort_values(
            "Date",
            ascending=False,
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "Date": st.column_config.DateColumn(
                format="MMM D, YYYY"
            ),
            weather_label: st.column_config.NumberColumn(
                format="%.2f"
            ),
            mobility_label: st.column_config.NumberColumn(
                format="%.2f"
            ),
        },
    )

st.caption(
    "Correlations describe association, not causation. Seasonal, roadway, "
    "service, and travel-demand conditions may affect both measures."
)
