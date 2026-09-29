from __future__ import annotations

import calendar

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.data_access.mobility_day_types import (
    get_mobility_day_type_summary,
    get_representative_mobility_days,
    load_mobility_day_type_date_features,
    load_mobility_day_type_profiles,
    load_mobility_day_type_qa,
    load_mobility_day_type_runs,
    load_mobility_day_types,
)
from app.utils.project_branding import (
    BRAND_COLORS,
    BRAND_DIVERGING_SEQUENCE,
    apply_branding,
    exploration_section,
    inject_app_css,
    render_chart_insight,
)


PAGE_CAPTION = "MOBILITY DAY TYPE CALENDAR"
PAGE_TITLE = "What kind of mobility day was NYC experiencing?"

DAY_TYPE_ORDER = [
    "faster_lighter",
    "holiday_like",
    "busier_slower",
]

DAY_TYPE_LABELS = {
    "faster_lighter": "Faster, Lighter",
    "holiday_like": "Holiday-Like",
    "busier_slower": "Busier, Slower",
}

DAY_TYPE_COLORS = {
    "faster_lighter": BRAND_COLORS["seafoam"],
    "holiday_like": BRAND_COLORS["terracotta"],
    "busier_slower": BRAND_COLORS["dark_teal"],
}

DAY_TYPE_DESCRIPTIONS = {
    "faster_lighter": (
        "Fewer trips and riders than usual for the same weekday, while "
        "taxis, FHVs, and buses generally move faster."
    ),
    "holiday_like": (
        "An unusually quiet daytime pattern with much faster AM and midday "
        "roads. Many of the clearest examples occur on or around holidays."
    ),
    "busier_slower": (
        "More trips and riders than usual for the same weekday, paired with "
        "slower movement on the streets."
    ),
}

METRIC_ORDER = [
    "taxi_trip_count",
    "taxi_avg_trip_speed",
    "fhvhv_trip_count",
    "fhvhv_avg_trip_speed",
    "subway_ridership",
    "avg_bus_speed",
]

METRIC_SHORT_LABELS = {
    "taxi_trip_count": "Taxi trips",
    "taxi_avg_trip_speed": "Taxi speed",
    "fhvhv_trip_count": "FHV trips",
    "fhvhv_avg_trip_speed": "FHV speed",
    "subway_ridership": "Subway riders",
    "avg_bus_speed": "Bus speed",
}

DAYPART_ORDER = [
    "overnight",
    "am_peak",
    "midday",
    "pm_peak",
    "evening",
]

DAYPART_LABELS = {
    "overnight": "Overnight",
    "am_peak": "AM peak",
    "midday": "Midday",
    "pm_peak": "PM peak",
    "evening": "Evening",
}

WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def _calendar_coordinates(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Create stable year × week × weekday positions for calendar cells."""
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result["calendar_year"] = result["date"].dt.year
    result["weekday_number"] = result["date"].dt.weekday

    # WHY: anchor each year's week index to the Monday of the week containing
    # January 1. This preserves contiguous seven-day columns across year edges.
    year_start = pd.to_datetime(
        result["calendar_year"].astype(str) + "-01-01"
    )
    first_monday = (
        year_start
        - pd.to_timedelta(
            year_start.dt.weekday,
            unit="D",
        )
    )
    date_monday = (
        result["date"]
        - pd.to_timedelta(
            result["weekday_number"],
            unit="D",
        )
    )
    result["week_index"] = (
        (date_monday - first_monday).dt.days // 7
    ).astype(int) + 1

    result["weekday_label"] = result[
        "weekday_number"
    ].map(
        dict(enumerate(WEEKDAY_ORDER))
    )
    result["hover_date"] = result["date"].dt.strftime(
        "%A, %B %-d, %Y"
    )

    # Windows does not support %-d in strftime.
    if result["hover_date"].str.contains("%-d", regex=False).any():
        result["hover_date"] = result["date"].map(
            lambda value: (
                f"{value.strftime('%A, %B')} "
                f"{value.day}, {value.year}"
            )
        )

    return result


def _calendar_figure(
    day_types: pd.DataFrame,
) -> go.Figure:
    """Render the fixed editorial year × week × weekday hero calendar."""
    frame = _calendar_coordinates(day_types)
    years = sorted(frame["calendar_year"].unique())

    fig = make_subplots(
        rows=len(years),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.055,
        subplot_titles=[
            str(year)
            for year in years
        ],
    )

    for row, year in enumerate(years, start=1):
        year_frame = frame.loc[
            frame["calendar_year"].eq(year)
        ].copy()

        for day_type_id in DAY_TYPE_ORDER:
            subset = year_frame.loc[
                year_frame["day_type_id"].eq(day_type_id)
            ]

            if subset.empty:
                continue

            fig.add_trace(
                go.Scatter(
                    x=subset["week_index"],
                    y=subset["weekday_number"],
                    mode="markers",
                    marker={
                        "symbol": "square",
                        "size": 12,
                        "color": DAY_TYPE_COLORS[
                            day_type_id
                        ],
                        "line": {
                            "color": "white",
                            "width": 0.7,
                        },
                    },
                    name=DAY_TYPE_LABELS[
                        day_type_id
                    ],
                    legendgroup=day_type_id,
                    showlegend=row == 1,
                    customdata=np.column_stack(
                        [
                            subset["date"].dt.strftime(
                                "%Y-%m-%d"
                            ),
                            subset["day_type_name"],
                            subset["day_name"],
                            subset["policy_period"],
                        ]
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "%{customdata[2]}<br>"
                        "%{customdata[1]}<br>"
                        "%{customdata[3]}"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=1,
            )

        fig.update_yaxes(
            row=row,
            col=1,
            tickmode="array",
            tickvals=list(range(7)),
            ticktext=[
                "Mon",
                "Tue",
                "Wed",
                "Thu",
                "Fri",
                "Sat",
                "Sun",
            ],
            autorange="reversed",
            range=[6.55, -0.55],
            title=None,
            gridcolor="rgba(0,0,0,0)",
            zeroline=False,
        )

        month_ticks = []
        month_labels = []

        for month in range(1, 13):
            first_day = pd.Timestamp(
                year=year,
                month=month,
                day=1,
            )
            first_monday = (
                pd.Timestamp(year=year, month=1, day=1)
                - pd.Timedelta(
                    days=pd.Timestamp(
                        year=year,
                        month=1,
                        day=1,
                    ).weekday()
                )
            )
            month_monday = (
                first_day
                - pd.Timedelta(
                    days=first_day.weekday()
                )
            )
            week = (
                (month_monday - first_monday).days // 7
            ) + 1
            month_ticks.append(week)
            month_labels.append(
                calendar.month_abbr[month]
            )

        fig.update_xaxes(
            row=row,
            col=1,
            tickmode="array",
            tickvals=month_ticks,
            ticktext=month_labels,
            range=[0, 54],
            title=None,
            showgrid=False,
            zeroline=False,
        )

    fig.update_layout(
        height=215 * len(years) + 85,
        hovermode="closest",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.04,
            "xanchor": "left",
            "x": 0,
        },
        margin={
            "l": 55,
            "r": 20,
            "t": 85,
            "b": 25,
        },
    )

    apply_branding(fig)
    fig.update_layout(
        title={"text": ""},
        plot_bgcolor="white",
    )

    return fig


def _profile_heatmap(
    profiles: pd.DataFrame,
) -> go.Figure:
    """Show the three recurring state signatures on a shared visual scale."""
    fig = make_subplots(
        rows=1,
        cols=3,
        horizontal_spacing=0.055,
        subplot_titles=[
            DAY_TYPE_LABELS[value]
            for value in DAY_TYPE_ORDER
        ],
    )

    max_abs = max(
        1.0,
        float(
            profiles["relative_level"]
            .abs()
            .quantile(0.98)
        ),
    )

    for column, day_type_id in enumerate(
        DAY_TYPE_ORDER,
        start=1,
    ):
        subset = profiles.loc[
            profiles["day_type_id"].eq(
                day_type_id
            )
        ].copy()

        matrix = (
            subset.pivot(
                index="metric",
                columns="daypart",
                values="relative_level",
            )
            .reindex(
                index=METRIC_ORDER,
                columns=DAYPART_ORDER,
            )
        )

        hover = np.empty(
            matrix.shape,
            dtype=object,
        )

        for row_index, metric in enumerate(
            matrix.index
        ):
            for col_index, daypart in enumerate(
                matrix.columns
            ):
                value = matrix.loc[
                    metric,
                    daypart,
                ]
                hover[
                    row_index,
                    col_index,
                ] = (
                    f"{METRIC_SHORT_LABELS[metric]} · "
                    f"{DAYPART_LABELS[daypart]}<br>"
                    f"{value:+.2f} same-weekday IQRs"
                )

        fig.add_trace(
            go.Heatmap(
                z=matrix.to_numpy(),
                x=[
                    DAYPART_LABELS[value]
                    for value in matrix.columns
                ],
                y=[
                    METRIC_SHORT_LABELS[value]
                    for value in matrix.index
                ],
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                zmin=-max_abs,
                zmax=max_abs,
                zmid=0,
                colorscale=[
                    [0.00, BRAND_DIVERGING_SEQUENCE[4]],
                    [0.25, BRAND_DIVERGING_SEQUENCE[3]],
                    [0.50, BRAND_DIVERGING_SEQUENCE[2]],
                    [0.75, BRAND_DIVERGING_SEQUENCE[1]],
                    [1.00, BRAND_DIVERGING_SEQUENCE[0]],
                ],
                colorbar={
                    "title": {
                        "text": "Relative to<br>same weekday"
                    },
                    "tickformat": "+.1f",
                }
                if column == 3
                else None,
                showscale=column == 3,
            ),
            row=1,
            col=column,
        )

        fig.update_xaxes(
            tickangle=-35,
            row=1,
            col=column,
        )

        if column > 1:
            fig.update_yaxes(
                showticklabels=False,
                row=1,
                col=column,
            )

    fig.update_layout(
        height=440,
        margin={
            "l": 95,
            "r": 80,
            "t": 75,
            "b": 85,
        },
    )
    apply_branding(fig)
    fig.update_layout(title={"text": ""})

    return fig


def _feature_phrase(
    row: pd.Series,
    *,
    include_magnitude: bool,
) -> str:
    """Translate one relative feature into compact reader-facing evidence."""
    direction = "above" if float(row["relative_level"]) > 0 else "below"
    metric = str(row["metric_label"])
    daypart = str(row["daypart_label"])

    if include_magnitude:
        return (
            f"{metric} · {daypart} was "
            f"{abs(float(row['relative_level'])):.1f} IQRs {direction} "
            "its same-weekday norm"
        )

    return f"{metric} · {daypart} {direction} normal"


def _diverse_top_features(
    frame: pd.DataFrame,
    *,
    n: int = 3,
) -> pd.DataFrame:
    """Choose strong signals without letting one metric monopolize the story."""
    ranked = (
        frame.assign(
            _strength=frame["relative_level"].abs()
        )
        .sort_values(
            "_strength",
            ascending=False,
        )
        .copy()
    )

    selected_indices = []
    used_metrics = set()

    # WHY: prefer different mobility measures first. If fewer than n distinct
    # metrics are available, fill the remaining slots by raw strength.
    for index, row in ranked.iterrows():
        metric = str(row["metric"])
        if metric in used_metrics:
            continue

        selected_indices.append(index)
        used_metrics.add(metric)

        if len(selected_indices) == n:
            break

    if len(selected_indices) < n:
        for index in ranked.index:
            if index in selected_indices:
                continue

            selected_indices.append(index)

            if len(selected_indices) == n:
                break

    return (
        ranked.loc[selected_indices]
        .drop(columns="_strength")
        .copy()
    )


def _archetype_signature(
    profiles: pd.DataFrame,
    day_type_id: str,
) -> str:
    """Summarize each frozen archetype with evidence from its profile table."""
    subset = profiles.loc[
        profiles["day_type_id"].eq(day_type_id)
    ].copy()

    strongest = _diverse_top_features(
        subset,
        n=2,
    )

    phrases = [
        _feature_phrase(
            row,
            include_magnitude=False,
        )
        for _, row in strongest.iterrows()
    ]

    if not phrases:
        return "Profile evidence unavailable"

    return " + ".join(phrases)


def _selected_date_heatmap(
    date_features: pd.DataFrame,
    selected_date: pd.Timestamp,
) -> go.Figure:
    """Explain one date across the same six metrics × five dayparts."""
    subset = date_features.loc[
        date_features["date"].eq(
            selected_date
        )
    ].copy()

    matrix = (
        subset.pivot(
            index="metric",
            columns="daypart",
            values="relative_level",
        )
        .reindex(
            index=METRIC_ORDER,
            columns=DAYPART_ORDER,
        )
    )

    max_abs = max(
        1.0,
        float(
            np.nanmax(
                np.abs(
                    matrix.to_numpy(
                        dtype=float
                    )
                )
            )
        ),
    )

    text = np.empty(
        matrix.shape,
        dtype=object,
    )

    for row_index, metric in enumerate(
        matrix.index
    ):
        for col_index, daypart in enumerate(
            matrix.columns
        ):
            value = matrix.loc[
                metric,
                daypart,
            ]
            text[
                row_index,
                col_index,
            ] = (
                f"{METRIC_SHORT_LABELS[metric]} · "
                f"{DAYPART_LABELS[daypart]}<br>"
                f"{value:+.2f} same-weekday IQRs"
            )

    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=[
                DAYPART_LABELS[value]
                for value in matrix.columns
            ],
            y=[
                METRIC_SHORT_LABELS[value]
                for value in matrix.index
            ],
            text=text,
            hovertemplate="%{text}<extra></extra>",
            zmin=-max_abs,
            zmax=max_abs,
            zmid=0,
            colorscale=[
                [0.00, BRAND_DIVERGING_SEQUENCE[4]],
                [0.25, BRAND_DIVERGING_SEQUENCE[3]],
                [0.50, BRAND_DIVERGING_SEQUENCE[2]],
                [0.75, BRAND_DIVERGING_SEQUENCE[1]],
                [1.00, BRAND_DIVERGING_SEQUENCE[0]],
            ],
            colorbar={
                "title": {
                    "text": "Same-weekday<br>IQRs"
                },
                "tickformat": "+.1f",
            },
        )
    )

    fig.update_layout(
        height=410,
        xaxis_title=None,
        yaxis_title=None,
        margin={
            "l": 110,
            "r": 80,
            "t": 25,
            "b": 55,
        },
    )
    apply_branding(fig)
    fig.update_layout(title={"text": ""})

    return fig


def _run_timeline(
    runs: pd.DataFrame,
    *,
    selected_run: pd.Series,
    selected_date: pd.Timestamp,
) -> go.Figure:
    """Show nearby state persistence while keeping the selected run prominent."""
    window_start = (
        selected_run["start_date"]
        - pd.Timedelta(days=35)
    )
    window_end = (
        selected_run["end_date"]
        + pd.Timedelta(days=35)
    )

    nearby = runs.loc[
        runs["end_date"].ge(window_start)
        & runs["start_date"].le(window_end)
    ].copy()

    fig = go.Figure()

    for day_type_id in DAY_TYPE_ORDER:
        subset = nearby.loc[
            nearby["day_type_id"].eq(
                day_type_id
            )
        ]

        if subset.empty:
            continue

        for _, row in subset.iterrows():
            is_selected = (
                int(row["run_number"])
                == int(
                    selected_run["run_number"]
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[
                        row["start_date"],
                        row["end_date"]
                        + pd.Timedelta(days=1),
                    ],
                    y=[
                        DAY_TYPE_LABELS[
                            day_type_id
                        ],
                        DAY_TYPE_LABELS[
                            day_type_id
                        ],
                    ],
                    mode="lines",
                    line={
                        "color": DAY_TYPE_COLORS[
                            day_type_id
                        ],
                        "width": 16
                        if is_selected
                        else 8,
                    },
                    opacity=1.0
                    if is_selected
                    else 0.55,
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{DAY_TYPE_LABELS[day_type_id]}</b><br>"
                        f"{row['start_date']:%Y-%m-%d} → "
                        f"{row['end_date']:%Y-%m-%d}<br>"
                        f"{int(row['duration_days'])} day(s)"
                        "<extra></extra>"
                    ),
                )
            )

    fig.add_vline(
        x=selected_date,
        line_width=1.5,
        line_dash="dot",
        line_color=BRAND_COLORS["dark_teal"],
        annotation_text="Selected date",
        annotation_position="top",
    )

    fig.update_layout(
        height=280,
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
        margin={
            "l": 120,
            "r": 25,
            "t": 25,
            "b": 45,
        },
    )
    apply_branding(fig)
    fig.update_layout(title={"text": ""})

    return fig


def _find_run_for_date(
    runs: pd.DataFrame,
    selected_date: pd.Timestamp,
) -> pd.Series:
    """Return the one contiguous state run containing the selected date."""
    match = runs.loc[
        runs["start_date"].le(selected_date)
        & runs["end_date"].ge(selected_date)
    ]

    if len(match) != 1:
        raise ValueError(
            "Expected exactly one mobility-state run "
            f"for {selected_date:%Y-%m-%d}; found {len(match)}."
        )

    return match.iloc[0]


def _format_date(
    value: pd.Timestamp,
) -> str:
    """Format a date without platform-specific strftime directives."""
    return (
        f"{value.strftime('%A, %B')} "
        f"{value.day}, {value.year}"
    )


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Mobility Day Type Calendar",
    page_icon="🗓️",
    layout="wide",
)

inject_app_css()

day_types = load_mobility_day_types()
profiles = load_mobility_day_type_profiles()
date_features = load_mobility_day_type_date_features()
runs = load_mobility_day_type_runs()
qa = load_mobility_day_type_qa()
summary = get_mobility_day_type_summary()
representatives = get_representative_mobility_days(
    per_type=5
)

st.caption(PAGE_CAPTION)
st.title(PAGE_TITLE)
st.markdown(
    """
    <div class="app-subtitle">
    NYC does not move the same way every day. After accounting for the normal
    differences between Mondays, Tuesdays, weekends, and so on, three recurring
    mobility patterns stand out. The calendar below shows when each pattern
    appeared from 2023 through March 2026.
    </div>
    """,
    unsafe_allow_html=True,
)

st.plotly_chart(
    _calendar_figure(day_types),
    width="stretch",
    config={
        "displayModeBar": False,
    },
    key="raw26_plotly_01",
)

st.caption(
    "Each square is one modeled date. The 2026 row ends in March because the "
    "study window ends March 31, 2026."
)

state_counts = (
    summary.set_index("day_type_id")["days"]
    .to_dict()
)

render_chart_insight(
    "The calendar repeatedly moves between two broad states — "
    "**Faster, Lighter** and **Busier, Slower** — while the much rarer "
    "**Holiday-Like** pattern punctuates that rhythm. Because every feature is "
    "compared with the norm for the same weekday, these colors are not simply "
    "a weekday-versus-weekend calendar."
)

st.subheader("Three recurring kinds of mobility day")

card_columns = st.columns(3)

for column, day_type_id in zip(
    card_columns,
    DAY_TYPE_ORDER,
):
    days = int(
        state_counts.get(
            day_type_id,
            0,
        )
    )
    share = (
        days / len(day_types)
        if len(day_types)
        else 0.0
    )

    examples = representatives.loc[
        representatives["day_type_id"].eq(
            day_type_id
        ),
        "date",
    ].head(3)

    example_text = ", ".join(
        value.strftime("%b %d, %Y")
        for value in examples
    )

    signature_text = _archetype_signature(
        profiles,
        day_type_id,
    )

    with column:
        st.markdown(
            f"""
            <div class="metric-card" style="border-top: 5px solid
            {DAY_TYPE_COLORS[day_type_id]}; min-height: 285px;">
                <div style="font-size: 1.15rem; font-weight: 750;
                color: {BRAND_COLORS["dark_teal"]};">
                    {DAY_TYPE_LABELS[day_type_id]}
                </div>
                <div style="font-size: 1.65rem; font-weight: 750;
                margin: 0.25rem 0;">
                    {days:,} days
                </div>
                <div style="color: {BRAND_COLORS["dark_teal"]}; margin-bottom: 0.65rem;">
                    {share:.1%} of modeled dates
                </div>
                <div style="line-height: 1.45;">
                    {DAY_TYPE_DESCRIPTIONS[day_type_id]}
                </div>
                <div style="margin-top: 0.75rem; font-size: 0.88rem;
                line-height: 1.4;">
                    <strong>Strongest signature:</strong> {signature_text}
                </div>
                <div style="margin-top: 0.75rem; font-size: 0.88rem;
                color: {BRAND_COLORS["dark_teal"]}; opacity: 0.72;">
                    Typical examples: {example_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("#### How the three day types differ across the day")

st.plotly_chart(
    _profile_heatmap(profiles),
    width="stretch",
    config={
        "displayModeBar": False,
    },
    key="raw26_plotly_02",
)

st.caption(
    "Each cell shows the median day-type profile in same-weekday IQR units. "
    "Positive values mean above the usual level for that weekday; negative "
    "values mean below it. For speed measures, positive means faster."
)

with st.expander(
    "How the three day types were found"
):
    pca_retained = float(
        qa["pca_variance_retained"].iloc[0]
    )

    st.markdown(
        f"""
        Each date is described with **six mobility measures across five parts
        of the day**, giving 30 measurements in all. Trip and ridership counts
        are log-transformed first so very large days do not dominate the
        comparison.

        Next, each Monday is compared with normal Mondays, each Tuesday with
        normal Tuesdays, and so on. For every measurement, the same-weekday
        median becomes the reference point and the interquartile range (IQR)
        provides the scale. This removes the ordinary weekly rhythm before the
        day types are formed.

        **PCA** then compresses the 30 overlapping measurements to **10 combined
        dimensions while preserving {pca_retained:.1%} of their variation**.
        Ten components sit just beyond the scree-plot elbow; retaining as much
        as 95–99% of the variation produced essentially the same three groups.

        Finally, **K-Means clustering** groups dates with similar compressed
        mobility patterns. Testing nearby cluster counts showed that three
        recurring states gave the clearest stable interpretation; a fourth
        cluster mainly isolated a handful of unusual dates rather than adding
        a fourth recurring kind of mobility day.

        The names **Faster, Lighter**, **Holiday-Like**, and **Busier, Slower**
        were assigned after clustering from the observed profiles. Holiday
        labels were not used to create the groups, so “Holiday-Like” describes
        the mobility pattern rather than declaring every date a holiday.
        """
    )

with exploration_section(
    key="mobility_day_type_exploration_area",
    title="Explore a mobility day",
    description=(
        "Choose any modeled date to see why it belongs to its day type and "
        "whether that state lasted for one day or persisted across several."
    ),
):
    available_dates = (
        day_types["date"]
        .sort_values()
        .tolist()
    )

    default_date = pd.Timestamp(
        "2024-07-05"
    )
    if default_date not in available_dates:
        default_date = available_dates[
            len(available_dates) // 2
        ]

    selected_date = st.selectbox(
        "Date",
        options=available_dates,
        index=available_dates.index(
            default_date
        ),
        format_func=_format_date,
        key="mobility_day_type_date",
    )

    selected_date = pd.Timestamp(
        selected_date
    ).normalize()

    selected_day = day_types.loc[
        day_types["date"].eq(
            selected_date
        )
    ].iloc[0]

    selected_run = _find_run_for_date(
        runs,
        selected_date,
    )

    state_id = str(
        selected_day["day_type_id"]
    )
    state_name = str(
        selected_day["day_type_name"]
    )

    info_columns = st.columns(
        [1.15, 1.0, 1.0, 1.0]
    )

    info_columns[0].metric(
        "Mobility day type",
        state_name,
    )
    info_columns[1].metric(
        "State duration",
        f"{int(selected_run['duration_days'])} day"
        + (
            ""
            if int(
                selected_run[
                    "duration_days"
                ]
            ) == 1
            else "s"
        ),
    )
    info_columns[2].metric(
        "Run began",
        selected_run[
            "start_date"
        ].strftime("%b %d, %Y"),
    )
    info_columns[3].metric(
        "Run ended",
        selected_run[
            "end_date"
        ].strftime("%b %d, %Y"),
    )

    st.markdown(
        f"""
        <div class="soft-callout">
        <strong>{_format_date(selected_date)}</strong> is classified as
        <strong>{state_name}</strong>. {DAY_TYPE_DESCRIPTIONS[state_id]}
        The chart below shows the selected date itself, not the cluster average.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### How this date differed from a typical day like it")

    st.plotly_chart(
        _selected_date_heatmap(
            date_features,
            selected_date,
        ),
        width="stretch",
        config={
            "displayModeBar": False,
        },
            key="raw26_plotly_03",
)

    selected_features = _diverse_top_features(
        date_features.loc[
            date_features["date"].eq(
                selected_date
            )
        ],
        n=3,
    )

    signal_phrases = [
        "**"
        + str(row["metric_label"])
        + " · "
        + str(row["daypart_label"])
        + "** "
        + _feature_phrase(
            row,
            include_magnitude=True,
        ).split(" was ", 1)[1]
        for _, row in selected_features.iterrows()
    ]

    render_chart_insight(
        "The strongest departures on this date were "
        + "; ".join(signal_phrases)
        + "."
    )

    st.markdown(
        "#### Did this state persist?"
    )

    st.plotly_chart(
        _run_timeline(
            runs,
            selected_run=selected_run,
            selected_date=selected_date,
        ),
        width="stretch",
        config={
            "displayModeBar": False,
        },
            key="raw26_plotly_04",
)

    st.caption(
        "The dotted line marks the selected date. The surrounding window shows "
        "nearby contiguous runs so transitions into and out of the selected "
        "state are easy to see."
    )

    # WHY: the chart already reports dates and duration. The takeaway earns
    # its space by explaining the transition around the selected mobility state.
    ordered_runs = (
        runs.sort_values(
            ["start_date", "end_date"]
        )
        .reset_index(drop=True)
    )
    run_matches = ordered_runs.index[
        ordered_runs["run_number"].eq(
            selected_run["run_number"]
        )
    ].tolist()

    if len(run_matches) != 1:
        raise RuntimeError(
            "Expected exactly one selected mobility-state run."
        )

    run_index = run_matches[0]
    previous_run = (
        ordered_runs.iloc[run_index - 1]
        if run_index > 0
        else None
    )
    next_run = (
        ordered_runs.iloc[run_index + 1]
        if run_index < len(ordered_runs) - 1
        else None
    )

    duration = int(
        selected_run["duration_days"]
    )
    previous_name = (
        str(previous_run["day_type_name"])
        if previous_run is not None
        else None
    )
    next_name = (
        str(next_run["day_type_name"])
        if next_run is not None
        else None
    )

    if duration == 1:
        if previous_name and next_name:
            if previous_name == next_name:
                persistence_copy = (
                    f"{state_name} briefly interrupted {previous_name} for "
                    "one day before NYC returned to the same mobility state."
                )
            else:
                persistence_copy = (
                    f"NYC moved from {previous_name} into a one-day "
                    f"{state_name} state, then shifted to {next_name}."
                )
        elif next_name:
            persistence_copy = (
                f"The study window opens with a one-day {state_name} state "
                f"before NYC shifted to {next_name}."
            )
        elif previous_name:
            persistence_copy = (
                f"The study window closes with a one-day {state_name} state "
                f"after NYC moved out of {previous_name}."
            )
        else:
            persistence_copy = (
                f"{state_name} appears here as a one-day mobility state."
            )
    else:
        stretch = (
            f"{duration}-day {state_name} stretch"
        )

        if previous_name and next_name:
            if previous_name == next_name:
                persistence_copy = (
                    f"A {stretch} interrupted {previous_name}, after which "
                    "NYC returned to the same mobility state."
                )
            else:
                persistence_copy = (
                    f"NYC shifted from {previous_name} into a {stretch}, "
                    f"then moved to {next_name}."
                )
        elif next_name:
            persistence_copy = (
                f"The study window opens in a {stretch}, followed by a "
                f"shift to {next_name}."
            )
        elif previous_name:
            persistence_copy = (
                f"The study window closes in a {stretch}, after NYC shifted "
                f"out of {previous_name}."
            )
        else:
            persistence_copy = (
                f"NYC remained in {state_name} for {duration} consecutive days."
            )

    render_chart_insight(
        persistence_copy
    )

st.divider()

st.markdown("### What this page establishes")
st.markdown(
    "NYC mobility repeatedly settles into three recognizable citywide patterns after "
    "the normal rhythm of weekdays is accounted for. **Faster, Lighter** and "
    "**Busier, Slower** make up most modeled dates, while the rarer **Holiday-Like** "
    "state marks a distinct mobility pattern. These states can persist across several "
    "days or change quickly, so the calendar captures both recurring kinds of mobility "
    "day and the transitions between them."
)

st.caption(
    "Study window: January 2023–March 2026. The day types describe recurring "
    "NYC-wide mobility patterns relative to the normal pattern for the same "
    "weekday. They are descriptive groupings, not causal estimates of "
    "congestion pricing or other events."
)
