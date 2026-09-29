from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data_access.mobility_day_types import (
    load_mobility_day_types,
)
from app.data_access.mobility_drivers import (
    GEOGRAPHY_LEVELS,
    METRICS,
    load_mobility_driver_geographies,
    load_mobility_driver_geography_context,
    load_mobility_driver_qa,
)
from app.utils.project_branding import (
    BRAND_COLORS,
    apply_branding,
    exploration_section,
    inject_app_css,
    render_chart_insight,
)


PAGE_CAPTION = "MOBILITY DRIVERS"
PAGE_TITLE = (
    "What was driving NYC away from its "
    "usual mobility pattern?"
)

CP_START = pd.Timestamp(
    "2025-01-05"
)

METRIC_LABELS = {
    "taxi_trip_count": "Taxi demand",
    "taxi_avg_trip_speed": "Taxi speed",
    "fhvhv_trip_count": "FHVHV demand",
    "fhvhv_avg_trip_speed": "FHVHV speed",
    "subway_ridership": "Subway ridership",
    "avg_bus_speed": "Bus speed",
}

COLORS = {
    "taxi_trip_count": BRAND_COLORS["dark_teal"],
    "taxi_avg_trip_speed": BRAND_COLORS["dark_teal"],
    "fhvhv_trip_count": BRAND_COLORS["terracotta"],
    "fhvhv_avg_trip_speed": BRAND_COLORS["terracotta"],
    "subway_ridership": BRAND_COLORS["seafoam"],
    "avg_bus_speed": BRAND_COLORS["seafoam"],
}

TIME_SCALE_WINDOWS = {
    "Daily": 1,
    "7-day centered mean": 7,
    "14-day centered mean": 14,
}

FORMULATION_IDS = {
    "Signed RMS": "signed_rms",
    "Mean": "mean",
}

SIGNAL_CLASSES = {
    "Both": METRICS,
    "Demand-only": (
        "taxi_trip_count",
        "fhvhv_trip_count",
        "subway_ridership",
    ),
    "Congestion-only": (
        "taxi_avg_trip_speed",
        "fhvhv_avg_trip_speed",
        "avg_bus_speed",
    ),
}


def _add_driver_diagnostics(
    frame: pd.DataFrame,
    *,
    active_metrics: tuple[str, ...] = METRICS,
) -> pd.DataFrame:
    """Add contribution, concentration, and leader fields for active signals."""
    out = frame.copy()

    magnitude = out[
        list(active_metrics)
    ].abs()

    out["total_departure"] = (
        magnitude.sum(
            axis=1,
            min_count=1,
        )
    )

    denominator = out[
        "total_departure"
    ].replace(
        0,
        np.nan,
    )

    # WHY: clear old shares first so changing signal class cannot leave stale
    # contribution values behind.
    for metric in METRICS:
        out[f"share__{metric}"] = np.nan

    for metric in active_metrics:
        out[
            f"share__{metric}"
        ] = (
            magnitude[metric]
            / denominator
        )

    out["dominant_metric"] = (
        magnitude.idxmax(
            axis=1
        )
    )
    out["dominant_label"] = (
        out["dominant_metric"]
        .map(METRIC_LABELS)
    )
    out["dominant_share"] = (
        magnitude.max(axis=1)
        / denominator
    )

    shares = (
        out[
            [
                f"share__{metric}"
                for metric in active_metrics
            ]
        ]
        .fillna(0)
        .to_numpy()
    )

    concentration = (
        np.square(shares)
        .sum(axis=1)
    )

    out[
        "effective_contributors"
    ] = np.where(
        concentration > 0,
        1 / concentration,
        np.nan,
    )

    return out


def _combine_dayparts(
    frame: pd.DataFrame,
    *,
    daypart: str,
    formulation: str,
) -> pd.DataFrame:
    """Reduce one selected geography to one row per date."""
    if daypart != "All day":
        daypart_value = {
            "Overnight": "overnight",
            "AM Peak": "am_peak",
            "Midday": "midday",
            "PM Peak": "pm_peak",
            "Evening": "evening",
        }[daypart]

        return (
            frame.loc[
                frame["daypart"].eq(daypart_value),
                ["date", *METRICS],
            ]
            .sort_values("date")
            .reset_index(drop=True)
        )

    rows = []

    for date, group in frame.groupby(
        "date",
        sort=True,
    ):
        row = {"date": date}

        for metric in METRICS:
            values = (
                pd.to_numeric(
                    group[metric],
                    errors="coerce",
                )
                .dropna()
                .to_numpy(dtype=float)
            )

            if not len(values):
                score = np.nan
            elif formulation == "Mean":
                score = float(
                    np.mean(values)
                )
            else:
                mean_value = float(
                    np.mean(values)
                )
                score = float(
                    np.sign(mean_value)
                    * np.sqrt(
                        np.mean(
                            np.square(values)
                        )
                    )
                )

            row[metric] = score

        rows.append(row)

    return pd.DataFrame(rows)


def _prepare_view(
    drivers: pd.DataFrame,
    *,
    geography_level: str,
    geography_value: str,
    daypart: str,
    formulation: str,
    window: int,
    active_metrics: tuple[str, ...] = METRICS,
) -> pd.DataFrame:
    """Prepare one geography's driver river at the requested temporal lens."""
    selected = drivers.loc[
        drivers["geography_level"].astype(str).eq(
            geography_level
        )
        & drivers["geography_value"].astype(str).eq(
            str(geography_value)
        )
    ].copy()

    view = _combine_dayparts(
        selected,
        daypart=daypart,
        formulation=formulation,
    )

    if window > 1:
        view[list(METRICS)] = (
            view[list(METRICS)]
            .rolling(
                window=window,
                center=True,
                min_periods=max(
                    1,
                    window // 2,
                ),
            )
            .mean()
        )

    return _add_driver_diagnostics(
        view,
        active_metrics=active_metrics,
    )


def _geography_options(
    context: pd.DataFrame,
    geography_level: str,
) -> pd.DataFrame:
    """Return canonical values and reader-facing labels for one level."""
    return (
        context.loc[
            context["geography_level"].astype(str).eq(
                geography_level
            ),
            [
                "geography_value",
                "display_label",
            ],
        ]
        .drop_duplicates()
        .sort_values("display_label")
        .reset_index(drop=True)
    )


def _mask_day_type(
    view: pd.DataFrame,
    day_types: pd.DataFrame,
    selected_day_type: str,
) -> pd.DataFrame:
    """Preserve calendar gaps when showing one NYC-wide Mobility Day Type."""
    if selected_day_type == "All day types":
        return view.copy()

    eligible_dates = set(
        day_types.loc[
            day_types["day_type_name"].eq(
                selected_day_type
            ),
            "date",
        ]
    )

    out = view.copy()
    keep = out["date"].isin(
        eligible_dates
    )

    columns_to_blank = [
        *METRICS,
        "total_departure",
        "dominant_metric",
        "dominant_label",
        "dominant_share",
        "effective_contributors",
        *[
            f"share__{metric}"
            for metric in METRICS
        ],
    ]

    existing = [
        column
        for column in columns_to_blank
        if column in out.columns
    ]
    out.loc[
        ~keep,
        existing,
    ] = np.nan

    return out


def _legend_html(
    active_metrics: tuple[str, ...] = METRICS,
) -> str:
    """Render a compact legend for the currently active signal class."""
    items = []

    for metric in active_metrics:
        items.append(
            "<span style='display:inline-flex;"
            "align-items:center;gap:0.35rem;"
            "margin-right:1.1rem;"
            "margin-bottom:0.35rem;'>"
            f"<span style='width:22px;height:6px;"
            f"background:{COLORS[metric]};"
            "display:inline-block;'></span>"
            f"<span>{METRIC_LABELS[metric]}</span>"
            "</span>"
        )

    return (
        "<div style='font-size:0.86rem;"
        "margin:0.15rem 0 0.45rem 0;'>"
        + "".join(items)
        + "</div>"
    )


def _river_figure(
    frame: pd.DataFrame,
    *,
    show_cp: bool = True,
    selected_date: pd.Timestamp | None = None,
    height: int = 525,
    x_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    active_metrics: tuple[str, ...] = METRICS,
) -> go.Figure:
    """Build the signed contribution river with one reader-facing hover card."""
    fig = go.Figure()
    x = frame["date"]

    positive_base = np.zeros(len(frame))
    negative_base = np.zeros(len(frame))

    # WHY: positive and negative bands require separate stacked traces. Those
    # geometry traces skip hover so the reader sees each signal only once.
    for metric in active_metrics:
        values = frame[metric].fillna(0).to_numpy(dtype=float)
        positive = np.clip(values, 0, None)
        upper = positive_base + positive

        fig.add_trace(
            go.Scatter(
                x=x,
                y=upper,
                mode="lines",
                line={
                    "width": 0.5,
                    "color": COLORS[metric],
                },
                fill=(
                    "tozeroy"
                    if not fig.data
                    else "tonexty"
                ),
                fillcolor=COLORS[metric],
                name=METRIC_LABELS[metric],
                showlegend=False,
                hoverinfo="skip",
            )
        )
        positive_base = upper

    for metric in active_metrics:
        values = frame[metric].fillna(0).to_numpy(dtype=float)
        negative = np.clip(values, None, 0)
        lower = negative_base + negative

        fig.add_trace(
            go.Scatter(
                x=x,
                y=lower,
                mode="lines",
                line={
                    "width": 0.5,
                    "color": COLORS[metric],
                },
                fill=(
                    "tozeroy"
                    if not np.any(negative_base)
                    else "tonexty"
                ),
                fillcolor=COLORS[metric],
                name=METRIC_LABELS[metric],
                showlegend=False,
                hoverinfo="skip",
            )
        )
        negative_base = lower

    # WHY: one invisible trace owns the hover card. This prevents the positive
    # and negative geometry traces from listing all six signals twice.
    hover_rows = []
    for _, row in frame.iterrows():
        details = []
        for metric in active_metrics:
            value = pd.to_numeric(
                pd.Series([row[metric]]),
                errors="coerce",
            ).iloc[0]

            if pd.isna(value):
                details.append("Not available")
                continue

            if metric in {
                "taxi_avg_trip_speed",
                "fhvhv_avg_trip_speed",
                "avg_bus_speed",
            }:
                meaning = (
                    "slower than usual"
                    if value > 0
                    else "faster than usual"
                    if value < 0
                    else "near usual"
                )
            else:
                meaning = (
                    "higher than usual"
                    if value > 0
                    else "lower than usual"
                    if value < 0
                    else "near usual"
                )

            details.append(
                f"{value:+.2f} IQR · {meaning}"
            )

        hover_rows.append(details)

    hover_custom = np.asarray(
        hover_rows,
        dtype=object,
    )

    hover_lines = [
        (
            f"{METRIC_LABELS[metric]}: "
            f"%{{customdata[{index}]}}<br>"
        )
        for index, metric in enumerate(
            active_metrics
        )
    ]

    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.zeros(len(frame)),
            mode="lines",
            line={
                "width": 12,
                "color": "rgba(0,0,0,0)",
            },
            showlegend=False,
            customdata=hover_custom,
            hovertemplate=(
                "<b>%{x|%b %d, %Y}</b><br>"
                + "".join(hover_lines)
                + "<extra></extra>"
            ),
        )
    )

    fig.add_hline(
        y=0,
        line_width=1.35,
        line_color=BRAND_COLORS["dark_teal"],
    )

    if show_cp:
        fig.add_vline(
            x=CP_START,
            line_dash="dash",
            line_width=1.1,
            line_color=BRAND_COLORS["terracotta"],
            annotation_text="Congestion pricing begins",
            annotation_position="top right",
        )

    if selected_date is not None:
        fig.add_vline(
            x=selected_date,
            line_width=1.4,
            line_dash="dot",
            line_color=BRAND_COLORS["dark_teal"],
            annotation_text="Selected date",
            annotation_position="top",
        )

    apply_branding(fig)

    xaxis = {
        "title": None,
        "domain": [0.0, 1.0],
        "automargin": False,
        "gridcolor": BRAND_COLORS["ice"],
    }
    if x_range is not None:
        xaxis["range"] = list(x_range)
        xaxis["autorange"] = False

    fig.update_layout(
        title={"text": ""},
        autosize=True,
        showlegend=False,
        hovermode="x",
        height=height,
        margin={
            "l": 72,
            "r": 24,
            "t": 38,
            "b": 50,
        },
        xaxis=xaxis,
        yaxis={
            "title": (
                "Departure from usual "
                "pattern (IQR units)"
            ),
            "automargin": True,
            "gridcolor": BRAND_COLORS["ice"],
            "zeroline": False,
        },
    )

    return fig


def _contribution_figure(
    row: pd.Series,
    *,
    active_metrics: tuple[str, ...] = METRICS,
) -> go.Figure:
    """Show absolute contribution shares for the active signal class."""
    values = [
        float(
            row[
                f"share__{metric}"
            ]
        )
        * 100
        for metric in active_metrics
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=[
                METRIC_LABELS[
                    metric
                ]
                for metric in active_metrics
            ],
            orientation="h",
            marker_color=[
                COLORS[
                    metric
                ]
                for metric in active_metrics
            ],
            customdata=[
                float(
                    row[metric]
                )
                for metric in active_metrics
            ],
            hovertemplate=(
                "%{y}: %{x:.1f}% "
                "of departure<br>"
                "Signed score: "
                "%{customdata:.2f} "
                "IQR units"
                "<extra></extra>"
            ),
        )
    )

    apply_branding(fig)
    fig.update_layout(
        title={
            "text": ""
        },
        height=335,
        showlegend=False,
        margin={
            "l": 120,
            "r": 25,
            "t": 15,
            "b": 45,
        },
        xaxis={
            "title": (
                "Share of total departure"
            ),
            "ticksuffix": "%",
            "range": [
                0,
                max(
                    55,
                    max(values) * 1.12,
                ),
            ],
        },
        yaxis={
            "title": None,
            "categoryorder": "array",
            "categoryarray": [
                METRIC_LABELS[
                    metric
                ]
                for metric
                in reversed(
                    active_metrics
                )
            ],
        },
    )

    return fig


def _selected_regime(
    frame: pd.DataFrame,
    selected_date: pd.Timestamp,
) -> tuple[
    pd.Timestamp,
    pd.Timestamp,
    int,
    str,
]:
    """Return the contiguous dominant-driver run containing the selected date."""
    marked = frame.copy()

    marked["run_id"] = (
        marked[
            "dominant_metric"
        ]
        .ne(
            marked[
                "dominant_metric"
            ].shift()
        )
        .cumsum()
    )

    row = marked.loc[
        marked["date"].eq(
            selected_date
        )
    ].iloc[0]

    run = marked.loc[
        marked["run_id"].eq(
            row["run_id"]
        )
    ]

    return (
        run["date"].min(),
        run["date"].max(),
        len(run),
        str(
            row[
                "dominant_label"
            ]
        ),
    )


def _fmt_date(
    value: pd.Timestamp,
) -> str:
    """Format a date consistently without platform-specific directives."""
    return (
        f"{value.strftime('%b')} "
        f"{value.day}, "
        f"{value.year}"
    )


st.set_page_config(
    page_title="Mobility Drivers",
    page_icon="〰️",
    layout="wide",
)
inject_app_css()

drivers = (
    load_mobility_driver_geographies()
)
geography_context = (
    load_mobility_driver_geography_context()
)
_qa = load_mobility_driver_qa()
day_types = (
    load_mobility_day_types()[
        [
            "date",
            "day_type_id",
            "day_type_name",
        ]
    ]
    .copy()
)

hero = _prepare_view(
    drivers,
    geography_level="Citywide",
    geography_value="NYC",
    daypart="All day",
    formulation="Signed RMS",
    window=14,
    active_metrics=SIGNAL_CLASSES["Demand-only"],
)

st.caption(
    PAGE_CAPTION
)
st.title(
    PAGE_TITLE
)
st.markdown(
    """
    <div class="app-subtitle">
    NYC's travel demand rarely changes because of just one mode.
    This river shows how Taxi demand, FHVHV demand, and Subway ridership
    moved above or below their usual levels for the same weekday and part
    of the day — and how those three modes combined over time.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "### How to read this river"
)
read_cols = st.columns(3)
read_cols[0].markdown(
    "**Direction tells you what changed**  \n"
    "Up means more travel demand than usual; down means less "
    "travel demand than usual."
)
read_cols[1].markdown(
    "**Thickness tells you how unusual it was**  \n"
    "A thicker band means that signal was farther from its normal "
    "level for the same weekday and part of the day."
)
read_cols[2].markdown(
    "**Color tells you which mode contributed**  \n"
    "Several thick bands at once mean the demand departure was shared "
    "across Taxi, FHVHV, and Subway."
)

st.markdown(
    _legend_html(
        SIGNAL_CLASSES["Demand-only"]
    ),
    unsafe_allow_html=True,
)
st.plotly_chart(
    _river_figure(
        hero,
        active_metrics=SIGNAL_CLASSES["Demand-only"],
    ),
    width="stretch",
    config={
        "displayModeBar": False,
    },
    key="mobility_drivers_hero_river",
)

st.caption(
    "The hero focuses on demand and uses a 14-day centered mean to make "
    "sustained changes easier to see. Band thickness preserves the size of "
    "each standardized departure; it is not forced to 100%."
)

render_chart_insight(
    "Citywide demand shifts noticeably around January 2025. Before congestion "
    "pricing begins, Taxi, FHVHV, and Subway demand spend long stretches below "
    "their usual levels; afterward, all three spend much more time above them, "
    "often together. Here, 'usual' means the typical level for the same weekday "
    "and same part of the day across the full study period, with all five "
    "dayparts combined into each daily score. The timing is striking, but this "
    "descriptive view alone does not show that congestion pricing caused the shift."
)

st.caption(
    "Each demand signal is shown in interquartile-range (IQR) units. In plain "
    "language, the scale asks how far a date sits from the middle of its usual "
    "range for the same weekday and part of the day. Larger values mean a more "
    "unusual departure. This scale describes what happened; it does not test "
    "statistical significance or prove what caused the change."
)

with st.expander(
    "Why use signed RMS for the default view?"
):
    st.markdown(
        """
        Signed RMS keeps a large departure visible when different parts of
        the day partly offset one another; the sign still follows the day's
        average direction.

        Across the study period, Mean and signed RMS chose the same leading
        signal on **85.6% of dates** and agreed on direction **95.6% of the
        time**. That makes signed RMS a useful default: it keeps the broad
        story similar to the simpler Mean view while preserving stronger
        within-day departures. Mean remains available below for comparison.
        """
    )

with exploration_section(
    key=(
        "mobility_drivers_"
        "exploration_area"
    ),
    title="Explore the drivers",
    description=(
        "Choose a place, signal class, Mobility Day Type, part of the day, "
        "and time scale to see how the mix of mobility drivers changes."
    ),
):
    geography_controls = st.columns(
        [1, 1.4]
    )

    geography_level = (
        geography_controls[0].selectbox(
            "Geography level",
            list(
                GEOGRAPHY_LEVELS
            ),
            index=0,
            help=(
                "Use the same Citywide, Taxi Zone, Borough, "
                "Mobility Environment, and Policy Geography "
                "levels used elsewhere in the Showcase."
            ),
        )
    )

    geography_options = (
        _geography_options(
            geography_context,
            geography_level,
        )
    )

    geography_values = (
        geography_options[
            "geography_value"
        ]
        .astype(str)
        .tolist()
    )
    geography_labels = dict(
        zip(
            geography_options[
                "geography_value"
            ].astype(str),
            geography_options[
                "display_label"
            ].astype(str),
        )
    )

    geography_value = (
        geography_controls[1].selectbox(
            "Geography",
            geography_values,
            index=0,
            format_func=lambda value: (
                geography_labels.get(
                    str(value),
                    str(value),
                )
            ),
            help=(
                "Taxi Zone choices include borough context in the "
                "display label while preserving the canonical zone ID."
            ),
        )
    )

    class_controls = st.columns(
        [1.15, 2.85]
    )
    signal_class = class_controls[0].selectbox(
        "Signal class",
        [
            "Both",
            "Demand-only",
            "Congestion-only",
        ],
        index=0,
        help=(
            "Demand-only shows Taxi demand, FHVHV demand, and Subway "
            "ridership. Congestion-only shows Taxi, FHVHV, and Bus street "
            "speeds. Both combines all six. This choice changes which "
            "signals build the river; it does not limit the view to "
            "stress-anomaly dates."
        ),
        key="mobility_drivers_signal_class",
    )
    class_controls[1].caption(
        "**Demand-only** uses Taxi demand, FHVHV demand, and Subway "
        "ridership. **Congestion-only** uses Taxi, FHVHV, and Bus street "
        "speeds. **Both** combines all six signals."
    )
    active_metrics = SIGNAL_CLASSES[
        signal_class
    ]

    filter_controls = st.columns(
        [1.15, 1, 1]
    )

    selected_day_type = (
        filter_controls[0].selectbox(
            "Mobility Day Type",
            [
                "All day types",
                "Faster, Lighter",
                "Holiday-Like",
                "Busier, Slower",
            ],
            index=0,
            help=(
                "Mobility Day Type is the NYC-wide classification "
                "of each date used on the Mobility Day Types page."
            ),
        )
    )

    daypart = (
        filter_controls[1].selectbox(
            "Daypart",
            [
                "All day",
                "Overnight",
                "AM Peak",
                "Midday",
                "PM Peak",
                "Evening",
            ],
            index=0,
        )
    )

    smoothing_label = (
        filter_controls[2].selectbox(
            "Time scale",
            list(
                TIME_SCALE_WINDOWS
            ),
            index=2,
            help=(
                "Daily preserves short shocks; smoothing makes "
                "sustained driver changes easier to see."
            ),
        )
    )

    if daypart == "All day":
        formulation = st.radio(
            "Combine dayparts",
            [
                "Signed RMS",
                "Mean",
            ],
            index=0,
            horizontal=True,
            help=(
                "Signed RMS gives extra weight to larger departures, so "
                "a strong morning or evening shift remains visible even if "
                "other parts of the day move the other way. Mean shows the "
                "simple average direction across the day."
            ),
        )
    else:
        formulation = "Signed RMS"
        st.caption(
            f"**{daypart} selected:** there is only one daypart "
            "to show, so no Mean/RMS combination is needed."
        )

    window = (
        TIME_SCALE_WINDOWS[
            smoothing_label
        ]
    )

    base_view = _prepare_view(
        drivers,
        geography_level=geography_level,
        geography_value=geography_value,
        daypart=daypart,
        formulation=formulation,
        window=window,
        active_metrics=active_metrics,
    )

    view = _mask_day_type(
        base_view,
        day_types,
        selected_day_type,
    )

    geography_label = (
        geography_labels.get(
            str(geography_value),
            str(geography_value),
        )
    )

    st.markdown(
        "#### Recreate the river for this view"
    )
    st.markdown(
        _legend_html(active_metrics),
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        _river_figure(
            view,
            active_metrics=active_metrics,
        ),
        width="stretch",
        config={
            "displayModeBar": False,
        },
        key=(
            "mobility_drivers_"
            "explorer_river"
        ),
    )

    if (
        selected_day_type
        != "All day types"
    ):
        st.caption(
            f"Showing **{geography_label}** only on dates when "
            f"NYC overall was **{selected_day_type}**. Blank stretches "
            "are other day types; separated dates are not connected."
        )

    st.markdown(
        "### Inspect a date"
    )
    st.caption(
        "The date control changes only the deep dive below. "
        "It does not filter the historical river above."
    )

    available_dates = (
        view.loc[
            view[
                "total_departure"
            ].notna(),
            "date",
        ]
        .tolist()
    )

    if not available_dates:
        st.info(
            "No dates are available for this combination of filters."
        )
    else:
        default_date = pd.Timestamp(
            "2025-12-10"
        )

        if (
            default_date
            not in available_dates
        ):
            default_date = (
                available_dates[
                    len(
                        available_dates
                    )
                    // 2
                ]
            )

        selected_date = st.selectbox(
            "Selected date",
            available_dates,
            index=(
                available_dates.index(
                    default_date
                )
            ),
            format_func=_fmt_date,
            key=(
                "mobility_drivers_"
                "selected_date"
            ),
        )
        selected_date = (
            pd.Timestamp(
                selected_date
            )
            .normalize()
        )

        selected = (
            view.loc[
                view["date"].eq(
                    selected_date
                )
            ]
            .iloc[0]
        )

        day_match = (
            day_types.loc[
                day_types[
                    "date"
                ].eq(
                    selected_date
                )
            ]
        )

        day_type_name = (
            str(
                day_match.iloc[0][
                    "day_type_name"
                ]
            )
            if not day_match.empty
            else "Not classified"
        )

        metrics = st.columns(
            4
        )
        metrics[0].metric(
            "Mobility Day Type",
            day_type_name,
        )
        metrics[1].metric(
            "Leading signal",
            selected[
                "dominant_label"
            ],
        )
        metrics[2].metric(
            "Leader share",
            (
                f"{selected['dominant_share']:.1%}"
            ),
        )
        metrics[3].metric(
            "Signals contributing",
            (
                (
                    f"{selected['effective_contributors']:.2f} "
                    f"of {len(active_metrics)}"
                )
            ),
        )

        detail_columns = st.columns(
            [1, 1.2]
        )

        with detail_columns[0]:
            st.markdown(
                "#### Which signals drove the departure?"
            )
            st.plotly_chart(
                _contribution_figure(
                    selected,
                    active_metrics=active_metrics,
                ),
                width="stretch",
                config={
                    "displayModeBar": False,
                },
                key=(
                    "mobility_drivers_"
                    "contribution"
                ),
            )

        with detail_columns[1]:
            st.markdown(
                "#### What was happening in the six weeks around this date?"
            )
            st.caption(
                "This zoom shows the 21 days before and after the selected "
                "date, so you can see whether its driver mix was a brief "
                "event or part of a longer pattern."
            )

            nearby_start = (
                selected_date
                - pd.Timedelta(days=21)
            )
            nearby_end = (
                selected_date
                + pd.Timedelta(days=21)
            )

            nearby = (
                base_view.loc[
                    base_view[
                        "date"
                    ].between(
                        nearby_start,
                        nearby_end,
                    )
                ]
                .copy()
            )

            st.plotly_chart(
                _river_figure(
                    nearby,
                    show_cp=(
                        nearby_start
                        <= CP_START
                        <= nearby_end
                    ),
                    selected_date=(
                        selected_date
                    ),
                    height=350,
                    x_range=(
                        nearby_start,
                        nearby_end,
                    ),
                    active_metrics=active_metrics,
                ),
                width="stretch",
                config={
                    "displayModeBar": False,
                },
                key=(
                    "mobility_drivers_"
                    "nearby_river"
                ),
            )

        daypart_phrase = (
            "across the day"
            if daypart == "All day"
            else (
                f"during the "
                f"{daypart}"
            )
        )

        regime_start, regime_end, regime_days, regime_label = (
            _selected_regime(
                base_view,
                selected_date,
            )
        )

        leader_value = float(
            selected[
                selected["dominant_metric"]
            ]
        )
        leader_is_speed = (
            selected["dominant_metric"]
            in {
                "taxi_avg_trip_speed",
                "fhvhv_avg_trip_speed",
                "avg_bus_speed",
            }
        )
        if leader_is_speed:
            leader_direction = (
                "slower than usual"
                if leader_value > 0
                else "faster than usual"
            )
        else:
            leader_direction = (
                "higher than usual"
                if leader_value > 0
                else "lower than usual"
            )

        broad_threshold = (
            2.25
            if len(active_metrics) == 3
            else 4
        )
        sharing_phrase = (
            "broadly shared"
            if selected["effective_contributors"] >= broad_threshold
            else "more concentrated"
        )

        persistence_phrase = (
            "only this date"
            if regime_days == 1
            else (
                f"{regime_days} consecutive days "
                f"({_fmt_date(regime_start)}–{_fmt_date(regime_end)})"
            )
        )

        render_chart_insight(
            f"On **{_fmt_date(selected_date)}**, NYC's Mobility Day Type "
            f"was **{day_type_name}**. For **{geography_label}** "
            f"{daypart_phrase}, **{selected['dominant_label']}** was the "
            f"largest contributor at **{selected['dominant_share']:.1%}** "
            f"and was **{leader_direction}**. The departure was "
            f"**{sharing_phrase}** across the selected "
            f"**{signal_class}** signals. Its contribution mix was "
            f"equivalent to about **{selected['effective_contributors']:.2f} "
            f"of {len(active_metrics)} signals contributing equally**. "
            f"**{regime_label}** remained the leading signal for "
            f"**{persistence_phrase}**."
        )

        if (
            geography_level
            != "Citywide"
        ):
            st.caption(
                "Mobility Day Type is defined for NYC as a whole. Changing "
                "the geography updates the driver signals for that place, "
                "but the date keeps its citywide Mobility Day Type."
            )

st.markdown(
    "### How Mobility Day Types and Mobility Drivers fit together"
)
st.markdown(
    "Mobility Day Types describe **what kind of mobility day** NYC was "
    "experiencing. Mobility Drivers explains **which signals helped produce "
    "that pattern**. Across the study period, all six signals took a turn as "
    "the leading contributor within each of the three day types. No single "
    "signal dominated most dates: the most common leader appeared on only "
    "**29.0%** of Busier, Slower dates, **27.6%** of Faster, Lighter dates, "
    "and **42.0%** of Holiday-Like dates. In other words, the same kind of "
    "mobility day can emerge from different combinations of Taxi, FHVHV, "
    "Subway, and Bus behavior."
)

with st.expander(
    "How this page works"
):
    st.markdown(
        """
        **1. Start with six mobility signals.** The page uses Taxi demand and
        speed, FHVHV demand and speed, Subway ridership, and Bus speed across
        five parts of the day. Trip and ridership counts are added across the
        selected geography. Speed measures are weighted by activity, so a
        high-volume area or period has more influence than a low-volume one.

        **2. Compare like with like.** A Monday morning should not be judged
        against a Saturday night. Each signal is therefore compared with its
        usual level for the **same weekday and same part of the day**. The
        result is expressed in interquartile-range (IQR) units: a larger
        absolute value means the signal was farther outside its usual range.
        Count measures use a `log1p` transformation first so unusually large
        counts do not overwhelm the comparison.

        **3. Put demand and congestion on one readable direction.** Higher
        demand points upward. Street speed is reversed for display, so
        **slower-than-usual streets also point upward** and faster-than-usual
        streets point downward. This gives the river a common visual language
        without pretending demand and speed are the same measure.

        **4. Choose which signals build the river.** **Demand-only** uses Taxi
        demand, FHVHV demand, and Subway ridership. **Congestion-only** uses
        Taxi, FHVHV, and Bus street speeds. **Both** uses all six. These labels
        match the terminology used elsewhere in the Showcase, but on this page
        they select metric families; they do **not** restrict the analysis to
        stress-anomaly dates.

        **5. Combine the parts of the day.** **Signed RMS** gives more weight
        to larger departures, which keeps a strong shift visible even when
        another part of the day moves in the opposite direction. The sign
        still follows the day's average direction. **Mean** is also available
        and simply averages the five dayparts.

        **6. Measure who contributed.** Contribution shares use each signal's
        absolute departure, so an upward signal and a downward signal do not
        cancel one another. The page also reports a concentration-based
        contributor count. A value near **1** means one signal did most of the
        work; a value near the number of available signals means the departure
        was broadly shared. Think of it as the number of equally strong signals
        that would produce a similar contribution mix.
        """
    )

st.caption(
    "Evidence scope: January 2023 through the latest date available in the "
    "Showcase data. Each place is compared with its own history for the same "
    "weekday and part of the day. The page describes how mobility differed "
    "from usual; it does not by itself show that congestion pricing or any "
    "other event caused those differences."
)
