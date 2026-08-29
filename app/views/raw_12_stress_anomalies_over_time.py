from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data_access.anomalies import (
    ANOMALY_EVENT_UNIVERSE_PATH,
    EVENT_ID_COLUMN,
    SELECTED_FINALIST_FLAG,
)
from app.utils.project_branding import (
    BRAND_COLORS,
    apply_branding,
    inject_app_css,
)


inject_app_css()


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------
CP_START_DATE = pd.Timestamp("2025-01-05")
PRESET_CONFIG_VERSION = 2
ALL_TEMPORAL_BUCKETS = "All temporal buckets"
CONGESTION_FLAG = "has_congestion_oriented"
DEMAND_FLAG = "has_positive_demand_shock"
FAMILY_COLUMN = "stress_family_exclusive"

FAMILY_ORDER = [
    "Congestion-only",
    "Demand-only",
    "Both",
]

FAMILY_COLORS = {
    "Congestion-only": BRAND_COLORS["terracotta"],
    "Demand-only": BRAND_COLORS["dark_teal"],
    "Both": BRAND_COLORS["seafoam"],
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

MEASURE_OPTIONS = [
    "Incidence per 1,000",
    "Composition share",
]

CADENCE_OPTIONS = [
    "Monthly",
    "Weekly",
]

TIME_SCOPE_OPTIONS = [
    "Full period",
    "Pre-CP",
    "Post-CP",
    "Custom dates",
]

SCOUTING_COLUMNS = [
    EVENT_ID_COLUMN,
    "date",
    "temporal_bucket",
    SELECTED_FINALIST_FLAG,
    CONGESTION_FLAG,
    DEMAND_FLAG,
]

SAVED_VIEWS = {
    "Full trajectory": {
        "families": FAMILY_ORDER,
        "measure": "Incidence per 1,000",
        "cadence": "Monthly",
        "time_scope": "Full period",
        "temporal_bucket": ALL_TEMPORAL_BUCKETS,
        "start_date": date(2023, 1, 1),
        "end_date": date(2026, 3, 31),
    },
    "Late-2024 congestion wave": {
        "families": ["Congestion-only", "Both"],
        "measure": "Incidence per 1,000",
        "cadence": "Monthly",
        "time_scope": "Custom dates",
        "temporal_bucket": ALL_TEMPORAL_BUCKETS,
        "start_date": date(2023, 10, 1),
        "end_date": date(2025, 3, 31),
    },
    "Early-2025 demand surge": {
        "families": ["Demand-only", "Both"],
        "measure": "Incidence per 1,000",
        "cadence": "Monthly",
        "time_scope": "Custom dates",
        "temporal_bucket": ALL_TEMPORAL_BUCKETS,
        "start_date": date(2024, 9, 1),
        "end_date": date(2025, 6, 30),
    },
}

SAVED_VIEW_DESCRIPTIONS = {
    "Full trajectory": (
        "The complete monthly record shows how the balance among all three "
        "stress families changed from January 2023 through March 2026."
    ),
    "Late-2024 congestion wave": (
        "The expanded window includes late 2023 and early 2025 so the elevated "
        "congestion-related period can be judged against its lead-in and decline."
    ),
    "Early-2025 demand surge": (
        "The expanded window includes four months of lead-in and three months "
        "of follow-through so the early-2025 demand surge is visible in context."
    ),
}


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------
@st.cache_data(show_spinner="Loading the stress-anomaly timeline...")
def _load_temporal_universe() -> pd.DataFrame:
    """Load the full denominator with only temporal-page columns."""
    frame = pd.read_parquet(
        ANOMALY_EVENT_UNIVERSE_PATH,
        columns=SCOUTING_COLUMNS,
    )

    missing = sorted(
        set(SCOUTING_COLUMNS).difference(frame.columns)
    )
    if missing:
        raise ValueError(
            "The 3.3.6 event universe is missing temporal-page columns: "
            + ", ".join(missing)
        )

    frame = frame.copy()
    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )
    if frame["date"].isna().any():
        raise ValueError(
            "The 3.3.6 event universe contains unparseable dates."
        )

    frame[SELECTED_FINALIST_FLAG] = (
        frame[SELECTED_FINALIST_FLAG]
        .fillna(False)
        .astype(bool)
    )
    for flag in [CONGESTION_FLAG, DEMAND_FLAG]:
        frame[flag] = frame[flag].fillna(False).astype(bool)

    # pre_post_cp is intentionally sparse outside finalist rows in the export.
    frame["policy_period"] = np.where(
        frame["date"].lt(CP_START_DATE),
        "Pre-CP",
        "Post-CP",
    )
    frame["month"] = (
        frame["date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    frame["week"] = (
        frame["date"]
        .dt.to_period("W-SUN")
        .dt.start_time
    )

    selected = frame[SELECTED_FINALIST_FLAG]
    congestion = frame[CONGESTION_FLAG]
    demand = frame[DEMAND_FLAG]

    frame[FAMILY_COLUMN] = pd.Series(
        pd.NA,
        index=frame.index,
        dtype="string",
    )
    frame.loc[
        selected & congestion & ~demand,
        FAMILY_COLUMN,
    ] = "Congestion-only"
    frame.loc[
        selected & demand & ~congestion,
        FAMILY_COLUMN,
    ] = "Demand-only"
    frame.loc[
        selected & congestion & demand,
        FAMILY_COLUMN,
    ] = "Both"
    frame.loc[
        selected & ~congestion & ~demand,
        FAMILY_COLUMN,
    ] = "Unclassified"

    return frame


def _incidence_per_1k(
    numerator: pd.Series | float | int,
    denominator: pd.Series | float | int,
) -> np.ndarray:
    """Calculate stress anomalies per 1,000 eligible observations."""
    numerator_array = np.asarray(numerator)
    denominator_array = np.asarray(denominator)

    return np.where(
        denominator_array > 0,
        numerator_array / denominator_array * 1_000,
        np.nan,
    )


def _build_group_summary(
    universe: pd.DataFrame,
    *,
    group_column: str,
) -> pd.DataFrame:
    """Build distinct-event incidence and exclusive family counts."""
    denominator = (
        universe.groupby(group_column, observed=True)[EVENT_ID_COLUMN]
        .nunique()
        .rename("eligible_observations")
        .reset_index()
    )

    selected = universe[
        universe[SELECTED_FINALIST_FLAG]
    ].copy()

    overall = (
        selected.groupby(group_column, observed=True)[EVENT_ID_COLUMN]
        .nunique()
        .rename("stress_anomalies")
        .reset_index()
    )

    families = (
        selected.groupby(
            [group_column, FAMILY_COLUMN],
            observed=True,
        )[EVENT_ID_COLUMN]
        .nunique()
        .unstack(fill_value=0)
        .reindex(columns=FAMILY_ORDER, fill_value=0)
        .reset_index()
    )

    result = (
        denominator
        .merge(overall, on=group_column, how="left")
        .merge(families, on=group_column, how="left")
        .fillna(0)
        .sort_values(group_column)
        .reset_index(drop=True)
    )

    result["incidence_per_1k"] = _incidence_per_1k(
        result["stress_anomalies"],
        result["eligible_observations"],
    )

    for family in FAMILY_ORDER:
        result[f"{family}_per_1k"] = _incidence_per_1k(
            result[family],
            result["eligible_observations"],
        )

    return result


def _relative_change(
    current: float,
    reference: float,
) -> float:
    """Return percent change from a nonzero reference value."""
    if reference == 0:
        return np.nan

    return (current / reference - 1) * 100


def _format_temporal_bucket(value: str) -> str:
    """Convert an internal temporal-bucket value into visitor-facing copy."""
    if value == ALL_TEMPORAL_BUCKETS:
        return value

    return (
        value
        .replace("_", " ")
        .title()
        .replace("Am ", "AM ")
        .replace("Pm ", "PM ")
    )


def _period_label(
    value: pd.Timestamp,
    cadence: str,
) -> str:
    """Format a monthly or weekly period for cards and captions."""
    if cadence == "Monthly":
        return f"{value:%b %Y}"

    week_end = value + pd.Timedelta(days=6)
    return f"{value:%b %d}–{week_end:%b %d, %Y}"


def _build_timeline_figure(
    summary: pd.DataFrame,
    *,
    period_column: str,
    cadence: str,
    families: list[str],
    measure: str,
    show_peak_annotation: bool,
) -> go.Figure:
    """Build a stacked monthly or weekly stress-family chart."""
    fig = go.Figure()

    selected_total = summary[families].sum(axis=1)

    for family in families:
        if measure == "Composition share":
            y_values = np.where(
                selected_total.gt(0),
                summary[family] / selected_total * 100,
                np.nan,
            )
            value_label = "Composition share"
            value_format = ".1f"
            value_suffix = "%"
        else:
            y_values = summary[f"{family}_per_1k"]
            value_label = "Incidence"
            value_format = ".1f"
            value_suffix = " per 1,000"

        customdata = np.column_stack(
            [
                summary[family],
            ]
        )

        period_hover = (
            "Month: %{x|%B %Y}<br>"
            if cadence == "Monthly"
            else "Week of: %{x|%b %d, %Y}<br>"
        )
        hovertemplate = (
            f"<b>{family}</b><br>"
            + period_hover
            + f"{value_label}: %{{y:{value_format}}}{value_suffix}<br>"
            + "Stress anomalies: %{customdata[0]:,.0f}"
            + "<extra></extra>"
        )

        fig.add_trace(
            go.Bar(
                x=summary[period_column],
                y=y_values,
                name=family,
                marker_color=FAMILY_COLORS[family],
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
        )

    chart_total = (
        summary[[f"{family}_per_1k" for family in families]].sum(axis=1)
        if measure == "Incidence per 1,000"
        else pd.Series(100.0, index=summary.index)
    )

    apply_branding(fig)
    fig.update_layout(
        # The shared branding template supplies title styling without title
        # text. Some Plotly/Streamlit frontend versions render that missing
        # value literally as "undefined". These charts use Streamlit headings,
        # so keep Plotly's internal title explicitly blank.
        title={"text": ""},
        barmode="stack",
        height=560,
        margin={"l": 65, "r": 30, "t": 55, "b": 75},
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.03,
            "yanchor": "bottom",
            "traceorder": "normal",
        },
        hovermode="closest",
        bargap=(0.16 if cadence == "Monthly" else 0.08),
    )
    xaxis_updates = {
        "title_text": (
            "Month"
            if cadence == "Monthly"
            else "Week starting"
        ),
        "tickformat": (
            "%b<br>%Y"
            if cadence == "Monthly"
            else "%b %d<br>%Y"
        ),
        "showgrid": False,
    }
    if cadence == "Monthly":
        xaxis_updates["dtick"] = "M3"

    fig.update_xaxes(
        **xaxis_updates,
    )

    yaxis_updates = {
        "title_text": (
            "Stress anomalies per 1,000 eligible observations"
            if measure == "Incidence per 1,000"
            else "Composition share of selected stress anomalies"
        ),
    }
    if measure == "Composition share":
        yaxis_updates["ticksuffix"] = "%"
        yaxis_updates["range"] = [0, 100]
    else:
        yaxis_updates["rangemode"] = "tozero"

    fig.update_yaxes(**yaxis_updates)

    period_min = pd.Timestamp(summary[period_column].min())
    period_max = pd.Timestamp(summary[period_column].max())
    if period_min <= CP_START_DATE <= period_max:
        fig.add_vline(
            x=CP_START_DATE,
            line_width=2,
            line_dash="dash",
            line_color="#335C67",
        )
        fig.add_annotation(
            x=CP_START_DATE,
            y=0.98,
            xref="x",
            yref="paper",
            text="Congestion pricing began<br>Jan 5, 2025",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font={"color": "#335C67", "size": 11},
            bgcolor="rgba(255,255,255,0.88)",
            borderpad=3,
        )

    if show_peak_annotation and not summary.empty:
        peak_index = chart_total.idxmax()
        peak = summary.loc[peak_index]
        fig.add_annotation(
            x=peak[period_column],
            y=chart_total.loc[peak_index],
            text=(
                f"Peak: {_period_label(peak[period_column], cadence)}<br>"
                f"{chart_total.loc[peak_index]:.1f} per 1,000"
            ),
            showarrow=True,
            arrowhead=2,
            ax=45,
            ay=-50,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=BRAND_COLORS["seafoam"],
            borderwidth=1,
            borderpad=5,
            font={"color": BRAND_COLORS["dark_teal"], "size": 12},
        )

    return fig


# ---------------------------------------------------------------------
# Saved-view state
# ---------------------------------------------------------------------
def _apply_saved_view() -> None:
    """Apply a saved view immediately when its selector changes."""
    view_name = st.session_state.get("raw12_saved_view")
    view = SAVED_VIEWS.get(str(view_name))
    if view is None:
        return

    st.session_state["raw12_families"] = list(view["families"])
    st.session_state["raw12_measure"] = str(view["measure"])
    st.session_state["raw12_cadence"] = str(view["cadence"])
    st.session_state["raw12_time_scope"] = str(view["time_scope"])
    st.session_state["raw12_temporal_bucket"] = str(view["temporal_bucket"])
    st.session_state["raw12_start_date"] = view["start_date"]
    st.session_state["raw12_end_date"] = view["end_date"]


def _mark_custom_view() -> None:
    """Mark the view as custom after any explorer control changes."""
    st.session_state["raw12_saved_view"] = "Custom"


# ---------------------------------------------------------------------
# Load and validate the canonical surface
# ---------------------------------------------------------------------
try:
    event_universe = _load_temporal_universe()
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(str(exc))
    st.stop()

selected_events = event_universe[
    event_universe[SELECTED_FINALIST_FLAG]
].copy()

eligible_event_count = int(
    event_universe[EVENT_ID_COLUMN].nunique()
)
selected_event_count = int(
    selected_events[EVENT_ID_COLUMN].nunique()
)
duplicate_event_ids = int(
    event_universe[EVENT_ID_COLUMN].duplicated().sum()
)
unclassified_event_count = int(
    selected_events[FAMILY_COLUMN].eq("Unclassified").sum()
)

if duplicate_event_ids:
    st.error(
        "The full event universe contains duplicate comparison_event_id values."
    )
    st.stop()

if unclassified_event_count:
    st.error(
        "Some selected stress anomalies do not map to a display family."
    )
    st.stop()

monthly_summary = _build_group_summary(
    event_universe,
    group_column="month",
)
period_summary = _build_group_summary(
    event_universe,
    group_column="policy_period",
)

pre_row = period_summary.loc[
    period_summary["policy_period"].eq("Pre-CP")
].iloc[0]
post_row = period_summary.loc[
    period_summary["policy_period"].eq("Post-CP")
].iloc[0]
peak_row = monthly_summary.loc[
    monthly_summary["incidence_per_1k"].idxmax()
]

overall_incidence = (
    selected_event_count
    / eligible_event_count
    * 1_000
)
overall_post_change = _relative_change(
    float(post_row["incidence_per_1k"]),
    float(pre_row["incidence_per_1k"]),
)
demand_post_change = _relative_change(
    float(post_row["Demand-only_per_1k"]),
    float(pre_row["Demand-only_per_1k"]),
)
congestion_post_change = _relative_change(
    float(post_row["Congestion-only_per_1k"]),
    float(pre_row["Congestion-only_per_1k"]),
)


# ---------------------------------------------------------------------
# Frozen answer
# ---------------------------------------------------------------------
st.caption("STRESS ANOMALY TEMPORAL EXPLORER")
st.title("When did stress anomalies intensify—and what kinds occurred?")
st.markdown(
    (
        "The final production surface contains "
        f"**{selected_event_count:,} stress anomalies** across "
        f"**{eligible_event_count:,} eligible Taxi Zone × date × daypart "
        "observations**. Each event is classified as congestion-only, "
        "demand-only, or both."
    )
)

card1, card2, card3, card4 = st.columns(4)
card1.metric(
    "Overall incidence",
    f"{overall_incidence:.1f} per 1,000",
)
card2.metric(
    "Peak month",
    f"{peak_row['month']:%b %Y}",
    help=(
        f"{peak_row['incidence_per_1k']:.1f} stress anomalies per 1,000 "
        "eligible observations."
    ),
)
card3.metric(
    "Post-CP demand shift",
    f"{demand_post_change:+.1f}%",
    help="Relative change in demand-only incidence from Pre-CP to Post-CP.",
)
card4.metric(
    "Post-CP congestion shift",
    f"{congestion_post_change:+.1f}%",
    help="Relative change in congestion-only incidence from Pre-CP to Post-CP.",
)

st.markdown("### Monthly stress-anomaly incidence")
hero_figure = _build_timeline_figure(
    monthly_summary,
    period_column="month",
    cadence="Monthly",
    families=FAMILY_ORDER,
    measure="Incidence per 1,000",
    show_peak_annotation=True,
)
st.plotly_chart(
    hero_figure,
    width="stretch",
    config={"displayModeBar": False},
)

st.markdown(
    f"""
    <div class="soft-callout">
        <strong>The overall rate changed less than the kind of stress.</strong><br>
        Stress-anomaly incidence was {abs(overall_post_change):.1f}% higher after congestion pricing began, but that modest change masks a reversal in composition. Demand-only incidence rose {abs(demand_post_change):.1f}%, while congestion-only incidence fell {abs(congestion_post_change):.1f}%. The highest concentration occurred in {peak_row['month']:%B %Y}, when demand-only stress accounted for the largest share of events.
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "The chart describes timing and composition around the January 2025 launch; "
    "it does not by itself establish that congestion pricing caused the changes."
)


# ---------------------------------------------------------------------
# Custom temporal explorer
# ---------------------------------------------------------------------
st.divider()
st.markdown("### Explore this view")
st.write(
    "Change the stress families, measure, time resolution, date scope, or "
    "daypart. Saved views apply immediately."
)

data_start_date = event_universe["date"].min().date()
data_end_date = event_universe["date"].max().date()

default_view = SAVED_VIEWS["Full trajectory"]
st.session_state.setdefault("raw12_saved_view", "Full trajectory")
st.session_state.setdefault("raw12_families", list(default_view["families"]))
st.session_state.setdefault("raw12_measure", str(default_view["measure"]))
st.session_state.setdefault("raw12_cadence", str(default_view["cadence"]))
st.session_state.setdefault("raw12_time_scope", str(default_view["time_scope"]))
st.session_state.setdefault(
    "raw12_temporal_bucket",
    str(default_view["temporal_bucket"]),
)
st.session_state.setdefault("raw12_start_date", data_start_date)
st.session_state.setdefault("raw12_end_date", data_end_date)

# Reapply named presets once when their definitions change so a hot-reloaded
# session does not retain an older preset's narrower date window.
if (
    st.session_state.get("raw12_preset_config_version")
    != PRESET_CONFIG_VERSION
):
    if st.session_state.get("raw12_saved_view") in SAVED_VIEWS:
        _apply_saved_view()
    st.session_state[
        "raw12_preset_config_version"
    ] = PRESET_CONFIG_VERSION

st.selectbox(
    "Saved view",
    options=[*SAVED_VIEWS.keys(), "Custom"],
    key="raw12_saved_view",
    on_change=_apply_saved_view,
)

active_saved_view = str(
    st.session_state.get("raw12_saved_view", "Custom")
)
if active_saved_view in SAVED_VIEW_DESCRIPTIONS:
    st.caption(SAVED_VIEW_DESCRIPTIONS[active_saved_view])

control1, control2, control3 = st.columns([2, 1, 1])

with control1:
    selected_families = st.multiselect(
        "Stress families",
        options=FAMILY_ORDER,
        key="raw12_families",
        on_change=_mark_custom_view,
        help=(
            "The three families are mutually exclusive. Select one or more "
            "segments to include in the chart."
        ),
    )

with control2:
    selected_measure = st.selectbox(
        "Measure",
        options=MEASURE_OPTIONS,
        key="raw12_measure",
        on_change=_mark_custom_view,
    )

with control3:
    selected_cadence = st.selectbox(
        "Time resolution",
        options=CADENCE_OPTIONS,
        key="raw12_cadence",
        on_change=_mark_custom_view,
    )

control4, control5 = st.columns(2)

with control4:
    selected_time_scope = st.selectbox(
        "Time scope",
        options=TIME_SCOPE_OPTIONS,
        key="raw12_time_scope",
        on_change=_mark_custom_view,
    )

with control5:
    selected_temporal_bucket = st.selectbox(
        "Temporal bucket",
        options=[ALL_TEMPORAL_BUCKETS, *TEMPORAL_BUCKET_ORDER],
        format_func=_format_temporal_bucket,
        key="raw12_temporal_bucket",
        on_change=_mark_custom_view,
    )

if selected_time_scope == "Custom dates":
    date1, date2 = st.columns(2)
    with date1:
        selected_start_date = st.date_input(
            "Start date",
            min_value=data_start_date,
            max_value=data_end_date,
            key="raw12_start_date",
            on_change=_mark_custom_view,
        )
    with date2:
        selected_end_date = st.date_input(
            "End date",
            min_value=data_start_date,
            max_value=data_end_date,
            key="raw12_end_date",
            on_change=_mark_custom_view,
        )
elif selected_time_scope == "Pre-CP":
    selected_start_date = data_start_date
    selected_end_date = (CP_START_DATE - pd.Timedelta(days=1)).date()
elif selected_time_scope == "Post-CP":
    selected_start_date = CP_START_DATE.date()
    selected_end_date = data_end_date
else:
    selected_start_date = data_start_date
    selected_end_date = data_end_date

if not selected_families:
    st.warning("Select at least one stress family to render the explorer.")
    st.stop()

if (
    selected_measure == "Composition share"
    and len(selected_families) == 1
):
    st.info(
        "Composition share requires at least two stress families. Select another "
        "family or switch the measure to Incidence per 1,000."
    )
    st.stop()

if selected_start_date > selected_end_date:
    st.warning("The start date must be on or before the end date.")
    st.stop()

filtered_universe = event_universe[
    event_universe["date"].between(
        pd.Timestamp(selected_start_date),
        pd.Timestamp(selected_end_date),
    )
].copy()

if selected_temporal_bucket != ALL_TEMPORAL_BUCKETS:
    filtered_universe = filtered_universe[
        filtered_universe["temporal_bucket"].eq(selected_temporal_bucket)
    ].copy()

if filtered_universe.empty:
    st.info("No eligible observations match this custom view.")
    st.stop()

period_column = "month" if selected_cadence == "Monthly" else "week"
custom_summary = _build_group_summary(
    filtered_universe,
    group_column=period_column,
)

custom_selected_count = int(
    custom_summary[selected_families].sum(axis=1).sum()
)
custom_eligible_count = int(
    filtered_universe[EVENT_ID_COLUMN].nunique()
)
custom_incidence = (
    custom_selected_count
    / custom_eligible_count
    * 1_000
)

if custom_selected_count == 0:
    st.info(
        "No stress anomalies from the selected families match this custom view."
    )
    st.stop()

custom_family_totals = custom_summary[selected_families].sum(axis=0)
leading_family = str(custom_family_totals.idxmax())
leading_family_count = int(custom_family_totals.loc[leading_family])
leading_family_share = (
    leading_family_count
    / custom_selected_count
)

custom_total_by_period = (
    custom_summary[
        [f"{family}_per_1k" for family in selected_families]
    ].sum(axis=1)
)
peak_index = custom_total_by_period.idxmax()
peak_period = pd.Timestamp(custom_summary.loc[peak_index, period_column])
peak_incidence = float(custom_total_by_period.loc[peak_index])

period_changes = custom_total_by_period.diff()
largest_increase_period: pd.Timestamp | None = None
largest_increase_value = np.nan
if period_changes.notna().any():
    largest_increase_index = period_changes.idxmax()
    largest_increase_value = float(
        period_changes.loc[largest_increase_index]
    )
    largest_increase_period = pd.Timestamp(
        custom_summary.loc[
            largest_increase_index,
            period_column,
        ]
    )

period_day_counts = (
    filtered_universe.groupby(
        "policy_period",
        observed=True,
    )["date"]
    .nunique()
)
scoped_period_summary = _build_group_summary(
    filtered_universe,
    group_column="policy_period",
)

pre_post_sentence = ""
if (
    {"Pre-CP", "Post-CP"}.issubset(
        set(scoped_period_summary["policy_period"])
    )
    and int(period_day_counts.get("Pre-CP", 0)) >= 28
    and int(period_day_counts.get("Post-CP", 0)) >= 28
):
    scoped_pre = scoped_period_summary.loc[
        scoped_period_summary["policy_period"].eq("Pre-CP")
    ].iloc[0]
    scoped_post = scoped_period_summary.loc[
        scoped_period_summary["policy_period"].eq("Post-CP")
    ].iloc[0]
    scoped_pre_incidence = float(
        scoped_pre[
            [f"{family}_per_1k" for family in selected_families]
        ].sum()
    )
    scoped_post_incidence = float(
        scoped_post[
            [f"{family}_per_1k" for family in selected_families]
        ].sum()
    )
    scoped_post_change = _relative_change(
        scoped_post_incidence,
        scoped_pre_incidence,
    )

    if np.isfinite(scoped_post_change):
        direction = "higher" if scoped_post_change >= 0 else "lower"
        pre_post_sentence = (
            f" Within this scope, Post-CP incidence was "
            f"{abs(scoped_post_change):.1f}% {direction} than Pre-CP incidence."
        )

scope_start_label = pd.Timestamp(selected_start_date).strftime("%b %d, %Y")
scope_end_label = pd.Timestamp(selected_end_date).strftime("%b %d, %Y")
bucket_label = _format_temporal_bucket(selected_temporal_bucket)

largest_increase_sentence = ""
if (
    largest_increase_period is not None
    and np.isfinite(largest_increase_value)
    and largest_increase_value > 0
):
    change_label = (
        "month-over-month"
        if selected_cadence == "Monthly"
        else "week-over-week"
    )
    largest_increase_sentence = (
        f" The largest {change_label} rise led into "
        f"{_period_label(largest_increase_period, selected_cadence)}, "
        f"increasing by {largest_increase_value:.1f} per 1,000."
    )

summary1, summary2, summary3, summary4 = st.columns(4)
summary1.metric("Stress anomalies", f"{custom_selected_count:,}")
summary2.metric("Incidence", f"{custom_incidence:.1f} per 1,000")
with summary3:
    if selected_cadence == "Weekly":
        st.markdown(
            """
            <style>
            div[data-testid="stColumn"]:has(#raw12-weekly-peak-card)
            div[data-testid="stMetricValue"] {
                font-size: 1.25rem !important;
                line-height: 1.15 !important;
                white-space: normal !important;
            }
            </style>
            <span id="raw12-weekly-peak-card"></span>
            """,
            unsafe_allow_html=True,
        )

    st.metric(
        "Peak period",
        _period_label(peak_period, selected_cadence),
        help=f"{peak_incidence:.1f} selected-family events per 1,000.",
    )
summary4.metric(
    "Leading family",
    leading_family,
    help=(
        f"{leading_family_count:,} events, or "
        f"{leading_family_share:.1%} of selected-family stress anomalies."
    ),
)

bucket_context = (
    "all temporal buckets"
    if selected_temporal_bucket == ALL_TEMPORAL_BUCKETS
    else bucket_label
)
st.markdown(
    f"""
    <div class="soft-callout">
        <strong>What stands out in this view.</strong><br>
        From {scope_start_label} through {scope_end_label}, {custom_selected_count:,} selected-family stress anomalies occurred across {bucket_context}, an incidence of {custom_incidence:.1f} per 1,000 eligible observations. {leading_family} was the leading family, accounting for {leading_family_share:.1%} of the selected events. Incidence peaked in {_period_label(peak_period, selected_cadence)} at {peak_incidence:.1f} per 1,000.{largest_increase_sentence}{pre_post_sentence}
    </div>
    """,
    unsafe_allow_html=True,
)

chart_measure_label = (
    "incidence"
    if selected_measure == "Incidence per 1,000"
    else "composition"
)
st.markdown(
    f"### {selected_cadence} stress-anomaly {chart_measure_label}"
)
st.caption(
    f"{scope_start_label}–{scope_end_label} · {bucket_label} · "
    + ", ".join(selected_families)
)

custom_figure = _build_timeline_figure(
    custom_summary,
    period_column=period_column,
    cadence=selected_cadence,
    families=selected_families,
    measure=selected_measure,
    show_peak_annotation=False,
)
st.plotly_chart(
    custom_figure,
    width="stretch",
    config={"displayModeBar": False},
)

with st.expander("View underlying data", expanded=False):
    display_data = custom_summary[
        [
            period_column,
            "eligible_observations",
            *selected_families,
            *[
                f"{family}_per_1k"
                for family in selected_families
            ],
        ]
    ].copy()

    display_data[period_column] = pd.to_datetime(
        display_data[period_column]
    ).dt.strftime(
        "%b %Y"
        if selected_cadence == "Monthly"
        else "%b %d, %Y"
    )

    selected_period_totals = display_data[
        selected_families
    ].sum(axis=1)
    for family in selected_families:
        rate_column = f"{family}_per_1k"
        display_data[rate_column] = display_data[rate_column].round(2)

        if selected_measure == "Composition share":
            display_data[
                f"{family}_composition_share"
            ] = np.where(
                selected_period_totals.gt(0),
                display_data[family]
                / selected_period_totals,
                np.nan,
            )

    rename_columns = {
        period_column: (
            "Month"
            if selected_cadence == "Monthly"
            else "Week starting"
        ),
        "eligible_observations": "Eligible observations",
    }
    for family in selected_families:
        rename_columns[family] = f"{family} events"
        rename_columns[
            f"{family}_per_1k"
        ] = f"{family} per 1,000"
        rename_columns[
            f"{family}_composition_share"
        ] = f"{family} composition share"

    display_data = display_data.rename(columns=rename_columns)

    st.dataframe(
        display_data,
        width="stretch",
        hide_index=True,
        column_config={
            column: st.column_config.NumberColumn(
                column,
                format="percent",
            )
            for column in display_data.columns
            if column.endswith("composition share")
        },
    )


# ---------------------------------------------------------------------
# Reading guide
# ---------------------------------------------------------------------
st.divider()
st.markdown("### How to read this view")
st.markdown(
    """
    - **Column height:** total incidence for the displayed stress families, or 100% of their composition when Composition share is selected.
    - **Column segments:** mutually exclusive congestion-only, demand-only, and both families.
    - **Incidence:** distinct stress anomalies per 1,000 eligible Taxi Zone × date × daypart observations in the same temporal scope.
    - **Dashed line:** January 5, 2025, when congestion pricing began.
    - **Hover:** the selected family’s incidence or composition share and event count.
    """
)

with st.expander("Methodology note", expanded=False):
    st.markdown(
        """
        The numerator is the distinct selected-finalist `comparison_event_id` count. The denominator is the distinct event count from the complete 3.3.6 event universe in the same scope. No `comparison_group_support_review_flag` or `support_status` filter is applied. The production surface retains stress anomalies identified by all three anomaly-detection frameworks.
        """
    )
