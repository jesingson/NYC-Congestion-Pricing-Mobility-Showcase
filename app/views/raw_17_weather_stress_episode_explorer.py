"""Raw 17 — Weather and Stress Episodes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from app.data_access.anomalies import load_selected_anomaly_events
from app.data_access.weather_relationships import MOBILITY_PANEL_PATH, WEATHER_PANEL_PATH
from app.utils.project_branding import BRAND_COLORS, apply_branding, inject_app_css


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HANDOFF_PATH = (
    PROJECT_ROOT
    / "data/processed/3.4.2.final_tables/weather_stress_analytical_handoff.parquet"
)
MIN_ZONE_EXPOSED_CONTEXTS = 25
MIN_ZONE_EXPOSED_DATES = 10
HERO_LABEL = "Unusually cold"

CONDITIONS = {
    "Measurable precipitation": {
        "flag": "measurable_precipitation_flag",
        "metric": "precipitation",
        "definition": "Any available observation with precipitation greater than zero.",
    },
    "Heavier precipitation": {
        "flag": "heavy_precipitation_flag",
        "metric": "precipitation",
        "definition": "The highest 10% of positive precipitation observations.",
    },
    "Low visibility": {
        "flag": "low_visibility_flag",
        "metric": "visibility",
        "definition": "The lowest 10% of available visibility observations.",
    },
    "High sustained wind": {
        "flag": "high_sustained_wind_flag",
        "metric": "wind_speed",
        "definition": "The highest 10% of available sustained-wind observations.",
    },
    "Strong wind gust": {
        "flag": "strong_wind_gust_flag",
        "metric": "wind_gust",
        "definition": "The highest 10% of available wind-gust observations.",
    },
    "Unusually cold": {
        "flag": "unusually_cold_flag",
        "metric": "temperature",
        "definition": "The lowest 5% of available temperature observations.",
    },
    "Unusually hot": {
        "flag": "unusually_hot_flag",
        "metric": "temperature",
        "definition": "The highest 5% of available temperature observations.",
    },
    "Large pressure change": {
        "flag": "large_pressure_change_flag",
        "metric": "pressure_3hr_change",
        "definition": "The highest 10% of absolute three-hour pressure changes.",
    },
}


@st.cache_data(show_spinner="Loading weather and stress-anomaly contexts...")
def load_weather_stress_surface() -> pd.DataFrame:
    """Enrich the 3.4.2 handoff with weather eligibility and readable geography."""
    flag_columns = [spec["flag"] for spec in CONDITIONS.values()]
    handoff_columns = [
        "comparison_event_id", "taxi_zone_id", "date", "temporal_bucket",
        "selected_finalist_flag", "weather_available_flag",
        "policy_geography_label", *flag_columns,
    ]
    frame = pd.read_parquet(HANDOFF_PATH, columns=handoff_columns)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["selected_finalist_flag"] = (
        frame["selected_finalist_flag"].fillna(False).astype(bool)
    )
    frame = frame.loc[frame["weather_available_flag"].fillna(False).astype(bool)].copy()

    # The 1.3.1 mobility panel preserves the authoritative raw-to-canonical
    # Taxi Zone bridge as well as reader-facing zone and borough names.
    zone_bridge = pd.read_parquet(
        MOBILITY_PANEL_PATH,
        columns=["taxi_zone_id", "canonical_location_id", "zone", "borough"],
    ).drop_duplicates("taxi_zone_id")
    zone_bridge["taxi_zone_id"] = pd.to_numeric(
        zone_bridge["taxi_zone_id"], errors="coerce"
    ).astype("Int64")
    zone_bridge["canonical_location_id"] = pd.to_numeric(
        zone_bridge["canonical_location_id"], errors="coerce"
    ).astype("Int64")

    frame["taxi_zone_id"] = pd.to_numeric(
        frame["taxi_zone_id"], errors="coerce"
    ).astype("Int64")
    frame = frame.merge(zone_bridge, on="taxi_zone_id", how="left", validate="many_to_one")

    # Actual weather measures distinguish a condition that is absent from a
    # condition that cannot be evaluated because its source measure is missing.
    weather_metrics = sorted({spec["metric"] for spec in CONDITIONS.values()})
    weather = pd.read_parquet(
        WEATHER_PANEL_PATH,
        columns=["taxi_zone_id", "date", "temporal_bucket", *weather_metrics],
    ).rename(columns={"taxi_zone_id": "canonical_location_id"})
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
    weather["canonical_location_id"] = pd.to_numeric(
        weather["canonical_location_id"], errors="coerce"
    ).astype("Int64")

    row_count = len(frame)
    frame = frame.merge(
        weather,
        on=["canonical_location_id", "date", "temporal_bucket"],
        how="left",
        validate="many_to_one",
    )
    if len(frame) != row_count:
        raise ValueError("The canonical weather join changed the event-context grain.")

    frame["calendar_month"] = frame["date"].dt.month
    frame["weekday_weekend"] = np.where(
        frame["date"].dt.dayofweek.ge(5), "Weekend", "Weekday"
    )
    frame["policy_geography_label"] = frame["policy_geography_label"].fillna("Unknown")
    return frame


def condition_comparison(frame: pd.DataFrame, condition_label: str) -> dict[str, float]:
    """Return overall and like-for-like anomaly incidence for one condition."""
    spec = CONDITIONS[condition_label]
    eligible = frame[spec["metric"]].notna()
    condition_frame = frame.loc[eligible].copy()
    condition_frame["condition_present"] = (
        condition_frame[spec["flag"]].fillna(False).astype(bool)
    )

    present = condition_frame["condition_present"]
    present_rate = 100 * condition_frame.loc[present, "selected_finalist_flag"].mean()
    absent_rate = 100 * condition_frame.loc[~present, "selected_finalist_flag"].mean()

    strata = [
        "temporal_bucket", "weekday_weekend", "calendar_month",
        "policy_geography_label",
    ]
    grouped = (
        condition_frame.groupby(strata + ["condition_present"], dropna=False)
        .agg(
            contexts=("comparison_event_id", "size"),
            stress_anomalies=("selected_finalist_flag", "sum"),
        )
        .reset_index()
    )
    grouped["incidence"] = 100 * grouped["stress_anomalies"] / grouped["contexts"]
    paired = grouped.loc[grouped["condition_present"]].merge(
        grouped.loc[~grouped["condition_present"]],
        on=strata,
        how="inner",
        suffixes=("_present", "_absent"),
    )
    paired["weight"] = paired["contexts_present"] + paired["contexts_absent"]
    adjusted_present = np.average(paired["incidence_present"], weights=paired["weight"])
    adjusted_absent = np.average(paired["incidence_absent"], weights=paired["weight"])

    return {
        "eligible_contexts": len(condition_frame),
        "present_contexts": int(present.sum()),
        "present_share": 100 * present.mean(),
        "present_dates": condition_frame.loc[present, "date"].nunique(),
        "overall_present": present_rate,
        "overall_absent": absent_rate,
        "overall_difference": present_rate - absent_rate,
        "adjusted_present": adjusted_present,
        "adjusted_absent": adjusted_absent,
        "adjusted_difference": adjusted_present - adjusted_absent,
        "contributing_strata": len(paired),
    }


def daily_condition_summary(frame: pd.DataFrame, condition_label: str) -> pd.DataFrame:
    """Build a date series using only contexts with an available source measure."""
    spec = CONDITIONS[condition_label]
    daily_frame = frame.loc[frame[spec["metric"]].notna()].copy()
    daily_frame["condition_present"] = daily_frame[spec["flag"]].fillna(False).astype(bool)
    daily_frame["present_anomaly"] = (
        daily_frame["condition_present"] & daily_frame["selected_finalist_flag"]
    )
    daily = (
        daily_frame.groupby("date")
        .agg(
            eligible_contexts=("comparison_event_id", "size"),
            present_contexts=("condition_present", "sum"),
            present_anomalies=("present_anomaly", "sum"),
        )
        .reset_index()
    )
    daily["Condition coverage %"] = (
        100 * daily["present_contexts"] / daily["eligible_contexts"]
    )
    daily["Anomaly incidence when present %"] = np.where(
        daily["present_contexts"].gt(0),
        100 * daily["present_anomalies"] / daily["present_contexts"],
        np.nan,
    )
    return daily


def zone_condition_summary(frame: pd.DataFrame, condition_label: str) -> pd.DataFrame:
    """Rank supported zones across the full study for one condition."""
    spec = CONDITIONS[condition_label]
    zone_frame = frame.loc[frame[spec["metric"]].notna()].copy()
    zone_frame["condition_present"] = zone_frame[spec["flag"]].fillna(False).astype(bool)
    zone_frame["present_anomaly"] = (
        zone_frame["condition_present"] & zone_frame["selected_finalist_flag"]
    )
    zone_frame["absent_anomaly"] = (
        ~zone_frame["condition_present"] & zone_frame["selected_finalist_flag"]
    )
    zones = (
        zone_frame.groupby(["taxi_zone_id", "zone", "borough"], dropna=False)
        .agg(
            present_contexts=("condition_present", "sum"),
            present_anomalies=("present_anomaly", "sum"),
            absent_contexts=("condition_present", lambda values: (~values).sum()),
            absent_anomalies=("absent_anomaly", "sum"),
            present_dates=("date", lambda values: values[zone_frame.loc[values.index, "condition_present"]].nunique()),
        )
        .reset_index()
    )
    zones["Condition-present anomaly incidence %"] = (
        100 * zones["present_anomalies"] / zones["present_contexts"]
    )
    zones["Condition-absent anomaly incidence %"] = (
        100 * zones["absent_anomalies"] / zones["absent_contexts"]
    )
    zones["Difference pp"] = (
        zones["Condition-present anomaly incidence %"]
        - zones["Condition-absent anomaly incidence %"]
    )
    return zones.loc[
        zones["present_contexts"].ge(MIN_ZONE_EXPOSED_CONTEXTS)
        & zones["present_dates"].ge(MIN_ZONE_EXPOSED_DATES)
    ].copy()


def normalize_modality_drivers(value: object) -> list[str]:
    """Normalize comma-separated or list-like modality-driver fields."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    values = value if isinstance(value, (list, tuple, set)) else str(value).split(",")
    return [str(item).strip() for item in values if str(item).strip()]


@st.cache_data(show_spinner=False)
def modality_composition(condition_label: str) -> pd.DataFrame:
    """Compare identified modality drivers when a condition is present or absent."""
    frame = load_weather_stress_surface()
    spec = CONDITIONS[condition_label]
    condition_events = frame.loc[
        frame[spec["metric"]].notna() & frame["selected_finalist_flag"],
        ["comparison_event_id", spec["flag"]],
    ].copy()
    condition_events["Condition group"] = np.where(
        condition_events[spec["flag"]].fillna(False), "Present", "Absent"
    )

    selected_events = load_selected_anomaly_events()
    driver_column = next(
        (
            column for column in [
                "event_modality_driver_list", "stress_modality_driver_list"
            ]
            if column in selected_events.columns
        ),
        None,
    )
    if driver_column is None:
        return pd.DataFrame()

    drivers = selected_events[["comparison_event_id", driver_column]].merge(
        condition_events[["comparison_event_id", "Condition group"]],
        on="comparison_event_id",
        how="inner",
        validate="one_to_one",
    )
    drivers["Modality"] = drivers[driver_column].apply(normalize_modality_drivers)
    drivers = drivers.explode("Modality")
    drivers = drivers.loc[drivers["Modality"].isin(["Taxi", "FHVHV", "Subway", "Bus"])]
    summary = (
        drivers.groupby(["Condition group", "Modality"])
        .size()
        .reset_index(name="Driver records")
    )
    summary["Composition share %"] = (
        100
        * summary["Driver records"]
        / summary.groupby("Condition group")["Driver records"].transform("sum")
    )
    return summary


def comparison_chart(summary: dict[str, float], title: str) -> go.Figure:
    """Show incidence in comparable contexts with visible values."""
    labels = ["Weather condition present", "Similar contexts without it"]
    values = [summary["adjusted_present"], summary["adjusted_absent"]]
    figure = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=[BRAND_COLORS["dark_teal"], BRAND_COLORS["seafoam"]],
        text=[f"{value:.1f}%" for value in values],
        textposition="outside",
        cliponaxis=False,
        customdata=[[summary["contributing_strata"]]] * 2,
        hovertemplate=(
            "<b>%{x}</b><br>Stress-anomaly incidence: %{y:.1f}%<br>"
            "Comparison groups: %{customdata[0]:,.0f}<extra></extra>"
        ),
    ))
    figure.update_layout(
        title=title, xaxis_title="", yaxis_title="Stress-anomaly incidence (%)",
        showlegend=False, height=390, margin=dict(t=70, r=30, b=50, l=60),
    )
    figure.update_yaxes(ticksuffix="%", rangemode="tozero")
    return apply_branding(figure)


def stability_message(summary: dict[str, float]) -> str:
    """Explain what changes after comparing similar contexts."""
    overall = summary["overall_difference"]
    adjusted = summary["adjusted_difference"]
    if np.sign(overall) != np.sign(adjusted):
        return (
            f"The initial comparison shows a **{overall:+.1f} pp** difference, but "
            f"after comparing similar contexts it changes direction to "
            f"**{adjusted:+.1f} pp**. The initial direction is not dependable."
        )
    if abs(adjusted) < 0.5 * abs(overall):
        return (
            f"The initial difference is **{overall:+.1f} pp** and falls to "
            f"**{adjusted:+.1f} pp** after comparing similar contexts. When and "
            "where the weather occurred explains much of the initial difference."
        )
    return (
        f"The initial difference is **{overall:+.1f} pp** and remains "
        f"**{adjusted:+.1f} pp** after comparing similar contexts. The relationship "
        "is not explained by the condition simply occurring at different times or places."
    )


inject_app_css()

for required_path in [HANDOFF_PATH, MOBILITY_PANEL_PATH, WEATHER_PANEL_PATH]:
    if not required_path.exists():
        st.error(f"Required Raw 17 input not found: {required_path}")
        st.stop()

weather_stress_df = load_weather_stress_surface()

# ---------------------------------------------------------------------
# Fixed answer view
# ---------------------------------------------------------------------

st.caption("WEATHER RELATIONSHIPS")
st.title("Do recurring weather conditions coincide with more stress anomalies?")
st.write(
    "Compare the share of mobility contexts containing a stress anomaly when a "
    "weather condition is present with otherwise similar contexts where it is "
    "absent. A mobility context is one Taxi Zone on one date during one time bucket."
)

hero = condition_comparison(weather_stress_df, HERO_LABEL)

st.divider()
st.header("What does the full study show?")
st.markdown(
    f"**Stress anomalies are more common during unusually cold weather.** After "
    f"comparing contexts from the same months, day types, time buckets, and policy "
    f"geographies, anomaly incidence is **{hero['adjusted_present']:.1f}%** when "
    f"unusually cold conditions are present and **{hero['adjusted_absent']:.1f}%** "
    f"when they are absent—a **{hero['adjusted_difference']:+.1f} percentage-point "
    f"difference**."
)
st.caption(CONDITIONS[HERO_LABEL]["definition"])

hero_cards = st.columns(4)
hero_cards[0].metric("During cold weather", f"{hero['adjusted_present']:.1f}%")
hero_cards[1].metric("Similar contexts without cold", f"{hero['adjusted_absent']:.1f}%")
hero_cards[2].metric("Difference after matching", f"{hero['adjusted_difference']:+.1f} pp")
hero_cards[3].metric("Cold-weather dates", f"{hero['present_dates']:,}")

st.plotly_chart(
    comparison_chart(hero, "Stress anomalies in similar contexts with and without cold weather"),
    use_container_width=True,
)

# ---------------------------------------------------------------------
# Condition comparison
# ---------------------------------------------------------------------

st.divider()
st.header("Compare another weather condition")
st.write(
    "Choose a condition to compare anomaly incidence when that condition is present "
    "with incidence at similar times and places where the condition is absent."
)

selected_label = st.selectbox(
    "Weather condition",
    list(CONDITIONS),
    index=list(CONDITIONS).index("Heavier precipitation"),
)
selected = condition_comparison(weather_stress_df, selected_label)
st.caption(f"**Condition definition:** {CONDITIONS[selected_label]['definition']}")

selected_cards = st.columns(4)
selected_cards[0].metric("When present", f"{selected['adjusted_present']:.1f}%")
selected_cards[1].metric("Similar contexts without it", f"{selected['adjusted_absent']:.1f}%")
selected_cards[2].metric("Difference after matching", f"{selected['adjusted_difference']:+.1f} pp")
selected_cards[3].metric("Mobility surface affected", f"{selected['present_share']:.1f}%")

st.markdown(stability_message(selected))

# ---------------------------------------------------------------------
# Temporal concentration
# ---------------------------------------------------------------------

st.subheader("When was the condition most widespread?")
st.write(
    "The upper panel shows how much of the city’s eligible mobility surface met the "
    "selected condition each day. The lower panel shows anomaly incidence only within "
    "the contexts where that condition was present."
)

selected_daily = daily_condition_summary(weather_stress_df, selected_label)
available_dates = selected_daily.loc[selected_daily["present_contexts"].gt(0)].copy()
available_dates = available_dates.sort_values(
    ["Condition coverage %", "Anomaly incidence when present %"], ascending=False
)
date_options = available_dates["date"].dt.date.tolist()
date_labels = {
    row.date.date(): (
        f"{row.date:%b %d, %Y} · {row['Condition coverage %']:.1f}% affected · "
        f"{row['Anomaly incidence when present %']:.1f}% anomaly incidence"
    )
    for _, row in available_dates.iterrows()
}
selected_date = st.selectbox(
    "Weather-active date to inspect", date_options, format_func=lambda value: date_labels[value]
)
selected_date_row = available_dates.loc[
    available_dates["date"].dt.date.eq(selected_date)
].iloc[0]

timeline = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
    subplot_titles=("Share of the mobility surface affected", "Anomaly incidence during the condition"),
)
timeline.add_trace(go.Scatter(
    x=selected_daily["date"], y=selected_daily["Condition coverage %"],
    mode="lines", line={"color": BRAND_COLORS["dark_teal"]},
    name="Mobility surface affected",
    hovertemplate="<b>%{x|%b %d, %Y}</b><br>Mobility surface affected: %{y:.1f}%<extra></extra>",
), row=1, col=1)
timeline.add_trace(go.Scatter(
    x=selected_daily["date"], y=selected_daily["Anomaly incidence when present %"],
    mode="lines", line={"color": BRAND_COLORS["terracotta"]},
    name="Anomaly incidence",
    hovertemplate="<b>%{x|%b %d, %Y}</b><br>Anomaly incidence where condition is present: %{y:.1f}%<extra></extra>",
), row=2, col=1)
timeline.add_hline(
    y=selected["overall_present"], line_dash="dot", line_color=BRAND_COLORS["terracotta"],
    annotation_text="Full-period incidence during this condition", row=2, col=1,
)
for row_number in [1, 2]:
    timeline.add_vline(
        x=pd.Timestamp(selected_date), line_dash="dot",
        line_color=BRAND_COLORS["dark_teal"], row=row_number, col=1,
    )
timeline.update_layout(
    title=f"{selected_label} across the study period", height=600,
    hovermode="x unified", showlegend=False,
    margin=dict(t=90, r=30, b=50, l=65),
)
timeline.update_yaxes(ticksuffix="%", rangemode="tozero", row=1, col=1)
timeline.update_yaxes(ticksuffix="%", rangemode="tozero", row=2, col=1)
st.plotly_chart(apply_branding(timeline), use_container_width=True)

date_cards = st.columns(4)
date_cards[0].metric("Mobility surface affected", f"{selected_date_row['Condition coverage %']:.1f}%")
date_cards[1].metric("Affected contexts", f"{int(selected_date_row['present_contexts']):,}")
date_cards[2].metric(
    "Anomalies in affected contexts", f"{int(selected_date_row['present_anomalies']):,}"
)
date_cards[3].metric(
    "Anomaly incidence", f"{selected_date_row['Anomaly incidence when present %']:.1f}%"
)
date_difference = (
    selected_date_row["Anomaly incidence when present %"] - selected["overall_present"]
)
st.markdown(
    f"On **{selected_date:%b %d, %Y}**, **{selected_date_row['Condition coverage %']:.1f}%** "
    f"of contexts with an available weather measurement met the "
    f"{selected_label.lower()} definition. Among those "
    f"contexts, **{selected_date_row['Anomaly incidence when present %']:.1f}%** contained "
    f"a stress anomaly, **{date_difference:+.1f} pp** from the full-period incidence "
    f"observed whenever this condition was present."
)

# ---------------------------------------------------------------------
# Spatial concentration
# ---------------------------------------------------------------------

st.subheader("Where did this condition coincide with the largest anomaly increases?")
st.write(
    "For each Taxi Zone, compare anomaly incidence during the selected weather "
    "condition with that zone’s incidence when the condition was absent. Only zones "
    "with at least 25 affected contexts across 10 dates are shown."
)

zone_summary_df = zone_condition_summary(weather_stress_df, selected_label)
zone_ranking = zone_summary_df.sort_values("Difference pp", ascending=False).head(15)
zone_figure = go.Figure(go.Bar(
    x=zone_ranking["Difference pp"],
    y=zone_ranking["zone"],
    orientation="h",
    marker_color=np.where(
        zone_ranking["Difference pp"].ge(0),
        BRAND_COLORS["dark_teal"], BRAND_COLORS["terracotta"],
    ),
    text=[f"{value:+.1f} pp" for value in zone_ranking["Difference pp"]],
    textposition="outside",
    cliponaxis=False,
    customdata=zone_ranking[[
        "borough", "present_contexts", "present_dates",
        "Condition-present anomaly incidence %", "Condition-absent anomaly incidence %",
    ]],
    hovertemplate=(
        "<b>%{y}</b><br>Borough: %{customdata[0]}<br>Difference: %{x:+.1f} pp<br>"
        "Incidence during condition: %{customdata[3]:.1f}%<br>"
        "Incidence without condition: %{customdata[4]:.1f}%<br>"
        "Affected contexts: %{customdata[1]:,.0f}<br>"
        "Affected dates: %{customdata[2]:,.0f}<extra></extra>"
    ),
))
zone_figure.update_layout(
    title=f"Where {selected_label.lower()} coincided with the largest anomaly increases",
    xaxis_title="Anomaly-incidence difference (percentage points)",
    yaxis_title="", height=620, margin=dict(t=80, r=80, b=60, l=180),
)
zone_figure.update_yaxes(autorange="reversed")
zone_figure.add_vline(x=0, line_dash="dash", line_color="#66747A")
st.plotly_chart(apply_branding(zone_figure), use_container_width=True)

if not zone_ranking.empty:
    leading_zone = zone_ranking.iloc[0]
    st.markdown(
        f"**{leading_zone['zone']}** shows the largest supported difference: anomaly "
        f"incidence was **{leading_zone['Condition-present anomaly incidence %']:.1f}%** "
        f"during {selected_label.lower()} and "
        f"**{leading_zone['Condition-absent anomaly incidence %']:.1f}%** when the "
        f"condition was absent—a **{leading_zone['Difference pp']:+.1f} pp** gap "
        f"across **{int(leading_zone['present_dates'])} affected dates**."
    )

zone_table = zone_ranking.rename(columns={
    "zone": "Taxi Zone",
    "borough": "Borough",
    "present_contexts": "Affected contexts",
    "present_dates": "Affected dates",
    "present_anomalies": "Stress anomalies",
    "Condition-present anomaly incidence %": "Incidence during condition %",
    "Condition-absent anomaly incidence %": "Incidence without condition %",
})[[
    "Taxi Zone", "Borough", "Affected contexts", "Affected dates",
    "Stress anomalies", "Incidence during condition %",
    "Incidence without condition %", "Difference pp",
]].copy()
numeric_columns = zone_table.select_dtypes(include="number").columns
zone_table[numeric_columns] = zone_table[numeric_columns].round(1)
st.dataframe(zone_table, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# Mobility-system composition
# ---------------------------------------------------------------------

st.subheader("Which mobility modes drove these anomalies?")
st.write(
    "See whether Taxi, FHVHV, Subway, or Bus accounted for a different share of "
    "identified anomaly drivers during the selected weather condition."
)

modality_df = modality_composition(selected_label)
if modality_df.empty:
    st.info("Modality-driver detail is unavailable in the local anomaly export.")
else:
    modalities = ["Taxi", "FHVHV", "Subway", "Bus"]
    modality_figure = go.Figure()
    for group_label, color in [
        ("Present", BRAND_COLORS["dark_teal"]),
        ("Absent", BRAND_COLORS["seafoam"]),
    ]:
        group = (
            modality_df.loc[modality_df["Condition group"].eq(group_label)]
            .set_index("Modality")
            .reindex(modalities)
            .fillna(0)
            .reset_index()
        )
        modality_figure.add_trace(go.Bar(
            x=group["Modality"], y=group["Composition share %"],
            name=("During condition" if group_label == "Present" else "Without condition"),
            marker_color=color,
            text=[f"{value:.1f}%" for value in group["Composition share %"]],
            textposition="outside", cliponaxis=False,
            customdata=group[["Driver records"]],
            hovertemplate=(
                "<b>%{x}</b><br>Weather: " + group_label
                + "<br>Share of identified drivers: %{y:.1f}%"
                + "<br>Driver records: %{customdata[0]:,.0f}<extra></extra>"
            ),
        ))
    modality_figure.update_layout(
        title=f"Which modes accounted for anomalies during {selected_label.lower()}?",
        xaxis_title="", yaxis_title="Share of identified modality drivers (%)",
        barmode="group", height=440, margin=dict(t=80, r=30, b=50, l=70),
    )
    modality_figure.update_yaxes(ticksuffix="%", rangemode="tozero")
    st.plotly_chart(apply_branding(modality_figure), use_container_width=True)

    driver_pivot = modality_df.pivot(
        index="Modality", columns="Condition group", values="Composition share %"
    ).fillna(0)
    driver_pivot["Shift pp"] = driver_pivot.get("Present", 0) - driver_pivot.get("Absent", 0)
    leading_driver = driver_pivot["Present"].idxmax()
    largest_shift_driver = driver_pivot["Shift pp"].abs().idxmax()
    largest_shift = driver_pivot.loc[largest_shift_driver, "Shift pp"]
    st.markdown(
        f"**{leading_driver}** was the largest identified driver during "
        f"{selected_label.lower()}, accounting for "
        f"**{driver_pivot.loc[leading_driver, 'Present']:.1f}%** of driver records. "
        f"The largest change in composition was **{largest_shift_driver}** at "
        f"**{largest_shift:+.1f} percentage points** compared with anomalies observed "
        f"without the condition."
    )

with st.expander("How to read this page", expanded=False):
    st.markdown(
        "- **Condition present** means the underlying weather measure is available "
        "and meets the displayed condition definition. **Condition absent** means "
        "the measure is available but does not meet that definition.\n"
        "- The headline comparison contrasts contexts from similar months, day types, "
        "times of day, and parts of the congestion-pricing geography.\n"
        "- The date view describes condition-active contexts; it does not claim that "
        "weather caused the anomalies observed on a particular date.\n"
        "- Taxi Zone estimates use weather assigned from a small station network and "
        "should be interpreted as broad spatial patterns rather than block-level weather."
    )

st.caption(
    "Weather relationships are descriptive. They show co-occurrence after accounting "
    "for broad timing and geography differences, not proof that weather caused an anomaly."
)
