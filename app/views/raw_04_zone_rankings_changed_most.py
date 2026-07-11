from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.data_access.loaders import CORE_METRICS
from app.data_access.spatial_aggregations import (
    ALL_TEMPORAL_BUCKETS_LABEL,
    get_zone_pre_post_metric_summary,
)
from app.data_access.spatial_visuals import add_reliability_flags
from app.data_access.zone_rankings import (
    METRIC_COLOR_MAP,
    prepare_zone_metric_ranking_base,
)
from app.utils.project_branding import (
    BRAND_COLORS,
    apply_branding,
    inject_app_css,
)


inject_app_css()

INCREASE_COLOR = BRAND_COLORS["dark_teal"]
DECREASE_COLOR = BRAND_COLORS["terracotta"]

METRIC_DISPLAY_ORDER = [
    "taxi_trip_count",
    "taxi_avg_trip_speed",
    "fhvhv_trip_count",
    "fhvhv_avg_trip_speed",
    "subway_ridership",
    "avg_bus_speed",
]

DEMAND_METRICS = {
    "taxi_trip_count",
    "fhvhv_trip_count",
    "subway_ridership",
}

SPEED_METRICS = {
    "taxi_avg_trip_speed",
    "fhvhv_avg_trip_speed",
    "avg_bus_speed",
}

WEEKDAY_BUCKETS = [
    "weekday_overnight",
    "weekday_am_peak",
    "weekday_midday",
    "weekday_pm_peak",
    "weekday_evening",
]

WEEKEND_BUCKETS = [
    "weekend_overnight",
    "weekend_am_peak",
    "weekend_midday",
    "weekend_pm_peak",
    "weekend_evening",
]

TEMPORAL_BUCKET_OPTIONS = [
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

TEMPORAL_BUCKET_LABELS = {
    "weekday_overnight": "Weekday · Overnight",
    "weekday_am_peak": "Weekday · AM peak",
    "weekday_midday": "Weekday · Midday",
    "weekday_pm_peak": "Weekday · PM peak",
    "weekday_evening": "Weekday · Evening",
    "weekend_overnight": "Weekend · Overnight",
    "weekend_am_peak": "Weekend · AM peak",
    "weekend_midday": "Weekend · Midday",
    "weekend_pm_peak": "Weekend · PM peak",
    "weekend_evening": "Weekend · Evening",
}

GEO_POLICY_LABELS = {
    "CBD": "CBD",
    "Gateway": "Gateway",
    "Adjacent": "Adjacent to CBD",
    "Non-CBD": "Non-CBD",
    "Non_CBD": "Non-CBD",
    "Non CBD": "Non-CBD",
    "non_cbd": "Non-CBD",
    "cbd": "CBD",
    "gateway": "Gateway",
    "adjacent": "Adjacent to CBD",
}


# ---------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------
def _format_signed_percent(value: float) -> str:
    if pd.isna(value):
        return "Unavailable"
    return f"{value:+,.1f}%"


def _format_average(value: float) -> str:
    if pd.isna(value):
        return "Unavailable"
    return f"{value:,.1f}"


def _format_signed_average(value: float) -> str:
    if pd.isna(value):
        return "Unavailable"
    return f"{value:+,.1f}"


def _format_geo_policy(value: object) -> str:
    text = str(value).strip()

    if text in GEO_POLICY_LABELS:
        return GEO_POLICY_LABELS[text]

    return text.replace("_", " ").title()


def _wrap_zone_label(
    value: str,
    *,
    max_line_length: int = 24,
) -> str:
    text = str(value).strip()

    if len(text) <= max_line_length:
        return text

    words = text.split()

    if len(words) <= 1:
        return text

    first_line: list[str] = []
    second_line: list[str] = []

    for word in words:
        candidate = " ".join(first_line + [word])

        if len(candidate) <= max_line_length or not first_line:
            first_line.append(word)
        else:
            second_line.append(word)

    if not second_line:
        return text

    return (
        " ".join(first_line)
        + "<br>"
        + " ".join(second_line)
    )


def _safe_percent_change(
    pre_value: float,
    post_value: float,
) -> float:
    if (
        pd.isna(pre_value)
        or pd.isna(post_value)
        or pre_value == 0
    ):
        return np.nan

    return (
        (post_value - pre_value)
        / pre_value
        * 100
    )


def _clean_ranking_base(
    ranking_base: pd.DataFrame,
) -> pd.DataFrame:
    if ranking_base.empty:
        return ranking_base

    result = ranking_base.copy()

    zone_text = (
        result["zone"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    borough_text = (
        result["borough"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    invalid_values = {
        "",
        "unknown",
        "nan",
        "none",
    }

    result = result[
        ~zone_text.isin(invalid_values)
        & ~borough_text.isin(invalid_values)
    ].copy()

    return result.reset_index(drop=True)


def _add_display_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    result["display_percent_change"] = (
        result["percent_change"].map(_format_signed_percent)
    )
    result["display_pre_average"] = (
        result["pre_daily_average"].map(_format_average)
    )
    result["display_post_average"] = (
        result["post_daily_average"].map(_format_average)
    )
    result["display_absolute_change"] = (
        result["absolute_change"].map(_format_signed_average)
    )
    result["wrapped_zone"] = (
        result["zone"].map(_wrap_zone_label)
    )

    return result


def _hover_template() -> str:
    return (
        "<b>%{customdata[0]}</b><br>"
        "Borough: %{customdata[1]}<br>"
        "Geo-policy group: %{customdata[2]}<br>"
        "Pre-CP daily average: %{customdata[3]}<br>"
        "Post-CP daily average: %{customdata[4]}<br>"
        "Daily-average change: %{customdata[5]}<br>"
        "Percent change: %{customdata[6]}"
        "<extra></extra>"
    )


def _metric_customdata(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    if "cbd_spatial_category" not in result.columns:
        result["cbd_spatial_category"] = "Unavailable"

    result["display_geo_policy"] = (
        result["cbd_spatial_category"]
        .fillna("Unavailable")
        .map(_format_geo_policy)
    )

    return result[
        [
            "zone",
            "borough",
            "display_geo_policy",
            "display_pre_average",
            "display_post_average",
            "display_absolute_change",
            "display_percent_change",
        ]
    ]


def _apply_chart_branding(
    fig: go.Figure,
) -> go.Figure:
    fig.update_layout(
        title={
            "text": " ",
            "x": 0,
            "y": 1,
        }
    )

    fig = apply_branding(fig)

    fig.update_layout(
        title={
            "text": "",
            "x": 0,
            "y": 1,
            "pad": {
                "t": 0,
                "b": 0,
                "l": 0,
                "r": 0,
            },
        }
    )

    clean_annotations = []

    for annotation in fig.layout.annotations or []:
        annotation_text = str(
            getattr(annotation, "text", "")
        ).strip()

        if annotation_text.lower() in {
            "",
            "none",
            "undefined",
            "nan",
        }:
            continue

        clean_annotations.append(annotation)

    fig.update_layout(
        annotations=clean_annotations,
    )

    return fig


# ---------------------------------------------------------------------
# Temporal rollups
# ---------------------------------------------------------------------
def _weighted_average(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    valid = values.notna() & weights.notna() & weights.gt(0)

    if valid.any():
        return float(
            np.average(
                values.loc[valid],
                weights=weights.loc[valid],
            )
        )

    valid_values = values.dropna()

    if valid_values.empty:
        return np.nan

    return float(valid_values.mean())


def _roll_up_metric_period(
    metric_df: pd.DataFrame,
    *,
    metric: str,
    period: str,
) -> tuple[float, float]:
    value_column = f"{period}_daily_average"
    support_column = f"{period}_support_daily_average"

    if metric in DEMAND_METRICS:
        value = metric_df[value_column].sum(
            min_count=1
        )
    elif metric in SPEED_METRICS:
        value = _weighted_average(
            metric_df[value_column],
            metric_df[support_column],
        )
    else:
        value = metric_df[value_column].mean()

    support = metric_df[support_column].sum(
        min_count=1
    )

    return value, support


def _build_temporal_rollup(
    *,
    metric: str,
    temporal_buckets: list[str],
    temporal_label: str,
) -> pd.DataFrame:
    frames = [
        get_zone_pre_post_metric_summary(
            metrics=[metric],
            temporal_bucket=bucket,
        )
        for bucket in temporal_buckets
    ]

    frames = [
        frame
        for frame in frames
        if not frame.empty
    ]

    if not frames:
        return pd.DataFrame()

    source = pd.concat(
        frames,
        ignore_index=True,
    )

    metadata_columns = [
        "taxi_zone_id",
        "zone",
        "borough",
        "cbd_spatial_category",
        "metric",
        "metric_label",
        "aggregation",
        "support_metric",
    ]

    rows: list[dict[str, object]] = []

    for _, zone_df in source.groupby(
        [
            "taxi_zone_id",
            "metric",
        ],
        observed=True,
        dropna=False,
        sort=False,
    ):
        first = zone_df.iloc[0]

        pre_average, pre_support = _roll_up_metric_period(
            zone_df,
            metric=metric,
            period="pre",
        )

        post_average, post_support = _roll_up_metric_period(
            zone_df,
            metric=metric,
            period="post",
        )

        absolute_change = (
            post_average - pre_average
            if pd.notna(pre_average)
            and pd.notna(post_average)
            else np.nan
        )

        percent_change = _safe_percent_change(
            pre_average,
            post_average,
        )

        row = {
            column: first.get(column, np.nan)
            for column in metadata_columns
        }

        row.update(
            {
                "temporal_bucket": temporal_label,
                "pre_support_daily_average": pre_support,
                "post_support_daily_average": post_support,
                "pre_daily_average": pre_average,
                "post_daily_average": post_average,
                "absolute_change": absolute_change,
                "percent_change": percent_change,
                "pre_observed_days": int(
                    zone_df["pre_observed_days"].max()
                ),
                "post_observed_days": int(
                    zone_df["post_observed_days"].max()
                ),
                "direction": (
                    "Increase"
                    if absolute_change > 0
                    else "Decrease"
                    if absolute_change < 0
                    else "No change"
                    if pd.notna(absolute_change)
                    else "Unknown"
                ),
                "has_both_periods": (
                    pd.notna(pre_average)
                    and pd.notna(post_average)
                    and zone_df["pre_observed_days"].max() > 0
                    and zone_df["post_observed_days"].max() > 0
                ),
            }
        )

        rows.append(row)

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    result = add_reliability_flags(result)

    result = result[
        result["has_both_periods"]
        & result["percent_change"].notna()
        & result["eligible_for_percent_change"]
    ].copy()

    return result.reset_index(drop=True)


def _get_explorer_base(
    *,
    metric: str,
    time_context: str,
    temporal_bucket: str | None,
) -> pd.DataFrame:
    if time_context == "Overall":
        result = prepare_zone_metric_ranking_base(
            metrics=[metric],
            temporal_bucket=ALL_TEMPORAL_BUCKETS_LABEL,
            ranking_value_column="percent_change",
            apply_reliability_thresholds=True,
        )
    elif time_context == "Weekday":
        result = _build_temporal_rollup(
            metric=metric,
            temporal_buckets=WEEKDAY_BUCKETS,
            temporal_label="Weekday",
        )
    elif time_context == "Weekend":
        result = _build_temporal_rollup(
            metric=metric,
            temporal_buckets=WEEKEND_BUCKETS,
            temporal_label="Weekend",
        )
    elif time_context == "Specific time bucket":
        result = prepare_zone_metric_ranking_base(
            metrics=[metric],
            temporal_bucket=temporal_bucket,
            ranking_value_column="percent_change",
            apply_reliability_thresholds=True,
        )
    else:
        raise ValueError(
            f"Unsupported time context: {time_context}"
        )

    return _clean_ranking_base(result)


# ---------------------------------------------------------------------
# Geography filters
# ---------------------------------------------------------------------
def _get_geography_values(
    ranking_base: pd.DataFrame,
    *,
    geography_scope: str,
) -> list[str]:
    if geography_scope == "Borough":
        return sorted(
            ranking_base["borough"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    if geography_scope == "Geo-policy group":
        return sorted(
            ranking_base["cbd_spatial_category"]
            .dropna()
            .astype(str)
            .unique()
            .tolist(),
            key=_format_geo_policy,
        )

    return []


def _filter_geography(
    ranking_base: pd.DataFrame,
    *,
    geography_scope: str,
    geography_value: str | None,
) -> pd.DataFrame:
    if geography_scope == "Citywide":
        return ranking_base.copy()

    if geography_scope == "Borough":
        return ranking_base[
            ranking_base["borough"].astype(str).eq(
                str(geography_value)
            )
        ].copy()

    if geography_scope == "Geo-policy group":
        return ranking_base[
            ranking_base[
                "cbd_spatial_category"
            ].astype(str).eq(
                str(geography_value)
            )
        ].copy()

    raise ValueError(
        f"Unsupported geography scope: {geography_scope}"
    )


def _build_context_label(
    *,
    geography_scope: str,
    geography_value: str | None,
    time_context: str,
    temporal_bucket: str | None,
) -> str:
    if geography_scope == "Citywide":
        geography_label = "Citywide"
    elif geography_scope == "Borough":
        geography_label = str(geography_value)
    else:
        geography_label = _format_geo_policy(
            geography_value
        )

    if time_context == "Specific time bucket":
        time_label = TEMPORAL_BUCKET_LABELS.get(
            temporal_bucket,
            str(temporal_bucket),
        )
    else:
        time_label = time_context

    return f"{geography_label} · {time_label}"


# ---------------------------------------------------------------------
# Interactive summary cards and insight
# ---------------------------------------------------------------------
def _build_explorer_summary(
    explorer_base: pd.DataFrame,
) -> dict[str, object]:
    """Summarize the currently filtered explorer population."""
    valid = explorer_base[
        explorer_base["percent_change"].notna()
    ].copy()

    if valid.empty:
        return {
            "eligible_zones": 0,
            "increase_count": 0,
            "decrease_count": 0,
            "median_change": np.nan,
            "largest_increase": None,
            "largest_decrease": None,
        }

    increases = valid[
        valid["percent_change"] > 0
    ].copy()

    decreases = valid[
        valid["percent_change"] < 0
    ].copy()

    largest_increase = (
        increases.loc[
            increases["percent_change"].idxmax()
        ]
        if not increases.empty
        else None
    )

    largest_decrease = (
        decreases.loc[
            decreases["percent_change"].idxmin()
        ]
        if not decreases.empty
        else None
    )

    return {
        "eligible_zones": int(
            valid["taxi_zone_id"].nunique()
        ),
        "increase_count": int(
            valid.loc[
                valid["percent_change"] > 0,
                "taxi_zone_id",
            ].nunique()
        ),
        "decrease_count": int(
            valid.loc[
                valid["percent_change"] < 0,
                "taxi_zone_id",
            ].nunique()
        ),
        "median_change": float(
            valid["percent_change"].median()
        ),
        "largest_increase": largest_increase,
        "largest_decrease": largest_decrease,
    }


def _render_explorer_summary_cards(
    summary: dict[str, object],
) -> None:
    """Render four compact cards for the active explorer filters."""
    largest_increase = summary["largest_increase"]
    largest_decrease = summary["largest_decrease"]

    increase_value = (
        f"{largest_increase['percent_change']:+,.1f}%"
        if largest_increase is not None
        else "None"
    )

    increase_zone = (
        str(largest_increase["zone"])
        if largest_increase is not None
        else "No eligible increases"
    )

    decrease_value = (
        f"{largest_decrease['percent_change']:+,.1f}%"
        if largest_decrease is not None
        else "None"
    )

    decrease_zone = (
        str(largest_decrease["zone"])
        if largest_decrease is not None
        else "No eligible decreases"
    )

    median_change = summary["median_change"]

    median_value = (
        f"{median_change:+,.1f}%"
        if pd.notna(median_change)
        else "Unavailable"
    )

    card1, card2, card3, card4 = st.columns(4)

    card1.metric(
        "Eligible zones",
        f"{summary['eligible_zones']:,}",
        help=(
            "Zones with sufficient pre- and post-period support "
            "for a reliable percent-change comparison."
        ),
    )

    card2.metric(
        "Largest increase",
        increase_value,
        increase_zone,
        delta_color="normal",
    )

    card3.metric(
        "Largest decrease",
        decrease_value,
        decrease_zone,
        delta_color="inverse",
    )

    card4.metric(
        "Median zone change",
        median_value,
        help=(
            "The median percent change across all eligible zones "
            "within the selected filters."
        ),
    )


def _build_explorer_insight(
    explorer_base: pd.DataFrame,
    *,
    metric_label: str,
    context_label: str,
) -> str:
    """Build a dynamic takeaway for the current explorer selection."""
    valid = explorer_base[
        explorer_base["percent_change"].notna()
    ].copy()

    if valid.empty:
        return (
            "No zones met the reliability requirements for this "
            "combination of filters."
        )

    increases = valid[
        valid["percent_change"] > 0
    ].copy()

    decreases = valid[
        valid["percent_change"] < 0
    ].copy()

    median_change = valid["percent_change"].median()

    increase_count = int(
        increases["taxi_zone_id"].nunique()
    )

    decrease_count = int(
        decreases["taxi_zone_id"].nunique()
    )

    eligible_count = int(
        valid["taxi_zone_id"].nunique()
    )

    sentences: list[str] = []

    if not increases.empty:
        strongest_increase = increases.loc[
            increases["percent_change"].idxmax()
        ]

        sentences.append(
            f"For **{metric_label}** in **{context_label}**, "
            f"the largest reliable increase was in "
            f"**{strongest_increase['zone']}** at "
            f"**{strongest_increase['percent_change']:+,.1f}%**."
        )

    if not decreases.empty:
        strongest_decrease = decreases.loc[
            decreases["percent_change"].idxmin()
        ]

        sentences.append(
            f"The largest decline was in "
            f"**{strongest_decrease['zone']}** at "
            f"**{strongest_decrease['percent_change']:+,.1f}%**."
        )

    if increase_count > decrease_count:
        direction_sentence = (
            f"Increases were more widespread: "
            f"**{increase_count} of {eligible_count} eligible zones** "
            f"increased, compared with **{decrease_count}** that declined."
        )
    elif decrease_count > increase_count:
        direction_sentence = (
            f"Declines were more widespread: "
            f"**{decrease_count} of {eligible_count} eligible zones** "
            f"declined, compared with **{increase_count}** that increased."
        )
    else:
        direction_sentence = (
            f"The pattern was evenly divided, with "
            f"**{increase_count} zones increasing** and "
            f"**{decrease_count} declining**."
        )

    sentences.append(direction_sentence)

    if pd.notna(median_change):
        if median_change > 1:
            median_sentence = (
                f"The typical eligible zone moved upward, with a "
                f"median change of **{median_change:+,.1f}%**."
            )
        elif median_change < -1:
            median_sentence = (
                f"The typical eligible zone moved downward, with a "
                f"median change of **{median_change:+,.1f}%**."
            )
        else:
            median_sentence = (
                f"Despite the extremes, the median eligible zone was "
                f"nearly unchanged at **{median_change:+,.1f}%**."
            )

        sentences.append(median_sentence)

    return " ".join(sentences)

# ---------------------------------------------------------------------
# Leaderboards
# ---------------------------------------------------------------------
def _get_metric_leaders(
    ranking_base: pd.DataFrame,
    *,
    metric: str,
    n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_df = ranking_base[
        ranking_base["metric"].eq(metric)
    ].copy()

    increases = (
        metric_df[metric_df["percent_change"] > 0]
        .nlargest(n, "percent_change")
        .sort_values(
            "percent_change",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    decreases = (
        metric_df[metric_df["percent_change"] < 0]
        .nsmallest(n, "percent_change")
        .sort_values(
            "percent_change",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return (
        _add_display_columns(increases),
        _add_display_columns(decreases),
    )


# ---------------------------------------------------------------------
# Editorial insight
# ---------------------------------------------------------------------
def _build_editorial_insight(
    ranking_base: pd.DataFrame,
    metric_labels: dict[str, str],
) -> str:
    valid = ranking_base[
        ranking_base["percent_change"].notna()
    ].copy()

    if valid.empty:
        return (
            "The largest reliable changes varied substantially across "
            "mobility measures and locations."
        )

    strongest_increase = valid.loc[
        valid["percent_change"].idxmax()
    ]

    strongest_decrease = valid.loc[
        valid["percent_change"].idxmin()
    ]

    demand = valid[
        valid["metric"].isin(DEMAND_METRICS)
    ].copy()

    speed = valid[
        valid["metric"].isin(SPEED_METRICS)
    ].copy()

    demand_extreme = (
        demand["percent_change"].abs().quantile(0.95)
        if not demand.empty
        else float("nan")
    )

    speed_extreme = (
        speed["percent_change"].abs().quantile(0.95)
        if not speed.empty
        else float("nan")
    )

    increase_metric_label = metric_labels.get(
        strongest_increase["metric"],
        strongest_increase["metric_label"],
    )

    decrease_metric_label = metric_labels.get(
        strongest_decrease["metric"],
        strongest_decrease["metric_label"],
    )

    first_sentence = (
        f"The sharpest reliable increase appeared in "
        f"**{strongest_increase['zone']}**, where "
        f"**{increase_metric_label} rose "
        f"{strongest_increase['percent_change']:+,.1f}%**. "
        f"The steepest decline was in "
        f"**{strongest_decrease['zone']}**, where "
        f"**{decrease_metric_label} changed "
        f"{strongest_decrease['percent_change']:+,.1f}%**."
    )

    if (
        pd.notna(demand_extreme)
        and pd.notna(speed_extreme)
        and demand_extreme > speed_extreme * 1.5
    ):
        second_sentence = (
            "Across the system, the most dramatic percentage swings were "
            "concentrated in trip and ridership measures, while speed changes "
            "were generally more restrained."
        )
    elif (
        pd.notna(demand_extreme)
        and pd.notna(speed_extreme)
        and speed_extreme > demand_extreme * 1.5
    ):
        second_sentence = (
            "Across the system, the most dramatic percentage swings appeared "
            "in speed measures rather than trip and ridership volumes."
        )
    else:
        second_sentence = (
            "The scale of change differed meaningfully by measure, reinforcing "
            "the value of comparing each mobility system on its own terms."
        )

    return f"{first_sentence} {second_sentence}"


# ---------------------------------------------------------------------
# Static hero
# ---------------------------------------------------------------------
def build_metric_card_chart(
    ranking_base: pd.DataFrame,
    *,
    metric: str,
    n_per_direction: int = 5,
) -> go.Figure:
    increases, decreases = _get_metric_leaders(
        ranking_base,
        metric=metric,
        n=n_per_direction,
    )

    combined = pd.concat(
        [
            decreases.assign(direction="Decrease"),
            increases.assign(direction="Increase"),
        ],
        ignore_index=True,
    ).sort_values(
        "percent_change",
        ascending=True,
    )

    metric_color = METRIC_COLOR_MAP.get(
        metric,
        INCREASE_COLOR,
    )

    colors = [
        DECREASE_COLOR if value < 0 else metric_color
        for value in combined["percent_change"]
    ]

    fig = go.Figure(
        layout={
            "title": {
                "text": " ",
            }
        }
    )

    fig.add_trace(
        go.Bar(
            x=combined["percent_change"],
            y=combined["wrapped_zone"],
            orientation="h",
            marker={
                "color": colors,
                "line": {
                    "width": 0,
                },
            },
            text=combined["display_percent_change"],
            textposition="outside",
            textfont={
                "size": 10,
            },
            cliponaxis=False,
            customdata=_metric_customdata(combined),
            hovertemplate=_hover_template(),
            showlegend=False,
        )
    )

    fig.add_vline(
        x=0,
        line_width=1,
        line_color="rgba(60,60,60,0.45)",
    )

    fig.update_xaxes(
        title_text="Percent change",
        ticksuffix="%",
        zeroline=False,
        showgrid=False,
        automargin=True,
        tickfont={
            "size": 10,
        },
        title_font={
            "size": 11,
        },
    )

    fig.update_yaxes(
        title_text="",
        automargin=True,
        tickfont={
            "size": 10,
        },
        showgrid=True,
        gridcolor="rgba(0,109,119,0.12)",
    )

    fig = _apply_chart_branding(fig)

    fig.update_layout(
        height=430,
        margin={
            "l": 8,
            "r": 65,
            "t": 0,
            "b": 50,
        },
        bargap=0.28,
        showlegend=False,
    )

    return fig


# ---------------------------------------------------------------------
# Interactive lollipop chart
# ---------------------------------------------------------------------
def _add_lollipop_trace(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    row: int,
    col: int,
    color: str,
    is_decrease: bool,
) -> None:
    for _, record in df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[
                    0,
                    record["percent_change"],
                ],
                y=[
                    record["wrapped_zone"],
                    record["wrapped_zone"],
                ],
                mode="lines",
                line={
                    "color": color,
                    "width": 3,
                },
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.add_trace(
        go.Scatter(
            x=df["percent_change"],
            y=df["wrapped_zone"],
            mode="markers+text",
            marker={
                "color": color,
                "size": 11,
                "line": {
                    "color": "white",
                    "width": 1,
                },
            },
            text=df["display_percent_change"],
            textposition=(
                "middle left"
                if is_decrease
                else "middle right"
            ),
            textfont={
                "size": 10,
            },
            customdata=_metric_customdata(df),
            hovertemplate=_hover_template(),
            showlegend=False,
            cliponaxis=False,
        ),
        row=row,
        col=col,
    )


def _space_subplot_titles(
    fig: go.Figure,
) -> go.Figure:
    fig.update_yaxes(
        domain=[0.0, 0.84],
        row=1,
        col=1,
    )

    fig.update_yaxes(
        domain=[0.0, 0.84],
        row=1,
        col=2,
    )

    for annotation in fig.layout.annotations or []:
        annotation_text = str(
            getattr(annotation, "text", "")
        ).strip()

        if annotation_text in {
            "Largest decreases",
            "Largest increases",
        }:
            annotation.update(
                y=0.98,
                yanchor="bottom",
                font={
                    "size": 15,
                },
            )

    return fig


def build_split_lollipop_leaderboard(
    increases: pd.DataFrame,
    decreases: pd.DataFrame,
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Largest decreases",
            "Largest increases",
        ),
        horizontal_spacing=0.18,
    )

    fig.update_layout(
        title={
            "text": " ",
        }
    )

    _add_lollipop_trace(
        fig,
        decreases,
        row=1,
        col=1,
        color=DECREASE_COLOR,
        is_decrease=True,
    )

    _add_lollipop_trace(
        fig,
        increases,
        row=1,
        col=2,
        color=INCREASE_COLOR,
        is_decrease=False,
    )

    fig.add_vline(
        x=0,
        line_width=1,
        line_color="rgba(60,60,60,0.45)",
        row=1,
        col=1,
    )

    fig.add_vline(
        x=0,
        line_width=1,
        line_color="rgba(60,60,60,0.45)",
        row=1,
        col=2,
    )

    fig.update_xaxes(
        title_text="Percent change",
        ticksuffix="%",
        zeroline=False,
        showgrid=False,
        tickfont={
            "size": 10,
        },
        row=1,
        col=1,
    )

    fig.update_xaxes(
        title_text="Percent change",
        ticksuffix="%",
        zeroline=False,
        showgrid=False,
        tickfont={
            "size": 10,
        },
        row=1,
        col=2,
    )

    fig.update_yaxes(
        title_text="",
        automargin=True,
        tickfont={
            "size": 10,
        },
        showgrid=True,
        gridcolor="rgba(0,109,119,0.10)",
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text="",
        automargin=True,
        tickfont={
            "size": 10,
        },
        showgrid=True,
        gridcolor="rgba(0,109,119,0.10)",
        row=1,
        col=2,
    )

    fig = _apply_chart_branding(fig)
    fig = _space_subplot_titles(fig)

    fig.update_layout(
        height=660,
        margin={
            "l": 8,
            "r": 65,
            "t": 90,
            "b": 50,
        },
        showlegend=False,
    )

    return fig


# ---------------------------------------------------------------------
# Detail table
# ---------------------------------------------------------------------
def build_detail_table(
    increases: pd.DataFrame,
    decreases: pd.DataFrame,
) -> pd.DataFrame:
    decrease_table = decreases.copy()
    decrease_table["Direction"] = "Decrease"
    decrease_table = decrease_table.sort_values(
        "percent_change",
        ascending=True,
    )

    increase_table = increases.copy()
    increase_table["Direction"] = "Increase"
    increase_table = increase_table.sort_values(
        "percent_change",
        ascending=False,
    )

    decrease_table["Rank"] = range(
        1,
        len(decrease_table) + 1,
    )

    increase_table["Rank"] = range(
        1,
        len(increase_table) + 1,
    )

    combined = pd.concat(
        [
            increase_table,
            decrease_table,
        ],
        ignore_index=True,
    )

    display = combined[
        [
            "Direction",
            "Rank",
            "zone",
            "borough",
            "cbd_spatial_category",
            "pre_daily_average",
            "post_daily_average",
            "absolute_change",
            "percent_change",
        ]
    ].copy()

    display["cbd_spatial_category"] = (
        display["cbd_spatial_category"]
        .fillna("Unavailable")
        .map(_format_geo_policy)
    )

    display = display.rename(
        columns={
            "zone": "Taxi Zone",
            "borough": "Borough",
            "cbd_spatial_category": "Geo-policy group",
            "pre_daily_average": "Pre-CP daily avg",
            "post_daily_average": "Post-CP daily avg",
            "absolute_change": "Daily-average change",
            "percent_change": "Percent change",
        }
    )

    return display


# ---------------------------------------------------------------------
# Page data
# ---------------------------------------------------------------------
st.caption("ZONE RANKINGS")
st.title("Which zones changed most?")

st.write(
    "The largest reliable pre- versus post-congestion-pricing changes "
    "across New York City's mobility system."
)

with st.spinner("Preparing zone rankings..."):
    hero_ranking_base = prepare_zone_metric_ranking_base(
        metrics=CORE_METRICS,
        temporal_bucket=ALL_TEMPORAL_BUCKETS_LABEL,
        ranking_value_column="percent_change",
        apply_reliability_thresholds=True,
    )

hero_ranking_base = _clean_ranking_base(
    hero_ranking_base
)

if hero_ranking_base.empty:
    st.warning(
        "No eligible zone-level rankings were produced."
    )
    st.stop()

metric_labels = (
    hero_ranking_base[
        [
            "metric",
            "metric_label",
        ]
    ]
    .drop_duplicates()
    .set_index("metric")["metric_label"]
    .to_dict()
)

available_metrics = [
    metric
    for metric in METRIC_DISPLAY_ORDER
    if metric in metric_labels
]


# ---------------------------------------------------------------------
# Editorial hero
# ---------------------------------------------------------------------
st.header("Where the biggest changes appeared")

st.write(
    "Each panel shows the five largest reliable increases and decreases "
    "within one mobility measure."
)

for row_start in range(0, len(available_metrics), 2):
    row_metrics = available_metrics[
        row_start : row_start + 2
    ]

    columns = st.columns(
        len(row_metrics),
        gap="large",
    )

    for column, metric in zip(
        columns,
        row_metrics,
    ):
        with column:
            st.subheader(
                metric_labels[metric]
            )

            metric_fig = build_metric_card_chart(
                hero_ranking_base,
                metric=metric,
                n_per_direction=5,
            )

            st.plotly_chart(
                metric_fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
                key=f"hero_metric_{metric}",
            )

st.markdown("#### What stands out")

st.info(
    _build_editorial_insight(
        hero_ranking_base,
        metric_labels,
    )
)


# ---------------------------------------------------------------------
# Interactive explorer
# ---------------------------------------------------------------------
st.divider()
st.header("Explore the rankings")

st.write(
    "Choose a mobility measure, geographic scope, and time context to "
    "compare the largest reliable zone-level increases and decreases."
)

control1, control2, control3 = st.columns(
    [1.2, 1.0, 1.0]
)

with control1:
    selected_metric = st.selectbox(
        "Mobility measure",
        options=available_metrics,
        format_func=lambda metric: metric_labels[metric],
        index=0,
        key="raw04_metric",
    )

with control2:
    geography_scope = st.selectbox(
        "Geography",
        options=[
            "Citywide",
            "Borough",
            "Geo-policy group",
        ],
        index=0,
        key="raw04_geography_scope",
    )

with control3:
    time_context = st.selectbox(
        "Time context",
        options=[
            "Overall",
            "Weekday",
            "Weekend",
            "Specific time bucket",
        ],
        index=0,
        key="raw04_time_context",
    )

temporal_bucket: str | None = None

if time_context == "Specific time bucket":
    temporal_bucket = st.selectbox(
        "Time bucket",
        options=TEMPORAL_BUCKET_OPTIONS,
        format_func=lambda bucket: TEMPORAL_BUCKET_LABELS[
            bucket
        ],
        index=1,
        key="raw04_temporal_bucket",
    )

with st.spinner("Updating zone rankings..."):
    explorer_base = _get_explorer_base(
        metric=selected_metric,
        time_context=time_context,
        temporal_bucket=temporal_bucket,
    )

geography_value: str | None = None

if geography_scope != "Citywide":
    geography_values = _get_geography_values(
        explorer_base,
        geography_scope=geography_scope,
    )

    if geography_values:
        geography_value = st.selectbox(
            (
                "Borough"
                if geography_scope == "Borough"
                else "Geo-policy group"
            ),
            options=geography_values,
            format_func=(
                (lambda value: value)
                if geography_scope == "Borough"
                else _format_geo_policy
            ),
            key="raw04_geography_value",
        )
    else:
        st.warning(
            f"No {geography_scope.lower()} values are available "
            "for the selected measure and time context."
        )

explorer_base = _filter_geography(
    explorer_base,
    geography_scope=geography_scope,
    geography_value=geography_value,
)

context_label = _build_context_label(
    geography_scope=geography_scope,
    geography_value=geography_value,
    time_context=time_context,
    temporal_bucket=temporal_bucket,
)

st.caption(
    f"{metric_labels[selected_metric]} · "
    f"{context_label} · Percent change · "
    f"Reliability thresholds on"
)

explorer_summary = _build_explorer_summary(
    explorer_base
)

_render_explorer_summary_cards(
    explorer_summary
)

st.markdown("#### What stands out in this view")

st.info(
    _build_explorer_insight(
        explorer_base,
        metric_label=metric_labels[selected_metric],
        context_label=context_label,
    )
)

explorer_increases, explorer_decreases = _get_metric_leaders(
    explorer_base,
    metric=selected_metric,
    n=10,
)

if explorer_increases.empty and explorer_decreases.empty:
    st.info(
        "No eligible increases or decreases were available for this "
        "combination of filters."
    )
else:
    explorer_fig = build_split_lollipop_leaderboard(
        explorer_increases,
        explorer_decreases,
    )

    st.plotly_chart(
        explorer_fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
        key=(
            f"explorer_{selected_metric}_"
            f"{geography_scope}_{geography_value}_"
            f"{time_context}_{temporal_bucket}"
        ),
    )

    st.caption(
        "The chart shows up to ten increases and ten decreases. "
        "Hover over a zone to compare its pre-CP average, post-CP average, "
        "daily-average change, and percent change."
    )

    with st.expander(
        "Compare the underlying daily averages",
        expanded=False,
    ):
        detail_table = build_detail_table(
            explorer_increases,
            explorer_decreases,
        )

        st.dataframe(
            detail_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn(
                    format="%d",
                ),
                "Pre-CP daily avg": st.column_config.NumberColumn(
                    format="%,.1f",
                ),
                "Post-CP daily avg": st.column_config.NumberColumn(
                    format="%,.1f",
                ),
                "Daily-average change": st.column_config.NumberColumn(
                    format="%+,.1f",
                ),
                "Percent change": st.column_config.NumberColumn(
                    format="%+,.1f%%",
                ),
            },
        )