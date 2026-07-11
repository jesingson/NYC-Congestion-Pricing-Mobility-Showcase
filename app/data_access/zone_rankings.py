"""Ranking and visual helpers for Raw View 04.

Phase 2 replaces raw z-scores with a bounded within-metric distinctiveness
score. Each eligible zone is ranked within its metric:

    -100 = most distinctive decrease
       0 = typical zone
    +100 = most distinctive increase

The score preserves ordering, remains comparable across metrics, and avoids
letting one heavy-tailed metric dominate purely because of scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.data_access.loaders import CORE_METRICS, METRIC_LABELS
from app.data_access.spatial_aggregations import (
    ALL_TEMPORAL_BUCKETS_LABEL,
    get_zone_pre_post_metric_summary,
)
from app.data_access.spatial_visuals import add_reliability_flags
from app.utils.project_branding import BRAND_COLORS, apply_branding


METRIC_COLOR_MAP = {
    "taxi_trip_count": BRAND_COLORS["dark_teal"],
    "taxi_avg_trip_speed": BRAND_COLORS["seafoam"],
    "fhvhv_trip_count": BRAND_COLORS["terracotta"],
    "fhvhv_avg_trip_speed": BRAND_COLORS["pale_peach"],
    "subway_ridership": "#5B5F97",
    "avg_bus_speed": "#6B8E23",
}


def _signed_percentile_score(values: pd.Series) -> pd.Series:
    """Return a bounded signed percentile score from -100 to +100."""
    valid = values.notna()
    result = pd.Series(np.nan, index=values.index, dtype="float64")

    if valid.sum() <= 1:
        result.loc[valid] = 0.0
        return result

    ranks = values.loc[valid].rank(
        method="average",
        pct=True,
    )

    # Map percentile ranks from (0, 1] to approximately [-100, 100].
    result.loc[valid] = (ranks - 0.5) * 200
    return result


def prepare_zone_metric_ranking_base(
    *,
    metrics: list[str] | None = None,
    temporal_bucket: str | None = None,
    ranking_value_column: str = "percent_change",
    apply_reliability_thresholds: bool = True,
) -> pd.DataFrame:
    """Return one eligible row per Taxi Zone × metric with distinctiveness."""
    metrics = metrics or CORE_METRICS
    temporal_bucket = temporal_bucket or ALL_TEMPORAL_BUCKETS_LABEL

    summary = get_zone_pre_post_metric_summary(
        metrics=metrics,
        temporal_bucket=temporal_bucket,
    ).copy()

    if summary.empty:
        return summary

    summary = add_reliability_flags(summary)

    base = summary[
        summary["has_both_periods"]
        & summary[ranking_value_column].notna()
    ].copy()

    if apply_reliability_thresholds:
        base = base[base["eligible_for_percent_change"]].copy()

    base["distinctiveness_score"] = (
        base.groupby("metric", observed=True)[ranking_value_column]
        .transform(_signed_percentile_score)
    )
    base["absolute_distinctiveness"] = (
        base["distinctiveness_score"].abs()
    )
    base["zone_metric_label"] = (
        base["zone"] + " · " + base["metric_label"]
    )

    return (
        base.sort_values(["metric", "taxi_zone_id"])
        .reset_index(drop=True)
    )


def get_balanced_multimodal_ranking(
    ranking_base: pd.DataFrame,
    *,
    per_metric_per_direction: int = 1,
) -> pd.DataFrame:
    """Return top increase/decrease entries for every represented metric."""
    rows: list[pd.DataFrame] = []

    for _, metric_df in ranking_base.groupby(
        "metric",
        observed=True,
        sort=False,
    ):
        positive = (
            metric_df[metric_df["distinctiveness_score"] > 0]
            .nlargest(
                per_metric_per_direction,
                "distinctiveness_score",
            )
        )
        negative = (
            metric_df[metric_df["distinctiveness_score"] < 0]
            .nsmallest(
                per_metric_per_direction,
                "distinctiveness_score",
            )
        )
        rows.extend([positive, negative])

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    result["direction"] = np.where(
        result["distinctiveness_score"] >= 0,
        "Increase",
        "Decrease",
    )

    return (
        result.sort_values("distinctiveness_score")
        .reset_index(drop=True)
    )


def get_unrestricted_ranking(
    ranking_base: pd.DataFrame,
    *,
    positive_n: int = 8,
    negative_n: int = 8,
) -> pd.DataFrame:
    """Return the strongest overall positive and negative entries."""
    positive = (
        ranking_base[
            ranking_base["distinctiveness_score"] > 0
        ]
        .nlargest(positive_n, "distinctiveness_score")
    )
    negative = (
        ranking_base[
            ranking_base["distinctiveness_score"] < 0
        ]
        .nsmallest(negative_n, "distinctiveness_score")
    )

    result = pd.concat([positive, negative], ignore_index=True)
    result["direction"] = np.where(
        result["distinctiveness_score"] >= 0,
        "Increase",
        "Decrease",
    )

    return (
        result.sort_values("distinctiveness_score")
        .reset_index(drop=True)
    )


def get_metric_contribution_summary(
    ranking_df: pd.DataFrame,
) -> pd.DataFrame:
    """Count metric representation within a ranking."""
    if ranking_df.empty:
        return pd.DataFrame()

    result = (
        ranking_df.groupby(
            ["metric", "metric_label"],
            observed=True,
            dropna=False,
        )
        .agg(
            ranking_positions=("taxi_zone_id", "size"),
            increases=(
                "distinctiveness_score",
                lambda values: int((values > 0).sum()),
            ),
            decreases=(
                "distinctiveness_score",
                lambda values: int((values < 0).sum()),
            ),
            strongest_abs_score=(
                "absolute_distinctiveness",
                "max",
            ),
        )
        .reset_index()
    )

    total = result["ranking_positions"].sum()
    result["ranking_share"] = np.where(
        total > 0,
        result["ranking_positions"] / total,
        np.nan,
    )

    return (
        result.sort_values(
            ["ranking_positions", "strongest_abs_score"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )


def _add_hover_strings(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    def fmt(value, *, signed=False, percent=False):
        if pd.isna(value):
            return "Unavailable"
        prefix = "+" if signed else ""
        suffix = "%" if percent else ""
        return f"{float(value):{prefix},.2f}{suffix}"

    result["hover_pre"] = result["pre_daily_average"].map(fmt)
    result["hover_post"] = result["post_daily_average"].map(fmt)
    result["hover_abs"] = result["absolute_change"].map(
        lambda value: fmt(value, signed=True)
    )
    result["hover_pct"] = result["percent_change"].map(
        lambda value: fmt(
            value,
            signed=True,
            percent=True,
        )
    )
    result["hover_score"] = result["distinctiveness_score"].map(
        lambda value: fmt(value, signed=True)
    )

    return result


def build_ranking_bar_chart(
    ranking_df: pd.DataFrame,
    *,
    title: str,
    height: int = 720,
) -> go.Figure:
    """Build a horizontal ranked bar chart colored by metric."""
    plot_df = _add_hover_strings(ranking_df)

    fig = px.bar(
        plot_df,
        x="distinctiveness_score",
        y="zone_metric_label",
        color="metric",
        orientation="h",
        color_discrete_map=METRIC_COLOR_MAP,
        custom_data=[
            "zone",
            "borough",
            "metric_label",
            "hover_pre",
            "hover_post",
            "hover_abs",
            "hover_pct",
            "hover_score",
        ],
        labels={
            "distinctiveness_score": "Distinctiveness score",
            "zone_metric_label": "",
            "metric": "",
        },
        title=title,
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Borough: %{customdata[1]}<br>"
            "Metric: %{customdata[2]}<br>"
            "Pre-CP daily average: %{customdata[3]}<br>"
            "Post-CP daily average: %{customdata[4]}<br>"
            "Daily-average change: %{customdata[5]}<br>"
            "Percent change: %{customdata[6]}<br>"
            "Distinctiveness score: %{customdata[7]}"
            "<extra></extra>"
        )
    )

    fig.add_vline(
        x=0,
        line_width=1,
        line_color="rgba(60,60,60,0.65)",
    )

    fig.update_xaxes(
        range=[-105, 105],
        tickvals=[-100, -50, 0, 50, 100],
        ticktext=[
            "Most distinctive decrease",
            "-50",
            "Typical",
            "+50",
            "Most distinctive increase",
        ],
    )

    fig = apply_branding(fig)

    fig.update_layout(
        height=height,
        margin={"l": 0, "r": 20, "t": 70, "b": 95},
        legend={
            "title": {"text": ""},
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.12,
            "yanchor": "top",
        },
        bargap=0.18,
    )

    return fig


def build_metric_contribution_chart(
    contribution_df: pd.DataFrame,
    *,
    title: str,
) -> go.Figure:
    """Build a compact horizontal metric-contribution chart."""
    plot_df = contribution_df.sort_values(
        "ranking_positions",
        ascending=True,
    )

    fig = px.bar(
        plot_df,
        x="ranking_positions",
        y="metric_label",
        color="metric",
        orientation="h",
        color_discrete_map=METRIC_COLOR_MAP,
        custom_data=[
            "increases",
            "decreases",
            "ranking_share",
        ],
        labels={
            "ranking_positions": "Ranking positions",
            "metric_label": "",
            "metric": "",
        },
        title=title,
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Positions: %{x:,}<br>"
            "Increases: %{customdata[0]:,}<br>"
            "Decreases: %{customdata[1]:,}<br>"
            "Share: %{customdata[2]:.1%}"
            "<extra></extra>"
        )
    )

    fig = apply_branding(fig)
    fig.update_layout(
        height=390,
        margin={"l": 0, "r": 20, "t": 65, "b": 20},
        showlegend=False,
    )

    return fig


def format_ranking_for_display(
    ranking_df: pd.DataFrame,
) -> pd.DataFrame:
    """Format ranking rows for Streamlit tables."""
    display = ranking_df.copy()

    for column in [
        "distinctiveness_score",
        "pre_daily_average",
        "post_daily_average",
        "absolute_change",
        "percent_change",
        "pre_support_daily_average",
    ]:
        display[column] = display[column].round(2)

    display = display.rename(
        columns={
            "zone": "Taxi Zone",
            "borough": "Borough",
            "metric_label": "Metric",
            "distinctiveness_score": "Distinctiveness score",
            "pre_daily_average": "Pre-CP daily avg",
            "post_daily_average": "Post-CP daily avg",
            "absolute_change": "Daily-average change",
            "percent_change": "% change",
            "pre_support_daily_average": "Pre-CP support avg",
        }
    )

    return display[
        [
            "Taxi Zone",
            "Borough",
            "Metric",
            "Distinctiveness score",
            "Pre-CP daily avg",
            "Post-CP daily avg",
            "Daily-average change",
            "% change",
            "Pre-CP support avg",
        ]
    ]