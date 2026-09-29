"""Raw 18 — Forecast Scorecard, Pass 3."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.data_access.forecasting import (
    load_forecast_geography_context,
    load_forecast_history,
    load_forecast_job_summary,
    load_forecast_records,
    load_forecast_zone_summary,
)
from app.data_access.anomalies import load_selected_anomaly_events
from app.utils.project_branding import (
    BRAND_COLORS,
    apply_branding,
    exploration_section,
    inject_app_css,
    render_chart_insight,
)


# ---------------------------------------------------------------------
# Raw 18 — Pass 2: Freeze the opening forecast story
# ---------------------------------------------------------------------
#
# PAGE QUESTION
#   How did the forecasts perform?
#
# PASS-2 STORY
#   1. Start with the most fundamental view: what actually happened versus
#      what the forecasting system predicted.
#   2. Use a Manhattan aggregate as the hero. It sits between a very smooth
#      citywide total and a single Taxi Zone, which better matches the spatial
#      spirit of the Showcase.
#   3. Follow with the 5 × 3 Forecast Quilt. The Quilt answers the broader
#      question: how accurate was the full system across every target and
#      forecast horizon?
#
# IMPORTANT INTERPRETATION
#   The hero is a WEEKLY AGGREGATE. Local over- and under-predictions can
#   partially cancel when summed. The Quilt therefore remains the stronger
#   system-wide accuracy view because its Relative MAE is calculated before
#   that geographic/weekly aggregation.
#
# THIS PASS BUILDS
#   * one tabbed Manhattan hero: Actual vs Forecast / Forecast Error,
#   * one tabbed Forecast Quilt: Seaport forecasts / NYC accuracy,
#   * and a consistent interactive explorer with one geography at a time,
#     All-temporal-buckets support, activity-weighted aggregate speeds,
#     saved views, and an optional stress-anomaly overlay.
# ---------------------------------------------------------------------

PAGE_CAPTION = "FORECASTING & RELIABILITY"
PAGE_TITLE = "How did the forecasts perform?"

FINAL_HOLDOUT_START_DATE = pd.Timestamp("2026-01-05")

HERO_BOROUGH = "Manhattan"
HERO_METRIC = "subway_ridership"
HORIZONS = (1, 2, 5)

METRIC_ORDER = [
    "taxi_trip_count",
    "taxi_avg_trip_speed",
    "fhvhv_trip_count",
    "fhvhv_avg_trip_speed",
    "subway_ridership",
]

METRIC_LABELS = {
    "taxi_trip_count": "Taxi trips",
    "taxi_avg_trip_speed": "Taxi average speed",
    "fhvhv_trip_count": "FHVHV trips",
    "fhvhv_avg_trip_speed": "FHVHV average speed",
    "subway_ridership": "Subway ridership",
}

METRIC_DESCRIPTIONS = {
    "taxi_trip_count": "Yellow-taxi demand",
    "taxi_avg_trip_speed": "Yellow-taxi movement",
    "fhvhv_trip_count": "For-hire demand",
    "fhvhv_avg_trip_speed": "For-hire movement",
    "subway_ridership": "Transit demand",
}

HORIZON_COLORS = {
    1: BRAND_COLORS["dark_teal"],
    2: BRAND_COLORS["seafoam"],
    5: BRAND_COLORS["terracotta"],
}

PLOTLY_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
}

RECORD_COLUMNS = [
    "metric",
    "horizon",
    "taxi_zone_id",
    "borough",
    "target_date",
    "target_temporal_bucket",
    "actual",
    "champion_prediction",
    "benchmark_prediction",
    "reader_facing_zone",
]

TARGET_ROW_KEYS = [
    "metric",
    "taxi_zone_id",
    "target_date",
    "target_temporal_bucket",
]


# ---------------------------------------------------------------------
# Validation + formatting helpers
# ---------------------------------------------------------------------

def metric_label(metric: str) -> str:
    """Return the reader-facing label used throughout the page."""
    return METRIC_LABELS.get(metric, metric)


def compact_number(value: float) -> str:
    """Use human-readable compact labels without hiding meaningful scale."""
    value = float(value)
    magnitude = abs(value)

    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if magnitude >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a six-digit hex color to RGB."""
    value = hex_color.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def mix_colors(
    low_color: str,
    high_color: str,
    weight: float,
) -> str:
    """Blend two brand colors for a subtle card background."""
    weight = float(np.clip(weight, 0, 1))
    low = np.array(hex_to_rgb(low_color), dtype=float)
    high = np.array(hex_to_rgb(high_color), dtype=float)
    mixed = np.rint(low * (1 - weight) + high * weight).astype(int)

    return f"rgb({mixed[0]}, {mixed[1]}, {mixed[2]})"


def strip_plotly_title_artifacts(
    figure: go.Figure,
) -> go.Figure:
    """
    Remove shared-template title artifacts from figures that intentionally
    use Streamlit headings instead of native Plotly titles.

    Some shared branding paths can serialize an absent title as the literal
    string "undefined". Supplying a zero-width title and removing any matching
    annotation keeps chart headers clean.
    """
    figure.update_layout(
        title={
            # A zero-width space is intentionally used instead of an empty
            # string. Some Plotly/Streamlit combinations can surface an empty
            # shared-template title as the literal word "undefined".
            "text": "\u200b",
        },
        legend_title_text="",
    )

    annotations = list(
        figure.layout.annotations
        if figure.layout.annotations
        else []
    )

    cleaned = [
        annotation
        for annotation in annotations
        if str(annotation.text).strip().lower()
        != "undefined"
    ]

    figure.update_layout(
        annotations=cleaned,
    )

    return figure


# ---------------------------------------------------------------------
# Load the frozen Chapter 4 handoff
# ---------------------------------------------------------------------

@st.cache_data(show_spinner="Loading the Manhattan forecast hero...")
def load_hero_records() -> pd.DataFrame:
    """Load the exact final-holdout records used by the Manhattan hero."""
    frame = load_forecast_records(
        columns=RECORD_COLUMNS,
        required_columns=RECORD_COLUMNS,
        metrics=HERO_METRIC,
        boroughs=HERO_BOROUGH,
        reader_facing_only=True,
        final_holdout_only=True,
    )

    return frame.loc[
        frame["horizon"].isin(HORIZONS)
        & frame["target_date"].notna()
        & frame["actual"].notna()
        & frame["champion_prediction"].notna()
        & frame["benchmark_prediction"].notna()
    ].copy()


@st.cache_data(show_spinner="Loading Forecast Quilt summaries...")
def load_quilt_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the frozen system-level and Taxi-Zone forecast summaries.

    WHY:
    Physical file ownership, schema validation, and dtype normalization belong
    to the shared forecasting access layer. Raw 18 only applies the Quilt's
    reader-facing analytical checks.
    """
    job_columns = [
        "metric",
        "horizon",
        "champion_family",
        "relative_mae_pct",
        "benchmark_skill_pct",
        "champion_better_row_pct",
        "severe_error_pct",
    ]

    zone_columns = [
        "metric",
        "horizon",
        "taxi_zone_id",
        "zone",
        "borough",
        "reader_facing_zone",
        "relative_mae_pct",
        "benchmark_skill_pct",
    ]

    jobs = load_forecast_job_summary(
        columns=job_columns,
        required_columns=job_columns,
    )

    zones = load_forecast_zone_summary(
        columns=zone_columns,
        required_columns=zone_columns,
        metrics=METRIC_ORDER,
        horizons=HORIZONS,
        reader_facing_only=True,
    )

    if jobs.duplicated(["metric", "horizon"]).any():
        st.error("Forecast job summary is not unique by metric × horizon.")
        st.stop()

    zones = zones.loc[
        zones["relative_mae_pct"].notna()
    ].copy()

    return jobs.copy(), zones

# ---------------------------------------------------------------------
# Hero: Manhattan weekly Actual vs Forecast
# ---------------------------------------------------------------------

@st.cache_data(show_spinner="Aggregating complete Manhattan weeks...")
def build_hero_weekly(records: pd.DataFrame) -> pd.DataFrame:
    """
    Build a comparable h=1 / h=2 / h=5 weekly Manhattan trajectory.

    We first keep only target rows available for all three horizons. This avoids
    comparing lines built from subtly different sets of places or dayparts.

    We then keep only complete Monday-Sunday weeks. The final study week begins
    March 30 but contains only two study dates, so it must not be mistaken for
    a genuine collapse in ridership.
    """
    horizon_counts = (
        records
        .groupby(
            TARGET_ROW_KEYS,
            observed=True,
        )["horizon"]
        .nunique()
        .rename("horizon_count")
        .reset_index()
    )

    complete_keys = horizon_counts.loc[
        horizon_counts["horizon_count"].eq(len(HORIZONS)),
        TARGET_ROW_KEYS,
    ]

    common = records.merge(
        complete_keys,
        on=TARGET_ROW_KEYS,
        how="inner",
        validate="many_to_one",
    )

    common["week_start"] = (
        common["target_date"]
        - pd.to_timedelta(
            common["target_date"].dt.weekday,
            unit="D",
        )
    ).dt.normalize()

    dates_per_week = (
        common[
            ["week_start", "target_date"]
        ]
        .drop_duplicates()
        .groupby("week_start", observed=True)
        .size()
        .rename("dates_in_week")
    )

    complete_weeks = dates_per_week.loc[
        dates_per_week.eq(7)
    ].index

    common = common.loc[
        common["week_start"].isin(complete_weeks)
    ].copy()

    weekly = (
        common
        .groupby(
            ["week_start", "horizon"],
            observed=True,
            sort=True,
        )
        .agg(
            actual=("actual", "sum"),
            forecast=("champion_prediction", "sum"),
            benchmark=("benchmark_prediction", "sum"),
            source_rows=("actual", "size"),
            zones=("taxi_zone_id", "nunique"),
        )
        .reset_index()
    )

    actual_check = weekly.pivot(
        index="week_start",
        columns="horizon",
        values="actual",
    )
    actual_spread = (
        actual_check.max(axis=1)
        - actual_check.min(axis=1)
    )

    if (actual_spread > 1e-6).any():
        st.error(
            "Manhattan hero horizons do not align on identical observed totals."
        )
        st.stop()

    benchmark_check = weekly.pivot(
        index="week_start",
        columns="horizon",
        values="benchmark",
    )
    benchmark_spread = (
        benchmark_check.max(axis=1)
        - benchmark_check.min(axis=1)
    )

    if (benchmark_spread > 1e-6).any():
        st.error(
            "Manhattan hero horizons do not align on identical Last-week "
            "baseline totals."
        )
        st.stop()

    forecast_wide = (
        weekly.pivot(
            index="week_start",
            columns="horizon",
            values="forecast",
        )
        .rename(
            columns={
                1: "forecast_h1",
                2: "forecast_h2",
                5: "forecast_h5",
            }
        )
    )

    result = pd.DataFrame(
        {
            "actual": actual_check.mean(axis=1),
            "benchmark": benchmark_check.mean(axis=1),
        }
    ).join(
        forecast_wide,
        how="inner",
    )

    result = result.reset_index().sort_values("week_start")
    return result


def hero_metrics(
    weekly: pd.DataFrame,
) -> dict[int, dict[str, float]]:
    """Calculate reader-facing weekly aggregate accuracy for each horizon."""
    denominator = weekly["actual"].abs().sum()
    benchmark_error = (
        weekly["benchmark"] - weekly["actual"]
    ).abs().sum()

    output: dict[int, dict[str, float]] = {}

    for horizon in HORIZONS:
        forecast = weekly[f"forecast_h{horizon}"]
        model_error = (
            forecast - weekly["actual"]
        ).abs().sum()

        relative_mae = (
            100 * model_error / denominator
            if denominator > 0
            else np.nan
        )

        benchmark_skill = (
            100 * (benchmark_error - model_error) / benchmark_error
            if benchmark_error > 0
            else np.nan
        )

        correlation = forecast.corr(weekly["actual"])

        output[horizon] = {
            "relative_mae_pct": relative_mae,
            "benchmark_skill_pct": benchmark_skill,
            "tracking_correlation": correlation,
        }

    return output


def hero_figure(
    weekly: pd.DataFrame,
    visible_horizons: set[int],
    show_benchmark: bool,
) -> go.Figure:
    """
    Show observed Manhattan ridership plus the forecast lines the visitor selected.

    Observed ridership always stays visible because it is the reference the
    forecasts are being judged against. The checkboxes above the chart let the
    visitor simplify the comparison by hiding any forecast horizon or the
    last-week baseline.
    """
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=weekly["week_start"],
            y=weekly["actual"],
            mode="lines+markers",
            name="Observed",
            line={
                "color": "#263238",
                "width": 4,
            },
            marker={
                "size": 7,
                "color": "#263238",
            },
            hovertemplate=(
                "<b>%{x|%b %d, %Y}</b><br>"
                "Observed riders: %{y:,.0f}<extra></extra>"
            ),
        )
    )

    for horizon in HORIZONS:
        if horizon not in visible_horizons:
            continue

        figure.add_trace(
            go.Scatter(
                x=weekly["week_start"],
                y=weekly[f"forecast_h{horizon}"],
                mode="lines",
                name=f"h={horizon}",
                line={
                    "color": HORIZON_COLORS[horizon],
                    "width": 2.5 if horizon == 1 else 2.1,
                },
                hovertemplate=(
                    "<b>%{x|%b %d, %Y}</b><br>"
                    f"h={horizon} forecast: "
                    "%{y:,.0f}<extra></extra>"
                ),
            )
        )

    if show_benchmark:
        figure.add_trace(
            go.Scatter(
                x=weekly["week_start"],
                y=weekly["benchmark"],
                mode="lines",
                name="Last-week baseline",
                line={
                    "color": "#7A878C",
                    "width": 1.7,
                    "dash": "dot",
                },
                hovertemplate=(
                    "<b>%{x|%b %d, %Y}</b><br>"
                    "Last-week baseline: %{y:,.0f}"
                    "<extra></extra>"
                ),
            )
        )

    figure = apply_branding(figure)
    figure = strip_plotly_title_artifacts(figure)

    # The title lives in Streamlit above the chart, so the legend gets its own
    # uncluttered strip and never competes with the headline.
    figure.update_layout(
        height=505,
        hovermode="x unified",
        margin=dict(t=58, r=35, b=60, l=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.015,
            xanchor="left",
            x=0,
        ),
        xaxis_title="",
        yaxis_title="Weekly riders",
    )
    figure.update_yaxes(
        tickformat="~s",
        rangemode="tozero",
    )

    return figure


# ---------------------------------------------------------------------
# Forecast Quilt: 15 small-multiple cards
# ---------------------------------------------------------------------

def row_display_limit(
    zones: pd.DataFrame,
    metric: str,
) -> float:
    """
    Use one display ceiling across the three horizons within a metric row.

    The 95th percentile prevents a few sparse-zone outliers from flattening
    the useful part of every mini distribution. The actual headline Relative
    MAE remains untrimmed.
    """
    values = zones.loc[
        zones["metric"].eq(metric),
        "relative_mae_pct",
    ].dropna()

    if values.empty:
        return 1.0

    limit = float(values.quantile(0.95))
    return max(limit, 1.0)


def build_distribution(
    values: pd.Series,
    upper_limit: float,
    bins: int = 13,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert zone-level errors into a normalized mini histogram."""
    clean = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    clean = clean.loc[
        clean.ge(0)
    ]

    if clean.empty:
        edges = np.linspace(0, upper_limit, bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        return centers, np.zeros(bins)

    clipped = clean.clip(upper=upper_limit)
    counts, edges = np.histogram(
        clipped,
        bins=bins,
        range=(0, upper_limit),
    )
    percentages = (
        100 * counts / counts.sum()
        if counts.sum() > 0
        else np.zeros_like(counts, dtype=float)
    )
    centers = (edges[:-1] + edges[1:]) / 2

    return centers, percentages


def quilt_figure(
    jobs: pd.DataFrame,
    zones: pd.DataFrame,
) -> go.Figure:
    """
    Render the 15 Forecast Quilt cells as compact, readable small multiples.

    Each card combines:
      * headline Relative MAE,
      * improvement over the last-week baseline,
      * share of zone summaries with positive benchmark skill,
      * the real distribution of Taxi-Zone-level Relative MAE.

    Each metric row uses one shared x-axis range across h=1 / h=2 / h=5.
    The displayed range ends at that metric row's 95th-percentile zone error,
    so sparse-zone outliers cannot flatten the useful part of the distribution.
    The final x-axis label includes "+" to make that clipping visible.
    """
    minimum = float(jobs["relative_mae_pct"].min())
    maximum = float(jobs["relative_mae_pct"].max())
    span = max(maximum - minimum, 1e-9)

    figure = make_subplots(
        rows=len(METRIC_ORDER),
        cols=len(HORIZONS),
        horizontal_spacing=0.032,
        vertical_spacing=0.038,
    )

    card_metadata = []

    for row_number, metric in enumerate(METRIC_ORDER, start=1):
        display_limit = row_display_limit(
            zones,
            metric,
        )

        tick_values = [
            0,
            display_limit / 4,
            3 * display_limit / 4,
            display_limit,
        ]
        tick_text = [
            "0%",
            f"{display_limit / 4:.0f}%",
            f"{3 * display_limit / 4:.0f}%",
            f"{display_limit:.0f}%+",
        ]

        for column_number, horizon in enumerate(HORIZONS, start=1):
            job_rows = jobs.loc[
                jobs["metric"].eq(metric)
                & jobs["horizon"].eq(horizon)
            ]

            if len(job_rows) != 1:
                st.error(
                    "Forecast Quilt requires one job row for every "
                    f"Metric × Horizon cell. Missing/duplicate: "
                    f"{metric}, h={horizon}."
                )
                st.stop()

            job = job_rows.iloc[0]

            zone_rows = zones.loc[
                zones["metric"].eq(metric)
                & zones["horizon"].eq(horizon)
            ].copy()

            centers, percentages = build_distribution(
                zone_rows["relative_mae_pct"],
                display_limit,
            )

            supported_skill = pd.to_numeric(
                zone_rows["benchmark_skill_pct"],
                errors="coerce",
            ).dropna()

            positive_zone_share = (
                100 * supported_skill.gt(0).mean()
                if not supported_skill.empty
                else np.nan
            )

            error_weight = (
                float(job["relative_mae_pct"]) - minimum
            ) / span

            background = mix_colors(
                BRAND_COLORS["ice"],
                BRAND_COLORS["pale_peach"],
                error_weight,
            )

            figure.add_trace(
                go.Bar(
                    x=centers,
                    y=percentages,
                    marker={
                        "color": BRAND_COLORS["seafoam"],
                        "line": {
                            "color": BRAND_COLORS["dark_teal"],
                            "width": 0.4,
                        },
                    },
                    opacity=0.78,
                    showlegend=False,
                    customdata=np.column_stack(
                        [
                            np.full(
                                len(centers),
                                metric_label(metric),
                            ),
                            np.full(
                                len(centers),
                                horizon,
                            ),
                        ]
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]} · "
                        "h=%{customdata[1]}</b><br>"
                        "Taxi-Zone Relative MAE bin: %{x:.1f}%<br>"
                        "Zones in bin: %{y:.1f}%"
                        "<extra></extra>"
                    ),
                ),
                row=row_number,
                col=column_number,
            )

            # Reserve the upper half of the tiny card for the reader-facing
            # numbers; the histogram stays visually anchored at the bottom.
            y_max = max(
                float(percentages.max()) * 2.15,
                12.0,
            )

            figure.update_xaxes(
                range=[0, display_limit],
                tickmode="array",
                tickvals=tick_values,
                ticktext=tick_text,
                tickfont={
                    "size": 8,
                    "color": "#60767B",
                },
                ticks="inside",
                ticklen=3,
                ticklabelposition="inside",
                ticklabelstandoff=-6,
                showgrid=False,
                zeroline=False,
                fixedrange=True,
                row=row_number,
                col=column_number,
            )
            figure.update_yaxes(
                range=[0, y_max],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                fixedrange=True,
                row=row_number,
                col=column_number,
            )

            figure.add_vline(
                x=float(job["relative_mae_pct"]),
                line={
                    "color": BRAND_COLORS["dark_teal"],
                    "width": 2.0,
                },
                row=row_number,
                col=column_number,
            )

            card_metadata.append(
                {
                    "row": row_number,
                    "col": column_number,
                    "metric": metric,
                    "horizon": horizon,
                    "relative_mae_pct": float(
                        job["relative_mae_pct"]
                    ),
                    "benchmark_skill_pct": float(
                        job["benchmark_skill_pct"]
                    ),
                    "positive_zone_share": float(
                        positive_zone_share
                    ),
                    "background": background,
                }
            )

    # Turn each subplot into a visual card.
    for card in card_metadata:
        axis_index = (
            (card["row"] - 1) * len(HORIZONS)
            + card["col"]
        )

        xaxis_name = (
            "xaxis"
            if axis_index == 1
            else f"xaxis{axis_index}"
        )
        yaxis_name = (
            "yaxis"
            if axis_index == 1
            else f"yaxis{axis_index}"
        )

        x_domain = getattr(
            figure.layout,
            xaxis_name,
        ).domain
        y_domain = getattr(
            figure.layout,
            yaxis_name,
        ).domain

        figure.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=x_domain[0],
            x1=x_domain[1],
            y0=y_domain[0],
            y1=y_domain[1],
            fillcolor=card["background"],
            line={
                "color": "rgba(0, 109, 119, 0.24)",
                "width": 1.0,
            },
            layer="below",
        )

        card_width = x_domain[1] - x_domain[0]
        card_height = y_domain[1] - y_domain[0]

        figure.add_annotation(
            x=x_domain[0] + 0.055 * card_width,
            y=y_domain[1] - 0.07 * card_height,
            xref="paper",
            yref="paper",
            text=f"<b>{card['relative_mae_pct']:.1f}%</b>",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font={
                "size": 18,
                "color": "#003F46",
            },
        )

        figure.add_annotation(
            x=x_domain[0] + 0.055 * card_width,
            y=y_domain[1] - 0.28 * card_height,
            xref="paper",
            yref="paper",
            text=(
                f"<b>{card['benchmark_skill_pct']:+.1f}%</b> "
                "vs baseline"
            ),
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font={
                "size": 9.5,
                "color": "#335C67",
            },
        )

        figure.add_annotation(
            x=x_domain[0] + 0.055 * card_width,
            y=y_domain[1] - 0.43 * card_height,
            xref="paper",
            yref="paper",
            text=(
                f"{card['positive_zone_share']:.0f}% "
                "of zones improved"
            ),
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font={
                "size": 8.5,
                "color": "#52666C",
            },
        )

    # Column labels sit in their own top strip. The visual title now lives
    # outside Plotly, so these headers cannot collide with it.
    top_card_y = getattr(
        figure.layout,
        "yaxis",
    ).domain[1]

    for column_number, horizon in enumerate(HORIZONS, start=1):
        xaxis_name = (
            "xaxis"
            if column_number == 1
            else f"xaxis{column_number}"
        )
        x_domain = getattr(
            figure.layout,
            xaxis_name,
        ).domain

        figure.add_annotation(
            x=sum(x_domain) / 2,
            y=top_card_y + 0.028,
            xref="paper",
            yref="paper",
            text=f"<b>h = {horizon}</b>",
            showarrow=False,
            font={
                "size": 13,
                "color": BRAND_COLORS["dark_teal"],
            },
        )

    # Reader-facing row labels. The final left margin is deliberately restored
    # after apply_branding(), whose shared template otherwise resets it to 50 px.
    for row_number, metric in enumerate(METRIC_ORDER, start=1):
        axis_index = (
            (row_number - 1) * len(HORIZONS)
            + 1
        )
        yaxis_name = (
            "yaxis"
            if axis_index == 1
            else f"yaxis{axis_index}"
        )
        y_domain = getattr(
            figure.layout,
            yaxis_name,
        ).domain

        figure.add_annotation(
            x=-0.028,
            y=sum(y_domain) / 2,
            xref="paper",
            yref="paper",
            text=(
                f"<b>{metric_label(metric)}</b><br>"
                f"<span style='font-size:9px'>"
                f"{METRIC_DESCRIPTIONS[metric]}"
                "</span>"
            ),
            showarrow=False,
            xanchor="right",
            align="right",
            font={
                "size": 12,
                "color": "#003F46",
            },
        )

    figure = apply_branding(figure)

    # The shared Plotly brand template sets generic margins and a pale plotting
    # background. Restore the card-specific layout *after* branding.
    figure.update_layout(
        height=735,
        margin=dict(
            t=48,
            r=20,
            b=28,
            l=205,
        ),
        paper_bgcolor="white",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.08,
        showlegend=False,
        hoverlabel=dict(
            bgcolor="white",
            font_color="#003F46",
        ),
    )

    return figure

SEAPORT_ZONE_NAME = "Seaport"
SEAPORT_WINDOW_POINTS = 12

SEAPORT_COLUMNS = [
    "metric",
    "horizon",
    "taxi_zone_id",
    "zone",
    "borough",
    "target_date",
    "target_temporal_bucket",
    "actual",
    "champion_prediction",
    "reader_facing_zone",
]


@st.cache_data(show_spinner="Finding the best-supported Seaport Quilt window...")
def load_seaport_quilt_records() -> pd.DataFrame:
    """
    Load the strongest Seaport small-multiple surface that the data supports.

    A Taxi Zone does not need to support all five mobility measures. In
    particular, zones without mapped Subway activity should not make the
    entire Quilt fail.

    For each temporal bucket, we:
      1. require all three horizons for a metric on a date,
      2. test every possible subset of supported metrics,
      3. keep the largest subset that shares a useful common date window,
      4. and render only those metric rows.

    This preserves honest h=1 / h=2 / h=5 comparisons without manufacturing
    blank Subway values for a zone that may not have Subway support.
    """
    frame = load_forecast_records(
        columns=SEAPORT_COLUMNS,
        required_columns=SEAPORT_COLUMNS,
        zones=SEAPORT_ZONE_NAME,
        reader_facing_only=True,
        final_holdout_only=True,
    )

    frame["target_date"] = pd.to_datetime(
        frame["target_date"],
        errors="coerce",
    )
    frame["horizon"] = pd.to_numeric(
        frame["horizon"],
        errors="coerce",
    ).astype("Int64")

    frame = frame.loc[
        frame["metric"].isin(METRIC_ORDER)
        & frame["horizon"].isin(HORIZONS)
        & frame["target_date"].notna()
        & frame["actual"].notna()
        & frame["champion_prediction"].notna()
    ].copy()

    if frame.empty:
        return frame

    # Identify dates where a given metric has all three horizons available.
    metric_date_support = (
        frame[
            [
                "target_temporal_bucket",
                "metric",
                "target_date",
                "horizon",
            ]
        ]
        .drop_duplicates()
        .groupby(
            [
                "target_temporal_bucket",
                "metric",
                "target_date",
            ],
            observed=True,
        )["horizon"]
        .nunique()
        .rename("horizon_count")
        .reset_index()
    )

    metric_date_support = metric_date_support.loc[
        metric_date_support["horizon_count"].eq(len(HORIZONS))
    ].copy()

    if metric_date_support.empty:
        return pd.DataFrame(columns=frame.columns)

    # With only five metrics, brute-forcing all non-empty subsets is tiny and
    # lets us find the largest set of rows that can honestly share one window.
    from itertools import combinations

    best_choice = None

    for bucket, bucket_support in metric_date_support.groupby(
        "target_temporal_bucket",
        observed=True,
        sort=False,
    ):
        available_metrics = [
            metric
            for metric in METRIC_ORDER
            if metric in set(bucket_support["metric"])
        ]

        for subset_size in range(len(available_metrics), 0, -1):
            for subset in combinations(available_metrics, subset_size):
                date_sets = []

                for metric in subset:
                    dates = set(
                        bucket_support.loc[
                            bucket_support["metric"].eq(metric),
                            "target_date",
                        ]
                    )
                    date_sets.append(dates)

                common_dates = set.intersection(*date_sets)

                if not common_dates:
                    continue

                choice = {
                    "bucket": bucket,
                    "metrics": list(subset),
                    "dates": sorted(common_dates),
                    "metric_count": len(subset),
                    "date_count": len(common_dates),
                }

                # Prefer more supported metric rows first, then a longer shared
                # window. This naturally squeezes out unsupported rows rather
                # than forcing the whole Quilt to disappear.
                if best_choice is None:
                    best_choice = choice
                else:
                    current_score = (
                        choice["metric_count"],
                        choice["date_count"],
                    )
                    best_score = (
                        best_choice["metric_count"],
                        best_choice["date_count"],
                    )

                    if current_score > best_score:
                        best_choice = choice

            # Once a subset size produced a valid choice for this bucket,
            # smaller subsets cannot improve the metric-count criterion.
            if (
                best_choice is not None
                and best_choice["bucket"] == bucket
                and best_choice["metric_count"] == subset_size
            ):
                break

    if best_choice is None:
        return pd.DataFrame(columns=frame.columns)

    chosen_dates = pd.Series(
        best_choice["dates"],
        dtype="datetime64[ns]",
    ).sort_values()

    chosen_dates = chosen_dates.tail(
        min(SEAPORT_WINDOW_POINTS, len(chosen_dates))
    )

    selected = frame.loc[
        frame["target_temporal_bucket"].eq(best_choice["bucket"])
        & frame["metric"].isin(best_choice["metrics"])
        & frame["target_date"].isin(chosen_dates)
    ].copy()

    selected.attrs["chosen_bucket"] = best_choice["bucket"]
    selected.attrs["window_start"] = chosen_dates.min()
    selected.attrs["window_end"] = chosen_dates.max()
    selected.attrs["window_points"] = len(chosen_dates)
    selected.attrs["supported_metrics"] = best_choice["metrics"]
    selected.attrs["excluded_metrics"] = [
        metric
        for metric in METRIC_ORDER
        if metric not in best_choice["metrics"]
    ]

    return selected


def seaport_quilt_context(
    records: pd.DataFrame,
) -> tuple[str, pd.Timestamp, pd.Timestamp, int]:
    """Recover the automatically selected common-window metadata."""
    if records.empty:
        return "Unavailable", pd.NaT, pd.NaT, 0

    bucket = records.attrs.get(
        "chosen_bucket",
        str(records["target_temporal_bucket"].iloc[0]),
    )
    start = records.attrs.get(
        "window_start",
        records["target_date"].min(),
    )
    end = records.attrs.get(
        "window_end",
        records["target_date"].max(),
    )
    points = int(
        records.attrs.get(
            "window_points",
            records["target_date"].nunique(),
        )
    )

    return bucket, start, end, points


def seaport_forecast_quilt(
    records: pd.DataFrame,
) -> go.Figure:
    """
    Show the supported Seaport metrics as tiny Actual-vs-Forecast time series.

    Unsupported mobility rows are omitted entirely rather than displayed as
    misleading blanks. Every remaining row shares the same dates and temporal
    bucket, and the three horizon cells within a metric row share one y-scale.
    """
    if records.empty:
        return go.Figure()

    supported_metrics = records.attrs.get(
        "supported_metrics",
        [
            metric
            for metric in METRIC_ORDER
            if metric in set(records["metric"])
        ],
    )

    # Preserve the canonical project ordering even when one or more rows are
    # absent in this Taxi Zone.
    supported_metrics = [
        metric
        for metric in METRIC_ORDER
        if metric in supported_metrics
    ]

    figure = make_subplots(
        rows=len(supported_metrics),
        cols=len(HORIZONS),
        horizontal_spacing=0.032,
        vertical_spacing=0.055,
    )

    row_ranges: dict[str, tuple[float, float]] = {}

    for metric in supported_metrics:
        subset = records.loc[
            records["metric"].eq(metric)
        ]

        values = pd.concat(
            [
                subset["actual"],
                subset["champion_prediction"],
            ],
            ignore_index=True,
        ).dropna()

        if values.empty:
            row_ranges[metric] = (0.0, 1.0)
            continue

        minimum = float(values.min())
        maximum = float(values.max())
        span = max(maximum - minimum, abs(maximum) * 0.05, 1e-9)

        row_ranges[metric] = (
            minimum - 0.10 * span,
            maximum + 0.10 * span,
        )

    for row_number, metric in enumerate(supported_metrics, start=1):
        y_min, y_max = row_ranges[metric]

        for column_number, horizon in enumerate(HORIZONS, start=1):
            cell = (
                records.loc[
                    records["metric"].eq(metric)
                    & records["horizon"].eq(horizon)
                ]
                .sort_values("target_date")
                .copy()
            )

            if cell.empty:
                continue

            value_decimals = (
                2
                if metric in SPEED_WEIGHT_METRIC
                else 0
            )
            difference = (
                cell["champion_prediction"]
                - cell["actual"]
            )

            observed_text = cell["actual"].map(
                lambda value: f"{value:,.{value_decimals}f}"
            )
            forecast_text = cell["champion_prediction"].map(
                lambda value: f"{value:,.{value_decimals}f}"
            )
            difference_text = difference.map(
                lambda value: f"{value:+,.{value_decimals}f}"
            )
            hover_data = np.column_stack(
                [
                    observed_text,
                    forecast_text,
                    difference_text,
                ]
            )
            hover_template = (
                "<b>%{x|%b %d, %Y}</b><br>"
                "Observed: %{customdata[0]}<br>"
                f"h={horizon} forecast: %{{customdata[1]}}<br>"
                "Difference: %{customdata[2]}"
                "<extra></extra>"
            )

            figure.add_trace(
                go.Scatter(
                    x=cell["target_date"],
                    y=cell["actual"],
                    mode="lines+markers",
                    line={
                        "color": "#263238",
                        "width": 2.5,
                    },
                    marker={
                        "size": 4.5,
                        "color": "#263238",
                    },
                    name="Observed",
                    legendgroup="observed",
                    showlegend=(
                        row_number == 1
                        and column_number == 1
                    ),
                    customdata=hover_data,
                    hovertemplate=hover_template,
                ),
                row=row_number,
                col=column_number,
            )

            figure.add_trace(
                go.Scatter(
                    x=cell["target_date"],
                    y=cell["champion_prediction"],
                    mode="lines",
                    line={
                        "color": HORIZON_COLORS[horizon],
                        "width": 2.1,
                    },
                    name=f"h={horizon} forecast",
                    legendgroup=f"h{horizon}",
                    showlegend=(
                        row_number == 1
                    ),
                    customdata=hover_data,
                    hovertemplate=hover_template,
                ),
                row=row_number,
                col=column_number,
            )

            figure.update_yaxes(
                range=[y_min, y_max],
                showgrid=True,
                gridcolor="rgba(0, 109, 119, 0.10)",
                zeroline=False,
                showticklabels=(
                    column_number == 1
                ),
                tickfont={
                    "size": 8,
                    "color": "#60767B",
                },
                fixedrange=True,
                row=row_number,
                col=column_number,
            )

            figure.update_xaxes(
                showgrid=False,
                zeroline=False,
                showticklabels=True,
                tickformat="%b %d",
                nticks=3,
                ticks="inside",
                ticklen=3,
                ticklabelposition="inside",
                ticklabelstandoff=-6,
                tickfont={
                    "size": 7.5,
                    "color": "#60767B",
                },
                fixedrange=True,
                row=row_number,
                col=column_number,
            )

    top_card_y = getattr(
        figure.layout,
        "yaxis",
    ).domain[1]

    for column_number, horizon in enumerate(HORIZONS, start=1):
        xaxis_name = (
            "xaxis"
            if column_number == 1
            else f"xaxis{column_number}"
        )
        x_domain = getattr(
            figure.layout,
            xaxis_name,
        ).domain

        figure.add_annotation(
            x=sum(x_domain) / 2,
            y=top_card_y + 0.03,
            xref="paper",
            yref="paper",
            text=f"<b>h = {horizon}</b>",
            showarrow=False,
            font={
                "size": 13,
                "color": BRAND_COLORS["dark_teal"],
            },
        )

    for row_number, metric in enumerate(supported_metrics, start=1):
        axis_index = (
            (row_number - 1) * len(HORIZONS)
            + 1
        )
        yaxis_name = (
            "yaxis"
            if axis_index == 1
            else f"yaxis{axis_index}"
        )
        y_domain = getattr(
            figure.layout,
            yaxis_name,
        ).domain

        figure.add_annotation(
            x=-0.03,
            y=sum(y_domain) / 2,
            xref="paper",
            yref="paper",
            text=(
                f"<b>{metric_label(metric)}</b><br>"
                f"<span style='font-size:9px'>"
                f"{METRIC_DESCRIPTIONS[metric]}"
                "</span>"
            ),
            showarrow=False,
            xanchor="right",
            align="right",
            font={
                "size": 12,
                "color": "#003F46",
            },
        )

    figure = apply_branding(figure)

    # Scale height to the number of supported rows instead of reserving space
    # for metrics that do not exist in this Taxi Zone.
    figure = strip_plotly_title_artifacts(figure)

    figure.update_layout(
        height=max(390, 135 * len(supported_metrics) + 115),
        margin=dict(
            t=55,
            r=20,
            b=45,
            l=205,
        ),
        hovermode="closest",
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.035,
            xanchor="left",
            x=0,
            font={"size": 10},
        ),
    )

    return figure


def accuracy_quilt_tabbed(
    jobs: pd.DataFrame,
    zones: pd.DataFrame,
) -> go.Figure:
    """
    Compact version of the NYC accuracy Quilt for the tabbed comparison.

    Each histogram carries its own compact Relative-MAE scale. Tick labels sit
    inside the bottom edge of each card so they remain visually attached to the
    histogram they describe instead of drifting toward the card below.
    """
    minimum = float(jobs["relative_mae_pct"].min())
    maximum = float(jobs["relative_mae_pct"].max())
    span = max(maximum - minimum, 1e-9)

    figure = make_subplots(
        rows=len(METRIC_ORDER),
        cols=len(HORIZONS),
        horizontal_spacing=0.032,
        vertical_spacing=0.038,
    )

    card_metadata = []
    row_limits: dict[str, float] = {}

    for row_number, metric in enumerate(METRIC_ORDER, start=1):
        display_limit = row_display_limit(
            zones,
            metric,
        )
        row_limits[metric] = display_limit
        tick_values = [
            0,
            display_limit / 4,
            3 * display_limit / 4,
            display_limit,
        ]
        tick_text = [
            "0%",
            f"{display_limit / 4:.0f}%",
            f"{3 * display_limit / 4:.0f}%",
            f"{display_limit:.0f}%+",
        ]

        for column_number, horizon in enumerate(HORIZONS, start=1):
            job = jobs.loc[
                jobs["metric"].eq(metric)
                & jobs["horizon"].eq(horizon)
            ].iloc[0]

            zone_rows = zones.loc[
                zones["metric"].eq(metric)
                & zones["horizon"].eq(horizon)
            ].copy()

            centers, percentages = build_distribution(
                zone_rows["relative_mae_pct"],
                display_limit,
            )

            supported_skill = pd.to_numeric(
                zone_rows["benchmark_skill_pct"],
                errors="coerce",
            ).dropna()

            positive_zone_share = (
                100 * supported_skill.gt(0).mean()
                if not supported_skill.empty
                else np.nan
            )

            error_weight = (
                float(job["relative_mae_pct"]) - minimum
            ) / span

            background = mix_colors(
                BRAND_COLORS["ice"],
                BRAND_COLORS["pale_peach"],
                error_weight,
            )

            figure.add_trace(
                go.Bar(
                    x=centers,
                    y=percentages,
                    marker={
                        "color": BRAND_COLORS["seafoam"],
                        "line": {
                            "color": BRAND_COLORS["dark_teal"],
                            "width": 0.4,
                        },
                    },
                    opacity=0.78,
                    showlegend=False,
                    hovertemplate=(
                        "<b>"
                        f"{metric_label(metric)} · h={horizon}"
                        "</b><br>"
                        "Taxi-Zone Relative MAE bin: %{x:.1f}%<br>"
                        "Zones in bin: %{y:.1f}%"
                        "<extra></extra>"
                    ),
                ),
                row=row_number,
                col=column_number,
            )

            y_max = max(
                float(percentages.max()) * 2.15,
                12.0,
            )

            figure.update_xaxes(
                range=[0, display_limit],
                tickmode="array",
                tickvals=tick_values,
                ticktext=tick_text,
                tickfont={
                    "size": 7.5,
                    "color": "#60767B",
                },
                ticks="inside",
                ticklen=3,
                ticklabelposition="inside",
                ticklabelstandoff=-6,
                showgrid=False,
                zeroline=False,
                showticklabels=True,
                fixedrange=True,
                row=row_number,
                col=column_number,
            )
            figure.update_yaxes(
                range=[0, y_max],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                fixedrange=True,
                row=row_number,
                col=column_number,
            )

            figure.add_vline(
                x=float(job["relative_mae_pct"]),
                line={
                    "color": BRAND_COLORS["dark_teal"],
                    "width": 2.0,
                },
                row=row_number,
                col=column_number,
            )

            card_metadata.append(
                {
                    "row": row_number,
                    "col": column_number,
                    "metric": metric,
                    "horizon": horizon,
                    "relative_mae_pct": float(
                        job["relative_mae_pct"]
                    ),
                    "benchmark_skill_pct": float(
                        job["benchmark_skill_pct"]
                    ),
                    "positive_zone_share": float(
                        positive_zone_share
                    ),
                    "background": background,
                }
            )

    for card in card_metadata:
        axis_index = (
            (card["row"] - 1) * len(HORIZONS)
            + card["col"]
        )

        xaxis_name = (
            "xaxis"
            if axis_index == 1
            else f"xaxis{axis_index}"
        )
        yaxis_name = (
            "yaxis"
            if axis_index == 1
            else f"yaxis{axis_index}"
        )

        x_domain = getattr(
            figure.layout,
            xaxis_name,
        ).domain
        y_domain = getattr(
            figure.layout,
            yaxis_name,
        ).domain

        figure.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=x_domain[0],
            x1=x_domain[1],
            y0=y_domain[0],
            y1=y_domain[1],
            fillcolor=card["background"],
            line={
                "color": "rgba(0, 109, 119, 0.24)",
                "width": 1.0,
            },
            layer="below",
        )

        card_width = x_domain[1] - x_domain[0]
        card_height = y_domain[1] - y_domain[0]

        figure.add_annotation(
            x=x_domain[0] + 0.055 * card_width,
            y=y_domain[1] - 0.07 * card_height,
            xref="paper",
            yref="paper",
            text=f"<b>{card['relative_mae_pct']:.1f}%</b>",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font={
                "size": 18,
                "color": "#003F46",
            },
        )

        figure.add_annotation(
            x=x_domain[0] + 0.055 * card_width,
            y=y_domain[1] - 0.28 * card_height,
            xref="paper",
            yref="paper",
            text=(
                f"<b>{card['benchmark_skill_pct']:+.1f}%</b> "
                "vs baseline"
            ),
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font={
                "size": 9.5,
                "color": "#335C67",
            },
        )

        figure.add_annotation(
            x=x_domain[0] + 0.055 * card_width,
            y=y_domain[1] - 0.43 * card_height,
            xref="paper",
            yref="paper",
            text=(
                f"{card['positive_zone_share']:.0f}% "
                "of zones improved"
            ),
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font={
                "size": 8.5,
                "color": "#52666C",
            },
        )

    top_card_y = getattr(
        figure.layout,
        "yaxis",
    ).domain[1]

    for column_number, horizon in enumerate(HORIZONS, start=1):
        xaxis_name = (
            "xaxis"
            if column_number == 1
            else f"xaxis{column_number}"
        )
        x_domain = getattr(
            figure.layout,
            xaxis_name,
        ).domain

        figure.add_annotation(
            x=sum(x_domain) / 2,
            y=top_card_y + 0.028,
            xref="paper",
            yref="paper",
            text=f"<b>h = {horizon}</b>",
            showarrow=False,
            font={
                "size": 13,
                "color": BRAND_COLORS["dark_teal"],
            },
        )

    for row_number, metric in enumerate(METRIC_ORDER, start=1):
        axis_index = (
            (row_number - 1) * len(HORIZONS)
            + 1
        )
        yaxis_name = (
            "yaxis"
            if axis_index == 1
            else f"yaxis{axis_index}"
        )
        y_domain = getattr(
            figure.layout,
            yaxis_name,
        ).domain

        limit = row_limits[metric]

        figure.add_annotation(
            x=-0.03,
            y=sum(y_domain) / 2,
            xref="paper",
            yref="paper",
            text=(
                f"<b>{metric_label(metric)}</b><br>"
                f"<span style='font-size:9px'>"
                f"{METRIC_DESCRIPTIONS[metric]}</span>"
            ),
            showarrow=False,
            xanchor="right",
            align="right",
            font={
                "size": 12,
                "color": "#003F46",
            },
        )

    figure = apply_branding(figure)
    figure = strip_plotly_title_artifacts(figure)

    figure.update_layout(
        height=735,
        margin=dict(
            t=48,
            r=20,
            b=22,
            l=225,
        ),
        paper_bgcolor="white",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.08,
        showlegend=False,
    )

    return figure



# =====================================================================
# Consolidated Pass 2 — Hero tabs + interactive forecast explorer
# =====================================================================

ALL_DAY_TYPES = "All days"
ALL_DAYPARTS = "All dayparts"
ALL_DAYPARTS_LEGACY_UNUSED = "All temporal buckets"

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

DAY_TYPE_OPTIONS = [
    ALL_DAY_TYPES,
    "Weekdays",
    "Weekends",
]

DAYPART_OPTIONS = [
    ALL_DAYPARTS,
    "Overnight",
    "AM peak",
    "Midday",
    "PM peak",
    "Evening",
]

TEMPORAL_BUCKET_META = {
    "weekday_overnight": ("Weekdays", "Overnight"),
    "weekday_am_peak": ("Weekdays", "AM peak"),
    "weekday_midday": ("Weekdays", "Midday"),
    "weekday_pm_peak": ("Weekdays", "PM peak"),
    "weekday_evening": ("Weekdays", "Evening"),
    "weekend_overnight": ("Weekends", "Overnight"),
    "weekend_am_peak": ("Weekends", "AM peak"),
    "weekend_midday": ("Weekends", "Midday"),
    "weekend_pm_peak": ("Weekends", "PM peak"),
    "weekend_evening": ("Weekends", "Evening"),
}

TEMPORAL_BUCKET_HOURS = {
    "weekday_overnight": 1,
    "weekend_overnight": 1,
    "weekday_am_peak": 8,
    "weekend_am_peak": 8,
    "weekday_midday": 12,
    "weekend_midday": 12,
    "weekday_pm_peak": 17,
    "weekend_pm_peak": 17,
    "weekday_evening": 21,
    "weekend_evening": 21,
}

GEOGRAPHY_LEVELS = [
    "Taxi Zone",
    "Borough",
    "Mobility Environment",
    "Policy Geography",
    "Citywide",
]

SPEED_WEIGHT_METRIC = {
    "taxi_avg_trip_speed": "taxi_trip_count",
    "fhvhv_avg_trip_speed": "fhvhv_trip_count",
}

EXPLORER_RECORD_COLUMNS = [
    "metric",
    "horizon",
    "taxi_zone_id",
    "zone",
    "borough",
    "target_date",
    "target_temporal_bucket",
    "target_observation_sequence_id",
    "actual",
    "champion_prediction",
    "benchmark_prediction",
    "evaluation_period",
    "reader_facing_zone",
]

PRESET_VIEWS = {
    "Choose your own view": None,
    "Seaport — representative FHVHV forecast": {
        "geography_level": "Taxi Zone",
        "geography_value": "Seaport",
        "metric": "fhvhv_trip_count",
        "day_type": "Weekdays",
        "daypart": "Overnight",
        "start_date": "2026-03-02",
        "end_date": "2026-03-17",
    },
    "Pelham Parkway — Taxi forecasts track closely": {
        "geography_level": "Taxi Zone",
        "geography_value": "Pelham Parkway",
        "metric": "taxi_trip_count",
        "day_type": "Weekdays",
        "daypart": "PM peak",
        "start_date": "2026-03-06",
        "end_date": "2026-03-23",
    },
    "Greenpoint — longer-horizon Subway drift": {
        "geography_level": "Taxi Zone",
        "geography_value": "Greenpoint",
        "metric": "subway_ridership",
        "day_type": "Weekdays",
        "daypart": "Midday",
        "start_date": "2026-03-12",
        "end_date": "2026-03-27",
    },
    "LIC/Hunters Point — difficult Taxi period": {
        "geography_level": "Taxi Zone",
        "geography_value": "Long Island City/Hunters Point",
        "metric": "taxi_trip_count",
        "day_type": "Weekdays",
        "daypart": "PM peak",
        "start_date": "2026-03-06",
        "end_date": "2026-03-23",
    },
    "East Chelsea — FHVHV horizons diverge": {
        "geography_level": "Taxi Zone",
        "geography_value": "East Chelsea",
        "metric": "fhvhv_trip_count",
        "day_type": "Weekdays",
        "daypart": "PM peak",
        "start_date": "2026-03-02",
        "end_date": "2026-03-17",
    },
}

PRESET_DESCRIPTIONS = {
    "Choose your own view": (
        "Pick any geography, mobility measure, day type, daypart, and date range."
    ),
    "Seaport — representative FHVHV forecast": (
        "A clean, good-but-not-perfect local example where all three forecast "
        "horizons broadly follow observed FHVHV demand."
    ),
    "Pelham Parkway — Taxi forecasts track closely": (
        "A low-error Taxi-demand example where h=1, h=2, and h=5 stay close "
        "to the observed series."
    ),
    "Greenpoint — longer-horizon Subway drift": (
        "A clear horizon lesson: the shorter Subway forecasts track better "
        "while the h=5 path drifts farther away."
    ),
    "LIC/Hunters Point — difficult Taxi period": (
        "A volatile Taxi-demand stretch where sharp observed moves are harder "
        "for every horizon to follow."
    ),
    "East Chelsea — FHVHV horizons diverge": (
        "A local FHVHV example where the three forecast horizons separate "
        "noticeably from one another."
    ),
}


def humanize_bucket(value: str) -> str:
    """Turn the canonical temporal-bucket key into compact reader copy."""
    if value == ALL_DAYPARTS_LEGACY_UNUSED:
        return value

    parts = str(value).split("_", 1)
    if len(parts) != 2:
        return str(value).replace("_", " ").title()

    weekpart, daypart = parts
    daypart = (
        daypart
        .replace("am_peak", "AM peak")
        .replace("pm_peak", "PM peak")
        .replace("midday", "Midday")
        .replace("overnight", "Overnight")
        .replace("evening", "Evening")
    )
    return f"{weekpart.title()} · {daypart}"


def selected_temporal_buckets(
    day_type: str,
    daypart: str,
) -> list[str]:
    """Translate reader-facing Day type + Daypart controls to canonical buckets."""
    selected = []

    for bucket in TEMPORAL_BUCKET_ORDER:
        bucket_day_type, bucket_daypart = TEMPORAL_BUCKET_META[bucket]

        day_type_match = (
            day_type == ALL_DAY_TYPES
            or bucket_day_type == day_type
        )
        daypart_match = (
            daypart == ALL_DAYPARTS
            or bucket_daypart == daypart
        )

        if day_type_match and daypart_match:
            selected.append(bucket)

    return selected


def daypart_filter_label(
    day_type: str,
    daypart: str,
) -> str:
    """Return compact copy for the currently selected temporal view."""
    if day_type == ALL_DAY_TYPES and daypart == ALL_DAYPARTS:
        return "All days · All dayparts"
    if day_type == ALL_DAY_TYPES:
        return f"All days · {daypart}"
    if daypart == ALL_DAYPARTS:
        return f"{day_type} · All dayparts"
    return f"{day_type} · {daypart}"


def evaluation_period_copy(
    start_date,
    end_date,
) -> str:
    """Explain whether the selected window is validation or final holdout."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    if end < FINAL_HOLDOUT_START_DATE:
        return (
            "**Evidence period: Ordinary validation.** These forecasts were "
            "available during model-family selection, so use them for "
            "longitudinal exploration rather than untouched final evaluation."
        )

    if start >= FINAL_HOLDOUT_START_DATE:
        return (
            "**Evidence period: Final holdout.** These forecasts were reserved "
            "for evaluation after the champion system was frozen."
        )

    return (
        "**Evidence periods: Ordinary validation + final holdout.** "
        "January 5, 2026 is the boundary: observations before it were "
        "available during model selection; observations from that date onward "
        "belong to the untouched final holdout."
    )


def add_evaluation_boundary(
    figure: go.Figure,
    series: pd.DataFrame,
) -> go.Figure:
    """Mark the validation-to-holdout boundary only when the view crosses it."""
    if series.empty or "target_date" not in series.columns:
        return figure

    dates = pd.to_datetime(
        series["target_date"],
        errors="coerce",
    ).dropna()

    if dates.empty:
        return figure

    if not (
        dates.min().normalize() < FINAL_HOLDOUT_START_DATE
        <= dates.max().normalize()
    ):
        return figure

    boundary = FINAL_HOLDOUT_START_DATE.to_pydatetime()

    # A paper-referenced vertical line survives changes in the y-scale and makes
    # the evidence boundary visible without implying a mobility discontinuity.
    figure.add_shape(
        type="line",
        x0=boundary,
        x1=boundary,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line={
            "color": "#7A878C",
            "width": 1.4,
            "dash": "dash",
        },
    )
    figure.add_annotation(
        x=boundary,
        y=1,
        xref="x",
        yref="paper",
        text="Final holdout begins",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font={
            "size": 10,
            "color": "#60767B",
        },
    )

    return figure

@st.cache_data(show_spinner="Loading the selected forecast history...")
def load_explorer_metric_records(
    metric: str,
) -> pd.DataFrame:
    """
    Load the selected longitudinal target and any activity-weight series.

    The compact runtime is stored wide by horizon, but the shared data-access
    loader restores Raw 18's original long-form forecasting contract.
    """
    source_metrics = [metric]

    if metric in SPEED_WEIGHT_METRIC:
        source_metrics.append(
            SPEED_WEIGHT_METRIC[
                metric
            ]
        )

    frame = load_forecast_history(
        columns=EXPLORER_RECORD_COLUMNS,
        required_columns=EXPLORER_RECORD_COLUMNS,
        metrics=source_metrics,
        horizons=HORIZONS,
        reader_facing_only=True,
    )

    allowed_periods = {
        "ordinary_validation",
        "final_holdout",
    }

    observed_periods = set(
        frame["evaluation_period"]
        .dropna()
        .astype(str)
        .unique()
    )

    unexpected_periods = sorted(
        observed_periods
        - allowed_periods
    )

    if unexpected_periods:
        st.error(
            "Forecast history contains unexpected evaluation periods: "
            f"{unexpected_periods}"
        )
        st.stop()

    return frame


def geography_options(
    context: pd.DataFrame,
    geography_level: str,
) -> list[str]:
    """Return exactly one selectable geography value for the chosen level."""
    if geography_level == "Citywide":
        return ["NYC"]

    column = {
        "Taxi Zone": "zone",
        "Borough": "borough",
        "Mobility Environment": "mobility_environment",
        "Policy Geography": "policy_geography",
    }[geography_level]

    return sorted(
        context[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def geography_option_label(
    context: pd.DataFrame,
    geography_level: str,
    geography_value: str,
) -> str:
    """
    Add borough context to Taxi Zone choices without changing stored values.

    Presets and downstream filtering continue to use the canonical zone name;
    only the dropdown label becomes more informative for the reader.
    """
    if geography_level != "Taxi Zone":
        return str(geography_value)

    boroughs = (
        context.loc[
            context["zone"].astype(str).eq(str(geography_value)),
            "borough",
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not boroughs:
        return str(geography_value)

    return f"{geography_value} · {', '.join(sorted(boroughs))}"


def geography_zone_ids(
    context: pd.DataFrame,
    geography_level: str,
    geography_value: str,
) -> set[int]:
    """Resolve the one selected geography into its member Taxi Zones."""
    if geography_level == "Citywide":
        selected = context
    else:
        column = {
            "Taxi Zone": "zone",
            "Borough": "borough",
            "Mobility Environment": "mobility_environment",
            "Policy Geography": "policy_geography",
        }[geography_level]

        selected = context.loc[
            context[column].astype(str).eq(
                str(geography_value)
            )
        ]

    return set(
        selected["taxi_zone_id"]
        .dropna()
        .astype(int)
        .tolist()
    )


def ordered_plot_time(
    dates: pd.Series,
    buckets: pd.Series,
) -> pd.Series:
    """
    Build a readable datetime axis that preserves the modeled daypart order.

    The full forecast surface is ordered by Date × Temporal Bucket rather than
    by one observation per day. Assigning each bucket a representative hour
    lets visitors see the complete ordered sequence while retaining a normal
    calendar x-axis.
    """
    offsets = (
        buckets
        .map(TEMPORAL_BUCKET_HOURS)
        .fillna(12)
        .astype(int)
    )

    return (
        pd.to_datetime(dates)
        + pd.to_timedelta(offsets, unit="h")
    )


def _weighted_average(
    frame: pd.DataFrame,
    value_column: str,
    weight_column: str,
) -> float:
    """Return a non-negative activity-weighted average."""
    values = pd.to_numeric(
        frame[value_column],
        errors="coerce",
    )
    weights = (
        pd.to_numeric(
            frame[weight_column],
            errors="coerce",
        )
        .clip(lower=0)
    )

    valid = values.notna() & weights.notna() & weights.gt(0)

    if not valid.any():
        return np.nan

    return float(
        np.average(
            values.loc[valid],
            weights=weights.loc[valid],
        )
    )


def aggregate_explorer_series(
    records: pd.DataFrame,
    *,
    metric: str,
    zone_ids: set[int],
    day_type: str,
    daypart: str,
    start_date,
    end_date,
) -> pd.DataFrame:
    """
    Build the selected Actual / Forecast / Benchmark trajectory.

    Counts and Subway ridership sum across the selected geography.
    Average speeds use the corresponding trip-count series as activity weights.
    """
    allowed_buckets = selected_temporal_buckets(
        day_type,
        daypart,
    )

    working = records.loc[
        records["taxi_zone_id"].astype("Int64").isin(zone_ids)
        & records["target_date"].dt.date.between(
            start_date,
            end_date,
        )
        & records["target_temporal_bucket"].isin(
            allowed_buckets
        )
    ].copy()

    target = working.loc[
        working["metric"].eq(metric)
    ].copy()

    if target.empty:
        return pd.DataFrame()

    grouping = [
        "horizon",
        "target_date",
        "target_temporal_bucket",
        "target_observation_sequence_id",
    ]

    if metric not in SPEED_WEIGHT_METRIC:
        aggregated = (
            target
            .groupby(
                grouping,
                observed=True,
                sort=False,
            )
            .agg(
                actual=("actual", "sum"),
                forecast=("champion_prediction", "sum"),
                benchmark=("benchmark_prediction", "sum"),
                source_zones=("taxi_zone_id", "nunique"),
            )
            .reset_index()
        )
    else:
        weight_metric = SPEED_WEIGHT_METRIC[metric]

        weights = working.loc[
            working["metric"].eq(weight_metric),
            [
                "horizon",
                "taxi_zone_id",
                "target_date",
                "target_temporal_bucket",
                "target_observation_sequence_id",
                "actual",
                "champion_prediction",
                "benchmark_prediction",
            ],
        ].rename(
            columns={
                "actual": "actual_weight",
                "champion_prediction": "forecast_weight",
                "benchmark_prediction": "benchmark_weight",
            }
        )

        joined = target.merge(
            weights,
            on=[
                "horizon",
                "taxi_zone_id",
                "target_date",
                "target_temporal_bucket",
                "target_observation_sequence_id",
            ],
            how="inner",
            validate="one_to_one",
        )

        rows = []

        for keys, group in joined.groupby(
            grouping,
            observed=True,
            sort=False,
        ):
            (
                horizon,
                target_date,
                bucket,
                sequence_id,
            ) = keys

            rows.append(
                {
                    "horizon": horizon,
                    "target_date": target_date,
                    "target_temporal_bucket": bucket,
                    "target_observation_sequence_id": sequence_id,
                    "actual": _weighted_average(
                        group,
                        "actual",
                        "actual_weight",
                    ),
                    "forecast": _weighted_average(
                        group,
                        "champion_prediction",
                        "forecast_weight",
                    ),
                    "benchmark": _weighted_average(
                        group,
                        "benchmark_prediction",
                        "benchmark_weight",
                    ),
                    "source_zones": group["taxi_zone_id"].nunique(),
                }
            )

        aggregated = pd.DataFrame(rows)

    if aggregated.empty:
        return aggregated

    actual_wide = aggregated.pivot(
        index=[
            "target_date",
            "target_temporal_bucket",
            "target_observation_sequence_id",
        ],
        columns="horizon",
        values="actual",
    )
    forecast_wide = aggregated.pivot(
        index=[
            "target_date",
            "target_temporal_bucket",
            "target_observation_sequence_id",
        ],
        columns="horizon",
        values="forecast",
    ).rename(
        columns={
            1: "forecast_h1",
            2: "forecast_h2",
            5: "forecast_h5",
        }
    )
    benchmark_wide = aggregated.pivot(
        index=[
            "target_date",
            "target_temporal_bucket",
            "target_observation_sequence_id",
        ],
        columns="horizon",
        values="benchmark",
    )

    result = (
        pd.DataFrame(
            {
                "actual": actual_wide.mean(axis=1),
                "benchmark": benchmark_wide.mean(axis=1),
            }
        )
        .join(
            forecast_wide,
            how="inner",
        )
        .reset_index()
    )

    result["plot_time"] = ordered_plot_time(
        result["target_date"],
        result["target_temporal_bucket"],
    )

    return result.sort_values(
        "target_observation_sequence_id"
    ).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_selected_stress_anomaly_events() -> pd.DataFrame:
    """Load every canonical selected stress anomaly used by the Showcase."""
    events = load_selected_anomaly_events().copy()

    if events.empty:
        return events

    required = [
        "taxi_zone_id",
        "date",
        "temporal_bucket",
        "zone",
        "event_metric_driver_list",
    ]
    missing = sorted(
        set(required) - set(events.columns)
    )

    if missing:
        raise KeyError(
            "Selected anomaly events are missing required columns: "
            f"{missing}"
        )

    events["date"] = pd.to_datetime(
        events["date"],
        errors="coerce",
    )
    events["taxi_zone_id"] = pd.to_numeric(
        events["taxi_zone_id"],
        errors="coerce",
    ).astype("Int64")

    return events.loc[
        events["date"].notna()
        & events["taxi_zone_id"].notna()
    ].copy()


def metric_is_event_driver(
    driver_list: object,
    metric: str,
) -> bool:
    """Match one metric exactly inside the stored event-driver string."""
    if pd.isna(driver_list):
        return False

    driver_text = str(driver_list).strip()

    if not driver_text:
        return False

    pattern = (
        rf"(?<![A-Za-z0-9_])"
        rf"{re.escape(metric)}"
        rf"(?![A-Za-z0-9_])"
    )

    return re.search(
        pattern,
        driver_text,
    ) is not None


def summarize_anomaly_zones(
    values: pd.Series,
    limit: int = 8,
) -> str:
    """Keep aggregate-geography anomaly hovers useful without becoming huge."""
    names = sorted(
        {
            str(value)
            for value in values.dropna()
            if str(value).strip()
        }
    )

    if not names:
        return "Unknown"

    if len(names) <= limit:
        return ", ".join(names)

    shown = ", ".join(names[:limit])
    return f"{shown}, +{len(names) - limit} more"


def anomaly_overlay(
    series: pd.DataFrame,
    *,
    metric: str,
    zone_ids: set[int],
    day_type: str,
    daypart: str,
    start_date,
    end_date,
) -> pd.DataFrame:
    """
    Align metric-relevant stress anomalies to the displayed forecast observations.

    A marker is shown only when the mobility measure currently on screen was a
    defining driver of the selected stress anomaly. Aggregate geographies retain
    the affected Taxi Zone names so the marker still explains where it occurred.
    """
    events = load_selected_stress_anomaly_events()

    if events.empty:
        return pd.DataFrame()

    allowed_buckets = selected_temporal_buckets(
        day_type,
        daypart,
    )

    events = events.loc[
        events["taxi_zone_id"].isin(zone_ids)
        & events["date"].dt.date.between(
            start_date,
            end_date,
        )
        & events["temporal_bucket"].isin(
            allowed_buckets
        )
    ].copy()

    if events.empty:
        return pd.DataFrame()

    events["selected_metric_is_driver"] = events[
        "event_metric_driver_list"
    ].map(
        lambda value: metric_is_event_driver(
            value,
            metric,
        )
    )

    # The overlay is metric-specific: an anomaly is relevant here only when
    # the mobility measure currently on screen was one of its defining drivers.
    events = events.loc[
        events["selected_metric_is_driver"]
    ].copy()

    if events.empty:
        return pd.DataFrame()

    # One anomaly event is defined at Taxi Zone × date × daypart grain. The
    # defensive de-duplication keeps aggregate counts stable if display aliases
    # ever create duplicate rows upstream.
    events = events.drop_duplicates(
        [
            "date",
            "temporal_bucket",
            "taxi_zone_id",
        ]
    )

    summary = (
        events
        .groupby(
            ["date", "temporal_bucket"],
            observed=True,
            as_index=False,
        )
        .agg(
            anomaly_zone_count=(
                "taxi_zone_id",
                "nunique",
            ),
            anomaly_zone_names=(
                "zone",
                summarize_anomaly_zones,
            ),
            metric_driver_zone_count=(
                "selected_metric_is_driver",
                "sum",
            ),
        )
        .rename(
            columns={
                "date": "target_date",
                "temporal_bucket": "target_temporal_bucket",
            }
        )
    )

    summary["metric_driver_zone_count"] = pd.to_numeric(
        summary["metric_driver_zone_count"],
        errors="coerce",
    ).fillna(0).astype(int)

    def driver_status(row: pd.Series) -> str:
        driver_count = int(row["metric_driver_zone_count"])
        zone_count = int(row["anomaly_zone_count"])

        if driver_count == 0:
            return (
                f"{metric_label(metric)} was not an event driver in the "
                "affected Taxi Zones."
            )

        return (
            f"{metric_label(metric)} was an event driver in "
            f"{driver_count} of {zone_count} affected Taxi Zones."
        )

    summary["metric_driver_status"] = summary.apply(
        driver_status,
        axis=1,
    )

    return series.merge(
        summary,
        on=[
            "target_date",
            "target_temporal_bucket",
        ],
        how="inner",
    )


def explorer_figure(
    series: pd.DataFrame,
    *,
    metric: str,
    geography_label: str,
    visible_horizons: set[int],
    show_benchmark: bool,
    anomaly_rows: pd.DataFrame | None = None,
) -> go.Figure:
    """Render the interactive trajectory using the same grammar as the hero."""
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=series["plot_time"],
            y=series["actual"],
            mode="lines",
            name="Observed",
            line={
                "color": "#263238",
                "width": 3.1,
            },
            hovertemplate=(
                "<b>%{x|%b %d, %Y %H:%M}</b><br>"
                "Observed: %{y:,.2f}<extra></extra>"
            ),
        )
    )

    for horizon in HORIZONS:
        if horizon not in visible_horizons:
            continue

        column = f"forecast_h{horizon}"

        if column not in series.columns:
            continue

        figure.add_trace(
            go.Scatter(
                x=series["plot_time"],
                y=series[column],
                mode="lines",
                name=f"h={horizon}",
                line={
                    "color": HORIZON_COLORS[horizon],
                    "width": 2.0,
                },
                hovertemplate=(
                    "<b>%{x|%b %d, %Y %H:%M}</b><br>"
                    f"h={horizon} forecast: "
                    "%{y:,.2f}<extra></extra>"
                ),
            )
        )

    if show_benchmark:
        figure.add_trace(
            go.Scatter(
                x=series["plot_time"],
                y=series["benchmark"],
                mode="lines",
                name="Last-week baseline",
                line={
                    "color": "#7A878C",
                    "width": 1.5,
                    "dash": "dot",
                },
                hovertemplate=(
                    "<b>%{x|%b %d, %Y %H:%M}</b><br>"
                    "Last-week baseline: %{y:,.2f}"
                    "<extra></extra>"
                ),
            )
        )

    if anomaly_rows is not None and not anomaly_rows.empty:
        marker_sizes = (
            8
            + 2
            * np.log1p(
                anomaly_rows["anomaly_zone_count"]
            )
        )

        figure.add_trace(
            go.Scatter(
                x=anomaly_rows["plot_time"],
                y=anomaly_rows["actual"],
                mode="markers",
                name="Selected stress anomaly",
                marker={
                    "symbol": "diamond-open",
                    "size": marker_sizes,
                    "color": BRAND_COLORS["terracotta"],
                    "line": {
                        "width": 2,
                        "color": BRAND_COLORS["terracotta"],
                    },
                },
                customdata=anomaly_rows[
                    [
                        "anomaly_zone_count",
                        "anomaly_zone_names",
                        "metric_driver_status",
                    ]
                ].to_numpy(),
                hovertemplate=(
                    "<b>Stress-anomaly context</b><br>"
                    "%{x|%b %d, %Y %H:%M}<br>"
                    "Observed: %{y:,.2f}<br>"
                    "Affected Taxi Zones (%{customdata[0]:.0f}): "
                    "%{customdata[1]}<br>"
                    "%{customdata[2]}"
                    "<extra></extra>"
                ),
            )
        )

    figure = add_evaluation_boundary(figure, series)

    figure = apply_branding(figure)
    figure = strip_plotly_title_artifacts(figure)
    figure.update_layout(
        title={"text": "\u200b"},
        height=520,
        hovermode="x unified",
        margin=dict(
            t=55,
            r=25,
            b=55,
            l=75,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.015,
            xanchor="left",
            x=0,
        ),
        xaxis_title="",
        yaxis_title=metric_label(metric),
    )

    return figure


def hero_error_figure(
    weekly: pd.DataFrame,
    *,
    visible_horizons: set[int],
    show_benchmark: bool,
) -> go.Figure:
    """Show signed weekly forecast error behind the primary Manhattan hero."""
    figure = go.Figure()

    denominator = weekly["actual"].replace(0, np.nan)

    for horizon in HORIZONS:
        if horizon not in visible_horizons:
            continue

        error = (
            100
            * (
                weekly[f"forecast_h{horizon}"]
                - weekly["actual"]
            )
            / denominator
        )

        formatted_error = error.map(
            lambda value: (
                f"{value:+.2f}%"
                if pd.notna(value)
                else "—"
            )
        )

        figure.add_trace(
            go.Scatter(
                x=weekly["week_start"],
                y=error,
                mode="lines+markers",
                name=f"h={horizon}",
                line={
                    "color": HORIZON_COLORS[horizon],
                    "width": 2.2,
                },
                marker={"size": 5},
                customdata=formatted_error.to_numpy(),
                hovertemplate=(
                    "<b>%{x|%b %d, %Y}</b><br>"
                    f"h={horizon} signed error: "
                    "%{customdata}<extra></extra>"
                ),
            )
        )

    if show_benchmark:
        benchmark_error = (
            100
            * (
                weekly["benchmark"]
                - weekly["actual"]
            )
            / denominator
        )

        formatted_benchmark_error = benchmark_error.map(
            lambda value: (
                f"{value:+.2f}%"
                if pd.notna(value)
                else "—"
            )
        )

        figure.add_trace(
            go.Scatter(
                x=weekly["week_start"],
                y=benchmark_error,
                mode="lines",
                name="Last-week baseline",
                line={
                    "color": "#7A878C",
                    "width": 1.6,
                    "dash": "dot",
                },
                customdata=formatted_benchmark_error.to_numpy(),
                hovertemplate=(
                    "<b>%{x|%b %d, %Y}</b><br>"
                    "Baseline signed error: %{customdata}"
                    "<extra></extra>"
                ),
            )
        )

    figure.add_hline(
        y=0,
        line={
            "color": "#607D84",
            "width": 1.1,
            "dash": "dash",
        },
    )

    figure = apply_branding(figure)
    figure = strip_plotly_title_artifacts(figure)
    figure.update_layout(
        title={
            "text": "\u200b",
        },
        height=505,
        hovermode="x unified",
        margin=dict(
            t=58,
            r=35,
            b=60,
            l=80,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.015,
            xanchor="left",
            x=0,
        ),
        xaxis_title="",
        yaxis_title="Signed forecast error",
    )
    figure.update_yaxes(
        ticksuffix="%",
        zeroline=False,
    )

    return figure



@st.cache_data(
    show_spinner="Loading the selected geography for Quilt views..."
)
def load_personalized_quilt_records(
    zone_ids: tuple[int, ...],
) -> pd.DataFrame:
    """
    Load all five targets for the selected geography from compact history.

    The returned shape intentionally matches the original longitudinal Chapter
    4 handoff so the existing Quilt calculations remain unchanged.
    """
    if not zone_ids:
        return pd.DataFrame(
            columns=EXPLORER_RECORD_COLUMNS
        )

    return load_forecast_history(
        columns=EXPLORER_RECORD_COLUMNS,
        required_columns=EXPLORER_RECORD_COLUMNS,
        metrics=METRIC_ORDER,
        horizons=HORIZONS,
        taxi_zone_ids=zone_ids,
        reader_facing_only=True,
    )


def build_personalized_quilt_series(
    records: pd.DataFrame,
    *,
    zone_ids: set[int],
    day_type: str,
    daypart: str,
    start_date,
    end_date,
) -> dict[str, pd.DataFrame]:
    """
    Build one aligned Actual / Forecast series for every supported metric.

    Unsupported metrics simply disappear from the personalized Quilt.
    """
    result: dict[str, pd.DataFrame] = {}

    for metric in METRIC_ORDER:
        series = aggregate_explorer_series(
            records,
            metric=metric,
            zone_ids=zone_ids,
            day_type=day_type,
            daypart=daypart,
            start_date=start_date,
            end_date=end_date,
        )

        if not series.empty:
            result[metric] = series

    return result


def personalized_forecast_quilt(
    series_by_metric: dict[str, pd.DataFrame],
) -> go.Figure:
    """
    Show the selected geography's Actual-vs-Forecast Quilt across all metrics.

    The three horizon cells in each metric row share one y-range.
    """
    supported_metrics = [
        metric
        for metric in METRIC_ORDER
        if metric in series_by_metric
        and not series_by_metric[metric].empty
    ]

    if not supported_metrics:
        return go.Figure()

    figure = make_subplots(
        rows=len(supported_metrics),
        cols=len(HORIZONS),
        horizontal_spacing=0.032,
        vertical_spacing=0.055,
    )

    for row_number, metric in enumerate(
        supported_metrics,
        start=1,
    ):
        series = series_by_metric[metric]

        value_columns = [
            "actual",
            *[
                f"forecast_h{horizon}"
                for horizon in HORIZONS
                if f"forecast_h{horizon}" in series.columns
            ],
        ]

        values = (
            series[value_columns]
            .stack()
            .dropna()
        )

        if values.empty:
            y_min, y_max = 0.0, 1.0
        else:
            minimum = float(values.min())
            maximum = float(values.max())
            span = max(
                maximum - minimum,
                abs(maximum) * 0.05,
                1e-9,
            )
            y_min = minimum - 0.10 * span
            y_max = maximum + 0.10 * span

        value_decimals = (
            2
            if metric in SPEED_WEIGHT_METRIC
            else 0
        )

        observed_text = series["actual"].map(
            lambda value: (
                f"{value:,.{value_decimals}f}"
                if pd.notna(value)
                else "—"
            )
        )

        for column_number, horizon in enumerate(
            HORIZONS,
            start=1,
        ):
            forecast_column = f"forecast_h{horizon}"

            if forecast_column not in series.columns:
                continue

            forecast_text = series[forecast_column].map(
                lambda value: (
                    f"{value:,.{value_decimals}f}"
                    if pd.notna(value)
                    else "—"
                )
            )
            difference_text = (
                series[forecast_column]
                - series["actual"]
            ).map(
                lambda value: (
                    f"{value:+,.{value_decimals}f}"
                    if pd.notna(value)
                    else "—"
                )
            )
            hover_data = np.column_stack(
                [
                    observed_text,
                    forecast_text,
                    difference_text,
                ]
            )
            hover_template = (
                "<b>%{x|%b %d, %Y %H:%M}</b><br>"
                "Observed: %{customdata[0]}<br>"
                f"h={horizon} forecast: %{{customdata[1]}}<br>"
                "Difference: %{customdata[2]}"
                "<extra></extra>"
            )

            figure.add_trace(
                go.Scatter(
                    x=series["plot_time"],
                    y=series["actual"],
                    mode="lines",
                    name="Observed",
                    legendgroup="observed",
                    showlegend=(
                        row_number == 1
                        and column_number == 1
                    ),
                    line={
                        "color": "#263238",
                        "width": 2.4,
                    },
                    customdata=hover_data,
                    hovertemplate=hover_template,
                ),
                row=row_number,
                col=column_number,
            )

            figure.add_trace(
                go.Scatter(
                    x=series["plot_time"],
                    y=series[forecast_column],
                    mode="lines",
                    name=f"h={horizon} forecast",
                    legendgroup=f"h{horizon}",
                    showlegend=(
                        row_number == 1
                    ),
                    line={
                        "color": HORIZON_COLORS[horizon],
                        "width": 2.0,
                    },
                    customdata=hover_data,
                    hovertemplate=hover_template,
                ),
                row=row_number,
                col=column_number,
            )

            figure.update_yaxes(
                range=[y_min, y_max],
                showgrid=True,
                gridcolor="rgba(0, 109, 119, 0.10)",
                showticklabels=(
                    column_number == 1
                ),
                tickfont={"size": 8},
                fixedrange=True,
                row=row_number,
                col=column_number,
            )
            figure.update_xaxes(
                showgrid=False,
                showticklabels=True,
                tickformat="%b %d",
                nticks=3,
                ticks="inside",
                ticklen=3,
                ticklabelposition="inside",
                ticklabelstandoff=-6,
                tickfont={
                    "size": 7.5,
                    "color": "#60767B",
                },
                fixedrange=True,
                row=row_number,
                col=column_number,
            )

    top_y = getattr(
        figure.layout,
        "yaxis",
    ).domain[1]

    for column_number, horizon in enumerate(
        HORIZONS,
        start=1,
    ):
        xaxis_name = (
            "xaxis"
            if column_number == 1
            else f"xaxis{column_number}"
        )
        x_domain = getattr(
            figure.layout,
            xaxis_name,
        ).domain

        figure.add_annotation(
            x=sum(x_domain) / 2,
            y=top_y + 0.03,
            xref="paper",
            yref="paper",
            text=f"<b>h = {horizon}</b>",
            showarrow=False,
            font={
                "size": 13,
                "color": BRAND_COLORS["dark_teal"],
            },
        )

    for row_number, metric in enumerate(
        supported_metrics,
        start=1,
    ):
        axis_index = (
            (row_number - 1) * len(HORIZONS)
            + 1
        )
        yaxis_name = (
            "yaxis"
            if axis_index == 1
            else f"yaxis{axis_index}"
        )
        y_domain = getattr(
            figure.layout,
            yaxis_name,
        ).domain

        figure.add_annotation(
            x=-0.03,
            y=sum(y_domain) / 2,
            xref="paper",
            yref="paper",
            text=(
                f"<b>{metric_label(metric)}</b><br>"
                f"<span style='font-size:9px'>"
                f"{METRIC_DESCRIPTIONS[metric]}"
                "</span>"
            ),
            showarrow=False,
            xanchor="right",
            align="right",
            font={
                "size": 12,
                "color": "#003F46",
            },
        )

    figure = apply_branding(figure)
    figure = strip_plotly_title_artifacts(figure)

    figure.update_layout(
        height=max(
            390,
            135 * len(supported_metrics) + 115,
        ),
        margin=dict(
            t=55,
            r=20,
            b=45,
            l=205,
        ),
        hovermode="closest",
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.035,
            xanchor="left",
            x=0,
            font={"size": 10},
        ),
    )

    return figure


def personalized_accuracy_quilt(
    series_by_metric: dict[str, pd.DataFrame],
) -> go.Figure:
    """
    Show a personalized 5 × 3 accuracy Quilt for the selected geography/window.

    Large number = Relative MAE for the displayed view.
    Secondary number = improvement over the displayed last-week baseline.
    Mini line = signed percentage error through the displayed observations.
    """
    supported_metrics = [
        metric
        for metric in METRIC_ORDER
        if metric in series_by_metric
        and not series_by_metric[metric].empty
    ]

    if not supported_metrics:
        return go.Figure()

    card_rows = []

    for metric in supported_metrics:
        series = series_by_metric[metric]
        denominator = series["actual"].abs().sum()

        for horizon in HORIZONS:
            forecast_column = f"forecast_h{horizon}"

            if (
                forecast_column not in series.columns
                or denominator <= 0
            ):
                continue

            model_error = (
                series[forecast_column]
                - series["actual"]
            ).abs().sum()

            benchmark_error = (
                series["benchmark"]
                - series["actual"]
            ).abs().sum()

            relative_mae = (
                100 * model_error / denominator
            )

            benchmark_skill = (
                100
                * (
                    benchmark_error
                    - model_error
                )
                / benchmark_error
                if benchmark_error > 0
                else np.nan
            )

            signed_error = (
                100
                * (
                    series[forecast_column]
                    - series["actual"]
                )
                / series["actual"].replace(
                    0,
                    np.nan,
                )
            )

            card_rows.append(
                {
                    "metric": metric,
                    "horizon": horizon,
                    "relative_mae_pct": relative_mae,
                    "benchmark_skill_pct": benchmark_skill,
                    "signed_error": signed_error,
                    "plot_time": series["plot_time"],
                }
            )

    if not card_rows:
        return go.Figure()

    all_mae = np.array(
        [
            row["relative_mae_pct"]
            for row in card_rows
            if np.isfinite(
                row["relative_mae_pct"]
            )
        ]
    )

    minimum = float(
        np.nanmin(all_mae)
    )
    maximum = float(
        np.nanmax(all_mae)
    )
    span = max(
        maximum - minimum,
        1e-9,
    )

    figure = make_subplots(
        rows=len(supported_metrics),
        cols=len(HORIZONS),
        horizontal_spacing=0.032,
        vertical_spacing=0.038,
    )

    metadata = []

    for row_number, metric in enumerate(
        supported_metrics,
        start=1,
    ):
        metric_cards = [
            row
            for row in card_rows
            if row["metric"] == metric
        ]

        abs_errors = pd.concat(
            [
                row["signed_error"].abs()
                for row in metric_cards
            ],
            ignore_index=True,
        ).dropna()

        y_limit = (
            max(
                float(
                    abs_errors.quantile(0.95)
                ),
                1.0,
            )
            if not abs_errors.empty
            else 1.0
        )

        for column_number, horizon in enumerate(
            HORIZONS,
            start=1,
        ):
            matches = [
                row
                for row in metric_cards
                if row["horizon"] == horizon
            ]

            if not matches:
                continue

            card = matches[0]
            signed_error = card["signed_error"].clip(
                lower=-y_limit,
                upper=y_limit,
            )

            formatted_error = card[
                "signed_error"
            ].map(
                lambda value: (
                    f"{value:+.2f}%"
                    if pd.notna(value)
                    else "—"
                )
            )

            error_weight = (
                card["relative_mae_pct"]
                - minimum
            ) / span

            background = mix_colors(
                BRAND_COLORS["ice"],
                BRAND_COLORS["pale_peach"],
                error_weight,
            )

            figure.add_trace(
                go.Scatter(
                    x=card["plot_time"],
                    y=signed_error,
                    mode="lines",
                    line={
                        "color": BRAND_COLORS[
                            "dark_teal"
                        ],
                        "width": 1.7,
                    },
                    fill="tozeroy",
                    fillcolor="rgba(131, 197, 190, 0.24)",
                    showlegend=False,
                    customdata=formatted_error.to_numpy(),
                    hovertemplate=(
                        "<b>%{x|%b %d, %Y %H:%M}</b><br>"
                        "Signed error: %{customdata}"
                        "<extra></extra>"
                    ),
                ),
                row=row_number,
                col=column_number,
            )

            figure.add_hline(
                y=0,
                line={
                    "color": "#607D84",
                    "width": 0.8,
                },
                row=row_number,
                col=column_number,
            )

            figure.update_xaxes(
                showgrid=False,
                showticklabels=True,
                tickformat="%b %d",
                nticks=3,
                ticks="inside",
                ticklen=3,
                ticklabelposition="inside",
                ticklabelstandoff=-6,
                tickfont={
                    "size": 7.5,
                    "color": "#60767B",
                },
                fixedrange=True,
                row=row_number,
                col=column_number,
            )
            figure.update_yaxes(
                range=[
                    -y_limit,
                    y_limit,
                ],
                showgrid=False,
                showticklabels=False,
                fixedrange=True,
                row=row_number,
                col=column_number,
            )

            metadata.append(
                {
                    "row": row_number,
                    "col": column_number,
                    "metric": metric,
                    "horizon": horizon,
                    "relative_mae_pct": card[
                        "relative_mae_pct"
                    ],
                    "benchmark_skill_pct": card[
                        "benchmark_skill_pct"
                    ],
                    "background": background,
                    "error_limit": y_limit,
                }
            )

    for card in metadata:
        axis_index = (
            (card["row"] - 1) * len(HORIZONS)
            + card["col"]
        )

        xaxis_name = (
            "xaxis"
            if axis_index == 1
            else f"xaxis{axis_index}"
        )
        yaxis_name = (
            "yaxis"
            if axis_index == 1
            else f"yaxis{axis_index}"
        )

        x_domain = getattr(
            figure.layout,
            xaxis_name,
        ).domain
        y_domain = getattr(
            figure.layout,
            yaxis_name,
        ).domain

        figure.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=x_domain[0],
            x1=x_domain[1],
            y0=y_domain[0],
            y1=y_domain[1],
            fillcolor=card["background"],
            line={
                "color": "rgba(0, 109, 119, 0.24)",
                "width": 1,
            },
            layer="below",
        )

        width = x_domain[1] - x_domain[0]
        height = y_domain[1] - y_domain[0]

        figure.add_annotation(
            x=x_domain[0] + 0.055 * width,
            y=y_domain[1] - 0.07 * height,
            xref="paper",
            yref="paper",
            text=(
                f"<b>{card['relative_mae_pct']:.1f}%</b>"
            ),
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font={
                "size": 18,
                "color": "#003F46",
            },
        )

        figure.add_annotation(
            x=x_domain[0] + 0.055 * width,
            y=y_domain[1] - 0.28 * height,
            xref="paper",
            yref="paper",
            text=(
                f"<b>{card['benchmark_skill_pct']:+.1f}%</b> "
                "vs baseline"
            ),
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font={
                "size": 9.5,
                "color": "#335C67",
            },
        )

    top_y = getattr(
        figure.layout,
        "yaxis",
    ).domain[1]

    for column_number, horizon in enumerate(
        HORIZONS,
        start=1,
    ):
        xaxis_name = (
            "xaxis"
            if column_number == 1
            else f"xaxis{column_number}"
        )
        x_domain = getattr(
            figure.layout,
            xaxis_name,
        ).domain

        figure.add_annotation(
            x=sum(x_domain) / 2,
            y=top_y + 0.028,
            xref="paper",
            yref="paper",
            text=f"<b>h = {horizon}</b>",
            showarrow=False,
            font={
                "size": 13,
                "color": BRAND_COLORS["dark_teal"],
            },
        )

    for row_number, metric in enumerate(
        supported_metrics,
        start=1,
    ):
        axis_index = (
            (row_number - 1) * len(HORIZONS)
            + 1
        )
        yaxis_name = (
            "yaxis"
            if axis_index == 1
            else f"yaxis{axis_index}"
        )
        y_domain = getattr(
            figure.layout,
            yaxis_name,
        ).domain

        row_cards = [
            row
            for row in metadata
            if row["metric"] == metric
        ]
        error_limit = (
            max(
                row["error_limit"]
                for row in row_cards
            )
            if row_cards
            else np.nan
        )

        figure.add_annotation(
            x=-0.03,
            y=sum(y_domain) / 2,
            xref="paper",
            yref="paper",
            text=(
                f"<b>{metric_label(metric)}</b><br>"
                f"<span style='font-size:9px'>"
                f"{METRIC_DESCRIPTIONS[metric]}</span><br>"
                f"<span style='font-size:8px;color:#60767B'>"
                f"Signed error display: ±{error_limit:.1f}%"
                "</span>"
            ),
            showarrow=False,
            xanchor="right",
            align="right",
            font={
                "size": 12,
                "color": "#003F46",
            },
        )

    figure = apply_branding(figure)
    figure = strip_plotly_title_artifacts(
        figure
    )

    figure.update_layout(
        height=max(
            390,
            135 * len(supported_metrics) + 115,
        ),
        margin=dict(
            t=48,
            r=20,
            b=22,
            l=225,
        ),
        paper_bgcolor="white",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )

    return figure


def explorer_error_figure(
    series: pd.DataFrame,
    *,
    visible_horizons: set[int],
    show_benchmark: bool,
    anomaly_rows: pd.DataFrame | None = None,
) -> go.Figure:
    """Personalized signed-error view for the selected explorer series."""
    figure = go.Figure()
    denominator = series["actual"].replace(
        0,
        np.nan,
    )

    for horizon in HORIZONS:
        if horizon not in visible_horizons:
            continue

        forecast_column = f"forecast_h{horizon}"

        if forecast_column not in series.columns:
            continue

        error = (
            100
            * (
                series[forecast_column]
                - series["actual"]
            )
            / denominator
        )

        formatted = error.map(
            lambda value: (
                f"{value:+.2f}%"
                if pd.notna(value)
                else "—"
            )
        )

        figure.add_trace(
            go.Scatter(
                x=series["plot_time"],
                y=error,
                mode="lines",
                name=f"h={horizon}",
                line={
                    "color": HORIZON_COLORS[
                        horizon
                    ],
                    "width": 2.0,
                },
                customdata=formatted.to_numpy(),
                hovertemplate=(
                    "<b>%{x|%b %d, %Y %H:%M}</b><br>"
                    f"h={horizon} signed error: "
                    "%{customdata}<extra></extra>"
                ),
            )
        )

    if show_benchmark:
        benchmark_error = (
            100
            * (
                series["benchmark"]
                - series["actual"]
            )
            / denominator
        )

        formatted_benchmark = benchmark_error.map(
            lambda value: (
                f"{value:+.2f}%"
                if pd.notna(value)
                else "—"
            )
        )

        figure.add_trace(
            go.Scatter(
                x=series["plot_time"],
                y=benchmark_error,
                mode="lines",
                name="Last-week baseline",
                line={
                    "color": "#7A878C",
                    "width": 1.5,
                    "dash": "dot",
                },
                customdata=formatted_benchmark.to_numpy(),
                hovertemplate=(
                    "<b>%{x|%b %d, %Y %H:%M}</b><br>"
                    "Baseline signed error: %{customdata}"
                    "<extra></extra>"
                ),
            )
        )

    figure.add_hline(
        y=0,
        line={
            "color": "#607D84",
            "width": 1.0,
            "dash": "dash",
        },
    )

    if (
        anomaly_rows is not None
        and not anomaly_rows.empty
    ):
        # Mark anomaly times at zero so the event context does not imply one
        # particular horizon caused or explained the anomaly.
        figure.add_trace(
            go.Scatter(
                x=anomaly_rows["plot_time"],
                y=np.zeros(
                    len(anomaly_rows)
                ),
                mode="markers",
                name="Selected stress anomaly",
                marker={
                    "symbol": "diamond-open",
                    "size": 9,
                    "color": BRAND_COLORS[
                        "terracotta"
                    ],
                    "line": {
                        "width": 2,
                        "color": BRAND_COLORS[
                            "terracotta"
                        ],
                    },
                },
                customdata=anomaly_rows[
                    [
                        "anomaly_zone_count",
                        "anomaly_zone_names",
                        "metric_driver_status",
                    ]
                ].to_numpy(),
                hovertemplate=(
                    "<b>Stress-anomaly context</b><br>"
                    "%{x|%b %d, %Y %H:%M}<br>"
                    "Affected Taxi Zones (%{customdata[0]:.0f}): "
                    "%{customdata[1]}<br>"
                    "%{customdata[2]}"
                    "<extra></extra>"
                ),
            )
        )

    figure = add_evaluation_boundary(figure, series)

    figure = apply_branding(figure)
    figure = strip_plotly_title_artifacts(figure)
    figure.update_layout(
        title={
            "text": "\u200b",
        },
        height=520,
        hovermode="x unified",
        margin=dict(
            t=55,
            r=25,
            b=55,
            l=75,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.015,
            xanchor="left",
            x=0,
        ),
        xaxis_title="",
        yaxis_title="Signed forecast error",
    )
    figure.update_yaxes(
        ticksuffix="%",
    )

    return figure


def relative_mae_for_series(
    series: pd.DataFrame,
    forecast_column: str,
) -> float:
    """Return the same denominator-safe Relative MAE used throughout the page."""
    denominator = series["actual"].abs().sum()

    if denominator <= 0 or forecast_column not in series.columns:
        return np.nan

    error_sum = (
        series[forecast_column]
        - series["actual"]
    ).abs().sum()

    return float(
        100 * error_sum / denominator
    )


def benchmark_skill_for_series(
    series: pd.DataFrame,
    forecast_column: str,
) -> float:
    """Return improvement over the displayed last-week baseline."""
    if forecast_column not in series.columns:
        return np.nan

    model_error = (
        series[forecast_column]
        - series["actual"]
    ).abs().sum()
    benchmark_error = (
        series["benchmark"]
        - series["actual"]
    ).abs().sum()

    if benchmark_error <= 0:
        return np.nan

    return float(
        100
        * (
            benchmark_error
            - model_error
        )
        / benchmark_error
    )


def pooled_relative_mae(
    series: pd.DataFrame,
    horizons: set[int],
    mask: pd.Series,
) -> float:
    """
    Pool visible-horizon absolute error over the selected observations.

    This preserves the Relative-MAE denominator logic rather than averaging
    volatile row-level percentages.
    """
    subset = series.loc[mask].copy()

    if subset.empty or not horizons:
        return np.nan

    numerator = 0.0
    denominator = 0.0

    for horizon in sorted(horizons):
        column = f"forecast_h{horizon}"

        if column not in subset.columns:
            continue

        valid = (
            subset["actual"].notna()
            & subset[column].notna()
        )

        if not valid.any():
            continue

        numerator += (
            subset.loc[valid, column]
            - subset.loc[valid, "actual"]
        ).abs().sum()

        denominator += (
            subset.loc[valid, "actual"]
            .abs()
            .sum()
        )

    if denominator <= 0:
        return np.nan

    return float(
        100 * numerator / denominator
    )


def build_hero_dynamic_insight(
    hero_stats: dict[int, dict[str, float]],
    visible_horizons: set[int],
    weekly: pd.DataFrame,
    *,
    show_baseline: bool,
) -> str:
    """
    Explain whether the forecasts tracked observed Manhattan Subway ridership.

    The takeaway leads with the forecasting result itself. Error metrics and
    benchmark comparisons support that interpretation instead of becoming the
    headline.
    """
    available = [
        horizon
        for horizon in sorted(visible_horizons)
        if horizon in hero_stats
        and np.isfinite(
            hero_stats[horizon]["relative_mae_pct"]
        )
    ]

    if not available:
        return (
            "Observed ridership remains visible. Select at least one forecast "
            "horizon to compare how closely the system tracked Manhattan."
        )

    best_horizon = min(
        available,
        key=lambda horizon: hero_stats[horizon]["relative_mae_pct"],
    )
    worst_horizon = max(
        available,
        key=lambda horizon: hero_stats[horizon]["relative_mae_pct"],
    )

    best_error = float(
        hero_stats[best_horizon]["relative_mae_pct"]
    )
    worst_error = float(
        hero_stats[worst_horizon]["relative_mae_pct"]
    )

    single_horizon = len(available) == 1

    # WHY: lead with what happened, not with a ranking of error statistics.
    if single_horizon:
        tracking_sentence = (
            f"The **h={best_horizon} forecast stayed close to observed Manhattan "
            f"Subway ridership** during the untouched final holdout, with "
            f"**{best_error:.1f}% Relative MAE**."
        )
    else:
        tracking_sentence = (
            "The forecasting system stayed **close to observed Manhattan Subway "
            "ridership across all visible forecast horizons** during the untouched "
            "final holdout. "
            f"The closest was **h={best_horizon} at {best_error:.1f}% Relative "
            f"MAE**, while even **h={worst_horizon} remained at "
            f"{worst_error:.1f}%**."
        )

    baseline_sentence = ""

    if show_baseline:
        skill = float(
            hero_stats[best_horizon]["benchmark_skill_pct"]
        )

        if np.isfinite(skill):
            if skill > 0:
                baseline_sentence = (
                    f" At h={best_horizon}, that forecast reduced absolute error "
                    f"by **{skill:.1f}%** compared with simply repeating the "
                    "previous week's ridership."
                )
            elif skill < 0:
                baseline_sentence = (
                    f" At h={best_horizon}, simply repeating the previous week's "
                    f"ridership produced **{abs(skill):.1f}% less error** than "
                    "the selected forecast."
                )
            else:
                baseline_sentence = (
                    f" At h={best_horizon}, the forecast and the simple "
                    "Last-week baseline produced the same total absolute error."
                )

    forecast_column = f"forecast_h{best_horizon}"

    valid = (
        weekly["actual"].notna()
        & weekly[forecast_column].notna()
        & weekly["actual"].abs().gt(0)
    )

    miss_sentence = ""

    if valid.any():
        signed_pct = (
            100
            * (
                weekly.loc[valid, forecast_column]
                - weekly.loc[valid, "actual"]
            )
            / weekly.loc[valid, "actual"].abs()
        )

        miss_index = signed_pct.abs().idxmax()
        miss_value = float(
            signed_pct.loc[miss_index]
        )
        miss_week = pd.Timestamp(
            weekly.loc[miss_index, "week_start"]
        )

        direction = (
            "above"
            if miss_value > 0
            else "below"
        )

        miss_sentence = (
            f" Its largest weekly miss began **{miss_week:%b %d}**, when the "
            f"h={best_horizon} forecast was **{abs(miss_value):.1f}% "
            f"{direction}** the ridership actually observed."
        )

    return (
        tracking_sentence
        + baseline_sentence
        + miss_sentence
    )


def build_quilt_dynamic_insight(
    jobs: pd.DataFrame,
    zones: pd.DataFrame,
) -> str:
    """Summarize the strongest system-wide contrasts in the frozen holdout."""
    usable_jobs = jobs.loc[
        jobs["relative_mae_pct"].notna()
    ].copy()

    if usable_jobs.empty:
        return (
            "The frozen holdout does not contain enough mobility-measure × horizon accuracy "
            "information to summarize the Quilt."
        )

    best = usable_jobs.loc[
        usable_jobs["relative_mae_pct"].idxmin()
    ]
    worst = usable_jobs.loc[
        usable_jobs["relative_mae_pct"].idxmax()
    ]

    skill_rows = usable_jobs.loc[
        usable_jobs["benchmark_skill_pct"].notna()
    ]
    strongest_skill = (
        skill_rows.loc[
            skill_rows["benchmark_skill_pct"].idxmax()
        ]
        if not skill_rows.empty
        else None
    )

    zone_skill = (
        zones
        .groupby(
            ["metric", "horizon"],
            observed=True,
        )["benchmark_skill_pct"]
        .apply(
            lambda values: (
                100
                * pd.to_numeric(values, errors="coerce")
                .dropna()
                .gt(0)
                .mean()
            )
        )
        .rename("zone_share_beating_baseline")
        .reset_index()
    )
    most_consistent = (
        zone_skill.loc[
            zone_skill["zone_share_beating_baseline"].idxmax()
        ]
        if not zone_skill.empty
        else None
    )

    sentences = [
        (
            f"Across the final holdout, **{metric_label(best['metric'])} · "
            f"h={int(best['horizon'])}** has the lowest system Relative MAE "
            f"at **{float(best['relative_mae_pct']):.1f}%**, while "
            f"**{metric_label(worst['metric'])} · h={int(worst['horizon'])}** "
            f"is highest at **{float(worst['relative_mae_pct']):.1f}%**."
        )
    ]

    if strongest_skill is not None:
        skill = float(strongest_skill["benchmark_skill_pct"])
        sentences.append(
            f"The largest system-level gain over the last-week baseline is "
            f"**{metric_label(strongest_skill['metric'])} · "
            f"h={int(strongest_skill['horizon'])}**, with **{skill:.1f}% lower "
            "error**."
        )

    if most_consistent is not None:
        sentences.append(
            f"The broadest zone-level advantage appears for "
            f"**{metric_label(most_consistent['metric'])} · "
            f"h={int(most_consistent['horizon'])}**, where "
            f"**{float(most_consistent['zone_share_beating_baseline']):.0f}% "
            "of Taxi Zones** beat the baseline."
        )

    return " ".join(sentences)


def _largest_displayed_miss(
    series: pd.DataFrame,
    horizons: list[int],
) -> dict[str, object] | None:
    """Return the largest percentage miss across the visible horizons."""
    candidates: list[dict[str, object]] = []

    for horizon in horizons:
        column = f"forecast_h{horizon}"

        if column not in series.columns:
            continue

        valid = (
            series["actual"].notna()
            & series[column].notna()
            & series["actual"].abs().gt(0)
        )

        if not valid.any():
            continue

        signed_pct = (
            100
            * (
                series.loc[valid, column]
                - series.loc[valid, "actual"]
            )
            / series.loc[valid, "actual"].abs()
        )
        index = signed_pct.abs().idxmax()

        candidates.append(
            {
                "horizon": horizon,
                "signed_pct": float(signed_pct.loc[index]),
                "target_date": pd.Timestamp(
                    series.loc[index, "target_date"]
                ),
                "bucket": series.loc[
                    index,
                    "target_temporal_bucket",
                ],
            }
        )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: abs(float(item["signed_pct"])),
    )


def build_explorer_dynamic_insight(
    series: pd.DataFrame,
    *,
    metric: str,
    visible_horizons: set[int],
    anomaly_rows: pd.DataFrame | None,
    show_anomalies: bool,
    show_baseline: bool,
) -> str:
    """Build a genuinely data-driven takeaway for the current explorer state."""
    horizon_metrics: dict[int, dict[str, float]] = {}

    for horizon in sorted(visible_horizons):
        column = f"forecast_h{horizon}"

        if column not in series.columns:
            continue

        horizon_metrics[horizon] = {
            "relative_mae_pct": relative_mae_for_series(
                series,
                column,
            ),
            "benchmark_skill_pct": benchmark_skill_for_series(
                series,
                column,
            ),
        }

    finite_horizons = [
        horizon
        for horizon, values in horizon_metrics.items()
        if np.isfinite(values["relative_mae_pct"])
    ]

    if not finite_horizons:
        return (
            "Select at least one forecast horizon to calculate an accuracy "
            "takeaway for this view."
        )

    best_horizon = min(
        finite_horizons,
        key=lambda horizon: horizon_metrics[horizon]["relative_mae_pct"],
    )
    worst_horizon = max(
        finite_horizons,
        key=lambda horizon: horizon_metrics[horizon]["relative_mae_pct"],
    )
    best_error = horizon_metrics[best_horizon]["relative_mae_pct"]
    worst_error = horizon_metrics[worst_horizon]["relative_mae_pct"]
    spread = worst_error - best_error

    single_horizon = len(finite_horizons) == 1

    if single_horizon:
        horizon_sentence = (
            f"With **h={best_horizon}** selected, this view has "
            f"**{best_error:.1f}% Relative MAE**."
        )
    elif spread < 0.5:
        horizon_sentence = (
            f"Forecast distance changes little in this selection: the visible "
            f"horizons span only **{spread:.1f} percentage points**, from "
            f"**{best_error:.1f}%** at h={best_horizon} to "
            f"**{worst_error:.1f}%** at h={worst_horizon}."
        )
    else:
        horizon_sentence = (
            f"h={best_horizon} is closest overall at **{best_error:.1f}% Relative "
            f"MAE**; h={worst_horizon} is highest at **{worst_error:.1f}%**, a "
            f"**{spread:.1f}-point** spread."
        )

    baseline_sentence = ""
    if show_baseline:
        finite_skills = {
            horizon: horizon_metrics[horizon]["benchmark_skill_pct"]
            for horizon in finite_horizons
            if np.isfinite(
                horizon_metrics[horizon]["benchmark_skill_pct"]
            )
        }

        if finite_skills:
            winners = [
                horizon
                for horizon, skill in finite_skills.items()
                if skill > 0
            ]
            strongest_horizon = max(
                finite_skills,
                key=finite_skills.get,
            )
            strongest_skill = finite_skills[strongest_horizon]

            if single_horizon:
                selected_skill = finite_skills[best_horizon]

                if selected_skill > 0:
                    baseline_sentence = (
                        f" It reduces absolute error by **{selected_skill:.1f}%** "
                        "versus the last-week baseline."
                    )
                elif selected_skill < 0:
                    baseline_sentence = (
                        f" The last-week baseline has **{abs(selected_skill):.1f}% "
                        "lower error** for this selection."
                    )
                else:
                    baseline_sentence = (
                        " It has the same total absolute error as the last-week "
                        "baseline."
                    )
            elif len(winners) == len(finite_skills):
                baseline_sentence = (
                    f" All {len(winners)} visible horizons beat the last-week "
                    f"baseline; the largest reduction in error is "
                    f"**{strongest_skill:.1f}% at h={strongest_horizon}**."
                )
            elif not winners:
                best_baseline_gap_horizon = max(
                    finite_skills,
                    key=finite_skills.get,
                )
                gap = abs(finite_skills[best_baseline_gap_horizon])
                baseline_sentence = (
                    " The last-week baseline has lower error than every visible "
                    f"horizon; the closest challenge is h={best_baseline_gap_horizon}, "
                    f"which trails by **{gap:.1f}%**."
                )
            else:
                winner_text = ", ".join(
                    f"h={horizon}"
                    for horizon in winners
                )
                baseline_sentence = (
                    f" Against the last-week baseline, **{winner_text}** "
                    f"{'beats' if len(winners) == 1 else 'beat'} it, while the "
                    "other visible horizons do not."
                )

    dates = pd.to_datetime(
        series["target_date"],
        errors="coerce",
    ).dt.normalize()
    validation_mask = dates.lt(FINAL_HOLDOUT_START_DATE)
    holdout_mask = dates.ge(FINAL_HOLDOUT_START_DATE)
    context_sentence = ""

    if validation_mask.any() and holdout_mask.any():
        validation_error = pooled_relative_mae(
            series,
            set(finite_horizons),
            validation_mask,
        )
        holdout_error = pooled_relative_mae(
            series,
            set(finite_horizons),
            holdout_mask,
        )

        if np.isfinite(validation_error) and np.isfinite(holdout_error):
            difference = holdout_error - validation_error
            direction = "higher" if difference > 0 else "lower"
            error_label = (
                "Relative MAE"
                if single_horizon
                else "Pooled Relative MAE"
            )
            context_sentence = (
                f" Across the evidence boundary, {error_label} is "
                f"**{validation_error:.1f}% in ordinary validation** and "
                f"**{holdout_error:.1f}% in the final holdout** "
                f"({abs(difference):.1f} points {direction} in the holdout)."
            )
    else:
        miss = _largest_displayed_miss(
            series,
            finite_horizons,
        )

        if miss is not None:
            miss_value = float(miss["signed_pct"])
            direction = "above" if miss_value > 0 else "below"
            miss_horizon = int(miss["horizon"])
            miss_lead = (
                "Its largest displayed miss"
                if single_horizon
                else f"The largest displayed miss, **h={miss_horizon}**,"
            )
            context_sentence = (
                f" {miss_lead} occurs on "
                f"**{pd.Timestamp(miss['target_date']):%b %d, %Y} · "
                f"{humanize_bucket(str(miss['bucket']))}**, when the forecast is "
                f"**{abs(miss_value):.1f}% {direction}** observed "
                f"{metric_label(metric).lower()}."
            )

    anomaly_sentence = ""

    if show_anomalies:
        if anomaly_rows is None or anomaly_rows.empty:
            anomaly_sentence = (
                " No metric-relevant selected stress anomalies overlap this view."
            )
        else:
            anomaly_keys = set(
                zip(
                    pd.to_datetime(
                        anomaly_rows["target_date"]
                    ),
                    anomaly_rows["target_temporal_bucket"],
                )
            )

            anomaly_mask = pd.Series(
                [
                    (
                        pd.Timestamp(date),
                        bucket,
                    )
                    in anomaly_keys
                    for date, bucket in zip(
                        pd.to_datetime(
                            series["target_date"]
                        ),
                        series["target_temporal_bucket"],
                    )
                ],
                index=series.index,
            )

            anomaly_error = pooled_relative_mae(
                series,
                set(finite_horizons),
                anomaly_mask,
            )
            ordinary_error = pooled_relative_mae(
                series,
                set(finite_horizons),
                ~anomaly_mask,
            )

            if np.isfinite(anomaly_error) and np.isfinite(ordinary_error):
                difference = anomaly_error - ordinary_error
                direction = "higher" if difference > 0 else "lower"
                anomaly_error_label = (
                    "Relative MAE"
                    if single_horizon
                    else "Pooled Relative MAE"
                )
                anomaly_sentence = (
                    f" Across **{int(anomaly_mask.sum())} anomaly-marked time "
                    f"steps**, {anomaly_error_label} is **{anomaly_error:.1f}%** "
                    f"versus **{ordinary_error:.1f}%** elsewhere "
                    f"({abs(difference):.1f} points {direction})."
                )

    return (
        horizon_sentence
        + baseline_sentence
        + context_sentence
        + anomaly_sentence
    )

def _set_explorer_defaults() -> None:
    """Initialize the explorer once without overwriting later reader choices."""
    defaults = {
        "raw18_preset": "Choose your own view",
        "raw18_geo_level": "Taxi Zone",
        "raw18_metric": "fhvhv_trip_count",
        "raw18_day_type": ALL_DAY_TYPES,
        "raw18_daypart": ALL_DAYPARTS,
        "raw18_explorer_h1": True,
        "raw18_explorer_h2": True,
        "raw18_explorer_h5": True,
        "raw18_explorer_benchmark": True,
        "raw18_explorer_anomalies": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _mark_explorer_custom() -> None:
    """Keep the saved-view label honest after the reader changes a control."""
    st.session_state["raw18_preset"] = "Choose your own view"


def apply_preset_to_state() -> None:
    """Apply every filter needed to reproduce one curated forecast story."""
    preset_name = st.session_state.get(
        "raw18_preset",
        "Choose your own view",
    )
    preset = PRESET_VIEWS.get(preset_name)

    if not preset:
        return

    # A story should be reproducible in one click. Reset every filter that can
    # materially change the view instead of inheriting stale custom controls.
    st.session_state["raw18_geo_level"] = preset["geography_level"]
    st.session_state["raw18_geo_value"] = preset["geography_value"]
    st.session_state["raw18_metric"] = preset["metric"]
    st.session_state["raw18_day_type"] = preset["day_type"]
    st.session_state["raw18_daypart"] = preset["daypart"]
    st.session_state["raw18_history_date_range"] = (
        pd.Timestamp(preset["start_date"]).date(),
        pd.Timestamp(preset["end_date"]).date(),
    )
    st.session_state["raw18_explorer_h1"] = True
    st.session_state["raw18_explorer_h2"] = True
    st.session_state["raw18_explorer_h5"] = True
    st.session_state["raw18_explorer_benchmark"] = True
    st.session_state["raw18_explorer_anomalies"] = False

# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------

inject_app_css()

st.caption("FORECASTING & RELIABILITY  ›  FORECAST SCORECARD")
st.title(PAGE_TITLE)

st.markdown(
    """
A forecast earns trust by staying close to what actually happened. But performance
that looks strong for one mobility measure, place, or forecast horizon may not hold
across the rest of the system. This page compares forecasts with observed mobility
across five measures and three horizons, using an untouched final holdout for the
main reliability assessment and earlier validation history only as additional context.
"""
)

# ------------------------------------------------------------------
# 1. Hero — one visual area, two lenses
# ------------------------------------------------------------------

hero_records = load_hero_records()
hero_weekly = build_hero_weekly(hero_records)
hero_stats = hero_metrics(hero_weekly)

st.subheader("How closely did the forecast follow observed Manhattan Subway ridership?")
st.markdown(
    "**Hero focus:** Manhattan Subway ridership · final holdout · weekly aggregate · "
    "complete Monday–Sunday weeks. **Observed** is what happened; **h=1, h=2, and h=5** "
    "are forecasts made progressively farther ahead. The last-week baseline repeats the "
    "corresponding value from one week earlier, giving the forecasting system a simple "
    "reference to beat."
)

# Keep the editorial hero fixed so the opening white section answers the
# forecasting question before the reader enters the exploration workspace.
hero_horizons = set(HORIZONS)
show_hero_benchmark = True

actual_tab, error_tab = st.tabs(
    [
        "Actual vs forecast",
        "Forecast error",
    ]
)

with actual_tab:
    st.plotly_chart(
        hero_figure(
            hero_weekly,
            visible_horizons=hero_horizons,
            show_benchmark=show_hero_benchmark,
        ),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

with error_tab:
    st.caption(
        "Signed error above zero means the forecast was too high; below zero "
        "means it was too low."
    )

    st.plotly_chart(
        hero_error_figure(
            hero_weekly,
            visible_horizons=hero_horizons,
            show_benchmark=show_hero_benchmark,
        ),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

st.caption(
    "Relative MAE = total absolute forecast error ÷ total observed mobility × 100. "
    "Lower is better. These hero values use the weekly Manhattan aggregate shown "
    "above, so local over- and under-predictions can partly cancel."
)

hero_metric_columns = st.columns(3)

for column, horizon in zip(
    hero_metric_columns,
    HORIZONS,
):
    column.metric(
        f"h={horizon} Relative MAE",
        f"{hero_stats[horizon]['relative_mae_pct']:.1f}%",
        help=(
            "Relative MAE at the weekly Manhattan aggregate shown above. "
            "Aggregation can smooth some local over- and under-predictions."
        ),
    )
    column.caption(
        f"{hero_stats[horizon]['benchmark_skill_pct']:+.0f}% "
        "vs last-week baseline"
    )

render_chart_insight(
    build_hero_dynamic_insight(
        hero_stats,
        hero_horizons,
        hero_weekly,
        show_baseline=show_hero_benchmark,
    )
)

with st.expander("How the hero works"):
    st.markdown(
        """
- **Observed** is what actually happened.
- **h=1, h=2, and h=5** are predictions made progressively farther ahead.
- **Relative MAE** summarizes absolute forecast error relative to the amount of
  mobility observed. **Lower is better.**
- **Last-week baseline** is a simple reference forecast. It assumes the
  corresponding mobility value will match one week earlier and gives the selected
  forecasting system a clear comparison to beat.
- The hero sums Subway ridership across Manhattan and uses only complete
  Monday–Sunday weeks. The incomplete week beginning March 30 is excluded.
- The **Forecast error** tab shows the same hero as signed percentage error:
  above zero means the forecast was high; below zero means it was low.
"""
    )

# ------------------------------------------------------------------
# 2. One Forecast Quilt component, with two interchangeable lenses
# ------------------------------------------------------------------

st.divider()
st.header("Forecast Quilt")

st.write(
    "The same 5 × 3 grid gives two complementary views of the forecasting "
    "system. **Forecasts in Seaport** makes the three horizons tangible in one "
    "real Taxi Zone. **Accuracy across NYC** then asks how those forecasts "
    "performed across the full reader-facing geography."
)

jobs, zones = load_quilt_sources()

seaport_records = load_seaport_quilt_records()
(
    seaport_bucket,
    seaport_start,
    seaport_end,
    seaport_points,
) = seaport_quilt_context(
    seaport_records
)

forecast_tab, accuracy_tab = st.tabs(
    [
        "Forecasts in Seaport",
        "Accuracy across NYC",
    ]
)

with forecast_tab:
    if seaport_records.empty:
        st.warning(
            "No Seaport mobility measure has enough aligned h=1 / h=2 / h=5 "
            "observations to construct this Quilt."
        )
    else:
        supported_metrics = seaport_records.attrs.get(
            "supported_metrics",
            [],
        )
        excluded_metrics = seaport_records.attrs.get(
            "excluded_metrics",
            [],
        )

        st.caption(
            f"Seaport · {humanize_bucket(seaport_bucket)} · "
            f"{seaport_start:%b %d, %Y} to {seaport_end:%b %d, %Y} · "
            f"{seaport_points} shared dates"
        )

        st.caption(
            "Why Seaport? It is a useful good-but-not-perfect local example: "
            "the forecasts generally follow observed mobility without sitting "
            "directly on top of it, so both strengths and misses stay visible."
        )

        if excluded_metrics:
            st.caption(
                "Rows not shown because Seaport does not provide a complete "
                "three-horizon comparison for them: "
                + ", ".join(
                    metric_label(metric)
                    for metric in excluded_metrics
                )
                + "."
            )

        st.plotly_chart(
            seaport_forecast_quilt(
                seaport_records,
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

        st.caption(
            "Seaport is useful because the forecasts usually move with observed "
            "mobility without tracing it perfectly. The remaining gaps make the "
            "trade-off between short- and longer-horizon forecasts visible rather "
            "than presenting an unusually easy success case."
        )

with accuracy_tab:
    st.caption(
        "Large number = system Relative MAE; lower is better. The small "
        "histogram shows how Taxi-Zone error is distributed for that "
        "mobility-measure × horizon comparison. The secondary number shows the reduction in "
        "error versus the last-week baseline; positive is better."
    )

    st.plotly_chart(
        accuracy_quilt_tabbed(
            jobs,
            zones,
        ),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    st.caption(
        "The highest-error tail above each row's 95th-percentile display limit "
        "is collected into the right edge of the histogram. Headline Relative "
        "MAE remains untrimmed."
    )

render_chart_insight(
    build_quilt_dynamic_insight(
        jobs,
        zones,
    )
)

# ------------------------------------------------------------------
# 3. Interactive explorer
# ------------------------------------------------------------------

_set_explorer_defaults()

with exploration_section(
    key="raw18_exploration_area",
    title="Explore the forecasts yourself",
    description=(
        "Start with a curated example or build your own view. Each story restores "
        "the filters needed to reproduce it; from there, change one control at a "
        "time to see what drives the forecast pattern."
    ),
):
    st.caption(
        "The explorer extends back through ordinary validation for context. The "
        "Manhattan hero and NYC accuracy Quilt above remain untouched final-holdout "
        "evidence."
    )

    zone_context = load_forecast_geography_context()

    selected_preset = st.selectbox(
        "Start with a story",
        options=list(PRESET_VIEWS),
        key="raw18_preset",
        on_change=apply_preset_to_state,
    )

    st.caption(
        PRESET_DESCRIPTIONS[selected_preset]
        + (
            " Choosing a story restores its geography, mobility measure, Day type, "
            "Daypart, date range, horizons, and chart overlays."
            if PRESET_VIEWS[selected_preset]
            else ""
        )
    )

    filter_row_1 = st.columns(
        [1.05, 1.45, 1.25]
    )

    with filter_row_1[0]:
        geography_level = st.selectbox(
            "Geography level",
            GEOGRAPHY_LEVELS,
            key="raw18_geo_level",
            on_change=_mark_explorer_custom,
        )

    geo_options = geography_options(
        zone_context,
        geography_level,
    )

    default_geo_value = (
        "Seaport"
        if geography_level == "Taxi Zone"
        and "Seaport" in geo_options
        else geo_options[0]
    )

    if (
        "raw18_geo_value"
        not in st.session_state
        or st.session_state[
            "raw18_geo_value"
        ]
        not in geo_options
    ):
        st.session_state[
            "raw18_geo_value"
        ] = default_geo_value

    with filter_row_1[1]:
        geography_value = st.selectbox(
            "Geography",
            geo_options,
            key="raw18_geo_value",
            format_func=lambda value: geography_option_label(
                zone_context,
                geography_level,
                value,
            ),
            on_change=_mark_explorer_custom,
        )

    with filter_row_1[2]:
        selected_metric = st.selectbox(
            "Mobility measure",
            METRIC_ORDER,
            format_func=metric_label,
            key="raw18_metric",
            on_change=_mark_explorer_custom,
        )

    metric_records = load_explorer_metric_records(
        selected_metric
    )

    minimum_date = (
        metric_records[
            "target_date"
        ].min().date()
    )
    maximum_date = (
        metric_records[
            "target_date"
        ].max().date()
    )

    if (
        "raw18_day_type"
        not in st.session_state
        or st.session_state[
            "raw18_day_type"
        ]
        not in DAY_TYPE_OPTIONS
    ):
        st.session_state[
            "raw18_day_type"
        ] = ALL_DAY_TYPES

    if (
        "raw18_daypart"
        not in st.session_state
        or st.session_state[
            "raw18_daypart"
        ]
        not in DAYPART_OPTIONS
    ):
        st.session_state[
            "raw18_daypart"
        ] = ALL_DAYPARTS

    filter_row_2 = st.columns(
        [1.0, 1.0, 1.6]
    )

    with filter_row_2[0]:
        day_type = st.selectbox(
            "Day type",
            DAY_TYPE_OPTIONS,
            key="raw18_day_type",
            on_change=_mark_explorer_custom,
        )

    with filter_row_2[1]:
        daypart = st.selectbox(
            "Daypart",
            DAYPART_OPTIONS,
            key="raw18_daypart",
            on_change=_mark_explorer_custom,
        )

    date_range_key = "raw18_history_date_range"
    stored_dates = st.session_state.get(
        date_range_key
    )

    if stored_dates is None:
        st.session_state[date_range_key] = (
            minimum_date,
            maximum_date,
        )
    elif isinstance(stored_dates, (tuple, list)) and len(stored_dates) == 2:
        stored_start = pd.Timestamp(
            stored_dates[0]
        ).date()
        stored_end = pd.Timestamp(
            stored_dates[1]
        ).date()

        clipped_start = min(
            max(stored_start, minimum_date),
            maximum_date,
        )
        clipped_end = min(
            max(stored_end, minimum_date),
            maximum_date,
        )

        if clipped_start > clipped_end:
            clipped_start = minimum_date
            clipped_end = maximum_date

        clipped_range = (
            clipped_start,
            clipped_end,
        )

        if tuple(stored_dates) != clipped_range:
            st.session_state[date_range_key] = clipped_range
    elif isinstance(stored_dates, (tuple, list)) and len(stored_dates) == 1:
        # Streamlit temporarily stores a one-date tuple after the first click in a
        # range selection. Preserve that intermediate state so the second click can
        # complete the range instead of snapping back to the full study window.
        pending_date = pd.Timestamp(
            stored_dates[0]
        ).date()
        clipped_pending = min(
            max(pending_date, minimum_date),
            maximum_date,
        )

        if pending_date != clipped_pending:
            st.session_state[date_range_key] = (
                clipped_pending,
            )
    else:
        st.session_state[date_range_key] = (
            minimum_date,
            maximum_date,
        )

    with filter_row_2[2]:
        selected_dates = st.date_input(
            "Date range",
            min_value=minimum_date,
            max_value=maximum_date,
            key=date_range_key,
            on_change=_mark_explorer_custom,
        )

    if (
        isinstance(
            selected_dates,
            tuple,
        )
        and len(selected_dates) == 2
    ):
        start_date, end_date = selected_dates
    else:
        start_date = minimum_date
        end_date = maximum_date

    st.caption(
        f"Temporal view: **{daypart_filter_label(day_type, daypart)}**. "
        "Choose All days + All dayparts to see the complete ordered "
        "Date × Daypart forecast sequence."
    )

    st.caption(
        evaluation_period_copy(
            start_date,
            end_date,
        )
    )

    line_controls = st.columns(5)

    show_explorer_h1 = line_controls[
        0
    ].checkbox(
        "h=1",
        key="raw18_explorer_h1",
        on_change=_mark_explorer_custom,
    )
    show_explorer_h2 = line_controls[
        1
    ].checkbox(
        "h=2",
        key="raw18_explorer_h2",
        on_change=_mark_explorer_custom,
    )
    show_explorer_h5 = line_controls[
        2
    ].checkbox(
        "h=5",
        key="raw18_explorer_h5",
        on_change=_mark_explorer_custom,
    )
    show_explorer_benchmark = (
        line_controls[3].checkbox(
            "Last-week baseline",
            key="raw18_explorer_benchmark",
            on_change=_mark_explorer_custom,
            help=(
                "Simple baseline forecast: use the corresponding mobility value "
                "from one week earlier."
            ),
        )
    )
    show_anomalies = line_controls[
        4
    ].checkbox(
        "Stress anomalies",
        key="raw18_explorer_anomalies",
        on_change=_mark_explorer_custom,
        help=(
            "Highlights selected stress anomalies in the chosen geography and "
            "time window only when the mobility measure currently on screen was "
            "one of the event's defining drivers."
        ),
    )

    selected_zone_ids = geography_zone_ids(
        zone_context,
        geography_level,
        geography_value,
    )

    explorer_series = aggregate_explorer_series(
        metric_records,
        metric=selected_metric,
        zone_ids=selected_zone_ids,
        day_type=day_type,
        daypart=daypart,
        start_date=start_date,
        end_date=end_date,
    )

    explorer_horizons = {
        horizon
        for horizon, visible in [
            (1, show_explorer_h1),
            (2, show_explorer_h2),
            (5, show_explorer_h5),
        ]
        if visible
    }

    geography_display = (
        "NYC"
        if geography_level
        == "Citywide"
        else geography_value
    )

    if explorer_series.empty:
        st.warning(
            "No forecast observations are available for this combination. "
            "Try another geography, mobility measure, day type, daypart, or "
            "date range."
        )
    else:
        anomaly_rows = pd.DataFrame()

        if show_anomalies:
            try:
                anomaly_rows = anomaly_overlay(
                    explorer_series,
                    metric=selected_metric,
                    zone_ids=selected_zone_ids,
                    day_type=day_type,
                    daypart=daypart,
                    start_date=start_date,
                    end_date=end_date,
                )
            except Exception as error:
                st.caption(
                    "Stress-anomaly overlay is unavailable for this view: "
                    f"{error}"
                )

        st.subheader(
            f"{geography_display} · "
            f"{metric_label(selected_metric)}"
        )

        st.caption(
            f"{daypart_filter_label(day_type, daypart)} · "
            f"{start_date:%b %d, %Y} to {end_date:%b %d, %Y}"
        )

        if (
            selected_metric
            in SPEED_WEIGHT_METRIC
            and geography_level
            != "Taxi Zone"
        ):
            st.caption(
                "Aggregate speed is activity-weighted: observed speed uses observed "
                "trip volume, each forecast uses its own forecast trip volume, and "
                "the last-week baseline uses its own baseline trip-volume prediction."
            )

        selected_tab, error_tab, forecast_quilt_tab, accuracy_quilt_tab = st.tabs(
            [
                "Actual vs forecast",
                "Forecast error",
                "Forecast quilt",
                "Accuracy quilt",
            ]
        )

        with selected_tab:
            st.plotly_chart(
                explorer_figure(
                    explorer_series,
                    metric=selected_metric,
                    geography_label=(
                        geography_display
                    ),
                    visible_horizons=(
                        explorer_horizons
                    ),
                    show_benchmark=(
                        show_explorer_benchmark
                    ),
                    anomaly_rows=(
                        anomaly_rows
                        if show_anomalies
                        else None
                    ),
                ),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )

        with error_tab:
            st.caption(
                "Positive error means the forecast was too high; negative error "
                "means it was too low. Stress-anomaly diamonds sit on the zero "
                "line as context rather than implying causation."
            )

            st.plotly_chart(
                explorer_error_figure(
                    explorer_series,
                    visible_horizons=(
                        explorer_horizons
                    ),
                    show_benchmark=(
                        show_explorer_benchmark
                    ),
                    anomaly_rows=(
                        anomaly_rows
                        if show_anomalies
                        else None
                    ),
                ),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )

        quilt_records = load_personalized_quilt_records(
            tuple(
                sorted(
                    selected_zone_ids
                )
            )
        )

        personalized_series = (
            build_personalized_quilt_series(
                quilt_records,
                zone_ids=selected_zone_ids,
                day_type=day_type,
                daypart=daypart,
                start_date=start_date,
                end_date=end_date,
            )
        )

        with forecast_quilt_tab:
            st.caption(
                "The same geography and temporal view, expanded across every "
                "supported mobility measure and all three forecast horizons."
            )

            if not personalized_series:
                st.warning(
                    "No supported Forecast Quilt is available for this view."
                )
            else:
                st.plotly_chart(
                    personalized_forecast_quilt(
                        personalized_series
                    ),
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )

        with accuracy_quilt_tab:
            st.caption(
                "Large number = Relative MAE for this geography and time window; "
                "lower is better. The mini line shows signed error through the same "
                "observations. The secondary number is the change in error versus "
                "the last-week baseline; positive is better."
            )

            if not personalized_series:
                st.warning(
                    "No personalized Accuracy Quilt is available for this view."
                )
            else:
                st.plotly_chart(
                    personalized_accuracy_quilt(
                        personalized_series
                    ),
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )

        render_chart_insight(
            build_explorer_dynamic_insight(
                explorer_series,
                metric=selected_metric,
                visible_horizons=explorer_horizons,
                anomaly_rows=(
                    anomaly_rows
                    if show_anomalies
                    else None
                ),
                show_anomalies=show_anomalies,
                show_baseline=show_explorer_benchmark,
            )
        )

        if show_anomalies:
            if anomaly_rows.empty:
                st.caption(
                    "No metric-relevant selected stress anomalies overlap this "
                    "geography and time window."
                )
            else:
                st.caption(
                    f"Diamond markers flag {len(anomaly_rows):,} displayed time "
                    "steps where the selected mobility measure helped define a "
                    "stress anomaly. Hover lists the affected Taxi Zones."
                )

        metric_cards = st.columns(3)
        denominator = (
            explorer_series[
                "actual"
            ].abs().sum()
        )

        for card, horizon in zip(
            metric_cards,
            HORIZONS,
        ):
            forecast_column = (
                f"forecast_h{horizon}"
            )

            if (
                forecast_column
                not in explorer_series.columns
                or denominator <= 0
            ):
                card.metric(
                    f"h={horizon} Relative MAE",
                    "—",
                )
                continue

            relative_mae = relative_mae_for_series(
                explorer_series,
                forecast_column,
            )

            card.metric(
                f"h={horizon} Relative MAE",
                f"{relative_mae:.1f}%",
            )

    with st.expander("About the explorer"):
        st.markdown(
            """
    **One geography at a time**  
    Choose one spatial level, then one place inside it. Borough, Taxi Zone, mobility
    environment, and policy geography are alternative lenses rather than stacked
    filters.

    **Day type + Daypart**  
    These controls translate the project's 10 recurring temporal buckets into
    reader-facing choices. **All days + All dayparts** restores the complete ordered
    Date × Daypart sequence; narrower choices isolate repeated slices such as
    Weekdays · PM peak.

    **Evidence periods**  
    The explorer can span ordinary validation and final holdout. Rows before
    January 5, 2026 were available during model selection; rows from that date onward
    belong to the untouched final holdout. A dashed boundary appears whenever the
    selected chart crosses that date.

    **Last-week baseline**  
    This is the page's simple comparison forecast: use the corresponding mobility
    value from one week earlier. It provides an easy-to-understand reference for
    judging whether the selected forecasting system adds useful predictive value.

    **Mobility environments**  
    The forecast history shown here is entirely Post-CP, so grouped views use the
    Post-CP mobility-environment assignment.

    **Aggregate speeds**  
    Taxi and FHVHV speeds are activity-weighted rather than averaged equally.
    Observed speed uses observed trip volume, forecast speed uses forecast trip
    volume, and baseline speed uses the baseline's own trip-volume prediction.

    **Personalized Quilts**  
    The Forecast Quilt and Accuracy Quilt inherit the same geography, Day type,
    Daypart, and date range as the main explorer view, then expand that selection
    across every supported mobility measure.

    **Stress anomalies**  
    Optional diamonds appear only when the mobility measure currently on screen
    helped define a selected stress anomaly. Hover identifies the affected Taxi
    Zones. When anomalies are shown, the takeaway compares forecast error on those
    marked time steps with the rest of the selected view. That comparison is
    descriptive, not evidence that the anomaly caused the forecast miss.
            """
        )

# ------------------------------------------------------------------
# Closing synthesis
# ------------------------------------------------------------------

st.divider()

st.markdown("### What this page establishes")
st.markdown(
    "Forecast quality is not one number. The selected forecasting system can track "
    "observed mobility closely in some measure × horizon combinations while error grows "
    "in others, and a citywide or borough aggregate can look better than the local "
    "forecasts underneath it because over- and under-predictions partly cancel. Reliability "
    "therefore has to be judged across measures, horizons, places, and time—not from a "
    "single representative series."
)

with st.expander("How this page works", expanded=False):
    st.markdown(
        """
        **1. Keep the main scorecard on unseen data.** The opening evidence uses the
        untouched final holdout. Earlier validation history appears only in the explorer
        as context and is visually separated when a selected window crosses into the
        holdout.

        **2. Compare three forecast horizons.** **h=1, h=2, and h=5** are predictions
        made progressively farther ahead. They are forecast horizons, not alternative
        scenarios.

        **3. Measure error relative to the amount of mobility observed.** **Relative
        MAE** is total absolute forecast error divided by total observed mobility × 100.
        Lower is better. This puts measures with very different natural scales onto a
        more comparable error scale.

        **4. Require the forecasting system to beat a simple reference.** The
        **last-week baseline** predicts that the corresponding mobility value will match
        one week earlier. Improvement versus that baseline asks whether the selected
        model adds predictive value beyond a straightforward recent-history rule.

        **5. Distinguish aggregate accuracy from local accuracy.** The Manhattan hero
        sums Subway ridership across the borough, so local over- and under-predictions
        can partly cancel. The Accuracy Quilt calculates error before geographic
        aggregation, making it the stronger systemwide view of local reliability.

        **6. Aggregate speed measures with activity weights.** Taxi and FHVHV speeds
        are weighted by their corresponding trip volumes. Observed, forecast, and
        baseline speeds each use the matching volume series for their own aggregation.

        **7. Treat anomaly overlays as context, not an explanation for forecast error.**
        Optional markers identify time steps where the displayed mobility measure helped
        define a selected stress anomaly. Comparing error on those observations with the
        rest of the view is descriptive and does not show that the anomaly caused a miss.
        """
    )

st.caption(
    "Evidence scope: predictive accuracy of the selected forecasting system on observed "
    "NYC mobility, with the main scorecard anchored to the untouched final holdout. "
    "Forecast error and improvement versus the last-week baseline measure predictive "
    "performance; they do not establish why mobility changed or why a particular forecast "
    "was wrong."
)
