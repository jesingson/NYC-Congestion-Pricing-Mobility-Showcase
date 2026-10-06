"""
Raw 25 — Gap Calibration

Phase 1 + 2 visual build.

Question:
    Is the no-CP gap unusual relative to normal forecasting error?

Raw 24 asks whether a conclusion survives reasonable analytical alternatives.
Raw 25 asks a different question: how large is the observed-versus-no-CP
divergence relative to the forecasting system's ordinary Pre-CP validation
error?

This page consumes only compact precomputed calibration artifacts. It does not refit models or recreate counterfactual forecasts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data_access.counterfactuals import (
    load_counterfactual_raw25_inputs,
    load_counterfactual_temporal_explorer_metric,
)
from app.utils.project_branding import (
    BRAND_COLORS,
    apply_branding,
    exploration_section,
    inject_app_css,
    render_chart_insight,
)


# ---------------------------------------------------------------------
# Page contract
# ---------------------------------------------------------------------

inject_app_css()

PAGE_CAPTION = "COUNTERFACTUAL CALIBRATION"
PAGE_TITLE = "How large is the no-CP gap relative to Pre-CP Reference error?"
PAGE_SUBTITLE = (
    "Put the observed-versus-no-CP gap on the same scale as the forecasting "
    "Reference system's Pre-CP validation error, then see where the typical gap "
    "is modest and where the upper tail becomes much larger."
)

PLOT_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
}

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

COUNT_METRICS = {
    "taxi_trip_count",
    "fhvhv_trip_count",
    "subway_ridership",
}

HORIZONS = [
    1,
    2,
    5,
]

GEOGRAPHY_TYPES = [
    "Systemwide",
    "Borough",
    "Policy geography",
    "Mobility environment",
    "Taxi Zone",
]

DAY_TYPES = [
    "All days",
    "Weekdays",
    "Weekends",
]

DAYPARTS = [
    "All dayparts",
    "Overnight",
    "AM peak",
    "Midday",
    "PM peak",
    "Evening",
]

POLICY_LABELS = {
    "cbd": "CBD",
    "adjacent_to_cbd": "Adjacent to CBD",
    "gateway_to_cbd": "Gateway to CBD",
    "non_cbd": "Outside CBD / adjacent / gateway",
}

# Match the physical-zone contract used by the corrected Raw 25 builders.
TAXI_ZONE_ALIASES = {
    57: 56,
    105: 103,
}


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


# ---------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------

def metric_label(
    metric: str,
) -> str:
    """Reader-facing metric label."""
    return METRIC_LABELS.get(
        metric,
        metric,
    )


def brand_figure(
    figure: go.Figure,
    *,
    height: int,
    left: int = 90,
    right: int = 30,
    top: int = 55,
    bottom: int = 65,
) -> go.Figure:
    """Apply shared branding and prevent Plotly title leakage."""
    figure = apply_branding(
        figure
    )

    figure.update_layout(
        height=height,
        margin={
            "l": left,
            "r": right,
            "t": top,
            "b": bottom,
        },
        title={
            "text": "",
        },
        legend_title_text="",
    )

    annotations = list(
        figure.layout.annotations
        if figure.layout.annotations
        else []
    )

    figure.update_layout(
        annotations=[
            annotation
            for annotation in annotations
            if str(
                annotation.text
            )
            .strip()
            .lower()
            != "undefined"
        ]
    )

    return figure


# ---------------------------------------------------------------------
# Hero: 15 calibration ridges
# ---------------------------------------------------------------------

def build_calibration_range_strip(
    summary: pd.DataFrame,
) -> go.Figure:
    """
    Compare the available calibration jobs on one Pre-CP error ruler.

    WHY:
    The systemwide hero contains all 15 Metric × Horizon jobs, while a selected
    geography may legitimately support fewer jobs because a mobility mode is
    unavailable there. The visual grammar stays identical in both cases; only
    supported rows are rendered.
    """
    working = summary.copy()

    # Keep only the known reader-facing metric / horizon contract and fail
    # loudly on duplicate jobs. Missing jobs are allowed for geography slices.
    working = working.loc[
        working["metric"].isin(METRIC_ORDER)
        & working["horizon"].isin(HORIZONS)
    ].copy()

    if working.empty:
        raise ValueError(
            "Calibration range strip received no supported Metric × Horizon rows."
        )

    if working.duplicated(["metric", "horizon"]).any():
        raise RuntimeError(
            "Calibration range strip received duplicate Metric × Horizon rows."
        )

    available_jobs = {
        (str(row.metric), int(row.horizon))
        for row in working.itertuples(index=False)
    }

    ordered_jobs = [
        (metric, horizon)
        for metric in METRIC_ORDER
        for horizon in HORIZONS
        if (metric, horizon) in available_jobs
    ]

    available_metrics = [
        metric
        for metric in METRIC_ORDER
        if any(
            (metric, horizon) in available_jobs
            for horizon in HORIZONS
        )
    ]

    # WHY: preserve the hero's grouping grammar while allocating space only to
    # rows that actually exist. This prevents an unavailable mode or horizon
    # from crashing the chart or leaving misleading blank rows.
    row_lookup: dict[tuple[str, int], float] = {}
    group_rows: dict[str, list[float]] = {}
    cursor = 0.0

    for metric in available_metrics:
        rows = []

        for horizon in HORIZONS:
            if (metric, horizon) not in available_jobs:
                continue

            row_lookup[(metric, horizon)] = cursor
            rows.append(cursor)
            cursor += 0.78

        group_rows[metric] = rows
        cursor += 0.58

    x_max = min(
        14.0,
        max(
            3.25,
            float(working["p95_abs_mae_units"].max()) * 1.06,
        ),
    )

    figure = go.Figure()

    # Calibration-ruler background. These are scale regions, not significance
    # thresholds.
    figure.add_vrect(
        x0=0.0,
        x1=min(1.0, x_max),
        fillcolor="rgba(131, 197, 190, 0.08)",
        line_width=0,
        layer="below",
    )
    if x_max > 1.0:
        figure.add_vrect(
            x0=1.0,
            x1=min(2.0, x_max),
            fillcolor="rgba(237, 246, 249, 0.58)",
            line_width=0,
            layer="below",
        )
    if x_max > 2.0:
        figure.add_vrect(
            x0=2.0,
            x1=x_max,
            fillcolor="rgba(255, 221, 210, 0.10)",
            line_width=0,
            layer="below",
        )

    # Alternate only among metric groups that are actually displayed.
    for index, metric in enumerate(available_metrics):
        rows = group_rows[metric]

        if index % 2 == 0:
            figure.add_hrect(
                y0=min(rows) - 0.34,
                y1=max(rows) + 0.34,
                fillcolor="rgba(255, 255, 255, 0.42)",
                line_width=0,
                layer="below",
            )

    for metric, horizon in ordered_jobs:
        matches = working.loc[
            working["metric"].eq(metric)
            & working["horizon"].eq(horizon)
        ]

        # Duplicate rows were already rejected above; this is a defensive
        # guard against future contract drift.
        if len(matches) != 1:
            raise RuntimeError(
                "Expected exactly one calibration row for "
                f"{metric} · h={horizon}; found {len(matches)}."
            )

        job = matches.iloc[0]

        y = float(row_lookup[(metric, horizon)])
        p50 = float(job["p50_abs_mae_units"])
        p75 = float(job["p75_abs_mae_units"])
        p90 = float(job["p90_abs_mae_units"])
        p95 = float(job["p95_abs_mae_units"])

        figure.add_trace(
            go.Scatter(
                x=[p50, p75],
                y=[y, y],
                mode="lines",
                line={
                    "color": BRAND_COLORS["dark_teal"],
                    "width": 8,
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

        figure.add_trace(
            go.Scatter(
                x=[p75, p95],
                y=[y, y],
                mode="lines",
                line={
                    "color": "rgba(0, 109, 119, 0.30)",
                    "width": 4,
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

        figure.add_trace(
            go.Scatter(
                x=[p50],
                y=[y],
                mode="markers",
                marker={
                    "size": 12,
                    "color": BRAND_COLORS["terracotta"],
                    "symbol": "diamond",
                    "line": {
                        "color": "white",
                        "width": 0.8,
                    },
                },
                customdata=[[
                    metric_label(metric),
                    int(horizon),
                    p75,
                    p90,
                    p95,
                    float(job["share_abs_ge_1_0_mae_pct"]),
                    float(job["share_abs_ge_2_0_mae_pct"]),
                ]],
                hovertemplate=(
                    "<b>%{customdata[0]} · h=%{customdata[1]}</b><br>"
                    "Median |gap|: %{x:.3f}× MAE<br>"
                    "P75: %{customdata[2]:.3f}× MAE<br>"
                    "P90: %{customdata[3]:.3f}× MAE<br>"
                    "P95: %{customdata[4]:.3f}× MAE<br>"
                    "Rows ≥1× MAE: %{customdata[5]:.3f}%<br>"
                    "Rows ≥2× MAE: %{customdata[6]:.3f}%"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

        figure.add_trace(
            go.Scatter(
                x=[p90, p90],
                y=[y - 0.19, y + 0.19],
                mode="lines",
                line={
                    "color": BRAND_COLORS["dark_teal"],
                    "width": 3,
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

        figure.add_trace(
            go.Scatter(
                x=[p95, p95],
                y=[y - 0.24, y + 0.24],
                mode="lines",
                line={
                    "color": "rgba(0, 109, 119, 0.52)",
                    "width": 2,
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

        figure.add_annotation(
            x=0.03,
            y=y,
            text=f"h={horizon}",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font={
                "size": 10,
                "color": "#527078",
            },
        )

    figure.add_vline(
        x=1.0,
        line_dash="dash",
        line_width=1.5,
        line_color="rgba(0, 109, 119, 0.64)",
    )
    figure.add_vline(
        x=2.0,
        line_dash="dot",
        line_width=1.3,
        line_color="rgba(226, 149, 120, 0.82)",
    )
    figure.add_annotation(
        x=1.0,
        y=1.075,
        xref="x",
        yref="paper",
        text="<b>1× Pre-CP Reference MAE</b>",
        showarrow=False,
        xanchor="right",
        font={"size": 11, "color": BRAND_COLORS["dark_teal"]},
    )
    figure.add_annotation(
        x=2.0,
        y=1.025,
        xref="x",
        yref="paper",
        text="<b>2× MAE</b>",
        showarrow=False,
        xanchor="left",
        font={"size": 11, "color": BRAND_COLORS["terracotta"]},
    )

    tickvals = []
    ticktext = []

    for metric in available_metrics:
        rows = group_rows[metric]
        tickvals.append(float(np.mean(rows)))
        ticktext.append(f"<b>{metric_label(metric)}</b>")

    figure.update_yaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        title=None,
        showgrid=False,
        autorange="reversed",
    )
    figure.update_xaxes(
        title_text="Absolute post-CP gap in Pre-CP Reference MAE units",
        range=[0, x_max],
        zeroline=False,
    )

    # Full hero remains 720 px; smaller geography slices contract naturally.
    chart_height = max(
        410,
        300 + (28 * len(ordered_jobs)),
    )

    figure = brand_figure(
        figure,
        height=chart_height,
        left=175,
        right=35,
        top=90,
        bottom=70,
    )

    # Two evidence landmarks orient the reader without labeling every row.
    highest_median = working.sort_values(
        "p50_abs_mae_units",
        ascending=False,
    ).iloc[0]
    median_key = (
        str(highest_median["metric"]),
        int(highest_median["horizon"]),
    )
    figure.add_annotation(
        x=float(highest_median["p50_abs_mae_units"]),
        y=row_lookup[median_key],
        text=(
            "<b>Largest median</b><br>"
            f"{float(highest_median['p50_abs_mae_units']):.3f}× MAE"
        ),
        showarrow=True,
        arrowhead=2,
        arrowsize=0.8,
        arrowwidth=1.1,
        arrowcolor=BRAND_COLORS["terracotta"],
        ax=52,
        ay=-28,
        bgcolor="rgba(255,255,255,0.90)",
        bordercolor="rgba(226,149,120,0.45)",
        borderwidth=1,
        font={"size": 10, "color": "#335C67"},
    )

    highest_p90 = working.sort_values(
        "p90_abs_mae_units",
        ascending=False,
    ).iloc[0]
    p90_key = (
        str(highest_p90["metric"]),
        int(highest_p90["horizon"]),
    )
    figure.add_annotation(
        x=float(highest_p90["p90_abs_mae_units"]),
        y=row_lookup[p90_key],
        text=(
            "<b>Largest P90</b><br>"
            f"{float(highest_p90['p90_abs_mae_units']):.3f}× MAE"
        ),
        showarrow=True,
        arrowhead=2,
        arrowsize=0.8,
        arrowwidth=1.1,
        arrowcolor=BRAND_COLORS["dark_teal"],
        ax=-58,
        ay=-28,
        bgcolor="rgba(255,255,255,0.90)",
        bordercolor="rgba(0,109,119,0.35)",
        borderwidth=1,
        font={"size": 10, "color": "#335C67"},
    )

    return figure


# ---------------------------------------------------------------------
# Secondary scout: typical versus upper-tail separation
# ---------------------------------------------------------------------

def render_calibration_key() -> None:
    """Render the custom quantile grammar as a compact reader-facing key."""
    st.markdown(
        f"""
        <div style="
            display:flex; flex-wrap:wrap; gap:0.55rem 1.05rem; align-items:center;
            padding:0.55rem 0.75rem; margin:0.20rem 0 0.55rem 0;
            border:1px solid rgba(0,109,119,0.18); border-radius:0.65rem;
            background:rgba(237,246,249,0.72); color:#335C67; font-size:0.88rem;
        ">
          <span style="font-weight:750;color:#003F46;">How to read one row</span>
          <span><b style="color:{BRAND_COLORS['terracotta']};font-size:1.05rem;">◆</b> Median / typical gap</span>
          <span><b style="color:{BRAND_COLORS['dark_teal']};">━━━━</b> Median → P75</span>
          <span><b style="color:{BRAND_COLORS['dark_teal']};font-size:1.05rem;">│</b> P90</span>
          <span><b style="color:rgba(0,109,119,0.50);">━━━━</b> P75 → P95</span>
          <span><b style="color:rgba(0,109,119,0.70);font-size:1.05rem;">│</b> P95 cap</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero_takeaway(
    summary: pd.DataFrame,
) -> str:
    """Explain which systemwide gaps stand apart from ordinary forecast error."""
    metric_rows = []

    for metric in METRIC_ORDER:
        scoped = (
            summary.loc[
                summary["metric"].eq(metric),
                ["horizon", "p50_abs_mae_units"],
            ]
            .sort_values("horizon")
        )

        if len(scoped) != len(HORIZONS):
            continue

        values = scoped["p50_abs_mae_units"].astype(float)

        metric_rows.append({
            "metric": metric,
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "spread": float(values.max() - values.min()),
        })

    if not metric_rows:
        return (
            "Compare each mobility measure's three forecast horizons together. "
            "Values near or below 1× MAE mean the typical observed-versus-no-CP "
            "gap is no larger than the forecasting system's ordinary Pre-CP "
            "error. Larger values mean the two paths are farther apart than the "
            "model would ordinarily miss by before congestion pricing."
        )

    metric_summary = pd.DataFrame(metric_rows)

    strongest = metric_summary.sort_values(
        ["minimum", "metric"],
        ascending=[False, True],
    ).iloc[0]

    most_consistent = metric_summary.sort_values(
        ["spread", "metric"],
        ascending=[True, True],
    ).iloc[0]

    strongest_label = metric_label(strongest["metric"])
    consistent_label = metric_label(most_consistent["metric"])

    strongest_minimum = float(strongest["minimum"])
    strongest_maximum = float(strongest["maximum"])
    consistent_spread = float(most_consistent["spread"])

    if strongest_minimum >= 1.0:
        strongest_sentence = (
            f"**{strongest_label} stands out most clearly from the model's "
            "ordinary forecast error:** across all three horizons, its typical "
            "observed-versus-no-CP gap remains at least "
            f"**{strongest_minimum:.2f}× the model's ordinary Pre-CP error**, "
            f"reaching **{strongest_maximum:.2f}×** at its largest."
        )
    else:
        strongest_sentence = (
            f"**{strongest_label} shows the largest typical separation**, but "
            "even its smallest cross-horizon gap remains below one ordinary "
            f"Pre-CP forecast error at **{strongest_minimum:.2f}× MAE**."
        )

    if strongest["metric"] == most_consistent["metric"]:
        consistency_sentence = (
            " It is also the least sensitive to forecast horizon among the "
            "mobility measures shown, so that magnitude changes relatively "
            "little between h=1, h=2, and h=5."
        )
    else:
        if consistent_spread < 0.01:
            consistency_sentence = (
                f" **{consistent_label} is remarkably consistent across forecast "
                "horizons**, with almost no change in its typical gap between "
                "h=1, h=2, and h=5."
            )
        else:
            consistency_sentence = (
                f" **{consistent_label} changes the least as forecast horizon "
                "changes**, with only "
                f"**{consistent_spread:.2f}× MAE** separating its smallest and "
                "largest typical gaps."
            )

    return (
        strongest_sentence
        + consistency_sentence
        + " The key question is therefore not whether a gap crosses a magic "
        "threshold, but whether it is large compared with the errors this "
        "forecasting system ordinarily made before congestion pricing—and "
        "whether that conclusion holds across horizons."
    )




# ---------------------------------------------------------------------
# Phase 3: interactive calibration explorer
# ---------------------------------------------------------------------


def format_geography_label(
    geography_type: str,
    geography_id: object,
    geography_label: object,
) -> str:
    """Return a concise reader-facing label for one supported geography."""
    label = str(geography_label)

    if geography_type == "Systemwide":
        return "NYC supported system"

    if geography_type == "Policy geography":
        return POLICY_LABELS.get(
            label,
            label.replace("_", " ").title(),
        )

    if geography_type == "Taxi Zone":
        return label

    return label.replace("_", " ").strip().title()


INVALID_GEOGRAPHY_TOKENS = {
    "",
    "<na>",
    "nan",
    "none",
    "null",
    "unknown",
    "not available",
}


def _valid_reader_geography(
    geography_id: object,
    geography_label: object,
) -> bool:
    """
    Return True only for geographies that should appear in reader-facing controls.

    WHY:
    The scout groups with dropna=False so it can audit the full analytical
    surface, including missing/unknown geography metadata. Those audit rows are
    useful upstream but should never become choices such as "<NA>" or "Unknown"
    in the production explorer.
    """
    if pd.isna(geography_id) or pd.isna(geography_label):
        return False

    id_text = str(geography_id).strip().lower()
    label_text = str(geography_label).strip().lower()

    return (
        id_text not in INVALID_GEOGRAPHY_TOKENS
        and label_text not in INVALID_GEOGRAPHY_TOKENS
    )


def supported_geographies(
    slice_scout: pd.DataFrame,
    *,
    metric: str,
    horizon: int,
    geography_type: str,
    day_type: str,
    daypart: str,
) -> pd.DataFrame:
    """Return only supported, reader-facing geographies for the selected context."""
    scoped = slice_scout.loc[
        slice_scout["metric"].eq(metric)
        & slice_scout["horizon"].eq(int(horizon))
        & slice_scout["geography_type"].eq(geography_type)
        & slice_scout["day_type"].eq(day_type)
        & slice_scout["daypart"].eq(daypart)
        & slice_scout["support_ok"]
    ].copy()

    if scoped.empty:
        return scoped

    # WHY: keep missing/unknown geography rows available in the scout artifact
    # for QA, but exclude them from the reader-facing dropdown.
    reader_mask = [
        _valid_reader_geography(
            row.geography_id,
            row.geography_label,
        )
        for row in scoped.itertuples(index=False)
    ]
    scoped = scoped.loc[reader_mask].copy()

    if scoped.empty:
        return scoped

    scoped["display_label"] = [
        format_geography_label(
            geography_type,
            row.geography_id,
            row.geography_label,
        )
        for row in scoped.itertuples(index=False)
    ]

    # Defensive second pass: formatted labels should also never expose an
    # unknown/missing token if an upstream label contract changes.
    scoped = scoped.loc[
        ~scoped["display_label"]
        .astype("string")
        .str.strip()
        .str.lower()
        .isin(INVALID_GEOGRAPHY_TOKENS)
    ].copy()

    return (
        scoped
        .sort_values(
            ["display_label", "geography_id"],
            kind="stable",
        )
        .drop_duplicates("geography_id")
        .reset_index(drop=True)
    )


def selected_slice_row(
    slice_scout: pd.DataFrame,
    *,
    metric: str,
    horizon: int,
    geography_type: str,
    geography_id: str,
    day_type: str,
    daypart: str,
) -> pd.Series:
    """Resolve one unique, supported explorer slice."""
    match = slice_scout.loc[
        slice_scout["metric"].eq(metric)
        & slice_scout["horizon"].eq(int(horizon))
        & slice_scout["geography_type"].eq(geography_type)
        & slice_scout["geography_id"].astype(str).eq(str(geography_id))
        & slice_scout["day_type"].eq(day_type)
        & slice_scout["daypart"].eq(daypart)
        & slice_scout["support_ok"]
    ]

    if len(match) != 1:
        raise RuntimeError(
            "Expected exactly one supported Raw 25 explorer slice; "
            f"found {len(match)}."
        )

    return match.iloc[0]


def explorer_x_max(summary: pd.DataFrame) -> float:
    """Reuse the hero's P95-bounded ruler for visual continuity."""
    return min(
        14.0,
        max(
            3.25,
            float(summary["p95_abs_mae_units"].max()) * 1.06,
        ),
    )


def build_selected_slice_strip(
    row: pd.Series,
    *,
    x_max: float,
) -> go.Figure:
    """Render one explorer slice using the exact hero quantile grammar."""
    p50 = float(row["p50_abs_mae_units"])
    p75 = float(row["p75_abs_mae_units"])
    p90 = float(row["p90_abs_mae_units"])
    p95 = float(row["p95_abs_mae_units"])

    figure = go.Figure()

    # Median -> P75.
    figure.add_trace(
        go.Scatter(
            x=[p50, p75],
            y=[1, 1],
            mode="lines",
            line={
                "color": BRAND_COLORS["dark_teal"],
                "width": 10,
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # P75 -> P95.
    figure.add_trace(
        go.Scatter(
            x=[p75, p95],
            y=[1, 1],
            mode="lines",
            line={
                "color": "rgba(0, 109, 119, 0.32)",
                "width": 5,
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )

    figure.add_trace(
        go.Scatter(
            x=[p50],
            y=[1],
            mode="markers",
            marker={
                "size": 16,
                "color": BRAND_COLORS["terracotta"],
                "symbol": "diamond",
            },
            customdata=[[p75, p90, p95, int(row["rows"])]],
            hovertemplate=(
                "Median |gap|: %{x:.3f}× MAE<br>"
                "P75: %{customdata[0]:.3f}× MAE<br>"
                "P90: %{customdata[1]:.3f}× MAE<br>"
                "P95: %{customdata[2]:.3f}× MAE<br>"
                "Eligible rows: %{customdata[3]:,d}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    # P90 tick.
    figure.add_trace(
        go.Scatter(
            x=[p90, p90],
            y=[0.78, 1.22],
            mode="lines",
            line={
                "color": BRAND_COLORS["dark_teal"],
                "width": 3,
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # P95 cap.
    figure.add_trace(
        go.Scatter(
            x=[p95, p95],
            y=[0.72, 1.28],
            mode="lines",
            line={
                "color": "rgba(0,109,119,0.60)",
                "width": 3,
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Calibration ruler.
    figure.add_vline(
        x=1.0,
        line_dash="dash",
        line_width=1.4,
        line_color="rgba(0, 109, 119, 0.62)",
    )
    figure.add_vline(
        x=2.0,
        line_dash="dot",
        line_width=1.2,
        line_color="rgba(226, 149, 120, 0.78)",
    )
    figure.add_annotation(
        x=1.0,
        y=1.06,
        xref="x",
        yref="paper",
        text="1× ordinary Pre-CP MAE",
        showarrow=False,
        xanchor="center",
        font={
            "size": 10,
            "color": BRAND_COLORS["dark_teal"],
        },
    )
    figure.add_annotation(
        x=2.0,
        y=1.06,
        xref="x",
        yref="paper",
        text="2× MAE",
        showarrow=False,
        xanchor="center",
        font={
            "size": 10,
            "color": BRAND_COLORS["terracotta"],
        },
    )

    figure.update_xaxes(
        title_text="Absolute post-CP gap in Pre-CP Reference MAE units",
        range=[0, x_max],
    )
    figure.update_yaxes(
        range=[0.45, 1.55],
        visible=False,
        fixedrange=True,
    )

    figure = brand_figure(
        figure,
        height=280,
        left=35,
        right=25,
        top=45,
        bottom=60,
    )

    return figure


def build_exceedance_profile(
    row: pd.Series,
) -> go.Figure:
    """Show how often the selected slice clears four calibration thresholds."""
    labels = [
        "≥0.5× MAE",
        "≥1× MAE",
        "≥2× MAE",
        "≥3× MAE",
    ]
    values = [
        float(row["share_abs_ge_0_5_mae_pct"]),
        float(row["share_abs_ge_1_0_mae_pct"]),
        float(row["share_abs_ge_2_0_mae_pct"]),
        float(row["share_abs_ge_3_0_mae_pct"]),
    ]

    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={
                "color": [
                    BRAND_COLORS["dark_teal"],
                    "rgba(0,109,119,0.74)",
                    BRAND_COLORS["seafoam"],
                    "rgba(131,197,190,0.72)",
                ],
            },
            text=[f"{value:.1f}%" for value in values],
            textposition="outside",
            cliponaxis=False,
            customdata=[int(row["rows"])] * len(labels),
            hovertemplate=(
                "%{y}: %{x:.3f}% of eligible rows<br>"
                "Eligible rows: %{customdata:,d}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    figure.update_xaxes(
        range=[0, 100],
        visible=False,
        fixedrange=True,
    )
    figure.update_yaxes(
        autorange="reversed",
        title=None,
        showgrid=False,
        tickfont={"size": 12},
    )

    figure = brand_figure(
        figure,
        height=270,
        left=95,
        right=55,
        top=18,
        bottom=18,
    )
    figure.update_layout(
        bargap=0.42,
    )

    return figure


def explorer_takeaway(
    row: pd.Series,
    *,
    metric: str,
    horizon: int,
    geography_label: str,
    day_type: str,
    daypart: str,
) -> str:
    """Translate the selected calibration slice into one concise finding."""
    scope = (
        f"{metric_label(metric)} · h={int(horizon)} · {geography_label} · "
        f"{day_type} · {daypart}"
    )

    return (
        f"For **{scope}**, the median absolute gap is "
        f"**{float(row['p50_abs_mae_units']):.3f}× MAE**. "
        f"**{float(row['share_abs_ge_1_0_mae_pct']):.1f}%** of eligible rows "
        "are at least 1× ordinary Pre-CP MAE from observed, and "
        f"**{float(row['share_abs_ge_2_0_mae_pct']):.1f}%** are at least 2×. "
        f"The P90 gap is **{float(row['p90_abs_mae_units']):.3f}× MAE** and "
        f"P95 is **{float(row['p95_abs_mae_units']):.3f}× MAE**."
    )


def supported_geographies_for_grid(
    slice_scout: pd.DataFrame,
    *,
    geography_type: str,
    day_type: str,
    daypart: str,
) -> pd.DataFrame:
    """
    Return reader-facing geographies with at least one supported mobility job.

    WHY:
    Some geographies legitimately do not support every mobility measure. A Taxi
    Zone without subway service, for example, should remain available for Taxi
    and FHVHV calibration rather than disappearing from the explorer entirely.
    """
    scoped = slice_scout.loc[
        slice_scout["geography_type"].eq(geography_type)
        & slice_scout["day_type"].eq(day_type)
        & slice_scout["daypart"].eq(daypart)
        & slice_scout["support_ok"]
    ].copy()

    if scoped.empty:
        return scoped

    reader_mask = [
        _valid_reader_geography(
            row.geography_id,
            row.geography_label,
        )
        for row in scoped.itertuples(index=False)
    ]
    scoped = scoped.loc[reader_mask].copy()

    if scoped.empty:
        return scoped

    scoped["display_label"] = [
        format_geography_label(
            geography_type,
            row.geography_id,
            row.geography_label,
        )
        for row in scoped.itertuples(index=False)
    ]

    scoped = scoped.loc[
        ~scoped["display_label"]
        .astype("string")
        .str.strip()
        .str.lower()
        .isin(INVALID_GEOGRAPHY_TOKENS)
    ].copy()

    if scoped.empty:
        return scoped

    geography_rows = (
        scoped.groupby(
            ["geography_id", "display_label"],
            observed=True,
            dropna=False,
        )
        .agg(
            supported_jobs=("metric", "size"),
            min_rows=("rows", "min"),
            median_rows=("rows", "median"),
        )
        .reset_index()
    )

    return (
        geography_rows
        .sort_values(
            ["display_label", "geography_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def selected_geography_grid(
    slice_scout: pd.DataFrame,
    *,
    geography_type: str,
    geography_id: str,
    day_type: str,
    daypart: str,
) -> pd.DataFrame:
    """Resolve all supported Metric × Horizon jobs for one geography."""
    result = slice_scout.loc[
        slice_scout["geography_type"].eq(geography_type)
        & slice_scout["geography_id"].astype(str).eq(str(geography_id))
        & slice_scout["day_type"].eq(day_type)
        & slice_scout["daypart"].eq(daypart)
        & slice_scout["support_ok"]
        & slice_scout["metric"].isin(METRIC_ORDER)
        & slice_scout["horizon"].isin(HORIZONS)
    ].copy()

    if result.empty:
        raise RuntimeError(
            "Selected geography has no supported Raw 25 Metric × Horizon jobs."
        )

    duplicate_jobs = result.duplicated(
        ["metric", "horizon"]
    )

    if duplicate_jobs.any():
        raise RuntimeError(
            "Selected geography has duplicate Metric × Horizon rows."
        )

    metric_rank = {
        metric: index
        for index, metric in enumerate(METRIC_ORDER)
    }
    horizon_rank = {
        horizon: index
        for index, horizon in enumerate(HORIZONS)
    }

    result["_metric_rank"] = result["metric"].map(metric_rank)
    result["_horizon_rank"] = result["horizon"].map(horizon_rank)

    return (
        result
        .sort_values(
            ["_metric_rank", "_horizon_rank"],
            kind="stable",
        )
        .drop(
            columns=["_metric_rank", "_horizon_rank"]
        )
        .reset_index(drop=True)
    )


def geography_grid_takeaway(
    grid: pd.DataFrame,
    *,
    geography_label: str,
    day_type: str,
    daypart: str,
) -> str:
    """Explain calibration for one place and time slice in reader-facing terms."""
    metric_rows = []

    for metric in METRIC_ORDER:
        scoped = (
            grid.loc[
                grid["metric"].eq(metric),
                ["horizon", "p50_abs_mae_units"],
            ]
            .sort_values("horizon")
        )

        if len(scoped) != len(HORIZONS):
            continue

        values = scoped["p50_abs_mae_units"].astype(float)

        metric_rows.append({
            "metric": metric,
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "spread": float(values.max() - values.min()),
        })

    context = f"**{geography_label} · {day_type} · {daypart}**"

    # WHY: some geography slices legitimately lack one or more horizons.
    # Describe the strongest available evidence without pretending that
    # cross-horizon consistency can be assessed.
    if not metric_rows:
        highest = grid.sort_values(
            "p50_abs_mae_units",
            ascending=False,
        ).iloc[0]

        highest_value = float(highest["p50_abs_mae_units"])
        highest_label = metric_label(highest["metric"])
        highest_horizon = int(highest["horizon"])

        if highest_value >= 1.0:
            magnitude_sentence = (
                f"The largest typical gap available is **{highest_label} at "
                f"h={highest_horizon}**, where the observed and estimated no-CP "
                "paths are separated by "
                f"**{highest_value:.2f}× the model's ordinary Pre-CP error**."
            )
        else:
            magnitude_sentence = (
                f"Even the largest typical gap available—**{highest_label} at "
                f"h={highest_horizon}**—is only **{highest_value:.2f}× the "
                "model's ordinary Pre-CP error**."
            )

        return (
            f"For {context}, {magnitude_sentence} "
            "Not every forecast horizon is available for the same mobility "
            "measure here, so this view cannot tell us whether that magnitude "
            "holds consistently across h=1, h=2, and h=5."
        )

    metric_summary = pd.DataFrame(metric_rows)

    strongest = metric_summary.sort_values(
        ["minimum", "metric"],
        ascending=[False, True],
    ).iloc[0]

    most_consistent = metric_summary.sort_values(
        ["spread", "metric"],
        ascending=[True, True],
    ).iloc[0]

    strongest_label = metric_label(strongest["metric"])
    consistent_label = metric_label(most_consistent["metric"])

    strongest_minimum = float(strongest["minimum"])
    strongest_maximum = float(strongest["maximum"])
    consistent_spread = float(most_consistent["spread"])

    if strongest_minimum >= 1.0:
        magnitude_sentence = (
            f"**{strongest_label} is the clearest separation from ordinary "
            "forecast error:** its typical observed-versus-no-CP gap stays "
            f"between **{strongest_minimum:.2f}× and "
            f"{strongest_maximum:.2f}× the model's ordinary Pre-CP error** "
            "across h=1, h=2, and h=5."
        )
    else:
        magnitude_sentence = (
            f"**{strongest_label} has the largest typical separation**, but "
            "at least one horizon remains within the model's ordinary Pre-CP "
            f"error scale at **{strongest_minimum:.2f}× MAE**."
        )

    if strongest["metric"] == most_consistent["metric"]:
        consistency_sentence = (
            " It is also the least sensitive to forecast horizon in this "
            "selection, so both its relative size and its cross-horizon "
            "consistency point in the same direction."
        )
    else:
        consistency_sentence = (
            f" **{consistent_label} is the least sensitive to forecast "
            "horizon**, with only "
            f"**{consistent_spread:.2f}× MAE** between its smallest and largest "
            "typical gaps."
        )

    return (
        f"For {context}, "
        + magnitude_sentence
        + consistency_sentence
        + " These comparisons show how large the counterfactual separation is "
        "relative to normal forecasting error; they are not statistical "
        "significance tests."
    )

# ---------------------------------------------------------------------
# Native-error orientation
# ---------------------------------------------------------------------

METRIC_UNITS = {
    "taxi_trip_count": "trips",
    "taxi_avg_trip_speed": "mph",
    "fhvhv_trip_count": "trips",
    "fhvhv_avg_trip_speed": "mph",
    "subway_ridership": "riders",
}


def build_native_mae_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Return Systemwide Pre-CP Reference MAE by mobility measure and horizon."""
    required = [
        "metric",
        "horizon",
        "pre_cp_reference_mae",
    ]
    missing = [
        column
        for column in required
        if column not in summary.columns
    ]

    if missing:
        raise RuntimeError(
            "Raw 25 summary is missing Reference-MAE fields: "
            + ", ".join(missing)
            + ". Available columns: "
            + ", ".join(str(column) for column in summary.columns)
        )

    # WHY: the rebuilt 15-row summary is explicitly Systemwide × All days ×
    # All dayparts, so these values are the correct hero/orientation yardsticks.
    working = summary.loc[:, required].copy()

    if working.duplicated(["metric", "horizon"]).any():
        raise RuntimeError(
            "Reference MAE table received duplicate Metric × Horizon rows."
        )

    pivot = working.pivot(
        index="metric",
        columns="horizon",
        values="pre_cp_reference_mae",
    ).reindex(METRIC_ORDER)

    rows = []

    for metric in METRIC_ORDER:
        rows.append({
            "Mobility measure": metric_label(metric),
            "Unit": METRIC_UNITS[metric],
            "h=1": float(pivot.loc[metric, 1]),
            "h=2": float(pivot.loc[metric, 2]),
            "h=5": float(pivot.loc[metric, 5]),
        })

    return pd.DataFrame(rows)


def format_native_mae_table(table: pd.DataFrame) -> pd.DataFrame:
    """Format heterogeneous native units without pretending they share one axis."""
    display = table.copy()
    for column in ["h=1", "h=2", "h=5"]:
        display[column] = [
            f"{value:,.2f} {unit}"
            for value, unit in zip(display[column], display["Unit"])
        ]
    return display.drop(columns="Unit")


def build_mae_ruler(example_mae: float) -> go.Figure:
    """Translate MAE units into native units using Taxi average speed h=1."""
    multiples = [0.0, 0.5, 1.0, 2.0]
    native_values = [multiple * example_mae for multiple in multiples]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=native_values,
            y=[1.0] * len(native_values),
            mode="lines+markers+text",
            line={"color": BRAND_COLORS["dark_teal"], "width": 5},
            marker={
                "size": [8, 13, 15, 15],
                "color": [
                    BRAND_COLORS["seafoam"],
                    BRAND_COLORS["seafoam"],
                    BRAND_COLORS["dark_teal"],
                    BRAND_COLORS["terracotta"],
                ],
                "line": {"color": "white", "width": 1},
            },
            text=["0×", "0.5× MAE", "1× MAE", "2× MAE"],
            textposition="top center",
            customdata=[[multiple, native] for multiple, native in zip(multiples, native_values)],
            hovertemplate=(
                "<b>%{customdata[0]:.1f}× MAE</b><br>"
                "%{customdata[1]:.2f} mph<extra></extra>"
            ),
            showlegend=False,
        )
    )
    figure.update_xaxes(
        title_text="Taxi average-speed gap (mph)",
        range=[-0.08 * example_mae, 2.18 * example_mae],
        zeroline=False,
    )
    figure.update_yaxes(visible=False, range=[0.72, 1.30], fixedrange=True)
    figure.add_annotation(
        x=example_mae,
        y=0.79,
        text=f"1× MAE = {example_mae:.2f} mph",
        showarrow=False,
        font={"size": 11, "color": BRAND_COLORS["dark_teal"]},
    )
    return brand_figure(
        figure,
        height=260,
        left=35,
        right=35,
        top=55,
        bottom=65,
    )


# ---------------------------------------------------------------------
# Worked example — make the ×MAE yardstick visible on a real series
# ---------------------------------------------------------------------

WORKED_METRIC = "taxi_avg_trip_speed"
WORKED_HORIZON = 1


@st.cache_data(show_spinner=False)
def build_worked_example_weekly() -> tuple[pd.DataFrame, str, str, str]:
    """
    Return one fixed Taxi Zone × temporal-bucket trajectory for the MAE lesson.

    WHY:
    The worked example stays at one Taxi Zone and one temporal bucket so its
    native-unit gap can be compared directly with the matching Taxi Zone ×
    Weekdays × PM peak Pre-CP Reference MAE.
    """
    source = load_counterfactual_temporal_explorer_metric(
        WORKED_METRIC
    ).copy()

    required = {
        "week_start",
        "target_temporal_bucket",
        "taxi_zone_id",
        "zone",
        "period_complete",
        "observed_level",
        f"no_cp_h{WORKED_HORIZON}",
    }
    missing = sorted(
        required.difference(source.columns)
    )

    if missing:
        raise RuntimeError(
            "Raw 25 worked example is missing temporal counterfactual columns: "
            + ", ".join(missing)
        )

    source["week_start"] = pd.to_datetime(
        source["week_start"],
        errors="coerce",
    )
    source = source.loc[
        source["week_start"].notna()
        & source["period_complete"].fillna(False).astype(bool)
        & source["target_temporal_bucket"].eq("weekday_pm_peak")
    ].copy()

    # Prefer a familiar high-volume location. The selected ID is returned so
    # the page can retrieve that exact slice's Reference MAE from slice_scout.
    preferred = source.loc[
        source["zone"]
        .astype(str)
        .str.contains(
            "Times Sq",
            case=False,
            na=False,
        )
    ].copy()

    if preferred.empty:
        support = (
            source.groupby(
                ["taxi_zone_id", "zone"],
                observed=True,
            )
            .size()
            .reset_index(name="rows")
            .sort_values(
                ["rows", "taxi_zone_id"],
                ascending=[False, True],
            )
        )

        if support.empty:
            raise RuntimeError(
                "No supported Taxi Zone is available for the worked example."
            )

        selected_zone_id = support.iloc[0]["taxi_zone_id"]
        selected_zone_label = str(
            support.iloc[0]["zone"]
        )

    else:
        selected_zone_id = preferred.iloc[0]["taxi_zone_id"]
        selected_zone_label = str(
            preferred.iloc[0]["zone"]
        )

    selected = source.loc[
        pd.to_numeric(
            source["taxi_zone_id"],
            errors="coerce",
        ).eq(float(selected_zone_id))
    ].copy()

    if selected.duplicated("week_start").any():
        raise RuntimeError(
            "Worked example expected one row per week for one "
            "Taxi Zone × temporal bucket."
        )

    selected["observed"] = pd.to_numeric(
        selected["observed_level"],
        errors="coerce",
    )
    selected["no_cp"] = pd.to_numeric(
        selected[f"no_cp_h{WORKED_HORIZON}"],
        errors="coerce",
    )

    selected = (
        selected.loc[
            :,
            [
                "week_start",
                "observed",
                "no_cp",
            ],
        ]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["observed", "no_cp"])
        .sort_values("week_start")
        .reset_index(drop=True)
    )

    return (
        selected,
        str(int(float(selected_zone_id))),
        selected_zone_label,
        TEMPORAL_BUCKET_LABELS["weekday_pm_peak"],
    )


def worked_example_mae(
    slice_scout: pd.DataFrame,
    *,
    zone_id: str,
) -> float:
    """Return the matching Taxi Zone × weekday PM-peak Reference MAE."""
    row = selected_slice_row(
        slice_scout,
        metric=WORKED_METRIC,
        horizon=WORKED_HORIZON,
        geography_type="Taxi Zone",
        geography_id=str(zone_id),
        day_type="Weekdays",
        daypart="PM peak",
    )

    value = float(
        row["pre_cp_reference_mae"]
    )

    if not np.isfinite(value) or value <= 0:
        raise RuntimeError(
            "Worked-example Pre-CP Reference MAE must be finite and positive."
        )

    return value


def build_worked_example_figure(
    weekly: pd.DataFrame,
    mae: float,
) -> tuple[go.Figure, pd.Series, float]:
    """
    Plot observed mobility, its ±1 MAE yardstick, and the estimated no-CP path.

    The band is intentionally centered on observed mobility. It answers the visual
    question: how far would the no-CP estimate have to sit from what actually happened
    before the separation exceeds one ordinary Pre-CP forecast miss?
    """
    plot = weekly.copy()
    plot["band_low"] = plot["observed"] - mae
    plot["band_high"] = plot["observed"] + mae
    plot["gap"] = plot["no_cp"] - plot["observed"]
    plot["gap_xmae"] = plot["gap"].abs() / mae

    largest = plot.loc[plot["gap"].abs().idxmax()].copy()
    outside_share = float(100.0 * plot["gap"].abs().gt(mae).mean())

    figure = go.Figure()

    # Draw the band first so both trajectories remain visually dominant.
    figure.add_trace(
        go.Scatter(
            x=plot["week_start"],
            y=plot["band_low"],
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=plot["week_start"],
            y=plot["band_high"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(131, 197, 190, 0.20)",
            line={"width": 0},
            name="±1 ordinary Pre-CP MAE",
            hoverinfo="skip",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=plot["week_start"],
            y=plot["observed"],
            mode="lines",
            name="Observed",
            line={
                "color": BRAND_COLORS["dark_teal"],
                "width": 3.2,
            },
            hovertemplate=(
                "<b>Observed</b><br>"
                "%{x|%b %d, %Y}<br>"
                "%{y:.3f} mph<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=plot["week_start"],
            y=plot["no_cp"],
            mode="lines",
            name="Estimated no-CP · h=1",
            line={
                "color": BRAND_COLORS["terracotta"],
                "width": 2.8,
                "dash": "dash",
            },
            hovertemplate=(
                "<b>Estimated no-CP · h=1</b><br>"
                "%{x|%b %d, %Y}<br>"
                "%{y:.3f} mph<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=[largest["week_start"]],
            y=[largest["no_cp"]],
            mode="markers",
            marker={
                "size": 12,
                "color": BRAND_COLORS["terracotta"],
                "line": {"color": "white", "width": 1.2},
            },
            name="Largest gap",
            showlegend=False,
            hovertemplate=(
                "<b>Largest absolute gap</b><br>"
                "%{x|%b %d, %Y}<br>"
                f"Observed: {float(largest['observed']):.3f} mph<br>"
                f"Estimated no-CP: {float(largest['no_cp']):.3f} mph<br>"
                f"Gap: {float(largest['gap']):+.3f} mph<br>"
                f"|Gap|: {float(largest['gap_xmae']):.2f}× MAE"
                "<extra></extra>"
            ),
        )
    )

    direction = (
        "no-CP above observed"
        if float(largest["gap"]) > 0
        else "observed above no-CP"
    )
    figure.add_annotation(
        x=largest["week_start"],
        y=largest["no_cp"],
        text=(
            "<b>Largest gap</b><br>"
            f"{abs(float(largest['gap'])):.2f} mph · "
            f"{float(largest['gap_xmae']):.2f}× MAE<br>"
            f"{direction}"
        ),
        showarrow=True,
        arrowhead=2,
        ax=-72,
        ay=-72,
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="rgba(226,149,120,0.45)",
        borderwidth=1,
        font={"size": 11, "color": "#335C67"},
    )

    figure = apply_branding(figure)
    figure.update_layout(
        title={"text": ""},
        height=475,
        hovermode="x unified",
        margin={"l": 70, "r": 25, "t": 35, "b": 65},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
    )
    figure.update_xaxes(title=None)
    figure.update_yaxes(
        title_text="Taxi average speed (mph)",
        gridcolor="rgba(131, 197, 190, 0.18)",
    )

    return figure, largest, outside_share


def build_worked_example_xmae_figure(
    weekly: pd.DataFrame,
    mae: float,
) -> go.Figure:
    """Plot the fixed worked example directly on the ×MAE calibration scale."""
    plot = weekly.copy()
    plot["gap_xmae"] = (plot["no_cp"] - plot["observed"]).abs() / mae

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=plot["week_start"],
            y=plot["gap_xmae"],
            mode="lines",
            name="Absolute gap",
            line={"color": BRAND_COLORS["dark_teal"], "width": 3},
            hovertemplate="%{x|%b %d, %Y}<br>|Gap|: %{y:.2f}× MAE<extra></extra>",
        )
    )
    figure.add_hline(
        y=1.0,
        line={"color": "rgba(51,92,103,0.50)", "width": 1.4, "dash": "dash"},
        annotation_text="1× Reference MAE",
        annotation_position="top left",
    )
    figure.add_hline(
        y=2.0,
        line={"color": "rgba(51,92,103,0.30)", "width": 1.2, "dash": "dot"},
        annotation_text="2× MAE",
        annotation_position="top left",
    )

    figure = apply_branding(figure)
    figure.update_layout(
        title={"text": ""},
        height=460,
        hovermode="x unified",
        margin={"l": 70, "r": 25, "t": 35, "b": 60},
        showlegend=False,
    )
    figure.update_xaxes(title=None)
    figure.update_yaxes(
        title_text="Absolute gap (× Pre-CP Reference MAE)",
        rangemode="tozero",
        gridcolor="rgba(131, 197, 190, 0.18)",
    )
    return figure


def worked_example_takeaway(
    largest: pd.Series,
    outside_share: float,
) -> str:
    """Translate the fixed visual calibration example into one plain-language finding."""
    gap = float(largest["gap"])
    xmae = float(largest["gap_xmae"])
    date_label = pd.Timestamp(largest["week_start"]).strftime("%b %d, %Y")

    relation = (
        "the estimated no-CP speed was above observed speed"
        if gap > 0
        else "observed speed was above the estimated no-CP speed"
    )

    if outside_share >= 60:
        persistence = (
            "The separation was therefore often larger than routine forecast error, "
            "not just at one isolated peak."
        )
    elif outside_share >= 25:
        persistence = (
            "The separation exceeded routine forecast error in a meaningful share of "
            "weeks, but not consistently across the full post-CP period."
        )
    else:
        persistence = (
            "Most weeks remained within one ordinary forecast-error unit, with larger "
            "separations appearing only occasionally."
        )

    return (
        f"For this fixed **Taxi average speed · h=1** example, the no-CP path sits "
        f"outside the **±1 MAE** band in **{outside_share:.1f}%** of post-CP weeks. "
        f"The largest gap occurs in the week of **{date_label}**, when {relation} by "
        f"**{abs(gap):.2f} mph**, or **{xmae:.2f}×** the model's ordinary Pre-CP "
        f"error. {persistence}"
    )



def selected_temporal_buckets(*, day_type: str, daypart: str) -> list[str]:
    """Translate the explorer's day-type/daypart controls into temporal-bucket IDs."""
    prefix = {"All days": None, "Weekdays": "weekday_", "Weekends": "weekend_"}[day_type]
    suffix = {
        "All dayparts": None,
        "Overnight": "overnight",
        "AM peak": "am_peak",
        "Midday": "midday",
        "PM peak": "pm_peak",
        "Evening": "evening",
    }[daypart]

    return [
        bucket for bucket in TEMPORAL_BUCKET_LABELS
        if (prefix is None or bucket.startswith(prefix))
        and (suffix is None or bucket.endswith(suffix))
    ]


def scope_trajectory_geography(
    frame: pd.DataFrame,
    *,
    geography_type: str,
    geography_id: object,
) -> pd.DataFrame:
    """Apply the reader's geography selection to the trajectory source."""
    if geography_type == "Systemwide":
        return frame.copy()

    column = {
        "Borough": "borough",
        "Policy geography": "cbd_spatial_category",
        "Mobility environment": "pre_cp_mobility_environment",
        "Taxi Zone": "taxi_zone_id",
    }.get(geography_type)

    if column is None:
        raise ValueError(
            f"Unsupported geography type: {geography_type}"
        )

    if geography_type == "Taxi Zone":
        # WHY: Raw 25's denominator and slice scout use physical Taxi Zones.
        # Apply the same 57→56 and 105→103 aliases before selecting the path.
        canonical = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).astype("Int64").replace(
            TAXI_ZONE_ALIASES
        )

        mask = canonical.eq(
            int(float(geography_id))
        )

    else:
        mask = (
            frame[column]
            .astype(str)
            .eq(str(geography_id))
        )

    return frame.loc[mask].copy()


def reference_mae_for_slice(
    slice_scout: pd.DataFrame,
    *,
    geography_type: str,
    geography_id: object,
    day_type: str,
    daypart: str,
    metric: str,
    horizon: int,
) -> float | None:
    """Return the exact supported Pre-CP Reference MAE for one Raw 25 slice."""
    match = slice_scout.loc[
        slice_scout["geography_type"].eq(geography_type)
        & slice_scout["geography_id"].astype(str).eq(str(geography_id))
        & slice_scout["day_type"].eq(day_type)
        & slice_scout["daypart"].eq(daypart)
        & slice_scout["metric"].eq(metric)
        & slice_scout["horizon"].eq(int(horizon))
        & slice_scout["support_ok"]
    ]

    if match.empty:
        return None

    if len(match) != 1:
        raise RuntimeError(
            "Expected at most one supported Reference-MAE row for "
            f"{geography_type} · {geography_id} · {day_type} · {daypart} · "
            f"{metric} · h={int(horizon)}; found {len(match)}."
        )

    value = float(
        match.iloc[0]["pre_cp_reference_mae"]
    )

    if not np.isfinite(value) or value <= 0:
        raise RuntimeError(
            "Supported Pre-CP Reference MAE must be finite and positive for "
            f"{metric} · h={int(horizon)}."
        )

    return value


def build_slice_native_trajectory(
    source: pd.DataFrame,
    *,
    geography_type: str,
    geography_id: object,
    day_type: str,
    daypart: str,
    metric: str,
) -> pd.DataFrame:
    """
    Aggregate observed and no-CP paths for the exact explorer slice.

    WHY:
    Count measures are additive, while speed measures must preserve the validated
    activity-weighted aggregation used by the counterfactual overview.
    """
    buckets = selected_temporal_buckets(day_type=day_type, daypart=daypart)
    scoped = scope_trajectory_geography(
        source,
        geography_type=geography_type,
        geography_id=geography_id,
    )
    scoped = scoped.loc[
        scoped["target_temporal_bucket"].isin(buckets)
        & scoped["period_complete"].fillna(False).astype(bool)
    ].copy()

    if scoped.empty:
        return pd.DataFrame()

    scoped["week_start"] = pd.to_datetime(scoped["week_start"], errors="coerce")

    if metric in COUNT_METRICS:
        named_aggs = {
            "support_rows": ("support_rows", "sum"),
            "observed_level": ("observed_level", "sum"),
        }
        for horizon in HORIZONS:
            named_aggs[f"no_cp_h{horizon}"] = (f"no_cp_h{horizon}", "sum")

        weekly = (
            scoped.groupby("week_start", observed=True, sort=False)
            .agg(**named_aggs)
            .reset_index()
        )
    else:
        scoped["observed_numerator"] = (
            pd.to_numeric(scoped["observed_level"], errors="coerce")
            * pd.to_numeric(scoped["observed_weight"], errors="coerce")
        )

        for horizon in HORIZONS:
            scoped[f"no_cp_numerator_h{horizon}"] = (
                pd.to_numeric(scoped[f"no_cp_h{horizon}"], errors="coerce")
                * pd.to_numeric(scoped[f"no_cp_weight_h{horizon}"], errors="coerce")
            )

        named_aggs = {
            "support_rows": ("support_rows", "sum"),
            "observed_numerator": ("observed_numerator", "sum"),
            "observed_weight": ("observed_weight", "sum"),
        }
        for horizon in HORIZONS:
            named_aggs[f"no_cp_numerator_h{horizon}"] = (
                f"no_cp_numerator_h{horizon}",
                "sum",
            )
            named_aggs[f"no_cp_weight_h{horizon}"] = (
                f"no_cp_weight_h{horizon}",
                "sum",
            )

        weekly = (
            scoped.groupby("week_start", observed=True, sort=False)
            .agg(**named_aggs)
            .reset_index()
        )
        weekly["observed_level"] = (
            weekly["observed_numerator"]
            / weekly["observed_weight"].where(weekly["observed_weight"].gt(0))
        )

        for horizon in HORIZONS:
            weekly[f"no_cp_h{horizon}"] = (
                weekly[f"no_cp_numerator_h{horizon}"]
                / weekly[f"no_cp_weight_h{horizon}"].where(
                    weekly[f"no_cp_weight_h{horizon}"].gt(0)
                )
            )

        weekly = weekly[
            [
                "week_start",
                "support_rows",
                "observed_level",
                *[f"no_cp_h{horizon}" for horizon in HORIZONS],
            ]
        ]

    return weekly.replace([np.inf, -np.inf], np.nan).sort_values(
        "week_start", kind="stable"
    ).reset_index(drop=True)


def build_native_trajectory_figure(
    trajectory: pd.DataFrame,
    *,
    metric: str,
) -> go.Figure:
    """Plot observed mobility beside the three no-CP horizon paths in native units."""
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=trajectory["week_start"],
            y=trajectory["observed_level"],
            mode="lines",
            name="Observed",
            line={"color": "#5A6B73", "width": 3.4},
            hovertemplate="%{x|%b %d, %Y}<br>Observed: %{y:,.2f}<extra></extra>",
        )
    )

    horizon_styles = {
        1: (BRAND_COLORS["dark_teal"], "solid", 3.0),
        2: (BRAND_COLORS["seafoam"], "dash", 2.2),
        5: (BRAND_COLORS["terracotta"], "dot", 2.2),
    }
    for horizon in HORIZONS:
        color, dash, width = horizon_styles[horizon]
        figure.add_trace(
            go.Scatter(
                x=trajectory["week_start"],
                y=trajectory[f"no_cp_h{horizon}"],
                mode="lines",
                name=f"No-CP · h={horizon}",
                line={"color": color, "width": width, "dash": dash},
                hovertemplate=(
                    f"%{{x|%b %d, %Y}}<br>No-CP · h={horizon}: "
                    "%{y:,.2f}<extra></extra>"
                ),
            )
        )

    figure = apply_branding(figure)
    figure.update_layout(
        title={"text": ""},
        height=455,
        hovermode="x unified",
        margin={"l": 70, "r": 25, "t": 35, "b": 60},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
    )
    figure.update_xaxes(title=None)
    figure.update_yaxes(
        title_text=f"{metric_label(metric)} ({METRIC_UNITS[metric]})",
        gridcolor="rgba(131, 197, 190, 0.18)",
    )
    return figure


def native_trajectory_takeaway(
    trajectory: pd.DataFrame,
    *,
    metric: str,
    geography_label: str,
    day_type: str,
    daypart: str,
) -> str:
    """Summarize the observed-versus-no-CP trajectory without causal overreach."""
    gap_columns = []
    for horizon in HORIZONS:
        column = f"gap_h{horizon}"
        trajectory[column] = trajectory[f"no_cp_h{horizon}"] - trajectory["observed_level"]
        gap_columns.append(column)

    absolute = trajectory[gap_columns].abs()
    row_index, col_index = np.unravel_index(
        np.nanargmax(absolute.to_numpy(dtype=float)),
        absolute.shape,
    )
    peak_row = trajectory.iloc[row_index]
    peak_horizon = HORIZONS[col_index]
    peak_gap = float(peak_row[f"gap_h{peak_horizon}"])
    peak_date = pd.Timestamp(peak_row["week_start"]).strftime("%b %d, %Y")

    direction = "above" if peak_gap < 0 else "below"
    return (
        f"For **{geography_label} · {day_type} · {daypart} · {metric_label(metric)}**, "
        f"the largest observed-versus-no-CP separation occurs in the week of "
        f"**{peak_date}** at **h={peak_horizon}**. Observed mobility is "
        f"**{abs(peak_gap):,.2f} {METRIC_UNITS[metric]} {direction}** the no-CP path. "
        "This view shows the native mobility trajectories; use the neighboring ×MAE tab "
        "to judge that separation against ordinary Pre-CP forecast error."
    )


def build_slice_calibration_trajectory(
    source: pd.DataFrame,
    slice_scout: pd.DataFrame,
    *,
    geography_type: str,
    geography_id: object,
    day_type: str,
    daypart: str,
    metric: str,
) -> pd.DataFrame:
    """
    Build weekly ×MAE trajectories with the matching slice-specific denominator.

    WHY:
    The corrected Raw 25 contract aggregates the post-CP numerator to the selected
    geography BEFORE calibration. Counts/ridership are summed. Speeds use separate
    observed and no-CP activity weights. The resulting geography-level gap is then
    divided by the Pre-CP Reference MAE for the same geography, day type, daypart,
    metric, and horizon.

    Within each week, selected temporal buckets are summarized with a median and
    IQR. This keeps broad choices such as All dayparts as a pooled set of eligible
    periods instead of inventing one synthetic all-day MAE denominator.
    """
    buckets = selected_temporal_buckets(
        day_type=day_type,
        daypart=daypart,
    )

    scoped = scope_trajectory_geography(
        source,
        geography_type=geography_type,
        geography_id=geography_id,
    )
    scoped = scoped.loc[
        scoped["target_temporal_bucket"].isin(buckets)
        & scoped["period_complete"].fillna(False).astype(bool)
    ].copy()

    if scoped.empty:
        return pd.DataFrame()

    scoped["week_start"] = pd.to_datetime(
        scoped["week_start"],
        errors="coerce",
    )

    period_key = [
        "week_start",
        "target_temporal_bucket",
    ]

    records = []

    for horizon in HORIZONS:
        mae = reference_mae_for_slice(
            slice_scout,
            geography_type=geography_type,
            geography_id=geography_id,
            day_type=day_type,
            daypart=daypart,
            metric=metric,
            horizon=horizon,
        )

        if mae is None:
            continue

        no_cp_column = f"no_cp_h{horizon}"

        if no_cp_column not in scoped.columns:
            continue

        if metric in COUNT_METRICS:
            working = scoped.loc[
                :,
                [
                    *period_key,
                    "observed_level",
                    no_cp_column,
                ],
            ].copy()

            working["observed_level"] = pd.to_numeric(
                working["observed_level"],
                errors="coerce",
            )
            working[no_cp_column] = pd.to_numeric(
                working[no_cp_column],
                errors="coerce",
            )

            working = working.replace(
                [np.inf, -np.inf],
                np.nan,
            ).dropna(
                subset=[
                    "observed_level",
                    no_cp_column,
                ]
            )

            periods = (
                working.groupby(
                    period_key,
                    observed=True,
                    sort=False,
                )
                .agg(
                    observed=(
                        "observed_level",
                        "sum",
                    ),
                    no_cp=(
                        no_cp_column,
                        "sum",
                    ),
                )
                .reset_index()
            )

        else:
            no_cp_weight_column = (
                f"no_cp_weight_h{horizon}"
            )

            required = {
                "observed_level",
                "observed_weight",
                no_cp_column,
                no_cp_weight_column,
            }

            if not required.issubset(scoped.columns):
                continue

            working = scoped.loc[
                :,
                [
                    *period_key,
                    "observed_level",
                    "observed_weight",
                    no_cp_column,
                    no_cp_weight_column,
                ],
            ].copy()

            for column in required:
                working[column] = pd.to_numeric(
                    working[column],
                    errors="coerce",
                )

            working = working.replace(
                [np.inf, -np.inf],
                np.nan,
            ).dropna(
                subset=list(required)
            )

            working = working.loc[
                working["observed_weight"].ge(0)
                & working[no_cp_weight_column].ge(0)
            ].copy()

            working["observed_numerator"] = (
                working["observed_level"]
                * working["observed_weight"]
            )
            working["no_cp_numerator"] = (
                working[no_cp_column]
                * working[no_cp_weight_column]
            )

            periods = (
                working.groupby(
                    period_key,
                    observed=True,
                    sort=False,
                )
                .agg(
                    observed_numerator=(
                        "observed_numerator",
                        "sum",
                    ),
                    observed_weight=(
                        "observed_weight",
                        "sum",
                    ),
                    no_cp_numerator=(
                        "no_cp_numerator",
                        "sum",
                    ),
                    no_cp_weight=(
                        no_cp_weight_column,
                        "sum",
                    ),
                )
                .reset_index()
            )

            periods = periods.loc[
                periods["observed_weight"].gt(0)
                & periods["no_cp_weight"].gt(0)
            ].copy()

            periods["observed"] = (
                periods["observed_numerator"]
                / periods["observed_weight"]
            )
            periods["no_cp"] = (
                periods["no_cp_numerator"]
                / periods["no_cp_weight"]
            )

        if periods.empty:
            continue

        periods["absolute_mae_units"] = (
            periods["no_cp"]
            - periods["observed"]
        ).abs() / mae

        periods = periods.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna(
            subset=["absolute_mae_units"]
        )

        if periods.empty:
            continue

        weekly = (
            periods.groupby(
                "week_start",
                observed=True,
            )["absolute_mae_units"]
            .agg(
                median="median",
                p25=lambda values: values.quantile(0.25),
                p75=lambda values: values.quantile(0.75),
                rows="size",
            )
            .reset_index()
        )

        weekly["horizon"] = int(horizon)
        weekly["pre_cp_reference_mae"] = mae
        records.append(weekly)

    if not records:
        return pd.DataFrame()

    return (
        pd.concat(
            records,
            ignore_index=True,
        )
        .sort_values(
            ["horizon", "week_start"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def build_slice_trajectory_figure(trajectory: pd.DataFrame) -> go.Figure:
    """Show weekly median absolute gap in ×MAE for h=1, h=2, and h=5."""
    figure = go.Figure()
    styles = {
        1: (BRAND_COLORS["dark_teal"], "solid"),
        2: (BRAND_COLORS["terracotta"], "dash"),
        5: (BRAND_COLORS["seafoam"], "dot"),
    }

    for horizon in HORIZONS:
        subset = trajectory.loc[trajectory["horizon"].eq(horizon)].copy()
        if subset.empty:
            continue

        color, dash = styles[horizon]
        figure.add_trace(
            go.Scatter(
                x=pd.concat([subset["week_start"], subset["week_start"].iloc[::-1]]),
                y=pd.concat([subset["p75"], subset["p25"].iloc[::-1]]),
                fill="toself",
                fillcolor=color,
                opacity=0.09,
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=subset["week_start"],
                y=subset["median"],
                mode="lines",
                name=f"h={horizon}",
                line={"color": color, "width": 2.7, "dash": dash},
                customdata=subset[["p25", "p75", "rows"]].to_numpy(),
                hovertemplate=(
                    f"<b>h={horizon}</b><br>%{{x|%b %d, %Y}}<br>"
                    "Median |gap|: %{y:.2f}× MAE<br>"
                    "P25–P75: %{customdata[0]:.2f}–%{customdata[1]:.2f}×<br>"
                    "Rows: %{customdata[2]:,.0f}<extra></extra>"
                ),
            )
        )

    figure.add_hline(
        y=1.0,
        line={"color": "rgba(51,92,103,0.45)", "width": 1.4, "dash": "dash"},
        annotation_text="1× Reference MAE",
        annotation_position="top left",
    )
    figure.add_hline(
        y=2.0,
        line={"color": "rgba(51,92,103,0.28)", "width": 1.2, "dash": "dot"},
        annotation_text="2× MAE",
        annotation_position="top left",
    )

    figure = apply_branding(figure)
    figure.update_layout(
        title={"text": ""},
        height=455,
        hovermode="x unified",
        margin={"l": 70, "r": 25, "t": 35, "b": 60},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
    )
    figure.update_xaxes(title=None)
    figure.update_yaxes(
        title_text="Weekly median absolute gap (× Pre-CP Reference MAE)",
        rangemode="tozero",
        gridcolor="rgba(131, 197, 190, 0.18)",
    )
    return figure


def slice_trajectory_takeaway(
    trajectory: pd.DataFrame,
    *,
    metric: str,
    geography_label: str,
    day_type: str,
    daypart: str,
) -> str:
    """Summarize the selected weekly calibration trajectory."""
    horizon_summary = (
        trajectory.groupby("horizon", observed=True)["median"]
        .agg(period_median="median", peak="max")
        .reset_index()
    )
    strongest = horizon_summary.loc[horizon_summary["period_median"].idxmax()]
    peak_row = trajectory.loc[trajectory["median"].idxmax()]
    peak_date = pd.Timestamp(peak_row["week_start"]).strftime("%b %d, %Y")

    return (
        f"For **{geography_label} · {day_type} · {daypart} · {metric_label(metric)}**, "
        f"the largest typical separation across the three horizons is **h={int(strongest['horizon'])}**, "
        f"with a median weekly calibration level of **{float(strongest['period_median']):.2f}× MAE**. "
        f"The strongest single weekly median occurs in the week of **{peak_date}** at "
        f"**{float(peak_row['median']):.2f}× MAE**. The chart summarizes absolute gap "
        "magnitude at the same row-level calibration grain as the range strip; it is not "
        "a confidence interval or significance test."
    )


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------

st.caption(PAGE_CAPTION)
st.title(PAGE_TITLE)
st.markdown(
    f'<p class="app-subtitle">{PAGE_SUBTITLE}</p>',
    unsafe_allow_html=True,
)

try:
    histogram, summary, slice_scout = load_counterfactual_raw25_inputs()
except FileNotFoundError:
    st.error("Counterfactual calibration data is temporarily unavailable.")
    st.stop()


# ---------------------------------------------------------------------
# Opening — minimum context needed to read the hero
# ---------------------------------------------------------------------

st.markdown("### Put the gap on the model's own error scale")
st.write(
    "Each post-CP counterfactual gap is divided by that forecasting job's "
    "**Pre-CP validation mean absolute error (MAE)**. A gap of **1× MAE** is the "
    "same size as one ordinary average absolute Pre-CP forecast miss; **2× MAE** "
    "is twice that error scale. Treat ×MAE as an **error yardstick**, not a "
    "significance test, confidence level, or causal threshold."
)

# ---------------------------------------------------------------------
# Hero — all 15 systemwide calibration jobs
# ---------------------------------------------------------------------

st.markdown("### How large are the post-CP gaps relative to ordinary forecast error?")
st.write(
    "Across the full NYC system, every mobility measure can share one calibration ruler. "
    "**Read each three-row mobility-measure group together**: h=1, h=2, and h=5 show "
    "whether the magnitude story holds as forecast horizon changes. Within each row, "
    "the diamond marks the **median absolute post-CP gap**; the thicker segment reaches "
    "P75, the vertical tick marks P90, and the lighter tail ends at P95."
)
render_calibration_key()
st.caption(
    "Systemwide · full post-CP period · all Taxi Zones · all days and dayparts · "
    "15 Metric × Horizon forecasting jobs"
)
st.plotly_chart(
    build_calibration_range_strip(summary),
    width="stretch",
    config=PLOT_CONFIG,
    key="raw25_calibration_range_strip",
)
render_chart_insight(hero_takeaway(summary))

st.caption(
    "A median below 1× MAE means the typical gap is smaller than one ordinary average "
    "absolute validation miss; a median above 1× means it is larger. The farther above "
    "that ordinary error scale — especially when the pattern persists across all three "
    "horizons — the clearer the descriptive separation. Crossing 1× is not a statistical "
    "significance threshold."
)


# ---------------------------------------------------------------------
# Fixed visual example — show what 1× MAE looks like through time
# ---------------------------------------------------------------------

st.markdown("### See the same fixed example two ways")
st.write(
    "The summary above puts all 15 jobs on one calibration ruler. The fixed example "
    "below keeps **Taxi average speed · h=1 · one Taxi Zone · weekday PM peak** so the "
    "native mobility path and the ×MAE calibration view refer to exactly the same series."
)

(
    worked_weekly,
    worked_zone_id,
    worked_zone_label,
    worked_bucket_label,
) = build_worked_example_weekly()
worked_mae = worked_example_mae(
    slice_scout,
    zone_id=worked_zone_id,
)
worked_figure, worked_largest, worked_outside_share = build_worked_example_figure(
    worked_weekly,
    worked_mae,
)
worked_xmae_figure = build_worked_example_xmae_figure(worked_weekly, worked_mae)

worked_cols = st.columns(4)
worked_cols[0].metric("Ordinary error scale", f"{worked_mae:.2f} mph")
worked_cols[1].metric(
    "Largest post-CP gap",
    f"{abs(float(worked_largest['gap'])):.2f} mph",
)
worked_cols[2].metric(
    "Largest gap in ×MAE",
    f"{float(worked_largest['gap_xmae']):.2f}×",
)
worked_cols[3].metric(
    "Weeks outside ±1 MAE",
    f"{worked_outside_share:.1f}%",
)

worked_actuals_tab, worked_mae_tab = st.tabs(
    ["Observed vs no-CP", "Gap in ×MAE"]
)

with worked_actuals_tab:
    st.caption(
        f"{worked_zone_label} · {worked_bucket_label} · Taxi average speed · h=1"
    )
    st.plotly_chart(
        worked_figure,
        width="stretch",
        config=PLOT_CONFIG,
        key="raw25_worked_native_timeseries",
    )
    st.caption(
        "Observed speed and the estimated no-CP path are shown in mph. The shaded "
        "region is ±1 ordinary Pre-CP MAE around observed speed — an error yardstick, "
        "not a confidence interval."
    )
    render_chart_insight(
        worked_example_takeaway(
            worked_largest,
            worked_outside_share,
        )
    )

with worked_mae_tab:
    st.caption(
        f"{worked_zone_label} · {worked_bucket_label} · Taxi average speed · h=1"
    )
    st.plotly_chart(
        worked_xmae_figure,
        width="stretch",
        config=PLOT_CONFIG,
        key="raw25_worked_xmae_timeseries",
    )
    st.caption(
        "The same weekly separation is now expressed directly in ordinary Pre-CP MAE "
        "units. 1× means the gap equals one average absolute validation miss."
    )
    render_chart_insight(
        "This is the same fixed trajectory as the neighboring tab, but the vertical "
        "axis removes the native mph scale and asks only how large each weekly gap is "
        "relative to ordinary forecasting error."
    )


# ---------------------------------------------------------------------
# Optional orientation — deeper explanation of the ×MAE scale
# ---------------------------------------------------------------------

with st.expander("How to read the ×MAE scale", expanded=False):
    st.markdown("### First, what does an ordinary forecast miss look like?")
    st.write(
        "The counterfactual gap is easier to interpret once we know the forecasting "
        "system's ordinary error in the original units. The table below shows the "
        "**Pre-CP validation mean absolute error (MAE)** for every mobility measure "
        "and forecast horizon. MAE is the average size of an absolute forecast miss: "
        "lower is better, and the units stay familiar — trips, mph, or riders."
    )

    native_mae_table = build_native_mae_table(summary)
    st.dataframe(
        format_native_mae_table(native_mae_table),
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "These numbers are not directly comparable across mobility measures because "
        "their native units differ. Their role is to establish the ordinary error "
        "scale for each mobility measure × forecast horizon."
    )


    # ---------------------------------------------------------------------
    # Bridge — convert a native gap into MAE units
    # ---------------------------------------------------------------------

    st.markdown("### Put the counterfactual gap on that same error scale")
    st.write(
        "For each row, we divide the counterfactual gap by that job's Pre-CP validation "
        "MAE. This gives a common **×MAE** ruler: a 1× gap is the same size as one "
        "ordinary average absolute validation error for that forecasting job. A 2× gap "
        "is twice that error scale. This is a magnitude comparison — **not** a "
        "significance test, confidence level, or causal threshold."
    )

    example_row = summary.loc[
        summary["metric"].eq("taxi_avg_trip_speed")
        & summary["horizon"].eq(1)
    ]
    if len(example_row) != 1:
        raise RuntimeError("Expected one Taxi average speed · h=1 calibration row.")
    example_mae = float(example_row.iloc[0]["pre_cp_reference_mae"])

    st.markdown("#### Worked example · Taxi average speed · h=1")
    st.plotly_chart(
        build_mae_ruler(example_mae),
        width="stretch",
        config=PLOT_CONFIG,
        key="raw25_mae_ruler",
    )
    st.caption(
        f"Here, ordinary Pre-CP MAE is {example_mae:.2f} mph. A 1.20 mph gap is about "
        f"0.5× MAE; {example_mae:.2f} mph is 1× MAE; and {2 * example_mae:.2f} mph "
        "is 2× MAE. The same conversion is done in each job's own native units."
    )

    with st.container(border=True):
        st.markdown("#### So what should I look for?")
        st.write(
            "Think of MAE as an **error yardstick**, not a pass/fail threshold. A typical "
            "gap well below 1× MAE is small compared with the errors this forecasting "
            "system ordinarily makes, so the observed and no-CP paths are not strongly "
            "separated on this descriptive error scale. As the gap grows to one, two, "
            "or several times MAE, that separation becomes increasingly large relative "
            "to ordinary forecast imprecision. **Bigger is not worse**: a real post-CP mobility "
            "shift—whatever its cause—could produce a very large gap."
        )
        st.write(
            "Then check **h=1, h=2, and h=5 together for the same mobility measure**. "
            "If all three tell a similar story, the magnitude conclusion is less dependent "
            "on forecast horizon. If they differ sharply, the apparent separation is more "
            "horizon-sensitive."
        )
        st.caption(
            "There is no magic cutoff at 1× or 2× MAE. These comparisons do not test a "
            "null hypothesis, establish statistical significance, or identify why the "
            "actual and no-CP paths differ."
        )

# ---------------------------------------------------------------------
# Interactive exploration — one place / time slice, same calibration grammar
# ---------------------------------------------------------------------

with exploration_section(
    key="raw25_calibration_exploration_area",
    title="Explore calibration by place and time",
    description=(
        "Choose a geography and time slice, then read each mobility measure's h=1, "
        "h=2, and h=5 rows together. This shows both how large the gap is relative to "
        "ordinary Pre-CP error and whether that magnitude story is stable across "
        "forecast horizons. Unsupported measures are omitted rather than treated as zero."
    ),
):
    control_a, control_b, control_c, control_d = st.columns([1.0, 1.35, 0.9, 0.9])

    with control_a:
        selected_geography_type = st.selectbox(
            "Geography level",
            GEOGRAPHY_TYPES,
            index=0,
            key="raw25_explorer_geography_type",
        )
    with control_c:
        selected_day_type = st.selectbox(
            "Day type",
            DAY_TYPES,
            index=0,
            key="raw25_explorer_day_type",
        )
    with control_d:
        selected_daypart = st.selectbox(
            "Daypart",
            DAYPARTS,
            index=0,
            key="raw25_explorer_daypart",
        )

    geography_options = supported_geographies_for_grid(
        slice_scout,
        geography_type=selected_geography_type,
        day_type=selected_day_type,
        daypart=selected_daypart,
    )

    if geography_options.empty:
        st.warning(
            "No reader-facing geography has supported calibration results for "
            "this time slice."
        )
    else:
        geography_lookup = dict(
            zip(
                geography_options["geography_id"].astype(str),
                geography_options["display_label"],
            )
        )
        with control_b:
            selected_geography_id = st.selectbox(
                "Geography",
                geography_options["geography_id"].astype(str).tolist(),
                format_func=lambda value: geography_lookup[str(value)],
                key="raw25_explorer_geography",
            )

        selected_geography_label = geography_lookup[str(selected_geography_id)]
        selected_grid = selected_geography_grid(
            slice_scout,
            geography_type=selected_geography_type,
            geography_id=str(selected_geography_id),
            day_type=selected_day_type,
            daypart=selected_daypart,
        )

        st.markdown("### How do the available mobility measures compare here?")
        st.caption(
            f"{selected_geography_label} · {selected_day_type} · {selected_daypart}"
        )
        render_calibration_key()
        st.plotly_chart(
            build_calibration_range_strip(selected_grid),
            width="stretch",
            config=PLOT_CONFIG,
            key="raw25_selected_geography_range_strip",
        )
        render_chart_insight(
            geography_grid_takeaway(
                selected_grid,
                geography_label=selected_geography_label,
                day_type=selected_day_type,
                daypart=selected_daypart,
            )
        )

        st.markdown("### Follow the selected slice through time")
        st.write(
            "Both views **inherit the geography, day type, and daypart selected above**. "
            "Choose the mobility measure once, then switch tabs to see the native "
            "observed-versus-no-CP paths or the same separation on the ×MAE scale."
        )

        available_metrics = [
            metric for metric in METRIC_ORDER
            if metric in set(selected_grid["metric"].astype(str))
        ]
        trajectory_metric = st.selectbox(
            "Trajectory measure",
            available_metrics,
            format_func=metric_label,
            key="raw25_trajectory_metric",
        )

        trajectory_source = load_counterfactual_temporal_explorer_metric(
            trajectory_metric
        ).copy()
        native_trajectory = build_slice_native_trajectory(
            trajectory_source,
            geography_type=selected_geography_type,
            geography_id=selected_geography_id,
            day_type=selected_day_type,
            daypart=selected_daypart,
            metric=trajectory_metric,
        )
        calibration_trajectory = build_slice_calibration_trajectory(
            trajectory_source,
            slice_scout,
            geography_type=selected_geography_type,
            geography_id=selected_geography_id,
            day_type=selected_day_type,
            daypart=selected_daypart,
            metric=trajectory_metric,
        )

        trajectory_scope = (
            f"{selected_geography_label} · {selected_day_type} · {selected_daypart} · "
            f"{metric_label(trajectory_metric)}"
        )
        actuals_tab, mae_tab = st.tabs(["Observed vs no-CP", "Gap in ×MAE"])

        with actuals_tab:
            if native_trajectory.empty:
                st.info("No supported native-unit trajectory is available for this selection.")
            else:
                st.caption(f"{trajectory_scope} · h=1 / h=2 / h=5")
                st.plotly_chart(
                    build_native_trajectory_figure(
                        native_trajectory,
                        metric=trajectory_metric,
                    ),
                    width="stretch",
                    config=PLOT_CONFIG,
                    key="raw25_interactive_native_trajectory",
                )
                render_chart_insight(
                    native_trajectory_takeaway(
                        native_trajectory.copy(),
                        metric=trajectory_metric,
                        geography_label=selected_geography_label,
                        day_type=selected_day_type,
                        daypart=selected_daypart,
                    )
                )

        with mae_tab:
            if calibration_trajectory.empty:
                st.info("No supported ×MAE trajectory is available for this selection.")
            else:
                st.caption(f"{trajectory_scope} · h=1 / h=2 / h=5")
                st.plotly_chart(
                    build_slice_trajectory_figure(calibration_trajectory),
                    width="stretch",
                    config=PLOT_CONFIG,
                    key="raw25_interactive_calibration_trajectory",
                )
                render_chart_insight(
                    slice_trajectory_takeaway(
                        calibration_trajectory,
                        metric=trajectory_metric,
                        geography_label=selected_geography_label,
                        day_type=selected_day_type,
                        daypart=selected_daypart,
                    )
                )


        min_support = int(selected_grid["rows"].min())
        available_jobs = int(len(selected_grid))
        shown_metrics = set(selected_grid["metric"].astype(str).unique())
        missing_metrics = [
            metric_label(metric)
            for metric in METRIC_ORDER
            if metric not in shown_metrics
        ]

        support_note = (
            f"{available_jobs} supported mobility measure × forecast horizon combinations are shown; "
            f"the smallest has {min_support:,} eligible rows."
        )
        if missing_metrics:
            support_note += " Not available here: " + ", ".join(missing_metrics) + "."
        support_note += " Missing measures are omitted, not treated as zero."
        st.caption(support_note)


# ---------------------------------------------------------------------
# Closing synthesis
# ---------------------------------------------------------------------

st.markdown("### What this page establishes")
st.write(
    "Calibration answers two related questions: **how large is the observed-versus-no-CP "
    "separation compared with ordinary forecast error, and does that magnitude story "
    "hold across forecast horizons?** When typical gaps remain small relative to the "
    "ordinary error scale, the page has limited descriptive leverage for separating "
    "the counterfactual gap from normal forecast imprecision. When large gaps persist "
    "across h=1, h=2, and h=5, the actual and no-CP paths are clearly farther apart "
    "on the model's own historical error scale. This still does not establish "
    "statistical significance, a null-hypothesis result, or causality."
)

with st.expander("How this page works"):
    st.write(
        "Each post-CP counterfactual gap is divided by the Pre-CP Reference MAE for "
        "the same geography, day type, daypart, mobility measure, and forecast horizon. "
        "The systemwide chart therefore uses Systemwide × All days × All dayparts "
        "Reference error, while the explorer switches to the matching local error scale. "
        "P50, P75, P90, and P95 summarize absolute gap magnitude, and h=1, h=2, and "
        "h=5 remain grouped so horizon consistency is visible."
    )

st.caption(
    "Evidence and interpretation boundary: MAE calibration is descriptive. It does not "
    "convert the counterfactual gap into a p-value, confidence interval, probability of "
    "a congestion-pricing effect, or causal significance threshold. A value above 1× MAE "
    "means the gap is larger than one ordinary average absolute Pre-CP validation error; "
    "it does not mean the gap is statistically significant. Robustness to alternative "
    "analytical choices is evaluated separately on the preceding page."
)
