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
PAGE_TITLE = "How large is the no-CP gap relative to ordinary forecast error?"
PAGE_SUBTITLE = (
    "Put the observed-versus-no-CP gap on the same scale as the forecasting "
    "system's ordinary Pre-CP validation error, then see where the typical gap "
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
        text="<b>1× ordinary Pre-CP MAE</b>",
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
        title_text="Absolute post-CP gap in Pre-CP validation MAE units",
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
        title_text="Absolute post-CP gap in Pre-CP validation MAE units",
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
    """Return one reader-facing row per mobility measure with native MAE by horizon."""
    required = ["metric", "horizon", "pre_cp_validation_system_mae"]
    missing = [column for column in required if column not in summary.columns]
    if missing:
        raise RuntimeError(
            "Raw 25 summary is missing native MAE fields: "
            + ", ".join(missing)
            + ". Available columns: "
            + ", ".join(str(column) for column in summary.columns)
        )

    # WHY: select the verified loader contract directly. The Raw 25 builder and
    # data-access layer guarantee one native Pre-CP MAE per Metric × Horizon.
    working = summary.loc[:, required].copy()
    if working.duplicated(["metric", "horizon"]).any():
        raise RuntimeError("Native MAE table received duplicate Metric × Horizon rows.")

    pivot = working.pivot(
        index="metric",
        columns="horizon",
        values="pre_cp_validation_system_mae",
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
# Opening — establish the native error scale before using ×MAE
# ---------------------------------------------------------------------

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
example_mae = float(example_row.iloc[0]["pre_cp_validation_system_mae"])

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
# Hero — all 15 systemwide calibration jobs
# ---------------------------------------------------------------------

st.markdown("### How large are the post-CP gaps relative to ordinary forecast error?")
st.write(
    "Now every mobility measure can share one calibration ruler. **Read each three-row "
    "mobility-measure group together**: h=1, h=2, and h=5 show whether the magnitude "
    "story holds "
    "as forecast horizon changes. Within each row, the diamond marks the **median "
    "absolute post-CP gap**; the thicker segment reaches P75, the vertical tick marks "
    "P90, and the lighter tail ends at P95."
)
render_calibration_key()
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
        "Each post-CP counterfactual gap is divided by the corresponding forecasting "
        "job's Pre-CP validation MAE. The denominator is fixed within each Metric × "
        "Horizon job, so the resulting ×MAE values express gap magnitude on that "
        "job's ordinary validation-error scale. The systemwide chart summarizes the "
        "absolute distribution with P50, P75, P90, and P95 and groups h=1, h=2, and "
        "h=5 together so horizon consistency is visible. The explorer repeats the "
        "same grammar for supported geography and time slices."
    )

st.caption(
    "Evidence and interpretation boundary: MAE calibration is descriptive. It does not "
    "convert the counterfactual gap into a p-value, confidence interval, probability of "
    "a congestion-pricing effect, or causal significance threshold. A value above 1× MAE "
    "means the gap is larger than one ordinary average absolute Pre-CP validation error; "
    "it does not mean the gap is statistically significant. Robustness to alternative "
    "analytical choices is evaluated separately on the preceding page."
)
