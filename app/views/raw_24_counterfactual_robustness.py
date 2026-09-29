"""
Raw 24 — Counterfactual Robustness

Pass 2 visual build:
- freeze Agreement Stripes as the hero;
- let hero point selection drive the evidence explorer;
- retain dimension-specific robustness diagnostics;
- scout Taxi Zone × mobility-measure strong / concerning examples;
- let readers verify a selected zone with a Raw-21-style weekly trajectory.

All analytical inputs come from frozen Chapter 5 exports already in the Showcase
repo. This page does not refit models or regenerate counterfactual forecasts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data_access.counterfactuals import (
    load_counterfactual_raw24_zone_scout,
    load_counterfactual_raw24_zone_weekly_inputs,
    load_counterfactual_robustness_inputs,
)
from app.utils.project_branding import (
    BRAND_COLORS,
    apply_branding,
    inject_app_css,
    exploration_section,
    render_chart_insight,
)


# ---------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------

inject_app_css()

CP_START = pd.Timestamp("2025-01-05")

METRIC_ORDER = [
    "taxi_trip_count",
    "taxi_avg_trip_speed",
    "fhvhv_trip_count",
    "fhvhv_avg_trip_speed",
    "subway_ridership",
]

HORIZON_ORDER = [1, 2, 5]

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

SPEED_METRICS = {
    "taxi_avg_trip_speed",
    "fhvhv_avg_trip_speed",
}

STATUS_LABELS = {
    "stable": "Stable",
    "mixed": "Mixed",
    "sensitive": "Sensitive",
}

STATUS_COLORS = {
    "stable": BRAND_COLORS["dark_teal"],
    "mixed": BRAND_COLORS["seafoam"],
    "sensitive": BRAND_COLORS["terracotta"],
}

EVIDENCE_DIMENSIONS = [
    "Population",
    "Weighting",
    "Temporal",
    "Geography",
    "Horizon",
    "Calibration",
]

# Reader-facing physical Taxi Zone aliases.
# WHY: source rows 56/57 and 103/105 represent the same physical places.

UNKNOWN_ZONE_IDS = {
    264,
    265,
}

# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def status_badge_html(
    status: str,
) -> str:
    """Return a compact branded status badge."""
    normalized = str(status).lower()
    label = STATUS_LABELS.get(
        normalized,
        str(status).title(),
    )
    color = STATUS_COLORS.get(
        normalized,
        BRAND_COLORS["dark_teal"],
    )

    return (
        "<span style='"
        f"background:{color};"
        "color:white;"
        "padding:0.18rem 0.55rem;"
        "border-radius:0.45rem;"
        "font-size:0.82rem;"
        "font-weight:700;"
        "'>"
        f"{label}"
        "</span>"
    )


def brand_figure(
    figure: go.Figure,
    *,
    height: int,
    left: int = 70,
    right: int = 30,
    top: int = 55,
    bottom: int = 55,
) -> go.Figure:
    """Apply shared project branding and page-specific dimensions."""
    figure = apply_branding(
        figure
    )

    # WHY: the shared template styles Plotly titles even when a figure has no
    # title text. Explicitly blank untitled figures so Plotly never renders the
    # browser-side placeholder "undefined". Preserve real chart titles.
    if figure.layout.title.text is None:
        figure.update_layout(
            title={"text": ""}
        )

    figure.update_layout(
        height=height,
        margin={
            "l": left,
            "r": right,
            "t": top,
            "b": bottom,
        },
    )
    return figure


def metric_label(
    metric: str,
) -> str:
    """Reader-facing mobility-measure label."""
    return METRIC_LABELS.get(
        metric,
        metric,
    )


# ---------------------------------------------------------------------
# Small frozen robustness package
# ---------------------------------------------------------------------

def build_robustness_qa(
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Validate exactly the contracts needed by the Raw 24 hero/explorer."""
    rows: list[dict[str, str]] = []

    def add(
        check_id: str,
        passed: bool,
        details: str,
    ) -> None:
        rows.append(
            {
                "check_id": check_id,
                "status": (
                    "PASS"
                    if bool(passed)
                    else "FAIL"
                ),
                "details": details,
            }
        )

    matrix = frames["matrix"]
    registry = frames["registry"]

    expected_jobs = {
        (metric, horizon)
        for metric in METRIC_ORDER
        for horizon in HORIZON_ORDER
    }

    validation_qa = frames["qa"]
    add(
        "upstream_5_3_1_qa",
        "status" in validation_qa.columns
        and validation_qa[
            "status"
        ].astype(str).str.upper().eq("PASS").all(),
        f"rows={len(validation_qa)}",
    )

    manifest = frames["manifest"]
    contract_ids = (
        manifest["contract_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
        if "contract_id" in manifest.columns
        else []
    )
    add(
        "handoff_contract",
        contract_ids
        == ["counterfactual_validation_handoff_v1"],
        f"contract_ids={contract_ids}",
    )

    for label in [
        "matrix",
        "registry",
        "population",
        "temporal",
        "calibration",
    ]:
        frame = frames[label]
        jobs = set(
            map(
                tuple,
                frame[
                    [
                        "metric",
                        "horizon",
                    ]
                ].drop_duplicates().to_numpy(),
            )
        )
        add(
            f"{label}_15_job_grid",
            len(frame) == 15
            and jobs == expected_jobs,
            (
                f"rows={len(frame)}; "
                f"unique_jobs={len(jobs)}"
            ),
        )

    matrix_registry = matrix.merge(
        registry[
            [
                "metric",
                "horizon",
                "primary_gap_pct",
                "evidence_status",
                "baseline_gap_direction",
            ]
        ].rename(
            columns={
                "primary_gap_pct": (
                    "registry_primary_gap_pct"
                ),
                "evidence_status": (
                    "registry_evidence_status"
                ),
                "baseline_gap_direction": (
                    "registry_baseline_gap_direction"
                ),
            }
        ),
        on=[
            "metric",
            "horizon",
        ],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    aligned = (
        matrix_registry[
            "_merge"
        ].eq("both").all()
        and matrix_registry[
            "evidence_status"
        ].astype(str).str.lower().eq(
            matrix_registry[
                "registry_evidence_status"
            ].astype(str).str.lower()
        ).all()
        and matrix_registry[
            "baseline_gap_direction"
        ].astype(str).eq(
            matrix_registry[
                "registry_baseline_gap_direction"
            ].astype(str)
        ).all()
        and np.isclose(
            pd.to_numeric(
                matrix_registry[
                    "baseline_abs_gap_pct"
                ],
                errors="coerce",
            ),
            pd.to_numeric(
                matrix_registry[
                    "registry_primary_gap_pct"
                ],
                errors="coerce",
            ).abs(),
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True,
        ).all()
    )

    add(
        "matrix_registry_alignment",
        bool(aligned),
        "15 jobs agree on status, direction, and absolute gap.",
    )

    add(
        "speed_weighting_6_jobs",
        len(
            frames[
                "speed_weighting"
            ]
        ) == 6
        and set(
            frames[
                "speed_weighting"
            ][
                "metric"
            ]
        ) == SPEED_METRICS,
        (
            f"rows="
            f"{len(frames['speed_weighting'])}"
        ),
    )

    add(
        "geography_45_rows",
        len(
            frames[
                "geography"
            ]
        ) == 45
        and frames[
            "geography"
        ][
            "geography_dimension"
        ].nunique() == 3,
        (
            f"rows={len(frames['geography'])}; "
            f"dimensions="
            f"{frames['geography']['geography_dimension'].nunique()}"
        ),
    )

    add(
        "horizon_5_metrics",
        len(
            frames[
                "horizon"
            ]
        ) == 5
        and set(
            frames[
                "horizon"
            ][
                "metric"
            ]
        ) == set(
            METRIC_ORDER
        ),
        f"rows={len(frames['horizon'])}",
    )

    return pd.DataFrame(
        rows
    )


frames = load_counterfactual_robustness_inputs()
qa = build_robustness_qa(
    frames
)

failed_qa = qa.loc[
    qa[
        "status"
    ].eq("FAIL")
]

if not failed_qa.empty:
    st.error(
        "Raw 24 is blocked because a frozen Chapter 5 "
        "robustness contract failed validation."
    )
    st.dataframe(
        failed_qa,
        hide_index=True,
        use_container_width=True,
    )
    st.stop()

matrix = frames["matrix"].copy()
registry = frames["registry"].copy()
population = frames["population"].copy()
speed_weighting = frames["speed_weighting"].copy()
temporal = frames["temporal"].copy()
geography = frames["geography"].copy()
horizon = frames["horizon"].copy()
calibration = frames["calibration"].copy()

metric_order_map = {
    metric: index
    for index, metric in enumerate(
        METRIC_ORDER
    )
}

matrix[
    "_metric_order"
] = matrix[
    "metric"
].map(
    metric_order_map
)

matrix = (
    matrix.sort_values(
        [
            "_metric_order",
            "horizon",
        ]
    )
    .drop(
        columns="_metric_order"
    )
    .reset_index(
        drop=True
    )
)

matrix[
    "job_label"
] = (
    matrix[
        "metric"
    ].map(
        METRIC_LABELS
    )
    + " · h="
    + matrix[
        "horizon"
    ].astype(str)
)


# ---------------------------------------------------------------------
# Frozen hero — Agreement Stripes
# ---------------------------------------------------------------------

def build_agreement_stripes() -> go.Figure:
    """
    Show the three evidence dimensions that truly share a 0–100% scale.

    Temporal, Geography, and Horizon each represent directional agreement.
    Population, Weighting, and Calibration remain side evidence because they use
    different units and should not be forced onto this axis.
    """
    figure = go.Figure()

    series = [
        (
            "Temporal",
            "monthly_direction_agreement_pct",
            BRAND_COLORS[
                "dark_teal"
            ],
        ),
        (
            "Geography",
            "geography_mean_direction_agreement_pct",
            BRAND_COLORS[
                "seafoam"
            ],
        ),
        (
            "Horizon",
            "row_direction_agreement_pct",
            BRAND_COLORS[
                "terracotta"
            ],
        ),
    ]

    # Thin horizontal connectors make each row read as a compact agreement
    # interval without implying time or ordered transitions.
    for _, row in matrix.iterrows():
        values = [
            float(
                row[
                    column
                ]
            )
            for _, column, _ in series
        ]

        figure.add_trace(
            go.Scatter(
                x=[
                    min(values),
                    max(values),
                ],
                y=[
                    row[
                        "job_label"
                    ],
                    row[
                        "job_label"
                    ],
                ],
                mode="lines",
                line={
                    "color": (
                        "rgba(0, 109, 119, 0.20)"
                    ),
                    "width": 4,
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

    for label, column, color in series:
        customdata = np.column_stack(
            [
                matrix[
                    "metric"
                ],
                matrix[
                    "horizon"
                ],
                matrix[
                    "evidence_status"
                ].map(
                    STATUS_LABELS
                ),
                matrix[
                    "baseline_abs_gap_pct"
                ],
            ]
        )

        figure.add_trace(
            go.Scatter(
                x=matrix[
                    column
                ],
                y=matrix[
                    "job_label"
                ],
                mode="markers",
                name=label,
                marker={
                    "size": 12,
                    "color": color,
                },
                customdata=customdata,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    f"{label} agreement: "
                    "%{x:.1f}%<br>"
                    "Overall: %{customdata[2]}<br>"
                    "|System gap|: "
                    "%{customdata[3]:.3f}%<br>"
                    "<i>Click to inspect this result below.</i>"
                    "<extra></extra>"
                ),
            )
        )

    # Metric separators reinforce the five groups while preserving the shared
    # scale and compact row structure.
    for boundary in [
        2.5,
        5.5,
        8.5,
        11.5,
    ]:
        figure.add_hline(
            y=boundary,
            line_width=1,
            line_color=(
                "rgba(0, 109, 119, 0.18)"
            ),
        )

    figure.update_xaxes(
        range=[
            0,
            102,
        ],
        dtick=20,
        ticksuffix="%",
        title_text=(
            "Share agreeing with the system-level direction"
        ),
    )

    figure.update_yaxes(
        autorange="reversed",
        title=None,
        tickfont={
            "size": 12,
        },
    )

    figure.update_layout(
        clickmode="event+select",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0,
        },
    )

    return brand_figure(
        figure,
        height=650,
        left=165,
        right=25,
        top=70,
        bottom=65,
    )


def selected_matrix_row(
    metric: str,
    horizon_value: int,
) -> pd.Series:
    """Return exactly one frozen robustness result."""
    match = matrix.loc[
        matrix[
            "metric"
        ].eq(
            metric
        )
        & matrix[
            "horizon"
        ].eq(
            horizon_value
        )
    ]

    if len(match) != 1:
        raise RuntimeError(
            "Expected one robustness row for "
            f"{metric} · h={horizon_value}; "
            f"found {len(match)}."
        )

    return match.iloc[0]


# ---------------------------------------------------------------------
# Evidence Explorer figures
# ---------------------------------------------------------------------

def build_population_figure(
    metric: str,
    horizon_value: int,
) -> go.Figure:
    """Compare primary and native population definitions."""
    row = population.loc[
        population[
            "metric"
        ].eq(
            metric
        )
        & population[
            "horizon"
        ].eq(
            horizon_value
        )
    ].iloc[0]

    values = [
        float(
            row[
                "primary_gap_pct"
            ]
        ),
        float(
            row[
                "native_gap_pct"
            ]
        ),
    ]

    figure = go.Figure(
        go.Bar(
            x=[
                "Primary supported population",
                "Native available population",
            ],
            y=values,
            marker={
                "color": [
                    BRAND_COLORS[
                        "dark_teal"
                    ],
                    BRAND_COLORS[
                        "seafoam"
                    ],
                ]
            },
            text=[
                f"{value:+.3f}%"
                for value in values
            ],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "No-CP − observed: %{y:+.3f}%"
                "<extra></extra>"
            ),
        )
    )

    figure.add_hline(
        y=0,
        line_width=1,
        line_dash="dot",
    )
    figure.update_yaxes(
        title_text="No-CP − observed (%)",
    )

    return brand_figure(
        figure,
        height=410,
        left=70,
        right=25,
        top=40,
        bottom=80,
    )


def build_weighting_figure(
    metric: str,
    horizon_value: int,
) -> go.Figure | None:
    """Compare the two frozen activity-weighting specifications."""
    match = speed_weighting.loc[
        speed_weighting[
            "metric"
        ].eq(
            metric
        )
        & speed_weighting[
            "horizon"
        ].eq(
            horizon_value
        )
    ]

    if match.empty:
        return None

    row = match.iloc[0]

    values = [
        float(
            row[
                "world_specific_gap_pct"
            ]
        ),
        float(
            row[
                "common_observed_gap_pct"
            ]
        ),
    ]

    figure = go.Figure(
        go.Bar(
            x=[
                "World-specific activity weights",
                "Common observed weights",
            ],
            y=values,
            marker={
                "color": [
                    BRAND_COLORS[
                        "dark_teal"
                    ],
                    BRAND_COLORS[
                        "seafoam"
                    ],
                ]
            },
            text=[
                f"{value:+.3f}%"
                for value in values
            ],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "No-CP − observed: %{y:+.3f}%"
                "<extra></extra>"
            ),
        )
    )

    figure.add_hline(
        y=0,
        line_width=1,
        line_dash="dot",
    )
    figure.update_yaxes(
        title_text="No-CP − observed (%)",
    )

    return brand_figure(
        figure,
        height=410,
        left=70,
        right=25,
        top=40,
        bottom=80,
    )


def build_temporal_figure(
    metric: str,
    horizon_value: int,
) -> go.Figure:
    """Show the frozen range of monthly gaps."""
    row = temporal.loc[
        temporal[
            "metric"
        ].eq(
            metric
        )
        & temporal[
            "horizon"
        ].eq(
            horizon_value
        )
    ].iloc[0]

    selected = selected_matrix_row(
        metric,
        horizon_value,
    )

    baseline_abs = float(
        selected[
            "baseline_abs_gap_pct"
        ]
    )
    signed_baseline = (
        baseline_abs
        if selected[
            "baseline_gap_direction"
        ]
        == "no_cp_above_observed"
        else -baseline_abs
    )

    minimum = float(
        row[
            "monthly_gap_pct_min"
        ]
    )
    maximum = float(
        row[
            "monthly_gap_pct_max"
        ]
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=[
                minimum,
                maximum,
            ],
            y=[
                "Monthly gaps",
                "Monthly gaps",
            ],
            mode="lines+markers",
            line={
                "width": 9,
                "color": BRAND_COLORS[
                    "seafoam"
                ],
            },
            marker={
                "size": 10,
                "color": BRAND_COLORS[
                    "dark_teal"
                ],
            },
            name="Monthly range",
            hovertemplate=(
                "Monthly endpoint: %{x:+.3f}%"
                "<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=[
                signed_baseline
            ],
            y=[
                "Monthly gaps"
            ],
            mode="markers",
            marker={
                "size": 17,
                "symbol": "diamond",
                "color": BRAND_COLORS[
                    "terracotta"
                ],
            },
            name="Full-period gap",
            hovertemplate=(
                "Full-period gap: %{x:+.3f}%"
                "<extra></extra>"
            ),
        )
    )

    figure.add_vline(
        x=0,
        line_width=1,
        line_dash="dot",
    )
    figure.update_xaxes(
        title_text="No-CP − observed (%)",
    )
    figure.update_yaxes(
        title=None,
    )

    return brand_figure(
        figure,
        height=300,
        left=95,
        right=25,
        top=40,
        bottom=60,
    )


def build_geography_figure(
    metric: str,
    horizon_value: int,
) -> go.Figure:
    """Show agreement across the three frozen geography definitions."""
    rows = geography.loc[
        geography[
            "metric"
        ].eq(
            metric
        )
        & geography[
            "horizon"
        ].eq(
            horizon_value
        )
    ].copy()

    rows = rows.sort_values(
        "direction_agreement_pct"
    )

    figure = go.Figure(
        go.Bar(
            x=rows[
                "direction_agreement_pct"
            ],
            y=rows[
                "geography_dimension"
            ],
            orientation="h",
            marker={
                "color": BRAND_COLORS[
                    "dark_teal"
                ],
            },
            customdata=np.column_stack(
                [
                    rows[
                        "group_count"
                    ],
                    rows[
                        "gap_pct_min"
                    ],
                    rows[
                        "gap_pct_max"
                    ],
                ]
            ),
            text=rows[
                "direction_agreement_pct"
            ].map(
                lambda value: (
                    f"{value:.1f}%"
                )
            ),
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Directional agreement: %{x:.1f}%<br>"
                "Groups: %{customdata[0]:.0f}<br>"
                "Gap range: %{customdata[1]:+.3f}% to "
                "%{customdata[2]:+.3f}%"
                "<extra></extra>"
            ),
        )
    )

    figure.update_xaxes(
        range=[
            0,
            105,
        ],
        ticksuffix="%",
        title_text=(
            "Groups sharing the system-level direction"
        ),
    )
    figure.update_yaxes(
        title=None,
    )

    return brand_figure(
        figure,
        height=360,
        left=160,
        right=35,
        top=35,
        bottom=60,
    )


def build_horizon_figure(
    metric: str,
) -> go.Figure:
    """Keep the three system-level horizons visible together."""
    row = horizon.loc[
        horizon[
            "metric"
        ].eq(
            metric
        )
    ].iloc[0]

    values = [
        float(
            row[
                "h1_gap_pct"
            ]
        ),
        float(
            row[
                "h2_gap_pct"
            ]
        ),
        float(
            row[
                "h5_gap_pct"
            ]
        ),
    ]

    figure = go.Figure(
        go.Bar(
            x=[
                "h=1",
                "h=2",
                "h=5",
            ],
            y=values,
            marker={
                "color": [
                    BRAND_COLORS[
                        "dark_teal"
                    ],
                    BRAND_COLORS[
                        "seafoam"
                    ],
                    BRAND_COLORS[
                        "terracotta"
                    ],
                ]
            },
            text=[
                f"{value:+.3f}%"
                for value in values
            ],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "No-CP − observed: %{y:+.3f}%"
                "<extra></extra>"
            ),
        )
    )

    figure.add_hline(
        y=0,
        line_width=1,
        line_dash="dot",
    )
    figure.update_yaxes(
        title_text="No-CP − observed (%)",
    )

    return brand_figure(
        figure,
        height=380,
        left=70,
        right=25,
        top=40,
        bottom=60,
    )


def build_calibration_figure(
    metric: str,
    horizon_value: int,
) -> go.Figure:
    """Put divergence on the frozen Pre-CP validation-MAE scale."""
    row = calibration.loc[
        calibration[
            "metric"
        ].eq(
            metric
        )
        & calibration[
            "horizon"
        ].eq(
            horizon_value
        )
    ].iloc[0]

    values = [
        float(
            row[
                "median_absolute_mae_units"
            ]
        ),
        float(
            row[
                "p90_absolute_mae_units"
            ]
        ),
    ]

    figure = go.Figure(
        go.Bar(
            x=[
                "Median absolute gap",
                "90th percentile absolute gap",
            ],
            y=values,
            marker={
                "color": [
                    BRAND_COLORS[
                        "dark_teal"
                    ],
                    BRAND_COLORS[
                        "seafoam"
                    ],
                ]
            },
            text=[
                f"{value:.3f}×"
                for value in values
            ],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "%{y:.3f} Pre-CP MAE units"
                "<extra></extra>"
            ),
        )
    )

    figure.add_hline(
        y=1,
        line_dash="dot",
        line_width=1,
        annotation_text=(
            "1× ordinary Pre-CP MAE"
        ),
        annotation_position="top right",
    )
    figure.update_yaxes(
        title_text="Pre-CP validation MAE units",
        rangemode="tozero",
    )

    return brand_figure(
        figure,
        height=400,
        left=70,
        right=25,
        top=45,
        bottom=80,
    )



# ---------------------------------------------------------------------
# One consistent Evidence Detail visual
# ---------------------------------------------------------------------

def build_evidence_detail_figure(
    metric: str,
    horizon_value: int,
    dimension: str,
) -> tuple[go.Figure, str]:
    """
    Render every evidence choice with the same visual grammar.

    WHY:
    Raw 24 should not change chart *type* when the reader changes Evidence.
    Every view is therefore a horizontal dot plot:
      - y = named alternatives / checks;
      - x = the quantity native to that evidence dimension;
      - dots = the frozen values being compared.

    The x-axis unit can change because the underlying evidence genuinely uses
    different units. The visual grammar does not.
    """
    labels: list[str]
    values: list[float]
    axis_title: str
    suffix: str
    caption: str
    zero_line = False
    reference_one = False

    if dimension == "Population":
        row = population.loc[
            population["metric"].eq(metric)
            & population["horizon"].eq(horizon_value)
        ].iloc[0]

        labels = [
            "Primary supported population",
            "Native available population",
        ]
        values = [
            float(row["primary_gap_pct"]),
            float(row["native_gap_pct"]),
        ]
        axis_title = "No-CP − observed (%)"
        suffix = "%"
        zero_line = True
        caption = (
            "The population alternative preserves the same direction; "
            f"the absolute shift is {float(row['absolute_gap_pct_shift_pp']):.3f} pp."
        )

    elif dimension == "Weighting":
        match = speed_weighting.loc[
            speed_weighting["metric"].eq(metric)
            & speed_weighting["horizon"].eq(horizon_value)
        ]

        if match.empty:
            raise ValueError(
                "Weighting evidence applies only to average-speed metrics."
            )

        row = match.iloc[0]
        labels = [
            "World-specific activity weights",
            "Common observed weights",
        ]
        values = [
            float(row["world_specific_gap_pct"]),
            float(row["common_observed_gap_pct"]),
        ]
        axis_title = "No-CP − observed (%)"
        suffix = "%"
        zero_line = True
        caption = (
            "The weighting alternative preserves the same direction; "
            f"the gap changes by {abs(float(row['gap_pct_shift_pp'])):.3f} pp."
        )

    elif dimension == "Temporal":
        row = temporal.loc[
            temporal["metric"].eq(metric)
            & temporal["horizon"].eq(horizon_value)
        ].iloc[0]
        selected = selected_matrix_row(metric, horizon_value)

        baseline_abs = float(selected["baseline_abs_gap_pct"])
        signed_baseline = (
            baseline_abs
            if selected["baseline_gap_direction"] == "no_cp_above_observed"
            else -baseline_abs
        )

        labels = [
            "Lowest monthly gap",
            "Full-period gap",
            "Highest monthly gap",
        ]
        values = [
            float(row["monthly_gap_pct_min"]),
            signed_baseline,
            float(row["monthly_gap_pct_max"]),
        ]
        axis_title = "No-CP − observed (%)"
        suffix = "%"
        zero_line = True
        caption = (
            f"{int(row['month_count'])} post-CP months · "
            f"{float(row['monthly_direction_agreement_pct']):.1f}% share "
            "the full-period direction · "
            f"{int(row['monthly_sign_flips'])} sign flips."
        )

    elif dimension == "Geography":
        rows = geography.loc[
            geography["metric"].eq(metric)
            & geography["horizon"].eq(horizon_value)
        ].copy()

        rows = rows.sort_values(
            "direction_agreement_pct",
            ascending=True,
        )

        labels = rows["geography_dimension"].astype(str).tolist()
        values = (
            pd.to_numeric(
                rows["direction_agreement_pct"],
                errors="coerce",
            )
            .astype(float)
            .tolist()
        )
        axis_title = "Groups sharing the system-level direction (%)"
        suffix = "%"
        caption = (
            "Each point is one frozen geography definition: Borough, "
            "Policy geography, or Pre-CP mobility environment."
        )

    elif dimension == "Horizon":
        row = horizon.loc[
            horizon["metric"].eq(metric)
        ].iloc[0]

        labels = [
            "h=1",
            "h=2",
            "h=5",
        ]
        values = [
            float(row["h1_gap_pct"]),
            float(row["h2_gap_pct"]),
            float(row["h5_gap_pct"]),
        ]
        axis_title = "No-CP − observed (%)"
        suffix = "%"
        zero_line = True
        caption = (
            "All horizons share the system-level direction: "
            f"{'Yes' if bool(row['aggregate_horizons_same_direction']) else 'No'} · "
            f"aggregate spread = {float(row['aggregate_horizon_spread_pp']):.3f} pp · "
            f"row-level direction agreement = "
            f"{float(row['row_direction_agreement_pct']):.1f}%."
        )

    elif dimension == "Calibration":
        row = calibration.loc[
            calibration["metric"].eq(metric)
            & calibration["horizon"].eq(horizon_value)
        ].iloc[0]

        labels = [
            "Median absolute gap",
            "90th percentile absolute gap",
        ]
        values = [
            float(row["median_absolute_mae_units"]),
            float(row["p90_absolute_mae_units"]),
        ]
        axis_title = "Pre-CP validation MAE units"
        suffix = "×"
        reference_one = True
        caption = (
            f"{float(row['share_abs_ge_1_mae_pct']):.1f}% of post-CP rows "
            "are at least 1× ordinary Pre-CP MAE from observed; "
            f"{float(row['share_abs_ge_2_mae_pct']):.1f}% are at least 2×."
        )

    else:
        raise ValueError(
            f"Unsupported evidence dimension: {dimension}"
        )

    display = pd.DataFrame(
        {
            "label": labels,
            "value": values,
        }
    )

    figure = go.Figure()

    # A quiet baseline lets every evidence choice retain the same visual form.
    for _, row in display.iterrows():
        start = (
            0.0
            if row["value"] >= 0
            else float(row["value"])
        )
        end = (
            float(row["value"])
            if row["value"] >= 0
            else 0.0
        )

        figure.add_trace(
            go.Scatter(
                x=[start, end],
                y=[row["label"], row["label"]],
                mode="lines",
                line={
                    "color": "rgba(0, 109, 119, 0.18)",
                    "width": 5,
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

    figure.add_trace(
        go.Scatter(
            x=display["value"],
            y=display["label"],
            mode="markers+text",
            marker={
                "size": 14,
                "color": BRAND_COLORS["dark_teal"],
            },
            text=[
                f"{value:+.3f}{suffix}"
                if (
                    suffix == "%"
                    and dimension
                    in {
                        "Population",
                        "Weighting",
                        "Temporal",
                        "Horizon",
                    }
                )
                else f"{value:.3f}{suffix}"
                for value in display["value"]
            ],
            textposition="middle right",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "%{x:.3f}"
                + suffix
                + "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    if zero_line:
        figure.add_vline(
            x=0,
            line_width=1,
            line_dash="dot",
            line_color="rgba(0, 109, 119, 0.45)",
        )

    if reference_one:
        figure.add_vline(
            x=1,
            line_width=1,
            line_dash="dot",
            line_color="rgba(0, 109, 119, 0.45)",
            annotation_text="1× ordinary Pre-CP MAE",
            annotation_position="top right",
        )

    figure.update_xaxes(
        title_text=axis_title,
    )
    figure.update_yaxes(
        title=None,
        autorange="reversed",
    )

    figure = brand_figure(
        figure,
        height=max(
            300,
            120 + 70 * len(display),
        ),
        left=185,
        right=90,
        top=35,
        bottom=60,
    )

    return figure, caption


# ---------------------------------------------------------------------
# Curated saved views
# ---------------------------------------------------------------------

SAVED_VIEWS = [
    {
        "label": (
            "Why Taxi speed h=1 is Sensitive"
        ),
        "metric": "taxi_avg_trip_speed",
        "horizon": 1,
        "dimension": "Temporal",
    },
    {
        "label": (
            "Why Taxi speed h=2 is Mixed"
        ),
        "metric": "taxi_avg_trip_speed",
        "horizon": 2,
        "dimension": "Weighting",
    },
    {
        "label": (
            "Stable demand example · FHVHV trips h=1"
        ),
        "metric": "fhvhv_trip_count",
        "horizon": 1,
        "dimension": "Geography",
    },
    {
        "label": (
            "Stable speed example · FHVHV speed h=1"
        ),
        "metric": "fhvhv_avg_trip_speed",
        "horizon": 1,
        "dimension": "Calibration",
    },
    {
        "label": (
            "Stable does not mean large · Taxi trips h=1"
        ),
        "metric": "taxi_trip_count",
        "horizon": 1,
        "dimension": "Calibration",
    },
]

def weighted_value(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    """Return an activity-weighted mean, or NaN when no positive weight exists."""
    value_array = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )
    weight_array = pd.to_numeric(
        weights,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    valid = (
        np.isfinite(
            value_array
        )
        & np.isfinite(
            weight_array
        )
        & (
            weight_array
            > 0
        )
    )

    if not valid.any():
        return np.nan

    return float(
        np.average(
            value_array[
                valid
            ],
            weights=weight_array[
                valid
            ],
        )
    )


def choose_zone_label(
    group: pd.DataFrame,
) -> str:
    """Prefer the label carried by the canonical source ID when available."""
    canonical_id = int(
        group[
            "canonical_taxi_zone_id"
        ].iloc[0]
    )

    canonical_rows = group.loc[
        group[
            "taxi_zone_id"
        ].eq(
            canonical_id
        )
        & group[
            "zone"
        ].notna()
    ]

    if not canonical_rows.empty:
        return str(
            canonical_rows[
                "zone"
            ].iloc[0]
        )

    names = (
        group[
            "zone"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if names:
        return names[
            0
        ]

    return (
        f"Taxi Zone {canonical_id}"
    )


@st.cache_data(show_spinner=False)
def build_zone_weekly_curves() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Aggregate all temporal buckets to physical Taxi Zone × metric × week.

    Counts/ridership sum. Speeds use the same activity-weighted logic as Raw 21.
    Canonical aliases are combined before aggregation so Corona does not appear
    twice in the reader-facing scout.
    """
    post, pre = load_counterfactual_raw24_zone_weekly_inputs()

    post = post.loc[
        post[
            "period_complete"
        ].fillna(False).astype(bool)
    ].copy()

    post_records: list[
        dict[str, object]
    ] = []

    post_keys = [
        "canonical_taxi_zone_id",
        "metric",
        "week_start",
    ]

    for (
        zone_id,
        metric,
        week_start,
    ), group in post.groupby(
        post_keys,
        observed=True,
        sort=False,
    ):
        label = choose_zone_label(
            group
        )

        borough_values = (
            group[
                "borough"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        borough = (
            borough_values[
                0
            ]
            if borough_values
            else ""
        )

        if metric in COUNT_METRICS:
            observed = pd.to_numeric(
                group[
                    "observed_level"
                ],
                errors="coerce",
            ).sum(
                min_count=1
            )
            no_cp_values = {
                horizon_value: pd.to_numeric(
                    group[
                        f"no_cp_h{horizon_value}"
                    ],
                    errors="coerce",
                ).sum(
                    min_count=1
                )
                for horizon_value in HORIZON_ORDER
            }
        else:
            observed = weighted_value(
                group[
                    "observed_level"
                ],
                group[
                    "observed_weight"
                ],
            )
            no_cp_values = {
                horizon_value: weighted_value(
                    group[
                        f"no_cp_h{horizon_value}"
                    ],
                    group[
                        f"no_cp_weight_h{horizon_value}"
                    ],
                )
                for horizon_value in HORIZON_ORDER
            }

        post_records.append(
            {
                "taxi_zone_id": int(
                    zone_id
                ),
                "zone": label,
                "borough": borough,
                "metric": metric,
                "week_start": pd.Timestamp(
                    week_start
                ),
                "support_rows": int(
                    pd.to_numeric(
                        group[
                            "support_rows"
                        ],
                        errors="coerce",
                    ).fillna(0).sum()
                ),
                "observed_level": float(
                    observed
                ),
                **{
                    f"no_cp_h{horizon_value}": float(
                        no_cp_values[
                            horizon_value
                        ]
                    )
                    for horizon_value in HORIZON_ORDER
                },
            }
        )

    pre_records: list[
        dict[str, object]
    ] = []

    pre_keys = [
        "canonical_taxi_zone_id",
        "metric",
        "week_start",
    ]

    for (
        zone_id,
        metric,
        week_start,
    ), group in pre.groupby(
        pre_keys,
        observed=True,
        sort=False,
    ):
        label = choose_zone_label(
            group
        )

        borough_values = (
            group[
                "borough"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        borough = (
            borough_values[
                0
            ]
            if borough_values
            else ""
        )

        if metric in COUNT_METRICS:
            observed = pd.to_numeric(
                group[
                    "observed_level"
                ],
                errors="coerce",
            ).sum(
                min_count=1
            )
        else:
            observed = weighted_value(
                group[
                    "observed_level"
                ],
                group[
                    "observed_weight"
                ],
            )

        pre_records.append(
            {
                "taxi_zone_id": int(
                    zone_id
                ),
                "zone": label,
                "borough": borough,
                "metric": metric,
                "week_start": pd.Timestamp(
                    week_start
                ),
                "support_rows": int(
                    pd.to_numeric(
                        group[
                            "support_rows"
                        ],
                        errors="coerce",
                    ).fillna(0).sum()
                ),
                "observed_level": float(
                    observed
                ),
            }
        )

    post_weekly = (
        pd.DataFrame(
            post_records
        )
        .sort_values(
            [
                "metric",
                "taxi_zone_id",
                "week_start",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    pre_weekly = (
        pd.DataFrame(
            pre_records
        )
        .sort_values(
            [
                "metric",
                "taxi_zone_id",
                "week_start",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return (
        post_weekly,
        pre_weekly,
    )


def safe_relative_pct(
    numerator: float,
    denominator: float,
) -> float:
    """Return an absolute percent ratio while avoiding unstable near-zero division."""
    if (
        not np.isfinite(
            numerator
        )
        or not np.isfinite(
            denominator
        )
        or abs(
            denominator
        )
        < 1e-12
    ):
        return np.nan

    return float(
        100.0
        * abs(
            numerator
            / denominator
        )
    )


def percentile_high(
    series: pd.Series,
) -> pd.Series:
    """Percentile where a larger raw value means more of the target property."""
    return series.rank(
        pct=True,
        method="average",
    )


def percentile_low(
    series: pd.Series,
) -> pd.Series:
    """Percentile where a smaller raw value means more of the target property."""
    return 1.0 - series.rank(
        pct=True,
        method="average",
    )


@st.cache_data(show_spinner=False)
def build_zone_candidate_scout() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Derive transparent zone-level trajectory diagnostics.

    No reader-facing "robustness score" is created. Two internal rank indices
    only order candidate examples for review:

    Strong-example rank:
      smooth launch + persistent direction + horizon agreement +
      historical-range plausibility + visible post-CP separation.

    Concern rank:
      launch jump + directional instability + horizon disagreement +
      historical-range departure.

    Each component is also retained so the ranking can be audited directly.
    """
    post_weekly, pre_weekly = build_zone_weekly_curves()

    records: list[
        dict[str, object]
    ] = []

    group_keys = [
        "metric",
        "taxi_zone_id",
        "zone",
        "borough",
    ]

    for (
        metric,
        zone_id,
        zone,
        borough,
    ), post_group in post_weekly.groupby(
        group_keys,
        observed=True,
        sort=False,
    ):
        pre_group = pre_weekly.loc[
            pre_weekly[
                "metric"
            ].eq(
                metric
            )
            & pre_weekly[
                "taxi_zone_id"
            ].eq(
                zone_id
            )
        ].sort_values(
            "week_start"
        )

        post_group = post_group.sort_values(
            "week_start"
        )

        if (
            len(
                pre_group
            )
            < 26
            or len(
                post_group
            )
            < 40
        ):
            continue

        pre_values = pd.to_numeric(
            pre_group[
                "observed_level"
            ],
            errors="coerce",
        ).dropna()

        if len(
            pre_values
        ) < 26:
            continue

        post_observed = pd.to_numeric(
            post_group[
                "observed_level"
            ],
            errors="coerce",
        )

        h1 = pd.to_numeric(
            post_group[
                "no_cp_h1"
            ],
            errors="coerce",
        )
        h2 = pd.to_numeric(
            post_group[
                "no_cp_h2"
            ],
            errors="coerce",
        )
        h5 = pd.to_numeric(
            post_group[
                "no_cp_h5"
            ],
            errors="coerce",
        )

        valid = (
            post_observed.notna()
            & h1.notna()
            & h2.notna()
            & h5.notna()
        )

        if valid.sum() < 40:
            continue

        post_observed = post_observed.loc[
            valid
        ]
        h1 = h1.loc[
            valid
        ]
        h2 = h2.loc[
            valid
        ]
        h5 = h5.loc[
            valid
        ]

        gap_h1 = h1 - post_observed
        gap_h2 = h2 - post_observed
        gap_h5 = h5 - post_observed

        median_gap = float(
            gap_h1.median()
        )

        if np.isclose(
            median_gap,
            0.0,
            atol=1e-12,
        ):
            baseline_sign = 0
        else:
            baseline_sign = int(
                np.sign(
                    median_gap
                )
            )

        weekly_sign = np.sign(
            gap_h1.to_numpy(
                dtype=float
            )
        )

        if baseline_sign == 0:
            direction_persistence = np.nan
        else:
            direction_persistence = float(
                100.0
                * np.mean(
                    weekly_sign
                    == baseline_sign
                )
            )

        signs = np.column_stack(
            [
                np.sign(
                    gap_h1.to_numpy(
                        dtype=float
                    )
                ),
                np.sign(
                    gap_h2.to_numpy(
                        dtype=float
                    )
                ),
                np.sign(
                    gap_h5.to_numpy(
                        dtype=float
                    )
                ),
            ]
        )

        all_nonzero = np.all(
            signs
            != 0,
            axis=1,
        )
        same_sign = (
            (
                signs[
                    :,
                    0
                ]
                == signs[
                    :,
                    1
                ]
            )
            & (
                signs[
                    :,
                    0
                ]
                == signs[
                    :,
                    2
                ]
            )
            & all_nonzero
        )

        horizon_agreement = float(
            100.0
            * np.mean(
                same_sign
            )
        )

        last_pre = float(
            pre_values.iloc[
                -1
            ]
        )
        first_h1 = float(
            h1.iloc[
                0
            ]
        )

        launch_jump_pct = safe_relative_pct(
            first_h1
            - last_pre,
            last_pre,
        )

        p05 = float(
            pre_values.quantile(
                0.05
            )
        )
        p95 = float(
            pre_values.quantile(
                0.95
            )
        )

        within_range = (
            h1.ge(
                p05
            )
            & h1.le(
                p95
            )
        )

        pre_range_share = float(
            100.0
            * within_range.mean()
        )

        pre_median_abs = float(
            pre_values.abs().median()
        )
        median_abs_gap = float(
            gap_h1.abs().median()
        )

        separation_pct = safe_relative_pct(
            median_abs_gap,
            pre_median_abs,
        )

        h1_h5_spread_pct = safe_relative_pct(
            float(
                (
                    h1
                    - h5
                ).abs().median()
            ),
            pre_median_abs,
        )

        records.append(
            {
                "metric": metric,
                "taxi_zone_id": int(
                    zone_id
                ),
                "zone": str(
                    zone
                ),
                "borough": str(
                    borough
                ),
                "pre_weeks": int(
                    len(
                        pre_values
                    )
                ),
                "post_weeks": int(
                    valid.sum()
                ),
                "launch_jump_pct": launch_jump_pct,
                "direction_persistence_pct": (
                    direction_persistence
                ),
                "horizon_agreement_pct": (
                    horizon_agreement
                ),
                "pre_range_share_pct": (
                    pre_range_share
                ),
                "median_abs_gap_relative_to_pre_pct": (
                    separation_pct
                ),
                "median_h1_h5_spread_relative_to_pre_pct": (
                    h1_h5_spread_pct
                ),
                "median_h1_gap_native": median_gap,
            }
        )

    scout = pd.DataFrame(
        records
    )

    if scout.empty:
        raise RuntimeError(
            "Taxi Zone candidate scouting produced no eligible rows."
        )

    ranked_parts: list[
        pd.DataFrame
    ] = []

    for metric, part in scout.groupby(
        "metric",
        observed=True,
        sort=False,
    ):
        part = part.copy()

        # Strong examples require both plausibility and a visible difference;
        # concern examples deliberately do not reward gap magnitude.
        part[
            "_smooth_rank"
        ] = percentile_low(
            part[
                "launch_jump_pct"
            ]
        )
        part[
            "_persistence_rank"
        ] = percentile_high(
            part[
                "direction_persistence_pct"
            ]
        )
        part[
            "_horizon_rank"
        ] = percentile_high(
            part[
                "horizon_agreement_pct"
            ]
        )
        part[
            "_plausibility_rank"
        ] = percentile_high(
            part[
                "pre_range_share_pct"
            ]
        )
        part[
            "_separation_rank"
        ] = percentile_high(
            part[
                "median_abs_gap_relative_to_pre_pct"
            ]
        )

        part[
            "_strong_scout_index"
        ] = (
            part[
                [
                    "_smooth_rank",
                    "_persistence_rank",
                    "_horizon_rank",
                    "_plausibility_rank",
                    "_separation_rank",
                ]
            ]
            .mean(
                axis=1,
                skipna=True,
            )
        )

        part[
            "_concern_scout_index"
        ] = (
            (
                1.0
                - part[
                    "_smooth_rank"
                ]
            )
            + (
                1.0
                - part[
                    "_persistence_rank"
                ]
            )
            + (
                1.0
                - part[
                    "_horizon_rank"
                ]
            )
            + (
                1.0
                - part[
                    "_plausibility_rank"
                ]
            )
        ) / 4.0

        ranked_parts.append(
            part
        )

    ranked = pd.concat(
        ranked_parts,
        ignore_index=True,
    )

    qa_rows: list[
        dict[str, object]
    ] = []

    def add_qa(
        check_id: str,
        passed: bool,
        details: str,
    ) -> None:
        qa_rows.append(
            {
                "check_id": check_id,
                "status": (
                    "PASS"
                    if bool(passed)
                    else "FAIL"
                ),
                "details": details,
            }
        )

    add_qa(
        "all_five_metrics_present",
        set(
            ranked[
                "metric"
            ]
        ) == set(
            METRIC_ORDER
        ),
        (
            "metrics="
            + ", ".join(
                sorted(
                    ranked[
                        "metric"
                    ].unique()
                )
            )
        ),
    )

    add_qa(
        "no_unknown_reader_zones",
        not ranked[
            "taxi_zone_id"
        ].isin(
            UNKNOWN_ZONE_IDS
        ).any(),
        "Unknown 264/265 are absent.",
    )

    duplicate_corona = (
        ranked.loc[
            ranked[
                "taxi_zone_id"
            ].eq(
                56
            )
        ]
        .groupby(
            "metric",
            observed=True,
        )
        .size()
        .gt(
            1
        )
        .any()
    )

    add_qa(
        "canonical_corona_once_per_metric",
        not bool(
            duplicate_corona
        ),
        (
            "Source 56/57 are combined before physical "
            "Taxi Zone diagnostics."
        ),
    )

    add_qa(
        "minimum_history_support",
        ranked[
            "pre_weeks"
        ].ge(
            26
        ).all()
        and ranked[
            "post_weeks"
        ].ge(
            40
        ).all(),
        (
            f"min_pre={int(ranked['pre_weeks'].min())}; "
            f"min_post={int(ranked['post_weeks'].min())}"
        ),
    )

    bounded_columns = [
        "direction_persistence_pct",
        "horizon_agreement_pct",
        "pre_range_share_pct",
    ]

    bounded_ok = all(
        ranked[
            column
        ].dropna().between(
            0,
            100,
        ).all()
        for column in bounded_columns
    )

    add_qa(
        "percentage_diagnostics_bounded",
        bounded_ok,
        ", ".join(
            bounded_columns
        ),
    )

    add_qa(
        "scout_indices_bounded",
        ranked[
            "_strong_scout_index"
        ].between(
            0,
            1,
        ).all()
        and ranked[
            "_concern_scout_index"
        ].between(
            0,
            1,
        ).all(),
        "Internal ordering indices stay within [0, 1].",
    )

    zone_qa = pd.DataFrame(
        qa_rows
    )

    failed = zone_qa.loc[
        zone_qa[
            "status"
        ].eq(
            "FAIL"
        )
    ]

    if not failed.empty:
        raise RuntimeError(
            "Taxi Zone scout QA failed:\n"
            + failed.to_string(
                index=False
            )
        )

    return (
        ranked,
        zone_qa,
    )


def candidate_view(
    ranked: pd.DataFrame,
    *,
    metric: str,
    kind: str,
    n: int = 5,
) -> pd.DataFrame:
    """Return top candidate examples with transparent diagnostics."""
    scoped = ranked.loc[
        ranked[
            "metric"
        ].eq(
            metric
        )
    ].copy()

    sort_column = (
        "_strong_scout_index"
        if kind
        == "strong"
        else "_concern_scout_index"
    )

    scoped = scoped.sort_values(
        sort_column,
        ascending=False,
    ).head(
        n
    )

    return scoped[
        [
            "taxi_zone_id",
            "zone",
            "borough",
            "launch_jump_pct",
            "direction_persistence_pct",
            "horizon_agreement_pct",
            "pre_range_share_pct",
            "median_abs_gap_relative_to_pre_pct",
            "median_h1_h5_spread_relative_to_pre_pct",
        ]
    ].reset_index(
        drop=True
    )


def build_zone_trajectory(
    post_weekly: pd.DataFrame,
    pre_weekly: pd.DataFrame,
    *,
    metric: str,
    taxi_zone_id: int,
    zone: str,
) -> go.Figure:
    """Show observed history and all three no-CP paths for one physical Taxi Zone."""
    pre = pre_weekly.loc[
        pre_weekly[
            "metric"
        ].eq(
            metric
        )
        & pre_weekly[
            "taxi_zone_id"
        ].eq(
            taxi_zone_id
        )
    ].sort_values(
        "week_start"
    )

    post = post_weekly.loc[
        post_weekly[
            "metric"
        ].eq(
            metric
        )
        & post_weekly[
            "taxi_zone_id"
        ].eq(
            taxi_zone_id
        )
    ].sort_values(
        "week_start"
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=pre[
                "week_start"
            ],
            y=pre[
                "observed_level"
            ],
            mode="lines",
            name="Observed · pre-CP",
            line={
                "color": (
                    "rgba(40, 60, 70, 0.65)"
                ),
                "width": 2,
            },
            hovertemplate=(
                "<b>%{x|%b %d, %Y}</b><br>"
                "Observed: %{y:.3f}"
                "<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=post[
                "week_start"
            ],
            y=post[
                "observed_level"
            ],
            mode="lines",
            name="Observed · post-CP",
            line={
                "color": BRAND_COLORS[
                    "dark_teal"
                ],
                "width": 3,
            },
            hovertemplate=(
                "<b>%{x|%b %d, %Y}</b><br>"
                "Observed: %{y:.3f}"
                "<extra></extra>"
            ),
        )
    )

    horizon_styles = {
        1: {
            "dash": "solid",
            "width": 2.5,
        },
        2: {
            "dash": "dash",
            "width": 2.0,
        },
        5: {
            "dash": "dot",
            "width": 2.0,
        },
    }

    for horizon_value in HORIZON_ORDER:
        style = horizon_styles[
            horizon_value
        ]

        figure.add_trace(
            go.Scatter(
                x=post[
                    "week_start"
                ],
                y=post[
                    f"no_cp_h{horizon_value}"
                ],
                mode="lines",
                name=(
                    f"Estimated no-CP · h={horizon_value}"
                ),
                line={
                    "color": BRAND_COLORS[
                        "terracotta"
                    ],
                    "dash": style[
                        "dash"
                    ],
                    "width": style[
                        "width"
                    ],
                },
                hovertemplate=(
                    "<b>%{x|%b %d, %Y}</b><br>"
                    f"No-CP h={horizon_value}: "
                    "%{y:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    figure.add_vline(
        x=CP_START,
        line_dash="dash",
        line_width=1,
        line_color=(
            "rgba(0, 109, 119, 0.55)"
        ),
        annotation_text=(
            "Jan 5, 2025"
        ),
        annotation_position="top left",
    )

    figure.update_layout(
        title={
            "text": (
                f"{zone} · "
                f"{metric_label(metric)}"
            ),
            "x": 0.01,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        hovermode="x unified",
    )

    figure.update_xaxes(
        title=None,
    )

    return brand_figure(
        figure,
        height=470,
        left=70,
        right=25,
        top=95,
        bottom=55,
    )


# ---------------------------------------------------------------------
# Opening robustness radar
# ---------------------------------------------------------------------

HERO_PROFILES = [
    ("taxi_avg_trip_speed", 1),
    ("taxi_avg_trip_speed", 2),
    ("fhvhv_avg_trip_speed", 1),
]

RADAR_TRACE_STYLES = [
    {
        "color": BRAND_COLORS["dark_teal"],
        "dash": "solid",
        "symbol": "circle",
    },
    {
        "color": BRAND_COLORS["terracotta"],
        "dash": "dash",
        "symbol": "diamond",
    },
    {
        "color": BRAND_COLORS["seafoam"],
        "dash": "dot",
        "symbol": "square",
    },
    {
        # WHY: pale_peach is a background/accent color and becomes nearly
        # invisible on the radar's light plotting surface. Use a darker,
        # high-contrast teal for the fourth comparison instead.
        "color": "#004F59",
        "dash": "dashdot",
        "symbol": "triangle-up",
    },
]


def robustness_profile(metric: str, horizon_value: int) -> dict[str, float | None]:
    """Return the directional-agreement profile for one Metric × Horizon job."""
    row = selected_matrix_row(metric, horizon_value)

    population_match = population.loc[
        population["metric"].eq(metric)
        & population["horizon"].eq(horizon_value)
    ]
    if len(population_match) != 1:
        raise RuntimeError(
            f"Expected one population row for {metric} · h={horizon_value}."
        )

    profile: dict[str, float | None] = {
        "Population": 100.0 if bool(population_match.iloc[0]["same_direction"]) else 0.0,
        "Weighting": None,
        "Temporal": float(row["monthly_direction_agreement_pct"]),
        "Geography": float(row["geography_mean_direction_agreement_pct"]),
        "Horizon": float(row["row_direction_agreement_pct"]),
    }

    if metric in SPEED_METRICS:
        weighting_match = speed_weighting.loc[
            speed_weighting["metric"].eq(metric)
            & speed_weighting["horizon"].eq(horizon_value)
        ]
        if len(weighting_match) != 1:
            raise RuntimeError(
                f"Expected one speed-weighting row for {metric} · h={horizon_value}."
            )
        profile["Weighting"] = (
            100.0 if bool(weighting_match.iloc[0]["same_direction"]) else 0.0
        )

    return profile


def build_robustness_radar(
    profiles: list[tuple[str, int]],
    *,
    height: int = 575,
) -> tuple[go.Figure, list[str]]:
    """Compare robustness profiles without collapsing dimensions into a score."""
    if not profiles:
        raise ValueError("At least one robustness profile is required.")

    # Weighting only exists for speed measures. If any selected profile lacks
    # that check, remove the spoke for everyone rather than encoding N/A as 0.
    profile_values = {
        key: robustness_profile(*key)
        for key in profiles
    }
    axes = ["Population", "Weighting", "Temporal", "Geography", "Horizon"]
    if any(profile_values[key]["Weighting"] is None for key in profiles):
        axes.remove("Weighting")

    figure = go.Figure()

    # Draw larger profiles first so smaller polygons remain visible. Keep fills
    # extremely faint; line/marker treatment carries the comparison.
    ordered_profiles = sorted(
        profiles,
        key=lambda key: sum(float(profile_values[key][axis]) for axis in axes),
        reverse=True,
    )

    # Collect one hover target per unique spoke/value intersection. This avoids
    # Plotly choosing an arbitrary trace when multiple profiles overlap exactly.
    hover_groups: dict[tuple[str, float], list[str]] = {}

    for trace_index, key in enumerate(ordered_profiles):
        metric, horizon_value = key
        row = selected_matrix_row(metric, horizon_value)
        values = [float(profile_values[key][axis]) for axis in axes]
        label = (
            f"{metric_label(metric)} · h={horizon_value} · "
            f"{STATUS_LABELS[str(row['evidence_status']).lower()]}"
        )
        style = RADAR_TRACE_STYLES[trace_index % len(RADAR_TRACE_STYLES)]

        for axis, value in zip(axes, values):
            hover_groups.setdefault((axis, round(value, 8)), []).append(label)

        figure.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=axes + [axes[0]],
                mode="lines+markers",
                fill="toself",
                fillcolor="rgba(0,0,0,0.018)",
                line={
                    "color": style["color"],
                    "width": 3,
                    "dash": style["dash"],
                },
                marker={
                    "color": style["color"],
                    "size": 9,
                    "symbol": style["symbol"],
                    "line": {"color": "white", "width": 1},
                },
                name=label,
                hoverinfo="skip",
            )
        )

    # Invisible, generous hover targets own the tooltip. If several profiles
    # occupy the same vertex, the tooltip lists every one of them together.
    for (axis, value), labels in hover_groups.items():
        joined_labels = "<br>".join(labels)
        figure.add_trace(
            go.Scatterpolar(
                r=[value],
                theta=[axis],
                mode="markers",
                marker={
                    "size": 24,
                    "color": "rgba(0,0,0,0.001)",
                    "line": {"width": 0},
                },
                showlegend=False,
                customdata=[joined_labels],
                hovertemplate=(
                    f"<b>{axis} · {value:.1f}%</b><br>"
                    "%{customdata}<extra></extra>"
                ),
            )
        )

    figure = brand_figure(
        figure,
        height=height,
        left=65,
        right=65,
        top=35,
        bottom=115,
    )

    # With four spokes, rotate the square 45° so labels sit on diagonals rather
    # than directly north/south/east/west. This also keeps the top clear.
    angular_rotation = 45 if len(axes) == 4 else 90

    figure.update_layout(
        title={"text": ""},
        polar={
            "radialaxis": {
                "range": [0, 100],
                "dtick": 20,
                "ticksuffix": "%",
                "angle": 90,
            },
            "angularaxis": {
                "direction": "counterclockwise",
                "rotation": angular_rotation,
            },
        },
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.12,
            "xanchor": "center",
            "x": 0.5,
        },
        hovermode="closest",
    )
    return figure, axes


# ---------------------------------------------------------------------
# Precomputed Taxi Zone scout access
# ---------------------------------------------------------------------

def candidate_view_precomputed(
    diagnostics: pd.DataFrame,
    *,
    metric: str,
    kind: str,
    n: int = 5,
) -> pd.DataFrame:
    """Return the strongest or most concerning precomputed candidates."""
    scoped = diagnostics.loc[
        diagnostics["metric"].eq(metric)
    ].copy()

    sort_column = (
        "strong_scout_index"
        if kind == "strong"
        else "concern_scout_index"
    )

    return (
        scoped.sort_values(
            sort_column,
            ascending=False,
        )
        .head(n)
        .reset_index(drop=True)
    )


def build_precomputed_zone_trajectory(
    weekly: pd.DataFrame,
    *,
    metric: str,
    taxi_zone_id: int,
    zone: str,
) -> go.Figure:
    """Render the compact precomputed weekly trajectory for one physical zone."""
    frame = weekly.loc[
        weekly["metric"].eq(metric)
        & weekly["taxi_zone_id"].eq(taxi_zone_id)
    ].sort_values("week_start")

    pre = frame.loc[
        frame["period"].eq("pre_cp")
    ].copy()

    post = frame.loc[
        frame["period"].eq("post_cp")
    ].copy()

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=pre["week_start"],
            y=pre["observed_level"],
            mode="lines",
            name="Observed · pre-CP",
            line={
                "color": "rgba(40, 60, 70, 0.65)",
                "width": 2,
            },
            hovertemplate=(
                "<b>%{x|%b %d, %Y}</b><br>"
                "Observed: %{y:.3f}"
                "<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=post["week_start"],
            y=post["observed_level"],
            mode="lines",
            name="Observed · post-CP",
            line={
                "color": BRAND_COLORS["dark_teal"],
                "width": 3,
            },
            hovertemplate=(
                "<b>%{x|%b %d, %Y}</b><br>"
                "Observed: %{y:.3f}"
                "<extra></extra>"
            ),
        )
    )

    styles = {
        1: "solid",
        2: "dash",
        5: "dot",
    }

    for horizon_value in HORIZON_ORDER:
        figure.add_trace(
            go.Scatter(
                x=post["week_start"],
                y=post[f"no_cp_h{horizon_value}"],
                mode="lines",
                name=f"Estimated no-CP · h={horizon_value}",
                line={
                    "color": BRAND_COLORS["terracotta"],
                    "dash": styles[horizon_value],
                    "width": 2.2,
                },
                hovertemplate=(
                    "<b>%{x|%b %d, %Y}</b><br>"
                    f"No-CP h={horizon_value}: "
                    "%{y:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    figure.add_vline(
        x=CP_START,
        line_dash="dash",
        line_width=1,
        line_color="rgba(0, 109, 119, 0.55)",
        annotation_text="Jan 5, 2025",
        annotation_position="top left",
    )

    figure.update_layout(
        title={
            "text": f"{zone} · {metric_label(metric)}",
            "x": 0.01,
        },
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
    )

    return brand_figure(
        figure,
        height=470,
        left=70,
        right=25,
        top=95,
        bottom=55,
    )




# ---------------------------------------------------------------------
# PAGE
# ---------------------------------------------------------------------

st.title(
    "Which counterfactual conclusions hold up?"
)

st.caption(
    "COUNTERFACTUAL ROBUSTNESS"
)

st.write(
    "A counterfactual result is more informative when its direction does not depend on "
    "one reasonable analytical choice. This page tests each forecast-based no-CP "
    "conclusion across population, speed weighting, time, geography, forecast horizon, "
    "and Pre-CP calibration. **Stable, Mixed, and Sensitive** summarize consistency "
    "across those checks; they are not causal-confidence scores."
)

status_counts = (
    matrix[
        "evidence_status"
    ]
    .astype(str)
    .str.lower()
    .value_counts()
)

summary_columns = st.columns(
    3
)

for column, status in zip(
    summary_columns,
    [
        "stable",
        "mixed",
        "sensitive",
    ],
):
    with column:
        st.metric(
            STATUS_LABELS[
                status
            ],
            int(
                status_counts.get(
                    status,
                    0,
                )
            ),
        )

st.markdown(
    "### Three robustness profiles show why the labels differ"
)

st.write(
    "Each spoke asks the same directional question: **how much of the tested evidence "
    "preserves the system-level gap direction?** Farther from the center means more "
    "directional agreement. Population and Weighting are binary specification checks "
    "for these speed examples; Temporal, Geography, and Horizon summarize many "
    "comparisons. The shape is a profile, not an overall robustness score—do not read "
    "its area as probability, confidence, or statistical significance."
)

hero_radar, _ = build_robustness_radar(
    HERO_PROFILES,
    height=575,
)

st.plotly_chart(
    hero_radar,
    use_container_width=True,
    key="raw24_robustness_radar_hero",
)

render_chart_insight(
    "**Taxi average speed at h=1 is the most sensitive of these three examples:** "
    "its observed-versus-no-CP direction changes more depending on **when** and "
    "**where** we look. The h=2 Taxi-speed result holds up across more of those "
    "checks, while **FHVHV average speed at h=1 is the most consistent of the "
    "three**, preserving the same direction across population, weighting, time, "
    "and geography and showing stronger agreement across forecast horizons. "
    "In practical terms, some counterfactual conclusions depend much more on the "
    "analytical lens than others."
)

st.markdown(
    "### Do the main directional checks tell the same story?"
)

st.write(
    "Each row is one mobility measure × forecast horizon. The three colored points "
    "show **Temporal, Geography, and Horizon agreement** on the same 0–100% scale; "
    "farther right means more of that evidence agrees with the system-level gap "
    "direction. The connector only shows the spread among those three checks—it is "
    "not a time path or uncertainty interval. Population, Weighting, and Calibration "
    "use different units and remain separate evidence below."
)

hero_event = st.plotly_chart(
    build_agreement_stripes(),
    use_container_width=True,
    key="raw24_agreement_stripes",
    on_select="rerun",
    selection_mode="points",
)

# A point click occurs before the explorer widgets below are instantiated, so
# it can safely update their Session State defaults.
try:
    selected_points = (
        hero_event.selection.points
    )
except Exception:
    selected_points = []

if selected_points:
    point = selected_points[
        0
    ]
    customdata = point.get(
        "customdata"
    )

    if (
        isinstance(
            customdata,
            (
                list,
                tuple,
            ),
        )
        and len(
            customdata
        )
        >= 2
    ):
        clicked_metric = str(
            customdata[
                0
            ]
        )
        clicked_horizon = int(
            customdata[
                1
            ]
        )

        if clicked_metric in METRIC_ORDER:
            st.session_state[
                "raw24_metric"
            ] = clicked_metric

        if clicked_horizon in HORIZON_ORDER:
            st.session_state[
                "raw24_horizon"
            ] = clicked_horizon

st.caption(
    "Click any colored point to load that mobility measure × forecast horizon result "
    "in the Evidence Explorer. Population and weighting are shown below as separate "
    "specification checks because they are not agreement percentages."
)


# ---------------------------------------------------------------------
# Unified robustness explorer
# ---------------------------------------------------------------------

st.divider()

with exploration_section(
    key="raw24_robustness_exploration_area",
    title="Explore the robustness evidence",
    description=(
        "Choose one mobility measure × forecast horizon result once, compare it with other "
        "robustness profiles, inspect the evidence behind it, and then examine Taxi Zone examples "
        "for that same mobility measure."
    ),
):
    # Saved views and Agreement-Strip clicks both feed this same primary state.
    saved_labels = ["Custom"] + [view["label"] for view in SAVED_VIEWS]
    saved_lookup = {view["label"]: view for view in SAVED_VIEWS}

    def _mark_raw24_custom() -> None:
        """Switch to Custom when a reader manually changes a primary control."""
        st.session_state["raw24_saved_view"] = "Custom"

    saved_label = st.selectbox(
        "Saved view",
        options=saved_labels,
        key="raw24_saved_view",
    )

    if saved_label != "Custom":
        preset = saved_lookup[saved_label]
        st.session_state["raw24_metric"] = preset["metric"]
        st.session_state["raw24_horizon"] = preset["horizon"]
        st.session_state["raw24_dimension"] = preset["dimension"]

    st.session_state.setdefault("raw24_metric", HERO_PROFILES[0][0])
    st.session_state.setdefault("raw24_horizon", HERO_PROFILES[0][1])
    st.session_state.setdefault("raw24_dimension", "Temporal")

    control_1, control_2 = st.columns([1.7, 0.8])

    with control_1:
        selected_metric = st.selectbox(
            "Mobility measure",
            options=METRIC_ORDER,
            format_func=metric_label,
            key="raw24_metric",
            on_change=_mark_raw24_custom,
        )

    with control_2:
        selected_horizon = st.selectbox(
            "Horizon",
            options=HORIZON_ORDER,
            format_func=lambda value: f"h={value}",
            key="raw24_horizon",
            on_change=_mark_raw24_custom,
        )

    available_dimensions = EVIDENCE_DIMENSIONS.copy()
    if selected_metric not in SPEED_METRICS:
        available_dimensions.remove("Weighting")

    current_dimension = st.session_state.get("raw24_dimension", "Temporal")
    if current_dimension not in available_dimensions:
        st.session_state["raw24_dimension"] = "Temporal"

    primary_profile = (selected_metric, selected_horizon)

    st.markdown("#### Compare robustness profiles")
    st.write(
        "The selected mobility measure × forecast horizon result is the primary profile. "
        "Add up to three others for comparison—you do not need to select the primary "
        "result again."
    )

    comparison_options = [
        (metric, horizon_value)
        for metric in METRIC_ORDER
        for horizon_value in HORIZON_ORDER
        if (metric, horizon_value) != primary_profile
    ]

    # Keep prior comparisons when possible, but never allow the primary profile
    # to appear a second time after the main controls change.
    # Keep prior comparisons when possible, but never allow the primary profile
    # to appear a second time after the main controls change.
    existing_comparisons = st.session_state.get(
        "raw24_radar_comparisons",
        [],
    )

    existing_comparisons = [
        tuple(profile)
        for profile in existing_comparisons
        if tuple(profile) in comparison_options
    ][:3]

    st.session_state["raw24_radar_comparisons"] = existing_comparisons

    comparison_selection = st.multiselect(
        "Add comparison profiles",
        options=comparison_options,
        max_selections=3,
        format_func=lambda value: f"{metric_label(value[0])} · h={value[1]}",
        key="raw24_radar_comparisons",
    )

    radar_profiles = [primary_profile] + list(comparison_selection)
    comparison_radar, comparison_axes = build_robustness_radar(
        radar_profiles,
        height=560,
    )

    if "Weighting" not in comparison_axes:
        st.caption(
            "This comparison uses four spokes because at least one selected profile is "
            "a count or ridership measure. Weighting applies only to speed measures, so "
            "it is omitted for every profile rather than treated as zero. The four-spoke "
            "radar is rotated onto the diagonals to keep labels clear of the legend."
        )

    st.plotly_chart(
        comparison_radar,
        use_container_width=True,
        key="raw24_robustness_radar_explorer",
    )

    st.markdown("#### Inspect the selected evidence")

    selected_dimension = st.selectbox(
        "Evidence dimension",
        options=available_dimensions,
        key="raw24_dimension",
        on_change=_mark_raw24_custom,
    )

    selected_row = selected_matrix_row(selected_metric, selected_horizon)
    st.markdown(
        f"#### {metric_label(selected_metric)} · h={selected_horizon} "
        f"{status_badge_html(selected_row['evidence_status'])}",
        unsafe_allow_html=True,
    )
    st.write(selected_row["baseline_result_summary"])

    context_1, context_2, context_3, context_4 = st.columns(4)
    signed_gap = (
        float(selected_row["baseline_abs_gap_pct"])
        if selected_row["baseline_gap_direction"] == "no_cp_above_observed"
        else -float(selected_row["baseline_abs_gap_pct"])
    )

    with context_1:
        st.metric("System gap", f"{signed_gap:+.3f}%")
    with context_2:
        st.metric(
            "Temporal agreement",
            f"{float(selected_row['monthly_direction_agreement_pct']):.1f}%",
        )
    with context_3:
        st.metric(
            "Geography agreement",
            f"{float(selected_row['geography_mean_direction_agreement_pct']):.1f}%",
        )
    with context_4:
        st.metric(
            "Median |gap|",
            f"{float(selected_row['median_absolute_mae_units']):.3f}× MAE",
        )

    evidence_figure, evidence_caption = build_evidence_detail_figure(
        selected_metric,
        selected_horizon,
        selected_dimension,
    )
    st.plotly_chart(
        evidence_figure,
        use_container_width=True,
        key=(
            "raw24_evidence_detail_"
            f"{selected_metric}_{selected_horizon}_{selected_dimension}"
        ),
    )
    st.caption(evidence_caption)

    st.markdown("##### What to keep in mind")
    st.write(selected_row["caution_note"])
    st.caption("Strongest supporting view: " + str(selected_row["supporting_view"]))

    st.markdown(
        "#### Where does the counterfactual look strongest — and where should we be cautious?"
    )
    st.write(
        f"For **{metric_label(selected_metric)}**, the Taxi Zone diagnostic looks for "
        "examples that combine a smoother launch, persistent direction, agreement "
        "across horizons, and a no-CP path that remains reasonably consistent with the "
        "zone's own Pre-CP weekly range. Cautionary examples surface the opposite "
        "patterns. The mobility measure above carries through automatically."
    )

    zone_scout = load_counterfactual_raw24_zone_scout()

    if zone_scout is None:
        st.info(
            "Taxi Zone diagnostic examples are temporarily unavailable."
        )
    else:
        zone_ranked, zone_weekly, zone_qa = zone_scout
        zone_metric = selected_metric

        strong = candidate_view_precomputed(
            zone_ranked,
            metric=zone_metric,
            kind="strong",
            n=5,
        )
        concern = candidate_view_precomputed(
            zone_ranked,
            metric=zone_metric,
            kind="concern",
            n=5,
        )

        left, right = st.columns(2)
        display_columns = {
            "zone": "Taxi Zone",
            "borough": "Borough",
            "launch_jump_pct": "Launch jump %",
            "direction_persistence_pct": "Direction persistence %",
            "horizon_agreement_pct": "Horizon agreement %",
            "pre_range_share_pct": "Within pre-CP range %",
            "median_abs_gap_relative_to_pre_pct": "Median gap / pre-CP level %",
            "median_h1_h5_spread_relative_to_pre_pct": "h1–h5 spread / pre-CP level %",
        }
        display_format = {
            "Launch jump %": "{:.3f}",
            "Direction persistence %": "{:.1f}",
            "Horizon agreement %": "{:.1f}",
            "Within pre-CP range %": "{:.1f}",
            "Median gap / pre-CP level %": "{:.3f}",
            "h1–h5 spread / pre-CP level %": "{:.3f}",
        }

        with left:
            st.markdown("##### More reassuring examples")
            st.dataframe(
                strong[list(display_columns)].rename(columns=display_columns).style.format(
                    display_format
                ),
                hide_index=True,
                use_container_width=True,
            )

        with right:
            st.markdown("##### Cautionary examples")
            st.dataframe(
                concern[list(display_columns)].rename(columns=display_columns).style.format(
                    display_format
                ),
                hide_index=True,
                use_container_width=True,
            )

        candidate_options: list[tuple[str, int, str]] = []
        for label, table in [("More reassuring", strong), ("Cautionary", concern)]:
            for _, row in table.iterrows():
                candidate_options.append(
                    (label, int(row["taxi_zone_id"]), str(row["zone"]))
                )

        selected_candidate = st.selectbox(
            "Verify a candidate",
            options=candidate_options,
            format_func=lambda value: f"{value[0]} · {value[2]}",
            key="raw24_zone_candidate",
        )

        if selected_candidate:
            _, candidate_zone_id, candidate_zone = selected_candidate
            selected_diagnostic = zone_ranked.loc[
                zone_ranked["metric"].eq(zone_metric)
                & zone_ranked["taxi_zone_id"].eq(candidate_zone_id)
            ].iloc[0]

            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.metric(
                    "Launch jump",
                    f"{float(selected_diagnostic['launch_jump_pct']):.3f}%",
                )
            with d2:
                st.metric(
                    "Direction persistence",
                    f"{float(selected_diagnostic['direction_persistence_pct']):.1f}%",
                )
            with d3:
                st.metric(
                    "Horizon agreement",
                    f"{float(selected_diagnostic['horizon_agreement_pct']):.1f}%",
                )
            with d4:
                st.metric(
                    "Within pre-CP range",
                    f"{float(selected_diagnostic['pre_range_share_pct']):.1f}%",
                )

            st.plotly_chart(
                build_precomputed_zone_trajectory(
                    zone_weekly,
                    metric=zone_metric,
                    taxi_zone_id=candidate_zone_id,
                    zone=candidate_zone,
                ),
                use_container_width=True,
                key=(
                    "raw24_zone_trajectory_"
                    f"{zone_metric}_{candidate_zone_id}"
                ),
            )
            st.caption(
                "The trajectory is diagnostic evidence, not proof of the true no-CP "
                "path. These zone-level labels summarize the displayed diagnostic "
                "criteria; they are separate from the system-level Stable, Mixed, and "
                "Sensitive classifications."
            )


# ---------------------------------------------------------------------
# Closing synthesis
# ---------------------------------------------------------------------

st.divider()

st.markdown("### What this page establishes")
st.markdown(
    "The counterfactual conclusions do not all have the same robustness profile. "
    "Some keep the same directional story across time, geography, and forecast horizon; "
    "others depend more strongly on a particular analytical choice or sit closer to the "
    "forecast system's ordinary Pre-CP error. Robustness therefore qualifies how much "
    "weight to place on a counterfactual pattern without converting that evidence into "
    "a causal-confidence score."
)

with st.expander("How this page works", expanded=False):
    st.markdown(
        """
        **1. Test six distinct robustness dimensions.** **Population** changes the
        supported observation universe. **Weighting** tests the speed-aggregation
        choice. **Temporal** checks month-to-month direction. **Geography** checks
        direction across spatial groupings. **Horizon** compares h=1, h=2, and h=5.
        **Calibration** expresses post-launch gaps relative to ordinary Pre-CP
        forecast error.

        **2. Only put genuinely comparable percentages on one axis.** Agreement Stripes
        combine **Temporal, Geography, and Horizon** because all three measure
        directional agreement on a 0–100% scale. Population, Weighting, and Calibration
        remain separate rather than being forced into an artificial common score.

        **3. Treat Stable, Mixed, and Sensitive as evidence summaries.** These labels
        come from the frozen robustness package and describe how consistently a result
        holds across its checks. They do **not** mean statistically significant,
        causally proven, or successful/unsuccessful policy outcomes.

        **4. Read calibration in units of ordinary forecast error.** A gap of one
        Pre-CP MAE unit is comparable in magnitude to the forecasting job's typical
        absolute error before congestion pricing began. That contextualizes effect
        separation without turning forecast error into a confidence interval.

        **5. Use Taxi Zone trajectories as diagnostic examples, not a second conclusion
        registry.** The zone-level views examine launch smoothness, directional
        persistence, horizon agreement, and consistency with the zone's own Pre-CP
        range. They help expose where a local no-CP trajectory looks more or less
        reassuring under those diagnostics.
        """
    )

st.caption(
    "Evidence scope: sensitivity and calibration checks on forecast-based no-CP "
    "comparisons. Robustness indicates whether a result survives reasonable analytical "
    "alternatives; it does not establish causal certainty, statistical significance, "
    "or whether congestion pricing was a policy success or failure."
)
