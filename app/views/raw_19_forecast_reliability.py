"""Raw 19 — Forecast Reliability, Pass 3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.data_access.mobility_environments import load_canonical_cluster_assignments
from app.data_access.spatial_visuals import get_zone_geojson
from app.data_access.forecasting import (
    load_forecast_records,
    load_forecast_temporal_zone_summary,
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
# Page 18 established whether the forecasting system tracks observed mobility.
# Page 19 asks the spatial follow-up: where are those forecasts more dependable,
# where are they less dependable, and does the model still add value in places
# that are intrinsically difficult to predict?
#
# Reliability claims stay on the untouched Jan–Mar 2026 final holdout. The page
# never mixes ordinary-validation history into the reliability ranking.
# ---------------------------------------------------------------------

PAGE_CAPTION = "FORECAST RELIABILITY"
PAGE_TITLE = "Where are forecasts most and least reliable?"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORECAST_DIR = PROJECT_ROOT / "data" / "processed" / "4.7.1.final_tables"

ZONE_SUMMARY_PATH = FORECAST_DIR / "showcase_forecast_zone_summary.parquet"

FINAL_HOLDOUT_START_DATE = pd.Timestamp("2026-01-05")
FINAL_HOLDOUT_END_DATE = pd.Timestamp("2026-03-31")

# Scouting selected this as the clearest reader-facing spatial reliability hero:
# broad support, substantial spatial variation, and no low-activity-tail warning.
HERO_METRIC = "taxi_avg_trip_speed"
HERO_HORIZON = 1

HERO_SCOPE_LABEL = (
    "Taxi average speed · h=1 · final holdout · Jan 5–Mar 31, 2026"
)

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

HORIZONS = (1, 2, 5)
MAP_CENTER = {"lat": 40.7128, "lon": -74.0060}
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
MIN_SUPPORT_RATIO = 0.50

DAY_TYPE_ORDER = ["All days", "Weekdays", "Weekends"]
DAYPART_ORDER = [
    "All dayparts",
    "AM peak",
    "Midday",
    "PM peak",
    "Evening",
    "Overnight",
]

MAP_VIEW_OPTIONS = [
    "Forecast error",
    "Improvement vs Last-week baseline",
    "Bivariate reliability",
]

ERROR_MEASURE_OPTIONS = [
    "Relative MAE (%)",
    "Native-unit MAE",
]

METRIC_NATIVE_UNITS = {
    "taxi_trip_count": "trips",
    "taxi_avg_trip_speed": "mph",
    "fhvhv_trip_count": "trips",
    "fhvhv_avg_trip_speed": "mph",
    "subway_ridership": "riders",
}

GEOGRAPHY_FILTER_TYPES = [
    "All NYC",
    "Borough",
    "Policy geography",
    "Mobility environment",
]

POLICY_GEOGRAPHY_MAP = {
    "cbd": "CBD",
    "adjacent_to_cbd": "Gateway + adjacent",
    "gateway_to_cbd": "Gateway + adjacent",
    "non_cbd": "Outside",
}

ZONE_REQUIRED_COLUMNS = [
    "metric",
    "horizon",
    "taxi_zone_id",
    "zone",
    "borough",
    "reader_facing_zone",
    "forecast_rows",
    "mae",
    "benchmark_mae",
    "relative_mae_pct",
    "benchmark_skill_pct",
]

RECORD_COLUMNS = [
    "metric",
    "horizon",
    "taxi_zone_id",
    "zone",
    "borough",
    "target_date",
    "target_temporal_bucket",
    "actual",
    "champion_prediction",
    "benchmark_prediction",
    "absolute_error",
    "benchmark_absolute_error",
    "reader_facing_zone",
]

ZONE_EVIDENCE_COLUMNS = [
    "metric",
    "horizon",
    "taxi_zone_id",
    "zone",
    "borough",
    "target_date",
    "target_temporal_bucket",
    "actual",
    "champion_prediction",
    "benchmark_prediction",
]

TEMPORAL_ZONE_COLUMNS = [
    "metric",
    "horizon",
    "taxi_zone_id",
    "zone",
    "borough",
    "day_type",
    "daypart",
    "forecast_rows",
    "observed_abs_sum",
    "absolute_error_sum",
    "benchmark_absolute_error_sum",
]

DAYPART_PLOT_HOUR = {
    "Overnight": 2,
    "AM peak": 8,
    "Midday": 12,
    "PM peak": 17,
    "Evening": 21,
}

# Saved views come directly from the Pass-1 scout. They are representative
# reader-facing examples, not literal extrema. Literal pathological extrema are
# kept for QA rather than promoted as stories.
STORY_PRESETS = {
    "Custom view": None,
    "Most reliable with substantial activity": {
        "metric": "fhvhv_avg_trip_speed",
        "horizon": 1,
        "map_view": "Forecast error",
        "day_type": "All days",
        "daypart": "All dayparts",
        "zone": "South Ozone Park",
    },
    "Least reliable with substantial activity": {
        "metric": "taxi_trip_count",
        "horizon": 2,
        "map_view": "Forecast error",
        "day_type": "All days",
        "daypart": "All dayparts",
        "zone": "Red Hook",
    },
    "Difficult, but the model still helps": {
        "metric": "taxi_trip_count",
        "horizon": 2,
        "map_view": "Bivariate reliability",
        "day_type": "All days",
        "daypart": "All dayparts",
        "zone": "Brooklyn Navy Yard",
    },
    "Low error, but the baseline was already competitive": {
        "metric": "taxi_avg_trip_speed",
        "horizon": 5,
        "map_view": "Bivariate reliability",
        "day_type": "All days",
        "daypart": "All dayparts",
        "zone": "LaGuardia Airport",
    },
}


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------


def require_file(path: Path) -> None:
    """Stop with a useful message when an authoritative handoff is missing."""
    if not path.exists():
        st.error(f"Required Forecast Reliability input not found: {path}")
        st.stop()


def require_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    """Protect reader-facing results from silent schema drift."""
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        st.error(f"{label} is missing required columns: {missing}")
        st.stop()


def metric_label(metric: str) -> str:
    """Return the page's reader-facing mobility-measure label."""
    return METRIC_LABELS.get(metric, metric)


def metric_native_unit(metric: str) -> str:
    """Return the native unit used by MAE for one mobility measure."""
    return METRIC_NATIVE_UNITS.get(metric, "units")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide while preserving unsupported rows as missing, never invented zeroes."""
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return numerator.div(denominator.where(denominator.ne(0)))


def _policy_geography(values: pd.Series) -> pd.Series:
    """Collapse canonical CBD tags into three reader-facing geography groups."""
    normalized = (
        values.astype("string")
        .str.strip()
        .str.lower()
        .replace(POLICY_GEOGRAPHY_MAP)
    )
    return normalized.fillna("Unknown")


def _day_type(bucket: pd.Series) -> pd.Series:
    """Translate the canonical temporal bucket into a simple Day type control."""
    values = bucket.astype("string").str.lower()
    return np.where(values.str.startswith("weekend"), "Weekends", "Weekdays")


def _daypart(bucket: pd.Series) -> pd.Series:
    """Translate the canonical temporal bucket into the five app Dayparts."""
    values = bucket.astype("string").str.lower()
    result = pd.Series("Unknown", index=bucket.index, dtype="string")

    mapping = {
        "overnight": "Overnight",
        "am_peak": "AM peak",
        "midday": "Midday",
        "pm_peak": "PM peak",
        "evening": "Evening",
    }
    for token, label in mapping.items():
        result.loc[values.str.contains(token, regex=False, na=False)] = label

    return result


def _format_pct(value: object, *, signed: bool = False) -> str:
    """Format percentages compactly without exposing floating-point noise."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    return f"{number:+.1f}%" if signed else f"{number:.1f}%"


def _format_number(value: object) -> str:
    """Format native-unit values with no more than three decimal places."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    if abs(number) >= 100:
        return f"{number:,.1f}"
    if abs(number) >= 10:
        return f"{number:,.2f}"
    return f"{number:,.3f}"


def _mix_hex(left: str, right: str, weight: float = 0.50) -> str:
    """Blend two canonical brand colors to create a brand-derived bivariate shade."""
    weight = min(max(float(weight), 0.0), 1.0)

    def rgb(color: str) -> tuple[int, int, int]:
        color = color.lstrip("#")
        return tuple(int(color[index:index + 2], 16) for index in (0, 2, 4))

    a = rgb(left)
    b = rgb(right)
    mixed = tuple(round((1 - weight) * x + weight * y) for x, y in zip(a, b))
    return "#{:02X}{:02X}{:02X}".format(*mixed)


def _build_bivariate_palette() -> dict[int, str]:
    """Build all nine bivariate shades only from canonical project colors.

    The four corners carry the semantics:
      * upper-left  = dark teal: low error + stronger model advantage;
      * upper-right = seafoam: higher error + stronger model advantage;
      * lower-left  = pale peach: low error + weaker model advantage;
      * lower-right = terracotta: high error + weaker model advantage.

    Intermediate cells are deterministic blends of those brand tokens. This
    keeps the 3 × 3 idea while avoiding a one-off blue that is outside the
    project's canonical palette.
    """
    top_left = BRAND_COLORS["dark_teal"]
    top_right = BRAND_COLORS["seafoam"]
    bottom_left = BRAND_COLORS["pale_peach"]
    bottom_right = BRAND_COLORS["terracotta"]

    top_middle = _mix_hex(top_left, top_right)
    bottom_middle = _mix_hex(bottom_left, bottom_right)

    return {
        0: bottom_left,
        1: bottom_middle,
        2: bottom_right,
        3: _mix_hex(bottom_left, top_left),
        4: _mix_hex(bottom_middle, top_middle),
        5: _mix_hex(bottom_right, top_right),
        6: top_left,
        7: top_middle,
        8: top_right,
    }


BIVARIATE_COLORS = _build_bivariate_palette()

ERROR_COLORSCALE = [
    [0.00, BRAND_COLORS["dark_teal"]],
    [0.35, BRAND_COLORS["seafoam"]],
    [0.68, BRAND_COLORS["pale_peach"]],
    [1.00, BRAND_COLORS["terracotta"]],
]

SKILL_COLORSCALE = [
    [0.00, BRAND_COLORS["terracotta"]],
    [0.25, BRAND_COLORS["pale_peach"]],
    [0.50, BRAND_COLORS["ice"]],
    [0.75, BRAND_COLORS["seafoam"]],
    [1.00, BRAND_COLORS["dark_teal"]],
]

BIVARIATE_ERROR_LABELS = {
    0: "Lower forecast error",
    1: "Typical forecast error",
    2: "Higher forecast error",
}

BIVARIATE_SKILL_LABELS = {
    0: "Lower baseline improvement",
    1: "Typical baseline improvement",
    2: "Higher baseline improvement",
}


# ---------------------------------------------------------------------
# Load and compact the authoritative final-holdout evidence
# ---------------------------------------------------------------------


@st.cache_data(show_spinner="Loading forecast reliability evidence...")
def load_page_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load final-holdout reliability summaries and compact temporal evidence.

    WHY:
    The main reliability explorer only needs additive statistics at
    Metric × Horizon × Taxi Zone × Day type × Daypart grain. Those statistics
    already exist in the compact temporal-zone handoff, so Raw 19 does not need
    to scan and regroup the full row-level final-holdout runtime on startup.
    """
    require_file(ZONE_SUMMARY_PATH)

    zones = pd.read_parquet(ZONE_SUMMARY_PATH)

    require_columns(
        zones,
        ZONE_REQUIRED_COLUMNS,
        "showcase_forecast_zone_summary",
    )

    temporal_base = load_forecast_temporal_zone_summary(
        columns=TEMPORAL_ZONE_COLUMNS,
        required_columns=TEMPORAL_ZONE_COLUMNS,
        metrics=METRIC_ORDER,
        horizons=HORIZONS,
    )

    zones["horizon"] = pd.to_numeric(
        zones["horizon"],
        errors="coerce",
    ).astype("Int64")

    zones["taxi_zone_id"] = pd.to_numeric(
        zones["taxi_zone_id"],
        errors="coerce",
    ).astype("Int64")

    temporal_base["horizon"] = pd.to_numeric(
        temporal_base["horizon"],
        errors="coerce",
    ).astype("Int64")

    temporal_base["taxi_zone_id"] = pd.to_numeric(
        temporal_base["taxi_zone_id"],
        errors="coerce",
    ).astype("Int64")

    zones = zones.loc[
        zones["reader_facing_zone"].fillna(False)
        & zones["metric"].isin(METRIC_ORDER)
        & zones["horizon"].isin(HORIZONS)
    ].copy()

    temporal_base = temporal_base.loc[
        temporal_base["metric"].isin(METRIC_ORDER)
        & temporal_base["horizon"].isin(HORIZONS)
    ].copy()

    assignments = load_canonical_cluster_assignments().copy()

    require_columns(
        assignments,
        [
            "taxi_zone_id",
            "cbd_spatial_category",
            "pre_post_cp",
            "canonical_cluster_name",
        ],
        "canonical mobility-environment assignments",
    )

    context = assignments.loc[
        assignments["pre_post_cp"].astype(str).eq("post_cp"),
        [
            "taxi_zone_id",
            "cbd_spatial_category",
            "canonical_cluster_name",
        ],
    ].drop_duplicates("taxi_zone_id")

    context["taxi_zone_id"] = pd.to_numeric(
        context["taxi_zone_id"],
        errors="coerce",
    ).astype("Int64")

    context["policy_geography"] = _policy_geography(
        context["cbd_spatial_category"]
    )

    context = context.rename(
        columns={
            "canonical_cluster_name": "mobility_environment",
        }
    )

    return zones, temporal_base, context


@st.cache_data(show_spinner=False)
def load_zone_evidence(
    metric: str,
    horizon: int,
    taxi_zone_id: int,
) -> pd.DataFrame:
    """Load exact held-out rows for the selected Taxi Zone proof chart."""
    frame = load_forecast_records(
        columns=ZONE_EVIDENCE_COLUMNS,
        required_columns=ZONE_EVIDENCE_COLUMNS,
        metrics=metric,
        horizons=int(horizon),
        taxi_zone_ids=int(taxi_zone_id),
        reader_facing_only=True,
        final_holdout_only=True,
    )

    if frame.empty:
        return frame

    frame["target_date"] = pd.to_datetime(
        frame["target_date"],
        errors="coerce",
    )

    frame["day_type"] = _day_type(
        frame["target_temporal_bucket"]
    )

    frame["daypart"] = _daypart(
        frame["target_temporal_bucket"]
    )

    return frame


def selected_reliability(
    temporal_base: pd.DataFrame,
    context: pd.DataFrame,
    *,
    metric: str,
    horizon: int,
    day_type: str,
    daypart: str,
) -> pd.DataFrame:
    """Rebuild exact zone reliability for one interactive temporal slice."""
    selected = temporal_base.loc[
        temporal_base["metric"].eq(metric)
        & temporal_base["horizon"].eq(horizon)
    ].copy()

    if day_type != "All days":
        selected = selected.loc[selected["day_type"].eq(day_type)].copy()
    if daypart != "All dayparts":
        selected = selected.loc[selected["daypart"].eq(daypart)].copy()

    if selected.empty:
        return pd.DataFrame()

    selected = (
        selected.groupby(
            ["metric", "horizon", "taxi_zone_id", "zone", "borough"],
            observed=True,
            as_index=False,
        )
        .agg(
            forecast_rows=("forecast_rows", "sum"),
            observed_abs_sum=("observed_abs_sum", "sum"),
            absolute_error_sum=("absolute_error_sum", "sum"),
            benchmark_absolute_error_sum=("benchmark_absolute_error_sum", "sum"),
        )
    )

    selected["observed_abs_mean"] = _safe_divide(
        selected["observed_abs_sum"], selected["forecast_rows"]
    )
    selected["relative_mae_pct"] = 100 * _safe_divide(
        selected["absolute_error_sum"], selected["observed_abs_sum"]
    )
    selected["benchmark_skill_pct"] = 100 * _safe_divide(
        selected["benchmark_absolute_error_sum"] - selected["absolute_error_sum"],
        selected["benchmark_absolute_error_sum"],
    )
    selected["mae"] = _safe_divide(
        selected["absolute_error_sum"], selected["forecast_rows"]
    )
    selected["benchmark_mae"] = _safe_divide(
        selected["benchmark_absolute_error_sum"], selected["forecast_rows"]
    )

    selected = selected.merge(
        context[
            ["taxi_zone_id", "policy_geography", "mobility_environment"]
        ],
        on="taxi_zone_id",
        how="left",
        validate="many_to_one",
    )
    selected["policy_geography"] = (
        selected["policy_geography"].astype("string").fillna("Unknown")
    )
    selected["mobility_environment"] = (
        selected["mobility_environment"].astype("string").fillna("Unknown")
    )

    median_rows = float(selected["forecast_rows"].median())
    selected["support_ratio"] = (
        selected["forecast_rows"] / median_rows if median_rows > 0 else np.nan
    )
    selected["map_eligible"] = (
        selected["support_ratio"].ge(MIN_SUPPORT_RATIO)
        & selected["benchmark_mae"].gt(0)
        & selected["observed_abs_sum"].gt(0)
        & selected["relative_mae_pct"].notna()
        & selected["benchmark_skill_pct"].notna()
    )

    eligible = selected.loc[selected["map_eligible"]].copy()
    if eligible.empty:
        return selected

    grouped = eligible.groupby(["metric", "horizon"], observed=True)
    eligible["error_percentile"] = grouped["relative_mae_pct"].rank(
        pct=True, method="average"
    )
    eligible["skill_percentile"] = grouped["benchmark_skill_pct"].rank(
        pct=True, method="average"
    )
    eligible["activity_percentile"] = grouped["observed_abs_mean"].rank(
        pct=True, method="average"
    )
    eligible["support_percentile"] = grouped["forecast_rows"].rank(
        pct=True, method="average"
    )

    selected = selected.drop(
        columns=[
            column
            for column in [
                "error_percentile",
                "skill_percentile",
                "activity_percentile",
                "support_percentile",
            ]
            if column in selected.columns
        ]
    )
    selected = selected.merge(
        eligible[
            [
                "taxi_zone_id",
                "error_percentile",
                "skill_percentile",
                "activity_percentile",
                "support_percentile",
            ]
        ],
        on="taxi_zone_id",
        how="left",
        validate="one_to_one",
    )

    return selected


# ---------------------------------------------------------------------
# Bivariate classification
# ---------------------------------------------------------------------


def _bivariate_cuts(frame: pd.DataFrame) -> dict[str, float]:
    """Return within-view tertile cuts for error and baseline improvement."""
    eligible = frame.loc[
        frame["map_eligible"].fillna(False)
        & frame["relative_mae_pct"].notna()
        & frame["benchmark_skill_pct"].notna()
    ].copy()

    if eligible.empty:
        return {
            "error_q1": np.nan,
            "error_q2": np.nan,
            "skill_q1": np.nan,
            "skill_q2": np.nan,
        }

    return {
        "error_q1": float(eligible["relative_mae_pct"].quantile(1 / 3)),
        "error_q2": float(eligible["relative_mae_pct"].quantile(2 / 3)),
        "skill_q1": float(eligible["benchmark_skill_pct"].quantile(1 / 3)),
        "skill_q2": float(eligible["benchmark_skill_pct"].quantile(2 / 3)),
    }


def apply_bivariate_classification(
    frame: pd.DataFrame,
    cuts: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Assign the nine Error × baseline-improvement cells."""
    result = frame.copy()
    cuts = _bivariate_cuts(result) if cuts is None else cuts

    if result.empty or any(not np.isfinite(value) for value in cuts.values()):
        result["error_band"] = pd.Series(dtype="Int64")
        result["skill_band"] = pd.Series(dtype="Int64")
        result["bivariate_code"] = pd.Series(dtype="Int64")
        result["bivariate_label"] = pd.Series(dtype="string")
        return result, cuts

    error = pd.to_numeric(result["relative_mae_pct"], errors="coerce")
    skill = pd.to_numeric(result["benchmark_skill_pct"], errors="coerce")

    result["error_band"] = np.select(
        [error.le(cuts["error_q1"]), error.le(cuts["error_q2"])],
        [0, 1],
        default=2,
    ).astype(int)
    result["skill_band"] = np.select(
        [skill.le(cuts["skill_q1"]), skill.le(cuts["skill_q2"])],
        [0, 1],
        default=2,
    ).astype(int)
    result["bivariate_code"] = (
        result["skill_band"] * 3 + result["error_band"]
    ).astype(int)
    result["bivariate_label"] = result.apply(
        lambda row: (
            f"{BIVARIATE_ERROR_LABELS[int(row['error_band'])]} · "
            f"{BIVARIATE_SKILL_LABELS[int(row['skill_band'])]}"
        ),
        axis=1,
    )

    return result, cuts


def _discrete_bivariate_colorscale() -> list[list[object]]:
    """Return a stepwise Plotly colorscale for bivariate integer codes 0–8."""
    scale: list[list[object]] = []
    for code in range(9):
        lower = max(0.0, (code - 0.5) / 8.0)
        upper = min(1.0, (code + 0.5) / 8.0)
        scale.append([lower, BIVARIATE_COLORS[code]])
        scale.append([upper, BIVARIATE_COLORS[code]])
    return scale


# ---------------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------------


def _add_bivariate_legend(
    figure: go.Figure,
    *,
    x0: float,
    y0: float,
    cell_w: float = 0.064,
    cell_h: float = 0.058,
    y_title_offset: float = 0.095,
) -> None:
    """Draw a readable 3 × 3 matrix legend in paper coordinates."""
    for skill_band in range(3):
        for error_band in range(3):
            code = skill_band * 3 + error_band
            figure.add_shape(
                type="rect",
                xref="paper",
                yref="paper",
                x0=x0 + error_band * cell_w,
                x1=x0 + (error_band + 1) * cell_w,
                y0=y0 + skill_band * cell_h,
                y1=y0 + (skill_band + 1) * cell_h,
                line={"color": "white", "width": 1.5},
                fillcolor=BIVARIATE_COLORS[code],
                layer="above",
            )

    # Leave a deliberate gap above the matrix so the multi-line legend title
    # never crowds the top row of color cells.
    figure.add_annotation(
        x=x0,
        y=y0 + 3 * cell_h + 0.125,
        xref="paper",
        yref="paper",
        text=(
            "<b>Bivariate legend</b><br>"
            "Error × gain vs<br>Last-week baseline"
        ),
        showarrow=False,
        xanchor="left",
        yanchor="top",
        align="left",
        font={"size": 11, "color": BRAND_COLORS["dark_teal"]},
    )

    for skill_band, label in [(2, "Higher"), (1, "Typical"), (0, "Lower")]:
        figure.add_annotation(
            x=x0 - 0.020,
            y=y0 + (skill_band + 0.5) * cell_h,
            xref="paper",
            yref="paper",
            text=label,
            showarrow=False,
            xanchor="right",
            font={"size": 9, "color": BRAND_COLORS["dark_teal"]},
        )

    # Keep the vertical title clearly separated from the row labels without
    # pushing it so far left that it feels detached from the matrix.
    figure.add_annotation(
        x=x0 - y_title_offset,
        y=y0 + 1.5 * cell_h,
        xref="paper",
        yref="paper",
        text="Baseline<br>improvement",
        showarrow=False,
        xanchor="center",
        textangle=-90,
        font={"size": 9, "color": BRAND_COLORS["dark_teal"]},
    )

    for error_band, label in enumerate(["Lower", "Typical", "Higher"]):
        figure.add_annotation(
            x=x0 + (error_band + 0.5) * cell_w,
            y=y0 - 0.046,
            xref="paper",
            yref="paper",
            text=label,
            showarrow=False,
            xanchor="center",
            yanchor="top",
            font={"size": 9, "color": BRAND_COLORS["dark_teal"]},
        )

    figure.add_annotation(
        x=x0 + 1.5 * cell_w,
        y=y0 - 0.102,
        xref="paper",
        yref="paper",
        text="Forecast error",
        showarrow=False,
        xanchor="center",
        yanchor="top",
        font={"size": 9, "color": BRAND_COLORS["dark_teal"]},
    )


def _hover_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Add presentation-ready hover strings without changing analytical values."""
    result = frame.copy()
    result["hover_error"] = result["relative_mae_pct"].map(_format_pct)
    result["hover_skill"] = result["benchmark_skill_pct"].map(
        lambda value: _format_pct(value, signed=True)
    )
    result["hover_mae"] = result["mae"].map(_format_number)
    result["hover_activity"] = result["observed_abs_mean"].map(_format_number)
    return result


def build_error_map(
    city_frame: pd.DataFrame,
    visible_frame: pd.DataFrame,
    *,
    title: str,
    error_measure: str = "Relative MAE (%)",
) -> go.Figure:
    """
    Map either Relative MAE or native-unit MAE.

    Relative MAE supports proportional cross-zone comparison. Native-unit MAE
    answers how large the average miss was in trips, riders, or mph. Hover keeps
    both measures together so tiny denominators or large-volume zones remain
    interpretable.
    """
    city = city_frame.loc[city_frame["map_eligible"]].copy()
    plot = _hover_table(visible_frame.loc[visible_frame["map_eligible"]].copy())

    if plot.empty:
        return go.Figure()

    metric = str(city_frame["metric"].iloc[0])
    use_native = error_measure == "Native-unit MAE"
    value_column = "mae" if use_native else "relative_mae_pct"

    city_values = pd.to_numeric(city[value_column], errors="coerce").dropna()
    if city_values.empty:
        return go.Figure()

    cap = float(city_values.quantile(0.95))
    floor = float(city_values.quantile(0.05))

    if not np.isfinite(cap) or not np.isfinite(floor) or cap <= floor:
        floor = float(city_values.min())
        cap = float(city_values.max())

    plot["map_value"] = (
        pd.to_numeric(plot[value_column], errors="coerce")
        .clip(lower=floor, upper=cap)
    )

    if use_native:
        primary_hover = (
            "Native-unit MAE: %{customdata[5]}<br>"
            "Relative MAE: %{customdata[3]}<br>"
        )
        colorbar = {
            "title": {
                "text": f"Native MAE ({metric_native_unit(metric)})",
            },
            "x": 1.01,
            "len": 0.68,
            "thickness": 13,
            "outlinewidth": 0,
        }
    else:
        primary_hover = (
            "Relative MAE: %{customdata[3]}<br>"
            "Native-unit MAE: %{customdata[5]}<br>"
        )
        colorbar = {
            "title": {"text": "Relative MAE (%)"},
            "ticksuffix": "%",
            "x": 1.01,
            "len": 0.68,
            "thickness": 13,
            "outlinewidth": 0,
        }

    figure = go.Figure(
        go.Choroplethmap(
            geojson=get_zone_geojson(),
            locations=plot["taxi_zone_id"].astype(str),
            featureidkey="properties.taxi_zone_id",
            z=plot["map_value"],
            zmin=floor,
            zmax=cap,
            colorscale=ERROR_COLORSCALE,
            marker={"line": {"width": 0.35, "color": "white"}},
            customdata=plot[
                [
                    "taxi_zone_id",
                    "zone",
                    "borough",
                    "hover_error",
                    "hover_skill",
                    "hover_mae",
                    "forecast_rows",
                    "hover_activity",
                ]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{customdata[1]}</b> · %{customdata[2]}<br><br>"
                + primary_hover
                + "vs Last-week baseline: %{customdata[4]}<br>"
                "Holdout observations: %{customdata[6]:,.0f}<br>"
                "Typical observed magnitude: %{customdata[7]}"
                "<extra></extra>"
            ),
            colorbar=colorbar,
        )
    )

    figure = apply_branding(figure)
    figure.update_layout(
        title={"text": title},
        map={"style": "carto-positron", "center": MAP_CENTER, "zoom": 9.1},
        height=650,
        margin={"l": 0, "r": 125, "t": 58, "b": 5},
        hovermode="closest",
    )
    return figure


def build_skill_map(
    city_frame: pd.DataFrame,
    visible_frame: pd.DataFrame,
    *,
    title: str,
) -> go.Figure:
    """Map improvement versus the Last-week baseline with robust tail clipping."""
    city = city_frame.loc[city_frame["map_eligible"]].copy()
    plot = _hover_table(visible_frame.loc[visible_frame["map_eligible"]].copy())
    if plot.empty:
        return go.Figure()

    low = float(city["benchmark_skill_pct"].quantile(0.05))
    high = float(city["benchmark_skill_pct"].quantile(0.95))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(city["benchmark_skill_pct"].min())
        high = float(city["benchmark_skill_pct"].max())

    plot["map_value"] = plot["benchmark_skill_pct"].clip(lower=low, upper=high)

    trace_kwargs = {
        "geojson": get_zone_geojson(),
        "locations": plot["taxi_zone_id"].astype(str),
        "featureidkey": "properties.taxi_zone_id",
        "z": plot["map_value"],
        "zmin": low,
        "zmax": high,
        "colorscale": SKILL_COLORSCALE,
        "marker": {"line": {"width": 0.35, "color": "white"}},
        "customdata": plot[
            [
                "taxi_zone_id",
                "zone",
                "borough",
                "hover_error",
                "hover_skill",
                "forecast_rows",
                "hover_activity",
            ]
        ].to_numpy(),
        "hovertemplate": (
            "<b>%{customdata[1]}</b> · %{customdata[2]}<br><br>"
            "vs Last-week baseline: %{customdata[4]}<br>"
            "Relative MAE: %{customdata[3]}<br>"
            "Holdout observations: %{customdata[5]:,.0f}<br>"
            "Typical observed magnitude: %{customdata[6]}"
            "<extra></extra>"
        ),
        "colorbar": {
            "title": {"text": "Error reduction vs baseline (%)"},
            "ticksuffix": "%",
            "x": 1.01,
            "len": 0.68,
            "thickness": 13,
            "outlinewidth": 0,
        },
    }
    if low < 0 < high:
        trace_kwargs["zmid"] = 0

    figure = go.Figure(go.Choroplethmap(**trace_kwargs))
    figure = apply_branding(figure)
    figure.update_layout(
        title={"text": title},
        map={"style": "carto-positron", "center": MAP_CENTER, "zoom": 9.1},
        height=650,
        margin={"l": 0, "r": 125, "t": 58, "b": 5},
        hovermode="closest",
    )
    return figure


def build_bivariate_map(
    city_frame: pd.DataFrame,
    visible_frame: pd.DataFrame,
    *,
    title: str,
) -> tuple[go.Figure, pd.DataFrame, dict[str, float]]:
    """Map one joint 3 × 3 Error × baseline-improvement classification."""
    classified_city, cuts = apply_bivariate_classification(city_frame)
    plot = visible_frame.merge(
        classified_city[
            [
                "taxi_zone_id",
                "error_band",
                "skill_band",
                "bivariate_code",
                "bivariate_label",
            ]
        ],
        on="taxi_zone_id",
        how="left",
        validate="one_to_one",
    )
    plot = _hover_table(plot.loc[plot["map_eligible"]].copy())
    if plot.empty:
        return go.Figure(), classified_city, cuts

    figure = go.Figure(
        go.Choroplethmap(
            geojson=get_zone_geojson(),
            locations=plot["taxi_zone_id"].astype(str),
            featureidkey="properties.taxi_zone_id",
            z=plot["bivariate_code"],
            zmin=0,
            zmax=8,
            colorscale=_discrete_bivariate_colorscale(),
            showscale=False,
            marker={"line": {"width": 0.35, "color": "white"}},
            customdata=plot[
                [
                    "taxi_zone_id",
                    "zone",
                    "borough",
                    "bivariate_label",
                    "hover_error",
                    "hover_skill",
                    "forecast_rows",
                ]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{customdata[1]}</b> · %{customdata[2]}<br><br>"
                "%{customdata[3]}<br>"
                "Relative MAE: %{customdata[4]}<br>"
                "vs Last-week baseline: %{customdata[5]}<br>"
                "Holdout observations: %{customdata[6]:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    figure = apply_branding(figure)
    figure.update_layout(
        title={"text": title},
        map={
            "style": "carto-positron",
            "center": MAP_CENTER,
            "zoom": 9.1,
            # Pull the map farther left than Pass 1 so the vertical legend label
            # has a clean gutter rather than touching the outer-borough polygons.
            "domain": {"x": [0.0, 0.64], "y": [0.0, 1.0]},
        },
        height=660,
        margin={"l": 0, "r": 8, "t": 60, "b": 10},
        hovermode="closest",
    )

    _add_bivariate_legend(figure, x0=0.79, y0=0.55)

    figure.add_annotation(
        x=0.79,
        y=0.30,
        xref="paper",
        yref="paper",
        text=(
            "<b>How the colors are defined</b><br>"
            "Supported NYC zones are grouped into thirds on both measures.<br><br>"
            "<b>Forecast error</b><br>"
            f"Lower: &lt;{cuts['error_q1']:.1f}%<br>"
            f"Typical: {cuts['error_q1']:.1f}%–{cuts['error_q2']:.1f}%<br>"
            f"Higher: &gt;{cuts['error_q2']:.1f}%<br><br>"
            "<b>Gain vs Last-week baseline</b><br>"
            f"Lower: &lt;{cuts['skill_q1']:+.1f}%<br>"
            f"Typical: {cuts['skill_q1']:+.1f}%–{cuts['skill_q2']:+.1f}%<br>"
            f"Higher: &gt;{cuts['skill_q2']:+.1f}%"
        ),
        showarrow=False,
        xanchor="left",
        yanchor="top",
        align="left",
        width=250,
        bgcolor="rgba(255,255,255,0.94)",
        borderpad=6,
        font={"size": 9, "color": BRAND_COLORS["dark_teal"]},
    )

    return figure, classified_city, cuts


def build_activity_scatter(
    city_frame: pd.DataFrame,
    visible_frame: pd.DataFrame,
    *,
    title: str,
    highlight_zone_id: int | None = None,
) -> tuple[go.Figure, pd.DataFrame, dict[str, float]]:
    """Relate observed activity to error while preserving the bivariate colors."""
    classified_city, cuts = apply_bivariate_classification(city_frame)
    plot = visible_frame.merge(
        classified_city[
            ["taxi_zone_id", "bivariate_code", "bivariate_label"]
        ],
        on="taxi_zone_id",
        how="left",
        validate="one_to_one",
    )
    plot = plot.loc[
        plot["map_eligible"].fillna(False)
        & plot["observed_abs_mean"].gt(0)
        & plot["relative_mae_pct"].gt(0)
    ].copy()
    if plot.empty:
        return go.Figure(), classified_city, cuts

    rho = plot["relative_mae_pct"].corr(
        np.log1p(plot["observed_abs_mean"]), method="spearman"
    )
    error_ratio = (
        float(plot["relative_mae_pct"].max())
        / max(float(plot["relative_mae_pct"].quantile(0.10)), 1e-9)
    )
    use_log_y = error_ratio > 20

    figure = go.Figure(
        go.Scatter(
            x=plot["observed_abs_mean"],
            y=plot["relative_mae_pct"],
            mode="markers",
            showlegend=False,
            marker={
                "size": np.clip(6 + 7 * plot["support_ratio"], 7, 16),
                "color": plot["bivariate_code"],
                "cmin": 0,
                "cmax": 8,
                "colorscale": _discrete_bivariate_colorscale(),
                "showscale": False,
                "opacity": 0.82,
                "line": {"width": 0.4, "color": "white"},
            },
            customdata=plot[
                [
                    "taxi_zone_id",
                    "zone",
                    "borough",
                    "forecast_rows",
                    "benchmark_skill_pct",
                    "bivariate_label",
                ]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{customdata[1]}</b> · %{customdata[2]}<br>"
                "%{customdata[5]}<br>"
                "Typical observed magnitude: %{x:,.3f}<br>"
                "Relative MAE: %{y:.3f}%<br>"
                "vs Last-week baseline: %{customdata[4]:+.3f}%<br>"
                "Holdout observations: %{customdata[3]:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    # A selected-zone ring links the explorer scatter back to the profile without
    # changing the bivariate color that carries the analytical meaning.
    if highlight_zone_id is not None:
        highlight = plot.loc[plot["taxi_zone_id"].eq(int(highlight_zone_id))]
        if not highlight.empty:
            row = highlight.iloc[0]
            figure.add_trace(
                go.Scatter(
                    x=[row["observed_abs_mean"]],
                    y=[row["relative_mae_pct"]],
                    mode="markers",
                    marker={
                        "size": 21,
                        "symbol": "circle-open",
                        "color": BRAND_COLORS["dark_teal"],
                        "line": {
                            "width": 3,
                            "color": BRAND_COLORS["dark_teal"],
                        },
                    },
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    figure = apply_branding(figure)
    figure.update_layout(
        title={"text": title, "x": 0.0, "xanchor": "left"},
        height=625,
        margin={"l": 84, "r": 80, "t": 80, "b": 88},
        xaxis={
            "title": {
                "text": "Typical observed level (log scale)",
                "standoff": 14,
            },
            "type": "log",
            # Match the choropleth's plot-versus-legend geometry. The smaller
            # external right margin keeps the scatter itself physically broad.
            "domain": [0.0, 0.64],
        },
        yaxis={
            "title": {
                "text": (
                    "Relative MAE (%) · log scale"
                    if use_log_y
                    else "Relative MAE (%)"
                ),
                "standoff": 14,
            },
            "type": "log" if use_log_y else "linear",
            "rangemode": None if use_log_y else "tozero",
        },
        hovermode="closest",
    )

    # Use exactly the same legend geometry as the bivariate choropleth. Keeping
    # one shared visual grammar avoids scatter-specific spacing drift.
    _add_bivariate_legend(
        figure,
        x0=0.79,
        y0=0.55,
    )

    figure.add_annotation(
        x=0.79,
        y=0.18,
        xref="paper",
        yref="paper",
        text=(
            "<b>Same grouping as the map</b><br>"
            "Each point uses the citywide thirds for this selected slice.<br><br>"
            "<b>Forecast error</b><br>"
            f"Lower: &lt;{cuts['error_q1']:.1f}% · "
            f"Typical: {cuts['error_q1']:.1f}%–{cuts['error_q2']:.1f}% · "
            f"Higher: &gt;{cuts['error_q2']:.1f}%<br><br>"
            "<b>Baseline gain</b><br>"
            f"Lower: &lt;{cuts['skill_q1']:+.1f}% · "
            f"Typical: {cuts['skill_q1']:+.1f}%–{cuts['skill_q2']:+.1f}% · "
            f"Higher: &gt;{cuts['skill_q2']:+.1f}%<br><br>"
            f"Activity/error rank relationship: ρ={rho:+.2f}"
        ),
        showarrow=False,
        xanchor="left",
        yanchor="top",
        align="left",
        width=275,
        bgcolor="rgba(255,255,255,0.94)",
        borderpad=6,
        font={"size": 9, "color": BRAND_COLORS["dark_teal"]},
    )

    return figure, classified_city, cuts


def build_zone_evidence_chart(
    frame: pd.DataFrame,
    *,
    metric: str,
    zone_name: str,
) -> go.Figure:
    """Show the held-out observed, forecast, and baseline traces behind one profile."""
    plot = frame.copy()
    if plot.empty:
        return go.Figure()

    plot = plot.loc[
        plot["actual"].notna()
        & plot["champion_prediction"].notna()
        & plot["benchmark_prediction"].notna()
    ].copy()
    if plot.empty:
        return go.Figure()

    plot["plot_hour"] = plot["daypart"].map(DAYPART_PLOT_HOUR).fillna(12)
    plot["plot_time"] = plot["target_date"] + pd.to_timedelta(
        plot["plot_hour"],
        unit="h",
    )
    plot = plot.sort_values(["plot_time", "target_temporal_bucket"])

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=plot["plot_time"],
            y=plot["actual"],
            mode="lines",
            name="Observed",
            line={"color": "#263238", "width": 2.2},
            customdata=plot[["day_type", "daypart"]].to_numpy(),
            hovertemplate=(
                "<b>%{x|%b %d, %Y}</b> · %{customdata[0]} · %{customdata[1]}<br>"
                "Observed: %{y:,.3f}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=plot["plot_time"],
            y=plot["champion_prediction"],
            mode="lines",
            name="Forecast",
            line={"color": BRAND_COLORS["dark_teal"], "width": 2.0},
            customdata=plot[["actual", "day_type", "daypart"]].to_numpy(),
            hovertemplate=(
                "<b>%{x|%b %d, %Y}</b> · %{customdata[1]} · %{customdata[2]}<br>"
                "Forecast: %{y:,.3f}<br>"
                "Observed: %{customdata[0]:,.3f}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=plot["plot_time"],
            y=plot["benchmark_prediction"],
            mode="lines",
            name="Last-week baseline",
            line={"color": "#7FAEB5", "width": 1.4, "dash": "dot"},
            customdata=plot[["actual", "day_type", "daypart"]].to_numpy(),
            hovertemplate=(
                "<b>%{x|%b %d, %Y}</b> · %{customdata[1]} · %{customdata[2]}<br>"
                "Last-week baseline: %{y:,.3f}<br>"
                "Observed: %{customdata[0]:,.3f}<extra></extra>"
            ),
        )
    )

    figure = apply_branding(figure)
    figure.update_layout(
        title={"text": "\u200b"},
        height=360,
        margin={"l": 70, "r": 24, "t": 54, "b": 58},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "x": 0.0,
            "y": 1.02,
            "xanchor": "left",
            "yanchor": "bottom",
            "title": {"text": ""},
        },
        xaxis={
            "title": "",
            "tickformat": "%b %d",
            "showgrid": False,
        },
        yaxis={
            "title": metric_label(metric),
            "rangemode": "tozero" if metric.endswith("trip_count") or metric == "subway_ridership" else "normal",
        },
    )
    return figure


def build_map_for_view(
    city_frame: pd.DataFrame,
    visible_frame: pd.DataFrame,
    *,
    map_view: str,
    title: str,
    error_measure: str = "Relative MAE (%)",
) -> tuple[go.Figure, pd.DataFrame | None, dict[str, float] | None]:
    """Dispatch the selected explorer map while keeping one map interaction model."""
    if map_view == "Forecast error":
        return (
            build_error_map(
                city_frame,
                visible_frame,
                title=title,
                error_measure=error_measure,
            ),
            None,
            None,
        )
    if map_view == "Improvement vs Last-week baseline":
        return build_skill_map(city_frame, visible_frame, title=title), None, None

    figure, classified, cuts = build_bivariate_map(
        city_frame,
        visible_frame,
        title=title,
    )
    return figure, classified, cuts


def _zone_id_from_plotly_event(event: object) -> int | None:
    """Return the selected Taxi Zone from either a map or scatter event."""
    try:
        points = event.selection.points
    except (AttributeError, TypeError):
        return None

    if not points:
        return None

    point = points[0]
    if not isinstance(point, dict):
        return None

    customdata = point.get("customdata")
    if customdata is not None and len(customdata) > 0:
        try:
            return int(customdata[0])
        except (TypeError, ValueError):
            pass

    location = point.get("location")
    if location is not None:
        try:
            return int(location)
        except (TypeError, ValueError):
            pass

    return None


# ---------------------------------------------------------------------
# Narrative helpers
# ---------------------------------------------------------------------


def representative_extremes(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Choose low/high-error examples without promoting tiny or sparse series."""
    eligible = frame.loc[frame["map_eligible"]].copy()
    if eligible.empty:
        return pd.Series(dtype=object), pd.Series(dtype=object)

    activity_cut = float(eligible["observed_abs_mean"].median())
    support_cut = float(eligible["forecast_rows"].median())
    representative = eligible.loc[
        eligible["observed_abs_mean"].ge(activity_cut)
        & eligible["forecast_rows"].ge(support_cut)
    ].copy()

    if representative.empty:
        representative = eligible

    reliable = representative.sort_values(
        ["relative_mae_pct", "benchmark_skill_pct"],
        ascending=[True, False],
    ).iloc[0]
    less_reliable = representative.sort_values(
        ["relative_mae_pct", "benchmark_skill_pct"],
        ascending=[False, True],
    ).iloc[0]
    return reliable, less_reliable


def hero_takeaway(frame: pd.DataFrame) -> str:
    """Generate the curated hero interpretation from the frozen hero job."""
    eligible = frame.loc[frame["map_eligible"]].copy()
    if eligible.empty:
        return "No supported Taxi Zones are available for the hero reliability view."

    reliable, less_reliable = representative_extremes(eligible)
    p10 = float(eligible["relative_mae_pct"].quantile(0.10))
    p90 = float(eligible["relative_mae_pct"].quantile(0.90))
    beat_share = 100 * float(eligible["benchmark_skill_pct"].gt(0).mean())

    boroughs = (
        eligible.groupby("borough", observed=True)["relative_mae_pct"]
        .median()
        .sort_values()
    )
    borough_sentence = ""
    if len(boroughs) >= 2:
        borough_sentence = (
            f" **{boroughs.index[0]}** has the lowest borough median "
            f"({_format_pct(boroughs.iloc[0])}), while **{boroughs.index[-1]}** "
            f"has the highest ({_format_pct(boroughs.iloc[-1])})."
        )

    baseline_sentence = (
        " Every mapped zone beats the Last-week baseline, so the contrast here is "
        "about *how close* forecasts stay to observed mobility—not whether the model "
        "adds value at all."
        if beat_share >= 99.5
        else (
            f" **{beat_share:.0f}%** of mapped zones beat the Last-week baseline."
        )
    )

    return (
        f"For **Taxi average speed at h=1**, the middle 80% of mapped zones span "
        f"**{p10:.1f}% to {p90:.1f}% Relative MAE**. Among well-supported, active "
        f"zones, **{reliable['zone']}** is a representative lower-error location "
        f"({_format_pct(reliable['relative_mae_pct'])}), while "
        f"**{less_reliable['zone']}** is substantially harder "
        f"({_format_pct(less_reliable['relative_mae_pct'])})."
        f"{borough_sentence}{baseline_sentence}"
    )


def scope_takeaway(
    city_frame: pd.DataFrame,
    visible_frame: pd.DataFrame,
    *,
    map_view: str,
    error_measure: str = "Relative MAE (%)",
) -> str:
    """Describe the current explorer selection rather than repeating static copy."""
    city = city_frame.loc[city_frame["map_eligible"]].copy()
    visible = visible_frame.loc[visible_frame["map_eligible"]].copy()
    if visible.empty:
        return "No supported Taxi Zones match the current filters."

    median_error = float(visible["relative_mae_pct"].median())
    city_median = float(city["relative_mae_pct"].median())
    beat_share = 100 * float(visible["benchmark_skill_pct"].gt(0).mean())
    reliable, less_reliable = representative_extremes(visible)

    filter_sentence = ""
    if len(visible) < len(city):
        delta = median_error - city_median
        direction = "higher" if delta > 0 else "lower"
        filter_sentence = (
            f" The filtered area has **{abs(delta):.1f} percentage points {direction}** "
            "median error than the same citywide mobility measure × horizon slice."
        )

    if map_view == "Bivariate reliability":
        classified, _ = apply_bivariate_classification(city)
        visible_classified = visible.merge(
            classified[["taxi_zone_id", "bivariate_code"]],
            on="taxi_zone_id",
            how="left",
            validate="one_to_one",
        )
        most_reassuring = int(visible_classified["bivariate_code"].eq(6).sum())
        most_caution = int(visible_classified["bivariate_code"].eq(2).sum())
        difficult_helpful = int(visible_classified["bivariate_code"].eq(8).sum())

        examples = []
        for code, label in [
            (6, "low error + stronger model gain"),
            (8, "high error + stronger model gain"),
            (2, "high error + weaker model gain"),
        ]:
            candidates = visible_classified.loc[
                visible_classified["bivariate_code"].eq(code)
            ].copy()
            if candidates.empty:
                continue

            if code == 6:
                example = candidates.sort_values(
                    ["relative_mae_pct", "benchmark_skill_pct"],
                    ascending=[True, False],
                ).iloc[0]
            elif code == 8:
                example = candidates.sort_values(
                    ["relative_mae_pct", "benchmark_skill_pct"],
                    ascending=[False, False],
                ).iloc[0]
            else:
                example = candidates.sort_values(
                    ["relative_mae_pct", "benchmark_skill_pct"],
                    ascending=[False, True],
                ).iloc[0]

            examples.append(f"**{example['zone']}** illustrates {label}")

        example_sentence = (
            " " + "; ".join(examples) + "."
            if examples
            else ""
        )

        return (
            f"Among **{len(visible):,} supported zones**, median Relative MAE is "
            f"**{median_error:.1f}%** and **{beat_share:.0f}%** beat the Last-week "
            f"baseline. The bivariate view places **{most_reassuring}** zones in the "
            "low-error / stronger-improvement corner, **"
            f"{most_caution}** in the high-error / weaker-improvement corner, and "
            f"**{difficult_helpful}** in the high-error / stronger-improvement case, "
            "where forecasting is difficult but the model still adds value."
            f"{example_sentence}{filter_sentence}"
        )

    if map_view == "Improvement vs Last-week baseline":
        strongest = visible.sort_values("benchmark_skill_pct", ascending=False).iloc[0]
        weakest = visible.sort_values("benchmark_skill_pct", ascending=True).iloc[0]
        weak_phrase = (
            f"**{abs(float(weakest['benchmark_skill_pct'])):.1f}% more error than** the baseline"
            if float(weakest["benchmark_skill_pct"]) < 0
            else f"only **{float(weakest['benchmark_skill_pct']):.1f}% less error** than the baseline"
        )
        return (
            f"**{beat_share:.0f}%** of the **{len(visible):,} supported zones** beat the "
            f"Last-week baseline. **{strongest['zone']}** shows the strongest improvement "
            f"({_format_pct(strongest['benchmark_skill_pct'], signed=True)}), while "
            f"**{weakest['zone']}** records {weak_phrase}.{filter_sentence}"
        )

    if error_measure == "Native-unit MAE":
        metric = str(visible["metric"].iloc[0])
        unit = metric_native_unit(metric)
        median_native = float(visible["mae"].median())
        city_median_native = float(city["mae"].median())

        low_native = visible.sort_values(
            ["mae", "relative_mae_pct"],
            ascending=[True, True],
        ).iloc[0]
        high_native = visible.sort_values(
            ["mae", "relative_mae_pct"],
            ascending=[False, False],
        ).iloc[0]

        native_filter_sentence = ""
        if len(visible) < len(city):
            delta = median_native - city_median_native
            direction = "higher" if delta > 0 else "lower"
            native_filter_sentence = (
                f" The filtered area has **{_format_number(abs(delta))} {unit} "
                f"{direction} median native MAE** than the same citywide slice."
            )

        return (
            f"Across **{len(visible):,} supported zones**, median native-unit MAE is "
            f"**{_format_number(median_native)} {unit}**. "
            f"**{low_native['zone']}** has the smallest native-unit miss "
            f"({_format_number(low_native['mae'])} {unit}), while "
            f"**{high_native['zone']}** has the largest "
            f"({_format_number(high_native['mae'])} {unit}). "
            "Relative MAE remains visible in hover because native-unit error "
            "naturally grows with the scale of mobility."
            f"{native_filter_sentence}"
        )

    return (
        f"Across **{len(visible):,} supported zones**, median Relative MAE is "
        f"**{median_error:.1f}%**. A representative lower-error location is "
        f"**{reliable['zone']}** ({_format_pct(reliable['relative_mae_pct'])}); "
        f"**{less_reliable['zone']}** is much harder "
        f"({_format_pct(less_reliable['relative_mae_pct'])}). "
        f"**{beat_share:.0f}%** of these zones still beat the Last-week baseline."
        f"{filter_sentence}"
    )


def activity_takeaway(
    city_frame: pd.DataFrame,
    visible_frame: pd.DataFrame,
) -> str:
    """Interpret the current activity/error relationship without implying causation."""
    city = city_frame.loc[
        city_frame["map_eligible"]
        & city_frame["observed_abs_mean"].gt(0)
        & city_frame["relative_mae_pct"].gt(0)
    ].copy()
    visible = visible_frame.loc[
        visible_frame["map_eligible"]
        & visible_frame["observed_abs_mean"].gt(0)
        & visible_frame["relative_mae_pct"].gt(0)
    ].copy()

    if visible.empty:
        return "No supported Taxi Zones are available for this activity comparison."

    rho = visible["relative_mae_pct"].corr(
        np.log1p(visible["observed_abs_mean"]),
        method="spearman",
    )

    if not np.isfinite(rho):
        relationship_sentence = (
            "There is not enough variation in this selection to summarize the "
            "observed-level/error relationship."
        )
    else:
        magnitude = abs(rho)

        if magnitude < 0.10:
            relationship_sentence = (
                "There is **little or no relationship** between typical observed "
                f"magnitude and forecast error in this selection "
                f"(Spearman ρ={rho:+.2f})."
            )
        else:
            strength = (
                "weak"
                if magnitude < 0.30
                else "moderate"
                if magnitude < 0.50
                else "strong"
            )

            direction = (
                "higher"
                if rho > 0
                else "lower"
            )

            relationship_sentence = (
                f"The relationship between typical observed magnitude and forecast "
                f"error is **{strength}** (Spearman ρ={rho:+.2f}); "
                f"higher-magnitude zones tend to have **{direction} Relative MAE** "
                "in this selection."
            )

    activity_q1 = float(visible["observed_abs_mean"].quantile(1 / 3))
    activity_q2 = float(visible["observed_abs_mean"].quantile(2 / 3))

    lower_activity = visible.loc[
        visible["observed_abs_mean"].le(activity_q1)
    ]
    higher_activity = visible.loc[
        visible["observed_abs_mean"].ge(activity_q2)
    ]

    comparison_sentence = ""
    if not lower_activity.empty and not higher_activity.empty:
        lower_error = float(lower_activity["relative_mae_pct"].median())
        higher_error = float(higher_activity["relative_mae_pct"].median())
        comparison_sentence = (
            f" The lowest-activity third has **{lower_error:.1f}% median "
            f"Relative MAE**, versus **{higher_error:.1f}%** in the "
            "highest observed-level third."
        )

    # A high-activity, high-error exception is useful because it prevents the
    # reader from reducing reliability to a simple sparse-data story.
    exception_sentence = ""
    if not higher_activity.empty:
        exception = higher_activity.sort_values(
            "relative_mae_pct",
            ascending=False,
        ).iloc[0]
        visible_p75 = float(visible["relative_mae_pct"].quantile(0.75))

        if float(exception["relative_mae_pct"]) >= visible_p75:
            exception_sentence = (
                f" **{exception['zone']}** is a useful exception: it sits in "
                f"the higher observed-level third but still records "
                f"**{float(exception['relative_mae_pct']):.1f}% Relative MAE**."
            )

    filter_sentence = ""
    if len(visible) < len(city):
        filter_sentence = (
            f" This comparison uses the **{len(visible):,} supported zones** "
            "remaining after the current geography filter."
        )

    return (
        relationship_sentence
        + comparison_sentence
        + exception_sentence
        + filter_sentence
        + " The relationship is descriptive, not evidence that activity level "
        "causes forecast error."
    )


def zone_profile_copy(
    selected_row: pd.Series,
    city_frame: pd.DataFrame,
    classified_city: pd.DataFrame,
) -> str:
    """Explain one selected zone in citywide context and flag fragile extremes."""
    city = city_frame.loc[city_frame["map_eligible"]].copy()
    city = city.sort_values("relative_mae_pct", ascending=True).reset_index(drop=True)

    zone_id = int(selected_row["taxi_zone_id"])
    rank_lookup = {
        int(value): index + 1
        for index, value in enumerate(city["taxi_zone_id"].tolist())
    }
    rank = rank_lookup.get(zone_id)
    total = len(city)

    category = classified_city.loc[
        classified_city["taxi_zone_id"].eq(zone_id),
        "bivariate_label",
    ]
    category_text = str(category.iloc[0]) if not category.empty else "Unclassified"

    skill = float(selected_row["benchmark_skill_pct"])
    if skill >= 0:
        skill_sentence = (
            f"The forecast reduces error by **{skill:.1f}%** versus the "
            "Last-week baseline."
        )
    else:
        skill_sentence = (
            f"The forecast has **{abs(skill):.1f}% more error** than the "
            "Last-week baseline."
        )

    error_pct = float(selected_row.get("error_percentile", np.nan))
    activity_pct = float(selected_row.get("activity_percentile", np.nan))
    support_pct = float(selected_row.get("support_percentile", np.nan))

    reliability_sentence = ""
    if np.isfinite(error_pct):
        if error_pct <= 0.25:
            reliability_sentence = (
                " It sits in the **lower-error quarter** of supported NYC zones."
            )
        elif error_pct >= 0.75:
            reliability_sentence = (
                " It sits in the **higher-error quarter** of supported NYC zones."
            )

    context_parts: list[str] = []
    if np.isfinite(activity_pct):
        if activity_pct >= 0.75:
            context_parts.append("observed magnitude is relatively high")
        elif activity_pct <= 0.25:
            context_parts.append("observed magnitude is relatively low")
    if np.isfinite(support_pct):
        if support_pct >= 0.75:
            context_parts.append("evaluation support is dense")
        elif support_pct <= 0.25:
            context_parts.append("evaluation support is lighter than most zones")

    context_sentence = (
        " In this slice, " + " and ".join(context_parts) + "."
        if context_parts
        else ""
    )

    rank_sentence = (
        f"Its error ranks **{rank} of {total}** supported NYC zones "
        "(lower is better). "
        if rank is not None
        else ""
    )

    caution_sentence = ""
    relative_mae = float(selected_row["relative_mae_pct"])
    if (
        np.isfinite(activity_pct)
        and activity_pct <= 0.10
        and relative_mae >= 100
    ):
        caution_sentence = (
            " Because observed magnitude is extremely small, this percentage "
            "error can become very large; the native-unit MAE and held-out trace "
            "below are the better way to judge the practical miss."
        )
    elif np.isfinite(support_pct) and support_pct <= 0.10:
        caution_sentence = (
            " This zone has unusually light evaluation support, so its rank "
            "deserves more caution than a densely supported zone."
        )

    return (
        f"**{selected_row['zone']}** has **{relative_mae:.1f}% Relative MAE**. "
        f"{rank_sentence}{skill_sentence}{reliability_sentence} Its bivariate "
        f"position is **{category_text.lower()}**.{context_sentence}"
        f"{caution_sentence}"
    )


# ---------------------------------------------------------------------
# Neighborhood ranking / comparison helpers
# ---------------------------------------------------------------------


def ranking_order_options(
    map_view: str,
) -> list[str]:
    """Return ordering choices that match the metric currently shown on the map."""
    if map_view == "Improvement vs Last-week baseline":
        return [
            "Largest baseline gain first",
            "Smallest baseline gain first",
        ]

    if map_view == "Forecast error":
        return [
            "Lower error first",
            "Higher error first",
        ]

    return [
        "Lower error groups first",
        "Higher error groups first",
    ]


def build_neighborhood_ranking(
    visible_frame: pd.DataFrame,
    *,
    map_view: str,
    error_measure: str,
    order: str,
) -> tuple[pd.DataFrame, str]:
    """
    Build a reader-facing neighborhood comparison that follows the active map.

    Forecast-error and baseline-improvement views have transparent numeric ranks.
    The bivariate view stays grouped rather than inventing a composite score.
    """
    frame = visible_frame.loc[visible_frame["map_eligible"]].copy()

    if frame.empty:
        return pd.DataFrame(), "No supported Taxi Zones match the current filters."

    frame["Taxi Zone"] = (
        frame["zone"].astype(str)
        + " · "
        + frame["borough"].astype(str)
    )
    frame["Relative MAE"] = frame["relative_mae_pct"].map(_format_pct)
    frame["Native MAE"] = frame["mae"].map(_format_number)
    frame["vs Last-week baseline"] = frame["benchmark_skill_pct"].map(
        lambda value: _format_pct(value, signed=True)
    )
    frame["Support"] = frame["forecast_rows"].map(lambda value: f"{int(value):,}")

    if map_view == "Forecast error":
        value_column = (
            "mae"
            if error_measure == "Native-unit MAE"
            else "relative_mae_pct"
        )
        frame["Rank"] = (
            frame[value_column]
            .rank(method="min", ascending=True)
            .astype(int)
        )

        ascending = order == "Lower error first"
        frame = frame.sort_values(
            [value_column, "zone"],
            ascending=[ascending, True],
        )

        metric = str(frame["metric"].iloc[0])
        unit = metric_native_unit(metric)
        explanation = (
            f"Rank follows **{error_measure}**, matching the Forecast error map. "
            "Rank 1 always means the lowest error; the order control only changes "
            "which end of the list appears first. Both Relative MAE and native "
            f"MAE ({unit}) stay visible so proportional and practical error can "
            "be judged together."
        )

        columns = [
            "Rank",
            "Taxi Zone",
            "Relative MAE",
            "Native MAE",
            "vs Last-week baseline",
            "Support",
        ]
        return frame[columns].reset_index(drop=True), explanation

    if map_view == "Improvement vs Last-week baseline":
        frame["Rank"] = (
            frame["benchmark_skill_pct"]
            .rank(method="min", ascending=False)
            .astype(int)
        )

        ascending = order == "Smallest baseline gain first"
        frame = frame.sort_values(
            ["benchmark_skill_pct", "zone"],
            ascending=[ascending, True],
        )

        explanation = (
            "Rank follows **improvement versus the Last-week baseline**, matching "
            "the map. Rank 1 means the largest reduction in forecast error versus "
            "that baseline; negative values mean the advanced forecast did worse."
        )

        columns = [
            "Rank",
            "Taxi Zone",
            "vs Last-week baseline",
            "Relative MAE",
            "Native MAE",
            "Support",
        ]
        return frame[columns].reset_index(drop=True), explanation

    classified, _ = apply_bivariate_classification(frame)
    frame = frame.merge(
        classified[
            [
                "taxi_zone_id",
                "error_band",
                "skill_band",
                "bivariate_label",
            ]
        ],
        on="taxi_zone_id",
        how="left",
        validate="one_to_one",
    )

    # The bivariate view has two dimensions. Sort transparently by error band,
    # then baseline-gain band, rather than pretending it has one numeric score.
    if order == "Higher error groups first":
        frame = frame.sort_values(
            ["error_band", "skill_band", "relative_mae_pct"],
            ascending=[False, True, False],
        )
    else:
        frame = frame.sort_values(
            ["error_band", "skill_band", "relative_mae_pct"],
            ascending=[True, False, True],
        )

    frame["Reliability group"] = frame["bivariate_label"]

    explanation = (
        "The bivariate map has **no single numeric reliability rank** because it "
        "combines forecast error with baseline gain. This table therefore uses "
        "the same nine groups and orders them by forecast-error band, then "
        "baseline-gain band, without inventing a composite score."
    )

    columns = [
        "Taxi Zone",
        "Reliability group",
        "Relative MAE",
        "Native MAE",
        "vs Last-week baseline",
        "Support",
    ]
    return frame[columns].reset_index(drop=True), explanation


# ---------------------------------------------------------------------
# Selected-zone horizon profile
# ---------------------------------------------------------------------


def zone_horizon_profile(
    temporal_base: pd.DataFrame,
    *,
    metric: str,
    taxi_zone_id: int,
    day_type: str,
    daypart: str,
) -> pd.DataFrame:
    """Rebuild one selected zone's held-out reliability across h=1, h=2, and h=5."""
    selected = temporal_base.loc[
        temporal_base["metric"].eq(metric)
        & temporal_base["taxi_zone_id"].eq(int(taxi_zone_id))
    ].copy()

    if day_type != "All days":
        selected = selected.loc[selected["day_type"].eq(day_type)].copy()
    if daypart != "All dayparts":
        selected = selected.loc[selected["daypart"].eq(daypart)].copy()

    if selected.empty:
        return pd.DataFrame()

    profile = (
        selected.groupby("horizon", observed=True, as_index=False)
        .agg(
            forecast_rows=("forecast_rows", "sum"),
            observed_abs_sum=("observed_abs_sum", "sum"),
            absolute_error_sum=("absolute_error_sum", "sum"),
            benchmark_absolute_error_sum=("benchmark_absolute_error_sum", "sum"),
        )
    )

    profile["relative_mae_pct"] = 100 * _safe_divide(
        profile["absolute_error_sum"],
        profile["observed_abs_sum"],
    )
    profile["benchmark_skill_pct"] = 100 * _safe_divide(
        profile["benchmark_absolute_error_sum"] - profile["absolute_error_sum"],
        profile["benchmark_absolute_error_sum"],
    )
    profile["mae"] = _safe_divide(
        profile["absolute_error_sum"],
        profile["forecast_rows"],
    )

    return profile.loc[
        profile["horizon"].isin(HORIZONS)
    ].sort_values("horizon").reset_index(drop=True)


def build_zone_horizon_profile_chart(
    profile: pd.DataFrame,
    *,
    zone_name: str,
) -> go.Figure:
    """Show forecast error and baseline gain across the three horizons separately."""
    figure = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.16,
        subplot_titles=(
            "Forecast error",
            "Improvement vs Last-week baseline",
        ),
    )

    labels = profile["horizon"].map(lambda value: f"h={int(value)}")

    figure.add_trace(
        go.Scatter(
            x=labels,
            y=profile["relative_mae_pct"],
            mode="lines+markers",
            line={"color": BRAND_COLORS["dark_teal"], "width": 2.5},
            marker={"size": 9},
            customdata=profile[["forecast_rows", "mae"]].to_numpy(),
            hovertemplate=(
                "<b>%{x}</b><br>Relative MAE: %{y:.2f}%<br>"
                "Native MAE: %{customdata[1]:.3f}<br>"
                "Holdout observations: %{customdata[0]:,.0f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=labels,
            y=profile["benchmark_skill_pct"],
            mode="lines+markers",
            line={"color": BRAND_COLORS["terracotta"], "width": 2.5},
            marker={"size": 9},
            customdata=profile[["forecast_rows"]].to_numpy(),
            hovertemplate=(
                "<b>%{x}</b><br>Error reduction vs Last-week baseline: "
                "%{y:+.2f}%<br>Holdout observations: %{customdata[0]:,.0f}"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    figure.add_hline(
        y=0,
        line={"color": "#607D84", "width": 1.0, "dash": "dash"},
        row=1,
        col=2,
    )

    figure = apply_branding(figure)
    figure.update_layout(
        title={"text": "​"},
        height=390,
        margin={"l": 70, "r": 30, "t": 65, "b": 55},
        showlegend=False,
    )
    figure.update_yaxes(
        title_text="Relative MAE (%)",
        ticksuffix="%",
        rangemode="tozero",
        row=1,
        col=1,
    )
    figure.update_yaxes(
        title_text="Error reduction (%)",
        ticksuffix="%",
        row=1,
        col=2,
    )
    figure.update_xaxes(title_text="Forecast horizon", row=1, col=1)
    figure.update_xaxes(title_text="Forecast horizon", row=1, col=2)

    return figure


def zone_horizon_profile_takeaway(profile: pd.DataFrame, zone_name: str) -> str:
    """Describe horizon sensitivity without assuming error worsens monotonically."""
    valid = profile.loc[profile["relative_mae_pct"].notna()].copy()
    if valid.empty:
        return f"No supported horizon comparison is available for **{zone_name}**."

    best = valid.loc[valid["relative_mae_pct"].idxmin()]
    worst = valid.loc[valid["relative_mae_pct"].idxmax()]
    spread = float(worst["relative_mae_pct"] - best["relative_mae_pct"])

    horizon_sentence = (
        f"For **{zone_name}**, **h={int(best['horizon'])}** has the lowest error "
        f"at **{float(best['relative_mae_pct']):.1f}% Relative MAE**, while "
        f"**h={int(worst['horizon'])}** is highest at "
        f"**{float(worst['relative_mae_pct']):.1f}%** "
        f"({spread:.1f} percentage points apart)."
    )

    supported_skill = valid.loc[valid["benchmark_skill_pct"].notna()].copy()
    if supported_skill.empty:
        return horizon_sentence

    wins = int(supported_skill["benchmark_skill_pct"].gt(0).sum())
    total = len(supported_skill)
    skill_sentence = (
        f" The selected forecast beats the Last-week baseline at "
        f"**{wins} of {total} supported horizons**."
    )

    return horizon_sentence + skill_sentence


# ---------------------------------------------------------------------
# Saved-view state helpers
# ---------------------------------------------------------------------


def _set_default_state() -> None:
    """Initialize the explorer once without overwriting the reader's selections."""
    defaults = {
        "raw19_story": "Custom view",
        "raw19_metric": HERO_METRIC,
        "raw19_horizon": HERO_HORIZON,
        "raw19_map_view": "Bivariate reliability",
        "raw19_error_measure": "Relative MAE (%)",
        "raw19_ranking_order": "Lower error groups first",
        "raw19_day_type": "All days",
        "raw19_daypart": "All dayparts",
        "raw19_geo_filter_type": "All NYC",
        "raw19_geo_filter_value": "All NYC",
        "raw19_zone_selector": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _mark_story_custom() -> None:
    """Keep the saved-story label honest after the reader changes the view."""
    st.session_state["raw19_story"] = "Custom view"


def _apply_story_preset() -> None:
    """Move all relevant explorer controls when a curated story is chosen."""
    story = st.session_state.get("raw19_story", "Custom view")
    preset = STORY_PRESETS.get(story)
    if not preset:
        return

    st.session_state["raw19_metric"] = preset["metric"]
    st.session_state["raw19_horizon"] = preset["horizon"]
    st.session_state["raw19_map_view"] = preset["map_view"]
    st.session_state["raw19_error_measure"] = "Relative MAE (%)"
    st.session_state["raw19_day_type"] = preset["day_type"]
    st.session_state["raw19_daypart"] = preset["daypart"]
    st.session_state["raw19_geo_filter_type"] = "All NYC"
    st.session_state["raw19_geo_filter_value"] = "All NYC"
    st.session_state["raw19_zone_name_request"] = preset["zone"]


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------

inject_app_css()

# Streamlit's default metric value is intentionally large, but these cards are
# compact orientation aids rather than billboard KPIs. Keep every metric on this
# page readable without allowing long geography labels to dominate the layout.
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        line-height: 1.15 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.86rem !important;
        line-height: 1.2 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_set_default_state()

require_file(ZONE_SUMMARY_PATH)

st.caption(PAGE_CAPTION)
st.title(PAGE_TITLE)
st.markdown(
    """
A citywide forecast can perform well overall and still be much easier to trust in
some neighborhoods than others. This page maps **held-out forecast reliability**
across NYC, then separates two ideas that are easy to confuse: **how large the
forecast errors were** and **how much the model improved on simply repeating last
week**.
"""
)

_, temporal_base, geography_context = load_page_data()

# ------------------------------------------------------------------
# Hero — one simple spatial answer
# ------------------------------------------------------------------

hero_frame = selected_reliability(
    temporal_base,
    geography_context,
    metric=HERO_METRIC,
    horizon=HERO_HORIZON,
    day_type="All days",
    daypart="All dayparts",
)
hero_eligible = hero_frame.loc[hero_frame["map_eligible"]].copy()

hero_median = float(hero_eligible["relative_mae_pct"].median())
hero_p10 = float(hero_eligible["relative_mae_pct"].quantile(0.10))
hero_p90 = float(hero_eligible["relative_mae_pct"].quantile(0.90))
hero_beat = 100 * float(hero_eligible["benchmark_skill_pct"].gt(0).mean())

hero_boroughs = (
    hero_eligible.groupby("borough", observed=True)["relative_mae_pct"]
    .median()
    .sort_values()
)
hero_borough_gap = (
    f"{hero_boroughs.index[0]} → {hero_boroughs.index[-1]}"
    if len(hero_boroughs) >= 2
    else "—"
)

st.subheader("Forecast reliability changes markedly from neighborhood to neighborhood")
st.markdown(f"**Hero focus:** {HERO_SCOPE_LABEL}")
st.caption(
    "All three hero views below use this same fixed mobility measure, forecast "
    "horizon, and final-holdout window. Taxi average speed at h=1 has broad zone "
    "coverage and enough spatial variation to make neighborhood differences easy "
    "to see without centering the story on very low-activity zones."
)

st.caption(
    "Relative MAE = total absolute forecast error ÷ total observed mobility × 100. "
    "Lower is better. Improvement versus the Last-week baseline is a separate "
    "question: positive values mean the selected forecast reduced error."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Median Relative MAE", f"{hero_median:.1f}%")
m2.metric("Middle 80% of zones", f"{hero_p10:.1f}%–{hero_p90:.1f}%")
m3.metric("Zones beating baseline", f"{hero_beat:.0f}%")
m4.metric("Lowest → highest borough median", hero_borough_gap)

st.plotly_chart(
    build_error_map(
        hero_eligible,
        hero_eligible,
        title="Taxi average speed · h=1 — Relative forecast error across NYC",
    ),
    width="stretch",
    config=PLOT_CONFIG,
    key="raw19_hero_error_map",
)
render_chart_insight(hero_takeaway(hero_eligible))

# ------------------------------------------------------------------
# Bivariate explanation — one map, two dimensions
# ------------------------------------------------------------------

st.divider()
st.subheader("Error alone does not tell the whole reliability story")
st.markdown(
    """
A neighborhood can be **hard to predict** and still benefit substantially from the
forecasting system. The bivariate map therefore combines **forecast error** with
**improvement over the Last-week baseline**. The companion scatter keeps those
same nine colors and asks a different question: **does reliability change with the
typical observed speed in a Taxi Zone?**

These are two additional lenses on the **same hero focus** above—not different
metrics or forecast horizons.
"""
)
st.caption(f"Same hero focus · {HERO_SCOPE_LABEL}")

biv_tab, scatter_tab = st.tabs(
    ["Bivariate reliability map", "Reliability vs observed speed"]
)

with biv_tab:
    hero_biv_fig, hero_classified, hero_cuts = build_bivariate_map(
        hero_eligible,
        hero_eligible,
        title="Taxi average speed · h=1 — Error × Last-week-baseline gain",
    )
    st.plotly_chart(
        hero_biv_fig,
        width="stretch",
        config=PLOT_CONFIG,
        key="raw19_hero_bivariate",
    )

    render_chart_insight(
        scope_takeaway(
            hero_eligible,
            hero_eligible,
            map_view="Bivariate reliability",
        )
    )

with scatter_tab:
    hero_scatter, _, _ = build_activity_scatter(
        hero_eligible,
        hero_eligible,
        title="Taxi average speed · h=1 — Reliability vs typical observed speed",
    )
    st.plotly_chart(
        hero_scatter,
        width="stretch",
        config=PLOT_CONFIG,
        key="raw19_hero_activity_scatter",
    )

    with st.expander("How to read this view"):
        st.markdown(
            """
- **Each dot is one supported Taxi Zone.**
- **Left → right:** the zone's typical observed **Taxi average speed** during
  the final holdout.
- The x-axis uses **logarithmic spacing** because zones can occupy different
  speed ranges. Moving the same visual distance to the right represents a
  multiplicative increase rather than adding one fixed number of mph.
- **Bottom → top:** Relative MAE. Lower means the forecasts stayed closer to
  observed mobility.
- **Color:** the same 3 × 3 Error × Last-week-baseline-gain grouping used on the
  bivariate map.
- **Dot size:** evaluation support. Larger dots have more final-holdout
  observations behind their reliability estimate.

The scatter is most useful for checking whether high forecast error is mostly
concentrated in low-level or lightly supported zones—and for spotting important
exceptions.
"""
        )

    render_chart_insight(
        activity_takeaway(
            hero_eligible,
            hero_eligible,
        )
    )

# ------------------------------------------------------------------
# Explorer — investigate another mobility measure × horizon or geography
# ------------------------------------------------------------------

with exploration_section(
    key="raw19_exploration_area",
    title="Explore reliability across NYC",
    description=(
        "Change the mobility measure, forecast horizon, time slice, or geography. "
        "The linked map, observed-level view, neighborhood ranking, and reliability "
        "profile update together as you investigate another part of NYC."
    ),
):
    st.caption(
        "Bivariate cut points remain citywide for the selected mobility measure × horizon and "
        "time slice, so lower, typical, and higher keep the same meaning when you "
        "focus on one part of NYC."
    )

    st.selectbox(
        "Start with a reliability story",
        options=list(STORY_PRESETS),
        key="raw19_story",
        on_change=_apply_story_preset,
    )

    control_1, control_2, control_3 = st.columns([1.35, 0.75, 1.35])
    with control_1:
        st.selectbox(
            "Mobility measure",
            options=METRIC_ORDER,
            key="raw19_metric",
            format_func=metric_label,
            on_change=_mark_story_custom,
        )

    with control_2:
        st.selectbox(
            "Forecast horizon",
            options=list(HORIZONS),
            key="raw19_horizon",
            format_func=lambda value: f"h={value}",
            on_change=_mark_story_custom,
        )

    with control_3:
        st.selectbox(
            "Map view",
            options=MAP_VIEW_OPTIONS,
            key="raw19_map_view",
            on_change=_mark_story_custom,
        )

    selected_map_view = str(st.session_state["raw19_map_view"])

    if selected_map_view == "Forecast error":
        control_4, control_5, control_6 = st.columns([1.0, 1.0, 1.15])
    else:
        control_4, control_5 = st.columns(2)
        control_6 = None

    with control_4:
        st.selectbox(
            "Day type",
            options=DAY_TYPE_ORDER,
            key="raw19_day_type",
            on_change=_mark_story_custom,
        )

    with control_5:
        st.selectbox(
            "Daypart",
            options=DAYPART_ORDER,
            key="raw19_daypart",
            on_change=_mark_story_custom,
        )

    if control_6 is not None:
        with control_6:
            st.selectbox(
                "Error measure",
                options=ERROR_MEASURE_OPTIONS,
                key="raw19_error_measure",
                on_change=_mark_story_custom,
                help=(
                    "Relative MAE compares error with the amount of mobility observed. "
                    "Native-unit MAE shows the average miss in trips, riders, or mph."
                ),
            )

    selected_metric = str(st.session_state["raw19_metric"])
    selected_horizon = int(st.session_state["raw19_horizon"])
    selected_day_type = str(st.session_state["raw19_day_type"])
    selected_daypart = str(st.session_state["raw19_daypart"])
    selected_error_measure = str(
        st.session_state.get("raw19_error_measure", "Relative MAE (%)")
    )

    city_scope = selected_reliability(
        temporal_base,
        geography_context,
        metric=selected_metric,
        horizon=selected_horizon,
        day_type=selected_day_type,
        daypart=selected_daypart,
    )

    if city_scope.empty or not city_scope["map_eligible"].any():
        st.warning("No supported Taxi Zones match this mobility measure × horizon and time slice.")
        st.stop()

    with st.expander("Filter the map by geography", expanded=False):
        filter_type = st.selectbox(
            "Geography filter",
            options=GEOGRAPHY_FILTER_TYPES,
            key="raw19_geo_filter_type",
            on_change=_mark_story_custom,
            help=(
                "Choose one spatial lens at a time. Borough, policy geography, and "
                "mobility environment are alternative filters and are never stacked."
            ),
        )

        if filter_type == "Borough":
            filter_column = "borough"
            filter_options = sorted(
                city_scope["borough"].dropna().astype(str).unique().tolist()
            )
        elif filter_type == "Policy geography":
            filter_column = "policy_geography"
            filter_options = sorted(
                city_scope.loc[
                    city_scope["policy_geography"].ne("Unknown"),
                    "policy_geography",
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        elif filter_type == "Mobility environment":
            filter_column = "mobility_environment"
            filter_options = sorted(
                city_scope.loc[
                    city_scope["mobility_environment"].ne("Unknown"),
                    "mobility_environment",
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        else:
            filter_column = None
            filter_options = ["All NYC"]

        if st.session_state.get("raw19_geo_filter_value") not in filter_options:
            st.session_state["raw19_geo_filter_value"] = filter_options[0]

        if filter_type == "All NYC":
            st.caption(
                "Showing all supported NYC Taxi Zones. Choose one geography filter "
                "above to focus the map without stacking spatial definitions."
            )
            selected_geo_filter = "All NYC"
        else:
            selected_geo_filter = st.selectbox(
                "Filter value",
                options=filter_options,
                key="raw19_geo_filter_value",
                on_change=_mark_story_custom,
            )

    visible_scope = city_scope.copy()
    if filter_column is not None:
        visible_scope = visible_scope.loc[
            visible_scope[filter_column].astype(str).eq(str(selected_geo_filter))
        ].copy()

    visible_eligible = visible_scope.loc[visible_scope["map_eligible"]].copy()
    if visible_eligible.empty:
        st.info("The current geography filters contain no supported mapped zones.")
        st.stop()

    # A saved story requests a zone by name once. Resolve it only after the current
    # metric/horizon/time slice has been rebuilt so the selector cannot point to an
    # unavailable zone. Map/scatter clicks use a pending value that is applied on
    # the next rerun BEFORE the selectbox is instantiated; this avoids Streamlit's
    # prohibition on mutating a widget key after that widget has already rendered.
    requested_zone_name = st.session_state.pop("raw19_zone_name_request", None)

    zone_lookup = (
        visible_eligible[["taxi_zone_id", "zone", "borough"]]
        .drop_duplicates("taxi_zone_id")
        .sort_values(["zone", "borough"])
    )
    zone_ids = zone_lookup["taxi_zone_id"].astype(int).tolist()
    zone_label_by_id = {
        int(row.taxi_zone_id): f"{row.zone} · {row.borough}"
        for row in zone_lookup.itertuples(index=False)
    }

    requested_zone_id = None
    if requested_zone_name:
        requested_match = visible_eligible.loc[
            visible_eligible["zone"].astype(str).eq(str(requested_zone_name))
        ]
        if not requested_match.empty:
            requested_zone_id = int(requested_match.iloc[0]["taxi_zone_id"])

    pending_zone_id = st.session_state.pop("raw19_pending_zone_id", None)
    if pending_zone_id in zone_ids:
        st.session_state["raw19_zone_selector"] = int(pending_zone_id)
    elif requested_zone_id in zone_ids:
        st.session_state["raw19_zone_selector"] = int(requested_zone_id)
    elif st.session_state.get("raw19_zone_selector") not in zone_ids:
        reliable_default, _ = representative_extremes(visible_eligible)
        st.session_state["raw19_zone_selector"] = int(
            reliable_default["taxi_zone_id"]
        )

    # Resolve the selected zone here because the linked scatter needs to know which
    # point to outline before it renders. The reader-facing selector itself appears
    # later with the Zone Profile.
    selected_zone_id = int(
        st.session_state["raw19_zone_selector"]
    )

    scope_label_parts = [
        metric_label(selected_metric),
        f"h={selected_horizon}",
    ]
    if selected_day_type != "All days":
        scope_label_parts.append(selected_day_type)
    if selected_daypart != "All dayparts":
        scope_label_parts.append(selected_daypart)
    scope_label = " · ".join(scope_label_parts)

    # The explorer keeps both bivariate views interactive. The map answers WHERE;
    # the scatter helps explain WHY by relating forecast error to typical activity.
    # Both respond to the same filters, and either visual can select the zone profile.
    st.caption(
        "Select a Taxi Zone on either visualization to update the reliability "
        "profile and held-out forecast evidence below. You can also choose a zone "
        "directly from the Zone profile menu above."
    )

    explorer_map_tab, explorer_scatter_tab = st.tabs(
        ["Reliability map", "Reliability vs observed level"]
    )

    with explorer_map_tab:
        explorer_map_title = f"{selected_map_view} · {scope_label}"
        if selected_map_view == "Forecast error":
            explorer_map_title = (
                f"{selected_map_view} · {selected_error_measure} · {scope_label}"
            )

        explorer_map, classified_city, explorer_cuts = build_map_for_view(
            city_scope,
            visible_scope,
            map_view=selected_map_view,
            title=explorer_map_title,
            error_measure=selected_error_measure,
        )

        map_event = st.plotly_chart(
            explorer_map,
            width="stretch",
            config=PLOT_CONFIG,
            key="raw19_explorer_map",
            on_select="rerun",
            selection_mode="points",
        )
        render_chart_insight(
            scope_takeaway(
                city_scope,
                visible_scope,
                map_view=selected_map_view,
                error_measure=selected_error_measure,
            )
        )

    with explorer_scatter_tab:
        explorer_scatter, _, _ = build_activity_scatter(
            city_scope,
            visible_scope,
            title=f"Reliability vs typical observed level · {scope_label}",
            highlight_zone_id=selected_zone_id,
        )
        scatter_event = st.plotly_chart(
            explorer_scatter,
            width="stretch",
            config=PLOT_CONFIG,
            key="raw19_explorer_scatter",
            on_select="rerun",
            selection_mode="points",
        )
        st.caption(
            "Each point is one supported Taxi Zone. Color uses the same citywide 3 × 3 "
            "reliability grouping as the map; the outlined point is the currently "
            "selected zone."
        )

        with st.expander("How to read this view"):
            st.markdown(
                """
    - **Left → right:** typical observed level for the selected mobility measure.
      Trip and ridership measures represent typical demand; speed measures represent
      typical observed speed.
    - The x-axis is **log-scaled** so very different zone scales can share one view.
      Equal horizontal spacing represents multiplicative rather than fixed-unit
      differences.
    - **Bottom → top:** Relative MAE; lower is better.
    - **Color:** the same nine-category reliability grouping as the bivariate map.
    - **Dot size:** final-holdout support.
    - **Outlined dot:** the Taxi Zone currently shown in the profile below.

    Click any point to carry that Taxi Zone into the reliability profile and its
    held-out Observed-vs-Forecast evidence.
    """
            )

        render_chart_insight(
            activity_takeaway(
                city_scope,
                visible_scope,
            )
        )

    clicked_id = _zone_id_from_plotly_event(map_event)
    scatter_clicked_id = _zone_id_from_plotly_event(scatter_event)
    if scatter_clicked_id in zone_ids and scatter_clicked_id != selected_zone_id:
        clicked_id = scatter_clicked_id

    if clicked_id in zone_ids and clicked_id != selected_zone_id:
        st.session_state["raw19_pending_zone_id"] = int(clicked_id)
        st.session_state["raw19_story"] = "Custom view"
        st.rerun()

    # ------------------------------------------------------------------
    # Neighborhood ranking / comparison
    # ------------------------------------------------------------------

    st.markdown("### Neighborhood reliability ranking")
    st.write(
        "Scan all supported neighborhoods under the current Metric × Horizon × "
        "time slice and geography filter. The ordering follows the active Map view "
        "instead of imposing one universal definition of reliability."
    )

    ranking_options = ranking_order_options(selected_map_view)

    if st.session_state.get("raw19_ranking_order") not in ranking_options:
        st.session_state["raw19_ranking_order"] = ranking_options[0]

    ranking_order = st.selectbox(
        "Ranking order",
        options=ranking_options,
        key="raw19_ranking_order",
    )

    ranking_table, ranking_explanation = build_neighborhood_ranking(
        visible_scope,
        map_view=selected_map_view,
        error_measure=selected_error_measure,
        order=ranking_order,
    )

    st.caption(ranking_explanation)

    if ranking_table.empty:
        st.info("No supported neighborhoods are available for this ranking.")
    else:
        st.dataframe(
            ranking_table,
            hide_index=True,
            width="stretch",
            height=420,
        )
        st.caption(
            "Use the map, scatter, or alphabetical Zone profile dropdown below to "
            "inspect any neighborhood in depth."
        )

    # ------------------------------------------------------------------
    # Selected-zone profile
    # ------------------------------------------------------------------

    st.subheader("Explore one neighborhood in detail")

    selected_zone_id = int(
        st.selectbox(
            "Taxi Zone",
            options=zone_ids,
            key="raw19_zone_selector",
            format_func=lambda value: zone_label_by_id.get(
                int(value),
                str(value),
            ),
            on_change=_mark_story_custom,
        )
    )

    selected_row = city_scope.loc[
        city_scope["taxi_zone_id"].eq(selected_zone_id)
    ].iloc[0]
    classified_for_profile, _ = apply_bivariate_classification(city_scope)

    st.subheader(f"{selected_row['zone']} · reliability profile")
    st.caption(
        f"{metric_label(selected_metric)} · h={selected_horizon} · "
        f"{selected_day_type} · {selected_daypart}"
    )

    city_rank_frame = city_scope.loc[city_scope["map_eligible"]].sort_values(
        "relative_mae_pct"
    ).reset_index(drop=True)
    rank_map = {
        int(zone_id): index + 1
        for index, zone_id in enumerate(city_rank_frame["taxi_zone_id"].tolist())
    }
    zone_rank = rank_map.get(selected_zone_id)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Relative MAE", _format_pct(selected_row["relative_mae_pct"]))
    p2.metric(
        "vs Last-week baseline",
        (
            f"{float(selected_row['benchmark_skill_pct']):.1f}% less error"
            if float(selected_row["benchmark_skill_pct"]) >= 0
            else f"{abs(float(selected_row['benchmark_skill_pct'])):.1f}% more error"
        ),
    )
    p3.metric(
        "NYC Relative-MAE rank",
        f"{zone_rank} of {len(city_rank_frame)}" if zone_rank is not None else "—",
    )
    p4.metric("Holdout observations", f"{int(selected_row['forecast_rows']):,}")

    profile_left, profile_right = st.columns(2)
    with profile_left:
        st.markdown(
            f"**Native-unit MAE:** {_format_number(selected_row['mae'])}  \n"
            f"**Typical observed magnitude:** {_format_number(selected_row['observed_abs_mean'])}"
        )
    with profile_right:
        st.markdown(
            f"**Borough:** {selected_row['borough']}  \n"
            f"**Policy geography:** {selected_row['policy_geography']}  \n"
            f"**Mobility environment:** {selected_row['mobility_environment']}"
        )

    render_chart_insight(
        zone_profile_copy(selected_row, city_scope, classified_for_profile)
    )

    st.markdown("### How does reliability change with forecast horizon?")
    st.caption(
        "The selected Taxi Zone, mobility measure, Day type, Daypart, and final-"
        "holdout window stay fixed. Only the forecast horizon changes. Lower "
        "Relative MAE is better; positive baseline improvement means less error "
        "than the Last-week baseline."
    )

    horizon_profile = zone_horizon_profile(
        temporal_base,
        metric=selected_metric,
        taxi_zone_id=selected_zone_id,
        day_type=selected_day_type,
        daypart=selected_daypart,
    )

    if horizon_profile.empty:
        st.info("No supported all-horizon profile is available for this selected slice.")
    else:
        st.plotly_chart(
            build_zone_horizon_profile_chart(
                horizon_profile,
                zone_name=str(selected_row["zone"]),
            ),
            width="stretch",
            config=PLOT_CONFIG,
            key=(
                f"raw19_horizon_profile_{selected_zone_id}_{selected_metric}_"
                f"{selected_day_type}_{selected_daypart}"
            ),
        )
        render_chart_insight(
            zone_horizon_profile_takeaway(
                horizon_profile,
                str(selected_row["zone"]),
            )
        )

    st.markdown("### See the held-out evidence behind this profile")
    st.write(
        "This compact trace shows the observations that produced the reliability "
        "summary above. It is supporting evidence rather than a second forecast "
        "explorer: the selected zone, metric, horizon, Day type, and Daypart stay fixed."
    )

    zone_evidence = load_zone_evidence(
        selected_metric,
        selected_horizon,
        selected_zone_id,
    )
    if selected_day_type != "All days":
        zone_evidence = zone_evidence.loc[
            zone_evidence["day_type"].eq(selected_day_type)
        ].copy()
    if selected_daypart != "All dayparts":
        zone_evidence = zone_evidence.loc[
            zone_evidence["daypart"].eq(selected_daypart)
        ].copy()

    if zone_evidence.empty:
        st.info("No held-out row-level evidence is available for this selected slice.")
    else:
        st.markdown(
            f"#### Observed vs forecast · {selected_row['zone']}"
        )
        st.plotly_chart(
            build_zone_evidence_chart(
                zone_evidence,
                metric=selected_metric,
                zone_name=str(selected_row["zone"]),
            ),
            width="stretch",
            config=PLOT_CONFIG,
            key=(
                f"raw19_zone_evidence_{selected_zone_id}_{selected_metric}_"
                f"{selected_horizon}_{selected_day_type}_{selected_daypart}"
            ),
        )
        st.caption(
            "The Last-week baseline stays visible as the simple reference forecast, "
            "so the profile can be checked against the held-out observations directly."
        )


# ------------------------------------------------------------------
# Closing synthesis
# ------------------------------------------------------------------

st.divider()

st.markdown("### What this page establishes")
st.markdown(
    "Forecast reliability is spatially uneven, and **forecast error** is not the same "
    "question as **forecast value**. A neighborhood can be difficult to predict while "
    "the selected model still improves substantially on Last-week, or it can have low "
    "error where the simple baseline was already competitive. Trust therefore depends "
    "on both the size of the miss and what the forecasting system adds beyond a simple "
    "recent-history rule."
)

with st.expander("How this page works", expanded=False):
    st.markdown(
        """
        **1. Keep every reliability claim on unseen data.** All maps, rankings, and
        bivariate comparison groups use the untouched **Jan 5–Mar 31, 2026 final
        holdout**.

        **2. Measure error in two complementary ways.** **Relative MAE** scales total
        absolute error by total observed mobility, making differently sized zones more
        comparable proportionally. **Native-unit MAE** reports the average miss in
        trips, riders, or mph, which is easier to interpret practically but naturally
        tends to be larger in higher-volume places.

        **3. Ask separately whether the model adds value.** **Improvement vs Last-week
        baseline** compares the selected forecast with the simple assumption that the
        corresponding mobility value repeats from one week earlier. Positive values
        mean the selected forecast reduced error; negative values mean Last-week was
        closer.

        **4. Combine error and model advantage without collapsing them into one score.**
        The bivariate map uses citywide tertiles for the selected mobility measure,
        horizon, day type, and daypart. Its nine classes preserve both dimensions so a
        difficult-but-helpful zone remains distinct from a low-error zone where the
        baseline was already competitive.

        **5. Keep filtered subsets comparable with the citywide frame.** Borough,
        policy-geography, and mobility-environment filters do not recalculate the
        bivariate cut points. A subset therefore retains its position relative to the
        same citywide reliability pattern.

        **6. Require enough evaluation support for a mapped claim.** A Taxi Zone must
        retain at least half of the typical support for the selected slice, a usable
        Last-week comparison, and nonzero observed activity before it can define a
        mapped reliability result.
        """
    )

st.caption(
    "Evidence scope: held-out predictive performance during Jan 5–Mar 31, 2026. "
    "Forecast error and improvement versus Last-week describe where the selected "
    "forecasting system was more or less dependable; they do not explain why mobility "
    "was difficult to predict in a particular place."
)
