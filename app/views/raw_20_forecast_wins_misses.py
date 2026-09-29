"""Raw 20 — Forecast Wins & Misses."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data_access.forecasting import (
    forecast_day_type,
    forecast_daypart,
    load_forecast_failure_cases,
    load_forecast_failure_context,
    load_forecast_failure_mechanisms,
    load_forecast_holdout_curated_cases,
    load_forecast_holdout_scatter_sample,
    load_forecast_holdout_slice_summary,
    load_forecast_records,
)

from app.utils.project_branding import (
    BRAND_COLORS,
    apply_branding,
    exploration_section,
    inject_app_css,
    render_chart_insight,
)


# ---------------------------------------------------------------------
# Page question
# ---------------------------------------------------------------------
# Pages 18 and 19 already answer whether the frozen forecasting system works
# overall and where its reliability varies spatially. Page 20 asks a different
# question: what does a forecast win look like, what does a real miss look like,
# and which recurring conditions make misses more likely or more costly?
#
# The dual lens is deliberate. "Win" asks whether the selected forecast beats
# the Last-week baseline. "Miss" asks how far the selected forecast is from what
# actually happened. Those are not opposites: a hard forecast can miss badly and
# still add value when the simple baseline misses by even more.
# ---------------------------------------------------------------------

PAGE_CAPTION = "FORECAST WINS & MISSES"
PAGE_TITLE = "A forecast can miss badly and still be useful"

PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
SCATTER_RANDOM_ROWS = 12_000
SCATTER_TAIL_ROWS = 350
CASE_TABLE_ROWS = 8
STRIP_ZONE_COUNT_OPTIONS = [8, 12, 16, 20]

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

METRIC_UNITS = {
    "taxi_trip_count": "trips",
    "taxi_avg_trip_speed": "mph",
    "fhvhv_trip_count": "trips",
    "fhvhv_avg_trip_speed": "mph",
    "subway_ridership": "riders",
}

DAYPART_LABELS = {
    "overnight": "Overnight",
    "am_peak": "AM peak",
    "midday": "Midday",
    "pm_peak": "PM peak",
    "evening": "Evening",
}
DAYPART_FILTER_VALUES = {
    "Overnight": "overnight",
    "AM peak": "am_peak",
    "Midday": "midday",
    "PM peak": "pm_peak",
    "Evening": "evening",
}

RECORD_COLUMNS = [
    "metric",
    "horizon",
    "taxi_zone_id",
    "zone",
    "borough",
    "target_date",
    "target_temporal_bucket",
    "champion_family",
    "actual",
    "champion_prediction",
    "benchmark_prediction",
    "absolute_error",
    "benchmark_absolute_error",
    "champion_better_than_benchmark",
    "severe_error",
    "failure_combination",
    "system_row_error_index",
    "benchmark_advantage_index",
    "reader_facing_zone",
]

FAILURE_CASE_REQUIRED = [
    "candidate_id",
    "failure_category",
    "metric",
    "horizon",
    "taxi_zone_id",
    "zone",
    "borough",
    "target_date",
    "target_temporal_bucket",
    "champion_family",
    "actual",
    "system_prediction",
    "benchmark_prediction",
    "absolute_error",
    "benchmark_absolute_error",
    "failure_combination",
]

FAILURE_CONTEXT_REQUIRED = [
    "candidate_id",
    "metric",
    "horizon",
    "zone",
    "borough",
    "target_date",
    "target_temporal_bucket",
    "actual",
    "system_prediction",
    "benchmark_prediction",
]

MECHANISM_REQUIRED = [
    "Mechanism",
    "Short label",
    "System forecast share (%)",
    "System error burden (%)",
    "Severe-error capture (%)",
    "Benchmark-loss capture (%)",
    "Burden / forecast share (×)",
]

LENS_ORDER = [
    "Forecast win · ordinary error",
    "Hard but helpful · severe error",
    "Last-week better · ordinary error",
    "Clear miss · severe error",
    "Tie · ordinary error",
    "Tie · severe error",
]

LENS_COLORS = {
    "Forecast win · ordinary error": BRAND_COLORS["dark_teal"],
    "Hard but helpful · severe error": BRAND_COLORS["seafoam"],
    "Last-week better · ordinary error": BRAND_COLORS["pale_peach"],
    "Clear miss · severe error": BRAND_COLORS["terracotta"],
}

PLANE_QUADRANT_ORDER = [
    "Lower-error forecast wins",
    "Higher-error but still helpful",
    "Lower-error Last-week wins",
    "Higher-error misses",
]

PLANE_TIE_LABEL = "Tie with Last-week"

PLANE_QUADRANT_COLORS = {
    "Lower-error forecast wins": BRAND_COLORS["dark_teal"],
    "Higher-error but still helpful": BRAND_COLORS["seafoam"],
    "Lower-error Last-week wins": BRAND_COLORS["pale_peach"],
    "Higher-error misses": BRAND_COLORS["terracotta"],
    PLANE_TIE_LABEL: "#7A878C",
}

MECHANISM_METRIC_DEFINITIONS = {
    "Forecast share": (
        "Share of all final-holdout forecast records that occur under this "
        "diagnostic condition."
    ),
    "Error burden": (
        "Share of total selected-forecast absolute error contributed by records "
        "under this diagnostic condition."
    ),
    "Severe-error capture": (
        "Share of all frozen severe-error records that occur under this "
        "diagnostic condition."
    ),
    "Benchmark-loss capture": (
        "Share of records where Last-week beats the selected forecast that occur "
        "under this diagnostic condition."
    ),
    "Burden / forecast share": (
        "Error-burden share divided by forecast share. Values above 1× mean the "
        "condition carries more error than its frequency alone would suggest."
    ),
}

DIAGNOSTIC_DEFINITIONS = {
    "Low / zero demand": (
        "Demand forecast made when recent same-daypart activity was in the "
        "calibrated low tail for that mobility measure × horizon, or an exact trip-count "
        "zero warning was present."
    ),
    "Structural low / zero demand": (
        "Demand forecast made when recent same-daypart activity was in the "
        "calibrated low tail for that mobility measure × horizon, or an exact trip-count "
        "zero warning was present."
    ),
    "High speed volatility": (
        "Speed forecast made when recent same-daypart 28-day volatility was in "
        "the calibrated high tail for that mobility measure × horizon."
    ),
    "Unstable recent conditions": (
        "Speed forecast made when recent same-daypart 28-day volatility was in "
        "the calibrated high tail for that mobility measure × horizon."
    ),
    "Shared-shock weeks": (
        "A broad difficult week: enough forecasting jobs were observed and even "
        "the 25th-percentile job had an error index above 100, meaning roughly "
        "three-quarters of jobs were running worse than their normal MAE."
    ),
    "Temporary shared shocks": (
        "A broad difficult week: enough forecasting jobs were observed and even "
        "the 25th-percentile job had an error index above 100, meaning roughly "
        "three-quarters of jobs were running worse than their normal MAE."
    ),
    "No identified mechanism": (
        "None of the frozen diagnostic conditions triggered for this record. "
        "That does not mean the miss had no cause; only that these diagnostics "
        "did not identify one."
    ),
}


# ---------------------------------------------------------------------
# Basic guards and formatting
# ---------------------------------------------------------------------

def metric_label(metric: str) -> str:
    """Return the reader-facing name for one forecasting target."""
    return METRIC_LABELS.get(metric, metric)


def metric_unit(metric: str) -> str:
    """Return the native unit for observed and predicted values."""
    return METRIC_UNITS.get(metric, "units")


def format_number(value: float | int | None, decimals: int = 3) -> str:
    """Format a finite number without exposing floating-point noise."""
    if value is None or not np.isfinite(float(value)):
        return "—"

    number = float(value)
    if abs(number) >= 1000:
        return f"{number:,.{min(decimals, 1)}f}"
    return f"{number:,.{decimals}f}".rstrip("0").rstrip(".")


def format_native(metric: str, value: float | int | None) -> str:
    """Format counts as whole values and speeds to at most three decimals."""
    if value is None or not np.isfinite(float(value)):
        return "—"

    number = float(value)
    if metric in {"taxi_trip_count", "fhvhv_trip_count", "subway_ridership"}:
        return f"{number:,.0f}"
    return format_number(number, decimals=3)


def format_pct(value: float | int | None, signed: bool = False) -> str:
    """Format a percentage-like value with no more than three decimals."""
    if value is None or not np.isfinite(float(value)):
        return "—"
    spec = "+.3f" if signed else ".3f"
    return f"{float(value):{spec}}%".replace("+0.000%", "0.000%")


def temporal_bucket_label(value: str) -> str:
    """Turn the compact temporal-bucket code into readable day-type/daypart text."""
    text = str(value)
    day_type = "Weekend" if text.startswith("weekend_") else "Weekday"
    daypart = text.replace("weekend_", "").replace("weekday_", "")
    return f"{day_type} · {DAYPART_LABELS.get(daypart, daypart.replace('_', ' ').title())}"


def analysis_lens(frame: pd.DataFrame) -> pd.Series:
    """
    Classify rows using two independent ideas: error severity and baseline gain.

    Exact ties are kept separate. A zero baseline-advantage index means the
    selected forecast and Last-week had the same absolute error; it is not a
    Last-week win.
    """
    severe = frame["severe_error"].fillna(False).astype(bool)
    advantage = pd.to_numeric(
        frame["benchmark_advantage_index"],
        errors="coerce",
    )

    model_wins = advantage.gt(0)
    baseline_wins = advantage.lt(0)
    ties = advantage.eq(0)

    result = np.select(
        [
            model_wins & ~severe,
            model_wins & severe,
            baseline_wins & ~severe,
            baseline_wins & severe,
            ties & ~severe,
            ties & severe,
        ],
        LENS_ORDER,
        default="Unavailable comparison",
    )
    return pd.Series(result, index=frame.index, dtype="string")


def plane_quadrant(frame: pd.DataFrame) -> pd.Series:
    """
    Classify records by the two boundaries drawn on the Win–Miss Plane.

    x=100 separates smaller- from larger-than-MAE misses. y=0 separates
    selected-forecast wins from Last-week wins. Exact y=0 ties remain a fifth,
    neutral outcome rather than being assigned to either side.
    """
    error_index = pd.to_numeric(frame["system_row_error_index"], errors="coerce")
    advantage = pd.to_numeric(frame["benchmark_advantage_index"], errors="coerce")

    valid = error_index.notna() & advantage.notna() & error_index.ge(0)
    lower_error = error_index.lt(100.0)
    forecast_wins = advantage.gt(0.0)
    baseline_wins = advantage.lt(0.0)
    ties = advantage.eq(0.0)

    result = np.select(
        [
            valid & lower_error & forecast_wins,
            valid & ~lower_error & forecast_wins,
            valid & lower_error & baseline_wins,
            valid & ~lower_error & baseline_wins,
            valid & ties,
        ],
        [
            *PLANE_QUADRANT_ORDER,
            PLANE_TIE_LABEL,
        ],
        default="Unavailable comparison",
    )
    return pd.Series(result, index=frame.index, dtype="string")


def diagnostic_context_definition(value: str) -> str:
    """Explain one frozen diagnostic-context label in reader-facing language."""
    label = str(value)

    if label in DIAGNOSTIC_DEFINITIONS:
        return DIAGNOSTIC_DEFINITIONS[label]

    matched = []
    for name, definition in DIAGNOSTIC_DEFINITIONS.items():
        if name == "No identified mechanism":
            continue
        if name.lower() in label.lower():
            matched.append(definition)

    if matched:
        return " ".join(dict.fromkeys(matched))

    return (
        "A frozen upstream diagnostic label used to describe the conditions "
        "surrounding this forecast record."
    )


def render_mechanism_definitions() -> None:
    """Show plain-language definitions beside the exact mechanism summary."""
    st.markdown(
        """
**Diagnostic conditions**

- **Low / zero demand:** Recent activity for the same time of day was unusually low for that mobility outcome and forecast horizon, or the trip-count history included an exact zero warning.
- **High speed volatility:** Recent same-time-of-day speeds were unusually unstable for that mobility outcome and forecast horizon.
- **Shared-shock weeks:** Broadly difficult weeks when forecast errors were elevated across much of the system at the same time.

These conditions help describe **when forecasts were difficult**. They do not remove records or change the forecasts after the fact.

**Chart measures**

- **Forecast share:** How often the condition appears in the final-holdout forecasts.
- **Error burden:** How much of the selected forecast's total absolute error comes from records under the condition.
- **Severe-error capture:** How much of the page's severe-error set occurs under the condition.
- **Benchmark-loss capture:** How much of the records where Last-week was closer occur under the condition.
- **Burden / forecast share:** Error burden divided by forecast share. Values above **1×** mean the condition carries more error than its frequency alone would suggest.
"""
    )

def render_band_rate_definitions() -> None:
    """Explain the Taxi Zone ordering measures at their point of use."""
    st.markdown(
        """
These measures are calculated **within each Taxi Zone using all forecast records that match the current Win–Miss Band filters**. They are not calculated from only the one representative record drawn for each date.

- **Severe-error rate:** Share of matching forecasts in the frozen **worst 10% of absolute errors for that mobility measure × horizon**.
- **Clear-miss rate:** Share of matching forecasts that are both **severe errors** and cases where **Last-week was closer** than the selected forecast.
- **Hard-but-helpful rate:** Share of matching forecasts that are **severe errors** but where the **selected forecast was still closer** than Last-week.
- **Model-win rate:** Share of matching forecasts where the **selected forecast beat Last-week**, whether the error was severe or ordinary.
- **Tie rate:** Share where the selected forecast and Last-week had exactly the same absolute error. Ties are neither model wins nor Last-week wins.
- **Median error severity:** Median **forecast error index** for the zone. **100 = one mobility-measure × horizon MAE**; values above 100 are larger than that job's typical absolute error.

Because the severe-error flag is defined within each mobility measure × horizon before these day-type/daypart filters are applied, a filtered Taxi Zone's severe-error rate does **not** have to equal 10%.
"""
    )


def comparison_population(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep records with usable error and baseline-comparison indices."""
    error_index = pd.to_numeric(
        frame["system_row_error_index"],
        errors="coerce",
    )
    advantage = pd.to_numeric(
        frame["benchmark_advantage_index"],
        errors="coerce",
    )

    return frame.loc[
        error_index.notna()
        & error_index.ge(0)
        & advantage.notna()
    ].copy()


def quadrant_share_series(frame: pd.DataFrame) -> pd.Series:
    """
    Return four quadrant shares using the complete supported population.

    Exact ties remain in the denominator but not in any quadrant, so the four
    quadrant shares plus the tie share sum to 100%.
    """
    supported = comparison_population(frame)

    if supported.empty:
        return pd.Series(
            0.0,
            index=PLANE_QUADRANT_ORDER,
            dtype=float,
        )

    counts = plane_quadrant(supported).value_counts()
    return (
        counts
        .reindex(PLANE_QUADRANT_ORDER, fill_value=0)
        .div(len(supported))
        .mul(100)
    )


def baseline_outcome_shares(frame: pd.DataFrame) -> dict[str, float]:
    """Return forecast-win, Last-week-win, and exact-tie shares."""
    supported = comparison_population(frame)

    if supported.empty:
        return {
            "forecast_win": np.nan,
            "baseline_win": np.nan,
            "tie": np.nan,
        }

    advantage = pd.to_numeric(
        supported["benchmark_advantage_index"],
        errors="coerce",
    )
    denominator = len(supported)

    return {
        "forecast_win": 100 * float(advantage.gt(0).sum()) / denominator,
        "baseline_win": 100 * float(advantage.lt(0).sum()) / denominator,
        "tie": 100 * float(advantage.eq(0).sum()) / denominator,
    }


def explorer_plane_takeaway(
    visible: pd.DataFrame,
    reference: pd.DataFrame,
) -> str:
    """Explain the current explorer slice relative to the full final holdout."""
    visible_shares = quadrant_share_series(visible)
    reference_shares = quadrant_share_series(reference)
    outcome_shares = baseline_outcome_shares(visible)

    dominant = str(visible_shares.idxmax())
    dominant_share = float(visible_shares.loc[dominant])

    miss_label = "Higher-error misses"
    miss_share = float(visible_shares.loc[miss_label])
    miss_delta = miss_share - float(reference_shares.loc[miss_label])

    severe = visible["severe_error"].fillna(False).astype(bool)
    advantage = pd.to_numeric(
        visible["benchmark_advantage_index"],
        errors="coerce",
    )
    severe_supported = severe & advantage.notna()
    severe_count = int(severe_supported.sum())
    severe_helpful_count = int(
        (severe_supported & advantage.gt(0)).sum()
    )

    severe_sentence = (
        f" Among severe errors with a usable baseline comparison, "
        f"**{100 * severe_helpful_count / severe_count:.1f}%** are still "
        "closer than Last-week."
        if severe_count
        else " This slice contains no severe errors with a usable baseline comparison."
    )

    if np.isclose(miss_delta, 0.0):
        comparison_sentence = (
            "the same share as in the full final holdout"
        )
    else:
        direction = "higher" if miss_delta > 0 else "lower"
        comparison_sentence = (
            f"**{abs(miss_delta):.1f} percentage points {direction}** "
            "than the full final holdout"
        )

    tie_sentence = (
        f" Exact ties account for **{outcome_shares['tie']:.2f}%**."
        if np.isfinite(outcome_shares["tie"]) and outcome_shares["tie"] > 0
        else ""
    )

    return (
        f"The largest quadrant in this slice is **{dominant}** at "
        f"**{dominant_share:.1f}%** of supported records. The selected forecast "
        f"is closer than Last-week on **{outcome_shares['forecast_win']:.1f}%** "
        f"of rows; Last-week is closer on **{outcome_shares['baseline_win']:.1f}%**. "
        f"**Higher-error misses** account for **{miss_share:.1f}%**, "
        f"{comparison_sentence}.{tie_sentence}{severe_sentence}"
    )

def explorer_plane_takeaway_from_summary(
    visible_summary: pd.DataFrame,
    reference_summary: pd.DataFrame,
) -> str:
    """Explain one explorer slice using exact additive summary statistics."""
    visible_shares = exact_quadrant_shares(visible_summary)
    reference_shares = exact_quadrant_shares(reference_summary)
    visible_stats = summary_statistics(visible_summary)

    dominant = str(visible_shares.idxmax())
    dominant_share = float(visible_shares.loc[dominant])

    miss_label = "Higher-error misses"
    miss_share = float(visible_shares.loc[miss_label])
    miss_delta = miss_share - float(reference_shares.loc[miss_label])

    if np.isclose(miss_delta, 0.0):
        comparison_sentence = "the same share as in the full final holdout"
    else:
        direction = "higher" if miss_delta > 0 else "lower"
        comparison_sentence = (
            f"**{abs(miss_delta):.1f} percentage points {direction}** "
            "than the full final holdout"
        )

    tie_share = visible_stats["tie_share"]
    tie_sentence = (
        f" Exact ties account for **{tie_share:.2f}%**."
        if np.isfinite(tie_share) and tie_share > 0
        else ""
    )

    severe_helpful = visible_stats["severe_helpful_share"]
    severe_sentence = (
        f" Among severe errors, **{severe_helpful:.1f}%** are still closer "
        "than Last-week."
        if np.isfinite(severe_helpful)
        else " This slice contains no severe errors."
    )

    return (
        f"The largest quadrant in this slice is **{dominant}** at "
        f"**{dominant_share:.1f}%** of supported records. The selected forecast "
        f"is closer than Last-week on **{visible_stats['model_win_share']:.1f}%** "
        f"of rows; Last-week is closer on "
        f"**{visible_stats['last_week_win_share']:.1f}%**. "
        f"**Higher-error misses** account for **{miss_share:.1f}%**, "
        f"{comparison_sentence}.{tie_sentence}{severe_sentence}"
    )

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_page_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load compact page-wide evidence through the shared forecasting layer."""
    summary = load_forecast_holdout_slice_summary()
    scatter = load_forecast_holdout_scatter_sample()
    curated = load_forecast_holdout_curated_cases()

    scatter["severe_error"] = scatter["severe_error"].fillna(False).astype(bool)
    scatter["failure_combination"] = (
        scatter["failure_combination"]
        .astype("string")
        .fillna("No identified mechanism")
    )
    scatter["day_type"] = forecast_day_type(
        scatter["target_temporal_bucket"]
    )
    scatter["daypart"] = forecast_daypart(
        scatter["target_temporal_bucket"]
    )

    curated["severe_error"] = curated["severe_error"].fillna(False).astype(bool)
    curated["failure_combination"] = (
        curated["failure_combination"]
        .astype("string")
        .fillna("No identified mechanism")
    )

    cases = load_forecast_failure_cases(
        required_columns=FAILURE_CASE_REQUIRED,
    )

    context = load_forecast_failure_context(
        required_columns=FAILURE_CONTEXT_REQUIRED,
    )

    mechanisms = load_forecast_failure_mechanisms(
        required_columns=MECHANISM_REQUIRED,
    )

    return (
        summary,
        scatter,
        curated,
        cases,
        context,
        mechanisms,
    )

# ---------------------------------------------------------------------
# Page filters
# ---------------------------------------------------------------------

def apply_page_filters(
    frame: pd.DataFrame,
    *,
    metric: str,
    horizon: str,
    day_type: str,
    daypart: str,
) -> pd.DataFrame:
    """Apply only the few filters needed to stress-test the visual concepts."""
    selected = frame

    if metric != "All targets":
        selected = selected.loc[selected["metric"].eq(metric)]
    if horizon != "All horizons":
        selected = selected.loc[selected["horizon"].eq(int(horizon))]
    if day_type != "All days":
        selected = selected.loc[selected["day_type"].eq(day_type)]
    if daypart != "All dayparts":
        selected = selected.loc[selected["daypart"].eq(daypart)]

    return selected.copy()

def apply_summary_filters(
    frame: pd.DataFrame,
    *,
    metric: str,
    horizon: str,
    day_type: str,
    daypart: str,
) -> pd.DataFrame:
    """Filter Raw 20's exact additive summary using reader-facing controls."""
    selected = frame

    if metric != "All targets":
        selected = selected.loc[selected["metric"].eq(metric)]

    if horizon != "All horizons":
        selected = selected.loc[selected["horizon"].eq(int(horizon))]

    if day_type != "All days":
        selected = selected.loc[selected["day_type"].eq(day_type)]

    if daypart != "All dayparts":
        selected = selected.loc[
            selected["daypart"].eq(DAYPART_FILTER_VALUES[daypart])
        ]

    return selected.copy()


def summary_statistics(frame: pd.DataFrame) -> dict[str, float]:
    """Recover exact additive Win-Miss statistics from one summary slice."""
    forecast_rows = int(frame["forecast_rows"].sum())
    supported_rows = int(frame["supported_rows"].sum())

    def supported_share(column: str) -> float:
        if not supported_rows:
            return np.nan
        return 100 * float(frame[column].sum()) / supported_rows

    severe_rows = int(frame["severe_rows"].sum())
    hard_helpful_rows = int(frame["hard_helpful_rows"].sum())

    return {
        "forecast_rows": forecast_rows,
        "supported_rows": supported_rows,
        "model_win_share": supported_share("model_win_rows"),
        "last_week_win_share": supported_share("last_week_win_rows"),
        "tie_share": supported_share("tie_rows"),
        "lower_error_model_win_share": supported_share(
            "lower_error_model_win_rows"
        ),
        "higher_error_model_win_share": supported_share(
            "higher_error_model_win_rows"
        ),
        "lower_error_last_week_win_share": supported_share(
            "lower_error_last_week_win_rows"
        ),
        "higher_error_last_week_win_share": supported_share(
            "higher_error_last_week_win_rows"
        ),
        "severe_share": (
            100 * severe_rows / forecast_rows
            if forecast_rows
            else np.nan
        ),
        "severe_helpful_share": (
            100 * hard_helpful_rows / severe_rows
            if severe_rows
            else np.nan
        ),
    }


def exact_quadrant_shares(frame: pd.DataFrame) -> pd.Series:
    """Return exact Plane quadrant percentages from the additive summary."""
    stats = summary_statistics(frame)

    return pd.Series(
        {
            "Lower-error forecast wins": stats[
                "lower_error_model_win_share"
            ],
            "Higher-error but still helpful": stats[
                "higher_error_model_win_share"
            ],
            "Lower-error Last-week wins": stats[
                "lower_error_last_week_win_share"
            ],
            "Higher-error misses": stats[
                "higher_error_last_week_win_share"
            ],
        },
        dtype=float,
    )

@st.cache_data(show_spinner=False)
def load_band_records(
    metric: str,
    horizon: int,
) -> pd.DataFrame:
    """Load only one metric × horizon slice for Win-Miss Bands."""
    frame = load_forecast_records(
        columns=RECORD_COLUMNS,
        required_columns=RECORD_COLUMNS,
        metrics=metric,
        horizons=int(horizon),
        reader_facing_only=True,
        final_holdout_only=True,
    )

    frame["severe_error"] = frame["severe_error"].fillna(False).astype(bool)
    frame["failure_combination"] = (
        frame["failure_combination"]
        .astype("string")
        .fillna("No identified mechanism")
    )
    frame["day_type"] = forecast_day_type(
        frame["target_temporal_bucket"]
    )

    frame["daypart"] = forecast_daypart(
        frame["target_temporal_bucket"]
    )

    return frame


@st.cache_data(show_spinner=False)
def load_record_context_rows(
    metric: str,
    horizon: int,
    taxi_zone_id: int,
    temporal_bucket: str,
) -> pd.DataFrame:
    """Load only the exact series needed around one curated forecast case."""
    return load_forecast_records(
        columns=RECORD_COLUMNS,
        required_columns=RECORD_COLUMNS,
        metrics=metric,
        horizons=int(horizon),
        taxi_zone_ids=int(taxi_zone_id),
        temporal_buckets=temporal_bucket,
        reader_facing_only=True,
        final_holdout_only=True,
    )

# ---------------------------------------------------------------------
# Win–Miss plane
# ---------------------------------------------------------------------


def scatter_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the plot responsive without hiding the tails that matter to Page 20.

    The random core is supplemented with the strongest forecast wins, strongest
    Last-week wins, largest selected-forecast errors, severe cases on both sides,
    and exact ties. Perfect zero-error forecasts remain eligible even though the
    log x-axis needs a small display offset for them.
    """
    eligible = comparison_population(frame)

    if eligible.empty:
        return eligible

    random_part = eligible.sample(
        n=min(SCATTER_RANDOM_ROWS, len(eligible)),
        random_state=696,
    )
    strongest_wins = eligible.nlargest(
        min(SCATTER_TAIL_ROWS, len(eligible)),
        "benchmark_advantage_index",
    )
    strongest_losses = eligible.nsmallest(
        min(SCATTER_TAIL_ROWS, len(eligible)),
        "benchmark_advantage_index",
    )
    largest_errors = eligible.nlargest(
        min(SCATTER_TAIL_ROWS, len(eligible)),
        "system_row_error_index",
    )
    severe_helpful = (
        eligible.loc[
            eligible["severe_error"]
            & eligible["benchmark_advantage_index"].gt(0)
        ]
        .nlargest(SCATTER_TAIL_ROWS, "system_row_error_index")
    )
    severe_misses = (
        eligible.loc[
            eligible["severe_error"]
            & eligible["benchmark_advantage_index"].lt(0)
        ]
        .nlargest(SCATTER_TAIL_ROWS, "system_row_error_index")
    )
    ties = eligible.loc[
        pd.to_numeric(
            eligible["benchmark_advantage_index"],
            errors="coerce",
        ).eq(0)
    ].head(SCATTER_TAIL_ROWS)

    return (
        pd.concat(
            [
                random_part,
                strongest_wins,
                strongest_losses,
                largest_errors,
                severe_helpful,
                severe_misses,
                ties,
            ],
            axis=0,
        )
        .loc[lambda df: ~df.index.duplicated(keep="first")]
        .copy()
    )


def build_win_miss_plane(
    frame: pd.DataFrame,
    *,
    exact_summary: pd.DataFrame | None = None,
    pre_sampled: bool = False,
) -> go.Figure:
    """
    Plot forecast-error magnitude against value added over Last-week.

    Population percentages use every supported record, including perfect
    zero-error forecasts and exact ties. The log x-axis cannot display zero,
    so perfect forecasts are drawn at a small left-edge display offset while
    their true zero error remains explicit in hover.
    """
    eligible = comparison_population(frame)
    figure = go.Figure()

    if eligible.empty:
        return figure

    error_index = pd.to_numeric(
        eligible["system_row_error_index"],
        errors="coerce",
    )
    positive_x = error_index.loc[error_index.gt(0)]

    if positive_x.empty:
        display_floor = 0.1
        x_low = 0.1
        x_high = 100.0
    else:
        smallest_positive = float(positive_x.min())
        display_floor = max(smallest_positive / 2.0, 1e-6)
        x_low = min(
            display_floor,
            max(float(positive_x.quantile(0.001)), 1e-6),
        )
        x_high = float(positive_x.quantile(0.9999))
        x_high = max(x_high, x_low * 10, 100.0)

    sample = (
        eligible.copy()
        if pre_sampled
        else scatter_sample(eligible)
    )
    if sample.empty:
        return figure

    sample["plane_quadrant"] = plane_quadrant(sample)
    sample["plot_error_index"] = (
        pd.to_numeric(
            sample["system_row_error_index"],
            errors="coerce",
        )
        .where(lambda values: values.gt(0), display_floor)
    )

    if exact_summary is None:
        quadrant_share = quadrant_share_series(eligible)
        outcome_shares = baseline_outcome_shares(eligible)
    else:
        exact_stats = summary_statistics(exact_summary)
        quadrant_share = exact_quadrant_shares(exact_summary)
        outcome_shares = {
            "forecast_win": exact_stats["model_win_share"],
            "baseline_win": exact_stats["last_week_win_share"],
            "tie": exact_stats["tie_share"],
        }

    full_y = pd.to_numeric(
        eligible["benchmark_advantage_index"],
        errors="coerce",
    )
    finite_y = full_y.loc[full_y.notna()]
    y_limit = (
        float(np.quantile(np.abs(finite_y), 0.995))
        if len(finite_y)
        else 100.0
    )
    y_limit = max(y_limit, 25.0)

    plot_order = [*PLANE_QUADRANT_ORDER, PLANE_TIE_LABEL]

    for outcome in plot_order:
        group = sample.loc[
            sample["plane_quadrant"].eq(outcome)
        ].copy()

        if group.empty:
            continue

        customdata = np.column_stack(
            [
                group["zone"].astype(str),
                group["borough"].astype(str),
                group["metric"].map(METRIC_LABELS).fillna(group["metric"]).astype(str),
                group["horizon"].astype(str),
                group["target_date"].dt.strftime("%b %d, %Y"),
                group["target_temporal_bucket"].map(temporal_bucket_label),
                group["actual"].map(lambda value: format_number(value, 3)),
                group["champion_prediction"].map(lambda value: format_number(value, 3)),
                group["benchmark_prediction"].map(lambda value: format_number(value, 3)),
                group["failure_combination"].astype(str),
                np.where(group["severe_error"], "Yes", "No"),
                group["system_row_error_index"].map(
                    lambda value: format_number(value, 3)
                ),
            ]
        )

        figure.add_trace(
            go.Scattergl(
                x=group["plot_error_index"],
                y=group["benchmark_advantage_index"],
                mode="markers",
                name=outcome,
                marker={
                    "size": 6.5 if outcome == PLANE_TIE_LABEL else 6,
                    "symbol": "diamond" if outcome == PLANE_TIE_LABEL else "circle",
                    "opacity": 0.72 if outcome == PLANE_TIE_LABEL else 0.58,
                    "color": PLANE_QUADRANT_COLORS[outcome],
                    "line": {
                        "width": 0.35,
                        "color": BRAND_COLORS["dark_teal"],
                    },
                },
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                    "%{customdata[2]} · h=%{customdata[3]}<br>"
                    "%{customdata[4]} · %{customdata[5]}<br><br>"
                    "Forecast error index: %{customdata[11]}<br>"
                    "Baseline advantage index: %{y:.3f}<br>"
                    "Observed: %{customdata[6]}<br>"
                    "Selected forecast: %{customdata[7]}<br>"
                    "Last-week baseline: %{customdata[8]}<br>"
                    "Severe-error flag: %{customdata[10]}<br>"
                    "Diagnostic context: %{customdata[9]}"
                    "<extra></extra>"
                ),
            )
        )

    figure.add_hline(
        y=0,
        line_width=1.3,
        line_dash="dash",
        line_color=BRAND_COLORS["dark_teal"],
    )
    figure.add_vline(
        x=100,
        line_width=1.3,
        line_dash="dot",
        line_color=BRAND_COLORS["terracotta"],
    )

    log_low = np.log10(x_low)
    log_high = np.log10(x_high)

    if log_low <= 2 <= log_high:
        mae_paper_x = (2 - log_low) / (log_high - log_low)
        figure.add_annotation(
            x=mae_paper_x,
            y=1.005,
            xref="paper",
            yref="paper",
            text="1× measure × horizon MAE",
            showarrow=False,
            yanchor="bottom",
            font={
                "size": 11,
                "color": BRAND_COLORS["terracotta"],
            },
        )

    annotation_specs = [
        ("Lower-error forecast wins", 0.04, 0.95, "left"),
        ("Higher-error but still helpful", 0.96, 0.95, "right"),
        ("Lower-error Last-week wins", 0.04, 0.06, "left"),
        ("Higher-error misses", 0.96, 0.06, "right"),
    ]

    for label, x, y, anchor in annotation_specs:
        negative_side = "misses" in label or "Last-week" in label
        figure.add_annotation(
            x=x,
            y=y,
            xref="paper",
            yref="paper",
            text=(
                f"{label}<br>"
                f"<b>{float(quadrant_share[label]):.1f}%</b> of records"
            ),
            showarrow=False,
            xanchor=anchor,
            align="left" if anchor == "left" else "right",
            font={
                "size": 12,
                "color": (
                    BRAND_COLORS["terracotta"]
                    if negative_side
                    else BRAND_COLORS["dark_teal"]
                ),
            },
            bgcolor=(
                "rgba(255,221,210,0.90)"
                if negative_side
                else "rgba(237,246,249,0.90)"
            ),
            borderpad=4,
        )

    figure.add_annotation(
        x=0.01,
        y=0.515,
        xref="paper",
        yref="paper",
        text="Selected forecast better ↑ · Last-week better ↓",
        showarrow=False,
        xanchor="left",
        font={
            "size": 11,
            "color": BRAND_COLORS["dark_teal"],
        },
        bgcolor="rgba(237,246,249,0.88)",
    )

    if np.isfinite(outcome_shares["tie"]) and outcome_shares["tie"] > 0:
        figure.add_annotation(
            x=0.99,
            y=0.515,
            xref="paper",
            yref="paper",
            text=f"Exact ties: {outcome_shares['tie']:.2f}%",
            showarrow=False,
            xanchor="right",
            font={
                "size": 10,
                "color": "#60767B",
            },
            bgcolor="rgba(255,255,255,0.90)",
        )

    figure = apply_branding(figure)
    figure.update_layout(
        title={
            "text": "Win–Miss Plane · scale-aware record-level forecast outcomes"
        },
        height=680,
        margin={
            "l": 70,
            "r": 25,
            "t": 92,
            "b": 70,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.04,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 11},
        },
        hovermode="closest",
        hoverlabel={"align": "left"},
    )
    figure.update_xaxes(
        type="log",
        title=(
            "Forecast error index · 100 = one mobility-measure × horizon MAE "
            "(log scale)"
        ),
        range=[log_low, log_high],
        dtick=1,
        gridcolor="rgba(0,109,119,0.12)",
    )
    figure.update_yaxes(
        title=(
            "Advantage vs Last-week baseline · positive = selected forecast better"
        ),
        range=[-y_limit, y_limit],
        zeroline=False,
    )

    return figure


# ---------------------------------------------------------------------
# Candidate mining
# ---------------------------------------------------------------------


def diverse_cases(
    frame: pd.DataFrame,
    *,
    sort_column: str,
    ascending: bool,
    n: int = CASE_TABLE_ROWS,
) -> pd.DataFrame:
    """Keep top cases diverse enough for editorial comparison."""
    if frame.empty:
        return frame

    chosen = []
    metric_counts: dict[str, int] = {}
    seen_zones: set[tuple[str, int]] = set()

    ordered = frame.sort_values(sort_column, ascending=ascending)
    for row in ordered.itertuples(index=False):
        metric = str(row.metric)
        zone_key = (metric, int(row.taxi_zone_id))

        if metric_counts.get(metric, 0) >= 2 or zone_key in seen_zones:
            continue

        chosen.append(row._asdict())
        metric_counts[metric] = metric_counts.get(metric, 0) + 1
        seen_zones.add(zone_key)

        if len(chosen) >= n:
            break

    return pd.DataFrame(chosen)


def candidate_shortlists(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Mine three intentionally different story types from the full filtered surface."""
    clean_win_pool = frame.loc[
        frame["benchmark_advantage_index"].gt(0)
        & ~frame["severe_error"]
        & frame["system_row_error_index"].le(100)
    ].copy()

    hard_helpful_pool = frame.loc[
        frame["benchmark_advantage_index"].gt(0)
        & frame["severe_error"]
    ].copy()

    clear_miss_pool = frame.loc[
        frame["benchmark_advantage_index"].lt(0)
        & frame["severe_error"]
    ].copy()

    return {
        "Clean wins": diverse_cases(
            clean_win_pool,
            sort_column="benchmark_advantage_index",
            ascending=False,
        ),
        "Hard but helpful": diverse_cases(
            hard_helpful_pool,
            sort_column="benchmark_advantage_index",
            ascending=False,
        ),
        "Clear misses": diverse_cases(
            clear_miss_pool,
            sort_column="system_row_error_index",
            ascending=False,
        ),
    }


def candidate_display(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a compact reader-facing shortlist table."""
    if frame.empty:
        return pd.DataFrame()

    display = frame.copy()
    display["Mobility measure"] = display["metric"].map(METRIC_LABELS).fillna(display["metric"])
    display["h"] = display["horizon"].astype(int)
    display["Date"] = display["target_date"].dt.strftime("%b %d, %Y")
    display["Time"] = display["target_temporal_bucket"].map(temporal_bucket_label)
    display["Taxi Zone"] = display["zone"].astype(str) + " · " + display["borough"].astype(str)
    display["Observed"] = [
        format_native(metric, value)
        for metric, value in zip(display["metric"], display["actual"])
    ]
    display["Selected forecast"] = [
        format_native(metric, value)
        for metric, value in zip(display["metric"], display["champion_prediction"])
    ]
    display["Last-week"] = [
        format_native(metric, value)
        for metric, value in zip(display["metric"], display["benchmark_prediction"])
    ]
    display["Error index"] = display["system_row_error_index"].map(format_number)
    display["Baseline advantage"] = display["benchmark_advantage_index"].map(
        lambda value: format_number(value, 3)
    )
    display["Diagnostic context"] = display["failure_combination"].astype(str)

    return display[
        [
            "Mobility measure",
            "h",
            "Date",
            "Time",
            "Taxi Zone",
            "Observed",
            "Selected forecast",
            "Last-week",
            "Error index",
            "Baseline advantage",
            "Diagnostic context",
        ]
    ].reset_index(drop=True)


# ---------------------------------------------------------------------
# Win–Miss Bands
# ---------------------------------------------------------------------


def hero_case(case_frame: pd.DataFrame) -> pd.Series | None:
    """Return the lead curated case from a shortlist, if available."""
    if case_frame.empty:
        return None
    return case_frame.iloc[0]


def curated_case_title(label: str, case: pd.Series) -> str:
    """Human-readable curated-case heading."""
    return (
        f"{label} · {case['zone']} ({case['borough']}) · "
        f"{metric_label(case['metric'])} · h={int(case['horizon'])}"
    )


def curated_case_why_it_matters(label: str) -> str:
    """Give each outcome example one compact reader-facing lesson."""
    if label == "Clean win":
        return (
            "This is the easiest outcome to trust: the selected forecast stayed relatively close "
            "to what happened and also improved on Last-week."
        )
    if label == "Hard but helpful":
        return (
            "This is the key reminder for the page: a forecast can miss badly and still be useful "
            "when Last-week misses by even more."
        )
    return (
        "This is the clearest caution case: the selected forecast missed badly and Last-week was "
        "closer, so both absolute accuracy and benchmark-relative usefulness broke down."
    )


def diagnostic_case_story_label(case: pd.Series) -> str:
    """Translate the frozen diagnostic category into a short case-study label."""
    category = str(case["failure_category"]).lower()

    if "structural" in category:
        return "Structural weakness"
    if "shared" in category or "shock" in category:
        return "Shared shock"
    if "sparse" in category or "zero" in category or "demand" in category:
        return "Sparse / zero demand"
    return str(case["failure_category"])


def diagnostic_case_why_it_matters(case: pd.Series) -> str:
    """Give each diagnostic example one compact mechanism-level lesson."""
    label = diagnostic_case_story_label(case)

    if label == "Structural weakness":
        return (
            "Some mobility-measure × horizon combinations are persistently harder than others, so a large "
            "miss does not always require a one-off local disruption."
        )
    if label == "Shared shock":
        return (
            "A broad disruption can make many forecasts struggle at the same time, which points to "
            "a systemwide shock rather than an isolated neighborhood failure."
        )
    if label == "Sparse / zero demand":
        return (
            "Very low local activity leaves less stable history to learn from, making an individual "
            "neighborhood forecast unusually fragile."
        )
    return (
        "This record shows a distinct condition under which the selected forecast deserves extra "
        "scrutiny."
    )


def six_case_synthesis(
    curated_cases: dict[str, pd.Series | None],
    failure_cases: pd.DataFrame,
) -> str:
    """Connect the six hero cases without creating another takeaway box."""
    outcome_count = sum(case is not None for case in curated_cases.values())
    diagnostic_count = int(len(failure_cases))

    if outcome_count < 3 or diagnostic_count < 3:
        return (
            "Together, these examples separate two questions that matter throughout the page: how "
            "large the forecast miss was, and whether the selected forecast still improved on "
            "Last-week. They also show that difficult forecasts can arise for different reasons, "
            "so caution should be tied to the evidence surrounding each forecast rather than to one "
            "universal failure rule."
        )

    return (
        "The first three cases separate **forecast accuracy** from **forecast usefulness**: a clean "
        "win does well on both, a hard-but-helpful forecast misses badly but still improves on "
        "Last-week, and a clear miss loses on both. The next three show that difficult forecasts "
        "also arise for different reasons — a persistently difficult mobility-measure × horizon problem, a "
        "shared shock affecting many forecasts at once, or sparse local demand. Together, the six "
        "cases provide the vocabulary for the rest of the page: judge the size of the miss, compare "
        "it with Last-week, and then look at the conditions surrounding the forecast."
    )


def zone_strip_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize each Taxi Zone under one mobility measure × horizon strip scope.

    The strip must compare like with like, so the page requires one target and
    one horizon before ranking zones. Rates remain support-aware because they are
    calculated from all currently visible holdout records for each zone.
    """
    working = frame.copy()
    advantage = pd.to_numeric(
        working["benchmark_advantage_index"],
        errors="coerce",
    )

    working["model_win"] = advantage.gt(0)
    working["last_week_win"] = advantage.lt(0)
    working["comparison_tie"] = advantage.eq(0)
    working["clear_miss"] = (
        working["severe_error"]
        & working["last_week_win"]
    )
    working["hard_helpful"] = (
        working["severe_error"]
        & working["model_win"]
    )

    summary = (
        working.groupby(
            ["taxi_zone_id", "zone", "borough"],
            as_index=False,
            observed=True,
        )
        .agg(
            forecast_rows=("system_row_error_index", "size"),
            median_error_index=("system_row_error_index", "median"),
            mean_error_index=("system_row_error_index", "mean"),
            severe_error_rate=("severe_error", "mean"),
            model_win_rate=("model_win", "mean"),
            last_week_win_rate=("last_week_win", "mean"),
            tie_rate=("comparison_tie", "mean"),
            clear_miss_rate=("clear_miss", "mean"),
            hard_helpful_rate=("hard_helpful", "mean"),
        )
    )
    return summary


def choose_strip_zones(
    summary: pd.DataFrame,
    *,
    sort_mode: str,
    zone_count: int,
) -> pd.DataFrame:
    """Choose the Taxi Zones shown as strip rows without inventing a composite score."""
    if summary.empty:
        return summary

    sort_specs = {
        "Clear-miss rate": (
            ["clear_miss_rate", "severe_error_rate", "median_error_index", "zone"],
            [False, False, False, True],
        ),
        "Severe-error rate": (
            ["severe_error_rate", "clear_miss_rate", "median_error_index", "zone"],
            [False, False, False, True],
        ),
        "Hard-but-helpful rate": (
            ["hard_helpful_rate", "severe_error_rate", "median_error_index", "zone"],
            [False, False, False, True],
        ),
        "Median error severity": (
            ["median_error_index", "severe_error_rate", "zone"],
            [False, False, True],
        ),
        "Alphabetical": (
            ["zone", "borough"],
            [True, True],
        ),
    }
    columns, ascending = sort_specs[sort_mode]

    return (
        summary.sort_values(columns, ascending=ascending)
        .head(int(zone_count))
        .reset_index(drop=True)
    )


def daily_strip_representatives(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Pick one exact forecast record per Taxi Zone × date for the strip.

    When the filter still contains several temporal buckets on a date, the strip
    shows that date's largest champion error. That choice makes the visual a
    caution finder while preserving an exact underlying record for hover/click.
    """
    working = frame.copy()
    working["system_row_error_index"] = pd.to_numeric(
        working["system_row_error_index"],
        errors="coerce",
    )
    working = working.loc[
        np.isfinite(working["system_row_error_index"])
        & working["system_row_error_index"].ge(0)
    ].copy()

    if working.empty:
        return working

    return (
        working.sort_values(
            [
                "taxi_zone_id",
                "target_date",
                "system_row_error_index",
                "benchmark_advantage_index",
            ],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(["taxi_zone_id", "target_date"], keep="first")
        .sort_values(["taxi_zone_id", "target_date"])
        .reset_index(drop=True)
    )


def record_key_from_row(row: pd.Series) -> tuple[str, int, int, str, str]:
    """Create the stable key used to carry a clicked strip point into the case explorer."""
    return (
        str(row["metric"]),
        int(row["horizon"]),
        int(row["taxi_zone_id"]),
        pd.Timestamp(row["target_date"]).strftime("%Y-%m-%d"),
        str(row["target_temporal_bucket"]),
    )


def record_from_key(
    frame: pd.DataFrame,
    key: tuple[str, int, int, str, str],
) -> pd.Series | None:
    """Recover one exact holdout record from a strip-selection key."""
    metric, horizon, zone_id, date_text, bucket = key
    target_date = pd.Timestamp(date_text)

    match = frame.loc[
        frame["metric"].eq(metric)
        & frame["horizon"].eq(horizon)
        & frame["taxi_zone_id"].eq(zone_id)
        & frame["target_date"].eq(target_date)
        & frame["target_temporal_bucket"].eq(bucket)
    ]

    if match.empty:
        return None

    return match.iloc[0]


def record_key_from_plotly_event(
    event: object,
) -> tuple[str, int, int, str, str] | None:
    """Return the exact forecast-record key from a clicked strip point."""
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
    if customdata is None or len(customdata) < 5:
        return None

    try:
        return (
            str(customdata[0]),
            int(customdata[1]),
            int(customdata[2]),
            str(customdata[3]),
            str(customdata[4]),
        )
    except (TypeError, ValueError):
        return None


def strip_zone_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Format the zone-ordering evidence without exposing long floating-point tails."""
    if summary.empty:
        return pd.DataFrame()

    shown = summary.copy()
    shown["Taxi Zone"] = shown["zone"].astype(str) + " · " + shown["borough"].astype(str)
    shown["Severe-error rate"] = shown["severe_error_rate"].map(lambda value: f"{100 * value:.3f}%")
    shown["Clear-miss rate"] = shown["clear_miss_rate"].map(lambda value: f"{100 * value:.3f}%")
    shown["Hard-but-helpful rate"] = shown["hard_helpful_rate"].map(
        lambda value: f"{100 * value:.3f}%"
    )
    shown["Model-win rate"] = shown["model_win_rate"].map(
        lambda value: f"{100 * value:.3f}%"
    )
    shown["Tie rate"] = shown["tie_rate"].map(
        lambda value: f"{100 * value:.3f}%"
    )
    shown["Median error index"] = shown["median_error_index"].map(format_number)
    shown["Records"] = shown["forecast_rows"].map(lambda value: f"{int(value):,}")

    return shown[
        [
            "Taxi Zone",
            "Severe-error rate",
            "Clear-miss rate",
            "Hard-but-helpful rate",
            "Model-win rate",
            "Tie rate",
            "Median error index",
            "Records",
        ]
    ].reset_index(drop=True)


def build_forecast_error_strips(
    all_daily: pd.DataFrame,
    selected_zones: pd.DataFrame,
    *,
    title: str,
    highlight_key: tuple[str, int, int, str, str] | None = None,
) -> go.Figure:
    """
    Draw folded Win–Miss Bands across Taxi Zones and holdout time.

    Sign answers who won: above the center means the selected forecast beat
    Last-week; below means Last-week was closer. Band depth answers by how much:
    the absolute baseline-advantage index is folded into repeated layers.

    Forecast-error severity remains available in hover and in the linked record
    explorer, so the band can focus on comparative usefulness without losing
    the underlying forecasting difficulty.
    """
    figure = go.Figure()
    if all_daily.empty or selected_zones.empty:
        return figure

    zone_ids = selected_zones["taxi_zone_id"].astype(int).tolist()
    plot = all_daily.loc[all_daily["taxi_zone_id"].isin(zone_ids)].copy()
    if plot.empty:
        return figure

    advantage = pd.to_numeric(
        all_daily["benchmark_advantage_index"],
        errors="coerce",
    ).abs()
    advantage = advantage[np.isfinite(advantage)]
    advantage_cap = max(float(advantage.quantile(0.995)), 100.0)

    plot["advantage_for_band"] = (
        pd.to_numeric(
            plot["benchmark_advantage_index"],
            errors="coerce",
        )
        .abs()
        .clip(lower=0, upper=advantage_cap)
    )
    plot["band_strength"] = (
        np.log1p(plot["advantage_for_band"])
        / np.log1p(advantage_cap)
    ).clip(lower=0, upper=1)
    band_advantage = pd.to_numeric(
        plot["benchmark_advantage_index"],
        errors="coerce",
    )
    plot["forecast_wins"] = band_advantage.gt(0)
    plot["last_week_wins"] = band_advantage.lt(0)
    plot["comparison_tie"] = band_advantage.eq(0)

    band_count = 4
    half_height = 0.39

    zone_position = {
        int(zone_id): len(zone_ids) - 1 - index
        for index, zone_id in enumerate(zone_ids)
    }

    min_date = pd.Timestamp(plot["target_date"].min())
    max_date = pd.Timestamp(plot["target_date"].max())

    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="Selected forecast better than Last-week",
            line={"width": 10, "color": BRAND_COLORS["seafoam"]},
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="Last-week better than selected forecast",
            line={"width": 10, "color": BRAND_COLORS["terracotta"]},
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="Exact tie",
            marker={
                "size": 7,
                "symbol": "diamond",
                "color": "#7A878C",
            },
            hoverinfo="skip",
        )
    )

    point_frames = []

    for zone_id in zone_ids:
        zone = plot.loc[plot["taxi_zone_id"].eq(zone_id)].sort_values("target_date")
        if zone.empty:
            continue

        base = float(zone_position[zone_id])
        date_strings = zone["target_date"].dt.strftime("%Y-%m-%d").tolist()
        strength = zone["band_strength"].to_numpy(dtype=float)
        forecast_wins = zone["forecast_wins"].to_numpy(dtype=bool)

        for band_index in range(band_count):
            band_fraction = np.clip(
                strength * band_count - band_index,
                0.0,
                1.0,
            )
            band_height = band_fraction * half_height

            positive_height = np.where(
                forecast_wins,
                band_height,
                0.0,
            )
            last_week_wins = zone["last_week_wins"].to_numpy(dtype=bool)
            negative_height = np.where(
                last_week_wins,
                band_height,
                0.0,
            )

            band_opacity = 0.22 + 0.16 * band_index
            baseline = np.full(len(zone), base, dtype=float)

            figure.add_trace(
                go.Scatter(
                    x=date_strings + date_strings[::-1],
                    y=np.concatenate(
                        [base + positive_height, baseline[::-1]]
                    ),
                    mode="lines",
                    fill="toself",
                    fillcolor=BRAND_COLORS["seafoam"],
                    line={"width": 0, "shape": "hv"},
                    opacity=band_opacity,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=date_strings + date_strings[::-1],
                    y=np.concatenate(
                        [base - negative_height, baseline[::-1]]
                    ),
                    mode="lines",
                    fill="toself",
                    fillcolor=BRAND_COLORS["terracotta"],
                    line={"width": 0, "shape": "hv"},
                    opacity=band_opacity,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        zone_points = zone.copy()
        zone_points["strip_y"] = base + np.select(
            [
                zone_points["forecast_wins"],
                zone_points["last_week_wins"],
            ],
            [
                half_height * 0.72,
                -half_height * 0.72,
            ],
            default=0.0,
        )
        point_frames.append(zone_points)

        figure.add_trace(
            go.Scatter(
                x=[
                    (min_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    (max_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                ],
                y=[base, base],
                mode="lines",
                line={
                    "width": 0.5,
                    "color": "rgba(0,109,119,0.20)",
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

    points = (
        pd.concat(point_frames, ignore_index=True)
        if point_frames
        else pd.DataFrame()
    )

    if points.empty:
        raise ValueError(
            "Win–Miss Bands produced no selectable points after zone/date "
            "preparation. Check the band scope and target_date parsing."
        )

    customdata = np.column_stack(
        [
            points["metric"].astype(str),
            points["horizon"].astype(int),
            points["taxi_zone_id"].astype(int),
            points["target_date"].dt.strftime("%Y-%m-%d"),
            points["target_temporal_bucket"].astype(str),
            points["zone"].astype(str),
            points["borough"].astype(str),
            points["target_date"].dt.strftime("%b %d, %Y"),
            points["target_temporal_bucket"].map(temporal_bucket_label),
            points["actual"].map(lambda value: format_number(value, 3)),
            points["champion_prediction"].map(lambda value: format_number(value, 3)),
            points["benchmark_prediction"].map(lambda value: format_number(value, 3)),
            points["system_row_error_index"].map(lambda value: format_number(value, 3)),
            points["benchmark_advantage_index"].map(lambda value: format_number(value, 3)),
            points["failure_combination"].astype(str),
            np.where(points["severe_error"], "Yes", "No"),
        ]
    )

    tie_points = points.loc[
        points["comparison_tie"].fillna(False)
    ].copy()

    if not tie_points.empty:
        figure.add_trace(
            go.Scattergl(
                x=tie_points["target_date"].dt.strftime("%Y-%m-%d"),
                y=tie_points["strip_y"],
                mode="markers",
                name="Exact tie",
                marker={
                    "size": 6,
                    "symbol": "diamond",
                    "color": "#7A878C",
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

    figure.add_trace(
        go.Scattergl(
            x=points["target_date"].dt.strftime("%Y-%m-%d"),
            y=points["strip_y"],
            mode="markers",
            name="Selectable forecast record",
            marker={
                "size": 13,
                "color": "rgba(0,0,0,0.001)",
                "line": {"width": 0},
            },
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[5]}</b> · %{customdata[6]}<br>"
                "%{customdata[7]} · %{customdata[8]}<br><br>"
                "Observed: %{customdata[9]}<br>"
                "Selected forecast: %{customdata[10]}<br>"
                "Last-week: %{customdata[11]}<br>"
                "Forecast error index: %{customdata[12]}<br>"
                "Baseline advantage: %{customdata[13]}<br>"
                "Severe error: %{customdata[15]}<br>"
                "Context: %{customdata[14]}<br><br>"
                "<b>Click to open this record below.</b>"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    if highlight_key is not None:
        metric, horizon, zone_id, date_text, bucket = highlight_key
        selected = points.loc[
            points["metric"].eq(metric)
            & points["horizon"].eq(horizon)
            & points["taxi_zone_id"].eq(zone_id)
            & points["target_date"].eq(pd.Timestamp(date_text))
            & points["target_temporal_bucket"].eq(bucket)
        ]
        if not selected.empty:
            chosen = selected.iloc[0]
            figure.add_trace(
                go.Scatter(
                    x=[
                        pd.Timestamp(chosen["target_date"]).strftime(
                            "%Y-%m-%d"
                        )
                    ],
                    y=[chosen["strip_y"]],
                    mode="markers",
                    name="Selected record",
                    marker={
                        "size": 16,
                        "symbol": "diamond",
                        "color": "white",
                        "line": {
                            "width": 3.2,
                            "color": BRAND_COLORS["dark_teal"],
                        },
                    },
                    hovertemplate="Selected record<extra></extra>",
                    showlegend=True,
                )
            )

    labels = (
        selected_zones.set_index("taxi_zone_id")
        .loc[zone_ids]
        .apply(
            lambda row: f"{row['zone']} · {row['borough']}",
            axis=1,
        )
        .tolist()
    )
    tickvals = [zone_position[zone_id] for zone_id in zone_ids]

    figure = apply_branding(figure)
    figure.update_layout(
        title={"text": title},
        height=max(560, 185 + 40 * len(zone_ids)),
        margin={"l": 220, "r": 35, "t": 82, "b": 120},
        hovermode="closest",
        hoverlabel={"align": "left"},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.14,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 11},
        },
    )
    figure.update_xaxes(
        type="date",
        title="Forecast target date",
        title_standoff=5,
        range=[
            (min_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            (max_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        ],
        dtick="M1",
        tickformat="%b\n%Y",
        gridcolor="rgba(0,109,119,0.08)",
    )
    figure.update_yaxes(
        title="Taxi Zone",
        tickmode="array",
        tickvals=tickvals,
        ticktext=labels,
        range=[-0.7, len(zone_ids) - 0.3],
        showgrid=False,
        zeroline=False,
    )
    return figure


def default_strip_record(strip_points: pd.DataFrame) -> pd.Series | None:
    """Choose a useful default case before the reader clicks a strip point."""
    if strip_points.empty:
        return None

    clear_misses = strip_points.loc[
        strip_points["severe_error"]
        & strip_points["benchmark_advantage_index"].lt(0)
    ]
    if not clear_misses.empty:
        return clear_misses.nlargest(1, "system_row_error_index").iloc[0]

    return strip_points.nlargest(1, "system_row_error_index").iloc[0]


def build_record_context_chart(
    case: pd.Series,
    records: pd.DataFrame,
) -> go.Figure:
    """Show nearby held-out evidence for any record selected from the error strips."""
    selected = records.loc[
        records["metric"].eq(case["metric"])
        & records["horizon"].eq(case["horizon"])
        & records["taxi_zone_id"].eq(case["taxi_zone_id"])
        & records["target_temporal_bucket"].eq(case["target_temporal_bucket"])
    ].copy()

    target_date = pd.Timestamp(case["target_date"])
    selected = selected.loc[
        selected["target_date"].between(
            target_date - pd.Timedelta(days=28),
            target_date + pd.Timedelta(days=28),
            inclusive="both",
        )
    ].sort_values("target_date")

    figure = go.Figure()
    if selected.empty:
        return figure

    figure.add_trace(
        go.Scatter(
            x=selected["target_date"],
            y=selected["actual"],
            mode="lines+markers",
            name="Observed",
            line={"width": 2.6, "color": "#263238"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=selected["target_date"],
            y=selected["champion_prediction"],
            mode="lines+markers",
            name="Selected forecast",
            line={"width": 2.2, "color": BRAND_COLORS["dark_teal"]},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=selected["target_date"],
            y=selected["benchmark_prediction"],
            mode="lines",
            name="Last-week baseline",
            line={"width": 1.8, "dash": "dot", "color": "#7A878C"},
        )
    )
    figure.add_vline(
        x=target_date.timestamp() * 1000,
        line_width=1.5,
        line_dash="dot",
        line_color=BRAND_COLORS["terracotta"],
    )

    figure = apply_branding(figure)
    figure.update_layout(
        title={
            "text": (
                f"{case['zone']} · {metric_label(case['metric'])} · "
                f"h={int(case['horizon'])} · "
                f"{temporal_bucket_label(case['target_temporal_bucket'])}"
            )
        },
        height=500,
        margin={"l": 65, "r": 25, "t": 75, "b": 105},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.16,
            "xanchor": "center",
            "x": 0.5,
        },
    )
    figure.update_xaxes(
        title="Forecast target date",
        title_standoff=5,
    )
    figure.update_yaxes(
        title=f"{metric_label(case['metric'])} ({metric_unit(case['metric'])})"
    )
    return figure


def selected_record_takeaway(case: pd.Series) -> str:
    """Translate one clicked strip record into exact Win–Miss language."""
    metric = str(case["metric"])
    champion_error = float(case["absolute_error"])
    baseline_error = float(case["benchmark_absolute_error"])
    advantage = float(case["benchmark_advantage_index"])
    severe = bool(case["severe_error"])

    if advantage > 0:
        comparison = (
            f"The selected forecast was **{format_native(metric, baseline_error - champion_error)} "
            f"{metric_unit(metric)} closer** than Last-week."
        )
        lens = "**Hard but helpful**" if severe else "**Forecast win · ordinary error**"
    elif advantage < 0:
        comparison = (
            f"Last-week was **{format_native(metric, champion_error - baseline_error)} "
            f"{metric_unit(metric)} closer** than the selected forecast."
        )
        lens = "**Clear miss**" if severe else "**Last-week better · ordinary error**"
    else:
        comparison = (
            "The selected forecast and Last-week had **exactly the same absolute error**."
        )
        lens = "**Severe error · exact tie**" if severe else "**Exact tie · ordinary error**"

    return (
        f"This is a {lens} case. Observed {metric_label(metric).lower()} was "
        f"**{format_native(metric, case['actual'])} {metric_unit(metric)}**; the selected "
        f"forecast was **{format_native(metric, case['champion_prediction'])}** and "
        f"Last-week was **{format_native(metric, case['benchmark_prediction'])}**. "
        f"{comparison} Its forecast-error index is "
        f"**{format_number(case['system_row_error_index'])}** "
        "(100 = one mobility-measure × horizon MAE). Diagnostic context: "
        f"**{case['failure_combination']}**."
    )


# ---------------------------------------------------------------------
# Formal 4.6.2 failure cases
# ---------------------------------------------------------------------


def build_failure_context_chart(
    case: pd.Series,
    context: pd.DataFrame,
) -> go.Figure:
    """Show the local time window around one formally selected failure record."""
    selected = context.loc[context["candidate_id"].eq(case["candidate_id"])].copy()
    selected = selected.sort_values("target_date")

    figure = go.Figure()
    if selected.empty:
        return figure

    figure.add_trace(
        go.Scatter(
            x=selected["target_date"],
            y=selected["actual"],
            mode="lines+markers",
            name="Observed",
            line={"width": 2.6, "color": "#263238"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=selected["target_date"],
            y=selected["system_prediction"],
            mode="lines+markers",
            name="Selected forecast",
            line={"width": 2.2, "color": BRAND_COLORS["dark_teal"]},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=selected["target_date"],
            y=selected["benchmark_prediction"],
            mode="lines",
            name="Last-week baseline",
            line={"width": 1.8, "dash": "dot", "color": "#7A878C"},
        )
    )

    figure.add_vline(
        x=pd.Timestamp(case["target_date"]).timestamp() * 1000,
        line_width=1.5,
        line_dash="dot",
        line_color=BRAND_COLORS["terracotta"],
    )

    figure = apply_branding(figure)
    figure.update_layout(
        title={
            "text": (
                f"{case['zone']} · {metric_label(case['metric'])} · "
                f"h={int(case['horizon'])} · {case['failure_category']}"
            )
        },
        height=500,
        margin={"l": 65, "r": 25, "t": 75, "b": 105},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.16,
            "xanchor": "center",
            "x": 0.5,
        },
    )
    figure.update_xaxes(
        title="Forecast target date",
        title_standoff=5,
    )
    figure.update_yaxes(title=f"{metric_label(case['metric'])} ({metric_unit(case['metric'])})")
    return figure


# ---------------------------------------------------------------------
# Failure-mechanism burden
# ---------------------------------------------------------------------


def build_mechanism_chart(mechanisms: pd.DataFrame) -> go.Figure:
    """Compare how common each diagnostic condition is with how much error it carries."""
    figure = go.Figure()

    specs = [
        ("System forecast share (%)", "Forecast share", BRAND_COLORS["seafoam"]),
        ("System error burden (%)", "Error burden", BRAND_COLORS["terracotta"]),
        ("Severe-error capture (%)", "Severe-error capture", BRAND_COLORS["dark_teal"]),
    ]

    for column, label, color in specs:
        figure.add_trace(
            go.Bar(
                y=mechanisms["Short label"],
                x=mechanisms[column],
                orientation="h",
                name=label,
                marker={"color": color},
                text=mechanisms[column].map(lambda value: f"{float(value):.1f}%"),
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    f"{label}: %{{x:.3f}}%"
                    "<extra></extra>"
                ),
            )
        )

    figure = apply_branding(figure)
    figure.update_layout(
        title={"text": "Which diagnostic conditions carry disproportionate forecast error?"},
        height=430,
        margin={"l": 170, "r": 70, "t": 70, "b": 105},
        barmode="group",
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.20,
            "xanchor": "center",
            "x": 0.5,
        },
    )
    figure.update_xaxes(title="Share of forecasts or error (%)")
    figure.update_yaxes(
        title="Diagnostic condition",
        tickmode="array",
        tickvals=mechanisms["Short label"].tolist(),
        ticktext=mechanisms["Short label"].tolist(),
    )
    return figure


def mechanism_takeaway(mechanisms: pd.DataFrame) -> str:
    """Summarize the strongest prevalence-vs-burden contrasts."""
    burden_leader = mechanisms.sort_values("System error burden (%)", ascending=False).iloc[0]
    concentration_leader = mechanisms.sort_values("Burden / forecast share (×)", ascending=False).iloc[0]

    return (
        f"**{burden_leader['Short label']}** appears in "
        f"**{float(burden_leader['System forecast share (%)']):.1f}%** of applicable forecasts "
        f"but accounts for **{float(burden_leader['System error burden (%)']):.1f}%** of system "
        f"error burden and captures **{float(burden_leader['Severe-error capture (%)']):.1f}%** "
        f"of severe errors. The most concentrated condition is **{concentration_leader['Short label']}**, "
        f"whose error burden is **{float(concentration_leader['Burden / forecast share (×)']):.2f}×** "
        "its forecast share."
    )


# ---------------------------------------------------------------------
# Scout summary text
# ---------------------------------------------------------------------


def top_case_line(label: str, frame: pd.DataFrame) -> str:
    """Return one paste-friendly line describing the first candidate in a shortlist."""
    if frame.empty:
        return f"{label}: none under the current filters"

    row = frame.iloc[0]
    return (
        f"{label}: {metric_label(row['metric'])} h={int(row['horizon'])} · "
        f"{row['zone']} ({row['borough']}) · {pd.Timestamp(row['target_date']).date()} · "
        f"error_index={float(row['system_row_error_index']):.3f} · "
        f"baseline_advantage_index={float(row['benchmark_advantage_index']):.3f} · "
        f"context={row['failure_combination']}"
    )


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------

inject_app_css()

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 1.45rem !important;
        line-height: 1.15 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.84rem !important;
        line-height: 1.2 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.caption(PAGE_CAPTION)
st.title(PAGE_TITLE)
st.markdown(
    """
Forecast accuracy is only part of the story. A forecast can land far from what actually
happened and still be useful if it is closer than simply repeating the previous week's
value. The reverse can happen too: a fairly small miss can still lose to Last-week.

For each mobility outcome and forecast horizon, the **selected forecast** is the model
chosen before this final test period. The views below compare its own error with how much
it improved — or failed to improve — on the **Last-week baseline**.
"""
)

with st.spinner("Loading the frozen final-holdout forecast diagnostics…"):
    (
        holdout_summary,
        scatter_records,
        curated_cases,
        failure_cases,
        failure_context,
        mechanisms,
    ) = load_page_data()

# ------------------------------------------------------------------
# At a Glance
# ------------------------------------------------------------------

st.subheader("At a Glance")

full_stats = summary_statistics(holdout_summary)

job_count = int(
    holdout_summary[["metric", "horizon"]]
    .drop_duplicates()
    .shape[0]
)

model_win_share = full_stats["model_win_share"]
last_week_win_share = full_stats["last_week_win_share"]
tie_share = full_stats["tie_share"]
severe_share = full_stats["severe_share"]
severe_helpful_within_severe = full_stats["severe_helpful_share"]

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(
    "Holdout records",
    f"{int(full_stats['forecast_rows']):,}",
)
m2.metric("Measure × horizon pairs", f"{job_count}")
m3.metric("Selected forecast closer", f"{model_win_share:.1f}%")
m4.metric("Last-week closer", f"{last_week_win_share:.1f}%")
m5.metric(
    "Severe errors still helpful",
    (
        f"{severe_helpful_within_severe:.1f}%"
        if np.isfinite(severe_helpful_within_severe)
        else "—"
    ),
)

tie_copy = (
    f" Exact ties account for {tie_share:.2f}% of supported comparisons."
    if np.isfinite(tie_share) and tie_share > 0
    else ""
)
st.caption(
    "All results use the untouched Jan 5–Mar 31, 2026 final holdout. "
    f"Severe errors make up {severe_share:.1f}% of records and are defined as "
    "the highest-error 10% within each mobility measure × horizon."
    + tie_copy
)

# ------------------------------------------------------------------
# Static hero
# ------------------------------------------------------------------

st.divider()
st.subheader("1. A big miss can still be useful")
st.write(
    "Each point answers two questions at once: **how far was the selected forecast from "
    "what actually happened, and was it still closer than Last-week?** Points to the "
    "right are larger forecast errors; points above zero still beat the Last-week baseline."
)

st.caption(
    "All five mobility outcomes · h=1/2/5 · weekdays and weekends · all dayparts · "
    "Jan 5–Mar 31, 2026"
)

hero_plane = build_win_miss_plane(
    scatter_records,
    exact_summary=holdout_summary,
    pre_sampled=True,
)
st.plotly_chart(
    hero_plane,
    width="stretch",
    config=PLOT_CONFIG,
    key="raw20_hero_plane",
)

hero_tie_sentence = (
    f" Exact ties account for **{tie_share:.2f}%**."
    if np.isfinite(tie_share) and tie_share > 0
    else ""
)

render_chart_insight(
    "Even some of the forecasting system's **largest misses still beat simply "
    "repeating the previous week's value**. Among the severe errors shown here, "
    f"**{severe_helpful_within_severe:.1f}%** are still closer to what actually "
    "happened than the Last-week baseline. Across the full final holdout, the "
    f"selected forecast is closer on **{model_win_share:.1f}%** of supported "
    f"comparisons, versus **{last_week_win_share:.1f}%** for Last-week."
    f"{hero_tie_sentence} The size of a forecast error and whether the forecast "
    "added value are therefore two different questions."
)

with st.expander("How to read the Win–Miss Plane", expanded=False):
    st.markdown(
        """
- **Forecast error index** = selected-forecast absolute error divided by that mobility-measure × horizon MAE, ×100. **100 means one job-level MAE.** Left of 100 is a below-average-sized miss for that job; right of 100 is an above-average-sized miss.
- **Baseline advantage index** = Last-week absolute error minus selected-forecast absolute error, divided by the same job-level MAE, ×100. **Positive means the selected forecast was closer; negative means Last-week was closer.**
- **The four quadrant colors match the visible x=100 and y=0 boundaries.** Exact ties sit on the y=0 center line in neutral gray and are not counted as either side winning.
- **Severe error is separate:** it marks the **highest-error 10% within each mobility measure × horizon**. A record can therefore sit right of 100 without being severe.
- The dots are sampled so the chart stays responsive. The quadrant percentages use the **full supported population**, including perfect zero-error forecasts and exact ties. Zero-error forecasts are drawn at a tiny left-edge offset because a logarithmic axis cannot display zero.
- The two axes share part of the same error calculation, so the plane's shape is partly mathematical structure rather than an ordinary correlation pattern.
"""
    )

# ------------------------------------------------------------------
# Six-case hero library
# ------------------------------------------------------------------

st.markdown("### 2. Six forecasts worth understanding")
st.write(
    "The Win–Miss Plane shows the overall pattern. These six records make that pattern concrete. "
    "The first three show **how a forecast can turn out**; the next three show **why a forecast can "
    "become difficult**."
)

# The first three examples teach the outcome vocabulary used throughout
# the page. They are selected from the complete final holdout rather than
# from the sampled hero scatter.
st.markdown("#### How forecasts can turn out")
st.write(
    "These examples separate the size of the selected forecast's miss from whether it still "
    "improved on Last-week."
)

hero_shortlists = {
    group: (
        curated_cases.loc[
            curated_cases["curated_group"].eq(group)
        ]
        .sort_values("curated_rank")
        .reset_index(drop=True)
    )
    for group in [
        "Clean wins",
        "Hard but helpful",
        "Clear misses",
    ]
}
curated_labels = [
    ("Clean win", "Clean wins"),
    ("Hard but helpful", "Hard but helpful"),
    ("Clear miss", "Clear misses"),
]
curated_lead_cases = {
    tab_label: hero_case(hero_shortlists[shortlist_key])
    for tab_label, shortlist_key in curated_labels
}
curated_tabs = st.tabs([label for label, _ in curated_labels])

for tab, (tab_label, shortlist_key) in zip(curated_tabs, curated_labels):
    with tab:
        lead_case = curated_lead_cases[tab_label]
        if lead_case is None:
            st.info("No final-holdout record is available for this outcome type.")
            continue

        st.markdown(f"#### {curated_case_title(tab_label, lead_case)}")

        c1, c2, c3, c4, c5 = st.columns(5)
        selected_metric = str(lead_case["metric"])
        selected_unit = metric_unit(selected_metric)
        c1.metric(
            "Observed",
            f"{format_native(selected_metric, lead_case['actual'])} {selected_unit}",
        )
        c2.metric(
            "Selected forecast",
            f"{format_native(selected_metric, lead_case['champion_prediction'])} {selected_unit}",
        )
        c3.metric(
            "Last-week",
            f"{format_native(selected_metric, lead_case['benchmark_prediction'])} {selected_unit}",
        )
        c4.metric(
            "Forecast error index",
            format_number(lead_case["system_row_error_index"]),
        )
        c5.metric(
            "Baseline advantage",
            format_number(lead_case["benchmark_advantage_index"]),
        )

        case_context = load_record_context_rows(
            str(lead_case["metric"]),
            int(lead_case["horizon"]),
            int(lead_case["taxi_zone_id"]),
            str(lead_case["target_temporal_bucket"]),
        )

        case_chart = build_record_context_chart(
            lead_case,
            case_context,
        )
        st.plotly_chart(
            case_chart,
            width="stretch",
            config=PLOT_CONFIG,
            key=f"raw20_curated_{shortlist_key.replace(' ', '_').lower()}",
        )
        render_chart_insight(
            curated_case_why_it_matters(tab_label)
        )

        with st.expander("See more outcomes like this", expanded=False):
            st.dataframe(
                candidate_display(hero_shortlists[shortlist_key]),
                hide_index=True,
                width="stretch",
                height=280,
            )

# The next three cases come from the frozen diagnostic analysis. Keeping
# them beside the outcome examples gives the reader the complete conceptual
# vocabulary before any free-form exploration begins.
st.markdown("#### Why forecasts can become difficult")
st.write(
    "A large miss can reflect very different forecasting problems. These three diagnostic "
    "examples show a persistent mobility measure × horizon weakness, a broad shared shock, and a sparse "
    "local-demand problem."
)

diagnostic_rows = [row for _, row in failure_cases.iterrows()]
diagnostic_tabs = st.tabs(
    [diagnostic_case_story_label(case) for case in diagnostic_rows]
)

for case_index, (tab, selected_case) in enumerate(
    zip(diagnostic_tabs, diagnostic_rows),
    start=1,
):
    with tab:
        story_label = diagnostic_case_story_label(selected_case)
        metric = str(selected_case["metric"])
        unit = metric_unit(metric)

        st.markdown(
            f"#### {story_label} · {selected_case['zone']} ({selected_case['borough']}) · "
            f"{metric_label(metric)} · h={int(selected_case['horizon'])}"
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(
            "Observed",
            f"{format_native(metric, selected_case['actual'])} {unit}",
        )
        c2.metric(
            "Selected forecast",
            f"{format_native(metric, selected_case['system_prediction'])} {unit}",
        )
        c3.metric(
            "Last-week",
            f"{format_native(metric, selected_case['benchmark_prediction'])} {unit}",
        )
        c4.metric(
            "Forecast abs. error",
            f"{format_native(metric, selected_case['absolute_error'])} {unit}",
        )
        c5.metric(
            "Last-week abs. error",
            f"{format_native(metric, selected_case['benchmark_absolute_error'])} {unit}",
        )

        diagnostic_chart = build_failure_context_chart(
            selected_case,
            failure_context,
        )
        st.plotly_chart(
            diagnostic_chart,
            width="stretch",
            config=PLOT_CONFIG,
            key=f"raw20_diagnostic_case_{case_index}",
        )
        render_chart_insight(
            diagnostic_case_why_it_matters(selected_case)
        )

        case_detail = pd.DataFrame(
            {
                "Field": [
                    "Taxi Zone",
                    "Borough",
                    "mobility measure × horizon",
                    "Forecast target date",
                    "Temporal bucket",
                    "Selected forecast family",
                    "Diagnostic category",
                    "Diagnostic combination",
                ],
                "Value": [
                    selected_case["zone"],
                    selected_case["borough"],
                    f"{metric_label(metric)} · h={int(selected_case['horizon'])}",
                    pd.Timestamp(selected_case["target_date"]).strftime("%b %d, %Y"),
                    temporal_bucket_label(selected_case["target_temporal_bucket"]),
                    str(selected_case["champion_family"]).title(),
                    selected_case["failure_category"],
                    selected_case["failure_combination"],
                ],
            }
        )
        with st.expander("Diagnostic details", expanded=False):
            st.dataframe(case_detail, hide_index=True, width="stretch")

st.markdown("#### What the six cases show")
st.markdown(six_case_synthesis(curated_lead_cases, failure_cases))

# ------------------------------------------------------------------
# Fixed systemwide mechanism burden
# ------------------------------------------------------------------

st.divider()
st.subheader("3. Which difficult conditions carry disproportionate error?")
st.write(
    "The case studies show what these conditions look like one forecast at a time. Across the "
    "full holdout, this chart asks a broader question: **which conditions occur often, and which "
    "carry more forecast error than their frequency would suggest?**"
)

mechanism_chart = build_mechanism_chart(mechanisms)
st.plotly_chart(
    mechanism_chart,
    width="stretch",
    config=PLOT_CONFIG,
    key="raw20_mechanism_chart",
)
render_chart_insight(mechanism_takeaway(mechanisms))

mechanism_table = mechanisms[
    [
        "Short label",
        "System forecast share (%)",
        "System error burden (%)",
        "Severe-error capture (%)",
        "Benchmark-loss capture (%)",
        "Burden / forecast share (×)",
    ]
].copy()
for column in mechanism_table.columns[1:]:
    mechanism_table[column] = mechanism_table[column].map(
        lambda value: round(float(value), 3)
    )

with st.expander("See the exact mechanism summary", expanded=False):
    render_mechanism_definitions()
    st.dataframe(mechanism_table, hide_index=True, width="stretch")

# ------------------------------------------------------------------
# Exploration boundary + interactive Win–Miss explorer
# ------------------------------------------------------------------

# The shared branding helper owns the exploration surface so every Showcase
# page can use the same subtle visual grammar without page-specific CSS.
with exploration_section(
    key="raw20_exploration_area",
    title="Explore the full holdout",
    description=(
        "Use the controls below to see how forecast outcomes change across "
        "mobility targets, horizons, weekdays and weekends, and different "
        "parts of the day."
    ),
):
    st.subheader("Explore the Win–Miss Plane")
    st.write(
        "Choose a mobility outcome, forecast horizon, day type, or daypart to see how the "
        "balance between forecast error and improvement over Last-week changes."
    )

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        metric_choice = st.selectbox(
            "Mobility measure",
            ["All targets", *METRIC_ORDER],
            format_func=lambda value: (
                "All measures"
                if value == "All targets"
                else metric_label(value)
            ),
            key="raw20_explorer_metric",
        )
    with f2:
        horizon_choice = st.selectbox(
            "Horizon",
            ["All horizons", "1", "2", "5"],
            key="raw20_explorer_horizon",
        )
    with f3:
        day_type_choice = st.selectbox(
            "Day type",
            ["All days", "Weekdays", "Weekends"],
            key="raw20_explorer_day_type",
        )
    with f4:
        daypart_choice = st.selectbox(
            "Daypart",
            ["All dayparts", "Overnight", "AM peak", "Midday", "PM peak", "Evening"],
            key="raw20_explorer_daypart",
        )

    visible_summary = apply_summary_filters(
        holdout_summary,
        metric=metric_choice,
        horizon=horizon_choice,
        day_type=day_type_choice,
        daypart=daypart_choice,
    )

    visible_scatter = apply_page_filters(
        scatter_records,
        metric=metric_choice,
        horizon=horizon_choice,
        day_type=day_type_choice,
        daypart=daypart_choice,
    )

    if visible_summary.empty:
        st.warning("No holdout records match the current explorer filters.")
        st.stop()

    explorer_plane = build_win_miss_plane(
        visible_scatter,
        exact_summary=visible_summary,
        pre_sampled=True,
    )

    st.plotly_chart(
        explorer_plane,
        width="stretch",
        config=PLOT_CONFIG,
        key="raw20_explorer_plane",
    )

    render_chart_insight(
        explorer_plane_takeaway_from_summary(
            visible_summary,
            holdout_summary,
        )
    )

    # ------------------------------------------------------------------
    # Win–Miss Bands + linked record explorer
    # ------------------------------------------------------------------

    st.divider()
    st.subheader("Where and when do wins and misses recur?")
    st.write(
        "The **Win–Miss Bands** show when the selected forecast beat Last-week — and when it did not — "
        "across multiple Taxi Zones. A band rises above its center line when the selected forecast "
        "was closer, and falls below it when Last-week was closer. Deeper folds mean a larger difference "
        "between the two forecasts."
    )

    st.caption(
        f"Using **{day_type_choice.lower()}** and **{daypart_choice.lower()}** from the filters above; "
        "choose the mobility outcome, horizon, ordering, and number of Taxi Zones here."
    )

    s1, s2, s3, s4 = st.columns([1.2, 0.8, 1.15, 0.75])

    with s1:
        default_metric = (
            metric_choice
            if metric_choice in METRIC_ORDER
            else "taxi_trip_count"
        )
        strip_metric = st.selectbox(
            "Band mobility measure",
            METRIC_ORDER,
            index=METRIC_ORDER.index(default_metric),
            format_func=metric_label,
            key="raw20_band_metric",
        )

    with s2:
        default_horizon = (
            int(horizon_choice)
            if horizon_choice != "All horizons"
            else 5
        )
        strip_horizon = st.selectbox(
            "Band horizon",
            [1, 2, 5],
            index=[1, 2, 5].index(default_horizon),
            format_func=lambda value: f"h={value}",
            key="raw20_band_horizon",
        )

    with s3:
        strip_sort = st.selectbox(
            "Order Taxi Zones by",
            [
                "Clear-miss rate",
                "Severe-error rate",
                "Hard-but-helpful rate",
                "Median error severity",
                "Alphabetical",
            ],
            key="raw20_band_sort",
        )

    with s4:
        strip_zone_count = st.selectbox(
            "Zones shown",
            STRIP_ZONE_COUNT_OPTIONS,
            index=1,
            key="raw20_band_zone_count",
        )

    with st.expander("What do the Taxi Zone ordering measures mean?", expanded=False):
        render_band_rate_definitions()

    band_records = load_band_records(
        strip_metric,
        strip_horizon,
    )

    strip_scope = apply_page_filters(
        band_records,
        metric=strip_metric,
        horizon=str(strip_horizon),
        day_type=day_type_choice,
        daypart=daypart_choice,
    )

    if strip_scope.empty:
        st.warning("No records match the current Win–Miss Band filters.")
    else:
        zone_summary = zone_strip_summary(strip_scope)
        selected_zone_summary = choose_strip_zones(
            zone_summary,
            sort_mode=strip_sort,
            zone_count=strip_zone_count,
        )

        all_daily_strip = daily_strip_representatives(strip_scope)
        shown_zone_ids = set(selected_zone_summary["taxi_zone_id"].astype(int))
        shown_strip_points = all_daily_strip.loc[
            all_daily_strip["taxi_zone_id"].isin(shown_zone_ids)
        ].copy()

        default_record = default_strip_record(shown_strip_points)
        available_keys = {
            record_key_from_row(row)
            for _, row in shown_strip_points.iterrows()
        }

        current_key = st.session_state.get("raw20_selected_record_key")
        if current_key is not None:
            current_key = tuple(current_key)

        if current_key not in available_keys:
            current_key = (
                record_key_from_row(default_record)
                if default_record is not None
                else None
            )
            st.session_state["raw20_selected_record_key"] = current_key

        strip_title = (
            f"Win–Miss Bands · {metric_label(strip_metric)} · h={strip_horizon} · "
            f"ordered by {strip_sort.lower()}"
        )
        strip_figure = build_forecast_error_strips(
            all_daily_strip,
            selected_zone_summary,
            title=strip_title,
            highlight_key=current_key,
        )

        st.caption(
            "Click any band position to inspect that forecast below. The **teal-outlined diamond** "
            "marks the record currently selected."
        )

        strip_event = st.plotly_chart(
            strip_figure,
            width="stretch",
            config=PLOT_CONFIG,
            key="raw20_error_bands",
            on_select="rerun",
            selection_mode="points",
        )

        clicked_key = record_key_from_plotly_event(strip_event)
        if clicked_key is not None and clicked_key != current_key:
            st.session_state["raw20_selected_record_key"] = clicked_key
            st.rerun()

        st.caption(
            "Each date is one discrete forecast outcome. If several dayparts remain for the same "
            "Taxi Zone and date, the chart shows the one with the largest selected-forecast error."
        )

        with st.expander("How to read the Win–Miss Bands", expanded=False):
            st.markdown(
                """
    - **Each row:** one Taxi Zone under the selected mobility measure × horizon.
    - **Left → right:** target date across the untouched Jan 5–Mar 31, 2026 final holdout.
    - **Above the row center:** the selected forecast was closer than Last-week.
    - **Below the row center:** Last-week was closer than the selected forecast.
    - **On the row center:** an exact tie; both forecasts had the same absolute error.
    - **Band depth:** the magnitude of the selected forecast's advantage or loss versus Last-week, using the same baseline-advantage index as the Win–Miss Plane.
    - **Forecast error remains visible in hover:** the bands emphasize comparative usefulness, while hover and the record detail below retain the selected forecast's own error index and severe-error flag.
    - **Teal-outlined diamond:** the exact record currently loaded below.
    - **Only one side is valid for an exact record.** The stepwise rendering prevents the visual from implying that a record can be both a forecast win and a Last-week win.
    - **Click a band position:** load that exact record into the explorer below.

    """
            )

        shown_advantage = pd.to_numeric(
            shown_strip_points["benchmark_advantage_index"],
            errors="coerce",
        )
        shown_supported = shown_advantage.notna()
        shown_denominator = int(shown_supported.sum())

        shown_rate = (
            100 * shown_advantage.gt(0).sum() / shown_denominator
            if shown_denominator
            else np.nan
        )
        shown_tie = (
            100 * shown_advantage.eq(0).sum() / shown_denominator
            if shown_denominator
            else np.nan
        )
        shown_severe = 100 * float(shown_strip_points["severe_error"].mean())
        shown_clear = 100 * float(
            (
                shown_strip_points["severe_error"]
                & shown_advantage.lt(0)
            ).mean()
        )

        median_advantage = float(
            pd.to_numeric(
                shown_strip_points["benchmark_advantage_index"],
                errors="coerce",
            ).abs().median()
        )
        lead_zone = (
            str(selected_zone_summary.iloc[0]["zone"])
            if not selected_zone_summary.empty
            else "the leading zone"
        )

        tie_sentence = (
            f" Exact ties account for **{shown_tie:.2f}%**."
            if np.isfinite(shown_tie) and shown_tie > 0
            else ""
        )

        render_chart_insight(
            f"Across these **{len(selected_zone_summary)} Taxi Zones**, the selected "
            f"forecast is closer than Last-week on **{shown_rate:.1f}%** of the "
            f"**{len(shown_strip_points):,} displayed daily outcomes**. "
            f"**{shown_severe:.1f}%** are severe errors and **{shown_clear:.1f}%** "
            f"are clear misses where Last-week was closer.{tie_sentence} The median "
            f"absolute baseline-advantage index is **{median_advantage:.1f}**, so "
            "deeper folds correspond to larger wins or losses."
        )

        with st.expander("See the Taxi Zone ordering evidence", expanded=False):
            st.dataframe(
                strip_zone_table(selected_zone_summary),
                hide_index=True,
                width="stretch",
            )

        st.markdown("### Inspect one forecast record")
        st.write(
            "This detail view puts the selected outcome back into native mobility units. Compare what "
            "actually happened with the selected forecast and Last-week, then use the timeline and "
            "diagnostic context to see what surrounded that forecast."
        )

        selected_key = st.session_state.get("raw20_selected_record_key")
        selected_record = (
            record_from_key(strip_scope, tuple(selected_key))
            if selected_key is not None
            else None
        )

        if selected_record is None:
            st.info("Click a band position to inspect that record.")
        else:
            selected_metric = str(selected_record["metric"])
            selected_unit = metric_unit(selected_metric)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric(
                "Observed",
                f"{format_native(selected_metric, selected_record['actual'])} {selected_unit}",
            )
            c2.metric(
                "Selected forecast",
                f"{format_native(selected_metric, selected_record['champion_prediction'])} {selected_unit}",
            )
            c3.metric(
                "Last-week",
                f"{format_native(selected_metric, selected_record['benchmark_prediction'])} {selected_unit}",
            )
            c4.metric(
                "Error index",
                format_number(selected_record["system_row_error_index"]),
            )
            c5.metric(
                "Baseline advantage",
                format_number(selected_record["benchmark_advantage_index"]),
            )

            record_chart = build_record_context_chart(selected_record, band_records)
            st.plotly_chart(
                record_chart,
                width="stretch",
                config=PLOT_CONFIG,
                key="raw20_selected_record_chart",
            )
            render_chart_insight(selected_record_takeaway(selected_record))

            with st.expander("Selected record details", expanded=False):
                record_details = pd.DataFrame(
                    {
                        "Field": [
                            "Taxi Zone",
                            "Borough",
                            "mobility measure × horizon",
                            "Forecast target date",
                            "Temporal bucket",
                            "Selected forecast family",
                            "Severe error",
                            "Diagnostic combination",
                        ],
                        "Value": [
                            selected_record["zone"],
                            selected_record["borough"],
                            (
                                f"{metric_label(selected_record['metric'])} · "
                                f"h={int(selected_record['horizon'])}"
                            ),
                            pd.Timestamp(selected_record["target_date"]).strftime("%b %d, %Y"),
                            temporal_bucket_label(selected_record["target_temporal_bucket"]),
                            str(selected_record["champion_family"]).title(),
                            "Yes" if bool(selected_record["severe_error"]) else "No",
                            selected_record["failure_combination"],
                        ],
                    }
                )
                st.dataframe(record_details, hide_index=True, width="stretch")

# ------------------------------------------------------------------
# Closing synthesis
# ------------------------------------------------------------------

st.divider()

st.markdown("### What this page establishes")
st.markdown(
    "A large forecast error and an unhelpful forecast are not the same thing. Some of "
    "the hardest observations still benefit substantially from the selected model "
    "because Last-week misses by even more, while other apparently modest errors lose "
    "to the simple baseline. The useful diagnostic question is therefore not just "
    "*how wrong was the forecast?* but *how wrong was it relative to a reasonable "
    "alternative, and under what conditions do those wins and misses recur?*"
)

with st.expander("How this page works", expanded=False):
    st.markdown(
        """
        **1. Freeze model choice before the final test.** For each mobility measure ×
        horizon, the selected model family was chosen from the earlier validation
        comparison. Every visual on this page then uses the untouched **Jan 5–Mar 31,
        2026 final holdout**.

        **2. Put different forecasting jobs on a common error scale.** **Forecast error
        index** is a record's absolute selected-forecast error divided by that mobility
        measure × horizon's overall selected-forecast MAE, ×100. **100** therefore means
        a miss equal to one full job-level MAE.

        **3. Measure usefulness against Last-week.** **Baseline advantage index** is
        Last-week absolute error minus selected-forecast absolute error, scaled by the
        same job-level MAE. Positive means the selected forecast was closer; negative
        means Last-week was closer; zero is an exact tie.

        **4. Read the Win–Miss Plane as a diagnostic map, not a correlation plot.** Its
        two axes share the selected-forecast error term and the same denominator, so part
        of the visible geometry is mathematical structure. The x=100 and y=0 lines are
        interpretive boundaries, not fitted thresholds.

        **5. Keep severe error separate from above-average error.** **Severe error** is
        the highest-error **10% within each mobility measure × horizon**. A record can
        sit to the right of x=100 without belonging to that severe tail.

        **6. Preserve the tails while keeping the scatter responsive.** The Win–Miss
        Plane displays a tail-preserving sample, but headline metrics and quadrant shares
        use the full supported population. Strong model wins, strong Last-week wins,
        large errors, severe cases, and ties are deliberately retained.

        **7. Return normalized diagnostics to native values.** Cross-target indices make
        different forecasting jobs comparable, but the case views keep the observed,
        selected-forecast, and Last-week values visible so practical interpretation does
        not depend on normalized scores alone.
        """
    )

st.caption(
    "Evidence scope: diagnostic performance of the preselected forecasting system on "
    "the untouched Jan 5–Mar 31, 2026 final holdout. Win–Miss patterns show when the "
    "selected forecast did or did not improve on Last-week; diagnostic conditions are "
    "associations with forecast difficulty, not causes of forecast error."
)
