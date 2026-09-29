"""
Build compact forecasting runtime tables for Raw 18-20.

The large Chapter 4 forecast surfaces remain authoritative build inputs.
The Showcase receives only the evidence required to reproduce its current
interactive views.

Outputs
-------
forecast_explorer_runtime.parquet
    Wide longitudinal evidence for Raw 18.

forecast_holdout_slice_summary.parquet
    Exact filterable summary evidence for Raw 20.

forecast_holdout_runtime.parquet
    Narrow exact row-level evidence for Raw 20's Win-Miss Bands and linked
    native-unit record explorer.

forecast_holdout_scatter_sample.parquet
    Deterministic visualization evidence for Raw 20's Win-Miss plane.

forecast_runtime_build_qa.parquet
    Build sizes, row counts, and validation results.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


# =====================================================================
# Paths
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FORECAST_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "4.7.1.final_tables"
)

APP_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "app_tables"
)

RECORD_SOURCE = (
    FORECAST_DIR
    / "showcase_forecast_record_surface.parquet"
)

HISTORY_SOURCE = (
    FORECAST_DIR
    / "showcase_forecast_history_surface.parquet"
)

EXPLORER_OUTPUT = (
    APP_DIR
    / "forecast_explorer_runtime.parquet"
)

SUMMARY_OUTPUT = (
    APP_DIR
    / "forecast_holdout_slice_summary.parquet"
)

HOLDOUT_OUTPUT = (
    APP_DIR
    / "forecast_holdout_runtime.parquet"
)

SCATTER_OUTPUT = (
    APP_DIR
    / "forecast_holdout_scatter_sample.parquet"
)

QA_OUTPUT = (
    APP_DIR
    / "forecast_runtime_build_qa.parquet"
)

# Superseded by HOLDOUT_OUTPUT. Remove only after the new build succeeds.
OBSOLETE_V2_OUTPUT = (
    APP_DIR
    / "forecast_holdout_daily_representatives.parquet"
)


# =====================================================================
# Frozen forecasting contract
# =====================================================================

FINAL_HOLDOUT_START = pd.Timestamp("2026-01-05")
FINAL_HOLDOUT_END = pd.Timestamp("2026-03-31")

METRICS = [
    "taxi_trip_count",
    "taxi_avg_trip_speed",
    "fhvhv_trip_count",
    "fhvhv_avg_trip_speed",
    "subway_ridership",
]

HORIZONS = [1, 2, 5]

SCATTER_RANDOM_ROWS = 12_000
SCATTER_TAIL_ROWS = 350
RANDOM_STATE = 696


HISTORY_COLUMNS = [
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


HOLDOUT_SOURCE_COLUMNS = [
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
    "severe_error",
    "failure_combination",
    "low_zero_demand",
    "unstable_conditions",
    "shared_shock",
    "system_row_error_index",
    "benchmark_advantage_index",
    "reader_facing_zone",
]


# This is the production contract proven by the Raw 20 storage scout.
# It preserves every exact holdout record while dropping fields that the
# Bands and linked record-context chart do not need.
HOLDOUT_RUNTIME_COLUMNS = [
    "metric",
    "horizon",
    "taxi_zone_id",
    "target_date",
    "target_temporal_bucket",
    "system_row_error_index",
    "benchmark_advantage_index",
    "severe_error",
    "failure_combination",
    "actual",
    "champion_prediction",
    "benchmark_prediction",
]


OBSERVATION_KEY = [
    "metric",
    "taxi_zone_id",
    "zone",
    "borough",
    "target_date",
    "target_temporal_bucket",
    "target_observation_sequence_id",
    "evaluation_period",
]

HOLDOUT_RECORD_KEY = [
    "metric",
    "horizon",
    "taxi_zone_id",
    "target_date",
    "target_temporal_bucket",
]


# =====================================================================
# Helpers
# =====================================================================

def elapsed(start: float) -> str:
    """Return a compact elapsed-time label."""
    return f"{perf_counter() - start:,.1f}s"


def size_mb(path: Path) -> float:
    """Return file size in MiB."""
    return path.stat().st_size / (1024 ** 2)


def require_file(path: Path) -> None:
    """Fail immediately when a frozen source is unavailable."""
    if not path.exists():
        raise FileNotFoundError(path)


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize shared runtime key types."""
    frame = frame.copy()

    frame["target_date"] = pd.to_datetime(
        frame["target_date"],
        errors="coerce",
    )

    frame["horizon"] = pd.to_numeric(
        frame["horizon"],
        errors="coerce",
    ).astype("Int64")

    frame["taxi_zone_id"] = pd.to_numeric(
        frame["taxi_zone_id"],
        errors="coerce",
    ).astype("Int64")

    frame["reader_facing_zone"] = (
        frame["reader_facing_zone"]
        .fillna(False)
        .astype(bool)
    )

    return frame


def day_type(series: pd.Series) -> pd.Series:
    """Translate temporal buckets into Raw 20 day types."""
    return pd.Series(
        np.where(
            series.astype(str).str.startswith("weekend_"),
            "Weekends",
            "Weekdays",
        ),
        index=series.index,
    )


def daypart(series: pd.Series) -> pd.Series:
    """Translate temporal buckets into Raw 20 dayparts."""
    return (
        series.astype(str)
        .str.replace("weekday_", "", regex=False)
        .str.replace("weekend_", "", regex=False)
    )


def write_table(
    frame: pd.DataFrame,
    path: Path,
) -> float:
    """Write one compressed runtime artifact and return its size."""
    frame.to_parquet(
        path,
        index=False,
        compression="zstd",
    )

    return size_mb(path)


# =====================================================================
# Raw 18 — reshape horizon repetition
# =====================================================================

def build_explorer_runtime() -> tuple[pd.DataFrame, dict]:
    """
    Reshape Raw 18 history from long horizons to one row per observation.

    Actual and Last-week values are collapsed exactly as Raw 18 currently does:
    mean across available horizons. Selected forecasts remain separate h=1/h=2/h=5
    columns.

    Taxi Zone grain is preserved because Raw 18 supports arbitrary geography
    aggregation and activity-weighted speed calculations.
    """
    start = perf_counter()

    print("\n[1/4] Raw 18 longitudinal explorer")
    print(
        f"      Source: {HISTORY_SOURCE.name} "
        f"({size_mb(HISTORY_SOURCE):,.2f} MB)"
    )

    read_start = perf_counter()

    history = pd.read_parquet(
        HISTORY_SOURCE,
        columns=HISTORY_COLUMNS,
    )

    print(
        f"      Read {len(history):,} projected rows "
        f"in {elapsed(read_start)}"
    )

    history = normalize(history)

    history = history.loc[
        history["reader_facing_zone"]
        & history["metric"].isin(METRICS)
        & history["horizon"].isin(HORIZONS)
        & history["target_date"].notna()
        & history["evaluation_period"].isin(
            ["ordinary_validation", "final_holdout"]
        )
    ].copy()

    print(
        f"      Eligible history rows: {len(history):,}"
    )

    reshape_start = perf_counter()

    # Actual mobility should describe the same target observation regardless
    # of forecast horizon. Verify that contract rather than silently averaging
    # horizon-specific values.
    actual_check = (
        history.groupby(
            OBSERVATION_KEY,
            observed=True,
            sort=False,
        )["actual"]
        .agg(["min", "max"])
    )

    actual_delta = (
        actual_check["max"]
        - actual_check["min"]
    ).abs()

    max_actual_horizon_delta = (
        float(actual_delta.max())
        if actual_delta.notna().any()
        else 0.0
    )

    if not np.isclose(
        max_actual_horizon_delta,
        0.0,
        equal_nan=True,
    ):
        raise AssertionError(
            "Actual mobility is not horizon-invariant. "
            f"Maximum within-observation delta: "
            f"{max_actual_horizon_delta:g}"
        )

    base = (
        history.groupby(
            OBSERVATION_KEY,
            observed=True,
            sort=False,
            as_index=False,
        )
        .agg(
            actual=("actual", "first"),
            horizon_rows=("horizon", "nunique"),
        )
    )

    # WHY: Raw 18 needs the forecast AND Last-week value separately for each
    # horizon. This is especially important for aggregate speed views because
    # each horizon uses its corresponding trip-count prediction as the weight.
    forecasts = (
        history.pivot(
            index=OBSERVATION_KEY,
            columns="horizon",
            values="champion_prediction",
        )
        .rename(
            columns={
                1: "forecast_h1",
                2: "forecast_h2",
                5: "forecast_h5",
            }
        )
        .reset_index()
    )

    benchmarks = (
        history.pivot(
            index=OBSERVATION_KEY,
            columns="horizon",
            values="benchmark_prediction",
        )
        .rename(
            columns={
                1: "benchmark_h1",
                2: "benchmark_h2",
                5: "benchmark_h5",
            }
        )
        .reset_index()
    )

    explorer = (
        base
        .merge(
            forecasts,
            on=OBSERVATION_KEY,
            how="inner",
            validate="one_to_one",
        )
        .merge(
            benchmarks,
            on=OBSERVATION_KEY,
            how="inner",
            validate="one_to_one",
        )
    )

    print(
        f"      Reshaped {len(history):,} → "
        f"{len(explorer):,} rows in {elapsed(reshape_start)}"
    )

    duplicate_keys = int(
        explorer.duplicated(OBSERVATION_KEY).sum()
    )

    if duplicate_keys:
        raise AssertionError(
            f"Raw 18 runtime has {duplicate_keys:,} "
            "duplicate observation keys."
        )

    expected_rows = (
        history[OBSERVATION_KEY]
        .drop_duplicates()
        .shape[0]
    )

    if len(explorer) != expected_rows:
        raise AssertionError(
            "Raw 18 wide reshape changed the observation population."
        )

    output_mb = write_table(
        explorer,
        EXPLORER_OUTPUT,
    )

    print(
        f"      Output: {output_mb:,.2f} MB"
    )
    print(
        f"      Date range: "
        f"{explorer['target_date'].min().date()} → "
        f"{explorer['target_date'].max().date()}"
    )
    print(
        f"      Observation keys duplicated: {duplicate_keys:,}"
    )
    print(
        f"      Stage runtime: {elapsed(start)}"
    )

    qa = {
        "artifact": EXPLORER_OUTPUT.name,
        "rows": len(explorer),
        "columns": len(explorer.columns),
        "output_mb": output_mb,
        "duplicate_keys": duplicate_keys,
        "qa_status": "PASS",
    }

    return explorer, qa


# =====================================================================
# Raw 20 — load frozen final holdout
# =====================================================================

def load_holdout() -> pd.DataFrame:
    """Read only final-holdout fields required by Raw 20."""
    start = perf_counter()

    holdout = pd.read_parquet(
        RECORD_SOURCE,
        columns=HOLDOUT_SOURCE_COLUMNS,
    )

    holdout = normalize(holdout)

    holdout = holdout.loc[
        holdout["reader_facing_zone"]
        & holdout["metric"].isin(METRICS)
        & holdout["horizon"].isin(HORIZONS)
        & holdout["target_date"].between(
            FINAL_HOLDOUT_START,
            FINAL_HOLDOUT_END,
            inclusive="both",
        )
    ].copy()

    holdout["severe_error"] = (
        holdout["severe_error"]
        .fillna(False)
        .astype(bool)
    )

    for column in [
        "low_zero_demand",
        "unstable_conditions",
        "shared_shock",
    ]:
        holdout[column] = (
            holdout[column]
            .fillna(False)
            .astype(bool)
        )

    holdout["day_type"] = day_type(
        holdout["target_temporal_bucket"]
    )

    holdout["daypart"] = daypart(
        holdout["target_temporal_bucket"]
    )

    advantage = pd.to_numeric(
        holdout["benchmark_advantage_index"],
        errors="coerce",
    )

    error_index = pd.to_numeric(
        holdout["system_row_error_index"],
        errors="coerce",
    )

    holdout["comparison_supported"] = (
        advantage.notna()
        & error_index.notna()
        & error_index.ge(0)
    )

    holdout["model_win"] = (
        holdout["comparison_supported"]
        & advantage.gt(0)
    )

    holdout["last_week_win"] = (
        holdout["comparison_supported"]
        & advantage.lt(0)
    )

    holdout["comparison_tie"] = (
        holdout["comparison_supported"]
        & advantage.eq(0)
    )

    holdout["clear_miss"] = (
        holdout["severe_error"]
        & holdout["last_week_win"]
    )

    holdout["hard_helpful"] = (
        holdout["severe_error"]
        & holdout["model_win"]
    )

    print(
        f"\n      Loaded {len(holdout):,} "
        f"final-holdout rows in {elapsed(start)}"
    )

    return holdout


# =====================================================================
# Raw 20 — exact summary cube
# =====================================================================

def build_holdout_summary(
    holdout: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Precompute exact Raw 20 filterable statistics.

    Grain:
        metric × horizon × day type × daypart × Taxi Zone

    Counts remain additive across reader-facing filters. The row-level runtime
    surface remains available wherever Raw 20 needs exact record distributions.
    """
    start = perf_counter()

    print("\n[2/4] Raw 20 exact slice summary")

    grouping = [
        "metric",
        "horizon",
        "day_type",
        "daypart",
        "taxi_zone_id",
        "zone",
        "borough",
    ]

    summary = (
        holdout.groupby(
            grouping,
            observed=True,
            sort=False,
            as_index=False,
        )
        .agg(
            forecast_rows=(
                "system_row_error_index",
                "size",
            ),
            supported_rows=(
                "comparison_supported",
                "sum",
            ),
            model_win_rows=(
                "model_win",
                "sum",
            ),
            last_week_win_rows=(
                "last_week_win",
                "sum",
            ),
            tie_rows=(
                "comparison_tie",
                "sum",
            ),
            severe_rows=(
                "severe_error",
                "sum",
            ),
            clear_miss_rows=(
                "clear_miss",
                "sum",
            ),
            hard_helpful_rows=(
                "hard_helpful",
                "sum",
            ),
            median_error_index=(
                "system_row_error_index",
                "median",
            ),
            mean_error_index=(
                "system_row_error_index",
                "mean",
            ),
            absolute_error_sum=(
                "absolute_error",
                "sum",
            ),
            benchmark_absolute_error_sum=(
                "benchmark_absolute_error",
                "sum",
            ),
        )
    )

    checks = {
        "forecast_rows": len(holdout),
        "supported_rows": int(
            holdout["comparison_supported"].sum()
        ),
        "model_win_rows": int(
            holdout["model_win"].sum()
        ),
        "last_week_win_rows": int(
            holdout["last_week_win"].sum()
        ),
        "tie_rows": int(
            holdout["comparison_tie"].sum()
        ),
        "severe_rows": int(
            holdout["severe_error"].sum()
        ),
        "clear_miss_rows": int(
            holdout["clear_miss"].sum()
        ),
        "hard_helpful_rows": int(
            holdout["hard_helpful"].sum()
        ),
    }

    for column, expected in checks.items():
        observed = int(
            summary[column].sum()
        )

        if observed != expected:
            raise AssertionError(
                f"{column} mismatch: "
                f"summary={observed:,}, "
                f"source={expected:,}"
            )

    output_mb = write_table(
        summary,
        SUMMARY_OUTPUT,
    )

    duplicate_keys = int(
        summary.duplicated(grouping).sum()
    )

    print(
        f"      {len(holdout):,} rows → "
        f"{len(summary):,} exact slices"
    )
    print(
        f"      Output: {output_mb:,.2f} MB"
    )
    print(
        "      Additive QA: PASS"
    )
    print(
        f"      Stage runtime: {elapsed(start)}"
    )

    qa = {
        "artifact": SUMMARY_OUTPUT.name,
        "rows": len(summary),
        "columns": len(summary.columns),
        "output_mb": output_mb,
        "duplicate_keys": duplicate_keys,
        "qa_status": "PASS",
    }

    return summary, qa


# =====================================================================
# Raw 20 — narrow exact row-level runtime
# =====================================================================

def build_holdout_runtime(
    holdout: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Preserve every exact final-holdout record needed by Raw 20.

    The storage scout showed that one unified narrow surface is essentially the
    same size as separate Band + Context surfaces, while avoiding duplicate
    keys and a more complicated runtime contract.
    """
    start = perf_counter()

    print("\n[3/4] Raw 20 exact row-level runtime")

    runtime = holdout[
        HOLDOUT_RUNTIME_COLUMNS
    ].copy()

    source_duplicates = int(
        holdout.duplicated(
            HOLDOUT_RECORD_KEY
        ).sum()
    )

    runtime_duplicates = int(
        runtime.duplicated(
            HOLDOUT_RECORD_KEY
        ).sum()
    )

    if runtime_duplicates != source_duplicates:
        raise AssertionError(
            "Raw 20 runtime changed exact record-key behavior."
        )

    if len(runtime) != len(holdout):
        raise AssertionError(
            "Raw 20 runtime changed the final-holdout row population."
        )

    # Exact value-preservation QA for every numeric field carried into
    # the production runtime table.
    numeric_columns = [
        "system_row_error_index",
        "benchmark_advantage_index",
        "actual",
        "champion_prediction",
        "benchmark_prediction",
    ]

    max_numeric_delta = 0.0

    for column in numeric_columns:
        source_values = pd.to_numeric(
            holdout[column],
            errors="coerce",
        )

        runtime_values = pd.to_numeric(
            runtime[column],
            errors="coerce",
        )

        delta = (
            source_values - runtime_values
        ).abs()

        if delta.notna().any():
            max_numeric_delta = max(
                max_numeric_delta,
                float(delta.max()),
            )

    if not np.isclose(
        max_numeric_delta,
        0.0,
        equal_nan=True,
    ):
        raise AssertionError(
            "Raw 20 runtime changed numeric values."
        )

    output_mb = write_table(
        runtime,
        HOLDOUT_OUTPUT,
    )

    print(
        f"      Retained all {len(runtime):,} "
        "exact final-holdout records"
    )
    print(
        f"      Columns: "
        f"{len(holdout.columns):,} working → "
        f"{len(runtime.columns):,} runtime"
    )
    print(
        f"      Output: {output_mb:,.2f} MB"
    )
    print(
        f"      Duplicate record keys: "
        f"{runtime_duplicates:,}"
    )
    print(
        f"      Max numeric delta: "
        f"{max_numeric_delta:g}"
    )
    print(
        f"      Stage runtime: {elapsed(start)}"
    )

    qa = {
        "artifact": HOLDOUT_OUTPUT.name,
        "rows": len(runtime),
        "columns": len(runtime.columns),
        "output_mb": output_mb,
        "duplicate_keys": runtime_duplicates,
        "qa_status": "PASS",
    }

    return runtime, qa


# =====================================================================
# Raw 20 — deterministic scatter evidence
# =====================================================================

def build_scatter_sample(
    holdout: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Materialize Raw 20's deterministic Win-Miss visualization evidence.

    Exact statistics remain based on the full population; this artifact exists
    only to keep the Plotly point layer responsive.
    """
    start = perf_counter()

    print("\n[4/4] Raw 20 deterministic scatter evidence")

    eligible = holdout.loc[
        holdout["comparison_supported"]
    ].copy()

    random_part = eligible.sample(
        n=min(
            SCATTER_RANDOM_ROWS,
            len(eligible),
        ),
        random_state=RANDOM_STATE,
    )

    strongest_wins = eligible.nlargest(
        min(
            SCATTER_TAIL_ROWS,
            len(eligible),
        ),
        "benchmark_advantage_index",
    )

    strongest_losses = eligible.nsmallest(
        min(
            SCATTER_TAIL_ROWS,
            len(eligible),
        ),
        "benchmark_advantage_index",
    )

    largest_errors = eligible.nlargest(
        min(
            SCATTER_TAIL_ROWS,
            len(eligible),
        ),
        "system_row_error_index",
    )

    severe_helpful = (
        eligible.loc[
            eligible["severe_error"]
            & eligible[
                "benchmark_advantage_index"
            ].gt(0)
        ]
        .nlargest(
            SCATTER_TAIL_ROWS,
            "system_row_error_index",
        )
    )

    severe_misses = (
        eligible.loc[
            eligible["severe_error"]
            & eligible[
                "benchmark_advantage_index"
            ].lt(0)
        ]
        .nlargest(
            SCATTER_TAIL_ROWS,
            "system_row_error_index",
        )
    )

    ties = eligible.loc[
        eligible[
            "benchmark_advantage_index"
        ].eq(0)
    ].head(SCATTER_TAIL_ROWS)

    sample = (
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
        .loc[
            lambda frame:
            ~frame.index.duplicated(
                keep="first"
            )
        ]
        .copy()
    )

    output_mb = write_table(
        sample,
        SCATTER_OUTPUT,
    )

    print(
        f"      {len(eligible):,} supported rows → "
        f"{len(sample):,} visualization rows"
    )
    print(
        f"      Output: {output_mb:,.2f} MB"
    )
    print(
        f"      Stage runtime: {elapsed(start)}"
    )

    qa = {
        "artifact": SCATTER_OUTPUT.name,
        "rows": len(sample),
        "columns": len(sample.columns),
        "output_mb": output_mb,
        "duplicate_keys": 0,
        "qa_status": "PASS",
    }

    return sample, qa


# =====================================================================
# Cleanup
# =====================================================================

def remove_obsolete_v2_output() -> None:
    """Remove the superseded 54 MB V2 artifact after all V3 outputs exist."""
    required_outputs = [
        EXPLORER_OUTPUT,
        SUMMARY_OUTPUT,
        HOLDOUT_OUTPUT,
        SCATTER_OUTPUT,
        QA_OUTPUT,
    ]

    if not all(
        path.exists()
        for path in required_outputs
    ):
        return

    if OBSOLETE_V2_OUTPUT.exists():
        obsolete_mb = size_mb(
            OBSOLETE_V2_OUTPUT
        )

        OBSOLETE_V2_OUTPUT.unlink()

        print(
            "\nRemoved superseded V2 artifact:"
        )
        print(
            f"  {OBSOLETE_V2_OUTPUT.name} "
            f"({obsolete_mb:,.2f} MB)"
        )


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    """Build all production forecasting runtime artifacts."""
    total_start = perf_counter()

    print("=" * 72)
    print("FORECASTING RUNTIME BUILD V3")
    print("=" * 72)

    require_file(RECORD_SOURCE)
    require_file(HISTORY_SOURCE)

    APP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    _, explorer_qa = (
        build_explorer_runtime()
    )

    holdout = load_holdout()

    _, summary_qa = (
        build_holdout_summary(
            holdout
        )
    )

    _, holdout_qa = (
        build_holdout_runtime(
            holdout
        )
    )

    _, scatter_qa = (
        build_scatter_sample(
            holdout
        )
    )

    qa = pd.DataFrame(
        [
            explorer_qa,
            summary_qa,
            holdout_qa,
            scatter_qa,
        ]
    )

    qa.to_parquet(
        QA_OUTPUT,
        index=False,
        compression="zstd",
    )

    source_mb = (
        size_mb(RECORD_SOURCE)
        + size_mb(HISTORY_SOURCE)
    )

    runtime_mb = float(
        qa["output_mb"].sum()
    )

    reduction_pct = (
        100
        * (
            1
            - runtime_mb
            / source_mb
        )
    )

    print("\n" + "=" * 72)
    print("BUILD COMPLETE")
    print("=" * 72)

    print(
        qa[
            [
                "artifact",
                "rows",
                "columns",
                "output_mb",
                "duplicate_keys",
                "qa_status",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Large source surfaces: "
        f"{source_mb:,.2f} MB"
    )
    print(
        f"Production runtime:    "
        f"{runtime_mb:,.2f} MB"
    )
    print(
        f"Size reduction:        "
        f"{reduction_pct:,.1f}%"
    )
    print(
        f"Total runtime:         "
        f"{elapsed(total_start)}"
    )

    print("\nProduction outputs:")

    for path in [
        EXPLORER_OUTPUT,
        SUMMARY_OUTPUT,
        HOLDOUT_OUTPUT,
        SCATTER_OUTPUT,
        QA_OUTPUT,
    ]:
        print(
            f"  {path.name:<46} "
            f"{size_mb(path):>7.2f} MB"
        )

    remove_obsolete_v2_output()


if __name__ == "__main__":
    main()