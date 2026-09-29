"""
Build compact, deployment-ready weather relationship tables for Raw 09.

WHY
---
Raw 09 currently reads two large analytical source tables at runtime:

- analysis_ready_mobility_panel.parquet
- weather_taxi_zone_context.parquet

Those files are appropriate analytical sources, but they are unnecessarily
large runtime dependencies for the deployed Streamlit Showcase.

This builder performs the expensive aggregation once and writes compact
app-ready tables that preserve:

- Citywide exploration
- Borough comparison
- Taxi Zone drill-down
- Date-range filtering
- Temporal-bucket filtering
- Scatterplots
- Time-series views
- Pre/post congestion-pricing comparisons
- Pearson and Spearman correlations

The builder is intentionally verbose. Long-running operations report their
input size, output size, elapsed stage time, and total build time so that
progress remains visible throughout execution.

Outputs
-------
data/processed/app_tables/weather_relationship_daily.parquet
data/processed/app_tables/weather_relationship_geography.parquet
data/processed/app_tables/weather_relationship_build_qa.parquet
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd


# =====================================================================
# 1. PATHS
# =====================================================================

ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "1.3.1.final_tables"
)

APP_TABLE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "app_tables"
)

MOBILITY_PANEL_PATH = (
    SOURCE_DIR
    / "analysis_ready_mobility_panel.parquet"
)

WEATHER_PANEL_PATH = (
    SOURCE_DIR
    / "weather_taxi_zone_context.parquet"
)

DAILY_OUTPUT_PATH = (
    APP_TABLE_DIR
    / "weather_relationship_daily.parquet"
)

GEOGRAPHY_OUTPUT_PATH = (
    APP_TABLE_DIR
    / "weather_relationship_geography.parquet"
)

QA_OUTPUT_PATH = (
    APP_TABLE_DIR
    / "weather_relationship_build_qa.parquet"
)


# =====================================================================
# 2. ANALYTICAL CONTRACT
# =====================================================================

CP_START_DATE = pd.Timestamp("2025-01-05")

VALID_BOROUGHS = [
    "Bronx",
    "Brooklyn",
    "Manhattan",
    "Queens",
    "Staten Island",
]

COUNT_METRICS = [
    "taxi_trip_count",
    "fhvhv_trip_count",
    "subway_ridership",
]

SPEED_WEIGHT_COLUMNS = {
    "taxi_avg_trip_speed": "taxi_trip_count",
    "fhvhv_avg_trip_speed": "fhvhv_trip_count",
    "avg_bus_speed": "bus_trip_count",
}

MOBILITY_METRICS = [
    "taxi_trip_count",
    "taxi_avg_trip_speed",
    "fhvhv_trip_count",
    "fhvhv_avg_trip_speed",
    "subway_ridership",
    "avg_bus_speed",
]

WEATHER_METRICS = [
    "temperature",
    "precipitation",
    "visibility",
    "wind_speed",
    "wind_gust",
    "relative_humidity",
    "ceiling_height",
    "pressure_3hr_change",
]

# =====================================================================
# SHOWCASE STORAGE PRECISION
# =====================================================================
#
# These tables are deployment artifacts, not the canonical analytical
# record. The source panels retain their original numerical precision.
#
# WHY:
# Raw floating-point weather values are nearly unique observation by
# observation and compress poorly in Parquet. Raw 09 does not require
# machine-level precision for visualization or relationship exploration.
#
# Quantization happens only AFTER all spatial/temporal aggregation is
# complete, so it cannot affect weighted-average construction.

DISPLAY_PRECISION = {
    # Demand / ridership counts
    "taxi_trip_count": 0,
    "fhvhv_trip_count": 0,
    "subway_ridership": 0,

    # Mobility speeds
    "taxi_avg_trip_speed": 2,
    "fhvhv_avg_trip_speed": 2,
    "avg_bus_speed": 2,

    # Weather context
    "temperature": 2,
    "precipitation": 2,
    "visibility": 4,
    "wind_speed": 2,
    "wind_gust": 2,
    "relative_humidity": 2,
    "ceiling_height": 0,
    "pressure_3hr_change": 3,
}

# =====================================================================
# 3. PROGRESS / TIMING HELPERS
# =====================================================================

BUILD_STARTED = time.perf_counter()


def _elapsed_total() -> float:
    """Return total elapsed build time in seconds."""
    return time.perf_counter() - BUILD_STARTED


def _format_seconds(seconds: float) -> str:
    """Format elapsed seconds for readable console progress."""
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    remaining = seconds % 60

    return f"{minutes}m {remaining:.1f}s"


def _progress(message: str) -> None:
    """
    Print a progress message immediately.

    flush=True is deliberate so PowerShell receives progress while the
    operation is running rather than after an output buffer fills.
    """
    print(
        f"[{_format_seconds(_elapsed_total()):>10}] {message}",
        flush=True,
    )


def _stage(number: int, total: int, message: str) -> float:
    """Start a major build stage and return its timer."""
    print(flush=True)
    _progress(f"[{number}/{total}] {message}")
    return time.perf_counter()


def _finish_stage(
    started: float,
    message: str,
) -> None:
    """Report completion time for a major build stage."""
    elapsed = time.perf_counter() - started

    _progress(
        f"      ✓ {message} "
        f"({_format_seconds(elapsed)})"
    )


def _file_size_mb(path: Path) -> float:
    """Return file size in MiB."""
    return path.stat().st_size / (1024 ** 2)


def _require_file(path: Path) -> None:
    """Fail immediately if an authoritative source is unavailable."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required source table not found: {path}"
        )


# =====================================================================
# 4. SOURCE PREPARATION
# =====================================================================

def _prepare_sources() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load only required columns and attach authoritative geography.

    Returns
    -------
    mobility
        Mobility source with Taxi Zone geography.

    weather
        Weather source with Taxi Zone geography attached.

    zone_lookup
        One authoritative row per Taxi Zone.
    """
    mobility_columns = sorted(
        {
            "date",
            "taxi_zone_id",
            "zone",
            "borough",
            "temporal_bucket",
            *MOBILITY_METRICS,
            *SPEED_WEIGHT_COLUMNS.values(),
        }
    )

    weather_columns = sorted(
        {
            "date",
            "taxi_zone_id",
            "temporal_bucket",
            *WEATHER_METRICS,
        }
    )

    _progress(
        "      Reading mobility source "
        f"({len(mobility_columns)} projected columns)..."
    )

    read_started = time.perf_counter()

    mobility = pd.read_parquet(
        MOBILITY_PANEL_PATH,
        columns=mobility_columns,
    )

    _progress(
        "      Mobility loaded: "
        f"{len(mobility):,} rows in "
        f"{_format_seconds(time.perf_counter() - read_started)}"
    )

    _progress(
        "      Reading weather source "
        f"({len(weather_columns)} projected columns)..."
    )

    read_started = time.perf_counter()

    weather = pd.read_parquet(
        WEATHER_PANEL_PATH,
        columns=weather_columns,
    )

    _progress(
        "      Weather loaded: "
        f"{len(weather):,} rows in "
        f"{_format_seconds(time.perf_counter() - read_started)}"
    )

    _progress("      Normalizing dates and Taxi Zone IDs...")

    normalize_started = time.perf_counter()

    mobility["date"] = pd.to_datetime(
        mobility["date"],
        errors="coerce",
    )

    weather["date"] = pd.to_datetime(
        weather["date"],
        errors="coerce",
    )

    mobility["taxi_zone_id"] = pd.to_numeric(
        mobility["taxi_zone_id"],
        errors="coerce",
    ).astype("Int64")

    weather["taxi_zone_id"] = pd.to_numeric(
        weather["taxi_zone_id"],
        errors="coerce",
    ).astype("Int64")

    _progress(
        "      Source normalization complete in "
        f"{_format_seconds(time.perf_counter() - normalize_started)}"
    )

    _progress("      Building authoritative Taxi Zone lookup...")

    lookup_started = time.perf_counter()

    zone_lookup = (
        mobility[
            [
                "taxi_zone_id",
                "zone",
                "borough",
            ]
        ]
        .dropna(
            subset=[
                "taxi_zone_id",
                "zone",
            ]
        )
        .drop_duplicates(
            subset=["taxi_zone_id"]
        )
        .sort_values(
            [
                "borough",
                "zone",
                "taxi_zone_id",
            ]
        )
        .reset_index(drop=True)
    )

    _progress(
        "      Taxi Zone lookup: "
        f"{len(zone_lookup):,} zones in "
        f"{_format_seconds(time.perf_counter() - lookup_started)}"
    )

    _progress("      Attaching geography to weather source...")

    merge_started = time.perf_counter()

    weather = weather.merge(
        zone_lookup,
        on="taxi_zone_id",
        how="left",
        validate="many_to_one",
    )

    _progress(
        "      Weather geography attached in "
        f"{_format_seconds(time.perf_counter() - merge_started)}"
    )

    return mobility, weather, zone_lookup


# =====================================================================
# 5. VECTORIZED MOBILITY AGGREGATION
# =====================================================================

def _aggregate_mobility(
    data: pd.DataFrame,
    group_columns: list[str],
    *,
    label: str,
) -> pd.DataFrame:
    """
    Aggregate mobility without groupby.apply().

    Count metrics are summed.

    Speed metrics use:

        weighted speed
        = sum(speed × valid weight)
          / sum(valid weight)

    WHY
    ---
    The previous implementation invoked Python once per group through
    groupby.apply(). That is acceptable for a few Citywide groups but
    becomes unnecessarily expensive at Taxi Zone grain.

    This implementation creates weighted numerators once and performs
    vectorized grouped sums.
    """
    started = time.perf_counter()

    _progress(
        f"          {label}: mobility aggregation starting "
        f"({len(data):,} source rows)"
    )

    required_columns = list(
        dict.fromkeys(
            [
                *group_columns,
                *COUNT_METRICS,
                *SPEED_WEIGHT_COLUMNS.keys(),
                *SPEED_WEIGHT_COLUMNS.values(),
            ]
        )
    )

    working = data[required_columns].copy()

    _progress(
        f"          {label}: preparing weighted-speed numerators..."
    )

    numerator_columns = {}
    denominator_columns = {}

    for speed_metric, weight_column in (
        SPEED_WEIGHT_COLUMNS.items()
    ):
        numerator_column = (
            f"__numerator_{speed_metric}"
        )
        denominator_column = (
            f"__denominator_{speed_metric}"
        )

        values = pd.to_numeric(
            working[speed_metric],
            errors="coerce",
        )

        weights = pd.to_numeric(
            working[weight_column],
            errors="coerce",
        )

        valid = (
            values.notna()
            & weights.notna()
            & weights.gt(0)
        )

        working[numerator_column] = np.where(
            valid,
            values * weights,
            0.0,
        )

        working[denominator_column] = np.where(
            valid,
            weights,
            0.0,
        )

        numerator_columns[
            speed_metric
        ] = numerator_column

        denominator_columns[
            speed_metric
        ] = denominator_column

    aggregation_columns = [
        *COUNT_METRICS,
        *numerator_columns.values(),
        *denominator_columns.values(),
    ]

    _progress(
        f"          {label}: grouping by "
        f"{', '.join(group_columns)}..."
    )

    group_started = time.perf_counter()

    grouped = (
        working
        .groupby(
            group_columns,
            observed=True,
            as_index=False,
            sort=False,
        )[aggregation_columns]
        .sum(min_count=1)
    )

    _progress(
        f"          {label}: groupby complete — "
        f"{len(grouped):,} groups in "
        f"{_format_seconds(time.perf_counter() - group_started)}"
    )

    _progress(
        f"          {label}: calculating weighted speeds..."
    )

    for speed_metric in SPEED_WEIGHT_COLUMNS:
        numerator_column = numerator_columns[
            speed_metric
        ]

        denominator_column = denominator_columns[
            speed_metric
        ]

        denominator = grouped[
            denominator_column
        ]

        grouped[speed_metric] = np.where(
            denominator.gt(0),
            grouped[numerator_column]
            / denominator,
            np.nan,
        )

    helper_columns = [
        *numerator_columns.values(),
        *denominator_columns.values(),
    ]

    grouped = grouped.drop(
        columns=helper_columns
    )

    _progress(
        f"          ✓ {label}: mobility complete — "
        f"{len(grouped):,} rows in "
        f"{_format_seconds(time.perf_counter() - started)}"
    )

    return grouped


# =====================================================================
# 6. WEATHER AGGREGATION
# =====================================================================

def _aggregate_weather(
    data: pd.DataFrame,
    group_columns: list[str],
    *,
    label: str,
) -> pd.DataFrame:
    """Average weather measures at the requested geography/time grain."""
    started = time.perf_counter()

    _progress(
        f"          {label}: weather aggregation starting "
        f"({len(data):,} source rows)"
    )

    _progress(
        f"          {label}: grouping by "
        f"{', '.join(group_columns)}..."
    )

    group_started = time.perf_counter()

    result = (
        data
        .groupby(
            group_columns,
            observed=True,
            as_index=False,
            sort=False,
        )[WEATHER_METRICS]
        .mean()
    )

    _progress(
        f"          {label}: groupby complete — "
        f"{len(result):,} groups in "
        f"{_format_seconds(time.perf_counter() - group_started)}"
    )

    _progress(
        f"          ✓ {label}: weather complete in "
        f"{_format_seconds(time.perf_counter() - started)}"
    )

    return result


# =====================================================================
# 7. ONE GEOGRAPHY / TEMPORAL SURFACE
# =====================================================================

def _build_surface(
    mobility: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    geography_level: str,
    geography_column: str | None,
    temporal_mode: str,
) -> pd.DataFrame:
    """
    Build one geography × temporal-mode surface.

    temporal_mode
    -------------
    "bucketed"
        Preserve the ten named temporal buckets.

    "overall"
        Collapse all temporal buckets into one daily Overall observation.
    """
    if temporal_mode not in {
        "bucketed",
        "overall",
    }:
        raise ValueError(
            f"Unsupported temporal mode: {temporal_mode}"
        )

    started = time.perf_counter()

    temporal_label = (
        "named buckets"
        if temporal_mode == "bucketed"
        else "Overall"
    )

    label = (
        f"{geography_level} · {temporal_label}"
    )

    _progress(
        f"      → {label} starting..."
    )

    if temporal_mode == "overall":
        # Work on shallow copies because temporal_bucket is overwritten.
        mobility_work = mobility.copy()
        weather_work = weather.copy()

        mobility_work["temporal_bucket"] = "all"
        weather_work["temporal_bucket"] = "all"

    else:
        mobility_work = mobility
        weather_work = weather

    if geography_column is None:
        group_columns = [
            "date",
            "temporal_bucket",
        ]
    else:
        group_columns = [
            geography_column,
            "date",
            "temporal_bucket",
        ]

    mobility_daily = _aggregate_mobility(
        mobility_work,
        group_columns,
        label=label,
    )

    weather_daily = _aggregate_weather(
        weather_work,
        group_columns,
        label=label,
    )

    _progress(
        f"          {label}: merging "
        f"{len(mobility_daily):,} mobility + "
        f"{len(weather_daily):,} weather rows..."
    )

    merge_started = time.perf_counter()

    result = mobility_daily.merge(
        weather_daily,
        on=group_columns,
        how="inner",
        validate="one_to_one",
    )

    _progress(
        f"          {label}: merge complete — "
        f"{len(result):,} matched rows in "
        f"{_format_seconds(time.perf_counter() - merge_started)}"
    )

    result["geography_level"] = geography_level

    if geography_level == "Citywide":
        result["geography_id"] = "citywide"
        result["geography_name"] = "Citywide"
        result["taxi_zone_id"] = pd.NA
        result["zone"] = pd.NA
        result["borough"] = pd.NA

    elif geography_level == "Borough":
        result["geography_id"] = (
            "borough:"
            + result["borough"]
            .astype(str)
            .str.lower()
            .str.replace(
                " ",
                "_",
                regex=False,
            )
        )

        result["geography_name"] = (
            result["borough"]
        )

        result["taxi_zone_id"] = pd.NA
        result["zone"] = pd.NA

    elif geography_level == "Taxi Zone":
        result["taxi_zone_id"] = pd.to_numeric(
            result["taxi_zone_id"],
            errors="coerce",
        ).astype("Int64")

        result["geography_id"] = (
            "zone:"
            + result["taxi_zone_id"].astype(
                "string"
            )
        )

        # Human-readable names are attached from the authoritative lookup
        # after the complete Taxi Zone surface is assembled.
        result["geography_name"] = pd.NA
        result["zone"] = pd.NA
        result["borough"] = pd.NA

    else:
        raise ValueError(
            "Unsupported geography level: "
            f"{geography_level}"
        )

    result["pre_post_cp"] = np.where(
        result["date"].lt(CP_START_DATE),
        "pre_cp",
        "post_cp",
    )

    _progress(
        f"      ✓ {label} complete — "
        f"{len(result):,} rows in "
        f"{_format_seconds(time.perf_counter() - started)}"
    )

    return result


# =====================================================================
# 8. COMPLETE GEOGRAPHY LEVEL
# =====================================================================

def _build_geography_level(
    mobility: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    geography_level: str,
    geography_column: str | None,
) -> pd.DataFrame:
    """
    Build both named temporal buckets and Overall for one geography level.
    """
    started = time.perf_counter()

    _progress(
        f"      {geography_level}: "
        f"{len(mobility):,} mobility rows + "
        f"{len(weather):,} weather rows"
    )

    bucketed = _build_surface(
        mobility,
        weather,
        geography_level=geography_level,
        geography_column=geography_column,
        temporal_mode="bucketed",
    )

    _progress(
        f"      {geography_level}: named buckets finished; "
        "starting Overall aggregation..."
    )

    overall = _build_surface(
        mobility,
        weather,
        geography_level=geography_level,
        geography_column=geography_column,
        temporal_mode="overall",
    )

    _progress(
        f"      {geography_level}: concatenating "
        f"{len(bucketed):,} bucketed + "
        f"{len(overall):,} Overall rows..."
    )

    result = pd.concat(
        [
            bucketed,
            overall,
        ],
        ignore_index=True,
    )

    _progress(
        f"      ✓ {geography_level}: complete — "
        f"{len(result):,} total rows in "
        f"{_format_seconds(time.perf_counter() - started)}"
    )

    return result


# =====================================================================
# 9. TAXI ZONE LABELS
# =====================================================================

def _attach_taxi_zone_labels(
    taxi_zone: pd.DataFrame,
    zone_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Attach authoritative Taxi Zone and borough names."""
    started = time.perf_counter()

    _progress(
        "      Attaching Taxi Zone names and boroughs..."
    )

    result = taxi_zone.drop(
        columns=[
            "zone",
            "borough",
        ],
        errors="ignore",
    ).merge(
        zone_lookup,
        on="taxi_zone_id",
        how="left",
        validate="many_to_one",
    )

    result["geography_name"] = (
        result["zone"].fillna(
            "Taxi Zone "
            + result["taxi_zone_id"].astype(
                "string"
            )
        )
        + " · "
        + result["borough"].fillna("Unknown")
    )

    _progress(
        "      ✓ Taxi Zone labels attached in "
        f"{_format_seconds(time.perf_counter() - started)}"
    )

    return result

# =====================================================================
# SHOWCASE QUANTIZATION
# =====================================================================

def _quantize_showcase_measures(
    daily: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Quantize deployment measures after analytical aggregation.

    The canonical analytical source remains untouched. This reduces
    Parquet cardinality and file size while retaining substantially more
    precision than Raw 09 needs for visualization.

    Returns
    -------
    quantized
        App-ready daily surface.

    precision_qa
        Per-metric precision contract and observed rounding error.
    """
    started = time.perf_counter()

    _progress(
        "      Quantizing app-ready measures..."
    )

    quantized = daily.copy()
    qa_rows = []

    for metric, decimals in DISPLAY_PRECISION.items():
        if metric not in quantized.columns:
            raise KeyError(
                "Quantization contract references missing column: "
                f"{metric}"
            )

        original = pd.to_numeric(
            quantized[metric],
            errors="coerce",
        )

        rounded = original.round(decimals)

        finite = (
            original.notna()
            & rounded.notna()
        )

        if finite.any():
            absolute_error = (
                original.loc[finite]
                - rounded.loc[finite]
            ).abs()

            max_absolute_error = float(
                absolute_error.max()
            )

            mean_absolute_error = float(
                absolute_error.mean()
            )

            changed_rows = int(
                original.loc[finite]
                .ne(rounded.loc[finite])
                .sum()
            )

        else:
            max_absolute_error = np.nan
            mean_absolute_error = np.nan
            changed_rows = 0

        quantized[metric] = rounded

        qa_rows.append(
            {
                "metric": metric,
                "decimal_places": decimals,
                "non_null_rows": int(
                    finite.sum()
                ),
                "changed_rows": changed_rows,
                "max_absolute_rounding_error": (
                    max_absolute_error
                ),
                "mean_absolute_rounding_error": (
                    mean_absolute_error
                ),
            }
        )

        _progress(
            f"          {metric:<25} "
            f"→ {decimals} decimal"
            f"{'' if decimals == 1 else 's'} "
            f"· max Δ "
            f"{max_absolute_error:,.6f}"
        )

    precision_qa = pd.DataFrame(
        qa_rows
    )

    _progress(
        "      ✓ Quantization complete in "
        f"{_format_seconds(time.perf_counter() - started)}"
    )

    return quantized, precision_qa

# =====================================================================
# 10. QA
# =====================================================================

def _build_qa(
    daily: pd.DataFrame,
    precision_qa: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate production grain and record deployment precision.

    The QA artifact contains both geography-level coverage checks and
    the deliberate numerical precision contract used by Raw 09.
    """
    started = time.perf_counter()

    _progress(
        f"      Validating {len(daily):,} combined rows..."
    )

    duplicate_key = [
        "geography_id",
        "date",
        "temporal_bucket",
    ]

    duplicate_count = int(
        daily.duplicated(
            subset=duplicate_key
        ).sum()
    )

    _progress(
        "      Duplicate geography/date/bucket keys: "
        f"{duplicate_count:,}"
    )

    if duplicate_count:
        raise RuntimeError(
            "Weather relationship app table contains "
            f"{duplicate_count:,} duplicate "
            "geography/date/bucket rows."
        )

    boroughs_found = sorted(
        daily.loc[
            daily["geography_level"].eq(
                "Borough"
            ),
            "borough",
        ]
        .dropna()
        .unique()
        .tolist()
    )

    _progress(
        "      Borough coverage: "
        + ", ".join(boroughs_found)
    )

    if boroughs_found != sorted(
        VALID_BOROUGHS
    ):
        raise RuntimeError(
            "Unexpected borough coverage. "
            f"Expected {sorted(VALID_BOROUGHS)}, "
            f"found {boroughs_found}."
        )

    qa_rows = []

    for geography_level, frame in daily.groupby(
        "geography_level",
        observed=True,
    ):
        qa_rows.append(
            {
                "qa_type": "geography",
                "item": geography_level,
                "rows": len(frame),
                "geographies": frame[
                    "geography_id"
                ].nunique(),
                "minimum_date": frame[
                    "date"
                ].min(),
                "maximum_date": frame[
                    "date"
                ].max(),
                "temporal_buckets": frame[
                    "temporal_bucket"
                ].nunique(),
                "duplicate_keys": int(
                    frame.duplicated(
                        subset=duplicate_key
                    ).sum()
                ),
                "decimal_places": np.nan,
                "max_absolute_rounding_error": np.nan,
                "mean_absolute_rounding_error": np.nan,
            }
        )

    for row in precision_qa.itertuples(
        index=False
    ):
        qa_rows.append(
            {
                "qa_type": "precision",
                "item": row.metric,
                "rows": row.non_null_rows,
                "geographies": np.nan,
                "minimum_date": pd.NaT,
                "maximum_date": pd.NaT,
                "temporal_buckets": np.nan,
                "duplicate_keys": np.nan,
                "decimal_places": (
                    row.decimal_places
                ),
                "max_absolute_rounding_error": (
                    row.max_absolute_rounding_error
                ),
                "mean_absolute_rounding_error": (
                    row.mean_absolute_rounding_error
                ),
            }
        )

    qa = pd.DataFrame(
        qa_rows
    )

    _progress(
        "      ✓ QA complete in "
        f"{_format_seconds(time.perf_counter() - started)}"
    )

    return qa


# =====================================================================
# 11. MAIN BUILD
# =====================================================================

def main() -> None:
    """Build Raw 09's compact deployment surfaces."""
    print("=" * 76)
    print("Raw 09 · Weather relationship app-table build")
    print("Optimized vectorized aggregation + transparent progress")
    print("=" * 76)
    print(flush=True)

    _progress("Build started.")

    _require_file(MOBILITY_PANEL_PATH)
    _require_file(WEATHER_PANEL_PATH)

    APP_TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Stage 1 — load sources
    # -----------------------------------------------------------------
    stage_started = _stage(
        1,
        8,
        "Load and prepare authoritative sources",
    )

    mobility, weather, zone_lookup = (
        _prepare_sources()
    )

    _finish_stage(
        stage_started,
        "Source preparation complete",
    )

    # Restrict Borough analysis to the five named NYC boroughs.
    # Citywide and Taxi Zone retain the valid source population.
    borough_mobility = mobility.loc[
        mobility["borough"].isin(
            VALID_BOROUGHS
        )
    ].copy()

    borough_weather = weather.loc[
        weather["borough"].isin(
            VALID_BOROUGHS
        )
    ].copy()

    # -----------------------------------------------------------------
    # Stage 2 — Citywide
    # -----------------------------------------------------------------
    stage_started = _stage(
        2,
        8,
        "Build Citywide surface",
    )

    citywide = _build_geography_level(
        mobility,
        weather,
        geography_level="Citywide",
        geography_column=None,
    )

    _finish_stage(
        stage_started,
        f"Citywide complete: {len(citywide):,} rows",
    )

    # -----------------------------------------------------------------
    # Stage 3 — Borough
    # -----------------------------------------------------------------
    stage_started = _stage(
        3,
        8,
        "Build Borough surface",
    )

    borough = _build_geography_level(
        borough_mobility,
        borough_weather,
        geography_level="Borough",
        geography_column="borough",
    )

    _finish_stage(
        stage_started,
        f"Borough complete: {len(borough):,} rows",
    )

    # -----------------------------------------------------------------
    # Stage 4 — Taxi Zone
    # -----------------------------------------------------------------
    stage_started = _stage(
        4,
        8,
        "Build Taxi Zone surface",
    )

    _progress(
        "      This is the largest aggregation stage. "
        "Progress will be reported for mobility, weather, merge, "
        "named buckets, Overall, and geography labels."
    )

    taxi_zone = _build_geography_level(
        mobility,
        weather,
        geography_level="Taxi Zone",
        geography_column="taxi_zone_id",
    )

    taxi_zone = _attach_taxi_zone_labels(
        taxi_zone,
        zone_lookup,
    )

    _finish_stage(
        stage_started,
        f"Taxi Zone complete: {len(taxi_zone):,} rows",
    )

    # -----------------------------------------------------------------
    # Stage 5 — combine + validate
    # -----------------------------------------------------------------
    stage_started = _stage(
        5,
        8,
        "Combine geography levels and validate",
    )

    _progress(
        "      Concatenating Citywide, Borough, and Taxi Zone..."
    )

    daily = pd.concat(
        [
            citywide,
            borough,
            taxi_zone,
        ],
        ignore_index=True,
    )

    _progress(
        f"      Combined surface: {len(daily):,} rows"
    )

    ordered_columns = [
        "date",
        "pre_post_cp",
        "geography_level",
        "geography_id",
        "geography_name",
        "taxi_zone_id",
        "zone",
        "borough",
        "temporal_bucket",
        *MOBILITY_METRICS,
        *WEATHER_METRICS,
    ]

    daily = daily[
        ordered_columns
    ].copy()

    _progress(
        "      Sorting production surface..."
    )

    sort_started = time.perf_counter()

    daily = (
        daily
        .sort_values(
            [
                "geography_level",
                "geography_name",
                "date",
                "temporal_bucket",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    _progress(
        "      Sort complete in "
        f"{_format_seconds(time.perf_counter() - sort_started)}"
    )

    _finish_stage(
        stage_started,
        "Combination and QA complete",
    )

    # -----------------------------------------------------------------
    # Stage 6 — geography lookup
    # -----------------------------------------------------------------
    stage_started = _stage(
        6,
        8,
        "Build compact geography lookup",
    )

    geography = (
        daily[
            [
                "geography_level",
                "geography_id",
                "geography_name",
                "taxi_zone_id",
                "zone",
                "borough",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "geography_level",
                "borough",
                "geography_name",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    _progress(
        f"      Geography lookup contains "
        f"{len(geography):,} rows."
    )

    _finish_stage(
        stage_started,
        "Geography lookup complete",
    )

    # -----------------------------------------------------------------
    # Stage 7 — quantize deployment measures + final QA
    # -----------------------------------------------------------------
    stage_started = _stage(
        7,
        8,
        "Quantize Showcase measures and finalize QA",
    )

    # TEMPORARY QA ARTIFACT:
    # Preserve the exact pre-quantization Borough surface so validation
    # isolates rounding from every other part of the build pipeline.
    validation_path = (
        APP_TABLE_DIR
        / "_weather_relationship_borough_full_precision.parquet"
    )

    daily.loc[
        daily["geography_level"].eq("Borough")
    ].to_parquet(
        validation_path,
        index=False,
        compression="zstd",
    )

    _progress(
        "      Temporary full-precision Borough QA surface written: "
        f"{_file_size_mb(validation_path):.2f} MB"
    )

    daily, precision_qa = (
        _quantize_showcase_measures(
            daily
        )
    )

    qa = _build_qa(
        daily,
        precision_qa,
    )

    _finish_stage(
        stage_started,
        "Quantization and final QA complete",
    )

    # -----------------------------------------------------------------
    # Stage 8 — writes
    # -----------------------------------------------------------------
    stage_started = _stage(
        8,
        8,
        "Write deployment artifacts",
    )

    _progress(
        f"      Writing daily surface "
        f"({len(daily):,} rows) with Zstandard compression..."
    )

    write_started = time.perf_counter()

    daily.to_parquet(
        DAILY_OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    _progress(
        "      ✓ Daily surface written in "
        f"{_format_seconds(time.perf_counter() - write_started)} "
        f"· {_file_size_mb(DAILY_OUTPUT_PATH):.2f} MB"
    )

    _progress(
        f"      Writing geography lookup "
        f"({len(geography):,} rows)..."
    )

    write_started = time.perf_counter()

    geography.to_parquet(
        GEOGRAPHY_OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    _progress(
        "      ✓ Geography lookup written in "
        f"{_format_seconds(time.perf_counter() - write_started)} "
        f"· {_file_size_mb(GEOGRAPHY_OUTPUT_PATH):.2f} MB"
    )

    _progress(
        f"      Writing QA table ({len(qa):,} rows)..."
    )

    write_started = time.perf_counter()

    qa.to_parquet(
        QA_OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    _progress(
        "      ✓ QA table written in "
        f"{_format_seconds(time.perf_counter() - write_started)} "
        f"· {_file_size_mb(QA_OUTPUT_PATH):.2f} MB"
    )

    _finish_stage(
        stage_started,
        "All deployment artifacts written",
    )

    # -----------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------
    total_elapsed = _elapsed_total()

    # WHY: The QA artifact now contains two different record types.
    # Keep geography coverage and quantization reporting separate so the
    # final build summary remains easy to read.
    geography_qa = (
        qa.loc[
            qa["qa_type"].eq("geography")
        ]
        .copy()
    )

    print(flush=True)
    print("=" * 76)
    print("BUILD COMPLETE")
    print("=" * 76)

    print(
        f"Daily relationship table : "
        f"{len(daily):,} rows · "
        f"{_file_size_mb(DAILY_OUTPUT_PATH):.2f} MB"
    )

    print(
        f"Geography lookup         : "
        f"{len(geography):,} rows · "
        f"{_file_size_mb(GEOGRAPHY_OUTPUT_PATH):.2f} MB"
    )

    print(
        f"Build QA                 : "
        f"{len(qa):,} rows · "
        f"{_file_size_mb(QA_OUTPUT_PATH):.2f} MB"
    )

    print(
        f"Date coverage            : "
        f"{daily['date'].min().date()} → "
        f"{daily['date'].max().date()}"
    )

    print(
        f"Unique geographies       : "
        f"{daily['geography_id'].nunique():,}"
    )

    print(
        f"Temporal buckets         : "
        f"{daily['temporal_bucket'].nunique():,}"
    )

    print(
        "Duplicate keys           : "
        f"{int(geography_qa['duplicate_keys'].sum()):,}"
    )

    print(
        "Total runtime            : "
        f"{_format_seconds(total_elapsed)}"
    )

    print()
    print("QA by geography level:")
    print(
        geography_qa.to_string(
            index=False,
        )
    )

    print()
    print("Showcase precision contract:")
    print(
        precision_qa[
            [
                "metric",
                "decimal_places",
                "non_null_rows",
                "changed_rows",
                "max_absolute_rounding_error",
                "mean_absolute_rounding_error",
            ]
        ].to_string(
            index=False,
        )
    )

    print()
    print("Output files:")
    print(f"  {DAILY_OUTPUT_PATH}")
    print(f"  {GEOGRAPHY_OUTPUT_PATH}")
    print(f"  {QA_OUTPUT_PATH}")
    print("=" * 76)


if __name__ == "__main__":
    main()