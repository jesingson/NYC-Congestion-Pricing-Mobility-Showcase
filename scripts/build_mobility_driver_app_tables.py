"""Build compact production tables for Raw 27 · Mobility Drivers.

Run once from the repository root:

    python scripts/build_mobility_driver_app_tables.py

The expensive work stays offline. The Streamlit page reads standardized
geography × date × daypart signals and only performs cheap filtering,
daypart-combination, and smoothing.

Supported Showcase geography levels:
- Citywide
- Taxi Zone
- Borough
- Mobility Environment
- Policy Geography

IMPORTANT:
Aggregate raw mobility measures FIRST, then establish each geography's own
same-weekday/daypart baseline. Do not average already-standardized zone scores.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE_PATH = (
    REPO_ROOT
    / "data"
    / "processed"
    / "1.3.1.final_tables"
    / "analysis_ready_mobility_panel.parquet"
)

CLUSTER_ASSIGNMENTS_PATH = (
    REPO_ROOT
    / "data"
    / "processed"
    / "3.2.2.final_tables"
    / "canonical_cluster_assignments-20260803-193800.parquet"
)

APP_TABLE_DIR = (
    REPO_ROOT
    / "data"
    / "processed"
    / "app_tables"
)

OUTPUT_PATH = (
    APP_TABLE_DIR
    / "mobility_driver_geographies.parquet"
)

CONTEXT_PATH = (
    APP_TABLE_DIR
    / "mobility_driver_geography_context.parquet"
)

QA_PATH = (
    APP_TABLE_DIR
    / "mobility_driver_qa.parquet"
)


# ---------------------------------------------------------------------
# Analytical contract
# ---------------------------------------------------------------------

CP_START = pd.Timestamp("2025-01-05")

DAYPART_ORDER = (
    "overnight",
    "am_peak",
    "midday",
    "pm_peak",
    "evening",
)

GEOGRAPHY_LEVEL_ORDER = (
    "Citywide",
    "Taxi Zone",
    "Borough",
    "Mobility Environment",
    "Policy Geography",
)

METRICS = (
    "taxi_trip_count",
    "taxi_avg_trip_speed",
    "fhvhv_trip_count",
    "fhvhv_avg_trip_speed",
    "subway_ridership",
    "avg_bus_speed",
)

COUNT_METRICS = (
    "taxi_trip_count",
    "fhvhv_trip_count",
    "subway_ridership",
)

SPEED_WEIGHTS = {
    "taxi_avg_trip_speed": "taxi_trip_count",
    "fhvhv_avg_trip_speed": "fhvhv_trip_count",
    "avg_bus_speed": "bus_trip_count",
}

POLICY_LABELS = {
    "cbd": "CBD",
    "CBD": "CBD",
    "gateway_to_cbd": "Gateway",
    "gateway": "Gateway",
    "Gateway": "Gateway",
    "adjacent_to_cbd": "Adjacent",
    "adjacent": "Adjacent",
    "Adjacent": "Adjacent",
    "non_cbd": "Non-CBD",
    "non-cbd": "Non-CBD",
    "Non-CBD": "Non-CBD",
    "Non CBD": "Non-CBD",
}


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def _require_file(path: Path) -> None:
    """Fail early when an authoritative input is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required source file:\n{path}"
        )


def _print_step(
    label: str,
    rows: int,
    started_at: float,
) -> None:
    """Print a compact timing line for one completed build step."""
    elapsed = perf_counter() - started_at

    print(
        f"  {label:<24}"
        f"{rows:>12,} rows · "
        f"{elapsed:>7.2f}s"
    )


# ---------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------

def _load_source() -> pd.DataFrame:
    """Read only the columns required by Raw 27."""
    _require_file(SOURCE_PATH)

    frame = pd.read_parquet(
        SOURCE_PATH,
        columns=[
            "taxi_zone_id",
            "zone",
            "borough",
            "cbd_spatial_category",
            "date",
            "daypart",
            *METRICS,
            "bus_trip_count",
        ],
    )

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="raise",
    ).dt.normalize()

    frame["taxi_zone_id"] = pd.to_numeric(
        frame["taxi_zone_id"],
        errors="coerce",
    ).astype("Int64")

    frame["daypart"] = (
        frame["daypart"]
        .astype(str)
    )

    frame["pre_post_cp"] = np.where(
        frame["date"].lt(CP_START),
        "pre_cp",
        "post_cp",
    )

    return frame.loc[
        frame["daypart"].isin(DAYPART_ORDER)
        & frame["taxi_zone_id"].notna()
    ].copy()


def _load_environment_assignments() -> pd.DataFrame:
    """Load the same period-specific Mobility Environment mapping as Raw 16."""
    _require_file(CLUSTER_ASSIGNMENTS_PATH)

    assignments = pd.read_parquet(
        CLUSTER_ASSIGNMENTS_PATH,
        columns=[
            "taxi_zone_id",
            "pre_post_cp",
            "canonical_cluster_name",
        ],
    ).drop_duplicates()

    assignments["taxi_zone_id"] = pd.to_numeric(
        assignments["taxi_zone_id"],
        errors="coerce",
    ).astype("Int64")

    assignments["pre_post_cp"] = (
        assignments["pre_post_cp"]
        .astype(str)
    )

    duplicate_count = int(
        assignments.duplicated(
            [
                "taxi_zone_id",
                "pre_post_cp",
            ]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            "Mobility Environment assignments are not unique at "
            "Taxi Zone × policy period."
        )

    return assignments.rename(
        columns={
            "canonical_cluster_name":
                "mobility_environment",
        }
    )


def _attach_context(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Attach canonical policy and period-specific environment labels."""
    out = panel.merge(
        _load_environment_assignments(),
        on=[
            "taxi_zone_id",
            "pre_post_cp",
        ],
        how="left",
        validate="many_to_one",
    )

    raw_policy = (
        out["cbd_spatial_category"]
        .astype(str)
    )

    out["policy_geography"] = (
        raw_policy
        .map(POLICY_LABELS)
        .fillna(raw_policy)
    )

    return out


# ---------------------------------------------------------------------
# Vectorized geography aggregation
# ---------------------------------------------------------------------

def _prepare_weighted_components(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Create numerator/denominator columns for vectorized speed aggregation.

    WHY:
    A geography's average speed must be activity weighted. Precomputing
    speed × activity lets Pandas sum both pieces in one grouped operation.
    """
    out = panel.copy()

    for metric, weight_col in SPEED_WEIGHTS.items():
        values = pd.to_numeric(
            out[metric],
            errors="coerce",
        )

        weights = (
            pd.to_numeric(
                out[weight_col],
                errors="coerce",
            )
            .clip(lower=0)
        )

        valid = (
            values.notna()
            & weights.notna()
            & weights.gt(0)
        )

        numerator_col = (
            f"__weighted_numerator__{metric}"
        )
        denominator_col = (
            f"__weighted_denominator__{metric}"
        )

        out[numerator_col] = np.where(
            valid,
            values * weights,
            0.0,
        )

        out[denominator_col] = np.where(
            valid,
            weights,
            0.0,
        )

    return out


def _aggregate_geography(
    panel: pd.DataFrame,
    *,
    geography_level: str,
    geography_column: str | None,
) -> pd.DataFrame:
    """Aggregate one entire geography level in a single Pandas groupby.

    Counts sum across the selected geography.

    Speeds use:
        sum(speed × activity) / sum(activity)

    This replaces the previous Python loop over every Taxi Zone and every
    date/daypart group.
    """
    if geography_column is None:
        working = panel.assign(
            __geography_value="NYC"
        )
        geography_column = "__geography_value"
    else:
        working = panel.loc[
            panel[geography_column].notna()
        ].copy()

    group_cols = [
        geography_column,
        "date",
        "daypart",
    ]

    sum_columns = [
        *COUNT_METRICS,
    ]

    for metric in SPEED_WEIGHTS:
        sum_columns.extend(
            [
                f"__weighted_numerator__{metric}",
                f"__weighted_denominator__{metric}",
            ]
        )

    grouped = (
        working
        .groupby(
            group_cols,
            observed=True,
            sort=False,
            dropna=True,
        )[sum_columns]
        .sum(min_count=1)
        .reset_index()
    )

    for metric in SPEED_WEIGHTS:
        numerator = grouped[
            f"__weighted_numerator__{metric}"
        ]

        denominator = grouped[
            f"__weighted_denominator__{metric}"
        ]

        grouped[metric] = (
            numerator
            .div(
                denominator.replace(
                    0,
                    np.nan,
                )
            )
        )

    grouped = grouped.rename(
        columns={
            geography_column:
                "geography_value",
        }
    )

    grouped[
        "geography_level"
    ] = geography_level

    grouped[
        "geography_value"
    ] = (
        grouped["geography_value"]
        .astype(str)
    )

    return grouped[
        [
            "geography_level",
            "geography_value",
            "date",
            "daypart",
            *METRICS,
        ]
    ]


def _build_all_geographies(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Build every supported Showcase geography with five bulk groupbys."""
    pieces: list[pd.DataFrame] = []

    geography_specs = (
        (
            "Citywide",
            None,
        ),
        (
            "Taxi Zone",
            "taxi_zone_id",
        ),
        (
            "Borough",
            "borough",
        ),
        (
            "Mobility Environment",
            "mobility_environment",
        ),
        (
            "Policy Geography",
            "policy_geography",
        ),
    )

    for (
        geography_level,
        geography_column,
    ) in geography_specs:
        started_at = perf_counter()

        result = _aggregate_geography(
            panel,
            geography_level=(
                geography_level
            ),
            geography_column=(
                geography_column
            ),
        )

        pieces.append(result)

        _print_step(
            geography_level,
            len(result),
            started_at,
        )

    return (
        pd.concat(
            pieces,
            ignore_index=True,
        )
        .sort_values(
            [
                "geography_level",
                "geography_value",
                "date",
                "daypart",
            ]
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Geography-specific normalization
# ---------------------------------------------------------------------

def _standardize(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Express each geography relative to its own normal weekday/daypart.

    Counts are log1p transformed before normalization.

    Each signal is centered against the median for the same:
        geography × weekday × daypart

    and divided by that group's IQR.

    Speed signs are inverted so the vertical interpretation is consistent:
        positive = more demand and/or slower roads
        negative = less demand and/or faster roads
    """
    out = frame.copy()

    out["weekday"] = (
        out["date"]
        .dt.dayofweek
        .astype("int8")
    )

    group_keys = [
        "geography_level",
        "geography_value",
        "weekday",
        "daypart",
    ]

    for metric in METRICS:
        started_at = perf_counter()

        values = pd.to_numeric(
            out[metric],
            errors="coerce",
        )

        if metric in COUNT_METRICS:
            values = np.log1p(
                values.clip(lower=0)
            )

        # Temporary Series keeps the original row index, allowing transform
        # results to align directly back to the production frame.
        working = out[
            group_keys
        ].copy()

        working["__value"] = values

        grouped = (
            working
            .groupby(
                group_keys,
                observed=True,
                sort=False,
            )["__value"]
        )

        median = grouped.transform(
            "median"
        )

        # Pandas quantile is still vectorized at the grouped-series level;
        # importantly, there is no Python lambda per geography/date group.
        q25 = grouped.transform(
            "quantile",
            q=0.25,
        )

        q75 = grouped.transform(
            "quantile",
            q=0.75,
        )

        iqr = (
            q75
            - q25
        ).replace(
            0,
            np.nan,
        )

        score = (
            values
            - median
        ).div(iqr)

        if metric in SPEED_WEIGHTS:
            score *= -1

        out[metric] = (
            score
            .astype("float32")
        )

        elapsed = (
            perf_counter()
            - started_at
        )

        print(
            f"    {metric:<24}"
            f"{elapsed:>7.2f}s"
        )

    return out[
        [
            "geography_level",
            "geography_value",
            "date",
            "daypart",
            *METRICS,
        ]
    ]


# ---------------------------------------------------------------------
# Geography selector context
# ---------------------------------------------------------------------

def _build_context(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Build lightweight labels for the standard two-stage geography selector."""
    zone_context = (
        panel[
            [
                "taxi_zone_id",
                "zone",
                "borough",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "borough",
                "zone",
                "taxi_zone_id",
            ]
        )
        .copy()
    )

    zone_context[
        "geography_value"
    ] = (
        zone_context[
            "taxi_zone_id"
        ]
        .astype(int)
        .astype(str)
    )

    zone_context[
        "display_label"
    ] = (
        zone_context["zone"]
        .astype(str)
        + " · "
        + zone_context["borough"]
        .astype(str)
    )

    zone_context[
        "geography_level"
    ] = "Taxi Zone"

    pieces = [
        pd.DataFrame(
            [
                {
                    "geography_level":
                        "Citywide",
                    "geography_value":
                        "NYC",
                    "display_label":
                        "NYC",
                }
            ]
        ),
        zone_context[
            [
                "geography_level",
                "geography_value",
                "display_label",
            ]
        ],
    ]

    for (
        geography_level,
        column,
    ) in (
        (
            "Borough",
            "borough",
        ),
        (
            "Mobility Environment",
            "mobility_environment",
        ),
        (
            "Policy Geography",
            "policy_geography",
        ),
    ):
        values = (
            panel[column]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
        )

        pieces.append(
            pd.DataFrame(
                {
                    "geography_level":
                        geography_level,
                    "geography_value":
                        values,
                    "display_label":
                        values,
                }
            )
        )

    context = pd.concat(
        pieces,
        ignore_index=True,
    )

    context[
        "geography_level"
    ] = pd.Categorical(
        context[
            "geography_level"
        ],
        categories=(
            GEOGRAPHY_LEVEL_ORDER
        ),
        ordered=True,
    )

    return (
        context
        .drop_duplicates()
        .sort_values(
            [
                "geography_level",
                "display_label",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ---------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------

def _build_qa(
    standardized: pd.DataFrame,
    context: pd.DataFrame,
) -> pd.DataFrame:
    """Store cheap build-time checks for runtime recovery."""
    grain = [
        "geography_level",
        "geography_value",
        "date",
        "daypart",
    ]

    duplicate_count = int(
        standardized
        .duplicated(grain)
        .sum()
    )

    observed_levels = set(
        standardized[
            "geography_level"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    expected_levels = set(
        GEOGRAPHY_LEVEL_ORDER
    )

    zone_count = int(
        context.loc[
            context[
                "geography_level"
            ]
            .astype(str)
            .eq("Taxi Zone"),
            "geography_value",
        ]
        .nunique()
    )

    metric_count = sum(
        metric
        in standardized.columns
        for metric in METRICS
    )

    return pd.DataFrame(
        [
            {
                "check":
                    "taxi_zone_drilldown_present",
                "observed":
                    zone_count,
                "expected":
                    1,
                "passed":
                    zone_count > 0,
            },
            {
                "check":
                    "five_geography_levels",
                "observed":
                    len(observed_levels),
                "expected":
                    len(expected_levels),
                "passed":
                    observed_levels
                    == expected_levels,
            },
            {
                "check":
                    "six_driver_metrics",
                "observed":
                    metric_count,
                "expected":
                    len(METRICS),
                "passed":
                    metric_count
                    == len(METRICS),
            },
        ]
    )


# ---------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------

def main() -> None:
    """Build Raw 27's compact geography-aware production assets."""
    total_started_at = (
        perf_counter()
    )

    print(
        "Raw 27 · Mobility Drivers"
    )

    print(
        "Loading source panel and geography context..."
    )
    started_at = perf_counter()

    panel = _attach_context(
        _load_source()
    )

    _print_step(
        "Source + context",
        len(panel),
        started_at,
    )

    print(
        "\nPreparing weighted speed components..."
    )
    started_at = perf_counter()

    panel = (
        _prepare_weighted_components(
            panel
        )
    )

    _print_step(
        "Weighted components",
        len(panel),
        started_at,
    )

    print(
        "\nAggregating supported Showcase geographies..."
    )

    aggregated = (
        _build_all_geographies(
            panel
        )
    )

    print(
        "\nStandardizing each geography against "
        "its own same-weekday/daypart history..."
    )
    started_at = perf_counter()

    standardized = (
        _standardize(
            aggregated
        )
    )

    _print_step(
        "All standardized rows",
        len(standardized),
        started_at,
    )

    print(
        "\nBuilding geography selector context..."
    )
    started_at = perf_counter()

    context = (
        _build_context(
            panel
        )
    )

    _print_step(
        "Geography context",
        len(context),
        started_at,
    )

    print(
        "\nRunning QA..."
    )

    qa = _build_qa(
        standardized,
        context,
    )

    print(
        qa.to_string(
            index=False
        )
    )

    if not qa[
        "passed"
    ].all():
        raise RuntimeError(
            "Raw 27 build failed QA."
        )

    print(
        "\nWriting production artifacts..."
    )
    started_at = perf_counter()

    APP_TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    standardized.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    context.to_parquet(
        CONTEXT_PATH,
        index=False,
        compression="zstd",
    )

    qa.to_parquet(
        QA_PATH,
        index=False,
        compression="zstd",
    )

    write_elapsed = (
        perf_counter()
        - started_at
    )

    print(
        f"  Driver table            "
        f"{OUTPUT_PATH.stat().st_size / 1_048_576:>8.2f} MB"
    )
    print(
        f"  Geography context       "
        f"{CONTEXT_PATH.stat().st_size / 1_048_576:>8.2f} MB"
    )
    print(
        f"  QA table                "
        f"{QA_PATH.stat().st_size / 1_048_576:>8.2f} MB"
    )
    print(
        f"  Write time              "
        f"{write_elapsed:>8.2f}s"
    )

    zone_count = int(
        context.loc[
            context[
                "geography_level"
            ]
            .astype(str)
            .eq("Taxi Zone"),
            "geography_value",
        ]
        .nunique()
    )

    total_elapsed = (
        perf_counter()
        - total_started_at
    )

    print(
        "\nBuild complete."
    )
    print(
        f"Taxi Zones available: {zone_count:,}"
    )
    print(
        f"Serving rows: {len(standardized):,}"
    )
    print(
        f"Total runtime: {total_elapsed:.2f}s "
        f"({total_elapsed / 60:.2f} min)"
    )


if __name__ == "__main__":
    main()