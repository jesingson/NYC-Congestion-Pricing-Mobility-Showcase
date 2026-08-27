from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Repository setup
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

from app.data_access.loaders import (
    BASE_METRICS,
    CONGESTION_PRICING_START_DATE,
    METRIC_LABELS,
)
from app.data_access.spatial_aggregations import (
    ALL_TEMPORAL_BUCKETS_LABEL,
)
from app.data_access.zone_profile_app_tables import (
    ZONE_PROFILE_DAILY_DIR,
)


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "app_tables"
    / "mobility_pulse_diagnostics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------
# Analysis configuration
# ---------------------------------------------------------------------

CP_START = pd.Timestamp(
    CONGESTION_PRICING_START_DATE
)

GEOGRAPHIES = [
    "Citywide",
    "Manhattan",
    "Brooklyn",
    "Queens",
    "Bronx",
    "Staten Island",
]

REQUIRED_SHORTLIST_GEOGRAPHIES = [
    "Citywide",
    "Manhattan",
    "Brooklyn",
    "Queens",
]

TEMPORAL_BUCKETS = [
    ALL_TEMPORAL_BUCKETS_LABEL,
    "weekday_am_peak",
    "weekday_midday",
    "weekday_pm_peak",
    "weekday_evening",
    "weekday_overnight",
    "weekend_midday",
    "weekend_evening",
    "weekend_overnight",
]

WINDOWS = {
    "Full study period": (
        "2023-01-01",
        "2026-03-31",
    ),
    "CP launch transition": (
        "2024-11-01",
        "2025-04-30",
    ),
    "First six months post-CP": (
        "2025-01-05",
        "2025-06-30",
    ),
    "Second half of 2025": (
        "2025-07-01",
        "2025-12-31",
    ),
    "Early 2026": (
        "2026-01-01",
        "2026-03-31",
    ),
}


# ---------------------------------------------------------------------
# Modality metadata
# ---------------------------------------------------------------------

MODE_BY_METRIC = {
    "taxi_trip_count": "Taxi",
    "taxi_avg_trip_speed": "Taxi",
    "taxi_avg_trip_duration": "Taxi",

    "fhvhv_trip_count": "FHVHV",
    "fhvhv_avg_trip_speed": "FHVHV",
    "fhvhv_avg_trip_duration": "FHVHV",

    "subway_ridership": "Subway",
    "subway_transfers": "Subway",

    "bus_trip_count": "Bus",
    "avg_bus_speed": "Bus",
}

REQUIRED_MODES = [
    "Taxi",
    "FHVHV",
    "Subway",
    "Bus",
]


# ---------------------------------------------------------------------
# Scoring configuration
# ---------------------------------------------------------------------
#
# The score is specifically intended to answer:
#
#     "Would this make an interesting Mobility Pulse animation?"
#
# It is NOT intended to rank the substantive importance of mobility
# metrics.
#
# Scores are normalized WITHIN EACH METRIC so volatile count metrics
# cannot automatically overwhelm smoother speed/duration metrics.

COMPONENT_WEIGHTS = {
    "median_spatial_spread": 0.25,
    "median_abs_change": 0.20,
    "median_temporal_motion": 0.25,
    "material_share": 0.10,
    "persistent_zone_share": 0.10,
    "policy_step": 0.10,
}

MATERIAL_CHANGE_THRESHOLD = 15.0

SHORTLIST_SIZE = 8

MAX_PER_MODE = 2
MAX_PER_GEOGRAPHY = 2


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def metric_label(metric: str) -> str:
    """Return the display label for a metric."""

    return METRIC_LABELS.get(
        metric,
        metric.replace("_", " ").title(),
    )


def robust_symmetric_bound(
    values: pd.Series,
    *,
    percentile: float = 0.95,
) -> float:
    """Calculate the robust absolute-value bound used by Mobility Pulse."""

    clean = (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .dropna()
        .abs()
    )

    if clean.empty:
        return np.nan

    return float(
        clean.quantile(percentile)
    )


def load_temporal_bucket(
    temporal_bucket: str,
) -> pd.DataFrame:
    """Load one temporal bucket once for all candidate metrics."""

    columns = [
        "taxi_zone_id",
        "date",
        "zone",
        "borough",
        *BASE_METRICS,
    ]

    print()
    print(
        f"Loading temporal bucket: {temporal_bucket}"
    )

    frame = pd.read_parquet(
        ZONE_PROFILE_DAILY_DIR,
        engine="pyarrow",
        columns=columns,
        filters=[
            (
                "temporal_bucket",
                "==",
                temporal_bucket,
            ),
        ],
    )

    frame = frame.copy()

    frame["date"] = pd.to_datetime(
        frame["date"]
    )

    return frame


def build_metric_panel(
    frame: pd.DataFrame,
    *,
    metric: str,
) -> pd.DataFrame:
    """Calculate each zone's change from its full pre-CP mean."""

    metric_frame = frame[
        [
            "taxi_zone_id",
            "date",
            "zone",
            "borough",
            metric,
        ]
    ].copy()

    metric_frame[metric] = pd.to_numeric(
        metric_frame[metric],
        errors="coerce",
    )

    pre_reference = (
        metric_frame.loc[
            metric_frame["date"].lt(
                CP_START
            ),
            [
                "taxi_zone_id",
                metric,
            ],
        ]
        .groupby(
            "taxi_zone_id",
            observed=True,
            dropna=False,
        )[metric]
        .mean()
        .rename("pre_reference")
        .reset_index()
    )

    metric_frame = metric_frame.merge(
        pre_reference,
        on="taxi_zone_id",
        how="left",
        validate="many_to_one",
    )

    valid_reference = (
        metric_frame[
            "pre_reference"
        ].notna()
        & metric_frame[
            "pre_reference"
        ].gt(0)
    )

    metric_frame[
        "pulse_change"
    ] = np.where(
        valid_reference,
        (
            metric_frame[metric]
            - metric_frame[
                "pre_reference"
            ]
        )
        / metric_frame[
            "pre_reference"
        ]
        * 100,
        np.nan,
    )

    return metric_frame


def filter_geography(
    frame: pd.DataFrame,
    geography: str,
) -> pd.DataFrame:
    """Apply one borough or citywide geography."""

    if geography == "Citywide":
        return frame.copy()

    return frame[
        frame[
            "borough"
        ]
        .astype(str)
        .eq(geography)
    ].copy()


# ---------------------------------------------------------------------
# Candidate diagnostics
# ---------------------------------------------------------------------

def score_candidate(
    frame: pd.DataFrame,
    *,
    metric: str,
    temporal_bucket: str,
    geography: str,
    window_name: str,
    start_date: str,
    end_date: str,
) -> dict[str, object] | None:
    """Calculate animation-interest diagnostics for one candidate."""

    start = pd.Timestamp(
        start_date
    )

    end = pd.Timestamp(
        end_date
    )

    candidate = frame[
        frame["date"].between(
            start,
            end,
        )
    ].copy()

    candidate = filter_geography(
        candidate,
        geography,
    )

    if candidate.empty:
        return None

    candidate[
        "pulse_change"
    ] = pd.to_numeric(
        candidate[
            "pulse_change"
        ],
        errors="coerce",
    )

    candidate = candidate[
        candidate[
            "pulse_change"
        ].notna()
    ].copy()

    if candidate.empty:
        return None

    distinct_dates = int(
        candidate[
            "date"
        ].nunique()
    )

    distinct_zones = int(
        candidate[
            "taxi_zone_id"
        ].nunique()
    )

    if distinct_dates < 20:
        return None

    if distinct_zones < 5:
        return None

    # -----------------------------------------------------------------
    # Coverage
    # -----------------------------------------------------------------

    expected_rows = (
        distinct_dates
        * distinct_zones
    )

    actual_rows = len(
        candidate[
            [
                "taxi_zone_id",
                "date",
            ]
        ]
        .drop_duplicates()
    )

    coverage = (
        actual_rows
        / expected_rows
        if expected_rows
        else np.nan
    )

    # -----------------------------------------------------------------
    # 1. Magnitude
    # -----------------------------------------------------------------

    median_abs_change = float(
        candidate[
            "pulse_change"
        ]
        .abs()
        .median()
    )

    robust_bound = (
        robust_symmetric_bound(
            candidate[
                "pulse_change"
            ]
        )
    )

    # -----------------------------------------------------------------
    # 2. Spatial contrast
    # -----------------------------------------------------------------
    #
    # For every date, calculate the robust 10th-to-90th percentile
    # spread across Taxi Zones. A high value means the map has strong
    # geographic contrast rather than moving uniformly.

    daily_spatial = (
        candidate.groupby(
            "date",
            observed=True,
            dropna=False,
        )[
            "pulse_change"
        ]
        .agg(
            q10=lambda x: (
                x.quantile(0.10)
            ),
            q90=lambda x: (
                x.quantile(0.90)
            ),
        )
        .reset_index()
    )

    daily_spatial[
        "spatial_spread"
    ] = (
        daily_spatial["q90"]
        - daily_spatial["q10"]
    )

    median_spatial_spread = float(
        daily_spatial[
            "spatial_spread"
        ].median()
    )

    # -----------------------------------------------------------------
    # 3. Temporal movement
    # -----------------------------------------------------------------
    #
    # Measure how much each zone's color is likely to change between
    # consecutive observations. This is crucial for an animation:
    # large static differences can make a good map but a poor movie.

    candidate = (
        candidate.sort_values(
            [
                "taxi_zone_id",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    candidate[
        "daily_change_in_pulse"
    ] = (
        candidate.groupby(
            "taxi_zone_id",
            observed=True,
            dropna=False,
        )[
            "pulse_change"
        ]
        .diff()
        .abs()
    )

    median_temporal_motion = float(
        candidate[
            "daily_change_in_pulse"
        ].median()
    )

    # -----------------------------------------------------------------
    # 4. Materiality
    # -----------------------------------------------------------------

    material_flag = (
        candidate[
            "pulse_change"
        ]
        .abs()
        .ge(
            MATERIAL_CHANGE_THRESHOLD
        )
    )

    material_share = float(
        material_flag.mean()
    )

    # -----------------------------------------------------------------
    # 5. Persistence
    # -----------------------------------------------------------------
    #
    # A zone counts as persistent when at least half of its observations
    # during the window show a material ±15% change.

    zone_material_share = (
        candidate.assign(
            material_flag=material_flag
        )
        .groupby(
            "taxi_zone_id",
            observed=True,
            dropna=False,
        )[
            "material_flag"
        ]
        .mean()
    )

    persistent_zone_share = float(
        zone_material_share
        .ge(0.50)
        .mean()
    )

    # -----------------------------------------------------------------
    # 6. Policy-step signal
    # -----------------------------------------------------------------
    #
    # This is only defined when the candidate window actually crosses
    # January 5, 2025. Post-only windows are NOT penalized for having
    # no policy-step measurement.

    policy_step = np.nan

    crosses_cp = (
        start < CP_START
        and end >= CP_START
    )

    if crosses_cp:
        pre = candidate[
            candidate[
                "date"
            ].lt(
                CP_START
            )
        ]

        post = candidate[
            candidate[
                "date"
            ].ge(
                CP_START
            )
        ]

        if (
            not pre.empty
            and not post.empty
        ):
            pre_zone = (
                pre.groupby(
                    "taxi_zone_id",
                    observed=True,
                )[
                    "pulse_change"
                ]
                .median()
            )

            post_zone = (
                post.groupby(
                    "taxi_zone_id",
                    observed=True,
                )[
                    "pulse_change"
                ]
                .median()
            )

            step = (
                post_zone
                - pre_zone
            ).dropna()

            if not step.empty:
                policy_step = float(
                    step
                    .abs()
                    .median()
                )

    return {
        "metric": metric,
        "metric_label": (
            metric_label(metric)
        ),
        "mode": (
            MODE_BY_METRIC.get(
                metric,
                "Other",
            )
        ),
        "geography": geography,
        "temporal_bucket": (
            temporal_bucket
        ),
        "window": window_name,
        "start_date": start.date(),
        "end_date": end.date(),
        "crosses_cp": crosses_cp,
        "dates": distinct_dates,
        "zones": distinct_zones,
        "coverage": coverage,
        "median_abs_change": (
            median_abs_change
        ),
        "robust_95pct_bound": (
            robust_bound
        ),
        "median_spatial_spread": (
            median_spatial_spread
        ),
        "median_temporal_motion": (
            median_temporal_motion
        ),
        "material_share": (
            material_share
        ),
        "persistent_zone_share": (
            persistent_zone_share
        ),
        "policy_step": policy_step,
    }


# ---------------------------------------------------------------------
# Metric-relative normalization
# ---------------------------------------------------------------------

def normalize_series_robustly(
    values: pd.Series,
) -> pd.Series:
    """Robustly scale one diagnostic to 0-1."""

    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    valid = numeric.dropna()

    if valid.empty:
        return pd.Series(
            np.nan,
            index=values.index,
            dtype=float,
        )

    low = valid.quantile(
        0.05
    )

    high = valid.quantile(
        0.95
    )

    if (
        pd.isna(low)
        or pd.isna(high)
        or high <= low
    ):
        result = pd.Series(
            0.5,
            index=values.index,
            dtype=float,
        )

        result[
            numeric.isna()
        ] = np.nan

        return result

    return (
        (
            numeric.clip(
                lower=low,
                upper=high,
            )
            - low
        )
        / (
            high
            - low
        )
    )


def add_metric_relative_scores(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize animation diagnostics separately within each metric."""

    scored_groups = []

    for metric, metric_frame in frame.groupby(
        "metric",
        observed=True,
        dropna=False,
    ):
        metric_frame = (
            metric_frame.copy()
        )

        for component in COMPONENT_WEIGHTS:
            metric_frame[
                f"{component}_score"
            ] = normalize_series_robustly(
                metric_frame[
                    component
                ]
            )

        scored_groups.append(
            metric_frame
        )

    result = pd.concat(
        scored_groups,
        ignore_index=True,
    )

    # -----------------------------------------------------------------
    # Composite score with row-specific weights.
    # -----------------------------------------------------------------
    #
    # policy_step is undefined for post-only windows. Rather than
    # assigning those rows a zero, calculate a weighted average from
    # the components that are actually applicable.

    weighted_sum = pd.Series(
        0.0,
        index=result.index,
    )

    available_weight = pd.Series(
        0.0,
        index=result.index,
    )

    for (
        component,
        weight,
    ) in COMPONENT_WEIGHTS.items():
        score_column = (
            f"{component}_score"
        )

        valid_component = (
            result[
                score_column
            ].notna()
        )

        weighted_sum = (
            weighted_sum
            + result[
                score_column
            ]
            .fillna(0.0)
            * weight
        )

        available_weight = (
            available_weight
            + valid_component
            .astype(float)
            * weight
        )

    result[
        "base_interest_score"
    ] = np.where(
        available_weight.gt(0),
        weighted_sum
        / available_weight,
        np.nan,
    )

    # -----------------------------------------------------------------
    # Animation-motion factor
    # -----------------------------------------------------------------
    #
    # A candidate with zero temporal motion should be heavily penalized
    # even if its static map has large spatial differences.
    #
    # 0 motion score -> 15% of base score
    # 1 motion score -> 100% of base score

    motion_score = (
        result[
            "median_temporal_motion_score"
        ]
        .fillna(0.0)
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    result[
        "motion_factor"
    ] = (
        0.15
        + 0.85
        * motion_score
    )

    # -----------------------------------------------------------------
    # Coverage factor
    # -----------------------------------------------------------------

    result[
        "coverage_factor"
    ] = (
        pd.to_numeric(
            result[
                "coverage"
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    result[
        "final_score"
    ] = (
        result[
            "base_interest_score"
        ]
        * result[
            "motion_factor"
        ]
        * result[
            "coverage_factor"
        ]
    )

    # -----------------------------------------------------------------
    # Useful ranks
    # -----------------------------------------------------------------

    result[
        "rank_within_metric"
    ] = (
        result.groupby(
            "metric",
            observed=True,
        )[
            "final_score"
        ]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype("Int64")
    )

    result[
        "rank_within_mode"
    ] = (
        result.groupby(
            "mode",
            observed=True,
        )[
            "final_score"
        ]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype("Int64")
    )

    return (
        result.sort_values(
            "final_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Diverse shortlist
# ---------------------------------------------------------------------

def candidate_key(
    row: pd.Series,
) -> tuple[str, str, str, str, str]:
    """Create a stable identity for one animation candidate."""

    return (
        str(row["metric"]),
        str(row["geography"]),
        str(row["temporal_bucket"]),
        str(row["window"]),
        str(row["start_date"]),
    )


def can_add_candidate(
    row: pd.Series,
    *,
    selected_keys: set[
        tuple[str, str, str, str, str]
    ],
    mode_counts: dict[str, int],
    geography_counts: dict[str, int],
    enforce_caps: bool = True,
) -> bool:
    """Check whether a candidate can be added to the shortlist."""

    key = candidate_key(
        row
    )

    if key in selected_keys:
        return False

    if not enforce_caps:
        return True

    mode = str(
        row["mode"]
    )

    geography = str(
        row["geography"]
    )

    if (
        mode_counts.get(
            mode,
            0,
        )
        >= MAX_PER_MODE
    ):
        return False

    if (
        geography_counts.get(
            geography,
            0,
        )
        >= MAX_PER_GEOGRAPHY
    ):
        return False

    return True


def register_candidate(
    row: pd.Series,
    *,
    selected_rows: list[
        pd.Series
    ],
    selected_keys: set[
        tuple[str, str, str, str, str]
    ],
    mode_counts: dict[str, int],
    geography_counts: dict[str, int],
    selection_reason: str,
) -> None:
    """Add one candidate and update shortlist counters."""

    selected = row.copy()

    selected[
        "selection_reason"
    ] = selection_reason

    selected_rows.append(
        selected
    )

    selected_keys.add(
        candidate_key(
            row
        )
    )

    mode = str(
        row["mode"]
    )

    geography = str(
        row["geography"]
    )

    mode_counts[mode] = (
        mode_counts.get(
            mode,
            0,
        )
        + 1
    )

    geography_counts[
        geography
    ] = (
        geography_counts.get(
            geography,
            0,
        )
        + 1
    )


def build_diverse_shortlist(
    scored: pd.DataFrame,
    *,
    target_size: int = SHORTLIST_SIZE,
) -> pd.DataFrame:
    """Build a shortlist with explicit modality and geography diversity.

    Selection order:

      1. Seed the best available candidate from every mobility mode.
      2. Add the best available candidate from Citywide, Manhattan,
         Brooklyn, and Queens when not already represented.
      3. Fill remaining slots by overall score while enforcing caps.

    Final caps:
      - no more than two animations from any one mode;
      - no more than two animations from any one geography.
    """

    ranked = (
        scored.sort_values(
            "final_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    selected_rows: list[
        pd.Series
    ] = []

    selected_keys: set[
        tuple[str, str, str, str, str]
    ] = set()

    mode_counts: dict[
        str,
        int,
    ] = {}

    geography_counts: dict[
        str,
        int,
    ] = {}

    # -----------------------------------------------------------------
    # Step 1: guarantee modality representation.
    # -----------------------------------------------------------------

    for mode in REQUIRED_MODES:
        candidates = ranked[
            ranked[
                "mode"
            ].eq(mode)
        ]

        for _, row in candidates.iterrows():
            if can_add_candidate(
                row,
                selected_keys=(
                    selected_keys
                ),
                mode_counts=(
                    mode_counts
                ),
                geography_counts=(
                    geography_counts
                ),
                enforce_caps=True,
            ):
                register_candidate(
                    row,
                    selected_rows=(
                        selected_rows
                    ),
                    selected_keys=(
                        selected_keys
                    ),
                    mode_counts=(
                        mode_counts
                    ),
                    geography_counts=(
                        geography_counts
                    ),
                    selection_reason=(
                        f"Best {mode} candidate"
                    ),
                )

                break

    # -----------------------------------------------------------------
    # Step 2: guarantee key geography representation.
    # -----------------------------------------------------------------

    represented_geographies = {
        str(
            row[
                "geography"
            ]
        )
        for row in selected_rows
    }

    for geography in (
        REQUIRED_SHORTLIST_GEOGRAPHIES
    ):
        if (
            geography
            in represented_geographies
        ):
            continue

        candidates = ranked[
            ranked[
                "geography"
            ].eq(geography)
        ]

        for _, row in candidates.iterrows():
            if can_add_candidate(
                row,
                selected_keys=(
                    selected_keys
                ),
                mode_counts=(
                    mode_counts
                ),
                geography_counts=(
                    geography_counts
                ),
                enforce_caps=True,
            ):
                register_candidate(
                    row,
                    selected_rows=(
                        selected_rows
                    ),
                    selected_keys=(
                        selected_keys
                    ),
                    mode_counts=(
                        mode_counts
                    ),
                    geography_counts=(
                        geography_counts
                    ),
                    selection_reason=(
                        f"Best {geography} candidate"
                    ),
                )

                represented_geographies.add(
                    geography
                )

                break

    # -----------------------------------------------------------------
    # Step 3: fill remaining positions by score.
    # -----------------------------------------------------------------

    for _, row in ranked.iterrows():
        if (
            len(selected_rows)
            >= target_size
        ):
            break

        if not can_add_candidate(
            row,
            selected_keys=(
                selected_keys
            ),
            mode_counts=(
                mode_counts
            ),
            geography_counts=(
                geography_counts
            ),
            enforce_caps=True,
        ):
            continue

        register_candidate(
            row,
            selected_rows=(
                selected_rows
            ),
            selected_keys=(
                selected_keys
            ),
            mode_counts=(
                mode_counts
            ),
            geography_counts=(
                geography_counts
            ),
            selection_reason=(
                "Highest remaining score"
            ),
        )

    if not selected_rows:
        return ranked.head(
            0
        ).copy()

    shortlist = pd.DataFrame(
        selected_rows
    )

    return (
        shortlist.sort_values(
            "final_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------

def print_table(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    max_rows: int | None = None,
) -> None:
    """Print a compact diagnostic table."""

    display = frame[
        columns
    ].copy()

    if max_rows is not None:
        display = display.head(
            max_rows
        )

    print(
        display.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.3f}"
            ),
        )
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """Score and shortlist candidate Mobility Pulse animations."""

    print(
        "=" * 80
    )
    print(
        "MOBILITY PULSE PRESET DIAGNOSTIC — METRIC-RELATIVE PASS"
    )
    print(
        "=" * 80
    )

    print()
    print(
        "Metrics:"
    )

    for metric in BASE_METRICS:
        print(
            f"  "
            f"{MODE_BY_METRIC.get(metric, 'Other'):<7} "
            f"{metric_label(metric)}"
        )

    results: list[
        dict[str, object]
    ] = []

    # -----------------------------------------------------------------
    # Evaluate candidate combinations.
    # -----------------------------------------------------------------

    for temporal_bucket in (
        TEMPORAL_BUCKETS
    ):
        try:
            bucket_frame = (
                load_temporal_bucket(
                    temporal_bucket
                )
            )

        except Exception as exc:
            print(
                f"Skipping "
                f"{temporal_bucket}: "
                f"{exc}"
            )

            continue

        for metric in BASE_METRICS:
            if (
                metric
                not in bucket_frame.columns
            ):
                continue

            print(
                f"  Scoring: "
                f"{metric_label(metric)}"
            )

            metric_frame = (
                build_metric_panel(
                    bucket_frame,
                    metric=metric,
                )
            )

            for geography in (
                GEOGRAPHIES
            ):
                for (
                    window_name,
                    (
                        start_date,
                        end_date,
                    ),
                ) in WINDOWS.items():

                    result = (
                        score_candidate(
                            metric_frame,
                            metric=metric,
                            temporal_bucket=(
                                temporal_bucket
                            ),
                            geography=(
                                geography
                            ),
                            window_name=(
                                window_name
                            ),
                            start_date=(
                                start_date
                            ),
                            end_date=(
                                end_date
                            ),
                        )
                    )

                    if (
                        result
                        is not None
                    ):
                        results.append(
                            result
                        )

    if not results:
        raise RuntimeError(
            "No candidate combinations could be scored."
        )

    raw_results = pd.DataFrame(
        results
    )

    scored = (
        add_metric_relative_scores(
            raw_results
        )
    )

    shortlist = (
        build_diverse_shortlist(
            scored,
            target_size=(
                SHORTLIST_SIZE
            ),
        )
    )

    # -----------------------------------------------------------------
    # Save diagnostics.
    # -----------------------------------------------------------------

    scores_path = (
        OUTPUT_DIR
        / "mobility_pulse_candidate_scores_metric_relative.csv"
    )

    shortlist_path = (
        OUTPUT_DIR
        / "mobility_pulse_recommended_shortlist_metric_relative.csv"
    )

    scored.to_csv(
        scores_path,
        index=False,
    )

    shortlist.to_csv(
        shortlist_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # Console reporting.
    # -----------------------------------------------------------------

    display_columns = [
        "final_score",
        "base_interest_score",
        "motion_factor",
        "mode",
        "metric_label",
        "geography",
        "temporal_bucket",
        "window",
        "rank_within_metric",
        "median_abs_change",
        "median_spatial_spread",
        "median_temporal_motion",
        "material_share",
        "persistent_zone_share",
        "policy_step",
        "coverage",
    ]

    shortlist_columns = [
        "final_score",
        "mode",
        "metric_label",
        "geography",
        "temporal_bucket",
        "window",
        "selection_reason",
        "rank_within_metric",
        "median_abs_change",
        "median_spatial_spread",
        "median_temporal_motion",
        "coverage",
    ]

    print()
    print(
        "=" * 80
    )
    print(
        "TOP 20 METRIC-RELATIVE ANIMATION CANDIDATES"
    )
    print(
        "=" * 80
    )

    print_table(
        scored,
        columns=display_columns,
        max_rows=20,
    )

    print()
    print(
        "=" * 80
    )
    print(
        "BEST CANDIDATE FOR EACH METRIC"
    )
    print(
        "=" * 80
    )

    best_by_metric = (
        scored[
            scored[
                "rank_within_metric"
            ].eq(1)
        ]
        .sort_values(
            "final_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print_table(
        best_by_metric,
        columns=display_columns,
    )

    print()
    print(
        "=" * 80
    )
    print(
        "BEST CANDIDATE BY MODE"
    )
    print(
        "=" * 80
    )

    best_by_mode = (
        scored.sort_values(
            "final_score",
            ascending=False,
        )
        .groupby(
            "mode",
            observed=True,
            as_index=False,
        )
        .first()
        .sort_values(
            "final_score",
            ascending=False,
        )
    )

    print_table(
        best_by_mode,
        columns=display_columns,
    )

    print()
    print(
        "=" * 80
    )
    print(
        "BEST CANDIDATE BY GEOGRAPHY"
    )
    print(
        "=" * 80
    )

    best_by_geography = (
        scored.sort_values(
            "final_score",
            ascending=False,
        )
        .groupby(
            "geography",
            observed=True,
            as_index=False,
        )
        .first()
        .sort_values(
            "final_score",
            ascending=False,
        )
    )

    print_table(
        best_by_geography,
        columns=display_columns,
    )

    print()
    print(
        "=" * 80
    )
    print(
        "DIVERSE RECOMMENDED SHORTLIST"
    )
    print(
        "=" * 80
    )

    print_table(
        shortlist,
        columns=shortlist_columns,
    )

    print()
    print(
        "Shortlist mode counts:"
    )

    print(
        shortlist[
            "mode"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Shortlist geography counts:"
    )

    print(
        shortlist[
            "geography"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        f"Full metric-relative candidate scores:\n"
        f"  {scores_path}"
    )

    print(
        f"Recommended shortlist:\n"
        f"  {shortlist_path}"
    )


if __name__ == "__main__":
    main()
