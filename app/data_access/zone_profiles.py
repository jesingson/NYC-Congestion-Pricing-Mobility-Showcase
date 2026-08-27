"""Zone-profile data helpers for Raw View 07.

Phase 1 purpose
---------------
Validate that a single Taxi Zone can drive a coordinated mobility profile:

- selected-zone daily series;
- citywide, Borough, or geo-policy peer baselines;
- indexed and actual-value trend compatibility;
- ten-metric pre/post profile;
- temporal-bucket profile;
- ranking context;
- strongest local pairwise divergence.

Important aggregation policy
----------------------------
- Count metrics are summed within each Taxi Zone × date.
- Speed and duration metrics use weighted averages where the matching
  activity-support metric exists.
- Peer baselines are averages across peer Taxi Zones after each peer zone has
  first been reduced to one value per date.
- The selected Taxi Zone is excluded from every comparison baseline.
- Percent-change eligibility uses the project's existing reliability helper.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from app.data_access.aggregations import (
    COUNT_METRICS,
    WEIGHTED_MEAN_METRICS,
    WEIGHT_COLUMNS,
    aggregate_metrics,
    get_required_columns,
)
from app.data_access.loaders import (
    BASE_METRICS,
    CONGESTION_PRICING_START_DATE,
    METRIC_LABELS,
    TEMPORAL_BUCKET_ORDER,
    load_analysis_panel,
)
from app.data_access.spatial_aggregations import (
    ALL_TEMPORAL_BUCKETS_LABEL,
    get_zone_pre_post_metric_summary,
)
from app.data_access.mobility_environments import (
    attach_mobility_regime_cluster_context,
    format_mobility_regime_cluster_label,
)
from app.data_access.spatial_visuals import (
    add_reliability_flags,
)
from app.data_access.zone_profile_app_tables import (
    ALL_TAXI_ZONES_GROUP,
    load_comparison_daily_totals,
    load_zone_daily_metric,
)


COMPARISON_LEVELS = [
    "No comparison",
    "Citywide",
    "Borough",
    "Geo-policy group",
    "Mobility regime cluster",
]

RECOMMENDED_ZONE_NAMES = [
    "Alphabet City",
    "Fort Greene",
    "Midtown Center",
    "JFK Airport",
    "Central Park",
    "Jamaica Bay",
    "Breezy Point/Fort Tilden/Riis Beach",
]

ZONE_METADATA_COLUMNS = [
    "taxi_zone_id",
    "zone",
    "borough",
    "cbd_spatial_category",
]


def _safe_percent_change(
    pre_value: float,
    post_value: float,
) -> float:
    if (
        pd.isna(pre_value)
        or pd.isna(post_value)
        or float(pre_value) == 0
    ):
        return np.nan

    return (
        (
            float(post_value)
            - float(pre_value)
        )
        / float(pre_value)
        * 100
    )


def _mode_or_first(
    values: pd.Series,
) -> object:
    non_null = values.dropna()

    if non_null.empty:
        return np.nan

    mode = non_null.mode()

    if not mode.empty:
        return mode.iloc[0]

    return non_null.iloc[0]


def format_geo_policy_label(
    value: object,
) -> str:
    """Return a presentation-ready policy-geography label."""
    if pd.isna(value):
        return "Unknown"

    normalized = (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    labels = {
        "cbd": "CBD",
        "gateway": "Gateway",
        "gateway to cbd": "Gateway",
        "adjacent": "Adjacent",
        "adjacent to cbd": "Adjacent",
        "non cbd": "Non-CBD",
        "noncbd": "Non-CBD",
    }

    return labels.get(
        normalized,
        str(value).strip().title(),
    )


def get_comparison_label(
    *,
    taxi_zone_id: int | float | str,
    comparison_level: str,
) -> str:
    """Return the user-facing geographic comparison label."""
    metadata = get_zone_metadata(
        taxi_zone_id
    )

    if comparison_level == "No comparison":
        return "No geographic comparison"

    if comparison_level == "Citywide":
        return "Citywide Taxi Zone average"

    if comparison_level == "Borough":
        return (
            f"{metadata['borough']} Taxi Zone average"
        )

    if comparison_level == "Geo-policy group":
        return (
            f"{format_geo_policy_label(metadata['cbd_spatial_category'])} "
            "Taxi Zone average"
        )

    if comparison_level == "Mobility regime cluster":
        cluster_context = attach_mobility_regime_cluster_context(
            pd.DataFrame({"taxi_zone_id": [taxi_zone_id]}),
            assignment_period="post_cp",
        )
        cluster_row = cluster_context.iloc[0]
        return (
            f"{format_mobility_regime_cluster_label(cluster_row['mobility_regime_cluster_label'])} "
            "Taxi Zone average"
        )

    raise ValueError(
        f"Unsupported comparison level: {comparison_level}"
    )


@st.cache_data(show_spinner=False)
def get_zone_catalog() -> pd.DataFrame:
    """Return one stable metadata row per Taxi Zone."""
    panel = load_analysis_panel(
        columns=ZONE_METADATA_COLUMNS,
    )

    catalog = (
        panel.groupby(
            "taxi_zone_id",
            observed=True,
            dropna=False,
        )
        .agg(
            zone=("zone", _mode_or_first),
            borough=("borough", _mode_or_first),
            cbd_spatial_category=(
                "cbd_spatial_category",
                _mode_or_first,
            ),
        )
        .reset_index()
    )

    catalog["zone"] = (
        catalog["zone"]
        .fillna("Unknown")
        .astype(str)
    )

    catalog["borough"] = (
        catalog["borough"]
        .fillna("Unknown")
        .astype(str)
    )

    catalog["cbd_spatial_category"] = (
        catalog["cbd_spatial_category"]
        .fillna("Unknown")
        .astype(str)
    )

    return (
        catalog.sort_values(
            [
                "zone",
                "taxi_zone_id",
            ]
        )
        .reset_index(drop=True)
    )


def get_recommended_zone_catalog() -> pd.DataFrame:
    """Return recommended zones first, then all remaining zones alphabetically."""
    catalog = get_zone_catalog().copy()

    recommendation_order = {
        zone_name: index
        for index, zone_name in enumerate(
            RECOMMENDED_ZONE_NAMES
        )
    }

    catalog["is_recommended"] = (
        catalog["zone"].isin(
            RECOMMENDED_ZONE_NAMES
        )
    )

    catalog["_recommendation_order"] = (
        catalog["zone"]
        .map(recommendation_order)
        .fillna(
            len(RECOMMENDED_ZONE_NAMES)
        )
    )

    return (
        catalog.sort_values(
            [
                "is_recommended",
                "_recommendation_order",
                "zone",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .drop(
            columns="_recommendation_order"
        )
        .reset_index(drop=True)
    )


def get_zone_metadata(
    taxi_zone_id: int | float | str,
) -> pd.Series:
    catalog = get_zone_catalog()

    match = catalog[
        catalog["taxi_zone_id"].astype(str).eq(
            str(taxi_zone_id)
        )
    ]

    if match.empty:
        raise KeyError(
            f"Taxi Zone {taxi_zone_id} was not found."
        )

    return match.iloc[0]

def _filter_temporal_bucket(
    panel: pd.DataFrame,
    temporal_bucket: str,
) -> pd.DataFrame:
    if temporal_bucket == ALL_TEMPORAL_BUCKETS_LABEL:
        return panel.copy()

    return panel[
        panel["temporal_bucket"].astype(str).eq(
            str(temporal_bucket)
        )
    ].copy()


def _aggregate_zone_daily(
    panel: pd.DataFrame,
    *,
    metric: str,
) -> pd.DataFrame:
    """Reduce panel rows to Taxi Zone × date using established metric rules."""
    return aggregate_metrics(
        panel,
        metrics=[metric],
        group_cols=[
            "taxi_zone_id",
            "date",
        ],
    )


def _attach_zone_metadata(
    daily: pd.DataFrame,
    metadata_source: pd.DataFrame,
) -> pd.DataFrame:
    metadata = (
        metadata_source[
            ZONE_METADATA_COLUMNS
        ]
        .groupby(
            "taxi_zone_id",
            observed=True,
            dropna=False,
        )
        .agg(
            zone=("zone", _mode_or_first),
            borough=("borough", _mode_or_first),
            cbd_spatial_category=(
                "cbd_spatial_category",
                _mode_or_first,
            ),
        )
        .reset_index()
    )

    return daily.merge(
        metadata,
        on="taxi_zone_id",
        how="left",
        validate="many_to_one",
    )


@st.cache_data(show_spinner=False)
def get_zone_and_baseline_daily_series(
    *,
    taxi_zone_id: int | float | str,
    metric: str,
    temporal_bucket: str = ALL_TEMPORAL_BUCKETS_LABEL,
    comparison_level: str = "Citywide",
) -> pd.DataFrame:
    """Return the selected-zone series and optional geographic comparison.

    Runtime behavior
    ----------------
    This function no longer loads and groups the full analysis panel. It reads:

    1. one selected-zone metric partition; and
    2. one additive comparison-total partition.

    The selected zone is excluded algebraically:

        (group sum - selected-zone value)
        ---------------------------------
        (valid group count - selected-zone presence)

    This preserves the original comparison methodology while avoiding the
    expensive all-zone daily groupby during every Streamlit interaction.
    """
    if metric not in BASE_METRICS:
        raise ValueError(
            f"Unsupported metric: {metric}"
        )

    if comparison_level not in COMPARISON_LEVELS:
        raise ValueError(
            "Unsupported comparison level: "
            f"{comparison_level}"
        )

    selected_metadata = get_zone_metadata(
        taxi_zone_id
    )

    resolved_zone_id = selected_metadata[
        "taxi_zone_id"
    ]

    selected_daily = load_zone_daily_metric(
        taxi_zone_id=resolved_zone_id,
        metric=metric,
        temporal_bucket=temporal_bucket,
    )

    if selected_daily.empty:
        raise ValueError(
            "No daily app-table rows were found for "
            f"Taxi Zone {resolved_zone_id}, "
            f"metric {metric}, and temporal bucket "
            f"{temporal_bucket}."
        )

    if comparison_level == "No comparison":
        baseline_daily = pd.DataFrame(
            {
                "date": selected_daily[
                    "date"
                ],
                "baseline_value": np.nan,
                "baseline_peer_zones": 0,
            }
        )

    else:
        if comparison_level == "Citywide":
            comparison_group = (
                ALL_TAXI_ZONES_GROUP
            )

        elif comparison_level == "Borough":
            comparison_group = str(
                selected_metadata[
                    "borough"
                ]
            )

        elif comparison_level == "Mobility regime cluster":
            cluster_context = attach_mobility_regime_cluster_context(
                pd.DataFrame({"taxi_zone_id": [resolved_zone_id]}),
                assignment_period="post_cp",
            )
            comparison_group = format_mobility_regime_cluster_label(
                cluster_context.iloc[0]["mobility_regime_cluster_label"]
            )

        else:
            comparison_group = str(
                selected_metadata[
                    "cbd_spatial_category"
                ]
            )

        comparison_totals = (
            load_comparison_daily_totals(
                comparison_level=(
                    comparison_level
                ),
                comparison_group=(
                    comparison_group
                ),
                metric=metric,
                temporal_bucket=(
                    temporal_bucket
                ),
            )
        )

        if comparison_totals.empty:
            raise ValueError(
                "No comparison app-table rows were found for "
                f"{comparison_level} / {comparison_group} / "
                f"{metric} / {temporal_bucket}."
            )

        baseline_daily = (
            comparison_totals.merge(
                selected_daily[
                    [
                        "date",
                        "zone_value",
                    ]
                ],
                on="date",
                how="outer",
                validate="one_to_one",
            )
            .sort_values("date")
        )

        selected_present = (
            baseline_daily[
                "zone_value"
            ]
            .notna()
            .astype("int64")
        )

        comparison_sum_excluding_zone = (
            baseline_daily[
                "metric_sum"
            ]
            - baseline_daily[
                "zone_value"
            ].fillna(0)
        )

        comparison_count_excluding_zone = (
            baseline_daily[
                "valid_zone_count"
            ]
            - selected_present
        )

        baseline_daily[
            "baseline_peer_zones"
        ] = (
            comparison_count_excluding_zone
            .clip(lower=0)
            .astype("int64")
        )

        baseline_daily[
            "baseline_value"
        ] = np.where(
            comparison_count_excluding_zone
            > 0,
            (
                comparison_sum_excluding_zone
                / comparison_count_excluding_zone
            ),
            np.nan,
        )

        baseline_daily = baseline_daily[
            [
                "date",
                "baseline_value",
                "baseline_peer_zones",
            ]
        ]

    aligned = selected_daily.merge(
        baseline_daily,
        on="date",
        how="outer",
        validate="one_to_one",
    ).sort_values("date")

    aligned["pre_post_cp"] = np.where(
        aligned["date"]
        >= CONGESTION_PRICING_START_DATE,
        "post_cp",
        "pre_cp",
    )

    aligned["paired_observation"] = (
        aligned[
            [
                "zone_value",
                "baseline_value",
            ]
        ]
        .notna()
        .all(axis=1)
    )

    for value_column, prefix in [
        (
            "zone_value",
            "zone",
        ),
        (
            "baseline_value",
            "baseline",
        ),
    ]:
        pre_reference = aligned.loc[
            aligned["date"]
            < CONGESTION_PRICING_START_DATE,
            value_column,
        ].mean(
            skipna=True
        )

        aligned[
            f"{prefix}_pre_reference"
        ] = pre_reference

        if (
            pd.isna(pre_reference)
            or pre_reference == 0
        ):
            aligned[
                f"{prefix}_index"
            ] = np.nan
        else:
            aligned[
                f"{prefix}_index"
            ] = (
                aligned[value_column]
                / pre_reference
                * 100
            )

    aligned[
        "zone_minus_baseline"
    ] = (
        aligned["zone_value"]
        - aligned["baseline_value"]
    )

    aligned[
        "zone_index_minus_baseline_index"
    ] = (
        aligned["zone_index"]
        - aligned["baseline_index"]
    )

    return (
        aligned.reset_index(
            drop=True
        )
    )


def summarize_daily_comparison(
    daily: pd.DataFrame,
) -> dict[str, object]:
    """Summarize aligned selected-zone and peer-baseline daily series."""
    zone_daily = daily[
        daily["zone_value"].notna()
    ].copy()

    paired = daily[
        daily["paired_observation"]
    ].copy()

    paired_pre = paired[
        paired["date"]
        < CONGESTION_PRICING_START_DATE
    ]

    paired_post = paired[
        paired["date"]
        >= CONGESTION_PRICING_START_DATE
    ]

    zone_pre = zone_daily[
        zone_daily["date"]
        < CONGESTION_PRICING_START_DATE
    ]

    zone_post = zone_daily[
        zone_daily["date"]
        >= CONGESTION_PRICING_START_DATE
    ]

    baseline_pre = paired_pre[
        "baseline_value"
    ].mean()

    baseline_post = paired_post[
        "baseline_value"
    ].mean()

    zone_pre = zone_pre[
        "zone_value"
    ].mean()

    zone_post = zone_post[
        "zone_value"
    ].mean()

    zone_change = _safe_percent_change(
        zone_pre,
        zone_post,
    )

    baseline_change = _safe_percent_change(
        baseline_pre,
        baseline_post,
    )

    return {
        "zone_pre_average": zone_pre,
        "zone_post_average": zone_post,
        "zone_percent_change": zone_change,
        "baseline_pre_average": baseline_pre,
        "baseline_post_average": baseline_post,
        "baseline_percent_change": baseline_change,
        "change_gap": (
            zone_change
            - baseline_change
            if pd.notna(zone_change)
            and pd.notna(
                baseline_change
            )
            else np.nan
        ),
        "zone_first_date": (
            zone_daily["date"].min()
        ),
        "zone_last_date": (
            zone_daily["date"].max()
        ),
        "baseline_first_date": (
            daily.loc[
                daily[
                    "baseline_value"
                ].notna(),
                "date",
            ].min()
        ),
        "baseline_last_date": (
            daily.loc[
                daily[
                    "baseline_value"
                ].notna(),
                "date",
            ].max()
        ),
        "paired_observations": int(
            paired["date"].nunique()
        ),
        "zone_observations": int(
            zone_daily["date"].nunique()
        ),
        "baseline_observations": int(
            daily.loc[
                daily[
                    "baseline_value"
                ].notna(),
                "date",
            ].nunique()
        ),
        "missing_paired_dates": int(
            (~daily[
                "paired_observation"
            ]).sum()
        ),
        "median_peer_zones": (
            daily[
                "baseline_peer_zones"
            ].median()
        ),
    }


def _peer_summary_from_zone_summary(
    source: pd.DataFrame,
    *,
    selected_zone_id: int | float | str,
    metric: str,
    comparison_level: str,
    selected_metadata: pd.Series,
) -> dict[str, float]:
    if comparison_level == "No comparison":
        return {
            "baseline_pre_daily_average": np.nan,
            "baseline_post_daily_average": np.nan,
            "baseline_peer_zones": 0,
        }

    peers = source[
        source["metric"].eq(
            metric
        )
        & ~source[
            "taxi_zone_id"
        ]
        .astype(str)
        .eq(str(selected_zone_id))
    ].copy()

    if comparison_level == "Borough":
        peers = peers[
            peers["borough"]
            .astype(str)
            .eq(
                str(
                    selected_metadata[
                        "borough"
                    ]
                )
            )
        ].copy()

    elif (
        comparison_level
        == "Geo-policy group"
    ):
        peers = peers[
            peers[
                "cbd_spatial_category"
            ]
            .astype(str)
            .eq(
                str(
                    selected_metadata[
                        "cbd_spatial_category"
                    ]
                )
            )
        ].copy()

    elif comparison_level == "Mobility regime cluster":
        selected_cluster = source[
            source["taxi_zone_id"]
            .astype(str)
            .eq(str(selected_zone_id))
        ]

        if not selected_cluster.empty:
            cluster_label = selected_cluster.iloc[0][
                "mobility_regime_cluster_label"
            ]
            if pd.notna(cluster_label):
                peers = peers[
                    peers["mobility_regime_cluster_label"]
                    .astype("Int64")
                    .eq(cluster_label)
                ].copy()

    usable_peers = peers[
        peers[
            [
                "pre_daily_average",
                "post_daily_average",
            ]
        ]
        .notna()
        .all(axis=1)
    ].copy()

    return {
        "baseline_pre_daily_average": (
            usable_peers[
                "pre_daily_average"
            ].mean()
        ),
        "baseline_post_daily_average": (
            usable_peers[
                "post_daily_average"
            ].mean()
        ),
        "baseline_peer_zones": int(
            usable_peers["taxi_zone_id"]
            .nunique()
        ),
    }


@st.cache_data(show_spinner=False)
def get_zone_pre_post_profile(
    *,
    taxi_zone_id: int | float | str,
    comparison_level: str,
    temporal_bucket: str = ALL_TEMPORAL_BUCKETS_LABEL,
) -> pd.DataFrame:
    """Return ten-metric selected-zone and peer-baseline pre/post profile."""
    selected_metadata = get_zone_metadata(
        taxi_zone_id
    )

    source = get_zone_pre_post_metric_summary(
        metrics=BASE_METRICS,
        temporal_bucket=temporal_bucket,
    )

    source = attach_mobility_regime_cluster_context(
        source,
        assignment_period="post_cp",
    )

    source = add_reliability_flags(
        source
    )

    selected = source[
        source["taxi_zone_id"]
        .astype(str)
        .eq(
            str(taxi_zone_id)
        )
    ].copy()

    rows: list[dict[str, object]] = []

    for metric in BASE_METRICS:
        selected_metric = selected[
            selected["metric"].eq(
                metric
            )
        ]

        if selected_metric.empty:
            continue

        row = selected_metric.iloc[0]

        baseline = (
            _peer_summary_from_zone_summary(
                source,
                selected_zone_id=taxi_zone_id,
                metric=metric,
                comparison_level=comparison_level,
                selected_metadata=selected_metadata,
            )
        )

        baseline_change = (
            _safe_percent_change(
                baseline[
                    "baseline_pre_daily_average"
                ],
                baseline[
                    "baseline_post_daily_average"
                ],
            )
        )

        zone_change = row[
            "percent_change"
        ]

        rows.append(
            {
                "metric": metric,
                "metric_label": (
                    METRIC_LABELS.get(
                        metric,
                        metric,
                    )
                ),
                "zone_pre_daily_average": row[
                    "pre_daily_average"
                ],
                "zone_post_daily_average": row[
                    "post_daily_average"
                ],
                "zone_absolute_change": row[
                    "absolute_change"
                ],
                "zone_percent_change": zone_change,
                "zone_eligible_for_percent_change": bool(
                    row.get(
                        "eligible_for_percent_change",
                        False,
                    )
                ),
                "baseline_pre_daily_average": baseline[
                    "baseline_pre_daily_average"
                ],
                "baseline_post_daily_average": baseline[
                    "baseline_post_daily_average"
                ],
                "baseline_percent_change": baseline_change,
                "change_gap": (
                    zone_change
                    - baseline_change
                    if pd.notna(zone_change)
                    and pd.notna(
                        baseline_change
                    )
                    else np.nan
                ),
                "baseline_peer_zones": baseline[
                    "baseline_peer_zones"
                ],
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _get_all_bucket_zone_summaries(
    *,
    metric: str,
) -> pd.DataFrame:
    """Load and combine all temporal-bucket summaries for one metric once."""
    frames: list[pd.DataFrame] = []

    for order, bucket in enumerate(
        TEMPORAL_BUCKET_ORDER,
        start=1,
    ):
        frame = get_zone_pre_post_metric_summary(
            metrics=[metric],
            temporal_bucket=bucket,
        ).copy()

        if frame.empty:
            continue

        frame["temporal_bucket_order"] = order
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    return add_reliability_flags(
        combined
    )


@st.cache_data(show_spinner=False)
def get_zone_temporal_profile(
    *,
    taxi_zone_id: int | float | str,
    metric: str,
    comparison_level: str,
) -> pd.DataFrame:
    """Return selected-zone versus peer-baseline change for all 10 buckets."""
    selected_metadata = get_zone_metadata(
        taxi_zone_id
    )

    source = _get_all_bucket_zone_summaries(
        metric=metric
    )

    if source.empty:
        return pd.DataFrame()

    source = attach_mobility_regime_cluster_context(
        source,
        assignment_period="post_cp",
    )

    rows: list[dict[str, object]] = []

    for bucket in TEMPORAL_BUCKET_ORDER:
        bucket_source = source[
            source["temporal_bucket"].astype(str).eq(
                str(bucket)
            )
        ].copy()

        selected = bucket_source[
            bucket_source["taxi_zone_id"]
            .astype(str)
            .eq(str(taxi_zone_id))
        ]

        if selected.empty:
            continue

        row = selected.iloc[0]

        baseline = _peer_summary_from_zone_summary(
            bucket_source,
            selected_zone_id=taxi_zone_id,
            metric=metric,
            comparison_level=comparison_level,
            selected_metadata=selected_metadata,
        )

        baseline_change = _safe_percent_change(
            baseline["baseline_pre_daily_average"],
            baseline["baseline_post_daily_average"],
        )

        rows.append(
            {
                "temporal_bucket": bucket,
                "temporal_bucket_order": int(
                    row["temporal_bucket_order"]
                ),
                "zone_percent_change": row[
                    "percent_change"
                ],
                "baseline_percent_change": baseline_change,
                "change_gap": (
                    row["percent_change"]
                    - baseline_change
                    if pd.notna(row["percent_change"])
                    and pd.notna(baseline_change)
                    else np.nan
                ),
                "zone_eligible_for_percent_change": bool(
                    row.get(
                        "eligible_for_percent_change",
                        False,
                    )
                ),
                "baseline_peer_zones": baseline[
                    "baseline_peer_zones"
                ],
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def get_zone_rank_context(
    *,
    taxi_zone_id: int | float | str,
    temporal_bucket: str = ALL_TEMPORAL_BUCKETS_LABEL,
) -> pd.DataFrame:
    """Return citywide and Borough rank/percentile for all eligible metrics."""
    source = get_zone_pre_post_metric_summary(
        metrics=BASE_METRICS,
        temporal_bucket=temporal_bucket,
    )

    source = add_reliability_flags(
        source
    )

    selected_metadata = get_zone_metadata(
        taxi_zone_id
    )

    rows: list[dict[str, object]] = []

    for metric in BASE_METRICS:
        eligible = source[
            source["metric"].eq(
                metric
            )
            & source[
                "eligible_for_percent_change"
            ]
            .fillna(False)
        ].copy()

        selected = eligible[
            eligible["taxi_zone_id"]
            .astype(str)
            .eq(
                str(taxi_zone_id)
            )
        ]

        if selected.empty:
            rows.append(
                {
                    "metric": metric,
                    "metric_label": (
                        METRIC_LABELS.get(
                            metric,
                            metric,
                        )
                    ),
                    "eligible": False,
                    "zone_percent_change": np.nan,
                    "citywide_rank": np.nan,
                    "citywide_eligible_zones": int(
                        eligible[
                            "taxi_zone_id"
                        ].nunique()
                    ),
                    "citywide_percentile": np.nan,
                    "borough_rank": np.nan,
                    "borough_eligible_zones": int(
                        eligible.loc[
                            eligible["borough"]
                            .astype(str)
                            .eq(
                                str(
                                    selected_metadata[
                                        "borough"
                                    ]
                                )
                            ),
                            "taxi_zone_id",
                        ].nunique()
                    ),
                    "borough_percentile": np.nan,
                }
            )

            continue

        selected_change = float(
            selected.iloc[0][
                "percent_change"
            ]
        )

        eligible = eligible.sort_values(
            "percent_change",
            ascending=False,
        ).reset_index(drop=True)

        city_matches = eligible[
            eligible["taxi_zone_id"]
            .astype(str)
            .eq(
                str(taxi_zone_id)
            )
        ]

        city_rank = int(
            city_matches.index[0]
            + 1
        )

        borough_eligible = eligible[
            eligible["borough"]
            .astype(str)
            .eq(
                str(
                    selected_metadata[
                        "borough"
                    ]
                )
            )
        ].reset_index(drop=True)

        borough_matches = borough_eligible[
            borough_eligible[
                "taxi_zone_id"
            ]
            .astype(str)
            .eq(
                str(taxi_zone_id)
            )
        ]

        borough_rank = (
            int(
                borough_matches.index[0]
                + 1
            )
            if not borough_matches.empty
            else np.nan
        )

        city_count = len(
            eligible
        )

        borough_count = len(
            borough_eligible
        )

        rows.append(
            {
                "metric": metric,
                "metric_label": (
                    METRIC_LABELS.get(
                        metric,
                        metric,
                    )
                ),
                "eligible": True,
                "zone_percent_change": selected_change,
                "citywide_rank": city_rank,
                "citywide_eligible_zones": city_count,
                "citywide_percentile": (
                    100
                    * (
                        1
                        - (
                            city_rank - 1
                        )
                        / max(
                            city_count - 1,
                            1,
                        )
                    )
                ),
                "borough_rank": borough_rank,
                "borough_eligible_zones": borough_count,
                "borough_percentile": (
                    100
                    * (
                        1
                        - (
                            borough_rank
                            - 1
                        )
                        / max(
                            borough_count
                            - 1,
                            1,
                        )
                    )
                    if pd.notna(
                        borough_rank
                    )
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def get_zone_pairwise_divergences(
    *,
    taxi_zone_id: int | float | str,
    temporal_bucket: str = ALL_TEMPORAL_BUCKETS_LABEL,
) -> pd.DataFrame:
    """Return eligible opposite-direction metric pairs for one selected zone."""
    profile = get_zone_pre_post_profile(
        taxi_zone_id=taxi_zone_id,
        comparison_level="Citywide",
        temporal_bucket=temporal_bucket,
    )

    eligible = profile[
        profile[
            "zone_eligible_for_percent_change"
        ]
        & profile[
            "zone_percent_change"
        ].notna()
    ].copy()

    rows: list[dict[str, object]] = []

    for (
        _,
        metric_a,
    ), (
        _,
        metric_b,
    ) in combinations(
        eligible.iterrows(),
        2,
    ):
        change_a = float(
            metric_a[
                "zone_percent_change"
            ]
        )

        change_b = float(
            metric_b[
                "zone_percent_change"
            ]
        )

        opposite = (
            np.sign(change_a)
            != np.sign(change_b)
            and np.sign(
                change_a
            )
            != 0
            and np.sign(
                change_b
            )
            != 0
        )

        if not opposite:
            continue

        rows.append(
            {
                "metric_a": metric_a[
                    "metric"
                ],
                "metric_a_label": metric_a[
                    "metric_label"
                ],
                "metric_a_change": change_a,
                "metric_b": metric_b[
                    "metric"
                ],
                "metric_b_label": metric_b[
                    "metric_label"
                ],
                "metric_b_change": change_b,
                "divergence_gap": (
                    change_a
                    - change_b
                ),
                "absolute_divergence": abs(
                    change_a
                    - change_b
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "absolute_divergence",
            ascending=False,
        )
        .reset_index(drop=True)
        if rows
        else pd.DataFrame(
            columns=[
                "metric_a",
                "metric_a_label",
                "metric_a_change",
                "metric_b",
                "metric_b_label",
                "metric_b_change",
                "divergence_gap",
                "absolute_divergence",
            ]
        )
    )
