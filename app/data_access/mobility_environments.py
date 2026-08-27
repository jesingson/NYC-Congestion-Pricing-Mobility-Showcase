from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


APP_ROOT = Path(__file__).resolve().parents[2]
CLUSTER_DATA_DIR = APP_ROOT / "data" / "processed" / "3.2.2.final_tables"

CANONICAL_CLUSTER_ASSIGNMENTS_PATH = (
    CLUSTER_DATA_DIR / "canonical_cluster_assignments-20260803-193800.parquet"
)
CANONICAL_CLUSTER_NAME_LOOKUP_PATH = (
    CLUSTER_DATA_DIR / "canonical_cluster_name_lookup.csv"
)

MOBILITY_REGIME_CLUSTER_OPTIONS = [
    0,
    1,
    2,
    3,
    4,
]

MOBILITY_REGIME_CLUSTER_PERIOD_OPTIONS = [
    "Post-CP assignment",
    "Pre-CP assignment",
]

MOBILITY_REGIME_CLUSTER_PERIOD_TO_VALUE = {
    "Post-CP assignment": "post_cp",
    "Pre-CP assignment": "pre_cp",
}

MOBILITY_ENVIRONMENT_PERIOD_OPTIONS = (
    MOBILITY_REGIME_CLUSTER_PERIOD_OPTIONS
)
MOBILITY_ENVIRONMENT_PERIOD_TO_VALUE = (
    MOBILITY_REGIME_CLUSTER_PERIOD_TO_VALUE
)


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected data file: {path}")


def _coerce_cluster_label(value: object) -> int | None:
    if pd.isna(value):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@st.cache_data(show_spinner="Loading mobility regime cluster lookup...")
def load_mobility_regime_cluster_lookup() -> pd.DataFrame:
    _require_file(CANONICAL_CLUSTER_NAME_LOOKUP_PATH)

    lookup = pd.read_csv(CANONICAL_CLUSTER_NAME_LOOKUP_PATH)

    expected = {"cluster_label", "canonical_cluster_name"}
    missing = expected.difference(lookup.columns)
    if missing:
        raise ValueError(
            "Unexpected mobility regime cluster lookup columns: "
            f"{sorted(lookup.columns)}"
        )

    lookup = lookup.copy()
    lookup["cluster_label"] = pd.to_numeric(
        lookup["cluster_label"],
        errors="coerce",
    ).astype("Int64")

    return lookup.sort_values("cluster_label").reset_index(drop=True)


@st.cache_data(show_spinner="Loading mobility regime cluster assignments...")
def load_mobility_regime_cluster_assignments() -> pd.DataFrame:
    _require_file(CANONICAL_CLUSTER_ASSIGNMENTS_PATH)

    assignments = pd.read_parquet(CANONICAL_CLUSTER_ASSIGNMENTS_PATH)

    expected = {
        "solution_id",
        "cluster_method",
        "cluster_count",
        "taxi_zone_id",
        "zone",
        "borough",
        "canonical_location_id",
        "cbd_spatial_category",
        "pre_post_cp",
        "cluster_label",
        "canonical_cluster_name",
    }
    missing = expected.difference(assignments.columns)
    if missing:
        raise ValueError(
            "Unexpected mobility regime cluster assignment columns: "
            f"{sorted(assignments.columns)}"
        )

    assignments = assignments.copy()
    assignments["cluster_label"] = pd.to_numeric(
        assignments["cluster_label"],
        errors="coerce",
    ).astype("Int64")

    return assignments


def get_mobility_regime_cluster_options() -> list[int]:
    lookup = load_mobility_regime_cluster_lookup()
    return [
        int(label)
        for label in lookup["cluster_label"].dropna().tolist()
    ]


def format_mobility_regime_cluster_label(
    cluster_label: object,
) -> str:
    numeric_label = _coerce_cluster_label(cluster_label)

    if numeric_label is None:
        return "Cluster Unknown"

    lookup = load_mobility_regime_cluster_lookup()
    match = lookup[
        lookup["cluster_label"].eq(numeric_label)
    ]

    if match.empty:
        return f"Cluster {numeric_label}"

    cluster_name = str(
        match.iloc[0]["canonical_cluster_name"]
    )
    return f"Cluster {numeric_label} · {cluster_name}"


def get_mobility_environment_options() -> list[str]:
    lookup = load_mobility_regime_cluster_lookup()
    return lookup["canonical_cluster_name"].tolist()


def attach_mobility_regime_cluster_context(
    df: pd.DataFrame,
    *,
    assignment_period: str,
) -> pd.DataFrame:
    period = assignment_period.strip().lower()
    if period not in {"pre_cp", "post_cp"}:
        raise ValueError(
            f"Unsupported assignment_period: {assignment_period}"
        )

    assignments = load_mobility_regime_cluster_assignments()
    assignments = assignments[
        assignments["pre_post_cp"].astype(str).eq(period)
    ][
        [
            "taxi_zone_id",
            "cluster_label",
            "canonical_cluster_name",
        ]
    ].drop_duplicates()

    result = df.merge(
        assignments,
        on="taxi_zone_id",
        how="left",
        validate="many_to_one",
    )

    result = result.rename(
        columns={
            "cluster_label": "mobility_regime_cluster_label",
            "canonical_cluster_name": "mobility_regime_cluster_name",
        }
    )
    result["mobility_environment"] = result[
        "mobility_regime_cluster_name"
    ]

    return result


def load_canonical_cluster_name_lookup() -> pd.DataFrame:
    return load_mobility_regime_cluster_lookup()


def load_canonical_cluster_assignments() -> pd.DataFrame:
    return load_mobility_regime_cluster_assignments()


def attach_mobility_environment(
    df: pd.DataFrame,
    *,
    assignment_period: str,
) -> pd.DataFrame:
    return attach_mobility_regime_cluster_context(
        df,
        assignment_period=assignment_period,
    )
