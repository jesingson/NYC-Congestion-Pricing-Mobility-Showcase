"""Runtime data access for Raw 26 · Mobility Day Type Calendar.

The expensive feature engineering, PCA, and clustering live in the standalone
offline builder. This module only validates and reads the compact production
Parquet artifacts used by Streamlit.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_TABLE_DIR = REPO_ROOT / "data" / "processed" / "app_tables"

DAY_TYPES_PATH = APP_TABLE_DIR / "mobility_day_types.parquet"
PROFILES_PATH = APP_TABLE_DIR / "mobility_day_type_profiles.parquet"
DATE_FEATURE_PATH = APP_TABLE_DIR / "mobility_day_type_date_features.parquet"
RUN_PATH = APP_TABLE_DIR / "mobility_day_type_runs.parquet"
QA_PATH = APP_TABLE_DIR / "mobility_day_type_qa.parquet"

DAY_TYPE_REQUIRED_COLUMNS = {
    "date",
    "cluster",
    "day_type_id",
    "day_type_name",
    "day_name",
    "day_of_week",
    "is_weekend",
    "year",
    "month",
    "season",
    "policy_period",
    "distance_to_centroid",
    "typicality_percentile",
    "calendar_week",
}

PROFILE_REQUIRED_COLUMNS = {
    "cluster",
    "day_type_id",
    "day_type_name",
    "feature",
    "metric",
    "metric_label",
    "daypart",
    "daypart_label",
    "relative_level",
    "absolute_relative_level",
    "direction",
    "feature_rank_within_type",
}

QA_REQUIRED_COLUMNS = {
    "check",
    "observed",
    "expected",
    "passed",
    "build_contract",
    "pca_variance_retained",
}


def _require_file(
    path: Path,
) -> None:
    """Explain how to recover when the offline production artifact is absent."""
    if path.exists():
        return

    raise FileNotFoundError(
        "Missing Raw 26 app table:\n"
        f"{path}\n\n"
        "Build the compact production assets from the repository root with:\n"
        "python scripts/build_mobility_day_type_app_tables.py"
    )


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    label: str,
) -> None:
    """Fail loudly if a production artifact no longer matches its contract."""
    missing = sorted(
        required.difference(
            frame.columns
        )
    )

    if missing:
        raise KeyError(
            f"{label} is missing columns: "
            + ", ".join(missing)
        )


@st.cache_data(
    show_spinner="Loading mobility day types..."
)
def load_mobility_day_types() -> pd.DataFrame:
    """Load the compact one-row-per-modeled-date Raw 26 artifact."""
    _require_file(
        DAY_TYPES_PATH
    )

    frame = pd.read_parquet(
        DAY_TYPES_PATH
    )

    _require_columns(
        frame,
        DAY_TYPE_REQUIRED_COLUMNS,
        label="mobility_day_types",
    )

    frame = frame.copy()
    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="raise",
    ).dt.normalize()

    frame["cluster"] = pd.to_numeric(
        frame["cluster"],
        errors="raise",
    ).astype("int64")

    duplicate_dates = int(
        frame["date"].duplicated().sum()
    )

    if duplicate_dates:
        raise ValueError(
            "mobility_day_types contains "
            f"{duplicate_dates:,} duplicate dates."
        )

    return (
        frame.sort_values("date")
        .reset_index(drop=True)
    )


@st.cache_data(
    show_spinner=False
)
def load_mobility_day_type_profiles() -> pd.DataFrame:
    """Load the 3-state × 30-feature interpretation table."""
    _require_file(
        PROFILES_PATH
    )

    frame = pd.read_parquet(
        PROFILES_PATH
    )

    _require_columns(
        frame,
        PROFILE_REQUIRED_COLUMNS,
        label="mobility_day_type_profiles",
    )

    frame = frame.copy()
    frame["cluster"] = pd.to_numeric(
        frame["cluster"],
        errors="raise",
    ).astype("int64")

    duplicate_features = int(
        frame.duplicated(
            [
                "cluster",
                "feature",
            ]
        ).sum()
    )

    if duplicate_features:
        raise ValueError(
            "mobility_day_type_profiles contains "
            f"{duplicate_features:,} duplicate "
            "cluster × feature rows."
        )

    return frame.reset_index(
        drop=True
    )


@st.cache_data(
    show_spinner=False
)
def load_mobility_day_type_qa() -> pd.DataFrame:
    """Load and verify the tiny build-time QA contract."""
    _require_file(
        QA_PATH
    )

    frame = pd.read_parquet(
        QA_PATH
    )

    _require_columns(
        frame,
        QA_REQUIRED_COLUMNS,
        label="mobility_day_type_qa",
    )

    if not frame["passed"].fillna(False).all():
        failed = frame.loc[
            ~frame["passed"].fillna(False)
        ]

        raise RuntimeError(
            "The stored Raw 26 build did not pass QA:\n"
            + failed.to_string(index=False)
        )

    return frame.reset_index(
        drop=True
    )


def get_mobility_day_type_date_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the first and last modeled dates available to the page."""
    frame = load_mobility_day_types()

    if frame.empty:
        raise ValueError(
            "mobility_day_types is empty."
        )

    return (
        frame["date"].min(),
        frame["date"].max(),
    )


def get_mobility_day_type_summary() -> pd.DataFrame:
    """Return compact state size, persistence, and period composition context."""
    frame = load_mobility_day_types().copy()

    summary = (
        frame.groupby(
            [
                "cluster",
                "day_type_id",
            ],
            observed=True,
        )
        .agg(
            days=("date", "size"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            weekend_share=("is_weekend", "mean"),
            post_cp_share=(
                "policy_period",
                lambda values: float(
                    values.eq("Post-CP").mean()
                ),
            ),
            median_typicality=(
                "typicality_percentile",
                "median",
            ),
        )
        .reset_index()
    )

    summary["share_of_days"] = (
        summary["days"]
        / summary["days"].sum()
    )

    return summary


def get_representative_mobility_days(
    *,
    per_type: int = 5,
) -> pd.DataFrame:
    """Return the most centroid-proximate dates for each frozen state."""
    if per_type < 1:
        raise ValueError(
            "per_type must be at least 1."
        )

    frame = load_mobility_day_types()

    return (
        frame.sort_values(
            [
                "cluster",
                "distance_to_centroid",
                "date",
            ]
        )
        .groupby(
            "cluster",
            observed=True,
            group_keys=False,
        )
        .head(per_type)
        .reset_index(drop=True)
    )


def get_top_profile_features(
    *,
    per_type: int = 6,
) -> pd.DataFrame:
    """Return the strongest absolute profile signals for each frozen state."""
    if per_type < 1:
        raise ValueError(
            "per_type must be at least 1."
        )

    profiles = (
        load_mobility_day_type_profiles()
        .sort_values(
            [
                "cluster",
                "feature_rank_within_type",
            ]
        )
    )

    return (
        profiles.groupby(
            "cluster",
            observed=True,
            group_keys=False,
        )
        .head(per_type)
        .reset_index(drop=True)
    )



@st.cache_data(show_spinner=False)
def load_mobility_day_type_date_features() -> pd.DataFrame:
    """Load per-date same-weekday-relative feature values for Raw 26."""
    _require_file(DATE_FEATURE_PATH)
    frame = pd.read_parquet(DATE_FEATURE_PATH)

    required = {
        "date",
        "cluster",
        "day_type_id",
        "day_type_name",
        "feature",
        "metric",
        "metric_label",
        "daypart",
        "daypart_label",
        "relative_level",
        "absolute_relative_level",
        "direction",
    }
    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            "Mobility day-type date-feature table is missing: "
            + ", ".join(sorted(missing))
        )

    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(
        ["date", "metric", "daypart"]
    ).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_mobility_day_type_runs() -> pd.DataFrame:
    """Load consecutive mobility-state runs for persistence views."""
    _require_file(RUN_PATH)
    frame = pd.read_parquet(RUN_PATH)

    required = {
        "run_number",
        "day_type_id",
        "day_type_name",
        "cluster",
        "start_date",
        "end_date",
        "duration_days",
        "start_year",
        "end_year",
        "crosses_year",
    }
    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            "Mobility day-type run table is missing: "
            + ", ".join(sorted(missing))
        )

    frame["start_date"] = pd.to_datetime(frame["start_date"])
    frame["end_date"] = pd.to_datetime(frame["end_date"])
    return frame.sort_values(
        "start_date"
    ).reset_index(drop=True)
