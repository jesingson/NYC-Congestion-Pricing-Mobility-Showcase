"""Runtime data access for Raw 27 · Mobility Drivers.

The expensive geography aggregation and normalization happen offline in
scripts/build_mobility_driver_app_tables.py. Streamlit reads only compact,
production-ready geography × date × daypart signals.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_TABLE_DIR = REPO_ROOT / "data" / "processed" / "app_tables"

DRIVER_PATH = APP_TABLE_DIR / "mobility_driver_geographies.parquet"
CONTEXT_PATH = APP_TABLE_DIR / "mobility_driver_geography_context.parquet"
QA_PATH = APP_TABLE_DIR / "mobility_driver_qa.parquet"

GEOGRAPHY_LEVELS = (
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

REQUIRED_DRIVER_COLUMNS = {
    "geography_level",
    "geography_value",
    "date",
    "daypart",
    *METRICS,
}

REQUIRED_CONTEXT_COLUMNS = {
    "geography_level",
    "geography_value",
    "display_label",
}

REQUIRED_QA_COLUMNS = {
    "check",
    "observed",
    "expected",
    "passed",
}


def _require_file(path: Path) -> None:
    """Explain the one-step recovery when a Raw 27 app table is absent."""
    if path.exists():
        return

    raise FileNotFoundError(
        "Missing Raw 27 app table:\n"
        f"{path}\n\n"
        "Build the production assets from the repository root with:\n"
        "python scripts/build_mobility_driver_app_tables.py"
    )


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    label: str,
) -> None:
    """Fail loudly when a production artifact no longer matches its contract."""
    missing = sorted(required.difference(frame.columns))

    if missing:
        raise KeyError(
            f"{label} is missing columns: "
            + ", ".join(missing)
        )


@st.cache_data(show_spinner="Loading mobility drivers...")
def load_mobility_driver_geographies() -> pd.DataFrame:
    """Load standardized geography × date × daypart driver signals."""
    _require_file(DRIVER_PATH)
    frame = pd.read_parquet(DRIVER_PATH)
    _require_columns(
        frame,
        REQUIRED_DRIVER_COLUMNS,
        label="mobility_driver_geographies",
    )

    frame = frame.copy()
    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="raise",
    ).dt.normalize()

    grain = [
        "geography_level",
        "geography_value",
        "date",
        "daypart",
    ]
    duplicates = int(frame.duplicated(grain).sum())

    if duplicates:
        raise ValueError(
            "mobility_driver_geographies contains "
            f"{duplicates:,} duplicate rows at its production grain."
        )

    return frame.sort_values(grain).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_mobility_driver_geography_context() -> pd.DataFrame:
    """Load values and display labels for the standard two-stage geography UI."""
    _require_file(CONTEXT_PATH)
    frame = pd.read_parquet(CONTEXT_PATH)
    _require_columns(
        frame,
        REQUIRED_CONTEXT_COLUMNS,
        label="mobility_driver_geography_context",
    )

    return frame.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_mobility_driver_qa() -> pd.DataFrame:
    """Load and require the tiny build-time QA contract."""
    _require_file(QA_PATH)
    frame = pd.read_parquet(QA_PATH)
    _require_columns(
        frame,
        REQUIRED_QA_COLUMNS,
        label="mobility_driver_qa",
    )

    if not frame["passed"].fillna(False).all():
        failed = frame.loc[
            ~frame["passed"].fillna(False)
        ]
        raise RuntimeError(
            "The stored Raw 27 build did not pass QA:\n"
            + failed.to_string(index=False)
        )

    return frame.reset_index(drop=True)
