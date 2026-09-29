"""Build compact production tables for Raw 26 · Mobility Day Type Calendar.

This is an OFFLINE build step. The Streamlit page should never refit PCA or
KMeans at runtime.

Frozen analytical design
------------------------
- Unit: one NYC-wide calendar date.
- Signature: six headline mobility metrics × five dayparts = 30 features.
- Counts: summed across Taxi Zones, then log1p transformed.
- Speeds: activity-weighted across Taxi Zones.
- Calendar adjustment: each feature is centered on its same-weekday median and
  divided by its same-weekday IQR.
- PCA: retain at least 90% of variance; this yields 10 components and 91.7% actual retained variance in the frozen source data.
- Clustering: KMeans, k=3, n_init=50, random_state=696.

The builder writes small app-facing Parquet files under
data/processed/app_tables/. Reader-facing archetype names are intentionally
kept separate from the frozen numeric cluster IDs so editorial naming can be
finalized without refitting the model.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from app.data_access.aggregations import (
    COUNT_METRICS,
    WEIGHT_COLUMNS,
    get_required_columns,
)
from app.data_access.loaders import (
    CONGESTION_PRICING_START_DATE,
    CORE_METRICS,
    DAYPART_ORDER,
    METRIC_LABELS,
    load_analysis_panel,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_TABLE_DIR = REPO_ROOT / "data" / "processed" / "app_tables"

DAY_TYPES_PATH = APP_TABLE_DIR / "mobility_day_types.parquet"
PROFILES_PATH = APP_TABLE_DIR / "mobility_day_type_profiles.parquet"
DATE_FEATURES_PATH = APP_TABLE_DIR / "mobility_day_type_date_features.parquet"
RUNS_PATH = APP_TABLE_DIR / "mobility_day_type_runs.parquet"
QA_PATH = APP_TABLE_DIR / "mobility_day_type_qa.parquet"

MODEL_METRICS = tuple(CORE_METRICS)
MODEL_DAYPARTS = tuple(DAYPART_ORDER)

PCA_VARIANCE = 0.90
CLUSTER_COUNT = 3
RANDOM_STATE = 696
KMEANS_N_INIT = 50

DAY_TYPE_NAMES = {
    "faster_lighter": "Faster, Lighter",
    "holiday_like": "Holiday-Like",
    "busier_slower": "Busier, Slower",
}

COUNT_MODEL_METRICS = tuple(
    metric
    for metric in MODEL_METRICS
    if metric in COUNT_METRICS
)

SPEED_MODEL_METRICS = tuple(
    metric
    for metric in MODEL_METRICS
    if metric in WEIGHT_COLUMNS
)


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    label: str,
) -> None:
    """Fail loudly if an upstream contract changes."""
    missing = sorted(required.difference(frame.columns))

    if missing:
        raise KeyError(
            f"{label} is missing required columns: "
            + ", ".join(missing)
        )


def _safe_weighted_average(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    """Match the Showcase weighted-mean policy, including mean fallback."""
    valid = (
        values.notna()
        & weights.notna()
        & weights.gt(0)
    )

    if valid.any():
        return float(
            np.average(
                values.loc[valid].astype(float),
                weights=weights.loc[valid].astype(float),
            )
        )

    finite = values.dropna()

    if finite.empty:
        return np.nan

    return float(finite.mean())


def _load_source_panel() -> pd.DataFrame:
    """Load only columns required to reproduce the frozen 30-feature design."""
    columns = get_required_columns(list(MODEL_METRICS))
    panel = load_analysis_panel(columns=columns).copy()

    _require_columns(
        panel,
        {
            "date",
            "temporal_bucket",
            *MODEL_METRICS,
            *[
                WEIGHT_COLUMNS[metric]
                for metric in SPEED_MODEL_METRICS
            ],
        },
        label="analysis_ready_mobility_panel",
    )

    panel["date"] = pd.to_datetime(
        panel["date"],
        errors="raise",
    ).dt.normalize()

    # The source temporal bucket carries weekday/weekend + daypart. Raw 26
    # deliberately strips the week-part prefix because same-weekday adjustment
    # handles ordinary weekly rhythm later.
    panel["daypart"] = (
        panel["temporal_bucket"]
        .astype(str)
        .str.replace(
            r"^(weekday|weekend)_",
            "",
            regex=True,
        )
    )

    panel = panel[
        panel["daypart"].isin(MODEL_DAYPARTS)
    ].copy()

    return panel


def _build_city_daypart_panel(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse Taxi Zone rows to NYC × date × daypart using app metric rules."""
    rows: list[dict[str, object]] = []

    for (date, daypart), group in panel.groupby(
        ["date", "daypart"],
        observed=True,
        sort=True,
    ):
        row: dict[str, object] = {
            "date": date,
            "daypart": daypart,
        }

        for metric in COUNT_MODEL_METRICS:
            row[metric] = group[metric].sum(
                min_count=1
            )

        for metric in SPEED_MODEL_METRICS:
            weight_column = WEIGHT_COLUMNS[metric]
            row[metric] = _safe_weighted_average(
                group[metric],
                group[weight_column],
            )

        rows.append(row)

    city = pd.DataFrame(rows)

    coverage = (
        city.groupby(
            "date",
            observed=True,
        )["daypart"]
        .nunique()
    )

    complete_dates = coverage[
        coverage.eq(len(MODEL_DAYPARTS))
    ].index

    return (
        city[
            city["date"].isin(complete_dates)
        ]
        .sort_values(["date", "daypart"])
        .reset_index(drop=True)
    )


def _build_feature_matrix(
    city: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return transformed model features and native-value features."""
    pivot = city.pivot(
        index="date",
        columns="daypart",
        values=list(MODEL_METRICS),
    )

    expected_columns = pd.MultiIndex.from_product(
        [MODEL_METRICS, MODEL_DAYPARTS]
    )

    pivot = pivot.reindex(
        columns=expected_columns
    )

    complete = pivot.dropna(
        axis=0,
        how="any",
    ).copy()

    native = complete.copy()

    # WHY: log1p keeps exceptionally busy dates from dominating Euclidean
    # distance while preserving the ordering of non-negative demand counts.
    for metric in COUNT_MODEL_METRICS:
        for daypart in MODEL_DAYPARTS:
            complete[(metric, daypart)] = np.log1p(
                complete[(metric, daypart)].clip(lower=0)
            )

    flat_columns = [
        f"{metric}__{daypart}"
        for metric, daypart in complete.columns
    ]

    complete.columns = flat_columns
    native.columns = flat_columns

    return complete, native


def _calendar_adjust(
    transformed: pd.DataFrame,
) -> pd.DataFrame:
    """Express every feature relative to what is normal for the same weekday."""
    adjusted = pd.DataFrame(
        index=transformed.index,
        columns=transformed.columns,
        dtype=float,
    )

    weekday = pd.Series(
        transformed.index.dayofweek,
        index=transformed.index,
        name="weekday",
    )

    for column in transformed.columns:
        values = transformed[column]

        median = values.groupby(
            weekday
        ).transform("median")

        q25 = values.groupby(
            weekday
        ).transform(
            lambda x: x.quantile(0.25)
        )

        q75 = values.groupby(
            weekday
        ).transform(
            lambda x: x.quantile(0.75)
        )

        iqr = (q75 - q25).mask(
            (q75 - q25).eq(0),
            1.0,
        )

        adjusted[column] = (
            values - median
        ) / iqr

    return adjusted


def _fit_frozen_solution(
    adjusted: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, PCA, KMeans]:
    """Fit the frozen calendar-adjusted PCA + k=3 solution."""
    # IMPORTANT: no second global RobustScaler is applied here. The features
    # are already in same-weekday IQR units; another scaling pass would alter
    # the interpretation that was selected during scouting.
    pca = PCA(
        n_components=PCA_VARIANCE,
        svd_solver="full",
        random_state=RANDOM_STATE,
    )

    embedded = pca.fit_transform(
        adjusted.to_numpy(dtype=float)
    )

    model = KMeans(
        n_clusters=CLUSTER_COUNT,
        n_init=KMEANS_N_INIT,
        random_state=RANDOM_STATE,
    )

    labels = model.fit_predict(
        embedded
    )

    return embedded, labels, pca, model



def _derive_semantic_cluster_map(
    adjusted: pd.DataFrame,
    labels: np.ndarray,
) -> dict[int, str]:
    """Map arbitrary KMeans labels to stable mobility-state identifiers.

    WHY: KMeans numeric labels have no semantic meaning. The production
    contract should survive a harmless label permutation, so names are derived
    from the frozen 30-feature profiles rather than from cluster number.
    """
    profile_source = adjusted.copy()
    profile_source["cluster"] = labels

    profiles = (
        profile_source.groupby("cluster", observed=True)
        .median()
        .T
    )

    count_features = [
        column
        for column in adjusted.columns
        if column.split("__", 1)[0] in COUNT_MODEL_METRICS
    ]
    speed_features = [
        column
        for column in adjusted.columns
        if column.split("__", 1)[0] in SPEED_MODEL_METRICS
    ]

    # Holiday-Like is the small, extreme state: its median profile departs far
    # more strongly from same-weekday norms than either broad recurring state.
    profile_magnitude = profiles.abs().mean(axis=0)
    holiday_cluster = int(profile_magnitude.idxmax())

    remaining = [
        int(cluster)
        for cluster in profiles.columns
        if int(cluster) != holiday_cluster
    ]

    contrast = {}

    for cluster in remaining:
        demand_level = float(
            profiles.loc[count_features, cluster].mean()
        )
        speed_level = float(
            profiles.loc[speed_features, cluster].mean()
        )

        # Positive = more demand + slower roads; negative = lighter + faster.
        contrast[cluster] = demand_level - speed_level

    busier_cluster = max(
        contrast,
        key=contrast.get,
    )
    faster_cluster = min(
        contrast,
        key=contrast.get,
    )

    mapping = {
        holiday_cluster: "holiday_like",
        busier_cluster: "busier_slower",
        faster_cluster: "faster_lighter",
    }

    if set(mapping.values()) != set(DAY_TYPE_NAMES):
        raise RuntimeError(
            "Could not derive all three frozen mobility-state identities."
        )

    return mapping


def _build_day_table(
    adjusted: pd.DataFrame,
    embedded: np.ndarray,
    labels: np.ndarray,
    model: KMeans,
    semantic_map: dict[int, str],
) -> pd.DataFrame:
    """Create the compact one-row-per-modeled-date runtime artifact."""
    dates = pd.DatetimeIndex(adjusted.index)

    day_table = pd.DataFrame(
        {
            "date": dates,
            "cluster": labels.astype("int64"),
        }
    )

    day_table["day_type_id"] = (
        day_table["cluster"].map(semantic_map)
    )
    day_table["day_type_name"] = (
        day_table["day_type_id"].map(DAY_TYPE_NAMES)
    )

    day_table["day_name"] = (
        day_table["date"].dt.day_name()
    )
    day_table["day_of_week"] = (
        day_table["date"].dt.dayofweek.astype("int8")
    )
    day_table["is_weekend"] = (
        day_table["day_of_week"].ge(5)
    )
    day_table["year"] = (
        day_table["date"].dt.year.astype("int16")
    )
    day_table["month"] = (
        day_table["date"].dt.month.astype("int8")
    )

    day_table["season"] = np.select(
        [
            day_table["month"].isin([12, 1, 2]),
            day_table["month"].isin([3, 4, 5]),
            day_table["month"].isin([6, 7, 8]),
        ],
        [
            "Winter",
            "Spring",
            "Summer",
        ],
        default="Fall",
    )

    day_table["policy_period"] = np.where(
        day_table["date"]
        < CONGESTION_PRICING_START_DATE,
        "Pre-CP",
        "Post-CP",
    )

    # Distances support representative-date selection and a reader-facing
    # "typicality" measure without shipping sklearn objects to Streamlit.
    centroid = model.cluster_centers_[labels]
    distances = np.linalg.norm(
        embedded - centroid,
        axis=1,
    )

    day_table["distance_to_centroid"] = distances

    distance_series = pd.Series(
        distances,
        index=day_table.index,
    )

    # Within each state, the closest date receives the highest typicality.
    day_table["typicality_percentile"] = (
        1.0
        - distance_series.groupby(
            day_table["cluster"]
        ).rank(
            method="average",
            pct=True,
        )
        + distance_series.groupby(
            day_table["cluster"]
        ).transform("count").rpow(-1)
    ).clip(0.0, 1.0)

    # PC coordinates are tiny to store and useful for optional details-on-demand.
    for component in range(
        min(3, embedded.shape[1])
    ):
        day_table[
            f"pc{component + 1}"
        ] = embedded[:, component]

    # Calendar coordinates are deterministic and save duplicate page logic.
    year_start = pd.to_datetime(
        day_table["year"].astype(str)
        + "-01-01"
    )

    day_table["calendar_week"] = (
        (
            day_table["date"] - year_start
        ).dt.days
        + year_start.dt.dayofweek
    ) // 7

    return (
        day_table.sort_values("date")
        .reset_index(drop=True)
    )


def _build_profile_table(
    adjusted: pd.DataFrame,
    labels: np.ndarray,
    semantic_map: dict[int, str],
) -> pd.DataFrame:
    """Create the full 3 × 30 descriptive profile used to explain each state."""
    profile_source = adjusted.copy()
    profile_source["cluster"] = labels

    wide = (
        profile_source.groupby(
            "cluster",
            observed=True,
        )
        .median()
        .T
    )

    wide.index.name = "feature"

    profile = (
        wide.rename_axis(
            columns="cluster"
        )
        .stack()
        .rename("relative_level")
        .reset_index()
    )

    split = profile["feature"].str.split(
        "__",
        n=1,
        expand=True,
    )

    profile["metric"] = split[0]
    profile["daypart"] = split[1]
    profile["metric_label"] = profile[
        "metric"
    ].map(METRIC_LABELS)

    daypart_labels = {
        "overnight": "Overnight",
        "am_peak": "AM peak",
        "midday": "Midday",
        "pm_peak": "PM peak",
        "evening": "Evening",
    }

    profile["daypart_label"] = profile[
        "daypart"
    ].map(daypart_labels)

    profile["day_type_id"] = profile[
        "cluster"
    ].map(semantic_map)
    profile["day_type_name"] = profile[
        "day_type_id"
    ].map(DAY_TYPE_NAMES)

    profile["direction"] = np.select(
        [
            profile["relative_level"].gt(0),
            profile["relative_level"].lt(0),
        ],
        [
            "Above same-weekday typical",
            "Below same-weekday typical",
        ],
        default="At same-weekday typical",
    )

    profile["absolute_relative_level"] = (
        profile["relative_level"].abs()
    )

    profile["feature_rank_within_type"] = (
        profile.groupby(
            "cluster",
            observed=True,
        )["absolute_relative_level"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype("int16")
    )

    return (
        profile[
            [
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
            ]
        ]
        .sort_values(
            [
                "cluster",
                "feature_rank_within_type",
            ]
        )
        .reset_index(drop=True)
    )



def _build_date_feature_table(
    adjusted: pd.DataFrame,
    day_table: pd.DataFrame,
) -> pd.DataFrame:
    """Preserve each date's 30 same-weekday-relative mobility values.

    WHY: The Streamlit page can explain a selected day without reopening the
    large source panel or rerunning preprocessing at page load.
    """
    feature_table = (
        adjusted.rename_axis(index="date", columns="feature")
        .stack()
        .rename("relative_level")
        .reset_index()
    )

    identity = day_table[
        [
            "date",
            "cluster",
            "day_type_id",
            "day_type_name",
        ]
    ]

    feature_table = feature_table.merge(
        identity,
        on="date",
        how="left",
        validate="many_to_one",
    )

    parts = feature_table["feature"].str.split(
        "__",
        n=1,
        expand=True,
    )
    feature_table["metric"] = parts[0]
    feature_table["daypart"] = parts[1]
    feature_table["metric_label"] = feature_table[
        "metric"
    ].map(METRIC_LABELS)
    feature_table["daypart_label"] = (
        feature_table["daypart"]
        .str.replace("_", " ", regex=False)
        .str.title()
    )
    feature_table["absolute_relative_level"] = feature_table[
        "relative_level"
    ].abs()
    feature_table["direction"] = np.select(
        [
            feature_table["relative_level"].gt(0),
            feature_table["relative_level"].lt(0),
        ],
        [
            "Above same-weekday norm",
            "Below same-weekday norm",
        ],
        default="At same-weekday norm",
    )

    return (
        feature_table[
            [
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
            ]
        ]
        .sort_values(
            ["date", "metric", "daypart"]
        )
        .reset_index(drop=True)
    )


def _build_run_table(
    day_table: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize consecutive dates that remain in the same mobility state."""
    ordered = (
        day_table.sort_values("date")
        .reset_index(drop=True)
        .copy()
    )

    date_gap = ordered["date"].diff().dt.days.ne(1)
    state_change = ordered["day_type_id"].ne(
        ordered["day_type_id"].shift()
    )

    ordered["run_number"] = (
        (date_gap | state_change)
        .cumsum()
        .astype(int)
    )

    runs = (
        ordered.groupby(
            "run_number",
            as_index=False,
            observed=True,
        )
        .agg(
            day_type_id=("day_type_id", "first"),
            day_type_name=("day_type_name", "first"),
            cluster=("cluster", "first"),
            start_date=("date", "min"),
            end_date=("date", "max"),
            duration_days=("date", "size"),
        )
    )

    runs["start_year"] = runs["start_date"].dt.year
    runs["end_year"] = runs["end_date"].dt.year
    runs["crosses_year"] = (
        runs["start_year"].ne(runs["end_year"])
    )

    return runs.sort_values(
        "start_date"
    ).reset_index(drop=True)


def _build_qa_table(
    *,
    panel: pd.DataFrame,
    city: pd.DataFrame,
    transformed: pd.DataFrame,
    adjusted: pd.DataFrame,
    day_table: pd.DataFrame,
    profile_table: pd.DataFrame,
    date_feature_table: pd.DataFrame,
    run_table: pd.DataFrame,
    pca: PCA,
) -> pd.DataFrame:
    """Record the frozen contract and fail-fast checks in a tiny artifact."""
    expected_feature_count = (
        len(MODEL_METRICS)
        * len(MODEL_DAYPARTS)
    )

    checks = [
        (
            "source_unique_dates",
            int(panel["date"].nunique()),
            1186,
            int(panel["date"].nunique()) == 1186,
        ),
        (
            "five_daypart_dates",
            int(city["date"].nunique()),
            1186,
            int(city["date"].nunique()) == 1186,
        ),
        (
            "modeled_dates",
            int(len(transformed)),
            1184,
            int(len(transformed)) == 1184,
        ),
        (
            "feature_count",
            int(transformed.shape[1]),
            expected_feature_count,
            int(transformed.shape[1]) == expected_feature_count,
        ),
        (
            "day_table_rows",
            int(len(day_table)),
            1184,
            int(len(day_table)) == 1184,
        ),
        (
            "day_table_duplicate_dates",
            int(day_table["date"].duplicated().sum()),
            0,
            not day_table["date"].duplicated().any(),
        ),
        (
            "cluster_count",
            int(day_table["cluster"].nunique()),
            CLUSTER_COUNT,
            int(day_table["cluster"].nunique()) == CLUSTER_COUNT,
        ),
        (
            "profile_rows",
            int(len(profile_table)),
            CLUSTER_COUNT * expected_feature_count,
            int(len(profile_table))
            == CLUSTER_COUNT * expected_feature_count,
        ),
        (
            "adjusted_nonfinite_cells",
            int(
                (~np.isfinite(
                    adjusted.to_numpy(dtype=float)
                )).sum()
            ),
            0,
            bool(
                np.isfinite(
                    adjusted.to_numpy(dtype=float)
                ).all()
            ),
        ),
        (
            "date_feature_rows",
            len(date_feature_table),
            len(day_table) * len(adjusted.columns),
            len(date_feature_table)
            == len(day_table) * len(adjusted.columns),
        ),
        (
            "date_feature_duplicate_keys",
            int(
                date_feature_table.duplicated(
                    ["date", "feature"]
                ).sum()
            ),
            0,
            int(
                date_feature_table.duplicated(
                    ["date", "feature"]
                ).sum()
            )
            == 0,
        ),
        (
            "run_rows",
            len(run_table),
            len(run_table),
            len(run_table) > 0,
        ),
        (
            "pca_components",
            int(pca.n_components_),
            10,
            int(pca.n_components_) == 10,
        ),
        (
            "faster_lighter_days",
            int(day_table["day_type_id"].eq("faster_lighter").sum()),
            500,
            int(day_table["day_type_id"].eq("faster_lighter").sum()) == 500,
        ),
        (
            "holiday_like_days",
            int(day_table["day_type_id"].eq("holiday_like").sum()),
            50,
            int(day_table["day_type_id"].eq("holiday_like").sum()) == 50,
        ),
        (
            "busier_slower_days",
            int(day_table["day_type_id"].eq("busier_slower").sum()),
            634,
            int(day_table["day_type_id"].eq("busier_slower").sum()) == 634,
        ),
    ]

    qa = pd.DataFrame(
        checks,
        columns=[
            "check",
            "observed",
            "expected",
            "passed",
        ],
    )

    qa["build_contract"] = (
        "calendar_adjusted_pca90_kmeans_k3_rs696"
    )
    qa["pca_variance_retained"] = float(
        pca.explained_variance_ratio_.sum()
    )

    return qa


def build_mobility_day_type_app_tables() -> dict[str, pd.DataFrame]:
    """Build and validate every Raw 26 production artifact in memory."""
    total_start = perf_counter()

    print(
        "Loading Raw 26 source columns...",
        flush=True,
    )
    panel = _load_source_panel()

    print(
        "Aggregating NYC × date × daypart...",
        flush=True,
    )
    city = _build_city_daypart_panel(
        panel
    )

    print(
        "Building 30-feature daily signatures...",
        flush=True,
    )
    transformed, _ = _build_feature_matrix(
        city
    )

    print(
        "Applying same-weekday median/IQR adjustment...",
        flush=True,
    )
    adjusted = _calendar_adjust(
        transformed
    )

    print(
        "Fitting frozen PCA + k=3 solution...",
        flush=True,
    )
    embedded, labels, pca, model = (
        _fit_frozen_solution(
            adjusted
        )
    )

    semantic_map = _derive_semantic_cluster_map(
        adjusted,
        labels,
    )

    day_table = _build_day_table(
        adjusted,
        embedded,
        labels,
        model,
        semantic_map,
    )

    profile_table = _build_profile_table(
        adjusted,
        labels,
        semantic_map,
    )

    date_feature_table = _build_date_feature_table(
        adjusted,
        day_table,
    )

    run_table = _build_run_table(
        day_table,
    )

    qa = _build_qa_table(
        panel=panel,
        city=city,
        transformed=transformed,
        adjusted=adjusted,
        day_table=day_table,
        profile_table=profile_table,
        date_feature_table=date_feature_table,
        run_table=run_table,
        pca=pca,
    )

    failed = qa.loc[
        ~qa["passed"]
    ]

    if not failed.empty:
        raise RuntimeError(
            "Raw 26 production build failed frozen-contract QA:\n"
            + failed.to_string(index=False)
        )

    print(
        "Raw 26 build validated in "
        f"{perf_counter() - total_start:,.1f} seconds.",
        flush=True,
    )

    return {
        "day_types": day_table,
        "profiles": profile_table,
        "date_features": date_feature_table,
        "runs": run_table,
        "qa": qa,
    }


def write_mobility_day_type_app_tables(
    tables: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    """Write validated Raw 26 app tables atomically enough for local builds."""
    APP_TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "day_types": DAY_TYPES_PATH,
        "profiles": PROFILES_PATH,
        "date_features": DATE_FEATURES_PATH,
        "runs": RUNS_PATH,
        "qa": QA_PATH,
    }

    for name, path in paths.items():
        frame = tables[name]

        # WHY: write to a sibling temp file first so an interrupted build does
        # not leave a half-written production parquet.
        temp_path = path.with_name(
            f".{path.name}.tmp"
        )

        frame.to_parquet(
            temp_path,
            index=False,
        )
        temp_path.replace(path)

        print(
            f"Wrote {name}: {len(frame):,} rows -> {path}",
            flush=True,
        )

    return paths


def main() -> None:
    """Build, validate, and write the frozen Raw 26 production package."""
    tables = build_mobility_day_type_app_tables()
    paths = write_mobility_day_type_app_tables(
        tables
    )

    print(
        "\nRaw 26 production artifacts are ready:",
        flush=True,
    )

    for name, path in paths.items():
        print(
            f"  {name}: {path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
