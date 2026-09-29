from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_TABLE_DIR = PROJECT_ROOT / "data" / "processed" / "app_tables"

RELATIONSHIP_DAILY_PATH = APP_TABLE_DIR / "weather_relationship_daily.parquet"
RELATIONSHIP_GEOGRAPHY_PATH = APP_TABLE_DIR / "weather_relationship_geography.parquet"

ALL_TEMPORAL_BUCKETS_LABEL = "All temporal buckets"
APP_OVERALL_BUCKET = "all"

MOBILITY_METRICS = (
    "taxi_trip_count", "taxi_avg_trip_speed", "fhvhv_trip_count",
    "fhvhv_avg_trip_speed", "subway_ridership", "avg_bus_speed",
)
WEATHER_METRICS = (
    "temperature", "precipitation", "visibility", "wind_speed", "wind_gust",
    "relative_humidity", "ceiling_height", "pressure_3hr_change",
)
MOBILITY_LABELS = {
    "taxi_trip_count": "Taxi Trips",
    "taxi_avg_trip_speed": "Taxi Average Speed",
    "fhvhv_trip_count": "FHVHV Trips",
    "fhvhv_avg_trip_speed": "FHVHV Average Speed",
    "subway_ridership": "Subway Ridership",
    "avg_bus_speed": "Bus Average Speed",
}
WEATHER_LABELS = {
    "temperature": "Temperature", "precipitation": "Precipitation",
    "visibility": "Visibility", "wind_speed": "Wind Speed",
    "wind_gust": "Wind Gust", "relative_humidity": "Relative Humidity",
    "ceiling_height": "Ceiling Height", "pressure_3hr_change": "3-Hour Pressure Change",
}
TEMPORAL_BUCKETS = (
    ALL_TEMPORAL_BUCKETS_LABEL,
    "weekday_overnight", "weekday_am_peak", "weekday_midday", "weekday_pm_peak",
    "weekday_evening", "weekend_overnight", "weekend_am_peak", "weekend_midday",
    "weekend_pm_peak", "weekend_evening",
)
TEMPORAL_BUCKET_LABELS = {
    ALL_TEMPORAL_BUCKETS_LABEL: "Overall",
    "weekday_overnight": "Weekday · Overnight",
    "weekday_am_peak": "Weekday · AM peak",
    "weekday_midday": "Weekday · Midday",
    "weekday_pm_peak": "Weekday · PM peak",
    "weekday_evening": "Weekday · Evening",
    "weekend_overnight": "Weekend · Overnight",
    "weekend_am_peak": "Weekend · AM peak",
    "weekend_midday": "Weekend · Midday",
    "weekend_pm_peak": "Weekend · PM peak",
    "weekend_evening": "Weekend · Evening",
}
BOROUGHS = ("Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island")


@lru_cache(maxsize=1)
def _load_relationship_daily() -> pd.DataFrame:
    """Load the compact Raw 09 production surface once per app process."""
    if not RELATIONSHIP_DAILY_PATH.exists():
        raise FileNotFoundError(f"Required Raw 09 app table not found: {RELATIONSHIP_DAILY_PATH}")
    frame = pd.read_parquet(RELATIONSHIP_DAILY_PATH)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if "taxi_zone_id" in frame.columns:
        frame["taxi_zone_id"] = pd.to_numeric(frame["taxi_zone_id"], errors="coerce").astype("Int64")
    # WHY: the app-ready builder stores reader-facing period labels, while the
    # existing Raw 09 chart/statistics contract uses compact internal keys.
    frame["pre_post_cp"] = frame["pre_post_cp"].replace(
        {"Pre-CP": "pre_cp", "Post-CP": "post_cp"}
    )
    return frame


@lru_cache(maxsize=1)
def _load_relationship_geography() -> pd.DataFrame:
    """Load the compact geography lookup once per app process."""
    if not RELATIONSHIP_GEOGRAPHY_PATH.exists():
        raise FileNotFoundError(f"Required Raw 09 geography table not found: {RELATIONSHIP_GEOGRAPHY_PATH}")
    frame = pd.read_parquet(RELATIONSHIP_GEOGRAPHY_PATH)
    if "taxi_zone_id" in frame.columns:
        frame["taxi_zone_id"] = pd.to_numeric(frame["taxi_zone_id"], errors="coerce").astype("Int64")
    return frame


def _validate_metric(metric: str, allowed_metrics: Iterable[str], metric_type: str) -> None:
    if metric not in allowed_metrics:
        raise ValueError(f"Unsupported {metric_type} metric: {metric}")


def _app_bucket(temporal_bucket: str) -> str:
    if temporal_bucket not in TEMPORAL_BUCKETS:
        raise ValueError(f"Unsupported temporal bucket: {temporal_bucket}")
    return APP_OVERALL_BUCKET if temporal_bucket == ALL_TEMPORAL_BUCKETS_LABEL else temporal_bucket


def load_zone_lookup() -> pd.DataFrame:
    """Return only Taxi Zones that have matched relationship data."""
    geography = _load_relationship_geography()
    zones = geography.loc[geography["geography_level"].eq("Taxi Zone")].copy()
    keep = [c for c in ["taxi_zone_id", "zone", "borough"] if c in zones.columns]
    zones = zones[keep].dropna(subset=["taxi_zone_id"]).drop_duplicates("taxi_zone_id")
    zones["taxi_zone_id"] = zones["taxi_zone_id"].astype(int)
    return zones.sort_values(["borough", "zone", "taxi_zone_id"], na_position="last").reset_index(drop=True)


def load_borough_lookup() -> pd.DataFrame:
    """Return Boroughs represented in the compact production surface."""
    geography = _load_relationship_geography()
    borough_rows = geography.loc[geography["geography_level"].eq("Borough")].copy()
    source = borough_rows["borough"] if "borough" in borough_rows.columns else borough_rows["geography_name"]
    available = set(source.dropna().astype(str))
    return pd.DataFrame({"borough": [b for b in BOROUGHS if b in available]})


@lru_cache(maxsize=1)
def load_relationship_date_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    frame = _load_relationship_daily()
    return pd.Timestamp(frame["date"].min()), pd.Timestamp(frame["date"].max())


def build_daily_weather_mobility_panel(
    *, temporal_bucket: str = ALL_TEMPORAL_BUCKETS_LABEL,
    taxi_zone_id: int | None = None, borough: str | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Filter the precomputed daily surface to one geography and time scope."""
    if taxi_zone_id is not None and borough is not None:
        raise ValueError("Choose either taxi_zone_id or borough, not both.")
    frame = _load_relationship_daily()
    bucket = _app_bucket(temporal_bucket)

    if taxi_zone_id is not None:
        mask = frame["geography_level"].eq("Taxi Zone") & frame["taxi_zone_id"].eq(int(taxi_zone_id))
        level = "Taxi Zone"
    elif borough is not None:
        if borough not in BOROUGHS:
            raise ValueError(f"Unsupported borough: {borough}")
        borough_values = frame["borough"] if "borough" in frame.columns else frame["geography_name"]
        mask = frame["geography_level"].eq("Borough") & borough_values.eq(borough)
        level = "Borough"
    else:
        mask = frame["geography_level"].eq("Citywide")
        level = "Citywide"

    mask &= frame["temporal_bucket"].eq(bucket)
    if start_date is not None:
        mask &= frame["date"].ge(pd.Timestamp(start_date).normalize())
    if end_date is not None:
        mask &= frame["date"].le(pd.Timestamp(end_date).normalize())

    result = frame.loc[mask].copy()
    result["geography_level"] = level
    # Preserve the old public contract: callers see "Overall", not the storage key "all".
    result["temporal_bucket"] = "Overall" if temporal_bucket == ALL_TEMPORAL_BUCKETS_LABEL else temporal_bucket
    return result.sort_values("date").reset_index(drop=True)


def build_weather_relationship_pair_data(
    *, mobility_metric: str, weather_metric: str,
    temporal_bucket: str = ALL_TEMPORAL_BUCKETS_LABEL,
    taxi_zone_id: int | None = None, borough: str | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    _validate_metric(mobility_metric, MOBILITY_METRICS, "mobility")
    _validate_metric(weather_metric, WEATHER_METRICS, "weather")
    panel = build_daily_weather_mobility_panel(
        temporal_bucket=temporal_bucket, taxi_zone_id=taxi_zone_id, borough=borough,
        start_date=start_date, end_date=end_date,
    )
    if panel.empty:
        return pd.DataFrame(columns=[
            "date", "pre_post_cp", "geography_level", "taxi_zone_id", "borough",
            "temporal_bucket", "mobility_value", "weather_value", "mobility_metric",
            "weather_metric", "mobility_label", "weather_label", "pair_label",
        ])
    keep = [c for c in [
        "date", "pre_post_cp", "geography_level", "geography_id", "geography_name",
        "taxi_zone_id", "zone", "borough", "temporal_bucket", mobility_metric, weather_metric,
    ] if c in panel.columns]
    result = panel[keep].copy().rename(columns={mobility_metric: "mobility_value", weather_metric: "weather_value"})
    result["mobility_metric"] = mobility_metric
    result["weather_metric"] = weather_metric
    result["mobility_label"] = MOBILITY_LABELS[mobility_metric]
    result["weather_label"] = WEATHER_LABELS[weather_metric]
    result["pair_label"] = result["weather_label"] + " vs " + result["mobility_label"]
    return result.sort_values("date").reset_index(drop=True)


def calculate_relationship_statistics(pair_data: pd.DataFrame, *, minimum_observations: int = 30) -> pd.DataFrame:
    required = {"date", "pre_post_cp", "mobility_value", "weather_value", "mobility_metric", "weather_metric", "mobility_label", "weather_label", "pair_label"}
    missing = sorted(required.difference(pair_data.columns))
    if missing:
        raise ValueError("Relationship data is missing columns: " + ", ".join(missing))

    rows = []
    periods = [
        ("all", pair_data),
        ("pre_cp", pair_data[pair_data["pre_post_cp"].eq("pre_cp")]),
        ("post_cp", pair_data[pair_data["pre_post_cp"].eq("post_cp")]),
    ]
    for period, period_data in periods:
        valid = period_data[["date", "mobility_value", "weather_value"]].replace([np.inf, -np.inf], np.nan).dropna()
        n = int(len(valid))
        supported = n >= minimum_observations and valid["mobility_value"].nunique() > 1 and valid["weather_value"].nunique() > 1
        if supported:
            pearson = float(valid["weather_value"].corr(valid["mobility_value"], method="pearson"))
            spearman = float(valid["weather_value"].corr(valid["mobility_value"], method="spearman"))
            slope = float(np.polyfit(valid["weather_value"], valid["mobility_value"], deg=1)[0])
        else:
            pearson = spearman = slope = np.nan
        rows.append({
            "period": period, "observation_count": n, "supported": supported,
            "pearson_correlation": pearson, "spearman_correlation": spearman,
            "linear_slope": slope,
            "minimum_date": valid["date"].min() if not valid.empty else pd.NaT,
            "maximum_date": valid["date"].max() if not valid.empty else pd.NaT,
        })
    result = pd.DataFrame(rows)
    for column in ["geography_level", "taxi_zone_id", "borough", "temporal_bucket", "mobility_metric", "weather_metric", "mobility_label", "weather_label", "pair_label"]:
        if column in pair_data.columns and not pair_data.empty:
            result[column] = pair_data.iloc[0][column]
    return result


def build_borough_relationship_summary(
    *, mobility_metric: str, weather_metric: str, temporal_bucket: str,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    minimum_observations: int = 30,
) -> pd.DataFrame:
    """Compare one weather–mobility relationship across all five Boroughs."""
    rows = []
    for borough in BOROUGHS:
        pair = build_weather_relationship_pair_data(
            mobility_metric=mobility_metric, weather_metric=weather_metric,
            temporal_bucket=temporal_bucket, borough=borough,
            start_date=start_date, end_date=end_date,
        )
        stats = calculate_relationship_statistics(pair, minimum_observations=minimum_observations)
        all_row = stats.loc[stats["period"].eq("all")]
        if all_row.empty:
            continue
        row = all_row.iloc[0]
        rows.append({
            "borough": borough, "observation_count": int(row["observation_count"]),
            "supported": bool(row["supported"]),
            "spearman_correlation": row["spearman_correlation"],
            "pearson_correlation": row["pearson_correlation"],
        })
    return pd.DataFrame(rows)


def scan_weather_relationships(
    *, temporal_bucket: str = ALL_TEMPORAL_BUCKETS_LABEL,
    taxi_zone_id: int | None = None, borough: str | None = None,
    minimum_observations: int = 30,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    panel = build_daily_weather_mobility_panel(
        temporal_bucket=temporal_bucket, taxi_zone_id=taxi_zone_id, borough=borough,
        start_date=start_date, end_date=end_date,
    )
    if panel.empty:
        return pd.DataFrame()
    rows = []
    for mobility_metric in MOBILITY_METRICS:
        for weather_metric in WEATHER_METRICS:
            pair = build_weather_relationship_pair_data(
                mobility_metric=mobility_metric, weather_metric=weather_metric,
                temporal_bucket=temporal_bucket, taxi_zone_id=taxi_zone_id, borough=borough,
                start_date=start_date, end_date=end_date,
            )
            rows.append(calculate_relationship_statistics(pair, minimum_observations=minimum_observations))
    result = pd.concat(rows, ignore_index=True)
    index_cols = [c for c in ["geography_level", "taxi_zone_id", "borough", "temporal_bucket", "mobility_metric", "weather_metric", "mobility_label", "weather_label", "pair_label"] if c in result.columns]
    wide = result.pivot_table(index=index_cols, columns="period", values=["observation_count", "pearson_correlation", "spearman_correlation", "supported"], aggfunc="first", dropna=False)
    wide.columns = [f"{measure}_{period}" for measure, period in wide.columns]
    wide = wide.reset_index().rename_axis(None, axis=1)
    for period in ("all", "pre_cp", "post_cp"):
        wide[f"absolute_spearman_{period}"] = wide[f"spearman_correlation_{period}"].abs()
    wide["spearman_post_minus_pre"] = wide["spearman_correlation_post_cp"] - wide["spearman_correlation_pre_cp"]
    wide["absolute_spearman_shift"] = wide["spearman_post_minus_pre"].abs()
    return wide.sort_values(["absolute_spearman_all", "absolute_spearman_shift"], ascending=[False, False]).reset_index(drop=True)


def scan_all_temporal_buckets(*, taxi_zone_id: int | None = None, borough: str | None = None, minimum_observations: int = 30) -> pd.DataFrame:
    frames = []
    for bucket in TEMPORAL_BUCKETS:
        frame = scan_weather_relationships(temporal_bucket=bucket, taxi_zone_id=taxi_zone_id, borough=borough, minimum_observations=minimum_observations)
        if not frame.empty:
            frame["temporal_bucket_label"] = TEMPORAL_BUCKET_LABELS[bucket]
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def validate_weather_relationship_panel(panel: pd.DataFrame) -> dict[str, object]:
    expected = {"date", "pre_post_cp", *MOBILITY_METRICS, *WEATHER_METRICS}
    missing = sorted(expected.difference(panel.columns))
    duplicate_dates = int(panel.duplicated(subset=["date"]).sum()) if "date" in panel.columns else 0
    return {
        "row_count": int(len(panel)),
        "minimum_date": panel["date"].min() if not panel.empty and "date" in panel.columns else pd.NaT,
        "maximum_date": panel["date"].max() if not panel.empty and "date" in panel.columns else pd.NaT,
        "duplicate_dates": duplicate_dates, "missing_columns": missing,
        "is_valid": not panel.empty and not missing and duplicate_dates == 0,
    }
