from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import duckdb


DEFAULT_SOURCE = Path(
    "data/processed/3.3.1.final_tables/temporal_residual_mobility_panel.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/processed/app_tables/stress_anomaly_metric_history"
)
DEFAULT_PILOT_METRIC = "avg_bus_speed"
VALUE_SCALE_FACTOR = 1_000
INT32_MAX = 2_147_483_647


def _sql_literal(value: str | Path) -> str:
    """Return a safely quoted DuckDB string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def _metric_filename(metric: str) -> str:
    """Convert a metric name into a stable Parquet filename."""
    safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", metric).strip("_").lower()
    return f"{safe_name}.parquet"


def _available_metrics(connection: duckdb.DuckDBPyConnection, source: Path) -> list[str]:
    rows = connection.execute(
        f"""
        SELECT DISTINCT metric
        FROM read_parquet({_sql_literal(source)})
        WHERE metric IS NOT NULL
        ORDER BY metric
        """
    ).fetchall()
    return [str(row[0]) for row in rows]


def _source_profile(
    connection: duckdb.DuckDBPyConnection,
    source: Path,
    metric: str,
) -> dict[str, object]:
    """Profile one source metric before writing its compact history file."""
    metric_sql = _sql_literal(metric)
    source_sql = _sql_literal(source)
    row = connection.execute(
        f"""
        WITH scoped AS (
            SELECT *
            FROM read_parquet({source_sql})
            WHERE metric = {metric_sql}
        ),
        checked AS (
            SELECT
                *,
                ROUND(observed_value * {VALUE_SCALE_FACTOR})
                    / {VALUE_SCALE_FACTOR} AS observed_quantized,
                ROUND(expected_value * {VALUE_SCALE_FACTOR})
                    / {VALUE_SCALE_FACTOR} AS expected_quantized
            FROM scoped
        ),
        errors AS (
            SELECT
                *,
                ABS(observed_value - observed_quantized) AS observed_error,
                ABS(expected_value - expected_quantized) AS expected_error,
                CASE
                    WHEN residual_scale IS NOT NULL
                         AND residual_scale > 0
                         AND residual_zscore IS NOT NULL
                    THEN ABS(
                        (
                            (
                                observed_quantized - expected_quantized
                                - residual_center
                            ) / residual_scale
                        ) - residual_zscore
                    )
                    ELSE NULL
                END AS reconstructed_z_error
            FROM checked
        )
        SELECT
            COUNT(*) AS source_rows,
            COUNT(DISTINCT (taxi_zone_id, date, temporal_bucket, metric))
                AS distinct_keys,
            COUNT(*) - COUNT(DISTINCT (
                taxi_zone_id, date, temporal_bucket, metric
            )) AS duplicate_keys,
            SUM(
                CASE
                    WHEN ABS(observed_value * {VALUE_SCALE_FACTOR}) > {INT32_MAX}
                      OR ABS(expected_value * {VALUE_SCALE_FACTOR}) > {INT32_MAX}
                    THEN 1 ELSE 0
                END
            ) AS int32_overflow_rows,
            MAX(observed_error) AS max_observed_quantization_error,
            MAX(expected_error) AS max_expected_quantization_error,
            QUANTILE_CONT(reconstructed_z_error, 0.50) AS median_z_error,
            QUANTILE_CONT(reconstructed_z_error, 0.99) AS p99_z_error,
            MAX(reconstructed_z_error) AS max_z_error,
            SUM(CASE WHEN support_status = 'modeled' THEN 1 ELSE 0 END)
                AS modeled_rows,
            SUM(CASE WHEN support_status = 'fallback' THEN 1 ELSE 0 END)
                AS fallback_rows,
            SUM(CASE WHEN support_status = 'unavailable' THEN 1 ELSE 0 END)
                AS unavailable_rows
        FROM errors
        """
    ).fetchone()
    columns = [description[0] for description in connection.description]
    return dict(zip(columns, row))


def _build_metric_file(
    connection: duckdb.DuckDBPyConnection,
    source: Path,
    output_dir: Path,
    metric: str,
) -> dict[str, object]:
    """Build and validate one compact metric-history partition."""
    profile = _source_profile(connection, source, metric)
    if profile["source_rows"] == 0:
        raise ValueError(f"No source rows found for metric {metric!r}.")
    if profile["duplicate_keys"] != 0:
        raise ValueError(
            f"Metric {metric!r} has {profile['duplicate_keys']:,} duplicate keys."
        )
    if profile["int32_overflow_rows"] != 0:
        raise OverflowError(
            f"Metric {metric!r} has {profile['int32_overflow_rows']:,} rows "
            "that cannot be represented as scaled int32 values."
        )

    output_path = output_dir / _metric_filename(metric)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    connection.execute(
        f"""
        COPY (
            SELECT
                CAST(taxi_zone_id AS SMALLINT) AS taxi_zone_id,
                CAST(date AS DATE) AS date,
                temporal_bucket,
                metric,
                CAST(
                    ROUND(observed_value * {VALUE_SCALE_FACTOR}) AS INTEGER
                ) AS observed_milli,
                CAST(
                    ROUND(expected_value * {VALUE_SCALE_FACTOR}) AS INTEGER
                ) AS expected_milli,
                CAST(residual_center AS FLOAT) AS residual_center,
                CAST(residual_scale AS FLOAT) AS residual_scale,
                support_status,
                support_reason,
                eligibility_pathway,
                scale_source,
                scale_method,
                scale_reference_window,
                CAST(scale_reference_count AS INTEGER) AS scale_reference_count
            FROM read_parquet({_sql_literal(source)})
            WHERE metric = {_sql_literal(metric)}
            ORDER BY taxi_zone_id, temporal_bucket, date
        )
        TO {_sql_literal(output_path)} (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 100000
        )
        """
    )

    output_qa = connection.execute(
        f"""
        SELECT
            COUNT(*) AS output_rows,
            COUNT(DISTINCT (taxi_zone_id, date, temporal_bucket, metric))
                AS distinct_output_keys
        FROM read_parquet({_sql_literal(output_path)})
        """
    ).fetchone()
    output_rows, distinct_output_keys = map(int, output_qa)
    if output_rows != profile["source_rows"]:
        raise ValueError(
            f"Metric {metric!r} row mismatch: source={profile['source_rows']:,}, "
            f"output={output_rows:,}."
        )
    if distinct_output_keys != output_rows:
        raise ValueError(
            f"Metric {metric!r} output key is not unique: "
            f"rows={output_rows:,}, keys={distinct_output_keys:,}."
        )

    result = {
        "metric": metric,
        "file": output_path.name,
        "rows": output_rows,
        "file_size_mb": round(output_path.stat().st_size / 1024**2, 3),
        "value_scale_factor": VALUE_SCALE_FACTOR,
        **profile,
    }
    return result


def _print_result(result: dict[str, object]) -> None:
    print()
    print(f"METRIC: {result['metric']}")
    print(f"file={result['file']}")
    print(f"rows={result['rows']:,}")
    print(f"file_size_mb={result['file_size_mb']:,.3f}")
    print(f"duplicate_keys={result['duplicate_keys']:,}")
    print(f"int32_overflow_rows={result['int32_overflow_rows']:,}")
    print(
        "max_observed_quantization_error="
        f"{result['max_observed_quantization_error']:.9f}"
    )
    print(
        "max_expected_quantization_error="
        f"{result['max_expected_quantization_error']:.9f}"
    )
    print(f"median_reconstructed_z_error={result['median_z_error']:.9f}")
    print(f"p99_reconstructed_z_error={result['p99_z_error']:.9f}")
    print(f"max_reconstructed_z_error={result['max_z_error']:.9f}")
    print(f"modeled_rows={result['modeled_rows']:,}")
    print(f"fallback_rows={result['fallback_rows']:,}")
    print(f"unavailable_rows={result['unavailable_rows']:,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build compact, metric-partitioned temporal histories for the Raw 15 "
            "Stress Anomaly Event Profiler."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--metric",
        action="append",
        help=(
            "Metric to build. Repeat for multiple metrics. Defaults to the "
            f"pilot metric {DEFAULT_PILOT_METRIC!r}."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build every metric found in the source panel.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source panel not found: {source}")

    connection = duckdb.connect()
    available_metrics = _available_metrics(connection, source)
    requested_metrics = (
        available_metrics
        if args.all
        else list(dict.fromkeys(args.metric or [DEFAULT_PILOT_METRIC]))
    )
    unknown_metrics = sorted(set(requested_metrics).difference(available_metrics))
    if unknown_metrics:
        raise ValueError(
            f"Unknown metrics: {unknown_metrics}. Available metrics: "
            f"{available_metrics}."
        )

    print("RAW 15 — COMPACT METRIC-HISTORY BUILD")
    print(f"source={source}")
    print(f"output_dir={output_dir}")
    print(f"value_scale_factor={VALUE_SCALE_FACTOR:,}")
    print(f"metrics={requested_metrics}")

    results = []
    for metric in requested_metrics:
        result = _build_metric_file(
            connection=connection,
            source=source,
            output_dir=output_dir,
            metric=metric,
        )
        results.append(result)
        _print_result(result)

    manifest = {
        "source": str(source),
        "value_scale_factor": VALUE_SCALE_FACTOR,
        "storage_contract": {
            "observed_value": "observed_milli / 1000",
            "expected_value": "expected_milli / 1000",
            "residual_value": "observed_value - expected_value",
            "residual_zscore": (
                "(residual_value - residual_center) / residual_scale"
            ),
        },
        "metrics": results,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print()
    print("BUILD COMPLETE")
    print(f"metric_files={len(results):,}")
    print(
        "total_size_mb="
        f"{sum(float(result['file_size_mb']) for result in results):,.3f}"
    )
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
