"""Independently check a published M1 artifact, its sources, and Parquet samples."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "stage1" / "config" / "ingestion_pilot.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_source_paths() -> set[Path]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    return {(ROOT / item["path"]).resolve() for item in config["sources"]}


def _check_clean_sample(file: pq.ParquetFile, year: int, month: int) -> int:
    columns = (
        "channel_id",
        "timestamp",
        "row_id",
        "source",
        "source_row",
        "sensor_type",
        "join_status",
    )
    if not set(columns).issubset(file.schema_arrow.names):
        raise ValueError("clean Parquet is missing required columns")
    if not file.metadata.num_row_groups:
        raise ValueError("clean Parquet has no row groups")
    sampled = 0
    previous_key = None
    for index in sorted({0, file.metadata.num_row_groups - 1}):
        batch = file.read_row_group(index, columns=list(columns))
        for row in batch.to_pylist():
            timestamp = row["timestamp"]
            if timestamp is None or (timestamp.year, timestamp.month) != (year, month):
                raise ValueError("clean row is in the wrong year/month partition")
            if not row["channel_id"] or not row["source"] or row["source_row"] < 1:
                raise ValueError("clean row lacks provenance")
            key = (row["channel_id"], timestamp, row["row_id"])
            if previous_key is not None and key < previous_key:
                raise ValueError("sampled clean rows are not sorted")
            previous_key = key
            sampled += 1
    return sampled


def _deep_aggregates(directory: Path) -> tuple[dict, dict, dict]:
    """Recompute type, join and calendar counts from actual clean rows."""
    glob = (directory / "clean" / "**" / "*.parquet").as_posix()
    connection = duckdb.connect()
    try:
        connection.execute("SET memory_limit='512MB'")
        connection.execute("SET threads=2")
        rows = connection.execute(
            """SELECT coalesce(sensor_type, 'unknown'), join_status,
                      year(timestamp), month(timestamp), count(*)
               FROM read_parquet(?, hive_partitioning=false)
               GROUP BY ALL""",
            [glob],
        ).fetchall()
    finally:
        connection.close()
    types, joins, partitions = {}, {}, {}
    for sensor_type, join_status, year, month, count in rows:
        types[sensor_type] = types.get(sensor_type, 0) + count
        joins[join_status] = joins.get(join_status, 0) + count
        partitions[(year, month)] = partitions.get((year, month), 0) + count
    return types, joins, partitions


def verify_artifact(
    directory: Path,
    *,
    require_full: bool = True,
    expected_source_paths: set[Path] | None = None,
    deep: bool = False,
) -> dict:
    directory = directory.resolve()
    if directory.name.endswith(".inprogress"):
        raise ValueError("unpublished .inprogress directory is not a completed artifact")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((directory / "data_quality.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("manifest is not complete")
    if require_full and manifest["scope"] != "full_supplied_sources":
        raise ValueError("artifact is a bounded probe, not full supplied history")
    if report["scope"] != manifest["scope"] or report["input_rows"] != manifest["input_rows"]:
        raise ValueError("manifest and quality report disagree")
    sources = manifest["sources"]
    if (
        sources != report["sources"]
        or sum(item["rows_read"] for item in sources) != manifest["input_rows"]
    ):
        raise ValueError("source row counts disagree")
    if require_full and any(item["max_rows"] is not None for item in sources):
        raise ValueError("at least one source was truncated")
    if require_full:
        expected = (
            expected_source_paths if expected_source_paths is not None else _expected_source_paths()
        )
        actual = [Path(item["path"]).resolve() for item in sources]
        if len(actual) != len(set(actual)) or set(actual) != {path.resolve() for path in expected}:
            raise ValueError("full artifact does not contain the expected source inventory")
        for item, path in zip(sources, actual):
            if not path.is_file() or path.stat().st_size != item["bytes"]:
                raise ValueError(f"source is missing or changed: {path}")
            if _sha256(path) != item["sha256"]:
                raise ValueError(f"source SHA-256 differs: {path}")
    dispositions = report["dispositions"]
    accepted = dispositions.get("accepted", 0)
    if sum(dispositions.values()) != manifest["input_rows"]:
        raise ValueError("disposition counts do not conserve input rows")
    for name, checks in (
        ("manifest", manifest["sanity_checks"]),
        ("quality report", report["sanity_checks"]),
    ):
        if not checks or not all(checks.values()):
            raise ValueError(f"{name} contains failed sanity checks")

    partition_rows = {}
    parquet_bytes = 0
    sampled_rows = 0
    for path in sorted((directory / "clean").rglob("*.parquet")):
        parts = path.relative_to(directory / "clean").parts
        if len(parts) != 3 or not parts[0].startswith("year=") or not parts[1].startswith("month="):
            raise ValueError(f"unexpected clean partition path: {path}")
        year = int(parts[0].split("=", 1)[1])
        month = int(parts[1].split("=", 1)[1])
        file = pq.ParquetFile(path)
        partition_rows[(year, month)] = (
            partition_rows.get((year, month), 0) + file.metadata.num_rows
        )
        sampled_rows += _check_clean_sample(file, year, month)
        parquet_bytes += path.stat().st_size
    if len(report["by_partition"]) != len(
        {(item["year"], item["month"]) for item in report["by_partition"]}
    ):
        raise ValueError("quality report has duplicate partitions")
    reported_partitions = {
        (item["year"], item["month"]): item["rows"] for item in report["by_partition"]
    }
    if partition_rows != reported_partitions or sum(partition_rows.values()) != accepted:
        raise ValueError("physical partition row counts disagree with report")
    excluded_path = directory / "excluded_rows.parquet"
    excluded = pq.ParquetFile(excluded_path).metadata.num_rows
    if excluded != manifest["input_rows"] - accepted:
        raise ValueError("excluded Parquet count disagrees with dispositions")
    statistics_path = directory / "sensor_statistics.parquet"
    parquet_bytes += excluded_path.stat().st_size + statistics_path.stat().st_size
    if parquet_bytes != report["parquet_bytes"]:
        raise ValueError("physical Parquet bytes disagree with report")
    if sum(item["rows"] for item in report["by_type"]) != accepted:
        raise ValueError("family counts disagree with accepted rows")
    if sum(item["rows"] for item in report["join_status"]) != accepted:
        raise ValueError("join counts disagree with accepted rows")
    if deep:
        types, joins, calendar = _deep_aggregates(directory)
        if types != {item["sensor_type"]: item["rows"] for item in report["by_type"]}:
            raise ValueError("real clean type counts disagree with report")
        if joins != {item["join_status"]: item["rows"] for item in report["join_status"]}:
            raise ValueError("real clean join counts disagree with report")
        if calendar != reported_partitions:
            raise ValueError("real clean calendar counts disagree with partitions")
    return {
        "status": "verified_deep" if deep else "verified_sampled",
        "scope": manifest["scope"],
        "input_rows": manifest["input_rows"],
        "accepted_rows": accepted,
        "excluded_rows": excluded,
        "sources": len(sources),
        "partitions": len(partition_rows),
        "sampled_clean_rows": sampled_rows,
        "parquet_bytes": parquet_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--allow-bounded", action="store_true")
    parser.add_argument("--deep", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify_artifact(args.directory, require_full=not args.allow_bounded, deep=args.deep),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
