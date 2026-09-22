"""Milestone 1 orchestration: streaming normalization and disk-backed relational work."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from stage1.ingestion.dictionaries import load_dictionaries
from stage1.ingestion.normalize import normalize_row
from stage1.ingestion.reporting import build_quality_report
from stage1.ingestion.schemas import CHANNEL_SCHEMA, OBJECT_SCHEMA, STAGING_SCHEMA
from stage1.ingestion.sources import iter_source_rows
from stage1.ingestion.sql import classify, export_tables


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parquet_sanity(output, expected_rows):
    """Read physical files in batches to verify exact counts and per-partition sort order."""
    total = 0
    ordered = True
    for path in sorted((output / "clean").rglob("*.parquet")):
        previous = None
        file = pq.ParquetFile(path)
        total += file.metadata.num_rows
        for batch in file.iter_batches(
            batch_size=25000, columns=["channel_id", "timestamp", "row_id"]
        ):
            for row in batch.to_pylist():
                key = (row["channel_id"], row["timestamp"], row["row_id"])
                if previous is not None and key < previous:
                    ordered = False
                previous = key
    return {"parquet_row_count_matches": total == expected_rows, "parquet_files_sorted": ordered}


def run_ingestion(config: dict, progress=None) -> dict:
    destination = Path(config["output"]).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    output = destination.with_name(destination.name + ".inprogress")
    sources = config["sources"]
    paths = [Path(item["path"]).resolve() for item in sources]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("Provide at least one source; repeated input paths are not allowed")
    batch_size = config.get("batch_size", 25000)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    for item, path in zip(sources, paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        limit = item.get("max_rows")
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        ):
            raise ValueError("source max_rows must be null or a positive integer")
    channels_path = Path(config["channels"])
    objects_path = Path(config["objects"])
    channel_rows, object_rows, dictionary_audit = load_dictionaries(channels_path, objects_path)
    output.mkdir(parents=True, exist_ok=False)  # Never mix a failed/older run into a new one.
    database = output / "work.duckdb"
    start = time.perf_counter()
    manifest = {
        "schema_version": "ingestion-v1",
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "bounded_probe"
        if any(s.get("max_rows") is not None for s in sources)
        else "full_supplied_sources",
        "configuration": config,
        "duckdb_version": duckdb.__version__,
        "pyarrow_version": pa.__version__,
        "memory_limit": config.get("memory_limit", "512MB"),
        "threads": 1,
        "sources": [],
        "input_rows": 0,
        "dictionaries": [
            {"path": str(p.resolve()), "sha256": sha256(p)} for p in (channels_path, objects_path)
        ],
        "limitations": [
            "No anomaly detection, labels or forecasting in milestone 1",
            "Absent archive years are not reconstructed",
            "Object links are never inferred from names/tags",
            "Rows are sorted within month partitions; glob concatenation is not a globally sorted stream",
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    con = duckdb.connect(str(database))
    try:
        con.execute("SET memory_limit = ?", [manifest["memory_limit"]])
        con.execute("SET temp_directory = ?", [str(output / "spill")])
        con.execute("SET threads = 1")
        for name, schema, items in (
            ("raw", STAGING_SCHEMA, []),
            ("channels", CHANNEL_SCHEMA, channel_rows),
            ("objects", OBJECT_SCHEMA, object_rows),
        ):
            con.register("incoming", pa.Table.from_pylist(items, schema=schema))
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM incoming")
            con.unregister("incoming")
        for path, spec in zip(paths, sources):
            before = path.stat()
            source_record = {
                "path": str(path),
                "bytes": before.st_size,
                "sha256": sha256(path),
                "max_rows": spec.get("max_rows"),
                "rows_read": 0,
            }
            pending = []
            for row in iter_source_rows(path, spec.get("max_rows")):
                manifest["input_rows"] += 1
                source_record["rows_read"] += 1
                pending.append(normalize_row(row, manifest["input_rows"]))
                if len(pending) >= batch_size:
                    _insert(con, pending)
                    pending.clear()
            if pending:
                _insert(con, pending)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or sha256(
                path
            ) != source_record["sha256"]:
                raise RuntimeError(f"Source changed during ingestion: {path}")
            manifest["sources"].append(source_record)
            if progress:
                progress(source_record)
        classify(con, object_mapping_available=dictionary_audit["object_mapping_available"])
        export_tables(con, output)
        accepted = con.execute("SELECT count(*) FROM clean").fetchone()[0]
        checks = parquet_sanity(output, accepted)
        if not all(checks.values()):
            raise ValueError(f"Parquet sanity checks failed: {checks}")
        report = build_quality_report(con, output, manifest, dictionary_audit)
        report["sanity_checks"].update(checks)
        report["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        report["disk_database_bytes"] = database.stat().st_size
        (output / "data_quality.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        manifest.update(
            status="complete",
            elapsed_seconds=report["elapsed_seconds"],
            sanity_checks=report["sanity_checks"],
        )
        return report
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        con.close()
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if manifest["status"] == "complete" and not config.get("keep_database", False):
            database.unlink()  # Exact database created by this run; no recursive cleanup.
        if manifest["status"] == "complete":
            if destination.exists():
                raise FileExistsError(destination)
            output.rename(destination)  # Publish the checked directory in one filesystem operation.


def _insert(connection, rows):
    connection.register("incoming", pa.Table.from_pylist(rows, schema=STAGING_SCHEMA))
    connection.execute("INSERT INTO raw SELECT * FROM incoming")
    connection.unregister("incoming")
