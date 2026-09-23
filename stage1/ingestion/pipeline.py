"""Milestone 1 orchestration: streaming normalization and disk-backed relational work."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from stage1.ingestion.checkpoints import atomic_write_json
from stage1.ingestion.dictionaries import load_dictionaries
from stage1.ingestion.normalize import normalize_row
from stage1.ingestion.reporting import build_quality_report, write_quality_outputs
from stage1.ingestion.schemas import CHANNEL_SCHEMA, OBJECT_SCHEMA, STAGING_SCHEMA
from stage1.ingestion.sources import iter_source_rows
from stage1.ingestion.sql import (
    AUXILIARY_EXPORTS,
    classify,
    export_auxiliary_table,
    export_tables,
)
from stage1.ingestion.sql_partitioned import classify_partitioned


CLASSIFICATION_STRATEGIES = frozenset({"global_window_v1", "partitioned_hash_v1"})
LEGACY_RECOVERY_STRATEGY_PLACEHOLDER = "recovered_existing_derived_tables"


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
        previous_tail = None
        file = pq.ParquetFile(path)
        total += file.metadata.num_rows
        for batch in file.iter_batches(
            batch_size=25000, columns=["channel_id", "timestamp", "row_id"]
        ):
            columns = tuple(batch.column(index) for index in range(batch.num_columns))
            if previous_tail is not None and _has_lexicographic_decrease(
                tuple(column.slice(0, 1) for column in columns), previous_tail
            ):
                ordered = False
            if batch.num_rows > 1 and _has_lexicographic_decrease(
                tuple(column.slice(1) for column in columns),
                tuple(column.slice(0, batch.num_rows - 1) for column in columns),
            ):
                ordered = False
            previous_tail = tuple(column.slice(batch.num_rows - 1, 1) for column in columns)
    return {"parquet_row_count_matches": total == expected_rows, "parquet_files_sorted": ordered}


def _has_lexicographic_decrease(current_columns, previous_columns):
    """Return whether any current key is lexicographically below its predecessor."""
    if any(column.null_count for column in (*current_columns, *previous_columns)):
        # Clean sort keys are non-null by contract. Retain the former Python tuple
        # semantics for malformed external fixtures instead of inventing a null order.
        current_rows = zip(*(column.to_pylist() for column in current_columns))
        previous_rows = zip(*(column.to_pylist() for column in previous_columns))
        return any(current < previous for current, previous in zip(current_rows, previous_rows))

    decrease = pc.less(current_columns[0], previous_columns[0])
    prefix_equal = pc.equal(current_columns[0], previous_columns[0])
    for current, previous in zip(current_columns[1:], previous_columns[1:]):
        decrease = pc.or_kleene(decrease, pc.and_kleene(prefix_equal, pc.less(current, previous)))
        prefix_equal = pc.and_kleene(prefix_equal, pc.equal(current, previous))
    return bool(pc.any(decrease).as_py())


def run_ingestion(
    config: dict,
    progress=None,
    *,
    resume: bool = False,
    resume_source_sha256: str | None = None,
    partitioned_classification: bool = False,
) -> dict:
    destination = Path(config["output"]).resolve()
    recovered_report = _recover_pending_publication(destination, config)
    if recovered_report is not None:
        return recovered_report
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
    database = output / "work.duckdb"
    start = time.perf_counter()
    fresh_manifest = {
        "schema_version": "ingestion-v1",
        "status": "running",
        "phase": "loading",
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
    if resume:
        if not output.is_dir() or not database.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("No existing .inprogress database and manifest to resume")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["status"] not in {"failed", "running"}:
            raise ValueError("Only failed or interrupted runs can be resumed")
        old_config = {
            key: value for key, value in manifest["configuration"].items() if key != "memory_limit"
        }
        new_config = {key: value for key, value in config.items() if key != "memory_limit"}
        if old_config != new_config or manifest["scope"] != fresh_manifest["scope"]:
            raise ValueError("Resume configuration differs from the original run")
        if manifest["dictionaries"] != fresh_manifest["dictionaries"]:
            raise ValueError("Dictionaries changed since the original run")
    else:
        output.mkdir(parents=True, exist_ok=False)  # Never mix runs into one output.
        manifest = fresh_manifest
        atomic_write_json(manifest_path, manifest)
    con = duckdb.connect(str(database))
    if resume:
        try:
            committed_by_source = _validate_resume(
                con, manifest, paths, resume_source_sha256=resume_source_sha256
            )
        except Exception:
            con.close()
            raise
    else:
        committed_by_source = {}
    try:
        con.execute("SET memory_limit = ?", [config.get("memory_limit", "512MB")])
        con.execute("SET temp_directory = ?", [str(output / "spill")])
        con.execute("SET threads = 1")
        con.execute("SET preserve_insertion_order = false")
        if resume:
            manifest.setdefault("resume_history", []).append(
                {
                    "at_utc": datetime.now(timezone.utc).isoformat(),
                    "committed_rows": sum(committed_by_source.values()),
                    "previous_memory_limit": manifest["memory_limit"],
                }
            )
            manifest.update(
                status="running",
                phase="loading",
                configuration=config,
                memory_limit=config.get("memory_limit", "512MB"),
                input_rows=sum(committed_by_source.values()),
            )
            manifest.pop("error", None)
            manifest.pop("failed_phase", None)
            atomic_write_json(manifest_path, manifest)
        else:
            for name, schema, items in (
                ("raw", STAGING_SCHEMA, []),
                ("channels", CHANNEL_SCHEMA, channel_rows),
                ("objects", OBJECT_SCHEMA, object_rows),
            ):
                con.register("incoming", pa.Table.from_pylist(items, schema=schema))
                con.execute(f"CREATE TABLE {name} AS SELECT * FROM incoming")
                con.unregister("incoming")
        completed_sources = len(manifest["sources"])
        for index, (path, spec) in enumerate(zip(paths, sources)):
            if index < completed_sources:
                continue
            before = path.stat()
            committed = committed_by_source.get(str(path), 0)
            source_record = {
                "path": str(path),
                "bytes": before.st_size,
                "sha256": sha256(path),
                "max_rows": spec.get("max_rows"),
                "rows_read": committed,
            }
            manifest["active_source"] = source_record
            atomic_write_json(manifest_path, manifest)
            pending = []
            skipped = 0
            for row in iter_source_rows(path, spec.get("max_rows")):
                if row["__source_row__"] <= committed + 1:
                    skipped += 1
                    continue
                manifest["input_rows"] += 1
                source_record["rows_read"] += 1
                pending.append(normalize_row(row, manifest["input_rows"]))
                if len(pending) >= batch_size:
                    _insert(con, pending)
                    pending.clear()
            if skipped != committed:
                raise ValueError(f"Resume prefix changed or shortened: {path}")
            if pending:
                _insert(con, pending)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or sha256(
                path
            ) != source_record["sha256"]:
                raise RuntimeError(f"Source changed during ingestion: {path}")
            manifest["sources"].append(source_record)
            manifest.pop("active_source", None)
            atomic_write_json(manifest_path, manifest)
            if progress:
                progress(source_record)
        classification_strategy = (
            "partitioned_hash_v1" if partitioned_classification else "global_window_v1"
        )
        manifest.update(phase="classification", classification_strategy=classification_strategy)
        if partitioned_classification:
            scratch = output.with_name(output.name + ".classify-scratch")
            manifest["classification_scratch"] = str(scratch)
        else:
            manifest.pop("classification_scratch", None)
        atomic_write_json(manifest_path, manifest)

        if partitioned_classification:
            classify_partitioned(
                con, scratch, object_mapping_available=dictionary_audit["object_mapping_available"]
            )
        else:
            classify(con, object_mapping_available=dictionary_audit["object_mapping_available"])
        manifest["phase"] = "export"
        atomic_write_json(manifest_path, manifest)
        export_tables(con, output)
        manifest["phase"] = "verifying"
        atomic_write_json(manifest_path, manifest)
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
            status="running",
            phase="ready_to_publish",
            elapsed_seconds=report["elapsed_seconds"],
            sanity_checks=report["sanity_checks"],
            publication={
                "destination": str(destination),
                "keep_database": bool(config.get("keep_database", False)),
            },
        )
        atomic_write_json(manifest_path, manifest)
    except Exception as error:
        failed_phase = manifest.get("phase")
        manifest.update(
            status="failed",
            phase="failed",
            failed_phase=failed_phase,
            error=f"{type(error).__name__}: {error}",
        )
        atomic_write_json(manifest_path, manifest)
        raise
    finally:
        con.close()

    _publish_ready_output(output, destination, manifest)
    return report


def recover_derived_ingestion(
    config: dict, *, trusted_classification_strategy: str | None = None
) -> dict:
    """Finalize an interrupted run whose derived DuckDB tables already committed.

    This path deliberately never reads event sources through ``iter_source_rows`` and
    never rebuilds ``classified`` or ``clean``.  It accepts only a fully materialized,
    internally consistent database.  Existing clean Parquet is reused only when its
    exact partition inventory, Arrow schema and per-partition row counts match DuckDB.
    A missing auxiliary export, or a zero-byte one, can then be rebuilt atomically;
    any non-empty conflicting artifact is left untouched and causes a hard failure.
    """
    destination = Path(config["output"]).resolve()
    recovered_report = _recover_pending_publication(destination, config)
    if recovered_report is not None:
        return recovered_report
    if destination.exists():
        raise FileExistsError(destination)
    output = destination.with_name(destination.name + ".inprogress")
    database = output / "work.duckdb"
    manifest_path = output / "manifest.json"
    if not output.is_dir() or not database.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("No existing .inprogress derived database and manifest")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"failed", "running"}:
        raise ValueError("Only failed or interrupted runs can be recovered")
    classification_strategy, classification_evidence = _resolve_recovery_classification_strategy(
        manifest, trusted_classification_strategy
    )
    paths = [Path(item["path"]).resolve() for item in config["sources"]]
    channels_path = Path(config["channels"]).resolve()
    objects_path = Path(config["objects"]).resolve()
    channel_rows, object_rows, dictionary_audit = load_dictionaries(channels_path, objects_path)
    _validate_recovery_inputs(
        config, manifest, paths, channels_path=channels_path, objects_path=objects_path
    )

    start = time.perf_counter()
    con = duckdb.connect(str(database))
    recovery_record = None
    try:
        con.execute("SET memory_limit = ?", [config.get("memory_limit", "512MB")])
        con.execute("SET temp_directory = ?", [str(output / "spill")])
        con.execute("SET threads = 1")
        con.execute("SET preserve_insertion_order = false")

        counts = _validate_derived_database(
            con,
            manifest,
            paths,
            expected_channels=len(channel_rows),
            expected_objects=len(object_rows),
        )
        clean_audit = _validate_clean_export(con, output, counts["clean"])
        _validate_report_targets(output)
        cleared_abandoned_export = _clear_abandoned_excluded_export(output, manifest)
        auxiliary_plan = _plan_auxiliary_recovery(con, output)

        recovery_started = datetime.now(timezone.utc).isoformat()
        history = manifest.setdefault("recovery_history", [])
        if history and history[-1].get("status") == "running":
            history[-1].update(status="interrupted", interrupted_utc=recovery_started)
        recovery_record = {
            "started_utc": recovery_started,
            "mode": "derived_tables_v1",
            "validated_table_rows": counts,
            "reused_clean_partitions": clean_audit["partitions"],
            "reused_clean_rows": clean_audit["rows"],
            "auxiliary_exports": auxiliary_plan,
            "cleared_abandoned_zero_byte_excluded_export": cleared_abandoned_export,
            "classification_strategy": classification_strategy,
            "classification_strategy_evidence": classification_evidence,
            "status": "running",
        }
        history.append(recovery_record)
        manifest.update(
            status="running",
            phase="export",
            configuration=config,
            memory_limit=config.get("memory_limit", "512MB"),
            classification_strategy=classification_strategy,
        )
        manifest.pop("error", None)
        manifest.pop("failed_phase", None)
        atomic_write_json(manifest_path, manifest)

        for name, action in auxiliary_plan.items():
            if action == "regenerate":
                _recover_auxiliary_export(con, output, name)

        manifest["phase"] = "verifying"
        atomic_write_json(manifest_path, manifest)
        checks = parquet_sanity(output, counts["clean"])
        if not all(checks.values()):
            raise ValueError(f"Parquet sanity checks failed: {checks}")
        report = build_quality_report(con, output, manifest, dictionary_audit, write_outputs=False)
        report["sanity_checks"].update(checks)
        report["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        report["disk_database_bytes"] = database.stat().st_size
        report["recovery"] = {
            "mode": recovery_record["mode"],
            "reused_clean_partitions": recovery_record["reused_clean_partitions"],
            "auxiliary_exports": auxiliary_plan,
            "classification_strategy": classification_strategy,
        }
        _write_recovery_reports(output, report)

        recovery_record.update(
            status="ready_to_publish", verified_utc=datetime.now(timezone.utc).isoformat()
        )
        manifest.update(
            status="running",
            phase="ready_to_publish",
            elapsed_seconds=report["elapsed_seconds"],
            sanity_checks=report["sanity_checks"],
            publication={
                "destination": str(destination),
                "keep_database": bool(config.get("keep_database", False)),
            },
        )
        atomic_write_json(manifest_path, manifest)
    except Exception as error:
        if recovery_record is not None:
            failed_phase = manifest.get("phase")
            recovery_record.update(
                status="failed",
                failed_utc=datetime.now(timezone.utc).isoformat(),
                error=f"{type(error).__name__}: {error}",
            )
            manifest.update(
                status="failed",
                phase="failed",
                failed_phase=failed_phase,
                error=recovery_record["error"],
            )
            atomic_write_json(manifest_path, manifest)
        raise
    finally:
        con.close()

    _publish_ready_output(output, destination, manifest)
    return report


def _recover_pending_publication(destination, config):
    """Finish only a verified publication checkpoint; never rerun heavy work."""
    output = destination.with_name(destination.name + ".inprogress")
    if destination.exists() and output.exists():
        raise FileExistsError(
            f"Both publication paths exist; refusing an ambiguous recovery: {output}, {destination}"
        )

    if destination.is_dir():
        manifest_path = destination / "manifest.json"
        if not manifest_path.is_file():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "running" and manifest.get("phase") == "publishing":
            _validate_publication_checkpoint(destination, destination, manifest, config)
            report = _load_publication_report(destination)
            _complete_published_output(destination, manifest)
            return report
        return None

    manifest_path = output / "manifest.json"
    if not output.is_dir() or not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "running" or manifest.get("phase") not in {
        "ready_to_publish",
        "publishing",
    }:
        return None
    _validate_publication_checkpoint(output, destination, manifest, config)
    report = _load_publication_report(output)
    _publish_ready_output(output, destination, manifest)
    return report


def _validate_publication_checkpoint(root, destination, manifest, config):
    publication = manifest.get("publication")
    if not isinstance(publication, dict):
        raise ValueError("Publication checkpoint lacks publication metadata")
    if publication.get("destination") != str(destination):
        raise ValueError("Publication checkpoint targets a different destination")

    recorded_config = dict(manifest.get("configuration", {}))
    requested_config = dict(config)
    recorded_output = Path(recorded_config.pop("output", "")).resolve()
    requested_output = Path(requested_config.pop("output", "")).resolve()
    recorded_config.pop("memory_limit", None)
    requested_config.pop("memory_limit", None)
    if (
        recorded_output != destination
        or requested_output != destination
        or recorded_config != requested_config
    ):
        raise ValueError("Publication retry configuration differs from the verified run")

    keep_database = publication.get("keep_database")
    if not isinstance(keep_database, bool):
        raise ValueError("Publication checkpoint has an invalid database policy")
    database_exists = (root / "work.duckdb").is_file()
    phase = manifest.get("phase")
    if keep_database and not database_exists:
        raise ValueError("Publication checkpoint requires the DuckDB database")
    if phase == "ready_to_publish" and not database_exists:
        raise ValueError("Ready publication checkpoint lost its DuckDB database")
    if root == destination and not keep_database and database_exists:
        raise ValueError("Published output unexpectedly retains its DuckDB database")
    sanity_checks = manifest.get("sanity_checks")
    if (
        not isinstance(sanity_checks, dict)
        or not sanity_checks
        or not all(value is True for value in sanity_checks.values())
    ):
        raise ValueError("Publication checkpoint lacks successful sanity checks")


def _load_publication_report(root):
    report_path = root / "data_quality.json"
    if not report_path.is_file():
        raise FileNotFoundError("Verified publication checkpoint lacks data_quality.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    sanity_checks = report.get("sanity_checks")
    if (
        not isinstance(sanity_checks, dict)
        or not sanity_checks
        or not all(value is True for value in sanity_checks.values())
    ):
        raise ValueError("Publication report lacks successful sanity checks")
    return report


def _publish_ready_output(output, destination, manifest):
    """Publish a verified sibling directory with a replay-safe ordering."""
    _validate_publication_checkpoint(output, destination, manifest, manifest["configuration"])
    _load_publication_report(output)
    if manifest.get("phase") == "ready_to_publish":
        manifest["phase"] = "publishing"
        manifest["status"] = "running"
        manifest["publication"]["publishing_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(output / "manifest.json", manifest)
    elif manifest.get("phase") != "publishing":
        raise ValueError(f"Output is not ready to publish: {manifest.get('phase')}")

    database = output / "work.duckdb"
    if not manifest["publication"]["keep_database"] and database.exists():
        database.unlink()  # Exact database owned by this run; retry treats absence as success.
    if destination.exists():
        raise FileExistsError(destination)
    _rename_publication_directory(output, destination)
    _complete_published_output(destination, manifest)


def _rename_publication_directory(output, destination):
    output.rename(destination)  # Sibling rename is the atomic publication boundary.


def _complete_published_output(destination, manifest):
    if manifest.get("phase") != "publishing" or manifest.get("status") != "running":
        raise ValueError("Only a publishing checkpoint can become complete")
    completed_utc = datetime.now(timezone.utc).isoformat()
    manifest.update(status="complete", phase="complete", completed_utc=completed_utc)
    publication = manifest["publication"]
    publication["completed_utc"] = completed_utc
    history = manifest.get("recovery_history", [])
    if history and history[-1].get("status") == "ready_to_publish":
        history[-1].update(status="complete", completed_utc=completed_utc)
    atomic_write_json(destination / "manifest.json", manifest)


def _resolve_recovery_classification_strategy(manifest, trusted_override):
    if trusted_override is not None and trusted_override not in CLASSIFICATION_STRATEGIES:
        raise ValueError(f"Unsupported recovery classification strategy: {trusted_override}")

    checkpoint = manifest.get("classification_strategy")
    if checkpoint == LEGACY_RECOVERY_STRATEGY_PLACEHOLDER:
        if trusted_override is None:
            raise ValueError(
                "Legacy recovery placeholder lacks a classification strategy; "
                "provide an explicit trusted recovery strategy"
            )
        return trusted_override, {
            "source": "explicit_trusted_legacy_placeholder_override",
            "manifest_checkpoint_present": True,
            "checkpoint": checkpoint,
            "override": trusted_override,
        }
    if checkpoint is not None and checkpoint not in CLASSIFICATION_STRATEGIES:
        raise ValueError(f"Unsupported manifest classification strategy: {checkpoint}")
    if checkpoint is None:
        if trusted_override is None:
            raise ValueError(
                "Legacy manifest lacks classification_strategy; provide an explicit trusted "
                "recovery strategy"
            )
        return trusted_override, {
            "source": "explicit_trusted_legacy_override",
            "manifest_checkpoint_present": False,
            "override": trusted_override,
        }
    if trusted_override is not None and trusted_override != checkpoint:
        raise ValueError(
            "Recovery classification strategy override disagrees with manifest checkpoint"
        )
    evidence = {
        "source": "manifest_checkpoint",
        "manifest_checkpoint_present": True,
        "checkpoint": checkpoint,
    }
    if trusted_override is not None:
        evidence["matching_override"] = trusted_override
    return checkpoint, evidence


def _validate_recovery_inputs(config, manifest, paths, *, channels_path, objects_path):
    old_config = {
        key: value for key, value in manifest["configuration"].items() if key != "memory_limit"
    }
    new_config = {key: value for key, value in config.items() if key != "memory_limit"}
    expected_scope = (
        "bounded_probe"
        if any(item.get("max_rows") is not None for item in config["sources"])
        else "full_supplied_sources"
    )
    if old_config != new_config or manifest.get("scope") != expected_scope:
        raise ValueError("Recovery configuration differs from the original run")
    expected_dictionaries = [
        {"path": str(path), "sha256": sha256(path)} for path in (channels_path, objects_path)
    ]
    if manifest.get("dictionaries") != expected_dictionaries:
        raise ValueError("Dictionaries changed since the original run")
    if manifest.get("active_source") is not None:
        raise ValueError("Recovery requires every configured source to be fully ingested")
    records = manifest.get("sources", [])
    if len(records) != len(paths):
        raise ValueError("Recovery requires every configured source to be fully ingested")
    total = 0
    for path, spec, record in zip(paths, config["sources"], records):
        if (
            record.get("path") != str(path)
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != sha256(path)
            or record.get("max_rows") != spec.get("max_rows")
        ):
            raise ValueError(f"Completed source changed since checkpoint: {path}")
        total += record["rows_read"]
    if manifest.get("input_rows") != total:
        raise ValueError("Manifest input row total disagrees with completed sources")


def _validate_derived_database(connection, manifest, paths, *, expected_channels, expected_objects):
    required = {"raw", "channels", "objects", "classified", "conflicts", "clean"}
    tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    if tables != required:
        raise ValueError(f"Unsafe recovery phase: database tables are {sorted(tables)}")
    counts = {
        table: connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in sorted(required)
    }
    input_rows = manifest["input_rows"]
    if counts["channels"] != expected_channels or counts["objects"] != expected_objects:
        raise ValueError("Dictionary tables disagree with the validated dictionary files")
    if counts["raw"] != input_rows or counts["classified"] != input_rows:
        raise ValueError("Raw/classified row counts disagree with the manifest")
    raw_range = connection.execute("SELECT min(row_id), max(row_id) FROM raw").fetchone()
    classified_range = connection.execute(
        "SELECT min(row_id), max(row_id) FROM classified"
    ).fetchone()
    expected_range = (1, input_rows) if input_rows else (None, None)
    if raw_range != expected_range or classified_range != expected_range:
        raise ValueError("Raw/classified row_id range is not contiguous with the manifest")
    groups = connection.execute(
        """SELECT source, count(*), min(source_row), max(source_row),
                  min(row_id), max(row_id)
           FROM raw GROUP BY source ORDER BY min(row_id)"""
    ).fetchall()
    if len(groups) != len(paths):
        raise ValueError("Raw source groups disagree with the completed manifest")
    previous_row_id = 0
    for path, record, group in zip(paths, manifest["sources"], groups):
        source, count, first_source_row, last_source_row, first_row_id, last_row_id = group
        if (
            source != str(path)
            or count != record["rows_read"]
            or first_source_row != 2
            or last_source_row != count + 1
            or first_row_id != previous_row_id + 1
            or last_row_id != previous_row_id + count
        ):
            raise ValueError("Raw source/row_id checkpoint is not contiguous")
        previous_row_id = last_row_id
    dispositions = dict(
        connection.execute(
            "SELECT disposition, count(*) FROM classified GROUP BY disposition"
        ).fetchall()
    )
    if set(dispositions) - {"accepted", "exact_duplicate", "quarantine", "repeated_header"}:
        raise ValueError("Classified contains an unknown disposition")
    if sum(dispositions.values()) != input_rows:
        raise ValueError("Classified disposition totals disagree with the manifest")
    if counts["clean"] != dispositions.get("accepted", 0):
        raise ValueError("Clean row count disagrees with accepted classified rows")
    return counts


def _validate_clean_export(connection, output, expected_rows):
    clean_dir = output / "clean"
    partitions = connection.execute(
        "SELECT year, month, count(*) FROM clean GROUP BY year, month ORDER BY year, month"
    ).fetchall()
    expected = {
        Path(f"year={int(year)}") / f"month={int(month)}" / "data_0.parquet": rows
        for year, month, rows in partitions
    }
    if not clean_dir.exists():
        if expected:
            raise ValueError("Clean export is missing")
        return {"partitions": 0, "rows": 0}
    files = {path.relative_to(clean_dir): path for path in clean_dir.rglob("*") if path.is_file()}
    if set(files) != set(expected):
        raise ValueError("Clean export file inventory disagrees with DuckDB partitions")
    expected_dirs = {relative.parent for relative in expected} | {
        relative.parent.parent for relative in expected
    }
    actual_dirs = {path.relative_to(clean_dir) for path in clean_dir.rglob("*") if path.is_dir()}
    if actual_dirs != expected_dirs:
        raise ValueError("Clean export directory inventory is ambiguous")
    expected_schema = (
        connection.execute("SELECT * EXCLUDE (year, month) FROM clean LIMIT 0")
        .to_arrow_table()
        .schema
    )
    total = 0
    observed_schema = None
    for relative, path in sorted(files.items(), key=lambda item: str(item[0])):
        try:
            parquet = pq.ParquetFile(path)
        except Exception as error:
            raise ValueError(f"Unreadable clean Parquet: {relative}") from error
        if parquet.metadata.num_rows != expected[relative]:
            raise ValueError(f"Clean partition row count disagrees: {relative}")
        schema = parquet.schema_arrow
        if not schema.equals(expected_schema, check_metadata=False):
            raise ValueError(f"Clean partition schema disagrees: {relative}")
        if observed_schema is not None and not schema.equals(observed_schema, check_metadata=False):
            raise ValueError("Clean partition schemas are inconsistent")
        observed_schema = schema
        total += parquet.metadata.num_rows
    if total != expected_rows:
        raise ValueError("Clean Parquet total disagrees with DuckDB clean rows")
    return {"partitions": len(files), "rows": total}


def _clear_abandoned_excluded_export(output, manifest):
    """Remove only an empty interrupted recovery export while holding the DB writer lock.

    The caller opens the DuckDB database for writing before reaching this function;
    another recovery process cannot hold that lock at the same time. Non-empty
    exports and any temporary file without a matching running recovery are ambiguous.
    """
    temporary = output / ".excluded_rows.parquet.recovering"
    if not temporary.exists() and not temporary.is_symlink():
        return False
    history = manifest.get("recovery_history", [])
    if (
        not history
        or history[-1].get("status") != "running"
        or history[-1].get("mode") != "derived_tables_v1"
        or temporary.is_symlink()
        or not temporary.is_file()
        or temporary.stat().st_size != 0
    ):
        raise ValueError(f"Ambiguous prior recovery temporary file: {temporary.name}")
    target = output / "excluded_rows.parquet"
    if target.is_symlink() or (
        target.exists() and (not target.is_file() or target.stat().st_size != 0)
    ):
        raise ValueError(f"Ambiguous prior recovery temporary file: {temporary.name}")
    temporary.unlink()
    return True


def _plan_auxiliary_recovery(connection, output):
    expected_rows = {
        "excluded_rows": connection.execute(
            "SELECT count(*) FROM classified WHERE disposition<>'accepted'"
        ).fetchone()[0],
        "sensor_statistics": connection.execute(
            "SELECT count(*) FROM (SELECT channel_id, sensor_type FROM clean GROUP BY ALL)"
        ).fetchone()[0],
    }
    plan = {}
    for name, query in AUXILIARY_EXPORTS.items():
        target = output / f"{name}.parquet"
        temporary = output / f".{name}.parquet.recovering"
        if temporary.exists():
            raise ValueError(f"Ambiguous prior recovery temporary file: {temporary.name}")
        if not target.exists() or target.stat().st_size == 0:
            plan[name] = "regenerate"
            continue
        try:
            parquet = pq.ParquetFile(target)
        except Exception as error:
            raise ValueError(f"Non-empty auxiliary export is unreadable: {target.name}") from error
        expected_schema = (
            connection.execute(f"SELECT * FROM ({query}) LIMIT 0").to_arrow_table().schema
        )
        if (
            parquet.metadata.num_rows != expected_rows[name]
            or parquet.schema_arrow.names != expected_schema.names
        ):
            raise ValueError(f"Non-empty auxiliary export is ambiguous: {target.name}")
        plan[name] = "reuse"
    return plan


def _recover_auxiliary_export(connection, output, name):
    target = output / f"{name}.parquet"
    temporary = output / f".{name}.parquet.recovering"
    try:
        export_auxiliary_table(connection, temporary, name)
        with temporary.open("rb") as stream:
            parquet = pq.ParquetFile(stream)
            actual_rows = parquet.metadata.num_rows
            actual_schema = parquet.schema_arrow
        query = AUXILIARY_EXPORTS[name]
        expected_rows = connection.execute(f"SELECT count(*) FROM ({query})").fetchone()[0]
        expected_schema = (
            connection.execute(f"SELECT * FROM ({query}) LIMIT 0").to_arrow_table().schema
        )
        if actual_rows != expected_rows or actual_schema.names != expected_schema.names:
            raise ValueError(
                f"Regenerated auxiliary export failed validation: {name}; "
                f"rows={actual_rows}/{expected_rows}"
            )
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_report_targets(output):
    for name in ("data_quality.json", "sensor_family_report.csv"):
        if (output / name).exists():
            raise ValueError(f"Ambiguous pre-existing recovery report: {name}")
    temporary = output / ".recovery-reports"
    if temporary.exists():
        raise ValueError("Ambiguous prior recovery report directory")


def _write_recovery_reports(output, report):
    temporary = output / ".recovery-reports"
    temporary.mkdir(exist_ok=False)
    try:
        write_quality_outputs(temporary, report)
        for name in ("sensor_family_report.csv", "data_quality.json"):
            (temporary / name).replace(output / name)
    finally:
        if temporary.exists():
            temporary.rmdir()


def _validate_resume(connection, manifest, paths, *, resume_source_sha256=None):
    """Recover only a raw-load checkpoint; never replay derived tables blindly."""
    tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    if tables != {"raw", "channels", "objects"}:
        raise ValueError(f"Unsafe resume phase: database tables are {sorted(tables)}")
    groups = connection.execute(
        """SELECT source, count(*), min(source_row), max(source_row),
                  min(row_id), max(row_id)
           FROM raw GROUP BY source ORDER BY min(row_id)"""
    ).fetchall()
    committed_by_source = {}
    previous_row_id = 0
    for index, (
        source,
        count,
        first_source_row,
        last_source_row,
        first_row_id,
        last_row_id,
    ) in enumerate(groups):
        if index >= len(paths) or source != str(paths[index]):
            raise ValueError("Resume raw source order differs from configuration")
        if (
            first_source_row != 2
            or last_source_row != count + 1
            or first_row_id != previous_row_id + 1
            or last_row_id != previous_row_id + count
        ):
            raise ValueError("Resume raw source/row_id checkpoint is not contiguous")
        committed_by_source[source] = count
        previous_row_id = last_row_id
    if len(groups) < len(manifest["sources"]) or len(groups) > len(manifest["sources"]) + 1:
        raise ValueError("Resume raw groups disagree with completed source records")
    for index, record in enumerate(manifest["sources"]):
        path = paths[index]
        if (
            record["path"] != str(path)
            or record["rows_read"] != committed_by_source.get(str(path), 0)
            or record["bytes"] != path.stat().st_size
            or record["sha256"] != sha256(path)
        ):
            raise ValueError(f"Completed source changed since checkpoint: {path}")
    next_index = len(manifest["sources"])
    if next_index < len(paths):
        path = paths[next_index]
        active = manifest.get("active_source")
        committed = committed_by_source.get(str(path), 0)
        expected_hash = active["sha256"] if active is not None else resume_source_sha256
        if active is not None and (
            active["path"] != str(path) or active["bytes"] != path.stat().st_size
        ):
            raise ValueError("Active source path or size differs from checkpoint")
        if committed and expected_hash is None:
            raise ValueError("Legacy partial source requires its pre-failure SHA-256")
        if expected_hash is not None and sha256(path) != expected_hash:
            raise ValueError("Active source SHA-256 differs from checkpoint")
    return committed_by_source


def _insert(connection, rows):
    connection.register("incoming", pa.Table.from_pylist(rows, schema=STAGING_SCHEMA))
    connection.execute("INSERT INTO raw SELECT * FROM incoming")
    connection.unregister("incoming")
