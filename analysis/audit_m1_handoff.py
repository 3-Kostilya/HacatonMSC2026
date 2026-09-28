"""B's read-only audit of an ingestion handoff, independently recomputing checks.

Only CSV/7z transport is shared with ingestion; normalization, classification,
joins and reporting are not called by this auditor. Scratch SQL lives on disk.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage1.ingestion.sources import iter_source_rows  # noqa: E402

RAW = ("event_id_raw", "channel_id_raw", "date_raw", "time_raw", "alarm_raw", "value_raw")
FIELDS = ("ид_события", "ид_канала_данных", "дата", "время", "тревожное", "значение_датчика")
YEARS = (2019, 2020, 2022, 2023, 2024, 2025, 2026)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def audit_handoff(directory, *, scratch=None):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or directory.name.endswith(".inprogress"):
        raise ValueError("Only published, complete runs can be audited")
    sources = manifest["sources"]
    # Enforce the user's exclusion before opening or hashing any source.
    if any("2021" in Path(s["path"]).name for s in sources):
        raise ValueError("2021 is excluded from this M1 audit")
    report = {
        "audit_version": "m1-b-v1",
        "scope": manifest["scope"],
        "excluded_years": [2021],
        "checks": {},
        "sources": [],
    }
    checks = report["checks"]
    quality = json.loads((directory / "data_quality.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(dir=scratch, prefix="m1-b-") as temp:
        con = duckdb.connect(str(Path(temp) / "audit.duckdb"))
        try:
            con.execute("SET memory_limit='512MB'")
            con.execute("SET threads=1")
            con.execute("SET temp_directory=?", [str(Path(temp) / "spill")])
            clean_paths = sorted((directory / "clean").rglob("*.parquet"))
            if clean_paths:
                con.read_parquet(
                    [str(p) for p in clean_paths], hive_partitioning=False
                ).create_view("clean")
            else:
                # An empty published dataset still carries typed normalized columns.
                con.read_parquet(str(directory / "excluded_rows.parquet")).create_view(
                    "empty_source"
                )
                con.execute(
                    "CREATE VIEW clean AS SELECT *, NULL::VARCHAR AS sensor_type, "
                    "NULL::VARCHAR AS object_id, NULL::VARCHAR AS join_status "
                    "FROM empty_source WHERE false"
                )
            con.read_parquet(str(directory / "excluded_rows.parquet")).create_view("excluded")
            selected = ",".join(("source", "source_row", *RAW))
            con.execute(
                f"CREATE VIEW delivered AS SELECT {selected} FROM clean UNION ALL "
                f"SELECT {selected} FROM excluded"
            )
            con.execute(
                "CREATE TABLE original (source VARCHAR, source_row BIGINT, "
                + ",".join(f"{name} VARCHAR" for name in RAW)
                + ")"
            )
            schema = pa.schema(
                [
                    ("source", pa.string()),
                    ("source_row", pa.int64()),
                    *[(name, pa.string()) for name in RAW],
                ]
            )

            def insert(items):
                con.register("batch", pa.Table.from_pylist(items, schema=schema))
                con.execute("INSERT INTO original SELECT * FROM batch")
                con.unregister("batch")

            for source in sources:
                path = Path(source["path"])
                before = digest(path)
                pending, count = [], 0
                for row in iter_source_rows(path, source.get("max_rows")):
                    count += 1
                    pending.append(
                        {
                            "source": row["__source__"],
                            "source_row": row["__source_row__"],
                            **dict(zip(RAW, (row[key] for key in FIELDS))),
                        }
                    )
                    if len(pending) == 25000:
                        insert(pending)
                        pending.clear()
                if pending:
                    insert(pending)
                report["sources"].append(
                    {
                        "file": path.name,
                        "rows_replayed": count,
                        "hash_matches": before == source["sha256"] == digest(path),
                        "count_matches": count == source["rows_read"],
                    }
                )
            checks["source_hashes_and_counts"] = all(
                row["hash_matches"] and row["count_matches"] for row in report["sources"]
            )

            def scalar(sql):
                return con.execute(sql).fetchone()[0]

            counts = {"accepted": scalar("SELECT count(*) FROM clean")}
            counts.update(
                dict(
                    con.execute(
                        "SELECT disposition,count(*) FROM excluded GROUP BY disposition"
                    ).fetchall()
                )
            )
            report["dispositions"] = counts
            report["input_rows"] = scalar("SELECT count(*) FROM original")
            checks["row_balance"] = (
                sum(counts.values()) == report["input_rows"] == manifest["input_rows"]
            )
            checks["reported_counts"] = {k: v for k, v in counts.items() if v} == {
                k: v for k, v in quality["dispositions"].items() if v
            }
            checks["unique_provenance"] = (
                scalar("SELECT count(*)-count(DISTINCT (source,source_row)) FROM delivered") == 0
            )
            checks["raw_fields_preserved"] = (
                scalar(
                    "SELECT count(*) FROM "
                    "((SELECT * FROM original EXCEPT ALL SELECT * FROM delivered) UNION ALL "
                    "(SELECT * FROM delivered EXCEPT ALL SELECT * FROM original))"
                )
                == 0
            )
            keys = ",".join(RAW)
            checks["no_exact_duplicates_in_clean"] = (
                scalar(
                    f"SELECT count(*) FROM (SELECT {keys} FROM clean GROUP BY {keys} HAVING count(*)>1)"
                )
                == 0
            )
            same = " AND ".join(f"e.{key} IS NOT DISTINCT FROM o.{key}" for key in RAW)
            checks["duplicate_references_match_all_six_fields"] = (
                scalar(
                    "SELECT count(*) FROM excluded e LEFT JOIN original o "
                    "ON e.duplicate_of_source=o.source AND e.duplicate_of_source_row=o.source_row "
                    f"WHERE e.disposition='exact_duplicate' AND (o.source IS NULL OR NOT ({same}))"
                )
                == 0
            )
            checks["value_representation"] = (
                scalar(
                    "SELECT count(*) FROM clean WHERE "
                    "is_numeric IS NULL OR is_numeric<>(value_numeric IS NOT NULL) OR "
                    "(is_numeric AND (value_state IS NOT NULL OR NOT isfinite(value_numeric))) OR "
                    "(NOT is_numeric AND value_state IS DISTINCT FROM value_raw)"
                )
                == 0
            )
            checks["special_numbers_preserved"] = (
                scalar(
                    "SELECT count(*) FROM clean WHERE "
                    "trim(value_raw) IN ('-3276','327.68','999') AND "
                    "value_numeric IS DISTINCT FROM try_cast(value_raw AS DOUBLE)"
                )
                == 0
            )
            checks["no_2021_events"] = (
                scalar("SELECT count(*) FROM clean WHERE year(timestamp)=2021") == 0
            )
            checks["valid_local_timestamps"] = (
                all(
                    pa.types.is_timestamp(pq.ParquetFile(p).schema_arrow.field("timestamp").type)
                    and pq.ParquetFile(p).schema_arrow.field("timestamp").type.tz is None
                    for p in clean_paths
                )
                and scalar("SELECT count(*) FROM clean WHERE timestamp IS NULL") == 0
            )
            ordered = True
            for path in clean_paths:
                previous = None
                for batch in pq.ParquetFile(path).iter_batches(
                    columns=["channel_id", "timestamp", "row_id"]
                ):
                    for row in batch.to_pylist():
                        key = (row["channel_id"], row["timestamp"], row["row_id"])
                        if previous is not None and key < previous:
                            ordered = False
                        previous = key
            checks["physical_sort_order"] = ordered
            dictionary_path = Path(manifest["configuration"]["channels"])
            with dictionary_path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                channels = list(reader)
            channel_ids = [r["ид_канала_данных"].strip() for r in channels]
            checks["unique_dictionary_channels"] = len(channel_ids) == len(set(channel_ids))
            con.register(
                "dictionary",
                pa.table(
                    {
                        "id": channel_ids,
                        "kind": [r["тип_датчика"] for r in channels],
                        "object": pa.array(
                            [r.get("ид_объект", "").strip() or None for r in channels],
                            type=pa.string(),
                        ),
                    }
                ),
            )
            checks["dictionary_types_and_unknown_channels"] = (
                scalar(
                    "SELECT count(*) FROM clean c LEFT JOIN dictionary d ON c.channel_id=d.id "
                    "WHERE (d.id IS NULL AND (c.join_status<>'unknown_channel' OR "
                    "NOT list_contains(c.quality_flags,'unknown_channel'))) OR "
                    "(d.id IS NOT NULL AND c.sensor_type IS DISTINCT FROM d.kind)"
                )
                == 0
            )
            report["object_mapping_available"] = "ид_объект" in fields
            with Path(manifest["configuration"]["objects"]).open(
                encoding="utf-8-sig", newline=""
            ) as stream:
                objects = list(csv.DictReader(stream))
            object_ids = [r["ид_объект"].strip() for r in objects]
            checks["unique_dictionary_objects"] = len(object_ids) == len(set(object_ids))
            con.register(
                "object_dictionary", pa.table({"id": pa.array(object_ids, type=pa.string())})
            )
            report["dictionary_objects"] = len(objects)
            report["channel_object_references"] = {
                "linked": sum(r.get("ид_объект", "").strip() in set(object_ids) for r in channels),
                "missing": sum(not r.get("ид_объект", "").strip() for r in channels),
                "not_found": sum(
                    bool(r.get("ид_объект", "").strip())
                    and r["ид_объект"].strip() not in set(object_ids)
                    for r in channels
                ),
            }
            if report["object_mapping_available"]:
                checks["object_links_match_dictionary"] = (
                    scalar(
                        "SELECT count(*) FROM clean c LEFT JOIN dictionary d ON c.channel_id=d.id "
                        "LEFT JOIN object_dictionary o ON d.object=o.id WHERE "
                        "c.object_id IS DISTINCT FROM d.object OR c.join_status IS DISTINCT FROM "
                        "CASE WHEN d.id IS NULL THEN 'unknown_channel' WHEN d.object IS NULL THEN "
                        "'object_id_missing' WHEN o.id IS NULL THEN 'object_not_found' ELSE 'linked' END"
                    )
                    == 0
                )
            if not report["object_mapping_available"]:
                checks["no_invented_object_links"] = (
                    scalar(
                        "SELECT count(*) FROM clean WHERE "
                        "object_id IS NOT NULL OR join_status NOT IN ('unknown_channel','object_mapping_unavailable')"
                    )
                    == 0
                )
            checks["dictionary_hashes"] = all(
                digest(item["path"]) == item["sha256"] for item in manifest["dictionaries"]
            )
            report["dictionary_channels"] = len(channels)
            report["dictionary_types"] = len({r["тип_датчика"] for r in channels})
            report["repeated_event_ids_retained"] = scalar(
                "SELECT coalesce(sum(n-1),0) FROM "
                "(SELECT count(*) n FROM clean WHERE event_id IS NOT NULL GROUP BY event_id HAVING count(*)>1)"
            )
            checks["conflict_flags_recomputed"] = (
                scalar(
                    "SELECT count(*) FROM clean c LEFT JOIN (SELECT channel_id,timestamp FROM clean "
                    "GROUP BY channel_id,timestamp HAVING count(DISTINCT (value_raw,alarm))>1) x "
                    "USING(channel_id,timestamp) WHERE list_contains(c.quality_flags,'channel_time_conflict') "
                    "IS DISTINCT FROM (x.channel_id IS NOT NULL)"
                )
                == 0
            )
            for name, sql in {
                "join_status": "SELECT join_status,count(*) AS rows FROM clean GROUP BY join_status",
                "by_type": "SELECT sensor_type,count(*) AS rows,count(DISTINCT channel_id) channels FROM clean GROUP BY sensor_type ORDER BY rows DESC",
                "by_month": "SELECT year(timestamp) AS year,month(timestamp) AS month,count(*) AS rows FROM clean GROUP BY 1,2 ORDER BY 1,2",
                "quality_flags": "SELECT flag,count(*) AS rows FROM (SELECT unnest(quality_flags) AS flag FROM clean) GROUP BY flag ORDER BY flag",
                "special_values": "SELECT value_raw,count(*) AS rows FROM clean WHERE value_numeric IN (-3276,327.68,999) OR list_contains(quality_flags,'nonfinite_numeric') GROUP BY value_raw",
            }.items():
                cursor = con.execute(sql)
                report[name] = [
                    dict(zip([c[0] for c in cursor.description], row)) for row in cursor.fetchall()
                ]
            report["passed"] = all(checks.values())
            return report
        finally:
            con.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new report path")
    report = audit_handoff(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": report["checks"]}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
