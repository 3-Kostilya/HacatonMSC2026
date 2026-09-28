"""Make a small, reproducible B1 sample from a published M1 clean artifact.

The selector scans all clean rows, retaining one deterministic hash-minimum
per category and hash bucket. It then chooses rows to cover as many observed
categories as the row budget permits. The output retains the clean schema.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

SELECTOR_VERSION = "m1-b1-sample-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _categories_sql(glob: str) -> str:
    # row_id is the global immutable ingestion identifier. DuckDB's hash is
    # deterministic in the pinned runtime; its version is recorded in report.
    return f"""
        WITH categorized AS (
          SELECT row_id, hash(row_id) AS selection_hash,
                 unnest(list_concat(
                   [
                     'sensor_type:' || coalesce(sensor_type, '<null>'),
                     'join_status:' || coalesce(join_status, '<null>'),
                     'value_kind:' || CASE
                       WHEN value_numeric IS NOT NULL AND value_state IS NOT NULL THEN 'mixed'
                       WHEN value_numeric IS NOT NULL THEN 'numeric'
                       WHEN value_state IS NOT NULL THEN 'state'
                       ELSE 'missing' END,
                     'sensor_type_value_kind:' || coalesce(sensor_type, '<null>') || ':' || CASE
                       WHEN value_numeric IS NOT NULL AND value_state IS NOT NULL THEN 'mixed'
                       WHEN value_numeric IS NOT NULL THEN 'numeric'
                       WHEN value_state IS NOT NULL THEN 'state'
                       ELSE 'missing' END,
                     'alarm:' || coalesce(cast(alarm AS VARCHAR), '<null>')
                   ],
                   list_transform(coalesce(quality_flags, []), flag -> 'quality_flag:' || flag)
                 )) AS category
          FROM read_parquet('{glob}', hive_partitioning=false)
        )
        SELECT category, selection_hash % ? AS bucket,
               arg_min(row_id, selection_hash) AS row_id,
               min(selection_hash) AS selection_hash
        FROM categorized
        GROUP BY category, bucket
        ORDER BY category, bucket
    """


def _choose_rows(candidates: list[tuple], max_rows: int) -> tuple[list[int], list[str], list[str]]:
    coverage: dict[int, set[str]] = collections.defaultdict(set)
    hashes: dict[int, int] = {}
    all_categories: set[str] = set()
    for category, _bucket, row_id, selection_hash in candidates:
        category = str(category)
        row_id = int(row_id)
        coverage[row_id].add(category)
        hashes[row_id] = int(selection_hash)
        all_categories.add(category)

    chosen: list[int] = []
    uncovered = set(all_categories)
    remaining = set(coverage)
    while uncovered and remaining and len(chosen) < max_rows:
        best = min(
            remaining,
            key=lambda row_id: (-len(coverage[row_id] & uncovered), hashes[row_id], row_id),
        )
        if not (coverage[best] & uncovered):
            break
        chosen.append(best)
        remaining.remove(best)
        uncovered.difference_update(coverage[best])
    for row_id in sorted(remaining, key=lambda row_id: (hashes[row_id], row_id)):
        if len(chosen) >= max_rows:
            break
        chosen.append(row_id)
    return chosen, sorted(all_categories - uncovered), sorted(uncovered)


def build_sample(
    artifact: Path,
    output: Path,
    *,
    max_rows: int = 512,
    buckets: int = 8,
    allow_bounded: bool = False,
) -> dict:
    if max_rows < 1 or buckets < 1:
        raise ValueError("max_rows and buckets must be positive")
    artifact = artifact.resolve()
    output = output.resolve()
    if artifact.name.endswith(".inprogress"):
        raise ValueError("cannot sample an unpublished .inprogress artifact")
    if artifact == output or artifact in output.parents:
        raise ValueError("sample output must be outside the published artifact")
    report_path = output.with_suffix(".report.json")
    if output.exists() or report_path.exists():
        raise FileExistsError("sample or report already exists; choose a new output path")
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("M1 manifest is not complete")
    if not allow_bounded and manifest.get("scope") != "full_supplied_sources":
        raise ValueError("B1 requires a full-history artifact; --allow-bounded is only for tests")

    files = sorted((artifact / "clean").rglob("*.parquet"))
    if not files:
        raise ValueError("published clean directory contains no Parquet files")
    schema = pq.read_schema(files[0])
    required = {
        "row_id",
        "source",
        "source_row",
        "sensor_type",
        "join_status",
        "quality_flags",
        "value_numeric",
        "value_state",
        "alarm",
        "channel_id",
        "timestamp",
    }
    if not required.issubset(schema.names):
        raise ValueError(f"clean schema lacks columns: {sorted(required - set(schema.names))}")
    for path in files[1:]:
        if not pq.read_schema(path).equals(schema, check_metadata=False):
            raise ValueError(f"inconsistent clean schema: {path}")

    glob = (artifact / "clean" / "**" / "*.parquet").as_posix().replace("'", "''")
    connection = duckdb.connect()
    try:
        connection.execute("SET threads=2")
        connection.execute("SET memory_limit='1GB'")
        candidates = connection.execute(_categories_sql(glob), [buckets]).fetchall()
        selected_ids, covered, missing = _choose_rows(candidates, max_rows)
        if not selected_ids:
            raise ValueError("clean dataset has no selectable rows")
        selected = connection.execute(
            f"""SELECT * FROM read_parquet('{glob}', hive_partitioning=false)
                WHERE row_id IN (SELECT unnest(?::BIGINT[]))
                ORDER BY channel_id, timestamp, row_id""",
            [selected_ids],
        ).to_arrow_table()
        duckdb_version = connection.execute("SELECT version()").fetchone()[0]
    finally:
        connection.close()
    if selected.num_rows != len(selected_ids):
        raise ValueError("row_id is not unique or selected rows were not recovered")
    if not selected.schema.equals(schema, check_metadata=False):
        selected = selected.cast(schema)
    if any(
        not row["source"] or row["source_row"] is None
        for row in selected.select(["source", "source_row"]).to_pylist()
    ):
        raise ValueError("sample contains rows without source provenance")

    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(selected, output, compression="zstd")
    counts_by_type = collections.Counter(
        value if value is not None else "<null>"
        for value in selected.column("sensor_type").to_pylist()
    )
    counts_by_join = collections.Counter(
        value if value is not None else "<null>"
        for value in selected.column("join_status").to_pylist()
    )
    counts_by_flag = collections.Counter(
        flag for flags in selected.column("quality_flags").to_pylist() for flag in (flags or [])
    )
    report = {
        "selector_version": SELECTOR_VERSION,
        "source_artifact": str(artifact),
        "source_scope": manifest["scope"],
        "source_manifest_sha256": sha256(manifest_path),
        "sample_parquet": str(output),
        "sample_sha256": sha256(output),
        "sample_rows": selected.num_rows,
        "max_rows": max_rows,
        "hash_buckets_per_category": buckets,
        "candidate_rows": len({int(row[2]) for row in candidates}),
        "observed_categories": sorted(set(covered) | set(missing)),
        "covered_categories": covered,
        "missing_categories": missing,
        "counts_by_sensor_type": dict(
            sorted(counts_by_type.items(), key=lambda item: str(item[0]))
        ),
        "counts_by_join_status": dict(
            sorted(counts_by_join.items(), key=lambda item: str(item[0]))
        ),
        "counts_by_quality_flag": dict(
            sorted(counts_by_flag.items(), key=lambda item: str(item[0]))
        ),
        "duckdb_version": duckdb_version,
        "pyarrow_version": pa.__version__,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-rows", type=int, default=512)
    parser.add_argument("--buckets", type=int, default=8)
    parser.add_argument("--allow-bounded", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build_sample(
                args.artifact,
                args.output,
                max_rows=args.max_rows,
                buckets=args.buckets,
                allow_bounded=args.allow_bounded,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
