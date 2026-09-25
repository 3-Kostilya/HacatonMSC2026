"""Publish conditional discrete-branch candidate keys from full A3/B3.

This is a separate admission index, never a feature table. Unknown channel
continuity remains explicit; the target is a future registered journal entry.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.build_r1_state_mapping import _sha256  # noqa: E402
from analysis.build_r3_full_month import FULL_PACK_VERSION  # noqa: E402
from stage1.state_labeling.forecast import LABEL_VERSION  # noqa: E402


ADMISSION_VERSION = "r3-b-conditional-discrete-v1"
CANDIDATE_SCHEMA = pa.schema([
    pa.field("channel_id", pa.string(), nullable=False),
    pa.field("prediction_time", pa.timestamp("us"), nullable=False),
    pa.field("sensor_type", pa.string()),
    pa.field("target", pa.int8(), nullable=False),
    pa.field("split", pa.string(), nullable=False),
    pa.field("target_episode_id", pa.string()),
    pa.field("label_available_at", pa.timestamp("us"), nullable=False),
    pa.field("availability_status", pa.string(), nullable=False),
    pa.field("admission_status", pa.string(), nullable=False),
])


def _sources(a3_dir: Path, b3_dir: Path) -> tuple[dict, dict, list[tuple[dict, dict]]]:
    a3_path, b3_path = a3_dir / "manifest.json", b3_dir / "manifest.json"
    a3 = json.loads(a3_path.read_text(encoding="utf-8"))
    b3 = json.loads(b3_path.read_text(encoding="utf-8"))
    if (
        a3.get("schema_version") != FULL_PACK_VERSION
        or a3.get("status") != "complete"
        or b3.get("schema_version") != LABEL_VERSION
        or b3.get("status") != "complete_full_labels"
        or b3.get("source_a3_full_manifest_sha256") != _sha256(a3_path)
        or b3.get("row_count") != a3.get("row_count")
        or b3.get("chunk_count") != a3.get("chunk_count")
        or _sha256(b3_dir / "report.json") != b3.get("report_sha256")
        or _sha256(a3_dir / a3["allowlist_file"]) != a3.get("allowlist_sha256")
    ):
        raise ValueError("full A3/B3 version or source lineage differs")
    a_chunks = {chunk["month"]: chunk for chunk in a3["chunks"]}
    b_chunks = {chunk["month"]: chunk for chunk in b3["chunks"]}
    if len(a_chunks) != len(a3["chunks"]) or set(a_chunks) != set(b_chunks):
        raise ValueError("full A3/B3 month sets differ")
    pairs = [(a_chunks[month], b_chunks[month]) for month in sorted(a_chunks)]
    if any(a["rows"] != b["rows"] for a, b in pairs):
        raise ValueError("full A3/B3 month row counts differ")
    if any(
        _sha256(a3_dir / a["manifest_file"]) != a["manifest_sha256"]
        or _sha256(b3_dir / b["manifest_file"]) != b["manifest_sha256"]
        for a, b in pairs
    ):
        raise ValueError("full A3/B3 month manifest SHA-256 mismatch")
    return a3, b3, pairs


def _eligible_rows(database: duckdb.DuckDBPyConnection, labels: Path,
                   statuses: Path) -> pa.RecordBatchReader:
    return database.execute(
        """SELECT l.channel_id, l.prediction_time, l.sensor_type,
                  CAST(l.target AS TINYINT) AS target, l.split,
                  l.target_episode_id, l.label_available_at,
                  s.availability_status,
                  'conditional_archive_assumption' AS admission_status
           FROM read_parquet(?) AS l
           JOIN read_parquet(?) AS s USING (channel_id, prediction_time)
           WHERE l.split_status = 'assigned'
             AND l.label_status IN ('positive', 'negative')
             AND l.target IN (0, 1)
             AND s.discrete_data_status = 'eligible'
             AND s.availability_status = 'unknown'
           ORDER BY l.channel_id, l.prediction_time""",
        [str(labels), str(statuses)],
    ).to_arrow_reader(batch_size=40_000)


def _completed(output: Path, a: dict, b: dict,
               a3_sha: str, b3_sha: str) -> bool:
    if not output.exists():
        return False
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    candidate = output / "conditional_discrete_keys.parquet"
    if (
        manifest.get("schema_version") != ADMISSION_VERSION
        or manifest.get("status") != "complete_month"
        or manifest.get("month") != a["month"]
        or manifest.get("source_a3_manifest_sha256") != a3_sha
        or manifest.get("source_b3_manifest_sha256") != b3_sha
        or manifest.get("source_a3_month_manifest_sha256") != a["manifest_sha256"]
        or manifest.get("source_b3_month_manifest_sha256") != b["manifest_sha256"]
        or _sha256(candidate) != manifest.get("candidate_sha256")
        or pq.ParquetFile(candidate).metadata.num_rows != manifest.get("row_count")
    ):
        raise ValueError(f"existing conditional admission month differs: {a['month']}")
    return True


def build(*, a3_dir: Path, b3_dir: Path, output_root: Path) -> dict[str, Any]:
    a3_dir, b3_dir, output_root = (path.resolve() for path in (a3_dir, b3_dir, output_root))
    a3, b3, pairs = _sources(a3_dir, b3_dir)
    a3_sha = _sha256(a3_dir / "manifest.json")
    b3_sha = _sha256(b3_dir / "manifest.json")
    output_root.mkdir(parents=True, exist_ok=True)
    completed, written = 0, 0
    with duckdb.connect(":memory:") as database:
        database.execute("SET threads=2")
        for a, b in pairs:
            month = a["month"]
            output = output_root / f"year={month[:4]}" / f"month={month[5:]}"
            if _completed(output, a, b, a3_sha, b3_sha):
                completed += 1
                continue
            pending = output.with_name(output.name + ".inprogress")
            if pending.exists():
                raise FileExistsError(f"conditional admission month is in progress: {month}")
            a_status = a3_dir / a["row_status_file"]
            b_labels = b3_dir / f"year={month[:4]}" / f"month={month[5:]}" / "registered_forecast_labels.parquet"
            if (
                _sha256(a_status) != a["row_status_sha256"]
                or _sha256(b_labels) != json.loads(
                    (b3_dir / b["manifest_file"]).read_text(encoding="utf-8")
                )["files"][b_labels.name]["sha256"]
            ):
                raise ValueError(f"A3/B3 month file SHA-256 mismatch: {month}")
            pending.mkdir(parents=True)
            candidate = pending / "conditional_discrete_keys.parquet"
            writer = pq.ParquetWriter(candidate, CANDIDATE_SCHEMA, compression="zstd")
            count = 0
            by_target: Counter[int] = Counter()
            try:
                for batch in _eligible_rows(database, b_labels, a_status):
                    table = pa.Table.from_batches([batch]).cast(CANDIDATE_SCHEMA)
                    writer.write_table(table)
                    count += table.num_rows
                    by_target.update(table.column("target").to_pylist())
            finally:
                writer.close()
            manifest = {
                "schema_version": ADMISSION_VERSION,
                "status": "complete_month",
                "month": month,
                "source_a3_manifest_sha256": a3_sha,
                "source_b3_manifest_sha256": b3_sha,
                "source_a3_month_manifest_sha256": a["manifest_sha256"],
                "source_b3_month_manifest_sha256": b["manifest_sha256"],
                "row_count": count,
                "target_counts": {str(key): value for key, value in sorted(by_target.items())},
                "candidate_sha256": _sha256(candidate),
                "candidate_bytes": candidate.stat().st_size,
            }
            (pending / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            pending.rename(output)
            written += 1
            print(json.dumps({"month": month, "rows": count, "targets": manifest["target_counts"]}),
                  flush=True)
    return {"months_total": len(pairs), "months_skipped": completed,
            "months_written": written, "a3_manifest_sha256": a3_sha,
            "b3_manifest_sha256": b3_sha,
            "a3_rows": a3["row_count"]}


def finalize(*, a3_dir: Path, b3_dir: Path, output_root: Path,
             refresh_summary: bool = False) -> dict[str, Any]:
    a3_dir, b3_dir, output_root = (path.resolve() for path in (a3_dir, b3_dir, output_root))
    _, _, pairs = _sources(a3_dir, b3_dir)
    a3_sha = _sha256(a3_dir / "manifest.json")
    b3_sha = _sha256(b3_dir / "manifest.json")
    report_path, manifest_path = output_root / "report.json", output_root / "manifest.json"
    if (report_path.exists() or manifest_path.exists()) and not refresh_summary:
        raise FileExistsError("conditional admission summary already exists")
    if refresh_summary:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            previous.get("source_a3_manifest_sha256") != a3_sha
            or previous.get("source_b3_manifest_sha256") != b3_sha
        ):
            raise ValueError("existing conditional admission summary has different sources")
    paths = []
    chunks = []
    expected_rows = 0
    for a, b in pairs:
        month = a["month"]
        directory = output_root / f"year={month[:4]}" / f"month={month[5:]}"
        if not _completed(directory, a, b, a3_sha, b3_sha):
            raise ValueError(f"missing conditional admission month: {month}")
        local_manifest = directory / "manifest.json"
        local = json.loads(local_manifest.read_text(encoding="utf-8"))
        candidate = directory / "conditional_discrete_keys.parquet"
        if not pq.ParquetFile(candidate).schema_arrow.equals(CANDIDATE_SCHEMA,
                                                                check_metadata=False):
            raise ValueError(f"conditional admission schema differs: {month}")
        expected_rows += local["row_count"]
        paths.append(str(candidate))
        chunks.append({"month": month, "manifest_file": local_manifest.relative_to(output_root).as_posix(),
                       "manifest_sha256": _sha256(local_manifest), "rows": local["row_count"]})
    with duckdb.connect(":memory:") as database:
        database.execute("SET threads=4")
        total, distinct_keys, invalid = database.execute(
            """SELECT count(*), count(DISTINCT (channel_id, prediction_time)),
                      count(*) FILTER (WHERE target NOT IN (0,1)
                        OR availability_status <> 'unknown'
                        OR admission_status <> 'conditional_archive_assumption'
                        OR label_available_at IS NULL)
               FROM read_parquet(?)""",
            [paths],
        ).fetchone()
        if total != expected_rows or distinct_keys != total or invalid:
            raise ValueError("conditional admission keys or statuses are invalid")
        grouped = database.execute(
            """SELECT split, sensor_type, target, count(*),
                      count(DISTINCT target_episode_id),
                      count(DISTINCT channel_id),
                      count(DISTINCT (channel_id, CAST(prediction_time AS DATE)))
               FROM read_parquet(?) GROUP BY split, sensor_type, target""",
            [paths],
        ).fetchall()
        split_details = database.execute(
            """SELECT split, target, count(*),
                      count(DISTINCT target_episode_id),
                      count(DISTINCT channel_id),
                      count(DISTINCT (channel_id, CAST(prediction_time AS DATE)))
               FROM read_parquet(?) GROUP BY split, target""",
            [paths],
        ).fetchall()
    by_split: dict[str, Counter[int]] = {}
    by_split_type: dict[str, dict[str, dict[str, Any]]] = {}
    for split, sensor_type, target, rows, episodes, channels, days in grouped:
        by_split.setdefault(split, Counter())[target] += rows
        by_split_type.setdefault(split, {}).setdefault(sensor_type or "<unknown>", {})[
            str(target)
        ] = {"rows": rows, "episodes": episodes, "channels": channels,
             "channel_days": days}
    all_types = set().union(*(set(types) for types in by_split_type.values()))
    both_all_splits = sorted(
        name for name in all_types
        if all(
            {"0", "1"} <= set(by_split_type.get(split, {}).get(name, {}))
            for split in ("train", "validation", "test")
        )
    )
    report = {
        "schema_version": ADMISSION_VERSION,
        "status": "conditional_discrete_candidates_not_jointly_accepted",
        "row_count": total,
        "month_count": len(chunks),
        "by_split": {split: {str(target): count for target, count in sorted(counts.items())}
                     for split, counts in sorted(by_split.items())},
        "by_split_details": {
            split: {
                str(target): {"rows": rows, "episodes": episodes,
                              "channels": channels, "channel_days": days}
                for item_split, target, rows, episodes, channels, days in split_details
                if item_split == split
            } for split in sorted(by_split)
        },
        "by_split_and_type": {split: dict(sorted(types.items()))
                              for split, types in sorted(by_split_type.items())},
        "types_with_both_classes_in_all_splits": both_all_splits,
        "admission_rule": {
            "target": "0_or_1_registered_event_only",
            "split_status": "assigned_after_24h_purge",
            "feature_branch": "discrete_data_status_eligible",
            "availability_status": "unknown_preserved",
            "channel_continuity": "not_verified",
            "archive_completeness": "conditional_operational_assumption",
            "physical_failure_label": False,
        },
        "limitations": [
            "A3 model feature allowlist still governs training; keys, labels and episode IDs are not features.",
            "This is conditional journal-event modeling, not verified physical sensor failure prediction.",
            "The test split stays sealed for model choice and threshold selection.",
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    manifest = {
        "schema_version": ADMISSION_VERSION,
        "status": "complete_conditional_candidates",
        "not_training_ready": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_a3_manifest_sha256": a3_sha,
        "source_b3_manifest_sha256": b3_sha,
        "row_count": total,
        "chunk_count": len(chunks),
        "report_sha256": _sha256(report_path),
        "chunks": chunks,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--b3-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--refresh-summary", action="store_true")
    args = parser.parse_args()
    build_report = build(a3_dir=args.a3_dir, b3_dir=args.b3_dir,
                         output_root=args.output_root)
    print(json.dumps(build_report, ensure_ascii=False), flush=True)
    if args.refresh_summary or not (args.output_root / "manifest.json").exists():
        report = finalize(a3_dir=args.a3_dir, b3_dir=args.b3_dir,
                          output_root=args.output_root,
                          refresh_summary=args.refresh_summary)
        print(json.dumps({"row_count": report["row_count"],
                          "by_split": report["by_split"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
