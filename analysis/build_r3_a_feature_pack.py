"""Publish an immutable, monthly partitioned R3 A feature-only pack from A2/R2.

This is a bounded handoff of already computed causal features, not a labeled
training dataset. B owns target, model admission, split and purge.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage1.features.r3 import (  # noqa: E402
    FEATURE_PACK_SCHEMA,
    MODEL_FEATURE_ALLOWLIST,
    R3_PACK_VERSION,
    ROW_STATUS_SCHEMA,
    build_pack_tables,
    summarize_partitions,
)
from stage1.features.r2 import R2_VERSION  # noqa: E402
from stage1.features.schema import FEATURE_VERSION  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(path: Path, version: str) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != version or manifest.get("status") != "complete":
        raise ValueError(f"expected complete {version} manifest: {path}")
    return manifest


def _verified_table(path: Path, expected_sha: str, expected_rows: int) -> pa.Table:
    if _sha256(path) != expected_sha:
        raise ValueError(f"source file SHA-256 mismatch: {path}")
    table = pq.read_table(path)
    if table.num_rows != expected_rows:
        raise ValueError(f"source row count mismatch: {path}")
    return table


def build(*, a2_dir: Path, r2_dir: Path, output: Path) -> dict[str, Any]:
    a2_dir, r2_dir, output = (path.resolve() for path in (a2_dir, r2_dir, output))
    pending = output.with_name(output.name + ".inprogress")
    if output.exists() or pending.exists():
        raise FileExistsError("R3 output or its .inprogress directory already exists")
    if output == a2_dir or output == r2_dir or a2_dir in output.parents or r2_dir in output.parents:
        raise ValueError("R3 output must be outside both source artifacts")

    a2_manifest_path = a2_dir / "manifest.json"
    r2_manifest_path = r2_dir / "manifest.json"
    a2 = _read_manifest(a2_manifest_path, FEATURE_VERSION)
    r2 = _read_manifest(r2_manifest_path, R2_VERSION)
    a2_manifest_sha = _sha256(a2_manifest_path)
    if r2.get("source_a2_manifest_sha256") != a2_manifest_sha:
        raise ValueError("R2 does not derive from this exact A2 artifact")
    if r2.get("source_m1_manifest_sha256") != a2.get("input_manifest_sha256"):
        raise ValueError("R2 and A2 do not derive from the same M1 manifest")
    catalog = r2.get("episode_catalog")
    if not isinstance(catalog, dict) or not catalog.get("catalog_manifest_sha256"):
        raise ValueError("R3 requires R2 with a verified B2 episode catalog")
    if catalog.get("catalog_input_manifest_sha256") != a2["input_manifest_sha256"]:
        raise ValueError("B2 episode catalog does not derive from the same M1")
    r2_report = r2["files"].get("report.json")
    if not isinstance(r2_report, dict) or _sha256(r2_dir / "report.json") != r2_report.get("sha256"):
        raise ValueError("R2 report SHA-256 mismatch")
    a2_table = _verified_table(
        a2_dir / a2["features_file"], a2["features_sha256"], a2["feature_rows"]
    )
    r2_file = r2["files"]["state_history.parquet"]
    r2_table = _verified_table(
        r2_dir / "state_history.parquet", r2_file["sha256"], r2["row_count"]
    )
    if set(r2_table.column("source_a2_manifest_sha256").to_pylist()) != {a2_manifest_sha}:
        raise ValueError("R2 rows have a different A2 manifest provenance")
    if set(r2_table.column("ruleset_version").to_pylist()) != {r2["ruleset_version"]}:
        raise ValueError("R2 row ruleset differs from its manifest")
    if set(r2_table.column("episode_history_status").to_pylist()) != {
        "unambiguous_completed_only"
    }:
        raise ValueError("R3 requires the B2-backed R2 episode-history policy")
    features, statuses = build_pack_tables(a2_table, r2_table)
    if features.num_rows != a2["feature_rows"]:
        raise ValueError("R3 feature count differs from A2")
    start_at = datetime.fromisoformat(a2["config"]["start_at"])
    end_at = datetime.fromisoformat(a2["config"]["end_at"])
    baseline_embargo = timedelta(seconds=a2["config"]["feature_config"]["baseline_embargo"])
    for row in statuses.select(["prediction_time", "baseline_fit_end_at"]).to_pylist():
        t = row["prediction_time"]
        if not start_at <= t < end_at or (t.minute, t.second, t.microsecond) != (0, 0, 0):
            raise ValueError("R3 row falls outside the source hourly prediction interval")
        if row["baseline_fit_end_at"] > t - timedelta(hours=168) - baseline_embargo:
            raise ValueError("R3 baseline overlaps a causal feature window")

    # The schema and allowlist are separate artifacts so a consumer cannot
    # accidentally train on identifiers, diagnostics or future target fields.
    allowlist = {
        "schema_version": R3_PACK_VERSION,
        "feature_columns": [
            {"name": name, "arrow_type": str(FEATURE_PACK_SCHEMA.field(name).type)}
            for name in MODEL_FEATURE_ALLOWLIST
        ],
        "key_columns_not_features": ["channel_id", "prediction_time"],
        "diagnostics_not_features": ROW_STATUS_SCHEMA.names[2:],
        "forbidden_even_if_joined_later": [
            "episode_id", "event_id", "split", "target", "target_status",
            "target_reason", "future_label_status", "future_label_reasons",
            "model_admission_status", "model_admission_reasons", "provenance",
        ],
    }
    by_month: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, t in enumerate(features.column("prediction_time").to_pylist()):
        by_month[t.year, t.month].append(index)
    if not by_month:
        raise ValueError("R3 source contains no rows")
    selection_mode = a2["config"]["selection_mode"]
    qa_only = selection_mode != "explicit_channels_file_v1"
    output.parent.mkdir(parents=True, exist_ok=True)
    pending.mkdir()
    chunks = []
    for (year, month), indices in sorted(by_month.items()):
        month_dir = pending / f"year={year}" / f"month={month:02d}"
        month_dir.mkdir(parents=True)
        take = pa.array(indices, type=pa.int64())
        feature_chunk = features.take(take)
        status_chunk = statuses.take(take)
        feature_path = month_dir / "features.parquet"
        status_path = month_dir / "row_status.parquet"
        pq.write_table(feature_chunk, feature_path, compression="zstd")
        pq.write_table(status_chunk, status_path, compression="zstd")
        chunks.append(
            {
                "year": year,
                "month": month,
                "features_file": feature_path.relative_to(pending).as_posix(),
                "features_sha256": _sha256(feature_path),
                "row_status_file": status_path.relative_to(pending).as_posix(),
                "row_status_sha256": _sha256(status_path),
                **summarize_partitions(feature_chunk, status_chunk),
            }
        )
    allowlist_path = pending / "model_feature_allowlist.json"
    allowlist_path.write_text(
        json.dumps(allowlist, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        "schema_version": R3_PACK_VERSION,
        "source_a2_manifest_sha256": a2_manifest_sha,
        "source_a2_features_sha256": a2["features_sha256"],
        "source_r2_manifest_sha256": _sha256(r2_manifest_path),
        "source_r2_features_sha256": r2_file["sha256"],
        "source_m1_manifest_sha256": a2["input_manifest_sha256"],
        "source_b2_catalog_manifest_sha256": catalog["catalog_manifest_sha256"],
        "ruleset_version": r2["ruleset_version"],
        "source_selection_mode": selection_mode,
    }
    run_id = "r3-a-" + hashlib.sha256(
        json.dumps(provenance, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    manifest = {
        **provenance,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "purpose": "qa_only" if qa_only else "feature_only_unlabeled",
        "not_training_ready": True,
        "reason_not_training_ready": (
            "source channels were selected using target-period presence; B target/split are absent"
            if qa_only else "B target, model admission and temporal split are absent"
        ),
        "feature_column_count": len(MODEL_FEATURE_ALLOWLIST),
        "allowlist_file": allowlist_path.name,
        "allowlist_sha256": _sha256(allowlist_path),
        "row_count": features.num_rows,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
    (pending / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pending.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a2-dir", required=True, type=Path)
    parser.add_argument("--r2-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = build(a2_dir=args.a2_dir, r2_dir=args.r2_dir, output=args.output)
    print(json.dumps({key: manifest[key] for key in (
        "run_id", "purpose", "row_count", "chunk_count", "feature_column_count"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
