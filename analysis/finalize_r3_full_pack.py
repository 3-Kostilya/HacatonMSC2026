"""Validate every R3 A month shard and publish one full-population manifest."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.build_r3_full_month import FULL_PACK_VERSION, POPULATION_VERSION  # noqa: E402
from stage1.features.r3 import (  # noqa: E402
    FEATURE_PACK_SCHEMA,
    MODEL_FEATURE_ALLOWLIST,
    ROW_STATUS_SCHEMA,
)
from stage1.state_labeling.operational import ARCHIVE_SEGMENTS  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _months() -> list[tuple[int, int]]:
    months = []
    for start, end in ARCHIVE_SEGMENTS:
        year, month = start.year, start.month
        while (year, month) < (end.year, end.month):
            months.append((year, month))
            year, month = year + (month == 12), month % 12 + 1
    return months


def finalize(*, output_root: Path, population_dir: Path, m1_manifest: Path,
             b2_manifest: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    population_dir = population_dir.resolve()
    m1_manifest = m1_manifest.resolve()
    b2_manifest = b2_manifest.resolve()
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("full R3 A pack already has a published manifest")
    population_manifest_path = population_dir / "manifest.json"
    population = json.loads(population_manifest_path.read_text(encoding="utf-8"))
    population_report = json.loads((population_dir / "report.json").read_text(encoding="utf-8"))
    if (
        population.get("schema_version") != POPULATION_VERSION
        or population.get("status") != "complete"
        or _sha256(population_dir / "report.json") != population["files"]["report.json"]["sha256"]
        or population.get("source_m1_manifest_sha256") != _sha256(m1_manifest)
    ):
        raise ValueError("population artifact or M1 lineage differs")
    expected_months = _months()
    chunks = []
    rows = 0
    by_split: Counter[str] = Counter()
    statuses: dict[str, Counter[str]] = {}
    for year, month in expected_months:
        directory = output_root / f"year={year}" / f"month={month:02d}"
        path = directory / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing full R3 month shard: {year}-{month:02d}")
        member = json.loads(path.read_text(encoding="utf-8"))
        if (
            member.get("schema_version") != FULL_PACK_VERSION
            or member.get("status") != "complete_month"
            or member.get("month") != f"{year}-{month:02d}"
            or member.get("source_m1_manifest_sha256") != _sha256(m1_manifest)
            or member.get("source_population_manifest_sha256") != _sha256(population_manifest_path)
            or member.get("source_b2_catalog_manifest_sha256") != _sha256(b2_manifest)
            or member.get("feature_count") != len(MODEL_FEATURE_ALLOWLIST)
        ):
            raise ValueError(f"full R3 month shard provenance differs: {year}-{month:02d}")
        for name, schema in (
            ("features.parquet", FEATURE_PACK_SCHEMA),
            ("row_status.parquet", ROW_STATUS_SCHEMA),
        ):
            file = directory / name
            expected = member["files"][name]
            if file.stat().st_size != expected["bytes"] or _sha256(file) != expected["sha256"]:
                raise ValueError(f"full R3 month file SHA-256 mismatch: {file}")
            parquet = pq.ParquetFile(file)
            if not parquet.schema_arrow.equals(schema, check_metadata=False) or (
                parquet.metadata.num_rows != member["row_count"]
            ):
                raise ValueError(f"full R3 month file schema or rows differ: {file}")
        relative = directory.relative_to(output_root)
        chunks.append({
            "month": member["month"],
            "manifest_file": (relative / "manifest.json").as_posix(),
            "manifest_sha256": _sha256(path),
            "features_file": (relative / "features.parquet").as_posix(),
            "features_sha256": member["files"]["features.parquet"]["sha256"],
            "row_status_file": (relative / "row_status.parquet").as_posix(),
            "row_status_sha256": member["files"]["row_status.parquet"]["sha256"],
            "rows": member["row_count"],
            "channels": member["channel_count"],
            "start_at": member["start_at"],
            "end_at": member["end_at"],
        })
        rows += member["row_count"]
        split = (
            "train" if year < 2025 else
            "validation" if year == 2025 else "test"
        )
        by_split[split] += member["row_count"]
        for name, counts in member["status_counts"].items():
            statuses.setdefault(name, Counter()).update(counts)
    if rows != population["candidate_hour_count"] or dict(by_split) != (
        population_report["hours_with_prior_normal_at_most_168h"]
    ):
        raise ValueError("full R3 month shards do not conserve all selected population hours")
    allowlist = {
        "schema_version": FULL_PACK_VERSION,
        "feature_columns": [
            {"name": name, "arrow_type": str(FEATURE_PACK_SCHEMA.field(name).type)}
            for name in MODEL_FEATURE_ALLOWLIST
        ],
        "key_columns_not_features": ["channel_id", "prediction_time"],
        "diagnostics_not_features": ROW_STATUS_SCHEMA.names[2:],
        "target_and_split_are_not_features": True,
    }
    allowlist_path = output_root / "model_feature_allowlist.json"
    report_path = output_root / "report.json"
    for path in (allowlist_path, report_path):
        if path.exists():
            raise FileExistsError(f"full R3 top-level file already exists: {path}")
    report = {
        "schema_version": FULL_PACK_VERSION,
        "population_version": POPULATION_VERSION,
        "candidate_hours_by_split": dict(sorted(by_split.items())),
        "candidate_hours_total": rows,
        "hours_outside_recent_normal_window_by_split": population_report[
            "outside_normal_window_hours"
        ],
        "source_m1_clean_files": population_report["source_clean_files"],
        "monthly_parts": len(chunks),
        "feature_count": len(MODEL_FEATURE_ALLOWLIST),
        "status_counts": {
            name: dict(sorted(counts.items())) for name, counts in sorted(statuses.items())
        },
        "limitations": [
            "Candidate grid uses only past same-channel exact Norma within 168h; B applies target and model admission separately.",
            "Hours outside the candidate grid are counted but have no heavy feature rows.",
            "Archive completeness remains a conditional assumption, not verified channel continuity.",
        ],
    }
    allowlist_path.write_text(json.dumps(allowlist, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": FULL_PACK_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "full_causal_population_unlabeled",
        "not_training_ready": True,
        "source_m1_manifest_sha256": _sha256(m1_manifest),
        "source_population_manifest_sha256": _sha256(population_manifest_path),
        "source_b2_catalog_manifest_sha256": _sha256(b2_manifest),
        "selection_policy": POPULATION_VERSION,
        "row_count": rows,
        "chunk_count": len(chunks),
        "feature_count": len(MODEL_FEATURE_ALLOWLIST),
        "allowlist_file": allowlist_path.name,
        "allowlist_sha256": _sha256(allowlist_path),
        "report_file": report_path.name,
        "report_sha256": _sha256(report_path),
        "chunks": chunks,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--population-dir", required=True, type=Path)
    parser.add_argument("--m1-manifest", required=True, type=Path)
    parser.add_argument("--b2-manifest", required=True, type=Path)
    args = parser.parse_args()
    manifest = finalize(
        output_root=args.output_root,
        population_dir=args.population_dir,
        m1_manifest=args.m1_manifest,
        b2_manifest=args.b2_manifest,
    )
    print(json.dumps({key: manifest[key] for key in (
        "status", "row_count", "chunk_count", "feature_count"
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
