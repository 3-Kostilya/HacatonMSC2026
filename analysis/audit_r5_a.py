"""Independent month-by-month integrity, coverage and leakage audit for R5 A."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from analysis.build_r5_a import DEFAULT_A3, DEFAULT_CANDIDATES, DEFAULT_OUTPUT, KEYS, _fold, _month_path
from stage1.features.r5 import MODEL_COLUMNS, R5_VERSION, STAT_COLUMNS


def _digest(path: Path) -> str:
    hash_ = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hash_.update(block)
    return hash_.hexdigest()


def audit(a3: Path, candidates: Path, output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    a_manifest = json.loads((a3 / "manifest.json").read_text(encoding="utf-8"))
    c_manifest = json.loads((candidates / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != R5_VERSION:
        raise ValueError("R5 version mismatch")
    if manifest["source_a3_manifest_sha256"] != _digest(a3 / "manifest.json"):
        raise ValueError("A3 root manifest mismatch")
    if manifest["source_candidate_manifest_sha256"] != _digest(candidates / "manifest.json"):
        raise ValueError("B candidate root manifest mismatch")
    if _digest(output / "fit_sample.parquet") != manifest["fit_sample_sha256"]:
        raise ValueError("R5 fit sample hash mismatch")
    sample = pq.read_table(output / "fit_sample.parquet", columns=[*KEYS, "split"])
    if set(sample["split"].to_pylist()) != {"train"} or sample.num_rows != manifest["fit_sample_rows"]:
        raise ValueError("R5 fit sample contains non-train rows")
    for details in manifest["folds"].values():
        for type_detail in details["types"].values():
            if type_detail["status"] == "fitted":
                path = output / type_detail["model_file"]
                if _digest(path) != type_detail["model_sha256"]:
                    raise ValueError(f"fitted model hash mismatch: {path}")

    a_chunks = {chunk["month"]: chunk for chunk in a_manifest["chunks"]}
    c_chunks = {chunk["month"]: chunk for chunk in c_manifest["chunks"]}
    totals = Counter()
    by_split = defaultdict(Counter)
    by_type = defaultdict(Counter)
    for chunk in manifest["chunks"]:
        month = chunk["month"]
        r_path = output / chunk["features_file"]
        a_path = a3 / a_chunks[month]["features_file"]
        c_month_path = _month_path(candidates, month, "manifest.json")
        if _digest(c_month_path) != c_chunks[month]["manifest_sha256"]:
            raise ValueError(f"B month manifest hash mismatch in {month}")
        c_month_manifest = json.loads(c_month_path.read_text(encoding="utf-8"))
        c_path = _month_path(candidates, month, "conditional_discrete_keys.parquet")
        for path, expected in ((r_path, chunk["features_sha256"]),
                               (a_path, a_chunks[month]["features_sha256"]),
                               (c_path, c_month_manifest["candidate_sha256"])):
            if _digest(path) != expected:
                raise ValueError(f"source or result file hash mismatch: {path}")
        left = pq.read_table(c_path, columns=[*KEYS, "sensor_type", "split"]).to_pandas()
        right = pq.read_table(r_path).to_pandas()
        if len(left) != chunk["rows"] or len(right) != len(left):
            raise ValueError(f"row-count mismatch in {month}")
        if right.duplicated(KEYS).any() or left.duplicated(KEYS).any():
            raise ValueError(f"duplicate R5 or B key in {month}")
        if not left.sort_values(KEYS).reset_index(drop=True)[KEYS + ["sensor_type"]].equals(
                right.sort_values(KEYS).reset_index(drop=True)[KEYS + ["sensor_type"]]):
            raise ValueError(f"R5/B key or type mismatch in {month}")
        if set(right.columns) & {"target", "target_episode_id", "label_available_at", "split"}:
            raise ValueError(f"target or split leaked into R5 output: {month}")
        if not right[list(STAT_COLUMNS)].notna().all().all():
            raise ValueError(f"missing statistical score in {month}")
        for name in (*STAT_COLUMNS, *MODEL_COLUMNS,
                     "r5_kmeans_mode_changed", "r5_hdbscan_proxy_mode_changed"):
            observed = right[name].dropna().to_numpy()
            if not np.isfinite(observed).all():
                raise ValueError(f"non-finite {name} in {month}")
        expected_fold = _fold(month)
        name = expected_fold[0] if expected_fold else "warmup-stat-only"
        if set(right["r5_model_fold"]) != {name}:
            raise ValueError(f"incorrect model fold in {month}")
        if expected_fold is None and right["r5_if_anomaly_score"].notna().any():
            raise ValueError(f"warm-up month has fitted-model scores: {month}")
        available = int(right["r5_if_anomaly_score"].notna().sum())
        if available != chunk["models_available_rows"]:
            raise ValueError(f"model coverage differs from manifest in {month}")
        totals["rows"] += len(right)
        totals["if_rows"] += available
        totals["hdbscan_rows"] += int(right["r5_hdbscan_centroid_distance"].notna().sum())
        totals["kmeans_transition_rows"] += int(right["r5_kmeans_mode_changed"].notna().sum())
        if left["split"].nunique() != 1:
            raise ValueError(f"mixed temporal splits in {month}")
        split = str(left["split"].iloc[0])
        by_split[split]["rows"] += len(right)
        by_split[split]["if_rows"] += available
        by_split[split]["hdbscan_rows"] += int(right["r5_hdbscan_centroid_distance"].notna().sum())
        for sensor_type, group in right.groupby("sensor_type", dropna=False):
            key = str(sensor_type)
            by_type[key]["rows"] += len(group)
            by_type[key]["if_rows"] += int(group["r5_if_anomaly_score"].notna().sum())
            by_type[key]["hdbscan_rows"] += int(group["r5_hdbscan_centroid_distance"].notna().sum())
        print(f"audited {month}: {len(right):,} rows", flush=True)
    if totals["rows"] != manifest["rows"]:
        raise ValueError("R5 total rows differ from manifest")
    report = {
        "schema_version": "r5-a-independent-audit-v1",
        "status": "passed",
        "months": len(manifest["chunks"]),
        "totals": dict(totals),
        "by_split": {key: dict(value) for key, value in sorted(by_split.items())},
        "by_type": {key: dict(value) for key, value in sorted(by_type.items())},
        "source_and_result_file_hashes_verified": True,
        "target_columns_absent": True,
        "note": "Coverage is method-specific; B must compare ablations on identical rows.",
    }
    (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a3", type=Path, default=DEFAULT_A3)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit(args.a3, args.candidates, args.output)
    print(json.dumps({"status": report["status"], "totals": report["totals"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
