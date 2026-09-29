"""Attribute old admission vetoes to assigned train/validation episodes, not new admission."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.build_quality_improvement_a import YEARS, expected_months, quoted, safe_path
from analysis.r6_provenance import frozen_rule_sha256
from analysis.train_r4_discrete_baselines import read_json, sha256


VERSION = "a-episode-admission-blockers-v1"
SCENARIOS = {
    "unchanged": frozenset(),
    "without_baseline_veto": frozenset({"baseline_unusable"}),
    "without_state_history_vetoes": frozenset(
        {"state_history_missing", "state_transitions_unavailable"}
    ),
    "without_baseline_and_state_history_vetoes": frozenset(
        {"baseline_unusable", "state_history_missing", "state_transitions_unavailable"}
    ),
}


def summarize_points(points) -> tuple[list[dict], dict]:
    grouped = {}
    seen_keys = set()
    for point in points:
        year = point["prediction_time"].year
        expected_split = "validation" if year == 2025 else "train"
        if year not in YEARS or point["split"] != expected_split:
            raise ValueError("test/excluded-year point in admission audit")
        identity = (point["channel_id"], point["prediction_time"])
        if identity in seen_keys:
            raise ValueError("duplicate positive key")
        seen_keys.add(identity)
        key = (point["split"], point["target_episode_id"])
        if not key[1]:
            raise ValueError("positive episode ID missing")
        status = point["discrete_data_status"]
        reasons = set(point["discrete_data_reasons"])
        if (
            status not in {"eligible", "unknown", "excluded"}
            or (status == "eligible" and reasons)
            or (status != "eligible" and not reasons)
        ):
            raise ValueError("discrete status/reasons are inconsistent")
        if point["availability_status"] != "unknown":
            reasons.add("r3_availability_status_not_unknown")
        expected = status == "eligible" and point["availability_status"] == "unknown"
        if point["old_admitted"] != expected:
            raise ValueError("saved R3 admission differs from status rule")
        item = grouped.setdefault(
            key,
            {
                "split": key[0],
                "target_episode_id": key[1],
                "channel_id": point["channel_id"],
                "sensor_type": point["sensor_type"],
                "hours": 0,
                "admitted": 0,
                "first": point["prediction_time"],
                "last": point["prediction_time"],
                "reason_counts": Counter(),
                "baseline_counts": Counter(),
                "cases": set(),
            },
        )
        if item["channel_id"] != point["channel_id"] or item["sensor_type"] != point["sensor_type"]:
            raise ValueError("episode channel/type is inconsistent")
        item["hours"] += 1
        item["admitted"] += int(expected)
        item["first"] = min(item["first"], point["prediction_time"])
        item["last"] = max(item["last"], point["prediction_time"])
        item["reason_counts"].update(reasons)
        item["baseline_counts"].update(set(point["baseline_reasons"]))
        item["cases"].add((status == "excluded", tuple(sorted(reasons))))
    episodes, summaries = [], {}
    for item in grouped.values():
        total, cases = item["hours"], item["cases"]
        smallest = min(len(reasons) for _, reasons in cases)
        record = {
            "split": item["split"],
            "target_episode_id": item["target_episode_id"],
            "channel_id": item["channel_id"],
            "sensor_type": item["sensor_type"],
            "positive_hours": total,
            "eligible_positive_hours": item["admitted"],
            "missed_at_old_admission": item["admitted"] == 0,
            "first_positive_hour": item["first"],
            "last_positive_hour": item["last"],
            "minimum_simultaneous_vetoes": smallest,
            "reasons_on_any_hour": sorted(item["reason_counts"]),
            "reasons_on_every_hour": sorted(
                r for r, n in item["reason_counts"].items() if n == total
            ),
            "baseline_reasons_on_any_hour": sorted(item["baseline_counts"]),
            "baseline_reasons_on_every_hour": sorted(
                r for r, n in item["baseline_counts"].items() if n == total
            ),
            "minimal_reason_sets": [
                list(r) for r in sorted({r for _, r in cases if len(r) == smallest})
            ],
        }
        episodes.append(record)
        summary = summaries.setdefault(
            item["split"],
            {
                "positive_hours": 0,
                "eligible_positive_hours": 0,
                "full_positive_episodes": 0,
                "available_positive_episodes": 0,
                "missed_episodes": 0,
                "by_type": defaultdict(Counter),
                "missed_reason_any_hour": Counter(),
                "missed_reason_every_hour": Counter(),
                "missed_baseline_reason_every_hour": Counter(),
                "missed_minimum_veto_histogram": Counter(),
                "counterfactual_episode_counts": Counter(),
            },
        )
        summary["positive_hours"] += total
        summary["eligible_positive_hours"] += item["admitted"]
        summary["full_positive_episodes"] += 1
        summary["available_positive_episodes"] += int(item["admitted"] > 0)
        missed = record["missed_at_old_admission"]
        summary["missed_episodes"] += int(missed)
        kind = summary["by_type"][item["sensor_type"] or "<unknown>"]
        kind["full_positive_episodes"] += 1
        kind["available_positive_episodes"] += int(not missed)
        kind["missed_episodes"] += int(missed)
        if missed:
            summary["missed_reason_any_hour"].update(record["reasons_on_any_hour"])
            summary["missed_reason_every_hour"].update(record["reasons_on_every_hour"])
            summary["missed_baseline_reason_every_hour"].update(
                record["baseline_reasons_on_every_hour"]
            )
            summary["missed_minimum_veto_histogram"][smallest] += 1
        for name, removed in SCENARIOS.items():
            reachable = any(
                not excluded and not (set(reasons) - removed) for excluded, reasons in cases
            )
            summary["counterfactual_episode_counts"][name] += int(reachable)
    for summary in summaries.values():
        if (
            summary["counterfactual_episode_counts"]["unchanged"]
            != summary["available_positive_episodes"]
        ):
            raise ValueError("unchanged counterfactual differs from saved admission")
        full = summary["full_positive_episodes"]
        summary["maximum_full_recall_at_old_admission"] = (
            summary["available_positive_episodes"] / full
        )
        summary["minimum_matched_episodes_for_recall_strictly_above_half"] = full // 2 + 1
    return sorted(episodes, key=lambda r: (r["split"], r["target_episode_id"])), summaries


def select_positive_labels(database, files: list[str]) -> None:
    database.execute(
        "CREATE TEMP TABLE labels AS SELECT channel_id,prediction_time,sensor_type,"
        "target_episode_id,split FROM read_parquet(?,hive_partitioning=false) "
        "WHERE target=1 AND label_status='positive' AND split_status='assigned' "
        "AND split IN ('train','validation')",
        [files],
    )


def build(*, a3_dir: Path, b3_dir: Path, admission_dir: Path, output_dir: Path) -> dict:
    started = time.perf_counter()
    pending = output_dir.with_name(output_dir.name + ".inprogress")
    if output_dir.exists() or pending.exists():
        raise FileExistsError(output_dir)
    contract = read_json(Path("ml/quality_improvement_feature_contract_v1.json"))
    base = read_json(Path("ml/r3_discrete_feature_allowlist_v1.json"))
    sources = (
        (a3_dir, contract["source_a3_manifest_sha256"]),
        (b3_dir, base["source_b3_manifest_sha256"]),
        (admission_dir, contract["source_admission_manifest_sha256"]),
    )
    if any(sha256(root / "manifest.json") != pin for root, pin in sources):
        raise ValueError("accepted source manifest differs")
    manifests = [read_json(root / "manifest.json") for root, _ in sources]
    full = read_json(b3_dir / "report.json")
    if sha256(b3_dir / "report.json") != manifests[1]["report_sha256"]:
        raise ValueError("full B3 report hash differs")
    chunks = [{c["month"]: c for c in m["chunks"]} for m in manifests]
    a_files, b_files, i_files, proofs = [], [], [], []
    for month in expected_months():
        a, b, index = [c[month] for c in chunks]
        am = safe_path(a3_dir, a["manifest_file"], month=month)
        bm = safe_path(b3_dir, b["manifest_file"], month=month)
        im = safe_path(admission_dir, index["manifest_file"], month=month)
        for path, expected in (
            (am, a["manifest_sha256"]),
            (bm, b["manifest_sha256"]),
            (im, index["manifest_sha256"]),
        ):
            if sha256(path) != expected:
                raise ValueError("source month manifest differs")
        metadata, admission = read_json(bm), read_json(im)
        if metadata["source_a3_month_manifest_sha256"] != a["manifest_sha256"]:
            raise ValueError("B3/A3 month lineage differs")
        ap = safe_path(a3_dir, a["row_status_file"], month=month)
        bp = bm.parent / "registered_forecast_labels.parquet"
        ip = im.parent / "conditional_discrete_keys.parquet"
        for path, expected in (
            (ap, a["row_status_sha256"]),
            (bp, metadata["files"][bp.name]["sha256"]),
            (ip, admission["candidate_sha256"]),
        ):
            actual = sha256(path)
            if actual != expected:
                raise ValueError("source Parquet hash differs")
            proofs.append({"month": month, "file": path.name, "sha256": actual})
        a_files.append(str(ap))
        b_files.append(str(bp))
        i_files.append(str(ip))
    pending.mkdir(parents=True)
    with duckdb.connect(":memory:") as database:
        database.execute("SET threads=2")
        database.execute("SET memory_limit='1GB'")
        select_positive_labels(database, b_files)
        database.execute(
            "CREATE TEMP TABLE old_positive_keys AS SELECT channel_id,prediction_time "
            "FROM read_parquet(?,hive_partitioning=false) WHERE target=1",
            [i_files],
        )
        database.execute(
            "CREATE TEMP TABLE positive AS SELECT l.*,s.availability_status,"
            "s.discrete_data_status,s.discrete_data_reasons,s.baseline_reasons,"
            "k.channel_id IS NOT NULL AS old_admitted FROM labels l "
            "JOIN read_parquet(?,hive_partitioning=false) s USING(channel_id,prediction_time) "
            "LEFT JOIN old_positive_keys k USING(channel_id,prediction_time)",
            [a_files],
        )
        count, unique = database.execute(
            "SELECT COUNT(*),COUNT(DISTINCT (channel_id,prediction_time)) FROM positive"
        ).fetchone()
        if (
            count != unique
            or count != database.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
        ):
            raise ValueError("positive status join duplicates or loses keys")
        if database.execute(
            "SELECT COUNT(*) FROM old_positive_keys k ANTI JOIN labels l USING(channel_id,prediction_time)"
        ).fetchone()[0]:
            raise ValueError("old admitted positive key missing from full labels")
        points = (
            database.execute(
                "SELECT * FROM positive ORDER BY split,target_episode_id,prediction_time"
            )
            .to_arrow_table()
            .to_pylist()
        )
        episodes, summaries = summarize_points(points)
        for split, summary in summaries.items():
            if (
                summary["full_positive_episodes"]
                != full["assigned_unique_positive_episodes_by_split"][split]
            ):
                raise ValueError("full episode denominator differs")
            actual_types = {k: v["full_positive_episodes"] for k, v in summary["by_type"].items()}
            if actual_types != full["assigned_unique_positive_episodes_by_split_and_type"][split]:
                raise ValueError("full episode type counts differ")
        hour_file = pending / "positive_hour_diagnostics.parquet"
        database.execute(
            "COPY (SELECT * FROM positive ORDER BY split,target_episode_id,prediction_time) TO "
            + quoted(str(hour_file))
            + " (FORMAT PARQUET,COMPRESSION ZSTD)"
        )
    episode_file = pending / "episode_blockers.parquet"
    pq.write_table(pa.Table.from_pylist(episodes), episode_file, compression="zstd")
    report = {
        "schema_version": VERSION,
        "status": "old_admission_diagnostics_complete_new_admission_not_approved",
        "by_split": summaries,
        "source_manifests": {root.name: pin for root, pin in sources},
        "source_month_files": proofs,
        "source_code_lf_sha256": frozen_rule_sha256(Path(__file__)),
        "diagnostic_scenarios_removed_vetoes": {k: sorted(v) for k, v in SCENARIOS.items()},
        "positive_hour_count": len(points),
        "episode_count": len(episodes),
        "elapsed_seconds": time.perf_counter() - started,
        "test_rows_used": False,
        "new_admission_approved": False,
        "new_training_population_created": False,
        "target_used_for_retrospective_diagnostics_only": True,
        "new_model_metrics_computed": False,
        "limitations": [
            "Reason counts overlap; every-hour refers only to assigned positive A3 grid points, not the complete physical 24-hour interval.",
            "Removing a veto is a mechanical counterfactual, not validation of new features, labels, or model quality.",
            "Excluded, quality-exclusion and other unremoved vetoes stay in force in diagnostic scenarios.",
            "Positive-only diagnostics and future episode IDs must never be used as causal admission or model features.",
            "Source A3 statuses precede QA correction; Q1 proved its overlay does not change old admitted discrete keys.",
        ],
    }
    report_file = pending / "report.json"
    report_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": VERSION,
        "status": report["status"],
        "diagnostics_not_model_inputs": True,
        "files": {
            p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in (hour_file, episode_file, report_file)
        },
    }
    (pending / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pending.rename(output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("a3-dir", "b3-dir", "admission-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = build(
        a3_dir=args.a3_dir,
        b3_dir=args.b3_dir,
        admission_dir=args.admission_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(report["by_split"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
