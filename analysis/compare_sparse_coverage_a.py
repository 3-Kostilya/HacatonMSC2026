"""Retrospective reconciliation of cached A3 scenarios with full-past Q2 admission."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import duckdb

from analysis.build_quality_improvement_a import expected_months, safe_path
from analysis.build_sparse_population_a import write_json
from analysis.train_r4_discrete_baselines import read_json, sha256


OLD_MANIFEST_SHA256 = "69a8fab59ea93f3ef175468ba6038d13c589c749c7f935622e4fc32274d038f2"


def summarize(rows, old_episodes):
    groups = Counter((split, old, new) for split, _, _, old, new in rows)
    details = []
    for split in ("train", "validation"):
        gained = {episode for s, episode, _, old, new in rows if s == split and new and not old}
        lost = {episode for s, episode, _, old, new in rows if s == split and old and not new}
        old_reasons = Counter(
            reason for item in old_episodes if item["split"] == split
            and item["target_episode_id"] in gained for reason in item["reasons_on_every_hour"]
        )
        details.append({
            "split": split,
            "cached_mechanical_available": sum(n for (s, old, _), n in groups.items() if s == split and old),
            "full_past_available": sum(n for (s, _, new), n in groups.items() if s == split and new),
            "gained_episodes": len(gained), "lost_episodes": len(lost),
            "gained_by_type": dict(Counter(kind for s, e, kind, _, _ in rows if s == split and e in gained)),
            "gained_cached_reasons_on_every_hour": dict(old_reasons),
        })
    return {"groups": [{"split": s, "cached_mechanical_available": old,
                        "full_past_available": new, "episodes": count}
                       for (s, old, new), count in sorted(groups.items())], "by_split": details}


def compare(package, old_audit, output):
    if output.exists():
        raise FileExistsError(output)
    if sha256(old_audit / "manifest.json") != OLD_MANIFEST_SHA256:
        raise ValueError("cached diagnostic source differs")
    old_manifest = read_json(old_audit / "manifest.json")
    for name, info in old_manifest["files"].items():
        if sha256(safe_path(old_audit, name)) != info["sha256"]:
            raise ValueError("cached diagnostic member differs")
    manifest = read_json(package / "manifest.json")
    if [m["month"] for m in manifest["months"]] != expected_months():
        raise ValueError("coverage comparison needs exactly all 72 train/validation months")
    files = []
    for month in manifest["months"]:
        folder = safe_path(package, month["manifest_file"]).parent
        file = folder / "admission.parquet"
        if sha256(file) != month["files"][file.name]["sha256"]:
            raise ValueError("candidate admission differs")
        files.append(str(file))
    positive = old_audit / "positive_hour_diagnostics.parquet"
    with duckdb.connect() as db:
        db.execute("SET memory_limit='2GB'")
        db.execute("SET threads=2")
        count, unique = db.execute(
            "SELECT COUNT(*),COUNT(DISTINCT (p.channel_id,p.prediction_time)) "
            "FROM read_parquet(?,hive_partitioning=false) p JOIN "
            "read_parquet(?,hive_partitioning=false) d USING(channel_id,prediction_time)",
            [str(positive), files],
        ).fetchone()
        if (count, unique) != (95217, 95217):
            raise ValueError("cached/candidate positive keys are incomplete or duplicated")
        rows = db.execute(
            "SELECT p.split,p.target_episode_id,p.sensor_type,"
            "BOOL_OR(p.discrete_data_status<>'excluded' AND len(list_filter("
            "p.discrete_data_reasons,x->x NOT IN ('baseline_unusable','state_history_missing',"
            "'state_transitions_unavailable')))=0),BOOL_OR(d.admission_status='eligible') "
            "FROM read_parquet(?,hive_partitioning=false) p JOIN "
            "read_parquet(?,hive_partitioning=false) d USING(channel_id,prediction_time) GROUP BY ALL",
            [str(positive), files],
        ).fetchall()
        old_episodes = db.execute("SELECT * FROM read_parquet(?,hive_partitioning=false)",
                                  [str(old_audit / "episode_blockers.parquet")]).to_arrow_table().to_pylist()
    result = summarize(rows, old_episodes)
    report = read_json(package / "report.json")
    for split in result["by_split"]:
        if split["full_past_available"] != report["label_audit"]["by_split"][split["split"]]["candidate_available_episodes"]:
            raise ValueError("independent direct join differs from candidate label audit")
    result.update({"status": "cached_vs_full_past_coverage_reconciled",
                   "candidate_manifest_sha256": sha256(package / "manifest.json"),
                   "cached_manifest_sha256": OLD_MANIFEST_SHA256,
                   "positive_keys_checked": count, "retrospective_diagnostics_not_model_inputs": True})
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package", "old-audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(compare(args.package, args.old_audit, args.output))


if __name__ == "__main__":
    main()
