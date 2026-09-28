"""Independently check round-7 warning subsets, labels and causal veto feature."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from analysis.prepare_ml_experiment import sha256


ROOT = Path("output/ml-experiment-round7")
PREVIOUS = Path("output/ml-experiment-round6")
Q2 = Path("output/q2-a-full-sparse-20260926-v5")
Q3 = Path("output/q3-a-coverage-reentry-20260927-v4")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_year(year: int) -> dict:
    suffix = "2024-unknown21-v1" if year == 2024 else "2025-unknown21-v2"
    candidate = ROOT / f"fullstream-{suffix}"
    before = PREVIOUS / f"fullstream-{year}-v2"
    report = read(candidate / "report.json")
    prior = read(before / "report.json")
    if (report["smoke_max_unknown_state_count_168h"] != 21
            or report["unknown_state_gated_decisions"] <= 0
            or report["year"] != year
            or report["standard_specialist_score_sha256"] !=
            prior["standard_specialist_score_sha256"]):
        raise AssertionError("candidate provenance differs")
    old = pd.read_parquet(before / "candidate_warnings.parquet")
    new = pd.read_parquet(candidate / "candidate_warnings.parquet")
    key = ["channel_id", "prediction_time"]
    old_keys = set(map(tuple, old[key].itertuples(index=False, name=None)))
    new_keys = set(map(tuple, new[key].itertuples(index=False, name=None)))
    old_hits = set(old.loc[old.outcome.eq("matched_known_episode"), "target_episode_id"])
    new_hits = set(new.loc[new.outcome.eq("matched_known_episode"), "target_episode_id"])
    old_removed = old.loc[[tuple(key) in old_keys - new_keys
                           for key in old[key].itertuples(index=False, name=None)]]
    new_added = new.loc[[tuple(item) in new_keys - old_keys
                         for item in new[key].itertuples(index=False, name=None)]]
    if (len(old_keys) != len(old) or len(new_keys) != len(new)
            or not old_hits <= new_hits or len(new) >= len(old)
            or len(new) != report["candidate"]["warnings"]
            or len(new_hits) != report["candidate"]["matched_known_episodes"]
            or len(new_hits) != len(old_hits)
            or old_removed.outcome.eq("matched_known_episode").any()):
        raise AssertionError("warning or known-episode metrics differ")
    q2_manifest = read(Q2 / "manifest.json")
    feature_paths = []
    for month in range(1, 13):
        folder = f"year={year}/month={month:02d}"
        q2_row = next(row for row in q2_manifest["months"]
                      if row["month"] == f"{year}-{month:02d}")
        q3_row = read(Q3 / folder / "manifest.json")
        for root, filename, expected in (
            (Q2, "model_features.parquet", q2_row["files"]["model_features.parquet"]["sha256"]),
            (Q3, "new_model_features.parquet", q3_row["files"]["new_model_features.parquet"]["sha256"]),
        ):
            path = root / folder / filename
            if sha256(path) != expected:
                raise AssertionError("feature provenance differs")
            feature_paths.append(str(path))
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        rows, bad_feature, suppressed_standard, altered_reset = db.execute('''
            WITH f AS (SELECT channel_id,prediction_time,sensor_type,
                       unknown_state_count_168h
                       FROM read_parquet(?,hive_partitioning=false))
            SELECT COUNT(*),
              COUNT(*) FILTER (WHERE w.sensor_type='Датчик дыма'
                AND (f.channel_id IS NULL OR f.sensor_type!=w.sensor_type)),
              COUNT(*) FILTER (WHERE w.reason='standard_24h_warning'
                AND w.sensor_type='Датчик дыма' AND f.unknown_state_count_168h>21),
              COUNT(*) FILTER (WHERE w.reason='recovered_episode_reset'
                AND w.sensor_type='Датчик дыма' AND f.unknown_state_count_168h>21)
            FROM read_parquet(?) w LEFT JOIN f USING(channel_id,prediction_time)''',
            [feature_paths, str(candidate / "candidate_warnings.parquet")]).fetchone()
    if rows != len(new) or bad_feature or suppressed_standard:
        raise AssertionError("smoke veto feature or standard-warning parity differs")
    return {
        "year": year,
        "old_warnings": len(old),
        "new_warnings": len(new),
        "removed_warning_keys": len(old_keys - new_keys),
        "new_warning_keys_after_feedback": len(new_keys - old_keys),
        "removed_warning_outcomes": old_removed.outcome.value_counts().to_dict(),
        "added_warning_outcomes": new_added.outcome.value_counts().to_dict(),
        "old_known_episodes": len(old_hits),
        "new_known_episodes": len(new_hits),
        "lost_known_episodes": len(old_hits - new_hits),
        "precision_lower_bound": report["candidate"]["precision_lower_bound"],
        "recall_lower_bound": report["candidate"]["full_recall_lower_bound"],
        "reset_warnings_above_limit_intentionally_retained": altered_reset,
        "test_2026_read": False,
        "data_2021_read": False,
    }


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    checks = [audit_year(year) for year in (2024, 2025)]
    result = {"status": "independent_round7_warning_and_feature_parity_passed",
              "checks": checks, "test_2026_read": False, "data_2021_read": False}
    output.mkdir(parents=True)
    (output / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "independent-audit-v1")
    run(parser.parse_args().output)
