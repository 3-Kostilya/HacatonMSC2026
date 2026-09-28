"""Independent parity and no-known-recall-loss audit for precision experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from analysis.prepare_ml_experiment import sha256


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(baseline: Path, candidate: Path, linear: Path, tree: Path) -> dict:
    base_report, report = read(baseline / "report.json"), read(candidate / "report.json")
    year = report["year"]
    if (year not in (2024, 2025) or base_report.get("year", 2025) != year
            or report["model_name"] != "linear"
            or report["full_context_scores_sha256"] != sha256(linear)
            or report["veto_tree_score_sha256"] != sha256(tree)):
        raise AssertionError("experiment provenance differs")
    old = pd.read_parquet(baseline / "candidate_warnings.parquet")
    new = pd.read_parquet(candidate / "candidate_warnings.parquet")
    old_hits = set(old.loc[old.outcome.eq("matched_known_episode"), "target_episode_id"])
    new_hits = set(new.loc[new.outcome.eq("matched_known_episode"), "target_episode_id"])
    if not old_hits <= new_hits:
        raise AssertionError(f"{year}: previously found known episodes were lost")
    metric = report["candidate"]
    full = 1204 if year == 2024 else 2142
    if (len(new) != metric["warnings"] or len(new_hits) != metric["matched_known_episodes"]
            or metric["full_recall_lower_bound"] != len(new_hits) / full
            or metric["precision_lower_bound"] != len(new_hits) / len(new)
            or new.duplicated(["channel_id", "prediction_time"]).any()
            or len(new) != sum((new.outcome == outcome).sum() for outcome in (
                "matched_known_episode", "known_no_target", "unknown_target"))):
        raise AssertionError(f"{year}: warning metric parity differs")
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        published = read(Path("output/q3-a-coverage-reentry-20260927-v4/report.json"))
        labels = []
        for month in range(1, 13):
            path = Path(f"output/r3-b-full-months-20260925-v2/"
                        f"year={year}/month={month:02d}/registered_forecast_labels.parquet")
            expected = next(item["sha256"] for item in published["source_b3_labels"]
                            if item["month"] == f"{year}-{month:02d}")
            if sha256(path) != expected:
                raise AssertionError("B3 label source changed")
            labels.append(str(path))
        count, bad_linear, bad_tree, bad_smoke_age, bad_reset, bad_label = db.execute('''
            SELECT COUNT(*),
            COUNT(*) FILTER(WHERE l.channel_id IS NULL OR l.sensor_type!=w.sensor_type
                OR l.score<? OR (w.sensor_type='Состояние фазы' AND l.score<?)),
            COUNT(*) FILTER(WHERE t.channel_id IS NULL OR t.sensor_type!=w.sensor_type
                OR (w.sensor_type!='Газовый датчик' AND t.score<?)),
            COUNT(*) FILTER(WHERE w.sensor_type='Датчик дыма' AND
                (w.last_explicit_normal_at IS NULL OR
                 w.prediction_time-w.last_explicit_normal_at>INTERVAL '5 hours')),
            COUNT(*) FILTER(WHERE w.reason='recovered_episode_reset' AND NOT(
                w.previous_warning_at<w.observed_onset_at
                AND w.observed_onset_at<w.observed_recovery_at
                AND w.observed_recovery_at<=w.prediction_time)),
            COUNT(*) FILTER(WHERE b.channel_id IS NULL OR b.sensor_type!=w.sensor_type
                OR b.target IS DISTINCT FROM w.target
                OR b.target_episode_id IS DISTINCT FROM w.target_episode_id
                OR b.label_available_at IS DISTINCT FROM w.label_available_at)
            FROM read_parquet(?) w LEFT JOIN read_parquet(?) l
            USING(channel_id,prediction_time)
            LEFT JOIN read_parquet(?) t USING(channel_id,prediction_time)
            LEFT JOIN read_parquet(?,hive_partitioning=false) b
            USING(channel_id,prediction_time)''', [
                report["threshold"], report["type_thresholds"]["Состояние фазы"],
                report["veto_tree_threshold"],
                str(candidate / "candidate_warnings.parquet"), str(linear), str(tree), labels
            ]).fetchone()
    if count != len(new) or any((bad_linear, bad_tree, bad_smoke_age, bad_reset, bad_label)):
        raise AssertionError(f"{year}: causal precision guard parity differs")
    return {"year": year, "old_warnings": len(old), "new_warnings": len(new),
            "old_known_episodes": len(old_hits), "new_known_episodes": len(new_hits),
            "lost_known_episodes": len(old_hits - new_hits),
            "unknown_outcome_warnings": metric["unknown_outcome_warnings"],
            "precision_lower_bound": metric["precision_lower_bound"],
            "full_recall_lower_bound": metric["full_recall_lower_bound"]}


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    root = Path("output/ml-experiment-round5")
    old = Path("output/ml-experiment-round4/q3-full-context-linear-th99544")
    tree_old = Path("output/ml-experiment-round4/q3-full-context-tree-th992-v2")
    pairs = [
        (root / "q3-full-context-linear-2024-base",
         root / "q3-full-context-linear-2024-phase-age5-tree95-gas",
         root / "q3-full-context-linear-2024-base/full_context_scores.parquet",
         root / "q3-full-context-tree-2024-base/full_context_scores.parquet"),
        (old, root / "q3-full-context-linear-2025-phase-age5-tree95-gas",
         old / "full_context_scores.parquet", tree_old / "full_context_scores.parquet"),
    ]
    checks = [audit(*pair) for pair in pairs]
    result = {"status": "independent_precision_guard_and_known_recall_parity_passed",
              "checks": checks, "test_2026_read": False, "data_2021_read": False}
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("output/ml-experiment-round5/independent-audit"))
    run(parser.parse_args().output)
