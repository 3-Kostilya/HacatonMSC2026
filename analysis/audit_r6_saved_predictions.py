"""Independently replay the frozen arithmetic for every saved R6 test row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from analysis.train_r4_discrete_baselines import read_json, sha256
from analysis.r6_provenance import frozen_rule_sha256
from ml.forecast.r6_rule import TERMS


def run(*, freeze_path: Path, a3_dir: Path, index_dir: Path,
        test_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    freeze = read_json(freeze_path)
    manifest = read_json(test_dir / "manifest.json")
    source = read_json(a3_dir / "manifest.json")
    index = read_json(index_dir / "manifest.json")
    if (manifest["source_freeze_sha256"] != frozen_rule_sha256(freeze_path)
            or sha256(a3_dir / "manifest.json")
            != freeze["source_a3_manifest_sha256"]
            or sha256(index_dir / "manifest.json")
            != freeze["source_r3_admission_manifest_sha256"]
            or len(manifest["monthly_predictions"]) != 6):
        raise ValueError("frozen prediction lineage differs")
    source_by_month = {item["month"]: item for item in source["chunks"]}
    index_by_month = {item["month"]: item for item in index["chunks"]}
    formula = " + ".join(
        f"{weight} * COALESCE(TRY_CAST(f.\"{name}\" AS DOUBLE),0.0)"
        for name, weight in TERMS.items()
    )
    monthly = []
    with duckdb.connect(":memory:") as database:
        database.execute("SET threads=2")
        for item in manifest["monthly_predictions"]:
            month = item["month"]
            if month not in source_by_month or month not in index_by_month:
                raise ValueError(f"month absent in source: {month}")
            prediction = test_dir / item["file"]
            if sha256(prediction) != item["sha256"]:
                raise ValueError(f"prediction hash differs: {month}")
            candidate = ((index_dir / index_by_month[month]["manifest_file"]).parent
                         / "conditional_discrete_keys.parquet")
            features = a3_dir / source_by_month[month]["features_file"]
            result = database.execute(
                f"""WITH joined AS (
                    SELECT p.channel_id,p.prediction_time,p.sensor_type,
                           p.target,p.target_episode_id,p.label_available_at,
                           p.rule_score,p.above_frozen_threshold,
                           c.channel_id AS source_channel_id,
                           c.sensor_type AS candidate_type,c.target AS candidate_target,
                           c.target_episode_id AS candidate_episode,
                           c.label_available_at AS candidate_onset,
                           f.channel_id AS feature_channel_id,
                           f.sensor_type AS feature_type,
                           {formula} AS replay_score
                    FROM read_parquet(?) p
                    FULL OUTER JOIN (
                        SELECT * FROM read_parquet(?) WHERE split='test'
                    ) c
                      ON p.channel_id=c.channel_id
                     AND p.prediction_time=c.prediction_time
                    LEFT JOIN read_parquet(?) f
                      ON p.channel_id=f.channel_id
                     AND p.prediction_time=f.prediction_time
                ) SELECT COUNT(*) AS rows,
                     COUNT(DISTINCT (channel_id,prediction_time)) AS distinct_keys,
                     COUNT(*) FILTER (WHERE channel_id IS NULL OR
                         source_channel_id IS NULL OR feature_channel_id IS NULL
                         OR sensor_type IS DISTINCT FROM candidate_type
                         OR sensor_type IS DISTINCT FROM feature_type
                         OR target IS DISTINCT FROM candidate_target
                         OR target_episode_id IS DISTINCT FROM candidate_episode
                         OR label_available_at IS DISTINCT FROM candidate_onset
                     ) AS lineage_mismatches,
                     COUNT(*) FILTER (WHERE rule_score IS NULL OR
                         ABS(rule_score-replay_score)>1e-12) AS score_mismatches,
                     COUNT(*) FILTER (WHERE above_frozen_threshold IS NULL OR
                         above_frozen_threshold IS DISTINCT FROM
                         (replay_score>=?)) AS alert_mismatches,
                     MAX(ABS(rule_score-replay_score)) AS max_score_difference
                FROM joined""",
                [str(prediction), str(candidate), str(features),
                 freeze["frozen_threshold"]],
            ).fetchone()
            row = dict(zip(("rows", "distinct_keys", "lineage_mismatches",
                            "score_mismatches", "alert_mismatches",
                            "max_score_difference"), result))
            row["month"] = month
            if (row["rows"] != item["rows"] or row["distinct_keys"] != item["rows"]
                    or row["lineage_mismatches"] or row["score_mismatches"]
                    or row["alert_mismatches"]):
                raise ValueError(f"saved prediction replay differs: {month}: {row}")
            monthly.append(row)
    report = {
        "schema_version": "r6-b-saved-prediction-replay-v1",
        "status": "passed",
        "source_freeze_sha256": frozen_rule_sha256(freeze_path),
        "source_test_manifest_sha256": sha256(test_dir / "manifest.json"),
        "months": monthly,
        "rows_checked": sum(item["rows"] for item in monthly),
        "maximum_score_difference": max(item["max_score_difference"]
                                        for item in monthly),
        "limitation": "Replays saved A3 features, not raw M1 history.",
    }
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(freeze_path=args.freeze, a3_dir=args.a3_dir,
                         index_dir=args.index_dir, test_dir=args.test_dir,
                         output_dir=args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
