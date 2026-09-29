"""A-side independent checks of B's R5 validation replay and R4 baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from catboost import CatBoostClassifier
import duckdb
import numpy as np
import pandas as pd

from analysis.run_r5_b_ablation import model_input, variants
from analysis.train_r4_discrete_baselines import read_json, sha256
from ml.forecast.alert_eval import evaluate_alerts


ROOT = Path(__file__).resolve().parents[1]
KEYS = ["channel_id", "prediction_time"]
BASE = [*KEYS, "sensor_type", "target", "target_episode_id", "label_available_at"]
TOLERANCE = 1e-12


def _selected(curve: list[dict], limit: int) -> dict:
    eligible = [item for item in curve if item["unmatched_warnings"] <= limit]
    if not eligible:
        raise ValueError(f"no curve point at {limit} unmatched warnings")
    return max(eligible, key=lambda item: (
        item["matched_episodes"], -item["unmatched_warnings"],
        item["episode_precision"], item["threshold"],
    ))


def _replay_sample(
    month: str, scored: pd.DataFrame, *, a3_dir: Path, r5_a_dir: Path,
    a3_chunks: dict, r5_a_chunks: dict, models: dict,
    allowlist: list[str], choices: dict,
) -> tuple[int, float]:
    sample = pd.concat([
        scored.head(2),
        scored.nlargest(2, "score_clusters_transitions"),
        scored.loc[scored.target.eq(1)].head(2),
    ]).drop_duplicates(KEYS).reset_index(drop=True)
    sample_keys = sample[KEYS].copy()
    sample_keys["_order"] = np.arange(len(sample_keys))
    source = a3_dir / a3_chunks[month]["features_file"]
    additions = r5_a_dir / r5_a_chunks[month]["features_file"]
    numeric = [name for name in allowlist if name != "sensor_type"]
    extra = choices["full"]
    projection = ", ".join([f'f."{name}"' for name in numeric] +
                           [f'r."{name}"' for name in extra])
    with duckdb.connect(":memory:") as database:
        database.register("sample_keys", sample_keys)
        joined = database.execute(
            f"""SELECT k.channel_id, k.prediction_time, f.sensor_type,
                       r.sensor_type AS r5_sensor_type, {projection}
                FROM sample_keys AS k
                JOIN read_parquet(?) AS f USING (channel_id, prediction_time)
                JOIN read_parquet(?) AS r USING (channel_id, prediction_time)
                ORDER BY k._order""",
            [str(source), str(additions)],
        ).fetch_df()
    if (len(joined) != len(sample)
            or not joined[KEYS].equals(sample[KEYS])
            or not joined.sensor_type.eq(joined.r5_sensor_type).all()):
        raise ValueError(f"sample replay keys/type differ in {month}")
    joined = joined.drop(columns="r5_sensor_type")
    max_error = 0.0
    for name, fields in choices.items():
        actual = models[name].predict_proba(
            model_input(joined, allowlist + fields))[:, 1]
        expected = sample[f"score_{name}"].to_numpy(dtype=np.float64)
        error = float(np.max(np.abs(actual - expected)))
        if error > TOLERANCE:
            raise ValueError(f"batch/replay score differs: {month} {name} {error}")
        max_error = max(max_error, error)
    return len(sample), max_error


def audit(*, replay_dir: Path, r4_audit_dir: Path, r4_model_dir: Path,
          a3_dir: Path, r5_a_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    manifest = read_json(replay_dir / "manifest.json")
    report = read_json(replay_dir / "report.json")
    decision = read_json(ROOT / "ml" / "r5_b_ablation_decision_v1.json")
    r4_manifest = read_json(r4_audit_dir / "manifest.json")
    r5_a_manifest = read_json(r5_a_dir / "manifest.json")
    a3_manifest = read_json(a3_dir / "manifest.json")
    contract = read_json(ROOT / "ml" / "r5_a_feature_contract_v1.json")
    allowlist = read_json(ROOT / "ml" / "r3_discrete_feature_allowlist_v1.json")["feature_names"]
    choices = variants(contract)
    if (manifest["schema_version"] != "r5-b-fixed-population-ablation-v1"
            or manifest["report_sha256"] != sha256(replay_dir / "report.json")
            or manifest["source_r5_manifest_sha256"] != sha256(r5_a_dir / "manifest.json")
            or report["source_r5_manifest_sha256"] != decision["source_r5_a_full_manifest_sha256"]
            or len(manifest["monthly_predictions"]) != 12):
        raise ValueError("R5 B lineage or monthly count differs")
    r4_chunks = {item["month"]: item for item in r4_manifest["files"]
                 if item["file"].startswith("validation_")}
    r5_chunks = {item["month"]: item for item in manifest["monthly_predictions"]}
    a3_chunks = {item["month"]: item for item in a3_manifest["chunks"]}
    r5_a_chunks = {item["month"]: item for item in r5_a_manifest["chunks"]}
    if set(r5_chunks) != set(r4_chunks):
        raise ValueError("R4/R5 validation month sets differ")
    models = {}
    for name in choices:
        path = (r4_model_dir / "catboost.cbm" if name == "baseline" else
                replay_dir / f"catboost_{name}.cbm")
        if sha256(path) != manifest["models"][name]:
            raise ValueError(f"model file hash differs: {name}")
        model = CatBoostClassifier()
        model.load_model(str(path))
        models[name] = model

    rows = 0
    positives = 0
    max_r4_baseline_error = 0.0
    max_batch_error = 0.0
    batch_rows = 0
    score_columns = [f"score_{name}" for name in choices]
    full_parts = []
    for month in sorted(r5_chunks):
        r5_item, r4_item = r5_chunks[month], r4_chunks[month]
        r5_path, r4_path = replay_dir / r5_item["file"], r4_audit_dir / r4_item["file"]
        if sha256(r5_path) != r5_item["sha256"] or sha256(r4_path) != r4_item["sha256"]:
            raise ValueError(f"validation file hash differs in {month}")
        scored = pd.read_parquet(r5_path, columns=[*BASE, *score_columns])
        old = pd.read_parquet(r4_path, columns=[*BASE, "catboost_score"])
        scored = scored.sort_values(KEYS).reset_index(drop=True)
        old = old.sort_values(KEYS).reset_index(drop=True)
        if (len(scored) != r5_item["rows"] or len(old) != len(scored)
                or not scored[BASE].equals(old[BASE])):
            raise ValueError(f"R4/R5 validation keys, labels or episodes differ in {month}")
        error = float(np.max(np.abs(scored.score_baseline.to_numpy() -
                                    old.catboost_score.to_numpy())))
        if error > TOLERANCE:
            raise ValueError(f"R4 CatBoost baseline differs in {month}: {error}")
        max_r4_baseline_error = max(max_r4_baseline_error, error)
        replayed, replay_error = _replay_sample(
            month, scored, a3_dir=a3_dir, r5_a_dir=r5_a_dir,
            a3_chunks=a3_chunks, r5_a_chunks=r5_a_chunks, models=models,
            allowlist=allowlist, choices=choices)
        batch_rows += replayed
        max_batch_error = max(max_batch_error, replay_error)
        rows += len(scored)
        positives += int(scored.target.sum())
        full_parts.append(scored)
        print(f"checked {month}: {len(scored):,} rows, {replayed} replayed", flush=True)
    if (rows != decision["validation_rows"]
            or positives != decision["validation_positive_hours"]):
        raise ValueError("R5 validation size or labels differ from B decision")
    full = pd.concat(full_parts, ignore_index=True)
    curves = read_json(replay_dir / "curves.json")
    chosen = {}
    for name in choices:
        at_177 = _selected(curves[name], 177)
        selected = report["alerts_at_budget"][name]
        if selected not in curves[name]:
            raise ValueError(f"reported threshold absent from grid: {name}")
        selected_again, _ = evaluate_alerts(
            full.rename(columns={f"score_{name}": "catboost_score"}),
            "catboost_score", selected["threshold"],
            channel_days=report["validation_channel_days"])
        for key in ("matched_episodes", "unmatched_warnings"):
            if selected_again[key] != selected[key]:
                raise ValueError(f"R5 warning replay differs: {name} {key}")
        chosen[name] = {
            "at_exploratory_budget": [selected["matched_episodes"],
                                      selected["unmatched_warnings"]],
            "best_on_tested_grid_at_177": [at_177["matched_episodes"],
                                           at_177["unmatched_warnings"]],
        }
    if (chosen["clusters_transitions"]["at_exploratory_budget"] != [79, 234]
            or chosen["clusters_transitions"]["best_on_tested_grid_at_177"] != [58, 177]
            or chosen["full"]["best_on_tested_grid_at_177"] != [60, 172]
            or report["r4_rule_reference_at_budget"]["matched_episodes"] != 78
            or report["r4_rule_reference_at_budget"]["unmatched_warnings"] != 177):
        raise ValueError("R5/R4 comparison differs from B's decision")
    output_dir.mkdir(parents=True)
    result = {
        "schema_version": "r5-a-independent-b-review-v1",
        "status": "passed",
        "source_replay_manifest_sha256": sha256(replay_dir / "manifest.json"),
        "matches_b_published_manifest_sha256": (
            sha256(replay_dir / "manifest.json") ==
            decision["source_r5_b_ablation_manifest_sha256"]),
        "validation_rows": rows,
        "validation_positive_hours": positives,
        "max_r4_baseline_abs_score_difference": max_r4_baseline_error,
        "batch_replay_rows": batch_rows,
        "max_batch_replay_abs_score_difference": max_batch_error,
        "methods": chosen,
        "note": "The 177-warning comparison is the best point on B's tested threshold grid, not a proof over every real threshold.",
    }
    (output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--r4-audit-dir", type=Path, required=True)
    parser.add_argument("--r4-model-dir", type=Path, required=True)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--r5-a-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(**vars(args))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
