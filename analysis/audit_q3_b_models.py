"""Check saved Q3 model outputs against source keys/labels and independent replay."""

from __future__ import annotations

import argparse
from pathlib import Path

from catboost import CatBoostClassifier
import duckdb
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from analysis.run_q3_b_models import MODELS, POLICIES, lf_hash, quoted, write_json
from analysis.run_r5_b_ablation import model_input
from analysis.train_r4_discrete_baselines import read_json, sha256
from analysis.verify_q2_b_metrics_a import WARNING_COLUMNS, chronological_warnings


def audit(*, experiment: Path, q2_dir: Path, package: Path, b3_dir: Path,
          output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    report = read_json(experiment / "report.json")
    manifest = read_json(experiment / "manifest.json")
    pins = report["source_pins"]
    if (sha256(experiment / "report.json") != manifest["report_sha256"]
            or sha256(experiment / "decision_2023.json") != manifest["decision_sha256"]
            or sha256(experiment / "inputs.json") != manifest["inputs_sha256"]
            or sha256(q2_dir / "manifest.json") != pins["q2_manifest_sha256"]
            or sha256(package / "manifest.json") != pins["q3_manifest_sha256"]
            or sha256(b3_dir / "manifest.json") != pins["b3_manifest_sha256"]):
        raise ValueError("experiment/source manifest changed")
    if read_json(experiment / "inputs.json") != pins:
        raise ValueError("report and experiment input pins disagree")
    allowlist_path = q2_dir / "model_feature_allowlist.json"
    q2_manifest = read_json(q2_dir / "manifest.json")
    q3_manifest = read_json(package / "manifest.json")
    if (sha256(allowlist_path) != q2_manifest["files"][allowlist_path.name]["sha256"]
            or sha256(package / "positive_hour_audit.parquet")
            != q3_manifest["files"]["positive_hour_audit.parquet"]["sha256"]):
        raise ValueError("accepted allowlist or positive-hour diagnostic changed")
    allowlist = read_json(allowlist_path)
    if pins["feature_sets"] != {"base51": allowlist["base_feature_names"],
                                "full121": allowlist["feature_names"],
                                "linear121": allowlist["feature_names"]}:
        raise ValueError("model inputs differ from accepted feature allowlist")
    for path, expected in pins["code_and_protocol_lf_sha256"].items():
        code_path = Path(path)
        # __file__ was pinned at B's absolute path; verify the same file in A's checkout.
        if code_path.is_absolute():
            if code_path.name != "run_q3_b_models.py" or code_path.parent.name != "analysis":
                raise ValueError("unexpected absolute experiment code pin")
            code_path = Path("analysis/run_q3_b_models.py")
        if lf_hash(code_path) != expected:
            raise ValueError("experiment code/protocol changed during execution")
    bchunks = {row["month"]: row for row in read_json(b3_dir / "manifest.json")["chunks"]}
    stages = [(experiment / f"selection_2023/{policy}", report["comparison_2023"][policy], MODELS)
              for policy in POLICIES]
    chosen = report["decision_2023"]["candidate_for_2024"]
    if (report["confirmation_2024"]["policy"] != chosen["policy"]
            or list(report["confirmation_2024"]["models"]) != [chosen["model"]]
            or report["confirmation_2024"]["models"][chosen["model"]]["candidate"]["threshold"]
            != chosen["threshold"]):
        raise ValueError("2024 did not retain selected policy/model/numeric threshold")
    stages.append((experiment / "confirmation_2024", report["confirmation_2024"], (chosen["model"],)))
    verified = []
    full_truth = {}
    with duckdb.connect(":memory:") as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='3GB'")
        for folder, stage, families in stages:
            year, policy = stage["year"], stage["policy"]
            end = pd.Timestamp(f"{year + 1}-01-01")
            if year not in full_truth:
                label_paths = [str(b3_dir / Path(item["manifest_file"]).parent
                                   / "registered_forecast_labels.parquet")
                               for label, item in bchunks.items() if label.startswith(str(year))]
                db.read_parquet(label_paths, hive_partitioning=False).create_view("full_labels", replace=True)
                full_truth[year] = db.execute("""SELECT DISTINCT target_episode_id,sensor_type
                    FROM full_labels WHERE target=1 AND split_status='assigned'
                    AND prediction_time>=? AND prediction_time<? AND horizon_end<?""",
                    [pd.Timestamp(f"{year}-01-01"), end, end]).fetch_df()
            episodes = full_truth[year]
            if (episodes.target_episode_id.duplicated().any()
                    or len(episodes) != stage["all_assigned_episodes"]):
                raise ValueError("full episode denominator differs from unchanged B3 labels")
            sample = pq.ParquetFile(folder / "sampled_train.parquet").read(columns=[
                "prediction_time", "horizon_end", "target", "label_available_at",
            ]).to_pandas()
            if (sample.horizon_end.max() >= pd.Timestamp(f"{year}-01-01")
                    or not sample.target.isin([0, 1]).all()
                    or sample.prediction_time.dt.year.isin([2021, 2026]).any()):
                raise ValueError("training sample leaks a boundary or contains forbidden labels/years")
            models = {}
            for family in families:
                path = folder / (family + (".joblib" if family == "linear121" else ".cbm"))
                fit = read_json(folder / f"{family}_fit.json")
                if fit["model_sha256"] != sha256(path) or fit["sample_sha256"] != sha256(folder / "sampled_train.parquet"):
                    raise ValueError("model/training cache hash changed")
                if family == "linear121":
                    models[family] = joblib.load(path)
                else:
                    models[family] = CatBoostClassifier()
                    models[family].load_model(str(path))
            all_scores = []
            total_rows = sampled_rows = 0
            for month in range(1, 13):
                label = f"{year}-{month:02d}"
                score = folder / f"scores_{label}.parquet"
                metadata = read_json(score.with_suffix(".json"))
                if metadata["score_sha256"] != sha256(score):
                    raise ValueError("saved score hash changed")
                bfolder = b3_dir / Path(bchunks[label]["manifest_file"]).parent
                labels = bfolder / "registered_forecast_labels.parquet"
                base = q2_dir / f"year={year}/month={month:02d}/model_features.parquet"
                delta = package / f"year={year}/month={month:02d}"
                feature_sql = "SELECT f.*,true AS source_base FROM read_parquet(" + quoted(base) + ",hive_partitioning=false) f"
                if policy != "base":
                    feature_sql += (" UNION ALL SELECT f.*,false AS source_base FROM read_parquet("
                                    + quoted(delta / "new_model_features.parquet") + ",hive_partitioning=false) f JOIN read_parquet("
                                    + quoted(delta / "new_admission.parquet") + ",hive_partitioning=false) d USING(channel_id,prediction_time) "
                                    + f"WHERE d.{policy}_status='eligible'")
                db.execute("CREATE OR REPLACE TEMP VIEW actual AS SELECT * FROM read_parquet("
                           + quoted(score) + ",hive_partitioning=false)")
                db.execute("CREATE OR REPLACE TEMP VIEW expected AS SELECT f.channel_id,f.prediction_time,"
                           "f.sensor_type,f.source_base,l.target,l.target_episode_id,l.label_available_at "
                           "FROM (" + feature_sql + ") f JOIN read_parquet(" + quoted(labels)
                           + ",hive_partitioning=false) l USING(channel_id,prediction_time) "
                           + "WHERE l.target IN (0,1) AND l.split_status='assigned' "
                           + f"AND l.prediction_time>=TIMESTAMP '{year}-01-01' "
                           + f"AND l.horizon_end<TIMESTAMP '{end}' "
                           + "AND f.sensor_type IS NOT DISTINCT FROM l.sensor_type")
                columns = "channel_id,prediction_time,sensor_type,source_base,target,target_episode_id,label_available_at"
                for left, right in (("actual", "expected"), ("expected", "actual")):
                    if db.execute(f"SELECT count(*) FROM (SELECT {columns} FROM {left} "
                                  f"EXCEPT ALL SELECT {columns} FROM {right})").fetchone()[0]:
                        raise ValueError("whole saved score keys/types/labels differ from accepted population")
                count, unique = db.execute("SELECT count(*),count(DISTINCT(channel_id,prediction_time)) FROM actual").fetchone()
                if count != unique or count != metadata["rows"]:
                    raise ValueError("score key multiplicity/count differs")
                selected = db.execute("SELECT channel_id,prediction_time FROM actual "
                                      "ORDER BY hash(channel_id,prediction_time),channel_id,prediction_time LIMIT 16").fetch_df()
                db.register("sample_keys", selected)
                x = db.execute("SELECT f.* FROM (" + feature_sql + ") f SEMI JOIN sample_keys "
                               "USING(channel_id,prediction_time) ORDER BY f.channel_id,f.prediction_time").fetch_df()
                saved = db.execute("SELECT a.* FROM actual a SEMI JOIN sample_keys USING(channel_id,prediction_time) "
                                   "ORDER BY a.channel_id,a.prediction_time").fetch_df()
                for family, model in models.items():
                    names = pins["feature_sets"][family]
                    data = (x[names].assign(sensor_type=x.sensor_type.fillna("<unknown>"))
                            if family == "linear121" else model_input(x, names))
                    repeated = model.predict_proba(data)[:, 1].astype("float32")
                    if not np.array_equal(repeated, saved[f"score_{family}"].to_numpy()):
                        raise ValueError("sample scores do not reproduce from source features/model")
                sampled_rows += len(saved)
                total_rows += count
                all_scores.append(str(score))
            if total_rows != stage["evaluation"]["rows"]:
                raise ValueError("full evaluation count changed")
            db.read_parquet(all_scores, hive_partitioning=False).create_view("whole_scores", replace=True)
            available = db.execute("SELECT count(DISTINCT target_episode_id) FROM whole_scores WHERE target=1").fetchone()[0]
            if available != stage["available_episodes"]:
                raise ValueError("available episode count differs from saved scores")
            available_ids = set(db.execute("SELECT DISTINCT target_episode_id FROM whole_scores WHERE target=1").fetch_df().target_episode_id)
            error_analysis = {}
            for family in families:
                candidate = stage["models"][family]["candidate"]
                threshold = candidate["threshold"]
                selected = db.execute(f"""SELECT channel_id,prediction_time,sensor_type,target,
                    target_episode_id,label_available_at,score_{family} AS score
                    FROM whole_scores WHERE score_{family}>=?""", [threshold]).fetch_df()
                independent = chronological_warnings(selected)
                canonical = pq.ParquetFile(folder / f"alerts_{family}.parquet").read().to_pandas()
                sort = ["channel_id", "prediction_time"]
                columns = list(WARNING_COLUMNS)
                if not independent.sort_values(sort).reset_index(drop=True)[columns].equals(
                    canonical.sort_values(sort).reset_index(drop=True)[columns]
                ):
                    raise ValueError("independent jump-based warning replay differs")
                matches = int(independent.outcome.eq("matched_episode").sum())
                if matches != candidate["matched_episodes"] or len(independent) != candidate["emitted_warnings"]:
                    raise ValueError("warning/episode counts differ from reported model choice")
                precision = matches / len(independent) if len(independent) else 0.0
                recall = matches / len(episodes)
                if (precision != candidate["episode_precision"]
                        or recall != candidate["full_episode_recall"]):
                    raise ValueError("reported precision/full recall differs from independent replay")
                above_ids = set(selected.loc[selected.target.eq(1), "target_episode_id"])
                matched_ids = set(independent.loc[independent.outcome.eq("matched_episode"), "target_episode_id"])
                all_ids = set(episodes.target_episode_id)
                if not matched_ids <= above_ids <= available_ids <= all_ids:
                    raise ValueError("episode error classes do not form nested sets")
                error_analysis[family] = {
                    "unavailable_episodes": len(all_ids - available_ids),
                    "below_threshold_episodes": len(available_ids - above_ids),
                    "cooldown_suppressed_episodes": len(above_ids - matched_ids),
                    "matched_episodes": len(matched_ids),
                    "all_above_threshold_rows_suppressed": len(selected) - len(independent),
                    "positive_above_threshold_rows_suppressed": int(selected.target.sum() - independent.target.sum()),
                }
            verified.append({"policy": policy, "year": year, "full_keys_labels_verified": total_rows,
                             "full_episode_denominator_verified": len(episodes),
                             "available_episodes_verified": available,
                             "model_scores_recomputed": sampled_rows * len(families),
                             "episode_error_analysis": error_analysis,
                             "warning_replays_verified": len(families)})
    result = {"schema_version": "q3-b-saved-model-audit-v1", "experiment_manifest_sha256": sha256(experiment / "manifest.json"),
              "stages": verified, "source_key_or_label_mismatches": 0,
              "sample_score_mismatches": 0, "warning_replay_mismatches": 0,
              "2024_policy_model_and_threshold_unchanged": True,
              "model_training_repeated": False, "test_data_read": False}
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("experiment", "q2-dir", "package", "b3-dir", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    print(audit(**vars(parser.parse_args())), flush=True)


if __name__ == "__main__":
    main()
