"""Read-only independent champion audit after 2024 policies have been frozen.

Checks full score metadata against prepared data using an exact full outer
join, then replays selected warnings with the unchanged production evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from analysis.ml_experiment_eval import EVALUATION_VERSION
from analysis.ml_experiment_metric_audit import replay
from ml.forecast.alert_eval import evaluate_alerts


METADATA = ["sensor_type", "target", "target_episode_id", "label_available_at"]


def metadata_parity(db, source: Path, data: Path, expected_count: int, year: int) -> dict:
    rows, minimum, maximum = db.execute("""SELECT COUNT(*),MIN(year(prediction_time)),
        MAX(year(prediction_time)) FROM read_parquet(?)""", [str(source)]).fetchone()
    if rows != expected_count or minimum != year or maximum != year:
        raise AssertionError(f"score rows/year differ for {source}: {rows}/{minimum}/{maximum}")
    differences = " OR ".join(f'a."{name}" IS DISTINCT FROM b."{name}"' for name in METADATA)
    bad = db.execute(f"""SELECT COUNT(*) FROM read_parquet(?) a
        FULL OUTER JOIN read_parquet(?) b USING(channel_id,prediction_time)
        WHERE a.channel_id IS NULL OR b.channel_id IS NULL OR {differences}""",
                     [str(source), str(data)]).fetchone()[0]
    if bad:
        raise AssertionError(f"full score/data metadata differs in {bad} rows: {source}")
    return {"rows": rows, "year": year, "full_outer_metadata_mismatches": bad}


def champion_cases(directory: Path, family: str):
    report_path = directory / "report_canonical_v2.json"
    if not report_path.exists():
        report_path = directory / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if family in {"pooled", "history"}:
        name = report["selected_variant"]
        experiment = report["experiments"][name]
        choice = experiment["tune"]["selected"]
        return [(name, "tune", directory / "scores_tune.parquet", f"score_{name}",
                 choice["threshold"], choice),
                (name, "validation", directory / "scores_validation.parquet", f"score_{name}",
                 choice["threshold"], experiment["validation_frozen"])]
    if family == "linear":
        name = report["selected_variant"]
        choice = report["tune"][name]
        return [(name, "tune", directory / "tune_scores.parquet", f"score_{name}",
                 choice["threshold"], choice),
                (name, "validation", directory / "validation_scores.parquet", f"score_{name}",
                 choice["threshold"], report["validation"][name])]
    if family == "ensemble":
        name = report["selected_column"]
        choice = report["tune"][name]
        return [(name, "tune", directory / "tune_scores.parquet", name,
                 choice["threshold"], choice),
                (name, "validation", directory / "validation_scores.parquet", name,
                 choice["threshold"], report["validation"][name])]
    if family == "linear-refit":
        choice = report["selection"]
        name = choice["selected_variant"]
        return [(name, "validation", directory / "validation_scores.parquet", f"score_{name}",
                 choice["threshold"], report["validation"])]
    if family == "online-linear":
        choice = report["selection"]
        return [(choice["selected_variant"], "validation", directory / "validation_scores.parquet",
                 "score_online", choice["threshold"], report["validation"])]
    if family in {"specialists", "specialists-refit"}:
        # Margin files encode the already frozen model/type threshold policy.
        # Tune policies are verified against selected candidate curves below.
        selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
        if family == "specialists" and not selection.get("canonical_float32_threshold_replay"):
            raise AssertionError("specialists must first replay canonical2024 selection")
        return [(name, "validation", directory / "validation_scores.parquet", f"score_{name}",
                 0.0, expected) for name, expected in report["validation"].items()]
    raise ValueError(family)


def refit_provenance(db, directory: Path, data: Path, family: str) -> dict:
    selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    original = (directory.with_name("specialists") / "selection.json" if family == "specialists-refit"
                else directory.with_name("linear") / "frozen_selection_canonical_v2.json")
    prior = json.loads(original.read_text(encoding="utf-8"))
    if family == "specialists-refit":
        if selection["policies"]["f1"] != prior["policies"]["f1"]:
            raise AssertionError("refit changed the frozen 2024 specialist policy")
        if selection["training_latest_year"] != 2024 or not selection.get("tune_metrics_apply_to_original_pre2024_models"):
            raise AssertionError("refit provenance or tune-metric interpretation differs")
        hash_sources = [("original_selection_sha256", original),
                        ("refit_source_manifest_sha256", data / "refit_manifest.json"),
                        ("refit_train_sha256", data / "refit_train.parquet")]
    else:
        name = selection["selected_variant"]
        if name != prior["selected_variant"] or selection["threshold"] != prior["selections"][name]["threshold"]:
            raise AssertionError("linear refit changed the frozen 2024 variant/threshold")
        if selection["fit_end_exclusive"] != "2025-01-01" or selection["selection_year"] != 2024:
            raise AssertionError("linear refit training cutoff or selection year differs")
        spec = next(item for item in prior["variants"] if item["name"] == name)
        if selection["variant_spec"] != spec:
            raise AssertionError("linear refit changed feature/clipping configuration")
        hash_sources = [("source_selection_sha256", original)]
    counts = db.execute("""SELECT COUNT(*),SUM(target),COUNT(DISTINCT target_episode_id)
        FILTER(WHERE target=1),COUNT(*) FILTER(WHERE year(prediction_time)
        NOT IN (2019,2020,2022,2023,2024) OR label_available_at>=TIMESTAMP '2025-01-01')
        FROM read_parquet(?)""", [str(data / "refit_train.parquet")]).fetchone()
    source = json.loads((data / "refit_manifest.json").read_text(encoding="utf-8"))
    if counts[3] or list(counts[:3]) != [source["rows"], source["positive_hours"], source["positive_episodes"]]:
        raise AssertionError("refit training rows/labels cross the 2025 cutoff or counts differ")
    for key, path in hash_sources:
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != selection[key]:
                raise AssertionError(f"refit source hash differs: {key}")
    return {"training_rows": counts[0], "positive_hours": counts[1],
            "positive_episodes": counts[2], "forbidden_or_future_rows": counts[3],
            "frozen_2024_policy_unchanged": True, "provenance_hashes_verified": True}


def online_provenance(db, directory: Path, data: Path) -> dict:
    """Verify available-label cutoffs and score routing without changing policy."""
    selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    prior_path = directory.with_name("linear") / "frozen_selection_canonical_v2.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    name = prior["selected_variant"]
    if (selection["selected_variant"] != name
            or selection["threshold"] != prior["selections"][name]["threshold"]
            or selection["threshold_adapted_using_2025_labels"]
            or selection["configuration_source_sha256"] != hashlib.sha256(prior_path.read_bytes()).hexdigest()):
        raise AssertionError("quarterly refits changed the original 2024 policy")
    fits = json.loads((directory / "fit_manifest.json").read_text(encoding="utf-8"))
    boundaries = ["2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01", "2026-01-01"]
    if len(fits) != 4:
        raise AssertionError("quarterly fit manifest is incomplete")
    base = db.execute("""SELECT COUNT(*),SUM(target),MAX(label_available_at),
        COUNT(*) FILTER(WHERE label_available_at IS NULL
        OR label_available_at>=TIMESTAMP '2025-01-01')
        FROM read_parquet(?)""", [str(data / "refit_train.parquet")]).fetchone()
    if base[3]:
        raise AssertionError("initial quarterly fit contains future or missing labels")
    results = []
    for quarter, fit in enumerate(fits, 1):
        cutoff, end = boundaries[quarter-1:quarter+1]
        past = db.execute("""SELECT COUNT(*),COALESCE(SUM(target),0),MAX(label_available_at)
            FROM read_parquet(?) WHERE label_available_at<CAST(? AS TIMESTAMP)
            AND prediction_time<CAST(? AS TIMESTAMP)
            AND (target=1 OR hash(channel_id,prediction_time)%10000<200)""",
                          [str(data / "validation.parquet"), cutoff, cutoff]).fetchone()
        maximum = max(value for value in [base[2], past[2]] if value is not None)
        if (fit["quarter"] != quarter or fit["cutoff"] != cutoff
                or fit["rows"] != base[0]+past[0] or fit["positive_hours"] != base[1]+past[1]
                or pd.Timestamp(fit["maximum_label_available_at"]) != maximum
                or maximum >= pd.Timestamp(cutoff)):
            raise AssertionError(f"quarter {quarter} training availability/counts differ")
        with (directory / f"quarter_{quarter}.joblib").open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != fit["model_sha256"]:
                raise AssertionError(f"quarter {quarter} model hash differs")
        rows, bad = db.execute("""SELECT COUNT(*),COUNT(*) FILTER(
            WHERE prediction_time<CAST(? AS TIMESTAMP) OR prediction_time>=CAST(? AS TIMESTAMP))
            FROM read_parquet(?) WHERE model_quarter=?""",
                              [cutoff, end, str(directory / "validation_scores.parquet"), quarter]).fetchone()
        if bad or rows == 0:
            raise AssertionError(f"quarter {quarter} score routing differs")
        results.append({**fit, "scored_rows": rows, "future_training_labels": 0})
    unexpected = db.execute("""SELECT COUNT(*) FROM read_parquet(?)
        WHERE model_quarter IS NULL OR model_quarter NOT IN (1,2,3,4)""",
                            [str(directory / "validation_scores.parquet")]).fetchone()[0]
    if unexpected:
        raise AssertionError("scores have an unknown quarterly model")
    return {"frozen_2024_policy_unchanged": True, "provenance_hashes_verified": True,
            "global_cooldown_replayed_without_quarterly_reset": True, "quarterly_fits": results}


def replay_specialist_tune(db, directory: Path, total: int):
    selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    source = directory / "tune_scores.parquet"
    days = db.execute("SELECT COUNT(DISTINCT (channel_id,CAST(prediction_time AS DATE))) "
                      "FROM read_parquet(?)", [str(source)]).fetchone()[0]
    results = []
    for policy, mapping in selection["policies"].items():
        candidates = []
        for kind, option in mapping.items():
            if kind == "__default__" or option["model"] is None:
                continue
            column = f"score_{option['model']}"
            threshold = float(np.float32(option["threshold"]))
            frame = db.execute(f'SELECT channel_id,prediction_time,sensor_type,target,'
                               f'target_episode_id,label_available_at,"{column}" AS catboost_score '
                               f'FROM read_parquet(?) WHERE sensor_type=? AND "{column}">=?',
                               [str(source), kind, threshold]).fetch_df()
            frame["catboost_score"] -= np.float32(threshold)
            candidates.append(frame)
        if not candidates:
            continue
        joined = pd.concat(candidates, ignore_index=True)
        metric, _ = evaluate_alerts(joined, "catboost_score", 0.0, channel_days=days)
        p = metric["episode_precision"]
        r = metric["matched_episodes"] / total
        metric.update({"full_episode_count": total, "full_episode_recall": r,
                       "full_episode_f1": 2*p*r/(p+r) if p+r else 0})
        expected = selection["tune_metrics"][policy]
        for key in ["matched_episodes", "emitted_warnings", "episode_precision",
                    "full_episode_recall", "full_episode_f1"]:
            if abs(metric[key] - expected[key]) > 1e-15:
                raise AssertionError(f"specialists/{policy}/tune {key} differs")
        results.append({"family": "specialists", "variant": policy, "fold": "tune",
                        "production_metrics": metric,
                        "reported_evaluation_version": EVALUATION_VERSION})
    return results


def run(root: Path, data: Path, output: Path, families: list[str]) -> dict:
    destination = output / "report.json"
    if destination.exists():
        raise FileExistsError(destination)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if manifest["full_episode_count"]["tune"] != 1204 or manifest["full_episode_count"]["validation"] != 2142:
        raise AssertionError("complete B3 episode denominators differ")
    checks = []
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        db.execute("SET temp_directory=?", [str(output / "duckdb-temp")])
        for family in families:
            directory = root / family
            cases = champion_cases(directory, family)
            folds_checked = {}
            provenance = refit_provenance(db, directory, data, family) if family.endswith("-refit") else None
            if family == "online-linear":
                provenance = online_provenance(db, directory, data)
            if family == "specialists":
                metadata = metadata_parity(db, directory / "tune_scores.parquet",
                                           data / "tune.parquet", manifest["row_stats"]["tune"]["rows"], 2024)
                tune_checks = replay_specialist_tune(db, directory, manifest["full_episode_count"]["tune"])
                for check in tune_checks:
                    check["metadata"] = metadata
                checks.extend(tune_checks)
                print("full metadata and production warning parity passed specialists/tune", flush=True)
            for name, fold, source, column, threshold, expected in cases:
                if fold not in folds_checked:
                    folds_checked[fold] = metadata_parity(
                        db, source, data / f"{fold}.parquet", manifest["row_stats"][fold]["rows"],
                        2024 if fold == "tune" else 2025)
                actual = replay(db, source, column, threshold, manifest["full_episode_count"][fold])
                keys = ["matched_episodes", "emitted_warnings", "unmatched_warnings",
                        "duplicate_episode_warnings", "suppressed_positive_score_rows",
                        "episode_precision", "full_episode_recall", "full_episode_f1", "median_lead_hours"]
                for key in keys:
                    if key in expected and actual[key] != expected[key]:
                        raise AssertionError(f"{family}/{name}/{fold}: {key} differs")
                if "matched_episode_ids" in expected and set(actual["matched_episode_ids"]) != set(expected["matched_episode_ids"]):
                    raise AssertionError(f"{family}/{name}/{fold}: matched episode IDs differ")
                checks.append({"family": family, "variant": name, "fold": fold,
                               "threshold": threshold, "metadata": folds_checked[fold],
                               "production_metrics": actual,
                               "reported_evaluation_version": expected.get("evaluation_version"),
                               "refit_provenance": provenance})
                print(f"full metadata and production warning parity passed {family}/{name}/{fold}", flush=True)
    result = {"status": "full_metadata_and_canonical_warning_parity_passed",
              "evaluation_version": EVALUATION_VERSION,
              "full_episode_count": {"tune": 1204, "validation": 2142},
              "checks": checks, "selection_changed": False, "test_2026_read": False,
              "data_2021_read": False,
              "evaluation_source_sha256": hashlib.sha256(Path(
                  "analysis/ml_experiment_eval.py").read_bytes()).hexdigest(),
              "production_source_sha256": hashlib.sha256(Path(
                  "ml/forecast/alert_eval.py").read_bytes()).hexdigest()}
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("output/ml-experiment"))
    parser.add_argument("--data", type=Path, default=Path("output/ml-experiment/data"))
    parser.add_argument("--output", type=Path, default=Path("output/ml-experiment/final-audit"))
    parser.add_argument("--families", nargs="+", choices=["pooled", "linear", "specialists", "history", "ensemble", "specialists-refit", "linear-refit", "online-linear"],
                        default=["pooled", "linear", "specialists"])
    args = parser.parse_args()
    run(args.root, args.data, args.output, args.families)
