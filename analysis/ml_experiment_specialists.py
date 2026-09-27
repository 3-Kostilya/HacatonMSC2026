"""Type specialists selected on 2024, with frozen open-transfer scoring on 2025.

Only declared causal Q2 features enter a model. Historical target information,
channel identifiers and dates are excluded. Equal episode weights and hard
negative weights use training labels only. Scores are generated in bounded
batches; curves load only the selected warning candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from catboost import CatBoostClassifier

from analysis.ml_experiment_eval import evaluate

KEYS = ["channel_id", "prediction_time", "sensor_type", "target",
        "target_episode_id", "label_available_at"]
KINDS = ["Датчик дыма", "Состояние фазы", "Газовый датчик"]
QUANTILES = [.98, .99, .995, .9975, .999, .9995, .99975, .9999, .99995,
             .999975, .99999, .999995]


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_manifest(output: Path, *, training_latest_year: int = 2023):
    artifacts = {}
    for path in sorted(output.glob("*")):
        if path.is_file() and path.name != "scores_manifest.json":
            item = {"bytes": path.stat().st_size, "sha256": fingerprint(path)}
            if path.suffix == ".parquet":
                parquet = pq.ParquetFile(path)
                item.update(rows=parquet.metadata.num_rows, columns=parquet.schema_arrow.names)
            artifacts[path.name] = item
    result = {"schema_version": "ml-experiment-specialists-v1", "artifacts": artifacts,
              "validation_score_semantics": "selected probability minus frozen per-type threshold; warn at zero",
              "training_latest_year": training_latest_year, "selection_year": 2024,
              "validation_year": 2025, "forbidden_years_read": [],
              "targets_and_identifiers_are_features": False, "cooldown_hours": 24}
    (output/"scores_manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def engineered_input(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Add rate contrasts computed solely from causal nested past windows."""
    out = frame[names].copy()
    out["sensor_type"] = out.sensor_type.fillna("<unknown>").astype(str)
    for name in names:
        if name == "sensor_type":
            continue
        out[name] = pd.to_numeric(out[name], errors="coerce").replace(
            [np.inf, -np.inf], np.nan).fillna(-1).astype("float32")
    blocks = ["event_count", "state_count", "state_transitions", "alarm_count",
              "technical_message_count", "registered_fault_text_count",
              "normal_message_count", "environmental_alarm_count", "unknown_state_count"]
    extras = {}
    for prefix in blocks:
        if f"{prefix}_168h" not in out:
            continue
        for short, long in [(1, 6), (6, 24), (24, 168)]:
            a = out[f"{prefix}_{short}h"].clip(lower=0)
            b = out[f"{prefix}_{long}h"].clip(lower=0)
            previous_rate = (b-a).clip(lower=0) / (long-short)
            extras[f"contrast__{prefix}_{short}_{long}"] = np.log1p(a/short) - np.log1p(previous_rate)
            extras[f"share__{prefix}_{short}_{long}"] = (a+1)/(b+1)
    for hours in [1, 6, 24, 168]:
        normal = out[f"normal_message_count_{hours}h"].clip(lower=0)
        technical = out[f"technical_message_count_{hours}h"].clip(lower=0)
        extras[f"technical_normal_ratio_{hours}h"] = (technical+1)/(normal+1)
    return pd.concat([out, pd.DataFrame(extras, index=out.index).astype("float32")], axis=1)


def training_weights(frame: pd.DataFrame, mode: str) -> np.ndarray:
    y = frame.target.to_numpy(dtype=np.int8)
    weights = np.ones(len(frame), dtype=np.float64)
    positives = y == 1
    if mode in {"episode", "hardnegative"}:
        counts = frame.loc[positives, "target_episode_id"].value_counts()
        weights[positives] = frame.loc[positives, "target_episode_id"].map(
            lambda episode: float(1/counts[episode])).to_numpy()
        weights[positives] *= positives.sum()/weights[positives].sum()
    balance = (np.sum(~positives)/np.sum(positives)) ** (.5 if mode == "moderate" else 1)
    weights[positives] *= balance
    return weights


def fit_one(frame: pd.DataFrame, names: list[str], mode: str,
            *, pilot: CatBoostClassifier | None = None) -> tuple[CatBoostClassifier, dict]:
    x = engineered_input(frame, names)
    weights = training_weights(frame, mode)
    negatives = frame.target.to_numpy() == 0
    mined = 0
    if mode == "hardnegative":
        assert pilot is not None
        scores = pilot.predict_proba(x, thread_count=2)[:, 1]
        cutoff = np.quantile(scores[negatives], .90)
        hard = negatives & (scores >= cutoff)
        weights[hard] *= 5
        mined = int(hard.sum())
    model = CatBoostClassifier(
        iterations=420, depth=5, learning_rate=.055, l2_leaf_reg=12,
        random_seed=193, thread_count=2, loss_function="Logloss",
        verbose=False, allow_writing_files=False, cat_features=["sensor_type"])
    model.fit(x, frame.target.to_numpy(dtype=np.int8), sample_weight=weights)
    stats = {"rows": len(frame), "positive_hours": int(frame.target.sum()),
             "positive_episodes": int(frame.loc[frame.target == 1, "target_episode_id"].nunique()),
             "mined_negative_rows": mined, "mode": mode, "iterations": 420, "depth": 5,
             "feature_names": list(x.columns), "negative_sample_fraction": .02,
             "hard_negative_scope": "in_sample_training_negatives_only" if mined else None}
    return model, stats


def score_batches(source: Path, target: Path, names: list[str],
                  models: dict, *, policies: dict | None = None) -> None:
    writer = None
    rows = 0
    try:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=100_000):
            frame = batch.to_pandas()
            out = frame[KEYS].copy()
            x = engineered_input(frame, names)
            if policies is None:
                for key, (kind, model) in models.items():
                    score = np.full(len(frame), np.nan, dtype="float32")
                    mask = np.ones(len(frame), dtype=bool) if kind is None else frame.sensor_type.eq(kind).to_numpy()
                    if mask.any():
                        score[mask] = model.predict_proba(x.loc[mask], thread_count=2)[:, 1].astype("float32")
                    out[f"score_{key}"] = score
            else:
                for policy, mapping in policies.items():
                    margin = np.full(len(frame), -1, dtype="float32")
                    for kind, option in mapping.items():
                        mask = (frame.sensor_type.eq(kind).to_numpy() if kind != "__default__"
                                else ~frame.sensor_type.isin([name for name in mapping if name != "__default__"]).to_numpy())
                        if mask.any() and option["model"] is not None:
                            model = models[option["model"]][1]
                            threshold = np.float32(option["threshold"])
                            probabilities = model.predict_proba(x.loc[mask], thread_count=2)[:, 1].astype("float32")
                            margin[mask] = probabilities - threshold
                    out[f"score_{policy}"] = margin
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(target, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(frame)
            if rows % 1_000_000 == 0:
                print(f"{source.stem}: scored {rows}", flush=True)
    finally:
        if writer is not None:
            writer.close()


def warning_metrics(db, source: Path, column: str, threshold: float,
                    *, total: int, available: int, days: int, kind: list[str] | None = None):
    threshold = float(np.float32(threshold))
    kind_sql = " AND sensor_type IN (SELECT UNNEST(?))" if kind is not None else ""
    args = [str(source), threshold] + ([kind] if kind is not None else [])
    frame = db.execute(f'SELECT {",".join(KEYS)}, "{column}" AS catboost_score '
                       f'FROM read_parquet(?) WHERE "{column}">=? {kind_sql}', args).fetch_df()
    metric = evaluate(frame, "catboost_score", threshold, full_episode_count=total, channel_days=days)
    metric["eligible_positive_episodes"] = available
    metric["available_episode_recall"] = metric["matched_episodes"]/available if available else 0
    return metric


def choose_joint(options: dict[str, list[dict]], total: int, objective: str) -> tuple[dict, dict]:
    """Coordinate search over type curves; all counts include unavailable truth."""
    chosen = {kind: choices[0] for kind, choices in options.items()}
    def statistics(mapping):
        tp = sum(x["matched_episodes"] for x in mapping.values())
        warnings = sum(x["emitted_warnings"] for x in mapping.values())
        return {"matched_episodes": tp, "emitted_warnings": warnings,
                "episode_precision": tp/warnings if warnings else 0,
                "full_episode_recall": tp/total,
                "full_episode_f1": 2*tp/(total+warnings)}
    def utility(mapping):
        stat = statistics(mapping)
        if objective == "precision70":
            if stat["emitted_warnings"] and stat["episode_precision"] < .70:
                return -1 + stat["episode_precision"] * .01
            return stat["full_episode_recall"]
        return stat["full_episode_f1"]
    for _ in range(12):
        changed = False
        for kind, choices in options.items():
            best, score = chosen[kind], utility(chosen)
            for option in choices:
                candidate = {**chosen, kind: option}
                value = utility(candidate)
                if value > score + 1e-12:
                    best, score = option, value
            if best is not chosen[kind]:
                chosen[kind], changed = best, True
        if not changed:
            break
    # Small specialist populations permit exact joint enumeration, including the
    # precision constraint where coordinate updates can otherwise get stuck.
    if np.prod([len(choices) for choices in options.values()]) <= 200_000:
        score = utility(chosen)
        for combination in product(*options.values()):
            candidate = dict(zip(options, combination))
            value = utility(candidate)
            if value > score + 1e-12:
                chosen, score = candidate, value
    mapping = {kind: {"model": option["model"], "threshold": option["threshold"]}
               for kind, option in chosen.items()}
    return mapping, statistics(chosen)


def run(data: Path, output: Path, *, minimum_tune_episodes: int = 20, resume: bool = False):
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((data/"manifest.json").read_text(encoding="utf-8"))
    names = manifest["all_features"]
    assert "channel_id" not in names and "target" not in names
    with duckdb.connect(":memory:") as db:
        db.execute("SET threads=2")
        support = dict(db.execute('SELECT sensor_type, count(distinct target_episode_id) '
                                 'FROM read_parquet(?) WHERE target=1 GROUP BY sensor_type',
                                 [str(data/"tune.parquet")]).fetchall())
    supported = [kind for kind in KINDS if support.get(kind, 0) >= minimum_tune_episodes]
    models, fit_stats = {}, {}
    if resume:
        fit_stats = json.loads((output/"fit_manifest.json").read_text(encoding="utf-8"))
        for key in fit_stats:
            model = CatBoostClassifier(thread_count=2)
            model.load_model(str(output/f"{key}.cbm"))
            kind = KINDS[int(key[4])] if key.startswith("type") else None
            models[key] = (kind, model)
    else:
        train = pq.read_table(data/"train.parquet").to_pandas()
        assert set(train.prediction_time.dt.year) <= {2019, 2020, 2022, 2023}
        pooled, stats = fit_one(train, names, "moderate")
        models["pooled"] = (None, pooled)
        fit_stats["pooled"] = stats
        pooled.save_model(str(output/"pooled.cbm"))
        for index, kind in enumerate(KINDS):
            if kind not in supported:
                print(f"pooled fallback for {kind}: only {support.get(kind, 0)} tuning episodes", flush=True)
                continue
            local = train.loc[train.sensor_type.eq(kind)]
            count = local.loc[local.target == 1, "target_episode_id"].nunique()
            print(f"specialist {kind}: {len(local)} rows, {count} episodes", flush=True)
            if count < 20 or local.target.nunique() != 2:
                continue
            pilot = None
            for mode in ["moderate", "episode", "hardnegative"]:
                key = f"type{index}_{mode}"
                model, stats = fit_one(local, names, mode, pilot=pilot)
                if mode == "moderate":
                    pilot = model
                models[key], fit_stats[key] = (kind, model), stats
                model.save_model(str(output/f"{key}.cbm"))
                print(f"fitted {key}", flush=True)
        del train
        (output/"fit_manifest.json").write_text(json.dumps(fit_stats, indent=2, ensure_ascii=False), encoding="utf-8")
    tune_scores = output/"tune_scores.parquet"
    if not resume:
        score_batches(data/"tune.parquet", tune_scores, names, models)
    elif pq.ParquetFile(tune_scores).metadata.num_rows != pq.ParquetFile(data/"tune.parquet").metadata.num_rows:
        raise ValueError("resume requires complete tune scores")
    curves = {}
    with duckdb.connect(":memory:") as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        all_kinds = [row[0] for row in db.execute('SELECT DISTINCT sensor_type FROM read_parquet(?)',
                                               [str(tune_scores)]).fetchall()]
        groups = {kind: [kind] for kind in supported}
        fallback = [kind for kind in all_kinds if kind not in supported]
        if fallback:
            groups["__pooled_fallback__"] = fallback
        type_stats = db.execute('SELECT CASE WHEN sensor_type IN (SELECT UNNEST(?)) THEN sensor_type '
                               "ELSE '__pooled_fallback__' END AS type_group, "
                               'count(distinct case when target=1 then target_episode_id end), '
                               'count(distinct (channel_id,cast(prediction_time AS DATE))) '
                               'FROM read_parquet(?) GROUP BY type_group', [supported, str(tune_scores)]).fetchall()
        bad = db.execute('SELECT COUNT(*) FROM (SELECT channel_id FROM read_parquet(?) '
                         'GROUP BY channel_id HAVING COUNT(DISTINCT sensor_type)>1)', [str(tune_scores)]).fetchone()[0]
        if bad:
            raise ValueError("type curve counts cannot be additive: channels change type")
        total = manifest["full_episode_count"]["tune"]
        for kind, available, days in type_stats:
            options = [{"model": None, "threshold": None, "matched_episodes": 0, "emitted_warnings": 0}]
            for key, (model_kind, _) in models.items():
                if model_kind not in {None, kind}:
                    continue
                column = f"score_{key}"
                quantiles = db.execute(f'SELECT quantile_cont("{column}",?) FROM read_parquet(?) '
                                       'WHERE sensor_type IN (SELECT UNNEST(?))',
                                       [QUANTILES, str(tune_scores), groups[kind]]).fetchone()[0]
                thresholds = sorted(set(float(x) for x in quantiles if x is not None))
                for threshold in thresholds:
                    metric = warning_metrics(db, tune_scores, column, threshold,
                                             total=total, available=available, days=max(days, 1), kind=groups[kind])
                    options.append({**metric, "model": key})
                print(f"curves {kind} {key} complete", flush=True)
            curves[kind] = options
        policies, selected_tune = {}, {}
        for objective in ["f1", "precision70"]:
            grouped, selected_tune[objective] = choose_joint(curves, total, objective)
            policies[objective] = {kind: grouped[group] for group, kinds in groups.items() for kind in kinds}
            if "__pooled_fallback__" in grouped:
                policies[objective]["__default__"] = grouped["__pooled_fallback__"]
    selection = {"selection_year": 2024, "training_latest_year": 2023,
                 "evaluation_version": "full-b3-canonical-native-score-precision-v2",
                 "canonical_float32_threshold_replay": True,
                 "policies": policies, "tune_metrics": selected_tune,
                 "minimum_tune_episodes": minimum_tune_episodes,
                 "type_tune_support": support, "rare_types_share_pooled_threshold": True,
                 "curves": curves, "input_manifest_sha256": fingerprint(data/"manifest.json"),
                 "notes": "2025 is open transfer data, previously inspected; not untouched test"}
    (output/"selection.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(selected_tune, ensure_ascii=False), flush=True)
    validation_scores = output/"validation_scores.parquet"
    score_batches(data/"validation.parquet", validation_scores, names, models, policies=policies)
    report = {"selection": {k:v for k,v in selection.items() if k != "curves"}, "validation": {}}
    with duckdb.connect(":memory:") as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        available, days = db.execute('SELECT count(distinct case when target=1 then target_episode_id end), '
                                    'count(distinct (channel_id,cast(prediction_time AS DATE))) '
                                    'FROM read_parquet(?)', [str(validation_scores)]).fetchone()
        for objective in policies:
            report["validation"][objective] = warning_metrics(
                db, validation_scores, f"score_{objective}", 0, total=manifest["full_episode_count"]["validation"],
                available=available, days=days)
    (output/"report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    artifact_manifest(output)
    print(json.dumps(report["validation"], ensure_ascii=False), flush=True)
    return report


def rescore_validation(data: Path, output: Path):
    """Replay frozen policies after adding the same rare-type default to unseen types."""
    report = (json.loads((output/"report.json").read_text(encoding="utf-8"))
              if (output/"report.json").exists() else {"validation": {}})
    selection = json.loads((output/"selection.json").read_text(encoding="utf-8"))
    manifest = json.loads((data/"manifest.json").read_text(encoding="utf-8"))
    for mapping in selection["policies"].values():
        if "__default__" not in mapping:
            mapping["__default__"] = mapping["Газовый датчик"]
    models = {}
    for path in output.glob("*.cbm"):
        model = CatBoostClassifier(thread_count=2)
        model.load_model(str(path))
        models[path.stem] = (None, model)
    validation_scores = output/"validation_scores.parquet"
    score_batches(data/"validation.parquet", validation_scores, manifest["all_features"],
                  models, policies=selection["policies"])
    with duckdb.connect(":memory:") as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        available, days = db.execute('SELECT count(distinct case when target=1 then target_episode_id end), '
                                    'count(distinct (channel_id,cast(prediction_time AS DATE))) '
                                    'FROM read_parquet(?)', [str(validation_scores)]).fetchone()
        for objective in selection["policies"]:
            report["validation"][objective] = warning_metrics(
                db, validation_scores, f"score_{objective}", 0,
                total=manifest["full_episode_count"]["validation"], available=available, days=days)
    report["selection"] = {key:value for key,value in selection.items() if key != "curves"}
    report["unseen_type_behavior"] = "same pooled model and 2024 rare-type threshold as observed rare types"
    (output/"selection.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
    (output/"report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    artifact_manifest(output)
    print(json.dumps(report["validation"], ensure_ascii=False), flush=True)


def reselect(data: Path, output: Path):
    """Re-evaluate the same 2024 score-only grid with canonical float32 boundaries."""
    selection = json.loads((output/"selection.json").read_text(encoding="utf-8"))
    curves = selection["curves"]
    tune_scores = output/"tune_scores.parquet"
    manifest = json.loads((data/"manifest.json").read_text(encoding="utf-8"))
    total = manifest["full_episode_count"]["tune"]
    with duckdb.connect(":memory:") as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        kinds = [row[0] for row in db.execute('SELECT DISTINCT sensor_type FROM read_parquet(?)',
                                           [str(tune_scores)]).fetchall()]
        groups = {group: ([group] if group != "__pooled_fallback__"
                          else [kind for kind in kinds if kind not in curves]) for group in curves}
        for group, options in curves.items():
            available, days = db.execute('SELECT count(distinct case when target=1 then target_episode_id end), '
                                        'count(distinct (channel_id,cast(prediction_time AS DATE))) '
                                        'FROM read_parquet(?) WHERE sensor_type IN (SELECT UNNEST(?))',
                                        [str(tune_scores), groups[group]]).fetchone()
            for index, option in enumerate(options):
                if option["model"] is not None:
                    options[index] = {**warning_metrics(db, tune_scores, f"score_{option['model']}",
                                                       option["threshold"], total=total, available=available,
                                                       days=days, kind=groups[group]), "model": option["model"]}
            print(f"canonical2024 curve replay {group} complete", flush=True)
        for objective in selection["policies"]:
            grouped, selection["tune_metrics"][objective] = choose_joint(curves, total, objective)
            selection["policies"][objective] = {kind: grouped[group] for group, group_kinds in groups.items()
                                               for kind in group_kinds}
            if "__pooled_fallback__" in grouped:
                selection["policies"][objective]["__default__"] = grouped["__pooled_fallback__"]
    selection["canonical_float32_threshold_replay"] = True
    (output/"selection.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
    rescore_validation(data, output)


def final_refit(data: Path, original: Path, output: Path):
    """Fit only the selected F1 configuration through 2024; preserve its thresholds."""
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((data/"manifest.json").read_text(encoding="utf-8"))
    selection = json.loads((original/"selection.json").read_text(encoding="utf-8"))
    original_fits = json.loads((original/"fit_manifest.json").read_text(encoding="utf-8"))
    policy = selection["policies"]["f1"]
    selected_names = sorted({option["model"] for option in policy.values() if option["model"] is not None})
    train = pq.read_table(data/"refit_train.parquet").to_pandas()
    if not set(train.prediction_time.dt.year) <= {2019, 2020, 2022, 2023, 2024}:
        raise ValueError("refit source contains a forbidden or future year")
    if (train.loc[train.target == 1, "label_available_at"] >= pd.Timestamp("2025-01-01")).any():
        raise ValueError("refit positive labels are unavailable at the 2025 fold cutoff")
    models, fit_stats = {}, {}
    for key in selected_names:
        mode = original_fits[key]["mode"]
        if mode not in {"moderate", "episode"}:
            raise ValueError("final-refit supports the selected moderate and episode configurations")
        kind = KINDS[int(key[4])] if key.startswith("type") else None
        local = train if kind is None else train.loc[train.sensor_type.eq(kind)]
        print(f"refitting {key}: {len(local)} rows", flush=True)
        model, stats = fit_one(local, manifest["all_features"], mode)
        model.save_model(str(output/f"{key}.cbm"))
        models[key], fit_stats[key] = (kind, model), stats
        print(f"refitted {key}", flush=True)
    del train
    selection = {"training_latest_year": 2024, "original_training_latest_year": 2023,
                 "selection_year": 2024, "evaluation_version": "full-b3-canonical-native-score-precision-v2",
                 "canonical_float32_threshold_replay": True, "policies": {"f1": policy},
                 "tune_metrics": {"f1": selection["tune_metrics"]["f1"]},
                 "tune_metrics_apply_to_original_pre2024_models": True,
                 "thresholds_identical_to_original": True,
                 "threshold_scale_transfer_assumption": "scores after refit share the original probability scale; weights are recomputed on expanded historical train",
                 "input_manifest_sha256": fingerprint(data/"manifest.json"),
                 "refit_source_manifest_sha256": fingerprint(data/"refit_manifest.json"),
                 "refit_train_sha256": fingerprint(data/"refit_train.parquet"),
                 "original_selection_sha256": fingerprint(original/"selection.json"),
                 "notes": "Final-refit configuration was fixed from the 2024 family winner; no 2025 threshold selection"}
    (output/"selection.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
    (output/"fit_manifest.json").write_text(json.dumps(fit_stats, indent=2, ensure_ascii=False), encoding="utf-8")
    scores = output/"validation_scores.parquet"
    score_batches(data/"validation.parquet", scores, manifest["all_features"], models, policies={"f1": policy})
    with duckdb.connect(":memory:") as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        available, days = db.execute('SELECT count(distinct case when target=1 then target_episode_id end), '
                                    'count(distinct (channel_id,cast(prediction_time AS DATE))) '
                                    'FROM read_parquet(?)', [str(scores)]).fetchone()
        metric = warning_metrics(db, scores, "score_f1", 0,
                                 total=manifest["full_episode_count"]["validation"], available=available, days=days)
    report = {"family": "specialists_final_refit_through2024", "selection": selection,
              "validation": {"f1": metric}, "original_output": str(original),
              "comparison_scope": "same frozen 2024 F1 policy; 2025 open transfer validation"}
    (output/"report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    artifact_manifest(output, training_latest_year=2024)
    print(json.dumps(metric, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("output/ml-experiment/data"))
    parser.add_argument("--output", type=Path, default=Path("output/ml-experiment/specialists"))
    parser.add_argument("--minimum-tune-episodes", type=int, default=20)
    parser.add_argument("--rescore-validation", action="store_true")
    parser.add_argument("--reselect", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--final-refit", action="store_true")
    parser.add_argument("--refit-output", type=Path, default=Path("output/ml-experiment/specialists-refit"))
    args = parser.parse_args()
    if args.final_refit:
        final_refit(args.data, args.output, args.refit_output)
    elif args.reselect:
        reselect(args.data, args.output)
    elif args.rescore_validation:
        rescore_validation(args.data, args.output)
    else:
        run(args.data, args.output, minimum_tune_episodes=args.minimum_tune_episodes, resume=args.resume)
