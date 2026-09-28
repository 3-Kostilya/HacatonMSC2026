"""Frozen cross-model type routing, selected entirely on full 2024 scores.

Run: python -m analysis.ml_experiment_round2_routing
Existing models fit through 2023; this script never fits a model. Rare and new
types share one globally selected pooled fallback. Candidate margins retain
native float32 semantics and the production 24h warning cooldown.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.ml_experiment_eval import EVALUATION_VERSION, PreparedEvaluation, evaluate
from analysis.prepare_ml_experiment import sha256
from ml.forecast.alert_eval import evaluate_alerts

META = ["channel_id", "prediction_time", "sensor_type", "target", "target_episode_id", "label_available_at"]
QUANTILES = (1-np.geomspace(.05, .000002, 31)).tolist()
MINIMUM_SUPPORT = 20
N = {"tune": 1204, "validation": 2142}


def specs(root: Path) -> list[dict]:
    models = []
    for family, tune, validation in [("pooled", "scores_tune.parquet", "scores_validation.parquet"),
                                     ("linear", "tune_scores.parquet", "validation_scores.parquet"),
                                     ("history", "scores_tune.parquet", "scores_validation.parquet")]:
        folder = root/family
        for column in pq.ParquetFile(folder/tune).schema_arrow.names:
            if column.startswith("score_"):
                models.append({"name": f"{family}__{column[6:]}", "family": family,
                               "column": column, "tune": str(folder/tune), "validation": str(folder/validation)})
    if len(models) != 11:
        raise ValueError(f"expected 11 pre2024 models, got {len(models)}")
    return models


def aligned_batches(models: list[dict], fold: str):
    """Full key AND label equality, not merely same lengths, at every row."""
    families = {item["family"]: item[fold] for item in models}
    readers = {family: iter(pq.ParquetFile(path).iter_batches(batch_size=100_000))
               for family, path in families.items()}
    while True:
        batches = {family: next(reader, None) for family, reader in readers.items()}
        if all(batch is None for batch in batches.values()):
            break
        if any(batch is None for batch in batches.values()):
            raise ValueError("score sources have different row counts")
        frames = {family: batch.to_pandas() for family, batch in batches.items()}
        meta = frames["pooled"][META]
        for family, frame in frames.items():
            if not meta.equals(frame[META]):
                raise ValueError(f"full source metadata/label/order mismatch: {fold} {family}")
        out = meta.copy()
        for item in models:
            out[item["name"]] = frames[item["family"]][item["column"]].to_numpy(dtype="float32")
        yield out


def write_raw_tune(models: list[dict], path: Path) -> int:
    writer = None
    rows = 0
    try:
        for frame in aligned_batches(models, "tune"):
            if not frame.prediction_time.dt.year.eq(2024).all():
                raise ValueError("selection reads only 2024")
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(frame)
    finally:
        if writer is not None:
            writer.close()
    return rows


def scalar32(value: float) -> float:
    return float(np.float32(value))


def add_fusions(frame: pd.DataFrame, fusion: list[dict]) -> pd.DataFrame:
    """Normalize family champions at their global 2024 operating points."""
    parts = []
    for item in fusion:
        p = np.clip(frame[item["model"]].to_numpy(dtype="float64"), 1e-7, 1-1e-7)
        q = np.clip(item["threshold"], 1e-7, 1-1e-7)
        parts.append(np.log(p/(1-p))-np.log(q/(1-q)))
    matrix = np.column_stack(parts)
    frame["fusion__mean"] = (1/(1+np.exp(-matrix.mean(axis=1)))).astype("float32")
    frame["fusion__max"] = (1/(1+np.exp(-matrix.max(axis=1)))).astype("float32")
    return frame


def write_fused_tune(source: Path, path: Path, fusion: list[dict]):
    writer = None
    try:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=100_000):
            frame = add_fusions(batch.to_pandas(), fusion)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def grid(db, path: Path, model: str, kinds: list[str] | None = None) -> list[float]:
    condition = " WHERE sensor_type IN (SELECT UNNEST(?))" if kinds is not None else ""
    args = [QUANTILES, str(path)] + ([kinds] if kinds is not None else [])
    values = db.execute(f'SELECT quantile_cont("{model}", ?) FROM read_parquet(?)'+condition, args).fetchone()[0]
    return sorted({scalar32(value) for value in values} | {scalar32(1.000001)})


def regularized_grid(reference: float, local: float, support: int) -> list[float]:
    """Seven bounded log-odds shifts plus one support-shrunk local threshold."""
    def logit(x):
        x = np.clip(x, 1e-7, 1-1e-7)
        return np.log(x/(1-x))
    def probability(x):
        return scalar32(1/(1+np.exp(-x)))
    shrink = support/(support+300)
    center = logit(reference)
    values = [probability(center+shift*shrink) for shift in [-.75, -.5, -.25, 0, .25, .5, .75]]
    values.append(probability(center+shrink*(logit(local)-center)))
    values.extend([scalar32(reference), scalar32(1.000001)])
    return sorted(set(values))


def compact(metric: dict) -> dict:
    return {key: metric[key] for key in ["threshold", "matched_episodes", "emitted_warnings",
                                       "episode_precision", "full_episode_recall", "full_episode_f1"]}


def group_curve(db, path: Path, model: str, thresholds: list[float], kinds: list[str],
                days: int) -> dict[float, dict]:
    thresholds = sorted(set(scalar32(x) for x in thresholds))
    frame = db.execute(f'SELECT {",".join(META)}, "{model}" FROM read_parquet(?) '
                       f'WHERE sensor_type IN (SELECT UNNEST(?)) AND "{model}">=?',
                       [str(path), kinds, thresholds[0]]).fetch_df()
    prepared = PreparedEvaluation(frame, N["tune"], channel_days=max(days, 1))
    result = {threshold: {**compact(prepared.evaluate(model, threshold)), "model": model}
              for threshold in thresholds}
    del prepared, frame
    gc.collect()
    return result


def aggregate(routes: dict[str, dict]) -> dict:
    tp = sum(route["matched_episodes"] for route in routes.values())
    warnings = sum(route["emitted_warnings"] for route in routes.values())
    return {"matched_episodes": tp, "emitted_warnings": warnings,
            "episode_precision": tp/warnings if warnings else 0,
            "full_episode_recall": tp/N["tune"], "full_episode_f1": 2*tp/(N["tune"]+warnings)}


def optimize(options: dict[str, list[dict]], fallback: dict) -> tuple[dict, dict]:
    """Dinkelbach solves the finite additive global F1 ratio without macro averaging."""
    ratio = 0.0
    for _ in range(100):
        chosen = {kind: max(items, key=lambda item: (2*item["matched_episodes"]-ratio*item["emitted_warnings"],
                                                   -item["emitted_warnings"])) for kind, items in options.items()}
        chosen["__default__"] = fallback
        metrics = aggregate(chosen)
        updated = metrics["full_episode_f1"]
        if abs(updated-ratio) < 1e-12:
            return chosen, metrics
        ratio = updated
    raise RuntimeError("F1 ratio optimization failed to converge")


def margin32(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(probabilities, dtype="float32") - np.float32(threshold)


def routed_frame(frame: pd.DataFrame, policies: dict[str, dict]) -> pd.DataFrame:
    out = frame[META].copy()
    for policy, mapping in policies.items():
        default = mapping["__default__"]
        margins = margin32(frame[default["model"]].to_numpy(), default["threshold"])
        names = np.full(len(frame), default["model"], dtype=object)
        thresholds = np.full(len(frame), default["threshold"], dtype="float32")
        for kind, option in mapping.items():
            if kind == "__default__":
                continue
            mask = frame.sensor_type.eq(kind).to_numpy()
            margins[mask] = margin32(frame.loc[mask, option["model"]].to_numpy(), option["threshold"])
            names[mask], thresholds[mask] = option["model"], option["threshold"]
        out[f"score_{policy}"] = margins
        out[f"route_{policy}"] = names
        out[f"threshold_{policy}"] = thresholds
    return out


def write_routes(frames, policies: dict, path: Path) -> int:
    writer = None
    rows = 0
    try:
        for frame in frames:
            out = routed_frame(frame, policies)
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(out)
    finally:
        if writer is not None:
            writer.close()
    return rows


def evaluate_saved(db, path: Path, column: str, fold: str, days: int, available: int) -> dict:
    candidates = db.execute(f'SELECT {",".join(META)}, "{column}" FROM read_parquet(?) WHERE "{column}">=0',
                            [str(path)]).fetch_df()
    metric = evaluate(candidates, column, 0, full_episode_count=N[fold], channel_days=days)
    metric["eligible_positive_episodes"] = available
    metric["available_episode_recall"] = metric["matched_episodes"]/available
    canonical = candidates.rename(columns={column: "catboost_score"})
    production, alerts = evaluate_alerts(canonical, "catboost_score", 0, channel_days=days)
    for key in ["matched_episodes", "emitted_warnings", "unmatched_warnings", "duplicate_episode_warnings",
                "suppressed_positive_score_rows", "episode_precision", "median_lead_hours"]:
        if metric[key] != production[key]:
            raise AssertionError(f"independent production evaluator mismatch: {key}")
    emitted_ids = set(alerts.loc[alerts.outcome.eq("matched_episode"), "target_episode_id"])
    if emitted_ids != set(metric["matched_episode_ids"]):
        raise AssertionError("independent production evaluator matches different episode IDs")
    metric["independent_production_count_and_precision_parity"] = True
    metric["independent_production_episode_identity_parity"] = True
    return metric


def run(root: Path, output: Path, *, resume: bool = False) -> dict:
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    models = specs(root)
    raw = output/"joined_tune.parquet"
    rows = write_raw_tune(models, raw) if not resume else pq.ParquetFile(raw).metadata.num_rows
    curves = {}
    with duckdb.connect(":memory:") as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        stats = db.execute('SELECT sensor_type, COUNT(DISTINCT CASE WHEN target=1 THEN target_episode_id END), '
                           'COUNT(DISTINCT (channel_id,CAST(prediction_time AS DATE))) '
                           'FROM read_parquet(?) GROUP BY sensor_type', [str(raw)]).fetchall()
        supports = {kind: int(count) for kind, count, _ in stats}
        supported = [kind for kind in supports if supports[kind] >= MINIMUM_SUPPORT]
        groups = {kind: [kind] for kind in supported}
        groups["__default__"] = [kind for kind in supports if kind not in supported]
        days_by_type = {kind: int(days) for kind, _, days in stats}
        days_by_group = {group: sum(days_by_type[kind] for kind in kinds) for group, kinds in groups.items()}
        if db.execute('SELECT COUNT(*) FROM (SELECT channel_id FROM read_parquet(?) GROUP BY channel_id '
                      'HAVING COUNT(DISTINCT sensor_type)>1)', [str(raw)]).fetchone()[0]:
            raise ValueError("type routes are not additive: channels change type")
        global_curves, references = {}, {}
        flexible, regularized = {kind: [] for kind in supported}, {kind: [] for kind in supported}
        fusion = []
        for spec in models+[{"name": "fusion__mean"}, {"name": "fusion__max"}]:
            model = spec["name"]
            if model == "fusion__mean":
                for family in ["pooled", "linear", "history"]:
                    name = max((item["name"] for item in models if item["family"] == family),
                               key=lambda name: references[name]["full_episode_f1"])
                    fusion.append({"family": family, "model": name, "threshold": references[name]["threshold"]})
                fused = output/"joined_fused_tune.parquet"
                write_fused_tune(raw, fused, fusion)
                raw = fused
            global_grid = grid(db, raw, model)
            per_group = {}
            for group, kinds in groups.items():
                local_grid = grid(db, raw, model, kinds) if group != "__default__" else []
                options = group_curve(db, raw, model, global_grid+local_grid, kinds, days_by_group[group])
                per_group[group] = options
            global_curves[model] = [{**aggregate({group: values[threshold] for group, values in per_group.items()}),
                                      "threshold": threshold, "model": model} for threshold in global_grid]
            reference = max(global_curves[model], key=lambda item: (item["full_episode_f1"], item["episode_precision"]))
            references[model] = reference
            for group in supported:
                options = per_group[group]
                flexible[group].extend(options.values())
                local_best = max(options.values(), key=lambda item: 2*item["matched_episodes"]/
                                 (supports[group]+item["emitted_warnings"]))
                bounded_grid = regularized_grid(reference["threshold"], local_best["threshold"], supports[group])
                bounded = group_curve(db, raw, model, bounded_grid, groups[group], days_by_group[group])
                regularized[group].extend(bounded.values())
            curves[model] = {group: list(options.values()) for group, options in per_group.items()}
            print(f"2024 curves complete: {model}", flush=True)
        pooled_name = max((spec["name"] for spec in models if spec["family"] == "pooled"),
                          key=lambda name: references[name]["full_episode_f1"])
        default_threshold = references[pooled_name]["threshold"]
        fallback = next(option for option in curves[pooled_name]["__default__"] if option["threshold"] == default_threshold)
        chosen, metrics = {}, {}
        for name, options in [("flexible", flexible), ("regularized", regularized)]:
            chosen[name], metrics[name] = optimize(options, fallback)
        global_name = max(references, key=lambda name: references[name]["full_episode_f1"])
        policies = {name: {kind: {"model": option["model"], "threshold": option["threshold"]}
                           for kind, option in mapping.items()} for name, mapping in chosen.items()}
        winner = max(metrics, key=lambda name: metrics[name]["full_episode_f1"])
        selection = {"selection_year": 2024, "models_fit_latest_year": 2023,
                     "full_episode_counts": N, "evaluation_version": EVALUATION_VERSION,
                     "supported_types": supports, "minimum_type_support": MINIMUM_SUPPORT,
                     "fallback": {"model": pooled_name, "threshold": default_threshold},
                     "policies": policies, "tune_metrics": metrics, "selected_policy": winner,
                     "global_baseline": references[global_name], "global_references": references,
                     "fusion": fusion, "fusion_methods": ["mean_logit_margin", "max_logit_margin"],
                     "regularization": "7 bounded support-shrunk logit shifts plus a shrunk local optimum; n/(n+300)",
                     "optimization": "exact finite additive F1 via Dinkelbach; fixed pooled fallback",
                     "models": models, "tune_source_hashes": {spec["tune"]: sha256(Path(spec["tune"])) for spec in models}}
        (output/"selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
        (output/"curves.json").write_text(json.dumps({"flexible": flexible, "regularized": regularized,
                                                    "global": global_curves}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metrics), flush=True)
        tune_scores = output/"tune_scores.parquet"
        write_routes((batch.to_pandas() for batch in pq.ParquetFile(raw).iter_batches(batch_size=100_000)), policies, tune_scores)
        available = sum(supports.values())
        tune_checks = {name: evaluate_saved(db, tune_scores, f"score_{name}", "tune", sum(days_by_type.values()), available)
                       for name in policies}
        for name in policies:
            for key in ["matched_episodes", "emitted_warnings"]:
                if tune_checks[name][key] != metrics[name][key]:
                    raise AssertionError("additive routing differs from full-row evaluator")
    # No validation source is opened until the frozen selection has been saved.
    validation_scores = output/"validation_scores.parquet"
    validation_rows = write_routes((add_fusions(frame, fusion) for frame in aligned_batches(models, "validation")),
                                   policies, validation_scores)
    with duckdb.connect(":memory:") as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        available, days = db.execute('SELECT COUNT(DISTINCT CASE WHEN target=1 THEN target_episode_id END), '
                                    'COUNT(DISTINCT (channel_id,CAST(prediction_time AS DATE))) FROM read_parquet(?)',
                                    [str(validation_scores)]).fetchone()
        transfers = {name: evaluate_saved(db, validation_scores, f"score_{name}", "validation", days, available)
                     for name in policies}
    report = {"scope": "2024 frozen selection; previously open 2025 exploratory temporal transfer",
              "selected_policy": winner, "selection": selection, "tune": tune_checks,
              "validation": transfers, "rows": {"tune": rows, "validation": validation_rows},
              "source_metadata_and_label_parity": "all rows across all 3 source families",
              "artifact_hashes": {path.name: sha256(path) for path in output.glob("*.parquet")},
              "selection_sha256": sha256(output/"selection.json")}
    (output/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: compact(metric) for name, metric in transfers.items()}), flush=True)
    return report


def audit(output: Path):
    """Fresh production replay, exact episode identities and full metadata checks."""
    selection = json.loads((output/"selection.json").read_text(encoding="utf-8"))
    report = json.loads((output/"report.json").read_text(encoding="utf-8"))
    if sha256(output/"selection.json") != report["selection_sha256"]:
        raise AssertionError("selection changed after transfer scoring")
    result = {"selection_sha256": report["selection_sha256"], "folds": {}}
    for fold in ["tune", "validation"]:
        source = Path(next(item[fold] for item in selection["models"] if item["family"] == "pooled"))
        output_path = output/("tune_scores.parquet" if fold == "tune" else "validation_scores.parquet")
        left = pq.ParquetFile(source).iter_batches(batch_size=100_000, columns=META)
        right = pq.ParquetFile(output_path).iter_batches(batch_size=100_000, columns=META)
        rows = 0
        while True:
            a, b = next(left, None), next(right, None)
            if a is None and b is None:
                break
            if a is None or b is None or not a.to_pandas().equals(b.to_pandas()):
                raise AssertionError(f"routing output changes source keys or labels: {fold}")
            rows += len(a)
        with duckdb.connect(":memory:") as db:
            db.execute("SET threads=2")
            db.execute("SET memory_limit='2GB'")
            available, days = db.execute('SELECT COUNT(DISTINCT CASE WHEN target=1 THEN target_episode_id END), '
                                        'COUNT(DISTINCT (channel_id,CAST(prediction_time AS DATE))) FROM read_parquet(?)',
                                        [str(output_path)]).fetchone()
            policies = {}
            for policy in selection["policies"]:
                metric = evaluate_saved(db, output_path, f"score_{policy}", fold, days, available)
                expected = report["tune" if fold == "tune" else "validation"][policy]
                for key in ["matched_episodes", "emitted_warnings", "episode_precision", "full_episode_recall",
                            "full_episode_f1", "eligible_positive_episodes", "matched_episode_ids"]:
                    if metric[key] != expected[key]:
                        raise AssertionError(f"saved report and fresh independent replay differ: {fold} {policy} {key}")
                policies[policy] = {**compact(metric), "exact_matched_episode_identity_parity": True}
        result["folds"][fold] = {"rows": rows, "complete_source_metadata_label_parity": True,
                                  "eligible_episodes": available, "full_episode_count": N[fold], "policies": policies}
    result["status"] = "passed"
    (output/"independent_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("output/ml-experiment"))
    parser.add_argument("--output", type=Path, default=Path("output/ml-experiment-round2/routing"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        audit(args.output)
    else:
        run(args.root, args.output, resume=args.resume)
