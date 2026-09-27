"""Explore existing OPEN 2025 scores; these choices cannot validate a new model.

Reads only 2025 monthly saved predictions and 2025 episode diagnostics. Search
per-type thresholds and model fusion to quantify headroom in already saved
scores. This is deliberately isolated from the 2024 model-selection workflow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from analysis.ml_experiment_eval import PreparedEvaluation, combine_scores, search_thresholds
from ml.forecast.alert_eval import evaluate_alerts


def compact(metric):
    return {key: value for key, value in metric.items()
            if key not in {"matched_episode_ids", "by_type"}}


def choose_joint(curves, full_count, beta=1.0):
    # Fractional programming: maximize (1+b²)TP / (b²N + warnings)
    # over one finite candidate per type. Channels belong to one type, so the
    # warning cooldown and TP/warning totals decompose exactly by type.
    ratio = 0.0
    selected = {}
    for _ in range(100):
        selected = {kind: max(options, key=lambda option:
                    ((1 + beta**2) * option["matched_episodes"]
                     - ratio * option["emitted_warnings"],
                     -option["emitted_warnings"]))
                    for kind, options in curves.items()}
        tp = sum(option["matched_episodes"] for option in selected.values())
        warnings = sum(option["emitted_warnings"] for option in selected.values())
        updated = (1 + beta**2) * tp / (beta**2 * full_count + warnings)
        if abs(updated - ratio) < 1e-12:
            break
        ratio = updated
    p = tp / warnings if warnings else 0.0
    r = tp / full_count
    return {"beta_optimized_on_open_2025": beta, "matched_episodes": tp,
            "emitted_warnings": warnings, "unmatched_warnings": warnings - tp,
            "episode_precision": p, "full_episode_recall": r,
            "full_episode_f1": 2 * p * r / (p + r) if p + r else 0.0,
            "selected_by_type": selected}


def audit_saved_policy(source: Path, output: Path) -> dict:
    """Independent production replay of the exploratory best-F1 saved policy."""
    destination = output / "parity_audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    policy = next(row for row in report["joint_research_choices"]
                  if row["beta_optimized_on_open_2025"] == 1.0)
    paths = [str(source / f"validation_2025-{month:02d}.parquet")
             for month in range(1, 13)]
    score_names = ["score_base51", "score_full121", "score_linear121"]
    checked = {}
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        for kind, choice in policy["selected_by_type"].items():
            column, threshold = choice["score_column"], choice["threshold"]
            # Mean/min/max cannot exceed the maximum source score, allowing a
            # small bounded projection. Float32 arithmetic is reproduced in
            # pandas exactly as in the exploratory original frame.
            frame = db.execute("""SELECT * FROM read_parquet(?) WHERE sensor_type=?
                AND greatest(score_base51,score_full121,score_linear121)>=?""",
                               [paths, kind, threshold-1e-6]).fetch_df()
            if column not in score_names:
                method = column.removeprefix("score_")
                frame[column] = (frame[score_names].mean(axis=1) if method == "mean"
                                 else combine_scores(frame, score_names, method=method))
            canonical, _ = evaluate_alerts(frame.rename(columns={column: "catboost_score"}),
                                           "catboost_score", threshold,
                                           channel_days=max(choice["channel_days"], 1))
            for key in ["matched_episodes", "emitted_warnings", "unmatched_warnings",
                        "suppressed_positive_score_rows", "duplicate_episode_warnings",
                        "median_lead_hours"]:
                if canonical[key] != choice[key]:
                    raise AssertionError(f"production parity differs: {kind} {key}")
            checked[kind] = {"matched_episodes": canonical["matched_episodes"],
                             "emitted_warnings": canonical["emitted_warnings"]}
    result = {"status": "production_warning_replay_exact_parity_passed",
              "checked_type_policies": len(checked), "by_type": checked,
              "full_episode_count": report["full_episode_count"],
              "matched_episodes": sum(row["matched_episodes"] for row in checked.values()),
              "emitted_warnings": sum(row["emitted_warnings"] for row in checked.values()),
              "data_2026_read": False, "data_2021_read": False}
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def run(source, diagnostics, output):
    if output.exists():
        raise FileExistsError(output)
    paths = [str(source / f"validation_2025-{month:02d}.parquet")
             for month in range(1, 13)]
    baseline = json.loads((source / "report.json").read_text(encoding="utf-8"))
    all_curves = {}
    by_type = {}
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        truth = db.execute("""SELECT sensor_type, COUNT(DISTINCT target_episode_id)
            FROM read_parquet(?) WHERE split='validation' GROUP BY sensor_type""",
                           [str(diagnostics)]).fetchall()
        multitype = db.execute("""SELECT COUNT(*) FROM (
            SELECT channel_id FROM read_parquet(?) GROUP BY channel_id
            HAVING COUNT(DISTINCT sensor_type)>1)""", [paths]).fetchone()[0]
        if multitype:
            raise ValueError("per-type cooldown does not decompose across channels")
        for kind, full_count in truth:
            frame = db.execute("""SELECT * FROM read_parquet(?)
                WHERE sensor_type IS NOT DISTINCT FROM ?""", [paths, kind]).fetch_df()
            if frame.empty:
                by_type[str(kind)] = {"full_episodes": full_count, "rows": 0}
                continue
            score_names = ["score_base51", "score_full121", "score_linear121"]
            frame["score_mean"] = frame[score_names].mean(axis=1)
            frame["score_max"] = frame[score_names].max(axis=1)
            frame["score_min"] = frame[score_names].min(axis=1)
            prepared = PreparedEvaluation(frame, full_count)
            options = []
            model_best = {}
            for column in score_names + ["score_mean", "score_max", "score_min"]:
                curve = [compact(item) for item in search_thresholds(
                    prepared, column, points=40)]
                options.extend(curve)
                model_best[column] = max(curve, key=lambda row: row["full_episode_f1"])
            all_curves[str(kind)] = options
            by_type[str(kind)] = {"full_episodes": full_count, "rows": len(frame),
                                  "best_by_model": model_best}
            print(f"{kind}: {len(frame)} rows, {full_count} episodes, best F1="
                  f"{max(item['full_episode_f1'] for item in options):.4f}", flush=True)
            del frame, prepared
    full_count = sum(count for _, count in truth)
    if full_count != 2142:
        raise ValueError("saved 2025 full denominator differs")
    report = {"status": "exploratory_open_2025_scores_only_not_model_validation",
              "source": str(source), "full_episode_count": full_count,
              "test_data_read": False, "data_2021_read": False,
              "existing_pooled_model_choices": baseline["threshold_goals"],
              "by_type": by_type,
              "joint_research_choices": [choose_joint(all_curves, full_count, beta)
                                          for beta in [.5, 1.0, 2.0]],
              "limitations": [
                  "Every choice here uses 2025 outcomes and is exploratory/optimistic.",
                  "No new model or final threshold may be selected from this report.",
                  "2026 and 2021 data were not read.",
                  "Fixed 24h cooldown and full assigned B3 episode denominator retained."]}
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    (output / "curves.json").write_text(json.dumps(all_curves, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=Path("output/q2-b-expanded-validation-20260927"))
    parser.add_argument("--diagnostics", type=Path,
                        default=Path("output/q2-a-full-sparse-20260926-v5/episode_diagnostics.parquet"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/ml-experiment/saved-score-diagnostics"))
    args = parser.parse_args()
    report = run(args.source, args.diagnostics, args.output)
    print(json.dumps([{key: value for key, value in row.items() if key != "selected_by_type"}
                      for row in report["joint_research_choices"]], ensure_ascii=False), flush=True)
