"""OPEN2025 optimistic diagnostic frontiers after 2024 policies were frozen.

No model/threshold is promoted by this report. The test, targets, admission and
warning policy stay unchanged. Projection loads only warning candidates for one
model at a time, retaining full episode and channel-day denominators.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.ml_experiment_eval import EVALUATION_VERSION, PreparedEvaluation


FULL_EPISODES = 2142
META = ["channel_id", "prediction_time", "sensor_type", "target",
        "target_episode_id", "label_available_at"]


def compact(metric):
    return {key: value for key, value in metric.items()
            if key not in {"matched_episode_ids", "by_type"}}


def pareto(points):
    unique = {}
    for row in points:
        unique.setdefault((row["episode_precision"], row["full_episode_recall"]), row)
    frontier = []
    for (p, r), row in unique.items():
        if not any(p2 >= p and r2 >= r and (p2 > p or r2 > r)
                   for p2, r2 in unique):
            frontier.append(row)
    return sorted(frontier, key=lambda row: row["full_episode_recall"])


def summarize(curve):
    at_precision = [row for row in curve if row["episode_precision"] > .7]
    at_recall = [row for row in curve if row["full_episode_recall"] > .5]
    return {
        "optimistic_best_full_f1": max(curve, key=lambda row: row["full_episode_f1"]),
        "max_recall_at_precision_gt_0_7": max(at_precision, key=lambda row: row["full_episode_recall"])
            if at_precision else None,
        "max_precision_at_recall_gt_0_5": max(at_recall, key=lambda row: row["episode_precision"])
            if at_recall else None,
        "joint_goal_met_anywhere_on_grid": any(row["episode_precision"] > .7
                                               and row["full_episode_recall"] > .5 for row in curve),
        "pareto": pareto(curve),
    }


def family_source(folder: Path) -> Path:
    if not (folder / "report.json").exists() and not (folder / "report_canonical_v2.json").exists():
        raise FileNotFoundError(f"family must finish frozen-policy reporting before diagnostic: {folder}")
    for filename in ["scores_validation.parquet", "validation_scores.parquet"]:
        candidate = folder / filename
        if candidate.exists():
            pq.ParquetFile(candidate)  # Require a closed complete file footer.
            return candidate
    raise FileNotFoundError(f"no complete 2025 score file in {folder}")


def run(root: Path, output: Path, families: list[str], points: int = 45):
    if output.exists():
        raise FileExistsError(output)
    if not 40 <= points <= 80:
        raise ValueError("40 to 80 threshold points are required")
    curves, summaries, all_points = {}, {}, []
    quantiles = (1 - np.geomspace(.20, .00001, points - 1)).tolist()
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='1GB'")
        for family in families:
            source = family_source(root / family)
            schema = pq.ParquetFile(source).schema_arrow
            names = [name for name in schema.names if name.startswith("score_")]
            if not set(META) <= set(schema.names) or not names:
                raise ValueError(f"score schema differs: {source}")
            n, available, days, min_year, max_year = db.execute("""SELECT COUNT(*),
                COUNT(DISTINCT CASE WHEN target=1 THEN target_episode_id END),
                COUNT(DISTINCT (channel_id,CAST(prediction_time AS DATE))),
                MIN(year(prediction_time)),MAX(year(prediction_time))
                FROM read_parquet(?)""", [str(source)]).fetchone()
            if (min_year, max_year) != (2025, 2025) or n != 13945520 or available != 1359:
                raise ValueError(f"frontier must use complete fixed 2025 population: {source}")
            for column in names:
                thresholds, maximum = db.execute(f'SELECT quantile_cont("{column}",?),'
                                                 f'max("{column}") FROM read_parquet(?) '
                                                 f'WHERE isfinite("{column}")',
                                                 [quantiles, str(source)]).fetchone()
                if maximum is None:
                    continue
                thresholds = sorted(set(float(value) for value in thresholds))
                thresholds.append(float(maximum + max(1.0, abs(maximum))*1e-6))
                minimum = float(np.float32(thresholds[0])) if pa.types.is_float32(schema.field(column).type) else thresholds[0]
                projection = ",".join(f'"{name}"' for name in META)
                frame = db.execute(f'SELECT {projection},"{column}" FROM read_parquet(?) '
                                   f'WHERE isfinite("{column}") AND "{column}">=?',
                                   [str(source), minimum]).fetch_df()
                prepared = PreparedEvaluation(frame, FULL_EPISODES, days)
                key = f"{family}/{column}"
                curve = []
                for threshold in thresholds:
                    metric = compact(prepared.evaluate(column, threshold))
                    metric.update({"family": family, "eligible_positive_episodes": available,
                                   "available_episode_recall": metric["matched_episodes"]/available})
                    curve.append(metric)
                curves[key] = curve
                summaries[key] = summarize(curve)
                all_points.extend(curve)
                best = summaries[key]["optimistic_best_full_f1"]
                print(f"OPEN2025 DIAGNOSTIC {key}: P={best['episode_precision']:.4f} "
                      f"R={best['full_episode_recall']:.4f} F1={best['full_episode_f1']:.4f}", flush=True)
                del frame, prepared
    original = json.loads(Path("output/q2-b-expanded-validation-20260927/report.json").read_text(encoding="utf-8"))
    report = {
        "status": "OPEN2025_DIAGNOSTIC_OPTIMISTIC_NO_MODEL_OR_THRESHOLD_PROMOTION",
        "evaluation_version": EVALUATION_VERSION, "full_episode_count": FULL_EPISODES,
        "threshold_points_requested": points, "families": families,
        "published_existing_open_2025_reference": original["threshold_goals"],
        "by_model": summaries, "pooled_optimistic_frontier": summarize(all_points),
        "test_2026_read": False, "data_2021_read": False,
        "selection_changed": False,
        "limitations": ["Every frontier choice uses previously opened 2025 outcomes.",
                        "These values are optimistic exploratory headroom, not confirmation.",
                        "The separately frozen 2024 policy remains the selected policy."]}
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "curves.json").write_text(json.dumps(curves, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("output/ml-experiment"))
    parser.add_argument("--output", type=Path, default=Path("output/ml-experiment/frontier"))
    parser.add_argument("--families", nargs="+", default=["pooled", "history", "linear", "specialists-refit"])
    parser.add_argument("--points", type=int, default=45)
    args = parser.parse_args()
    run(args.root, args.output, args.families, args.points)
