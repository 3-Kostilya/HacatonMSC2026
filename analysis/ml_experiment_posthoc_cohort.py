"""Optimize a clearly labeled post-hoc channel cohort under diversity constraints.

Selection uses open 2025 outcomes, so its 2025 Precision is an in-sample maximum,
not an independent validation result or a replacement for full-stream metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from analysis.prepare_ml_experiment import sha256


WARNINGS = {
    2024: Path("output/ml-experiment-round7/fullstream-2024-unknown21-v1/candidate_warnings.parquet"),
    2025: Path("output/ml-experiment-round7/fullstream-2025-unknown21-v2/candidate_warnings.parquet"),
}
LABELS = Path("output/r3-b-full-months-20260925-v2")
FULL_ASSIGNED_EPISODES = {2024: 1204, 2025: 2142}


def scored_warnings(db: duckdb.DuckDBPyConnection, year: int) -> pd.DataFrame:
    path = str(LABELS / f"year={year}" / "month=*" / "registered_forecast_labels.parquet")
    frame = db.execute('''SELECT w.*,l.split_status,l.sensor_type AS label_sensor_type
        FROM read_parquet(?) w JOIN read_parquet(?,hive_partitioning=false) l
        USING(channel_id,prediction_time)''', [str(WARNINGS[year]), path]).fetch_df()
    if (len(frame) != len(pd.read_parquet(WARNINGS[year], columns=["channel_id"]))
            or frame.duplicated(["channel_id", "prediction_time"]).any()
            or not frame.sensor_type.eq(frame.label_sensor_type).all()):
        raise AssertionError("warning/B3 exact key and sensor parity differs")
    return frame.drop(columns="label_sensor_type")


def channel_stats(frame: pd.DataFrame, minimum_warning_count: int) -> pd.DataFrame:
    grouped = frame.groupby("channel_id", sort=True)
    stats = grouped.agg(sensor_type=("sensor_type", "first"),
                        warnings=("outcome", "size"))
    stats["assigned_tp"] = grouped.apply(
        lambda group: int((group.outcome.eq("matched_known_episode") &
                           group.split_status.eq("assigned")).sum()),
        include_groups=False)
    stats["months"] = grouped.prediction_time.apply(
        lambda times: sorted(int(month) for month in times.dt.month.unique()))
    if (frame.groupby("channel_id").sensor_type.nunique() > 1).any():
        raise AssertionError("channel has multiple sensor types")
    return stats.loc[stats.warnings.ge(minimum_warning_count)].reset_index()


def optimize(stats: pd.DataFrame, *, min_warnings: int, min_channels: int,
             min_types: int, min_type_warnings: int, min_months: int,
             max_channel_share: float, max_type_share: float) -> tuple[list[str], dict]:
    n = stats.warnings.to_numpy(dtype=float)
    tp = stats.assigned_tp.to_numpy(dtype=float)
    kinds = stats.sensor_type.tolist()
    month_sets = [set(value) for value in stats.months]
    kind_names = sorted(set(kinds))
    channels, types = len(stats), len(kind_names)
    variables = channels + types + 12
    rows: list[dict[int, float]] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(coefficients: dict[int, float], minimum: float = -np.inf,
            maximum: float = np.inf) -> None:
        rows.append(coefficients)
        lower.append(minimum)
        upper.append(maximum)

    add({index: n[index] for index in range(channels)}, min_warnings)
    add({index: 1 for index in range(channels)}, min_channels)
    add({channels + index: 1 for index in range(types)}, min_types)
    add({channels + types + index: 1 for index in range(12)}, min_months)
    for index, kind in enumerate(kind_names):
        support = {i: n[i] for i in range(channels) if kinds[i] == kind}
        support[channels + index] = -min_type_warnings
        add(support, 0)
        add({i: max_type_share * n[i] - (n[i] if kinds[i] == kind else 0)
             for i in range(channels)}, 0)
    for month in range(1, 13):
        support = {i: 1 for i in range(channels) if month in month_sets[i]}
        support[channels + types + month - 1] = -1
        add(support, 0)
    # At the minimum sample size, smaller channels cannot breach this share.
    for index in range(channels):
        if n[index] > max_channel_share * min_warnings:
            add({i: (n[i] if i == index else 0) - max_channel_share * n[i]
                 for i in range(channels)}, maximum=0)
    matrix = lil_matrix((len(rows), variables), dtype=float)
    for row_index, coefficients in enumerate(rows):
        for column_index, coefficient in coefficients.items():
            matrix[row_index, column_index] = coefficient
    constraints = LinearConstraint(matrix.tocsr(), lower, upper)
    bounds = Bounds(np.zeros(variables), np.ones(variables))
    probability = .7
    history = []
    for iteration in range(12):
        objective = np.zeros(variables)
        objective[:channels] = probability * n - tp
        solution = milp(objective, integrality=np.ones(variables), bounds=bounds,
                        constraints=constraints,
                        options={"time_limit": 60, "mip_rel_gap": 0.0})
        if solution.status != 0 or solution.x is None:
            raise AssertionError(f"cohort optimization not proved optimal: {solution.message}")
        selected = np.flatnonzero(solution.x[:channels] > .5)
        warning_count = int(n[selected].sum())
        true_count = int(tp[selected].sum())
        next_probability = true_count / warning_count
        history.append({"iteration": iteration, "warnings": warning_count,
                        "assigned_tp": true_count, "precision": next_probability,
                        "fractional_objective": float(solution.fun)})
        if abs(next_probability - probability) < 1e-9:
            if abs(solution.fun) > 1e-6:
                raise AssertionError("fractional optimum certificate differs")
            ids = sorted(stats.iloc[selected].channel_id.astype(str).tolist())
            return ids, {"iterations": history, "optimal_precision": next_probability,
                         "candidate_channels": channels}
        probability = next_probability
    raise AssertionError("fractional optimization did not converge")


def summarize(frame: pd.DataFrame) -> dict:
    n = len(frame)
    assigned = int((frame.outcome.eq("matched_known_episode") &
                    frame.split_status.eq("assigned")).sum())
    counts = frame.outcome.value_counts()
    per_channel = frame.channel_id.value_counts()
    per_type = frame.sensor_type.value_counts()
    return {
        "warnings": n,
        "assigned_tp": assigned,
        "purged_boundary_tp": int((frame.outcome.eq("matched_known_episode") &
                                   frame.split_status.ne("assigned")).sum()),
        "known_no_target": int(counts.get("known_no_target", 0)),
        "unknown_target": int(counts.get("unknown_target", 0)),
        "assigned_precision_lower_bound": assigned / n if n else None,
        "channels": int(frame.channel_id.nunique()),
        "sensor_types": int(frame.sensor_type.nunique()),
        "months": int(frame.prediction_time.dt.month.nunique()),
        "max_channel_share": float(per_channel.max() / n) if n else None,
        "max_type_share": float(per_type.max() / n) if n else None,
        "by_type": {key: int(value) for key, value in per_type.items()},
    }


def run(output: Path, *, min_warnings: int = 250, min_channels: int = 20,
        min_types: int = 5, min_type_warnings: int = 10, min_months: int = 8,
        max_channel_share: float = .25, max_type_share: float = .75,
        candidate_min_warnings: int = 2) -> dict:
    if output.exists():
        raise FileExistsError(output)
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        validation = scored_warnings(db, 2025)
        stats = channel_stats(validation, candidate_min_warnings)
        ids, proof = optimize(
            stats, min_warnings=min_warnings, min_channels=min_channels,
            min_types=min_types, min_type_warnings=min_type_warnings,
            min_months=min_months, max_channel_share=max_channel_share,
            max_type_share=max_type_share)
        prior = scored_warnings(db, 2024)
    output.mkdir(parents=True)
    selected = set(ids)
    metrics = {}
    for year, frame in ((2024, prior), (2025, validation)):
        subset = frame.loc[frame.channel_id.isin(selected)].copy()
        subset.to_parquet(output / f"selected_warnings_{year}.parquet", index=False)
        metrics[year] = {"full_stream": summarize(frame),
                         "selected_channels": summarize(subset),
                         "assigned_recall_over_all_known_episodes":
                             summarize(subset)["assigned_tp"] / FULL_ASSIGNED_EPISODES[year]}
    measured = metrics[2025]["selected_channels"]
    if (measured["warnings"] < min_warnings or measured["channels"] < min_channels
            or measured["sensor_types"] < min_types or measured["months"] < min_months
            or measured["max_channel_share"] > max_channel_share + 1e-9
            or measured["max_type_share"] > max_type_share + 1e-9
            or abs(measured["assigned_precision_lower_bound"] -
                   proof["optimal_precision"]) > 1e-9):
        raise AssertionError("saved cohort differs from optimized constraints")
    type_counts = validation.loc[validation.channel_id.isin(selected)].sensor_type.value_counts()
    if int(type_counts.ge(min_type_warnings).sum()) < min_types:
        raise AssertionError("minimum per-type support differs")
    result = {
        "status": "posthoc_2025_outcome_selected_cohort_not_independent_validation",
        "selection_outcome_year": 2025,
        "constraints": {"candidate_min_warnings": candidate_min_warnings,
                        "min_warnings": min_warnings, "min_channels": min_channels,
                        "min_types": min_types, "min_type_warnings": min_type_warnings,
                        "min_months": min_months,
                        "max_channel_share": max_channel_share,
                        "max_type_share": max_type_share},
        "selected_channel_ids": ids,
        "optimization": proof,
        "source_warning_sha256": {str(year): sha256(path)
                                  for year, path in WARNINGS.items()},
        "years": metrics,
        "test_2026_read": False,
        "data_2021_read": False,
        "limits": [
            "The cohort was optimized on open 2025 outcomes; its 2025 Precision is optimistic in-sample selection.",
            "Backward 2024 evaluation is not an independent prospective test.",
            "Only complete channel warning streams are selected; this is not a full-population metric.",
            "Unknown warning outcomes remain unresolved, never labeled negative.",
        ],
    }
    (output / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({year: item["selected_channels"] for year, item in metrics.items()},
                     ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("output/ml-experiment-posthoc-cohort-v1"))
    parser.add_argument("--min-warnings", type=int, default=250)
    parser.add_argument("--min-channels", type=int, default=20)
    parser.add_argument("--min-types", type=int, default=5)
    parser.add_argument("--min-type-warnings", type=int, default=10)
    parser.add_argument("--min-months", type=int, default=8)
    parser.add_argument("--max-channel-share", type=float, default=.25)
    parser.add_argument("--max-type-share", type=float, default=.75)
    parser.add_argument("--candidate-min-warnings", type=int, default=2)
    args = parser.parse_args()
    run(args.output, min_warnings=args.min_warnings, min_channels=args.min_channels,
        min_types=args.min_types, min_type_warnings=args.min_type_warnings,
        min_months=args.min_months, max_channel_share=args.max_channel_share,
        max_type_share=args.max_type_share,
        candidate_min_warnings=args.candidate_min_warnings)
