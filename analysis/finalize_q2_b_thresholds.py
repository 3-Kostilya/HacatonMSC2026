"""Apply the agreed episode-level goal rule to both Q2 validation grids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.run_q2_b_expanded import ALL_VALIDATION_EPISODES, SCORE_NAMES
from analysis.train_r4_discrete_baselines import read_json, sha256
from ml.forecast.v2_threshold import choose_threshold


def run(*, experiment: Path, refinement: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    original = read_json(experiment / "curves.json")
    dense = read_json(refinement / "curves.json")
    refined = read_json(refinement / "report.json")
    if (refined["source_experiment_manifest_sha256"]
            != sha256(experiment / "manifest.json")
            or set(original) != set(SCORE_NAMES)
            or set(dense) != set(SCORE_NAMES)):
        raise ValueError("threshold curves or source experiment differ")
    decisions = {}
    for name in SCORE_NAMES:
        merged = {float(item["threshold"]): item
                  for item in [*original[name], *dense[name]]}
        curve = list(merged.values())
        decision = choose_threshold(curve,
                                    full_positive_episodes=ALL_VALIDATION_EPISODES)
        diagnostic = decision["diagnostic_best_full_f1"]
        decisions[name] = {
            "goal_check": decision,
            "max_precision_checked": max(x["episode_precision"] for x in curve),
            "max_full_recall_checked": max(x["matched_episodes"]
                                           / ALL_VALIDATION_EPISODES for x in curve),
            "max_full_recall_at_precision_above_0_7": max(
                (x["matched_episodes"] / ALL_VALIDATION_EPISODES for x in curve
                 if x["episode_precision"] > 0.7), default=0.0),
            "diagnostic_type_counts_source": (
                "original_experiment" if diagnostic["threshold"] in
                {x["threshold"] for x in original[name]} else "refinement"),
        }
    report = {
        "schema_version": "q2-b-final-validation-threshold-choice-v1",
        "source_experiment_manifest_sha256": sha256(experiment / "manifest.json"),
        "source_refinement_manifest_sha256": sha256(refinement / "manifest.json"),
        "full_positive_episodes": ALL_VALIDATION_EPISODES,
        "models": decisions,
        "any_model_meets_both_strict_goals": any(
            x["goal_check"]["requirements_feasible_on_checked_grid"]
            for x in decisions.values()),
        "threshold_frozen_for_independent_test": False,
        "test_data_read": False,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--refinement", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({"any_model_meets_both_strict_goals":
                      result["any_model_meets_both_strict_goals"]}), flush=True)


if __name__ == "__main__":
    main()
