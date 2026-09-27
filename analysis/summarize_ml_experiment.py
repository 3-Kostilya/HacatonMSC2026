"""Assemble completed reports, without fitting, threshold selection or promotion."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

from analysis.ml_experiment_eval import EVALUATION_VERSION
from analysis.ml_experiment_frontier import summarize


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def compact(metric):
    keys = ["threshold", "matched_episodes", "emitted_warnings", "episode_precision",
            "full_episode_recall", "full_episode_f1", "median_lead_hours",
            "unmatched_warnings_per_1000_channel_days"]
    return {key: metric[key] for key in keys if key in metric}


def run(root: Path) -> dict:
    rows = []
    for family in ["pooled", "history"]:
        report = read(root / family / "report.json")
        for name, experiment in report["experiments"].items():
            rows.append({"family": family, "variant": name,
                         "family_champion_by_2024": name == report["selected_variant"],
                         "tune_2024": compact(experiment["tune"]["selected"]),
                         "transfer_2025": compact(experiment["validation_frozen"])})
    linear = read(root / "linear/report_canonical_v2.json")
    for name, tune in linear["tune"].items():
        rows.append({"family": "linear", "variant": name,
                     "family_champion_by_2024": name == linear["selected_variant"],
                     "tune_2024": compact(tune), "transfer_2025": compact(linear["validation"][name])})
    specialist = read(root / "specialists/report.json")
    for name, metric in specialist["selection"]["tune_metrics"].items():
        rows.append({"family": "specialists", "variant": name,
                     "family_champion_by_2024": name == "f1",
                     "tune_2024": compact(metric), "transfer_2025": compact(specialist["validation"][name])})
    ensemble = read(root / "ensemble/report.json")
    for name, tune in ensemble["tune"].items():
        rows.append({"family": "ensemble", "variant": name,
                     "family_champion_by_2024": name == ensemble["selected_column"],
                     "tune_2024": compact(tune), "transfer_2025": compact(ensemble["validation"][name])})
    # This reports the original 2024 ranking. It does not rerank using 2025.
    champion = max((row for row in rows if row["family_champion_by_2024"]),
                   key=lambda row: row["tune_2024"]["full_episode_f1"])
    refits = {}
    for family in ["linear-refit", "specialists-refit", "online-linear"]:
        report = read(root / family / "report.json")
        metric = report["validation"]["f1"] if family == "specialists-refit" else report["validation"]
        refits[family] = {"selection": report["selection"], "transfer_2025": compact(metric)}
    curves = {}
    for folder in ["frontier", "frontier-additional"]:
        curves.update(read(root / folder / "curves.json"))
    points = [point for curve in curves.values() for point in curve]
    diagnostics = summarize(points)
    diagnostics["grid_points"] = len(points)
    diagnostics["score_columns"] = len(curves)
    baseline = read(Path("output/q2-b-expanded-validation-20260927/report.json"))[
        "threshold_goals"]["linear121"]["diagnostic_best_full_f1"]
    old_mixes = read(root / "saved-score-diagnostics-canonical-v2/report.json")["joint_research_choices"]
    audits = {}
    for folder in ["final-audit", "ensemble-audit", "online-audit"]:
        audit = read(root / folder / "report.json")
        if audit["status"] != "full_metadata_and_canonical_warning_parity_passed":
            raise AssertionError(f"incomplete audit: {folder}")
        audits[folder] = {"status": audit["status"], "cases": len(audit["checks"])}
    models = {}
    for path in sorted(root.glob("*/*")):
        if path.suffix in {".cbm", ".joblib"}:
            with path.open("rb") as stream:
                models[str(path.relative_to(root))] = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {"status": "COMPLETED_OPEN2025_RESEARCH_NOT_DEPLOYED", "branch": "codex/ml-experiment",
              "evaluation_version": EVALUATION_VERSION, "full_episodes_2025": 2142,
              "existing_open2025_reference": compact(baseline),
              "frozen_2024_variants": rows, "global_2024_champion": champion,
              "same_threshold_refits": refits, "optimistic_open2025_diagnostic": diagnostics,
              "optimistic_old_score_type_mix": old_mixes,
              "audits": audits, "model_sha256": models,
              "environment": {"python": platform.python_version(), **{name: version(name) for name in
                  ["numpy", "pandas", "scikit-learn", "catboost", "duckdb", "pyarrow", "joblib"]}},
              "test_2026_read": False, "data_2021_used": False,
              "limitations": ["2025 was already open before this task; no independent confirmation.",
                              "New grid is denser than the historical reference grid.",
                              "Online refits learn only past2025 labels; this is not a fixed-model holdout.",
                              "No model or threshold is promoted based on 2025 scores."]}
    destination = root / "summary.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"global_2024_champion": f"{champion['family']}/{champion['variant']}",
                      "trained_model_artifacts": len(models), "diagnostic_points": len(points),
                      "diagnostic_best": compact(diagnostics["optimistic_best_full_f1"])}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output/ml-experiment"))
    run(parser.parse_args().root)
