"""Run frozen baseline variants on the synthetic holdout and evaluate them."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage1.context_events import detect_shared_state  # noqa: E402
from stage1.contracts import Decision, NormalizedEvent, Origin  # noqa: E402
from stage1.detectors import (  # noqa: E402
    DiscreteDetectorConfig,
    NumericDetectorConfig,
    RULESET_VERSION,
)
from stage1.evaluation import EvaluationInterval, TruthEpisode, compare_variants  # noqa: E402
from stage1.pipeline import evaluate_channel  # noqa: E402


DEFAULT_EVENTS = ROOT / "output" / "stage1" / "synthetic_holdout_events.jsonl"
DEFAULT_TRUTH = ROOT / "output" / "stage1" / "synthetic_holdout_truth_manifest.json"
DEFAULT_OUTPUT = ROOT / "output" / "stage1" / "synthetic_benchmark.json"


VARIANTS = {
    "baseline": (NumericDetectorConfig(), DiscreteDetectorConfig(), True, True, True),
    "numeric_mad_5_5": (
        NumericDetectorConfig(mad_multiplier=5.5),
        DiscreteDetectorConfig(),
        True,
        True,
        True,
    ),
    "numeric_mad_6_5": (
        NumericDetectorConfig(mad_multiplier=6.5),
        DiscreteDetectorConfig(),
        True,
        True,
        True,
    ),
    "numeric_confirmation_2": (
        NumericDetectorConfig(min_sustained=2),
        DiscreteDetectorConfig(),
        True,
        True,
        True,
    ),
    "numeric_confirmation_4": (
        NumericDetectorConfig(min_sustained=4),
        DiscreteDetectorConfig(),
        True,
        True,
        True,
    ),
    "discrete_transitions_3": (
        NumericDetectorConfig(),
        DiscreteDetectorConfig(transition_count=3),
        True,
        True,
        True,
    ),
    "discrete_transitions_5": (
        NumericDetectorConfig(),
        DiscreteDetectorConfig(transition_count=5),
        True,
        True,
        True,
    ),
    "discrete_window_8m": (
        NumericDetectorConfig(),
        DiscreteDetectorConfig(transition_window=timedelta(minutes=8)),
        True,
        True,
        True,
    ),
    "discrete_window_12m": (
        NumericDetectorConfig(),
        DiscreteDetectorConfig(transition_window=timedelta(minutes=12)),
        True,
        True,
        True,
    ),
    "without_numeric": (
        NumericDetectorConfig(),
        DiscreteDetectorConfig(),
        False,
        True,
        True,
    ),
    "without_discrete": (
        NumericDetectorConfig(),
        DiscreteDetectorConfig(),
        True,
        False,
        True,
    ),
    "without_context": (
        NumericDetectorConfig(),
        DiscreteDetectorConfig(),
        True,
        True,
        False,
    ),
}


def _event(record: dict[str, Any]) -> NormalizedEvent:
    return NormalizedEvent(
        channel_id=record["channel_id"],
        timestamp=datetime.fromisoformat(record["timestamp"]),
        raw_value=record["raw_value"],
        alarm=record["alarm"],
        sensor_type=record["sensor_type"],
        source=record["source"],
        event_id=record.get("event_id"),
        object_id=record.get("object_id"),
        numeric_value=record.get("numeric_value"),
        quality_flags=tuple(record.get("quality_flags", ())),
    )


def load_inputs(events_path: Path, truth_path: Path):
    events = [
        _event(json.loads(line))
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads(truth_path.read_text(encoding="utf-8"))
    by_scenario: dict[str, list[NormalizedEvent]] = {}
    for event in events:
        scenario_id = event.source.removeprefix("synthetic:")
        by_scenario.setdefault(scenario_id, []).append(event)
    return by_scenario, manifest


def evaluation_inputs(by_scenario, manifest, observability_statuses=None):
    truths = []
    intervals = []
    for scenario in manifest["scenarios"]:
        scenario_id = scenario["scenario_id"]
        events = by_scenario[scenario_id]
        cadence_seconds = scenario.get("expected_cadence_seconds")
        cadence = timedelta(seconds=cadence_seconds or 1)
        if scenario["detector_applicability"] == "context":
            object_id = next(event.object_id for event in events if event.object_id)
            channel_id = f"context:{object_id}"
        else:
            channel_id = scenario["channel_ids"][0]
        start_at = datetime.fromisoformat(scenario["intervention_start"])
        end_at = datetime.fromisoformat(scenario["end"])
        truths.append(
            TruthEpisode(
                episode_id="truth:" + scenario_id,
                channel_id=channel_id,
                scenario_id=scenario_id,
                start_at=start_at,
                end_at=end_at,
                expected_cadence=cadence,
                sensor_type=scenario.get("sensor_type", events[0].sensor_type),
                is_control=scenario["label"] == "control",
            )
        )
        interval_start = min(event.timestamp for event in events)
        interval_end = max(end_at + cadence, max(event.timestamp for event in events) + cadence)
        actual_status = (observability_statuses or {}).get(scenario_id)
        interval_status = actual_status or ("unknown" if cadence_seconds is None else "include")
        intervals.append(
            EvaluationInterval(
                channel_id=channel_id,
                scenario_id=scenario_id,
                start_at=interval_start,
                end_at=interval_end,
                status=interval_status,
                sensor_type=scenario.get("sensor_type", events[0].sensor_type),
                reasons=("cadence_unknown",) if cadence_seconds is None else (),
            )
        )
    return truths, intervals


def run_variant(by_scenario, manifest, settings):
    numeric_cfg, discrete_cfg, use_numeric, use_discrete, use_context = settings
    episodes = []
    for scenario in manifest["scenarios"]:
        scenario_id = scenario["scenario_id"]
        events = by_scenario[scenario_id]
        applicability = scenario["detector_applicability"]
        results = []
        if applicability == "numeric" and use_numeric:
            results = list(evaluate_channel(events, numeric_config=numeric_cfg).detector_results)
        elif applicability == "discrete" and use_discrete:
            results = list(evaluate_channel(events, discrete_config=discrete_cfg).detector_results)
        elif applicability == "context" and use_context:
            results = [detect_shared_state(events, state="Нет связи")]
        elif applicability.startswith("observability_only"):
            results = list(evaluate_channel(events, expected_cadence=None).detector_results)
        for result in results:
            episodes.append(
                replace(
                    result,
                    origin=Origin.SYNTHETIC,
                    metadata={**result.metadata, "scenario_id": scenario_id},
                )
            )
    return episodes


def check_scenarios(by_scenario, manifest, episodes):
    """Verify each manifest expectation against an actually executed code path."""

    by_id: dict[str, list] = {}
    for episode in episodes:
        by_id.setdefault(episode.metadata.get("scenario_id"), []).append(episode)
    checks = []
    for scenario in manifest["scenarios"]:
        scenario_id = scenario["scenario_id"]
        applicability = scenario["detector_applicability"]
        scenario_episodes = by_id.get(scenario_id, [])
        candidates = [
            episode for episode in scenario_episodes if episode.decision is Decision.CANDIDATE
        ]
        cadence_seconds = scenario.get("expected_cadence_seconds")
        intervention_start = datetime.fromisoformat(scenario["intervention_start"])
        evaluation_end = datetime.fromisoformat(scenario["end"])
        match_end = evaluation_end + timedelta(seconds=cadence_seconds or 0)
        timely_candidates = [
            episode
            for episode in candidates
            if intervention_start <= episode.confirmed_at < match_end
        ]
        actual = (
            ",".join(episode.decision.value for episode in scenario_episodes)
            if scenario_episodes
            else "not_executed"
        )
        observability = None
        if applicability.startswith("observability_only"):
            outcome = evaluate_channel(by_scenario[scenario_id], expected_cadence=None)
            observability = outcome.observability.status.value
            passed = observability == "unknown" and bool(scenario_episodes) and not candidates
        else:
            expected = scenario["expected_detector_behavior"]
            if expected in {"candidate", "single_common_context_candidate"}:
                passed = len(candidates) == len(timely_candidates) == 1
            elif expected == "no_candidate":
                passed = bool(scenario_episodes) and all(
                    episode.decision is Decision.NO_CANDIDATE for episode in scenario_episodes
                )
            else:
                raise ValueError(
                    f"Unsupported expected detector behavior for {scenario_id!r}: {expected!r}"
                )
        checks.append(
            {
                "scenario_id": scenario_id,
                "scenario_name": scenario.get("scenario_name", scenario_id),
                "applicability": applicability,
                "expected": scenario["expected_detector_behavior"],
                "actual": actual,
                "episode_count": len(scenario_episodes),
                "candidate_count": len(candidates),
                "timely_candidate_count": len(timely_candidates),
                "candidate_confirmed_at": [
                    episode.confirmed_at.isoformat(sep=" ") for episode in candidates
                ],
                "expected_candidate_interval": {
                    "start_at": intervention_start.isoformat(sep=" "),
                    "end_at_exclusive": match_end.isoformat(sep=" "),
                },
                "episodes": [
                    {
                        "episode_id": episode.episode_id,
                        "decision": episode.decision.value,
                        "start_at": episode.start_at.isoformat(sep=" "),
                        "confirmed_at": episode.confirmed_at.isoformat(sep=" "),
                    }
                    for episode in scenario_episodes
                ],
                "observability": observability,
                "passed": passed,
            }
        )
    return {
        "executed": sum(item["actual"] != "not_executed" for item in checks),
        "passed": sum(item["passed"] for item in checks),
        "total": len(checks),
        "all_passed": all(item["passed"] for item in checks),
        "details": checks,
    }


def build_benchmark(events_path: Path, truth_path: Path) -> dict[str, Any]:
    by_scenario, manifest = load_inputs(events_path, truth_path)
    runs = {
        name: run_variant(by_scenario, manifest, settings) for name, settings in VARIANTS.items()
    }
    baseline_checks = check_scenarios(by_scenario, manifest, runs["baseline"])
    observability_statuses = {
        item["scenario_id"]: "unknown"
        for item in baseline_checks["details"]
        if item["applicability"].startswith("observability_only")
        and "observability=unknown" in item["actual"]
    }
    truths, intervals = evaluation_inputs(by_scenario, manifest, observability_statuses)
    result = compare_variants(truths, runs, intervals=intervals, baseline="baseline")
    result["scenario_checks"] = baseline_checks
    result["inputs"] = {
        "simulation_version": manifest["simulation_version"],
        "ruleset_version": RULESET_VERSION,
        "suite": manifest["suite"],
        "seed": manifest["seed"],
        "scenarios_declared": len(manifest["scenarios"]),
        "scenarios_executed": baseline_checks["executed"],
        "scenarios_passed": baseline_checks["passed"],
        "events": sum(len(events) for events in by_scenario.values()),
    }
    result["variant_parameters"] = {
        name: {
            "numeric": {
                "baseline_size": settings[0].baseline_size,
                "min_sustained": settings[0].min_sustained,
                "mad_multiplier": settings[0].mad_multiplier,
            },
            "discrete": {
                "repeated_state_count": settings[1].repeated_state_count,
                "transition_count": settings[1].transition_count,
                "transition_window_seconds": int(settings[1].transition_window.total_seconds()),
            },
            "components": {
                "numeric": settings[2],
                "discrete": settings[3],
                "context": settings[4],
            },
        }
        for name, settings in VARIANTS.items()
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_benchmark(args.events, args.truth)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    baseline = result["variants"]["baseline"]["report"]
    print(
        json.dumps(
            {
                "inputs": result["inputs"],
                "baseline": {
                    key: baseline[key]
                    for key in (
                        "true_positives",
                        "false_positives",
                        "false_negatives",
                        "precision",
                        "recall",
                        "f1",
                        "delay_seconds_median",
                        "delay_seconds_p90",
                    )
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
