"""Evaluate a synthetic holdout artifact under frozen stage1-eval-v1 rules."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage1.contracts import Decision, Episode, Origin  # noqa: E402
from stage1.evaluation import compare_variants, evaluate_episodes  # noqa: E402


def _episode(record: dict[str, Any]) -> Episode:
    return Episode(
        episode_id=record["episode_id"],
        channel_id=record["channel_id"],
        sensor_type=record["sensor_type"],
        sensor_group=record["sensor_group"],
        anomaly_type=record["anomaly_type"],
        decision=Decision(record["decision"]),
        start_at=datetime.fromisoformat(record["start_at"]),
        confirmed_at=datetime.fromisoformat(record["confirmed_at"]),
        ruleset_version=record["ruleset_version"],
        evidence=tuple(record.get("evidence", ())),
        observation_quality=tuple(record.get("observation_quality", ())),
        cause_hypothesis=record.get("cause_hypothesis", "unknown"),
        object_id=record.get("object_id"),
        end_at=datetime.fromisoformat(record["end_at"]) if record.get("end_at") else None,
        score=record.get("score"),
        origin=Origin(record.get("origin", "synthetic")),
        metadata=record.get("metadata", {}),
    )


def evaluate_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    truth = payload.get("truth", payload.get("truth_episodes", payload.get("scenarios", ())))
    intervals = payload.get("intervals", payload.get("evaluation_intervals", ()))
    if not intervals and payload.get("scenarios"):
        intervals = [
            {
                "channel_id": channel_id,
                "scenario_id": scenario["scenario_id"],
                "start_at": scenario["intervention_start"],
                "end_at": scenario["end"],
                "status": (
                    "unknown"
                    if str(scenario.get("expected_detector_behavior", "")).lower() == "unknown"
                    else "include"
                ),
                "sensor_type": scenario.get("suite", "unknown"),
                "reasons": ["synthetic_manifest_expected_behavior"],
            }
            for scenario in payload["scenarios"]
            for channel_id in scenario["channel_ids"]
        ]
    variants = payload.get("variants")
    if variants:
        runs = {}
        for name, value in variants.items():
            records = value.get("episodes", ()) if isinstance(value, dict) else value
            runs[name] = [_episode(item) for item in records]
        return compare_variants(
            truth,
            runs,
            intervals=intervals,
            baseline=payload.get("baseline_variant", "baseline"),
        )
    episodes = [_episode(item) for item in payload.get("episodes", ())]
    return evaluate_episodes(truth, episodes, intervals=intervals).to_record()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="synthetic artifact JSON")
    parser.add_argument(
        "--episodes",
        type=Path,
        help="optional Episode JSON array/object to merge with a truth manifest",
    )
    parser.add_argument("--output", type=Path, help="result JSON (stdout when omitted)")
    args = parser.parse_args()
    if not args.artifact.exists():
        parser.error(f"synthetic artifact is not available yet: {args.artifact}")
    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    if args.episodes:
        episode_payload = json.loads(args.episodes.read_text(encoding="utf-8"))
        if isinstance(episode_payload, list):
            payload["episodes"] = episode_payload
        else:
            payload.update(episode_payload)
    if "episodes" not in payload and "variants" not in payload:
        parser.error(
            "truth manifest is ready, but no detector Episode artifact was supplied; "
            "use --episodes to avoid silently treating an absent run as zero warnings"
        )
    result = evaluate_artifact(payload)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
