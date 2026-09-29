"""Run B3's detector-independent episode state-machine acceptance scenarios."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage1.score_episodes import (  # noqa: E402
    KindTruth,
    ScoreObservation,
    build_notifications,
    build_score_episodes,
    evaluate_anomaly_kinds,
)


BASE = datetime(2026, 4, 1)


def point(channel, hour, score, *, kind="level_shift", status="eligible", reasons=()):
    return ScoreObservation(
        channel_id=channel,
        as_of=BASE + timedelta(hours=hour),
        anomaly_type=kind,
        score=score,
        status=status,
        status_reasons=tuple(reasons),
        sensor_type="Датчик температуры",
        sensor_group="numeric_environment",
        method="fixed-b3-score",
        method_version="b3-fixture-v1",
        evidence=(f"fixture_hour={hour}",) if status == "eligible" else (),
    )


def build_acceptance_report() -> dict:
    fragmented = [
        point("sustained", hour, score)
        for hour, score in enumerate((0.1, 0.85, 0.9, 0.95, 0.88, 0.92, 0.86, 0.7))
    ]
    recurrence = [
        point("recurrence", hour, score)
        for hour, score in enumerate((0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9))
    ]
    unknown = [point("unknown-gap", hour, 0.9) for hour in range(3)]
    unknown.append(point("unknown-gap", 3, None, status="unknown", reasons=("source_gap",)))
    unknown.extend(point("unknown-gap", hour, 0.2) for hour in (4, 5))
    kind_rows = [point("kind", hour, 0.9, kind="variance") for hour in range(3)]

    fragmented_episodes = build_score_episodes(fragmented)
    recurrence_episodes = build_score_episodes(recurrence)
    unknown_episodes = build_score_episodes(unknown)
    kind_episodes = build_score_episodes(kind_rows)
    notifications = build_notifications(recurrence_episodes, cooldown=timedelta(days=1))
    kind_report = evaluate_anomaly_kinds(
        [KindTruth("variance-truth", "kind", BASE, BASE + timedelta(hours=4), "variance")],
        kind_episodes,
    )
    prefix = build_score_episodes(fragmented[:5])[0]
    extended = fragmented_episodes[0]
    checks = {
        "sustained_impact_is_one_episode": len(fragmented_episodes) == 1,
        "start_and_confirmation_are_distinct": extended.start_at < extended.confirmed_at,
        "future_preserves_episode_identity": (
            prefix.episode_id,
            prefix.start_at,
            prefix.confirmed_at,
        )
        == (extended.episode_id, extended.start_at, extended.confirmed_at),
        "recovery_allows_two_episodes": len(recurrence_episodes) == 2
        and recurrence_episodes[0].end_at is not None,
        "cooldown_does_not_hide_new_episode": len(notifications.emitted) == 2,
        "unknown_gap_does_not_close_episode": len(unknown_episodes) == 1
        and unknown_episodes[0].end_at is None
        and "source_gap" in unknown_episodes[0].observation_quality,
        "impact_and_kind_are_scored_separately": kind_report["impact_recall"] == 1.0
        and kind_report["kind_accuracy_on_detected"] == 1.0,
        "year_2021_absent": all(
            row.as_of.year != 2021 for row in (*fragmented, *recurrence, *unknown, *kind_rows)
        ),
    }
    return {
        "validation_version": "b3-score-sequences-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "episode_counts": {
            "sustained": len(fragmented_episodes),
            "recurrence": len(recurrence_episodes),
            "unknown_gap": len(unknown_episodes),
            "kind": len(kind_episodes),
        },
        "episodes": [
            episode.to_record()
            for episode in (
                *fragmented_episodes,
                *recurrence_episodes,
                *unknown_episodes,
                *kind_episodes,
            )
        ],
        "notifications": [asdict(item) for item in notifications.emitted],
        "kind_evaluation": kind_report,
        "limitations": [
            "A3 anomaly_scores are not available; fixed score sequences were used",
            "Thresholds are fixture defaults, not tuned operational thresholds",
            "2021 is excluded",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/b3/b3-validation.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new path")
    result = build_acceptance_report()
    if not result["passed"]:
        raise RuntimeError(f"B3 acceptance failed: {result['checks']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"passed": True, **result["episode_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
