from datetime import datetime, timedelta
from types import SimpleNamespace
import unittest

from analysis.run_synthetic_benchmark import check_scenarios
from stage1.contracts import Decision


BASE = datetime(2026, 3, 8)


def episode(identifier: str, decision: Decision, confirmed_at: datetime):
    return SimpleNamespace(
        episode_id=identifier,
        decision=decision,
        start_at=confirmed_at,
        confirmed_at=confirmed_at,
        metadata={"scenario_id": "scenario"},
    )


def manifest(expected: str) -> dict:
    return {
        "scenarios": [
            {
                "scenario_id": "scenario",
                "detector_applicability": "numeric",
                "expected_detector_behavior": expected,
                "intervention_start": BASE.isoformat(),
                "end": (BASE + timedelta(hours=4)).isoformat(),
                "expected_cadence_seconds": 3600,
            }
        ]
    }


class ScenarioCheckTests(unittest.TestCase):
    def test_positive_scenario_rejects_early_or_duplicate_warning(self):
        for extra_time in (BASE - timedelta(minutes=1), BASE + timedelta(minutes=1)):
            with self.subTest(extra_time=extra_time):
                result = check_scenarios(
                    {},
                    manifest("candidate"),
                    [
                        episode("first", Decision.CANDIDATE, BASE),
                        episode("extra", Decision.CANDIDATE, extra_time),
                    ],
                )
                self.assertFalse(result["all_passed"])

    def test_candidate_before_intervention_does_not_pass_positive_scenario(self):
        result = check_scenarios(
            {},
            manifest("candidate"),
            [episode("early", Decision.CANDIDATE, BASE - timedelta(seconds=1))],
        )

        detail = result["details"][0]
        self.assertFalse(detail["passed"])
        self.assertEqual(detail["candidate_count"], 1)
        self.assertEqual(detail["timely_candidate_count"], 0)
        self.assertEqual(detail["episodes"][0]["episode_id"], "early")

    def test_context_scenario_rejects_extra_candidate_and_reports_every_episode(self):
        result = check_scenarios(
            {},
            manifest("single_common_context_candidate"),
            [
                episode("first", Decision.CANDIDATE, BASE),
                episode("extra", Decision.CANDIDATE, BASE + timedelta(minutes=1)),
            ],
        )

        detail = result["details"][0]
        self.assertFalse(detail["passed"])
        self.assertEqual(detail["candidate_count"], 2)
        self.assertEqual(detail["timely_candidate_count"], 2)
        self.assertEqual([item["episode_id"] for item in detail["episodes"]], ["first", "extra"])


if __name__ == "__main__":
    unittest.main()
