from datetime import datetime, timedelta
import unittest

from stage1.contracts import Decision, Episode
from stage1.evaluation import (
    EvaluationInterval,
    TruthEpisode,
    compare_variants,
    evaluate_episodes,
)


BASE = datetime(2026, 2, 1)


def truth(name="t1", start=10, end=20, scenario="shift", **kwargs):
    return TruthEpisode(
        episode_id=name,
        channel_id="c1",
        scenario_id=scenario,
        start_at=BASE + timedelta(minutes=start),
        end_at=BASE + timedelta(minutes=end),
        expected_cadence=timedelta(minutes=5),
        **kwargs,
    )


def warning(name, confirmed, scenario="shift", decision=Decision.CANDIDATE, start=None):
    moment = BASE + timedelta(minutes=confirmed)
    return Episode(
        episode_id=name,
        channel_id="c1",
        sensor_type="temperature",
        sensor_group="numeric",
        anomaly_type="numeric_level_shift",
        decision=decision,
        start_at=BASE + timedelta(minutes=confirmed if start is None else start),
        confirmed_at=moment,
        ruleset_version="test",
        evidence=("hit",) if decision is not Decision.UNKNOWN else (),
        observation_quality=("gap",) if decision is Decision.UNKNOWN else (),
        metadata={"scenario_id": scenario},
    )


class EvaluationTests(unittest.TestCase):
    def test_greedy_one_warning_per_truth_and_duplicate_is_fp(self):
        report = evaluate_episodes(
            [truth(), truth("t2", start=30, end=40)],
            [warning("w2", 17), warning("w1", 12), warning("w3", 34)],
        )
        self.assertEqual([match.warning_episode_id for match in report.matches], ["w1", "w3"])
        self.assertEqual(report.unmatched_warning_ids, ("w2",))
        self.assertEqual((report.true_positives, report.false_positives), (2, 1))

    def test_scope_keys_prevent_cross_scenario_match(self):
        report = evaluate_episodes([truth()], [warning("wrong", 12, scenario="drift")])
        self.assertEqual(
            (report.true_positives, report.false_positives, report.false_negatives), (0, 1, 1)
        )
        self.assertEqual(report.f1, 0.0)

    def test_matching_boundaries_are_inclusive(self):
        start = evaluate_episodes([truth()], [warning("at-start", 10)])
        cadence_end = evaluate_episodes([truth()], [warning("at-end-plus-step", 25)])
        after = evaluate_episodes([truth()], [warning("after", 25.001)])
        self.assertEqual(start.true_positives, 1)
        self.assertEqual(cadence_end.true_positives, 1)
        self.assertEqual((after.false_positives, after.false_negatives), (1, 1))

    def test_unknown_is_coverage_not_true_negative_or_false_positive(self):
        interval = EvaluationInterval(
            channel_id="c1",
            scenario_id="shift",
            start_at=BASE,
            end_at=BASE + timedelta(days=1),
            status="unknown",
        )
        report = evaluate_episodes([truth()], [warning("w", 12)], intervals=[interval])
        self.assertEqual(
            (report.true_positives, report.false_positives, report.false_negatives), (0, 0, 0)
        )
        self.assertEqual(report.ignored_warning_ids, ("w",))
        self.assertEqual(report.coverage["fractions"]["unknown"], 1.0)

    def test_missing_eligibility_scope_is_unknown_not_implicitly_included(self):
        unrelated = EvaluationInterval(
            channel_id="other",
            scenario_id="shift",
            start_at=BASE,
            end_at=BASE + timedelta(days=1),
            status="include",
        )
        report = evaluate_episodes([truth()], [warning("w", 12)], intervals=[unrelated])
        self.assertEqual((report.false_positives, report.false_negatives), (0, 0))
        self.assertEqual(report.ignored_warning_ids, ("w",))

    def test_control_warning_is_false_positive_without_control_fn(self):
        report = evaluate_episodes(
            [truth(is_control=True, scenario="single_spike")],
            [warning("w", 12, scenario="single_spike")],
        )
        self.assertEqual((report.false_positives, report.false_negatives), (1, 0))

    def test_group_context_truth_matches_only_one_channel_warning(self):
        group_truth = truth(
            scenario="context",
            scope_channel_ids=("c1", "c2", "c3"),
        )
        second = warning("w2", 11, scenario="context")
        object.__setattr__(second, "channel_id", "c2")
        report = evaluate_episodes(
            [group_truth],
            [warning("w1", 12, scenario="context"), second],
        )
        self.assertEqual((report.true_positives, report.false_positives), (1, 1))
        self.assertEqual(report.matches[0].warning_episode_id, "w2")

    def test_generator_manifest_truth_schema_is_accepted(self):
        item = {
            "scenario_id": "s-1",
            "suite": "numeric",
            "channel_ids": ["c1"],
            "intervention_start": (BASE + timedelta(minutes=10)).isoformat(),
            "end": (BASE + timedelta(minutes=20)).isoformat(),
            "label": "positive",
            "expected_cadence_seconds": 300,
        }
        report = evaluate_episodes([item], [warning("w", 12, scenario="s-1")])
        self.assertEqual(report.true_positives, 1)

    def test_causal_warning_time_is_confirmation_not_backdated_start(self):
        report = evaluate_episodes([truth()], [warning("late", 30, start=12)])
        self.assertEqual(
            (report.true_positives, report.false_positives, report.false_negatives), (0, 1, 1)
        )

    def test_delay_and_channel_day_rate(self):
        interval = EvaluationInterval(
            channel_id="c1",
            scenario_id="shift",
            start_at=BASE,
            end_at=BASE + timedelta(days=2),
            status="include",
            sensor_type="temperature",
        )
        report = evaluate_episodes(
            [truth()], [warning("matched", 12), warning("duplicate", 13)], intervals=[interval]
        )
        self.assertEqual(report.delay_seconds_median, 120)
        self.assertEqual(report.false_warnings_per_100_channel_days, 50)
        self.assertEqual(report.coverage["by_sensor_type"]["temperature"]["include"], 1)

    def test_isolation_forest_status_is_explicit_when_run_is_absent(self):
        comparison = compare_variants([truth()], {"baseline": [warning("w", 12)]})
        status = comparison["method_comparisons"]["isolation_forest"]
        self.assertEqual(status["status"], "not_comparable")
        self.assertIn("pre-frozen", status["reason"])
        self.assertFalse(comparison["sensitivity_plan"]["complete"])


if __name__ == "__main__":
    unittest.main()
