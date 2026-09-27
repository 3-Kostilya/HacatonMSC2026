"""Warning evaluation is separate from causal decisions and keeps full Recall."""

from datetime import datetime, timedelta
import unittest

import pandas as pd

from analysis.audit_q2_recovery_reset_a import assess_emitted


class RecoveryEvaluationTests(unittest.TestCase):
    def fixture(self):
        start = datetime(2025, 1, 1)
        decisions = [
            {
                "channel_id": "a",
                "prediction_time": start + timedelta(hours=h),
                "sensor_type": "Датчик дыма",
                "warning_emitted": True,
            }
            for h in range(3)
        ]
        labels = pd.DataFrame(decisions).drop(columns="warning_emitted")
        labels["target"] = [1, 1, 0]
        labels["target_episode_id"] = ["one", "one", None]
        labels["label_available_at"] = start + timedelta(hours=3)
        full = pd.DataFrame({"target_episode_id": ["one", "unavailable", "below"]})
        return decisions, labels, full

    def test_one_warning_one_episode_duplicate_is_not_second_true_positive(self):
        decisions, labels, full = self.fixture()
        metrics, alerts, matched = assess_emitted(decisions, labels, full, 10)
        self.assertEqual(metrics["matched_episodes"], 1)
        self.assertEqual(metrics["precision"], 1 / 3)
        self.assertEqual(metrics["full_episode_recall"], 1 / 3)
        self.assertEqual(
            alerts.outcome.tolist(),
            ["matched_episode", "duplicate_episode_warning", "no_target_in_horizon"],
        )
        self.assertEqual(matched, {"one"})

    def test_unknown_labels_or_dropped_keys_are_not_silently_accepted(self):
        decisions, labels, full = self.fixture()
        with self.assertRaises(ValueError):
            assess_emitted(decisions, labels.iloc[:2], full, 10)
        labels.loc[2, "target"] = -1
        with self.assertRaises(ValueError):
            assess_emitted(decisions, labels, full, 10)


if __name__ == "__main__":
    unittest.main()
