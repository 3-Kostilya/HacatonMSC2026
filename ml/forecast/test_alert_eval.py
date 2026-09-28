"""Operational alert semantics for R4 validation."""

from datetime import datetime, timedelta
import unittest

import pandas as pd

from ml.forecast.alert_eval import evaluate_alerts


BASE = datetime(2025, 1, 1)


def row(channel: str, hour: int, target: int, episode: str | None,
        onset_hour: int, score: float) -> dict:
    return {
        "channel_id": channel,
        "prediction_time": BASE + timedelta(hours=hour),
        "sensor_type": "Датчик дыма",
        "target": target,
        "target_episode_id": episode,
        "label_available_at": BASE + timedelta(hours=onset_hour),
        "rule_score": score,
    }


class AlertEvaluationTest(unittest.TestCase):
    def test_false_warning_suppresses_later_true_warning_on_same_channel(self) -> None:
        frame = pd.DataFrame([
            row("A", 0, 0, None, 24, 5),
            row("A", 2, 1, "E1", 20, 5),
            row("B", 3, 1, "E2", 21, 5),
        ])
        result, alerts = evaluate_alerts(frame, "rule_score", 4, channel_days=2)
        self.assertEqual(result["emitted_warnings"], 2)
        self.assertEqual(result["matched_episodes"], 1)
        self.assertEqual(result["suppressed_positive_score_rows"], 1)
        self.assertEqual(result["unmatched_warnings"], 1)
        self.assertEqual(alerts.loc[alerts.channel_id == "B", "lead_hours"].iloc[0], 18)

    def test_one_episode_is_not_counted_twice_after_cooldown(self) -> None:
        frame = pd.DataFrame([
            row("A", 0, 1, "E1", 24, 5),
            row("A", 12, 1, "E1", 24, 5),
        ])
        result, _ = evaluate_alerts(
            frame, "rule_score", 4, channel_days=2, cooldown=timedelta(hours=12)
        )
        self.assertEqual(result["matched_episodes"], 1)
        self.assertEqual(result["duplicate_episode_warnings"], 1)
        self.assertEqual(result["episode_recall"], 1)

    def test_invalid_future_label_is_rejected(self) -> None:
        frame = pd.DataFrame([row("A", 0, 1, "E1", 25, 5)])
        with self.assertRaisesRegex(ValueError, "lead time"):
            evaluate_alerts(frame, "rule_score", 4, channel_days=1)


if __name__ == "__main__":
    unittest.main()
