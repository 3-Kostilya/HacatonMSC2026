"""Checks that Q2 threshold evaluation keeps the full episode denominator."""

from __future__ import annotations

import unittest

import pandas as pd

from analysis.run_q2_b_expanded import selected_metrics, threshold_grid
from ml.forecast.alert_eval import evaluate_alerts


class Q2ExpandedExperimentTest(unittest.TestCase):
    def test_filtered_warning_evaluation_matches_canonical_counts(self) -> None:
        frame = pd.DataFrame({
            "channel_id": ["a", "a", "a", "b", "b"],
            "prediction_time": pd.to_datetime([
                "2025-01-01 00:00", "2025-01-01 01:00", "2025-01-02 03:00",
                "2025-01-01 00:00", "2025-01-01 02:00",
            ]),
            "sensor_type": ["Датчик дыма"] * 5,
            "target": [0, 1, 1, 0, 1],
            "target_episode_id": [None, "one", "two", None, "three"],
            "label_available_at": pd.to_datetime([
                None, "2025-01-01 10:00", "2025-01-02 10:00", None,
                "2025-01-01 12:00",
            ]),
            "score": [0.9, 0.8, 0.8, 0.2, 0.95],
        })
        compact, alerts = selected_metrics(frame, "score", 0.8, 2)
        canonical, expected = evaluate_alerts(
            frame.rename(columns={"score": "catboost_score"}),
            "catboost_score", 0.8, channel_days=2)
        self.assertEqual(compact["matched_episodes"], canonical["matched_episodes"])
        self.assertEqual(compact["emitted_warnings"], canonical["emitted_warnings"])
        self.assertEqual(compact["suppressed_positive_score_rows"], canonical[
            "suppressed_positive_score_rows"])
        self.assertEqual(len(alerts), len(expected))
        self.assertEqual(compact["eligible_positive_episodes"], 1359)
        self.assertEqual(compact["episode_recall"], compact["matched_episodes"] / 1359)
    def test_score_only_threshold_grid_is_deterministic(self) -> None:
        import numpy as np

        values = np.arange(1_000, dtype=float) / 1_000
        self.assertEqual(threshold_grid(values), threshold_grid(values[::-1]))
        self.assertEqual(len(threshold_grid(values)), 12)
