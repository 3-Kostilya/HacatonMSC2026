"""Small exact checks for the explicitly post-hoc cohort optimizer."""
from __future__ import annotations

import unittest

import pandas as pd

from analysis.ml_experiment_posthoc_cohort import optimize, summarize


class PosthocCohortTests(unittest.TestCase):
    def test_fractional_optimum_with_diversity_constraints(self):
        stats = pd.DataFrame({
            "channel_id": ["a", "b", "c"],
            "sensor_type": ["Датчик дыма", "Состояние фазы", "Состояние фазы"],
            "warnings": [2, 2, 2],
            "assigned_tp": [2, 1, 0],
            "months": [[1], [2], [1]],
        })
        ids, proof = optimize(
            stats, min_warnings=3, min_channels=2, min_types=2,
            min_type_warnings=1, min_months=2,
            max_channel_share=.75, max_type_share=.75)
        self.assertEqual(ids, ["a", "b"])
        self.assertEqual(proof["optimal_precision"], .75)

    def test_unknown_and_purged_are_not_assigned_hits(self):
        frame = pd.DataFrame({
            "channel_id": ["a", "b", "c"],
            "prediction_time": pd.to_datetime(["2025-01-01"] * 3),
            "sensor_type": ["Датчик дыма"] * 3,
            "outcome": ["matched_known_episode", "matched_known_episode",
                        "unknown_target"],
            "split_status": ["assigned", "purged_boundary", "assigned"],
        })
        result = summarize(frame)
        self.assertEqual(result["assigned_tp"], 1)
        self.assertEqual(result["purged_boundary_tp"], 1)
        self.assertEqual(result["unknown_target"], 1)
        self.assertEqual(result["assigned_precision_lower_bound"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
