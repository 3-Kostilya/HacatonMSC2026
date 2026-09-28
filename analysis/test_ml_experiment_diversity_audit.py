"""Selection for the diversity audit must be outcome-blind and deterministic."""
from __future__ import annotations

import unittest

import pandas as pd

from analysis.ml_experiment_diversity_audit import counts, select_keys


class DiversityAuditTests(unittest.TestCase):
    def test_rejects_outcome_bearing_input(self):
        frame = pd.DataFrame({
            "channel_id": ["a"],
            "prediction_time": [pd.Timestamp("2025-01-01")],
            "sensor_type": ["Датчик дыма"],
            "outcome": ["matched_known_episode"],
        })
        with self.assertRaises(ValueError):
            select_keys(frame)

    def test_one_per_channel_month_and_order_independent(self):
        frame = pd.DataFrame({
            "channel_id": ["a", "a", "b", "a"],
            "prediction_time": pd.to_datetime([
                "2025-01-01", "2025-01-02", "2025-01-03", "2025-02-01"]),
            "sensor_type": ["Датчик дыма"] * 4,
        })
        first = select_keys(frame)
        second = select_keys(frame.iloc[::-1].reset_index(drop=True))
        self.assertEqual(len(first), 3)
        self.assertEqual(first[["channel_id", "prediction_time"]].to_dict("records"),
                         second[["channel_id", "prediction_time"]].to_dict("records"))

    def test_type_month_cap_and_unknown_outcomes(self):
        frame = pd.DataFrame({
            "channel_id": [str(index) for index in range(40)],
            "prediction_time": pd.to_datetime(["2025-01-01"] * 40),
            "sensor_type": ["Датчик дыма"] * 40,
        })
        selected = select_keys(frame)
        self.assertEqual(len(selected), 30)
        scored = pd.DataFrame({
            "channel_id": ["a", "b", "c"],
            "prediction_time": pd.to_datetime(["2025-01-01"] * 3),
            "sensor_type": ["Датчик дыма"] * 3,
            "outcome": ["matched_known_episode", "known_no_target", "unknown_target"],
            "split_status": ["assigned"] * 3,
        })
        result = counts(scored)
        self.assertEqual(result["known_no_target"], 1)
        self.assertEqual(result["unknown_target"], 1)
        self.assertEqual(result["precision_lower_bound"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
