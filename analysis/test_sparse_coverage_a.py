"""Cached scenarios are diagnostics, not bounds on longer causal history."""

import unittest
from datetime import datetime

from analysis.compare_sparse_coverage_a import summarize
from analysis.verify_sparse_population_a import qa_counts_from_legacy_events
from stage1.features.hourly import FeatureEvent


class CoverageComparisonTests(unittest.TestCase):
    def test_verifier_derives_qa_from_undecorated_frozen_events(self):
        at = datetime(2025, 1, 1)
        numeric = FeatureEvent("gas", at, False, value_numeric=1.5, sensor_type="Газовый датчик")
        epoch = FeatureEvent("gas", at, False, value_state="01.01.1970 03:00:01",
                             sensor_type="Газовый датчик")
        self.assertIsNone(numeric.qa_value_category)
        counts = qa_counts_from_legacy_events([numeric, epoch], at)
        self.assertEqual(counts["qa_gas_alarm_level_candidate_count_168h"], 1)
        self.assertEqual(counts["qa_epoch_value_artifact_count_24h"], 1)
        self.assertIsNone(numeric.qa_value_category)

    def test_reconcile_gain_loss_and_nonexclusive_cached_reasons(self):
        rows = [("train", "t", "smoke", True, True),
                ("validation", "v1", "smoke", False, True),
                ("validation", "v2", "smoke", True, False),
                ("validation", "v3", "phase", True, True)]
        old = [{"split": "validation", "target_episode_id": "v1",
                "reasons_on_every_hour": ["insufficient_history", "baseline_unusable"]}]
        result = summarize(rows, old)["by_split"][1]
        self.assertEqual(result["cached_mechanical_available"], 2)
        self.assertEqual(result["full_past_available"], 2)
        self.assertEqual(result["gained_episodes"], 1)
        self.assertEqual(result["lost_episodes"], 1)
        self.assertEqual(result["gained_by_type"], {"smoke": 1})
        self.assertEqual(result["gained_cached_reasons_on_every_hour"],
                         {"insufficient_history": 1, "baseline_unusable": 1})


if __name__ == "__main__":
    unittest.main()
