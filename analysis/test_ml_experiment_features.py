import numpy as np
import pandas as pd
import unittest

from analysis.ml_experiment_features import engineered_input


def feature_frame():
    names = ["sensor_type", "last_observation_age_seconds", "last_completed_episode_end_age_seconds"]
    for family in ("event_count", "technical_message_count", "registered_fault_text_count", "normal_message_count",
                   "environmental_alarm_count", "unknown_state_count", "state_transitions"):
        names.extend(f"{family}_{window}h" for window in (1, 6, 24, 168))
    frame = pd.DataFrame({name: [0, 0] for name in names})
    frame["sensor_type"] = ["smoke", "smoke"]
    return frame, names


class ExperimentFeaturesTest(unittest.TestCase):
    def test_recent_burst_differs_from_old_history_with_same_total(self):
        frame, names = feature_frame()
        frame["event_count_6h"] = [6, 6]
        frame["event_count_1h"] = [6, 0]
        out = engineered_input(frame, names)
        self.assertGreater(out.loc[0, "disjoint_log_rate__event_count_1v6h"], 0)
        self.assertLess(out.loc[1, "disjoint_log_rate__event_count_1v6h"], 0)
        self.assertNotIn("channel_id", out)
        self.assertNotIn("target", out)

    def test_missing_history_is_not_a_zero_count_and_age_decays(self):
        frame, names = feature_frame()
        frame["event_count_1h"] = [np.nan, 0]
        frame["last_observation_age_seconds"] = [0, 24 * 3600]
        out = engineered_input(frame, names)
        self.assertEqual(out.loc[0, "log1p__event_count_1h"], -1)
        self.assertEqual(out.loc[1, "log1p__event_count_1h"], 0)
        self.assertTrue(np.isclose(out.loc[1, "freshness24__last_observation_age_seconds"], np.exp(-1)))
        self.assertTrue(out.drop(columns="sensor_type").apply(np.isfinite).all().all())

    def test_episode_weights_assign_equal_total_mass(self):
        from analysis.ml_experiment_pooled import training_weights
        frame = pd.DataFrame({"target": [1, 1, 1, 1, 1, 0, 0],
                              "target_episode_id": ["a", "a", "a", "b", "b", None, None]})
        weights, classes = training_weights(frame, "episode")
        self.assertAlmostEqual(float(weights[:3].sum()), float(weights[3:5].sum()))
        self.assertEqual(classes, [1, 2 / 5])
        self.assertEqual(weights[5:].tolist(), [1, 1])
