"""Training weights and shared-denominator specialist policy regression checks."""
import unittest

import numpy as np
import pandas as pd

from analysis.ml_experiment_specialists import (
    choose_joint, engineered_input, training_weights,
)


class SpecialistsTest(unittest.TestCase):
    def test_equal_episode_weight_does_not_reward_long_positive_windows(self):
        frame = pd.DataFrame({"target": [1, 1, 1, 0, 0, 0],
                              "target_episode_id": ["a", "a", "b", None, None, None]})
        weights = training_weights(frame, "episode")
        self.assertAlmostEqual(weights[:2].sum(), weights[2])
        self.assertAlmostEqual(weights[:3].sum(), weights[3:].sum())

    def test_contrast_uses_disjoint_previous_window_and_excludes_labels(self):
        data = {"sensor_type": ["smoke"], "target": [1], "channel_id": ["x"]}
        for prefix in ["event_count", "technical_message_count", "normal_message_count"]:
            for hours, count in [(1, 4), (6, 14), (24, 50), (168, 338)]:
                data[f"{prefix}_{hours}h"] = [count]
        names = [key for key in data if key not in {"target", "channel_id"}]
        result = engineered_input(pd.DataFrame(data), names)
        self.assertAlmostEqual(result.loc[0, "contrast__event_count_1_6"],
                               np.log1p(4)-np.log1p(2), places=6)
        self.assertNotIn("target", result)
        self.assertNotIn("channel_id", result)

    def test_joint_choice_uses_full_episode_denominator_and_precision_constraint(self):
        empty = {"model": None, "threshold": None, "matched_episodes": 0, "emitted_warnings": 0}
        options = {
            "smoke": [empty, {"model": "many", "threshold": .4,
                                "matched_episodes": 100, "emitted_warnings": 200},
                      {"model": "precise", "threshold": .8,
                       "matched_episodes": 40, "emitted_warnings": 50}],
            "phase": [empty, {"model": "phase", "threshold": .6,
                                "matched_episodes": 20, "emitted_warnings": 20}],
        }
        chosen, metrics = choose_joint(options, 1000, "f1")
        self.assertEqual(chosen["smoke"]["model"], "many")
        self.assertAlmostEqual(metrics["full_episode_recall"], .12)
        chosen, metrics = choose_joint(options, 1000, "precision70")
        self.assertEqual(chosen["smoke"]["model"], "precise")
        self.assertGreaterEqual(metrics["episode_precision"], .7)


if __name__ == "__main__":
    unittest.main()
