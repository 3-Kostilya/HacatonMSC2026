"""Independent warning implementation preserves boundaries and error accounting."""

import unittest

import pandas as pd

from analysis.verify_q2_b_metrics_a import chronological_warnings, episode_classes


class MetricsAcceptanceTests(unittest.TestCase):
    def points(self):
        return pd.DataFrame(
            {
                "channel_id": ["a"] * 4 + ["b"],
                "prediction_time": pd.to_datetime(
                    [
                        "2025-01-01 00:00",
                        "2025-01-01 01:00",
                        "2025-01-02 00:00",
                        "2025-01-02 01:00",
                        "2025-01-01 01:00",
                    ]
                ),
                "sensor_type": ["smoke"] * 5,
                "target": [0, 1, 1, 1, 1],
                "target_episode_id": [None, "early", "next", "next", "b-one"],
                "label_available_at": pd.to_datetime(
                    [
                        "2025-01-02",
                        "2025-01-01 02:00",
                        "2025-01-02 02:00",
                        "2025-01-02 02:00",
                        "2025-01-01 02:00",
                    ],
                    format="mixed",
                ),
                "score": [0.9] * 5,
            }
        )

    def test_false_warning_suppresses_true_and_exact_24h_is_allowed(self):
        result = chronological_warnings(self.points().sample(frac=1, random_state=5))
        self.assertEqual(len(result), 3)
        self.assertEqual(
            result.outcome.tolist(), ["no_target_in_horizon", "matched_episode", "matched_episode"]
        )
        self.assertEqual(set(result.target_episode_id.dropna()), {"next", "b-one"})

    def test_duplicate_nonbinary_and_invalid_horizon_fail(self):
        points = self.points()
        variants = [pd.concat([points, points.iloc[[0]]], ignore_index=True)]
        unknown = points.copy()
        unknown.loc[0, "target"] = -1
        variants.append(unknown)
        wrong = points.copy()
        wrong.loc[1, "label_available_at"] = wrong.loc[1, "prediction_time"]
        variants.append(wrong)
        for data in variants:
            with self.assertRaises(ValueError):
                chronological_warnings(data)

    def test_partition_covers_full_denominator_not_only_available(self):
        episodes = pd.DataFrame(
            {
                "target_episode_id": list("abcd"),
                "channel_id": ["one"] * 4,
                "sensor_type": ["smoke"] * 4,
            }
        )
        result = episode_classes(episodes, set("bcd"), set("cd"), {"d"})
        self.assertEqual(
            result.error_class.tolist(),
            ["unavailable", "below_threshold", "cooldown_suppressed", "matched"],
        )
        with self.assertRaises(ValueError):
            episode_classes(episodes, set("bcd"), set("cd"), {"a"})


if __name__ == "__main__":
    unittest.main()
