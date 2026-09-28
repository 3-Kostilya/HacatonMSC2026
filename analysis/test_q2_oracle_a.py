"""Exact oracle scheduling requires explicit episode/window invariants."""

from datetime import timedelta
from itertools import combinations
import random
import unittest

import pandas as pd

from analysis.audit_q2_oracle_a import optimal_schedule


def points(hours, *, channel="a"):
    start = pd.Timestamp("2025-01-01")
    return pd.DataFrame(
        [
            {
                "channel_id": channel,
                "prediction_time": start + timedelta(hours=h),
                "sensor_type": "Датчик дыма",
                "target": 1,
                "target_episode_id": f"{channel}-{i}",
                "label_available_at": start + timedelta(hours=h + 1),
            }
            for i, h in enumerate(hours)
        ]
    )


class OracleSchedulingTests(unittest.TestCase):
    def test_exact_cooldown_boundary_is_allowed(self):
        selected, channels = optimal_schedule(points([0, 23, 24, 47, 48]))
        self.assertEqual([r["prediction_time"].hour for r in selected], [0, 0, 0])
        self.assertEqual(channels[0]["dynamic_program_matches"], 3)

    def test_arbitrary_order_and_separate_channels(self):
        frame = pd.concat([points([0, 1, 24]), points([0, 1, 24], channel="b")], ignore_index=True)
        selected, rows = optimal_schedule(frame.sample(frac=1, random_state=4))
        self.assertEqual(len(selected), 4)
        self.assertEqual([r["greedy_matches"] for r in rows], [2, 2])

    def test_many_positive_hours_do_not_reuse_one_episode(self):
        frame = points([0, 1, 2, 23])
        frame["target_episode_id"] = "one"
        frame["label_available_at"] = pd.Timestamp("2025-01-02")
        selected, _ = optimal_schedule(frame)
        self.assertEqual(len(selected), 1)

    def test_long_episode_span_rejects_simple_scheduling_proof(self):
        frame = points([0, 12])
        frame["target_episode_id"] = "one"
        frame["label_available_at"] = pd.Timestamp("2025-01-01 13:00")
        with self.assertRaisesRegex(ValueError, "episode span"):
            optimal_schedule(frame, cooldown=timedelta(hours=12))

    def test_invalid_lead_duplicate_unknown_and_metadata_are_rejected(self):
        base = points([0, 1])
        variants = []
        duplicate = pd.concat([base, base.iloc[[0]]], ignore_index=True)
        variants.append(duplicate)
        unknown = base.copy()
        unknown.loc[0, "target"] = 0
        variants.append(unknown)
        bad = base.copy()
        bad.loc[0, "label_available_at"] = bad.loc[0, "prediction_time"]
        variants.append(bad)
        cross = base.copy()
        cross["target_episode_id"] = "same"
        cross.loc[1, "channel_id"] = "b"
        variants.append(cross)
        for frame in variants:
            with self.subTest(frame=frame.to_dict("records")), self.assertRaises(ValueError):
                optimal_schedule(frame)

    def test_greedy_and_dp_equal_exhaustive_search(self):
        rng = random.Random(1729)
        for _ in range(40):
            hours = sorted(rng.sample(range(100), 7))
            optimum = 0
            for n in range(len(hours) + 1):
                for subset in combinations(hours, n):
                    if all(b - a >= 24 for a, b in zip(subset, subset[1:])):
                        optimum = max(optimum, n)
            selected, _ = optimal_schedule(points(hours))
            self.assertEqual(len(selected), optimum)


if __name__ == "__main__":
    unittest.main()
