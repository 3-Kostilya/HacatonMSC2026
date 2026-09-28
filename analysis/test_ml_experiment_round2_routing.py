"""Global F1 optimization, rare-type defaults and native score tie checks."""
import itertools
import unittest

import numpy as np
import pandas as pd

from analysis.ml_experiment_round2_routing import (
    META, N, add_fusions, margin32, optimize, regularized_grid, routed_frame,
)


class RoutingTest(unittest.TestCase):
    def test_fractional_optimizer_matches_exhaustive_grid(self):
        options = {"a": [{"matched_episodes": tp, "emitted_warnings": warnings}
                         for tp, warnings in [(1, 1), (20, 30), (50, 200)]],
                   "b": [{"matched_episodes": tp, "emitted_warnings": warnings}
                         for tp, warnings in [(3, 4), (10, 20), (40, 100)]]}
        fallback = {"matched_episodes": 4, "emitted_warnings": 10}
        _, metrics = optimize(options, fallback)
        exhaustive = max(2*(sum(x["matched_episodes"] for x in pair)+4)/
                         (N["tune"]+sum(x["emitted_warnings"] for x in pair)+10)
                         for pair in itertools.product(*options.values()))
        self.assertAlmostEqual(metrics["full_episode_f1"], exhaustive)

    def test_unknown_type_and_rare_type_share_exact_same_fallback(self):
        frame = pd.DataFrame({"channel_id": ["a", "b", "c"], "sensor_type": ["smoke", "rare", "new"],
                              "target": [0, 0, 0], "target_episode_id": [None]*3,
                              "prediction_time": [pd.Timestamp("2024-01-01")]*3,
                              "label_available_at": [pd.NaT]*3,
                              "pool": [.4, .8, .8], "special": [.9, .2, .2]})
        policy = {"f": {"smoke": {"model": "special", "threshold": .8},
                         "__default__": {"model": "pool", "threshold": .7}}}
        out = routed_frame(frame, policy)
        self.assertEqual(out.loc[1, "route_f"], out.loc[2, "route_f"])
        self.assertEqual(out.loc[1, "score_f"], out.loc[2, "score_f"])
        self.assertTrue(out[META].equals(frame[META]))

    def test_float32_threshold_tie_matches_native_score_comparison(self):
        p = np.float32(.731)
        threshold = float(p)+1e-9
        self.assertTrue(margin32(np.array([p]), threshold)[0] >= 0)

    def test_regularization_uses_smaller_support_for_more_shrinkage(self):
        weak = regularized_grid(.9, .6, 20)
        strong = regularized_grid(.9, .6, 1000)
        self.assertLess(max(abs(x-.9) for x in weak if x < 1),
                        max(abs(x-.9) for x in strong if x < 1))

    def test_fusion_centers_frozen_champions_and_ignores_targets(self):
        frame = pd.DataFrame({"a": [.2, .2], "b": [.8, .8], "target": [0, 1]})
        out = add_fusions(frame, [{"model": "a", "threshold": .2}, {"model": "b", "threshold": .8}])
        np.testing.assert_array_equal(out.fusion__mean.to_numpy(), [.5, .5])
        np.testing.assert_array_equal(out.fusion__max.to_numpy(), [.5, .5])


if __name__ == "__main__":
    unittest.main()
