"""Regression checks for the research-only no-known-hit-loss rule screen."""
from __future__ import annotations

import unittest

import pandas as pd

from analysis.ml_experiment_round7_rule_screen import candidates


class RuleScreenTests(unittest.TestCase):
    def test_unknowns_do_not_count_as_negatives_or_expand_positive_boundary(self):
        frame = pd.DataFrame({
            "sensor_type": ["Датчик дыма"] * 8,
            "outcome": ["matched_known_episode", "matched_known_episode",
                        "known_no_target", "known_no_target", "known_no_target",
                        "known_no_target", "known_no_target", "unknown_target"],
            "unknown_state_count_168h": [3, 21, 22, 23, 24, 25, 26, 99],
            "target_episode_id": ["a", "b", None, None, None, None, None, None],
        })
        rules = candidates(frame)
        found = next(rule for rule in rules
                     if rule["feature"] == "unknown_state_count_168h"
                     and rule["direction"] == "above")
        self.assertEqual(found["bound"], 21)
        self.assertEqual(found["removed_negative_2024"], 5)
        self.assertEqual(found["removed_unknown_2024"], 1)
        self.assertEqual(found["removed_positive_2024"], 0)
        self.assertFalse(any(rule["feature"] == "target_episode_id" for rule in rules))


if __name__ == "__main__":
    unittest.main()
