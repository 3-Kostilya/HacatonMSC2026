"""Protect the strict episode targets and full-population denominator."""

import unittest

from analysis.audit_v2_validation_start import summarize


def point(*, threshold: float, matched: int, emitted: int) -> dict:
    precision = matched / emitted
    return {
        "eligible_positive_episodes": 261,
        "matched_episodes": matched,
        "emitted_warnings": emitted,
        "unmatched_warnings": emitted - matched,
        "threshold": threshold,
        "episode_precision": precision,
        "episode_recall": matched / 261,
        "episode_f1": 2 * precision * matched / 261 / (precision + matched / 261),
    }


class ValidationStartTest(unittest.TestCase):
    def test_strict_targets_use_all_assigned_episodes(self) -> None:
        curve = [
            point(threshold=0.1, matched=252, emitted=360),
            point(threshold=0.2, matched=250, emitted=350),
            point(threshold=0.3, matched=251, emitted=350),
        ]
        result = summarize(curve, all_episodes=500)
        self.assertTrue(result["requirements_feasible_on_checked_grid"])
        self.assertEqual(result["selected_threshold_if_feasible"], 0.3)

    def test_conditional_recall_cannot_hide_coverage_limit(self) -> None:
        result = summarize([point(threshold=0.9, matched=200, emitted=250)],
                           all_episodes=2142)
        self.assertFalse(result["requirements_feasible_on_checked_grid"])
        self.assertIsNone(result["selected_threshold_if_feasible"])
        self.assertLess(result["diagnostic_best_full_f1_on_checked_grid"][
            "full_episode_recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
