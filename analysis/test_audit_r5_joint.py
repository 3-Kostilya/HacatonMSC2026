"""Small decision-boundary tests for the independent R5 review."""

import unittest

from analysis.audit_r5_joint import _selected


class R5JointAuditTests(unittest.TestCase):
    def test_equal_warning_count_prefers_more_distinct_episodes(self) -> None:
        curve = [
            {"matched_episodes": 58, "unmatched_warnings": 177,
             "episode_precision": 0.24, "threshold": 0.91},
            {"matched_episodes": 60, "unmatched_warnings": 172,
             "episode_precision": 0.26, "threshold": 0.93},
            {"matched_episodes": 79, "unmatched_warnings": 234,
             "episode_precision": 0.25, "threshold": 0.88},
        ]
        self.assertEqual(_selected(curve, 177)["matched_episodes"], 60)
        self.assertEqual(_selected(curve, 234)["matched_episodes"], 79)

    def test_no_eligible_threshold_is_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "no curve point"):
            _selected([{"matched_episodes": 1, "unmatched_warnings": 5,
                        "episode_precision": 0.5, "threshold": 0.9}], 4)


if __name__ == "__main__":
    unittest.main()
