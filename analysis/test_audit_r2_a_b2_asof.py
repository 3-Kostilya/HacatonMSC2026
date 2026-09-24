"""Independent R2 episode as-of audit arithmetic."""

from datetime import datetime, timedelta
import unittest

from analysis.audit_r2_a_b2_asof import _expected, _matches
from stage1.features.r2 import CompletedEpisode


class R2AsOfAuditTests(unittest.TestCase):
    def test_episode_appears_only_at_its_end(self) -> None:
        end = datetime(2025, 6, 10, 12)
        episode = CompletedEpisode("c", end - timedelta(hours=2), end)
        before = _expected([episode], "c", end - timedelta(microseconds=1))
        at = _expected([episode], "c", end)
        self.assertEqual(before, (None, 0, None))
        self.assertEqual(at, (0.0, 1, 7200.0))
        self.assertTrue(
            _matches(
                {
                    "last_completed_episode_end_age_seconds": 0.0,
                    "completed_episode_count_168h": 1,
                    "completed_episode_mean_duration_seconds_168h": 7200.0,
                    "episode_history_status": "unambiguous_completed_only",
                },
                at,
            )
        )

    def test_missing_year_severs_previous_episode_age(self) -> None:
        episode = CompletedEpisode("c", datetime(2020, 12, 30), datetime(2020, 12, 31))
        self.assertEqual(_expected([episode], "c", datetime(2022, 1, 2)), (None, 0, None))


if __name__ == "__main__":
    unittest.main()
