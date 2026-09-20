from datetime import datetime, timedelta
from dataclasses import replace
import unittest

from stage1.contracts import Decision, Episode
from stage1.episodes import EpisodeAggregationConfig, consolidate_episodes


BASE = datetime(2026, 1, 1)


def candidate(
    minute,
    *,
    episode_id=None,
    channel="c-1",
    anomaly="numeric_level_shift",
    cause="local",
):
    start = BASE + timedelta(minutes=minute)
    return Episode(
        episode_id=episode_id or f"ep-{channel}-{minute}-{anomaly}",
        channel_id=channel,
        sensor_type="Датчик температуры",
        sensor_group="numeric",
        anomaly_type=anomaly,
        decision=Decision.CANDIDATE,
        start_at=start,
        confirmed_at=start + timedelta(minutes=1),
        ruleset_version="test",
        evidence=(f"hit_at={minute}",),
        observation_quality=(),
        cause_hypothesis=cause,
    )


class EpisodeAggregationTests(unittest.TestCase):
    def test_merges_close_detections_of_same_cause(self):
        result = consolidate_episodes(
            [candidate(0), candidate(5)],
            config=EpisodeAggregationConfig(merge_gap=timedelta(minutes=5)),
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].metadata["merged_detection_count"], 2)
        self.assertEqual(result[0].confirmed_at, BASE + timedelta(minutes=1))
        self.assertEqual(
            result[0].metadata["last_confirmed_at"],
            (BASE + timedelta(minutes=6)).isoformat(sep=" "),
        )
        self.assertEqual(len(result[0].evidence), 2)

    def test_never_merges_different_channels_anomalies_or_causes(self):
        result = consolidate_episodes(
            [
                candidate(0),
                candidate(1, channel="c-2"),
                candidate(2, anomaly="rapid_switching"),
                candidate(3, cause="environmental"),
            ],
            config=EpisodeAggregationConfig(merge_gap=timedelta(hours=1)),
        )
        self.assertEqual(len(result), 4)

    def test_recovery_closes_episode(self):
        recovery = BASE + timedelta(minutes=8)
        result = consolidate_episodes([candidate(0)], recovery_by_channel={"c-1": recovery})
        self.assertEqual(result[0].end_at, recovery)
        self.assertIn("recovery_observed", result[0].evidence)

    def test_suppresses_recurrence_during_cooldown(self):
        result = consolidate_episodes(
            [candidate(0), candidate(10)],
            recovery_by_channel={"c-1": BASE + timedelta(minutes=5)},
            config=EpisodeAggregationConfig(
                merge_gap=timedelta(minutes=1),
                suppression_window=timedelta(minutes=10),
            ),
        )
        self.assertEqual(len(result), 1)

    def test_non_candidates_pass_through(self):
        ordinary = Episode(
            episode_id="ordinary",
            channel_id="c-1",
            sensor_type="Датчик температуры",
            sensor_group="numeric",
            anomaly_type="numeric_level_shift",
            decision=Decision.NO_CANDIDATE,
            start_at=BASE,
            confirmed_at=BASE,
            ruleset_version="test",
            evidence=(),
            observation_quality=(),
        )
        self.assertEqual(consolidate_episodes([ordinary]), [ordinary])

    def test_later_open_recurrence_keeps_merged_episode_open(self):
        first = candidate(0)
        first = replace(first, end_at=BASE + timedelta(minutes=2))
        second = candidate(5)
        result = consolidate_episodes(
            [first, second],
            config=EpisodeAggregationConfig(merge_gap=timedelta(minutes=5)),
        )
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].end_at)

    def test_overlapping_closed_recurrence_keeps_later_known_end(self):
        first = candidate(0)
        first = replace(first, end_at=BASE + timedelta(minutes=10))
        second = replace(candidate(5), end_at=BASE + timedelta(minutes=7))
        result = consolidate_episodes(
            [first, second],
            config=EpisodeAggregationConfig(merge_gap=timedelta(minutes=5)),
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].end_at, BASE + timedelta(minutes=10))

    def test_merge_preserves_later_last_confirmed_metadata(self):
        first = replace(
            candidate(0),
            metadata={"last_confirmed_at": (BASE + timedelta(minutes=20)).isoformat(sep=" ")},
        )
        result = consolidate_episodes(
            [first, candidate(5)],
            config=EpisodeAggregationConfig(merge_gap=timedelta(minutes=5)),
        )
        self.assertEqual(
            result[0].metadata["last_confirmed_at"],
            (BASE + timedelta(minutes=20)).isoformat(sep=" "),
        )

    def test_merge_preserves_current_aggregate_last_confirmed_metadata(self):
        current = replace(
            candidate(5),
            metadata={"last_confirmed_at": (BASE + timedelta(minutes=20)).isoformat(sep=" ")},
        )
        result = consolidate_episodes(
            [candidate(0), current],
            config=EpisodeAggregationConfig(merge_gap=timedelta(minutes=5)),
        )
        self.assertEqual(
            result[0].metadata["last_confirmed_at"],
            (BASE + timedelta(minutes=20)).isoformat(sep=" "),
        )


if __name__ == "__main__":
    unittest.main()
