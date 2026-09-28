from dataclasses import replace
from datetime import datetime, timedelta
import unittest

from stage1.score_episodes import (
    KindTruth,
    ScoreEpisodeConfig,
    ScoreObservation,
    build_notifications,
    build_score_episodes,
    evaluate_anomaly_kinds,
    score_observation_from_record,
)


BASE = datetime(2026, 1, 1)


def point(hour, score, *, kind="level_shift", status="eligible", reasons=(), channel="c1"):
    return ScoreObservation(
        channel_id=channel,
        as_of=BASE + timedelta(hours=hour),
        anomaly_type=kind,
        score=score,
        status=status,
        status_reasons=reasons,
        sensor_type="Датчик температуры",
        sensor_group="numeric",
        method="synthetic-score",
        method_version="v1",
        evidence=(f"point={hour}",) if status == "eligible" else (),
    )


class ScoreEpisodeTests(unittest.TestCase):
    def test_three_high_scores_confirm_one_episode_without_fragmentation(self):
        rows = [point(hour, score) for hour, score in enumerate((0.1, 0.85, 0.9, 0.95, 0.88, 0.9))]
        episodes = build_score_episodes(rows)
        self.assertEqual(len(episodes), 1)
        episode = episodes[0]
        self.assertEqual(episode.start_at, BASE + timedelta(hours=1))
        self.assertEqual(episode.confirmed_at, BASE + timedelta(hours=3))
        self.assertIsNone(episode.end_at)
        self.assertEqual(episode.score, 0.95)

    def test_hysteresis_and_sustained_recovery_close_at_confirmation(self):
        scores = (0.85, 0.9, 0.95, 0.6, 0.3, 0.7, 0.35, 0.3, 0.2)
        episode = build_score_episodes(point(hour, score) for hour, score in enumerate(scores))[0]
        self.assertEqual(episode.end_at, BASE + timedelta(hours=8))
        self.assertEqual(
            episode.metadata["recovery_started_at"],
            (BASE + timedelta(hours=6)).isoformat(sep=" "),
        )
        self.assertIn("sustained_recovery_confirmed", episode.evidence)

    def test_two_episodes_after_recovery_are_both_kept_despite_notification_cooldown(self):
        scores = (0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9)
        episodes = build_score_episodes(point(hour, score) for hour, score in enumerate(scores))
        self.assertEqual(len(episodes), 2)
        self.assertIsNotNone(episodes[0].end_at)
        self.assertIsNone(episodes[1].end_at)
        notifications = build_notifications(episodes, cooldown=timedelta(days=1))
        self.assertEqual(len(notifications.emitted), 2)
        self.assertNotEqual(episodes[0].episode_id, episodes[1].episode_id)

    def test_notification_replay_of_same_episode_is_suppressed(self):
        episode = build_score_episodes([point(0, 0.9), point(1, 0.9), point(2, 0.9)])[0]
        result = build_notifications([episode, episode], cooldown=timedelta(hours=1))
        self.assertEqual(len(result.emitted), 1)
        self.assertEqual(result.suppressed_duplicate_episode_ids, (episode.episode_id,))

    def test_unknown_does_not_close_active_episode_or_turn_into_zero(self):
        rows = [point(0, 0.9), point(1, 0.9), point(2, 0.9)]
        rows.append(point(3, None, status="unknown", reasons=("source_gap",)))
        rows.extend([point(4, 0.2), point(5, 0.2)])
        episode = build_score_episodes(rows)[0]
        self.assertIsNone(episode.end_at)
        self.assertIn("source_gap", episode.observation_quality)

    def test_large_gap_breaks_entry_confirmation(self):
        config = ScoreEpisodeConfig(maximum_confirmation_gap=timedelta(hours=2))
        rows = [point(0, 0.9), point(1, 0.9), point(10, 0.9)]
        self.assertEqual(build_score_episodes(rows, config), [])

    def test_anomaly_types_are_independent_state_machines(self):
        rows = []
        for hour in range(3):
            rows.extend([point(hour, 0.9), point(hour, 0.9, kind="variance")])
        episodes = build_score_episodes(rows)
        self.assertEqual(
            {episode.anomaly_type for episode in episodes}, {"level_shift", "variance"}
        )

    def test_future_rows_do_not_change_past_start_or_confirmation(self):
        prefix = [point(0, 0.9), point(1, 0.9), point(2, 0.9), point(3, 0.7)]
        early = build_score_episodes(prefix)[0]
        extended = build_score_episodes(prefix + [point(4, 0.1), point(5, 0.1), point(6, 0.1)])[0]
        self.assertEqual(
            (early.episode_id, early.start_at, early.confirmed_at),
            (extended.episode_id, extended.start_at, extended.confirmed_at),
        )
        self.assertIsNone(early.end_at)
        self.assertEqual(extended.end_at, BASE + timedelta(hours=6))

    def test_kind_accuracy_is_separate_from_impact_detection(self):
        correct = build_score_episodes([point(0, 0.9), point(1, 0.9), point(2, 0.9)])[0]
        wrong = replace(correct, episode_id="wrong-kind", anomaly_type="variance")
        truth = [KindTruth("t1", "c1", BASE, BASE + timedelta(hours=4), "level_shift")]
        correct_report = evaluate_anomaly_kinds(truth, [correct])
        wrong_report = evaluate_anomaly_kinds(truth, [wrong])
        self.assertEqual(correct_report["impact_recall"], 1.0)
        self.assertEqual(wrong_report["impact_recall"], 1.0)
        self.assertEqual(correct_report["kind_accuracy_on_detected"], 1.0)
        self.assertEqual(wrong_report["kind_accuracy_on_detected"], 0.0)

    def test_invalid_unavailable_score_and_duplicate_timestamp_are_rejected(self):
        with self.assertRaises(ValueError):
            point(0, 0.0, status="unknown", reasons=("gap",))
        duplicate = [point(0, 0.9), point(0, 0.95), point(1, 0.9)]
        with self.assertRaisesRegex(ValueError, "duplicate score timestamp"):
            build_score_episodes(duplicate)

    def test_m0_score_record_adapter_is_ready_for_a3(self):
        observation = score_observation_from_record(
            {
                "channel_id": "c1",
                "as_of": "2026-01-01 00:00:00",
                "method": "cusum",
                "method_version": "v1",
                "score": 0.9,
                "score_status": "eligible",
                "score_reasons": [],
                "evidence": ["persistent_shift"],
                "sensor_type": "Датчик температуры",
                "sensor_group": "numeric_environment",
                "anomaly_type": "level_shift",
            }
        )
        self.assertEqual(observation.anomaly_type, "level_shift")
        self.assertEqual(observation.as_of, BASE)


if __name__ == "__main__":
    unittest.main()
