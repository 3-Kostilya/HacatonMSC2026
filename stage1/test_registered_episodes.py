"""Behavioral checks for B2 event ordering and R1 episode boundaries."""

from datetime import datetime
import unittest

from stage1.state_labeling.registered_episodes import StateEvent, build_episodes


def event(row_id, day, hour, text, *, channel="c", sensor_type="Датчик дыма", alarm=False):
    return StateEvent(row_id, channel, sensor_type, datetime(2025, 1, day, hour), text, alarm)


class RegisteredEpisodeTests(unittest.TestCase):
    def test_repeats_and_false_alarm_do_not_create_or_close_episode(self):
        result = build_episodes(
            [
                event(1, 1, 0, "Норма"),
                event(2, 1, 1, "Неисправен", alarm=False),
                event(3, 1, 2, "Неисправен", alarm=False),
                event(4, 1, 3, "Рычаг сдернут"),
                event(5, 1, 4, "Норма"),
            ]
        )
        self.assertEqual(len(result.episodes), 1)
        episode = result.episodes[0]
        self.assertEqual(episode.fault_message_count, 2)
        self.assertEqual(episode.onset_status, "candidate_new_onset")
        self.assertEqual(episode.end_status, "exact_norma_after_uncertainty")
        self.assertEqual(episode.end_at, datetime(2025, 1, 1, 4))

    def test_first_fault_stale_normal_and_return_are_distinct(self):
        result = build_episodes(
            [
                event(1, 1, 0, "Неисправен"),
                event(2, 1, 1, "Норма"),
                event(3, 1, 2, "Неисправен"),
                event(4, 1, 3, "Норма"),
                event(5, 10, 0, "Неисправен"),
            ]
        )
        self.assertEqual(
            [row.onset_status for row in result.episodes],
            ["left_censored", "candidate_new_onset", "stale_normal"],
        )

    def test_same_timestamp_conflict_ignores_row_id_order(self):
        for rows in (
            [event(1, 1, 1, "Норма"), event(2, 1, 1, "Неисправен")],
            [event(1, 1, 1, "Неисправен"), event(2, 1, 1, "Норма")],
        ):
            result = build_episodes([event(0, 1, 0, "Норма"), *rows])
            self.assertEqual(result.episodes[0].onset_status, "conflicted_same_timestamp")
            self.assertIsNone(result.episodes[0].end_at)
            self.assertEqual(result.message_counts["conflicting_target_timestamps"], 1)

    def test_unknown_text_clears_prior_normal_but_environmental_text_does_not(self):
        uncertain = build_episodes(
            [event(1, 1, 0, "Норма"), event(2, 1, 1, "Обесточен"),
             event(3, 1, 2, "Неисправен")]
        )
        self.assertEqual(uncertain.episodes[0].onset_status, "uncertain_prior_state")
        neutral = build_episodes(
            [event(1, 1, 0, "Норма"), event(2, 1, 1, "Обнаружен дым"),
             event(3, 1, 2, "Неисправен")]
        )
        self.assertEqual(neutral.episodes[0].onset_status, "candidate_new_onset")

    def test_unknown_type_fault_is_not_positive(self):
        result = build_episodes([event(1, 1, 0, "Неисправен", sensor_type=None)])
        self.assertEqual(result.episodes, [])
        self.assertEqual(result.message_counts["unknown_type_fault_rows"], 1)

    def test_archive_boundary_never_inherits_a_normal_or_closes_open_episode(self):
        result = build_episodes(
            [
                StateEvent(1, "c", "Датчик дыма", datetime(2020, 12, 31, 20), "Норма"),
                StateEvent(2, "c", "Датчик дыма", datetime(2022, 1, 1), "Неисправен"),
            ]
        )
        self.assertEqual(result.episodes[0].onset_status, "left_censored")

    def test_open_episode_is_severed_on_type_change(self):
        result = build_episodes(
            [event(1, 1, 0, "Неисправен"),
             event(2, 1, 1, "Норма", sensor_type="Газовый датчик")]
        )
        self.assertEqual(result.episodes[0].end_status, "open_type_or_archive_boundary")
        self.assertIsNone(result.episodes[0].end_at)

    def test_missing_type_is_uncertainty_without_forcing_second_episode(self):
        result = build_episodes(
            [event(1, 1, 0, "Неисправен"),
             event(2, 1, 1, "Неопределен", sensor_type=None),
             event(3, 1, 2, "Неисправен"),
             event(4, 1, 3, "Норма")]
        )
        self.assertEqual(len(result.episodes), 1)
        self.assertEqual(result.episodes[0].fault_message_count, 2)
        self.assertEqual(result.episodes[0].end_status, "exact_norma_after_uncertainty")

    def test_out_of_order_input_is_rejected(self):
        with self.assertRaises(ValueError):
            build_episodes([event(1, 1, 2, "Норма"), event(2, 1, 1, "Неисправен")])


if __name__ == "__main__":
    unittest.main()
