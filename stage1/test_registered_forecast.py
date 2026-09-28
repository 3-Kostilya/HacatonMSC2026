"""R3 fixtures for causal state eligibility, outcomes, and split edges."""

from datetime import datetime
import unittest

from stage1.state_labeling.forecast import (
    PredictionPoint,
    RegisteredTargetIndex,
    split_at,
)
from stage1.state_labeling.registered_episodes import RegisteredEpisode, StateEvent
from stage1.state_labeling.rules import TARGET_DEFINITION


SMOKE = "Датчик дыма"


def event(row_id, at, text, *, kind=SMOKE, channel="c"):
    return StateEvent(row_id, channel, kind, at, text, False)


def episode(row_id, start, end=None, *, status="candidate_new_onset", kind=SMOKE):
    return RegisteredEpisode(
        episode_id=f"c:{row_id}",
        channel_id="c",
        sensor_type=kind,
        target_kind=TARGET_DEFINITION["target_kind"],
        start_at=start,
        confirmed_at=start,
        end_at=end,
        onset_status=status,
        end_status="exact_norma" if end else "open_unknown",
        prior_normal_at=None,
        first_row_id=row_id,
        last_fault_at=start,
    )


def make_index(channel, events, episodes, *, observed_until=datetime(2026, 7, 2)):
    return RegisteredTargetIndex(
        channel, events, episodes, observed_until=observed_until
    )


class RegisteredForecastTests(unittest.TestCase):
    def test_positive_is_known_at_onset_and_active_episode_is_excluded(self):
        normal = datetime(2025, 6, 1, 0)
        start = datetime(2025, 6, 1, 2)
        end = datetime(2025, 6, 1, 3)
        index = make_index(
            "c",
            [event(1, normal, "Норма"), event(2, start, "Неисправен"),
             event(3, end, "Норма")],
            [episode(2, start, end)],
        )
        positive = index.label(PredictionPoint("c", SMOKE, datetime(2025, 6, 1, 1)))
        self.assertEqual((positive.target, positive.label_status), (1, "positive"))
        self.assertEqual(positive.label_available_at, start)
        self.assertEqual(positive.target_episode_id, "c:2")
        self.assertEqual(
            index.label(PredictionPoint("c", SMOKE, start)).label_status, "excluded"
        )
        after = index.label(PredictionPoint("c", SMOKE, datetime(2025, 6, 1, 4)))
        self.assertEqual((after.target, after.label_status), (0, "negative"))
        self.assertEqual(after.label_available_at, after.horizon_end)

    def test_uncertainty_blocks_negative_and_current_eligibility(self):
        n = datetime(2025, 6, 1, 0)
        u = datetime(2025, 6, 1, 2)
        index = make_index(
            "c", [event(1, n, "Норма"), event(2, u, "Обесточен")], []
        )
        self.assertEqual(
            index.label(PredictionPoint("c", SMOKE, datetime(2025, 6, 1, 1))).reason,
            "future_state_or_onset_uncertain",
        )
        self.assertEqual(
            index.label(PredictionPoint("c", SMOKE, datetime(2025, 6, 1, 3))).reason,
            "uncertain_state_at_t",
        )

    def test_environmental_text_is_neutral_and_silence_expires_normal(self):
        n = datetime(2025, 6, 1, 0)
        index = make_index(
            "c", [event(1, n, "Норма"),
                  event(2, datetime(2025, 6, 1, 2), "Обнаружен дым")], []
        )
        self.assertEqual(
            index.label(PredictionPoint("c", SMOKE, datetime(2025, 6, 1, 3))).target,
            0,
        )
        self.assertEqual(
            index.label(PredictionPoint("c", SMOKE, datetime(2025, 6, 9))).reason,
            "no_recent_explicit_normal_at_t",
        )

    def test_conflicting_same_second_is_not_recovery_or_positive(self):
        n = datetime(2025, 6, 1, 0)
        conflict = datetime(2025, 6, 1, 2)
        index = make_index(
            "c", [event(1, n, "Норма"), event(2, conflict, "Норма"),
                  event(3, conflict, "Неисправен")],
            [episode(3, conflict, status="conflicted_same_timestamp")],
        )
        self.assertEqual(
            index.label(PredictionPoint("c", SMOKE, datetime(2025, 6, 1, 1))).label_status,
            "unknown",
        )
        self.assertEqual(
            index.label(PredictionPoint("c", SMOKE, conflict)).label_status,
            "excluded",
        )

    def test_positive_near_archive_end_does_not_require_whole_future_window(self):
        n = datetime(2026, 6, 30, 21)
        start = datetime(2026, 6, 30, 23, 30)
        index = make_index(
            "c", [event(1, n, "Норма"), event(2, start, "Неисправен")],
            [episode(2, start)],
        )
        result = index.label(PredictionPoint("c", SMOKE, datetime(2026, 6, 30, 22)))
        self.assertEqual(result.target, 1)
        self.assertEqual(result.split_status, "purged_boundary")
        no_event = make_index("c", [event(1, n, "Норма")], [])
        self.assertEqual(
            no_event.label(PredictionPoint("c", SMOKE, datetime(2026, 6, 30, 22))).reason,
            "future_archive_window_incomplete",
        )

    def test_archive_gap_clears_history(self):
        index = make_index(
            "c", [event(1, datetime(2020, 12, 31, 22), "Норма"),
                  event(2, datetime(2022, 1, 1), "Дыма нет")], []
        )
        self.assertEqual(
            index.label(PredictionPoint("c", SMOKE, datetime(2022, 1, 1, 1))).label_status,
            "unknown",
        )

    def test_unknown_type_and_wrong_type_are_never_positive(self):
        n = datetime(2025, 6, 1, 0)
        start = datetime(2025, 6, 1, 2)
        index = make_index(
            "c", [event(1, n, "Норма"), event(2, start, "Неисправен")],
            [episode(2, start, kind="Газовый датчик")],
        )
        self.assertEqual(
            index.label(PredictionPoint("c", None, datetime(2025, 6, 1, 1))).label_status,
            "excluded",
        )
        self.assertNotEqual(
            index.label(PredictionPoint("c", SMOKE, datetime(2025, 6, 1, 1))).target,
            1,
        )

    def test_old_open_episode_is_excluded_even_without_recent_event_rows(self):
        old_start = datetime(2025, 1, 1)
        index = make_index("c", [], [episode(1, old_start)])
        label = index.label(PredictionPoint("c", SMOKE, datetime(2025, 6, 1)))
        self.assertEqual(label.label_status, "excluded")

    def test_negative_needs_loaded_future_but_positive_can_be_known_early(self):
        n = datetime(2025, 6, 1, 0)
        start = datetime(2025, 6, 1, 2)
        cutoff = datetime(2025, 6, 1, 3)
        point = PredictionPoint("c", SMOKE, datetime(2025, 6, 1, 1))
        no_event = make_index(
            "c", [event(1, n, "Норма")], [], observed_until=cutoff
        )
        self.assertEqual(no_event.label(point).reason, "future_events_not_loaded")
        positive = make_index(
            "c", [event(1, n, "Норма"), event(2, start, "Неисправен")],
            [episode(2, start)], observed_until=cutoff,
        )
        self.assertEqual(positive.label(point).target, 1)

    def test_future_uncertainty_before_candidate_keeps_outcome_unknown(self):
        n = datetime(2025, 6, 1, 0)
        unknown = datetime(2025, 6, 1, 2)
        refreshed = datetime(2025, 6, 1, 3)
        start = datetime(2025, 6, 1, 4)
        index = make_index(
            "c",
            [event(1, n, "Норма"), event(2, unknown, "Обесточен"),
             event(3, refreshed, "Норма"), event(4, start, "Неисправен")],
            [episode(4, start)],
        )
        result = index.label(PredictionPoint("c", SMOKE, datetime(2025, 6, 1, 1)))
        self.assertEqual(result.label_status, "unknown")

    def test_episode_across_split_edge_cannot_label_both_sides(self):
        n = datetime(2024, 12, 31, 20)
        start = datetime(2025, 1, 1, 0, 30)
        index = make_index(
            "c", [event(1, n, "Норма"), event(2, start, "Неисправен")],
            [episode(2, start)],
        )
        before = index.label(PredictionPoint("c", SMOKE, datetime(2024, 12, 31, 23)))
        after = index.label(PredictionPoint("c", SMOKE, datetime(2025, 1, 1, 1)))
        self.assertEqual((before.target, before.split_status), (1, "purged_boundary"))
        self.assertEqual(after.label_status, "excluded")

    def test_split_purges_only_the_future_horizon(self):
        self.assertEqual(
            split_at(datetime(2024, 12, 30, 23)), ("train", "assigned")
        )
        self.assertEqual(
            split_at(datetime(2024, 12, 31)), ("train", "purged_boundary")
        )
        self.assertEqual(
            split_at(datetime(2025, 1, 1)), ("validation", "assigned")
        )
        self.assertEqual(
            split_at(datetime(2021, 6, 1)), (None, "outside_accepted_archive")
        )


if __name__ == "__main__":
    unittest.main()
