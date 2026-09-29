"""Causal replay boundaries of the four fixed R6 rule inputs."""

from datetime import datetime, timedelta
import unittest

from stage1.features.hourly import FeatureEvent
from stage1.features.r2 import CompletedEpisode
from stage1.features.r6_history import iter_rule_history


T = datetime(2026, 2, 10, 12)


def event(at, state="Неисправен", flags=(), kind="Датчик дыма"):
    return FeatureEvent("c", at, False, value_state=state, sensor_type=kind, quality_flags=flags)


class R6HistoryTests(unittest.TestCase):
    def test_left_open_right_closed_and_completed_end_known_at_t(self):
        events = [event(T - timedelta(hours=168)), event(T - timedelta(hours=24)), event(T)]
        episodes = [CompletedEpisode("c", T - timedelta(hours=2), T)]
        row = list(iter_rule_history(events, episodes, "c", [T]))[0]
        self.assertEqual(row["registered_fault_text_count_24h"], 1)
        self.assertEqual(row["registered_fault_text_count_168h"], 2)
        self.assertEqual(row["completed_episode_count_168h"], 1)

    def test_future_observation_or_episode_end_cannot_change_past(self):
        old = [event(T)]
        before = list(iter_rule_history(old, [], "c", [T]))
        after = list(
            iter_rule_history(
                [*old, event(T + timedelta(microseconds=1))],
                [CompletedEpisode("c", T - timedelta(hours=2), T + timedelta(seconds=1))],
                "c",
                [T],
            )
        )
        self.assertEqual(before, after)

    def test_quality_unknown_type_and_other_channel_do_not_add_faults(self):
        events = [
            event(T, flags=("channel_time_conflict",)),
            event(T, kind=None),
            FeatureEvent("other", T, False, value_state="Неисправен", sensor_type="Датчик дыма"),
        ]
        row = list(iter_rule_history(events, [], "c", [T]))[0]
        self.assertEqual(row["registered_fault_text_count_168h"], 0)

    def test_technical_message_is_not_always_target_message(self):
        row = list(
            iter_rule_history(
                [event(T, "Батарея неисправна", kind="Состояние вентилятора")],
                [],
                "c",
                [T],
            )
        )[0]
        self.assertEqual(row["technical_message_count_24h"], 1)
        self.assertEqual(row["registered_fault_text_count_24h"], 0)

    def test_batch_snapshots_equal_truncated_independent_replays(self):
        events = [event(T), event(T + timedelta(hours=1)), event(T + timedelta(hours=26))]
        times = [T, T + timedelta(hours=1), T + timedelta(hours=26)]
        batch = list(iter_rule_history(events, [], "c", times))
        replay = [
            list(iter_rule_history([e for e in events if e.timestamp <= at], [], "c", [at]))[0]
            for at in times
        ]
        self.assertEqual(batch, replay)

    def test_excluded_year_and_unsorted_clock_are_rejected(self):
        with self.assertRaises(ValueError):
            list(iter_rule_history([], [], "c", [datetime(2021, 1, 1)]))
        with self.assertRaises(ValueError):
            list(iter_rule_history([], [], "c", [T, T]))
        with self.assertRaises(ValueError):
            list(
                iter_rule_history(
                    [],
                    [CompletedEpisode("c", datetime(2020, 12, 31), datetime(2022, 1, 1))],
                    "c",
                    [T],
                )
            )


if __name__ == "__main__":
    unittest.main()
