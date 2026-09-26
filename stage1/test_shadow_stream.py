"""Causal shadow admission, bounded history and frozen warning decisions."""

from datetime import datetime, timedelta
from itertools import groupby
import unittest

from ml.forecast.r6_rule import TERMS
from stage1.features.hourly import FeatureEvent
from stage1.features.r2 import CompletedEpisode, _branch_status
from stage1.features.r3_full import build_selected_hourly_rows
from stage1.features.r6_history import iter_rule_history
from stage1.shadow.stream import Observation, ShadowStream


T = datetime(2025, 12, 1)


def observation(at, text="Норма", *, channel="c", kind="Датчик дыма", flags=(), row_id=1):
    return Observation(
        row_id,
        FeatureEvent(channel, at, False, value_state=text, sensor_type=kind, quality_flags=flags),
        f"ext-journal-{at.year}.7z",
    )


def baseline():
    return [observation(T - timedelta(days=35 - i), row_id=i + 1) for i in range(26)]


def feed(stream, observations):
    ordered = sorted(observations, key=lambda row: (row.event.timestamp, row.event.channel_id))
    for _, group in groupby(ordered, key=lambda row: (row.event.timestamp, row.event.channel_id)):
        stream.observe_group(list(group))


class ShadowStreamTests(unittest.TestCase):
    def test_no_history_and_unknown_type_never_create_zero_risk(self):
        stream = ShadowStream(threshold=7.1)
        row = stream.predict(T, ["c"])[0]
        self.assertIsNone(row["rule_score"])
        self.assertIsNone(row["above_frozen_threshold"])
        self.assertTrue(row["admission_reasons"])
        stream = ShadowStream(threshold=7.1)
        feed(stream, [observation(T, "Неисправен", kind=None)])
        row = stream.predict(T, ["c"])[0]
        self.assertIsNone(row["rule_score"])
        self.assertEqual(row["registered_fault_text_count_168h"], 0)

    def test_recent_normal_and_legacy_past_data_gate(self):
        events = [*baseline(), observation(T)]
        stream = ShadowStream(threshold=7.1)
        feed(stream, events)
        row = stream.predict(T, ["c"])[0]
        old = build_selected_hourly_rows([o.event for o in events], "c", [T])[0]
        self.assertEqual(_branch_status(old, discrete=True), ("eligible", []))
        self.assertEqual(row["admission_status"], "eligible")
        self.assertEqual(row["availability_status"], "unknown")
        self.assertEqual(row["rule_score"], 0.0)
        self.assertEqual(row["baseline_fit_end_at"], old["baseline_fit_end_at"])
        # A normal alone does not waive baseline/history requirements.
        other = ShadowStream(threshold=7.1)
        feed(other, [observation(T)])
        self.assertIsNone(other.predict(T, ["c"])[0]["rule_score"])

    def test_active_fault_silence_alarm_false_and_unknown_text_do_not_recover(self):
        stream = ShadowStream(threshold=7.1)
        feed(
            stream,
            [
                *baseline(),
                observation(T - timedelta(hours=3)),
                observation(T - timedelta(hours=2), "Неисправен"),
                observation(T - timedelta(hours=1), "Непонятный текст"),
            ],
        )
        row = stream.predict(T, ["c"])[0]
        self.assertIn("registered_episode_active_at_t", row["admission_reasons"])
        self.assertIsNone(row["rule_score"])
        self.assertEqual(row["completed_episode_count_168h"], 0)

    def test_only_unambiguous_completed_episodes_enter_past_counts(self):
        before = observation(T - timedelta(hours=3))
        fault = observation(T - timedelta(hours=2), "Неисправен")
        recovered = observation(T - timedelta(hours=1))
        stream = ShadowStream(threshold=7.1)
        feed(stream, [*baseline(), before, fault])
        self.assertEqual(
            stream.predict(T - timedelta(hours=2), ["c"])[0]["completed_episode_count_168h"], 0
        )
        stream.observe_group([recovered])
        row = stream.predict(T, ["c"])[0]
        expected = list(
            iter_rule_history(
                [o.event for o in [*baseline(), before, fault, recovered]],
                [CompletedEpisode("c", fault.event.timestamp, recovered.event.timestamp)],
                "c",
                [T],
            )
        )[0]
        self.assertEqual(
            {name: row[name] for name in TERMS}, {name: expected[name] for name in TERMS}
        )
        # Left/stale-censored onset or an intervening unknown is not a completed feature.
        for extra in ([], [observation(T - timedelta(minutes=90), "Неизвестно")]):
            other = ShadowStream(threshold=7.1)
            feed(other, [*baseline(), fault, *extra, recovered])
            self.assertEqual(other.predict(T, ["c"])[0]["completed_episode_count_168h"], 0)

    def test_same_second_conflict_is_order_independent_and_blocks_prediction(self):
        group = [observation(T, "Норма", row_id=2), observation(T, "Неисправен", row_id=1)]
        rows = []
        for order in (group, list(reversed(group))):
            stream = ShadowStream(threshold=7.1)
            feed(stream, baseline())
            stream.observe_group(order)
            rows.append(stream.predict(T, ["c"])[0])
        self.assertEqual(rows[0], rows[1])
        self.assertIsNone(rows[0]["rule_score"])

    def test_quality_exclusions_and_unknown_metadata_block_admission(self):
        for event in (
            observation(T - timedelta(hours=1), flags=("channel_time_conflict",)),
            observation(T - timedelta(hours=1), text=None, kind=None),
        ):
            stream = ShadowStream(threshold=7.1)
            feed(stream, [*baseline(), observation(T - timedelta(hours=2)), event])
            row = stream.predict(T, ["c"])[0]
            self.assertIsNone(row["rule_score"])
            self.assertTrue(row["admission_reasons"])

    def test_sources_excluded_year_and_gap_never_carry_history(self):
        stream = ShadowStream(threshold=7.1)
        bad = observation(datetime(2021, 1, 1))
        stream.observe_group([bad])
        self.assertEqual(stream.ignored_source_rows, 1)
        self.assertEqual(
            stream.predict(datetime(2021, 1, 1), ["c"])[0]["admission_status"], "excluded"
        )
        stream.observe_group([observation(datetime(2022, 1, 1))])
        row = stream.predict(datetime(2022, 1, 1), ["c"])[0]
        self.assertIn("insufficient_history", row["admission_reasons"])
        example = ShadowStream(threshold=7.1)
        example.observe_group([Observation(1, observation(T).event, "журнал_событий_пример.csv")])
        self.assertIsNone(example.predict(T, ["c"])[0]["rule_score"])

    def test_late_unordered_partial_groups_and_consumed_future_are_rejected(self):
        stream = ShadowStream(threshold=7.1)
        stream.predict(T, ["c"])
        with self.assertRaisesRegex(ValueError, "late event"):
            stream.observe_group([observation(T)])
        future = ShadowStream(threshold=7.1)
        future.observe_group([observation(T + timedelta(hours=1))])
        with self.assertRaisesRegex(ValueError, "future observations"):
            future.predict(T, ["c"])
        with self.assertRaisesRegex(ValueError, "unique and sorted"):
            future.observe_group([observation(T + timedelta(hours=1))])
        with self.assertRaisesRegex(ValueError, "one complete"):
            future.observe_group([observation(T), observation(T + timedelta(hours=1))])

    def test_prefix_replay_and_future_mutation_leave_published_snapshot_identical(self):
        prefix = [*baseline(), observation(T)]
        published = []
        for tail in (
            [],
            [observation(T + timedelta(hours=1), "Неисправен")],
            [observation(T + timedelta(hours=1), "Неизвестно", kind=None)],
        ):
            stream = ShadowStream(threshold=7.1)
            feed(stream, [row for row in [*prefix, *tail] if row.event.timestamp <= T])
            published.append(stream.predict(T, ["c"])[0])
        self.assertEqual(published[0], published[1])
        self.assertEqual(published[0], published[2])

    def test_cooldown_uses_only_past_emitted_warnings(self):
        stream = ShadowStream(threshold=7.1)
        feed(
            stream,
            [
                *baseline(),
                observation(T - timedelta(hours=3)),
                *[
                    observation(T - timedelta(hours=2), "Неисправен", row_id=50 + i)
                    for i in range(4)
                ],
                observation(T - timedelta(hours=1)),
            ],
        )
        self.assertTrue(stream.predict(T, ["c"])[0]["warning_emitted"])
        self.assertEqual(
            stream.predict(T + timedelta(hours=1), ["c"])[0]["warning_status"],
            "suppressed_cooldown",
        )
        stream.observe_group(
            [observation(T + timedelta(hours=22), "Неисправен", row_id=60 + i) for i in range(4)]
        )
        stream.observe_group([observation(T + timedelta(hours=23))])
        self.assertTrue(stream.predict(T + timedelta(hours=24), ["c"])[0]["warning_emitted"])

    def test_retention_is_bounded_without_erasing_accumulated_past_metadata(self):
        stream = ShadowStream(threshold=7.1)
        feed(stream, [observation(T - timedelta(days=100)), *baseline(), observation(T)])
        stream.predict(T, ["c"])
        state = stream.channels["c"]
        self.assertTrue(all(event.timestamp > T - timedelta(days=37) for event in state.events))
        self.assertEqual(state.prefix.first_usable_at, T - timedelta(days=100))
        self.assertEqual(stream.retained_event_count, 27)

    def test_labels_and_changed_frozen_settings_are_rejected(self):
        row = {"row_id": 1, "source": "ext-journal-2025.7z", "target": 1}
        with self.assertRaisesRegex(ValueError, "forbidden inputs"):
            Observation.from_record(row)
        with self.assertRaisesRegex(ValueError, "frozen R6"):
            ShadowStream(threshold=6.1)


if __name__ == "__main__":
    unittest.main()
