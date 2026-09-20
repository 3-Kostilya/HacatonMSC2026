from datetime import datetime, timedelta
from dataclasses import replace
import unittest

from stage1.contracts import Decision, NormalizedEvent
from stage1.detectors import (
    ContextDetectorConfig,
    DiscreteDetectorConfig,
    NumericDetectorConfig,
    detect_context_coincidence,
    detect_discrete_pattern,
    detect_discrete_patterns,
    detect_numeric_level_shift,
)
from stage1.normalization import CHANNEL_TIME_CONFLICT, iter_accepted, normalize_chunks


BASE = datetime(2026, 1, 1)


def numeric_events(values, *, quality=()):
    return [
        NormalizedEvent(
            channel_id="n-1",
            timestamp=BASE + timedelta(minutes=index),
            raw_value=str(value),
            numeric_value=float(value),
            alarm=False,
            sensor_type="Датчик температуры",
            source="synthetic",
            quality_flags=quality if index == 0 else (),
        )
        for index, value in enumerate(values)
    ]


def discrete_events(values, *, channel="d-1", object_id=None):
    return [
        NormalizedEvent(
            channel_id=channel,
            timestamp=BASE + timedelta(minutes=index),
            raw_value=value,
            alarm=value == "alarm",
            sensor_type="Датчик дыма",
            source="synthetic",
            object_id=object_id,
        )
        for index, value in enumerate(values)
    ]


class NumericDetectorTests(unittest.TestCase):
    def setUp(self):
        self.config = NumericDetectorConfig(
            baseline_size=5,
            min_sustained=3,
            mad_multiplier=4,
            zero_mad_absolute_delta=1,
        )

    def test_sustained_shift_uses_earlier_baseline_and_separates_times(self):
        episode = detect_numeric_level_shift(
            numeric_events([10, 10, 10, 10, 10, 14, 15, 16]), self.config
        )
        self.assertIs(episode.decision, Decision.CANDIDATE)
        self.assertEqual(episode.start_at, BASE + timedelta(minutes=5))
        self.assertEqual(episode.confirmed_at, BASE + timedelta(minutes=7))
        self.assertLess(
            datetime.fromisoformat(episode.metadata["baseline_end_at"]), episode.start_at
        )
        self.assertIn("baseline_mad=0", episode.evidence)
        self.assertTrue(any("absolute_fallback" in item for item in episode.evidence))

    def test_single_spike_is_not_candidate(self):
        episode = detect_numeric_level_shift(
            numeric_events([10, 10, 10, 10, 10, 30, 10, 10]), self.config
        )
        self.assertIs(episode.decision, Decision.NO_CANDIDATE)

    def test_insufficient_history_is_unknown(self):
        episode = detect_numeric_level_shift(numeric_events([1, 2, 3]), self.config)
        self.assertIs(episode.decision, Decision.UNKNOWN)
        self.assertIn("insufficient_numeric_history", episode.observation_quality[0])

    def test_blocking_quality_is_unknown(self):
        episode = detect_numeric_level_shift(
            numeric_events([10] * 8, quality=(CHANNEL_TIME_CONFLICT,)), self.config
        )
        self.assertIs(episode.decision, Decision.UNKNOWN)
        self.assertIn(f"blocking_quality:{CHANNEL_TIME_CONFLICT}", episode.observation_quality)

    def test_real_normalizer_conflict_is_unknown(self):
        def row(event_id, value):
            return {
                "ид_события": event_id,
                "ид_канала_данных": "n-1",
                "дата": "2026-01-01",
                "время": "00:00:00",
                "тревожное": "false",
                "значение_датчика": value,
            }

        results = normalize_chunks(
            [[row("1", "10"), row("2", "11")]],
            {"n-1": "Датчик температуры"},
            "synthetic.csv",
        )
        episode = detect_numeric_level_shift(iter_accepted(results), self.config)
        self.assertIs(episode.decision, Decision.UNKNOWN)
        self.assertIn(f"blocking_quality:{CHANNEL_TIME_CONFLICT}", episode.observation_quality)

    def test_future_quality_cannot_erase_an_already_confirmed_detection(self):
        events = numeric_events([10] * 5 + [20] * 3)
        prefix = detect_numeric_level_shift(events, self.config)
        future_bad = replace(
            events[-1],
            timestamp=BASE + timedelta(days=1),
            quality_flags=(CHANNEL_TIME_CONFLICT,),
        )
        extended = detect_numeric_level_shift(events + [future_bad], self.config)
        self.assertEqual(prefix.decision, Decision.CANDIDATE)
        self.assertEqual(
            (extended.decision, extended.start_at, extended.confirmed_at),
            (prefix.decision, prefix.start_at, prefix.confirmed_at),
        )

    def test_blocking_nonnumeric_message_breaks_numeric_run(self):
        events = numeric_events([10] * 5)
        events.extend(numeric_events([20])[:1])
        events[-1] = replace(events[-1], timestamp=BASE + timedelta(minutes=5))
        events.append(
            NormalizedEvent(
                channel_id="n-1",
                timestamp=BASE + timedelta(minutes=6),
                raw_value="NaN",
                numeric_value=None,
                alarm=False,
                sensor_type="Датчик температуры",
                source="synthetic",
                quality_flags=("nonfinite_numeric",),
            )
        )
        events.extend(
            replace(event, timestamp=BASE + timedelta(minutes=7 + index))
            for index, event in enumerate(numeric_events([20, 20]))
        )
        episode = detect_numeric_level_shift(events, self.config)
        self.assertIs(episode.decision, Decision.UNKNOWN)
        self.assertIn("blocking_quality:nonfinite_numeric", episode.observation_quality)

    def test_quality_is_checked_on_numeric_baseline_not_raw_prefix(self):
        text = [
            replace(event, numeric_value=None, raw_value="status")
            for event in numeric_events([0] * 3)
        ]
        numeric = [
            replace(event, timestamp=BASE + timedelta(minutes=index + 3))
            for index, event in enumerate(numeric_events([10] * 5 + [20] * 3))
        ]
        numeric[3] = replace(numeric[3], quality_flags=(CHANNEL_TIME_CONFLICT,))
        episode = detect_numeric_level_shift(text + numeric, self.config)
        self.assertIs(episode.decision, Decision.UNKNOWN)
        self.assertIn(f"blocking_quality:{CHANNEL_TIME_CONFLICT}", episode.observation_quality)


class DiscreteDetectorTests(unittest.TestCase):
    def setUp(self):
        self.config = DiscreteDetectorConfig(
            baseline_size=4,
            repeated_state_count=3,
            transition_count=4,
            transition_window=timedelta(minutes=5),
            known_states=frozenset({"normal", "alarm"}),
        )

    def test_repeated_state_requires_sustained_messages(self):
        episode = detect_discrete_pattern(
            discrete_events(["normal", "alarm", "normal", "alarm", "alarm", "alarm", "alarm"]),
            self.config,
        )
        self.assertIs(episode.decision, Decision.CANDIDATE)
        self.assertEqual(episode.anomaly_type, "repeated_state_burst")
        self.assertEqual(episode.start_at, BASE + timedelta(minutes=4))
        self.assertEqual(episode.confirmed_at, BASE + timedelta(minutes=6))

    def test_one_rare_message_is_not_candidate(self):
        episode = detect_discrete_pattern(
            discrete_events(["normal", "alarm", "normal", "alarm", "alarm"]), self.config
        )
        self.assertIs(episode.decision, Decision.NO_CANDIDATE)

    def test_same_messages_far_apart_are_not_a_burst(self):
        events = discrete_events(["normal", "alarm", "normal", "alarm"])
        for offset in (1, 30, 60):
            events.append(
                NormalizedEvent(
                    channel_id="d-1",
                    timestamp=BASE + timedelta(days=offset),
                    raw_value="alarm",
                    alarm=True,
                    sensor_type="Датчик дыма",
                    source="synthetic",
                )
            )
        episode = detect_discrete_pattern(events, self.config)
        self.assertIs(episode.decision, Decision.NO_CANDIDATE)

    def test_rapid_switching(self):
        config = DiscreteDetectorConfig(
            baseline_size=4,
            repeated_state_count=5,
            transition_count=4,
            transition_window=timedelta(minutes=5),
            known_states=frozenset({"normal", "alarm"}),
        )
        episode = detect_discrete_pattern(
            discrete_events(
                ["normal", "alarm", "normal", "alarm", "normal", "alarm", "normal", "alarm"]
            ),
            config,
        )
        self.assertIs(episode.decision, Decision.CANDIDATE)
        self.assertEqual(episode.anomaly_type, "rapid_switching")

    def test_unknown_state_is_unknown(self):
        episode = detect_discrete_pattern(
            discrete_events(["normal", "alarm", "normal", "alarm", "???"]), self.config
        )
        self.assertIs(episode.decision, Decision.UNKNOWN)
        self.assertEqual(episode.observation_quality, ("unknown_states:???",))

    def test_sequence_api_keeps_short_history_unknown(self):
        from stage1.detectors import detect_discrete_patterns

        results = detect_discrete_patterns(
            discrete_events(["normal"] * self.config.baseline_size), self.config
        )
        self.assertEqual(len(results), 1)
        self.assertIs(results[0].decision, Decision.UNKNOWN)

    def test_later_repeat_does_not_replace_earlier_switching(self):
        values = ["normal"] * 4 + ["alarm", "normal", "alarm", "normal"]
        early = detect_discrete_pattern(discrete_events(values), self.config)
        later = discrete_events(values)
        later.extend(
            replace(
                later[-1],
                timestamp=BASE + timedelta(minutes=minute),
                raw_value="normal",
            )
            for minute in (60, 61, 62)
        )
        extended = detect_discrete_pattern(later, self.config)
        self.assertEqual(early.anomaly_type, "rapid_switching")
        self.assertEqual(
            (extended.anomaly_type, extended.start_at, extended.confirmed_at),
            (early.anomaly_type, early.start_at, early.confirmed_at),
        )

    def test_regular_repeated_normal_state_is_not_anomaly(self):
        episode = detect_discrete_pattern(discrete_events(["normal"] * 100), self.config)
        self.assertIs(episode.decision, Decision.NO_CANDIDATE)

    def test_normal_fast_cycle_seen_in_baseline_is_not_anomaly(self):
        config = replace(self.config, baseline_size=6)
        episode = detect_discrete_pattern(discrete_events(["normal", "alarm"] * 10), config)
        self.assertIs(episode.decision, Decision.NO_CANDIDATE)

    def test_same_state_acceleration_is_candidate(self):
        events = discrete_events(["normal"] * 6)
        events = [
            replace(event, timestamp=BASE + timedelta(hours=index))
            for index, event in enumerate(events)
        ]
        events.extend(
            replace(events[-1], timestamp=BASE + timedelta(hours=6, minutes=index))
            for index in range(3)
        )
        episode = detect_discrete_pattern(events, replace(self.config, baseline_size=6))
        self.assertIs(episode.decision, Decision.CANDIDATE)
        self.assertEqual(episode.anomaly_type, "repeated_state_burst")

    def test_rapid_switching_recovers_and_can_recur(self):
        values = ["normal"] * 6 + ["alarm", "normal", "alarm", "normal"]
        events = discrete_events(values)
        events.extend(
            NormalizedEvent(
                channel_id="d-1",
                timestamp=BASE + timedelta(minutes=minute),
                raw_value="normal",
                alarm=False,
                sensor_type="Датчик дыма",
                source="synthetic",
            )
            for minute in range(10, 60)
        )
        events.extend(
            NormalizedEvent(
                channel_id="d-1",
                timestamp=BASE + timedelta(minutes=60 + index),
                raw_value=value,
                alarm=value == "alarm",
                sensor_type="Датчик дыма",
                source="synthetic",
            )
            for index, value in enumerate(["alarm", "normal", "alarm", "normal"])
        )
        episodes = detect_discrete_patterns(events, replace(self.config, baseline_size=6))
        self.assertEqual(len(episodes), 2)
        self.assertIsNotNone(episodes[0].end_at)
        self.assertIsNone(episodes[1].end_at)


class ContextDetectorTests(unittest.TestCase):
    def test_missing_relation_is_unknown(self):
        target = discrete_events(["normal", "alarm"], channel="target")
        context = discrete_events(["normal", "alarm"], channel="power")
        episode = detect_context_coincidence(target, context)
        self.assertIs(episode.decision, Decision.UNKNOWN)
        self.assertEqual(episode.observation_quality, ("unconfirmed_context_link",))

    def test_confirmed_relation_requires_multiple_causal_coincidences(self):
        target = discrete_events(["normal", "alarm"], channel="target", object_id="obj-1")
        context = discrete_events(["normal", "alarm"], channel="power", object_id="obj-1")
        episode = detect_context_coincidence(
            target,
            context,
            ContextDetectorConfig(coincidence_window=timedelta(minutes=1), min_coincidences=2),
        )
        self.assertIs(episode.decision, Decision.CANDIDATE)
        self.assertTrue(any("confirmed_object_id=obj-1" == item for item in episode.evidence))


if __name__ == "__main__":
    unittest.main()
