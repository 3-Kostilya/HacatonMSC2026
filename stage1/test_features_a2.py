"""Independent A2 feature checks against hand calculations and B2 events.

The B2 truth manifest is deliberately never supplied to the feature builder.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import unittest

from stage1.contracts import NormalizedEvent
from stage1.features import (
    A2_SCHEMA,
    FEATURE_VERSION,
    FeatureEvent,
    HourlyConfig,
    build_hourly_rows,
    feature_at,
)
from stage1.simulation import build_b2_suite


def _event(
    channel_id: str,
    timestamp: datetime,
    *,
    number: float | None = None,
    state: str | None = None,
    alarm: bool = False,
    quality_flags: tuple[str, ...] = (),
    sensor_type: str | None = None,
) -> FeatureEvent:
    return FeatureEvent(
        channel_id=channel_id,
        timestamp=timestamp,
        alarm=alarm,
        value_numeric=number,
        value_state=state,
        sensor_type=sensor_type or ("Датчик температуры" if number is not None else "КД Дверь"),
        quality_flags=quality_flags,
    )


class A2FeatureTests(unittest.TestCase):
    def test_versioned_schema_has_windows_and_unavailability(self):
        self.assertTrue(FEATURE_VERSION)
        for field in (
            "channel_id",
            "prediction_time",
            "availability_status",
            "availability_reasons",
            "event_count_1h",
            "event_count_6h",
            "event_count_24h",
            "event_count_168h",
            "numeric_median_24h",
            "state_transitions_24h",
        ):
            self.assertIn(field, A2_SCHEMA.names)

    def test_left_open_right_closed_boundaries_in_all_four_windows(self):
        t = datetime(2026, 6, 15, 12)
        events = [
            _event("c", t - timedelta(hours=168), number=1, alarm=True),
            _event("c", t - timedelta(hours=24), number=2, alarm=True),
            _event("c", t - timedelta(hours=6), number=10, alarm=True),
            _event("c", t - timedelta(hours=1), number=20),
            _event("c", t, number=30, alarm=True),
            _event("c", t + timedelta(microseconds=1), number=999, alarm=True),
        ]
        row = feature_at(events, "c", t)
        self.assertEqual(
            [row[f"event_count_{h}h"] for h in (1, 6, 24, 168)],
            [1, 2, 3, 4],
        )
        self.assertEqual(
            [row[f"alarm_count_{h}h"] for h in (1, 6, 24, 168)],
            [1, 1, 2, 3],
        )
        self.assertEqual(row["numeric_median_1h"], 30)
        self.assertEqual(row["numeric_median_6h"], 25)
        self.assertEqual(row["numeric_median_24h"], 20)
        self.assertEqual(row["numeric_mad_24h"], 10)
        self.assertEqual(row["numeric_min_24h"], 10)
        self.assertEqual(row["numeric_max_24h"], 30)
        self.assertEqual(row["numeric_range_24h"], 20)

    def test_empty_hour_is_unknown_not_normal_zero(self):
        t = datetime(2026, 6, 15, 12)
        row = feature_at([], "silent", t)
        self.assertEqual(row["event_count_1h"], 0)
        self.assertEqual(row["alarm_count_1h"], 0)
        self.assertIsNone(row["numeric_median_24h"])
        self.assertIsNone(row["state_transitions_24h"])
        self.assertEqual(row["availability_status"], "unknown")
        self.assertTrue(row["availability_reasons"])
        self.assertIsNone(row["last_observation_age_seconds"])

    def test_mixed_channel_preserves_numeric_and_state_branches(self):
        t = datetime(2026, 6, 15, 12)
        events = [
            _event("ups", t - timedelta(hours=3), number=10, sensor_type="ИБП"),
            _event("ups", t - timedelta(hours=2), state="Сеть", sensor_type="ИБП"),
            _event("ups", t - timedelta(hours=1), number=20, sensor_type="ИБП"),
            _event("ups", t, state="Батарея", sensor_type="ИБП"),
        ]
        row = feature_at(events, "ups", t)
        self.assertEqual(row["numeric_count_24h"], 2)
        self.assertEqual(row["state_count_24h"], 2)
        self.assertEqual(row["numeric_median_24h"], 15)
        self.assertEqual(row["state_transitions_24h"], 1)
        self.assertEqual(row["state_distinct_count_24h"], 2)

    def test_conflicted_values_are_counted_but_not_used_in_statistics(self):
        t = datetime(2026, 6, 15, 12)
        events = [
            _event("c", t - timedelta(hours=2), number=5),
            _event(
                "c",
                t - timedelta(hours=1),
                number=100,
                quality_flags=("channel_time_conflict",),
            ),
            _event("c", t, number=200, alarm=True, quality_flags=("channel_time_conflict",)),
        ]
        row = feature_at(events, "c", t)
        self.assertEqual(row["event_count_24h"], 3)
        self.assertEqual(row["alarm_count_24h"], 1)
        self.assertEqual(row["excluded_quality_count_24h"], 2)
        self.assertEqual(row["numeric_count_24h"], 1)
        self.assertEqual(row["numeric_median_24h"], 5)
        self.assertIn("quality_exclusions_present", row["window_reasons_24h"])

    def test_zero_mad_is_reported_without_division_or_fake_variance(self):
        t = datetime(2026, 6, 15, 12)
        events = [_event("c", t - timedelta(hours=h), number=7) for h in (3, 2, 1)]
        row = feature_at(events, "c", t)
        self.assertEqual(row["numeric_median_24h"], 7)
        self.assertEqual(row["numeric_mad_24h"], 0)
        self.assertEqual(row["numeric_range_24h"], 0)

    def test_baseline_is_frozen_before_current_window_and_embargo(self):
        t = datetime(2026, 6, 15, 12)
        cutoff = t - timedelta(hours=192)
        old = [_event("c", cutoff - timedelta(days=14 - day), number=10) for day in range(12)]
        recent = [
            _event("c", t - timedelta(hours=170), number=100),
            _event("c", t - timedelta(hours=1), number=100),
        ]
        config = HourlyConfig(minimum_baseline_events=10)
        row = feature_at(old + recent, "c", t, config=config, baseline_fit_end_at=cutoff)
        self.assertEqual(row["baseline_fit_end_at"], cutoff)
        self.assertEqual(row["baseline_event_count"], 12)
        self.assertEqual(row["baseline_numeric_median"], 10)
        self.assertEqual(row["baseline_numeric_mad"], 0)
        self.assertEqual(
            row,
            feature_at(
                old + recent + [_event("c", t + timedelta(hours=1), number=500)],
                "c",
                t,
                config=config,
                baseline_fit_end_at=cutoff,
            ),
        )
        with self.assertRaises(ValueError):
            feature_at(old + recent, "c", t, baseline_fit_end_at=t - timedelta(hours=24))

    def test_future_append_does_not_change_past_row(self):
        t = datetime(2026, 6, 15, 12)
        prefix = [
            _event("c", t - timedelta(hours=2), number=10),
            _event("c", t, number=20),
        ]
        future = [_event("c", t + timedelta(seconds=1), number=1_000_000, alarm=True)]
        self.assertEqual(feature_at(prefix, "c", t), feature_at(prefix + future, "c", t))

    def test_source_event_id_and_channel_spelling_cannot_encode_features(self):
        t = datetime(2026, 6, 15, 12)

        def normalized(channel: str, source: str, event_id: str) -> NormalizedEvent:
            return NormalizedEvent(
                channel_id=channel,
                timestamp=t - timedelta(minutes=20),
                raw_value="17",
                alarm=False,
                sensor_type="Датчик температуры",
                source=source,
                event_id=event_id,
                numeric_value=17,
            )

        first = FeatureEvent.from_normalized(normalized("ordinary", "source-a", "event-a"))
        renamed = FeatureEvent.from_normalized(
            normalized("sim-tuning-numeric_level_shift", "synthetic:positive", "truth-1")
        )
        a = feature_at([first], first.channel_id, t)
        b = feature_at([renamed], renamed.channel_id, t)
        a.pop("channel_id")
        b.pop("channel_id")
        self.assertEqual(a, b)

    def test_b2_tuning_events_only_have_causal_prefix_equivalence(self):
        suite = build_b2_suite("tuning")
        for channel_id in (
            "sim-tuning-numeric_gradual_drift",
            "sim-tuning-mixed_numeric_state",
            "sim-tuning-known_cadence_dropout",
        ):
            with self.subTest(channel_id=channel_id):
                events = [
                    FeatureEvent.from_normalized(event)
                    for event in suite.events
                    if event.channel_id == channel_id
                ]
                self.assertTrue(events)
                t = events[0].timestamp + timedelta(days=7, hours=1)
                prefix = [event for event in events if event.timestamp <= t]
                self.assertLess(len(prefix), len(events))
                self.assertEqual(
                    feature_at(prefix, channel_id, t),
                    feature_at(events, channel_id, t),
                )

    def test_bounded_hourly_grid_keeps_silent_hours_and_unique_keys(self):
        start = datetime(2026, 6, 15, 12)
        end = start + timedelta(hours=4)
        events = [_event("c", start, state="Закрыто")]
        rows = build_hourly_rows(events, "c", start, end)
        keys = [(row["channel_id"], row["prediction_time"]) for row in rows]
        self.assertEqual(keys, [("c", start + timedelta(hours=h)) for h in range(4)])
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(rows[-1]["event_count_1h"], 0)

    def test_hourly_grid_uses_one_frozen_baseline(self):
        start = datetime(2026, 6, 15, 12)
        cutoff = start - timedelta(hours=192)
        old = [_event("c", cutoff - timedelta(days=14 - day), number=10) for day in range(12)]
        recent = [_event("c", start + timedelta(hours=1), number=100)]
        rows = build_hourly_rows(old + recent, "c", start, start + timedelta(hours=4))
        self.assertEqual({row["baseline_fit_end_at"] for row in rows}, {cutoff})
        self.assertEqual({row["baseline_event_count"] for row in rows}, {12})
        self.assertEqual({row["baseline_numeric_median"] for row in rows}, {10})


if __name__ == "__main__":
    unittest.main()
