"""Sparse full-grid R3 A features match independently calculated snapshots."""

from datetime import datetime, timedelta
import unittest

from stage1.features import FeatureEvent, feature_at
from stage1.features.r2 import build_state_history_rows
from stage1.features.r3_full import build_selected_hourly_rows


T = datetime(2025, 6, 10, 12)


def event(at: datetime, state: str) -> FeatureEvent:
    return FeatureEvent("c", at, False, value_state=state, sensor_type="Датчик дыма")


class SparseR3FeaturesTests(unittest.TestCase):
    def test_matches_independent_asof_baseline_for_selected_hours(self) -> None:
        events = [
            event(T - timedelta(days=20), "Норма"),
            event(T - timedelta(hours=24), "Норма"),
            event(T, "Норма"),
            event(T + timedelta(hours=1), "Неисправен"),
            event(T + timedelta(days=1), "Норма"),
        ]
        hours = [T, T + timedelta(hours=1), T + timedelta(days=1)]
        actual = build_selected_hourly_rows(events, "c", hours)
        for at, row in zip(hours, actual):
            day = at.replace(hour=0)
            fit_end = day - timedelta(hours=168 + 24)
            self.assertEqual(row, feature_at(events, "c", at, baseline_fit_end_at=fit_end))

    def test_future_events_cannot_change_existing_rows(self) -> None:
        old = [event(T - timedelta(hours=1), "Норма")]
        hours = [T, T + timedelta(hours=2)]
        before = build_selected_hourly_rows(old, "c", hours)
        after = build_selected_hourly_rows(
            [*old, event(T + timedelta(hours=3), "Неисправен")], "c", hours
        )
        self.assertEqual(before, after)

    def test_unsorted_or_subhour_prediction_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique and sorted"):
            build_selected_hourly_rows([], "c", [T + timedelta(hours=1), T])
        with self.assertRaisesRegex(ValueError, "whole hours"):
            build_selected_hourly_rows([], "c", [T + timedelta(minutes=1)])

    def test_batched_r2_validation_switch_does_not_change_values(self) -> None:
        events = [event(T - timedelta(hours=1), "Норма")]
        rows = build_selected_hourly_rows(events, "c", [T])
        rows[0]["run_id"] = "r3-full-test"
        strict = build_state_history_rows(
            rows, events, source_a2_manifest_sha256="a" * 64,
            completed_episodes=[],
        )
        batched = build_state_history_rows(
            rows, events, source_a2_manifest_sha256="a" * 64,
            completed_episodes=[], validate=False,
        )
        self.assertEqual(strict.to_pylist(), batched.to_pylist())


if __name__ == "__main__":
    unittest.main()
