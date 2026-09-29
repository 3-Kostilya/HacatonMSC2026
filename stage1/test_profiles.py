from datetime import datetime, timedelta
import unittest

from stage1.contracts import NormalizedEvent
from stage1.profiles import fit_behavior_profiles, numeric_deviation_mad, state_is_known


def event(hour: int, value: str, numeric: float | None = None) -> NormalizedEvent:
    return NormalizedEvent(
        channel_id="1",
        timestamp=datetime(2025, 1, 1) + timedelta(hours=hour),
        raw_value=value,
        numeric_value=numeric,
        alarm=False,
        sensor_type="Датчик температуры",
        source="fixture.csv",
    )


class BehaviorProfileTests(unittest.TestCase):
    def test_future_events_do_not_change_frozen_profile(self):
        past = [event(hour, str(value), float(value)) for hour, value in enumerate([9, 10, 11])]
        future = event(30, "1000", 1000.0)
        cutoff = datetime(2025, 1, 2)
        first = fit_behavior_profiles(past, cutoff, min_events=1, min_numeric=1)["1"]
        second = fit_behavior_profiles([*past, future], cutoff, min_events=1, min_numeric=1)["1"]
        self.assertEqual(first, second)
        self.assertEqual(first.numeric_median, 10.0)

    def test_zero_mad_is_unknown_not_infinite_score(self):
        profile = fit_behavior_profiles(
            [event(0, "10", 10.0), event(1, "10", 10.0)],
            datetime(2025, 1, 1, 3),
            min_events=1,
            min_numeric=1,
        )["1"]
        self.assertIn("zero_numeric_mad", profile.quality_flags)
        self.assertIsNone(numeric_deviation_mad(event(4, "11", 11.0), profile))

    def test_unseen_state_is_reported_without_calling_it_failure(self):
        profile = fit_behavior_profiles(
            [event(0, "Норма"), event(1, "Норма")],
            datetime(2025, 1, 1, 3),
            min_events=1,
        )["1"]
        self.assertFalse(state_is_known(event(4, "Новый код"), profile))

    def test_scoring_before_cutoff_is_rejected(self):
        profile = fit_behavior_profiles([event(0, "Норма")], datetime(2025, 1, 1, 2), min_events=1)[
            "1"
        ]
        with self.assertRaisesRegex(ValueError, "precedes"):
            state_is_known(event(1, "Норма"), profile)


if __name__ == "__main__":
    unittest.main()
