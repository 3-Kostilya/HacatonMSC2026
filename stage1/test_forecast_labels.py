from datetime import datetime, timedelta
from dataclasses import replace
import unittest

from stage1.contracts import Decision, Episode
from stage1.forecast_labels import label_future_onsets


BASE = datetime(2026, 1, 1)


def episode(start_hour: int, end_hour: int | None = None) -> Episode:
    return Episode(
        episode_id=f"ep-{start_hour}",
        channel_id="1",
        sensor_type="Датчик дыма",
        sensor_group="fire_discrete",
        anomaly_type="rapid_switching",
        decision=Decision.CANDIDATE,
        start_at=BASE + timedelta(hours=start_hour),
        confirmed_at=BASE + timedelta(hours=start_hour + 1),
        end_at=BASE + timedelta(hours=end_hour) if end_hour is not None else None,
        ruleset_version="test-v1",
        evidence=("fixture",),
        observation_quality=(),
    )


class ForecastLabelTests(unittest.TestCase):
    def test_new_onset_inside_horizon_is_positive(self):
        label = label_future_onsets(
            "1", [BASE], [episode(12, 14)], observed_until=BASE + timedelta(days=2)
        )[0]
        self.assertEqual(label.value, 1)
        self.assertEqual(label.reason, "new_episode_onset")

    def test_ongoing_episode_is_not_a_successful_prediction(self):
        label = label_future_onsets(
            "1",
            [BASE + timedelta(hours=13)],
            [episode(12, 20)],
            observed_until=BASE + timedelta(days=2),
        )[0]
        self.assertEqual(label.value, -1)
        self.assertEqual(label.reason, "episode_already_ongoing")

    def test_end_of_export_and_unknown_future_are_censored(self):
        prediction = BASE + timedelta(hours=12)
        end_label = label_future_onsets(
            "1", [prediction], [], observed_until=BASE + timedelta(hours=20)
        )[0]
        unknown_label = label_future_onsets(
            "1",
            [prediction],
            [],
            observed_until=BASE + timedelta(days=3),
            unknown_intervals=[(BASE + timedelta(hours=18), BASE + timedelta(hours=19))],
        )[0]
        self.assertEqual((end_label.value, unknown_label.value), (-1, -1))

    def test_horizon_boundary_is_inclusive(self):
        label = label_future_onsets(
            "1", [BASE], [episode(24, 25)], observed_until=BASE + timedelta(days=2)
        )[0]
        self.assertEqual(label.value, 1)

    def test_onset_is_censored_until_its_confirmation_is_observed(self):
        unconfirmed = episode(23, 30)
        unconfirmed = replace(unconfirmed, confirmed_at=BASE + timedelta(hours=26))
        label = label_future_onsets(
            "1", [BASE], [unconfirmed], observed_until=BASE + timedelta(hours=24)
        )[0]
        self.assertEqual(label.value, -1)
        self.assertEqual(label.reason, "episode_confirmation_not_observed")


if __name__ == "__main__":
    unittest.main()
