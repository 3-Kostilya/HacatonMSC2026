"""QA value rules preserve observed events while guarding physical statistics."""

from __future__ import annotations

from datetime import datetime, timedelta
import unittest

from stage1.features.hourly import FeatureEvent, feature_at
from stage1.features.qa_values import qa_window_counts
from stage1.value_quality import assess_value


class ValueQualityTests(unittest.TestCase):
    def test_temperature_service_code_is_not_a_physical_reading(self):
        assessment = assess_value("Датчик температуры", "-127", -127.0)
        self.assertEqual(assessment.category, "temperature_service_code_candidate")
        self.assertFalse(assessment.numeric_measurement_usable)
        self.assertTrue(assessment.technical_code_candidate)
        self.assertFalse(assessment.environmental_alarm_level_candidate)
        self.assertTrue(assess_value("Датчик температуры", "17", 17).numeric_measurement_usable)

    def test_epoch_value_is_not_a_state_transition_or_failure(self):
        for text in ("01.01.1970 03:00:00", "01.01.1970 03:00:01"):
            assessment = assess_value("Состояние охраны", text, None)
            self.assertEqual(assessment.category, "epoch_value_artifact")
            self.assertFalse(assessment.state_transition_usable)
            self.assertFalse(assessment.technical_code_candidate)

    def test_gas_levels_are_environmental_candidates_not_device_faults(self):
        self.assertEqual(
            assess_value("Газовый датчик", "-0.01", -0.01).category, "gas_negative_reading"
        )
        self.assertTrue(assess_value("Газовый датчик", "-0.01", -0.01).numeric_measurement_usable)
        self.assertFalse(
            assess_value("Газовый датчик", "0.99", 0.99).environmental_alarm_level_candidate
        )
        self.assertTrue(
            assess_value("Газовый датчик", "1", 1.0).environmental_alarm_level_candidate
        )
        self.assertTrue(assess_value("Газовый датчик", "100", 100.0).numeric_measurement_usable)
        self.assertFalse(
            assess_value("Газовый датчик", "327.68", 327.68).numeric_measurement_usable
        )
        self.assertFalse(
            assess_value("Датчик температуры", "1", 1.0).environmental_alarm_level_candidate
        )

    def test_opt_in_feature_recalculation_keeps_event_and_alarm_counts(self):
        t = datetime(2025, 6, 15, 12)
        records = [
            {
                "channel_id": "temp",
                "timestamp": t - timedelta(hours=3),
                "alarm": False,
                "value_numeric": 18.0,
                "value_state": None,
                "sensor_type": "Датчик температуры",
            },
            {
                "channel_id": "temp",
                "timestamp": t - timedelta(hours=2),
                "alarm": True,
                "value_numeric": -127.0,
                "value_state": None,
                "sensor_type": "Датчик температуры",
            },
            {
                "channel_id": "temp",
                "timestamp": t - timedelta(hours=1),
                "alarm": False,
                "value_numeric": 20.0,
                "value_state": None,
                "sensor_type": "Датчик температуры",
            },
        ]
        old = feature_at([FeatureEvent.from_clean_record(row) for row in records], "temp", t)
        adjusted = feature_at(
            [FeatureEvent.from_clean_record(row, apply_qa_value_policy=True) for row in records],
            "temp",
            t,
        )
        self.assertEqual((old["event_count_24h"], old["alarm_count_24h"]), (3, 1))
        self.assertEqual((adjusted["event_count_24h"], adjusted["alarm_count_24h"]), (3, 1))
        self.assertEqual(old["numeric_median_24h"], 18)
        self.assertEqual(adjusted["numeric_median_24h"], 19)
        self.assertEqual(adjusted["numeric_count_24h"], 2)
        self.assertEqual(old["numeric_count_24h"], 3)

    def test_future_value_does_not_change_past_qa_features(self):
        t = datetime(2025, 6, 15, 12)
        prefix = FeatureEvent.from_clean_record(
            {
                "channel_id": "gas",
                "timestamp": t,
                "alarm": False,
                "value_numeric": 1.2,
                "value_state": None,
                "sensor_type": "Газовый датчик",
            },
            apply_qa_value_policy=True,
        )
        future = FeatureEvent.from_clean_record(
            {
                "channel_id": "gas",
                "timestamp": t + timedelta(seconds=1),
                "alarm": False,
                "value_numeric": 327.68,
                "value_state": None,
                "sensor_type": "Газовый датчик",
            },
            apply_qa_value_policy=True,
        )
        self.assertEqual(feature_at([prefix], "gas", t), feature_at([prefix, future], "gas", t))
        self.assertEqual(qa_window_counts([prefix], t), qa_window_counts([prefix, future], t))
        self.assertEqual(
            qa_window_counts([prefix, future], t)["qa_gas_alarm_level_candidate_count_1h"], 1
        )

    def test_epoch_value_does_not_count_as_real_state(self):
        t = datetime(2025, 6, 15, 12)
        records = [
            {
                "channel_id": "security",
                "timestamp": t - timedelta(minutes=2),
                "alarm": False,
                "value_numeric": None,
                "value_state": "Норма",
                "sensor_type": "Состояние охраны",
            },
            {
                "channel_id": "security",
                "timestamp": t - timedelta(minutes=1),
                "alarm": False,
                "value_numeric": None,
                "value_state": "01.01.1970 03:00:00",
                "sensor_type": "Состояние охраны",
            },
        ]
        old = feature_at([FeatureEvent.from_clean_record(row) for row in records], "security", t)
        adjusted = feature_at(
            [FeatureEvent.from_clean_record(row, apply_qa_value_policy=True) for row in records],
            "security",
            t,
        )
        self.assertEqual(old["state_count_1h"], 2)
        self.assertEqual(adjusted["state_count_1h"], 1)
        self.assertEqual(adjusted["event_count_1h"], 2)
        self.assertEqual(adjusted["state_transitions_1h"], 0)

    def test_qa_count_window_boundary(self):
        t = datetime(2025, 6, 15, 12)
        events = [
            FeatureEvent(
                "c", t - timedelta(hours=1), False, qa_value_category="epoch_value_artifact"
            ),
            FeatureEvent("c", t, False, qa_value_category="epoch_value_artifact"),
        ]
        self.assertEqual(qa_window_counts(events, t)["qa_epoch_value_artifact_count_1h"], 1)


if __name__ == "__main__":
    unittest.main()
