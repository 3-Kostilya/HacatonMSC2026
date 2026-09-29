"""R1 accepted archive assumption boundaries and as-of cohort checks."""

from datetime import datetime, timedelta
import unittest

from stage1.state_labeling.operational import (
    future_window_in_archive, recent_explicit_normal, recent_normal_for_prediction,
    registered_state_effect, same_timestamp_state_conflict, segment_at,
    source_is_full_archive,
)


class OperationalTargetTests(unittest.TestCase):
    def test_explicit_archive_segments_exclude_gaps_and_isolated_example(self):
        self.assertEqual(segment_at(datetime(2020, 12, 31, 23)), 0)
        self.assertIsNone(segment_at(datetime(2021, 1, 1)))
        self.assertEqual(segment_at(datetime(2022, 1, 1)), 1)
        self.assertIsNone(segment_at(datetime(2026, 7, 1)))
        self.assertIsNone(segment_at(datetime(2026, 8, 1)))

    def test_full_archive_source_only(self):
        at = datetime(2025, 6, 1)
        self.assertTrue(source_is_full_archive(r"C:\data\ext-journal-2025.7z", at))
        self.assertFalse(source_is_full_archive("журнал_событий_пример.csv", at))
        self.assertFalse(source_is_full_archive("ext-journal-2024.7z", at))
        self.assertFalse(source_is_full_archive("ext-journal-2026.7z",
                                                datetime(2026, 8, 1)))

    def test_future_window_stays_inside_archive(self):
        self.assertTrue(future_window_in_archive(datetime(2025, 6, 1)))
        self.assertFalse(future_window_in_archive(datetime(2020, 12, 31, 0)))
        self.assertFalse(future_window_in_archive(datetime(2026, 6, 30, 0)))
        self.assertFalse(future_window_in_archive(datetime(2026, 8, 1)))

    def test_recent_normal_is_causal_and_cannot_cross_missing_segment(self):
        at = datetime(2023, 1, 8)
        self.assertTrue(recent_explicit_normal(at - timedelta(hours=168), at))
        self.assertFalse(recent_explicit_normal(at - timedelta(hours=169), at))
        self.assertFalse(recent_explicit_normal(at, at))
        self.assertTrue(recent_normal_for_prediction(at, at))
        self.assertTrue(recent_normal_for_prediction(at - timedelta(hours=168), at))
        self.assertFalse(recent_normal_for_prediction(at - timedelta(hours=169), at))
        self.assertFalse(recent_explicit_normal(at + timedelta(minutes=1), at))
        self.assertFalse(recent_explicit_normal(datetime(2020, 12, 31),
                                                datetime(2022, 1, 1)))

    def test_same_second_conflict_ignores_numeric_but_not_other_state(self):
        self.assertFalse(same_timestamp_state_conflict(["Норма", "Норма", None]))
        self.assertTrue(same_timestamp_state_conflict(["Неисправен", "Норма"]))
        self.assertTrue(same_timestamp_state_conflict(["Неисправен", "Обесточен"]))

    def test_only_exact_registered_text_changes_health_state(self):
        smoke = "Датчик дыма"
        self.assertEqual(registered_state_effect(smoke, "Норма", True), "normal")
        self.assertEqual(registered_state_effect(smoke, "Неисправен", False), "fault")
        self.assertEqual(registered_state_effect(smoke, "Обнаружен дым", True), "neutral")
        self.assertEqual(registered_state_effect(smoke, "Дыма нет", False), "neutral")
        self.assertEqual(registered_state_effect(smoke, "Рычаг норма", False), "uncertain")
        self.assertEqual(registered_state_effect("Состояние вентилятора",
                                                 "Батарея неисправна", False), "uncertain")
        self.assertEqual(registered_state_effect("Состояние насоса", "Выключен", False),
                         "uncertain")
        self.assertEqual(registered_state_effect(smoke, None, False), "neutral")
        self.assertEqual(registered_state_effect(None, "Неисправен", True), "uncertain")


if __name__ == "__main__":
    unittest.main()
