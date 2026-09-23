"""R1/B1 fixtures for state semantics and the proposed registered-message target."""

import csv
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from stage1.state_labeling.rules import (
    DICTIONARY_PAIRS,
    RULESET_VERSION,
    TARGET_DEFINITION,
    classify_message,
    in_scope_year,
    review_dictionary,
)
from stage1.ml_m0_contracts import in_future_window


class StateLabelingTests(unittest.TestCase):
    def test_exact_fault_is_an_observed_target_for_known_types_even_with_false_alarm(self):
        for alarm in (False, True):
            for sensor_type in ("Датчик дыма", "Состояние насоса", "КД Дверь"):
                with self.subTest(sensor_type=sensor_type, alarm=alarm):
                    result = classify_message(sensor_type, "Неисправен", alarm)
                    self.assertEqual(result.category, "technical_fault")
                    self.assertEqual(result.source, "project_rule")
                    self.assertEqual(result.status, "project_defined")
                    self.assertTrue(result.target_message_candidate)
                    self.assertIs(result.observed_alarm, alarm)
                    self.assertEqual(result.ruleset_version, RULESET_VERSION)

    def test_other_technical_message_is_not_silently_added_to_target(self):
        battery = classify_message("Состояние вентилятора", "Батарея неисправна", False)
        self.assertEqual(battery.category, "technical_fault")
        self.assertEqual(battery.source, "dictionary_candidate")
        self.assertEqual(battery.status, "candidate_only")
        self.assertFalse(battery.target_message_candidate)

    def test_smoke_alarm_and_working_state_are_not_fault_targets(self):
        smoke = classify_message("Датчик дыма", "Обнаружен дым", True)
        normal = classify_message("Датчик дыма", "Норма", False)
        self.assertEqual(smoke.category, "environmental_alarm")
        self.assertEqual(normal.category, "normal")
        self.assertFalse(smoke.target_message_candidate)
        self.assertFalse(normal.target_message_candidate)

    def test_unknown_type_and_nonexact_text_do_not_become_positive(self):
        for sensor_type, state in (
            (None, "Неисправен"),
            ("unverified type", "Неисправен"),
            ("Датчик дыма", " Неисправен"),
            ("Датчик дыма", "Неисправен "),
            ("Датчик дыма", "неисправен"),
        ):
            with self.subTest(sensor_type=sensor_type, state=state):
                result = classify_message(sensor_type, state, True)
                self.assertEqual(result.category, "unknown")
                self.assertFalse(result.target_message_candidate)

    def test_ambiguous_dictionary_entries_remain_unresolved_regardless_of_alarm(self):
        for alarm in (False, True):
            result = classify_message("Газовый датчик", "Температура ниже 3ºC1", alarm)
            self.assertEqual(result.category, "unknown")
            self.assertEqual(result.source, "unresolved")
            self.assertEqual(result.reason, "conflicting_dictionary_alarm")
            self.assertFalse(result.target_message_candidate)
        mismatched = classify_message("КД АВ", "Обнаружен дым", True)
        self.assertEqual(mismatched.category, "unknown")
        self.assertEqual(mismatched.reason, "type_state_association_requires_review")

    def test_alarm_does_not_choose_category(self):
        first = classify_message("Датчик движения", "Включен", False)
        second = classify_message("Датчик движения", "Включен", True)
        self.assertEqual((first.category, first.source), (second.category, second.source))
        self.assertEqual(first.category, "operational")

    def test_unmapped_and_numeric_states_stay_unknown(self):
        self.assertEqual(
            classify_message("Датчик дыма", "неизвестный код", False).reason,
            "unmapped_state",
        )
        self.assertEqual(
            classify_message("Газовый датчик", None, True).status,
            "not_state_message",
        )

    def test_dictionary_review_detects_conflict_and_does_not_resolve_it_by_alarm(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "states.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    ["тип_датчика", "ид_набор_состояний", "название_состояния", "тревожное"]
                )
                writer.writerow(["Газовый датчик", "13", "Температура ниже 3ºC1", "false"])
                writer.writerow(["Газовый датчик", "13", "Температура ниже 3ºC1", "true"])
                writer.writerow(["Газовый датчик", "13", "Температура ниже 3ºC1", "true"])
                writer.writerow(["КД Дверь", "1", "Норма", "false"])
                writer.writerow(["КД Дверь", "2", "Норма", "false"])
                writer.writerow(["КД Дверь", "1", "новое", "false"])
            report = review_dictionary(path)
        self.assertEqual((report["rows"], report["unique_rows"]), (6, 5))
        self.assertEqual(
            report["contradictory_type_set_states"],
            [("Газовый датчик", "13", "Температура ниже 3ºC1")],
        )
        self.assertEqual(report["unreviewed_pairs"], [("КД Дверь", "новое")])
        self.assertEqual(report["multiple_candidate_sets"], [("КД Дверь", "Норма", ["1", "2"])])
        self.assertEqual(len(report["review_rows"]), 5)
        self.assertEqual(
            {
                row["category"]
                for row in report["review_rows"]
                if row["value_state"] == "Температура ниже 3ºC1"
            },
            {"unknown"},
        )
        self.assertFalse(report["ready_for_joint_review"])

    def test_scope_and_rule_inventory_are_versioned(self):
        self.assertEqual(len(DICTIONARY_PAIRS), 54)
        self.assertEqual(TARGET_DEFINITION["status"], "proposed_pending_joint_r1_review")
        self.assertEqual(TARGET_DEFINITION["horizon_hours"], 24)
        self.assertFalse(in_scope_year(2021))
        self.assertTrue(all(in_scope_year(y) for y in (2019, 2020, 2022, 2023, 2024, 2025, 2026)))

    def test_target_window_excludes_prediction_instant_and_includes_24h_boundary(self):
        at = datetime(2025, 5, 1, 12)
        self.assertFalse(in_future_window(at, at, TARGET_DEFINITION["horizon_hours"]))
        self.assertTrue(in_future_window(at + timedelta(hours=24), at, 24))
        self.assertFalse(in_future_window(at + timedelta(hours=24, seconds=1), at, 24))


if __name__ == "__main__":
    unittest.main()
