"""Independent R1 checks for technical dictionary matching, not fault labels."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from stage1.state_mapping import (
    MAPPING_VERSION,
    STATE_MAPPING_SCHEMA,
    build_audit_table,
    load_state_dictionary,
    match_state,
)


_HEADER = ("тип_датчика", "ид_набор_состояний", "название_состояния", "тревожное")
_ROWS = (
    ("КД Дверь", "1", "Норма", "false"),
    ("КД Дверь", "1", "Норма", "false"),  # exact source duplicate
    ("КД Дверь", "2", "Норма", "false"),  # another candidate set
    ("КД Дверь", "1", "Неисправен", "true"),
    ("Состояние вентилятора", "16", "Батарея неисправна", "false"),
    ("КД АВ", "5", "Обнаружен дым", "true"),
    ("Газовый датчик", "13", "Температура ниже 3ºC1", "false"),
    ("Газовый датчик", "13", "Температура ниже 3ºC1", "true"),
    ("Газовый датчик", "13", "Температура ниже 3ºC1", "true"),  # exact duplicate
    ("Датчик дыма", "11", "Рычаг норма", "false"),
)


class StateMappingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.csv_path = Path(self.temporary.name) / "справочник_состояний.csv"
        with self.csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(_HEADER)
            writer.writerows(_ROWS)
        self.original_bytes = self.csv_path.read_bytes()
        self.dictionary = load_state_dictionary(self.csv_path)

    def match(self, sensor_type: str, state: str, observed_alarm: bool):
        return match_state(self.dictionary, sensor_type, state, observed_alarm)

    def test_exact_duplicate_does_not_create_another_candidate_or_mutate_csv(self):
        result = self.match("КД Дверь", "Неисправен", True)
        self.assertEqual(result["match_status"], "exact_candidate")
        self.assertEqual(result["candidate_set_ids"], ["1"])
        self.assertEqual(result["expected_alarm_values"], [True])
        self.assertIs(result["expected_alarm"], True)
        self.assertEqual(result["alarm_consistency"], "agree")
        self.assertEqual(self.csv_path.read_bytes(), self.original_bytes)

    def test_multiple_sets_remain_ambiguous_even_when_alarm_agrees(self):
        result = self.match("КД Дверь", "Норма", False)
        self.assertEqual(result["match_status"], "multiple_candidates")
        self.assertEqual(result["candidate_set_ids"], ["1", "2"])
        self.assertEqual(result["expected_alarm_values"], [False])
        self.assertIs(result["expected_alarm"], False)
        self.assertEqual(result["alarm_consistency"], "agree")
        self.assertEqual(result["definition_source_rows"], [2, 3, 4])

    def test_conflicting_definition_is_not_resolved_by_observed_alarm(self):
        no_alarm = self.match("Газовый датчик", "Температура ниже 3ºC1", False)
        alarm = self.match("Газовый датчик", "Температура ниже 3ºC1", True)
        for result in (no_alarm, alarm):
            self.assertEqual(result["match_status"], "conflicting_definition")
            self.assertEqual(result["candidate_set_ids"], ["13"])
            self.assertEqual(result["expected_alarm_values"], [False, True])
            self.assertIsNone(result["expected_alarm"])
            self.assertEqual(result["alarm_consistency"], "undetermined")
            self.assertEqual(result["definition_source_rows"], [8, 9, 10])

    def test_observed_alarm_mismatch_changes_consistency_not_dictionary_status(self):
        matching = self.match("КД Дверь", "Неисправен", True)
        mismatching = self.match("КД Дверь", "Неисправен", False)
        self.assertEqual(matching["match_status"], mismatching["match_status"])
        self.assertEqual(matching["candidate_set_ids"], mismatching["candidate_set_ids"])
        self.assertIs(mismatching["expected_alarm"], True)
        self.assertEqual(mismatching["alarm_consistency"], "disagree")

    def test_fault_word_and_smoke_alarm_are_not_fault_labels(self):
        battery = self.match("Состояние вентилятора", "Батарея неисправна", False)
        smoke = self.match("КД АВ", "Обнаружен дым", True)
        self.assertEqual(battery["match_status"], "exact_candidate")
        self.assertIs(battery["expected_alarm"], False)
        self.assertEqual(smoke["match_status"], "exact_candidate")
        self.assertIs(smoke["expected_alarm"], True)
        # A technical mapping supplies expected alarm only; it cannot supply a
        # physical-failure label or silently move a message between types.
        self.assertNotIn("target", battery)
        self.assertNotIn("target", smoke)
        self.assertEqual(
            self.match("Датчик дыма", "Обнаружен дым", True)["match_status"],
            "unmapped_state",
        )

    def test_unknown_type_and_unknown_state_are_distinct(self):
        self.assertEqual(self.match("ИБП", "Неисправен", True)["match_status"], "unmapped_type")
        self.assertEqual(
            self.match("КД Дверь", "Состояние без определения", False)["match_status"],
            "unmapped_state",
        )

    def test_whitespace_match_keeps_original_observation_in_audit(self):
        result = self.match("КД Дверь", "  Неисправен  ", True)
        self.assertEqual(result["match_status"], "exact_candidate")
        self.assertEqual(result["state_match_key"], "Неисправен")
        table = build_audit_table(
            [
                {
                    "year": 2025,
                    "sensor_type": "КД Дверь",
                    "state_text_raw": "  Неисправен  ",
                    "observed_alarm": True,
                    "row_count": 7,
                }
            ],
            self.dictionary,
            input_manifest_sha256="a" * 64,
        )
        row = table.to_pylist()[0]
        self.assertEqual(row["state_text_raw"], "  Неисправен  ")
        self.assertEqual(row["state_match_key"], "Неисправен")
        self.assertEqual(row["row_count"], 7)

    def test_aggregated_rows_are_not_multiplied_by_dictionary_duplicates(self):
        aggregates = [
            {
                "year": 2025,
                "sensor_type": "КД Дверь",
                "state_text_raw": "Норма",
                "observed_alarm": False,
                "row_count": 100,
            },
            {
                "year": 2025,
                "sensor_type": "Газовый датчик",
                "state_text_raw": "Температура ниже 3ºC1",
                "observed_alarm": True,
                "row_count": 9,
            },
            {
                "year": 2026,
                "sensor_type": "ИБП",
                "state_text_raw": "Новый режим",
                "observed_alarm": False,
                "row_count": 5,
            },
        ]
        table = build_audit_table(aggregates, self.dictionary, input_manifest_sha256="b" * 64)
        self.assertTrue(table.schema.equals(STATE_MAPPING_SCHEMA, check_metadata=False))
        rows = table.to_pylist()
        self.assertEqual(len(rows), len(aggregates))
        self.assertEqual(sum(row["row_count"] for row in rows), 114)
        self.assertEqual(
            {row["row_count"]: row["match_status"] for row in rows},
            {100: "multiple_candidates", 9: "conflicting_definition", 5: "unmapped_type"},
        )
        self.assertTrue(MAPPING_VERSION)

    def test_invalid_manifest_hash_and_fractional_count_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "SHA-256 hex"):
            build_audit_table([], self.dictionary, input_manifest_sha256="z" * 64)
        aggregate = {
            "year": 2025,
            "sensor_type": "КД Дверь",
            "state_text_raw": "Норма",
            "observed_alarm": False,
            "row_count": 1.5,
        }
        with self.assertRaisesRegex(ValueError, "must be integers"):
            build_audit_table([aggregate], self.dictionary, input_manifest_sha256="a" * 64)


if __name__ == "__main__":
    unittest.main()
