"""Raw pairing proposals must not silently broaden admission or create targets."""

from datetime import datetime, timedelta
import unittest

import duckdb

from analysis.audit_quality_blockers_a import (
    classify_group,
    match_hours,
    mechanical_hour,
    merged_windows,
)


def message(
    raw,
    number=None,
    *,
    kind="Датчик температуры",
    alarm=False,
    flags=("channel_time_conflict",),
    rows=1,
    excluded_rows=None,
):
    return {
        "sensor_type": kind,
        "value_raw": raw,
        "value_numeric": number,
        "value_state": raw if number is None else None,
        "alarm": alarm,
        "excluded_flags": list(flags),
        "rows": rows,
        "excluded_rows": rows if excluded_rows is None and flags else (excluded_rows or 0),
    }


class PairingTests(unittest.TestCase):
    def test_numeric_temperature_range_is_only_candidate(self):
        result = classify_group([message("9", 9), message("В норме от +3 до +40")])
        self.assertEqual(result["pair_category"], "temperature_range_companion_candidate")
        self.assertTrue(result["compatibility_candidate_requires_review"])
        self.assertEqual(result["excluded_rows"], 2)

    def test_all_same_second_messages_are_considered(self):
        result = classify_group(
            [
                message("9", 9),
                message("В норме от +3 до +40"),
                message("Норма"),
                message("Неисправен", alarm=True),
            ]
        )
        self.assertEqual(result["pair_category"], "registered_fault_and_normal_same_second")
        self.assertTrue(result["registered_state_contradiction"])
        self.assertFalse(result["compatibility_candidate_requires_review"])

    def test_temperature_service_code_stays_protected(self):
        result = classify_group([message("-127", -127), message("Не определено", alarm=True)])
        self.assertEqual(result["pair_category"], "qa_technical_artifact_or_code")
        self.assertIn("temperature_service_code_candidate", result["protected_evidence"])
        self.assertFalse(result["compatibility_candidate_requires_review"])

    def test_temperature_code_with_normal_number_stays_protected(self):
        result = classify_group(
            [message("9", 9), message("-127", -127), message("В норме от +3 до +40")]
        )
        self.assertTrue(result["multiple_numeric_values"])
        self.assertEqual(result["pair_category"], "qa_technical_artifact_or_code")

    def test_disagreeing_temperature_text_is_not_candidate(self):
        result = classify_group([message("2", 2), message("В норме от +3 до +40")])
        self.assertEqual(result["pair_category"], "numeric_temperature_range_disagreement")
        self.assertFalse(result["compatibility_candidate_requires_review"])

    def test_equivalent_numeric_formats_require_review(self):
        result = classify_group(
            [message("0.0", 0, kind="Газовый датчик"), message("0,00", 0, kind="Газовый датчик")]
        )
        self.assertEqual(result["pair_category"], "equivalent_numeric_format_candidate")

    def test_single_visible_variant_retains_scope_ambiguity(self):
        result = classify_group([message("9", 9, rows=3)])
        self.assertEqual(result["pair_category"], "single_variant_visible_in_full_archive")
        self.assertFalse(result["compatibility_candidate_requires_review"])

    def test_alarm_difference_is_not_cleared(self):
        result = classify_group([message("Норма"), message("Норма", alarm=True)])
        self.assertEqual(result["pair_category"], "same_text_alarm_disagreement")
        self.assertTrue(result["alarm_difference"])

    def test_two_numbers_do_not_become_physical_fault(self):
        result = classify_group([message("8", 8), message("9", 9)])
        self.assertEqual(result["pair_category"], "multiple_numeric_values_unordered")
        self.assertFalse(result["registered_state_contradiction"])

    def test_unknown_or_mixed_type_stays_blocked(self):
        for kind in (None, "Газовый датчик"):
            with self.subTest(kind=kind):
                result = classify_group([message("9", 9), message("Норма", kind=kind)])
                self.assertEqual(result["pair_category"], "unknown_or_conflicting_type")

    def test_gas_alarm_and_negative_values_are_not_compatible_norma(self):
        for number in (-0.01, 1.0):
            with self.subTest(number=number):
                result = classify_group(
                    [
                        message(str(number), number, kind="Газовый датчик"),
                        message("Норма", kind="Газовый датчик"),
                    ]
                )
                self.assertEqual(result["pair_category"], "numeric_and_text_unresolved")
                self.assertFalse(result["registered_state_contradiction"])

    def test_nonfinite_quality_flag_is_protected(self):
        result = classify_group([message("NaN", flags=("nonfinite_numeric",)), message("Норма")])
        self.assertEqual(result["pair_category"], "nonfinite_or_invalid_time")

    def test_epoch_artifact_is_protected(self):
        result = classify_group([message("01.01.1970 03:00:00"), message("Норма")])
        self.assertIn("epoch_value_artifact", result["protected_evidence"])

    def test_exact_excluded_row_count_not_boolean_or_overcount(self):
        result = classify_group(
            [message("9", 9, rows=3, excluded_rows=1), message("В норме от +3 до +40", rows=2)]
        )
        self.assertEqual(result["excluded_rows"], 3)


class DiagnosticWindowsTests(unittest.TestCase):
    def test_union_overlaps_preserves_gaps_and_channels(self):
        at = datetime(2025, 2, 1)
        rows = [
            {"channel_id": channel, "prediction_time": time}
            for channel, time in (
                ("x", at),
                ("x", at),
                ("x", at + timedelta(hours=1)),
                ("x", at + timedelta(hours=50)),
                ("y", at),
            )
        ]
        result = merged_windows(rows)
        self.assertEqual(len(result), 3)
        self.assertEqual(
            result[0],
            {
                "channel_id": "x",
                "lower": at - timedelta(hours=24),
                "upper": at + timedelta(hours=1),
            },
        )

    def test_prefix_matches_direct_windows_across_months_and_archive_gap(self):
        at = datetime(2025, 2, 1)
        after_gap = datetime(2022, 1, 1)
        groups = [
            {
                "channel_id": channel,
                "timestamp": time,
                "excluded_rows": count,
                "noncandidate": blocked,
            }
            for channel, time, count, blocked in (
                ("x", at - timedelta(hours=24), 3, 1),
                ("x", at - timedelta(hours=23), 2, 0),
                ("x", at, 4, 1),
                ("x", at + timedelta(seconds=1), 8, 1),
                ("z", datetime(2020, 12, 31), 100, 1),
                ("z", after_gap, 2, 0),
            )
        ]
        positive = [
            {"channel_id": channel, "prediction_time": time}
            for channel, time in (("x", at), ("none", at), ("z", after_gap))
        ]
        with duckdb.connect() as db:
            results = match_hours(db, positive, groups)
        for row in results:
            expected = [
                g
                for g in groups
                if g["channel_id"] == row["channel_id"]
                and row["prediction_time"] - timedelta(hours=24)
                < g["timestamp"]
                <= row["prediction_time"]
            ]
            self.assertEqual(
                row["matched_excluded_rows_24h"], sum(g["excluded_rows"] for g in expected)
            )
            self.assertEqual(
                row["noncandidate_groups_24h"], sum(g["noncandidate"] for g in expected)
            )

    def test_mechanical_proposal_keeps_every_other_guard(self):
        base = {
            "admission_status": "unknown",
            "admission_reasons": ["quality_exclusions_24h"],
            "matched_excluded_rows_24h": 2,
            "noncandidate_groups_24h": 0,
        }
        self.assertTrue(mechanical_hour(base))
        for replacement in (
            {"admission_status": "excluded"},
            {"admission_reasons": ["quality_exclusions_24h", "insufficient_history"]},
            {"admission_reasons": []},
            {"matched_excluded_rows_24h": 0},
            {"noncandidate_groups_24h": 1},
        ):
            with self.subTest(replacement=replacement):
                self.assertFalse(mechanical_hour({**base, **replacement}))
        self.assertTrue(mechanical_hour({**base, "admission_status": "eligible"}))


if __name__ == "__main__":
    unittest.main()
