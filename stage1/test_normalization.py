from __future__ import annotations

import unittest

from stage1.normalization import (
    CHANNEL_TIME_CONFLICT,
    EXACT_DUPLICATE,
    INVALID_ALARM,
    NONFINITE_NUMERIC,
    TIMEZONE_PRESENT,
    UNKNOWN_CHANNEL,
    iter_accepted,
    normalize_chunks,
    parse_alarm,
)


def row(
    event_id="1",
    channel="10",
    date="2026-08-01",
    time="03:09:27",
    alarm="false",
    value="28",
):
    return {
        "ид_события": event_id,
        "ид_канала_данных": channel,
        "дата": date,
        "время": time,
        "тревожное": alarm,
        "значение_датчика": value,
    }


class NormalizationTests(unittest.TestCase):
    types = {"10": "Датчик температуры", "11": "Датчик дыма"}

    def normalize(self, *chunks):
        return list(normalize_chunks(chunks, self.types, "fixture.csv"))

    def test_numeric_and_lossless_text_are_separate(self):
        result = self.normalize([row(value=" 28,50 ")])[0]
        self.assertEqual(result.event.raw_value, " 28,50 ")
        self.assertEqual(result.event.numeric_value, 28.5)
        self.assertEqual(result.source_row, 2)

    def test_explicit_source_row_survives_upstream_filtering(self):
        source = row()
        source["__source_row__"] = 101
        result = self.normalize([source])[0]
        self.assertEqual(result.source_row, 101)

    def test_text_state_has_no_numeric_value(self):
        result = self.normalize([row(channel="11", value="Неисправен")])[0]
        self.assertIsNone(result.event.numeric_value)
        self.assertEqual(result.event.raw_value, "Неисправен")

    def test_alarm_parser_is_explicit(self):
        for value in ("true", "T", "1"):
            self.assertTrue(parse_alarm(value))
        for value in ("false", "F", "0"):
            self.assertFalse(parse_alarm(value))
        with self.assertRaises(ValueError):
            parse_alarm("yes")

    def test_invalid_alarm_is_rejected(self):
        result = self.normalize([row(alarm="yes")])[0]
        self.assertEqual(result.disposition, "rejected")
        self.assertIsNone(result.event)
        self.assertIn(INVALID_ALARM, result.quality_flags)

    def test_timezone_timestamp_is_rejected(self):
        result = self.normalize([row(time="03:09:27+03:00")])[0]
        self.assertEqual(result.disposition, "rejected")
        self.assertIn(TIMEZONE_PRESENT, result.quality_flags)

    def test_unknown_channel_is_retained_and_flagged(self):
        result = self.normalize([row(channel="999")])[0]
        self.assertEqual(result.disposition, "accepted")
        self.assertEqual(result.event.sensor_type, "unknown")
        self.assertIn(UNKNOWN_CHANNEL, result.quality_flags)

    def test_nonfinite_number_does_not_enter_numeric_series(self):
        result = self.normalize([row(value="NaN")])[0]
        self.assertIsNone(result.event.numeric_value)
        self.assertIn(NONFINITE_NUMERIC, result.quality_flags)

    def test_duplicate_rule_works_across_chunk_boundary(self):
        duplicate = row()
        results = self.normalize([duplicate], [dict(duplicate)])
        self.assertEqual([r.disposition for r in results], ["accepted", EXACT_DUPLICATE])
        self.assertEqual(results[1].duplicate_of_source_row, 2)
        self.assertIn(EXACT_DUPLICATE, results[1].event.quality_flags)
        self.assertEqual(len(list(iter_accepted(results))), 1)

    def test_event_id_alone_is_not_a_key(self):
        results = self.normalize(
            [row(event_id="same", channel="10"), row(event_id="same", channel="11")]
        )
        self.assertEqual([r.disposition for r in results], ["accepted", "accepted"])

    def test_conflict_marks_every_row_and_does_not_choose(self):
        results = self.normalize([row(value="28")], [row(event_id="2", value="99", alarm="true")])
        self.assertEqual(len(list(iter_accepted(results))), 2)
        self.assertTrue(all(CHANNEL_TIME_CONFLICT in r.quality_flags for r in results))

    def test_same_semantics_with_different_ids_is_not_conflict(self):
        results = self.normalize([row(event_id="1"), row(event_id="2")])
        self.assertTrue(all(CHANNEL_TIME_CONFLICT not in r.quality_flags for r in results))


if __name__ == "__main__":
    unittest.main()
