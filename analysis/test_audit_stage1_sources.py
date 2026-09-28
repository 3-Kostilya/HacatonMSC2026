import io
import unittest

from audit_stage1_sources import (
    EVENT_SCHEMA,
    _probe_csv_stream,
    audit_object_links,
    detect_utf8_encoding,
    parse_7z_listing,
    validate_grouping,
)


class SourceAuditTests(unittest.TestCase):
    def test_parse_7z_listing_extracts_member_metadata(self):
        listing = "header\n----------\nPath = journal.csv\nSize = 42\nCRC = ABCD\n"
        self.assertEqual(
            parse_7z_listing(listing), [{"Path": "journal.csv", "Size": "42", "CRC": "ABCD"}]
        )

    def test_encoding_accepts_utf8_with_and_without_bom(self):
        self.assertEqual(detect_utf8_encoding(b"\xef\xbb\xbfabc"), "utf-8-sig")
        self.assertEqual(detect_utf8_encoding("датчик".encode()), "utf-8")
        with self.assertRaises(UnicodeDecodeError):
            detect_utf8_encoding(b"\xff")

    def test_probe_is_bounded_and_maps_type_to_group(self):
        header = ",".join(f'"{c}"' for c in EVENT_SCHEMA)
        data = (
            header
            + "\n1,10,2026-01-01,00:00:00,false,12.5"
            + "\n2,99,2026-01-02,00:00:00,true,Неисправен\n"
        ).encode()
        result = _probe_csv_stream(io.BytesIO(data), {"10": "Датчик температуры"}, 1)
        self.assertEqual(result["rows_probed"], 1)
        self.assertEqual(result["numeric_value_rows"], 1)
        self.assertEqual(result["rows_by_group"], {"Числовые измерения среды": 1})

    def test_unknown_dictionary_type_is_reported(self):
        result = validate_grouping(["Датчик температуры", "Новый тип"])
        self.assertEqual(result["unmapped_dictionary_types"], ["Новый тип"])

    def test_probe_rejects_schema_drift(self):
        with self.assertRaisesRegex(ValueError, "Unexpected event schema"):
            _probe_csv_stream(io.BytesIO(b"a,b\n1,2\n"), {}, 1)

    def test_object_links_report_external_parent_and_cycle(self):
        rows = [
            {"ид_объект": "1", "родитель": "99"},
            {"ид_объект": "2", "родитель": "3"},
            {"ид_объект": "3", "родитель": "2"},
        ]
        result = audit_object_links(rows)
        self.assertEqual(result["missing_parent_ids"], ["99"])
        self.assertEqual(set(result["cycle_start_ids"]), {"2", "3"})


if __name__ == "__main__":
    unittest.main()
