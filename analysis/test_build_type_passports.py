import collections
import unittest

from analysis.build_type_passports import exact_type_year, parse_number, pct, top_values


class TypePassportTest(unittest.TestCase):
    def test_parse_number_accepts_only_finite_values(self):
        self.assertEqual(parse_number("-12.5"), -12.5)
        self.assertIsNone(parse_number("Неисправен"))
        self.assertIsNone(parse_number("NaN"))
        self.assertIsNone(parse_number("inf"))

    def test_exact_type_year_uses_channel_mapping(self):
        profile = {
            "types": {"A": 8},
            "numeric_by_type": {"A": {"count": 3, "min": 1.0, "max": 2.0}},
            "channels": {"1": 4, "2": 4, "3": 10},
        }
        mapping = {
            "1": {"тип_датчика": "A"},
            "2": {"тип_датчика": "A"},
            "3": {"тип_датчика": "B"},
        }
        row = exact_type_year("A", 2025, profile, mapping)
        self.assertEqual(row["active_channels"], 2)
        self.assertEqual(row["text"], 5)
        self.assertEqual(row["numeric_min"], 1.0)

    def test_top_values_is_explicitly_sample_based(self):
        bucket = {"text": collections.Counter({"Норма": 3, "Неисправен": 1})}
        self.assertEqual(top_values(bucket, "text", 1), "Норма")
        self.assertEqual(pct(1, 4), "25.00%")
        self.assertEqual(pct(0, 0), "—")


if __name__ == "__main__":
    unittest.main()
