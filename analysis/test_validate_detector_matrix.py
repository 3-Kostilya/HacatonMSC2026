import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from validate_detector_matrix import (
    DEFAULT_MATRIX,
    MatrixValidationError,
    load_matrix,
    validate_matrix,
)


class DetectorMatrixValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = load_matrix(DEFAULT_MATRIX)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        dictionary = cls.root / cls.matrix["source_dictionary"]
        dictionary.parent.mkdir(parents=True)
        with dictionary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([cls.matrix["dictionary_type_column"]])
            for entry in cls.matrix["types"]:
                for _ in range(entry["expected_channels"]):
                    writer.writerow([entry["sensor_type"]])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def validate_changed(self, change):
        matrix = copy.deepcopy(self.matrix)
        change(matrix)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "matrix.yaml"
            path.write_text(json.dumps(matrix, ensure_ascii=False), encoding="utf-8")
            return validate_matrix(path, self.root)

    def test_current_matrix_matches_all_19_dictionary_types(self):
        result = validate_matrix(DEFAULT_MATRIX, self.root)
        self.assertEqual(result["types"], 19)
        self.assertEqual(result["channels"], 11_485)
        self.assertEqual(result["modes"], ["context", "discrete", "numeric"])

    def test_missing_dictionary_type_is_rejected(self):
        with self.assertRaisesRegex(MatrixValidationError, "exactly 19 type rows"):
            self.validate_changed(lambda matrix: matrix["types"].pop())

    def test_duplicate_type_is_rejected(self):
        def duplicate(matrix):
            matrix["types"][1]["sensor_type"] = matrix["types"][0]["sensor_type"]

        with self.assertRaisesRegex(MatrixValidationError, "duplicate sensor types"):
            self.validate_changed(duplicate)

    def test_missing_processing_mode_is_rejected(self):
        def remove_mode(matrix):
            del matrix["types"][0]["modes"]["context"]

        with self.assertRaisesRegex(MatrixValidationError, "modes must define exactly"):
            self.validate_changed(remove_mode)

    def test_unknown_feature_is_rejected(self):
        def add_feature(matrix):
            matrix["types"][0]["features"].append("imaginary_detector")

        with self.assertRaisesRegex(MatrixValidationError, "unknown features"):
            self.validate_changed(add_feature)

    def test_changed_dictionary_count_is_rejected(self):
        def change_count(matrix):
            matrix["types"][0]["expected_channels"] += 1

        with self.assertRaisesRegex(MatrixValidationError, "dictionary has"):
            self.validate_changed(change_count)

    def test_empty_unknown_rule_is_rejected(self):
        def empty_unknown(matrix):
            matrix["types"][0]["unknown_if"] = []

        with self.assertRaisesRegex(MatrixValidationError, "unknown_if"):
            self.validate_changed(empty_unknown)


if __name__ == "__main__":
    unittest.main()
