from pathlib import Path
import tempfile
import unittest

from stage1.registry import load_registry


class TypeRegistryTests(unittest.TestCase):
    def test_loads_all_types_and_channel_counts(self):
        registry = load_registry()
        self.assertEqual(len(registry), 19)
        self.assertEqual(sum(policy.expected_channels for policy in registry), 11_485)

    def test_mode_queries_respect_not_applicable_and_conditional(self):
        registry = load_registry()
        numeric = {policy.sensor_type for policy in registry.by_mode("numeric")}
        strict_numeric = {
            policy.sensor_type for policy in registry.by_mode("numeric", include_conditional=False)
        }
        self.assertEqual(numeric, {"Датчик температуры", "Газовый датчик", "ИБП"})
        self.assertEqual(strict_numeric, {"Датчик температуры", "Газовый датчик"})
        self.assertTrue(registry.get("КД АВ").requires_confirmation("discrete"))

    def test_unknown_type_is_not_silently_accepted(self):
        with self.assertRaisesRegex(KeyError, "absent from the validated registry"):
            load_registry().get("Новый датчик")

    def test_invalid_matrix_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"schema_version": 1, "types": []}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must contain 19"):
                load_registry(path)


if __name__ == "__main__":
    unittest.main()
