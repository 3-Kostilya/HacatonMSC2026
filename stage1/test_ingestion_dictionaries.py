from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from stage1.ingestion.dictionaries import load_dictionaries


CHANNEL_COLUMNS = [
    "ид_канала_данных",
    "тип_инж_системы",
    "тип_датчика",
    "тег_инженерной_системы",
    "название_датчика",
]
OBJECT_COLUMNS = [
    "ид_объект",
    "иерархия_уровень",
    "родитель",
    "вид_объекта",
    "диспетчерское_название_объекта",
]


class DictionaryIngestionTests(unittest.TestCase):
    def write_csv(self, path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def load(self, channels, objects, channel_columns=CHANNEL_COLUMNS):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            channels_path, objects_path = root / "channels.csv", root / "objects.csv"
            self.write_csv(channels_path, channel_columns, channels)
            self.write_csv(objects_path, OBJECT_COLUMNS, objects)
            return load_dictionaries(channels_path, objects_path)

    def test_missing_channel_object_mapping_is_not_inferred(self):
        channels, _, audit = self.load(
            [
                {
                    "ид_канала_данных": " 42 ",
                    "тип_инж_системы": "fire",
                    "тип_датчика": "smoke",
                    "тег_инженерной_системы": "object-1",
                    "название_датчика": "Smoke detector",
                }
            ],
            [],
        )
        self.assertIsNone(channels[0]["object_id"])
        self.assertEqual(channels[0]["channel_id"], "42")
        self.assertFalse(audit["object_mapping_available"])

    def test_unknown_object_link_is_preserved_and_audited(self):
        columns = [*CHANNEL_COLUMNS, "ид_объект"]
        channels, _, audit = self.load(
            [{**self.channel("42"), "ид_объект": " missing "}], [], columns
        )
        self.assertEqual(channels[0]["object_id"], "missing")
        self.assertEqual(audit["absent_channel_object_foreign_keys"], 1)

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate channel_id"):
            self.load([self.channel("42"), self.channel("42")], [])
        with self.assertRaisesRegex(ValueError, "duplicate object_id"):
            self.load([], [self.object("1"), self.object("1")])

    def test_short_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            channels_path, objects_path = root / "channels.csv", root / "objects.csv"
            channels_path.write_text(
                ",".join(CHANNEL_COLUMNS) + "\n42,fire,smoke\n", encoding="utf-8"
            )
            self.write_csv(objects_path, OBJECT_COLUMNS, [])
            with self.assertRaisesRegex(ValueError, "record 2 has too few fields"):
                load_dictionaries(channels_path, objects_path)

    def test_long_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            channels_path, objects_path = root / "channels.csv", root / "objects.csv"
            channels_path.write_text(
                ",".join(CHANNEL_COLUMNS) + "\n42,fire,smoke,tag,name,extra\n", encoding="utf-8"
            )
            self.write_csv(objects_path, OBJECT_COLUMNS, [])
            with self.assertRaisesRegex(ValueError, "record 2 has too many fields"):
                load_dictionaries(channels_path, objects_path)

    def test_known_object_link_and_external_parent_are_reported(self):
        columns = [*CHANNEL_COLUMNS, "ид_объект"]
        channels, objects, audit = self.load(
            [{**self.channel("42"), "ид_объект": "1"}],
            [self.object("1", parent="outside")],
            columns,
        )
        self.assertEqual(channels[0]["object_id"], objects[0]["object_id"])
        self.assertEqual(audit["absent_channel_object_foreign_keys"], 0)
        self.assertEqual(audit["missing_parent_ids"], ["outside"])

    def test_hierarchy_cycles_are_reported(self):
        _, _, audit = self.load(
            [], [self.object("a", parent="b"), self.object("b", parent="a"), self.object("c")]
        )
        self.assertEqual(audit["hierarchy_cycles"], [["a", "b"]])

    @staticmethod
    def channel(channel_id: str) -> dict[str, str]:
        return {
            "ид_канала_данных": channel_id,
            "тип_инж_системы": "fire",
            "тип_датчика": "smoke",
            "тег_инженерной_системы": "tag",
            "название_датчика": "Smoke detector",
        }

    @staticmethod
    def object(object_id: str, parent: str = "") -> dict[str, str]:
        return {
            "ид_объект": object_id,
            "иерархия_уровень": "1",
            "родитель": parent,
            "вид_объекта": "building",
            "диспетчерское_название_объекта": "Building",
        }


if __name__ == "__main__":
    unittest.main()
