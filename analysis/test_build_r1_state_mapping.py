"""End-to-end tests for the bounded technical state-mapping publisher."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from analysis.build_r1_state_mapping import build


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class R1BuildTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.m1 = self.root / "m1"
        clean = self.m1 / "clean" / "year=2025" / "month=6"
        clean.mkdir(parents=True)
        self.clean_file = clean / "data_0.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "timestamp": datetime(2025, 6, 2, 12),
                        "sensor_type": "Датчик дыма",
                        "value_state": "Неисправен",
                        "alarm": False,
                    },
                    {
                        "timestamp": datetime(2025, 6, 2, 13),
                        "sensor_type": "Датчик дыма",
                        "value_state": "Норма",
                        "alarm": False,
                    },
                    {
                        "timestamp": datetime(2025, 6, 2, 14),
                        "sensor_type": "Датчик дыма",
                        "value_state": None,
                        "alarm": False,
                    },
                ]
            ),
            self.clean_file,
        )
        self.channels = self.root / "channels.csv"
        self.objects = self.root / "objects.csv"
        self.channels.write_text(
            "ид_канала_данных,тип_инж_системы,тип_датчика,"
            "тег_инженерной_системы,название_датчика\n"
            "1,пожарная система,Датчик дыма,tag,датчик\n",
            encoding="utf-8",
        )
        self.objects.write_text(
            "ид_объект,иерархия_уровень,родитель,вид_объекта,"
            "диспетчерское_название_объекта\n"
            "1,1,,здание,тест\n",
            encoding="utf-8",
        )
        self.state_dictionary = self.root / "states.csv"
        self.state_dictionary.write_text(
            "тип_датчика,ид_набор_состояний,название_состояния,тревожное\n"
            "Датчик дыма,1,Норма,false\n",
            encoding="utf-8",
        )
        (self.m1 / "data_quality.json").write_text(
            json.dumps(
                {
                    "scope": "full_supplied_sources",
                    "input_rows": 3,
                    "dispositions": {"accepted": 3},
                    "output_files": [
                        {
                            "path": "clean\\year=2025\\month=6\\data_0.parquet",
                            "bytes": self.clean_file.stat().st_size,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.input_manifest = self.m1 / "manifest.json"
        self._write_manifest()

    def _write_manifest(self) -> None:
        self.input_manifest.write_text(
            json.dumps(
                {
                    "schema_version": "ingestion-v1",
                    "status": "complete",
                    "scope": "full_supplied_sources",
                    "input_rows": 3,
                    "dictionaries": [
                        {"path": str(self.channels), "sha256": _sha256(self.channels)},
                        {"path": str(self.objects), "sha256": _sha256(self.objects)},
                    ],
                }
            ),
            encoding="utf-8",
        )

    def test_build_conserves_text_and_numeric_rows_without_inventing_labels(self) -> None:
        output = self.root / "r1"
        manifest = build(
            input_manifest=self.input_manifest,
            state_dictionary=self.state_dictionary,
            output=output,
        )
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        table = pq.read_table(output / "state_mapping_audit.parquet")
        self.assertEqual(manifest["row_count"], 2)
        self.assertEqual(report["text_rows"], 2)
        self.assertEqual(report["numeric_rows_not_applicable"], 1)
        self.assertEqual(report["match_status_rows"], {"exact_candidate": 1, "unmapped_state": 1})
        self.assertEqual(report["exact_text_neispraven_by_match_status"], {"unmapped_state": 1})
        self.assertEqual(report["top_unmapped_type_state"][0]["state_text_raw"], "Неисправен")
        self.assertNotIn("target", table.schema.names)
        with self.assertRaises(FileExistsError):
            build(
                input_manifest=self.input_manifest,
                state_dictionary=self.state_dictionary,
                output=output,
            )

    def test_changed_m1_dictionary_is_rejected(self) -> None:
        self.channels.write_text("channel\n2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed since ingestion"):
            build(
                input_manifest=self.input_manifest,
                state_dictionary=self.state_dictionary,
                output=self.root / "invalid",
            )

    def test_clean_file_size_change_is_rejected(self) -> None:
        with self.clean_file.open("ab") as stream:
            stream.write(b"altered")
        with self.assertRaisesRegex(ValueError, "byte size differs"):
            build(
                input_manifest=self.input_manifest,
                state_dictionary=self.state_dictionary,
                output=self.root / "invalid-size",
            )

    def test_unreported_clean_file_is_rejected(self) -> None:
        extra = self.clean_file.with_name("data_1.parquet")
        extra.write_bytes(self.clean_file.read_bytes())
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            build(
                input_manifest=self.input_manifest,
                state_dictionary=self.state_dictionary,
                output=self.root / "invalid-inventory",
            )


if __name__ == "__main__":
    unittest.main()
