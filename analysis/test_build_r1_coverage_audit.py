"""Checks that observed bounds never become certified coverage."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from analysis.build_r1_coverage_audit import _missing_months, build


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CoverageAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.m1 = self.root / "m1"
        clean = self.m1 / "clean" / "year=2019" / "month=1"
        clean.mkdir(parents=True)
        self.clean_file = clean / "data_0.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "channel_id": "1",
                        "sensor_type": "Датчик дыма",
                        "source": "source.csv",
                        "timestamp": datetime(2019, 1, 1, 10),
                        "value_state": "Норма",
                    },
                    {
                        "channel_id": "1",
                        "sensor_type": "Датчик дыма",
                        "source": "source.csv",
                        "timestamp": datetime(2019, 1, 3, 10),
                        "value_state": "Неисправен",
                    },
                    {
                        "channel_id": "2",
                        "sensor_type": None,
                        "source": "source.csv",
                        "timestamp": datetime(2019, 1, 2, 10),
                        "value_state": "Неисправен",
                    },
                ]
            ),
            self.clean_file,
        )
        self.channels = self.root / "channels.csv"
        self.channels.write_text(
            "ид_канала_данных,тип_инж_системы,тип_датчика,"
            "тег_инженерной_системы,название_датчика\n"
            "1,пожарная система,Датчик дыма,tag,датчик\n",
            encoding="utf-8",
        )
        self.objects = self.root / "objects.csv"
        self.objects.write_text(
            "ид_объект,иерархия_уровень,родитель,вид_объекта,"
            "диспетчерское_название_объекта\n"
            "1,1,,здание,тест\n",
            encoding="utf-8",
        )
        (self.m1 / "data_quality.json").write_text(
            json.dumps(
                {
                    "scope": "full_supplied_sources",
                    "input_rows": 3,
                    "dispositions": {"accepted": 3},
                    "by_partition": [{"year": 2019, "month": 1, "rows": 3}],
                    "output_files": [
                        {
                            "path": "clean\\year=2019\\month=1\\data_0.parquet",
                            "bytes": self.clean_file.stat().st_size,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.manifest = self.m1 / "manifest.json"
        self.manifest.write_text(
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

    def test_only_observed_bounds_are_published(self) -> None:
        output = self.root / "coverage"
        build(input_manifest=self.manifest, output=output)
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        rows = pq.read_table(output / "channel_observation_bounds.parquet").to_pylist()
        self.assertEqual(report["accepted_rows"], 3)
        self.assertEqual(report["known_type_fault_text_rows"], 1)
        self.assertEqual(report["unknown_type_fault_text_rows"], 1)
        self.assertEqual(report["observed_unknown_channel_rows"], 1)
        self.assertEqual(report["confirmed_continuous_channel_intervals"], 0)
        self.assertFalse(report["future_negative_labels_authorized"])
        self.assertEqual(sum(row["event_rows"] for row in rows), 3)
        self.assertTrue(
            all(row["continuous_observation_status"] == "unverified_events_only" for row in rows)
        )
        self.assertEqual(rows[0]["active_days"], 2)
        self.assertEqual(rows[0]["first_observed_at"], datetime(2019, 1, 1, 10))
        with self.assertRaises(FileExistsError):
            build(input_manifest=self.manifest, output=output)

    def test_missing_months_include_excluded_year(self) -> None:
        quality = {"by_partition": [{"year": 2020, "month": 12}, {"year": 2022, "month": 1}]}
        self.assertEqual(len(_missing_months(quality)), 12)
        self.assertEqual(_missing_months(quality)[0], "2021-01")
        self.assertEqual(_missing_months(quality)[-1], "2021-12")


if __name__ == "__main__":
    unittest.main()
