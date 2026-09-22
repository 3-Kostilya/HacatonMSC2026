"""B verifies both real invariants and detection of corrupted handoffs."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from analysis.audit_m1_handoff import FIELDS, audit_handoff
from stage1.ingestion.pipeline import run_ingestion


class HandoffAuditTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        fixture = Path(__file__).resolve().parents[1] / "stage1/fixtures/m0"
        source = self.root / "events.csv"
        with source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(FIELDS)
            writer.writerows(
                [
                    ["same-id", "numeric", "2026-01-01", "00:00:00", "false", "-3276"],
                    ["same-id", "numeric", "2026-01-01", "00:00:00", "false", "-3276"],
                    ["same-id", "numeric", "2026-01-01", "00:00:00", "true", "327.68"],
                    ["third", "numeric", "2026-01-01", "01:00:00", "false", "999"],
                    ["text", "state", "2026-01-01", "01:00:00", "false", "Норма"],
                    ["nan", "numeric", "2026-01-01", "02:00:00", "false", "NaN"],
                    ["unknown", "missing", "2026-01-01", "01:00:00", "false", "Норма"],
                    ["bad", "numeric", "bad-date", "01:00:00", "false", "save me"],
                    list(FIELDS),
                ]
            )
        self.output = self.root / "handoff"
        run_ingestion(
            {
                "sources": [{"path": str(source), "max_rows": None}],
                "channels": str(fixture / "channels.csv"),
                "objects": str(fixture / "objects.csv"),
                "output": str(self.output),
                "batch_size": 2,
                "memory_limit": "128MB",
            }
        )

    def test_audit_preserves_special_values_repeated_id_and_raw_rows(self):
        report = audit_handoff(self.output)
        self.assertTrue(report["passed"], report["checks"])
        self.assertEqual(report["input_rows"], 9)
        self.assertEqual(
            report["dispositions"],
            {"accepted": 6, "exact_duplicate": 1, "quarantine": 1, "repeated_header": 1},
        )
        self.assertEqual(report["repeated_event_ids_retained"], 1)
        self.assertFalse(report["object_mapping_available"])

    def test_changed_raw_value_is_detected_even_if_counts_match(self):
        path = next((self.output / "clean").rglob("*.parquet"))
        table = pq.ParquetFile(path).read()
        rows = table.to_pylist()
        rows[0]["value_raw"] = "silently changed"
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
        report = audit_handoff(self.output)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["raw_fields_preserved"])

    def test_same_id_row_deletion_is_detected(self):
        path = next((self.output / "clean").rglob("*.parquet"))
        table = pq.ParquetFile(path).read()
        rows = table.to_pylist()
        index = next(i for i, row in enumerate(rows) if row["value_raw"] == "327.68")
        del rows[index]
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
        report = audit_handoff(self.output)
        self.assertFalse(report["checks"]["row_balance"])
        self.assertFalse(report["checks"]["raw_fields_preserved"])

    def test_2021_rejected_before_source_access(self):
        path = self.output / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["sources"][0]["path"] = str(self.root / "ext-journal-2021.7z")
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "2021 is excluded"):
            audit_handoff(self.output)

    def test_real_object_mapping_and_corrupted_link(self):
        config = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))[
            "configuration"
        ]
        channels = self.root / "channels-with-links.csv"
        with Path(config["channels"]).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        with channels.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[*fields, "ид_объект"])
            writer.writeheader()
            writer.writerows({**row, "ид_объект": "fixture-object"} for row in rows)
        config.update(channels=str(channels), output=str(self.root / "linked"))
        run_ingestion(config)
        report = audit_handoff(config["output"])
        self.assertTrue(report["passed"], report["checks"])
        path = next((Path(config["output"]) / "clean").rglob("*.parquet"))
        table = pq.ParquetFile(path).read()
        rows = table.to_pylist()
        rows[0]["object_id"] = "invented-object"
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
        report = audit_handoff(config["output"])
        self.assertFalse(report["checks"]["object_links_match_dictionary"])

    def test_incomplete_run_is_rejected(self):
        path = self.output / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["status"] = "failed"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "complete"):
            audit_handoff(self.output)


if __name__ == "__main__":
    unittest.main()
