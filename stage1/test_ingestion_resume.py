"""Crash/restart contracts for the disk-backed milestone-one ingestion."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import duckdb
import pyarrow.parquet as pq

from stage1.ingestion.dictionaries import CHANNEL_REQUIRED_COLUMNS, OBJECT_REQUIRED_COLUMNS
from stage1.ingestion.pipeline import run_ingestion
from stage1.ingestion.sources import iter_source_rows
from stage1.normalization import EVENT_FIELDS


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


class IngestionResumeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "events.csv"
        self.channels = self.root / "channels.csv"
        self.objects = self.root / "objects.csv"
        write_csv(
            self.source,
            EVENT_FIELDS,
            [
                [f"event-{n}", "channel-1", "2026-01-01", f"00:0{n}:00", "false", str(n)]
                for n in range(5)
            ],
        )
        write_csv(
            self.channels,
            CHANNEL_REQUIRED_COLUMNS,
            [["channel-1", "system", "numeric", "tag", "name"]],
        )
        write_csv(self.objects, OBJECT_REQUIRED_COLUMNS, [])

    def config(self, name="resumed"):
        return {
            "sources": [{"path": str(self.source), "max_rows": None}],
            "channels": str(self.channels),
            "objects": str(self.objects),
            "output": str(self.root / name),
            "memory_limit": "128MB",
            "batch_size": 2,
            "keep_database": True,
        }

    def interrupt_after_partial_insert(self, config):
        def interrupted_rows(path, max_rows):
            for index, row in enumerate(iter_source_rows(path, max_rows)):
                if index == 3:
                    raise RuntimeError("injected interruption after a committed batch")
                yield row

        with patch("stage1.ingestion.pipeline.iter_source_rows", side_effect=interrupted_rows):
            with self.assertRaisesRegex(RuntimeError, "injected interruption"):
                run_ingestion(config)

        inprogress = Path(config["output"] + ".inprogress")
        self.assertTrue(inprogress.is_dir())
        self.assertFalse(Path(config["output"]).exists())
        manifest = json.loads((inprogress / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "failed")
        with duckdb.connect(str(inprogress / "work.duckdb"), read_only=True) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM raw").fetchone()[0], 2)
        return inprogress

    def test_resume_after_partial_insert_has_no_missing_or_repeated_raw_rows(self):
        config = self.config()
        self.interrupt_after_partial_insert(config)

        report = run_ingestion(config, resume=True)

        self.assertEqual(report["input_rows"], 5)
        self.assertEqual(report["dispositions"], {"accepted": 5})
        output = Path(config["output"])
        self.assertTrue(output.is_dir())
        self.assertFalse(Path(config["output"] + ".inprogress").exists())
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["input_rows"], 5)

        with duckdb.connect(str(output / "work.duckdb"), read_only=True) as connection:
            raw = connection.execute(
                "SELECT row_id, source_row FROM raw ORDER BY row_id"
            ).fetchall()
        self.assertEqual(raw, [(n, n + 1) for n in range(1, 6)])
        clean = [
            row
            for path in sorted((output / "clean").rglob("*.parquet"))
            for row in pq.read_table(path).to_pylist()
        ]
        self.assertEqual(len(clean), 5)
        self.assertEqual({row["row_id"] for row in clean}, {1, 2, 3, 4, 5})

        baseline = run_ingestion(self.config(name="baseline"))
        self.assertEqual(report["dispositions"], baseline["dispositions"])
        baseline_clean = [
            row
            for path in sorted((self.root / "baseline" / "clean").rglob("*.parquet"))
            for row in pq.read_table(path).to_pylist()
        ]
        self.assertEqual(
            sorted(clean, key=lambda row: row["row_id"]),
            sorted(baseline_clean, key=lambda row: row["row_id"]),
        )

    def test_resume_rejects_changed_configuration_without_mutating_checkpoint(self):
        config = self.config()
        inprogress = self.interrupt_after_partial_insert(config)
        changed = {**config, "batch_size": 3}
        manifest_before = (inprogress / "manifest.json").read_bytes()

        with self.assertRaises((ValueError, RuntimeError)):
            run_ingestion(changed, resume=True)

        self.assertFalse(Path(config["output"]).exists())
        self.assertEqual((inprogress / "manifest.json").read_bytes(), manifest_before)
        with duckdb.connect(str(inprogress / "work.duckdb"), read_only=True) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM raw").fetchone()[0], 2)

    def test_resume_accepts_running_checkpoint_left_by_abrupt_termination(self):
        config = self.config()
        inprogress = self.interrupt_after_partial_insert(config)
        manifest_path = inprogress / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "running"
        manifest.pop("error", None)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        report = run_ingestion(config, resume=True)

        self.assertEqual(report["input_rows"], 5)
        self.assertEqual(report["dispositions"], {"accepted": 5})
        self.assertTrue(Path(config["output"]).is_dir())

    def test_resume_rejects_changed_source_without_mutating_checkpoint(self):
        config = self.config()
        inprogress = self.interrupt_after_partial_insert(config)
        with self.source.open("a", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow(
                ["event-5", "channel-1", "2026-01-01", "00:05:00", "false", "5"]
            )
        manifest_before = (inprogress / "manifest.json").read_bytes()

        with self.assertRaises((ValueError, RuntimeError)):
            run_ingestion(config, resume=True)

        self.assertFalse(Path(config["output"]).exists())
        self.assertEqual((inprogress / "manifest.json").read_bytes(), manifest_before)
        with duckdb.connect(str(inprogress / "work.duckdb"), read_only=True) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM raw").fetchone()[0], 2)

    def test_completed_output_is_never_overwritten_by_resume(self):
        config = self.config()
        run_ingestion(config)
        manifest_path = Path(config["output"]) / "manifest.json"
        before = manifest_path.read_bytes()

        with self.assertRaises(FileExistsError):
            run_ingestion(config, resume=True)

        self.assertEqual(manifest_path.read_bytes(), before)
        self.assertFalse(Path(config["output"] + ".inprogress").exists())


if __name__ == "__main__":
    unittest.main()
