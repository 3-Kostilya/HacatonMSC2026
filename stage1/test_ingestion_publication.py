"""Crash-safety contracts for ingestion manifest checkpoints and publication."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from stage1.ingestion.checkpoints import atomic_write_json
from stage1.ingestion.dictionaries import CHANNEL_REQUIRED_COLUMNS, OBJECT_REQUIRED_COLUMNS
from stage1.ingestion.pipeline import recover_derived_ingestion, run_ingestion
from stage1.normalization import EVENT_FIELDS


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


class IngestionPublicationTests(unittest.TestCase):
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
            [["event-1", "channel-1", "2026-01-01", "00:01:00", "false", "1"]],
        )
        write_csv(
            self.channels,
            CHANNEL_REQUIRED_COLUMNS,
            [["channel-1", "system", "numeric", "tag", "name"]],
        )
        write_csv(self.objects, OBJECT_REQUIRED_COLUMNS, [])
        self.config = {
            "sources": [{"path": str(self.source), "max_rows": None}],
            "channels": str(self.channels),
            "objects": str(self.objects),
            "output": str(self.root / "published"),
            "memory_limit": "128MB",
            "batch_size": 1,
            "keep_database": False,
        }

    def test_atomic_json_failure_preserves_previous_checkpoint(self):
        checkpoint = self.root / "manifest.json"
        atomic_write_json(checkpoint, {"generation": 1})

        with (
            patch("stage1.ingestion.checkpoints.os.replace", side_effect=OSError("crash")),
            self.assertRaisesRegex(OSError, "crash"),
        ):
            atomic_write_json(checkpoint, {"generation": 2})

        self.assertEqual(json.loads(checkpoint.read_text(encoding="utf-8")), {"generation": 1})
        self.assertEqual(list(self.root.glob(".manifest.json.*.tmp")), [])

    def test_retry_finishes_publication_after_database_deletion(self):
        destination = Path(self.config["output"])
        inprogress = destination.with_name(destination.name + ".inprogress")
        with (
            patch(
                "stage1.ingestion.pipeline._rename_publication_directory",
                side_effect=OSError("rename interrupted"),
            ),
            self.assertRaisesRegex(OSError, "rename interrupted"),
        ):
            run_ingestion(self.config)

        manifest = json.loads((inprogress / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual((manifest["status"], manifest["phase"]), ("running", "publishing"))
        self.assertFalse((inprogress / "work.duckdb").exists())
        self.assertFalse(destination.exists())

        with patch(
            "stage1.ingestion.pipeline.iter_source_rows",
            side_effect=AssertionError("publication retry must not re-ingest"),
        ):
            report = run_ingestion(self.config)

        self.assertEqual(report["input_rows"], 1)
        self.assertFalse(inprogress.exists())
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual((manifest["status"], manifest["phase"]), ("complete", "complete"))

    def test_retry_finishes_manifest_after_directory_was_published(self):
        from stage1.ingestion.pipeline import atomic_write_json as real_atomic_write_json

        destination = Path(self.config["output"])

        def interrupt_final_manifest(path, payload, **kwargs):
            if Path(path).parent == destination and payload.get("status") == "complete":
                raise OSError("final manifest interrupted")
            return real_atomic_write_json(path, payload, **kwargs)

        with (
            patch(
                "stage1.ingestion.pipeline.atomic_write_json",
                side_effect=interrupt_final_manifest,
            ),
            self.assertRaisesRegex(OSError, "final manifest interrupted"),
        ):
            run_ingestion(self.config)

        on_disk = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual((on_disk["status"], on_disk["phase"]), ("running", "publishing"))

        with patch(
            "stage1.ingestion.pipeline.iter_source_rows",
            side_effect=AssertionError("publication retry must not re-ingest"),
        ):
            run_ingestion(self.config)
        completed = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual((completed["status"], completed["phase"]), ("complete", "complete"))

    def test_derived_recovery_uses_same_replayable_publication(self):
        config = dict(self.config)
        config["output"] = str(self.root / "recovered")
        config["keep_database"] = True
        run_ingestion(config)
        destination = Path(config["output"])
        inprogress = destination.with_name(destination.name + ".inprogress")
        destination.rename(inprogress)
        manifest_path = inprogress / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(status="failed", phase="failed", failed_phase="verifying")
        manifest.pop("publication", None)
        atomic_write_json(manifest_path, manifest)
        (inprogress / "data_quality.json").unlink()
        (inprogress / "sensor_family_report.csv").unlink()
        (inprogress / "excluded_rows.parquet").write_bytes(b"")

        with (
            patch(
                "stage1.ingestion.pipeline._rename_publication_directory",
                side_effect=OSError("recovery rename interrupted"),
            ),
            self.assertRaisesRegex(OSError, "recovery rename interrupted"),
        ):
            recover_derived_ingestion(config)

        checkpoint = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual((checkpoint["status"], checkpoint["phase"]), ("running", "publishing"))
        with patch(
            "stage1.ingestion.pipeline._validate_derived_database",
            side_effect=AssertionError("publication retry must not revalidate derived tables"),
        ):
            recover_derived_ingestion(config)

        completed = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual((completed["status"], completed["phase"]), ("complete", "complete"))
        self.assertEqual(completed["recovery_history"][-1]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
