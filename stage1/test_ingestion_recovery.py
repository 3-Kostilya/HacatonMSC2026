"""Safe finalization contracts for an interrupted post-classification run."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow.parquet as pq

from stage1.ingestion.dictionaries import CHANNEL_REQUIRED_COLUMNS, OBJECT_REQUIRED_COLUMNS
from stage1.ingestion.pipeline import recover_derived_ingestion, run_ingestion
from stage1.normalization import EVENT_FIELDS


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class IngestionDerivedRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "events.csv"
        self.channels = self.root / "channels.csv"
        self.objects = self.root / "objects.csv"
        duplicate = ["event-1", "channel-1", "2026-01-01", "00:01:00", "false", "1"]
        write_csv(
            self.source,
            EVENT_FIELDS,
            [
                duplicate,
                duplicate,
                ["event-bad", "channel-1", "not-a-date", "00:02:00", "false", "2"],
                ["event-3", "channel-1", "2026-02-01", "00:03:00", "false", "3"],
            ],
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
            "output": str(self.root / "recovered"),
            "memory_limit": "128MB",
            "batch_size": 2,
            "keep_database": True,
        }

    def make_interrupted_derived_run(self):
        run_ingestion(self.config)
        destination = Path(self.config["output"])
        inprogress = destination.with_name(destination.name + ".inprogress")
        destination.rename(inprogress)
        manifest_path = inprogress / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "running"
        manifest.pop("sanity_checks", None)
        manifest.pop("elapsed_seconds", None)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (inprogress / "data_quality.json").unlink()
        (inprogress / "sensor_family_report.csv").unlink()
        (inprogress / "sensor_statistics.parquet").unlink()
        (inprogress / "excluded_rows.parquet").write_bytes(b"")
        return inprogress

    def test_recovery_reuses_exact_clean_and_repairs_only_safe_exports(self):
        inprogress = self.make_interrupted_derived_run()
        clean = next((inprogress / "clean").rglob("*.parquet"))
        clean_before = (file_hash(clean), clean.stat().st_mtime_ns)

        with (
            patch(
                "stage1.ingestion.pipeline.iter_source_rows",
                side_effect=AssertionError("recovery must not re-ingest"),
            ),
            patch(
                "stage1.ingestion.pipeline.classify",
                side_effect=AssertionError("recovery must not reclassify"),
            ),
        ):
            report = recover_derived_ingestion(self.config)

        output = Path(self.config["output"])
        self.assertTrue(output.is_dir())
        self.assertFalse(inprogress.exists())
        clean_after = next((output / "clean").rglob("*.parquet"))
        self.assertEqual((file_hash(clean_after), clean_after.stat().st_mtime_ns), clean_before)
        self.assertEqual(pq.ParquetFile(output / "excluded_rows.parquet").metadata.num_rows, 2)
        self.assertGreater(
            pq.ParquetFile(output / "sensor_statistics.parquet").metadata.num_rows, 0
        )
        self.assertEqual(report["recovery"]["auxiliary_exports"]["excluded_rows"], "regenerate")
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "complete")
        recovery = manifest["recovery_history"][-1]
        self.assertEqual(recovery["status"], "complete")
        self.assertEqual(recovery["reused_clean_rows"], 2)
        self.assertEqual(recovery["classification_strategy"], "global_window_v1")
        self.assertEqual(
            recovery["classification_strategy_evidence"]["source"], "manifest_checkpoint"
        )

    def test_failed_classification_keeps_intended_strategy_checkpoint(self):
        cases = (
            (False, "stage1.ingestion.pipeline.classify", "global_window_v1"),
            (True, "stage1.ingestion.pipeline.classify_partitioned", "partitioned_hash_v1"),
        )
        for index, (partitioned, target, expected) in enumerate(cases):
            with self.subTest(strategy=expected):
                config = dict(self.config)
                config["output"] = str(self.root / f"failed-{index}")
                with (
                    patch(target, side_effect=RuntimeError("classification failed")),
                    self.assertRaisesRegex(RuntimeError, "classification failed"),
                ):
                    run_ingestion(config, partitioned_classification=partitioned)

                manifest_path = Path(config["output"] + ".inprogress") / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(manifest["status"], "failed")
                self.assertEqual(manifest["classification_strategy"], expected)

    def test_legacy_recovery_requires_explicit_strategy_without_mutation(self):
        inprogress = self.make_interrupted_derived_run()
        manifest_path = inprogress / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.pop("classification_strategy")
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest_before = manifest_path.read_bytes()

        with self.assertRaisesRegex(ValueError, "Legacy manifest lacks classification_strategy"):
            recover_derived_ingestion(self.config)

        self.assertEqual(manifest_path.read_bytes(), manifest_before)
        self.assertFalse(Path(self.config["output"]).exists())

    def test_legacy_recovery_records_trusted_strategy_override(self):
        inprogress = self.make_interrupted_derived_run()
        manifest_path = inprogress / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.pop("classification_strategy")
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        recover_derived_ingestion(self.config, trusted_classification_strategy="global_window_v1")

        output = Path(self.config["output"])
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["classification_strategy"], "global_window_v1")
        evidence = manifest["recovery_history"][-1]["classification_strategy_evidence"]
        self.assertEqual(evidence["source"], "explicit_trusted_legacy_override")
        self.assertFalse(evidence["manifest_checkpoint_present"])
        self.assertEqual(evidence["override"], "global_window_v1")

    def test_interrupted_legacy_recovery_retries_only_empty_export(self):
        inprogress = self.make_interrupted_derived_run()
        manifest_path = inprogress / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["classification_strategy"] = "recovered_existing_derived_tables"
        manifest["recovery_history"] = [
            {"mode": "derived_tables_v1", "status": "running", "started_utc": "earlier"}
        ]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        abandoned = inprogress / ".excluded_rows.parquet.recovering"
        abandoned.write_bytes(b"")

        with self.assertRaisesRegex(ValueError, "Legacy recovery placeholder"):
            recover_derived_ingestion(self.config)
        self.assertTrue(abandoned.exists())
        self.assertEqual(
            json.loads(manifest_path.read_text(encoding="utf-8"))["recovery_history"][0]["status"],
            "running",
        )

        with (
            patch(
                "stage1.ingestion.pipeline.iter_source_rows",
                side_effect=AssertionError("retry must not re-ingest"),
            ),
            patch(
                "stage1.ingestion.pipeline.classify",
                side_effect=AssertionError("retry must not reclassify"),
            ),
        ):
            recover_derived_ingestion(
                self.config, trusted_classification_strategy="global_window_v1"
            )

        output = Path(self.config["output"])
        self.assertFalse((output / abandoned.name).exists())
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        earlier, retried = manifest["recovery_history"]
        self.assertEqual(earlier["status"], "interrupted")
        self.assertIn("interrupted_utc", earlier)
        self.assertEqual(retried["status"], "complete")
        self.assertTrue(retried["cleared_abandoned_zero_byte_excluded_export"])
        self.assertEqual(retried["classification_strategy"], "global_window_v1")
        self.assertEqual(
            retried["classification_strategy_evidence"]["source"],
            "explicit_trusted_legacy_placeholder_override",
        )

    def test_interrupted_recovery_preserves_nonempty_or_unattributed_temp(self):
        for index, history in enumerate(([], [{"mode": "derived_tables_v1", "status": "running"}])):
            with self.subTest(index=index):
                config = dict(self.config)
                config["output"] = str(self.root / f"ambiguous-{index}")
                run_ingestion(config)
                destination = Path(config["output"])
                inprogress = destination.with_name(destination.name + ".inprogress")
                destination.rename(inprogress)
                manifest_path = inprogress / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["status"] = "running"
                manifest["recovery_history"] = history
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                (inprogress / "data_quality.json").unlink()
                (inprogress / "sensor_family_report.csv").unlink()
                abandoned = inprogress / ".excluded_rows.parquet.recovering"
                abandoned.write_bytes(b"" if index == 0 else b"partial")
                before = manifest_path.read_bytes()

                with self.assertRaisesRegex(ValueError, "Ambiguous prior recovery temporary file"):
                    recover_derived_ingestion(config)

                self.assertEqual(manifest_path.read_bytes(), before)
                self.assertEqual(abandoned.read_bytes(), b"" if index == 0 else b"partial")
                self.assertFalse(destination.exists())

    def test_empty_abandoned_temp_does_not_replace_nonempty_export(self):
        inprogress = self.make_interrupted_derived_run()
        manifest_path = inprogress / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["recovery_history"] = [{"mode": "derived_tables_v1", "status": "running"}]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        target = inprogress / "excluded_rows.parquet"
        target.write_bytes(b"nonempty")
        abandoned = inprogress / ".excluded_rows.parquet.recovering"
        abandoned.write_bytes(b"")
        before = manifest_path.read_bytes()

        with self.assertRaisesRegex(ValueError, "Ambiguous prior recovery temporary file"):
            recover_derived_ingestion(self.config)

        self.assertEqual(manifest_path.read_bytes(), before)
        self.assertEqual(target.read_bytes(), b"nonempty")
        self.assertTrue(abandoned.exists())

    def test_recovery_rejects_override_that_disagrees_with_manifest(self):
        inprogress = self.make_interrupted_derived_run()
        manifest_path = inprogress / "manifest.json"
        manifest_before = manifest_path.read_bytes()

        with self.assertRaisesRegex(ValueError, "disagrees with manifest checkpoint"):
            recover_derived_ingestion(
                self.config, trusted_classification_strategy="partitioned_hash_v1"
            )

        self.assertEqual(manifest_path.read_bytes(), manifest_before)
        self.assertFalse(Path(self.config["output"]).exists())

    def test_recovery_refuses_nonempty_ambiguous_auxiliary_without_mutation(self):
        inprogress = self.make_interrupted_derived_run()
        target = inprogress / "excluded_rows.parquet"
        target.write_bytes(b"not parquet")
        manifest_path = inprogress / "manifest.json"
        manifest_before = manifest_path.read_bytes()

        with self.assertRaisesRegex(ValueError, "Non-empty auxiliary export is unreadable"):
            recover_derived_ingestion(self.config)

        self.assertEqual(manifest_path.read_bytes(), manifest_before)
        self.assertEqual(target.read_bytes(), b"not parquet")
        self.assertFalse(Path(self.config["output"]).exists())

    def test_recovery_refuses_incomplete_clean_inventory_without_mutation(self):
        inprogress = self.make_interrupted_derived_run()
        next((inprogress / "clean").rglob("*.parquet")).unlink()
        manifest_path = inprogress / "manifest.json"
        manifest_before = manifest_path.read_bytes()

        with self.assertRaisesRegex(ValueError, "Clean export file inventory"):
            recover_derived_ingestion(self.config)

        self.assertEqual(manifest_path.read_bytes(), manifest_before)
        self.assertFalse(Path(self.config["output"]).exists())


if __name__ == "__main__":
    unittest.main()
