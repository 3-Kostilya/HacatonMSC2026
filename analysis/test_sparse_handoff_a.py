"""Small handoff/export checks independent of the full local data run."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.build_sparse_population_a import verify_month
from analysis.package_sparse_population_a import package_handoff
from analysis.train_r4_discrete_baselines import sha256
from analysis.verify_sparse_population_a import active_package, month_snapshots
from analysis.build_quality_improvement_a import expected_months


class SparseHandoffTests(unittest.TestCase):
    def fixture(self, root):
        package = root / "package"
        month = package / "year=2025" / "month=01"
        month.mkdir(parents=True)
        for path in (package / "report.json", month / "manifest.json"):
            path.write_text("{}\n", encoding="utf-8")
        data = month / "admission.parquet"
        pq.write_table(pa.table({"channel_id": ["c"]}), data)
        manifest = {
            "files": {"report.json": {"sha256": sha256(package / "report.json")}},
            "months": [{
                "manifest_file": "year=2025/month=01/manifest.json",
                "manifest_sha256": sha256(month / "manifest.json"),
                "files": {data.name: {"sha256": sha256(data)}},
            }],
        }
        (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return package, manifest

    def test_archive_hashes_and_spill_exclusion(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            package, _ = self.fixture(root)
            spill = package / "db-spill"
            spill.mkdir()
            (spill / "not-a-deliverable.tmp").write_text("scratch", encoding="utf-8")
            destination = root / "handoff.zip"
            result = package_handoff(package, destination)
            self.assertEqual(result["verified_members"], 4)
            with zipfile.ZipFile(destination) as archive:
                self.assertFalse(any("db-spill" in name for name in archive.namelist()))
                self.assertIsNone(archive.testzip())
            with self.assertRaises(FileExistsError):
                package_handoff(package, destination)

    def test_changed_content_and_escape_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            package, manifest = self.fixture(root)
            (package / "report.json").write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                package_handoff(package, root / "changed.zip")
            outside = root / "outside.json"
            outside.write_text("private", encoding="utf-8")
            manifest["files"] = {"../outside.json": {"sha256": sha256(outside)}}
            (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                package_handoff(package, root / "escape.zip")

    def test_verifier_rejects_extra_years_before_reading_months(self):
        with TemporaryDirectory() as temporary:
            package = Path(temporary)
            (package / "manifest.json").write_text(
                json.dumps({"months": [{"month": "2026-01"}]}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "before any"):
                list(month_snapshots(package, False))

    def test_completed_incremental_snapshots_and_pending_root(self):
        with TemporaryDirectory() as temporary:
            package = Path(temporary) / "package"
            pending = Path(temporary) / "package.inprogress"
            pending.mkdir()
            self.assertEqual(active_package(package), pending)
            pending.rename(package)
            self.assertEqual(active_package(package), package)
            items = []
            for month in expected_months():
                relative = f"year={month[:4]}/month={month[5:]}/manifest.json"
                path = package / relative
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"month": month}), encoding="utf-8")
                items.append({"month": month, "manifest_file": relative,
                              "manifest_sha256": sha256(path)})
            (package / "manifest.json").write_text(json.dumps({"months": items}), encoding="utf-8")
            snapshots = list(month_snapshots(package, True))
            self.assertEqual(len(snapshots), 72)
            self.assertEqual([(r["month"], r["manifest_sha256"]) for _, r in snapshots],
                             [(r["month"], r["manifest_sha256"]) for r in items])

    def test_export_rejects_missing_or_ineligible_features(self):
        with TemporaryDirectory() as temporary, duckdb.connect() as db:
            root = Path(temporary)
            admission = root / "admission.parquet"
            feature_file = root / "features.parquet"
            db.execute("CREATE TEMP TABLE decisions AS SELECT 'c' AS channel_id, "
                       "TIMESTAMP '2025-01-01' prediction_time,'Датчик дыма' sensor_type, "
                       "'eligible' admission_status, []::VARCHAR[] admission_reasons, "
                       "0 blocking_qa_count_24h, 'unknown' availability_status, "
                       "TIMESTAMP '2025-01-01' admission_evidence_through, "
                       "TIMESTAMP '2025-01-01' last_explicit_normal_at, "
                       "TIMESTAMP '2024-12-01' first_usable_at, "
                       "TIMESTAMP '2024-12-02' second_usable_at")
            db.execute("COPY decisions TO ? (FORMAT PARQUET)", [str(admission)])
            db.execute("COPY (SELECT channel_id,prediction_time,sensor_type FROM decisions) "
                       "TO ? (FORMAT PARQUET)", [str(feature_file)])
            self.assertEqual(verify_month(db, 1, admission, feature_file, ["sensor_type"])["feature_rows"], 1)
            db.execute("COPY (SELECT channel_id,prediction_time,sensor_type FROM decisions "
                       "WHERE false) TO ? (FORMAT PARQUET)", [str(feature_file)])
            with self.assertRaises(ValueError):
                verify_month(db, 1, admission, feature_file, ["sensor_type"])
            db.execute("COPY (SELECT 'other' channel_id,prediction_time,sensor_type "
                       "FROM decisions) TO ? (FORMAT PARQUET)", [str(feature_file)])
            with self.assertRaises(ValueError):
                verify_month(db, 1, admission, feature_file, ["sensor_type"])


if __name__ == "__main__":
    unittest.main()
