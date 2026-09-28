"""Transferred R6 files cannot pass with corrupt, missing or altered content."""

from datetime import datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.audit_r6_a_b_handoff import MONTHS, compare_parquets, verified_file, verified_package
from analysis.audit_r6_saved_predictions import run as replay_saved_predictions
from analysis.r6_provenance import frozen_rule_sha256
from analysis.train_r4_discrete_baselines import sha256
from ml.forecast.r6_rule import TERMS


T = datetime(2026, 1, 1)
SCHEMA = pa.schema(
    [
        ("channel_id", pa.string()),
        ("prediction_time", pa.timestamp("us")),
        ("rule_score", pa.float64()),
        ("target_episode_id", pa.string()),
    ]
)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class R6HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.database = duckdb.connect()
        self.addCleanup(self.database.close)
        self.rows = [
            {"channel_id": "c", "prediction_time": T, "rule_score": 7.1, "target_episode_id": None},
            {"channel_id": "d", "prediction_time": T, "rule_score": 1.0, "target_episode_id": "ep"},
        ]
        self.left = self.write_parquet("a.parquet", self.rows)

    def write_parquet(self, name, rows, schema=SCHEMA):
        path = self.directory / name
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
        return path

    def test_exact_keyed_comparison_accepts_reordered_rows_and_nulls(self):
        right = self.write_parquet("b.parquet", list(reversed(self.rows)))
        result = compare_parquets(self.database, self.left, right)
        self.assertEqual(result["rows_checked"], 2)
        self.assertEqual(result["key_mismatches"], 0)
        self.assertEqual(result["value_mismatches"], {"rule_score": 0, "target_episode_id": 0})
        self.assertFalse(result["file_bytes_equal"])

    def test_changed_score_is_rejected_even_for_a_small_difference(self):
        rows = [dict(row) for row in self.rows]
        rows[0]["rule_score"] += 1e-13
        with self.assertRaisesRegex(ValueError, "content differs"):
            compare_parquets(self.database, self.left, self.write_parquet("b.parquet", rows))

    def test_null_to_value_change_is_rejected(self):
        rows = [dict(row) for row in self.rows]
        rows[0]["target_episode_id"] = "ep"
        with self.assertRaisesRegex(ValueError, "content differs"):
            compare_parquets(self.database, self.left, self.write_parquet("b.parquet", rows))

    def test_missing_and_extra_keys_are_rejected(self):
        for rows in (
            self.rows[:1],
            [dict(row, channel_id=row["channel_id"] + "x") for row in self.rows],
        ):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "content differs"):
                compare_parquets(self.database, self.left, self.write_parquet("b.parquet", rows))

    def test_duplicate_and_null_keys_are_rejected(self):
        for rows in ([self.rows[0], self.rows[0]], [dict(self.rows[0], channel_id=None)]):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "duplicate or null"):
                compare_parquets(self.database, self.left, self.write_parquet("b.parquet", rows))

    def test_logical_schema_change_is_rejected(self):
        schema = SCHEMA.set(2, pa.field("rule_score", pa.int64()))
        rows = [dict(row, rule_score=1) for row in self.rows]
        right = self.write_parquet("b.parquet", rows, schema)
        with self.assertRaisesRegex(ValueError, "schemas differ"):
            compare_parquets(self.database, self.left, right)

    def test_hash_mismatch_and_path_escape_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "hash differs"):
            verified_file(self.directory, self.left.name, "0" * 64)
        with self.assertRaisesRegex(ValueError, "escapes package"):
            verified_file(self.directory, "../outside.parquet", "0" * 64)

    def test_package_rejects_duplicate_months_and_modified_alert_file(self):
        report = self.directory / "report.json"
        write_json(report, {})
        manifest = {
            "report_sha256": sha256(report),
            "monthly_predictions": [
                {"month": month, "file": self.left.name, "sha256": sha256(self.left)}
                for month in MONTHS
            ],
            "emitted_alerts_sha256": sha256(self.left),
        }
        alerts = self.directory / "emitted_alerts.parquet"
        alerts.write_bytes(self.left.read_bytes())
        path = self.directory / "manifest.json"
        write_json(path, manifest)
        verified_package(self.directory, sha256(path))
        alerts.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            verified_package(self.directory, sha256(path))
        manifest["monthly_predictions"][1]["month"] = MONTHS[0]
        write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            verified_package(self.directory, sha256(path))

    def test_saved_replay_filters_population_before_full_join(self):
        # Other splits (including null) must not enter test; a missing test key must fail.
        features, index, predictions = (self.directory / name for name in ("a3", "index", "b"))
        for directory in (features, index, predictions):
            directory.mkdir()
        a3_chunks, index_chunks, prediction_chunks = [], [], []
        for month in MONTHS:
            at = datetime.fromisoformat(month + "-01")
            base = {
                "channel_id": "c",
                "prediction_time": at,
                "sensor_type": "smoke",
                "target": 0,
                "target_episode_id": None,
                "label_available_at": at,
            }
            candidate_rows = [
                dict(base, split="test"),
                dict(base, channel_id="d", split="train"),
                dict(base, channel_id="e", split=None),
            ]
            candidate_dir = index / month
            candidate_dir.mkdir()
            pq.write_table(
                pa.Table.from_pylist(candidate_rows),
                candidate_dir / "conditional_discrete_keys.parquet",
            )
            index_chunks.append({"month": month, "manifest_file": month + "/manifest.json"})
            feature_name, prediction_name = f"f-{month}.parquet", f"p-{month}.parquet"
            pq.write_table(
                pa.Table.from_pylist([dict(base, **dict.fromkeys(TERMS, 1))]),
                features / feature_name,
            )
            a3_chunks.append({"month": month, "features_file": feature_name})
            pq.write_table(
                pa.Table.from_pylist(
                    [dict(base, rule_score=sum(TERMS.values()), above_frozen_threshold=False)]
                ),
                predictions / prediction_name,
            )
            prediction_chunks.append(
                {
                    "month": month,
                    "file": prediction_name,
                    "rows": 1,
                    "sha256": sha256(predictions / prediction_name),
                }
            )
        write_json(features / "manifest.json", {"chunks": a3_chunks})
        write_json(index / "manifest.json", {"chunks": index_chunks})
        freeze = self.directory / "freeze.json"
        write_json(
            freeze,
            {
                "frozen_threshold": 7.1,
                "source_a3_manifest_sha256": sha256(features / "manifest.json"),
                "source_r3_admission_manifest_sha256": sha256(index / "manifest.json"),
            },
        )
        b_manifest = {
            "source_freeze_sha256": frozen_rule_sha256(freeze),
            "monthly_predictions": prediction_chunks,
        }
        write_json(predictions / "manifest.json", b_manifest)
        args = dict(freeze_path=freeze, a3_dir=features, index_dir=index, test_dir=predictions)
        self.assertEqual(
            replay_saved_predictions(**args, output_dir=self.directory / "ok")["rows_checked"], 6
        )
        # Same file hash, but index gains a true test key absent from predictions.
        candidate_rows.append(dict(base, channel_id="missing", split="test"))
        pq.write_table(
            pa.Table.from_pylist(candidate_rows),
            candidate_dir / "conditional_discrete_keys.parquet",
        )
        with self.assertRaisesRegex(ValueError, "saved prediction replay differs"):
            replay_saved_predictions(**args, output_dir=self.directory / "bad")


if __name__ == "__main__":
    unittest.main()
