"""Checks for the read-only A2 eligibility diagnostic."""

from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from analysis.audit_a2_eligibility import audit, summarize_rows
from stage1.features import A2_SCHEMA, FEATURE_VERSION, FeatureEvent, build_hourly_rows


class AuditA2EligibilityTests(unittest.TestCase):
    def _state_only_row(self) -> dict:
        decision_at = datetime(2025, 6, 1)
        events = [
            FeatureEvent(
                "state-channel",
                decision_at - timedelta(days=35 - day),
                False,
                value_state="Норма",
                sensor_type="КД Дверь",
            )
            for day in range(27)
        ]
        events.append(
            FeatureEvent(
                "state-channel",
                decision_at - timedelta(hours=1),
                True,
                value_state="Неисправен",
                sensor_type="КД Дверь",
            )
        )
        return build_hourly_rows(
            events, "state-channel", decision_at, decision_at + timedelta(hours=1)
        )[0]

    def test_state_only_history_is_descriptive_candidate_not_approved_eligible(self) -> None:
        row = self._state_only_row()
        self.assertEqual(row["availability_status"], "unknown")
        self.assertIsNone(row["baseline_numeric_median"])
        self.assertIsNone(row["numeric_median_24h"])

        report = summarize_rows([row])
        self.assertEqual(report["original_availability"]["status_counts"], {"unknown": 1})
        self.assertEqual(report["discrete_history"]["data_candidate_rows"], 1)
        self.assertEqual(
            report["discrete_history"]["data_candidates_without_any_baseline_or_24h_numeric"],
            1,
        )
        self.assertEqual(report["numeric_profile"]["data_candidate_rows"], 0)
        self.assertEqual(report["model_usability"]["status"], "not_yet_approved")
        self.assertIsNone(report["model_usability"]["eligible_rows"])
        self.assertEqual(
            report["future_label_eligibility"]["status"], "not_computable_until_b_target"
        )
        self.assertIsNone(report["future_label_eligibility"]["eligible_rows"])

    def test_reason_counts_overlap_instead_of_becoming_new_statuses(self) -> None:
        row = self._state_only_row()
        row["availability_reasons"] = [
            "cadence_unknown",
            "one_or_more_windows_unusable",
            "cadence_unknown",
        ]
        report = summarize_rows([row])
        self.assertEqual(
            report["original_availability"]["reason_counts_overlapping"],
            {"cadence_unknown": 1, "one_or_more_windows_unusable": 1},
        )
        self.assertEqual(report["original_availability"]["status_counts"], {"unknown": 1})

    def test_artifact_provenance_and_bounded_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            a2_dir = Path(directory)
            row = self._state_only_row()
            row.update(
                schema_version=FEATURE_VERSION,
                run_id="fixture",
                config_sha256="a" * 64,
                input_manifest_sha256="b" * 64,
            )
            features = a2_dir / "features.parquet"
            pq.write_table(pa.Table.from_pylist([row], schema=A2_SCHEMA), features)
            manifest = {
                "status": "complete",
                "schema_version": FEATURE_VERSION,
                "features_file": "features.parquet",
                "features_sha256": hashlib.sha256(features.read_bytes()).hexdigest(),
                "feature_rows": 1,
                "run_id": "fixture",
                "config_sha256": "a" * 64,
                "input_manifest_sha256": "b" * 64,
                "config": {"selection_mode": "explicit_channels_file_v1"},
            }
            (a2_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (a2_dir / "validation_report.json").write_text(
                json.dumps({"availability_counts": {"unknown": 1}}), encoding="utf-8"
            )

            report = audit(a2_dir)
            self.assertEqual(report["row_count"], 1)
            self.assertEqual(report["discrete_history"]["data_candidate_rows"], 1)
            with self.assertRaisesRegex(ValueError, "row safety limit"):
                audit(a2_dir, max_rows=0)
            manifest["features_sha256"] = "0" * 64
            (a2_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                audit(a2_dir)


if __name__ == "__main__":
    unittest.main()
