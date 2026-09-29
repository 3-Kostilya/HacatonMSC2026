"""Candidate replay is complete, past-only and never an automatic training index."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import json

import pyarrow as pa
import pyarrow.parquet as pq

from analysis.replay_shadow_pilot import observation_groups
from analysis.replay_sparse_admission_a import SCHEMA, compare_saved_legacy, create_handoff, replay
from analysis.replay_sparse_admission_a import validate_bounds, verify_decisions
from analysis.test_shadow_pilot import row
from analysis.train_r4_discrete_baselines import sha256
from stage1.features.sparse_admission import SparseAdmissionStream


T = datetime(2025, 12, 1)


def points(tail=()):
    source = [row(T - timedelta(days=10)), row(T - timedelta(hours=30)), *tail]
    return [
        hour
        for hour in replay(
            SparseAdmissionStream(),
            observation_groups(source),
            ["c", "empty"],
            T,
            T + timedelta(hours=2),
        )
    ]


class SparseReplayTests(unittest.TestCase):
    def test_future_tail_and_future_type_do_not_change_prior_decisions(self):
        initial = points()[0]
        self.assertEqual(initial, points([row(T + timedelta(hours=1), text="Неисправен")])[0])
        self.assertEqual(initial, points([row(T + timedelta(hours=1), kind=None)])[0])
        self.assertNotEqual(
            points()[1], points([row(T + timedelta(hours=1), text="Неисправен")])[1]
        )

    def test_whole_grid_includes_unknowns_and_parquet_reread_validates(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "decisions.parquet"
            rows = [point for hour in points() for point in hour]
            pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
            result = verify_decisions(path, ["c", "empty"], T, T + timedelta(hours=2))
            self.assertEqual(result["rows_checked"], 4)
            self.assertEqual(result["candidate_status_counts"], {"eligible": 2, "unknown": 2})
            self.assertEqual(result["legacy_status_counts"], {"unknown": 4})
            self.assertFalse(
                {"target", "split", "rule_score", "warning_emitted"}.intersection(SCHEMA.names)
            )

    def test_missing_duplicate_and_unprotected_veto_fail_validation(self):
        rows = [point for hour in points() for point in hour]
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.parquet"
            for bad in (rows[:-1], [*rows, rows[0]]):
                pq.write_table(pa.Table.from_pylist(bad, schema=SCHEMA), path)
                with self.assertRaisesRegex(ValueError, "grid incomplete or duplicated"):
                    verify_decisions(path, ["c", "empty"], T, T + timedelta(hours=2))
            bad = [{**row} for row in rows]
            bad[0]["legacy_admission_reasons"] = [
                *bad[0]["legacy_admission_reasons"],
                "quality_exclusions_24h",
            ]
            pq.write_table(pa.Table.from_pylist(bad, schema=SCHEMA), path)
            with self.assertRaisesRegex(ValueError, "non-statistical veto"):
                verify_decisions(path, ["c", "empty"], T, T + timedelta(hours=2))

    def test_bounds_reject_test_excluded_year_cross_gap_and_bad_hours(self):
        for start, end in (
            (datetime(2026, 1, 1), datetime(2026, 1, 2)),
            (datetime(2021, 1, 1), datetime(2021, 1, 2)),
            (datetime(2020, 12, 31), datetime(2021, 1, 2)),
            (T, T + timedelta(days=32)),
            (T, T),
            (T, T + timedelta(minutes=1)),
            (T.replace(tzinfo=timezone.utc), T.replace(tzinfo=timezone.utc) + timedelta(hours=1)),
        ):
            with self.assertRaisesRegex(ValueError, "train/validation whole hours"):
                validate_bounds(start, end)

    def test_qa_is_an_additional_veto_not_a_new_positive_label(self):
        source = [
            row(T - timedelta(days=10), kind="Датчик температуры"),
            row(T - timedelta(hours=30), kind="Датчик температуры"),
            {**row(T, text=None, kind="Датчик температуры"), "value_numeric": -127.0},
        ]
        hours = list(
            replay(
                SparseAdmissionStream(),
                observation_groups(source),
                ["c"],
                T,
                T + timedelta(hours=1),
            )
        )
        actual = hours[0][0]
        self.assertEqual(actual["blocking_qa_count_24h"], 1)
        self.assertEqual(actual["admission_status"], "unknown")
        self.assertEqual(actual["registered_fault_text_count_24h"], 0)

    def test_saved_legacy_parity_checks_keys_counters_scope_and_hashes(self):
        rows = [point for hour in points() for point in hour]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate.parquet"
            old = root / "old"
            old.mkdir()
            pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), candidate)
            legacy_rows = [
                {
                    **point,
                    "admission_status": point["legacy_admission_status"],
                    "admission_reasons": point["legacy_admission_reasons"],
                }
                for point in rows
            ]
            predictions = old / "predictions.parquet"
            pq.write_table(pa.Table.from_pylist(legacy_rows, schema=SCHEMA), predictions)
            report_file = old / "report.json"
            report_file.write_text(
                json.dumps(
                    {
                        "source_m1_manifest_sha256": "m1",
                        "start": T.isoformat(),
                        "end_exclusive": (T + timedelta(hours=2)).isoformat(),
                        "selected_channels": [{"channel_id": "c"}, {"channel_id": "empty"}],
                    }
                ),
                encoding="utf-8",
            )
            manifest_file = old / "manifest.json"
            manifest = {
                "report_sha256": sha256(report_file),
                "prediction_sha256": sha256(predictions),
                "prediction_file": predictions.name,
                "prediction_rows": 4,
            }
            manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
            args = {
                "m1_sha256": "m1",
                "channels": ["c", "empty"],
                "start": T,
                "end": T + timedelta(hours=2),
            }
            self.assertEqual(
                compare_saved_legacy(candidate, old, **args)["legacy_rows_compared"], 4
            )
            legacy_rows[0]["registered_fault_text_count_24h"] = 1
            pq.write_table(pa.Table.from_pylist(legacy_rows, schema=SCHEMA), predictions)
            with self.assertRaisesRegex(ValueError, "hash differs"):
                compare_saved_legacy(candidate, old, **args)
            manifest["prediction_sha256"] = sha256(predictions)
            manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "past evidence mismatch"):
                compare_saved_legacy(candidate, old, **args)

    def test_zip_checks_hashes_and_never_overwrites_existing_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "candidate"
            output.mkdir()
            decisions = output / "candidate_admission.parquet"
            pq.write_table(pa.Table.from_pylist(points()[0], schema=SCHEMA), decisions)
            report = output / "report.json"
            report.write_text("{}", encoding="utf-8")
            manifest = output / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "decision_file": decisions.name,
                        "decision_sha256": sha256(decisions),
                        "report_sha256": sha256(report),
                    }
                ),
                encoding="utf-8",
            )
            destination = root / "handoff.zip"
            result = create_handoff(output, destination)
            self.assertEqual(result["member_hashes_verified"], 3)
            with self.assertRaises(FileExistsError):
                create_handoff(output, destination)
            report.write_text('{"tampered":true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                create_handoff(output, root / "bad.zip")


if __name__ == "__main__":
    unittest.main()
