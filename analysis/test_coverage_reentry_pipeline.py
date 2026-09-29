"""72-month small fixture tests publication and future-label isolation, not quality."""

from contextlib import redirect_stdout
from datetime import datetime, timedelta
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from analysis.build_coverage_reentry_a import build, summarize_episodes
from analysis.build_quality_improvement_a import QA_NAMES, expected_months
from analysis.build_sparse_population_a import write_json
from analysis.package_sparse_population_a import package_handoff
from analysis.test_sparse_population_a import row
from analysis.train_r4_discrete_baselines import sha256


class CoveragePipelineTests(unittest.TestCase):
    def test_complete_delta_before_labels_unknown_preserved_and_verified_zip(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            q2, m1, a3, b3, qa = [root / n for n in ("q2", "m1", "a3", "b3", "qa")]
            for directory in (q2, m1, a3, b3, qa):
                directory.mkdir()
            write_json(m1 / "manifest.json", {"fixture": True})
            months, raw_files, a_chunks, b_chunks, positives = [], [], [], [], []
            firsts = {0: datetime(2019, 1, 27, 22), 1: datetime(2022, 1, 27, 22)}
            for month in expected_months():
                year, number = map(int, month.split("-"))
                at = datetime(year, number, 28)
                raw = m1 / f"year={year}/month={number:02}/raw.parquet"
                raw.parent.mkdir(parents=True)
                pq.write_table(
                    pa.Table.from_pylist(
                        [
                            {
                                **row(at - timedelta(hours=2)),
                                "object_id": None,
                                "join_status": "unlinked",
                            },
                            {
                                **row(at - timedelta(hours=1), identity=2),
                                "object_id": None,
                                "join_status": "unlinked",
                            },
                        ]
                    ),
                    raw,
                )
                raw_files.append({"file": raw.relative_to(m1).as_posix(), "sha256": sha256(raw)})
                directory = q2 / f"year={year}/month={number:02}"
                directory.mkdir(parents=True)
                cold = month in {"2019-01", "2022-01"}
                first = firsts[0 if year < 2021 else 1]
                past = {
                    "channel_id": "c",
                    "prediction_time": at,
                    "sensor_type": "Датчик дыма",
                    "admission_status": "unknown" if cold else "eligible",
                    "admission_reasons": ["insufficient_history"] if cold else [],
                    "availability_status": "unknown",
                    "last_explicit_normal_at": at - timedelta(hours=1),
                    "admission_evidence_through": at - timedelta(hours=1),
                    "first_usable_at": first,
                    "second_usable_at": first + timedelta(hours=1),
                    "excluded_quality_count_24h": 0,
                    "ambiguous_seconds_24h": 0,
                    "blocking_qa_count_24h": 0,
                }
                admission = directory / "admission.parquet"
                pq.write_table(pa.Table.from_pylist([past]), admission)
                meta = {
                    "month": month,
                    "decision_rows": 1,
                    "files": {admission.name: {"sha256": sha256(admission)}},
                }
                write_json(directory / "manifest.json", meta)
                months.append(
                    {
                        **meta,
                        "manifest_file": (directory / "manifest.json").relative_to(q2).as_posix(),
                        "manifest_sha256": sha256(directory / "manifest.json"),
                    }
                )
                ad = a3 / f"year={year}/month={number:02}"
                ad.mkdir(parents=True)
                features = ad / "features.parquet"
                pq.write_table(
                    pa.table(
                        {
                            "channel_id": ["c"],
                            "prediction_time": [at],
                            "sensor_type": ["Датчик дыма"],
                            "event_count_1h": [0],
                        }
                    ),
                    features,
                )
                a_chunks.append(
                    {
                        "month": month,
                        "features_file": features.relative_to(a3).as_posix(),
                        "features_sha256": sha256(features),
                    }
                )
                bd = b3 / f"year={year}/month={number:02}"
                bd.mkdir(parents=True)
                positive = month in {"2020-10", "2025-12"}
                negative = month == "2022-01"
                label = {
                    "channel_id": "c",
                    "prediction_time": at,
                    "sensor_type": "Датчик дыма",
                    "target": 1 if positive else 0 if negative else None,
                    "label_status": "positive"
                    if positive
                    else "negative"
                    if negative
                    else "unknown",
                    "split": "validation" if year == 2025 else "train",
                    "split_status": "assigned" if positive or negative else "unassigned",
                    "target_episode_id": month if positive else None,
                }
                schema = pa.schema(
                    [
                        (
                            n,
                            pa.timestamp("us")
                            if n == "prediction_time"
                            else pa.int64()
                            if n == "target"
                            else pa.string(),
                        )
                        for n in label
                    ]
                )
                labels = bd / "registered_forecast_labels.parquet"
                pq.write_table(pa.Table.from_pylist([label], schema=schema), labels)
                write_json(
                    bd / "manifest.json", {"files": {labels.name: {"sha256": sha256(labels)}}}
                )
                b_chunks.append(
                    {
                        "month": month,
                        "manifest_file": (bd / "manifest.json").relative_to(b3).as_posix(),
                        "manifest_sha256": sha256(bd / "manifest.json"),
                    }
                )
                if positive:
                    positives.append({**label, **past})
            write_json(a3 / "manifest.json", {"chunks": a_chunks})
            write_json(b3 / "manifest.json", {"chunks": b_chunks})
            correction = qa / "feature_corrections.parquet"
            pq.write_table(
                pa.table(
                    {
                        "channel_id": pa.array([], type=pa.string()),
                        "prediction_time": pa.array([], type=pa.timestamp("us")),
                        "sensor_type": pa.array([], type=pa.string()),
                        "event_count_1h": pa.array([], type=pa.int64()),
                    }
                ),
                correction,
            )
            positive_file = q2 / "positive_hour_diagnostics.parquet"
            pq.write_table(pa.Table.from_pylist(positives), positive_file)
            allowlist = q2 / "model_feature_allowlist.json"
            write_json(
                allowlist,
                {
                    "base_feature_names": ["sensor_type", "event_count_1h"],
                    "feature_names": [
                        "sensor_type",
                        "event_count_1h",
                        *QA_NAMES,
                        "missing__event_count_1h",
                    ],
                },
            )
            pins = {
                "m1": sha256(m1 / "manifest.json"),
                "a3": sha256(a3 / "manifest.json"),
                "b3": sha256(b3 / "manifest.json"),
                "corrections": sha256(correction),
            }
            report_path = q2 / "report.json"
            write_json(
                report_path,
                {
                    "source_m1_files": raw_files,
                    "label_audit": {
                        "by_split": {
                            "train": {"all_episodes": 1, "candidate_available_episodes": 1},
                            "validation": {"all_episodes": 1, "candidate_available_episodes": 1},
                        }
                    },
                },
            )
            write_json(
                q2 / "manifest.json",
                {
                    "months": months,
                    "source_manifests": pins,
                    "files": {
                        p.name: {"sha256": sha256(p)}
                        for p in (positive_file, allowlist, report_path)
                    },
                },
            )
            output = root / "result"

            def assert_causal_artifacts_closed(rows):
                for month in expected_months():
                    folder = (
                        output.with_name("result.inprogress")
                        / f"year={month[:4]}/month={month[5:]}"
                    )
                    self.assertTrue((folder / "new_model_features.parquet").exists())
                    self.assertTrue((folder / "new_admission.parquet").exists())
                    self.assertFalse(
                        {"target", "split", "target_episode_id"}.intersection(
                            pq.read_schema(folder / "new_admission.parquet").names
                        )
                    )
                return summarize_episodes(rows)

            with (
                patch("analysis.build_coverage_reentry_a.Q2_PIN", sha256(q2 / "manifest.json")),
                patch(
                    "analysis.build_coverage_reentry_a.summarize_episodes",
                    side_effect=assert_causal_artifacts_closed,
                ),
                redirect_stdout(io.StringIO()),
            ):
                result = build(
                    q2_dir=q2,
                    m1_dir=m1,
                    a3_dir=a3,
                    b3_dir=b3,
                    corrections_dir=qa,
                    output_dir=output,
                )
            self.assertEqual(result["all_decisions_checked"], 72)
            self.assertEqual(result["new_feature_rows"], 2)
            self.assertEqual(result["episode_coverage"]["train"]["combined_available"], 1)
            self.assertTrue(any(r["target"] == 0 for r in result["new_hour_label_population"]))
            self.assertTrue(
                any(
                    r["target"] is None and r["label_status"] == "unknown"
                    for r in result["new_hour_label_population"]
                )
            )
            self.assertFalse(result["labels_read_for_causal_phase"])
            self.assertFalse(result["production_gate_changed"])
            self.assertEqual(package_handoff(output, root / "handoff.zip")["verified_members"], 221)
            with (
                patch("analysis.build_coverage_reentry_a.Q2_PIN", sha256(q2 / "manifest.json")),
                redirect_stdout(io.StringIO()),
            ):
                recovered = build(
                    q2_dir=q2,
                    m1_dir=m1,
                    a3_dir=a3,
                    b3_dir=b3,
                    corrections_dir=qa,
                    output_dir=root / "recovered",
                    resume_from=output,
                )
            self.assertEqual(recovered["episode_coverage"], result["episode_coverage"])
            self.assertEqual(recovered["resources"]["reused_causal_months"], 72)
            for month in expected_months():
                relative = f"year={month[:4]}/month={month[5:]}/new_model_features.parquet"
                self.assertEqual(sha256(output / relative), sha256(root / "recovered" / relative))
            with self.assertRaises(FileExistsError):
                build(
                    q2_dir=q2,
                    m1_dir=m1,
                    a3_dir=a3,
                    b3_dir=b3,
                    corrections_dir=qa,
                    output_dir=output,
                )


if __name__ == "__main__":
    unittest.main()
