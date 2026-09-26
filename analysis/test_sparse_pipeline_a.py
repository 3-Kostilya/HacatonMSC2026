"""Small 72-month end-to-end fixture, not a substitute for the real M1 audit."""

from datetime import datetime, timedelta
from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from analysis.build_quality_improvement_a import expected_months
from analysis.build_sparse_population_a import audit_labels, build, write_json
from analysis.package_sparse_population_a import package_handoff
from analysis.test_sparse_population_a import row
from analysis.train_r4_discrete_baselines import read_json, sha256


class SparsePipelineTests(unittest.TestCase):
    def test_cold_build_all_months_and_package_after_freeing_temporaries(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            m1, a3, b3, qa = [root / name for name in ("m1", "a3", "b3", "qa")]
            for path in (m1, a3, b3, qa):
                path.mkdir()
            write_json(m1 / "manifest.json", {"synthetic_fixture": True})
            a_chunks, b_chunks = [], []
            for month in expected_months():
                at = datetime.strptime(month, "%Y-%m")
                raw = m1 / "clean" / f"year={at.year}" / f"month={at.month}"
                raw.mkdir(parents=True)
                pq.write_table(pa.Table.from_pylist([row(at, numeric=0.0)]), raw / "data_0.parquet")
                a = a3 / f"year={at.year}" / f"month={at.month:02}"
                b = b3 / f"year={at.year}" / f"month={at.month:02}"
                a.mkdir(parents=True)
                b.mkdir(parents=True)
                features = a / "features.parquet"
                pq.write_table(pa.table({"channel_id": ["c"], "prediction_time": [at],
                                         "sensor_type": ["Датчик дыма"], "state_count_1h": [1]}), features)
                write_json(a / "manifest.json", {"month": month, "rows": 1})
                a_chunks.append({"month": month, "rows": 1,
                                 "manifest_file": (a / "manifest.json").relative_to(a3).as_posix(),
                                 "manifest_sha256": sha256(a / "manifest.json"),
                                 "features_file": features.relative_to(a3).as_posix(),
                                 "features_sha256": sha256(features)})
                split = "validation" if at.year == 2025 else "train"
                positive = month in {"2020-01", "2025-01"}
                # Deliberately synthetic immutable labels, read only after inference.
                label = {"channel_id": "c", "sensor_type": "Датчик дыма", "prediction_time": at,
                         "horizon_end": at + timedelta(hours=24), "target": 1 if positive else None,
                         "label_status": "positive" if positive else "unknown", "reason": "fixture",
                         "target_episode_id": f"{split}-episode" if positive else None,
                         "label_available_at": at + timedelta(hours=1), "split": split,
                         "split_status": "assigned" if positive else "unassigned"}
                schema = pa.schema([(name, pa.timestamp("us") if name in {
                    "prediction_time", "horizon_end", "label_available_at"
                } else pa.int64() if name == "target" else pa.string()) for name in label])
                labels = b / "registered_forecast_labels.parquet"
                pq.write_table(pa.Table.from_pylist([label], schema=schema), labels)
                write_json(b / "manifest.json", {
                    "source_a3_month_manifest_sha256": a_chunks[-1]["manifest_sha256"],
                    "files": {labels.name: {"sha256": sha256(labels)}}})
                b_chunks.append({"month": month,
                                 "manifest_file": (b / "manifest.json").relative_to(b3).as_posix(),
                                 "manifest_sha256": sha256(b / "manifest.json")})
            write_json(a3 / "manifest.json", {"chunks": a_chunks})
            write_json(b3 / "report.json", {"assigned_unique_positive_episodes_by_split": {
                "train": 1, "validation": 1}})
            write_json(b3 / "manifest.json", {"chunks": b_chunks,
                                              "report_sha256": sha256(b3 / "report.json")})
            corrections = qa / "feature_corrections.parquet"
            pq.write_table(pa.Table.from_pylist([], schema=pa.schema([
                ("channel_id", pa.string()), ("prediction_time", pa.timestamp("us")),
                ("sensor_type", pa.string()), ("state_count_1h", pa.int64())])), corrections)
            contract = {"source_m1_manifest_sha256": sha256(m1 / "manifest.json"),
                        "source_a3_manifest_sha256": sha256(a3 / "manifest.json"),
                        "source_qa_correction_features_sha256": sha256(corrections)}
            base = {"feature_names": ["sensor_type", "state_count_1h"],
                    "source_b3_manifest_sha256": sha256(b3 / "manifest.json")}

            def fixture_json(path):
                if path == Path("ml/quality_improvement_feature_contract_v1.json"):
                    return contract
                if path == Path("ml/r3_discrete_feature_allowlist_v1.json"):
                    return base
                return read_json(path)

            output = root / "result"
            def labels_after_inference(package, source_b3, source_a3):
                for month in expected_months():
                    directory = package / f"year={month[:4]}" / f"month={month[5:]}"
                    self.assertTrue((directory / "manifest.json").exists())
                    self.assertTrue((directory / "model_features.parquet").exists())
                    self.assertTrue((directory / "admission.parquet").exists())
                return audit_labels(package, source_b3, source_a3)

            with patch("analysis.build_sparse_population_a.read_json", side_effect=fixture_json), \
                 patch("analysis.build_sparse_population_a.audit_labels", side_effect=labels_after_inference), \
                 redirect_stdout(io.StringIO()):
                result = build(m1_dir=m1, a3_dir=a3, b3_dir=b3, corrections_dir=qa,
                               output_dir=output, memory_limit="256MB", threads=2)
            self.assertEqual(len(result["months"]), 72)
            self.assertEqual(sum(m["decision_rows"] for m in result["months"]), 72)
            self.assertEqual(sum(m["feature_rows"] for m in result["months"]), 70)
            self.assertEqual(result["label_audit"]["episodes"], 2)
            self.assertEqual(result["label_audit"]["by_split"]["validation"]["candidate_available_episodes"], 1)
            self.assertFalse(result["labels_read_for_inference"])
            self.assertFalse(result["test_events_read"])
            self.assertFalse(result["training_ready"])
            self.assertFalse(output.with_name("result.inprogress").exists())
            packaged = package_handoff(output, root / "handoff.zip")
            self.assertEqual(packaged["verified_members"], 221)


if __name__ == "__main__":
    unittest.main()
