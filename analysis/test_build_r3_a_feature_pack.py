"""R3 A pack tests: immutable lineage, safe allowlist and time chunks."""

from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from analysis.build_r3_a_feature_pack import build
from stage1.features import A2_SCHEMA, FEATURE_VERSION, FeatureEvent, feature_at
from stage1.features.r2 import R2_VERSION, build_state_history_rows
from stage1.features.r3 import MODEL_FEATURE_ALLOWLIST, build_pack_tables


T = datetime(2025, 6, 30, 23)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(root: Path, *, add_future: bool = False) -> tuple[Path, Path]:
    a2_dir = root / "a2"
    r2_dir = root / "r2"
    a2_dir.mkdir()
    r2_dir.mkdir()
    events = [FeatureEvent("c", T - timedelta(hours=2), False, value_state="Норма", sensor_type="Датчик дыма")]
    if add_future:
        events.append(FeatureEvent("c", T + timedelta(hours=3), True, value_state="Неисправен", sensor_type="Датчик дыма"))
    rows = []
    for t in (T, T + timedelta(hours=1)):
        row = feature_at(events, "c", t)
        row.update({
            "schema_version": FEATURE_VERSION,
            "run_id": "a2-test",
            "config_sha256": "a" * 64,
            "input_manifest_sha256": "b" * 64,
        })
        rows.append(row)
    a2_path = a2_dir / "features.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=A2_SCHEMA), a2_path)
    a2_manifest_path = a2_dir / "manifest.json"
    a2_manifest_path.write_text(json.dumps({
        "schema_version": FEATURE_VERSION,
        "status": "complete",
        "features_file": "features.parquet",
        "features_sha256": _sha(a2_path),
        "feature_rows": 2,
        "input_manifest_sha256": "b" * 64,
        "config": {
            "selection_mode": "seeded_type_stratified_real_20_v1",
            "start_at": T.isoformat(),
            "end_at": (T + timedelta(hours=2)).isoformat(),
            "feature_config": {"baseline_embargo": 86400.0},
        },
    }), encoding="utf-8")
    r2_table = build_state_history_rows(
        rows, events, source_a2_manifest_sha256=_sha(a2_manifest_path), completed_episodes=[]
    )
    r2_path = r2_dir / "state_history.parquet"
    pq.write_table(r2_table, r2_path)
    report_path = r2_dir / "report.json"
    report_path.write_text("{}", encoding="utf-8")
    (r2_dir / "manifest.json").write_text(json.dumps({
        "schema_version": R2_VERSION,
        "status": "complete",
        "source_a2_manifest_sha256": _sha(a2_manifest_path),
        "source_m1_manifest_sha256": "b" * 64,
        "ruleset_version": r2_table.column("ruleset_version")[0].as_py(),
        "row_count": 2,
        "files": {
            "state_history.parquet": {"sha256": _sha(r2_path)},
            "report.json": {"sha256": _sha(report_path)},
        },
        "episode_catalog": {
            "catalog_manifest_sha256": "c" * 64,
            "catalog_input_manifest_sha256": "b" * 64,
        },
    }), encoding="utf-8")
    return a2_dir, r2_dir


class R3PackTests(unittest.TestCase):
    def test_allowlist_excludes_keys_diagnostics_provenance_and_target(self) -> None:
        self.assertEqual(len(MODEL_FEATURE_ALLOWLIST), len(set(MODEL_FEATURE_ALLOWLIST)))
        for forbidden in (
            "channel_id", "prediction_time", "availability_status", "run_id",
            "baseline_fit_end_at", "future_label_status", "target", "split", "episode_id",
        ):
            self.assertNotIn(forbidden, MODEL_FEATURE_ALLOWLIST)

    def test_monthly_packs_keep_qa_flag_and_exact_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a2_dir, r2_dir = _source(root)
            output = root / "pack"
            result = build(a2_dir=a2_dir, r2_dir=r2_dir, output=output)
            self.assertEqual(result["purpose"], "qa_only")
            self.assertTrue(result["not_training_ready"])
            self.assertEqual(result["row_count"], 2)
            self.assertEqual(result["chunk_count"], 2)
            self.assertEqual([(x["year"], x["month"]) for x in result["chunks"]], [(2025, 6), (2025, 7)])
            allowlist = json.loads((output / result["allowlist_file"]).read_text(encoding="utf-8"))
            self.assertEqual([x["name"] for x in allowlist["feature_columns"]], list(MODEL_FEATURE_ALLOWLIST))
            for chunk in result["chunks"]:
                feature_file = output / chunk["features_file"]
                status_file = output / chunk["row_status_file"]
                self.assertEqual(_sha(feature_file), chunk["features_sha256"])
                self.assertEqual(_sha(status_file), chunk["row_status_sha256"])
                feature_table = pq.read_table(feature_file)
                status_table = pq.read_table(status_file)
                self.assertEqual(feature_table.num_rows, 1)
                self.assertEqual(status_table.num_rows, 1)
                self.assertEqual(
                    feature_table.select(["channel_id", "prediction_time"]).to_pylist(),
                    status_table.select(["channel_id", "prediction_time"]).to_pylist(),
                )
                self.assertFalse({"target", "split", "episode_id", "future_label_status"} & set(feature_table.schema.names))

    def test_future_event_does_not_change_past_pack_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_a = root / "a"
            source_b = root / "b"
            source_a.mkdir()
            source_b.mkdir()
            a2_a, r2_a = _source(source_a)
            a2_b, r2_b = _source(source_b, add_future=True)
            first = build_pack_tables(
                pq.read_table(a2_a / "features.parquet"), pq.read_table(r2_a / "state_history.parquet")
            )[0]
            second = build_pack_tables(
                pq.read_table(a2_b / "features.parquet"), pq.read_table(r2_b / "state_history.parquet")
            )[0]
            self.assertEqual(first.to_pylist(), second.to_pylist())

    def test_hash_and_key_mismatch_rejected_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a2_dir, r2_dir = _source(root)
            manifest_path = r2_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source_a2_manifest_sha256"] = "d" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact A2"):
                build(a2_dir=a2_dir, r2_dir=r2_dir, output=root / "pack")
            self.assertFalse((root / "pack").exists())


if __name__ == "__main__":
    unittest.main()
