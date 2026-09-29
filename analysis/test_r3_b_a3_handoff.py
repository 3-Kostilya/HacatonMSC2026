"""B3 labels must join exactly to the reviewed A3 feature-only artifact."""

from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from analysis.build_r3_a_feature_pack import build as build_a3
from analysis.build_r3_registered_labels import _audit_a3_pack, _sha256
from analysis.test_build_r3_a_feature_pack import T, _source


class A3B3HandoffTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[dict, set, list, dict]:
        a2_dir, r2_dir = _source(root)
        pack_dir = root / "pack"
        build_a3(a2_dir=a2_dir, r2_dir=r2_dir, output=pack_dir)
        m1_manifest = root / "m1.json"
        b2_manifest = root / "b2.json"
        m1_manifest.write_text("{}", encoding="utf-8")
        b2_manifest.write_text("{}", encoding="utf-8")
        manifest_path = pack_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source_m1_manifest_sha256"] = _sha256(m1_manifest)
        manifest["source_b2_catalog_manifest_sha256"] = _sha256(b2_manifest)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        keys = {("c", T), ("c", T + timedelta(hours=1))}
        labels = [
            SimpleNamespace(channel_id=channel, prediction_time=at, label_status="unknown")
            for channel, at in sorted(keys)
        ]
        paths = {
            "a3_dir": pack_dir,
            "a2_manifest": a2_dir / "manifest.json",
            "r2_manifest": r2_dir / "manifest.json",
            "m1_manifest": m1_manifest,
            "b2_manifest": b2_manifest,
        }
        return paths, keys, labels, manifest

    def test_exact_join_and_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths, keys, labels, _ = self._fixture(Path(temp))
            result = _audit_a3_pack(**paths, expected_keys=keys, labels=labels)
            self.assertEqual(result["matched_rows"], 2)
            self.assertEqual(result["feature_count"], 111)
            self.assertEqual(result["purpose"], "qa_only")
            self.assertTrue(result["not_training_ready"])

    def test_rejects_missing_key_and_wrong_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths, keys, labels, manifest = self._fixture(Path(temp))
            with self.assertRaisesRegex(ValueError, "prediction keys differ"):
                _audit_a3_pack(**paths, expected_keys={next(iter(keys))}, labels=labels)
            manifest["source_b2_catalog_manifest_sha256"] = "f" * 64
            (paths["a3_dir"] / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "source lineage"):
                _audit_a3_pack(**paths, expected_keys=keys, labels=labels)


if __name__ == "__main__":
    unittest.main()
