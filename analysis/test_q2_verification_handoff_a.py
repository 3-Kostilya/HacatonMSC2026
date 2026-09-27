"""Small Q2 handoff validates hashes and ignores unrelated spill/raw files."""

from pathlib import Path
import tempfile
import unittest

from analysis.build_sparse_population_a import write_json
from analysis.package_q2_verification_a import verified_members
from analysis.train_r4_discrete_baselines import sha256


class VerificationHandoffTests(unittest.TestCase):
    def test_uses_only_manifest_members_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            write_json(folder / "report.json", {"verified": True})
            write_json(folder / "unrelated.json", {"do_not_copy": True})
            write_json(
                folder / "manifest.json",
                {"files": {"report.json": {"sha256": sha256(folder / "report.json")}}},
            )
            self.assertEqual(set(verified_members(folder)), {"report.json", "manifest.json"})
            write_json(folder / "report.json", {"verified": False})
            with self.assertRaises(ValueError):
                verified_members(folder)

    def test_legacy_B_manifest_supported_and_escaping_paths_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "package"
            folder.mkdir()
            write_json(folder / "report.json", {})
            write_json(folder / "manifest.json", {"report_sha256": sha256(folder / "report.json")})
            self.assertEqual(len(verified_members(folder)), 2)
            write_json(root / "outside.json", {})
            write_json(
                folder / "manifest.json",
                {
                    "files": {
                        "report.json": {"sha256": sha256(folder / "report.json")},
                        "../outside.json": {"sha256": sha256(root / "outside.json")},
                    }
                },
            )
            with self.assertRaises(ValueError):
                verified_members(folder)


if __name__ == "__main__":
    unittest.main()
