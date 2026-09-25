"""Cross-platform R4 feature allowlist provenance checks."""

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.train_r4_discrete_baselines import sha256_pinned_text


class AllowlistHashTests(unittest.TestCase):
    def test_lf_and_crlf_copies_match_pinned_hash(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "allowlist.json"
            crlf = b'{"feature_names": ["sensor_type"]}\r\n'
            expected = hashlib.sha256(crlf).hexdigest()
            for copy in (crlf, crlf.replace(b"\r\n", b"\n")):
                path.write_bytes(copy)
                self.assertEqual(sha256_pinned_text(path), expected)

    def test_changed_content_is_rejected(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "allowlist.json"
            expected = hashlib.sha256(b'{"feature_names": ["sensor_type"]}\r\n').hexdigest()
            path.write_bytes(b'{"feature_names": ["different_type"]}\n')
            self.assertNotEqual(sha256_pinned_text(path), expected)

    def test_checked_in_r3_files_match_both_line_endings(self) -> None:
        root = Path(__file__).resolve().parents[1]
        files = (
            ("ml/r3_discrete_feature_allowlist_v1.json",
             "201913367eab87800fa9921e92b6bf6a9aa52cf9dd595b4ede3c3c009abfac0a"),
            ("ml/r3_conditional_training_contract_v1.json",
             "c3dd160446e22fb6072b6ef10d331edc31d409ad5b2e3705bd7c5aa95e9eff36"),
        )
        with TemporaryDirectory() as temp:
            copy = Path(temp) / "copy.json"
            for name, expected in files:
                source = (root / name).read_bytes()
                for value in (source.replace(b"\r\n", b"\n"),
                              source.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")):
                    copy.write_bytes(value)
                    self.assertEqual(sha256_pinned_text(copy), expected)
    def test_non_line_ending_carriage_return_is_rejected(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "allowlist.json"
            path.write_bytes(b'{"feature_names": ["sensor_type"]}\r')
            with self.assertRaises(ValueError):
                sha256_pinned_text(path)


if __name__ == "__main__":
    unittest.main()