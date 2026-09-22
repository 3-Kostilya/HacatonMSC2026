from __future__ import annotations

import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from stage1.ingestion.sources import iter_source_rows
from stage1.normalization import EVENT_FIELDS


def csv_text(*rows: list[str]) -> str:
    fields = ",".join(EVENT_FIELDS)
    return fields + "\n" + "\n".join(",".join(row) for row in rows) + "\n"


class SourceReaderTests(unittest.TestCase):
    def test_csv_preserves_quoted_multiline_value_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            path.write_bytes(
                (",".join(EVENT_FIELDS) + '\n1,c,d,t,true,"line 1\nline 2"\n').encode("utf-8")
            )
            rows = list(iter_source_rows(path))
        self.assertEqual(rows[0]["значение_датчика"], "line 1\nline 2")
        self.assertEqual(rows[0]["__source_row__"], 2)
        self.assertEqual(rows[0]["__source__"], str(path.resolve()))

    def test_bounded_and_repeated_header_are_raw_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            path.write_text(
                csv_text(list(EVENT_FIELDS), ["id", "c", "d", "t", "x", "v"]), encoding="utf-8"
            )
            rows = list(iter_source_rows(path, max_rows=1))
        self.assertEqual(rows[0]["ид_события"], EVENT_FIELDS[0])

    def test_wrong_header_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            path.write_text(",".join(EVENT_FIELDS[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source row 1"):
                list(iter_source_rows(path))

    def test_max_rows_must_be_positive(self) -> None:
        self.assertRaises(ValueError, iter_source_rows, Path("events.csv"), 0)

    @patch("stage1.ingestion.sources.find_seven_zip", return_value="7z")
    @patch("stage1.ingestion.sources.subprocess.run")
    def test_crlf_listing_with_two_csv_members_is_rejected(self, run, _find) -> None:
        listing = "----------\r\nPath = one.csv\r\nAttributes = A\r\n\r\nPath = two.csv\r\nAttributes = A\r\n"
        run.return_value = subprocess.CompletedProcess([], 0, listing.encode(), b"")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.7z"
            path.touch()
            with self.assertRaisesRegex(ValueError, "exactly one CSV"):
                list(iter_source_rows(path))

    @patch("stage1.ingestion.sources.find_seven_zip", return_value="7z")
    @patch("stage1.ingestion.sources.subprocess.Popen")
    @patch("stage1.ingestion.sources.subprocess.run")
    def test_archive_eof_before_limit_checks_nonzero_exit(self, run, popen, _find) -> None:
        listing = "----------\r\nPath = one.csv\r\nAttributes = A\r\n"
        run.return_value = subprocess.CompletedProcess([], 0, listing.encode(), b"")
        process = unittest.mock.Mock()
        process.stdout = io.BytesIO(csv_text(["id", "c", "d", "t", "x", "v"]).encode())
        process.poll.return_value = 2
        process.wait.return_value = 2
        popen.return_value = process
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.7z"
            path.touch()
            with self.assertRaisesRegex(RuntimeError, "extraction failed"):
                list(iter_source_rows(path, max_rows=2))


if __name__ == "__main__":
    unittest.main()
