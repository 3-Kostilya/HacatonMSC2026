import unittest
from unittest.mock import patch

from extract import find_seven_zip


class ArchiveExecutableTests(unittest.TestCase):
    @patch.dict("os.environ", {"SEVEN_ZIP": "/custom/7z"}, clear=True)
    @patch("extract.shutil.which", return_value="/custom/7z")
    def test_explicit_executable(self, which):
        self.assertEqual(find_seven_zip(), "/custom/7z")
        which.assert_called_once_with("/custom/7z")

    @patch.dict("os.environ", {}, clear=True)
    @patch("extract.shutil.which", side_effect=[None, "/usr/bin/7z"])
    def test_path_fallback(self, which):
        self.assertEqual(find_seven_zip(), "/usr/bin/7z")
        self.assertEqual(which.call_count, 2)

    @patch.dict("os.environ", {"SEVEN_ZIP": "/missing/7z"}, clear=True)
    @patch("extract.shutil.which", return_value=None)
    def test_missing_configured_executable(self, which):
        with self.assertRaisesRegex(FileNotFoundError, "SEVEN_ZIP"):
            find_seven_zip()
        which.assert_called_once_with("/missing/7z")


if __name__ == "__main__":
    unittest.main()
