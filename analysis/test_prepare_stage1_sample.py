import csv
from pathlib import Path
import tempfile
import unittest

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # Optional locally; installed by the project CI requirements.
    pq = None


class CrossFileNormalizationTests(unittest.TestCase):
    @unittest.skipIf(pq is None, "pyarrow is not installed")
    def test_duplicate_and_conflict_state_is_global_across_input_files(self):
        from analysis.prepare_stage1_sample import build_sample

        channel = "temperature-1"
        fields = [
            "ид_события",
            "ид_канала_данных",
            "дата",
            "время",
            "тревожное",
            "значение_датчика",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dictionary = root / "dictionary.csv"
            with dictionary.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["ид_канала_данных", "тип_датчика"])
                writer.writerow([channel, "Датчик температуры"])
            paths = []
            for name, records in (("a.csv", (("1", "10"),)), ("b.csv", (("1", "10"), ("2", "20")))):
                path = root / name
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(fields)
                    for event_id, value in records:
                        writer.writerow(
                            [event_id, channel, "2026-01-01", "00:00:00", "false", value]
                        )
                paths.append(path)
            target = root / "sample.parquet"
            build_sample(paths, target, 100, 2, dictionary_path=dictionary)
            rows = pq.read_table(target).to_pylist()

        self.assertEqual(
            [row["disposition"] for row in rows], ["accepted", "exact_duplicate", "accepted"]
        )
        self.assertTrue(all("channel_time_conflict" in row["quality_flags"] for row in rows))
        self.assertEqual(rows[1]["duplicate_of_source"], "a.csv")


if __name__ == "__main__":
    unittest.main()
