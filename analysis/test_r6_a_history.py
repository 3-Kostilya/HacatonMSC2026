"""Independent raw SQL count path matches the causal clock implementation."""

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.audit_r6_a_history import recount_inputs
from analysis.r6_provenance import frozen_rule_sha256
from stage1.features.hourly import FeatureEvent
from stage1.features.r2 import CompletedEpisode
from stage1.features.r6_history import iter_rule_history


T = datetime(2026, 2, 10, 12)
SOURCE_SCHEMA = pa.schema(
    [
        pa.field("row_id", pa.int64()),
        pa.field("channel_id", pa.string()),
        pa.field("timestamp", pa.timestamp("us")),
        pa.field("sensor_type", pa.string()),
        pa.field("value_state", pa.string()),
        pa.field("alarm", pa.bool_()),
        pa.field("quality_flags", pa.list_(pa.string())),
        pa.field("source", pa.string()),
    ]
)


class R6RawHistoryTests(unittest.TestCase):
    def test_frozen_text_hash_accepts_only_line_ending_conversion(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "freeze.json"
            lf = b'{"threshold":7.1}\n'
            path.write_bytes(lf)
            expected = frozen_rule_sha256(path)
            path.write_bytes(lf.replace(b"\n", b"\r\n"))
            self.assertEqual(frozen_rule_sha256(path), expected)
            path.write_bytes(b'{"threshold":7.2}\n')
            self.assertNotEqual(frozen_rule_sha256(path), expected)
            path.write_bytes(b'{"threshold":7.1}\r')
            with self.assertRaises(ValueError):
                frozen_rule_sha256(path)

    def test_raw_sql_and_clock_include_only_past_valid_archive_observations(self):
        source = [
            {"timestamp": T - timedelta(hours=168)},
            {"timestamp": T - timedelta(hours=24)},
            {"timestamp": T},
            {"timestamp": T + timedelta(hours=1)},
            {"timestamp": T, "quality_flags": ["channel_time_conflict"]},
            {"timestamp": T, "source": "журнал_событий_пример.csv"},
            {"timestamp": T, "source": "ext-journal-2025.7z"},
            {
                "timestamp": T,
                "sensor_type": "Состояние вентилятора",
                "value_state": "Батарея неисправна",
            },
            {"timestamp": T, "sensor_type": None},
        ]
        rows = [
            {
                "row_id": i,
                "channel_id": "c",
                "sensor_type": "Датчик дыма",
                "value_state": "Неисправен",
                "alarm": False,
                "quality_flags": [],
                "source": "ext-journal-2026.7z",
                **row,
            }
            for i, row in enumerate(source)
        ]
        completed = [
            CompletedEpisode("c", T - timedelta(hours=2), T),
            CompletedEpisode("c", T - timedelta(hours=1), T + timedelta(hours=1)),
        ]
        times = [T, T + timedelta(hours=1)]
        with TemporaryDirectory() as temporary, duckdb.connect() as database:
            path = Path(temporary) / "source.parquet"
            pq.write_table(pa.Table.from_pylist(rows, schema=SOURCE_SCHEMA), path)
            database.register(
                "prediction_keys",
                pa.Table.from_pylist([{"channel_id": "c", "prediction_time": at} for at in times]),
            )
            recount_inputs(database, [path], completed)
            actual = (
                database.execute("SELECT * FROM raw_rule_inputs ORDER BY prediction_time")
                .to_arrow_table()
                .to_pylist()
            )
        archive = [row for row in rows if row["source"] == "ext-journal-2026.7z"]
        expected = list(
            iter_rule_history(
                [FeatureEvent.from_clean_record(row) for row in archive],
                completed,
                "c",
                times,
            )
        )
        self.assertEqual(actual, expected)
        self.assertEqual(actual[0]["registered_fault_text_count_24h"], 1)
        self.assertEqual(actual[0]["technical_message_count_24h"], 2)
        self.assertEqual(actual[0]["completed_episode_count_168h"], 1)

    def test_targets_are_not_accepted_by_raw_feature_builder(self):
        with duckdb.connect() as database:
            database.register(
                "prediction_keys",
                pa.Table.from_pylist([{"channel_id": "c", "prediction_time": T, "target": 1}]),
            )
            with self.assertRaisesRegex(ValueError, "never target"):
                recount_inputs(database, [], [])


if __name__ == "__main__":
    unittest.main()
