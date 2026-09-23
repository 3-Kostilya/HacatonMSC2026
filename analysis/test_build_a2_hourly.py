"""Focused checks for the bounded M1-to-A2 real-data command."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from analysis.build_a2_hourly import (
    _monthly_files,
    build_slice,
    select_real_20,
)
from stage1.features import A2_SCHEMA, FEATURE_VERSION
from stage1.registry import load_registry


START = datetime(2025, 6, 10)
END = START + timedelta(hours=3)


def _clean_artifact(root: Path) -> tuple[Path, Path]:
    artifact = root / "m1"
    partition = artifact / "clean" / "year=2025" / "month=6"
    partition.mkdir(parents=True)
    manifest_path = artifact / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "ingestion-v1",
                "status": "complete",
                "scope": "full_supplied_sources",
            }
        ),
        encoding="utf-8",
    )
    (artifact / "data_quality.json").write_text(
        json.dumps({"dictionary_audit": {"object_mapping_available": False}}),
        encoding="utf-8",
    )
    stats = pa.Table.from_pylist(
        [
            {
                "channel_id": "n1",
                "sensor_type": "Датчик температуры",
                "rows": 4,
                "numeric_count": 4,
                "state_count": 0,
                "alarm_count": 0.0,
                "first_at": START - timedelta(days=8),
                "last_at": START,
                "distinct_values": 4,
            }
        ]
    )
    pq.write_table(stats, artifact / "sensor_statistics.parquet")
    clean_schema = pa.schema(
        [
            ("channel_id", pa.string()),
            ("timestamp", pa.timestamp("us")),
            ("alarm", pa.bool_()),
            ("value_numeric", pa.float64()),
            ("value_state", pa.string()),
            ("sensor_type", pa.string()),
            ("object_id", pa.string()),
            ("join_status", pa.string()),
            ("quality_flags", pa.list_(pa.string())),
        ]
    )
    records = [
        {
            "channel_id": "n1",
            "timestamp": timestamp,
            "alarm": False,
            "value_numeric": float(index),
            "value_state": None,
            "sensor_type": "Датчик температуры",
            "object_id": None,
            "join_status": "object_mapping_unavailable",
            "quality_flags": [],
        }
        for index, timestamp in enumerate(
            (
                START - timedelta(days=8),
                START - timedelta(days=1),
                START,
                START + timedelta(hours=1),
            ),
            start=1,
        )
    ]
    pq.write_table(pa.Table.from_pylist(records, schema=clean_schema), partition / "data_0.parquet")
    channels = root / "channels.json"
    channels.write_text('["n1"]', encoding="utf-8")
    return manifest_path, channels


class BoundedA2Tests(unittest.TestCase):
    def test_month_selection_is_bounded_and_reports_missing_months(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            present = root / "clean" / "year=2025" / "month=6" / "data_0.parquet"
            present.parent.mkdir(parents=True)
            present.touch()
            files, missing = _monthly_files(root, datetime(2025, 5, 10), datetime(2025, 7, 1))
            self.assertEqual(files, [present])
            self.assertEqual(missing, ["2025-05"])

    def test_real_20_covers_registry_and_unknown_with_target_presence(self) -> None:
        types = sorted(policy.sensor_type for policy in load_registry())
        self.assertEqual(len(types), 19)
        stats = [
            {
                "channel_id": f"c{index}",
                "sensor_type": sensor_type,
                "rows": 100,
                "first_at": START - timedelta(days=30),
                "last_at": END,
            }
            for index, sensor_type in enumerate([*types, None])
        ]
        counts = Counter({row["channel_id"]: 1 for row in stats})
        selected = select_real_20(stats, START, END, seed=17, target_counts=counts)
        self.assertEqual(len(selected), 20)
        self.assertEqual(set(selected.values()), {*types, None})
        self.assertEqual(selected, select_real_20(stats, START, END, seed=17, target_counts=counts))
        counts["c0"] = 0
        with self.assertRaisesRegex(ValueError, "no active M1 channel"):
            select_real_20(stats, START, END, seed=17, target_counts=counts)

    def test_complete_run_provenance_schema_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_manifest, channels = _clean_artifact(root)
            output = root / "a2"
            result = build_slice(
                input_manifest=input_manifest,
                start_at=START,
                end_at=END,
                output=output,
                channels_file=channels,
            )
            self.assertTrue((output / "features.parquet").is_file())
            self.assertFalse((root / "a2.inprogress").exists())
            self.assertEqual(result["manifest"]["status"], "complete")
            self.assertEqual(result["manifest"]["feature_rows"], 3)
            self.assertEqual(result["validation"]["channel_count"], 1)
            self.assertEqual(result["validation"]["peer_features_status"], "unavailable")
            self.assertEqual(result["validation"]["missing_context_or_target_months"], ["2025-05"])
            table = pq.read_table(output / "features.parquet")
            self.assertTrue(table.schema.equals(A2_SCHEMA, check_metadata=False))
            self.assertEqual(table.num_rows, 3)
            first = table.slice(0, 1).to_pylist()[0]
            self.assertEqual(first["schema_version"], FEATURE_VERSION)
            self.assertEqual(
                first["input_manifest_sha256"],
                hashlib.sha256(input_manifest.read_bytes()).hexdigest(),
            )
            self.assertEqual(first["run_id"], result["manifest"]["run_id"])
            with self.assertRaises(FileExistsError):
                build_slice(
                    input_manifest=input_manifest,
                    start_at=START,
                    end_at=END,
                    output=output,
                    channels_file=channels,
                )

    def test_rejects_unbounded_range_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_manifest, channels = _clean_artifact(root)
            output = root / "a2"
            with self.assertRaisesRegex(ValueError, "at most"):
                build_slice(
                    input_manifest=input_manifest,
                    start_at=START,
                    end_at=START + timedelta(days=32),
                    output=output,
                    channels_file=channels,
                )
            self.assertFalse(output.exists())

    def test_rejects_excluded_2021_even_with_explicit_channels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_manifest, channels = _clean_artifact(root)
            with self.assertRaisesRegex(ValueError, "2021 is an explicitly excluded"):
                build_slice(
                    input_manifest=input_manifest,
                    start_at=datetime(2021, 6, 1),
                    end_at=datetime(2021, 6, 2),
                    output=root / "a2",
                    channels_file=channels,
                )


if __name__ == "__main__":
    unittest.main()
