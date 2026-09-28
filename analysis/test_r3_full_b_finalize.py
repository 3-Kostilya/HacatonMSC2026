"""The full B3 audit must reject a positive episode shared across splits."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from analysis.build_r1_state_mapping import _sha256
from analysis.build_r3_full_month import FULL_PACK_VERSION
from analysis.build_r3_registered_labels import LABEL_SCHEMA
from analysis.finalize_r3_full_registered_labels import finalize
from stage1.state_labeling.forecast import LABEL_VERSION, RegisteredForecastLabel


def _fixture(root: Path, *, shared_episode: bool) -> tuple[Path, Path]:
    a3_dir, b3_dir = root / "a3", root / "b3"
    a3_dir.mkdir()
    b3_dir.mkdir()
    months = [("2024-12", datetime(2024, 12, 1)),
              ("2025-01", datetime(2025, 1, 1))]
    chunks = [{"month": month, "rows": 1, "manifest_sha256": month} for month, _ in months]
    a3_manifest = a3_dir / "manifest.json"
    a3_manifest.write_text(json.dumps({
        "schema_version": FULL_PACK_VERSION, "status": "complete",
        "row_count": 2, "chunk_count": 2, "chunks": chunks,
        "source_m1_manifest_sha256": "m1", "source_b2_catalog_manifest_sha256": "b2",
    }), encoding="utf-8")
    a3_sha = _sha256(a3_manifest)
    for month, at in months:
        directory = b3_dir / f"year={month[:4]}" / f"month={month[5:]}"
        directory.mkdir(parents=True)
        episode = "same" if shared_episode or month == "2024-12" else "other"
        label = RegisteredForecastLabel(
            "c", "Датчик дыма", at, at + timedelta(hours=24), 1,
            "positive", "new_confident_registered_onset", at + timedelta(hours=1),
            episode, at - timedelta(hours=1),
            "train" if month == "2024-12" else "validation", "assigned",
        )
        label_path = directory / "registered_forecast_labels.parquet"
        pq.write_table(pa.Table.from_pylist([asdict(label)], schema=LABEL_SCHEMA), label_path)
        report_path = directory / "report.json"
        report_path.write_text(json.dumps({
            "label_status": {"positive": 1}, "assigned_label_status": {"positive": 1},
            "split_status": {"assigned": 1},
            "by_type": {"Датчик дыма": {"positive": 1}},
            "row_status_by_label": {"positive": {"discrete_data_status": {"eligible": 1}}},
            "unique_positive_episode_ids": [episode],
            "assigned_positive_episode_ids": [episode],
            "positive_channel_ids": ["c"],
            "channel_ids_by_label": {"positive": ["c"]},
            "channel_days_by_type_and_label": {"positive": {"Датчик дыма": 1}},
        }), encoding="utf-8")
        (directory / "manifest.json").write_text(json.dumps({
            "schema_version": LABEL_VERSION, "status": "complete_month", "month": month,
            "source_a3_full_manifest_sha256": a3_sha,
            "source_a3_month_manifest_sha256": month,
            "row_count": 1,
            "files": {path.name: {"sha256": _sha256(path)}
                      for path in (label_path, report_path)},
        }), encoding="utf-8")
    return a3_dir, b3_dir


class FullB3FinalizeTests(unittest.TestCase):
    def test_rejects_episode_in_two_assigned_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            a3_dir, b3_dir = _fixture(Path(temp), shared_episode=True)
            with self.assertRaisesRegex(ValueError, "two assigned splits"):
                finalize(a3_dir=a3_dir, labels_dir=b3_dir)

    def test_conserves_rows_and_distinct_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            a3_dir, b3_dir = _fixture(Path(temp), shared_episode=False)
            result = finalize(a3_dir=a3_dir, labels_dir=b3_dir)
            self.assertEqual(result["row_count"], 2)
            self.assertEqual(result["assigned_unique_positive_episodes_by_split"], {
                "train": 1, "validation": 1, "test": 0,
            })


if __name__ == "__main__":
    unittest.main()
