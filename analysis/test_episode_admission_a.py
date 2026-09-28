"""Episode-level attribution must not invent a new admission or ignore purge."""

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.audit_episode_admission_a import select_positive_labels, summarize_points


T = datetime(2025, 2, 1)


def point(episode, offset=0, reasons=("baseline_unusable",), **overrides):
    result = {
        "split": "validation",
        "target_episode_id": episode,
        "channel_id": episode,
        "sensor_type": "Датчик дыма",
        "prediction_time": T + timedelta(hours=offset),
        "discrete_data_status": "unknown" if reasons else "eligible",
        "discrete_data_reasons": list(reasons),
        "baseline_reasons": [],
        "availability_status": "unknown",
        "old_admitted": not reasons,
    }
    result.update(overrides)
    return result


class EpisodeAdmissionTests(unittest.TestCase):
    def test_any_every_and_counterfactuals_use_all_assigned_hours(self):
        points = [
            point("mixed", 0),
            point("mixed", 1, ("state_history_missing", "state_transitions_unavailable")),
            point("quality", 2, ("baseline_unusable", "quality_exclusions_24h")),
            point("quality", 3, ("baseline_unusable", "quality_exclusions_24h")),
            point("available", 4, ()),
        ]
        episodes, results = summarize_points(points)
        mixed = next(r for r in episodes if r["target_episode_id"] == "mixed")
        quality = next(r for r in episodes if r["target_episode_id"] == "quality")
        self.assertEqual(mixed["reasons_on_every_hour"], [])
        self.assertEqual(mixed["minimum_simultaneous_vetoes"], 1)
        self.assertEqual(mixed["minimal_reason_sets"], [["baseline_unusable"]])
        self.assertEqual(
            quality["reasons_on_every_hour"], ["baseline_unusable", "quality_exclusions_24h"]
        )
        summary = results["validation"]
        self.assertEqual(summary["full_positive_episodes"], 3)
        self.assertEqual(summary["available_positive_episodes"], 1)
        self.assertEqual(summary["positive_hours"], 5)
        self.assertEqual(summary["missed_reason_any_hour"]["baseline_unusable"], 2)
        self.assertEqual(summary["missed_reason_every_hour"]["baseline_unusable"], 1)
        self.assertEqual(summary["missed_minimum_veto_histogram"], {1: 1, 2: 1})
        for name, count in summary["counterfactual_episode_counts"].items():
            self.assertEqual(count, 1 if name == "unchanged" else 2)
        self.assertEqual(summary["minimum_matched_episodes_for_recall_strictly_above_half"], 2)

    def test_duplicate_reasons_do_not_multiply_episode_counts(self):
        _, summaries = summarize_points(
            [
                point(
                    "e",
                    0,
                    ("baseline_unusable", "baseline_unusable"),
                    baseline_reasons=["insufficient_baseline_events"],
                ),
                point(
                    "e",
                    1,
                    baseline_reasons=[
                        "insufficient_baseline_events",
                        "insufficient_baseline_events",
                    ],
                ),
            ]
        )
        result = summaries["validation"]
        self.assertEqual(result["missed_reason_every_hour"]["baseline_unusable"], 1)
        self.assertEqual(
            result["missed_baseline_reason_every_hour"]["insufficient_baseline_events"], 1
        )

    def test_excluded_quality_and_other_vetoes_remain_protected(self):
        _, summaries = summarize_points(
            [
                point("excluded", discrete_data_status="excluded", availability_status="excluded"),
                point("quality", reasons=("quality_exclusions_24h",)),
                point("history", reasons=("insufficient_history",)),
            ]
        )
        self.assertTrue(
            all(n == 0 for n in summaries["validation"]["counterfactual_episode_counts"].values())
        )

    def test_r3_availability_condition_is_not_silently_dropped(self):
        episodes, summaries = summarize_points(
            [point("e", reasons=(), availability_status="eligible", old_admitted=False)]
        )
        self.assertEqual(
            episodes[0]["reasons_on_every_hour"], ["r3_availability_status_not_unknown"]
        )
        self.assertEqual(summaries["validation"]["available_positive_episodes"], 0)

    def test_inconsistent_status_admission_type_and_duplicate_keys_fail(self):
        broken = [
            [point("e", reasons=(), old_admitted=False)],
            [point("e", discrete_data_status="eligible")],
            [point("e", reasons=(), discrete_data_status="unknown")],
            [point("e"), point("e")],
            [point("e"), point("e", 1, sensor_type="Газовый датчик")],
            [point("e"), point("e", 1, channel_id="different")],
            [point("e", target_episode_id=None)],
        ]
        for rows in broken:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                summarize_points(rows)

    def test_test_excluded_year_and_wrong_temporal_split_fail(self):
        for year, split in (
            (2026, "validation"),
            (2021, "train"),
            (2025, "train"),
            (2024, "validation"),
        ):
            with self.subTest(year=year, split=split), self.assertRaises(ValueError):
                summarize_points([point("e", prediction_time=datetime(year, 2, 1), split=split)])

    def test_positive_diagnostic_selection_respects_purge_and_label_status(self):
        rows = []
        for name, target, label, assigned, split in (
            ("valid", 1, "positive", "assigned", "validation"),
            ("purged", 1, "positive", "purged_boundary", "validation"),
            ("unknown", None, "unknown", "assigned", "validation"),
            ("wrong_label", 1, "unknown", "assigned", "validation"),
            ("test", 1, "positive", "assigned", "test"),
            ("negative", 0, "negative", "assigned", "validation"),
        ):
            rows.append(
                {
                    "channel_id": name,
                    "prediction_time": T,
                    "sensor_type": "Датчик дыма",
                    "target_episode_id": name,
                    "target": target,
                    "label_status": label,
                    "split_status": assigned,
                    "split": split,
                }
            )
        with TemporaryDirectory() as temporary, duckdb.connect() as database:
            path = Path(temporary) / "labels.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path)
            select_positive_labels(database, [str(path)])
            self.assertEqual(
                database.execute("SELECT channel_id FROM labels").fetchall(), [("valid",)]
            )


if __name__ == "__main__":
    unittest.main()
