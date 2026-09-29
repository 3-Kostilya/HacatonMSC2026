"""Pilot selection and replay cannot select or score using future observations."""

from datetime import datetime, timedelta
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.replay_shadow_pilot import observation_groups, replay, select_channels, to_b_input
from analysis.replay_shadow_pilot import OUTPUT_SCHEMA
from analysis.audit_shadow_pilot import run as audit_pilot
from analysis.r6_provenance import frozen_rule_sha256
from analysis.train_r4_discrete_baselines import sha256
from stage1.shadow.stream import ShadowStream
from ml.forecast.shadow_pilot import ShadowPolicy, ShadowState, run_shadow_batch
import json


T = datetime(2025, 12, 1)


def row(at, *, channel="c", text="Норма", kind="Датчик дыма", row_id=1):
    return {
        "row_id": row_id,
        "channel_id": channel,
        "timestamp": at,
        "alarm": False,
        "value_numeric": None,
        "value_state": text,
        "sensor_type": kind,
        "source": f"ext-journal-{at.year}.7z",
    }


def history():
    return [row(T - timedelta(days=35 - i), row_id=i + 1) for i in range(26)]


def results(rows, *, end=T + timedelta(hours=3)):
    groups = observation_groups(sorted(rows, key=lambda r: (r["timestamp"], r["channel_id"])))
    return [
        value for value, _ in replay(ShadowStream(threshold=7.1), groups, ["c", "empty"], T, end)
    ]


class ShadowPilotReplayTests(unittest.TestCase):
    def test_a_rows_feed_b_without_future_labels_and_preserve_all_decisions(self):
        source = [
            *history(),
            row(T - timedelta(hours=3)),
            *[row(T - timedelta(hours=2), text="Неисправен", row_id=100 + i) for i in range(4)],
            row(T - timedelta(hours=1)),
        ]
        policy = ShadowPolicy.from_freeze(Path("ml/r6_frozen_rule_v1.json"))
        state = ShadowState()
        for hour in results(source):
            decisions = run_shadow_batch([to_b_input(item) for item in hour], state, policy)
            for expected, actual in zip(hour, decisions, strict=True):
                self.assertEqual(expected["rule_score"], actual["rule_score"])
                self.assertEqual(expected["above_frozen_threshold"], actual["threshold_crossed"])
                self.assertEqual(expected["warning_emitted"], actual["shadow_warning"])
                self.assertFalse(actual["automatic_action_taken"])
            state = ShadowState.restore(state.checkpoint(policy), policy)

    def test_selection_never_uses_future_channel_or_future_type(self):
        with TemporaryDirectory() as temporary, duckdb.connect() as database:
            path = Path(temporary) / "observations.parquet"
            rows = [
                row(T - timedelta(hours=1)),
                row(T, channel="future", kind="ИБП"),
                row(T + timedelta(hours=1), kind="Газовый датчик"),
            ]
            pq.write_table(pa.Table.from_pylist(rows), path)
            selected = select_channels(database, [path], T - timedelta(days=1), T, 20)
            self.assertEqual(selected, [{"channel_id": "c", "observed_type": "Датчик дыма"}])
            pq.write_table(pa.Table.from_pylist(rows[:1]), path)
            self.assertEqual(
                select_channels(database, [path], T - timedelta(days=1), T, 20), selected
            )

    def test_selection_does_not_resolve_same_second_type_conflict_using_row_ids(self):
        with TemporaryDirectory() as temporary, duckdb.connect() as database:
            path = Path(temporary) / "observations.parquet"
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        row(T - timedelta(hours=1), row_id=1),
                        row(T - timedelta(hours=1), kind="ИБП", row_id=2),
                    ]
                ),
                path,
            )
            self.assertEqual(
                select_channels(database, [path], T - timedelta(days=1), T, 20),
                [{"channel_id": "c", "observed_type": "<conflicting>"}],
            )

    def test_complete_groups_survive_source_batch_boundaries(self):
        first = [row(T, text="Норма", row_id=1)]
        second = [row(T, text="Неисправен", row_id=2), row(T + timedelta(hours=1))]
        groups = list(observation_groups(chain(first, second)))
        self.assertEqual([len(group) for group in groups], [2, 1])
        result = results([*history(), *first, *second])[0][0]
        self.assertIsNone(result["rule_score"])
        self.assertIn("registered_episode_active_at_t", result["admission_reasons"])

    def test_every_hour_and_empty_channel_are_reported(self):
        actual = results([*history(), row(T)])
        self.assertEqual(len(actual), 3)
        self.assertEqual([len(hour) for hour in actual], [2, 2, 2])
        self.assertTrue(all(hour[1]["rule_score"] is None for hour in actual))
        self.assertEqual(
            [hour[0]["prediction_time"] for hour in actual],
            [T + timedelta(hours=i) for i in range(3)],
        )

    def test_future_tail_mutation_leaves_all_earlier_outputs_unchanged(self):
        old = [*history(), row(T)]
        first = results(old)
        after = results([*old, row(T + timedelta(hours=3), text="Неисправен")])
        self.assertEqual(first, after)

    def test_long_batch_and_short_prefix_replay_are_identical(self):
        source = [
            *history(),
            row(T),
            row(T + timedelta(hours=1), text="Неисправен"),
            row(T + timedelta(hours=2)),
        ]
        batch = results(source)
        for i in range(3):
            cutoff = T + timedelta(hours=i)
            truncated = [event for event in source if event["timestamp"] <= cutoff]
            self.assertEqual(results(truncated, end=cutoff + timedelta(hours=1)), batch[: i + 1])

    def test_invalid_and_unbounded_replay_is_rejected(self):
        for end in (T, T + timedelta(days=32), T + timedelta(minutes=10)):
            with self.subTest(end=end), self.assertRaisesRegex(ValueError, "whole local hours"):
                results([], end=end)

    def test_post_inference_audit_accepts_missing_noneligible_a3_keys_but_rejects_bad_score(self):
        def write_json(path, value):
            path.write_text(json.dumps(value) + "\n", encoding="utf-8")

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pilot, a3 = root / "pilot", root / "a3"
            pilot.mkdir()
            a3.mkdir()
            actual = results([*history(), row(T)], end=T + timedelta(hours=1))[0]
            predictions = pilot / "shadow_predictions.parquet"
            pq.write_table(pa.Table.from_pylist(actual, schema=OUTPUT_SCHEMA), predictions)
            feature = a3 / "features.parquet"
            status = a3 / "row_status.parquet"
            names = [
                "channel_id",
                "prediction_time",
                "registered_fault_text_count_24h",
                "registered_fault_text_count_168h",
                "technical_message_count_24h",
                "completed_episode_count_168h",
            ]
            pq.write_table(
                pa.Table.from_pylist([{name: actual[0][name] for name in names}]), feature
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [{"channel_id": "c", "prediction_time": T, "discrete_data_status": "eligible"}]
                ),
                status,
            )
            write_json(
                a3 / "manifest.json",
                {
                    "source_m1_manifest_sha256": "m1",
                    "chunks": [
                        {
                            "month": "2025-12",
                            "start_at": T.isoformat(),
                            "end_at": "2026-01-01T00:00:00",
                            "features_file": feature.name,
                            "row_status_file": status.name,
                            "features_sha256": sha256(feature),
                            "row_status_sha256": sha256(status),
                        }
                    ],
                },
            )
            freeze = root / "freeze.json"
            write_json(freeze, {"source_a3_manifest_sha256": sha256(a3 / "manifest.json")})
            write_json(
                pilot / "report.json",
                {
                    "start": T.isoformat(),
                    "end_exclusive": "2025-12-01T01:00:00",
                    "source_freeze_lf_sha256": frozen_rule_sha256(freeze),
                    "source_m1_manifest_sha256": "m1",
                    "conditionally_scored_hours": 1,
                },
            )
            manifest = {
                "prediction_file": predictions.name,
                "prediction_rows": 2,
                "prediction_sha256": sha256(predictions),
                "report_sha256": sha256(pilot / "report.json"),
            }
            write_json(pilot / "manifest.json", manifest)
            args = {"pilot_dir": pilot, "a3_dir": a3, "freeze_path": freeze}
            audit = audit_pilot(**args, output_dir=root / "ok")
            self.assertEqual(audit["rows_checked"], 2)
            self.assertEqual(audit["common_a3_rows"], 1)
            actual[0]["rule_score"] = 0.1
            pq.write_table(pa.Table.from_pylist(actual, schema=OUTPUT_SCHEMA), predictions)
            manifest["prediction_sha256"] = sha256(predictions)
            write_json(pilot / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "missing-prediction contract"):
                audit_pilot(**args, output_dir=root / "bad")


if __name__ == "__main__":
    unittest.main()
