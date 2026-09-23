"""Both owners read the same committed M0 fixture and contract envelopes."""

from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from stage1.contracts import Decision, Episode, NormalizedEvent
from stage1.forecast_labels import label_future_onsets
from stage1.ingestion.m0_projection import iter_m0_clean_batches
from stage1.ingestion.pipeline import run_ingestion
from stage1.ml_m0_contracts import (
    CONTRACT_VERSION,
    SCHEMAS,
    in_feature_window,
    in_future_window,
    validate_table,
)
from stage1.pipeline import evaluate_channel


FIXTURE = Path(__file__).parent / "fixtures" / "m0"
PROVENANCE = {
    "schema_version": CONTRACT_VERSION,
    "run_id": "m0-fixture",
    "config_sha256": "0" * 64,
    "input_manifest_sha256": "1" * 64,
}


class SharedM0FixtureTests(unittest.TestCase):
    def test_a_and_b_read_identical_clean_table_and_two_episodes(self):
        expected = json.loads((FIXTURE / "expected.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "ingested"
            report = run_ingestion(
                {
                    "sources": [{"path": str(FIXTURE / "events.csv"), "max_rows": None}],
                    "channels": str(FIXTURE / "channels.csv"),
                    "objects": str(FIXTURE / "objects.csv"),
                    "output": str(destination),
                    "memory_limit": "128MB",
                    "batch_size": 3,
                }
            )
            rows = [
                row
                for path in sorted((destination / "clean").rglob("*.parquet"))
                for row in pq.read_table(path).to_pylist()
            ]
            shared = pa.Table.from_batches(list(iter_m0_clean_batches(destination, batch_size=3)))
            shared_path = Path(directory) / "shared.parquet"
            pq.write_table(shared, shared_path)
            reread = pq.read_table(shared_path)

        self.assertEqual(report["input_rows"], expected["input_rows"])
        self.assertEqual(report["dispositions"]["accepted"], expected["accepted_rows"])
        self.assertEqual(report["dispositions"]["exact_duplicate"], expected["exact_duplicates"])
        self.assertEqual(
            sum("unknown_channel" in row["quality_flags"] for row in rows),
            expected["unknown_channel_rows"],
        )
        self.assertEqual(
            sum("channel_time_conflict" in row["quality_flags"] for row in rows),
            expected["conflict_rows"],
        )
        self.assertFalse(report["dictionary_audit"]["object_mapping_available"])
        validate_table("clean", reread)
        self.assertEqual(reread.to_pylist(), shared.to_pylist())

        numeric = [
            NormalizedEvent(
                channel_id=row["channel_id"],
                timestamp=row["timestamp"],
                raw_value=row["value_raw"],
                numeric_value=row["value_numeric"],
                alarm=row["alarm"],
                sensor_type=row["sensor_type"],
                source=row["source"],
                quality_flags=tuple(row["quality_flags"]),
            )
            for row in rows
            if row["channel_id"] == "numeric"
        ]
        episodes = [
            episode
            for episode in evaluate_channel(numeric).detector_results
            if episode.decision is Decision.CANDIDATE
        ]
        self.assertEqual(
            [episode.start_at.isoformat(sep=" ") for episode in episodes],
            expected["numeric_episode_starts"],
        )

    def test_feature_and_target_boundaries(self):
        t = datetime.fromisoformat("2026-01-01 12:00:00")
        self.assertFalse(in_feature_window(t - timedelta(hours=1), t, 1))
        self.assertTrue(in_feature_window(t, t, 1))
        self.assertFalse(in_feature_window(t + timedelta(microseconds=1), t, 1))
        self.assertFalse(in_future_window(t, t, 24))
        self.assertTrue(in_future_window(t + timedelta(hours=24), t, 24))
        self.assertFalse(in_future_window(t + timedelta(hours=24, microseconds=1), t, 24))

        episode = Episode(
            episode_id="future-1",
            channel_id="numeric",
            sensor_type="Датчик температуры",
            sensor_group="numeric",
            anomaly_type="level_shift",
            decision=Decision.CANDIDATE,
            start_at=t + timedelta(hours=24),
            confirmed_at=t + timedelta(hours=25),
            ruleset_version="fixture",
            evidence=("fixture",),
            observation_quality=(),
        )
        label = label_future_onsets(
            "numeric", [t], [episode], observed_until=t + timedelta(hours=24)
        )[0]
        self.assertEqual((label.value, label.reason), (-1, "episode_confirmation_not_observed"))

    def test_b_envelopes_nullable_target_and_unavailable_prediction(self):
        t = datetime(2026, 1, 1, 12)
        training_row = {
            **PROVENANCE,
            "channel_id": "numeric",
            "prediction_time": t,
            "horizon_hours": 24,
            "target": None,
            "eligibility": "unknown",
            "eligibility_reasons": ["future_window_not_observed"],
            "split": "train",
            "episode_id": None,
        }
        table = pa.Table.from_pylist([training_row], schema=SCHEMAS["training_dataset"])
        validate_table("training_dataset", table)
        bad = pa.Table.from_pylist(
            [{**training_row, "target": 0}], schema=SCHEMAS["training_dataset"]
        )
        with self.assertRaisesRegex(ValueError, "unavailable target"):
            validate_table("training_dataset", bad)

        prediction = {
            **PROVENANCE,
            "channel_id": "numeric",
            "prediction_time": t,
            "horizon_hours": 24,
            "score": None,
            "probability": None,
            "risk_level": None,
            "status": "unknown",
            "status_reasons": ["insufficient_history"],
            "model_version": "fixture",
        }
        validate_table(
            "predictions", pa.Table.from_pylist([prediction], schema=SCHEMAS["predictions"])
        )

    def test_all_six_schemas_have_versions_provenance_and_keys(self):
        self.assertEqual(len(SCHEMAS), 6)
        for schema in SCHEMAS.values():
            self.assertEqual(
                schema.names[:4],
                ["schema_version", "run_id", "config_sha256", "input_manifest_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
