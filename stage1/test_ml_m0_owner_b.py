"""B's negative checks for the shared handoff; no forecast model is fitted."""

from datetime import datetime, timezone
import hashlib
import unittest

import pyarrow as pa

from stage1.ml_m0_contracts import SCHEMAS, in_feature_window, in_future_window, validate_table
from stage1.test_ml_m0_contracts import FIXTURE, PROVENANCE


class OwnerBContractTests(unittest.TestCase):
    def prediction(self, **changes):
        return {
            **PROVENANCE,
            "channel_id": "new-channel",
            "prediction_time": datetime(2026, 1, 1),
            "horizon_hours": 24,
            "score": None,
            "probability": None,
            "risk_level": None,
            "status": "unknown",
            "status_reasons": ["insufficient_history"],
            "model_version": "m0-test",
            **changes,
        }

    def validate(self, name, rows):
        validate_table(name, pa.Table.from_pylist(rows, schema=SCHEMAS[name]))

    def test_fixture_matches_shared_revision(self):
        expected = {
            "events.csv": "f477ec9ea6054392738e4621e4d5e4f12baa10c17ec086aeb86c41d7de3049e4",
            "channels.csv": "09c0ac061c6d4420cf31a2e4af24addae1cd177654a899df0ed03b93acf68f49",
            "objects.csv": "79aae4ae67153fb3d0e86f43604ca2a0865cf623fbbb838bfd9e53772d4db0df",
            "expected.json": "b683221fa4f8b37a19d12f8eff081becea7e8c8217f354b5fb334cc16869b8a6",
        }
        for name, digest in expected.items():
            with self.subTest(file=name):
                # Git may check text out as CRLF on Windows; hash canonical LF bytes.
                data = (FIXTURE / name).read_bytes().replace(b"\r\n", b"\n")
                self.assertEqual(hashlib.sha256(data).hexdigest(), digest)

    def test_new_channel_has_no_fabricated_risk(self):
        self.validate("predictions", [self.prediction()])
        for changes in ({"score": 0}, {"probability": 0}, {"risk_level": "LOW"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.validate("predictions", [self.prediction(**changes)])

    def test_unavailable_requires_reason(self):
        with self.assertRaisesRegex(ValueError, "requires reasons"):
            self.validate("predictions", [self.prediction(status_reasons=[])])

    def test_nonfinite_scores_and_invalid_probabilities_are_rejected(self):
        for score in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(score=score), self.assertRaises(ValueError):
                self.validate("predictions", [self.prediction(status="eligible", score=score)])
        for probability in (-0.01, 1.01, float("nan")):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                self.validate(
                    "predictions",
                    [self.prediction(status="eligible", score=0.4, probability=probability)],
                )

    def test_duplicate_prediction_key_and_wrong_type_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            self.validate("predictions", [self.prediction(), self.prediction()])
        table = pa.Table.from_pylist([self.prediction()], schema=SCHEMAS["predictions"])
        index = table.schema.get_field_index("horizon_hours")
        table = table.set_column(index, "horizon_hours", pa.array([24], type=pa.int64()))
        with self.assertRaisesRegex(ValueError, "schema differs"):
            validate_table("predictions", table)

    def test_local_time_contract_rejects_timezone(self):
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for function in (in_feature_window, in_future_window):
            with self.subTest(function=function.__name__), self.assertRaises(ValueError):
                function(t, t, 24)

    def test_target_zero_one_and_unknown(self):
        base = {
            **PROVENANCE,
            "channel_id": "numeric",
            "prediction_time": datetime(2026, 1, 1),
            "horizon_hours": 24,
            "split": "train",
            "episode_id": None,
        }
        for target, eligibility, reasons in (
            (0, "eligible", []),
            (1, "eligible", []),
            (None, "unknown", ["future_window_not_observed"]),
            (None, "excluded", ["episode_already_ongoing"]),
        ):
            with self.subTest(target=target, eligibility=eligibility):
                self.validate(
                    "training_dataset",
                    [
                        {
                            **base,
                            "target": target,
                            "eligibility": eligibility,
                            "eligibility_reasons": reasons,
                        }
                    ],
                )

    def test_available_distance_requires_real_measure(self):
        row = {
            **PROVENANCE,
            "channel_id": "numeric",
            "as_of": datetime(2026, 1, 1),
            "method": "cluster",
            "method_version": "fixture",
            "score": 0.5,
            "score_measure": "rarity",
            "score_status": "eligible",
            "score_reasons": [],
            "evidence": ["fixture"],
            "distance_status": "eligible",
        }
        with self.assertRaisesRegex(ValueError, "requires value and measure"):
            self.validate("anomaly_scores", [row])
        self.validate(
            "anomaly_scores", [{**row, "cluster_distance": 1.2, "distance_measure": "euclidean"}]
        )


if __name__ == "__main__":
    unittest.main()
