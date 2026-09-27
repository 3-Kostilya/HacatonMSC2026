"""Independent randomized parity and operational warning edge cases."""

import unittest
from pathlib import Path
import tempfile

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.audit_ml_experiment import metadata_parity
from analysis.ml_experiment_eval import (
    PreparedEvaluation, combine_scores, evaluate, search_thresholds,
)
from ml.forecast.alert_eval import evaluate_alerts
from analysis.ml_experiment_metric_audit import replay


def fixture(rows):
    frame = pd.DataFrame(rows, columns=["channel_id", "prediction_time", "target",
                                        "target_episode_id", "score"])
    start = pd.Timestamp("2024-01-01")
    frame["prediction_time"] = start + pd.to_timedelta(frame.prediction_time, unit="h")
    frame["label_available_at"] = frame.prediction_time + pd.Timedelta(hours=1)
    frame["sensor_type"] = "smoke"
    return frame


class ExperimentEvaluationTest(unittest.TestCase):
    def test_false_warning_suppresses_true_and_boundary_is_allowed(self):
        frame = fixture([("a", 0, 0, None, .9), ("a", 23, 1, "e1", .9),
                         ("a", 24, 1, "e2", .9), ("a", 48, 1, "e2", .9)])
        result = evaluate(frame, "score", .5, 4)
        self.assertEqual(result["emitted_warnings"], 3)
        self.assertEqual(result["suppressed_positive_score_rows"], 1)
        self.assertEqual(result["matched_episodes"], 1)
        self.assertEqual(result["duplicate_episode_warnings"], 1)
        self.assertEqual(result["full_episode_recall"], .25)
        self.assertEqual(result["available_episode_recall"], .5)
        self.assertEqual(result["matched_episode_ids"], ["e2"])

    def test_randomized_production_parity(self):
        rng = np.random.default_rng(88)
        rows = []
        for channel in ["z", "a", "b"]:
            for hour in range(24 * 15):
                target = int(rng.random() < .1)
                rows.append((channel, hour, target,
                             f"{channel}-{hour // 36}" if target else None,
                             rng.uniform()))
        frame = fixture(rows).sample(frac=1, random_state=1)
        prepared = PreparedEvaluation(frame, 100)
        for threshold in [.1, .5, .9, 1.1]:
            reference, alerts = evaluate_alerts(frame.rename(columns={"score": "catboost_score"}),
                                               "catboost_score", threshold,
                                               channel_days=prepared.channel_days)
            result = prepared.evaluate("score", threshold)
            for field in ["emitted_warnings", "suppressed_positive_score_rows",
                          "matched_episodes", "unmatched_warnings",
                          "duplicate_episode_warnings", "episode_precision",
                          "median_lead_hours", "unmatched_warnings_per_1000_channel_days"]:
                self.assertEqual(result[field], reference[field], field)
            matched = alerts.loc[alerts.outcome.eq("matched_episode"), "target_episode_id"]
            self.assertEqual(set(result["matched_episode_ids"]), set(matched))
            self.assertEqual(result["episode_recall"], reference["matched_episodes"] / 100)

    def test_all_available_denominator_does_not_hide_unavailable(self):
        frame = fixture([("a", 0, 1, "e1", .8), ("b", 0, 1, "e2", .8)])
        result = evaluate(frame, "score", .5, 10)
        self.assertEqual(result["episode_recall"], .2)
        self.assertEqual(result["available_episode_recall"], 1)
        self.assertAlmostEqual(result["episode_f1"], 1 / 3)

    def test_reverse_order_and_integer_nanosecond_timestamps(self):
        frame = fixture([("b", 24, 1, "b1", .8), ("a", 0, 0, None, .8),
                         ("b", 0, 1, "b2", .8), ("a", 24, 1, "a1", .8)])
        expected = evaluate(frame, "score", .5, 5)
        reverse = frame.iloc[::-1].copy()
        self.assertEqual(evaluate(reverse, "score", .5, 5), expected)
        reverse["prediction_time"] = reverse.prediction_time.to_numpy(
            dtype="datetime64[ns]").astype("int64")
        reverse["label_available_at"] = reverse.label_available_at.to_numpy(
            dtype="datetime64[ns]").astype("int64")
        self.assertEqual(evaluate(reverse, "score", .5, 5), expected)

    def test_float32_threshold_comparison_matches_production(self):
        frame = fixture([("a", 0, 1, "e1", 1), ("a", 24, 1, "e2", 1)])
        frame["score"] = frame.score.astype("float32")
        threshold = float(np.nextafter(1.0, float("inf")))
        canonical, _ = evaluate_alerts(frame.rename(columns={"score": "catboost_score"}),
                                      "catboost_score", threshold, channel_days=2)
        fast = evaluate(frame, "score", threshold, 2)
        self.assertEqual(fast["emitted_warnings"], canonical["emitted_warnings"])
        typed = evaluate(frame, "score", {"smoke": threshold}, 2)
        self.assertEqual(typed["emitted_warnings"], canonical["emitted_warnings"])
        curve = search_thresholds(frame, "score", 2, points=40)
        self.assertEqual(curve[-1]["emitted_warnings"], 0)

    def test_per_type_thresholds_and_score_fusion(self):
        frame = fixture([("a", 0, 1, "e1", .8), ("b", 0, 0, None, .8)])
        frame.loc[1, "sensor_type"] = "gas"
        result = evaluate(frame, "score", {"smoke": .7, "gas": .9}, 2)
        self.assertEqual(result["emitted_warnings"], 1)
        frame["second"] = [.4, .2]
        np.testing.assert_allclose(combine_scores(frame, ["score", "second"]), [.6, .5])
        np.testing.assert_allclose(combine_scores(frame, ["score", "second"], method="max"),
                                   [.8, .8])
        with self.assertRaises(ValueError):
            evaluate(frame, "score", {"smoke": .7}, 2)

    def test_invalid_leads_unknown_targets_and_denominator(self):
        frame = fixture([("a", 0, 1, "e1", .8)])
        with self.assertRaises(ValueError):
            evaluate(frame, "score", .5, 0)
        for lead in [0, 25]:
            altered = frame.copy()
            altered.label_available_at = altered.prediction_time + pd.Timedelta(hours=lead)
            with self.assertRaises(ValueError):
                evaluate(altered, "score", .5, 1)
        frame.target = -1
        with self.assertRaises(ValueError):
            evaluate(frame, "score", .5, 1)

    def test_score_only_grid_and_empty(self):
        frame = fixture([("a", i, 0, None, i / 100) for i in range(100)])
        curve = search_thresholds(frame, "score", 20, points=40)
        self.assertEqual(len(curve), 40)
        self.assertEqual(curve[-1]["emitted_warnings"], 0)
        shuffled_targets = frame.assign(target=1, target_episode_id="x")
        other = search_thresholds(shuffled_targets, "score", 20, points=40)
        self.assertEqual([x["threshold"] for x in curve], [x["threshold"] for x in other])
        result = evaluate(frame.iloc[:0], "score", .5, 20)
        self.assertEqual(result["emitted_warnings"], 0)

    def test_audit_sql_native_float32_filter_and_full_metadata(self):
        frame = fixture([("a", 0, 1, "e1", 1), ("a", 24, 1, "e2", 1)])
        frame["score"] = frame.score.astype("float32")
        with tempfile.TemporaryDirectory(prefix="ml-evaluation-test-") as folder:
            source = Path(folder) / "scores.parquet"
            data = Path(folder) / "data.parquet"
            pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), source)
            pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), data)
            with duckdb.connect() as db:
                check = metadata_parity(db, source, data, 2, 2024)
                self.assertEqual(check["full_outer_metadata_mismatches"], 0)
                audited = replay(db, source, "score", float(np.nextafter(1.0, np.inf)), 4)
                self.assertEqual(audited["matched_episodes"], 2)
                self.assertEqual(audited["full_episode_recall"], .5)
                changed = frame.copy()
                changed.loc[0, "target_episode_id"] = "incorrect"
                pq.write_table(pa.Table.from_pandas(changed, preserve_index=False), data)
                with self.assertRaises(AssertionError):
                    metadata_parity(db, source, data, 2, 2024)


if __name__ == "__main__":
    unittest.main()
