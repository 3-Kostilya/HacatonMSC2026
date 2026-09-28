"""Research router must join complete forecast identity, not row position."""
from pathlib import Path
import tempfile
import unittest

import duckdb
import pandas as pd

from analysis.ml_experiment_round2_coverage_routing import align_sources


def fixture():
    return pd.DataFrame({"channel_id": ["a", "b"],
        "prediction_time": pd.to_datetime(["2024-02-01", "2024-02-02"]),
        "sensor_type": ["Датчик дыма", "Датчик дыма"],
        "target": [1, 0], "target_episode_id": ["ep1", None],
        "label_available_at": pd.to_datetime(["2024-02-02", "2024-02-03"])})


class RoutingAlignmentTests(unittest.TestCase):
    def paths(self, folder, second):
        root = Path(folder)
        for name in ["frozen-models", "retrained", "data"]:
            (root/name).mkdir()
        first = fixture()
        first.assign(score_tree=[.9,.1],score_linear=[.7,.2]).to_parquet(
            root/"frozen-models/scores_tune.parquet", index=False)
        second.assign(score_engineered_episode=[.8,.2]).to_parquet(
            root/"retrained/scores_tune.parquet", index=False)
        first.to_parquet(root/"data/tune.parquet", index=False)
        return root

    def test_key_alignment_is_independent_of_file_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.paths(folder, fixture().iloc[::-1].reset_index(drop=True))
            with duckdb.connect() as db:
                align_sources(db,root,"tune",root/"joined.parquet",2)
                result = db.execute("SELECT channel_id,score_retrained FROM read_parquet(?) ORDER BY 1",
                                    [str(root/"joined.parquet")]).fetchall()
            self.assertEqual(result,[('a',.2),('b',.8)])

    def test_missing_key_cannot_silently_drop_a_row(self):
        with tempfile.TemporaryDirectory() as folder:
            second = fixture()
            second.loc[1,"channel_id"] = "elsewhere"
            root = self.paths(folder,second)
            with duckdb.connect() as db:
                with self.assertRaises(AssertionError):
                    align_sources(db,root,"tune",root/"joined.parquet",2)

    def test_conflicting_outcomes_cannot_join_same_key(self):
        with tempfile.TemporaryDirectory() as folder:
            second = fixture()
            second.loc[0,"target_episode_id"] = "future-different"
            root = self.paths(folder,second)
            with duckdb.connect() as db:
                with self.assertRaises(AssertionError):
                    align_sources(db,root,"tune",root/"joined.parquet",2)


if __name__ == "__main__":
    unittest.main()
