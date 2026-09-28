"""Causality, contiguity and metadata tests for trailing score policies."""

import unittest

import duckdb
import numpy as np
import pandas as pd

from analysis.ml_experiment_round2_temporal import META, temporal_projection_sql


def rows(channel,hours,probabilities):
    start = pd.Timestamp("2024-01-01")
    return pd.DataFrame({"channel_id": channel,
                         "prediction_time": start+pd.to_timedelta(hours,unit="h"),
                         "sensor_type": "smoke","target": 0,"target_episode_id": None,
                         "label_available_at": start+pd.to_timedelta(np.asarray(hours)+24,unit="h"),
                         "score_pooled_raw": probabilities,"score_linear_raw": probabilities})


def project(frame):
    with duckdb.connect() as db:
        db.register("fixture",frame)
        return db.execute(temporal_projection_sql("SELECT * FROM fixture")).fetch_df().sort_values(
            ["channel_id","prediction_time"]).reset_index(drop=True)


class TemporalPolicyTest(unittest.TestCase):
    def test_complete_trailing_windows_include_current_and_strict_past(self):
        probabilities = [.1,.2,.3,.4,.5,.6]
        out = project(rows("a",range(6),probabilities))
        logit = np.log(np.asarray(probabilities)/(1-np.asarray(probabilities)))
        self.assertTrue(np.isnan(out.score_pooled_meanlogit_2h.iloc[0]))
        self.assertAlmostEqual(out.score_pooled_meanlogit_2h.iloc[1],np.mean(logit[:2]),places=6)
        self.assertAlmostEqual(out.score_pooled_meanlogit_3h.iloc[2],np.mean(logit[:3]),places=6)
        self.assertAlmostEqual(out.score_pooled_meanlogit_6h.iloc[5],np.mean(logit),places=6)
        self.assertAlmostEqual(out.score_pooled_min_2h.iloc[2],.2,places=6)
        self.assertAlmostEqual(out.score_pooled_min_3h.iloc[2],.1,places=6)
        self.assertAlmostEqual(out.score_pooled_rising1h.iloc[2],logit[2]+.5*(logit[2]-logit[1]),places=6)

    def test_gaps_and_channel_boundaries_do_not_fill_or_carry(self):
        frame = pd.concat([rows("a",[0,1,4,5],[.1,.2,.8,.9]),
                           rows("b",[5,6],[.9,.95])],ignore_index=True)
        out = project(frame)
        a = out.loc[out.channel_id.eq("a")].reset_index(drop=True)
        self.assertTrue(np.isnan(a.score_pooled_meanlogit_2h.iloc[2]))
        self.assertTrue(np.isnan(a.score_pooled_rising1h.iloc[2]))
        self.assertFalse(np.isnan(a.score_pooled_meanlogit_2h.iloc[3]))
        self.assertTrue(np.isnan(a.score_pooled_meanlogit_3h.iloc[3]))
        b = out.loc[out.channel_id.eq("b")].reset_index(drop=True)
        self.assertTrue(np.isnan(b.score_pooled_min_2h.iloc[0]))
        self.assertTrue(np.isnan(b.score_pooled_rising1h.iloc[0]))

    def test_future_change_and_future_append_leave_all_previous_scores_unchanged(self):
        original = rows("a",range(8),np.linspace(.1,.8,8))
        first = project(original)
        changed = original.copy()
        changed.loc[changed.index>=5,["score_pooled_raw","score_linear_raw"]] = .999
        second = project(pd.concat([changed,rows("a",[8,9],[.99,.999])],ignore_index=True))
        pd.testing.assert_frame_equal(first.iloc[:5],second.iloc[:5])

    def test_labels_do_not_enter_aggregation_and_metadata_are_preserved(self):
        frame = rows("a",range(7),np.linspace(.1,.7,7))
        frame["target_episode_id"] = [None,None,"e1",None,None,None,None]
        frame.loc[2,"target"] = 1
        out = project(frame.sample(frac=1,random_state=11))
        pd.testing.assert_frame_equal(out[META],frame[META],check_dtype=False)
        altered = frame.copy()
        altered.target = 1-altered.target
        altered.target_episode_id = "different-target"
        other = project(altered)
        names = [column for column in out if column.startswith("score_")]
        pd.testing.assert_frame_equal(out[names],other[names])

    def test_missing_probability_is_unavailable_instead_of_extreme_logit(self):
        frame = rows("a",[0,1,2],[.1,np.nan,.9])
        out = project(frame)
        self.assertTrue(out.score_pooled_meanlogit_2h.isna().all())
        self.assertTrue(out.score_pooled_rising1h.isna().all())


if __name__=="__main__":
    unittest.main()
