"""Unknown outcomes remain unresolved in the full-stream alert metric."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest

import duckdb
import pandas as pd

from analysis.ml_experiment_round4_full_context import assess


class FullContextMetricTests(unittest.TestCase):
    def test_unknown_warning_affects_denominator_but_is_not_a_negative_label(self):
        at=datetime(2025,1,3)
        rows=[]
        labels=[]
        for index,target in enumerate([1,None,0]):
            time=at+timedelta(hours=index)
            rows.append({"channel_id":str(index),"prediction_time":time,
                         "sensor_type":"Датчик дыма","warning_emitted":True,
                         "reason":"standard_24h_warning"})
            labels.append({"channel_id":str(index),"prediction_time":time,
                           "sensor_type":"Датчик дыма","target":target,
                           "target_episode_id":"e1" if target==1 else None,
                           "label_available_at":time+timedelta(hours=1) if target==1 else None})
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"labels.parquet"
            frame=pd.DataFrame(labels)
            frame["target"]=pd.array(frame["target"],dtype="Int8")
            frame.to_parquet(path,index=False)
            with duckdb.connect() as db:
                metric,alerts=assess(db,rows,[str(path)],full=2)
                capped,_=assess(db,rows,[str(path)],full=1)
        self.assertEqual(metric["warnings"],3)
        self.assertEqual(metric["matched_known_episodes"],1)
        self.assertEqual(metric["unknown_outcome_warnings"],1)
        self.assertEqual(metric["known_no_target_warnings"],1)
        self.assertEqual(metric["precision_lower_bound"],1/3)
        self.assertEqual(metric["precision_known_evaluable"],1/2)
        self.assertEqual(metric["full_recall_lower_bound"],1/2)
        self.assertEqual(capped["precision_loose_upper_bound"],1/3)
        self.assertEqual(alerts.outcome.tolist(),[
            "matched_known_episode","unknown_target","known_no_target"])


if __name__=="__main__":
    unittest.main()
