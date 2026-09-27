import unittest

import duckdb
import pandas as pd

from analysis.ml_experiment_online_linear import available_training_rows


class QuarterlyAvailabilityTests(unittest.TestCase):
    def test_future_label_and_boundary_label_cannot_enter_update(self):
        import tempfile
        from pathlib import Path
        frame = pd.DataFrame({'channel_id':['a','b','c'],
                              'prediction_time': pd.to_datetime(['2025-03-29']*3),
                              'label_available_at': pd.to_datetime(['2025-03-30','2025-04-01','2025-04-02']),
                              'target':[1,1,1]})
        with tempfile.TemporaryDirectory() as directory, duckdb.connect() as db:
            path = Path(directory)/'past.parquet'
            frame.to_parquet(path, index=False)
            selected = available_training_rows(db, path, '2025-04-01')
        self.assertEqual(selected.channel_id.tolist(), ['a'])


if __name__ == '__main__':
    unittest.main()
