import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from analysis.ml_experiment_history import CompletedHistory


class CompletedHistoryTests(unittest.TestCase):
    def catalog(self, rows):
        frame = pd.DataFrame(rows, columns=['channel_id', 'start_at', 'end_at'])
        frame['start_at'] = pd.to_datetime(frame.start_at)
        frame['end_at'] = pd.to_datetime(frame.end_at)
        frame['onset_status'] = 'candidate_new_onset'
        frame['end_status'] = 'exact_norma'
        frame['uncertain_intervening_state'] = False
        return frame

    def test_future_closure_is_invisible(self):
        past = self.catalog([('a', '2023-01-01', '2023-01-02')])
        future = self.catalog([('a', '2023-01-03', '2023-01-10')])
        frame = pd.DataFrame({'channel_id': ['a'], 'prediction_time': [pd.Timestamp('2023-01-04')]})
        left = CompletedHistory(past).features(frame)
        right = CompletedHistory(pd.concat([past, future])).features(frame)
        assert_frame_equal(left, right)
        self.assertEqual(left.history_completed_count_7d.iloc[0], 1)

    def test_segment_reset_and_exact_window_boundary(self):
        catalog = self.catalog([('a', '2020-12-01', '2020-12-02'),
                                ('a', '2022-01-01', '2022-01-02')])
        frame = pd.DataFrame({'channel_id': ['a', 'a'], 'prediction_time': pd.to_datetime(['2022-01-01', '2022-01-09'])})
        out = CompletedHistory(catalog).features(frame)
        self.assertEqual(out.history_completed_count_segment.iloc[0], 0)
        self.assertEqual(out.history_completed_count_segment.iloc[1], 1)
        self.assertEqual(out.history_completed_count_7d.iloc[1], 0)

    def test_ambiguous_closure_is_excluded(self):
        catalog = self.catalog([('a', '2023-01-01', '2023-01-02')])
        catalog['uncertain_intervening_state'] = True
        frame = pd.DataFrame({'channel_id': ['a'], 'prediction_time': [pd.Timestamp('2023-01-04')]})
        self.assertEqual(CompletedHistory(catalog).features(frame).history_completed_count_segment.iloc[0], 0)


if __name__ == '__main__':
    unittest.main()
