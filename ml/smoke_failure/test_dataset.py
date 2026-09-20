import unittest

import numpy as np
import pandas as pd

from dataset import (
    AMBIGUOUS,
    FAULT,
    TARGET,
    assign_split,
    canonicalize,
    features_at,
    label_at,
    timeline,
)


def history(hours, states):
    return canonicalize(
        pd.DataFrame(
            {
                "channel_id": "1",
                "timestamp": pd.Timestamp("2022-01-01") + pd.to_timedelta(hours, unit="h"),
                "state": states,
                "alarm": False,
            }
        )
    )


class LabelTests(unittest.TestCase):
    def test_repeats_recovery_and_horizon_boundaries(self):
        g = history(
            [0, 12, 24, 30, 31, 35, 36, 40, 64, 88],
            ["Норма", "Норма", "Норма", FAULT, FAULT, "Норма", FAULT, "Норма", "Норма", "Норма"],
        )
        self.assertEqual(np.flatnonzero(timeline(g)["onset"]).tolist(), [3, 6])
        q = pd.Timestamp("2022-01-01") + pd.to_timedelta([6, 24, 30, 35, 40, 65], unit="h")
        self.assertEqual(label_at(g, q)[TARGET].tolist(), [1, 1, -1, 1, 0, -1])

    def test_unknown_and_long_gap_censor(self):
        g = history(
            [0, 12, 24, 26, 28, 50, 80],
            ["Норма", "Норма", "Норма", "Неопределен", FAULT, "Норма", FAULT],
        )
        self.assertFalse(timeline(g)["onset"].any())
        q = pd.Timestamp("2022-01-01") + pd.to_timedelta([24, 50], unit="h")
        self.assertEqual(label_at(g, q)[TARGET].tolist(), [-1, -1])

    def test_simultaneous_conflicting_states_are_unknown(self):
        g = history([0, 12, 24, 24, 30], ["Норма", "Норма", FAULT, "Норма", FAULT])
        self.assertEqual(g.state.iloc[2], AMBIGUOUS)
        self.assertFalse(timeline(g)["onset"].any())

    def test_features_do_not_change_when_future_is_changed(self):
        g = history(
            [0, 12, 24, 30, 36, 48, 60], ["Норма", "Норма", "Норма", FAULT, "Норма", "Норма", FAULT]
        )
        q = [pd.Timestamp("2022-01-02")]
        truncated = g.loc[g.timestamp <= q[0]].copy()
        pd.testing.assert_frame_equal(features_at(g, q), features_at(truncated, q))
        mutated = g.copy()
        mutated.loc[mutated.timestamp > q[0], "state"] = "Неопределен"
        pd.testing.assert_frame_equal(features_at(g, q), features_at(mutated, q))

    def test_purge_prevents_labels_crossing_splits(self):
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2022-12-30 23:00",
                        "2022-12-31",
                        "2023-01-01",
                        "2023-06-30",
                        "2023-07-01",
                        "2023-12-31",
                    ],
                    format="mixed",
                )
            }
        )
        self.assertEqual(
            assign_split(frame).split.tolist(),
            ["train", "excluded", "validation", "excluded", "test", "excluded"],
        )

    def test_first_observed_fault_is_not_an_onset(self):
        self.assertFalse(timeline(history([0, 1, 2], [FAULT, FAULT, FAULT]))["onset"].any())

    def test_incomplete_positive_window_is_also_censored(self):
        g = history([0, 12, 24, 25], ["Норма", "Норма", "Норма", FAULT])
        self.assertEqual(label_at(g, [pd.Timestamp("2022-01-02")])[TARGET].tolist(), [-1])


if __name__ == "__main__":
    unittest.main()
