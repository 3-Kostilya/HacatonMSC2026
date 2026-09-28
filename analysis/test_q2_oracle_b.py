"""Small exact-capacity checks independent of A's oracle implementation."""

from __future__ import annotations

from datetime import timedelta
import unittest

import pandas as pd

from analysis.verify_q2_oracle_b import channel_capacity


class Q2OracleBCapacityTest(unittest.TestCase):
    def test_24_hour_boundary_and_blocked_neighbour(self) -> None:
        base = pd.Timestamp("2025-01-01 00:00:00")
        times = [base + timedelta(hours=hour) for hour in (0, 1, 23, 24, 47, 48)]
        self.assertEqual(channel_capacity(times), (3, 3))

    def test_empty_channel_and_invalid_order(self) -> None:
        self.assertEqual(channel_capacity([]), (0, 0))
        with self.assertRaises(ValueError):
            channel_capacity([pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-01")])


if __name__ == "__main__":
    unittest.main()
