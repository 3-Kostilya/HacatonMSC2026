import unittest

from analysis.run_b3_score_sequences import build_acceptance_report


class B3AcceptanceReportTests(unittest.TestCase):
    def test_fixed_score_acceptance_passes_and_keeps_recurrence(self):
        report = build_acceptance_report()
        self.assertTrue(report["passed"], report["checks"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["episode_counts"]["sustained"], 1)
        self.assertEqual(report["episode_counts"]["recurrence"], 2)
        self.assertEqual(len(report["notifications"]), 2)


if __name__ == "__main__":
    unittest.main()
