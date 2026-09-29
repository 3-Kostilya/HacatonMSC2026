import unittest

from analysis.build_stage1_final_report import _causal_checks_passed, _passed_scenario_ids


class FinalReportEvidenceTests(unittest.TestCase):
    def test_same_scenario_across_suite_specific_ids_counts_once(self):
        def checks(suite):
            return {
                "details": [
                    {"scenario_id": f"{suite}:drift:001", "scenario_name": "drift", "passed": True}
                ]
            }

        self.assertEqual(
            _passed_scenario_ids(checks("tuning"))
            & _passed_scenario_ids(checks("synthetic_holdout")),
            {"drift"},
        )

    def test_causal_evidence_requires_complete_executable_checks(self):
        self.assertFalse(
            _causal_checks_passed(
                {"all_passed": True, "passed": 1, "total": 1, "checks": {"causal": False}}
            )
        )
        self.assertTrue(
            _causal_checks_passed(
                {"all_passed": True, "passed": 1, "total": 1, "checks": {"causal": True}}
            )
        )

    def test_scenarios_count_only_when_the_same_id_passed_in_both_suites(self):
        tuning = _passed_scenario_ids(
            {
                "details": [
                    {"scenario_id": "one", "passed": True},
                    {"scenario_id": "two", "passed": True},
                ]
            }
        )
        holdout = _passed_scenario_ids(
            {
                "details": [
                    {"scenario_id": "one", "passed": True},
                    {"scenario_id": "three", "passed": True},
                ]
            }
        )

        self.assertEqual(tuning & holdout, {"one"})


if __name__ == "__main__":
    unittest.main()
