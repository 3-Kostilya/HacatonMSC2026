import unittest

from stage1.readiness import ReadinessEvidence, assess_readiness


def evidence(**changes) -> ReadinessEvidence:
    values = {
        "types_covered": 19,
        "groups_covered": 7,
        "synthetic_scenarios_covered": 9,
        "real_examples_reviewed": 38,
        "causal_checks_passed": True,
        "reproducible_run_available": True,
        "temporal_protocol_frozen": True,
        "full_history_candidate_catalog": True,
        "candidate_diversity_sufficient": True,
        "candidates_traceable": True,
        "external_validation_available": False,
    }
    values.update(changes)
    return ReadinessEvidence(**values)


class ReadinessTests(unittest.TestCase):
    def test_method_can_finish_without_operational_claim(self):
        decision = assess_readiness(evidence())
        self.assertTrue(decision.stage_complete)
        self.assertTrue(decision.research_forecast_ready)
        self.assertFalse(decision.operational_claim_ready)
        self.assertIn("missing:external_validation_available", decision.blockers)

    def test_bounded_sample_is_not_full_forecast_input(self):
        decision = assess_readiness(
            evidence(full_history_candidate_catalog=False, candidate_diversity_sufficient=False)
        )
        self.assertTrue(decision.stage_complete)
        self.assertFalse(decision.research_forecast_ready)
        self.assertIn("missing:full_history_candidate_catalog", decision.blockers)

    def test_incomplete_method_cannot_advance(self):
        decision = assess_readiness(evidence(real_examples_reviewed=10))
        self.assertFalse(decision.stage_complete)
        self.assertFalse(decision.research_forecast_ready)


if __name__ == "__main__":
    unittest.main()
