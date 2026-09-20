import unittest

from stage1.verification import verify_causal_invariants


class CausalVerificationTests(unittest.TestCase):
    def test_all_readiness_causal_invariants_pass(self):
        result = verify_causal_invariants()
        self.assertTrue(result["all_passed"], result["checks"])
        self.assertEqual(result["passed"], result["total"])


if __name__ == "__main__":
    unittest.main()
