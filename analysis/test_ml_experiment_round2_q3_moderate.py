"""Equal episode mass with a milder positive class correction."""
import unittest

import numpy as np
import pandas as pd

from analysis.ml_experiment_round2_q3_moderate import episode_sqrt_weights


class ModerateWeightTests(unittest.TestCase):
    def test_positive_episode_mass_is_equal(self):
        frame = pd.DataFrame({"target":[1,1,1,1,0,0,0,0,0,0],
                              "target_episode_id":["a","a","a","b",None,None,None,None,None,None]})
        weight,classes = episode_sqrt_weights(frame,"episode_sqrt")
        self.assertAlmostEqual(weight[:3].sum(),weight[3])
        self.assertEqual(weight[4:].tolist(),[1]*6)
        self.assertAlmostEqual(classes[1],np.sqrt(6/4))

    def test_missing_positive_episode_id_is_rejected(self):
        frame = pd.DataFrame({"target":[1,0],"target_episode_id":[None,None]})
        with self.assertRaises(ValueError):
            episode_sqrt_weights(frame,"episode_sqrt")


if __name__ == "__main__":
    unittest.main()
