import unittest

import sys

sys.path.append("src/")

from numpy.random import seed
from src.bases.root import Root
from src.examples.example_setups import setup_stat_scm
from src.utils.sem_utils.toy_sems import StationaryDependentSEM as StatSEM
from src.utils.sequential_sampling import sequentially_sample_model

seed(seed=0)


class TestRootClass(unittest.TestCase):
    #  Do NOT change the setUp method -- setUp is reserved by unittest.
    def setUp(self):
        #  Use STAT DAG to test Root class
        self.T = 3  # Time-steps in DAG
        self.n = 4  # Number of observational samples per variable per time-step
        self.N = 5  # Number of trials per time-step for method
        (
            self.init_sem,
            self.sem,
            _,
            self.G,
            self.exploration_sets,
            self.intervention_domain,
            self.true_objective_values,
        ) = setup_stat_scm(T=self.T)
        #  Sample observational data using SEM
        D_O = sequentially_sample_model(
            self.init_sem, self.sem, total_timesteps=self.T, sample_count=self.n, epsilon=None,
        )
        root_inputs = {
            "G": self.G,
            "sem": StatSEM,
            "base_target_variable": "Y",
            "observation_samples": D_O,  # Observational samples
            "intervention_domain": self.intervention_domain,
            "number_of_trials": self.N,
        }
        self.root = Root(**root_inputs)

    def test_setup_STAT_function(self):
        self.assertEqual(self.exploration_sets, [("X",), ("Z",), ("X", "Z")])
        self.assertEqual(self.intervention_domain, {"X": [-4, 1], "Z": [-3, 3]})
        self.assertAlmostEqual(
            self.true_objective_values, [-2.1518267393287287, -4.303653478657457, -6.455480217986186], places=7
        )
        self.assertEqual(self.init_sem.keys(), self.sem.keys())

    def test_root_methods(self):
        self.assertEqual(
            self.root.node_pars,
            {
                "X_0": (),
                "Z_0": ("X_0",),
                "Y_0": ("Z_0",),
                "X_1": ("X_0",),
                "Z_1": ("Z_0", "X_1"),
                "Y_1": ("Y_0", "Z_1"),
                "X_2": ("X_1",),
                "Z_2": ("Z_1", "X_2"),
                "Y_2": ("Y_1", "Z_2"),
            },
        )
        self.assertEqual(self.root.outcome_values, {0: [10000000.0], 1: [10000000.0], 2: [10000000.0]})
        self.assertEqual(
            self.root.sorted_nodes,
            {"X_0": 0, "Z_0": 1, "X_1": 2, "Y_0": 3, "Z_1": 4, "X_2": 5, "Y_1": 6, "Z_2": 7, "Y_2": 8},
        )


if __name__ == "__main__":
    unittest.main()
