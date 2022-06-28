import unittest

from numpy import arange, linspace
from numpy.random import seed
from src.bases.root import Root
from src.examples.example_setups import setup_stat_scm
from src.utils.sem_utils.toy_sems import StationaryDependentSEM as StatSEM
from src.utils.sequential_intervention_functions import get_interventional_grids
from src.utils.sequential_sampling import sequentially_sample_model
from src.utils.utilities import convert_to_dict_of_temporal_lists, powerset

seed(seed=0)


class TestRoot(unittest.TestCase):
    #  Do NOT change the setUp method -- setUp is reserved by unittest.
    def setUp(self):
        #  Use STAT DAG to test Root class
        self.T = 3  # Time-steps in DAG
        self.n = 4  # Number of observational samples per variable per time-step
        self.N = 5  # Number of trials per time-step for method
        (
            self.init_sem,
            self.sem,
            _,  #  view of the DAG
            self.G,
            self.exploration_sets,
            self.intervention_domain,
            self.true_objective_values,
            _,  # optimal interventions
            _,  # all causal effects
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
        self.assertEqual(self.root.interventional_variable_limits, {"X": [-4, 1], "Z": [-3, 3]})
        #  If we do not pass any exploration set, then by default the Root class will assign all manipulative variables as the intervention set.
        self.assertEqual(self.root.exploration_sets, [("X", "Z")])
        self.assertEqual(
            self.root.interventional_data_y, {0: {("X", "Z"): None}, 1: {("X", "Z"): None}, 2: {("X", "Z"): None}}
        )
        self.assertEqual(
            self.root.interventional_data_x, {0: {("X", "Z"): None}, 1: {("X", "Z"): None}, 2: {("X", "Z"): None}}
        )

    def test_dict_to_list_conversion_of_observational_samples(self):
        observational_samples = {
            "X": arange(0, 9).reshape(3, -1),
            "Y": arange(3, 12).reshape(3, -1),
            "Z": arange(6, 15).reshape(3, -1),
        }
        out = convert_to_dict_of_temporal_lists(observational_samples)
        self.assertEqual(len(out["X"]), 3)
        self.assertEqual(len(out["Z"][0]), 3)
        self.assertEqual(sum([len(out["Y"][t]) for t in range(3)]), 9)

    def test_interventional_grids(self):
        nr_samples = 10
        interventional_variable_limits = {"X": [-15, 3], "Z": [-1, 10]}
        exploration_sets = list(powerset(self.root.manipulative_variables))
        grids = get_interventional_grids(exploration_sets, interventional_variable_limits, nr_samples)
        compare_vector = linspace(
            interventional_variable_limits["X"][0], interventional_variable_limits["X"][1], num=nr_samples
        ).reshape(-1, 1)
        self.assertEqual(compare_vector.shape, grids[exploration_sets[0]].shape)
        self.assertTrue((compare_vector == grids[exploration_sets[0]]).all())

    def test_target_variables(self):
        self.assertEqual(self.root.all_target_variables, ["Y_0", "Y_1", "Y_2"])

    def test_canonical_variables(self):
        self.assertEqual(self.root.observational_samples.keys(), {"X", "Y", "Z"})

    def test_number_of_nodes_per_time_slice(self):
        # Number of nodes per time-slice
        v_n = len(self.root.G.nodes()) / self.root.G.T
        nodes = list(self.root.G.nodes())
        self.assertEqual(v_n, 3)
        for t in range(self.G.T):
            self.assertEqual(len([v for v in nodes if v.split("_")[1] == str(t)]), v_n)


if __name__ == "__main__":
    unittest.main()
