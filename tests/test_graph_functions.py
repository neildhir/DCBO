import unittest
from numpy import linspace, arange, array, inf
import pygraphviz
from networkx.drawing import nx_agraph

canonical_exploration_sets import generate_temporal_sub_CGM
from src.gaussian_process_utils import fit_conditional_distributions
from src.sequential_causal_functions import powerset, sequentially_sample_model
from src.structural_equation_models import DynamicalBayesianNetwork
from src.sequential_intervention_functions import get_interventional_grids
from src.utilities import (
    initialise_DCBO_parameters_and_objects_at_start_only,
    initialise_SCIBO_parameters_and_objects_full_graph,
    initialise_global_outcome_dict_new,
    convert_to_dict_of_temporal_lists,
)


class InitialisationFunctionality(unittest.TestCase):
    def setUp(self):
        # We'll use items related to a DBN
        dot_format_graph = generate_temporal_sub_CGM(0, 2, False)
        self.graph = nx_agraph.from_agraph(pygraphviz.AGraph(dot_format_graph))
        model = DynamicalBayesianNetwork()
        self.obs_samples = sequentially_sample_model(model.sem(), self.graph, sample_count=5, total_timesteps=3)
        self.base_target = "Y"

    def test_number_of_timesteps(self):
        total_timesteps = self.graph.number_of_nodes() // len(
            set("".join(filter(str.isalpha, "".join(list(self.graph.nodes)))))
        )
        self.assertEqual(total_timesteps, 3, "Should be 3")

    def test_canonical_variables(self):
        self.assertEqual(self.obs_samples.keys(), {"X", "Y", "Z"})

    @unittest.skip("Old way of construction emission functions.")
    def test_keys_of_emission_functions_old(self):
        initial_emission_functions, emission_functions, transition_functions = fit_conditional_distributions(
            self.graph, self.obs_samples
        )
        self.assertEqual(initial_emission_functions.keys(), {("X",), ("Z",), ("X", "Z")})
        self.assertEqual(transition_functions.keys(), {"X", "Y", "Z"})
        self.assertEqual(
            emission_functions.keys(),
            {("X",), ("Z",), ("X", "Z"), ("X", "y_{t-1}"), ("Z", "y_{t-1}"), ("X", "Z", "y_{t-1}")},
        )

    def test_keys_of_emission_functions(self):
        initial_emission_functions, emission_functions, transition_functions = fit_conditional_distributions(
            self.graph, self.obs_samples
        )
        self.assertEqual(initial_emission_functions.keys(), {("X",), ("Z",), ("X", "Z")})
        self.assertEqual(transition_functions.keys(), {"X", "Y", "Z"})
        self.assertEqual(
            emission_functions.keys(), {("X",), ("Z",), ("X", "Z")},
        )

    def test_target_variables(self):
        all_target_variables = list(filter(lambda k: self.base_target in k, self.graph.nodes))
        self.assertEqual(all_target_variables, ["Y_0", "Y_1", "Y_2"])

    def test_which_canonical_exploration_set(self):
        canonical_manipulative_variables = list(filter(lambda k: self.base_target not in k, self.obs_samples.keys()))
        canonical_exploration_sets = list(powerset(canonical_manipulative_variables))
        self.assertEqual(canonical_exploration_sets, [("X",), ("Z",), ("X", "Z")])

    def test_interventional_grids(self):
        nr_samples = 10
        interventional_variable_limits = {"X": [-15, 3], "Z": [-1, 10]}
        canonical_manipulative_variables = list(filter(lambda k: self.base_target not in k, self.obs_samples.keys()))
        canonical_exploration_sets = list(powerset(canonical_manipulative_variables))
        grids = get_interventional_grids(canonical_exploration_sets, interventional_variable_limits, nr_samples)
        compare_vector = linspace(
            interventional_variable_limits["X"][0], interventional_variable_limits["X"][1], num=nr_samples
        ).reshape(-1, 1)
        self.assertEqual(compare_vector.shape, grids[canonical_exploration_sets[0]].shape)
        self.assertTrue((compare_vector == grids[canonical_exploration_sets[0]]).all())

    def test_SCIBO_initialisation_function_full_graph(self):
        exploration_sets = [("X",), ("Z",), ("X", "Z")]
        interventional_data = {key: None for key in exploration_sets}
        nr_interventions = 2
        base_target = "Y"
        index_name = 0
        task = "min"
        for i, es in enumerate(exploration_sets):
            interventional_data[es] = {
                "X": arange(0 + i, 9 + i).reshape(3, -1),
                "Y": arange(3 + i, 12 + i).reshape(3, -1),
                "Z": arange(6 + i, 15 + i).reshape(3, -1),
            }
            interventional_data[es][base_target][-1, -1] = 0

        (
            initial_optimal_intervention_set,
            initial_optimal_target_values,
            optimal_intervention_levels,
            interventional_data_X,
            interventional_data_Y,
        ) = initialise_SCIBO_parameters_and_objects_full_graph(
            exploration_sets,
            interventional_data,
            nr_interventions,
            base_target,
            index_name,
            task,
            turn_off_shuffle=True,
        )

canonical_exploration_setsal(initial_optimal_intervention_set, ("X",))
        self.assertTrue((initial_optimal_target_values == array([3, 4, 0])).all())
        self.assertCountEqual(optimal_intervention_levels, {"X": array([0, 1, 8]), "Z": None})
        self.assertCountEqual(
            interventional_data_X,
            {
                ("X",): array([[0], [1], [8]]),
                ("Z",): array([[7], [8], [15]]),
                ("X", "Z"): array([[2, 8], [3, 9], [10, 16]]),
            },
        )
        self.assertCountEqual(
            interventional_data_Y,
            {("X",): array([[3], [4], [0]]), ("Z",): array([[4], [5], [0]]), ("X", "Z"): array([[5], [6], [0]])},
        )

    def test_SCIBO_initialisation_function_initial_time_step(self):
        exploration_sets = [("X",), ("Z",), ("X", "Z")]
        interventional_data = {key: None for key in exploration_sets}
        nr_interventions = 2
        base_target = "Y"
        index_name = 0
        task = "min"
        for i, es in enumerate(exploration_sets):
            interventional_data[es] = {
                "X": arange(0 + i, 9 + i).reshape(3, -1),
                "Y": arange(3 + i, 12 + i).reshape(3, -1),
                "Z": arange(6 + i, 15 + i).reshape(3, -1),
            }
            interventional_data[es][base_target][-1, -1] = 0

        (
            initial_optimal_intervention_set,
            initial_optimal_target_values,
            optimal_intervention_levels,
            interventional_data_X,
            interventional_data_Y,
        ) = initialise_DCBO_parameters_and_objects_at_start_only(
            exploration_sets,
            interventional_data,
            nr_interventions,
            base_target,
            index_name,
            task,
            turn_off_shuffle=True,
        )

        self.assertEqual(initial_optimal_intervention_set, ("X",))
        self.assertEqual(initial_optimal_target_values, 3)
        self.assertCountEqual(optimal_intervention_levels, {"X": 0, "exploration_setsl(
            interventional_data_X, {("X",): array([[0]]), ("Z",): array([[7]]), ("X", "Z"): array([[2, 8]])}
        )
        self.assertCountEqual(
            interventional_data_Y, {("X",): array([[3]]), ("Z",): array([[4]]), ("X", "Z"): array([[5]])}
        )

    def test_trial_updated_objects(self):
        max_T = 3
        vals = [1, 1, 1]
        best_target_values_expected = {t: [inf, 1] for t in range(max_T)}
        result = initialise_global_outcome_dict_new(task="min", max_T=max_T, initial_optimal_target_values=vals)
        self.assertEqual(result, best_target_values_expected)

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

    def test_target_function_initialisation(self):
        pass


if __name__ == "__main__":
    unittest.main()
