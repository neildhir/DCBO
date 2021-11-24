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

if __name__ == "__main__":
    unittest.main()
