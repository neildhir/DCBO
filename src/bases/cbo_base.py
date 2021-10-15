from copy import deepcopy
from itertools import combinations
from random import choice
from typing import Callable

from networkx.algorithms.dag import topological_sort
from networkx.classes.multidigraph import MultiDiGraph
from src.bayes_opt.cost_functions import define_costs
from src.utils.sem_utils.emissions import fit_sem_complex
from src.utils.sequential_intervention_functions import (
    evaluate_target_function,
    get_interventional_grids,
    make_sequential_intervention_dictionary,
)
from src.utils.utilities import (
    convert_to_dict_of_temporal_lists,
    create_intervention_exploration_domain,
    initialise_DCBO_parameters_and_objects_filtering,
    initialise_global_outcome_dict_new,
    initialise_optimal_intervention_level_list,
    update_emission_pairs_keys,
)


class BaseClassCBO:
    """
    Base class for the CBO system.
    """

    def __init__(
        self,
        G: str,
        sem: classmethod,
        make_sem_hat: Callable,
        observational_samples: dict,
        intervention_domain: dict,
        interventional_samples: dict = None,  # interventional data collected for specific intervention sets
        exploration_sets: list = None,
        number_of_trials=10,
        base_target_variable: str = "Y",
        ground_truth: list = None,
        estimate_sem: bool = False,
        task: str = "min",
        n_restart: int = 1,
        cost_type: int = 1,  # There are multiple options here
        use_mc: bool = False,
        debug_mode: bool = False,
        online: bool = False,
        num_anchor_points: int = 100,
        args_sem=None,
        manipulative_variables=None,
        change_points: list = None,
        root_instrument: bool = None,
    ):
        self.debug_mode = debug_mode
        if args_sem is None and change_points is None:
            true_sem = sem()
        elif args_sem and change_points is None:
            true_sem = sem(args_sem[0], args_sem[1])
        else:
            true_sem = sem(change_points.index(True))

        # These will be used in the target function evaluation
        self.true_initial_sem = true_sem.static()  # for t = 0
        self.true_sem = true_sem.dynamic()  # for t > 0
        self.make_sem_hat = make_sem_hat

        # Make sure data has been normalised/centred
        self.observational_samples = observational_samples

        # Number of optimization restart for GPs
        self.n_restart = n_restart
        self.online = online
        self.use_mc = use_mc

        # Total time-steps and sample count per time-step
        _, T = observational_samples[list(observational_samples.keys())[0]].shape
        assert isinstance(G, MultiDiGraph)
        self.graph = G
        self.sem_variables = [v.split("_")[0] for v in [v for v in G.nodes if v.split("_")[1] == "0"]]
        self.root_instrument = root_instrument
        self.node_children = {node: None for node in self.graph.nodes}
        self.node_parents = {node: None for node in self.graph.nodes}
        self.emission_pairs = {}

        # Children of all nodes
        for node in self.graph.nodes:
            self.node_children[node] = list(self.graph.successors(node))

        #  Parents of all nodes
        for node in self.graph.nodes:
            self.node_parents[node] = tuple(self.graph.predecessors(node))

        emissions = {t: [] for t in range(T)}
        for e in G.edges:
            _, inn_time = e[0].split("_")
            _, out_time = e[1].split("_")
            # Emission edgee
            if out_time == inn_time:
                emissions[int(out_time)].append((e[0], e[1]))

        new_emissions = deepcopy(emissions)
        self.emissions = emissions
        for t in range(T):
            for a, b in combinations(emissions[t], 2):
                if a[1] == b[1]:
                    new_emissions[t].append(((a[0], b[0]), a[1]))
                    cond = [v for v in list(G.predecessors(b[0])) if v.split("_")[1] == str(t)]
                    if len(cond) != 0 and cond[0] == a[0]:
                        # Remove from list
                        new_emissions[t].remove(a)

        self.emission_pairs = {}
        for t in range(T):
            for pair in new_emissions[t]:
                if isinstance(pair[0], tuple):
                    self.emission_pairs[pair[0]] = pair[1]
                else:
                    self.emission_pairs[(pair[0],)] = pair[1]

        # Sometimes the input and output pair order does not match because of NetworkX internal issues,
        # so we need adjust the keys so that they do match.
        self.emission_pairs = update_emission_pairs_keys(T, self.node_parents, self.emission_pairs)

        self.estimate_sem = estimate_sem
        self.sem_emit_fncs = fit_sem_complex(observational_samples, self.emission_pairs)

        # XXX: assumes that we have the same initial obs count per variable. Not true for most real problems.
        self.number_of_trials = number_of_trials

        #  Induced sub-graph on the nodes in the first time-slice -- it doesn't matter which time-slice we consider since one of the main assumptions is that time-slice topology does not change in the DBN.
        gg = self.G.subgraph([v + "_0" for v in observational_samples.keys()])
        #  Causally ordered nodes in the first time-slice
        self.causal_order = list(v.split("_")[0] for v in topological_sort(gg))
        #  See page 199 of 'Elements of Causal Inference' for a reference on summary graphs.
        self.summary_graph_node_parents = {
            v.split("_")[0]: tuple([vv.split("_")[0] for vv in gg.predecessors(v)]) for v in gg.nodes
        }
        assert self.causal_order == list(self.summary_graph_node_parents.keys())

        # Check that we are either minimising or maximising the objective function
        assert task in ["min", "max"], task
        self.task = task
        if task == "min":
            self.blank_val = 1e7  # Positive infinity
        elif task == "max":
            self.blank_val = -1e7  # Negative infinity
        self.base_target_variable = base_target_variable  # This has to be reflected in the CGM
        self.index_name = 0

        # Instantiate blanket that will form final solution
        (self.optimal_blanket, self.total_timesteps,) = make_sequential_intervention_dictionary(self.graph)
        self.assigned_blanket = deepcopy(self.optimal_blanket)
        self.empty_intervention_blanket, _ = make_sequential_intervention_dictionary(self.graph)

        # Canonical manipulative variables
        if manipulative_variables is None:
            self.manipulative_variables = list(
                filter(lambda k: self.base_target_variable not in k, self.observational_samples.keys(),)
            )
        else:
            self.manipulative_variables = manipulative_variables

        self.interventional_variable_limits = intervention_domain
        assert self.manipulative_variables == list(intervention_domain.keys())
        assert isinstance(exploration_sets, list)
        self.exploration_sets = exploration_sets

        # Extract all target variables from the causal graphical model
        self.all_target_variables = list(filter(lambda k: self.base_target_variable in k, self.graph.nodes))

        # Get the interventional grids
        self.interventional_grids = get_interventional_grids(
            self.exploration_sets, intervention_domain, size_intervention_grid=100
        )

        # Objective function params
        self.bo_model = {t: {es: None for es in self.exploration_sets} for t in range(self.total_timesteps)}

        self.ground_truth = ground_truth

        # Number of points where to evaluate acquisition function
        self.num_anchor_points = num_anchor_points

        # Assigned during optimisation
        self.mean_function = deepcopy(self.bo_model)
        self.variance_function = deepcopy(self.bo_model)

        # Store the dict for mean and var values computed in the acquisition function
        self.mean_dict_store = {t: {es: {} for es in self.exploration_sets} for t in range(self.total_timesteps)}
        self.var_dict_store = deepcopy(self.mean_dict_store)

        # For logging
        self.sequence_of_interventions_during_trials = [[] for _ in range(self.total_timesteps)]
        # Initial optimal solutions
        if interventional_samples:
            # Provide initial interventional data
            (
                initial_optimal_sequential_intervention_sets,
                initial_optimal_target_values,
                initial_optimal_sequential_intervention_levels,
                self.interventional_data_x,
                self.interventional_data_y,
            ) = initialise_DCBO_parameters_and_objects_filtering(
                self.exploration_sets,
                interventional_samples,
                self.base_target_variable,
                self.total_timesteps,
                self.task,
                index_name=0,
                nr_interventions=None,  # There are interventions, we just don't sub-sample them.
            )
        else:
            # No initial interventional data
            initial_optimal_sequential_intervention_sets = [choice(self.exploration_sets)] + (T - 1) * [None]
            initial_optimal_target_values = T * [None]
            initial_optimal_sequential_intervention_levels = T * [None]
            self.interventional_data_x = deepcopy(self.bo_model)
            self.interventional_data_y = deepcopy(self.bo_model)

        assert (
            len(initial_optimal_sequential_intervention_levels)
            == len(initial_optimal_target_values)
            == len(initial_optimal_sequential_intervention_levels)
            == self.total_timesteps
        )

        # Dict indexed by the global exploration sets, stores the best
        self.outcome_values = initialise_global_outcome_dict_new(
            self.total_timesteps, initial_optimal_target_values, self.blank_val
        )
        self.optimal_outcome_values_during_trials = [[] for _ in range(self.total_timesteps)]

        self.optimal_intervention_levels = initialise_optimal_intervention_level_list(
            self.total_timesteps,
            self.exploration_sets,
            initial_optimal_sequential_intervention_sets,
            initial_optimal_sequential_intervention_levels,
            number_of_trials,
        )
        self.best_initial_es = initial_optimal_sequential_intervention_sets[0]  # 0 indexes the first time-step

        # Target functions for Bayesian optimisation - ground truth
        self.target_functions = {t: {es: None for es in self.exploration_sets} for t in range(T)}

        for temporal_index in range(T):
            for es in self.exploration_sets:
                self.target_functions[temporal_index][es] = evaluate_target_function(
                    self.true_initial_sem, self.true_sem, self.graph, es, self.observational_samples.keys(), T,
                )

        # Parameter space for optimisation
        self.intervention_exploration_domain = create_intervention_exploration_domain(
            self.exploration_sets, intervention_domain,
        )

        # Optimisation specific parameters to initialise
        self.trial_type = [[] for _ in range(self.total_timesteps)]
        self.cost_functions = define_costs(self.manipulative_variables, self.base_target_variable, cost_type)
        self.per_trial_cost = [[] for _ in range(self.total_timesteps)]
        self.optimal_intervention_sets = [None for _ in range(self.total_timesteps)]

        # Convert observational samples to dict of temporal lists.
        # We do this because at each time-index we may have a different number of samples.
        # Because of this, samples need to be stored one lists per time-step.
        self.observational_samples = convert_to_dict_of_temporal_lists(self.observational_samples)
        # Acquisition function specifics
        self.y_acquired = {es: None for es in self.exploration_sets}
        self.corresponding_x = deepcopy(self.y_acquired)
        if self.estimate_sem:
            self.assigned_blanket_hat = deepcopy(self.optimal_blanket)
