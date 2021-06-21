from copy import deepcopy
from random import choice
from src.cost_functions import define_costs

from src.sequential_intervention_functions import (
    evaluate_target_function,
    get_interventional_grids,
    make_sequential_intervention_dictionary,
)
from src.utilities import (
    convert_to_dict_of_temporal_lists,
    create_intervention_exploration_domain,
    initialise_DCBO_parameters_and_objects_filtering,
    initialise_global_outcome_dict_new,
    initialise_optimal_intervention_level_list,
    standard_mean_function,
    zero_variance_adjustment,
)
from networkx.classes.multidigraph import MultiDiGraph


class BaseClassBO:
    """
    Base class for the BO system.

    BO is implemented only with standard prior.

    """

    def __init__(
        self,
        graph: str,
        sem: classmethod,
        observational_samples: dict,
        intervention_domain: dict,
        interventional_samples: dict = None,
        base_target_variable: str = "Y",
        task: str = "min",
        cost_type: int = 1,
        number_of_trials=10,
        n_restart: int = 1,
        debug_mode: bool = False,
        num_anchor_points: int = 100,
        args_sem=None,
        manipulative_variables=None,
        change_points: list = None,
    ):
        self.debug_mode = debug_mode
        if args_sem is None and change_points is None:
            true_sem = sem()
        elif args_sem and change_points is None:
            true_sem = sem(args_sem[0], args_sem[1])
        else:
            true_sem = sem(change_points.index(True))

        # These will be used in the target function evaluation
        self.true_initial_structural_equation_model = true_sem.static()  # for t = 0
        self.true_structural_equation_model = true_sem.dynamic()  # for t > 0

        self.observational_samples = observational_samples

        # Number of optimization restart for GPs
        self.n_restart = n_restart

        # Total time-steps and sample count per time-step
        N, T = observational_samples[list(observational_samples.keys())[0]].shape

        assert isinstance(graph, MultiDiGraph)
        self.graph = graph

        # XXX: assumes that we have the same initial obs count per variable. Not true for most real problems.
        self.number_of_trials = number_of_trials

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
        (
            self.optimal_blanket,
            self.total_timesteps,
        ) = make_sequential_intervention_dictionary(self.graph)

        # This is needed to compute the ground truth
        self.assigned_blanket = deepcopy(self.optimal_blanket)
        self.empty_intervention_blanket, _ = make_sequential_intervention_dictionary(self.graph)

        # Canonical manipulative variables
        if manipulative_variables is None:
            self.manipulative_variables = list(
                filter(
                    lambda k: self.base_target_variable not in k,
                    self.observational_samples.keys(),
                )
            )
        else:
            self.manipulative_variables = manipulative_variables

        self.interventional_variable_limits = intervention_domain
        assert self.manipulative_variables == list(intervention_domain.keys())
        #  that is when the only intervention is on the full set of vars
        self.exploration_sets = [tuple(self.manipulative_variables)]
        # Extract all target variables from the causal graphical model
        self.all_target_variables = list(filter(lambda k: self.base_target_variable in k, self.graph.nodes))
        # Get the interventional grids
        self.interventional_grids = get_interventional_grids(
            self.exploration_sets, intervention_domain, size_intervention_grid=100
        )
        # Objective function params
        self.bo_model = {t: {es: None for es in self.exploration_sets} for t in range(self.total_timesteps)}
        self.mean_function = deepcopy(self.bo_model)
        self.variance_function = deepcopy(self.bo_model)

        # Number of points where to evaluate acquisition function
        self.num_anchor_points = num_anchor_points

        # Only one set for BO
        assert len(self.exploration_sets) == 1

        for temporal_index in range(T):
            self.mean_function[temporal_index][self.exploration_sets[0]] = standard_mean_function
            self.variance_function[temporal_index][self.exploration_sets[0]] = zero_variance_adjustment

        # For logging
        self.sequence_of_interventions_during_trials = [[] for _ in range(self.total_timesteps)]
        # Initial optimal solutions
        if interventional_samples:
            #  Provide initial interventional data
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
                nr_interventions=None,
            )
        else:
            #  No initial interventional data
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
                    self.true_initial_structural_equation_model,
                    self.true_structural_equation_model,
                    self.graph,
                    es,
                    self.observational_samples.keys(),
                    T,
                )

        # Parameter space for optimisation
        self.intervention_exploration_domain = create_intervention_exploration_domain(
            self.exploration_sets,
            intervention_domain,
        )
        #  Optimisation specific parameters to initialise
        self.trial_type = [[] for _ in range(self.total_timesteps)]  # If we observed or intervened during the trial
        self.cost_functions = define_costs(self.manipulative_variables, self.base_target_variable, cost_type)
        self.per_trial_cost = [[] for _ in range(self.total_timesteps)]
        self.optimal_intervention_sets = [None for _ in range(self.total_timesteps)]
        # Convert observational samples to dict of temporal lists.
        self.observational_samples = convert_to_dict_of_temporal_lists(self.observational_samples)
        # Acquisition function specifics
        self.y_acquired = {es: None for es in self.exploration_sets}
        self.corresponding_x = deepcopy(self.y_acquired)
