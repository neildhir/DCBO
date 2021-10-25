"""
Main CBO class.
"""
from typing import Callable

import numpy as np
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from GPy.core import Mapping
from GPy.core.parameterization import priors
from GPy.kern.src.rbf import RBF
from GPy.models import GPRegression
from numpy import squeeze
from numpy.core.multiarray import ndarray
from src.bases.root import Root
from src.bayes_opt.causal_kernels import CausalRBF
from src.bayes_opt.cost_functions import total_intervention_cost
from src.bayes_opt.intervention_computations import evaluate_acquisition_function
from src.utils.gp_utils import fit_gp, update_sufficient_statistics, update_sufficient_statistics_hat
from src.utils.sequential_causal_functions import sequentially_sample_model
from src.utils.utilities import (
    assign_blanket,
    check_blanket,
    convert_to_dict_of_temporal_lists,
    make_column_shape_2D,
    standard_mean_function,
    zero_variance_adjustment,
)
from tqdm import trange


class CBO(Root):
    def __init__(
        self,
        G: str,
        sem: classmethod,
        make_sem_estimator: Callable,
        observation_samples: dict,
        intervention_domain: dict,
        intervention_samples: dict,
        exploration_sets: dict,
        number_of_trials: int,
        base_target_variable: str,
        ground_truth: list = None,
        estimate_sem: bool = True,
        task: str = "min",
        n_restart: int = 1,
        cost_type: int = 1,
        use_mc: bool = False,
        debug_mode: bool = False,
        online: bool = False,
        concat: bool = False,
        optimal_assigned_blankets: dict = None,
        n_obs_t: int = None,
        hp_i_prior: bool = True,
        num_anchor_points=100,
        seed: int = 1,
        sample_anchor_points: bool = False,
        seed_anchor_points=None,
        args_sem=None,
        manipulative_variables: list = None,
        change_points: list = None,
    ):
        args = {
            "G": G,
            "sem": sem,
            "make_sem_estimator": make_sem_estimator,
            "observation_samples": observation_samples,
            "intervention_domain": intervention_domain,
            "intervention_samples": intervention_samples,
            "exploration_sets": exploration_sets,
            "estimate_sem": estimate_sem,
            "base_target_variable": base_target_variable,
            "task": task,
            "cost_type": cost_type,
            "use_mc": use_mc,
            "number_of_trials": number_of_trials,
            "ground_truth": ground_truth,
            "n_restart": n_restart,
            "debug_mode": debug_mode,
            "online": online,
            "num_anchor_points": num_anchor_points,
            "args_sem": args_sem,
            "manipulative_variables": manipulative_variables,
            "change_points": change_points,
        }
        super().__init__(**args)

        self.concat = concat
        self.optimal_assigned_blankets = optimal_assigned_blankets
        self.n_obs_t = n_obs_t
        self.hp_i_prior = hp_i_prior
        self.seed = seed
        self.sample_anchor_points = sample_anchor_points
        self.seed_anchor_points = seed_anchor_points
        # Convert observational samples to dict of temporal lists.
        # We do this because at each time-index we may have a different number of samples.
        # Because of this, samples need to be stored one lists per time-step.
        self.observational_samples = convert_to_dict_of_temporal_lists(self.observational_samples)

    def run_optimization(self):

        if self.debug_mode:
            assert self.ground_truth is not None, "Provide ground truth to plot surrogate models"

        # Walk through the graph, from left to right, i.e. the temporal dimension
        for temporal_index in trange(self.T, desc="Time index"):

            if self.debug_mode:
                print("\n\t\t\t\t###########################")
                print("\t\t\t\t# Time: {}".format(temporal_index))
                print("\t\t\t\t###########################\n")

            # Evaluate each target
            target = self.all_target_variables[temporal_index]
            # Check which current target we are dealing with, and in initial_sem sequence where we are in time
            _, target_temporal_index = target.split("_")
            assert int(target_temporal_index) == temporal_index
            best_es = self.best_initial_es

            # Updating the observational and interventional data based on the online and concat options.
            self._update_observational_data(temporal_index=temporal_index)
            self._update_interventional_data(temporal_index=temporal_index)

            if temporal_index > 0 and (self.online or isinstance(self.n_obs_t, list)):
                self._update_sem_emit_fncs(temporal_index)

            # Get blanket to compute y_new
            assigned_blanket = self._get_assigned_blanket(temporal_index)

            for it in range(self.number_of_trials):

                if self.debug_mode:
                    print("\n\n>>>")
                    print("Iteration:", it)
                    print("<<<\n\n")

                if it == 0:
                    # >>>OBSERVE<<<

                    self.trial_type[temporal_index].append("o")  # For 'o'bserve

                    if self.estimate_sem:
                        # Check which current target we are dealing with
                        _, target_temporal_index = target.split("_")
                        assert int(target_temporal_index) == temporal_index
                        sem_hat = self.make_sem_hat(
                            summary_graph_node_parents=self.summary_graph_node_parents,
                            independent_causes=self.independent_causes,
                            emission_functions=self.sem_emit_fncs,
                        )
                    else:
                        sem_hat = None

                    # Create mean functions and var functions given the observational data. This updates the prior.
                    self._update_sufficient_statistics(target, temporal_index, sem_hat)
                    # Update optimisation related parameters
                    self._update_opt_params(it, temporal_index, best_es)

                else:
                    # >>>INTERVENE<<<

                    # Presently find the optimal value of Y_t
                    current_best_global_target = eval(self.task)(self.outcome_values[temporal_index])

                    # Surrogate models
                    if self.trial_type[temporal_index][-1] == "o":
                        for es in self.exploration_sets:
                            if (
                                self.interventional_data_x[temporal_index][es] is not None
                                and self.interventional_data_y[temporal_index][es] is not None
                            ):
                                self._update_bo_model(temporal_index, es)

                    # Surrogate model
                    if self.debug_mode:
                        self._plot_surrogate_model(temporal_index)

                    self.trial_type[temporal_index].append("i")  # For 'i'ntervene

                    # Compute acquisition function
                    self._evaluate_acquisition_functions(temporal_index, current_best_global_target, it)

                    # Best exploration set based on acquired target-values
                    best_es = eval("max")(self.y_acquired, key=self.y_acquired.get)
                    new_interventional_data_x = self.corresponding_x[best_es]

                    self._check_new_point(best_es, temporal_index)

                    # Compute target value for selected intervention
                    y_new = self.target_functions[temporal_index][best_es](
                        current_target=target,
                        intervention_levels=squeeze(new_interventional_data_x),
                        assigned_blanket=assigned_blanket,
                    )

                    if self.debug_mode:
                        print("Selected set:", best_es)
                        print("Intervention value:", new_interventional_data_x)
                        print("Outcome:", y_new)

                    # Update interventional data
                    self._get_updated_interventional_data(new_interventional_data_x, y_new, best_es, temporal_index)

                    # Evaluate cost of intervention
                    self.per_trial_cost[temporal_index].append(
                        total_intervention_cost(
                            best_es, self.cost_functions, self.interventional_data_x[temporal_index][best_es],
                        )
                    )

                    # Store local optimal exploration set corresponding intervention levels
                    self.outcome_values[temporal_index].append(y_new)
                    self.optimal_outcome_values_during_trials[temporal_index].append(
                        eval(self.task)(y_new, current_best_global_target)
                    )

                    # Store the intervention
                    if len(new_interventional_data_x.shape) != 2:
                        self.optimal_intervention_levels[temporal_index][best_es][it] = make_column_shape_2D(
                            new_interventional_data_x
                        )
                    else:
                        self.optimal_intervention_levels[temporal_index][best_es][it] = new_interventional_data_x

                    # Store the currently best intervention set
                    self.sequence_of_interventions_during_trials[temporal_index].append(best_es)

                    # Create BO model if it does not exist
                    self._update_bo_model(temporal_index, best_es)

                    if self.debug_mode:
                        print("########################### results of optimization ##################")
                        self._plot_surrogate_model(temporal_index)

                    if self.debug_mode:
                        print(
                            "### Optimized model: ###", best_es, self.bo_model[temporal_index][best_es].model,
                        )

            # Post optimisation assignments (post this time-index that is)
            # Index of the best value of the objective function
            best_objective_fnc_value_idx = (
                self.outcome_values[temporal_index].index(eval(self.task)(self.outcome_values[temporal_index])) - 1
            )

            # 1) Best intervention for this temporal index
            for es in self.exploration_sets:

                if isinstance(
                    self.optimal_intervention_levels[temporal_index][es][best_objective_fnc_value_idx], ndarray,
                ):
                    # Check to see that the optimal intervention is not None
                    check_val = self.optimal_intervention_levels[temporal_index][es][best_objective_fnc_value_idx]

                    assert check_val is not None, (
                        temporal_index,
                        self.optimal_intervention_sets[temporal_index],
                        best_objective_fnc_value_idx,
                        es,
                    )
                    # This is the, overall, best intervention set for this temporal index.
                    self.optimal_intervention_sets[temporal_index] = es
                    break  # There is only one so we can break here

            # 2) Blanket stores optimal values (interventions and targets) found during DCBO.
            self.optimal_blanket[self.base_target_variable][temporal_index] = eval(self.task)(
                self.outcome_values[temporal_index]
            )

            # 3) Write optimal interventions to the optimal blanket
            for i, es_member in enumerate(set(es).intersection(self.manipulative_variables)):
                self.optimal_blanket[es_member][temporal_index] = float(
                    self.optimal_intervention_levels[temporal_index][self.optimal_intervention_sets[temporal_index]][
                        best_objective_fnc_value_idx
                    ][:, i]
                )

            # 4) Finally, populate the summary blanket with info found in (1) to (3)
            assign_blanket(
                self.true_initial_sem,
                self.true_sem,
                self.assigned_blanket,
                self.optimal_intervention_sets[temporal_index],
                self.optimal_intervention_levels[temporal_index][self.optimal_intervention_sets[temporal_index]][
                    best_objective_fnc_value_idx
                ],
                target=target,
                target_value=self.optimal_blanket[self.base_target_variable][temporal_index],
                node_children=self.node_children,
            )
            check_blanket(
                self.assigned_blanket, self.base_target_variable, temporal_index, self.manipulative_variables,
            )

            # Check optimization results for the current temporal index before moving on
            self._check_optimization_results(temporal_index)

    def _evaluate_acquisition_functions(self, temporal_index, current_best_global_target, it):

        for es in self.exploration_sets:
            if (
                self.interventional_data_x[temporal_index][es] is not None
                and self.interventional_data_y[temporal_index][es] is not None
            ):
                bo_model = self.bo_model[temporal_index][es]
            else:
                bo_model = None
                if isinstance(self.n_obs_t, list) and self.n_obs_t[temporal_index] == 1:
                    self.mean_function[temporal_index][es] = standard_mean_function
                    self.variance_function[temporal_index][es] = zero_variance_adjustment

            # We evaluate this function IF there is interventional data at this time index
            if self.seed_anchor_points is None:
                seed_to_pass = None
            else:
                seed_to_pass = int(self.seed_anchor_points * (temporal_index + 1) * it)
            (self.y_acquired[es], self.corresponding_x[es],) = evaluate_acquisition_function(
                self.intervention_exploration_domain[es],
                bo_model,
                self.mean_function[temporal_index][es],
                self.variance_function[temporal_index][es],
                current_best_global_target,
                es,
                self.cost_functions,
                self.task,
                self.base_target_variable,
                dynamic=False,
                causal_prior=True,
                temporal_index=temporal_index,
                previous_variance=1.0,
                num_anchor_points=self.num_anchor_points,
                sample_anchor_points=self.sample_anchor_points,
                seed_anchor_points=seed_to_pass,
            )

    def _update_interventional_data(self, temporal_index):

        if temporal_index > 0 and self.concat:
            for var in self.interventional_data_x[0].keys():
                self.interventional_data_x[temporal_index][var] = self.interventional_data_x[temporal_index - 1][var]
                self.interventional_data_y[temporal_index][var] = self.interventional_data_y[temporal_index - 1][var]

    def _update_sem_emit_fncs(self, temporal_index: int) -> None:

        for inputs in self.sem_emit_fncs[temporal_index].keys():
            output = self.emission_pairs[inputs].split("_")[0]
            if len(inputs) > 1:
                xx = []
                for node in inputs:
                    start_node, time = node.split("_")
                    time = int(time)
                    #  Input
                    x = make_column_shape_2D(self.observational_samples[start_node][time])
                    xx.append(x)
                xx = np.hstack(xx)
                #  Output
                yy = make_column_shape_2D(self.observational_samples[output][time])
            elif len(inputs) == 1:
                start_node, time = inputs[0].split("_")
                time = int(time)
                #  Input
                xx = make_column_shape_2D(self.observational_samples[start_node][time])
                #  Output
                yy = make_column_shape_2D(self.observational_samples[output][time])
            else:
                raise ValueError("The length of the tuple is: {}".format(len(inputs)))

            assert len(xx.shape) == 2
            assert len(yy.shape) == 2
            assert xx.shape[0] == yy.shape[0]  # Column arrays

            if xx.shape[0] != yy.shape[0]:
                min_rows = np.min((xx.shape[0], yy.shape[0]))
                xx = xx[: int(min_rows)]
                yy = yy[: int(min_rows)]

            if not self.sem_emit_fncs[temporal_index][inputs]:
                self.sem_emit_fncs[temporal_index][inputs] = fit_gp(x=xx, y=yy)
            else:
                # Update in-place
                self.sem_emit_fncs[temporal_index][inputs].set_XY(X=xx, Y=yy)
                self.sem_emit_fncs[temporal_index][inputs].optimize()

    def _update_sufficient_statistics(self, target: str, temporal_index: int, updated_sem=None) -> None:

        # Check which current target we are dealing with, and in consequence where we are in time
        target_variable, target_temporal_index = target.split("_")
        assert int(target_temporal_index) == temporal_index
        blanket = self.empty_intervention_blanket

        for es in self.exploration_sets:
            #  Use estimates of sem
            if self.estimate_sem:
                (
                    self.mean_function[temporal_index][es],
                    self.variance_function[temporal_index][es],
                ) = update_sufficient_statistics_hat(
                    temporal_index,
                    target_variable,
                    es,
                    updated_sem,
                    self.node_parents,
                    dynamic=False,
                    assigned_blanket=blanket,
                    mean_dict_store=self.mean_dict_store,
                    var_dict_store=self.var_dict_store,
                )
            # Use true sem
            else:
                # At the first time-slice we do not have any previous fixed interventions to consider.
                (
                    self.mean_function[temporal_index][es],
                    self.variance_function[temporal_index][es],
                ) = update_sufficient_statistics(
                    temporal_index,
                    es,
                    self.node_children,
                    self.true_initial_sem,
                    self.true_sem,
                    dynamic=False,
                    assigned_blanket=blanket,  # At t=0 this is a dummy variable as it has not been assigned yet.
                )

    def _update_bo_model(
        self, temporal_index: int, exploration_set: tuple, alpha: float = 2, beta: float = 0.5,
    ) -> None:

        assert self.interventional_data_x[temporal_index][exploration_set] is not None
        assert self.interventional_data_y[temporal_index][exploration_set] is not None

        input_dim = len(exploration_set)

        # if not self.bo_model[temporal_index][exploration_set]:
        # Specify mean function
        mf = Mapping(input_dim, 1)
        mf.f = self.mean_function[temporal_index][exploration_set]
        mf.update_gradients = lambda a, b: None

        # Set kernel
        causal_kernel = CausalRBF(
            input_dim=input_dim,  # Indexed before passed to this function
            variance_adjustment=self.variance_function[temporal_index][exploration_set],  # Variance function here
            lengthscale=1.0,
            variance=1.0,
            ARD=False,
        )

        if temporal_index > 0 and isinstance(self.n_obs_t, list) and self.n_obs_t[temporal_index] == 1:
            #  Set model, the standard prior
            model = GPRegression(
                X=self.interventional_data_x[temporal_index][exploration_set],
                Y=self.interventional_data_y[temporal_index][exploration_set],
                kernel=RBF(input_dim, lengthscale=1.0, variance=1.0),
                noise_var=1e-5,
            )
        else:
            #  Set model
            model = GPRegression(
                X=self.interventional_data_x[temporal_index][exploration_set],
                Y=self.interventional_data_y[temporal_index][exploration_set],
                kernel=causal_kernel,
                noise_var=1e-5,
                mean_function=mf,
            )

        if self.hp_i_prior:
            gamma = priors.Gamma(a=alpha, b=beta)  # See https://github.com/SheffieldML/GPy/issues/735
            model.kern.variance.set_prior(gamma)

        # Store model
        model.likelihood.variance.fix()

        old_seed = np.random.get_state()

        np.random.seed(self.seed)
        model.optimize()
        np.random.set_state(old_seed)

        self.bo_model[temporal_index][exploration_set] = GPyModelWrapper(model)
        self._safe_optimization(temporal_index, exploration_set)

    def _update_observational_data(self, temporal_index):
        if temporal_index > 0:
            if self.online:
                if isinstance(self.n_obs_t, list):
                    local_n_t = self.n_obs_t[temporal_index]
                else:
                    local_n_t = self.n_obs_t
                assert local_n_t is not None

                # Sample new data
                set_observational_samples = sequentially_sample_model(
                    static_sem=self.true_initial_sem,
                    dynamic_sem=self.true_sem,
                    total_timesteps=temporal_index + 1,
                    sample_count=local_n_t,
                    use_sem_estimate=False,
                    interventions=self.assigned_blanket,
                )

                # Reshape data
                set_observational_samples = convert_to_dict_of_temporal_lists(set_observational_samples)

                for var in self.observational_samples.keys():
                    self.observational_samples[var][temporal_index] = set_observational_samples[var][temporal_index]
            else:
                if isinstance(self.n_obs_t, list):
                    local_n_obs = self.n_obs_t[temporal_index]

                    n_stored_observations = len(
                        self.observational_samples[list(self.observational_samples.keys())[0]][temporal_index]
                    )

                    if self.online is False and local_n_obs != n_stored_observations:
                        # We already have the same number of observations stored
                        set_observational_samples = sequentially_sample_model(
                            static_sem=self.true_initial_sem,
                            dynamic_sem=self.true_sem,
                            total_timesteps=temporal_index + 1,
                            sample_count=local_n_obs,
                            use_sem_estimate=False,
                        )
                        # Reshape data
                        set_observational_samples = convert_to_dict_of_temporal_lists(set_observational_samples)

                        for var in self.observational_samples.keys():
                            self.observational_samples[var][temporal_index] = set_observational_samples[var][
                                temporal_index
                            ]

    def _get_assigned_blanket(self, temporal_index):
        if temporal_index > 0:
            if self.optimal_assigned_blankets is not None:
                assigned_blanket = self.optimal_assigned_blankets[temporal_index]
            else:
                assigned_blanket = self.assigned_blanket
        else:
            assigned_blanket = self.assigned_blanket
        return assigned_blanket
