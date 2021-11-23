from typing import Callable
from numpy import squeeze
from src.bases.dcbo_base import BaseClassDCBO
from src.bayes_opt.cost_functions import total_intervention_cost
from src.utils.utilities import (
    convert_to_dict_of_temporal_lists,
    make_column_shape_2D,
)
from tqdm import trange


class DCBO(BaseClassDCBO):
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
        task: str = "min",
        estimate_sem: bool = True,
        cost_type: int = 1,
        ground_truth: list = None,
        n_restart: int = 1,
        use_mc: bool = False,
        debug_mode: bool = False,
        online: bool = False,
        optimal_assigned_blankets: dict = None,
        use_di: bool = False,
        transfer_hp_o: bool = False,
        transfer_hp_i: bool = False,
        hp_i_prior: bool = True,
        n_obs_t: int = None,
        num_anchor_points=100,
        seed: int = 1,
        sample_anchor_points: bool = False,
        seed_anchor_points=None,
        args_sem=None,
        manipulative_variables: list = None,
        change_points: list = None,
    ):
        base_args = {
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
            "number_of_trials": number_of_trials,
            "ground_truth": ground_truth,
            "n_restart": n_restart,
            "use_mc": use_mc,
            "debug_mode": debug_mode,
            "online": online,
            "num_anchor_points": num_anchor_points,
            "args_sem": args_sem,
            "manipulative_variables": manipulative_variables,
            "change_points": change_points,
        }
        super().__init__(**base_args)

        self.optimal_assigned_blankets = optimal_assigned_blankets
        self.use_di = use_di
        self.transfer_hp_o = transfer_hp_o
        self.transfer_hp_i = transfer_hp_i
        self.hp_i_prior = hp_i_prior
        self.hyperparam_obs_emit = {}
        self.hyperparam_obs_transf = {}
        self.n_obs_t = n_obs_t
        self.seed = seed
        self.sample_anchor_points = sample_anchor_points
        self.seed_anchor_points = seed_anchor_points
        # Convert observational samples to dict of temporal lists. We do this because at each time-index we may have a different number of samples. Because of this, samples need to be stored one lists per time-step.
        self.observational_samples = convert_to_dict_of_temporal_lists(self.observational_samples)

    def run(self):

        if self.debug_mode is True:
            assert self.ground_truth is not None, "Provide ground truth to plot"

        # Walk through the graph, from left to right, i.e. the temporal dimension
        for temporal_index in trange(self.T, desc="Time index"):

            # Evaluate each target
            target = self.all_target_variables[temporal_index]
            # Check which current target we are dealing with, and in initial_sem sequence where we are in time
            _, target_temporal_index = target.split("_")
            assert int(target_temporal_index) == temporal_index
            best_es = self.best_initial_es

            # Updating the observational based on the online option.
            self._update_observational_data(temporal_index=temporal_index)

            # Forward propagation to generate observational data from interventional data
            # We can do this only once we have collected interventional data (t>0)
            if self.use_di and temporal_index > 0:
                self._forward_propagation(temporal_index)

            # Update hp and refit functions in sem given new obs data [only valid with stationary problems]
            if self.transfer_hp_o and temporal_index > 0:  # DCBO with transfer of obs parameters
                self._get_observational_hp_emissions(
                    emission_functions=self.sem_emit_fncs, temporal_index=temporal_index
                )
                self._get_observational_hp_transition(self.sem_trans_fncs)

            # Refit the functions if we have added extra data with use_di or we are online therefore we have not fitted the data yet
            if temporal_index > 0 and (self.use_di or self.online or isinstance(self.n_obs_t, list)):
                if isinstance(self.n_obs_t, list) and self.n_obs_t[temporal_index] == 1:
                    self._update_sem_fncs(temporal_index, temporal_index_data=temporal_index - 1)
                else:
                    self._update_sem_fncs(temporal_index)

            # Get blanket to compute y_new
            assigned_blanket = self._get_assigned_blanket(temporal_index)

            for it in range(self.number_of_trials):

                if self.debug_mode:
                    print("\n\n>>>")
                    print("Iteration:", it)
                    print("Trials undertaken so far:", self.trial_type)
                    print("<<<\n\n")

                if it == 0:

                    self.trial_type[temporal_index].append("o")  # For 'o'bserve
                    sem_hat = self.make_sem_hat(
                        G=self.G, emission_fncs=self.sem_emit_fncs, transition_fncs=self.sem_trans_fncs
                    )
                    self.static_sem = sem_hat().static(moment=0)  # for t = 0
                    self.sem = sem_hat().dynamic(moment=0)  # for t > 0

                    # New mean functions and var functions given the observational data. This updates the prior.
                    self._update_sufficient_statistics(
                        target=target,
                        temporal_index=temporal_index,
                        dynamic=True,
                        assigned_blanket=assigned_blanket,
                        updated_sem=sem_hat,
                    )

                    # Update optimisation related parameters
                    self._update_opt_params(it, temporal_index, best_es)

                    if self.debug_mode:
                        if any(len(s) == 1 for s in self.exploration_sets):
                            self._plot_conditional_distributions(
                                temporal_index, it,
                            )

                else:

                    # Presently find the optimal value of Y_t
                    current_best_global_target = eval(self.task)(self.outcome_values[temporal_index])

                    # Set surrogate models
                    if self.trial_type[temporal_index][-1] == "o":
                        # If we observed, we _create_ new BO models with updated mean functions and var functions
                        for es in self.exploration_sets:
                            if (
                                self.interventional_data_x[temporal_index][es] is not None
                                and self.interventional_data_y[temporal_index][es] is not None
                            ):
                                self._update_bo_model(temporal_index, es)
                    if self.debug_mode:
                        self._plot_surrogate_model(temporal_index)
                    self.trial_type[temporal_index].append("i")  # for intervene

                    # Compute acquisition function given the updated BO models for the interventional data. Notice that we use current_global and the costs to compute the acquisition functions.
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

                    #  Store the currently best intervention set
                    self.sequence_of_interventions_during_trials[temporal_index].append(best_es)

                    # Update model and optimize given the collected point
                    self._update_bo_model(temporal_index, best_es)
                    if self.debug_mode:
                        print(">>> Results of optimization")
                        self._plot_surrogate_model(temporal_index)
                        print(
                            "### Optimized model: ###", best_es, self.bo_model[temporal_index][best_es].model,
                        )

            # Post optimisation assignments (post this time-index that is)
            self._post_optimisation_assignments(target, temporal_index, True)
