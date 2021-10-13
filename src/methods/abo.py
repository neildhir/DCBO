"""
Main ABO class.
"""
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from GPy.core.parameterization import priors
from GPy.kern.src.rbf import RBF
from GPy.models import GPRegression
from numpy import arange, asarray, hstack, nan, random, repeat, squeeze
from numpy.core import vstack
from numpy.core.multiarray import ndarray
from src.bases.abo_base import BaseClassABO
from src.bayes_opt.cost_functions import total_intervention_cost
from src.bayes_opt.intervention_computations import evaluate_acquisition_function
from src.utils.utilities import assign_blanket, check_blanket, check_reshape_add_data, make_column_shape_2D
from tqdm import trange


class ABO(BaseClassABO):
    def __init__(
        self,
        graph: str,
        sem: classmethod,
        observational_samples: dict,
        intervention_domain: dict,
        interventional_samples: dict,
        number_of_trials: int,
        base_target_variable: str = "Y",
        task: str = "min",
        cost_type: int = 1,
        n_restart: int = 1,
        hp_i_prior: bool = True,
        debug_mode: bool = False,
        optimal_assigned_blankets: dict = None,
        num_anchor_points=100,
        seed: int = 1,
        sample_anchor_points: bool = False,
        seed_anchor_points=None,
        args_sem=None,
        manipulative_variables=None,
        change_points: list = None,
    ):
        super().__init__(
            graph,
            sem,
            observational_samples,
            intervention_domain,
            interventional_samples,
            base_target_variable,
            task,
            cost_type,
            number_of_trials,
            n_restart,
            debug_mode,
            num_anchor_points,
            args_sem,
            manipulative_variables,
            change_points,
        )
        self.optimal_assigned_blankets = optimal_assigned_blankets
        self.sample_anchor_points = sample_anchor_points
        self.seed_anchor_points = seed_anchor_points
        self.hp_i_prior = hp_i_prior
        self.seed = seed

    def run_optimization(self):

        # Walk through the graph, from left to right, i.e. the temporal dimension
        for temporal_index in trange(self.total_timesteps, desc="Time index"):

            if self.debug_mode:
                print("Time:", temporal_index)

            # Evaluate each target
            target = self.all_target_variables[temporal_index]
            # Check which current target we are dealing with, and in initial_sem sequence where we are in time
            _, target_temporal_index = target.split("_")
            assert int(target_temporal_index) == temporal_index
            best_es = self.best_initial_es

            # Get blanket to compute y_new
            assigned_blanket = self._get_assigned_blanket(temporal_index)

            for it in range(self.number_of_trials):

                if self.debug_mode:
                    print("\n\n>>>")
                    print("Iteration:", it)
                    print("<<<\n\n")

                # Presently find the optimal value of Y_t
                current_best_global_target = eval(self.task)(self.outcome_values[temporal_index])

                # if temporal_index == 1: print(stop)
                self.trial_type[temporal_index].append("i")

                # Compute acquisition function given the updated BO models for the interventional data.
                # Notice that we use current_global and the costs to compute the acquisition functions.
                self._evaluate_acquisition_functions(temporal_index, current_best_global_target, it)

                # Discard the time dimension
                self.corresponding_x[best_es] = self.corresponding_x[best_es][:, :-1]
                new_interventional_data_x = self.corresponding_x[best_es]

                # Best exploration set based on acquired target-values
                self._check_new_point(best_es, temporal_index)

                y_new = self.target_functions[temporal_index][best_es](
                    current_target=target,
                    intervention_levels=squeeze(new_interventional_data_x),
                    assigned_blanket=assigned_blanket,
                )

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

                self._update_bo_model(temporal_index, best_es)

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
                self.true_initial_structural_equation_model,
                self.true_structural_equation_model,
                self.assigned_blanket,
                self.optimal_intervention_sets[temporal_index],
                self.optimal_intervention_levels[temporal_index][self.optimal_intervention_sets[temporal_index]][
                    best_objective_fnc_value_idx
                ],
                target=target,
                target_value=self.optimal_blanket[self.base_target_variable][temporal_index],
                node_children=None,
            )
            check_blanket(
                self.assigned_blanket, self.base_target_variable, temporal_index, self.manipulative_variables,
            )

            # Check optimization results for the current temporal index before moving on
            self._check_optimization_results(temporal_index)

    def _check_optimization_results(self, temporal_index):
        # Check everything went well with the trials
        assert len(self.optimal_outcome_values_during_trials[temporal_index]) == self.number_of_trials, (
            len(self.optimal_outcome_values_during_trials[temporal_index]),
            self.number_of_trials,
        )
        assert len(self.per_trial_cost[temporal_index]) == self.number_of_trials, len(self.per_trial_cost)

        if temporal_index > 0:
            assert all(
                len(self.optimal_intervention_levels[temporal_index][es]) == self.number_of_trials
                for es in self.exploration_sets
            ), [len(self.optimal_intervention_levels[temporal_index][es]) for es in self.exploration_sets]

        assert self.optimal_intervention_sets[temporal_index] is not None, (
            self.optimal_intervention_sets,
            self.optimal_intervention_levels,
            temporal_index,
        )

    def _get_updated_interventional_data(self, new_interventional_data_x, y_new, best_es, temporal_index):
        data_x, data_y = check_reshape_add_data(
            self.interventional_data_x,
            self.interventional_data_y,
            new_interventional_data_x,
            y_new,
            best_es,
            temporal_index,
        )
        self.interventional_data_x[temporal_index][best_es] = data_x
        self.interventional_data_y[temporal_index][best_es] = data_y

    def _get_assigned_blanket(self, temporal_index):
        if temporal_index > 0:
            if self.optimal_assigned_blankets is not None:
                assigned_blanket = self.optimal_assigned_blankets[temporal_index]
            else:
                assigned_blanket = self.assigned_blanket
        else:
            assigned_blanket = self.assigned_blanket
        return assigned_blanket

    def _check_new_point(self, best_es, temporal_index):
        assert best_es is not None, (best_es, self.y_acquired)
        assert best_es in self.exploration_sets

        # Check that new intervention point is in the allowed intervention domain
        assert self.intervention_exploration_domain[best_es].check_points_in_domain(self.corresponding_x[best_es])[0], (
            best_es,
            temporal_index,
            self.y_acquired,
            self.corresponding_x,
        )

    def _evaluate_acquisition_functions(self, temporal_index, current_best_global_target, it):
        for es in self.exploration_sets:
            if (
                self.interventional_data_x[temporal_index][es] is not None
                and self.interventional_data_y[temporal_index][es] is not None
            ):
                bo_model = self.bo_model[temporal_index][es]
            else:
                bo_model = None

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
                dynamic=True,
                causal_prior=False,
                temporal_index=temporal_index,
                previous_variance=1.0,
                num_anchor_points=self.num_anchor_points,
                sample_anchor_points=self.sample_anchor_points,
                seed_anchor_points=seed_to_pass,
            )

    def _update_opt_params(self, it: int, temporal_index: int, best_es: tuple) -> None:

        # When observe append previous optimal values for logs
        # Outcome values at previous step
        self.outcome_values[temporal_index].append(self.outcome_values[temporal_index][-1])

        if it == 0:
            # Special case for first time index
            # Assign an outcome values that is the same as the initial value
            self.optimal_outcome_values_during_trials[temporal_index].append(self.outcome_values[temporal_index][-1])

            if self.interventional_data_x[temporal_index][best_es] is None:
                self.optimal_intervention_levels[temporal_index][best_es][it] = nan

            self.per_trial_cost[temporal_index].append(0.0)

        elif it > 0:
            # Get previous one cause we are observing thus we no need to recompute it
            self.optimal_outcome_values_during_trials[temporal_index].append(
                self.optimal_outcome_values_during_trials[temporal_index][-1]
            )
            self.optimal_intervention_levels[temporal_index][best_es][it] = self.optimal_intervention_levels[
                temporal_index
            ][best_es][it - 1]
            # The cost of observation is the same as the previous trial.
            self.per_trial_cost[temporal_index].append(self.per_trial_cost[temporal_index][-1])

    def _update_bo_model(
        self, temporal_index: int, exploration_set: tuple, alpha: float = 2, beta: float = 0.5,
    ) -> None:

        assert self.interventional_data_x[temporal_index][exploration_set] is not None
        assert self.interventional_data_y[temporal_index][exploration_set] is not None

        input_dim = len(exploration_set)
        time_indices = arange(0, temporal_index + 1)
        counts = asarray([self.interventional_data_x[t][exploration_set].shape[0] for t in range(temporal_index + 1)])
        time_vec = make_column_shape_2D(repeat(time_indices, counts))

        if temporal_index > 0:
            data_x = vstack([self.interventional_data_x[t][exploration_set] for t in range(temporal_index + 1)])
            data_y = vstack([self.interventional_data_y[t][exploration_set] for t in range(temporal_index + 1)])
        else:
            data_x = self.interventional_data_x[temporal_index][exploration_set]
            data_y = self.interventional_data_y[temporal_index][exploration_set]

        x = hstack((data_x, time_vec))
        y = data_y

        if not self.bo_model[temporal_index][exploration_set]:

            input_dim = input_dim + 1

            model = GPRegression(X=x, Y=y, kernel=RBF(input_dim, lengthscale=1.0, variance=1.0), noise_var=1e-5,)

            # Store model
            model.likelihood.variance.fix()

            if self.hp_i_prior:
                gamma = priors.Gamma(a=alpha, b=beta)  # See https://github.com/SheffieldML/GPy/issues/735
                model.kern.variance.set_prior(gamma)

            old_seed = random.get_state()

            random.seed(self.seed)
            model.optimize()
            random.set_state(old_seed)

            self.bo_model[temporal_index][exploration_set] = GPyModelWrapper(model)
        else:
            # Model exists so we simply update it
            self.bo_model[temporal_index][exploration_set].set_data(
                X=x, Y=y,
            )
            old_seed = random.get_state()

            random.seed(self.seed)
            self.bo_model[temporal_index][exploration_set].optimize()

            random.set_state(old_seed)
