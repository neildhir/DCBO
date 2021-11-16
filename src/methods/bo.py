"""
Main BO class.
"""
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from GPy.core.parameterization import priors
from GPy.kern.src.rbf import RBF
from GPy.models import GPRegression
from numpy import random, squeeze
from numpy.core.multiarray import ndarray
from src.bases.root import Root
from src.bayes_opt.cost_functions import total_intervention_cost
from src.bayes_opt.intervention_computations import evaluate_acquisition_function
from src.utils.utilities import (
    assign_blanket,
    check_blanket,
    convert_to_dict_of_temporal_lists,
    make_column_shape_2D,
    standard_mean_function,
    zero_variance_adjustment,
)
from tqdm import trange


class BO(Root):
    def __init__(
        self,
        G: str,
        sem: classmethod,
        observation_samples: dict,
        intervention_domain: dict,
        intervention_samples: dict,
        number_of_trials: int,
        base_target_variable: str,
        task: str = "min",
        exploration_sets: list = None,
        cost_type: int = 1,
        n_restart: int = 1,
        hp_i_prior: bool = True,
        debug_mode: bool = False,
        seed: int = 1,
        optimal_assigned_blankets: dict = None,
        num_anchor_points=100,
        sample_anchor_points: bool = False,
        seed_anchor_points=None,
        args_sem=None,
        manipulative_variables=None,
        change_points: list = None,
    ):
        args = {
            "G": G,
            "sem": sem,
            "observation_samples": observation_samples,
            "intervention_domain": intervention_domain,
            "intervention_samples": intervention_samples,
            "exploration_sets": exploration_sets,
            "base_target_variable": base_target_variable,
            "task": task,
            "cost_type": cost_type,
            "number_of_trials": number_of_trials,
            "n_restart": n_restart,
            "debug_mode": debug_mode,
            "num_anchor_points": num_anchor_points,
            "args_sem": args_sem,
            "manipulative_variables": manipulative_variables,
            "change_points": change_points,
        }
        super().__init__(**args)

        self.optimal_assigned_blankets = optimal_assigned_blankets
        self.sample_anchor_points = sample_anchor_points
        self.seed_anchor_points = seed_anchor_points
        self.seed = seed
        self.hp_i_prior = hp_i_prior
        # Convert observational samples to dict of temporal lists.
        # We do this because at each time-index we may have a different number of samples.
        # Because of this, samples need to be stored one lists per time-step.
        self.observational_samples = convert_to_dict_of_temporal_lists(self.observational_samples)
        # This is partiuclar to BO, why these lines override the standards in the Root class.
        for temporal_index in range(self.T):
            self.mean_function[temporal_index][self.exploration_sets[0]] = standard_mean_function
            self.variance_function[temporal_index][self.exploration_sets[0]] = zero_variance_adjustment

    def run(self):

        # Walk through the graph, from left to right, i.e. the temporal dimension
        for temporal_index in trange(self.T, desc="Time index"):

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

                self.trial_type[temporal_index].append("i")

                # Compute acquisition function given the updated BO models for the interventional data. Notice that we use current_global and the costs to compute the acquisition functions.
                self._evaluate_acquisition_functions(temporal_index, current_best_global_target, it)

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

                if self.debug_mode:
                    print("Selected set:", best_es)
                    print("Intervention value:", new_interventional_data_x)
                    print("Outcome:", y_new)

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

                if self.debug_mode:
                    print(
                        "### Optimized model: ###", self.bo_model[temporal_index][best_es].model,
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
                node_children=None,
            )
            check_blanket(
                self.assigned_blanket, self.base_target_variable, temporal_index, self.manipulative_variables,
            )

            # Check optimization results for the current temporal index before moving on
            self._check_optimization_results(temporal_index)

    def _get_assigned_blanket(self, temporal_index):
        if temporal_index > 0:
            if self.optimal_assigned_blankets is not None:
                assigned_blanket = self.optimal_assigned_blankets[temporal_index]
            else:
                assigned_blanket = self.assigned_blanket
        else:
            assigned_blanket = self.assigned_blanket
        return assigned_blanket

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
                dynamic=False,
                causal_prior=False,
                temporal_index=temporal_index,
                previous_variance=1.0,
                num_anchor_points=self.num_anchor_points,
                sample_anchor_points=self.sample_anchor_points,
                seed_anchor_points=seed_to_pass,
            )

    def _update_bo_model(
        self, temporal_index: int, exploration_set: tuple, alpha: float = 2, beta: float = 0.5,
    ) -> None:

        assert self.interventional_data_x[temporal_index][exploration_set] is not None
        assert self.interventional_data_y[temporal_index][exploration_set] is not None

        input_dim = len(exploration_set)
        if not self.bo_model[temporal_index][exploration_set]:
            x = self.interventional_data_x[temporal_index][exploration_set]
            y = self.interventional_data_y[temporal_index][exploration_set]

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
                X=self.interventional_data_x[temporal_index][exploration_set],
                Y=self.interventional_data_y[temporal_index][exploration_set],
            )
            old_seed = random.get_state()

            random.seed(self.seed)
            self.bo_model[temporal_index][exploration_set].optimize()

            random.set_state(old_seed)
