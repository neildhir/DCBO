from typing import Callable

import numpy as np
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from GPy.core import Mapping
from GPy.core.parameterization import priors
from GPy.kern.src.rbf import RBF
from GPy.models import GPRegression
from src.bases.root import Root
from src.bayes_opt.causal_kernels import CausalRBF
from src.bayes_opt.intervention_computations import evaluate_acquisition_function
from src.utils.sem_utils.sem_estimate import fit_arcs
from src.utils.utilities import (
    convert_to_dict_of_temporal_lists,
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
        exploration_sets: list,
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
        # Fit Gaussian processes to emissions
        self.sem_emit_fncs = fit_arcs(self.G, self.observational_samples, emissions=True)
        # Convert observational samples to dict of temporal lists. We do this because at each time-index we may have a different number of samples. Because of this, samples need to be stored one lists per time-step.
        self.observational_samples = convert_to_dict_of_temporal_lists(self.observational_samples)

    def run(self):

        if self.debug_mode:
            assert self.ground_truth is not None, "Provide ground truth to plot surrogate models"

        # Walk through the graph, from left to right, i.e. the temporal dimension
        for temporal_index in trange(self.T, desc="Time index"):

            # Evaluate each target
            target = self.all_target_variables[temporal_index]
            # Check which current target we are dealing with, and in initial_sem sequence where we are in time
            _, target_temporal_index = target.split("_")
            assert int(target_temporal_index) == temporal_index
            best_es = self.best_initial_es

            # Updating the observational and interventional data based on the online and concat options.
            self._update_observational_data(temporal_index=temporal_index)
            self._update_interventional_data(temporal_index=temporal_index)

            #  Online run option
            if temporal_index > 0 and (self.online or isinstance(self.n_obs_t, list)):
                self._update_sem_emit_fncs(temporal_index)

            # Get blanket to compute y_new
            assigned_blanket = self._get_assigned_blanket(temporal_index)

            for it in range(self.number_of_trials):

                if it == 0:

                    self.trial_type[temporal_index].append("o")  # For 'o'bserve
                    sem_hat = self.make_sem_hat(G=self.G, emission_fncs=self.sem_emit_fncs,)

                    # Create mean functions and var functions given the observational data. This updates the prior.
                    self._update_sufficient_statistics(
                        target=target,
                        temporal_index=temporal_index,
                        dynamic=False,
                        assigned_blanket=self.empty_intervention_blanket,
                        updated_sem=sem_hat,
                    )
                    # Update optimisation related parameters
                    self._update_opt_params(it, temporal_index, best_es)

                else:

                    # Surrogate models
                    if self.trial_type[temporal_index][-1] == "o":
                        for es in self.exploration_sets:
                            if (
                                self.interventional_data_x[temporal_index][es] is not None
                                and self.interventional_data_y[temporal_index][es] is not None
                            ):
                                self._update_bo_model(temporal_index, es)

                    # This function runs the actual computation -- calls are identical for all methods
                    self._per_trial_computations(temporal_index, it, target, assigned_blanket)

            # Post optimisation assignments (post this time-index that is)
            self._post_optimisation_assignments(target, temporal_index)

    def _evaluate_acquisition_functions(self, temporal_index, current_best_global_target, it):

        # Loop over all given exploration sets
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

    def _update_sem_emit_fncs(self, t: int) -> None:

        # Loop over all emission functions in this time-slice
        for pa in self.sem_emit_fncs[t]:
            # Get relevant data for updating emission functions
            xx, yy = self._get_sem_emit_obs(t, pa)
            if xx and yy:
                # Update in-place
                self.sem_emit_fncs[t][pa].set_XY(X=xx, Y=yy)
                self.sem_emit_fncs[t][pa].optimize()

    def _update_bo_model(
        self, temporal_index: int, exploration_set: tuple, alpha: float = 2, beta: float = 0.5,
    ) -> None:

        assert self.interventional_data_x[temporal_index][exploration_set] is not None
        assert self.interventional_data_y[temporal_index][exploration_set] is not None

        input_dim = len(exploration_set)

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

        model.likelihood.variance.fix()
        old_seed = np.random.get_state()
        np.random.seed(self.seed)
        model.optimize()
        np.random.set_state(old_seed)

        self.bo_model[temporal_index][exploration_set] = GPyModelWrapper(model)
        self._safe_optimization(temporal_index, exploration_set)

    def _get_assigned_blanket(self, temporal_index):
        if temporal_index > 0:
            if self.optimal_assigned_blankets is not None:
                assigned_blanket = self.optimal_assigned_blankets[temporal_index]
            else:
                assigned_blanket = self.assigned_blanket
        else:
            assigned_blanket = self.assigned_blanket

        return assigned_blanket
