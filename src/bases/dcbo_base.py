from copy import deepcopy

import numpy as np
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from GPy.core import Mapping
from GPy.core.parameterization import priors
from GPy.models import GPRegression
from src.bayes_opt.causal_kernels import CausalRBF
from src.bayes_opt.intervention_computations import evaluate_acquisition_function
from src.utils.gp_utils import fit_gp, sequential_sample_from_SEM_hat
from src.utils.sem_utils.emissions import fit_sem_emit_fncs
from src.utils.sem_utils.transitions import fit_sem_trans_fncs
from src.utils.sequential_intervention_functions import make_sequential_intervention_dictionary

from .root import Root


class BaseClassDCBO(Root):
    def __init__(
        self,
        G: str,
        sem: classmethod,
        make_sem_estimator: callable,
        observation_samples: dict,
        intervention_domain: dict,
        intervention_samples: dict = None,
        exploration_sets: list = None,
        estimate_sem: bool = False,
        base_target_variable: str = "Y",
        task: str = "min",
        cost_type: int = 1,
        number_of_trials=10,
        ground_truth=None,
        n_restart: int = 1,
        use_mc: bool = False,
        debug_mode: bool = False,
        online: bool = False,
        num_anchor_points: int = 100,
        args_sem=None,
        manipulative_variables=None,
        change_points: list = None,
    ):
        root_args = {
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
        super().__init__(**root_args)
        # Fit Gaussian processes to emissions
        self.sem_emit_fncs = fit_sem_emit_fncs(self.G, self.observational_samples)
        #  The fit emission function lives in the main class of the method
        self.sem_trans_fncs = fit_sem_trans_fncs(self.G, self.observational_samples)

    def _update_sem_fncs(self, temporal_index: int, temporal_index_data: int = None) -> None:
        """
        Function to update transition and emission functions for the online inference case.

        Parameters
        ----------
        temporal_index : int
            Current temporal index of the run method.
        temporal_index_data : int, optional
            [description], by default None
        """

        if temporal_index_data is None:
            temporal_index_data = temporal_index

        #  Update
        self._update_sem_emit_fncs(temporal_index, t_index_data=temporal_index_data)
        self._update_sem_trans_fncs(temporal_index, t_index_data=temporal_index_data)

    def _update_sem_emit_fncs(self, t: int, t_index_data: int = None) -> None:

        for pa in self.sem_emit_fncs[t]:
            # Get relevant data for updating emission functions
            xx, yy = self._get_sem_emit_obs(t, pa, t_index_data)
            if xx and yy:
                if t_index_data == t:
                    # Online / we have data
                    if any(self.hyperparam_obs_emit.values()):
                        self.sem_emit_fncs[t][pa] = fit_gp(
                            x=xx,
                            y=yy,
                            lengthscale=self.hyperparam_obs_emit[pa][1],
                            variance=self.hyperparam_obs_emit[pa][0],
                        )
                    else:
                        # Update in-place
                        self.sem_emit_fncs[t][pa].set_XY(X=xx, Y=yy)
                        self.sem_emit_fncs[t][pa].optimize()
                else:
                    # No data for this time-step
                    assert t_index_data != t and t_index_data < t, (
                        t_index_data,
                        t,
                    )
                    temporal_index_pa = tuple(v.split("_")[0] + "_" + str(t_index_data) for v in pa)
                    assert temporal_index_pa in self.sem_emit_fncs[t_index_data], (
                        temporal_index_pa,
                        self.sem_emit_fncs[t_index_data].keys(),
                    )
                    self.sem_emit_fncs[t][pa] = self.sem_emit_fncs[t_index_data][temporal_index_pa]

    def _update_sem_trans_fncs(self, t: int, t_index_data: int = None) -> None:

        assert t > 0

        for y, pa in zip(self.sem, self.sem_trans_fncs[t]):
            # Independent vars: transition parents to var
            xx = np.hstack([self.observational_samples[v.split("_")[0]][:, t - 1].reshape(-1, 1) for v in pa])
            # Estimand
            yy = self.observational_samples[y][:, t].reshape(-1, 1)

            min_size = np.min((xx.shape[0], yy.shape[0]))  # Check rows
            xx = xx[: int(min_size), :]
            yy = yy[: int(min_size), :]

            if t_index_data == t:
                if any(self.hyperparam_obs_transf.values()):
                    self.sem_trans_fncs[pa] = fit_gp(
                        x=xx,
                        y=yy,
                        lengthscale=self.hyperparam_obs_transf[pa][1],
                        variance=self.hyperparam_obs_transf[pa][0],
                    )
                else:
                    self.sem_trans_fncs[pa].set_XY(X=xx, Y=yy)
                    self.sem_trans_fncs[pa].optimize()
            else:
                # No data for this time-step
                assert t_index_data != t and t_index_data < t, (
                    t_index_data,
                    t,
                )
                # Find the input for the past index
                temporal_index_data_pa = tuple(v.split("_")[0] + "_" + str(t_index_data - 1) for v in pa)
                assert temporal_index_data_pa in self.time_indexed_trans_fncs_pa[t_index_data], (
                    t,
                    t_index_data,
                    pa,
                    temporal_index_data_pa,
                    self.time_indexed_trans_fncs_pa[t_index_data],
                )
                # We use the previous transition function for this time-index
                self.sem_trans_fncs[pa] = self.sem_trans_fncs[temporal_index_data_pa]

    def _get_observational_hp_emissions(self, emission_functions, temporal_index):
        # XXX: this is _ONLY_ a valid option for stationary time-series.
        assert temporal_index > 0
        for inputs in emission_functions[temporal_index].keys():
            for past_inputs, inputs in zip(
                self.sem_emit_fncs[temporal_index - 1].keys(), self.sem_emit_fncs[temporal_index].keys(),
            ):
                if len(past_inputs) == len(inputs):
                    assert all([v1.split("_")[0] == v2.split("_")[0] for v1, v2 in zip(past_inputs, inputs)])
                    self.hyperparam_obs_emit[inputs] = [
                        emission_functions[temporal_index - 1][past_inputs].kern.variance[0],
                        emission_functions[temporal_index - 1][past_inputs].kern.lengthscale[0],
                    ]
                else:
                    raise ValueError("This is not a valid option for non-stationary problems.")

    def _get_observational_hp_transition(self, emission_functions):
        for inputs in emission_functions.keys():
            self.hyperparam_obs_transf[inputs] = [
                emission_functions[inputs].kern.variance[0],
                emission_functions[inputs].kern.lengthscale[0],
            ]

    def _forward_propagation(self, temporal_index):

        empty_blanket, _ = make_sequential_intervention_dictionary(self.G)
        res = []

        assert temporal_index > 0

        for t in range(temporal_index):
            for es in self.exploration_sets:
                interventional_data_es_x = self.interventional_data_x[t][es]
                interventional_data_es_y = self.interventional_data_y[t][es]

                if interventional_data_es_x is not None and interventional_data_es_y is not None:
                    for xx, yy in zip(interventional_data_es_x, interventional_data_es_y):
                        this_blanket = deepcopy(empty_blanket)

                        for i, intervention_variable in enumerate(es):
                            this_blanket[intervention_variable][t] = float(xx[i])

                        this_blanket["Y"][t] = float(yy)

                        out = sequential_sample_from_SEM_hat(
                            static_sem=self.static_sem,
                            dynamic_sem=self.sem,
                            timesteps=self.T,
                            interventions=this_blanket,
                            node_parents=self.node_parents,
                        )

                        res.append(out)

        for dic in res:
            for var in self.observational_samples.keys():
                self.observational_samples[var][temporal_index].extend([dic[var][temporal_index]])
                print(len(self.observational_samples[var][temporal_index]))

    def _get_interventional_hp(self, temporal_index, exploration_set, prior_var, prior_lengthscale):
        if temporal_index > 0 and self.transfer_hp_i:
            if self.bo_model[temporal_index][exploration_set] is None:
                if self.bo_model[temporal_index - 1][exploration_set] is not None:
                    variance = self.bo_model[temporal_index - 1][exploration_set].model.kern.variance[0]
                    lengthscale = self.bo_model[temporal_index - 1][exploration_set].model.kern.lengthscale[0]
                else:
                    variance = prior_var
                    lengthscale = prior_lengthscale

            else:
                variance = self.bo_model[temporal_index][exploration_set].model.kern.variance[0]
                lengthscale = self.bo_model[temporal_index][exploration_set].model.kern.lengthscale[0]
        else:
            if self.bo_model[temporal_index][exploration_set] is not None:
                variance = self.bo_model[temporal_index][exploration_set].model.kern.variance[0]
                lengthscale = self.bo_model[temporal_index][exploration_set].model.kern.lengthscale[0]
            else:
                variance = prior_var
                lengthscale = prior_lengthscale

        return variance, lengthscale

    def _evaluate_acquisition_functions(self, temporal_index, current_best_global_target, it):
        for es in self.exploration_sets:
            if (
                self.interventional_data_x[temporal_index][es] is not None
                and self.interventional_data_y[temporal_index][es] is not None
            ):
                bo_model = self.bo_model[temporal_index][es]
                previous_variance = 1.0
            else:
                bo_model = None
                if temporal_index > 0 and self.transfer_hp_i and self.bo_model[temporal_index - 1][es] is not None:
                    previous_variance = self.bo_model[temporal_index - 1][es].model.kern.variance[0]
                else:
                    previous_variance = 1.0
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
                causal_prior=True,
                temporal_index=temporal_index,
                previous_variance=previous_variance,
                num_anchor_points=self.num_anchor_points,
                sample_anchor_points=self.sample_anchor_points,
                seed_anchor_points=seed_to_pass,
            )

    def _update_bo_model(
        self,
        temporal_index: int,
        exploration_set: tuple,
        alpha: float = 2,
        beta: float = 0.5,
        noise_var: float = 1e-5,
        prior_var: float = 1.0,
        prior_lengthscale: float = 1.0,
    ) -> None:

        assert self.interventional_data_x[temporal_index][exploration_set] is not None
        assert self.interventional_data_y[temporal_index][exploration_set] is not None

        # Model does not exist so we create it
        if not self.bo_model[temporal_index][exploration_set]:
            input_dim = len(exploration_set)
            # Specify mean function
            mf = Mapping(input_dim, 1)
            mf.f = self.mean_function[temporal_index][exploration_set]
            mf.update_gradients = lambda a, b: None

            variance, lengthscale = self._get_interventional_hp(
                temporal_index, exploration_set, prior_var, prior_lengthscale
            )

            # Set kernel
            causal_kernel = CausalRBF(
                input_dim=input_dim,  # Indexed before passed to this function
                variance_adjustment=self.variance_function[temporal_index][exploration_set],  # Variance function here
                lengthscale=lengthscale,
                variance=variance,
                ARD=False,
            )

            #  Set model
            model = GPRegression(
                X=self.interventional_data_x[temporal_index][exploration_set],
                Y=self.interventional_data_y[temporal_index][exploration_set],
                kernel=causal_kernel,
                noise_var=noise_var,
                mean_function=mf,
            )

            # Store model
            model.likelihood.variance.fix()
            if self.hp_i_prior:
                gamma = priors.Gamma(a=alpha, b=beta)  # See https://github.com/SheffieldML/GPy/issues/735
                model.kern.variance.set_prior(gamma)

            old_seed = np.random.get_state()

            np.random.seed(self.seed)
            model.optimize()

            self.bo_model[temporal_index][exploration_set] = GPyModelWrapper(model)

            np.random.set_state(old_seed)

        else:
            # Model exists so we simply update it
            self.bo_model[temporal_index][exploration_set].set_data(
                X=self.interventional_data_x[temporal_index][exploration_set],
                Y=self.interventional_data_y[temporal_index][exploration_set],
            )
            old_seed = np.random.get_state()

            np.random.seed(self.seed)
            self.bo_model[temporal_index][exploration_set].optimize()

            np.random.set_state(old_seed)

        self._safe_optimization(temporal_index, exploration_set)

    def _get_assigned_blanket(self, temporal_index):
        if temporal_index > 0:
            if self.optimal_assigned_blankets is not None:
                assigned_blanket = self.optimal_assigned_blankets[temporal_index]
            else:
                assigned_blanket = self.assigned_blanket_hat
        else:
            assigned_blanket = self.assigned_blanket_hat

        return assigned_blanket
