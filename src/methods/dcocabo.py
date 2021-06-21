# -*- coding: utf-8 -*-
# ==========================================
# Title:  Dynamic Causal Continuous and Categorical Bayesian Optimisation (DCoCaBO)
# File:   DCoCaBO.py
# Date:   17 June 2021
# ==========================================

from typing import Callable
from GPy.core.mapping import Mapping
from GPy.core.parameterization import priors
from GPy.models.gp_regression import GPRegression
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper

from numpy import squeeze, array
from numpy.core.multiarray import ndarray

# from causal_cocabo.methods import CoCaBO
from src.bases.dcbo_base import BaseClassDCBO
from src.bayes_opt.cost_functions import total_intervention_cost
from src.utils.gp_utils import update_sufficient_statistics, update_sufficient_statistics_hat
from src.utils.utilities import assign_blanket, assign_blanket_hat, check_blanket
from tqdm import trange
from numpy import random
from causal_cocabo.utils.ml_utils.models.additive_gp import MixtureViaSumAndProduct, CategoryOverlapKernel
from src.bayes_opt.causal_kernels import CausalRBF


# TODO: write optimal parameters with **kwargs instead, it is currently far too messy.
class DCoCaBO(BaseClassDCBO):
    def __init__(
        self,
        graph: str,
        sem: classmethod,
        make_sem_hat: Callable,
        observational_samples: dict,
        intervention_domain: dict,
        interventional_samples: dict,
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
        n_obs_t=None,
        num_anchor_points=100,
        seed: int = 1,
        sample_anchor_points: bool = False,
        seed_anchor_points=None,
        args_sem=None,
        manipulative_variables: list = None,
        change_points: list = None,
        root_instrument: bool = None,
        batch: bool = False,
    ):
        # TODO: write optimal parameters with **kwargs instead, it is currently far too messy.
        super(DCoCaBO).__init__(
            graph,
            sem,
            make_sem_hat,
            observational_samples,
            intervention_domain,
            interventional_samples,
            exploration_sets,
            estimate_sem,
            base_target_variable,
            task,
            cost_type,
            number_of_trials,
            ground_truth,
            n_restart,
            use_mc,
            debug_mode,
            online,
            num_anchor_points,
            args_sem,
            manipulative_variables,
            change_points,
            root_instrument,
        )

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
        self.batch = batch
        self.name = "DCoCaBO"

    def run_optimization(self):
        """
        Pseudo-algo
        for t in {0,...,T}:
            Best ES at t <--- Causal CoCaBO(D^O, D^I, t)
        Returns: sequence of optimal (discrete and continuous) interventions at each time-step t.
        """

        if self.debug_mode is True:
            assert self.ground_truth is not None, "Provide ground truth to plot"

        # Walk through the graph, from left to right, i.e. the temporal dimension
        # TODO: replace with counter with tdqm proper
        for temporal_index in trange(self.total_timesteps, desc="Time index"):

            if self.debug_mode:
                print("\n# Time-step: {}\n".format(temporal_index))

            # Evaluate each target
            target = self.all_target_variables[temporal_index]

            # Check which current target we are dealing with, and in initial_sem sequence where we are in time
            _, target_temporal_index = target.split("_")
            assert int(target_temporal_index) == temporal_index
            optimal_exploration_set = self.best_initial_es

            # Updating the observational based on the online option; i.e. DCoCaBO is collecting data in real-time which it will use to make an optimal causal decision a this time-index.
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

            # Refit the functions if we have added extra data with use_di or we are online
            # therefore we have not fitted the data yet
            if temporal_index > 0 and (self.use_di or self.online or isinstance(self.n_obs_t, list)):
                if isinstance(self.n_obs_t, list) and self.n_obs_t[temporal_index] == 1:
                    self._update_sem_functions(temporal_index, temporal_index_data=temporal_index - 1)
                else:
                    self._update_sem_functions(temporal_index)

            # Get blanket to compute y_new
            assigned_blanket = self._get_assigned_blanket(temporal_index)

            # XXX: this is where the old iteration loop for BO used to start

            # Check which current target we are dealing with
            _, target_temporal_index = target.split("_")
            assert int(target_temporal_index) == temporal_index

            updated_sem_hat = self.make_sem_hat(
                variables=self.sem_variables,
                root_instrument=self.root_instrument,
                emission_functions=self.sem_emit_fncs,
                transition_functions=self.sem_trans_fncs,
            )
            self.static_sem = updated_sem_hat().static(moment=0)  # for t = 0
            self.sem = updated_sem_hat().dynamic(moment=0)  # for t > 0

            # Set mean functions and var functions given the observational data. This updates the prior.
            self._update_sufficient_statistics(target, temporal_index, updated_sem_hat, assigned_blanket)

            # Presently find the optimal value of Y_t
            best_outcome_value = eval(self.task)(self.outcome_values[temporal_index])

            if self.debug_mode:
                self._plot_surrogate_model(temporal_index)

            # Build the surrogate model if interventional data exists (which is not true for the first iteration)
            for es in self.exploration_sets:
                if (
                    self.interventional_data_x[temporal_index][es] is not None
                    and self.interventional_data_y[temporal_index][es] is not None
                ):
                    # Has interventional data with which to build BO model, collected using manual EI acq. func.
                    self._update_bo_model(temporal_index, es)

            # TODO: this needs to be run _one_ with the manual EI as there is likely no intervention data, and once that is done we can use the normal BO model in the acquisition function.
            # >>>>>>>>
            # XXX: Causal CoCaBO is called here
            self._evaluate_acquisition_functions()
            # <<<<<<<<

            # Best exploration set based on acquired target-values
            optimal_exploration_set = eval("max")(self.y_acquired, key=self.y_acquired.get)
            new_interventional_data_x = self.corresponding_x[optimal_exploration_set]

            self._check_new_point(optimal_exploration_set, temporal_index)

            # Compute target value for selected intervention
            y_new = self.target_functions[temporal_index][optimal_exploration_set](
                current_target=target,
                intervention_levels=squeeze(new_interventional_data_x),
                assigned_blanket=assigned_blanket,
            )

            if self.debug_mode:
                print("Selected set:", optimal_exploration_set)
                print("Intervention value:", new_interventional_data_x)
                print("Outcome:", y_new)

            # Update interventional data
            self._get_updated_interventional_data(
                new_interventional_data_x, y_new, optimal_exploration_set, temporal_index
            )

            # Evaluate cost of intervention
            self.per_trial_cost[temporal_index].append(
                total_intervention_cost(
                    optimal_exploration_set,
                    self.cost_functions,
                    self.interventional_data_x[temporal_index][optimal_exploration_set],
                )
            )

            # Store local optimal exploration set corresponding intervention levels
            self.outcome_values[temporal_index].append(y_new)

            self.optimal_outcome_values_during_trials[temporal_index].append(eval(self.task)(y_new, best_outcome_value))

            #  Store the currently best intervention set
            self.sequence_of_interventions_during_trials[temporal_index].append(optimal_exploration_set)

            # Update model and optimize given the collected intervention point
            self._build_BO_model(temporal_index, optimal_exploration_set)

            if self.debug_mode:
                print("###results of optimization ###")
                self._plot_surrogate_model(temporal_index)
                print(
                    "### Optimized model: ###",
                    optimal_exploration_set,
                    self.bo_model[temporal_index][optimal_exploration_set].model,
                )

            # Post optimisation assignments (post this time-index that is)
            # Index of the best value of the objective function
            best_objective_fnc_value_idx = (
                self.outcome_values[temporal_index].index(eval(self.task)(self.outcome_values[temporal_index])) - 1
            )

            # TODO: many of the below steps are not needed if CoCaBO handles exploration-set selection internally -- e.g. (1) can most likely be dropped altogether.

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
            # TODO: move blanket operations to a separate method, there is too much happening here
            assign_blanket_hat(
                self.assigned_blanket_hat,
                self.optimal_intervention_sets[temporal_index],  # Exploration set
                self.optimal_intervention_levels[temporal_index][self.optimal_intervention_sets[temporal_index]][
                    best_objective_fnc_value_idx
                ],  # Intervention level
                target=target,
                target_value=self.optimal_blanket[self.base_target_variable][temporal_index],
            )
            check_blanket(
                self.assigned_blanket_hat, self.base_target_variable, temporal_index, self.manipulative_variables,
            )
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

    def _update_sufficient_statistics(
        self, target: str, temporal_index: int, updated_sem, assigned_blanket: dict
    ) -> None:
        # Check which current target we are dealing with, and in consequence where we are in time
        target_variable, target_temporal_index = target.split("_")
        assert int(target_temporal_index) == temporal_index

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
                    dynamic=True,
                    assigned_blanket=assigned_blanket,
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
                    dynamic=True,
                    assigned_blanket=self.assigned_blanket,
                )

    def _evaluate_acquisition_functions(self, temporal_index):

        for es in self.exploration_sets:

            # TODO: re-write this whole block, it is too messy.

            if (
                self.interventional_data_x[temporal_index][es] is not None
                and self.interventional_data_y[temporal_index][es] is not None
            ):
                # Use Causal Expected Improvement for acquisition function

                bo_model = self.bo_model[temporal_index][es]
                previous_variance = 1.0
            else:
                # Use Manual Causal Expected Improvement for acquisition function

                bo_model = None
                if temporal_index > 0 and self.transfer_hp_i and self.bo_model[temporal_index - 1][es] is not None:
                    previous_variance = self.bo_model[temporal_index - 1][es].model.kern.variance[0]
                else:
                    previous_variance = 1.0

            # Run causal CoCaBO Algorithm for len(self.exploration_sets) times for this temporal index.
            if self.batch:
                """
                Things that need to be created outside of CoCaBO, i.e. come from DCBO:
                1. The objective function (which in our case will be pass the mean and variance function)
                2. The bounds (interventional domain to causal people)
                3. We'll need to be specific w.r.t. which acquisition function type we are giving it
                4. Depending on which exploration we are considering, the number of categories therein will change
                5. Depending on which exploration we are considering, the kernel mix ratio will change to reflect the number of discrete variables present in the exploration set.
                """
                bounds, categories, kernel_mix = get_CoCaBO_inputs()
                """
                Kernel mix settings:
                - If ES is of mixed type: 0 < km < 1
                - If ES is of discrete type: use k_h only
                - If ES is of contnuous type: use k_x only

                Will be set dynamically as we loop over exploration sets.
                """

                # We pass a DCBO type model to CoCaBO and let it evaluate the object for each es
                mabbo = CoCaBO(objfn=f, initN=initN, bounds=bounds, acq_type="LCB", C=categories, kernel_mix=kernel_mix)

            else:
                # batch CoCaBO
                mabbo = BatchCoCaBO(
                    objfn=f,
                    initN=initN,
                    bounds=bounds,
                    acq_type="LCB",
                    C=categories,
                    kernel_mix=kernel_mix,
                    batch_size=batch,
                )
            mabbo.runTrials(trials, budget, saving_path)

    def _make_causal_mixed_numerical_type_kernel(self, temporal_index, exploration_set):
        # This method originally lived inside the CoCaBO class, but _this_ method is very different and hence to allow more control it is moved to the main body.

        # TODO: set these properly
        nDim = len(self.bounds)
        continuous_dims = list(range(len(self.C_list), nDim))
        categorical_dims = list(range(len(self.C_list)))

        # create surrogate
        if self.ARD:
            hp_bounds = array(
                [*[[1e-4, 3]] * len(continuous_dims), [1e-6, 1],]  # cont lengthscale  # likelihood variance
            )
        else:
            hp_bounds = array([[1e-4, 3], [1e-6, 1],])  # cont lengthscale  # likelihood variance

        fix_mix_in_this_iter, mix_value, hp_bounds = self.get_mix(hp_bounds)

        # Categorical part
        # TODO: what to do here when data is nominal, when data is ordinal?
        categorical_kernel = CategoryOverlapKernel(len(categorical_dims), active_dims=categorical_dims)

        # Continous part
        # continuous_kernel = GPy.kern.Matern52(
        #     len(continuous_dims), lengthscale=self.default_cont_lengthscale, active_dims=continuous_dims, ARD=self.ARD
        # )
        # TODO: Probably need to swap these for the ones from cocabo
        variance, lengthscale = self._get_interventional_hp(
            temporal_index, exploration_set, prior_var, prior_lengthscale
        )

        continuous_kernel = CausalRBF(
            input_dim=len(continuous_dims),
            variance_adjustment=self.variance_function[temporal_index][exploration_set],  # Variance function here
            lengthscale=lengthscale,
            variance=variance,
            ARD=self.ARD,
        )

        mixture_kernel = MixtureViaSumAndProduct(
            len(categorical_dims) + len(continuous_dims),
            categorical_kernel,
            continuous_kernel,
            mix=mix_value,
            fix_inner_variances=True,
            fix_mix=fix_mix_in_this_iter,
        )
        return mixture_kernel, hp_bounds

    def _build_BO_model(
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
            mf = Mapping(input_dim, 1)  #  Many-to-one mapping
            mf.f = self.mean_function[temporal_index][exploration_set]
            mf.update_gradients = lambda a, b: None

            # Set kernel
            causal_kernel = self._make_causal_mixed_numerical_type_kernel(temporal_index, exploration_set)

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

            old_seed = random.get_state()

            random.seed(self.seed)
            model.optimize()

            self.bo_model[temporal_index][exploration_set] = GPyModelWrapper(model)

            random.set_state(old_seed)

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

        self._safe_optimization(temporal_index, exploration_set)
