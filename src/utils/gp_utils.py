from copy import deepcopy
import numpy as np
from GPy.core.mapping import Mapping
from GPy.core.parameterization import priors
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression
from ..bayes_opt.causal_kernels import CausalRBF
from .sequential_causal_functions import sequential_sample_from_model_hat, sequential_sample_from_model


def update_sufficient_statistics_hat(
    temporal_index: int,
    target_variable: str,
    exploration_set: tuple,
    sem_hat,
    node_parents,
    dynamic: bool,
    assigned_blanket: dict,
    mean_dict_store: dict,
    var_dict_store: dict,
    seed: int = 1,
):

    if dynamic:
        # This relies on the correct blanket being passed from outside this function.
        intervention_blanket = deepcopy(assigned_blanket)
        dynamic_sem_mean = sem_hat().dynamic(moment=0)
        dynamic_sem_var = sem_hat().dynamic(moment=1)
    else:
        # static: no backward dependency on past targets or interventions
        intervention_blanket = deepcopy(assigned_blanket)
        # This is empty if passed correctly.
        # This relies on the correct blanket being passed from outside this function.
        assert [all(intervention_blanket[key] is None for key in intervention_blanket.keys())]
        dynamic_sem_mean = None  # CBO does not have horizontal information hence gets static model every time
        dynamic_sem_var = None  # CBO does not have horizontal information hence gets static model every time

    #  Mean vars
    kwargs1 = {
        "static_sem": sem_hat().static(moment=0),  # Get the mean
        "dynamic_sem": dynamic_sem_mean,
        "node_parents": node_parents,
        "timesteps": temporal_index + 1,
    }
    #  Variance vars
    kwargs2 = {
        "static_sem": sem_hat().static(moment=1),  # Gets the variance
        "dynamic_sem": dynamic_sem_var,
        "node_parents": node_parents,
        "timesteps": temporal_index + 1,
    }

    def mean_function_internal(x_vals, mean_dict_store):
        out = []
        for x in x_vals:
            # Check if it is already computed
            if str(x) in mean_dict_store[temporal_index][exploration_set].keys():
                out.append(mean_dict_store[temporal_index][exploration_set][str(x)])
            else:
                # Otherwise compute it and store it
                for intervention_variable, xx in zip(exploration_set, x):
                    intervention_blanket[intervention_variable][temporal_index] = xx

                # TODO: parallelise all sampling functions, this is much too slow [GPyTorch]
                sample = sequential_sample_from_model_hat(interventions=intervention_blanket, **kwargs1, seed=seed)
                out.append(sample[target_variable][temporal_index])

                mean_dict_store[temporal_index][exploration_set][str(x)] = sample[target_variable][temporal_index]

        return np.vstack(out)

    def mean_function(x_vals):
        return mean_function_internal(x_vals, mean_dict_store)

    def variance_function_internal(x_vals, var_dict_store):
        out = []
        for x in x_vals:
            # Check if it is already computed
            if str(x) in var_dict_store[temporal_index][exploration_set].keys():
                out.append(var_dict_store[temporal_index][exploration_set][str(x)])
            else:
                # Otherwise compute it and store it
                for intervention_variable, xx in zip(exploration_set, x):
                    intervention_blanket[intervention_variable][temporal_index] = xx

                # TODO: parallelise all sampling functions, this is much too slow
                sample = sequential_sample_from_model_hat(interventions=intervention_blanket, **kwargs2, seed=seed)
                out.append(sample[target_variable][temporal_index])

                var_dict_store[temporal_index][exploration_set][str(x)] = sample[target_variable][temporal_index]

        return np.vstack(out)

    def variance_function(x_vals):
        return variance_function_internal(x_vals, var_dict_store)

    return mean_function, variance_function


def update_sufficient_statistics(
    temporal_index: int,
    exploration_set: tuple,
    time_slice_children: dict,
    initial_sem: dict,
    sem: dict,
    dynamic: bool,
    assigned_blanket: dict,
):

    if dynamic:
        # This relies on the correct blanket being passed from outside this function.
        intervention_blanket = deepcopy(assigned_blanket)
    else:
        # static: no backward dependency on past targets or interventions
        # This relies on the correct blanket being passed from outside this function.
        # This is empty if passed correctly.
        intervention_blanket = deepcopy(assigned_blanket)

        assert [all(intervention_blanket[key] is None for key in intervention_blanket.keys())]

    if len(exploration_set) == 1:
        #  Check which variable is currently being intervened upon
        intervention_variable = exploration_set[0]  # Input variable to SEM
        child_var = time_slice_children[intervention_variable]

        def mean_function(x_vals):
            out = []
            for x in x_vals:
                intervention_blanket[intervention_variable][temporal_index] = x
                sample = sequential_sample_from_model(
                    initial_sem, sem, temporal_index + 1, interventions=intervention_blanket,
                )
                out.append(sample[child_var][temporal_index])
            return np.vstack(out)

        def variance_function(x_vals):
            return np.zeros(x_vals.shape)

    else:

        def mean_function(x_vals):
            out = []
            for x in x_vals:
                for i, inter_var in enumerate(exploration_set):
                    intervention_blanket[inter_var][temporal_index] = x[i]
                    sample = sequential_sample_from_model(
                        initial_sem, sem, temporal_index + 1, interventions=intervention_blanket,
                    )
                    out.append(sample[child_var][temporal_index])
            return np.vstack(out)

        def variance_function(x_vals):
            return np.zeros(x_vals.shape)

    return mean_function, variance_function


def fit_gp(
    x, y, lengthscale=1.0, variance=1.0, noise_var=1.0, ard=False, n_restart=10, seed: int = 0,
):
    # This seed ensures that if you have the same date the optimization of the ML leads to the same hyper-parameters
    np.random.seed(seed)
    kernel = RBF(x.shape[1], ARD=ard, lengthscale=lengthscale, variance=variance)

    model = GPRegression(X=x, Y=y, kernel=kernel, noise_var=noise_var)
    model.optimize_restarts(n_restart, verbose=False, robust=True)
    return model


def fit_causal_gp(mean_function, variance_function, X, Y):
    input_dim = X.shape[1]
    # Specify mean function
    mf = Mapping(input_dim, 1)
    mf.f = mean_function
    mf.update_gradients = lambda a, b: None

    kernel = CausalRBF(
        input_dim=input_dim, variance_adjustment=variance_function, lengthscale=1.0, variance=1.0, ARD=False,
    )

    model = GPRegression(X=X, Y=Y, kernel=kernel, noise_var=1e-10, mean_function=mf)
    gamma = priors.Gamma(a=3, b=0.5)  # See https://github.com/SheffieldML/GPy/issues/735

    model.kern.variance.set_prior(gamma)

    model.optimize()

    return model
