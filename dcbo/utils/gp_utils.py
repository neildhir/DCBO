from copy import deepcopy
from typing import Callable, OrderedDict, Tuple

import numpy as np
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression

from .sequential_sampling import sequential_sample_from_SEM_hat, sequential_sample_from_true_SEM


def update_sufficient_statistics_hat(
    temporal_index: int,
    target_variable: str,
    exploration_set: tuple,
    sem_hat: OrderedDict,
    node_parents: Callable,
    dynamic: bool,
    assigned_blanket: dict,
    mean_dict_store: dict,
    var_dict_store: dict,
    seed: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Updates the mean and variance functions (priors) on our causal effects given the current exploration set.

    Parameters
    ----------
    temporal_index : int
        The current time index in the causal Bayesian network.
    target_variable : str
        The current target variable e.g Y_1
    exploration_set : tuple
        The current exploration set
    sem_hat : OrderedDict
        Contains our estimated SEMs
    node_parents : Callable
        Function with returns parents of the passed argument at the given time-slice
    dynamic : bool
        Tells the method to use horizontal information or not
    assigned_blanket : dict
        The assigned blanket thus far (i.e. up until the temporal index)
    mean_dict_store : dict
        Stores the updated mean function for this time index and exploration set
    var_dict_store : dict
        Stores the updated variance function for this time index and exploration set
    seed : int, optional
        The random seet, by default 1

    Returns
    -------
    Tuple
        Returns the updated mean and variance function
    """

    if dynamic:
        # This relies on the correct blanket being passed from outside this function.
        intervention_blanket = deepcopy(assigned_blanket)
        dynamic_sem_mean = sem_hat().dynamic(moment=0)
        dynamic_sem_var = sem_hat().dynamic(moment=1)
    else:
        # static: no backward dependency on past targets or interventions
        intervention_blanket = deepcopy(
            assigned_blanket
        )  # This is empty if passed correctly. This relies on the correct blanket being passed from outside this function.
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

    def mean_function_internal(x_vals, mean_dict_store) -> np.ndarray:
        samples = []
        for x in x_vals:
            # Check if it has already been computed
            if str(x) in mean_dict_store[temporal_index][exploration_set]:
                samples.append(mean_dict_store[temporal_index][exploration_set][str(x)])
            else:
                # Otherwise compute it and store it
                for intervention_variable, xx in zip(exploration_set, x):
                    intervention_blanket[intervention_variable][temporal_index] = xx

                # TODO: parallelise all sampling functions, this is much too slow [GPyTorch] -- see https://docs.gpytorch.ai/en/v1.5.0/examples/08_Advanced_Usage/Simple_Batch_Mode_GP_Regression.html#Setting-up-the-model
                sample = sequential_sample_from_SEM_hat(interventions=intervention_blanket, **kwargs1, seed=seed)
                samples.append(sample[target_variable][temporal_index])
                mean_dict_store[temporal_index][exploration_set][str(x)] = sample[target_variable][temporal_index]
        return np.vstack(samples)

    def mean_function(x_vals) -> np.ndarray:
        return mean_function_internal(x_vals, mean_dict_store)

    def variance_function_internal(x_vals, var_dict_store):
        out = []
        for x in x_vals:
            # Check if it is already computed
            if str(x) in var_dict_store[temporal_index][exploration_set]:
                out.append(var_dict_store[temporal_index][exploration_set][str(x)])
            else:
                # Otherwise compute it and store it
                for intervention_variable, xx in zip(exploration_set, x):
                    intervention_blanket[intervention_variable][temporal_index] = xx
                # TODO: parallelise all sampling functions, this is much too slow
                sample = sequential_sample_from_SEM_hat(interventions=intervention_blanket, **kwargs2, seed=seed)
                out.append(sample[target_variable][temporal_index])
                var_dict_store[temporal_index][exploration_set][str(x)] = sample[target_variable][temporal_index]
        return np.vstack(out)

    def variance_function(x_vals) -> np.ndarray:
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
) -> Tuple[np.ndarray, np.ndarray]:

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
                sample = sequential_sample_from_true_SEM(
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
                    sample = sequential_sample_from_true_SEM(
                        initial_sem, sem, temporal_index + 1, interventions=intervention_blanket,
                    )
                    out.append(sample[child_var][temporal_index])
            return np.vstack(out)

        def variance_function(x_vals):
            return np.zeros(x_vals.shape)

    return mean_function, variance_function


def fit_gp(x, y, lengthscale=1.0, variance=1.0, noise_var=1.0, ard=False, n_restart=10, seed: int = 0,) -> Callable:
    # This seed ensures that if you have the same date the optimization of the ML leads to the same hyper-parameters
    np.random.seed(seed)
    kernel = RBF(x.shape[1], ARD=ard, lengthscale=lengthscale, variance=variance)

    model = GPRegression(X=x, Y=y, kernel=kernel, noise_var=noise_var)
    model.optimize_restarts(n_restart, verbose=False, robust=True)
    return model

