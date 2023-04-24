from typing import Tuple

import numpy as np
from emukit.core.parameter_space import ParameterSpace
from numpy import argmax, argmin, ndarray
from dcbo.bayes_opt.causal_acquisition_functions import CausalExpectedImprovement, ManualCausalExpectedImprovement
from dcbo.bayes_opt.cost_functions import COST
from dcbo.utils.sequential_intervention_functions import create_n_dimensional_intervention_grid
from dcbo.utils.utilities import make_column_shape_2D


def numerical_optimization(acquisition, inputs: ndarray, task: str, exploration_set,) -> ndarray:

    # Finds the new best point by evaluating the function in a set of given inputs
    _, D = inputs.shape

    improvements = acquisition.evaluate(inputs)

    if task == "min":
        idx = argmax(improvements)
    else:
        idx = argmin(improvements)

    # Get point with best improvement, the x new should be taken from the inputs
    x_new = inputs[idx]

    # Reshape point
    if len(x_new.shape) == 1 and len(exploration_set) == 1:
        x_new = make_column_shape_2D(x_new)
    elif len(exploration_set) > 1 and len(x_new.shape) == 1:
        x_new = x_new.reshape(1, -1)
    else:
        raise ValueError("The new point is not an array. Or something else fishy is going on.")

    # TODO: consider removing
    if x_new.shape[0] == D:
        # The function make_column_shape_2D might convert a (D, ) array in a (D,1) array that needs to be reshaped
        x_new = np.transpose(x_new)

    assert x_new.shape[1] == inputs.shape[1], "New point has a wrong dimension"

    return x_new


def evaluate_acquisition_function(
    parameter_intervention_domain: ndarray,
    bo_model,
    mean_function,
    variance_function,
    optimal_target_value_at_current_time: float,
    exploration_set: tuple,
    cost_functions,
    task: str,
    base_target: str,
    dynamic: bool,
    causal_prior: bool,
    temporal_index: int,
    previous_variance: float = 1.0,
    num_anchor_points: int = 100,
    sample_anchor_points: bool = False,
    seed_anchor_points=None,
) -> Tuple[ndarray, ndarray]:

    assert isinstance(parameter_intervention_domain, ParameterSpace)
    dim = parameter_intervention_domain.dimensionality
    assert dim == len(exploration_set)

    cost_of_acquisition = COST(cost_functions, exploration_set, base_target)

    if bo_model:
        acquisition = (
            CausalExpectedImprovement(
                optimal_target_value_at_current_time, task, dynamic, causal_prior, temporal_index, bo_model,
            )
            / cost_of_acquisition
        )
    else:
        acquisition = (
            ManualCausalExpectedImprovement(
                optimal_target_value_at_current_time, task, mean_function, variance_function, previous_variance,
            )
            / cost_of_acquisition
        )

    if dim > 1:
        num_anchor_points = int(np.sqrt(num_anchor_points))

    if sample_anchor_points:
        # This is to ensure the points are different every time we call the function
        if seed_anchor_points is not None:
            np.random.seed(seed_anchor_points)
        else:
            np.random.seed()

        sampled_points = parameter_intervention_domain.sample_uniform(point_count=num_anchor_points)
    else:
        limits = [list(tup) for tup in parameter_intervention_domain.get_bounds()]
        sampled_points = create_n_dimensional_intervention_grid(limits=limits, size_intervention_grid=num_anchor_points)

    if causal_prior is False and dynamic:
        # ABO
        sampled_points = np.hstack((sampled_points, np.repeat(temporal_index, sampled_points.shape[0])[:, np.newaxis],))

    x_new = numerical_optimization(acquisition, sampled_points, task, exploration_set)
    y_acquisition = acquisition.evaluate(x_new)
    return y_acquisition, x_new
