import copy
from typing import Tuple
import matplotlib.pyplot as plt

import numpy as np
from emukit.core import ContinuousParameter, ParameterSpace
from numpy import cumsum, vstack, array, meshgrid
from src.sequential_causal_functions import sequential_sample_from_model
from numpy.core import hstack

from copy import deepcopy
import seaborn as sns
import datetime
from pandas import read_csv, DataFrame


def make_surface_plot(
    interventional_grids, causal_effects, interventional_variable_limits, variables=None, filename=None
):
    sns.set_context("paper", font_scale=1.7)
    if variables is None:
        variables = ["X", "Z"]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d", proj_type="ortho")

    X, Y = meshgrid(interventional_grids[(variables[0],)], interventional_grids[(variables[1],)])
    CE = array(causal_effects)
    Z = CE.reshape(X.shape)
    surf = ax.plot_surface(X, Y, Z, rstride=5, cstride=1, alpha=0.85, cmap="viridis", edgecolor="none")

    ax.set_xlim(interventional_variable_limits[variables[0]][0], interventional_variable_limits[variables[0]][1])
    ax.set_ylim(
        interventional_variable_limits[variables[1]][0],
        interventional_variable_limits[variables[1]][1],
    )
    # ax.set_zlim(-10, 1)

    ax.set_xlabel(r"$X$", labelpad=10)
    ax.set_ylabel(r"$Z$", labelpad=10)
    ax.set_zlabel(r"$\mathbb{E}[Y \mid do(X,Z)]$", labelpad=10)

    if filename:
        # Set reference time for save
        now = datetime.datetime.now()
        fig.savefig(
            "../figures/synthetic/intervene_XZ_" + filename + "_" + now.strftime("%Y-%m-%d-%H:%M") + ".pdf",
            bbox_inches="tight",
        )

    plt.show()


def get_mean_and_std(data, t_steps, repeats=5):
    out = {key: [] for key in data.keys()}
    for model in data.keys():
        for t in range(t_steps):
            tmp = []
            for ex in range(repeats):
                tmp.append(data[model][ex][t])
            tmp = np.vstack(tmp)
            out[model].append((tmp.mean(axis=0), tmp.std(axis=0)))
    return out


def standard_mean_function(x):
    return np.zeros_like(x)


def zero_variance_adjustment(x):
    return np.zeros_like(x)


def check_reshape_add_data(
    interventional_data_x,
    interventional_data_y,
    new_interventional_data_x,
    y_new,
    best_es,
    temporal_index,
):
    if (
        interventional_data_x[temporal_index][best_es] is not None
        and interventional_data_y[temporal_index][best_es] is not None
    ):
        assert interventional_data_x[temporal_index][best_es].shape[1] == new_interventional_data_x.shape[1]

        # Update interventional data X
        interventional_data_x[temporal_index][best_es] = vstack(
            (interventional_data_x[temporal_index][best_es], new_interventional_data_x)
        )
        # Update interventional data Y
        interventional_data_y[temporal_index][best_es] = vstack(
            (
                interventional_data_y[temporal_index][best_es],
                make_column_shape_2D(y_new),
            )
        )
    else:
        # Assign new interventional data
        if len(new_interventional_data_x.shape) == 1 and len(best_es) == 1:
            reshaped_new_interventional_data_x = make_column_shape_2D(new_interventional_data_x)
        elif len(best_es) > 1 and len(new_interventional_data_x.shape) == 1:
            reshaped_new_interventional_data_x = new_interventional_data_x.reshape(1, -1)
        elif new_interventional_data_x.shape[0] == len(best_es):  # ABO
            # TODO This might not be needed
            reshaped_new_interventional_data_x = np.transpose(new_interventional_data_x)
        else:
            reshaped_new_interventional_data_x = new_interventional_data_x

        #  Assign X and Y
        interventional_data_x[temporal_index][best_es] = reshaped_new_interventional_data_x
        interventional_data_y[temporal_index][best_es] = make_column_shape_2D(y_new)

        assert (
            interventional_data_x[temporal_index][best_es].shape[0]
            == interventional_data_y[temporal_index][best_es].shape[0]
        )

    return (
        interventional_data_x[temporal_index][best_es],
        interventional_data_y[temporal_index][best_es],
    )


def extract_relevant_data_for_plotting(results, repeats=10, T=3):
    assert isinstance(results, dict)

    per_trial_cost = {key: [] for key in results.keys()}
    optimal_outcome_values_during_trials = {key: [] for key in results.keys()}

    for i, key in enumerate(results.keys()):

        model = results[key]

        for n in range(repeats):

            exp_cost = []
            outcomes = []
            for time_index in range(T):
                # Loop over all experiments per model
                exp_cost.append(model[n].per_trial_cost[time_index])
                outcomes.append(model[n].optimal_outcome_values_during_trials[time_index])

            per_trial_cost[key].append(exp_cost)
            optimal_outcome_values_during_trials[key].append(outcomes)

    return per_trial_cost, optimal_outcome_values_during_trials


def get_rmse(m, gt, sample=False):

    if sample is False:
        rmse = np.sqrt(np.mean((gt - m) ** 2))
    else:
        raise NotImplementedError("Not implemented for samples")

    return rmse


def gestatic_semstd(data, t_steps, repeats=5):
    out = {key: [] for key in data.keys()}
    for model in data.keys():
        for t in range(t_steps):
            tmp = []
            for ex in range(repeats):
                tmp.append(data[model][ex][t])
            tmp = np.vstack(tmp)
            out[model].append((tmp.mean(axis=0), tmp.std(axis=0)))
    return out


def get_cumulative_cost_mean_and_std(data, t_steps, repeats=5):
    out = {key: [] for key in data.keys()}

    for model in data.keys():
        for t in range(t_steps):
            tmp = []
            for ex in range(repeats):
                tmp.append(data[model][ex][t])
            tmp = np.vstack(tmp)
            # Calculate the cumulative sum here
            out[model].append(cumsum(tmp.mean(axis=0)))
    return out


def get_monte_carlo_expectation(intervention_samples):
    assert isinstance(intervention_samples, dict)
    new = {k: None for k in intervention_samples.keys()}
    for es in new.keys():
        new[es] = intervention_samples[es].mean(axis=0)

    # Returns the expected value of the intervention via MC sampling
    return new


def create_intervention_exploration_domain(
    exploration_sets,
    interventional_variable_limits,
) -> dict:
    intervention_exploration_domain = {es: None for es in exploration_sets}
    for es in exploration_sets:
        if len(es) == 1:
            assert es[0] in interventional_variable_limits.keys()
            LL = float(min(interventional_variable_limits[es[0]]))
            UL = float(max(interventional_variable_limits[es[0]]))
        else:
            LL, UL = [], []  # lower-limit and upper-limit
            for i, var in enumerate(es):
                LL.append(float(min(interventional_variable_limits[var])))
                UL.append(float(max(interventional_variable_limits[var])))
            assert len(es) == len(UL) == len(LL)
        # Assign
        intervention_exploration_domain[es] = make_parameter_space_for_intervention_set(es, LL, UL)

    return intervention_exploration_domain


def make_parameter_space_for_intervention_set(
    exploration_set: tuple,
    lower_limit,
    upper_limit,
) -> ParameterSpace:
    assert isinstance(exploration_set, tuple)
    if len(exploration_set) == 1:
        assert isinstance(lower_limit, float)
        assert isinstance(upper_limit, float)
        return ParameterSpace([ContinuousParameter(str(exploration_set), lower_limit, upper_limit)])
    else:
        multivariate_limits = []
        assert len(exploration_set) == len(lower_limit), exploration_set
        assert len(exploration_set) == len(upper_limit), exploration_set
        for i, var in enumerate(exploration_set):
            multivariate_limits.append(ContinuousParameter(str(var), lower_limit[i], upper_limit[i]))
        return ParameterSpace(multivariate_limits)


def get_canonical_variables_only(es, base_target):
    mark = None
    for var in es:
        if var.find(base_target.lower()) == 0:
            mark = True
    if mark:
        return es[:-1]
    else:
        return es


def initialise_DCBO_parameters_and_objects_at_start_only(
    canonical_exploration_sets: list,
    interventional_data: dict,
    nr_interventions: int,
    base_target: str,
    index_name: int,
    task="min",
    turn_off_shuffle=False,
):
    assert isinstance(interventional_data, dict)
    assert sorted(canonical_exploration_sets) == list(sorted(interventional_data.keys()))
    target_values = {es: None for es in canonical_exploration_sets}
    interventions = copy.deepcopy(target_values)
    intervention_data_X = copy.deepcopy(target_values)
    intervention_data_Y = copy.deepcopy(target_values)

    for es in canonical_exploration_sets:

        # Interventional data contains a dictionary of dictionaries,
        # each corresponding to one type (es) of intervention.
        interventional_samples = interventional_data[es]  # es on keys and nd.array on values
        assert isinstance(interventional_samples, dict)
        assert base_target in interventional_samples.keys()
        assert isinstance(interventional_samples[base_target], np.ndarray)

        # Need to reset the global seed
        state = np.random.get_state()
        np.random.seed(index_name)
        data_subset = get_shuffled_dict_sample_subsets(interventional_samples, nr_interventions)
        np.random.set_state(state)
        if turn_off_shuffle:
            data_subset = interventional_samples

        # Under this intervention: find the optimal target value,
        # the corresponding interventions level(s) and the corresponding exploration set
        if task == "min":
            # For this exploration set find the smallest target value coordinates [array]
            target_coordinates = data_subset[base_target][:, 0].argmin(axis=0)
        elif task == "max":
            # For this exploration find the largest target value coordinates [array]
            target_coordinates = data_subset[base_target][:, 0].argmax(axis=0)

        # Find the corresponding target values at these coordinates [array]
        target_values[es] = data_subset[base_target][target_coordinates, 0]
        # Find the corresponding interventions [array]
        interventions[es] = get_corresponding_interventions(
            es,
            target_coordinates,
            data_subset,
            base_target,
            only_initial_time_step=True,
        )
        # Set the interventional data for use in DCBO
        (
            intervention_data_X[es],
            intervention_data_Y[es],
        ) = make_intervention_data_for_DCBO(es, interventions[es], target_values[es])

    # Best exploration set
    best_es = eval(task)(target_values, key=target_values.get)
    # Best target value
    optimal_target_values = target_values[best_es]

    return (
        best_es,
        optimal_target_values,
        interventions[best_es],
        intervention_data_X,
        intervention_data_Y,
    )


def convert_to_dict_of_temporal_lists(observational_samples: dict) -> dict:
    assert isinstance(observational_samples[list(observational_samples.keys())[0]], np.ndarray)
    assert len(observational_samples[list(observational_samples.keys())[0]].shape) == 2
    new = {k: None for k in observational_samples.keys()}
    for key in observational_samples.keys():
        new[key] = observational_samples[key].T.tolist()
    return new


def initialise_DCBO_parameters_and_objects_filtering(
    exploration_sets: list,
    interventional_data: dict,
    base_target: str,
    total_timesteps: int,
    task="min",
    index_name: int = None,
    nr_interventions: int = None,
) -> Tuple[list, list, list, dict, dict]:

    assert isinstance(interventional_data, dict)
    target_values = {t: {es: None for es in exploration_sets} for t in range(total_timesteps)}
    interventions = copy.deepcopy(target_values)

    intervention_data_X = copy.deepcopy(target_values)
    intervention_data_Y = copy.deepcopy(target_values)
    temporal_index = 0
    for es in exploration_sets:

        if es not in interventional_data.keys():

            pass

        else:

            # Interventional data contains a dictionary of dictionaries,
            # each corresponding to one type (es) of intervention.
            interventional_samples = interventional_data[es]  # es on keys and nd.array on values

            assert isinstance(interventional_samples, dict)
            assert base_target in interventional_samples.keys()
            assert isinstance(interventional_samples[base_target], np.ndarray)

            # This option exist _if_ we have more than one intervention per es
            if nr_interventions:
                assert index_name is not None
                # Need to reset the global seed
                state = np.random.get_state()
                np.random.seed(index_name)
                data_subset = get_shuffled_dict_sample_subsets(interventional_samples, nr_interventions)
                assert data_subset[list(data_subset.keys())[0]].shape[0] == nr_interventions

                np.random.set_state(state)

            # If we only have one sample per intervention we just use that
            else:
                data_subset = interventional_samples
            # Find the corresponding target values at these coordinates [array]
            target_values[temporal_index][es] = np.array(data_subset[base_target][temporal_index]).reshape(-1, 1)
            assert target_values[temporal_index][es] is not None

            # Find the corresponding interventions [array]
            if len(es) == 1:
                interventions[temporal_index][es] = np.array(data_subset[es[0]][temporal_index]).reshape(-1, 1)
            else:
                tmp = []
                for var in es:
                    tmp.append(data_subset[var][temporal_index])
                interventions[temporal_index][es] = np.expand_dims(np.hstack(tmp), axis=0)
            assert interventions[temporal_index][es] is not None

            # Set the interventional data for use in DCBO
            intervention_data_Y[temporal_index][es] = target_values[temporal_index][es]
            intervention_data_X[temporal_index][es] = interventions[temporal_index][es]

            assert intervention_data_X[temporal_index][es] is not None
            assert intervention_data_Y[temporal_index][es] is not None

    # Get best intervention set at each time index
    best_es = eval(task)(target_values[temporal_index], key=target_values[temporal_index].get)

    # Interventions
    best_intervention_level = interventions[temporal_index][best_es]
    # Outcomes
    best_target_value = target_values[temporal_index][best_es]

    # PRIORS
    # Use the best outcome level at t=0 as a prior for all the other timesteps
    best_es_sequence = total_timesteps * [None]
    best_es_sequence[0] = best_es
    best_intervention_levels = total_timesteps * [None]
    best_intervention_levels[0] = best_intervention_level
    best_target_levels = total_timesteps * [None]
    best_target_levels[0] = best_target_value

    return (
        best_es_sequence,
        best_target_levels,
        best_intervention_levels,
        intervention_data_X,
        intervention_data_Y,
    )


def initialise_DCBO_parameters_and_objects_full_graph_smoothing(
    canonical_exploration_sets: list,
    interventional_data: dict,
    base_target: str,
    index_name: int,
    task="min",
    autoregressive=True,
    nr_interventions: int = None,
) -> Tuple[list, list, list, dict, dict]:

    assert isinstance(interventional_data, dict)
    target_values = {es: None for es in canonical_exploration_sets}
    interventions = copy.deepcopy(target_values)

    if autoregressive:
        canonical_exploration_sets_VAR = [i + (base_target.lower() + "_{t-1}",) for i in canonical_exploration_sets]
        intervention_data_X = {es: None for es in (canonical_exploration_sets + canonical_exploration_sets_VAR)}
        intervention_data_Y = copy.deepcopy(intervention_data_X)
    else:
        intervention_data_X = copy.deepcopy(target_values)
        intervention_data_Y = copy.deepcopy(target_values)

    # TODO: investigate the issues w.r.t. the temporal dimension being ignored on instantiation,
    #  when we are considering a smoothing problem.
    for es in canonical_exploration_sets:

        if es not in interventional_data.keys():

            pass

        else:

            # Interventional data contains a dictionary of dictionaries,
            # each corresponding to one type (es) of intervention.
            interventional_samples = interventional_data[es]  # es on keys and nd.array on values
            assert isinstance(interventional_samples, dict)
            assert base_target in interventional_samples.keys()
            assert isinstance(interventional_samples[base_target], np.ndarray)

            # This option exist _if_ we have more than one intervention per es
            if nr_interventions:

                # Need to reset the global seed
                state = np.random.get_state()
                np.random.seed(index_name)
                data_subset = get_shuffled_dict_sample_subsets(interventional_samples, nr_interventions)
                assert data_subset[list(data_subset.keys())[0]].shape[0] == nr_interventions
                np.random.set_state(state)

            # If we only have one sample per intervention we just use that
            else:

                data_subset = interventional_samples

            # Under this intervention: find the optimal target value,
            # the corresponding interventions level(s) and the corresponding exploration set
            if task == "min":
                # For this exploration set find the smallest target value coordinates [array],
                # on each column (each time-step)
                # XXX: re-visit this; it is not clear if this should just select the "minimal" row (by sum e.g.)
                # or do it the current way.
                # TODO: investigate the issues w.r.t. the temporal dimension being ignored on instantiation.
                #  Particularly here there arg min should appear on the actually index of the intervention?
                target_coordinates = data_subset[base_target].argmin(axis=0)
            elif task == "max":
                # For this exploration find the largest target value coordinates [array]
                target_coordinates = data_subset[base_target].argmax(axis=0)

            # Find the corresponding target values at these coordinates [array]
            target_values[es] = data_subset[base_target][target_coordinates, range(len(target_coordinates))]
            assert target_values[es] is not None
            # Find the corresponding interventions [array]
            # TODO: temporal issue again; this intervention may _not_ be the one that was actually done in the data it.
            interventions[es] = get_corresponding_interventions(es, target_coordinates, data_subset, base_target)
            assert interventions[es] is not None
            # Set the interventional data for use in DCBO
            (
                intervention_data_X[es],
                intervention_data_Y[es],
            ) = make_intervention_data_for_DCBO(es, interventions[es], target_values[es])
            assert intervention_data_X[es] is not None
            assert intervention_data_Y[es] is not None

    if autoregressive:
        for es, es_VAR in zip(canonical_exploration_sets, canonical_exploration_sets_VAR):
            intervention_data_X[es_VAR] = np.hstack((intervention_data_X[es], intervention_data_Y[es]))
            intervention_data_Y[es_VAR] = intervention_data_Y[es]

    all_best_outcome_levels = []
    # Search of the canonical set
    for es in canonical_exploration_sets:
        all_best_outcome_levels.append(intervention_data_Y[es])
    all_best_outcome_levels = np.hstack(all_best_outcome_levels)

    # Get best intervention set at each time index
    if task == "min":
        best_es_sequence_indices = all_best_outcome_levels.argmin(axis=1)
    else:
        best_es_sequence_indices = all_best_outcome_levels.argmax(axis=1)

    # Best sequence of interventions
    best_es_sequence = [canonical_exploration_sets[i] for i in best_es_sequence_indices]
    # Corresponding intervention levels
    best_intervention_levels = []
    best_outcome_levels = []
    for t, es in enumerate(best_es_sequence):
        # Interventions
        best_intervention_levels.append(intervention_data_X[es][t, :])
        # Outcomes
        best_outcome_levels.append(intervention_data_Y[es][t, :])

    # These are ordered lists/sequences
    assert len(best_es_sequence) == len(best_intervention_levels) == len(best_outcome_levels)

    return (
        best_es_sequence,
        best_outcome_levels,
        best_intervention_levels,
        intervention_data_X,
        intervention_data_Y,
    )


def get_corresponding_interventions(
    exploration_set: tuple,
    target_coordinates: list,
    interventional_samples: dict,
    base_target: str,
    only_initial_time_step: bool = False,
):
    assert base_target in interventional_samples.keys()
    out = {key: None for key in interventional_samples.keys() if key != base_target}
    if only_initial_time_step:
        col = 0
    else:
        tmp = list(interventional_samples.keys())[0]
        max_T = interventional_samples[tmp].shape[1]
        col = range(max_T)
    for i in exploration_set:
        out[i] = interventional_samples[i][target_coordinates, col]
    return out


def create_time_mask(samples):
    assert isinstance(samples, dict), type(samples)
    N, T = samples[list(samples.keys())[0]].shape
    return np.tile(range(T), N).reshape(-1, T)


def get_shuffled_dict_sample_subsets(samples, nr_interventions):
    assert isinstance(samples, dict), type(samples)
    for key in samples.keys():
        D = samples[key]
        # Means that all columns have the same number of samples
        assert isinstance(D, np.ndarray)
    # Rows and total timesteps
    N, T = samples[list(samples.keys())[0]].shape
    shuffled_row_ids = np.random.permutation(N)
    assert nr_interventions <= N
    new = {key: None for key in samples.keys()}
    for key in samples.keys():
        new[key] = samples[key][shuffled_row_ids][:nr_interventions]
    return new


def make_intervention_data_for_DCBO(es, intervention_levels, target_levels):

    YY = np.ravel(target_levels, order="F").reshape(-1, 1)

    # Need to transform the time-series data
    if len(es) == 1:
        XX = np.ravel(intervention_levels[es[0]], order="F").reshape(-1, 1)
    else:
        XX = []
        for iv in es:
            XX.append(np.ravel(intervention_levels[iv], order="F").reshape(-1, 1))
        XX = np.hstack(XX)

    return XX, YY


def initialise_optimal_intervention_level_list(
    total_graph_timesteps: int,
    exploration_sets: list,
    initial_optimal_sequential_intervention_sets: list,
    initial_optimal_sequential_intervention_levels: list,
    number_of_trials: int,
) -> list:
    assert len(initial_optimal_sequential_intervention_levels) == total_graph_timesteps
    intervention_levels = [
        {es: number_of_trials * [None] for es in exploration_sets} for _ in range(total_graph_timesteps)
    ]

    #  Add interventional data that we have at start
    for es in exploration_sets:
        if es == initial_optimal_sequential_intervention_sets[0]:
            intervention_levels[0][es].insert(0, initial_optimal_sequential_intervention_levels[0])
        else:
            intervention_levels[0][es].insert(0, None)

    return intervention_levels


def initialise_global_outcome_dict_new(
    total_graph_timesteps: int, initial_optimal_target_values: list, blank_val
) -> dict:
    assert isinstance(total_graph_timesteps, int)
    assert isinstance(initial_optimal_target_values, list)
    assert total_graph_timesteps > 0
    assert len(initial_optimal_target_values) == total_graph_timesteps
    # Remember that Python lists are mutable objects, hence this construction.
    targets = {t: [] for t in range(total_graph_timesteps)}

    for t in range(total_graph_timesteps):
        if initial_optimal_target_values[t]:
            targets[t].append(float(initial_optimal_target_values[t]))
        else:
            # No interventional data was provided so this is empty.
            targets[t].append(blank_val)
    return targets


def find_global_optima_at_current_time_new(target_values, task, time_index) -> float:
    if task == "min":
        return min(target_values[time_index])
    else:
        return max(target_values[time_index])


def initialise_global_target_dict(exploration_set, task, max_T: int) -> dict:
    assert isinstance(max_T, int)
    assert max_T > 0
    if task == "min":
        # Remember that Python lists are mutable objects, hence this construction.
        return {es: [[np.inf] for _ in range(max_T)] for es in exploration_set}
    else:
        return {es: [[-np.inf] for _ in range(max_T)] for es in exploration_set}


def find_global_optima_at_current_time(best_y_hitherto, task, time_index):
    if task == "min":
        return min(
            [(i, min(j[time_index])) for i, j in best_y_hitherto.items()],
            key=lambda k: k[1],
        )
    else:
        return max(
            [(i, max(j[time_index])) for i, j in best_y_hitherto.items()],
            key=lambda k: k[1],
        )


def make_column_shape_2D(x):
    return np.array([x]).reshape(-1, 1)


def get_valid_children_in_time_slice(graph, exploration_set, time_slice_index, target_canonical_variable):
    return [
        s
        for s in graph.successors(exploration_set[0] + "_" + str(time_slice_index))
        if s.split("_")[1] == str(time_slice_index) and s.split("_")[0] != target_canonical_variable
    ]


def assign_blanket_hat(
    blanket_hat: dict,
    exploration_set,
    intervention_level,
    target,
    target_value,
):

    # Split current target
    target_variable, temporal_index = target.split("_")
    temporal_index = int(temporal_index)
    assert len(exploration_set) == intervention_level.shape[1], (
        exploration_set,
        intervention_level,
    )
    assert intervention_level is not None
    #  Assign target value
    blanket_hat[target_variable][temporal_index] = float(target_value)  # TARGET
    #  Assign intervention
    for intervention_variable, xx in zip(exploration_set, intervention_level.ravel()):
        blanket_hat[intervention_variable][temporal_index] = xx

    return


def assign_blanket(
    initial_sem: dict,  # OBS: true SEM
    sem: dict,  #  OBS: true SEM
    blanket: dict,
    exploration_set: list,
    intervention_level,
    target: str,
    target_value,
    node_children: dict,
):

    # Split current target
    target_canonical_variable, temporal_index = target.split("_")
    temporal_index = int(temporal_index)
    assert len(exploration_set) == intervention_level.shape[1], (
        exploration_set,
        intervention_level,
    )
    assert intervention_level is not None

    #  Assign target value
    blanket[target_canonical_variable][temporal_index] = float(target_value)

    if len(exploration_set) == 1:
        # Intervention only happening on _one_ variable, assign it
        intervention_variable = exploration_set[0]
        # Intervention only happening on _one_ variable, assign it
        blanket[intervention_variable][temporal_index] = float(intervention_level)
        # The target and intervention value have already assigned
        # so we check to see if anything else is missing in this time-slice
        intervention_node = intervention_variable + "_" + str(temporal_index)
        children = [
            v.split("_")[0] for v in node_children[intervention_node] if v.split("_")[0] != target_canonical_variable
        ]
        if len(children) != 0:
            for child in children:
                if blanket[child][temporal_index] is None:  # Only valid when t > 0
                    # Value is None so we sample a value for this node
                    sample = sequential_sample_from_model(initial_sem, sem, temporal_index + 1, interventions=blanket)
                    blanket[child][temporal_index] = sample[child][temporal_index]
    else:
        for i, intervention_variable in enumerate(exploration_set):
            blanket[intervention_variable][temporal_index] = float(intervention_level[:, i])


def optimisation_type_reminder(causal_prior, dynamic):
    """
    Optimisation options:

    1) DCBO         : (causal_prior == True, dynamic == True)
    2) Dynamic BO   : (causal_prior == False, dynamic == True)
    3) CBO          : (causal_prior == True, dynamic == False)
    4) BO           : (causal_prior == False, dynamic == False)

    Parameters
    ----------
    causal_prior : bool
        [description]
    dynamic : bool
        [description]
    """
    opt = {
        (True, True): "DCBO",
        (False, True): "ABO",
        (True, False): "CBO",
        (False, False): "BO",
    }
    print("\n\t\t>>> {}\n".format(opt[(causal_prior, dynamic)]))


def check_blanket(blanket, base_target_variable, temporal_index, manipulative_variables):
    # Check that the target has been properly assigned.
    assert blanket[base_target_variable][temporal_index] is not None, temporal_index
    # Check that at least one intervention has been assigned. E.g. if X was intervened upon then Z should have a value.
    assert any(x is not None for x in [blanket[var][temporal_index] for var in manipulative_variables]), blanket


def select_sample(sample, input_variables, outside_time):
    if isinstance(input_variables, str):
        return sample[input_variables][outside_time].reshape(-1, 1)
    else:
        #  Takes either a tuple() or a list()
        samp = []
        for node in input_variables:
            var, time = node.split("_")[0], int(node.split("_")[1])
            assert time == outside_time, (sample, input_variables, time, outside_time)
            samp.append(sample[var][time].reshape(-1, 1))
        return hstack(samp)


def calculate_best_intervention_and_effect(
    static_sem,
    dynamic_sem,
    exploration_sets,
    interventional_grids,
    time,
    intervention_domain,
    blanket,
    T,
    plot=False,
    target="Y",
    print_option=False,
):
    true_causal_effect = {key: None for key in exploration_sets}
    static_noise_model = {k: np.zeros(T) for k in static_sem.keys()}
    for es in exploration_sets:
        res = []
        this_blanket = deepcopy(blanket)
        for xx in interventional_grids[es]:
            for intervention_variable, x in zip(es, xx):
                this_blanket[intervention_variable][time] = x
            out = sequential_sample_from_model(
                static_sem=static_sem,
                dynamic_sem=dynamic_sem,
                timesteps=time + 1,
                epsilon=static_noise_model,
                interventions=this_blanket,
            )

            res.append(out[target][time])

        true_causal_effect[es] = np.array(res)

    #  Plot results
    if plot:
        for es in exploration_sets:
            if len(es) == 1:
                fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                fig.suptitle("True causal effect at $t={}$".format(time))
                ax.plot(
                    interventional_grids[es],
                    true_causal_effect[es],
                    lw=2,
                    alpha=0.5,
                    label="$do{}$".format(es),
                )
                plt.legend()
            # elif len(es) == 2:
            #     make_surface_plot(interventional_grids, true_causal_effect[es], intervention_domain, variables=list(es))

    opt_vals = {es: None for es in exploration_sets}
    # Find best causal effect
    for es in exploration_sets:
        Y = true_causal_effect[es].tolist()
        # Min value
        outcome_min_val = min(Y)
        # Corresponding intervention at min value
        idx = Y.index(outcome_min_val)
        opt_vals[es] = (outcome_min_val, interventional_grids[es][idx, :])

    # Get best
    minval = min(k[0] for k in opt_vals.values())
    best_es = [k for k, v in opt_vals.items() if v[0] == minval]

    # Indexed by zero so we take the shortest ES first
    if print_option is True:
        print("\nBest exploration set: {}".format(best_es))
        print("Best intervention level: {}".format(opt_vals[best_es[0]][1]))
        print("Best best outcome value: {}".format(opt_vals[best_es[0]][0]))

    for intervention_variable, x in zip(best_es[0], opt_vals[best_es[0]][1]):
        blanket[intervention_variable][time] = x
    blanket[target][time] = opt_vals[best_es[0]][0]

    if print_option is True:
        print("\nNext blanket:\n")
        print(blanket)

    return blanket, true_causal_effect


def create_plankton_dataset(start: int, end: int) -> dict:
    """Function to create dataset for plankton experiment.

    Uses data from experiments C1 to C4 from [1].

    A series of ten chemostat experiments was performed, constituting a total of 1,948 measurement days (corresponding to 5.3 years of measurement) and covering 3 scenarios.

    Constant environmental conditions (C1–C7, 1,428 measurement days). This scenario consisted of 4 trials with the alga M. minutum (C1–C4) which is what we use in these experiments. All data is lives in `data/plankton` and is freely available online.

    [1] Blasius B, Rudolf L, Weithoff G, Gaedke U, Fussmann G.F. Long-term cyclic persistence in an experimental predator-prey system. Nature (2019).

    Parameters
    ----------
    start : int
        Start time-index
    end : int
        End time-index

    Returns
    -------
    dict
        State-variables as the keys with data as a ndarray
    """

    #  Constants from paper
    v_algal = 28e-9  # nitrogen content per algal cell
    v_Brachionus = 0.57 * 1e-3  # nitrogen content per adult female Brachionus
    beta = 5
    data = DataFrame()

    ds = []

    files = ["C1", "C2", "C3", "C4"]  # , "C6", "C7"]
    for file in files:
        df = read_csv("../data/plankton/{}.csv".format(file))

        # Impute missing values
        df.interpolate(method="cubic", inplace=True)  #  There are better imputation methods

        data["M"] = df[" external medium (mu mol N / l)"]
        data["N"] = df[" algae (10^6 cells/ml)"] * 1e6 * 1000 * v_algal
        data["P"] = df[" rotifers (animals/ml)"] * 1000 * v_Brachionus
        data["D"] = df[" dead animals (per ml)"] * 1000 * v_Brachionus
        data["E"] = df[" eggs (per ml)"] * 1000 * v_Brachionus
        data["B"] = df.apply(
            lambda row: beta * row[" eggs (per ml)"] / row[" egg-ratio"] if row[" egg-ratio"] > 0 else 0.0, axis=1
        )

        # Derivative state variables (function of other state variables)
        data["A"] = data.apply(lambda row: (row.B * 0.5) * 1000 * v_Brachionus, axis=1)
        data["J"] = data.apply(lambda row: (row.B / (2 * beta)) * 1000 * v_Brachionus, axis=1)

        # Replace NaN values at t=0 with 0.0
        data.fillna(value=0.0, inplace=True)
        assert data.isnull().sum().sum() == 0, (file, df.isnull().sum())

        tmp_dict = data[["M", "N", "P", "J", "A", "E", "D"]].iloc[start:end, :].to_dict("list")
        # print(tmp_dict)
        ds.append({item[0]: np.array(item[1]).reshape(1, -1) for item in tmp_dict.items()})

    # Merge all observations from all datasets
    d = {}
    for k in tmp_dict.keys():
        d[k] = np.concatenate(list(d[k] for d in ds), axis=0)

    print("Units of all observation variables is (mu mol N / L).")
    return d


def update_emission_pairs_keys(T: int, node_parents: dict, emission_pairs: dict) -> dict:
    """
    Sometimes the input and output pair order does not match because of NetworkX internal issues,
    so we need adjust the keys so that they do match.

    Parameters
    ----------
    T : int
        [description]
    node_parents : dict
        [description]
    emission_pairs : dict
        [description]

    Returns
    -------
    dict
        [description]

    Raises
    ------
    ValueError
        [description]
    """
    for t in range(T):
        nodes = [v for v in node_parents.keys() if v.split("_")[1] == str(t)]
        for node in nodes:
            if len(node_parents[node]) > 1:
                #  Get only parents from this time-slice
                parents = (*[v for v in node_parents[node] if v.split("_")[1] == str(t)],)
                # Check if parents live in the emission pair dictionary
                if not parents in emission_pairs.keys():
                    #  Check if reverse tuple live in the emission pair dictionary
                    if tuple(reversed(parents)) in emission_pairs.keys():
                        # Remove the wrong key and replace it with correct one
                        emission_pairs[parents] = emission_pairs.pop(tuple(reversed(parents)))
                    else:
                        raise ValueError("This key is erroneous.", parents, tuple(reversed(parents)))

    return emission_pairs
