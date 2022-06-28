from copy import deepcopy
from itertools import chain, combinations
from typing import Iterable, OrderedDict, Tuple
from networkx.classes.multidigraph import MultiDiGraph
import numpy as np
from emukit.core import ContinuousParameter, ParameterSpace
from numpy.core import hstack, vstack
from .sequential_sampling import sequential_sample_from_true_SEM
import matplotlib.pyplot as plt


def standard_mean_function(x):
    return np.zeros_like(x)


def zero_variance_adjustment(x):
    return np.zeros_like(x)


def check_reshape_add_data(
    interventional_data_x, interventional_data_y, new_interventional_data_x, y_new, best_es, temporal_index,
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
            (interventional_data_y[temporal_index][best_es], make_column_shape_2D(y_new),)
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


def get_monte_carlo_expectation(intervention_samples):
    assert isinstance(intervention_samples, dict)
    new = {k: None for k in intervention_samples.keys()}
    for es in new.keys():
        new[es] = intervention_samples[es].mean(axis=0)

    # Returns the expected value of the intervention via MC sampling
    return new


def create_intervention_exploration_domain(exploration_sets, interventional_variable_limits,) -> dict:
    intervention_exploration_domain = {es: None for es in exploration_sets}
    for es in exploration_sets:
        if len(es) == 1:
            assert es[0] in interventional_variable_limits.keys()
            LL = float(min(interventional_variable_limits[es[0]]))
            UL = float(max(interventional_variable_limits[es[0]]))
        else:
            LL, UL = [], []  # lower-limit and upper-limit
            for var in es:
                LL.append(float(min(interventional_variable_limits[var])))
                UL.append(float(max(interventional_variable_limits[var])))
            assert len(es) == len(UL) == len(LL)
        # Assign
        intervention_exploration_domain[es] = make_parameter_space_for_intervention_set(es, LL, UL)

    return intervention_exploration_domain


def make_parameter_space_for_intervention_set(exploration_set: tuple, lower_limit, upper_limit,) -> ParameterSpace:
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


def convert_to_dict_of_temporal_lists(observational_samples: dict) -> dict:
    assert isinstance(observational_samples[list(observational_samples.keys())[0]], np.ndarray)
    assert len(observational_samples[list(observational_samples.keys())[0]].shape) == 2
    new = {k: None for k in observational_samples.keys()}
    for key in observational_samples.keys():
        new[key] = observational_samples[key].T.tolist()
    return new


def get_shuffled_dict_sample_subsets(samples, nr_interventions):
    assert isinstance(samples, dict), type(samples)
    for key in samples.keys():
        D = samples[key]
        # Means that all columns have the same number of samples
        assert isinstance(D, np.ndarray)
    # Rows and total timesteps
    N, _ = samples[list(samples.keys())[0]].shape
    shuffled_row_ids = np.random.permutation(N)
    assert nr_interventions <= N
    new = {key: None for key in samples.keys()}
    for key in samples.keys():
        new[key] = samples[key][shuffled_row_ids][:nr_interventions]
    return new


def initialise_interventional_objects(
    exploration_sets: list,
    D_I: dict,  # Interventional data
    base_target: str,
    total_timesteps: int,
    task="min",
    index_name: int = None,
    nr_interventions: int = None,
) -> Tuple[list, list, list, dict, dict]:

    assert isinstance(D_I, dict)
    target_values = {t: {es: None for es in exploration_sets} for t in range(total_timesteps)}
    interventions = deepcopy(target_values)

    intervention_data_X = deepcopy(target_values)
    intervention_data_Y = deepcopy(target_values)
    temporal_index = 0
    for es in exploration_sets:
        if es not in D_I:
            pass
        else:
            # Interventional data contains a dictionary of dictionaries, each corresponding to one type (es) of intervention.
            interventional_samples = D_I[es]  # es on keys and nd.array on values

            assert isinstance(interventional_samples, dict), (es, type(interventional_samples), D_I)
            assert base_target in interventional_samples
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
    print(target_values)
    best_es = eval(task)(target_values[temporal_index], key=target_values[temporal_index].get)

    # Interventions
    best_intervention_level = interventions[temporal_index][best_es]
    # Outcomes
    best_target_value = target_values[temporal_index][best_es]

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


def make_column_shape_2D(x):
    return np.array([x]).reshape(-1, 1)


def assign_blanket_hat(
    blanket_hat: dict, exploration_set, intervention_level, target, target_value,
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
    static_sem: dict,  # OBS: true SEM
    dynamic_sem: dict,  #  OBS: true SEM
    blanket: dict,
    exploration_set: list,
    intervention_level: np.array,
    target: str,
    target_value: float,
    G: MultiDiGraph,
) -> None:

    # Split current target
    target_var, temporal_index = target.split("_")
    t = int(temporal_index)
    assert len(exploration_set) == intervention_level.shape[1], (
        exploration_set,
        intervention_level,
    )
    assert intervention_level is not None

    #  Assign target value
    blanket[target_var][t] = float(target_value)

    if len(exploration_set) == 1:
        # Intervention only happening on _one_ variable, assign it
        intervention_variable = exploration_set[0]
        # Intervention only happening on _one_ variable, assign it
        blanket[intervention_variable][t] = float(intervention_level)
        # The target and intervention value have already assigned so we check to see if anything else is missing in this time-slice
        intervention_node = intervention_variable + "_" + str(t)
        children = [
            v.split("_")[0]
            for v in G.successors(intervention_node)
            if v.split("_")[0] != target_var and v.split("_")[1] == temporal_index
        ]
        if children:
            for child in children:
                if blanket[child][t] is None:  # Only valid when t > 0
                    # Value is None so we sample a value for this node
                    sample = sequential_sample_from_true_SEM(static_sem, dynamic_sem, t + 1, interventions=blanket)
                    blanket[child][t] = sample[child][t]
    else:
        for i, intervention_variable in enumerate(exploration_set):
            blanket[intervention_variable][t] = float(intervention_level[:, i])


def check_blanket(blanket, base_target_variable, temporal_index, manipulative_variables):
    # Check that the target has been properly assigned.
    assert blanket[base_target_variable][temporal_index] is not None, temporal_index
    # Check that at least one intervention has been assigned. E.g. if X was intervened upon then Z should have a value.
    assert any(x is not None for x in [blanket[var][temporal_index] for var in manipulative_variables]), blanket


def select_sample(sample: OrderedDict, input_variables: Iterable, outside_time: int) -> np.ndarray:
    """Select the part of the sample which will be used as the input for the GP regression.

    Parameters
    ----------
    sample : OrderedDict
        The sample as it is being created
    input_variables : Iterable
        The input variables fort he GP regression
    outside_time : int
        The current time indexed being worked on

    Returns
    -------
    np.ndarray
        The input formatted as an ndarray of shape N x D
    """

    if isinstance(input_variables, str):
        return sample[input_variables][outside_time].reshape(-1, 1)
    elif len(input_variables) == 3 and isinstance(input_variables[1], int):
        # Special case when the input looks like (pa_V,ID,V)
        pa_V = input_variables[0]
        var, time = pa_V.split("_")[0], int(pa_V.split("_")[1])
        assert time == outside_time, (sample, input_variables, time, outside_time)
        return sample[var][time].reshape(-1, 1)
    else:
        # Takes either a tuple() or a list()
        samp = []
        for node in input_variables:
            var, time = node.split("_")[0], int(node.split("_")[1])
            assert time == outside_time, (sample, input_variables, time, outside_time)
            samp.append(sample[var][time].reshape(-1, 1))
        return hstack(samp)


def powerset(iterable):
    # this returns e.g. powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def calculate_best_intervention_and_effect(
    static_sem,
    dynamic_sem,
    exploration_sets,
    interventional_grids,
    time,
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
            out = sequential_sample_from_true_SEM(
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
                    interventional_grids[es], true_causal_effect[es], lw=2, alpha=0.5, label="$do{}$".format(es),
                )
                plt.legend()

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
