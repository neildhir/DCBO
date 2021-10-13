from collections import OrderedDict

import numpy as np
from numpy.random import randn
from scipy.spatial import ConvexHull


def sequential_sample_from_model(
    static_sem,
    dynamic_sem,
    timesteps: int,
    initial_values: dict = None,
    interventions: dict = None,
    epsilon=None,
    seed=None,
):
    """Function to sequentially sample a dynamic Bayesian network.

    Parameters
    ----------
    initial_values : dict
        Start values for SEM
    interventions : dict
        Dictionary of interventions (variables on keys and time-steps on horizontal array dim)

    Returns
    -------
    dict
        Sequential sample from DBN.
    """

    # A specific noise-model has not been provided so we use standard Gaussian noise
    if seed is not None:
        np.random.seed(seed)

    if not epsilon:
        epsilon = {k: np.random.randn(timesteps) for k in static_sem.keys()}
    assert isinstance(epsilon, dict)

    # Notice that we call it 'sample' in singular since we only want one sample of the whole graph
    sample = OrderedDict([(k, np.zeros(timesteps)) for k in static_sem.keys()])

    if initial_values:
        assert sample.keys() == initial_values.keys()

    for temporal_idx in range(timesteps):

        if temporal_idx == 0:

            for var, function in static_sem.items():
                # Check that interventions and initial values at t=0 are not both provided
                if interventions and initial_values:
                    if interventions[var][temporal_idx] is not None and initial_values[var] is not None:
                        raise ValueError(
                            "You cannot provided an initial value "
                            "and an intervention for the same location "
                            "(var,time) in the graph."
                        )

                # If interventions exist they take precedence
                if interventions is not None and interventions[var][temporal_idx] is not None:
                    sample[var][temporal_idx] = interventions[var][temporal_idx]

                # If initial values are passed then we use these, if no interventions
                elif initial_values:
                    sample[var][temporal_idx] = initial_values[var]

                # If neither interventions nor initial values are provided
                # sample the model with provided epsilon, if exists
                else:
                    # XXX: this SEM does _not_ have backwards temporal dependency
                    sample[var][temporal_idx] = function(epsilon[var][temporal_idx], temporal_idx, sample)

        else:
            # Dynamically propagate the samples through the graph using
            # the static (with noise) structural equation models
            # If we have found an optimal interventional target response from t-1,
            # it has been included in the intervention dictionary at the correct index.
            for var, function in dynamic_sem.items():
                if interventions is not None and interventions[var][temporal_idx] is not None:
                    sample[var][temporal_idx] = interventions[var][temporal_idx]
                else:
                    # TODO: fix this construction, it is very slow to pass the whole sample.
                    # TODO: evaluate input params in this loop and then pass as blind arguments
                    sample[var][temporal_idx] = function(epsilon[var][temporal_idx], temporal_idx, sample)
                # print("\n sample inside sampling functin:", var, temporal_idx, sample)

    return sample


def sequential_sample_from_model_hat(
    initial_sem, dynamic_sem, timesteps: int, initial_values: dict = None, interventions: dict = None,
):
    """
    Function to sequentially sample a dynamic Bayesian network using ESTIMATED SEMs.
    Currently function approximations are done using Gaussian processes.

    Parameters
    ----------
    initial_values : dict
        Start values for SEM
    interventions : dict
        Dictionary of interventions (variables on keys and time-steps on horizontal array dim)

    Returns
    -------
    dict
        Sequential sample from DBN.
    """

    # Notice that we call it 'sample' in singular since we only want one sample of
    # the whole graph
    sample = OrderedDict([(k, np.zeros(timesteps)) for k in initial_sem.keys()])

    if initial_values:
        assert sample.keys() == initial_values.keys()
    parent = None
    for temporal_idx in range(timesteps):
        if temporal_idx == 0:

            for i, (var, function) in enumerate(initial_sem.items()):
                # Check that interventions and initial values at t=0 are not both provided
                if interventions and initial_values:
                    if interventions[var][temporal_idx] is not None and initial_values[var] is not None:
                        raise ValueError(
                            "You cannot provided an initial value "
                            "and an intervention for the same location "
                            "(var,time) in the graph."
                        )

                # If interventions exist they take precedence
                if interventions and interventions[var][temporal_idx]:
                    sample[var][temporal_idx] = interventions[var][temporal_idx]

                # If initial values are passed then we use these, if no interventions
                elif initial_values:
                    sample[var][temporal_idx] = initial_values[var]

                # If neither interventions nor initial values are provided
                # sample the model with provided epsilon, if exists
                else:
                    #  This is a special assignment for the first node and the first time-step of the CGM.
                    if i == 0:
                        sample[var][temporal_idx] = function(randn())
                    else:
                        sample[var][temporal_idx] = function(temporal_idx, parent, sample)

                parent = var

        else:

            # Dynamically propagate the samples through the graph using estimated SEMs
            # If we have found an optimal interventional target response from t-1,
            # it has been included in the intervention dictionary at the correct index.
            for i, (var, function) in enumerate(dynamic_sem.items()):
                if interventions and interventions[var][temporal_idx]:
                    sample[var][temporal_idx] = interventions[var][temporal_idx]
                else:
                    sample[var][temporal_idx] = function(temporal_idx, parent, sample)

                parent = var

    return sample


def sequential_sample_from_complex_model_hat(
    static_sem,
    dynamic_sem,
    timesteps: int,
    node_parents: dict,
    initial_values: dict = None,
    interventions: dict = None,
    seed=None,
):
    """
    Function to sequentially sample a dynamic Bayesian network using ESTIMATED SEMs.
    Currently function approximations are done using Gaussian processes.
    This function handles far more complex graphs.
    """
    # A specific noise-model has not been provided so we use standard Gaussian noise
    if seed is not None:
        np.random.seed(seed)

    # Notice that we call it 'sample' in singular since we only want one sample of
    # the whole graph
    sample = OrderedDict([(k, np.zeros(timesteps)) for k in static_sem.keys()])

    if initial_values:
        assert sample.keys() == initial_values.keys()

    for temporal_index in range(timesteps):

        if temporal_index == 0 or dynamic_sem is None:

            for var, function in static_sem.items():

                time = str(temporal_index)
                node = var + "_" + time

                # Check that interventions and initial values at t=0 are not both provided
                if interventions and initial_values:
                    if interventions[var][temporal_index] is not None and initial_values[var] is not None:
                        raise ValueError(
                            "You cannot provided an initial value "
                            "and an intervention for the same location "
                            "(var,time) in the graph."
                        )

                # If interventions exist they take precedence
                if interventions and interventions[var][temporal_index]:
                    sample[var][temporal_index] = interventions[var][temporal_index]

                # If initial values are passed then we use these, if no interventions
                elif initial_values:
                    sample[var][temporal_index] = initial_values[var]

                # If neither interventions nor initial values are provided
                # so sample the model with provided epsilon, if exists
                else:
                    if node_parents[node]:
                        if temporal_index == 0:
                            sample[var][temporal_index] = function(temporal_index, node_parents[node], sample)
                        else:
                            emit_vars = (*[v for v in node_parents[node] if v.split("_")[1] == time],)
                            if emit_vars:
                                sample[var][temporal_index] = function(temporal_index, emit_vars, sample)
                            elif not emit_vars:
                                sample[var][temporal_index] = function()
                            else:
                                raise ValueError("There are no parents!", emit_vars)
                    else:
                        sample[var][temporal_index] = function()

        else:
            assert dynamic_sem is not None
            # Dynamically propagate the samples through the graph using estimated SEMs.
            # If we have found an optimal interventional target response from t-1,
            # it has been included in the intervention dictionary at the correct index.

            for var, function in dynamic_sem.items():
                time = str(temporal_index)
                node = var + "_" + time

                #  XXX: note the weird syntax used here; it converts a list to a tuple (e.g. Y_0 --> Y_1)
                transfer_vars = (*[v for v in node_parents[node] if v.split("_")[1] != time],)
                # Get conditioning variables from conditional distribution (input-variables at this time-step only)
                emit_vars = (*[v for v in node_parents[node] if v.split("_")[1] == time],)

                #  Sample
                if interventions and interventions[var][temporal_index]:
                    sample[var][temporal_index] = interventions[var][temporal_index]
                elif node_parents[node]:
                    sample[var][temporal_index] = function(temporal_index, transfer_vars, emit_vars, sample)
                else:
                    sample[var][temporal_index] = function()

    return sample


def sequentially_sample_model(
    static_sem,
    dynamic_sem,
    total_timesteps: int,
    initial_values=None,
    interventions=None,
    node_parents=None,
    emission_parents=None,
    sample_count=100,
    epsilon=None,
    use_sem_estimate=False,
    seed=None,
) -> dict:
    """
    Draws multiple samples from DBN.

    Per variable the returned array is of the format: n_samples x timesteps in DBN.

    Returns
    -------
    dict
        Dictionary of n_samples per node in graph.
    """

    new_samples = {k: [] for k in static_sem.keys()}
    # TODO: should this sample_count loop really be there for the estimated SEM?
    for i in range(sample_count):
        # This option uses the estimates of the SEMs, estimates found through use of GPs.
        if use_sem_estimate:
            tmp = sequential_sample_from_complex_model_hat(
                static_sem=static_sem,
                dynamic_sem=dynamic_sem,
                timesteps=total_timesteps,
                node_parents=node_parents,
                initial_values=initial_values,
                interventions=interventions,
                seed=seed,
            )
        # This option uses the true SEMs.
        else:
            if epsilon is not None and isinstance(epsilon, list):
                epsilon_term = epsilon[i]
            else:
                epsilon_term = epsilon

            tmp = sequential_sample_from_model(
                static_sem=static_sem,
                dynamic_sem=dynamic_sem,
                timesteps=total_timesteps,
                initial_values=initial_values,
                interventions=interventions,
                epsilon=epsilon_term,
                seed=seed,
            )
        for var in static_sem.keys():
            new_samples[var].append(tmp[var])

    for var in static_sem.keys():
        new_samples[var] = np.vstack(new_samples[var])

    return new_samples


def extract_data_streams_from_multivariate_time_series(variables, data, new_samples=False):
    if len(variables) == 1 or isinstance(variables, str):
        try:
            canonical_variable, temporal_index = variables[0].split("_")
        except ValueError:
            canonical_variable, temporal_index = variables.split("_")
        if new_samples:
            # This option is used when we have different number of samples per time step
            return np.expand_dims(data[canonical_variable][int(temporal_index)], axis=1)
        else:
            return np.expand_dims(data[canonical_variable][:, int(temporal_index)], axis=1)
    else:
        tmp = []
        for my_var in variables:
            canonical_variable, temporal_index = my_var.split("_")
            if new_samples:
                tmp.append(np.expand_dims(data[canonical_variable][int(temporal_index)], axis=1))
            else:
                tmp.append(np.expand_dims(data[canonical_variable][:, int(temporal_index)], axis=1))
        return np.hstack(tmp)


def calculate_epsilon(
    observational_samples,
    canonical_manipulative_variables,
    temporal_index,
    initial_observation_count,
    total_observation_coverage,
):

    assert isinstance(temporal_index, int)
    observational_coverage = update_convex_hull(observational_samples, canonical_manipulative_variables, temporal_index)
    """
    1) need to update the observational samples after each opt loop
    2) this assumes that each variables has the same number of observations which is not at all true.
    Currently the code only allows for this.
    """
    N = len(observational_samples[canonical_manipulative_variables[0]][temporal_index])
    scale_coverage_factor = (
        # Note that we have the same number of samples per manipulative variable, per time-slice
        N
        / initial_observation_count
    )

    # Check N_max >= N
    assert initial_observation_count >= N, (
        "Max number of observations smaller than current N",
        observational_samples,
        temporal_index,
        initial_observation_count,
        N,
    )

    # observational_coverage needs to be <= than total_observation_coverage
    clipped_observational_coverage = np.clip(observational_coverage, 0.0, total_observation_coverage)

    return (clipped_observational_coverage / total_observation_coverage) * scale_coverage_factor


def update_convex_hull(observational_samples, manipulative_variables, temporal_index):
    # obs = extract_data_streams_from_multivariate_time_series(manipulative_variables, observational_samples)
    obs = extract_data_streams_from_multivariate_time_series_list(
        manipulative_variables, observational_samples, temporal_index
    )
    assert obs.shape[0] > 2
    if all([np.isclose(obs[:, 0], obs[:, t]).all() for t in range(obs.shape[1])]):
        # This is invoked IF an intervention or an initial condition has been used in which case the convex hull
        # does not exist.
        raise ValueError(
            "This is invoked IF an intervention or "
            "an initial condition has been used in which case "
            "the convex hull does not exist."
        )
    return ConvexHull(obs).volume


def extract_data_streams_from_multivariate_time_series_list(canonical_variables, samples, temporal_index):
    """
    We use this function when samples has already been converted from a 2D array
    per canonical variable in the graph, to a list of lists per canonical variable.
    """
    assert isinstance(temporal_index, int)
    assert isinstance(samples[list(samples.keys())[0]][0], list)

    if isinstance(canonical_variables, str):
        assert isinstance(samples[canonical_variables][temporal_index], list)
        return np.array(samples[canonical_variables][temporal_index]).reshape(-1, 1)
    elif isinstance(canonical_variables, list):
        tmp = []
        for my_var in canonical_variables:
            tmp.append(np.array(samples[my_var][temporal_index]).reshape(-1, 1))
        return np.hstack(tmp)
    else:
        raise ValueError
