from collections import OrderedDict
from networkx.classes.multidigraph import MultiDiGraph
import numpy as np


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


def sequential_sample_from_model_hat_v2(
    static_sem: OrderedDict,
    dynamic_sem: OrderedDict,
    timesteps: int,
    G: MultiDiGraph,
    initial_values: dict = None,
    interventions: dict = None,
) -> OrderedDict:
    """
    Function to sequentially sample a dynamic Bayesian network using ESTIMATED SEMs. Currently function approximations are done using Gaussian processes.

    Parameters
    ----------
    static_sem : OrderedDict
        SEMs used at t=0
    dynamic_sem : OrderedDict
        SEMs used at t>0
    timesteps : int
        Total number of time-steps up until now (i.e. we do not sample the DAG beyond the current time-step)
    G : MultiDiGraph
        Causal DAG
    initial_values : dict, optional
        Initial values of nodes at t=0, by default None
    interventions : dict, optional
        Blanket which contains the interventions implemented thus far, by default None

    Returns
    -------
    OrderedDict
        A sample from the CBN given previously implemented interventions as well as the current one

    Raises
    ------
    ValueError
        If internventions and initial values are passed at t=0 -- they are equivalent so both cannot be passed.
    ValueError
        [description]
    """

    # Notice that we call it 'sample' in singular since we only receive one sample of the whole graph
    sample = OrderedDict([(k, np.zeros(timesteps)) for k in static_sem.keys()])

    if initial_values:
        assert sample.keys() == initial_values.keys()

    for t in range(timesteps):
        if t == 0 or dynamic_sem is None:
            for var, function in static_sem.items():
                time = str(t)
                node = var + "_" + time

                # Check that interventions and initial values at t=0 are not both provided
                if interventions and initial_values:
                    if interventions[var][t] is not None and initial_values[var] is not None:
                        raise ValueError(
                            "You cannot provided an initial value "
                            "and an intervention for the same location "
                            "(var,time) in the graph."
                        )

                # If interventions exist they take precedence
                if interventions and interventions[var][t]:
                    sample[var][t] = interventions[var][t]

                # If initial values are passed then we use these, if no interventions
                elif initial_values:
                    sample[var][t] = initial_values[var]

                # If neither interventions nor initial values are provided; sample the model
                else:
                    pa = G.predecessors(node)
                    if pa:
                        if t == 0:
                            sample[var][t] = function(t, pa, sample)
                        else:
                            emit_vars = (*[v for v in pa if v.split("_")[1] == time],)
                            if emit_vars:
                                sample[var][t] = function(t, emit_vars, sample)
                            else:
                                #  Sample source node marginal
                                sample[var][t] = function(t, (None, var), None)
                    else:
                        #  Sample source node marginal
                        sample[var][t] = function(t, (None, var), None)

        else:
            assert dynamic_sem is not None
            # Dynamically propagate the samples through the graph using estimated SEMs. If we have found an optimal interventional target response from t-1, it has been included in the intervention dictionary at the correct index.

            for var, function in dynamic_sem.items():
                time = str(t)
                node = var + "_" + time

                #  XXX: note the weird syntax used here; it converts a list to a tuple (e.g. Y_0 --> Y_1)
                transfer_vars = (*[v for v in node_parents[node] if v.split("_")[1] != time],)
                # Get conditioning variables from conditional distribution (input-variables at this time-step only)
                emit_vars = (*[v for v in node_parents[node] if v.split("_")[1] == time],)

                #  Sample
                if interventions and interventions[var][t]:
                    sample[var][t] = interventions[var][t]
                elif node_parents[node]:
                    sample[var][t] = function(t, transfer_vars, emit_vars, sample)
                else:
                    sample[var][t] = function()

    return sample


def sequential_sample_from_model_hat(
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

                # If neither interventions nor initial values are provided; sample the model
                else:
                    if node_parents[node]:
                        if temporal_index == 0:
                            sample[var][temporal_index] = function(temporal_index, node_parents[node], sample)
                        else:
                            emit_vars = (*[v for v in node_parents[node] if v.split("_")[1] == time],)
                            if emit_vars:
                                sample[var][temporal_index] = function(temporal_index, emit_vars, sample)
                            elif not emit_vars:
                                # TODO: this needs to be replaced by the marginal
                                sample[var][temporal_index] = function()
                            else:
                                raise ValueError("There are no parents!", emit_vars)
                    else:
                        # TODO: this needs to be replaced by the marginal
                        sample[var][temporal_index] = function()

        else:
            assert dynamic_sem is not None
            # Dynamically propagate the samples through the graph using estimated SEMs. If we have found an optimal interventional target response from t-1, it has been included in the intervention dictionary at the correct index.

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
    for i in range(sample_count):
        # This option uses the estimates of the SEMs, estimates found through use of GPs.
        if use_sem_estimate:
            tmp = sequential_sample_from_model_hat(
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

