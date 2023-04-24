from collections import OrderedDict
from typing import Callable
import numpy as np


def sequential_sample_from_true_SEM(
    static_sem: OrderedDict,
    dynamic_sem: OrderedDict,
    timesteps: int,
    initial_values: dict = None,
    interventions: dict = None,
    epsilon=None,
    seed=None,
) -> OrderedDict:

    if seed is not None:
        np.random.seed(seed)
    # A specific noise-model has not been provided so we use standard Gaussian noise
    if not epsilon:
        epsilon = {k: np.random.randn(timesteps) for k in static_sem.keys()}
    assert isinstance(epsilon, dict)
    # Notice that we call it 'sample' in singular since we only want one sample of the whole graph
    sample = OrderedDict([(k, np.zeros(timesteps)) for k in static_sem.keys()])
    if initial_values:
        assert sample.keys() == initial_values.keys()

    for t in range(timesteps):
        if t == 0 or dynamic_sem is None:
            for var, function in static_sem.items():
                # Check that interventions and initial values at t=0 are not both provided
                if interventions and initial_values:
                    if interventions[var][t] is not None and initial_values[var] is not None:
                        raise ValueError(
                            "You cannot provided an initial value and an intervention for the same location(var,time) in the graph."
                        )
                # If interventions exist they take precedence
                if interventions is not None and interventions[var][t] is not None:
                    sample[var][t] = interventions[var][t]
                # If initial values are passed then we use these, if no interventions
                elif initial_values:
                    sample[var][t] = initial_values[var]
                # If neither interventions nor initial values are provided sample the model with provided epsilon, if exists
                else:
                    sample[var][t] = function(epsilon[var][t], t, sample)
        else:
            for var, function in dynamic_sem.items():
                if interventions is not None and interventions[var][t] is not None:
                    sample[var][t] = interventions[var][t]
                else:
                    sample[var][t] = function(epsilon[var][t], t, sample)

    return sample


def sequential_sample_from_SEM_hat(
    static_sem: OrderedDict,
    dynamic_sem: OrderedDict,
    timesteps: int,
    node_parents: Callable,
    initial_values: dict = None,
    interventions: dict = None,
    seed: int = None,
) -> OrderedDict:
    """
    Function to sequentially sample a dynamic Bayesian network using ESTIMATED SEMs. Currently function approximations are done using Gaussian processes.

    Parameters
    ----------
    static_sem : OrderedDict
        SEMs used at t=0 and used for CBO since CBO does not have access to the dynamic model
    dynamic_sem : OrderedDict
        SEMs used at t>0
    timesteps : int
        Total number of time-steps up until now (i.e. we do not sample the DAG beyond the current time-step)
    node_parents : Callable
        Function with returns parents of the passed argument at the given time-slice
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
    """
    if seed:
        np.random.seed(seed)
    # Notice that we call it 'sample' in singular since we only receive one sample of the whole graph
    sample = OrderedDict([(k, np.zeros(timesteps)) for k in static_sem])
    if initial_values:
        assert sample.keys() == initial_values.keys()

    for t in range(timesteps):
        if t == 0 or dynamic_sem is None:
            for var, function in static_sem.items():
                # Check that interventions and initial values at t=0 are not both provided
                if interventions and initial_values:
                    if interventions[var][t] is not None and initial_values[var] is not None:
                        raise ValueError(
                            "You cannot provided an initial value and an intervention for the same location(var,time) in the graph."
                        )
                # If interventions exist they take precedence
                if interventions and interventions[var][t]:
                    sample[var][t] = interventions[var][t]
                # If initial values are passed then we use these, if no interventions
                elif initial_values:
                    sample[var][t] = initial_values[var]
                # If neither interventions nor initial values are provided; sample the model
                else:
                    V = var + "_" + str(t)
                    if node_parents(V, t):
                        sample[var][t] = function(t, None, node_parents(V, t), sample)
                    else:
                        # Sample source node marginal
                        sample[var][t] = function(t, (None, V))
        else:
            assert dynamic_sem is not None
            for var, function in dynamic_sem.items():
                V = var + "_" + str(t)  # E.g. X_1
                if interventions and interventions[var][t]:
                    sample[var][t] = interventions[var][t]
                # TODO: pass the graph to not have to deal with this.
                elif not node_parents(V, t - 1) and not node_parents(V, t):
                    # Sample source node marginal
                    sample[var][t] = function(t, (None, V))
                # TODO: pass the graph to not have to deal with this.
                else:
                    # input args: time index, parents which are transfer vars, parents which are emission vars and the sample
                    sample[var][t] = function(t, node_parents(V, t - 1), node_parents(V, t), sample)
    return sample


def sequentially_sample_model(
    static_sem,
    dynamic_sem,
    total_timesteps: int,
    initial_values=None,
    interventions=None,
    node_parents=None,
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
            tmp = sequential_sample_from_SEM_hat(
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

            tmp = sequential_sample_from_true_SEM(
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

