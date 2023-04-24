from copy import deepcopy
import numpy as np
from .intervention_assignments import assign_initial_intervention_level, assign_intervention_level
from .sequential_sampling import sequential_sample_from_true_SEM
from networkx import MultiDiGraph


def create_n_dimensional_intervention_grid(limits: list, size_intervention_grid: int = 100):
    """
    Usage: combine_n_dimensional_intervention_grid([[-2,2],[-5,10]],10)
    """
    if any(isinstance(el, list) for el in limits) is False:
        # We are just passing a single list
        return np.linspace(limits[0], limits[1], size_intervention_grid)[:, None]
    else:
        extrema = np.vstack(limits)
        inputs = [np.linspace(i, j, size_intervention_grid) for i, j in zip(extrema[:, 0], extrema[:, 1])]
        return np.dstack(np.meshgrid(*inputs)).ravel("F").reshape(len(inputs), -1).T


def get_interventional_grids(exploration_set, intervention_limits, size_intervention_grid=100) -> dict:
    """Builds the n-dimensional interventional grids for the respective exploration sets.

    Parameters
    ----------
    exploration_set : iterable
        All the exploration sets
    intervention_limits : [type]
        The intervention range per canonical manipulative variable in the causal graph.
    size_intervention_grid : int, optional
        The size of the intervention grid (i.e. number of points on the grid)

    Returns
    -------
    dict
        Dict containing all the grids, indexed by the exploration sets
    """

    # Create grids
    intervention_grid = {k: None for k in exploration_set}
    for es in exploration_set:
        if len(es) == 1:
            # Notice that we are splitting by the underscore and this is a hard-coded feature
            intervention_grid[es] = create_n_dimensional_intervention_grid(
                intervention_limits[es[0]], size_intervention_grid
            )
        else:
            if size_intervention_grid >= 100 and len(es) > 2:
                size_intervention_grid = 10

            intervention_grid[es] = create_n_dimensional_intervention_grid(
                [intervention_limits[j] for j in es], size_intervention_grid
            )

    return intervention_grid


def reproduce_empty_intervention_blanket(T, keys):
    return {key: T * [None] for key in keys}


def evaluate_target_function(
    initial_structural_equation_model, structural_equation_model, graph, exploration_set: tuple, all_vars, T: int,
):
    # Initialise temporal intervention dictionary
    intervention_blanket = make_sequential_intervention_dict(graph, T)
    keys = intervention_blanket.keys()

    def compute_target_function(current_target: str, intervention_levels: np.array, assigned_blanket: dict):

        # Split current target
        target_canonical_variable, target_temporal_index = current_target.split("_")
        target_temporal_index = int(target_temporal_index)

        # Populate the blanket in place
        if target_temporal_index == 0:
            intervention_blanket = reproduce_empty_intervention_blanket(T, keys)
            assign_initial_intervention_level(
                exploration_set=exploration_set,
                intervention_level=intervention_levels,
                intervention_blanket=intervention_blanket,
                target_temporal_index=target_temporal_index,
            )
        else:
            # Takes into account the interventions, assignments and target outcomes from the {t-1}
            intervention_blanket = deepcopy(assigned_blanket)
            assign_intervention_level(
                exploration_set=exploration_set,
                intervention_level=intervention_levels,
                intervention_blanket=intervention_blanket,
                target_temporal_index=target_temporal_index,
            )

        static_noise_model = {k: np.zeros(T) for k in list(all_vars)}

        interventional_samples = sequential_sample_from_true_SEM(
            static_sem=initial_structural_equation_model,
            dynamic_sem=structural_equation_model,
            timesteps=T,
            epsilon=static_noise_model,
            interventions=intervention_blanket,
        )

        # Compute the effect of intervention(s)
        target_response = compute_sequential_target_function(
            intervention_samples=interventional_samples,
            temporal_index=target_temporal_index,
            target_variable=target_canonical_variable,
        )
        return target_response.mean()

    return compute_target_function


def compute_sequential_target_function(
    intervention_samples: np.array, temporal_index: int, target_variable: str = "Y"
) -> np.array:
    if ~isinstance(temporal_index, int):
        temporal_index = int(temporal_index)
    # Function calculates the target provided the time-index is correct
    # assert intervention_samples[target_variable].shape[1]
    return intervention_samples[target_variable][temporal_index]


def make_sequential_intervention_dict(G: MultiDiGraph, T: int) -> dict:
    """
    Makes an intervention dictionary so that we know _where_ (var) and _when_ (time step) to intervene and with what magnitude

    Parameters
    ----------
    G : MultiDiGraph
        A structural causal graph
    T : int
        Total time-series length

    Returns
    -------
    dict
        Dictionary of (empty) sequential interventions
    """
    nodes = "".join(G.nodes)
    variables = sorted(set([s for s in nodes if s.isalpha()]))
    return {v: T * [None] for v in variables}
