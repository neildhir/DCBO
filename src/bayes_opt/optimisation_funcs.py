import numpy as np
import itertools
import scipy


def calculate_convex_hull_ratio(graph, observations_variables, N_max=None) -> float:
    """
    Open questions:
    1. Currently unclear what to do if we intervene on variables with different dimensionality e.g. univariate and multivariate intervention mixing.
    2. Does this work when we update the number of observations taken?

    usage:

    tg = ToyGraph(observational_samples)
    calculate_convex_hull_ratio(tg, [X,Z])
    """

    # List of variables that have been observed to some degree
    if len(observations_variables) >= 2:
        assert isinstance(observations_variables, tuple) or isinstance(observations_variables, list)
        # This is not necessarily a strict criterion, but only for the toy example
        assert all(x.shape == observations_variables[0].shape for x in observations_variables)

    # From specified graph get the internventional ranges
    interventional_ranges = list(graph.get_interventional_ranges().values())
    if observations_variables[0].shape[1] == 1:
        # Cartesian product of interventional ranges for univariate manipulative variables
        points = np.array(list(itertools.product(*interventional_ranges)))
    else:
        # Cartesian product for multivariate manipulative variables
        points = np.array([sum(i, ()) for i in list(itertools.product(*interventional_ranges))])

    # Calculate coverage of the observations
    obs_points = np.hstack(observations_variables)  # [X,Y,Z,...]
    assert obs_points.shape[1] == points.shape[1]
    # Convex hull calculation done here
    hull_ratio = scipy.spatial.ConvexHull(obs_points).volume / scipy.spatial.ConvexHull(points).volume
    if N_max:
        # Return epsilon factor
        return hull_ratio * (observations_variables[0].shape[0] / N_max)
    else:
        return hull_ratio
