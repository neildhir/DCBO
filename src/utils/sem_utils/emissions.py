from copy import deepcopy

from networkx import MultiDiGraph
from numpy import array, hstack, where
from sklearn.neighbors import KernelDensity  #  StatsModels works better
from src.utils.dag_utils.adjacency_matrix_utils import get_emit_and_trans_adjacency_mats

from ..gp_utils import fit_gp


def fit_sem_emit_fncs(G: MultiDiGraph, D_obs: dict) -> dict:
    """
    Fit within time-slice estimated SEM.

    Parameters
    ----------
    G : MultiDiGraph
        Causal DAG
    D_obs : dict
        Observational samples from the true system

    Returns
    -------
    dict
        Dictionary containing the estimated SEM functions
    """

    # Emission adjacency matrix (doesn't contain entries for transition edges)
    emit_adj_mat, _ = get_emit_and_trans_adjacency_mats(G)
    #  Binary matrix which keeps track of which edges have been fitted
    edge_fit_mat = deepcopy(emit_adj_mat)
    T = G.T
    nodes = array(G.nodes())
    fncs = {t: {} for t in range(T)}  # SEM functions

    # Each node in this list is a parent to more than one child node i.e. Y <-- X --> Z
    fork_idx = where(emit_adj_mat.sum(axis=1) > 1)[0]
    fork_nodes = nodes[fork_idx]

    if any(fork_nodes):
        for i, v in zip(fork_idx, fork_nodes):
            #  Get children / independents
            coords = where(emit_adj_mat[i, :] == 1)[0]
            ch = nodes[coords].tolist()  # Gets variables names
            var, t = v.split("_")
            t = int(t)
            xx = D_obs[var][:, t].reshape(-1, 1)  #  Independent regressor
            for j, y in enumerate(ch):
                # Estimand
                var_y, _ = y.split("_")
                yy = D_obs[var_y][:, t].reshape(-1, 1)
                # Fit estimator
                fncs[t][(v, j, y)] = fit_gp(x=xx, y=yy)
                # Update edge tracking matrix (removes entry (i,j) if it has been fitted)
                edge_fit_mat[i, coords[j]] -= 1

    # Assign estimators to source nodes (these don't exist in edge_fit_mat)
    for v in nodes[where(emit_adj_mat.sum(axis=0) == 0)]:
        var, t = v.split("_")
        t = int(t)
        # This is a source node so we need to find the marginal from the observational data.
        xx = D_obs[var][:, t].reshape(-1, 1)
        # Fit estimator
        fncs[t][(None, v)] = KernelDensity(kernel="gaussian").fit(xx)

    # Fit remaining un-estimated edges
    for i, j in zip(*where(edge_fit_mat == 1)):
        pa_y, t_pa = nodes[i].split("_")
        y, t_y = nodes[j].split("_")
        assert t_pa == t_y
        t = int(t_y)
        # Regressor/Independent
        xx = D_obs[pa_y][:, t].reshape(-1, 1)
        # Estimand
        yy = D_obs[y][:, t].reshape(-1, 1)
        #  Fit estimator
        fncs[t][(nodes[i],)] = fit_gp(x=xx, y=yy)
        # Update edge tracking matrix
        edge_fit_mat[i, j] -= 1

    #  The edge-tracking matrix should be zero at the end.
    assert edge_fit_mat.sum() == 0

    # Finally, fit many-to-one function estimates (i.e. nodes with more than one parent) to account for multivariate intervention
    many_to_one = where(emit_adj_mat.sum(axis=0) > 1)[0]
    if any(many_to_one):
        for i, v in zip(many_to_one, nodes[many_to_one]):
            y, y_t = v.split("_")
            t = int(y_t)
            pa_y = nodes[where(emit_adj_mat[:, i] == 1)]
            assert len(pa_y) > 1, (pa_y, y, many_to_one)
            xx = hstack([D_obs[vv.split("_")[0]][:, t].reshape(-1, 1) for vv in pa_y])
            # Estimand
            yy = D_obs[y][:, t].reshape(-1, 1)
            #  Fit estimator
            fncs[t][tuple(pa_y)] = fit_gp(x=xx, y=yy)

    return fncs
