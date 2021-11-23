from networkx.classes import MultiDiGraph
from src.utils.dag_utils.adjacency_matrix_utils import get_emit_and_trans_adjacency_mats
from ..gp_utils import fit_gp
from numpy import array, hstack, where


def fit_sem_trans_fncs(G: MultiDiGraph, D_obs: dict) -> dict:
    """
    Fit transition functions connecting time-slices in causal Bayesian network.

    OBS: this function assumes that transition relationships are first-order Markov only. The code needs to modified if longer-range dependencies need to be encoded i.e. we are modelling p(X_t | X_{t-1}, X_{t-2} etc).

    Parameters
    ----------
    G : multidigraph
        DAG
    D_obs : dict
        Observational data samples from true SEM

    Returns
    -------
    dict
        A dictionary of transition functions.
    """
    # Emission adjacency matrix
    _, trans_adj_mat = get_emit_and_trans_adjacency_mats(G)
    nodes = array(G.nodes())
    # Number of nodes per time-slice
    v_n = len(nodes) / G.T
    assert v_n.is_integer()
    fncs = {t: {} for t in range(G.T)}

    for i, v in enumerate(nodes[int(v_n) :], start=G.T):
        var, time = v.split("_")
        t = int(time)
        assert t > 0
        # Estimand
        yy = D_obs[var][:, t].reshape(-1, 1)
        # Parents of estimand variable, we could use G.predecessors(v) but it includes the emission variables.
        pa_y = nodes[where(trans_adj_mat[:, i] == 1)]
        assert all([vv.split("_")[1] == str(t - 1) for vv in pa_y])
        if len(pa_y) == 0:
            # This node has no incoming edges from the past time-slice
            continue
        else:
            xx = hstack([D_obs[vv.split("_")[0]][:, t - 1].reshape(-1, 1) for vv in pa_y])
            #  Fit estimator
            fncs[t][tuple(pa_y)] = fit_gp(x=xx, y=yy)

    return fncs
