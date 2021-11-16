from networkx.classes import multidigraph
from src.utils.dag_utils.adjacency_matrix_utils import get_emit_and_trans_adjacency_mats
from ..gp_utils import fit_gp
from typing import Dict
from numpy import array, hstack, where


def fit_sem_trans_fncs(G: multidigraph, D_obs: dict) -> dict:
    # OBS: this function assumes that transition relationships are first-order Markov only. The code needs to modified if longer-range dependencies need to be encoded.

    T = G.T
    # Emission adjacency matrix
    _, trans_adj_mat = get_emit_and_trans_adjacency_mats(G)
    nodes = array(G.nodes())
    # Number of nodes per time-slice
    v_n = len(nodes) / T
    assert v_n.is_integer()
    fncs = {t: {} for t in range(T)}

    for i, v in enumerate(nodes[int(v_n) :], start=T):
        var, t = v.split("_")
        t = int(t)
        # Parents of estimand variable, we could use G.predecessors(v) but it includes the emission variables.
        pa_y = [vv.split("_")[0] for vv in nodes[where(trans_adj_mat[:, i] == 1)[0]]]
        # Estimand
        yy = D_obs[var][:, t].reshape(-1, 1)
        if len(pa_y) == 0:
            # This node has no incoming edges from the past time-slice
            continue
        else:
            xx = hstack([D_obs[vv][:, t].reshape(-1, 1) for vv in pa_y])
            #  Fit estimator
            fncs[t][tuple(pa_y)] = fit_gp(x=xx, y=yy)

    return fncs


def fit_sem_trans_fncs_old(observational_samples, transfer_pairs: dict) -> dict:

    # Store function which concern t-1 --> t
    transition_functions = {}

    for input_vars in transfer_pairs:
        # Transfer input
        if len(input_vars) > 1:
            # many to one mapping
            iin_vars = [var.split("_")[0] for var in input_vars]
            iin_times = [int(var.split("_")[1]) for var in input_vars]
            # Auto-regressive structure
            xx = []
            for in_var, in_time in zip(iin_vars, iin_times):
                xx.append(observational_samples[in_var][:, in_time].reshape(-1, 1))
            xx = hstack(xx)
        else:
            in_var, in_time = input_vars[0].split("_")
            xx = observational_samples[in_var][:, int(in_time)].reshape(-1, 1)

        # Transfer target
        output = transfer_pairs[input_vars]
        out_var, out_time = output.split("_")
        yy = observational_samples[out_var][:, int(out_time)].reshape(-1, 1)
        # Store funcs in dict for later usage
        transition_functions[input_vars] = fit_gp(x=xx, y=yy)

    return transition_functions


def get_transition_input_output_pairs(node_parents: dict) -> Dict[tuple, tuple]:

    #  Find all inputs and outputs for transition functions
    transfer_pairs = {}
    for node in node_parents:
        _, time = node.split("_")
        if node_parents[node] and time > "0":
            tmp = [parent for parent in node_parents[node] if parent.split("_")[1] != time]
            assert len(tmp) != 0, (node, node_parents[node], tmp)
            transfer_pairs[node] = tmp

    # Flip keys and values to get explicit input-output order
    transfer_pairs = dict((tuple(v), k) for k, v in transfer_pairs.items())

    return transfer_pairs
