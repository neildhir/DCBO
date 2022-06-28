from collections import OrderedDict
from copy import deepcopy
from typing import Callable

from networkx import MultiDiGraph
from numpy import array, hstack, where
from sklearn.neighbors import KernelDensity  # StatsModels works better
from src.utils.dag_utils.adjacency_matrix_utils import get_emit_and_trans_adjacency_mats

from ..gp_utils import fit_gp
from ..utilities import select_sample


def fit_arcs(G: MultiDiGraph, data: dict, emissions: bool) -> dict:
    """
    Fit within (inter) time-slice arcs and between (intra) transitions functions connecting causal dynamic network.

    Note
    ----
    This function assumes that arc relationships are first-order Markov only. The code needs to modified if longer-range dependencies need to be encoded i.e. when modelling $p(X_t \mid X_{t-1}, X_{t-2}$ etc).

    Parameters
    ----------
    G : MultiDiGraph
        Causal DAG
    data: dict
        Observational samples from the true system
    emissions: bool
        Which edge-type we are working with; emissions or transitions - if True we are fitting emissions

    Returns
    -------
    dict
        Dictionary containing the estimated functions
    """

    if emissions:
        # Emission adjacency matrix (doesn't contain entries for transition edges)
        A, _ = get_emit_and_trans_adjacency_mats(G)
    else:
        # Transition adjacency matrix
        _, A = get_emit_and_trans_adjacency_mats(G)

    #  Binary matrix which keeps track of which edges have been fitted
    edge_fit_track_mat = deepcopy(A)
    T = G.T
    nodes = array(G.nodes())
    fncs = {t: {} for t in range(T)}  # Estimated functions

    # Each node in this list is a parent to more than one child node i.e. Y <-- X --> Z; a node with multiple children in other words
    fork_idx = where(A.sum(axis=1) > 1)[0]
    fork_nodes = nodes[fork_idx]

    if any(fork_nodes):
        for i, v in zip(fork_idx, fork_nodes):
            #  Get children / independents
            coords = where(A[i, :] == 1)[0]
            ch = nodes[coords].tolist()  # Get children variable names
            var, t = v.split("_")
            t = int(t)
            xx = data[var][:, t].reshape(-1, 1)  #  Independent regressor
            for j, y in enumerate(ch):
                # Estimand
                var_y, _ = y.split("_")
                yy = data[var_y][:, t].reshape(-1, 1)
                # Fit estimator
                if v.split("_")[1] == y.split("_")[1]:
                    # Emissions
                    fncs[t][(v, j, y)] = fit_gp(x=xx, y=yy)
                else:
                    # Transitions
                    fncs[t + 1][(v, j, y)] = fit_gp(x=xx, y=yy)
                # Update edge tracking matrix (removes entry (i,j) if it has been fitted)
                edge_fit_track_mat[i, coords[j]] -= 1

    if emissions:
        # Assign estimators to source/root nodes (these don't exist in edge_fit_mat)
        for v in nodes[where(A.sum(axis=0) == 0)]:
            var, t = v.split("_")
            t = int(t)
            # This is a source node so we need to find the marginal from the observational data.
            xx = data[var][:, t].reshape(-1, 1)
            # Fit estimator
            fncs[t][(None, v)] = KernelDensity(kernel="gaussian").fit(xx)

    # Fit remaining un-estimated edges
    for i, j in zip(*where(edge_fit_track_mat == 1)):
        pa_y, t_pa = nodes[i].split("_")
        y, t_y = nodes[j].split("_")
        if emissions:
            #  To fit this function emission pairs must live within the same slice
            assert t_pa == t_y, (i, j, nodes[i], nodes[j])
        else:
            #  Transitions per definition are defined between time indices
            assert t_pa != t_y, (i, j, nodes[i], nodes[j])
        t = int(t_y)
        # Regressor/Independent
        xx = data[pa_y][:, t].reshape(-1, 1)
        # Estimand
        yy = data[y][:, t].reshape(-1, 1)
        #  Fit estimator
        fncs[t][(nodes[i],)] = fit_gp(x=xx, y=yy)
        # Update edge tracking matrix
        edge_fit_track_mat[i, j] -= 1

    #  The edge-tracking matrix should be zero at the end.
    assert edge_fit_track_mat.sum() == 0

    # Finally, fit many-to-one function estimates (i.e. nodes with more than one parent) to account for multivariate intervention
    many_to_one = where(A.sum(axis=0) > 1)[0]
    if any(many_to_one):
        for i, v in zip(many_to_one, nodes[many_to_one]):
            y, y_t = v.split("_")
            t = int(y_t)
            pa_y = nodes[where(A[:, i] == 1)]
            assert len(pa_y) > 1, (pa_y, y, many_to_one)
            xx = hstack([data[vv.split("_")[0]][:, t].reshape(-1, 1) for vv in pa_y])
            # Estimand
            yy = data[y][:, t].reshape(-1, 1)
            #  Fit estimator
            if y_t == pa_y[0].split("_")[1]:
                # Emissions
                fncs[t][tuple(pa_y)] = fit_gp(x=xx, y=yy)
            else:
                # Transitions
                assert t != 0, (t, pa_y, y)
                fncs[t][tuple(pa_y)] = fit_gp(x=xx, y=yy)

    return fncs


def build_sem_hat(G: MultiDiGraph, emission_fncs: dict, transition_fncs: dict = None,) -> classmethod:
    """
    This function is used to automatically create the SEM -function estimates for the edges in a given graph.

    Parameters
    ----------
    G : MultiDiGraph
        Causal graphical model
    emission_functions : dict
        A dictionary of fitted emission functions (roughly most horizontal edges in the DAG).
    transition_functions : dict
        A dictionary of fitted transition functions (roughly most vertical edges in the DAG).

    Notes
    -----
    We have _NOT_ covered all network topologies with this function. Beware.

    Returns
    -------
    classmethod
        A SEM estimate found using observational data only; used in finding the optimal intervention set and values for CBO and DCBO.
    """

    class SEMHat:
        def __init__(self):
            self.G = G
            nodes = G.nodes()
            n_t = len(nodes) / G.T  # Number of nodes per time-slice
            assert n_t.is_integer()
            self.n_t = int(n_t)

        @staticmethod
        def _make_marginal() -> Callable:
            #  Assigns the KDE for the marginal
            return lambda t, margin_id: emission_fncs[t][margin_id].sample()

        @staticmethod
        def _make_emit_fnc(moment: int) -> Callable:
            #  Within time-slice emission only
            return lambda t, _, emit_input_vars, sample: emission_fncs[t][emit_input_vars].predict(
                select_sample(sample, emit_input_vars, t)
            )[moment]

        @staticmethod
        def _make_trans_fnc(moment: int) -> Callable:
            #  Only transition between time-slices (only valid for first-order Markov assumption)
            return lambda t, transfer_input_vars, _, sample: transition_fncs[t][transfer_input_vars].predict(
                select_sample(sample, transfer_input_vars, t - 1)
            )[moment]

        @staticmethod
        def _make_emit_plus_trans_fnc(moment: int) -> Callable:
            #  Transition plus within-slice emission(s)
            return (
                lambda t, transfer_input_vars, emit_input_vars, sample: transition_fncs[t][transfer_input_vars].predict(
                    select_sample(sample, transfer_input_vars, t - 1)
                )[moment]
                + emission_fncs[t][emit_input_vars].predict(select_sample(sample, emit_input_vars, t))[moment]
            )

        def static(self, moment: int) -> OrderedDict:
            assert moment in [0, 1], moment
            # SEM functions
            f = OrderedDict()
            # Assumes that the keys are causally ordered
            for v in list(self.G.nodes)[: self.n_t]:
                vv = v.split("_")[0]
                if self.G.in_degree[v] == 0:
                    f[vv] = self._make_marginal()
                else:
                    f[vv] = self._make_emit_fnc(moment)
            return f

        def dynamic(self, moment: int) -> OrderedDict:
            assert moment in [0, 1], moment
            # SEM functions
            f = OrderedDict()
            # Variables are causally ordered
            for v in list(self.G.nodes)[self.n_t : 2 * self.n_t]:
                vv = v.split("_")[0]
                if self.G.in_degree[v] == 0:
                    # Single source node
                    f[vv] = self._make_marginal()
                elif all(int(vv.split("_")[1]) + 1 == int(v.split("_")[1]) for vv in G.predecessors(v)):
                    # Depends only on incoming transition edge(s)
                    f[vv] = self._make_trans_fnc(moment)
                elif all(vv.split("_")[1] == v.split("_")[1] for vv in G.predecessors(v)):
                    # Depends only on incoming emission edge(s) from this time-slice
                    f[vv] = self._make_emit_fnc(moment)
                else:
                    # Depends incoming emission AND transition edges
                    f[vv] = self._make_emit_plus_trans_fnc(moment)
            return f

    return SEMHat
