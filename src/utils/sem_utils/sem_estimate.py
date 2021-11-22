from collections import OrderedDict
from typing import Callable
from networkx import MultiDiGraph

from ..utilities import select_sample


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
        A dictionary of fitted transition functions (roughly most vertical edges in the DAG)..

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
            return lambda t, emit_input_vars, sample: emission_fncs[t][emit_input_vars].predict(
                select_sample(sample, emit_input_vars, t)
            )[moment]

        @staticmethod
        def _make_trans_fnc(moment: int) -> Callable:
            #  Only transition between time-slices (only valid for first-order Markov assumption)
            return lambda t, transfer_input_vars, _, sample: transition_fncs[transfer_input_vars].predict(
                select_sample(sample, transfer_input_vars, t - 1)
            )[moment]

        @staticmethod
        def _make_emit_plus_trans_fnc(moment: int) -> Callable:
            #  Transition plus within-slice emission(s)
            return (
                lambda t, transfer_input_vars, emit_input_vars, sample: transition_fncs[transfer_input_vars].predict(
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
