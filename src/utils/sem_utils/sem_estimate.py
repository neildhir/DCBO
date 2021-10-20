from collections import OrderedDict
from typing import Callable

from numpy.random import randn

from ..utilities import select_sample


def auto_sem_hat(
    summary_graph_node_parents: dict,
    emission_functions: dict,
    transition_functions: dict = None,
    independent_causes: dict = None,
) -> classmethod:
    """
    This function is used to automatically create the estimates for the edges in a given graph.

    Parameters
    ----------
    summary_graph_node_parents: Iterable
        A dictionary with causally ordered keys for the summary graph, with values containing the parents of the variables in that graph
    emission_functions : dict
        A dictionary of fitted emission functions.
    transition_functions : dict
        A dictionary of fitted transition functions.
    independent_causes: bool
        Tells the function if the first varible should be treated as an independent cause node (instrument variable) and thus has no incoming edges from the previous time-slice. This is always true for t=0. However, independent causes can appear at any time-slice including on the target node itself.

    Returns
    -------
    classmethod
        A SEM estimate found using observational data only; used in finding the optimal intervention set and values for CBO and DCBO.
    """

    # XXX: this function will eventually be passed the full adjacency matrix for the graph.

    class SEMHat:
        @staticmethod
        def _make_white_noise_fnc() -> Callable:
            #  Instrument variable samples the exogenous model which we take to be white noise
            return lambda: randn()

        @staticmethod
        def _make_static_fnc(moment: int) -> Callable:
            #  No temporal dependence
            return lambda t, emit_input_var, sample: emission_functions[t][emit_input_var].predict(
                select_sample(sample, emit_input_var, t)
            )[moment]

        @staticmethod
        def _make_dynamic_fnc(moment: int) -> Callable:
            #  Temporal dependence
            return (
                lambda t, transfer_input_vars, emit_input_vars, sample: transition_functions[
                    transfer_input_vars
                ].predict(select_sample(sample, transfer_input_vars, t - 1))[moment]
                + emission_functions[t][emit_input_vars].predict(select_sample(sample, emit_input_vars, t))[moment]
            )

        @staticmethod
        def _make_only_dynamic_transfer_fnc(moment: int) -> Callable:
            return lambda t, transfer_input_vars, _, sample: transition_functions[transfer_input_vars].predict(
                select_sample(sample, transfer_input_vars, t - 1)
            )[moment]

        def static(self, moment: int):
            assert moment in [0, 1], moment
            functions = OrderedDict()
            # Assume variables are causally ordered
            for i, v in enumerate(summary_graph_node_parents):
                if i == 0 or not summary_graph_node_parents[v]:
                    # This is how CBO 'views' the graph.
                    # Always sample from the exogenous model at the root node and instrument variables -- unless other models are specified
                    functions[v] = self._make_white_noise_fnc()
                else:
                    functions[v] = self._make_static_fnc(moment)
            return functions

        def dynamic(self, moment: int):
            assert moment in [0, 1], moment
            functions = OrderedDict()
            # Assume variables are causally ordered
            for i, v in enumerate(summary_graph_node_parents):
                if i == 0 and independent_causes[v]:
                    """
                    Instrument variable at the root of the time-slice, without any time dependence
                    o   x Node at time t
                        |
                        v
                        o Child node at time t
                    """
                    functions[v] = self._make_white_noise_fnc()
                elif i == 0 and not independent_causes[v]:
                    """
                    Root node in the time-slice, with time dependence
                    x-->o Node at time t with dependence from time t-1
                    """
                    functions[v] = self._make_only_dynamic_transfer_fnc(moment)
                elif i > 0 and independent_causes[v] and not summary_graph_node_parents[v]:
                    """
                    Instrument variable in the time-slice, without any time dependence
                        o Node at time t
                    x   |
                      \ v
                        o Child node at time t
                    """
                    functions[v] = self._make_white_noise_fnc()
                elif i > 0 and not independent_causes[v] and not summary_graph_node_parents[v]:
                    """
                    Node in the time-slice, with time dependence
                    x-->o Node at time t with dependence from time t-1
                    """
                    functions[v] = self._make_only_dynamic_transfer_fnc(moment)
                else:
                    functions[v] = self._make_dynamic_fnc(moment)
            return functions

    return SEMHat

