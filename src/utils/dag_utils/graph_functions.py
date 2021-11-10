from graphviz import Source
from collections import Counter
from numpy import repeat
from itertools import cycle, chain
from networkx import topological_sort, MultiDiGraph
from typing import Dict, Tuple, List, Union


def make_graphical_model(
    start_time: int, stop_time: int, topology: str, nodes: List[str], target_node: str = None, verbose: bool = False
) -> Union[MultiDiGraph, str]:
    """
    Generic temporal Bayesian network with two types of connections.

    Parameters
    ----------
    start : int
        Index of first time-step
    stop : int
        Index of the last time-step
    topology: str, optional
        Choice of independent and dependent causal topology
    nodes: list
        List containing the nodes of the time-slice of the CGM e.g. nodes=['X', 'Z', 'Y']
    target_node: str, optional
        If we are using a independent spatial topology then we need to specify the target node
    verbose : bool, optional
        To print the graph or not.

    Returns
    -------
    Union[MultiDiGraph, str]
        Returns the DOT format of the graph or a networkx object
    """

    assert start_time <= stop_time
    assert topology in ["dependent", "independent"]
    assert nodes

    if topology == "independent":
        assert target_node is not None
        assert isinstance(target_node, str)

    ## Time-slice connections

    spatial_edges = []
    ranking = []
    # Check if target node is in the list of nodes, and if so remove it
    if topology == "independent":
        if target_node in nodes:
            nodes.remove(target_node)
        node_count = len(nodes)
        assert target_node not in nodes
        connections = node_count * "{}_{} -> {}_{}; "
        edge_pairs = list(sum([(item, target_node) for item in nodes], ()))
    else:
        node_count = len(nodes)
        connections = (node_count - 1) * "{}_{} -> {}_{}; "
        edge_pairs = [item for pair in list(zip(nodes, nodes[1:])) for item in pair]

    pair_count = len(edge_pairs)

    if topology == "independent":
        # X_0 --> Y_0; Z_0 --> Y_0
        all_nodes = nodes + [target_node]
        for t in range(start_time, stop_time + 1):
            space_idx = pair_count * [t]
            iters = [iter(edge_pairs), iter(space_idx)]
            inserts = list(chain(map(next, cycle(iters)), *iters))
            spatial_edges.append(connections.format(*inserts))
            ranking.append("{{ rank=same; {} }} ".format(" ".join([item + "_{}".format(t) for item in all_nodes])))
    elif topology == "dependent":
        # X_0 --> Z_0; Z_0 --> Y_0
        for t in range(start_time, stop_time + 1):
            space_idx = pair_count * [t]
            iters = [iter(edge_pairs), iter(space_idx)]
            inserts = list(chain(map(next, cycle(iters)), *iters))
            spatial_edges.append(connections.format(*inserts))
            ranking.append("{{ rank=same; {} }} ".format(" ".join([item + "_{}".format(t) for item in nodes])))
    else:
        raise ValueError("Not a valid spatial topology.")

    ranking = "".join(ranking)
    spatial_edges = "".join(spatial_edges)

    ## Temporal connections

    temporal_edges = []
    if topology == "independent":
        node_count += 1
        nodes += [target_node]

    connections = node_count * "{}_{} -> {}_{}; "
    for t in range(stop_time):
        edge_pairs = repeat(nodes, 2).tolist()
        temporal_idx = node_count * [t, t + 1]
        iters = [iter(edge_pairs), iter(temporal_idx)]
        inserts = list(chain(map(next, cycle(iters)), *iters))
        temporal_edges.append(connections.format(*inserts))

    temporal_edges = "".join(temporal_edges)

    graph = "digraph {{ rankdir=LR; {} {} {} }}".format(spatial_edges, temporal_edges, ranking)

    if verbose:
        return Source(graph)
    else:
        return graph


def get_independent_causes(time_slice_vars: List[str], G: MultiDiGraph) -> Dict[str, bool]:
    """
    Function to find the "independent causes" in each time-slice. These are variables which have no dependence on the other nodes or on the past. But they are still part of endogenous model. They are potential instrument variables but can also just be nodes that affect the target-variable directly.

    Parameters
    ----------
    time_slice_vars : list
        List of variables that belong to the time-slice.
    G : MultiDiGraph
        Causal graphical model of networkx type.

    Returns
    -------
    Dict[str, bool]
        Dict which tells which vars are independent causes.
    """

    independent_causes = {v: False for v in time_slice_vars}

    # Get induced sub-graph from first two time-slices
    gg = G.subgraph([v + "_0" for v in time_slice_vars] + [v + "_1" for v in time_slice_vars])
    # Find parents of all nodes in this sub-graph
    gg_parents = {v: tuple([vv for vv in gg.predecessors(v)]) for v in gg.nodes}
    # Check which nodes are independent singelton cause variables in both time-slices
    possible_singletons = [v.split("_")[0] for v in [key for key in gg if not gg_parents[key]]]
    singleton_causes = [v for v in set(possible_singletons) if possible_singletons.count(v) > 1]
    #  Indicate found singleton
    for v in singleton_causes:
        independent_causes[v] = True

    return independent_causes


def get_summary_graph_node_parents(time_slice_vars: List[str], G: MultiDiGraph) -> Tuple[Dict[str, tuple], List[str]]:
    """
    Finds the summary graph of the CBN a la page 199 of `Elements of Causal Inference'.

    Parameters
    ----------
    time_slice_vars : List[str]
        List of variables that belong to the time-slice.
    G : MultiDiGraph
        Causal graphical model of networkx type.

    Returns
    -------
    Tuple[Dict[str, tuple], List[str]]
        The parents of the nodes in the summary graph and a list of the variables in the time-slice, but causally ordered using topological sort.
    """
    gg = G.subgraph([v + "_0" for v in time_slice_vars])
    #  Causally ordered nodes in the first time-slice
    causal_order = list(v.split("_")[0] for v in topological_sort(gg))
    #  See page 199 of 'Elements of Causal Inference' for a reference on summary graphs.
    summary_graph_node_parents = {
        v.split("_")[0]: tuple([vv.split("_")[0] for vv in gg.predecessors(v)]) for v in gg.nodes
    }
    #  Re-order dict to follow causal order of time-slices
    summary_graph_node_parents = {k: summary_graph_node_parents[k] for k in causal_order}

    return summary_graph_node_parents, causal_order


def get_subgraph(t: int, time_slice_vars: set, G: MultiDiGraph) -> MultiDiGraph:
    """Extracts the sub-graph related to a given time-slice indexed by t.

    Parameters
    ----------
    t : int
        Time index for which we want the time-slice.
    time_slice_vars : set
        The [un-indexed] vertices of the time-slice e.g. ['X','Z','Y'] -- for the SCM example used throughout the paper.
    G : MultiDiGraph
        DAG

    Returns
    -------
    MultiDiGraph
        Sub-graph of the time-index.
    """
    return G.subgraph([v + "_{}".format(t) for v in time_slice_vars])


def query_common_cause(summary_graph_node_parents: dict) -> Union[str, list]:
    # Counts how many times each value appears
    cnt = Counter(summary_graph_node_parents.values())
    #  If this list is populated then one or more common cause relationships exist in the time-slice
    common_cause = [k for (k, v) in cnt.items() if v > 1]

    if len(common_cause) > 1:
        raise NotImplementedError("Graphs this complex have not yet been catered for.")
    elif len(common_cause) == 0:
        return common_cause
    else:
        # Returns the nodes which have the common cause
        return [k for k, v in summary_graph_node_parents.items() if v == common_cause[0]]

