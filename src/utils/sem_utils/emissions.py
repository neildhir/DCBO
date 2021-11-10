from copy import deepcopy
from itertools import combinations
from typing import Dict, Tuple
from networkx import MultiDiGraph
from numpy.core import hstack

from ..gp_utils import fit_gp
from ..utilities import update_emission_pairs_keys
from ..dag_utils.graph_functions import get_subgraph


def fit_sem_emit_fncs(observational_samples: dict, emission_pairs: dict) -> dict:
    """
    Function fits a Gaussian process to all source-sink (cause-effect) relationships in the causal Bayesian network provided.

    Parameters
    ----------
    observational_samples : dict
        Contains time-series (though not required to have time-stamp) of each summary-graph variable in the CBN.
    emission_pairs : dict
        Dictionary contains the source-sink relationships on the items.

    Returns
    -------
    dict
        Dictionary containing all the fitted GPs.
    """

    #  This is an ordered list
    timesteps = observational_samples[list(observational_samples.keys())[0]].shape[1]
    fncs = {t: {key: None for key in emission_pairs.keys()} for t in range(timesteps)}

    for input_nodes in emission_pairs.keys():

        output_tuple = False
        if isinstance(emission_pairs[input_nodes], tuple):
            #  This does not mean an undetermined system, it means that if we have e.g 'X': ('Z','Y') as as an item in emission_pairs then we are looking for two functions: f_1:'X'-->'Z' and f_2:'X'-->'Y'.
            target_variable = emission_pairs[input_nodes]
            output_tuple = True
        else:
            target_variable = emission_pairs[input_nodes].split("_")[0]

        if len(input_nodes) > 1:
            xx = []
            for node in input_nodes:
                start_node, time = node.split("_")
                time = int(time)
                #  Input
                x = observational_samples[start_node][:, time].reshape(-1, 1)
                xx.append(x)
            xx = hstack(xx)
            #  Output
            yy = observational_samples[target_variable][:, time].reshape(-1, 1)

        elif len(input_nodes) == 1:
            start_node, time = input_nodes[0].split("_")
            time = int(time)
            #  Input
            xx = observational_samples[start_node][:, time].reshape(-1, 1)

            if output_tuple:
                #  This option happens when a vertex has multiple outgoing edges in the same time-slice.

                # Output
                yy_obs = {}
                for target in target_variable:
                    var = target.split("_")[0]
                    yy_obs[var] = observational_samples[var][:, time].reshape(-1, 1)
            else:
                #  This is the standard option when only one edge leaves each node in each time-slice.

                # Output
                yy = observational_samples[target_variable][:, time].reshape(-1, 1)

        else:
            raise ValueError("The length of the tuple is: {}".format(len(input_nodes)))

        #  Basic checks
        assert len(xx.shape) == 2
        if output_tuple:
            for target in yy_obs:
                assert len(yy_obs[target].shape) == 2
        else:
            assert len(yy.shape) == 2

        if input_nodes in fncs[time]:
            if output_tuple:
                #  We map one function to each item in the output, with the input
                fncs[time][input_nodes] = {}
                for target, yy in yy_obs.items():
                    fncs[time][input_nodes][target] = fit_gp(x=xx, y=yy)
            else:
                #  Default option
                fncs[time][input_nodes] = fit_gp(x=xx, y=yy)
        else:
            raise ValueError(input_nodes)

    # To remove any None values
    return {t: {k: v for k, v in fncs[t].items() if v is not None} for t in range(timesteps)}


def get_emissions_input_output_pairs(
    T: int, G: MultiDiGraph
) -> Tuple[Dict[str, tuple], Dict[str, tuple], Dict[str, tuple]]:
    """
    Creates input and output pairs for the emission functions - i.e. the functions that operate within a time-slice.

    Parameters
    ----------
    T : int
        The total number of time-steps in the causal Bayesian network.
    G : MultiDiGraph
        DAG

    Returns
    -------
    Tuple[Dict[str, tuple], Dict[str, tuple], Dict[str, tuple]]
        Children, parents and pairs

    Obs
    -------
    The intended usage for this function is when the emission vertice outgoing degree is exactly equal to one. E.g. a DAG structure like X <-- Z --> Y is supported but X <-- Z --> Y <-- W is currently not.
    """

    node_children = {node: None for node in G.nodes}
    node_parents = {node: None for node in G.nodes}

    # Children of all nodes
    for node in G.nodes:
        node_children[node] = list(G.successors(node))

    #  Parents of all nodes
    for node in G.nodes:
        node_parents[node] = tuple(G.predecessors(node))

    time_slice_vars = set(v[0] for v in G.nodes)
    nodes_with_high_out_degree = []
    subgraphs = []
    emission_pairs = {}
    # Check if we have any complex vertex relationships (i.e. out-degree in time-slice > 1) in each time-slice (allowing for a non-stationary CBN).
    for t in range(T):
        # Subgraph for time-slice t
        gg = get_subgraph(t, time_slice_vars, G)
        subgraphs.append(gg)
        #  Get all nodes which has an out degree higher than 1
        Vs = [v for v in gg.out_degree if gg.out_degree[v[0]] > 1]
        nodes_with_high_out_degree.append(Vs)

    if len(nodes_with_high_out_degree) == 0:
        # The normal case where there are no special vertex cases with high out-degree.
        emissions = get_emissions(T, G)
        new_emissions = deepcopy(emissions)
        for t in range(T):
            for a, b in combinations(emissions[t], 2):
                if a[1] == b[1]:
                    new_emissions[t].append(((a[0], b[0]), a[1]))
                    cond = [v for v in list(G.predecessors(b[0])) if v.split("_")[1] == str(t)]
                    if len(cond) != 0 and cond[0] == a[0]:
                        # Remove from list
                        new_emissions[t].remove(a)

        for t in range(T):
            for pair in new_emissions[t]:
                if isinstance(pair[0], tuple):
                    emission_pairs[pair[0]] = pair[1]
                else:
                    emission_pairs[(pair[0],)] = pair[1]

        # Sometimes the input and output pair order does not match because of NetworkX internal issues, so we need adjust the keys so that they do match.
        emission_pairs = update_emission_pairs_keys(T, node_parents, emission_pairs)

    else:
        # XXX: currently only allows for one-to-one mappings from one vertex to another vertex. Subsets of outgoing edges, heading for different targets, is currently not supported when vertices have a higher degree than one. E.g. a DAG structure like X <-- Z --> Y is supported but X <-- Z --> Y <-- W is currently not.

        emissions = {t: [] for t in range(T)}
        for t, gg in enumerate(subgraphs):
            for e in gg.edges:
                emissions[t].append(e[:-1])

        for t in range(T):
            for tup in emissions[t]:
                if (tup[0],) in emission_pairs.keys():
                    emission_pairs[(tup[0],)] = (emission_pairs[(tup[0],)],) + (tup[1],)
                else:
                    emission_pairs[(tup[0],)] = tup[1]

    return node_children, node_parents, emission_pairs


def get_emissions(T: int, G: MultiDiGraph) -> dict:
    """
    Get the emission (input, output) per time-slice.

    Parameters
    ----------
    T : int
        The total number of time-steps in the causal Bayesian network.
    G : MultiDiGraph
        DAG

    Returns
    -------
    dict
        First pass at emissions pairs per time-slice.
    """
    emissions = {t: [] for t in range(T)}
    for e in G.edges:
        _, inn_time = e[0].split("_")
        _, out_time = e[1].split("_")
        # Emission edge
        if out_time == inn_time:
            emissions[int(out_time)].append((e[0], e[1]))

    return emissions

