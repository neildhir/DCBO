from copy import deepcopy
from itertools import combinations
from typing import Dict, Tuple
from networkx import MultiDiGraph
from numpy.core import hstack

from ..gp_utils import fit_gp
from ..utilities import update_emission_pairs_keys


def fit_sem_emit_fncs(observational_samples: dict, emission_pairs: dict) -> dict:

    #  This is an ordered list
    timesteps = observational_samples[list(observational_samples.keys())[0]].shape[1]
    emit_fncs = {t: {key: None for key in emission_pairs.keys()} for t in range(timesteps)}

    for input_nodes in emission_pairs.keys():
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
            #  Output
            yy = observational_samples[target_variable][:, time].reshape(-1, 1)
        else:
            raise ValueError("The length of the tuple is: {}".format(len(input_nodes)))

        assert len(xx.shape) == 2
        assert len(yy.shape) == 2
        if input_nodes in emit_fncs[time]:
            emit_fncs[time][input_nodes] = fit_gp(x=xx, y=yy)
        else:
            raise ValueError(input_nodes)

    # To remove any None values
    return {t: {k: v for k, v in emit_fncs[t].items() if v is not None} for t in range(timesteps)}


def get_emissions_input_output_pairs(
    T: int, G: MultiDiGraph
) -> Tuple[Dict[str, tuple], Dict[str, tuple], Dict[str, tuple]]:

    node_children = {node: None for node in G.nodes}
    node_parents = {node: None for node in G.nodes}
    emission_pairs = {}

    # Children of all nodes
    for node in G.nodes:
        node_children[node] = list(G.successors(node))

    #  Parents of all nodes
    for node in G.nodes:
        node_parents[node] = tuple(G.predecessors(node))

    emissions = {t: [] for t in range(T)}
    for e in G.edges:
        _, inn_time = e[0].split("_")
        _, out_time = e[1].split("_")
        # Emission edge
        if out_time == inn_time:
            emissions[int(out_time)].append((e[0], e[1]))

    new_emissions = deepcopy(emissions)
    emissions = emissions
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

    return node_children, node_parents, emission_pairs
