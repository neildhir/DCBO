from src.gaussian_process_utils import fit_gp
from numpy.core import hstack
from itertools import chain


def fit_initial_emission_functions(
    observational_samples, canonical_exploration_sets, base_target, n_restart: int = 10
) -> dict:
    #  The previous target must be a lower-case version of the base_target variable
    assert base_target.lower() not in chain(*canonical_exploration_sets)
    # Store functions which concern t = 0
    initial_emission_functions = {k: None for k in canonical_exploration_sets}
    yy = observational_samples[base_target][:, 0].reshape(-1, 1)
    for es in canonical_exploration_sets:
        if len(es) == 1:
            initial_emission_functions[es] = fit_gp(
                x=observational_samples[es[0]][:, 0].reshape(-1, 1),
                y=yy,
                n_restart=n_restart,
            )
        else:
            xx = []
            for iv in es:
                xx.append(observational_samples[iv][:, 0].reshape(-1, 1))
            # Fit function
            initial_emission_functions[es] = fit_gp(x=hstack(xx), y=yy, n_restart=n_restart)

    return initial_emission_functions


def fit_sem(observational_samples: dict, time_slice_children: dict, n_restart: int = 10) -> dict:
    assert isinstance(observational_samples, dict)
    assert isinstance(time_slice_children, dict)
    #  This is an ordered list
    _, T = observational_samples[list(observational_samples.keys())[0]].shape
    fncs = {t: {key: None for key in time_slice_children.keys()} for t in range(T)}
    for t in range(T):
        for key in fncs[t].keys():
            # TODO: This is hard coded
            if key == "Z" and t > 0:
                xx = observational_samples[key][:, t].reshape(-1, 1)
                yy = observational_samples[time_slice_children[key]][:, t].reshape(-1, 1)
            else:
                xx = observational_samples[key][:, t].reshape(-1, 1)
                yy = observational_samples[time_slice_children[key]][:, t].reshape(-1, 1)

            fncs[t][key] = fit_gp(x=xx, y=yy, n_restart=n_restart)

    return fncs


def fit_sem_complex(observational_samples: dict, emission_pairs: dict) -> dict:

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
