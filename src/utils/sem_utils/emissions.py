from numpy.core import hstack
from ..gp_utils import fit_gp


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
