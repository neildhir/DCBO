from ..gp_utils import fit_gp
from numpy import hstack


def fit_sem_trans_fncs(observational_samples, transfer_pairs: dict) -> dict:

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
