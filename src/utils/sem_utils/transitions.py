from ..gp_utils import fit_gp
from numpy import newaxis
from numpy.core import concatenate, hstack


def fit_sem_transition_functions(observational_samples, n_restart=10) -> dict:
    # Store function which concern t-1 --> t
    transition_functions = {k: None for k in observational_samples.keys()}
    # Total time-steps
    _, T = observational_samples[list(observational_samples.keys())[0]].shape
    # Fit transition functions first
    for var in observational_samples.keys():
        xx = []
        yy = []
        for t1, t2 in zip(range(T - 1), range(1, T)):
            # Auto-regressive structure
            xx.append(observational_samples[var][:, t1])
            yy.append(observational_samples[var][:, t2])

        # Store funcs in dict for later usage
        transition_functions[var] = fit_gp(
            x=concatenate(xx, axis=0)[:, newaxis], y=concatenate(yy, axis=0)[:, newaxis], n_restart=n_restart,
        )

    return transition_functions


def update_transition_function(transition_functions, new_observational_samples, total_time_steps_in_graph: int) -> None:
    for var in transition_functions.keys():
        xx = []
        yy = []
        # Explicitly defining what makes this a transition domain (first-order Markov)
        for t1, t2 in zip(range(total_time_steps_in_graph - 1), range(1, total_time_steps_in_graph)):
            # Auto-regressive structure
            xx.append(new_observational_samples[var][:, t1])
            yy.append(new_observational_samples[var][:, t2])

        # Re-fit the GP without creating a new object
        transition_functions[var].set_XY(
            X=concatenate(xx, axis=0)[:, newaxis], Y=concatenate(yy, axis=0)[:, newaxis],
        )
        transition_functions[var].optimise()


def fit_sem_transition_functions_complex(observational_samples, transfer_pairs: dict) -> dict:

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
