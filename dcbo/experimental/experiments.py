import pickle
from copy import deepcopy
from typing import Callable, Tuple

import numpy as np
from networkx.classes.multidigraph import MultiDiGraph
from pandas import DataFrame, read_csv
from tqdm import trange

from dcbo.methods.abo import ABO
from dcbo.methods.bo import BO
from dcbo.methods.cbo import CBO
from dcbo.methods.dcbo import DCBO
from dcbo.utils.sequential_sampling import sequentially_sample_model
from dcbo.utils.sequential_intervention_functions import make_sequential_intervention_dict
from dcbo.utils.utilities import get_monte_carlo_expectation


def run_methods_replicates(
    G,
    sem,
    make_sem_estimator,
    intervention_domain,
    methods_list,
    obs_samples,
    exploration_sets,
    base_target_variable,
    ground_truth=None,
    total_timesteps=3,
    reps=3,
    number_of_trials=3,
    number_of_trials_BO_ABO=None,
    n_restart=1,
    save_data=False,
    n_obs=100,
    cost_structure: int = 1,
    optimal_assigned_blankets=None,
    debug_mode=False,
    use_mc=False,
    online=False,
    concat=False,
    use_di=False,
    transfer_hp_o=False,
    transfer_hp_i=False,
    hp_i_prior=True,
    estimate_sem=True,
    folder=None,
    num_anchor_points=100,
    n_obs_t=None,
    sample_anchor_points=False,
    seed: int = 0,
    controlled_experiment: bool = True,
    noise_experiment: bool = False,
    args_sem=None,
    manipulative_variables=None,
    change_points: list = None,
):

    #  Structural equation model
    if args_sem is None and change_points is None:
        true_sem = sem()
    elif args_sem and change_points is None:
        true_sem = sem(args_sem[0], args_sem[1])
    else:
        true_sem = sem(change_points.index(True))

    initial_structural_equation_model = true_sem.static()
    structural_equation_model = true_sem.dynamic()

    # Optimise
    results = {}
    opt_results_for_pickle = {}

    np.random.seed(seed)

    # Sample observational data
    if obs_samples is None:
        if noise_experiment:
            np.random.seed(seed)
            epsilon_list = []
            new_mean = 2.0
            new_std = 4.0
            for i in range(n_obs):
                epsilon = {
                    k: (np.random.randn(total_timesteps) + new_mean) * new_std
                    for k in initial_structural_equation_model.keys()
                }
                epsilon[base_target_variable] = np.asarray(np.random.randn(total_timesteps))
                epsilon_list.append(epsilon)
        else:
            epsilon_list = None

        np.random.seed(seed)
        observation_samples = sequentially_sample_model(
            initial_structural_equation_model,
            structural_equation_model,
            total_timesteps=total_timesteps,
            sample_count=n_obs,
            epsilon=epsilon_list,
        )
    else:
        observation_samples = obs_samples

    for ex in trange(reps, desc="Experiment count"):
        if controlled_experiment:
            seed_anchor_points = ex + 1
        else:
            seed_anchor_points = None

        if number_of_trials_BO_ABO is None:
            number_of_trials_BO_ABO = number_of_trials

        # Set parameters common to all methods
        input_params = {
            "G": G,
            "sem": sem,
            "base_target_variable": base_target_variable,
            "observation_samples": observation_samples,
            "intervention_domain": intervention_domain,
            "intervention_samples": None,
            "number_of_trials": number_of_trials,
            "task": "min",
            "cost_type": cost_structure,
            "n_restart": n_restart,
            "debug_mode": debug_mode,
            "optimal_assigned_blankets": optimal_assigned_blankets,
            "num_anchor_points": num_anchor_points,
            "sample_anchor_points": sample_anchor_points,
            "seed_anchor_points": seed_anchor_points,
            "hp_i_prior": hp_i_prior,
            "args_sem": args_sem,
            "manipulative_variables": manipulative_variables,
            "change_points": change_points,
        }

        models, names = run_all_opt_models(
            methods_list,
            input_params,
            exploration_sets,
            online,
            use_di,
            transfer_hp_o,
            transfer_hp_i,
            concat,
            estimate_sem,
            use_mc,
            ground_truth,
            n_obs_t,
            make_sem_estimator,
            number_of_trials_BO_ABO,
        )

        del input_params

        for i, key in enumerate(names):
            if key in results:
                results[key].append(models[i])
            else:
                results[key] = [models[i]]

            if save_data:
                if key in opt_results_for_pickle:
                    opt_results_for_pickle[key].append(
                        (
                            models[i].per_trial_cost,
                            models[i].optimal_outcome_values_during_trials,
                            models[i].optimal_intervention_sets,
                            models[i].assigned_blanket,
                        )
                    )
                else:
                    opt_results_for_pickle[key] = [
                        (
                            models[i].per_trial_cost,
                            models[i].optimal_outcome_values_during_trials,
                            models[i].optimal_intervention_sets,
                            models[i].assigned_blanket,
                        )
                    ]
    if isinstance(n_obs_t, list):
        missing = True
    else:
        missing = False
    if save_data:
        if optimal_assigned_blankets:
            optimal_assigned_blankets = True
        with open(
            "../data/"
            + folder
            + "/method_{}_T_{}_it_{}_reps_{}_Nobs_{}_online_{}_concat_{}_transfer_{}_usedi_{}_hpiprior_{}_missing_{}_noise_{}_optimal_assigned_blanket_{}_seed_{}.pickle".format(
                "".join(methods_list),
                total_timesteps,
                number_of_trials,
                reps,
                n_obs,
                online,
                concat,
                transfer_hp_i,
                use_di,
                hp_i_prior,
                missing,
                noise_experiment,
                seed,
                optimal_assigned_blankets,
            ),
            "wb",
        ) as handle:
            pickle.dump(opt_results_for_pickle, handle)
            handle.close()

    return results


def run_all_opt_models(
    methods_list,
    input_params,
    exploration_sets,
    online,
    use_di,
    transfer_hp_o,
    transfer_hp_i,
    concat,
    estimate_sem,
    use_mc,
    ground_truth,
    n_obs_t,
    make_sem_estimator,
    number_of_trials_BO_ABO,
):

    # Sequential run of experiments
    models_list = []
    names_list = []
    for method in methods_list:

        assert method in ["ABO", "DCBO", "BO", "CBO"], (
            "Method not implemented",
            method,
            methods_list,
        )
        alg_input_params = deepcopy(input_params)

        names_list.append(method)

        if method in ["DCBO", "CBO"]:
            # Args common to DCBO and CBO
            alg_input_params["estimate_sem"] = estimate_sem
            alg_input_params["exploration_sets"] = exploration_sets
            alg_input_params["online"] = online
            alg_input_params["ground_truth"] = ground_truth
            alg_input_params["use_mc"] = use_mc
            alg_input_params["n_obs_t"] = n_obs_t
            alg_input_params["make_sem_estimator"] = make_sem_estimator

        if method == "DCBO":
            algorithm = DCBO
            alg_input_params["use_di"] = use_di
            alg_input_params["transfer_hp_o"] = transfer_hp_o
            alg_input_params["transfer_hp_i"] = transfer_hp_i

        elif method == "CBO":
            algorithm = CBO
            alg_input_params["concat"] = concat

        elif method == "ABO":
            algorithm = ABO
            alg_input_params["number_of_trials"] = number_of_trials_BO_ABO
        else:
            algorithm = BO
            alg_input_params["number_of_trials"] = number_of_trials_BO_ABO

        print("\n\t>>>" + method + "\n")
        model = algorithm(**alg_input_params)
        model.run()
        models_list.append(model)

    return models_list, names_list


def optimal_sequence_of_interventions(
    exploration_sets: list,
    interventional_grids: dict,
    initial_structural_equation_model: Callable,
    structural_equation_model: Callable,
    G: MultiDiGraph,
    T: int = 3,
    model_variables: list = None,
    target_variable: str = None,
    task: str = "min",
) -> Tuple:
    if model_variables is None:
        static_noise_model = {k: np.zeros(T) for k in ["X", "Z", "Y"]}
    else:
        static_noise_model = {k: np.zeros(T) for k in model_variables}

    assert target_variable is not None
    assert target_variable in model_variables

    range_T = range(T)
    shift_range_T = range(T - 1)
    best_s_sequence = []
    best_s_values = []
    best_objective_values = []

    optimal_interventions = {setx: [None] * T for setx in exploration_sets}

    y_stars = deepcopy(optimal_interventions)
    all_CE = []
    blank_intervention_blanket = make_sequential_intervention_dict(G, T)

    for t in range_T:

        CE = {es: [] for es in exploration_sets}

        # E[Y_0 | do( . _0)]
        if t == 0:

            for s in exploration_sets:

                # Reset blanket so as to not carry over levels from previous exploration set
                intervention_blanket = deepcopy(blank_intervention_blanket)
                for level in interventional_grids[s]:

                    # Univariate intervention
                    if len(s) == 1:
                        intervention_blanket[s[0]][t] = float(level)

                    # Multivariate intervention
                    else:
                        for var, val in zip(s, level):
                            intervention_blanket[var][t] = val

                    intervention_samples = sequentially_sample_model(
                        initial_structural_equation_model,
                        structural_equation_model,
                        total_timesteps=T,
                        interventions=intervention_blanket,
                        sample_count=1,
                        epsilon=static_noise_model,
                    )
                    out = get_monte_carlo_expectation(intervention_samples)

                    CE[s].append((out[target_variable][t]))

        # E[Y_t | do( . _t), did( . _{t-1},...,_{t-T})]
        else:
            # do()
            for s in exploration_sets:

                # Reset blanket so as to not carry over levels from previous exploration set
                intervention_blanket = deepcopy(blank_intervention_blanket)
                # Assign previous optimal interventions --> 'did()'
                for best_s, best_s_value, tt in zip(best_s_sequence, best_s_values, shift_range_T):
                    if len(best_s) == 1:
                        intervention_blanket[best_s[0]][tt] = float(best_s_value)
                    else:
                        for var, val in zip(best_s, best_s_value):
                            intervention_blanket[var][tt] = val

                for level in interventional_grids[s]:

                    if len(s) == 1:
                        intervention_blanket[s[0]][t] = float(level)
                    else:
                        for var, val in zip(s, level):
                            intervention_blanket[var][t] = val

                    intervention_samples = sequentially_sample_model(
                        initial_structural_equation_model,
                        structural_equation_model,
                        total_timesteps=T,
                        interventions=intervention_blanket,
                        sample_count=1,
                        epsilon=static_noise_model,
                    )
                    out = get_monte_carlo_expectation(intervention_samples)

                    CE[s].append((out[target_variable][t]))

        local_target_values = []
        for s in exploration_sets:
            if task == "min":
                idx = np.array(CE[s]).argmin()
            else:
                idx = np.array(CE[s]).argmax()
            local_target_values.append((s, idx, CE[s][idx]))
            y_stars[s][t] = CE[s][idx]
            optimal_interventions[s][t] = interventional_grids[s][idx]

        # Find best intervention at time t
        best_s, best_idx, best_objective_value = min(local_target_values, key=lambda t: t[2])
        best_s_value = interventional_grids[best_s][best_idx]

        best_s_sequence.append(best_s)
        best_s_values.append(best_s_value)
        best_objective_values.append(best_objective_value)
        all_CE.append(CE)

    return (
        best_s_values,
        best_s_sequence,
        best_objective_values,
        y_stars,
        optimal_interventions,
        all_CE,
    )


def create_plankton_dataset(start: int, end: int) -> dict:
    """Function to create dataset for plankton experiment.

    Uses data from experiments C1 to C4 from [1].

    A series of ten chemostat experiments was performed, constituting a total of 1,948 measurement days (corresponding to 5.3 years of measurement) and covering 3 scenarios.

    Constant environmental conditions (C1–C7, 1,428 measurement days). This scenario consisted of 4 trials with the alga M. minutum (C1–C4) which is what we use in these experiments. All data lives in `data/plankton` and is freely available online.

    [1] Blasius B, Rudolf L, Weithoff G, Gaedke U, Fussmann G.F. Long-term cyclic persistence in an experimental predator-prey system. Nature (2019).

    Parameters
    ----------
    start : int
        Start time-index
    end : int
        End time-index

    Returns
    -------
    dict
        State-variables as the keys with data as a ndarray
    """

    #  Constants from paper
    v_algal = 28e-9  # nitrogen content per algal cell
    v_Brachionus = 0.57 * 1e-3  # nitrogen content per adult female Brachionus
    beta = 5
    data = DataFrame()

    ds = []

    files = ["C1", "C2", "C3", "C4"]  # , "C6", "C7"]
    for file in files:
        df = read_csv("../data/plankton/{}.csv".format(file))

        # Impute missing values
        df.interpolate(method="cubic", inplace=True)  #  There are better imputation methods

        data["M"] = df[" external medium (mu mol N / l)"]
        data["N"] = df[" algae (10^6 cells/ml)"] * 1e6 * 1000 * v_algal
        data["P"] = df[" rotifers (animals/ml)"] * 1000 * v_Brachionus
        data["D"] = df[" dead animals (per ml)"] * 1000 * v_Brachionus
        data["E"] = df[" eggs (per ml)"] * 1000 * v_Brachionus
        data["B"] = df.apply(
            lambda row: beta * row[" eggs (per ml)"] / row[" egg-ratio"] if row[" egg-ratio"] > 0 else 0.0, axis=1
        )

        # Derivative state variables (function of other state variables)
        data["A"] = data.apply(lambda row: (row.B * 0.5) * 1000 * v_Brachionus, axis=1)
        data["J"] = data.apply(lambda row: (row.B / (2 * beta)) * 1000 * v_Brachionus, axis=1)

        # Replace NaN values at t=0 with 0.0
        data.fillna(value=0.0, inplace=True)
        assert data.isnull().sum().sum() == 0, (file, df.isnull().sum())

        tmp_dict = data[["M", "N", "P", "J", "A", "E", "D"]].iloc[start:end, :].to_dict("list")
        # print(tmp_dict)
        ds.append({item[0]: np.array(item[1]).reshape(1, -1) for item in tmp_dict.items()})

    # Merge all observations from all datasets
    d = {}
    for k in tmp_dict.keys():
        d[k] = np.concatenate(list(d[k] for d in ds), axis=0)

    print("Units of all observation variables is (mu mol N / L).")
    return d
