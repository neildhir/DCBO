import pickle
from copy import deepcopy
from pathlib import Path
from random import sample

import pygraphviz
from matplotlib import pyplot as plt
from networkx.drawing import nx_agraph
import numpy as np
from tqdm import trange

from ..methods.dcbo import DCBO
from ..methods.cbo import CBO
from ..methods.abo import ABO
from ..methods.bo import BO

from ..utils.sequential_causal_functions import sequentially_sample_model
from ..utils.sequential_intervention_functions import (
    get_interventional_grids,
    make_sequential_intervention_dictionary,
)
from ..utils.utilities import get_monte_carlo_expectation, powerset
from ..utils.gp_utils import fit_causal_gp


def run_methods_replicates(
    graph,
    sem,
    make_sem_hat,
    root_instrument,
    intervention_domain,
    methods_list,
    obs_samples,
    ground_truth,
    exploration_sets,
    base_target_variable,
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
    folder="toy_stationary",
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
    if args_sem is None and True not in change_points:
        true_sem = sem()
    elif args_sem and True not in change_points:
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
                epsilon["Y"] = np.asarray(np.random.randn(total_timesteps))
                epsilon_list.append(epsilon)
        else:
            epsilon_list = None

        np.random.seed(seed)
        observational_samples = sequentially_sample_model(
            initial_structural_equation_model,
            structural_equation_model,
            total_timesteps=total_timesteps,
            sample_count=n_obs,
            epsilon=epsilon_list,
        )
    else:
        observational_samples = obs_samples

    for ex in trange(reps, desc="Experiment count"):
        if controlled_experiment:
            seed_anchor_points = ex + 1
        else:
            seed_anchor_points = None

        if number_of_trials_BO_ABO is None:
            number_of_trials_BO_ABO = number_of_trials

        # Set parameters common to all methods
        input_params = {
            "graph": graph,
            "sem": sem,
            "base_target_variable": base_target_variable,
            "observational_samples": observational_samples,
            "intervention_domain": intervention_domain,
            "interventional_samples": None,
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
            make_sem_hat,
            number_of_trials_BO_ABO,
            root_instrument,
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
            #  We have to use dill because our object contains lambda functions.
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
    make_sem_hat,
    number_of_trials_BO_ABO,
    root_instrument,
):

    """
    Sequential run of experiments
    """
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
            alg_input_params["make_sem_hat"] = make_sem_hat
            alg_input_params["root_instrument"] = root_instrument

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
        model.run_optimization()

        models_list.append(model)

    return models_list, names_list


def run_methods_replicates_parallel(
    seed,
    graph,
    sem,
    make_sem_hat,
    intervention_domain,
    method,
    observational_samples,
    gt,
    save_metrics=True,
    total_timesteps=3,
    number_of_trials=3,
    initial_interventions=False,
    n_restart=1,
    save_data=True,
    online=True,
    N_obs=100,
    N_t=50,
    cost_structure=1,
    sample_count=10,
    CBO_concat_DO=False,
    CBO_concat_DI=False,
    DCBO_concat_DO=False,
):

    #  Structural equation model
    initial_structural_equation_model = sem().static()
    structural_equation_model = sem().dynamic()

    Graph = nx_agraph.from_agraph(pygraphviz.AGraph(graph))

    opt_result_for_pickle = {}

    if initial_interventions is False:
        interventional_samples = None
        number_of_interventions = None
    else:
        interventional_samples = simulate_interventional_data_for_sequential_LinearToyDBN(
            Graph, intervention_domain, initial_structural_equation_model, structural_equation_model, seed=seed,
        )
        number_of_interventions = 1

    np.random.seed(seed)
    if observational_samples is None:
        observational_samples = sequentially_sample_model(
            initial_structural_equation_model,
            structural_equation_model,
            total_timesteps=total_timesteps,
            sample_count=N_obs,
        )

    input_params = {
        "graph": graph,
        "sem": sem,
        "make_sem_hat": make_sem_hat,
        "observational_samples": observational_samples,
        "intervention_domain": intervention_domain,
        "interventional_samples": interventional_samples,
        "estimate_sem": True,
        "base_target_variable": "Y",
        "task": "min",
        "cost_type": cost_structure,
        "number_of_trials": number_of_trials,
        "number_of_interventions": number_of_interventions,  # Allows us to sub-sample interventions if more than one
        "filtering": True,
        "gt": gt,
        "save_metrics": save_metrics,
        "n_restart": n_restart,
        "use_mc": False,
        "debug_mode": False,
        "N_t": N_t,
        "N_epsilon": 1,
        "online": online,
    }

    if method == "DCBO":
        input_params["dynamic"] = True
        input_params["causal_prior"] = True
        input_params["concat_DO"] = DCBO_concat_DO
    elif method == "CBO":
        input_params["dynamic"] = False
        input_params["causal_prior"] = True
        input_params["concat_DO"] = CBO_concat_DO
        input_params["concat_DI"] = CBO_concat_DI
    elif method == "ABO":
        input_params["dynamic"] = True
        input_params["causal_prior"] = False
    else:
        input_params["dynamic"] = False
        input_params["causal_prior"] = False

    model = DCBO(**input_params)
    model.run_optimization()

    del input_params

    if save_metrics is False:
        model_rmse = [None]
    else:
        model_rmse = model.model_rmse

    opt_result_for_pickle[method] = (
        model.per_trial_cost,
        model.optimal_outcome_values_during_trials,
        model_rmse,
    )

    if save_data:
        # Check if pickle file is alreay there, if not store it in location
        my_file = Path(
            "../data/synthetic/{}_synthetic_runs_s_{}_t_{}_cost_{}.pickle".format(
                method, seed, number_of_trials, cost_structure
            )
        )
        if not my_file.is_file():
            # File is not there, so we create
            with open(
                "../data/synthetic/{}_synthetic_runs_s_{}_t_{}_cost_{}.pickle".format(
                    method, seed, number_of_trials, cost_structure
                ),
                "wb",
            ) as handle:
                #  We have to use dill because our object contains lambda functions.
                pickle.dump(opt_result_for_pickle, handle)
                handle.close()

    return model


def simulate_interventional_data_for_sequential_LinearToyDBN(
    graph, intervention_domain, initial_structural_equation_model, structural_equation_model, seed=0,
):
    np.random.seed(seed)

    interventional_data = {k: None for k in powerset(["X", "Z"])}

    canonical_exploration_sets = list(powerset(intervention_domain.keys()))

    # Get the interventional grids
    interventional_grids = get_interventional_grids(
        canonical_exploration_sets, intervention_domain, size_intervention_grid=100
    )
    levels = {es: None for es in canonical_exploration_sets}
    for es in canonical_exploration_sets:
        idx = np.random.randint(0, interventional_grids[es].shape[0])  # Random indices
        levels[es] = interventional_grids[es][idx, :]

    """
    do(Z_0)
    """
    interv, T = make_sequential_intervention_dictionary(graph)
    # Univariate intervention at time 0
    interv["Z"][0] = float(levels[("Z",)])
    static_noise_model = {k: np.zeros(T) for k in ["X", "Z", "Y"]}
    # Sample this model with one intervention
    intervention_samples = sequentially_sample_model(
        initial_structural_equation_model,
        structural_equation_model,
        total_timesteps=T,
        interventions=interv,
        sample_count=1,
        epsilon=static_noise_model,
    )
    interventional_data[("Z",)] = get_monte_carlo_expectation(intervention_samples)

    """
    do(X_0)
    """
    interv, T = make_sequential_intervention_dictionary(graph)
    # Univariate intervention
    interv["X"][0] = float(levels[("X",)])
    # Sample this model with one intervention
    intervention_samples = sequentially_sample_model(
        initial_structural_equation_model,
        structural_equation_model,
        total_timesteps=T,
        interventions=interv,
        sample_count=1,
        epsilon=static_noise_model,
    )
    interventional_data[("X",)] = get_monte_carlo_expectation(intervention_samples)

    """
    do(Z_0, X_0)
    """
    interv, T = make_sequential_intervention_dictionary(graph)
    # Multivariate intervention
    interv["X"][0] = float(levels[("X",)][0])
    interv["Z"][0] = float(levels[("Z",)][0])

    intervention_samples = sequentially_sample_model(
        initial_structural_equation_model,
        structural_equation_model,
        total_timesteps=T,
        interventions=interv,
        sample_count=1,
        epsilon=static_noise_model,
    )

    interventional_data[("X", "Z")] = get_monte_carlo_expectation(intervention_samples)

    return interventional_data


def simulate_interventional_data_for_sequential_ComplexDBN(
    graph, intervention_domain, initial_structural_equation_model, structural_equation_model, variables, T, seed=0,
):
    np.random.seed(seed)

    interventional_data = {k: None for k in powerset(variables)}

    canonical_exploration_sets = list(powerset(intervention_domain.keys()))

    # Get the interventional grids
    interventional_grids = get_interventional_grids(
        canonical_exploration_sets, intervention_domain, size_intervention_grid=100
    )
    levels = {es: None for es in canonical_exploration_sets}
    for es in canonical_exploration_sets:
        idx = np.random.randint(0, interventional_grids[es].shape[0])  # Random indices
        levels[es] = interventional_grids[es][idx, :]

    static_noise_model = {k: np.zeros(T) for k in variables + ["Y"]}

    for var in canonical_exploration_sets:
        var = list(var)

        intervention, _ = make_sequential_intervention_dictionary(graph)
        if len(var) == 1:
            intervention[var[0]][0] = float(levels[tuple(var)])
        elif len(var) == 2:
            intervention[var[0]][0] = float(levels[tuple(var[0])])
            intervention[var[1]][0] = float(levels[tuple(var[1])])
        else:
            intervention[var[0]][0] = float(levels[tuple(var[0])])
            intervention[var[1]][0] = float(levels[tuple(var[1])])
            intervention[var[2]][0] = float(levels[tuple(var[2])])

        # Sample this model with one intervention
        intervention_samples = sequentially_sample_model(
            initial_structural_equation_model,
            structural_equation_model,
            total_timesteps=T,
            interventions=intervention,
            sample_count=1,
            epsilon=static_noise_model,
        )
        interventional_data[tuple(var)] = get_monte_carlo_expectation(intervention_samples)

    return interventional_data


def optimal_sequence_of_interventions(
    exploration_sets,
    interventional_grids,
    initial_structural_equation_model,
    structural_equation_model,
    graph,
    timesteps=4,
    model_variables=None,
    target_variable=None,
    task="min",
):
    if model_variables is None:
        static_noise_model = {k: np.zeros(timesteps) for k in ["X", "Z", "Y"]}
    else:
        static_noise_model = {k: np.zeros(timesteps) for k in model_variables}

    assert target_variable is not None
    assert target_variable in model_variables

    range_T = range(timesteps)
    shift_range_T = range(timesteps - 1)
    best_s_sequence = []
    best_s_values = []
    best_objective_values = []

    # optimal_interventions, _ = make_sequential_intervention_dictionary(graph)
    optimal_interventions = {setx: [None] * timesteps for setx in exploration_sets}

    y_stars = deepcopy(optimal_interventions)
    all_CE = []
    blank_intervention_blanket, _ = make_sequential_intervention_dictionary(graph)

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
                        total_timesteps=timesteps,
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
                        total_timesteps=timesteps,
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


def plot_optimisation_outcomes(
    ax,
    i,
    interventional_grids,
    mean_function,
    time_index,
    exploration_set,
    true_causal_effects,
    vmin,
    vmax,
    X_I_train,
    Y_I_train,
    observational_samples,
    time_slice_batch_indices,
    objective_values,
    n,
):
    #  Mean function
    ax[i].plot(
        interventional_grids[exploration_set],
        mean_function,
        lw=2,
        color="b",
        ls="--",
        label=r"$m_{{{}_{}}}$".format(exploration_set[0], time_index),
    )
    # True causal effect at time_index
    ax[i].plot(
        interventional_grids[exploration_set],
        np.array(true_causal_effects[time_index][exploration_set]),
        lw=4,
        color="r",
        alpha=0.5,
        label="Target CE",
    )
    # Confidence interval
    ax[i].fill_between(
        interventional_grids[exploration_set].squeeze(),
        vmin.squeeze(),
        vmax.squeeze(),
        color="b",
        alpha=0.25,
        label="95\% CI",
    )
    # Interventions
    ax[i].scatter(
        X_I_train,
        Y_I_train,
        s=200,
        marker=".",
        c="g",
        label=r"$\mathcal{{D}}^I_{}, |\mathcal{{D}}^I_{}| = {}$".format(time_index, time_index, n),
        linewidths=2,
        zorder=10,
    )
    # Observations
    ax[i].scatter(
        observational_samples[exploration_set[0]][time_slice_batch_indices[time_index], time_index],
        1.5
        * objective_values[exploration_set][time_index]
        * np.ones_like(observational_samples[exploration_set[0]][time_slice_batch_indices[time_index], time_index]),
        s=100,
        marker="x",
        c="k",
        label=r"$\mathcal{{D}}^O_{}, |\mathcal{{D}}^O_{}| = {}$".format(
            time_index,
            time_index,
            len(observational_samples[exploration_set[0]][time_slice_batch_indices[time_index], time_index]),
        ),
        alpha=0.5,
        linewidths=1,
    )
    ax[i].set_xlabel("${}_{}$".format(exploration_set[0], time_index))
    if time_index == 0:
        ax[i].set_ylabel(
            r"$\mathbb{{E}}[{}_{} \mid \textrm{{do}}({}_{})]$".format("Y", time_index, exploration_set[0], time_index)
        )
    else:
        es_star, _ = max(objective_values.items(), key=lambda x: x[time_index - 1])
        ax[i].set_ylabel(
            r"$\mathbb{{E}}[{}_{} \mid \textrm{{do}}({}_{}),\textrm{{did}}({}_{}) ]$".format(
                "Y", time_index, exploration_set[0], time_index, es_star[0], time_index - 1,
            )
        )
    ax[i].legend(
        ncol=1, fontsize="medium", loc="lower center", frameon=False, bbox_to_anchor=(1.2, 0.4),
    )


def optimise_one_time_step(
    time_index,
    exploration_set,
    mean_fnc,
    var_fnc,
    expectation_estimates,
    objective_values,
    intervention_levels,
    interventional_grids: dict,
    observational_samples,
    time_slice_batch_indices,
    true_causal_effects,
    plot_me=True,
    index_data=None,
    intervention_points=[1, 1, 1],
) -> None:

    N = range(interventional_grids[exploration_set].shape[0])

    k = len(intervention_points)
    plt.rcParams.update({"font.size": 20, "text.usetex": True, "font.family": "serif"})

    plt.plot(
        interventional_grids[exploration_set], mean_fnc(interventional_grids[exploration_set]),
    )
    plt.plot(
        interventional_grids[exploration_set], true_causal_effects[time_index][exploration_set], color="red",
    )
    plt.show()

    fig, ax = plt.subplots(k, figsize=(11, k * 5))
    index = []
    X_I_train_list = []
    Y_I_train_list = []

    for i, n in enumerate(intervention_points):
        # Select points to explore
        if index_data is None:
            idx = sample(N, n)  # No replacement
        else:
            idx = [index_data[i]]

        index.extend(idx)
        N = list(set(N) - set(idx))

        # Interventional data
        X_I_train = interventional_grids[exploration_set][index]
        X_I_train_list.append(X_I_train)
        # Samples drawn from true causal effect
        Y_I_train = np.array(true_causal_effects[time_index][exploration_set])[index].reshape(-1, 1)
        Y_I_train_list.append(Y_I_train)

        # Estimate of expected causal effect
        expectation_estimates[exploration_set][time_index] = fit_causal_gp(
            mean_function=mean_fnc, variance_function=var_fnc, X=X_I_train, Y=Y_I_train
        )
        # Get mean and variance of estimate
        # if time_index == 0:
        M, V = expectation_estimates[exploration_set][time_index].predict(interventional_grids[exploration_set])

        # Optimal value (y^*_{time_index})
        idx = M.argmin()
        objective_values[exploration_set][time_index] = float(M[idx])

        # TODO: this is not yet specified for multivariate interventions.
        intervention_levels[exploration_set][time_index] = interventional_grids[exploration_set][idx][0]
        # Variance
        CI = np.sqrt(V)
        # Confidence intervals
        vmax = M + CI
        vmin = M - CI

        if plot_me:
            plot_optimisation_outcomes(
                ax,
                i,
                interventional_grids,
                M,
                time_index,
                exploration_set,
                true_causal_effects,
                vmin,
                vmax,
                X_I_train,
                Y_I_train,
                observational_samples,
                time_slice_batch_indices,
                objective_values,
                n,
            )
    fig.tight_layout()

    return (
        objective_values,
        expectation_estimates,
        intervention_levels,
        X_I_train_list,
        Y_I_train_list,
    )
