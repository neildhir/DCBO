from typing import Callable, Dict, Tuple
import numpy as np
from numpy import cumsum
from copy import deepcopy
from ..utils.utilities import calculate_best_intervention_and_effect


def get_relevant_results(results: Callable, replicates: int) -> Dict[str, tuple]:
    """
    When we get results from a notebook they are in a different format from when we pickle them. This function converts the results into the correct format so that we can analyse them.

    Parameters
    ----------
    results : Callable
        The results from running the function 'run_methods_replicates()'
    replicates : int
        How many replicates we used.

    Returns
    -------
    Dict[str, tuple]
        A dictionary with the methods on the keys with results from each replicates on the values.
    """
    data = {m: [] for m in results}
    for m in results:
        for r in range(replicates):
            data[m].append(
                (
                    results[m][r].per_trial_cost,
                    results[m][r].optimal_outcome_values_during_trials,
                    results[m][r].optimal_intervention_sets,
                    results[m][r].assigned_blanket,
                )
            )
    return data


def get_mean_and_std(data, t_steps, repeats=5):
    out = {key: [] for key in data}
    for model in data.keys():
        for t in range(t_steps):
            tmp = []
            for ex in range(repeats):
                tmp.append(data[model][ex][t])
            tmp = np.vstack(tmp)
            out[model].append((tmp.mean(axis=0), tmp.std(axis=0)))
    return out


def get_cumulative_cost_mean_and_std(data, t_steps, repeats=5):
    out = {key: [] for key in data}
    for model in data.keys():
        for t in range(t_steps):
            tmp = []
            for ex in range(repeats):
                tmp.append(data[model][ex][t])
            tmp = np.vstack(tmp)
            # Calculate the cumulative sum here
            out[model].append(cumsum(tmp.mean(axis=0)))
    return out


def elaborate(
    number_of_interventions: int, n_replicates: int, data: dict, best_objective_values: list, T: int
) -> Tuple[dict, dict]:

    # Replace initial data point
    if number_of_interventions is None:
        for model in data:
            for r in range(n_replicates):
                for t in range(T):
                    if data[model][r][1][t][0] == 10000000.0:
                        data[model][r][1][t][0] = data[model][r][1][t][1]

    # Aggregate data
    per_trial_cost = {model: [] for model in data.keys()}
    optimal_outcome_values_during_trials = {model: [] for model in data.keys()}

    for i in range(n_replicates):
        for model in data:
            per_trial_cost[model].append(data[model][i][0])
            optimal_outcome_values_during_trials[model].append(data[model][i][1])

    # Aggregate data
    exp_per_trial_cost = get_cumulative_cost_mean_and_std(per_trial_cost, T, repeats=n_replicates)
    exp_optimal_outcome_values_during_trials = get_mean_and_std(
        optimal_outcome_values_during_trials, T, repeats=n_replicates
    )

    for model in exp_per_trial_cost:
        if model == "BO" or model == "ABO":
            costs = exp_per_trial_cost[model]
            values = exp_optimal_outcome_values_during_trials[model]
            for t in range(T):
                values_t = values[t]
                exp_per_trial_cost[model][t] = np.asarray([0] + list(costs[t]))

                exp_optimal_outcome_values_during_trials[model][t] = tuple(
                    [np.asarray([values_t[i][0]] + list(values_t[i])) for i in range(2)]
                )

    # Clip values so they are not lower than the min
    clip_max = 1000
    for model in exp_per_trial_cost:
        for t in range(T):
            clipped = np.clip(
                exp_optimal_outcome_values_during_trials[model][t][0], a_min=best_objective_values[t], a_max=clip_max
            )
            exp_optimal_outcome_values_during_trials[model][t] = (
                clipped,
                exp_optimal_outcome_values_during_trials[model][t][1],
            )

    return exp_optimal_outcome_values_during_trials, exp_per_trial_cost


def get_converge_trial(best_objective_values, exp_optimal_outcome_values_during_trials, n_trials, T, n_decimal=1):
    where_converge_dict = {method: [None] * T for method in list(exp_optimal_outcome_values_during_trials.keys())}

    for method in exp_optimal_outcome_values_during_trials.keys():
        for t in range(T):
            if isinstance(best_objective_values, dict):
                comparison_values = np.mean(np.vstack(best_objective_values[method])[:, t])
            else:
                comparison_values = best_objective_values[t]

            bool_results = np.round(exp_optimal_outcome_values_during_trials[method][t][0], n_decimal) == np.round(
                comparison_values, n_decimal
            )

            if np.all(~np.array(bool_results)):
                where_method = n_trials
            else:
                where_method = np.argmax(bool_results)

            where_converge_dict[method][t] = where_method
    return where_converge_dict


def get_common_initial_values(
    T, data, n_replicates,
):
    total_initial_list = []
    for t in range(T):
        reps_initial_list = []
        for r in range(n_replicates):
            initial_list = []
            for method in list(data.keys()):
                values = data[method][r][1][t]
                initial = values[0]
                if initial == 10000000.0:
                    initial = values[1]
                initial_list.append(initial)
            reps_initial_list.append(np.max(initial_list))
        total_initial_list.append(reps_initial_list)
    return total_initial_list


def get_table_values(dict_gap_summary, T, n_decimal_mean=2, n_decimal_std=2):
    total_list_mean = []
    for method in dict_gap_summary.keys():
        list_method_mean = [method]
        list_method_std = [" "]
        for t in range(T):
            list_method_mean.append(np.round(dict_gap_summary[method][t][0], n_decimal_mean))
            std_value = np.round(dict_gap_summary[method][t][1], n_decimal_std)
            if std_value == 0.0:
                std_value = "0.00"
            list_method_std.append("(" + str(std_value) + ")")

        total_list_mean.append(list_method_mean)
        total_list_mean.append(list_method_std)
    return total_list_mean


def count_optimal_intervention_set(n_replicates, T, data, optimal_set):
    dict_count = {method: None for method in list(data.keys())}

    for method in list(data.keys()):
        count_list = [None] * T
        for t in range(T):
            count_time = 0.0
            for r in range(n_replicates):
                intervened_set = data[method][r][2][t]
                if isinstance(optimal_set, dict):
                    count_time += int(optimal_set[method][r][t] == intervened_set)
                else:
                    count_time += int(optimal_set[t] == intervened_set)
            count_list[t] = count_time

        dict_count[method] = count_list
    return dict_count


def gap_metric_standard(
    T, data, best_objective_values, total_initial_list, n_replicates, n_trials, where_converge_dict=None,
):
    dict_gap = {method: [None] * T for method in list(data.keys())}
    for method in list(data.keys()):
        for t in range(T):
            for r in range(n_replicates):
                values = data[method][r][1][t]
                initial = total_initial_list[t][r]
                last = values[-1]
                if last - initial == 0.0:
                    gap = 0.0
                else:
                    gap = np.clip((last - initial) / (best_objective_values[t] - initial), 0.0, 1.0)
                if dict_gap[method][t] is None:
                    dict_gap[method][t] = [gap]
                else:
                    dict_gap[method][t].append(gap)
    dict_gap_iters_summary = {method: [None] * T for method in list(data.keys())}
    for t in range(T):
        for method in data.keys():
            percent_iters = (n_trials - where_converge_dict[method][t]) / n_trials
            normaliser = 1.0 + (n_trials - 1) / n_trials
            values_gap_standard = list((np.asarray(dict_gap[method][t]) + percent_iters) / normaliser)
            dict_gap_iters_summary[method][t] = [np.mean(values_gap_standard), np.std(values_gap_standard)]

    return dict_gap_iters_summary


def get_stored_blanket(T, data, n_replicates, list_var):
    store_blankets = {
        model: [[{var: [None] * T for var in list_var} for _ in range(T)] for _ in range(n_replicates)]
        for model in data.keys()
    }
    for method in data.keys():
        for r in range(n_replicates):
            for t in range(1, T):
                values = data[method][r][3]
                store_blankets[method][r][t] = deepcopy(store_blankets[method][r][t - 1])
                for var in list_var:
                    store_blankets[method][r][t][var][t - 1] = values[var][t - 1]

                if store_blankets[method][r][t]["X"][t - 1] is not None and method in ["CBO", "DCBO"]:
                    store_blankets[method][r][t]["Z"][t - 1] = None
    return store_blankets


def get_optimal_set_value(GT, T, exploration_sets_list):
    opt_set_list = [None] * T
    opt_values_list = [None] * T

    for t in range(T):
        values_min = []
        for setx in exploration_sets_list:
            values_min.append(np.min(GT[t][setx]))

        opt_set_index = np.argmin(values_min)

        opt_set_list[t] = exploration_sets_list[opt_set_index]
        opt_values_list[t] = values_min[opt_set_index]
    return opt_set_list, opt_values_list


def get_average_performance_t(data, dict_values, T):
    average_metric = {method: [None, None] for method in list(data.keys())}
    for method in dict_values.keys():
        sum_method = 0.0
        sum_method_std = 0.0
        for t in range(T):
            # if t > 0:
            sum_method += dict_values[method][t][0]
            sum_method_std += dict_values[method][t][1]

        average_metric[method] = [[sum_method / (T), sum_method_std / (T)]]
    return average_metric


def store_optimal_set_values(
    store_blankets,
    data,
    n_replicates,
    T,
    init_sem,
    sem,
    exploration_sets,
    interventional_grids,
    intervention_domain,
    exploration_sets_dict,
):
    optimal_intervention_values = {model: [] for model in data.keys()}
    optimal_intervention_sets = {model: [] for model in data.keys()}
    for model in data.keys():
        for r in range(n_replicates):
            GT, _ = get_ground_truth(
                deepcopy(store_blankets[model][r]),
                T,
                init_sem,
                sem,
                exploration_sets,
                interventional_grids,
                intervention_domain,
            )
            opt_set_list, opt_values_list = get_optimal_set_value(GT, T, exploration_sets_dict[model])

            optimal_intervention_sets[model].append(opt_set_list)
            optimal_intervention_values[model].append(opt_values_list)

    return optimal_intervention_sets, optimal_intervention_values


def get_ground_truth(
    blanket, T, init_sem, sem, exploration_sets, interventional_grids, intervention_domain,
):
    optimal_assigned_blankets = [None] * T
    ground_truth = []
    for t in range(T):
        new_blanket, true_causal_effect = calculate_best_intervention_and_effect(
            static_sem=init_sem,
            dynamic_sem=sem,
            exploration_sets=exploration_sets,
            interventional_grids=interventional_grids,
            time=t,
            intervention_domain=intervention_domain,
            blanket=blanket[t],
            T=T,
            plot=False,
        )
        if t < T - 1:
            optimal_assigned_blankets[t + 1] = new_blanket
        ground_truth.append(true_causal_effect)

    return ground_truth, optimal_assigned_blankets
