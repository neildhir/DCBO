import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from utilities import get_cumulative_cost_mean_and_std
from utilities import calculate_best_intervention_and_effect


def get_mean_and_std(data, t_steps, repeats=5):
    out = {key: [] for key in data.keys()}
    for model in data.keys():
        for t in range(t_steps):
            tmp = []
            for ex in range(repeats):
                tmp.append(data[model][ex][t])
            tmp = np.vstack(tmp)
            out[model].append((tmp.mean(axis=0), tmp.std(axis=0)))
    return out


def elaborate(number_of_interventions, n_replicates, n_trials, data, best_objective_values, T):
    # Replace initial data point
    if number_of_interventions is None:
        for model in data.keys():
            for r in range(n_replicates):
                for t in range(T):
                    if data[model][r][1][t][0] == 10000000.0:
                        data[model][r][1][t][0] = data[model][r][1][t][1]

    # Aggregate data
    per_trial_cost = {model: [] for model in data.keys()}
    optimal_outcome_values_during_trials = {model: [] for model in data.keys()}

    for i in range(n_replicates):
        for model in data.keys():
            per_trial_cost[model].append(data[model][i][0])
            optimal_outcome_values_during_trials[model].append(data[model][i][1])

    # Aggregate data
    exp_per_trial_cost = get_cumulative_cost_mean_and_std(per_trial_cost, T, repeats=n_replicates)
    exp_optimal_outcome_values_during_trials = get_mean_and_std(
        optimal_outcome_values_during_trials, T, repeats=n_replicates
    )

    # For ABO and BO we make the cost start from 0 as in the competing models
    # We then augement the dimension of the y values to plot to ensure they can be plotted
    if "BO" in exp_per_trial_cost.keys() and "ABO" in exp_per_trial_cost.keys():
        initial_value_BO_ABO = np.max(
            (
                exp_optimal_outcome_values_during_trials["BO"][0][0][0],
                exp_optimal_outcome_values_during_trials["ABO"][0][0][0],
            )
        )

    for model in exp_per_trial_cost.keys():
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
    for model in exp_per_trial_cost.keys():
        for t in range(T):
            clipped = np.clip(
                exp_optimal_outcome_values_during_trials[model][t][0], a_min=best_objective_values[t], a_max=clip_max
            )
            exp_optimal_outcome_values_during_trials[model][t] = (
                clipped,
                exp_optimal_outcome_values_during_trials[model][t][1],
            )

    return exp_optimal_outcome_values_during_trials, exp_per_trial_cost


def plot_expected_opt_curve_paper(
    T,
    ground_truth,
    cost,
    outcome,
    plot_params,
    ground_truth_dict=None,
    filename=None,
    fig_size=(15, 3),
    save_fig=False,
    y_lim_list=None,
):

    sns.set_theme(
        context="paper",
        style="ticks",
        palette="deep",
        font="sans-serif",
        font_scale=1.3,
    )
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

    if T > 3:
        fig, axs = plt.subplots(2, int(T / 2), figsize=fig_size, facecolor="w", edgecolor="k")
    else:
        fig, axs = plt.subplots(1, T, figsize=fig_size, facecolor="w", edgecolor="k")

    fig.subplots_adjust(hspace=0.5, wspace=0.13)

    axs = axs.ravel()
    for time_index in range(T):
        cs_all = []
        out_all = []
        linet_list = []

        for i, model in enumerate(cost.keys()):
            cs = cost[model][time_index]
            cs_max = round(max(cs))
            cs_all.append(cs_max)

            out_max = np.max(np.array(outcome[model][time_index]))
            out_all.append(out_max)

            # Mean
            (linet,) = axs[time_index].plot(
                cs,
                outcome[model][time_index][0],
                linewidth=plot_params["linewidth"],
                ls=plot_params["line_styles"][model],
                label=plot_params["labels"][model],
                color=plot_params["colors"][model],
            )
            linet_list.append(linet)

            # plus/minus one std
            lower = outcome[model][time_index][0] - outcome[model][time_index][1]
            upper = outcome[model][time_index][0] + outcome[model][time_index][1]
            axs[time_index].fill_between(
                cs, lower, upper, alpha=plot_params["alpha"], color=plot_params["colors"][model]
            )

        axs[time_index].tick_params(axis="both", which="major", labelsize=plot_params["size_ticks"])

        # Ground truth
        if ground_truth_dict is not None:
            for model in outcome.keys():
                values = np.mean(np.vstack(ground_truth_dict[model])[:, time_index])
                gt_line = axs[time_index].hlines(
                    y=values,
                    xmin=0,
                    xmax=np.floor(max(cs_all)) + 1.0,
                    linewidth=plot_params["linewidth_opt"],
                    color=plot_params["colors"][model],
                    ls=plot_params["line_styles"]["True"],
                    label=model + ", " + plot_params["labels"]["True"],
                    zorder=15,
                    alpha=plot_params["alpha"] + 0.3,
                )
                linet_list.append(gt_line)
        # else:
        gt_line = axs[time_index].hlines(
            y=ground_truth[time_index],
            xmin=0,
            xmax=np.floor(max(cs_all)) + 1.0,
            linewidth=plot_params["linewidth_opt"],
            color=plot_params["colors"]["True"],
            ls=plot_params["line_styles"]["True"],
            label=plot_params["labels"]["True"],
            zorder=10,
        )
        linet_list.append(gt_line)

        # Cost
        axs[time_index].set_xlabel(plot_params["xlabel"], fontsize=plot_params["size_labels"])
        axs[time_index].set_xlim(0, plot_params["xlim_max"])
        if isinstance(ground_truth, dict):
            axs[time_index].set_ylim(
                ground_truth["DCBO"][time_index] - 0.2,
                np.max(outcome[model][time_index][0] + outcome[model][time_index][1] + 0.3),
            )
        else:
            if y_lim_list and time_index in y_lim_list:
                axs[time_index].set_ylim(y_lim_list[time_index][0], y_lim_list[time_index][1])
            else:
                axs[time_index].set_ylim(ground_truth[time_index] - 0.1, np.max(out_all) + 0.3)

        # Annotate (1, 3)
        bbox_props = dict(boxstyle="round,pad=0.3", fc="w", ec="k", lw=1.5)

        axs[time_index].annotate(
            "$t = {}$".format(time_index + 1),
            xy=(0, 0.0),
            xycoords="axes fraction",
            xytext=(0.45, 0.85),
            bbox=bbox_props,
            fontsize=plot_params["size_labels"],
        )

        # Outcome value
        if time_index == 0:
            axs[time_index].set_ylabel("$y_t^\star$", fontsize=plot_params["size_labels"])

        axs[time_index].yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))

    lgd = plt.legend(
        ncol=plot_params["ncols"], handles=linet_list, bbox_to_anchor=(-1.75, -0.3), loc="upper left", fontsize="large"
    )

    if save_fig and filename:
        # Set reference time for save
        now = datetime.datetime.now()
        fig.savefig(
            "../figures/synthetic/" + filename + "_" + now.strftime("%d%m%Y_%H%M") + ".pdf",
            bbox_extra_artists=(lgd,),
            bbox_inches="tight",
        )


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
    T,
    data,
    best_objective_values,
    n_replicates,
    where_converge_dict=None,
    count_optimal_set=None,
    use_initial_value=True,
):
    dict_gap = {method: [None] * T for method in list(data.keys())}

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
    total_list_std = []

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
    T,
    data,
    best_objective_values,
    total_initial_list,
    n_replicates,
    n_trials,
    where_converge_dict=None,
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
        model: [[{var: [None] * T for var in list_var} for t in range(T)] for n in range(n_replicates)]
        for model in data.keys()
    }
    # list_var.remove('Y')
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
    initial_structural_equation_model,
    structural_equation_model,
    exploration_sets,
    interventional_grids,
    intervention_domain,
    exploration_sets_dict,
):
    optimal_intervention_values = {model: [] for model in data.keys()}
    optimal_intervention_sets = {model: [] for model in data.keys()}
    for model in data.keys():
        for r in range(n_replicates):
            GT, _ = get_GT(
                deepcopy(store_blankets[model][r]),
                T,
                initial_structural_equation_model,
                structural_equation_model,
                exploration_sets,
                interventional_grids,
                intervention_domain,
            )
            opt_set_list, opt_values_list = get_optimal_set_value(GT, T, exploration_sets_dict[model])

            optimal_intervention_sets[model].append(opt_set_list)
            optimal_intervention_values[model].append(opt_values_list)

    return optimal_intervention_sets, optimal_intervention_values


def get_GT(
    blanket,
    T,
    initial_structural_equation_model,
    structural_equation_model,
    exploration_sets,
    interventional_grids,
    intervention_domain,
):
    optimal_assigned_blankets = [None] * T
    GT = []
    for t in range(T):
        new_blanket, true_causal_effect = calculate_best_intervention_and_effect(
            static_sem=initial_structural_equation_model,
            dynamic_sem=structural_equation_model,
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

        GT.append(true_causal_effect)

    return GT, optimal_assigned_blankets


# def compute_gap(T, data, best_objective_values, n_replicates,
#                 where_converge_dict = None, count_optimal_set = None,
#                 use_initial_value = True):
#     dict_gap = {method: [None]*T for method in list(data.keys())}

#     for method in list(data.keys()):
#         for t in range(T):
#             for r in range(n_replicates):

#                 values = data[method][r][1][t]

#                 initial = values[0]
#                 if initial == 10000000.:
#                     initial = values[1]

#                 last = values[-1]

#                 if last-initial == 0.:
#                     gap = 0.
#                 else:
#                     if isinstance(best_objective_values, dict):
#                         if use_initial_value:
#                             gap =  np.clip((last-initial)/(best_objective_values[method][r][t]-initial), 0., 1.)
#                         else:
#                             gap =  (last)/(best_objective_values[method][r][t])
#                     else:
#                         if use_initial_value:
#                             gap =  np.clip((last-initial)/(best_objective_values[t]-initial), 0., 1.)
#                         else:
#                             gap =  (last)/(best_objective_values[t])

#                 if dict_gap[method][t] is None:
#                     dict_gap[method][t] = [gap]
#                 else:
#                     dict_gap[method][t].append(gap)

#     dict_gap_summary = {method: [None]*T for method in list(data.keys())}
#     for method in list(data.keys()):
#         for t in range(T):
#             dict_gap_summary[method][t] = [np.mean(dict_gap[method][t]), np.std(dict_gap[method][t])]

#     if where_converge_dict is not None:
#         dict_gap_iters_summary = deepcopy(dict_gap_summary)
#         for t in range(T):
#             for method in dict_gap_summary.keys():
#                 percent_iters = n_trials/where_converge_dict[method][t]
#                 dict_gap_iters_summary[method][t] = list(np.asarray(dict_gap_summary[method][t])*percent_iters)

#     if count_optimal_set is not None:
#         dict_gap_iters_set_summary = deepcopy(dict_gap_iters_summary)
#         for t in range(T):
#             for method in dict_gap_iters_summary.keys():
#                 percent_iters = count_optimal_set[method][t]/n_replicates
#                 dict_gap_iters_set_summary[method][t] = list(np.asarray(dict_gap_iters_summary[method][t])*percent_iters)

#     return dict_gap, dict_gap_summary, dict_gap_iters_summary, dict_gap_iters_set_summary

# def alternative_gap(T, data, best_objective_values, total_initial_list, n_replicates,
#                 where_converge_dict = None, count_optimal_set = None,
#                 use_initial_value = True):
#     dict_gap = {method: [None]*T for method in list(data.keys())}

#     for method in list(data.keys()):
#         for t in range(T):
#             for r in range(n_replicates):
#                 values = data[method][r][1][t]

#                 initial = total_initial_list[t][r]

#                 last = values[-1]

#                 if last-initial == 0.:
#                     gap = 0.
#                 else:
#                     gap =  np.clip((last-initial)/(best_objective_values[t]-initial), 0., 1.)

#                 if dict_gap[method][t] is None:
#                     dict_gap[method][t] = [gap]
#                 else:
#                     dict_gap[method][t].append(gap)

#     dict_gap_summary = {method: [None]*T for method in list(data.keys())}
#     for method in list(data.keys()):
#         for t in range(T):
#             dict_gap_summary[method][t] = [np.mean(dict_gap[method][t]), np.std(dict_gap[method][t])]

#     if where_converge_dict is not None:
#         dict_gap_iters_summary = deepcopy(dict_gap_summary)
#         for t in range(T):
#             for method in dict_gap_summary.keys():
#                 percent_iters = n_trials/where_converge_dict[method][t]
#                 dict_gap_iters_summary[method][t] = list(np.asarray(dict_gap_summary[method][t])*percent_iters)

#     if count_optimal_set is not None:
#         dict_gap_iters_set_summary = deepcopy(dict_gap_iters_summary)
#         for t in range(T):
#             for method in dict_gap_iters_summary.keys():
#                 percent_iters = count_optimal_set[method][t]/n_replicates
#                 dict_gap_iters_set_summary[method][t] = list(np.asarray(dict_gap_iters_summary[method][t])*percent_iters)

#     return dict_gap, dict_gap_summary, dict_gap_iters_summary, dict_gap_iters_set_summary
