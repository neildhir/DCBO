from numpy import hstack, newaxis, array
import numpy as np
from pandas import DataFrame
from seaborn import jointplot, set_style, set_context
import matplotlib.pyplot as plt
import datetime
from numpy import linspace, meshgrid, sqrt
from src.sequential_causal_functions import sequentially_sample_model
from src.utilities import get_cumulative_cost_mean_and_std, get_mean_and_std
import seaborn as sns


# def elaborate(number_of_interventions, n_replicates, n_trials, data, best_objective_values, T):
#     # Replace initial data point
#     if number_of_interventions is None:
#         for model in data.keys():
#             for r in range(n_replicates):
#                 for t in range(T):
#                     if data[model][r][1][t][0] == 10000000.0:
#                         data[model][r][1][t][0] = data[model][r][1][t][1]

#     # Aggregate data
#     per_trial_cost = {model: [] for model in data.keys()}
#     optimal_outcome_values_during_trials = {model: [] for model in data.keys()}

#     for i in range(n_replicates):
#         for model in data.keys():
#             per_trial_cost[model].append(data[model][i][0])
#             optimal_outcome_values_during_trials[model].append(data[model][i][1])

#     # Aggregate data
#     exp_per_trial_cost = get_cumulative_cost_mean_and_std(per_trial_cost, T, repeats=n_replicates)
#     exp_optimal_outcome_values_during_trials = get_mean_and_std(
#         optimal_outcome_values_during_trials, T, repeats=n_replicates
#     )

#     # For ABO and BO we make the cost start from 0 as in the competing models
#     # We then augement the dimension of the y values to plot to ensure they can be plotted
#     if "BO" in exp_per_trial_cost.keys() and "ABO" in exp_per_trial_cost.keys():
#         initial_value_BO_ABO = np.max(
#             (
#                 exp_optimal_outcome_values_during_trials["BO"][0][0][0],
#                 exp_optimal_outcome_values_during_trials["ABO"][0][0][0],
#             )
#         )

#     for model in exp_per_trial_cost.keys():
#         if model == "BO" or model == "ABO":
#             costs = exp_per_trial_cost[model]
#             values = exp_optimal_outcome_values_during_trials[model]
#             for t in range(T):
#                 values_t = values[t]
#                 exp_per_trial_cost[model][t] = np.asarray([0] + list(costs[t]))

#                 exp_optimal_outcome_values_during_trials[model][t] = tuple(
#                     [np.asarray([values_t[i][0]] + list(values_t[i])) for i in range(2)]
#                 )

#     # # Clip values so they are not lower than the min
#     # for model in exp_per_trial_cost.keys():
#     #     for t in range(T):
#     #         clipped = np.clip(
#     #             exp_optimal_outcome_values_during_trials[model][t][0], a_min=best_objective_values[t], a_max=0
#     #         )
#     #         exp_optimal_outcome_values_during_trials[model][t] = (
#     #             clipped,
#     #             exp_optimal_outcome_values_during_trials[model][t][1],
#     #         )

#     return exp_optimal_outcome_values_during_trials, exp_per_trial_cost


def select_data(samples, variables, time_indices):
    # Extract data
    joint_sample = []
    for v, t in zip(variables, time_indices):
        joint_sample.append(samples[v][:, t][:, newaxis])

    return hstack(joint_sample)


def plot_joint(sem, graph, x, y, samples=None, num_samples=100):
    assert all([i in graph.nodes for i in (x, y)])
    # No samples passed so we sample
    if not samples:
        samples = sequentially_sample_model(sem, graph, num_samples)
    # Check that we have passed an actual temporal index for the x and y before we plot
    var1, time_index1 = x.split("_")
    assert int(time_index1) is not None
    var2, time_index2 = y.split("_")
    assert int(time_index2) is not None
    # Get data
    data = select_data(samples, (var1, var2), (int(time_index1), int(time_index2)))
    # Plot
    jointplot(data=DataFrame(data, columns=[x, y]), x=x, y=y)


def plot_target_intervention_response(intervention_domain, mean_causal_effect, target, iv_set):

    set_context("notebook", font_scale=1.5)
    set_style("darkgrid")

    if len(iv_set) == 1:
        plt.figure(figsize=(10, 6))
        iv, _ = iv_set[0].split("_")  # iv == intervention variable

        plt.plot(intervention_domain, mean_causal_effect)
        plt.xlabel("${}$".format(iv_set[0].lower()))
        plt.ylabel("$E[{} \mid do({} = {})]$".format(target, iv_set[0], iv_set[0].lower()))
        plt.xlim(min(intervention_domain), max(intervention_domain))

    elif len(iv_set) == 2:

        set_style("whitegrid")

        fig = plt.figure(figsize=(12, 12))
        ax = plt.axes(projection="3d")
        size_intervention_grid = int(sqrt(len(mean_causal_effect)))
        a, b = min(intervention_domain[:, 0]), max(intervention_domain[:, 0])
        c, d = min(intervention_domain[:, 1]), max(intervention_domain[:, 1])

        # Data to plot
        iv_range_1 = linspace(a, b, size_intervention_grid)[:, None]
        iv_range_2 = linspace(c, d, size_intervention_grid)[:, None]
        iv_range_1, iv_range_2 = meshgrid(iv_range_1, iv_range_2)
        Y_for_plot = mean_causal_effect.reshape(size_intervention_grid, size_intervention_grid)  # The mean

        # PLOT
        ax.plot_surface(
            iv_range_1,
            iv_range_2,
            Y_for_plot,
            cmap="viridis",
            edgecolor="none",
            linewidth=0,
            antialiased=False,
        )

        # Axes titles
        ax.set_xlabel("${}$".format(iv_set[0].lower()), labelpad=30)
        ax.set_ylabel("${}$".format(iv_set[1].lower()), labelpad=30)
        ax.set_zlabel(
            "$E[{} \mid do({} = {}), do({} = {})]$".format(
                target, iv_set[0], iv_set[0].lower(), iv_set[1], iv_set[1].lower()
            ),
            labelpad=20,
        )
        ax.grid(True)
        plt.tight_layout()
    else:
        raise ValueError("We can only plot interventions spaces of max two dimensions.")

    plt.show()


def make_contour_surface_plot(
    interventional_grids,
    causal_effects,
    interventional_variable_limits,
    optimal_int_level=None,
    filename=None,
):
    sns.set_context("paper", font_scale=1.7)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")  # , proj_type="ortho")

    X, Y = meshgrid(interventional_grids[("X",)], interventional_grids[("Z",)])
    CE = array(causal_effects)
    Z = CE.reshape(X.shape)

    ce_ceil = np.ceil(max(causal_effects))
    ce_floor = np.floor(min(causal_effects))

    # Surface
    ax.plot_surface(
        X,
        Y,
        Z,
        rstride=3,
        cstride=3,
        alpha=0.4,
        cmap="autumn_r",
        antialiased=False,
        linewidth=0,
        zorder=1,
    )
    # Contour
    ax.contour(
        X,
        Y,
        Z,
        10,
        zdir="z",
        cmap="autumn_r",
        linestyles="solid",
        offset=ce_floor,
        zorder=2,
    )
    # ax.contour(X, Y, Z, 20, colors="k", linestyles="solid")
    # ax.set_xlim(interventional_variable_limits["X"][0], interventional_variable_limits["X"][1])
    # ax.set_ylim(
    #     interventional_variable_limits["Z"][0], interventional_variable_limits["Z"][1],
    # )
    ax.set_zlim(ce_floor, ce_ceil)

    ax.set_xlabel(r"$X$", labelpad=10)
    ax.set_ylabel(r"$Z$", labelpad=10)
    ax.set_zlabel(r"$\mathbb{E}[Y \mid do(X,Z)]$", labelpad=10)

    if optimal_int_level:
        ax.scatter(
            optimal_int_level[0],  # X
            optimal_int_level[1],  # Z
            ce_floor,
            s=100,
            marker="o",
            c="g",
            label="Optimal intervention level",
            linewidths=3,
            alpha=0.5,
            zorder=10,
        )
        plt.legend(
            ncol=2, fontsize="large", loc="upper right", framealpha=1.0
        )  # , bbox_to_anchor=(0.5, 0.125), framealpha=1.0)

    if filename:
        # Set reference time for save
        now = datetime.datetime.now()
        fig.savefig(
            "../figures/synthetic/intervene_XZ_" + filename + "_" + now.strftime("%Y-%m-%d-%H:%M") + ".pdf",
            bbox_inches="tight",
        )

    plt.show()


def opt_results_outcome_per_temporal_index(time_index, ground_truth, bo, cbo, sbo, scibo, filename=None):
    sns.set_theme(
        context="paper",
        style={"ticks", {"xtick.direction": "in", "ytick.direction": "in"}},
        palette="deep",
        font="sans-serif",
        font_scale=1.3,
    )

    nr_trials = bo.number_of_trials

    fig = plt.figure(figsize=(7, 3))  # Change to golden ratio
    ax = fig.add_subplot(111)

    ax.hlines(
        y=ground_truth,
        xmin=0,
        xmax=nr_trials,
        linewidth=2,
        color="k",
        ls="--",
        label="Ground truth",
    )
    ax.set_xlim(0, nr_trials)
    labels = ["BO", "CBO", "SCIBO w.o. CP", "SCIBO"]
    for i, model in enumerate([bo, cbo, sbo, scibo]):
        ax.plot(
            np.cumsum(model.per_trial_cost[time_index])[1:],
            model.optimal_outcome_values_during_trials[time_index],
            lw=2,
            alpha=0.75,
            label=labels[i],
        )

    ax.legend(ncol=3, loc="upper right", fontsize="small", frameon=False)
    ax.set_xlabel("Cost")
    ax.set_ylabel(r"$\mathbb{E}[Y \mid do(X_s = x_s)]$")

    if filename:
        # Set reference time for save
        now = datetime.datetime.now()
        fig.savefig(
            "../figures/synthetic/opt_curves_" + filename + "_" + now.strftime("%Y-%m-%d-%H:%M") + ".pdf",
            bbox_inches="tight",
        )
    plt.show()


def plot_opt_curve(time_index, ground_truth, cost, outcome, filename=None):
    """
    Plots the outcome of _one_ call of SCIBO/CBO/BO and so on.

    Ground truth for the toy DBN with four time-steps:
    ground_truth = [-2.2598164061785635,-4.598172847301528,-6.948783927331318,-9.272325039971896]

    Parameters
    ----------
    time_index : int
        Which time index we're interested in plotting
    ground_truth : float
        The ground truth for this SEM
    cost : list
        This is the cost array from this experiment, already indexed by time_index
    outcome : list
        This is the value of the outcome variable, already indexed by time_index
    filename : str, optional
        The name of the file we want to save this as
    """
    assert len(outcome) == len(cost)

    sns.set_theme(
        context="paper",
        style="ticks",
        palette="deep",
        font="sans-serif",
        font_scale=1.3,
    )
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

    width = 6
    fig = plt.figure(figsize=(width, width / 1.61803398875))  # Change to golden ratio
    ax = fig.add_subplot(111)

    # Cum sum
    cs = np.cumsum(cost)

    # Ground truth
    ax.hlines(
        y=ground_truth,
        xmin=0,
        xmax=round(max(cs)),
        linewidth=2,
        color="red",
        ls="-",
        label="Ground truth",
    )

    a = np.array(outcome)
    # Only valid when task is 'min'
    out_max = np.ceil(a[np.isfinite(a)].max())

    # Cost vs outcome
    ax.plot(cs, outcome, lw=2, ls="-", alpha=1)

    ax.set_xlabel("Cost")
    ax.set_xlim(0, round(max(cs)))
    ax.set_ylabel("$E[Y_{} \mid do(X^s_{} = x^s_{})]$".format(time_index, time_index, time_index))
    ax.set_ylim(np.floor(ground_truth), out_max)

    if filename:
        # Set reference time for save
        fig.savefig(
            "../figures/synthetic/main_opt_curves_" + filename + ".pdf",
            bbox_inches="tight",
        )

    plt.show()


def plot_expected_opt_curve(time_index, ground_truth, cost, outcome, plot_params, filename=None):

    sns.set_theme(
        context="paper",
        style="ticks",
        palette="deep",
        font="sans-serif",
        font_scale=1.3,
    )
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

    assert cost.keys() == outcome.keys()

    width = 5
    # fig = plt.figure(figsize=(width, width / 1.61803398875))
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(111)

    ax.tick_params(axis="both", which="major", labelsize=plot_params["size_ticks"])

    cs_all = []
    out_all = []
    for i, model in enumerate(cost.keys()):

        cs = cost[model][time_index]
        cs_max = round(max(cs))
        cs_all.append(cs_max)

        a = np.array(outcome[model][time_index])
        # Only valid when task is 'min'
        out_max = np.ceil(a[np.isfinite(a)].max())
        out_all.append(out_max)

        # Mean
        ax.plot(
            cs,
            outcome[model][time_index][0],
            linewidth=plot_params["linewidth"],
            ls=plot_params["line_styles"][model],
            label=plot_params["labels"][model],
            color=plot_params["colors"][model],
        )
        # plus/minus one std
        lower = outcome[model][time_index][0] - outcome[model][time_index][1]
        upper = outcome[model][time_index][0] + outcome[model][time_index][1]

        ax.fill_between(
            cs,
            lower,
            upper,
            alpha=plot_params["alpha"],
            color=plot_params["colors"][model],
        )

    # Ground truth
    if isinstance(ground_truth, dict):
        for method in ground_truth.keys():
            ax.hlines(
                y=ground_truth[method],
                xmin=0,
                xmax=np.floor(max(cs_all)) + 1.0,
                linewidth=plot_params["linewidth_opt"],
                color=plot_params["colors"][method],
                ls=plot_params["line_styles"]["True"],
                label=plot_params["labels"]["True"],
                zorder=10,
                alpha=0.3,
            )
    else:
        ax.hlines(
            y=ground_truth,
            xmin=0,
            xmax=np.floor(max(cs_all)) + 1.0,
            linewidth=plot_params["linewidth_opt"],
            color=plot_params["colors"]["True"],
            ls=plot_params["line_styles"]["True"],
            label=plot_params["labels"]["True"],
            zorder=10,
        )

    # Cost
    ax.set_xlabel("Cumulative Cost", fontsize=plot_params["size_labels"])
    ax.set_xlim(0, plot_params["xlim_max"])
    if isinstance(ground_truth, dict):
        ax.set_ylim(
            ground_truth["DCBO"] - 0.2,
            np.max(outcome[model][time_index][0] + outcome[model][time_index][1] + 2.0),
        )
    else:
        ax.set_ylim(
            ground_truth - 0.2,
            np.max(outcome[model][time_index][0] + outcome[model][time_index][1] + 1.0),
        )
    # Outcome value
    ax.set_ylabel("$y_{}^\star$".format(time_index), fontsize=plot_params["size_labels"])
    ax.legend(
        ncol=plot_params["ncols"],
        loc=plot_params["loc_legend"],
        fontsize="large",
        frameon=True,
    )

    if filename:
        # Set reference time for save
        now = datetime.datetime.now()
        fig.savefig(
            "../figures/synthetic/opt_curves_" + filename + "_" + now.strftime("%d%m%Y_%H%M") + ".pdf",
            bbox_inches="tight",
        )
    return fig


def plot_average_curve(
    T,
    ground_truth,
    cost,
    outcome,
    plot_params,
    online_option,
    n_obs,
    n_replicates,
    optionCBO,
    optionDCBO,
    filename=None,
):

    sns.set_theme(
        context="paper",
        style="ticks",
        palette="deep",
        font="sans-serif",
        font_scale=1.3,
    )
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

    assert cost.keys() == outcome.keys()

    fig = plt.figure(figsize=(7, 5))

    cs_all = []
    out_all = []
    for i, model in enumerate(cost.keys()):

        cs = np.mean(np.vstack([cost[model][i] for i in range(T)]), axis=0)
        cs_max = round(max(cs))
        cs_all.append(cs_max)

        mean_values = np.mean(np.vstack([outcome[model][i][0] for i in range(T)]), axis=0)
        # Only valid when task is 'min'
        out_max = np.ceil(mean_values[np.isfinite(mean_values)].max())
        out_all.append(out_max)

        # Mean
        plt.plot(
            cs,
            mean_values,
            linewidth=plot_params["linewidth"],
            ls=plot_params["line_styles"][model],
            label=plot_params["labels"][model],
            color=plot_params["colors"][model],
        )
        # plus/minus one std
        std_values = np.mean(np.vstack([outcome[model][i][1] for i in range(T)]), axis=0)
        lower = mean_values - std_values
        upper = mean_values + std_values

        plt.fill_between(
            cs,
            lower,
            upper,
            alpha=plot_params["alpha"],
            color=plot_params["colors"][model],
        )

    # Ground truth
    plt.hlines(
        y=np.mean([ground_truth[i] for i in range(T)]),
        xmin=0,
        xmax=np.floor(max(cs_all)) + 1.0,
        linewidth=plot_params["linewidth_opt"],
        color=plot_params["colors"]["True"],
        ls=plot_params["line_styles"]["True"],
        label=plot_params["labels"]["True"],
        zorder=10,
    )

    # Cost
    plt.xlabel("Cumulative Cost", fontsize=plot_params["size_labels"])
    plt.xlim(0, plot_params["xlim_max"])
    # plt.xlim(0, 2)
    plt.ylim(
        np.mean([ground_truth[i] for i in range(T)]) - 0.1,
        np.max(mean_values + std_values),
    )
    plt.legend(
        ncol=plot_params["ncols"],
        loc=plot_params["loc_legend"],
        fontsize="large",
        frameon=True,
    )
    plt.title(
        online_option + ", " + n_obs + ", " + n_replicates + " OptionCBO:" + optionCBO + " OptionDCBO:" + optionDCBO,
        size=plot_params["size_labels"],
    )
    if filename:
        # Set reference time for save
        now = datetime.datetime.now()
        fig.savefig(
            "../figures/synthetic/average_opt_curves_" + filename + "_" + now.strftime("%d%m%Y_%H%M") + ".pdf",
            bbox_inches="tight",
        )


def plot_expected_opt_curve_paper(
    T: int,
    ground_truth,
    cost,
    outcome,
    plot_params,
    ground_truth_dict=None,
    filename: str = None,
    fig_size: tuple = (15, 3),
    save_fig=False,
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

            a = np.array(outcome[model][time_index])
            out_max = np.ceil(a[np.isfinite(a)].max())
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
                    alpha=plot_params["alpha"] + 0.7,
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
            axs[time_index].set_ylim(
                ground_truth[time_index] - 0.2,
                ground_truth[time_index] + 3,
            )

        # Annotate (1, 3)
        bbox_props = dict(boxstyle="round,pad=0.3", fc="w", ec="k", lw=1.5)

        axs[time_index].annotate(
            "$t = {}$".format(time_index),
            xy=(0, 0.0),
            xycoords="axes fraction",
            xytext=(0.45, 0.85),
            bbox=bbox_props,
            fontsize=plot_params["size_labels"],
        )

        # Outcome value
        if time_index == 0:
            axs[time_index].set_ylabel("$y_t^\star$", fontsize=plot_params["size_labels"])

    lgd = plt.legend(
        ncol=plot_params["ncols"],
        handles=linet_list,
        bbox_to_anchor=(0, -0.6),
        loc=plot_params["loc_legend"],
        fontsize="large",
    )

    if save_fig and filename:
        # Set reference time for save
        now = datetime.datetime.now()
        fig.savefig(
            "../figures/synthetic/opt_curves_" + filename + "_" + now.strftime("%d%m%Y_%H%M") + ".pdf",
            bbox_extra_artists=(lgd,),
            bbox_inches="tight",
        )
