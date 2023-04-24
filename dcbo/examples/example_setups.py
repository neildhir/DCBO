import pygraphviz
from networkx.drawing import nx_agraph
from dcbo.experimental.experiments import optimal_sequence_of_interventions
from dcbo.utils.dag_utils.graph_functions import make_graphical_model
from dcbo.utils.sem_utils.toy_sems import (
    PISHCAT_SEM,
    LinearMultipleChildrenSEM,
    NonStationaryDependentSEM,
    StationaryDependentMultipleChildrenSEM,
    StationaryDependentSEM,
    StationaryIndependentSEM,
)
from dcbo.utils.sequential_intervention_functions import get_interventional_grids
from dcbo.utils.utilities import powerset


def setup_PISHCAT(T: int = 3):
    # Setup used in `Developing Optimal Causal Cyber-Defence Agents via Cyber Security Simulation`.

    SEM = PISHCAT_SEM()
    init_sem = SEM.static()
    sem = SEM.dynamic()

    slice_node_set = ["P", "I", "S", "H", "C", "A", "T"]
    dag_view = make_graphical_model(0, T - 1, topology="dependent", nodes=slice_node_set, verbose=True)
    G = nx_agraph.from_agraph(pygraphviz.AGraph(dag_view.source))

    # Modulate base structure to fit PISHCAT framework

    # Transitions
    for t in range(T - 1):
        # Add
        G.add_edge("S_{}".format(t), "H_{}".format(t + 1))
        # Remove
        G.remove_edge("P_{}".format(t), "P_{}".format(t + 1))
        G.remove_edge("I_{}".format(t), "I_{}".format(t + 1))
        G.remove_edge("H_{}".format(t), "H_{}".format(t + 1))
        G.remove_edge("C_{}".format(t), "C_{}".format(t + 1))
        G.remove_edge("A_{}".format(t), "A_{}".format(t + 1))
        G.remove_edge("T_{}".format(t), "T_{}".format(t + 1))

    # Emissions
    for t in range(T):
        # Remove
        G.remove_edge("P_{}".format(t), "I_{}".format(t))
        G.remove_edge("S_{}".format(t), "H_{}".format(t))
        G.remove_edge("C_{}".format(t), "A_{}".format(t))
        # Add
        G.add_edge("P_{}".format(t), "H_{}".format(t))
        G.add_edge("P_{}".format(t), "A_{}".format(t))
        G.add_edge("I_{}".format(t), "A_{}".format(t))
        G.add_edge("C_{}".format(t), "T_{}".format(t))

    #  Specifiy all the exploration sets based on the manipulative variables in the DAG
    exploration_sets = list(powerset(["P", "I"]))
    # Specify the intervention domain for each variable
    intervention_domain = {"P": [0, 1], "I": [0, 3]}
    # Specify a grid over each exploration and use the grid to find the best intevention value for that ES
    interventional_grids = get_interventional_grids(exploration_sets, intervention_domain, size_intervention_grid=100)

    _, optimal_interventions, true_objective_values, _, _, all_causal_effects = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=init_sem,
        structural_equation_model=sem,
        G=G,
        T=T,
        model_variables=slice_node_set,
        target_variable="T",
    )

    return (
        init_sem,
        sem,
        dag_view,
        G,
        exploration_sets,
        intervention_domain,
        true_objective_values,
        optimal_interventions,
        all_causal_effects,
    )


def setup_linear_multiple_children_scm(T: int = 3):

    #  Load SEM from figure 1 in paper (upper left quadrant)
    SEM = LinearMultipleChildrenSEM()
    init_sem = SEM.static()
    sem = SEM.dynamic()

    dag_view = make_graphical_model(
        0, T - 1, topology="dependent", nodes=["X", "Z", "Y"], verbose=True
    )  # Base topology that we build on
    G = nx_agraph.from_agraph(pygraphviz.AGraph(dag_view.source))
    G.add_edges_from([("X_0", "Y_0"), ("X_1", "Y_1"), ("X_2", "Y_2")])

    #  Specifiy all the exploration sets based on the manipulative variables in the DAG
    exploration_sets = list(powerset(["X", "Z"]))
    # Specify the intervention domain for each variable
    intervention_domain = {"X": [-4, 1], "Z": [-3, 3]}
    # Specify a grid over each exploration and use the grid to find the best intevention value for that ES
    interventional_grids = get_interventional_grids(exploration_sets, intervention_domain, size_intervention_grid=100)

    _, optimal_interventions, true_objective_values, _, _, all_causal_effects = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=init_sem,
        structural_equation_model=sem,
        G=G,
        T=T,
        model_variables=["X", "Z", "Y"],
        target_variable="Y",
    )

    return (
        init_sem,
        sem,
        dag_view,
        G,
        exploration_sets,
        intervention_domain,
        true_objective_values,
        optimal_interventions,
        all_causal_effects,
    )


def setup_stat_multiple_children_scm(T: int = 3):

    #  Load SEM from figure 1 in paper (upper left quadrant)
    SEM = StationaryDependentMultipleChildrenSEM()
    init_sem = SEM.static()
    sem = SEM.dynamic()

    dag_view = make_graphical_model(
        0, T - 1, topology="dependent", nodes=["X", "Z", "Y"], verbose=True
    )  # Base topology that we build on
    G = nx_agraph.from_agraph(pygraphviz.AGraph(dag_view.source))
    G.add_edges_from([("X_0", "Y_0"), ("X_1", "Y_1"), ("X_2", "Y_2")])

    #  Specifiy all the exploration sets based on the manipulative variables in the DAG
    exploration_sets = list(powerset(["X", "Z"]))
    # Specify the intervention domain for each variable
    intervention_domain = {"X": [-4, 1], "Z": [-3, 3]}
    # Specify a grid over each exploration and use the grid to find the best intevention value for that ES
    interventional_grids = get_interventional_grids(exploration_sets, intervention_domain, size_intervention_grid=100)

    _, optimal_interventions, true_objective_values, _, _, all_causal_effects = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=init_sem,
        structural_equation_model=sem,
        G=G,
        T=T,
        model_variables=["X", "Z", "Y"],
        target_variable="Y",
    )

    return (
        init_sem,
        sem,
        dag_view,
        G,
        exploration_sets,
        intervention_domain,
        true_objective_values,
        optimal_interventions,
        all_causal_effects,
    )


def setup_stat_scm(T: int = 3):

    #  Load SEM from figure 1 in paper (upper left quadrant)
    SEM = StationaryDependentSEM()
    init_sem = SEM.static()
    sem = SEM.dynamic()

    G_view = make_graphical_model(
        0, T - 1, topology="dependent", nodes=["X", "Z", "Y"], verbose=True
    )  # Base topology that we build on
    G = nx_agraph.from_agraph(pygraphviz.AGraph(G_view.source))

    #  Specifiy all the exploration sets based on the manipulative variables in the DAG
    exploration_sets = list(powerset(["X", "Z"]))
    # Specify the intervention domain for each variable
    intervention_domain = {"X": [-4, 1], "Z": [-3, 3]}
    # Specify a grid over each exploration and use the grid to find the best intevention value for that ES
    interventional_grids = get_interventional_grids(exploration_sets, intervention_domain, size_intervention_grid=100)

    _, optimal_interventions, true_objective_values, _, _, all_causal_effects = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=init_sem,
        structural_equation_model=sem,
        G=G,
        T=T,
        model_variables=["X", "Z", "Y"],
        target_variable="Y",
    )

    return (
        init_sem,
        sem,
        G_view,
        G,
        exploration_sets,
        intervention_domain,
        true_objective_values,
        optimal_interventions,
        all_causal_effects,
    )


def setup_ind_scm(T: int = 3):

    #  Load SEM from figure 1 in paper (upper left quadrant)
    SEM = StationaryIndependentSEM()
    init_sem = SEM.static()
    sem = SEM.dynamic()

    G_view = make_graphical_model(
        0, T - 1, topology="independent", nodes=["X", "Z", "Y"], target_node="Y", verbose=True
    )  # Base topology that we build on
    G = nx_agraph.from_agraph(pygraphviz.AGraph(G_view.source))

    #  Specifiy all the exploration sets based on the manipulative variables in the DAG
    exploration_sets = list(powerset(["X", "Z"]))
    # Specify the intervention domain for each variable
    intervention_domain = {"X": [-4, 1], "Z": [-3, 3]}
    # Specify a grid over each exploration and use the grid to find the best intevention value for that ES
    interventional_grids = get_interventional_grids(exploration_sets, intervention_domain, size_intervention_grid=100)

    _, _, true_objective_values, _, _, _ = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=init_sem,
        structural_equation_model=sem,
        G=G,
        T=T,
        model_variables=["X", "Z", "Y"],
        target_variable="Y",
    )

    return init_sem, sem, G_view, G, exploration_sets, intervention_domain, true_objective_values


def setup_nonstat_scm(T: int = 3):

    #  Load SEM from figure 3(c) in paper (upper left quadrant)
    SEM = NonStationaryDependentSEM(change_point=1)  #  Explicitly tell SEM to change at t=1
    init_sem = SEM.static()
    sem = SEM.dynamic()

    dag_view = make_graphical_model(
        0, T - 1, topology="dependent", nodes=["X", "Z", "Y"], verbose=True
    )  # Base topology that we build on
    dag = nx_agraph.from_agraph(pygraphviz.AGraph(dag_view.source))

    #  Specifiy all the exploration sets based on the manipulative variables in the DAG
    exploration_sets = list(powerset(["X", "Z"]))
    # Specify the intervention domain for each variable
    intervention_domain = {"X": [-4, 1], "Z": [-3, 3]}
    # Specify a grid over each exploration and use the grid to find the best intevention value for that ES
    interventional_grids = get_interventional_grids(exploration_sets, intervention_domain, size_intervention_grid=100)

    _, _, true_objective_values, _, _, all_causal_effects = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=init_sem,
        structural_equation_model=sem,
        G=dag,
        T=T,
        model_variables=["X", "Z", "Y"],
        target_variable="Y",
    )

    return (
        init_sem,
        sem,
        dag_view,
        dag,
        exploration_sets,
        intervention_domain,
        true_objective_values,
        all_causal_effects,
    )

