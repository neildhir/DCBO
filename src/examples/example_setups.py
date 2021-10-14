from networkx.drawing import nx_agraph
import pygraphviz
from src.utils.sem_utils.toy_sems import StationaryDependentSEM, NonStationaryDependentSEM
from src.utils.dag_utils.graph_functions import make_graphical_model
from src.experimental.experiments import optimal_sequence_of_interventions
from src.utils.sequential_intervention_functions import get_interventional_grids
from src.utils.utilities import powerset


def setup_stat_scm(T: int = 3):

    #  Load SEM from figure 1 in paper (upper left quadrant)
    SEM = StationaryDependentSEM()
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

    _, _, true_objective_values, _, _, _ = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=init_sem,
        structural_equation_model=sem,
        graph=dag,
        timesteps=T,
        model_variables=["X", "Z", "Y"],
        target_variable="Y",
    )

    return init_sem, sem, dag_view, dag, exploration_sets, intervention_domain, true_objective_values


def setup_nonstat_scm(T: int = 3):

    #  Load SEM from figure 3(c) in paper (upper left quadrant)
    SEM = NonStationaryDependentSEM(change_point=1) # Explicitly tell SEM to change at t=1
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

    _, _, true_objective_values, _, _, _ = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=init_sem,
        structural_equation_model=sem,
        graph=dag,
        timesteps=T,
        model_variables=["X", "Z", "Y"],
        target_variable="Y",
    )

    return init_sem, sem, dag_view, dag, exploration_sets, intervention_domain, true_objective_values
