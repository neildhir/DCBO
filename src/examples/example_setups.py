from networkx.drawing import nx_agraph
import pygraphviz
from src.utils.sem_utils.toy_sems import StationaryDependentSEM
from src.utils.dag_utils.graph_functions import make_graphical_model


def setup_stat_scm(T: int = 3):

    #  Load SEM from figure 1 in paper (upper left quadrant)
    SEM = StationaryDependentSEM()
    init_sem = SEM.static()
    sem = SEM.dynamic()

    dag_view = make_graphical_model(
        0, T - 1, topology="dependent", nodes=["X", "Z", "Y"], verbose=True
    )  # Base topology that we build on
    dag = nx_agraph.from_agraph(pygraphviz.AGraph(dag_view.source))

    return init_sem, sem, dag_view, dag

