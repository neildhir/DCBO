import sys

sys.path.append("../src/")
sys.path.append("..")

import numpy as np
import argparse
from src.experiments import run_methods_replicates
from src.structural_equation_models import StationaryIndependentSEM, StationaryComplexSEM
from graph_functions import generate_CGM
from networkx import nx_agraph
from sem_estimate import make_sem_complex_hat, make_sem_independent_hat
import pygraphviz
from sequential_causal_functions import powerset

parser = argparse.ArgumentParser()
parser.add_argument("--T", default=3, type=int, help="T")
parser.add_argument("--n_rep", default=3, type=int, help="n_rep")
parser.add_argument("--number_of_trials", default=3, type=int, help="number_of_trials")
parser.add_argument("--method", default="DCBO", type=str, help="method to run")
parser.add_argument("--online", default=0, type=int, help="online")
parser.add_argument("--n_obs", default=100, type=int, help="n obs to start")
parser.add_argument("--n_t", default=1, type=int, help="n obs to follow")
parser.add_argument("--concat", default=0, type=int, help="concat")
parser.add_argument("--use_di", default=0, type=int, help="use_di")
parser.add_argument("--transfer_hp_i", default=0, type=int, help="transfer_hp_i")
parser.add_argument("--transfer_hp_o", default=0, type=int, help="transfer_hp_o")
parser.add_argument("--hp_i_prior", default=1, type=int, help="hp_i_prior")
parser.add_argument("--num_anchor_points", default=100, type=int, help="num_anchor_points")

parser.add_argument("--folder", default="toy_stationary_independent/seed_runs/", type=str, help="where to save data")
parser.add_argument("--graph", default="independent", type=str, help="which graph to consider")


args = parser.parse_args()

# Set experiment folder for results
folder = args.folder

# Set parameters
T = args.T
n_rep = args.n_rep
number_of_trials = args.number_of_trials
method = args.method
n_obs = args.n_obs
n_t = args.n_t
num_anchor_points = args.num_anchor_points

online_option = bool(args.online)
concat = bool(args.concat)
use_di = bool(args.use_di)
transfer_hp_i = bool(args.transfer_hp_i)
transfer_hp_o = bool(args.transfer_hp_o)
hp_i_prior = bool(args.hp_i_prior)

if args.graph == "independent":
    SEM = StationaryIndependentSEM
    make_SEM_hat = make_sem_independent_hat
    intervention_domain = {"X": [-3, 3], "Z": [-3, 3]}
    graph_view = generate_CGM(0, T - 1, spatial_connection_topo="independent", verbose=True)  # Base topology
    Graph = nx_agraph.from_agraph(pygraphviz.AGraph(graph_view.source))
    number_of_trials_BO_ABO = number_of_trials
    exploration_sets = list(powerset(["X", "Z"]))
else:
    SEM = StationaryComplexSEM
    make_SEM_hat = make_sem_complex_hat
    intervention_domain = {"X": [-3, 3], "W": [-3, 3], "Z": [-3, 3]}
    graph_view = generate_CGM(0, T - 1, spatial_connection_topo="independent", verbose=True)  # Base topology
    graph_view_modified = "digraph { rankdir=LR; W_0 -> Z_0;  W_1 -> Z_1; W_2 -> Z_2; X_0 -> Y_0; Z_0 -> Y_0; X_1 -> Y_1; Z_1 -> Y_1; X_2 -> Y_2; Z_2 -> Y_2; X_0 -> X_1; Y_0 -> Y_1; Z_0 -> Z_1;X_1 -> X_2; Y_1 -> Y_2; Z_1 -> Z_2; { rank=same; X_0 Z_0 Y_0 } { rank=same; X_1 Z_1 Y_1 } { rank=same; X_2 Z_2 Y_2 }  }"
    Graph = nx_agraph.from_agraph(pygraphviz.AGraph(graph_view_modified))
    number_of_trials_BO_ABO = 40
    exploration_sets = list(powerset(["X", "W", "Z"]))
    exploration_sets_dcbo_cbo = exploration_sets[:-1]
    del exploration_sets_dcbo_cbo[5]
    exploration_sets = exploration_sets_dcbo_cbo


GT_path = "../data/" + folder + "/GT.npy"
assigned_blanket_path = "../data/" + folder + "/optimal_assigned_blankets.npy"
n_restart = 1
initial_interventions = False
n_obs_t = None
noise_experiment = False
# methods_list = [method]
methods_list = ["DCBO", "CBO", "ABO", "BO"]
assign_optimal_blanket = False

GT = np.load(GT_path, allow_pickle=True)
optimal_assigned_blankets = np.load(assigned_blanket_path, allow_pickle=True)

for s in range(0, 10):
    run_methods_replicates(
        Graph,
        SEM,
        make_SEM_hat,
        intervention_domain,
        methods_list,
        obs_samples=None,
        ground_truth=GT,
        exploration_sets=exploration_sets,
        total_timesteps=T,
        reps=n_rep,
        number_of_trials=number_of_trials,
        number_of_trials_BO_ABO=number_of_trials_BO_ABO,
        n_restart=n_restart,
        save_data=True,
        n_obs=n_obs,
        cost_structure=1,
        optimal_assigned_blankets=optimal_assigned_blankets,
        debug_mode=False,
        online=online_option,
        concat=concat,
        use_di=use_di,
        transfer_hp_o=transfer_hp_o,
        transfer_hp_i=transfer_hp_i,
        hp_i_prior=hp_i_prior,
        estimate_sem=True,
        folder=folder,
        num_anchor_points=num_anchor_points,
        n_obs_t=n_obs_t,
        sample_anchor_points=True,
        controlled_experiment=True,
        noise_experiment=noise_experiment,
        seed=s,
    )

print("Done!")
