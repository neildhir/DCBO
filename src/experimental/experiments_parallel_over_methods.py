import sys

sys.path.append("../src/")
sys.path.append("..")

import numpy as np
import argparse

from src.experiments import run_methods_replicates
from src.structural_equation_models import LinearToyDBN
from graph_functions import generate_CGM
from networkx import nx_agraph
from sem_estimate import make_sem_hat
import pygraphviz

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
parser.add_argument(
    "--num_anchor_points", default=100, type=int, help="num_anchor_points"
)
args = parser.parse_args()

# Set experiment folder for results
folder = "toy_stationary"

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

SEM = LinearToyDBN
make_SEM_hat = make_sem_hat
intervention_domain = {"X": [-5, 5], "Z": [-5, 20]}
GT_path = "../data/" + folder + "/GT.npy"
assigned_blanket_path = "../data/" + folder + "/optimal_assigned_blankets.npy"
n_restart = 1
initial_interventions = False
n_obs_t = None
noise_experiment = False
methods_list = [method]
assign_optimal_blanket = False


# Run algorithm
graph_view = generate_CGM(0, T - 1, spatial_connection_topo="chain", verbose=True)
Graph = nx_agraph.from_agraph(pygraphviz.AGraph(graph_view.source))

GT = np.load(GT_path, allow_pickle=True)


run_methods_replicates(
    Graph,
    SEM,
    make_SEM_hat,
    intervention_domain,
    methods_list,
    obs_samples=None,
    ground_truth=GT,
    total_timesteps=T,
    reps=n_rep,
    number_of_trials=number_of_trials,
    n_restart=n_restart,
    save_data=True,
    n_obs=n_obs,
    n_t=n_t,
    cost_structure=1,
    optimal_assigned_blankets=assign_optimal_blanket,
    debug_mode=False,
    online=online_option,
    concat=concat,
    use_di=use_di,
    transfer_hp_o=transfer_hp_o,
    transfer_hp_i=transfer_hp_i,
    hp_i_prior=hp_i_prior,
    n_obs_t=n_obs_t,
    folder=folder,
    sample_anchor_points=True,
    controlled_experiment=True,
    noise_experiment=noise_experiment,
)

print("Done!")
