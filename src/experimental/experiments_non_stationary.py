import numpy as np
import argparse
import sys
from networkx.drawing import nx_agraph
import pygraphviz

sys.path.append("../src/")
sys.path.append("..")
from src.experiments import run_methods_replicates
from src.structural_equation_models import NonStationarySEM, make_non_stationary_SEM_hat
from src.dot_functions import generate_CGM

parser = argparse.ArgumentParser()
parser.add_argument("--T", default=3, type=int, help="T")
parser.add_argument("--n_rep", default=3, type=int, help="n_rep")
parser.add_argument("--number_of_trials", default=3, type=int, help="number_of_trials")
parser.add_argument("--method", default="DCBO", type=str, help="method to run")
parser.add_argument("--N_obs", default=10, type=int, help="n obs to start")
parser.add_argument("--concat_DI", default=0, type=int, help="concat")

args = parser.parse_args()


# Set parameters
T = args.T
n_rep = args.n_rep
number_of_trials = args.number_of_trials
method = args.method
N_obs = args.N_obs
concat_DI = args.concat_DI


if concat_DI == 1:
    concat_DI = True
else:
    concat_DI = False


intervention_domain = {"X": [-5, 5], "Z": [-5, 20]}
optimal_assigned_blankets_path = "../data/non_stationary/optimal_assigned_blankets.npy"
n_restart = 1
initial_interventions = False
methods_list = [method]


SEM = NonStationarySEM
make_SEM_hat = make_non_stationary_SEM_hat

non_stationary_toy_graph = generate_CGM(
    0, T - 1, spatial_connection_topo="chain", verbose=False
)

optimal_assigned_blankets = np.load(optimal_assigned_blankets_path, allow_pickle=True)


run_methods_replicates(
    non_stationary_toy_graph,
    SEM,
    make_SEM_hat,
    intervention_domain,
    methods_list,
    observational_samples=None,
    GT=None,
    save_metrics=False,
    total_timesteps=T,
    n=n_rep,
    number_of_trials=number_of_trials,
    initial_interventions=initial_interventions,
    n_restart=n_restart,
    save_data=True,
    N_obs=N_obs,
    N_t=1,
    cost_structure=1,
    sample_count=10,
    concat_DI=concat_DI,
    optimal_assigned_blankets=optimal_assigned_blankets,
)


print("Done!")
