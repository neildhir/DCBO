import numpy as np
import argparse
import sys

sys.path.append("../src/")
sys.path.append("..")
from src.experiments import run_methods_replicates_parallel
from src.structural_equation_models import LinearToyDBN
from src.structural_equation_models import make_LinearToyDBN_SEM_hat
from src.dot_functions import generate_CGM

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=1, type=int, help="seed")
parser.add_argument("--number_of_trials", default=3, type=int, help="number_of_trials")
parser.add_argument("--method", default="BO", type=str, help="method to run")
args = parser.parse_args()


# Set parameters
seed = args.seed
number_of_trials = args.number_of_trials
method = args.method

SEM = LinearToyDBN
make_SEM_hat = make_LinearToyDBN_SEM_hat
intervention_domain = {"X": [-5, 5], "Z": [-5, 20]}
GT_path = "../data/synthetic/all_CE.npy"
n_restart = 10
initial_interventions = True
online = True
N_obs = 100
N_t = 50
CBO_concat_DO = False
CBO_concat_DI = False
DCBO_concat_DO = False
observational_samples = None

T = 3
graph = generate_CGM(0, T - 1, spatial_connection_topo="chain", verbose=False)
GT = np.load(GT_path, allow_pickle=True)


run_methods_replicates_parallel(
    seed=seed,
    graph=graph,
    SEM=SEM,
    make_SEM_hat=make_SEM_hat,
    intervention_domain=intervention_domain,
    method=method,
    observational_samples=observational_samples,
    GT=GT,
    number_of_trials=number_of_trials,
    initial_interventions=initial_interventions,
    n_restart=n_restart,
    online=online,
    N_obs=N_obs,
    N_t=N_t,
    CBO_concat_DO=CBO_concat_DO,
    CBO_concat_DI=CBO_concat_DI,
    DCBO_concat_DO=DCBO_concat_DO,
)


print("Done!")
