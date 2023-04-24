from collections import OrderedDict
import numpy as np
from emukit.core.acquisition import Acquisition


class COST(Acquisition):
    def __init__(self, costs_functions, exploration_set, base_target: str):
        self.costs_functions = costs_functions
        self.exploration_set = exploration_set
        self.base_target = base_target

    def evaluate(self, x):
        if len(self.exploration_set) == 1:
            #  Univariate intervention
            return self.costs_functions[self.exploration_set[0]](x)
        else:
            # Multivariate intervention
            cost = []
            for i, es_member in enumerate(self.exploration_set):
                cost.append(self.costs_functions[es_member](x[:, i]))
            return sum(cost)

    @property
    def has_gradients(self):
        return True

    def evaluate_with_gradients(self, x):
        return self.evaluate(x), np.zeros(x.shape)


## Define a cost variable for each intervention
def cost_fix_equal(intervention_value):
    # The argument for this function is a dummy variable.
    fix_cost = 1.0
    return fix_cost


## Define a cost variable for each intervention
def cost_fix_different(intervention_value):
    # The argument for this function is a dummy variable.
    fix_cost = int(np.random.randint(1, 10, 1))
    return fix_cost


## Define a cost variable for each intervention
def cost_fix_equal_variable(intervention_value):
    fix_cost = 1.0
    return np.sum(np.abs(intervention_value)) + fix_cost


def define_costs(canonical_manipulative_variables, base_target, type_cost: int):
    assert base_target not in canonical_manipulative_variables

    if type_cost == 1:
        costs = OrderedDict([(var, cost_fix_equal) for var in canonical_manipulative_variables])

    if type_cost == 2:
        costs = OrderedDict([(var, cost_fix_different) for var in canonical_manipulative_variables])

    if type_cost == 3:
        costs = OrderedDict([(var, cost_fix_equal_variable) for var in canonical_manipulative_variables])

    return costs


def total_intervention_cost(
    exploration_set: tuple, costs, intervention_levels,
):
    total_intervention_cost = 0.0

    for i, es_member in enumerate(exploration_set):
        total_intervention_cost += costs[es_member](intervention_levels[:, i])
    return total_intervention_cost
