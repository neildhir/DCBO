from copy import deepcopy
from src.utils.sequential_intervention_functions import make_sequential_intervention_dictionary
import numpy as np
from itertools import combinations
from src.utils.utilities import convert_to_dict_of_temporal_lists, make_column_shape_2D, update_emission_pairs_keys
from .root import Root
from src.utils.sem_utils.emissions import fit_sem_complex
from src.utils.sem_utils.transitions import fit_sem_transition_functions_complex
from src.utils.gp_utils import fit_gp, sequential_sample_from_complex_model_hat
from src.utils.sequential_causal_functions import sequentially_sample_model

# TODO: we probably want a separate base class for DCBO and CBO as they share so many methods.


class BaseClassDCBO(Root):
    """
    Base class for the DCBO system.
    """

    def __init__(
        self,
        graph: str,
        sem: classmethod,
        make_sem_hat: callable,
        observational_samples: dict,
        intervention_domain: dict,
        interventional_samples: dict = None,  # Interventional data collected for specific intervention sets
        exploration_sets: list = None,
        estimate_sem: bool = False,
        base_target_variable: str = "y",
        task: str = "min",
        cost_type: int = 1,  # There are multiple options here
        number_of_trials=10,
        ground_truth=None,
        n_restart: int = 1,
        use_mc: bool = False,
        debug_mode: bool = False,
        online: bool = False,
        num_anchor_points: int = 100,
        args_sem=None,
        manipulative_variables=None,
        change_points: list = None,
        root_instrument: bool = None,
    ):
        super(BaseClassDCBO, self).__init__(
            graph,
            sem,
            make_sem_hat,
            observational_samples,
            intervention_domain,
            interventional_samples,
            exploration_sets,
            base_target_variable,
            task,
            cost_type,
            number_of_trials,
            ground_truth,
            n_restart,
            debug_mode,
            online,
            num_anchor_points,
            args_sem,
            manipulative_variables,
            change_points,
        )

        self.use_mc = use_mc

        self.sem_variables = [v.split("_")[0] for v in [v for v in self.graph.nodes if v.split("_")[1] == "0"]]
        self.root_instrument = root_instrument
        self.node_children = {node: None for node in self.graph.nodes}
        self.node_parents = {node: None for node in self.graph.nodes}
        self.emission_pairs = {}

        # Children of all nodes
        for node in self.graph.nodes:
            self.node_children[node] = list(self.graph.successors(node))

        #  Parents of all nodes
        for node in self.graph.nodes:
            self.node_parents[node] = tuple(self.graph.predecessors(node))

        emissions = {t: [] for t in range(self.T)}
        for e in self.graph.edges:
            _, inn_time = e[0].split("_")
            _, out_time = e[1].split("_")
            # Emission edge
            if out_time == inn_time:
                emissions[int(out_time)].append((e[0], e[1]))

        new_emissions = deepcopy(emissions)
        self.emissions = emissions
        for t in range(self.T):
            for a, b in combinations(emissions[t], 2):
                if a[1] == b[1]:
                    new_emissions[t].append(((a[0], b[0]), a[1]))
                    cond = [v for v in list(self.graph.predecessors(b[0])) if v.split("_")[1] == str(t)]
                    if len(cond) != 0 and cond[0] == a[0]:
                        # Remove from list
                        new_emissions[t].remove(a)

        self.emission_pairs = {}

        for t in range(self.T):
            for pair in new_emissions[t]:
                if isinstance(pair[0], tuple):
                    self.emission_pairs[pair[0]] = pair[1]
                else:
                    self.emission_pairs[(pair[0],)] = pair[1]

        #  Find all inputs and outputs for transition functions
        transfer_pairs = {}
        for node in self.node_parents:
            _, time = node.split("_")
            if self.node_parents[node] and time > "0":
                tmp = [parent for parent in self.node_parents[node] if parent.split("_")[1] != time]
                assert len(tmp) != 0, (node, self.node_parents[node], tmp)
                transfer_pairs[node] = tmp

        # Flip keys and values to get explicit input-output order
        self.transfer_pairs = dict((tuple(v), k) for k, v in transfer_pairs.items())

        # Sometimes the input and output pair order does not match because of NetworkX internal issues,
        # so we need adjust the keys so that they do match.
        self.emission_pairs = update_emission_pairs_keys(self.T, self.node_parents, self.emission_pairs)
        # Estimates of structural equation models [spatial models]
        self.estimate_sem = estimate_sem
        self.sem_trans_fncs = fit_sem_transition_functions_complex(self.observational_samples, self.transfer_pairs)
        self.sem_emit_fncs = fit_sem_complex(self.observational_samples, self.emission_pairs)

        self.time_indexed_trans_fncs_inputs = {t: [] for t in range(1, self.T)}
        for t in range(1, self.T):
            for key in self.transfer_pairs.keys():
                tt = int(self.transfer_pairs[key].split("_")[1])
                if t == tt:
                    self.time_indexed_trans_fncs_inputs[t].append(key)

    def _update_sem_emit_fncs(self, temporal_index: int, temporal_index_data=None) -> None:

        for inputs in list(self.sem_emit_fncs[temporal_index].keys()):
            output = self.emission_pairs[inputs].split("_")[0]

            # Multivariate conditional
            if len(inputs) > 1:
                xx = []
                for node in inputs:
                    start_node, time = node.split("_")
                    if temporal_index_data is not None:
                        time = temporal_index_data
                    else:
                        time = int(time)
                        assert time == temporal_index_data, (time, temporal_index_data)

                    #  Input
                    x = make_column_shape_2D(self.observational_samples[start_node][time])
                    xx.append(x)
                xx = np.hstack(xx)
                #  Output
                yy = make_column_shape_2D(self.observational_samples[output][time])

            # Univariate conditional
            elif len(inputs) == 1:
                start_node, time = inputs[0].split("_")
                if temporal_index_data is not None:
                    #  Use past conditional
                    time = temporal_index_data
                else:
                    time = int(time)
                    assert time == temporal_index_data, (time, temporal_index_data)

                #  Input
                xx = make_column_shape_2D(self.observational_samples[start_node][time])
                #  Output
                yy = make_column_shape_2D(self.observational_samples[output][time])
            else:
                raise ValueError("The length of the tuple is: {}".format(len(inputs)))

            assert len(xx.shape) == 2
            assert len(yy.shape) == 2
            assert xx.shape[0] == yy.shape[0]  # Column arrays

            if xx.shape[0] != yy.shape[0]:
                min_rows = np.min((xx.shape[0], yy.shape[0]))
                xx = xx[: int(min_rows)]
                yy = yy[: int(min_rows)]

            if temporal_index_data == temporal_index:
                # Online / we have data
                if any(self.hyperparam_obs_emit.values()):
                    self.sem_emit_fncs[temporal_index][inputs] = fit_gp(
                        x=xx,
                        y=yy,
                        lengthscale=self.hyperparam_obs_emit[inputs][1],
                        variance=self.hyperparam_obs_emit[inputs][0],
                    )
                else:
                    # Update in-place
                    self.sem_emit_fncs[temporal_index][inputs].set_XY(X=xx, Y=yy)
                    self.sem_emit_fncs[temporal_index][inputs].optimize()
            else:
                # No data for this time-step
                assert temporal_index_data != temporal_index and temporal_index_data < temporal_index, (
                    temporal_index_data,
                    temporal_index,
                )
                temporal_index_inputs = tuple(v.split("_")[0] + "_" + str(temporal_index_data) for v in inputs)
                assert temporal_index_inputs in self.sem_emit_fncs[temporal_index_data].keys(), (
                    temporal_index_inputs,
                    self.sem_emit_fncs[temporal_index_data].keys(),
                )
                self.sem_emit_fncs[temporal_index][inputs] = self.sem_emit_fncs[temporal_index_data][
                    temporal_index_inputs
                ]

    def _update_sem_transmission_functions(self, temporal_index: int, temporal_index_data: int = None) -> None:

        assert temporal_index > 0

        # Fit transition functions first
        # for inputs in self.transfer_pairs.keys():
        for inputs in self.time_indexed_trans_fncs_inputs[temporal_index]:
            # Transfer input
            if len(inputs) > 1:
                # many to one mapping
                iin_vars = [var.split("_")[0] for var in inputs]
                iin_times = [int(var.split("_")[1]) for var in inputs]
                # Auto-regressive structure
                xx = []
                for in_var, in_time in zip(iin_vars, iin_times):
                    xx.append(make_column_shape_2D(self.observational_samples[in_var][in_time]))
                xx = np.hstack(xx)
            else:
                in_var = inputs[0].split("_")[0]
                in_time = int(inputs[0].split("_")[1])
                xx = make_column_shape_2D(self.observational_samples[in_var][in_time])

            # Transfer target
            output = self.transfer_pairs[inputs]
            # one to one mapping
            out_var = output.split("_")[0]
            out_time = int(output.split("_")[1])
            # Auto-regressive structure
            yy = make_column_shape_2D(self.observational_samples[out_var][out_time])

            min_size = np.min((xx.shape[0], yy.shape[0]))  # Check rows
            xx = xx[: int(min_size), :]
            yy = yy[: int(min_size), :]

            if temporal_index_data == temporal_index:
                if any(self.hyperparam_obs_transf.values()):
                    self.sem_trans_fncs[inputs] = fit_gp(
                        x=xx,
                        y=yy,
                        lengthscale=self.hyperparam_obs_transf[inputs][1],
                        variance=self.hyperparam_obs_transf[inputs][0],
                    )
                else:
                    self.sem_trans_fncs[inputs].set_XY(X=xx, Y=yy)
                    self.sem_trans_fncs[inputs].optimize()
            else:
                # No data for this time-step
                assert temporal_index_data != temporal_index and temporal_index_data < temporal_index, (
                    temporal_index_data,
                    temporal_index,
                )
                # Find the input for the past index
                temporal_index_data_inputs = tuple(v.split("_")[0] + "_" + str(temporal_index_data - 1) for v in inputs)
                assert temporal_index_data_inputs in self.time_indexed_trans_fncs_inputs[temporal_index_data], (
                    temporal_index,
                    temporal_index_data,
                    inputs,
                    temporal_index_data_inputs,
                    self.time_indexed_trans_fncs_inputs[temporal_index_data],
                )
                # We use the previous transition function for this time-index
                self.sem_trans_fncs[inputs] = self.sem_trans_fncs[temporal_index_data_inputs]

    def _get_observational_hp_emissions(self, emission_functions, temporal_index):
        # XXX: this is _ONLY_ a valid option for stationary time-series.
        assert temporal_index > 0
        for inputs in emission_functions[temporal_index].keys():
            for past_inputs, inputs in zip(
                self.sem_emit_fncs[temporal_index - 1].keys(), self.sem_emit_fncs[temporal_index].keys(),
            ):
                if len(past_inputs) == len(inputs):
                    assert all([v1.split("_")[0] == v2.split("_")[0] for v1, v2 in zip(past_inputs, inputs)])
                    self.hyperparam_obs_emit[inputs] = [
                        emission_functions[temporal_index - 1][past_inputs].kern.variance[0],
                        emission_functions[temporal_index - 1][past_inputs].kern.lengthscale[0],
                    ]
                else:
                    raise ValueError("This is not a valid option for non-stationary problems.")

    def _get_observational_hp_transition(self, emission_functions):
        for inputs in emission_functions.keys():
            self.hyperparam_obs_transf[inputs] = [
                emission_functions[inputs].kern.variance[0],
                emission_functions[inputs].kern.lengthscale[0],
            ]

    def _forward_propagation(self, temporal_index):

        empty_blanket, _ = make_sequential_intervention_dictionary(self.graph)
        res = []

        assert temporal_index > 0

        for t in range(temporal_index):
            for es in self.exploration_sets:
                interventional_data_es_x = self.interventional_data_x[t][es]
                interventional_data_es_y = self.interventional_data_y[t][es]

                if interventional_data_es_x is not None and interventional_data_es_y is not None:
                    for xx, yy in zip(interventional_data_es_x, interventional_data_es_y):
                        this_blanket = deepcopy(empty_blanket)

                        for i, intervention_variable in enumerate(es):
                            this_blanket[intervention_variable][t] = float(xx[i])

                        this_blanket["Y"][t] = float(yy)

                        out = sequential_sample_from_complex_model_hat(
                            static_sem=self.static_sem,
                            dynamic_sem=self.sem,
                            timesteps=self.total_timesteps,
                            interventions=this_blanket,
                            node_parents=self.node_parents,
                        )

                        res.append(out)

        for dic in res:
            for var in self.observational_samples.keys():
                self.observational_samples[var][temporal_index].extend([dic[var][temporal_index]])
                print(len(self.observational_samples[var][temporal_index]))

    def _update_sem_functions(self, temporal_index, temporal_index_data=None):

        if temporal_index_data is None:
            temporal_index_data = temporal_index

        self._update_sem_emit_fncs(temporal_index, temporal_index_data=temporal_index_data)
        self._update_sem_transmission_functions(temporal_index, temporal_index_data=temporal_index_data)

    def _update_observational_data(self, temporal_index):
        if temporal_index > 0:
            if self.online:
                if isinstance(self.n_obs_t, list):
                    local_n_t = self.n_obs_t[temporal_index]
                else:
                    local_n_t = self.n_obs_t
                assert local_n_t is not None

                # Sample new data
                set_observational_samples = sequentially_sample_model(
                    static_sem=self.true_initial_sem,
                    dynamic_sem=self.true_sem,
                    total_timesteps=temporal_index + 1,
                    sample_count=local_n_t,
                    use_sem_estimate=False,
                    interventions=self.assigned_blanket,
                )

                # Reshape data
                set_observational_samples = convert_to_dict_of_temporal_lists(set_observational_samples)

                for var in self.observational_samples.keys():
                    self.observational_samples[var][temporal_index] = set_observational_samples[var][temporal_index]
            else:
                if isinstance(self.n_obs_t, list):
                    local_n_obs = self.n_obs_t[temporal_index]

                    n_stored_observations = len(
                        self.observational_samples[list(self.observational_samples.keys())[0]][temporal_index]
                    )

                    if self.online is False and local_n_obs != n_stored_observations:
                        # We already have the same number of observations stored
                        set_observational_samples = sequentially_sample_model(
                            static_sem=self.true_initial_sem,
                            dynamic_sem=self.true_sem,
                            total_timesteps=temporal_index + 1,
                            sample_count=local_n_obs,
                            use_sem_estimate=False,
                        )
                        # Reshape data
                        set_observational_samples = convert_to_dict_of_temporal_lists(set_observational_samples)

                        for var in self.observational_samples.keys():
                            self.observational_samples[var][temporal_index] = set_observational_samples[var][
                                temporal_index
                            ]

    def _get_assigned_blanket(self, temporal_index):
        if temporal_index > 0:
            if self.optimal_assigned_blankets is not None:
                assigned_blanket = self.optimal_assigned_blankets[temporal_index]
            else:
                assigned_blanket = self.assigned_blanket_hat
        else:
            assigned_blanket = self.assigned_blanket_hat
        return assigned_blanket

    def _safe_optimization(self, temporal_index, exploration_set, bound_var=1e-02, bound_len=20.0):
        if self.bo_model[temporal_index][exploration_set].model.kern.variance[0] < bound_var:
            self.bo_model[temporal_index][exploration_set].model.kern.variance[0] = 1.0

        if self.bo_model[temporal_index][exploration_set].model.kern.lengthscale[0] > bound_len:
            self.bo_model[temporal_index][exploration_set].model.kern.lengthscale[0] = 1.0

    def _get_interventional_hp(self, temporal_index, exploration_set, prior_var, prior_lengthscale):
        if temporal_index > 0 and self.transfer_hp_i:
            if self.bo_model[temporal_index][exploration_set] is None:
                if self.bo_model[temporal_index - 1][exploration_set] is not None:
                    variance = self.bo_model[temporal_index - 1][exploration_set].model.kern.variance[0]
                    lengthscale = self.bo_model[temporal_index - 1][exploration_set].model.kern.lengthscale[0]
                else:
                    variance = prior_var
                    lengthscale = prior_lengthscale

            else:
                variance = self.bo_model[temporal_index][exploration_set].model.kern.variance[0]
                lengthscale = self.bo_model[temporal_index][exploration_set].model.kern.lengthscale[0]
        else:
            if self.bo_model[temporal_index][exploration_set] is not None:
                variance = self.bo_model[temporal_index][exploration_set].model.kern.variance[0]
                lengthscale = self.bo_model[temporal_index][exploration_set].model.kern.lengthscale[0]
            else:
                variance = prior_var
                lengthscale = prior_lengthscale

        return variance, lengthscale
