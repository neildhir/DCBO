from collections import OrderedDict
from typing import Callable, Iterable

from numpy.random import randn
from src.utilities import select_sample


def make_sem_hat(emission_fncs: dict, transition_fncs: dict) -> classmethod:
    """
    This function creates a full SEM using functions estimated from the observational samples.
    Parameters
    ----------
    emission_fncs : dict
        These are the emission functions which operate within a given time-slice.
    transition_fncs : dict
        These are the transition functions which moves samples forward from one time-step to the next.
    Returns
    -------
    class
        A full SEM estimate.
    """

    class semhat:
        def __init__(self):
            pass

        def static(self, moment: int):
            """
            Find the static updates (no conditioning on the past).
            Parameters
            ----------
            moment : int
                Ordinal moment, either 0 (mean) or 1 (variance) if estimator is a Gaussian process
            Returns
            -------
            np.ndarray
                A sample from the first time-index
            """
            assert moment in [0, 1], moment

            X = lambda noise: noise
            Z = lambda t, emit_input_var, sample: emission_fncs[t][emit_input_var].predict(
                select_sample(sample, emit_input_var, t)
            )[moment]
            Y = lambda t, emit_input_var, sample: emission_fncs[t][emit_input_var].predict(
                select_sample(sample, emit_input_var, t)
            )[moment]
            return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])

        def dynamic(self, moment: int):
            """
            Find the static updates (no conditioning on the past).
            Parameters
            ----------
            moment : int
                Ordinal moment, either 0 (mean) or 1 (variance) if estimator is a Gaussian process
            Returns
            -------
            np.ndarray
                A sample from the time-index t
            """
            assert moment in [0, 1], moment

            X = lambda t, transfer_input_vars, emit_input_vars, sample: transition_fncs[transfer_input_vars].predict(
                select_sample(sample, transfer_input_vars, t - 1)
            )[moment]

            Z = (
                lambda t, transfer_input_vars, emit_input_vars, sample: transition_fncs[transfer_input_vars].predict(
                    select_sample(sample, transfer_input_vars, t - 1)
                )[moment]
                + emission_fncs[t][emit_input_vars].predict(select_sample(sample, emit_input_vars, t))[moment]
            )

            Y = (
                lambda t, transfer_input_vars, emit_input_vars, sample: transition_fncs[transfer_input_vars].predict(
                    select_sample(sample, transfer_input_vars, t - 1)
                )[moment]
                + emission_fncs[t][emit_input_vars].predict(select_sample(sample, emit_input_vars, t))[moment]
            )

            return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])

    return semhat


def make_sem_independent_hat(emission_fncs: dict, transition_fncs: dict) -> classmethod:
    """
    This function creates a full SEM using functions estimated from the observational samples.
    Parameters
    ----------
    emission_fncs : dict
        These are the emission functions which operate within a given time-slice.
    transition_fncs : dict
        These are the transition functions which moves samples forward from one time-step to the next.
    Returns
    -------
    class
        A full SEM estimate.
    """

    class semhat:
        @staticmethod
        def static(moment: int):
            """
            Find the static updates (no conditioning on the past).
            Parameters
            ----------
            moment : int
                Moment, either 0 (mean) or 1 (variance) if estimator is a Gaussian process
            Returns
            -------
            np.ndarray
                A sample from the first time-index
            """
            assert moment in [0, 1], moment

            X = lambda noise: noise
            Z = lambda noise: noise
            Y = lambda t, emit_input_var, sample: emission_fncs[t][emit_input_var].predict(
                select_sample(sample, emit_input_var, t)
            )[moment]
            return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])

        @staticmethod
        def dynamic(moment: int):
            """
            Find the dynamic updates.

            Parameters
            ----------
            moment : int
                Moment, either 0 (mean) or 1 (variance) if estimator is a Gaussian process
            Returns
            -------
            np.ndarray
                A sample from the time-index t
            """
            assert moment in [0, 1], moment

            # The nature of this CGM means that there are no emission input variables
            X = lambda t, transfer_input_vars, emit_input_vars, sample: transition_fncs[transfer_input_vars].predict(
                select_sample(sample, transfer_input_vars, t - 1)
            )[moment]

            # The nature of this CGM means that there are no emission input variables
            Z = lambda t, transfer_input_vars, emit_input_vars, sample: transition_fncs[transfer_input_vars].predict(
                select_sample(sample, transfer_input_vars, t - 1)
            )[moment]

            Y = (
                lambda t, transfer_input_vars, emit_input_vars, sample: transition_fncs[transfer_input_vars].predict(
                    select_sample(sample, transfer_input_vars, t - 1)
                )[moment]
                + emission_fncs[t][emit_input_vars].predict(select_sample(sample, emit_input_vars, t))[moment]
            )

            return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])

    return semhat


def make_sem_complex_hat(emission_fncs: dict, transition_fncs: dict) -> classmethod:
    """
    This function creates a full SEM using functions estimated from the observational samples.
    Parameters
    ----------
    emission_fncs : dict
        These are the emission functions which operate within a given time-slice.
    transition_fncs : dict
        These are the transition functions which moves samples forward from one time-step to the next.
    Returns
    -------
    class
        A full SEM estimate.
    """

    class semhat:
        def __init__(self):
            pass

        def static(self, moment: int):
            """
            Find the static updates (no conditioning on the past).
            Parameters
            ----------
            moment : int
                Moment, either 0 (mean) or 1 (variance) if estimator is a Gaussian process
            Returns
            -------
            np.ndarray
                A sample from the first time-index
            """
            assert moment in [0, 1], moment

            X = lambda noise: noise
            W = lambda noise: noise

            Z = lambda t, emit_input_var, sample: emission_fncs[t][emit_input_var].predict(
                select_sample(sample, emit_input_var, t)
            )[moment]
            Y = lambda t, emit_input_var, sample: emission_fncs[t][emit_input_var].predict(
                select_sample(sample, emit_input_var, t)
            )[moment]
            return OrderedDict([("X", X), ("W", W), ("Z", Z), ("Y", Y)])

        def dynamic(self, moment: int):
            """
            Find the dynamic updates.

            Parameters
            ----------
            moment : int
                Moment, either 0 (mean) or 1 (variance) if estimator is a Gaussian process
            Returns
            -------
            np.ndarray
                A sample from the time-index t
            """
            assert moment in [0, 1], moment

            W = lambda noise: noise

            # The nature of this CGM means that there are no emission input variables
            X = lambda t, transfer_input_vars, emit_input_vars, sample: transition_fncs[transfer_input_vars].predict(
                select_sample(sample, transfer_input_vars, t - 1)
            )[moment]

            # The nature of this CGM means that there are no emission input variables
            Z = lambda t, transfer_input_vars, emit_input_vars, sample: transition_fncs[transfer_input_vars].predict(
                select_sample(sample, transfer_input_vars, t - 1)
            )[moment]

            Y = (
                lambda t, transfer_input_vars, emit_input_vars, sample: transition_fncs[transfer_input_vars].predict(
                    select_sample(sample, transfer_input_vars, t - 1)
                )[moment]
                + emission_fncs[t][emit_input_vars].predict(select_sample(sample, emit_input_vars, t))[moment]
            )

            return OrderedDict([("X", X), ("W", W), ("Z", Z), ("Y", Y)])

    return semhat


def auto_sem_dependent_stationary_hat(
    variables: Iterable, root_instrument: bool, emission_functions: dict, transition_functions: dict
) -> classmethod:

    # TODO: this function will eventually be passed the full adjacency matrix for the graph.

    class semhat:
        @staticmethod
        def _make_some_white_noise_function() -> Callable:
            return lambda: randn()

        @staticmethod
        def _make_static_function(moment: int) -> Callable:
            return lambda t, emit_input_var, sample: emission_functions[t][emit_input_var].predict(
                select_sample(sample, emit_input_var, t)
            )[moment]

        @staticmethod
        def _make_dynamic_function(moment: int) -> Callable:
            return (
                lambda t, transfer_input_vars, emit_input_vars, sample: transition_functions[
                    transfer_input_vars
                ].predict(select_sample(sample, transfer_input_vars, t - 1))[moment]
                + emission_functions[t][emit_input_vars].predict(select_sample(sample, emit_input_vars, t))[moment]
            )

        @staticmethod
        def _make_dynamic_transfer_only_function(moment: int) -> Callable:
            return lambda t, transfer_input_vars, sample: transition_functions[transfer_input_vars].predict(
                select_sample(sample, transfer_input_vars, t - 1)
            )[moment]

        def static(self, moment: int):
            assert moment in [0, 1], moment
            functions = OrderedDict()
            for i, var in enumerate(variables):
                if i == 0:
                    if root_instrument:
                        """
                        Noise as root node in time-slice, without any time dependence
                        """
                        functions[var] = self._make_some_white_noise_function()
                    else:
                        """
                        Root node as instrument variable.
                        """
                        functions[var] = self._make_static_function(moment)
                else:
                    functions[var] = self._make_static_function(moment)
            return functions

        def dynamic(self, moment: int):
            assert moment in [0, 1], moment
            functions = OrderedDict()
            for i, var in enumerate(variables):
                if i == 0:
                    if root_instrument:
                        """
                        Root node in the time-slice, without any time dependence
                        o   o Node at time t
                            |
                            |
                            V
                            o Child node at time t
                        """
                        functions[var] = self._make_some_white_noise_function()
                    else:
                        """
                        Root node in the time-slice, with time dependence
                        o-->o Node at time t with dependence from time t-1
                            |
                            |
                            V
                            o Child node at time t
                        """
                        functions[var] = self._make_dynamic_transfer_only_function(moment)
                else:
                    functions[var] = self._make_dynamic_function(moment)
            return functions

    return semhat


def make_sem_predator_prey_hat(emission_fncs: dict, transition_fncs: dict) -> classmethod:
    class semhat:
        @staticmethod
        def static(moment: int):
            assert moment in [0, 1], moment
            M = lambda noise: noise
            N = lambda t, emit_input_var, sample: emission_fncs[t][emit_input_var].predict(
                select_sample(sample, emit_input_var, t)
            )[moment]
            P = lambda t, emit_input_var, sample: emission_fncs[t][emit_input_var].predict(
                select_sample(sample, emit_input_var, t)
            )[moment]
            J = lambda t, emit_input_var, sample: emission_fncs[t][emit_input_var].predict(
                select_sample(sample, emit_input_var, t)
            )[moment]
            A = lambda t, emit_input_var, sample: emission_fncs[t][emit_input_var].predict(
                select_sample(sample, emit_input_var, t)
            )[moment]
            E = lambda t, emit_input_var, sample: emission_fncs[t][emit_input_var].predict(
                select_sample(sample, emit_input_var, t)
            )[moment]
            D = lambda t, emit_input_var, sample: emission_fncs[t][emit_input_var].predict(
                select_sample(sample, emit_input_var, t)
            )[moment]
            return OrderedDict([("M", M), ("N", N), ("P", P), ("J", J), ("A", A), ("E", E), ("D", D)])

        @staticmethod
        def dynamic(moment: int):
            assert moment in [0, 1], moment

            M = lambda noise: noise
            N = (
                lambda t, transfer_input_vars, emit_input_vars, sample: transition_fncs[transfer_input_vars].predict(
                    select_sample(sample, transfer_input_vars, t - 1)
                )[moment]
                + emission_fncs[t][emit_input_vars].predict(select_sample(sample, emit_input_vars, t))[moment]
            )
            P = (
                lambda t, transfer_input_vars, emit_input_vars, sample: transition_fncs[transfer_input_vars].predict(
                    select_sample(sample, transfer_input_vars, t - 1)
                )[moment]
                + emission_fncs[t][emit_input_vars].predict(select_sample(sample, emit_input_vars, t))[moment]
            )
            J = (
                lambda t, transfer_input_vars, emit_input_vars, sample: transition_fncs[transfer_input_vars].predict(
                    select_sample(sample, transfer_input_vars, t - 1)
                )[moment]
                + emission_fncs[t][emit_input_vars].predict(select_sample(sample, emit_input_vars, t))[moment]
            )
            A = (
                lambda t, transfer_input_vars, emit_input_vars, sample: transition_fncs[transfer_input_vars].predict(
                    select_sample(sample, transfer_input_vars, t - 1)
                )[moment]
                + emission_fncs[t][emit_input_vars].predict(select_sample(sample, emit_input_vars, t))[moment]
            )
            E = (
                lambda t, transfer_input_vars, emit_input_vars, sample: transition_fncs[transfer_input_vars].predict(
                    select_sample(sample, transfer_input_vars, t - 1)
                )[moment]
                + emission_fncs[t][emit_input_vars].predict(select_sample(sample, emit_input_vars, t))[moment]
            )
            D = (
                lambda t, transfer_input_vars, emit_input_vars, sample: transition_fncs[transfer_input_vars].predict(
                    select_sample(sample, transfer_input_vars, t - 1)
                )[moment]
                + emission_fncs[t][emit_input_vars].predict(select_sample(sample, emit_input_vars, t))[moment]
            )

            return OrderedDict([("M", M), ("N", N), ("P", P), ("J", J), ("A", A), ("E", E), ("D", D)])

    return semhat
