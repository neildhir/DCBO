from collections import OrderedDict
import numpy as np


class PISHCAT_SEM:
    @staticmethod
    def static():

        P = lambda noise, t, sample: noise
        I = lambda noise, t, sample: noise
        S = lambda noise, t, sample: sample["I"][t] + noise
        H = lambda noise, t, sample: sample["P"][t] + noise
        C = lambda noise, t, sample: sample["H"][t] + noise
        A = lambda noise, t, sample: sample["I"][t] + sample["P"][t] + noise
        T = lambda noise, t, sample: sample["C"][t] + sample["A"][t] + noise
        return OrderedDict([("P", P), ("I", I), ("S", S), ("H", H), ("C", C), ("A", A), ("T", T)])

    @staticmethod
    def dynamic():

        P = lambda noise, t, sample: sample["P"][t - 1] + noise
        I = lambda noise, t, sample: sample["I"][t - 1] + noise
        S = lambda noise, t, sample: sample["S"][t - 1] + sample["I"][t] + noise
        H = lambda noise, t, sample: sample["S"][t - 1] + sample["P"][t] + noise
        C = lambda noise, t, sample: sample["H"][t] + noise
        A = lambda noise, t, sample: sample["I"][t] + sample["P"][t] + noise
        T = lambda noise, t, sample: sample["C"][t] + sample["A"][t] + noise
        return OrderedDict([("P", P), ("I", I), ("S", S), ("H", H), ("C", C), ("A", A), ("T", T)])


class StationaryDependentSEM:
    @staticmethod
    def static():

        X = lambda noise, t, sample: noise
        Z = lambda noise, t, sample: np.exp(-sample["X"][t]) + noise
        Y = lambda noise, t, sample: np.cos(sample["Z"][t]) - np.exp(-sample["Z"][t] / 20.0) + noise
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])

    @staticmethod
    def dynamic():

        # We get temporal innovation by introducing transfer functions between temporal indices
        X = lambda noise, t, sample: sample["X"][t - 1] + noise
        Z = lambda noise, t, sample: np.exp(-sample["X"][t]) + sample["Z"][t - 1] + noise
        Y = (
            lambda noise, t, sample: np.cos(sample["Z"][t])
            - np.exp(-sample["Z"][t] / 20.0)
            + sample["Y"][t - 1]
            + noise
        )
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])


class LinearMultipleChildrenSEM:
    """
    Test DAG for nodes within a slice that have more than one child _within_ the slice.

    Returns
    -------
        None
    """

    @staticmethod
    def static() -> OrderedDict:

        X = lambda noise, t, sample: 1 + noise
        Z = lambda noise, t, sample: 2 * sample["X"][t] + noise
        Y = lambda noise, t, sample: 2 * sample["Z"][t] - sample["X"][t] + noise
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])

    @staticmethod
    def dynamic() -> OrderedDict:

        # We get temporal innovation by introducing transfer functions between temporal indices
        X = lambda noise, t, sample: sample["X"][t - 1] + 1 + noise
        Z = lambda noise, t, sample: 2 * sample["X"][t] + sample["Z"][t - 1] + noise
        Y = lambda noise, t, sample: 2 * sample["Z"][t] + sample["Y"][t - 1] - sample["X"][t] + noise
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])


class StationaryDependentMultipleChildrenSEM:
    """
    Test DAG for nodes within a slice that have more than one child _within_ the slice.

    Returns
    -------
        None
    """

    @staticmethod
    def static() -> OrderedDict:

        X = lambda noise, t, sample: noise
        Z = lambda noise, t, sample: np.exp(-sample["X"][t]) + noise
        Y = (
            lambda noise, t, sample: np.cos(sample["Z"][t])
            - np.exp(-sample["Z"][t] / 20.0)
            - np.sin(sample["X"][t])
            + noise
        )
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])

    @staticmethod
    def dynamic() -> OrderedDict:

        # We get temporal innovation by introducing transfer functions between temporal indices
        X = lambda noise, t, sample: sample["X"][t - 1] + noise
        Z = lambda noise, t, sample: np.exp(-sample["X"][t]) + sample["Z"][t - 1] + noise
        Y = (
            lambda noise, t, sample: np.cos(sample["Z"][t])
            - np.exp(-sample["Z"][t] / 20.0)
            + sample["Y"][t - 1]
            - np.sin(sample["X"][t])
            + noise
        )
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])


class StationaryIndependentSEM:
    @staticmethod
    def static():
        X = lambda noise, t, sample: noise
        Z = lambda noise, t, sample: noise
        Y = (
            lambda noise, t, sample: -2 * np.exp(-((sample["X"][t] - 1) ** 2) - (sample["Z"][t] - 1) ** 2)
            - np.exp(-((sample["X"][t] + 1) ** 2) - sample["Z"][t] ** 2)
            + noise
        )
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])

    @staticmethod
    def dynamic():
        X = lambda noise, t, sample: -sample["X"][t - 1] + noise
        Z = lambda noise, t, sample: -sample["Z"][t - 1] + noise
        Y = (
            lambda noise, t, sample: -2 * np.exp(-((sample["X"][t] - 1) ** 2) - (sample["Z"][t] - 1) ** 2)
            - np.exp(-((sample["X"][t] + 1) ** 2) - sample["Z"][t] ** 2)
            + sample["Y"][t - 1]
            + noise
        )
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])


class NonStationaryDependentSEM:
    """
    This SEM currently supports one change point.

    This SEM changes topology over t.

    with: intervention_domain = {'X':[-4,1],'Z':[-3,3]}
    """

    def __init__(self, change_point):
        """
        Initialise change point(s).

        Parameters
        ----------
        cp : int
            The temporal index of the change point (cp).
        """
        self.cp = change_point

    @staticmethod
    def static():
        """
        noise: e
        sample: s
        time index: t
        """
        X = lambda e, t, s: e
        Z = lambda e, t, s: s["X"][t] + e
        Y = lambda e, t, s: np.sqrt(abs(36 - (s["Z"][t] - 1) ** 2)) + 1 + e
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])

    def dynamic(self):
        X = lambda e, t, s: s["X"][t - 1] + e
        Z = (
            lambda e, t, s: -s["X"][t] / s["X"][t - 1] + s["Z"][t - 1] + e
            if t == self.cp
            else s["X"][t] + s["Z"][t - 1] + e
        )
        Y = (
            lambda e, t, s: s["Z"][t] * np.cos(np.pi * s["Z"][t]) - s["Y"][t - 1] + e
            if t == self.cp
            else abs(s["Z"][t]) - s["Y"][t - 1] - s["Z"][t - 1] + e
        )
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])


class NonStationaryIndependentSEM:
    """
    This SEM currently supports one change point.

    This SEM changes topology over t.
    """

    def __init__(self, change_point):
        self.change_point = change_point

    @staticmethod
    def static():
        X = lambda noise, t, sample: noise
        Z = lambda noise, t, sample: noise
        Y = (
            lambda noise, t, sample: -(
                2 * np.exp(-((sample["X"][t] - 1) ** 2) - (sample["Z"][t] - 1) ** 2)
                + np.exp(-((sample["X"][t] + 1) ** 2) - sample["Z"][t] ** 2)
            )
            + noise
        )
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])

    def dynamic(self):
        #  X_t | X_{t-1}
        X = lambda noise, t, sample: sample["X"][t - 1] + noise
        Z = (
            lambda noise, t, sample: np.cos(sample["Z"][t - 1]) + noise
            if t == self.change_point
            else np.sin(sample["Z"][t - 1] ** 2) * sample["X"][t - 1] + noise
        )
        #  if t <= 1: Y_t | Z_t, Y_{t-1} else: Y_t | Z_t, X_t, Y_{t-1}
        Y = (
            lambda noise, t, sample:
            # np.exp(-np.cos(sample["X"][t]) ** 2)
            -np.exp(-(sample["Z"][t]) / 3.0)
            + np.exp(-sample["X"][t] / 3.0)
            + sample["Y"][t - 1]
            + sample["X"][t - 1]
            + noise
            if t == self.change_point
            else -2 * np.exp(-((sample["X"][t]) ** 2) - (sample["Z"][t] - sample["Z"][t - 1]) ** 2)
            - np.exp(-((sample["X"][t] - sample["Z"][t]) ** 2))
            + np.cos(sample["Z"][t])
            - sample["Y"][t - 1]
            + noise
        )
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])
