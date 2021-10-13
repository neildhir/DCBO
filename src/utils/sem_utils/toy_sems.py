from collections import OrderedDict
import numpy as np


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


class StationaryDependentSEM_v2:
    """
    Has an extra edge between X and Y.
    """

    @staticmethod
    def static():

        X = lambda noise, t, sample: noise
        Z = lambda noise, t, sample: np.exp(-sample["X"][t]) + noise
        Y = (
            lambda noise, t, sample: np.cos(sample["Z"][t])
            - np.exp(-sample["Z"][t] / 20.0)
            - sample["X"][t]  # Extra edge
            + noise
        )
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
            - sample["X"][t]  # Extra edge
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


class StationaryComplexSEM:
    def __init__(self):
        pass

    def static(self):

        X = lambda noise, t, sample: noise
        W = lambda noise, t, sample: noise
        Z = lambda noise, t, sample: np.sin(sample["W"][t]) + noise

        Y = (
            lambda noise, t, sample: -2 * np.exp(-((sample["X"][t] - 1) ** 2) - (sample["Z"][t] - 1) ** 2)
            - np.exp(-((sample["X"][t] + 1) ** 2) - sample["Z"][t] ** 2)
            + np.cos(sample["Z"][t])
            + noise
        )
        return OrderedDict([("X", X), ("W", W), ("Z", Z), ("Y", Y)])

    def dynamic(self):

        # We get temporal innovation by introducing transfer functions between temporal indices
        W = lambda noise, t, sample: noise
        X = lambda noise, t, sample: -sample["X"][t - 1] + noise
        Z = lambda noise, t, sample: np.sin(sample["W"][t]) - sample["Z"][t - 1] + noise
        Y = (
            lambda noise, t, sample: -2 * np.exp(-((sample["X"][t] - 1) ** 2) - (sample["Z"][t] - 1) ** 2)
            - np.exp(-((sample["X"][t] + 1) ** 2) - sample["Z"][t] ** 2)
            + np.cos(sample["Z"][t] * sample["Y"][t - 1])
            - sample["Y"][t - 1]
            + noise
        )
        return OrderedDict([("X", X), ("W", W), ("Z", Z), ("Y", Y)])


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
