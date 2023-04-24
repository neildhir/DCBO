from collections import OrderedDict
import numpy as np


class PredatorPreySEM:
    """
    ODE from "Long-term cyclic persistence in an experimental predator–prey system" (2019).

    See methods section for ODE.
    """

    def __init__(self):
        """
        Parameter values of ODE.
        """
        self.beta = 5  #  Mass ratio adult/juveniles
        self.delta = 0.55  #  dilution rate
        self.theta = 1  # egg development time (set to one to fit the sampling frequenzy; one day)
        self.kappa = 1.25  # Hill coefficient of the functional response
        self.tau = 0  # juvenile maturation time (set to zero for simulation)
        self.epsilon = 0.25  # predator assimilation efficiency
        self.r_P = 3.3  # phytoplankton maximal growth rate
        self.K_P = 4.3  # phytoplankton half-saturation constant
        self.r_B = 2.25  # rotifer maximal egg-recruitment rate
        self.K_B = 15  # rotifer half-saturation constant
        self.m = 0.15  # rotifer mortality rate
        self.M = 80  # Nitrogen concentration in the external medium
        self.v_algal = 28e-9  # nitrogen content per algal cell
        self.v_Brachionus = 0.57 * 1e-3  # nitrogen content per adult female Brachionus

    def _F_P(self, N, noise):
        """
        Algal nutrient uptake is modelled as a Monod function.

        Parameters
        ----------
        N : float
            concentration of nitrogen
        noise : float
           autocorrelated noise
        """
        return self.r_P * (1 + noise) * (N / (self.K_P + N))

    def _F_B(self, P, noise):
        """
        Predator recruitment is modelled as a type-3 function response with a Hill coefficient.

        Parameters
        ----------
        P : float
            concentration of phytoplankton
        noise : float
           autocorrelated noise
        """
        return self.r_B * (1 + noise) * (P ** self.kappa / (self.K_B ** self.kappa + P ** self.kappa))

    def _R_E(self, t, s, e):
        """
        Egg recruitment rate.

        Parameters
        ----------
        t : int
            time-index
        s : dict
            sample
        e : float
            noise

        """
        assert isinstance(t, int)
        return self._F_B(s["P"][t], e) * s["A"][t]

    def _R_J(self, t, s, e):
        """
        Juvenile recruitment rate.

        Parameters
        ----------
        t : int
            time-index
        s : dict
            sample
        e : float
            noise
        """
        return self._R_E(t - self.theta, s, e) * np.exp(-self.delta * self.theta)

    def _R_A(self, t, s, e):
        """
        Adult recruitment rate.

        Parameters
        ----------
        t : int
            time-index
        s : dict
            sample
        e : float
            noise
        """
        return self._R_J(t - self.tau, s, e) * np.exp(-self.delta * self.tau)

    def _B(self, J, A):
        """
        Total predator density.

        Parameters
        ----------
        J : float
            concentration of juveniles
        A : float
            concentration of adults
        """
        return self.beta * J + A

    def static(self):
        """
        noise: e
        sample: s
        time index: t

        Constants particular to experiment 2 (ref. 'C2' in paper) [row-index 0 of dataframe].
        """

        # We use expected values for N, P, J, and A at t=0 -- calculated from {C1,C2,C3,C4}.csv
        N_init = 41.3
        P_init = 2.71605
        J_init = 1.2303571428571427
        A_init = 6.151785714285714
        E_init = 0.17099999999999999
        D_init = 0.0

        # Nitrogen concentration in the external medium
        M = lambda e, t, s: self.M

        # Prey (algae)
        N = lambda e, t, s: self.delta * s["M"][t] - self._F_P(N_init, e) * P_init  # - self.delta * N_init

        # Predator (phytoplankton)
        P = (
            lambda e, t, s: self._F_P(s["N"][t], e)
            * P_init
            # - self._F_B(P_init, e) * self._B(J_init, A_init) / self.epsilon
            # - self.delta * P_init
        )

        # Predator Juveniles
        J = (
            lambda e, t, s: (self._F_B(s["P"][t], e) * A_init) * np.exp(-self.delta * self.theta)
            # - (self._F_B(s["P"][t], e) * A_init) * np.exp(-self.delta * self.tau)
            - (self.m + self.delta) * J_init
        )

        # Predator Adults
        A = (
            lambda e, t, s: self.beta * self._F_B(s["P"][t], e) * A_init * np.exp(-self.delta * self.theta)
            - (self.m + self.delta) * A_init
        )

        # Predator Eggs
        E = (
            lambda e, t, s: self._F_B(s["P"][t], e) * s["A"][t] * (1 - np.exp(-self.delta * self.theta))
            - self.delta * E_init
        )

        # Dead animals
        D = lambda e, t, s: self.m * (s["J"][t] + s["A"][t]) - self.delta * D_init

        return OrderedDict([("M", M), ("N", N), ("P", P), ("J", J), ("A", A), ("E", E), ("D", D)])

    def dynamic(self):
        """
        noise: e
        sample: s
        time index: t
        """

        # Nitrogen concentration in the external medium
        M = lambda e, t, s: self.M  #  Instrument variable - no dependence on past

        # Prey (algae)
        N = (
            lambda e, t, s: self.delta * s["M"][t]
            - self._F_P(s["N"][t - 1], e) * s["P"][t - 1]
            - self.delta * s["N"][t - 1]
        )

        # Predator (phytoplankton)
        P = (
            lambda e, t, s: self._F_P(s["N"][t], e)
            * s["P"][t - 1]
            # - self._F_B(s["P"][t - 1], e) * self._B(s["J"][t - 1], s["A"][t - 1]) / self.epsilon
            # - self.delta * s["P"][t - 1]
        )

        # Predator Juveniles
        J = lambda e, t, s: self._R_J(t, s, e) - (self.m + self.delta) * s["J"][t - 1]  # -  self._R_A(t, s, e)

        # Predator Adults
        A = lambda e, t, s: self.beta * self._R_A(t, s, e) - (self.m + self.delta) * s["A"][t - 1]

        # Predator Eggs
        E = lambda e, t, s: self._R_E(t, s, e) - self.delta * s["E"][t - 1]  # - self._R_J(t, s, e)

        # Dead animals
        D = lambda e, t, s: self.m * (s["J"][t] + s["A"][t]) - self.delta * s["D"][t - 1]

        return OrderedDict([("M", M), ("N", N), ("P", P), ("J", J), ("A", A), ("E", E), ("D", D)])
