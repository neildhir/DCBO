from typing import Tuple, Union
import numpy as np
import scipy.stats
from emukit.core.acquisition import Acquisition
from emukit.core.interfaces import IDifferentiable, IModel


class ManualCausalExpectedImprovement(Acquisition):
    def __init__(
        self, current_global_min, task, mean_function, variance_function, previous_variance, jitter: float = float(0),
    ) -> None:
        """
        The improvement when a BO model has not yet been instantiated.

        Efficient Global Optimization of Expensive Black-Box Functions
        Jones, Donald R. and Schonlau, Matthias and Welch, William J.
        Journal of Global Optimization

        :param mean_function: the mean function for the current DCBO exploration at given temporal index
        :param variance_function: the mean function for the current DCBO exploration at given temporal index
        :param jitter: parameter to encourage extra exploration.
        """
        self.mean_function = mean_function
        self.variance_function = variance_function
        self.jitter = jitter
        self.current_global_min = current_global_min
        self.task = task
        self.previous_variance = previous_variance

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """

        mean = self.mean_function(x)

        # adjustment term          #initial kernel variance
        variance = self.previous_variance * np.ones((x.shape[0], 1)) + self.variance_function(
            x
        )  # See Causal GP def in paper

        standard_deviation = np.sqrt(variance.clip(0))
        mean += self.jitter

        u, pdf, cdf = get_standard_normal_pdf_cdf(self.current_global_min, mean, standard_deviation)
        if self.task == "min":
            improvement = standard_deviation * (u * cdf + pdf)
        else:
            improvement = -(standard_deviation * (u * cdf + pdf))

        return improvement

    @property
    def has_gradients(self) -> bool:
        """
        Returns that this acquisition does not have gradients.
        """
        return False


class CausalExpectedImprovement(Acquisition):
    def __init__(
        self,
        current_global_min,
        task,
        dynamic,
        causal_prior,
        temporal_index,
        model: Union[IModel, IDifferentiable],
        jitter: float = float(0),
    ) -> None:
        """
        This acquisition computes for a given input the improvement over the current best observed value in
        expectation. For more information see:

        Efficient Global Optimization of Expensive Black-Box Functions
        Jones, Donald R. and Schonlau, Matthias and Welch, William J.
        Journal of Global Optimization

        :param model: model that is used to compute the improvement.
        :param jitter: parameter to encourage extra exploration.
        """
        self.model = model
        self.jitter = jitter
        self.current_global_min = current_global_min
        self.task = task
        self.dynamic = dynamic
        self.causal_prior = causal_prior
        self.temporal_index = temporal_index

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """
        # Adding an extra time dimension for ABO
        if self.dynamic and self.causal_prior is False:
            x = np.hstack((x, np.repeat(self.temporal_index, x.shape[0])[:, np.newaxis]))

        mean, variance = self.model.predict(x)

        # Variance is computed with MonteCarlo so we might have some numerical stability
        # This is ensuring that negative values or nan values are not generated
        if np.any(np.isnan(variance)):
            variance[np.isnan(variance)] = 0
        elif np.any(variance < 0):
            variance = variance.clip(0.0001)

        standard_deviation = np.sqrt(variance)

        mean += self.jitter

        u, pdf, cdf = get_standard_normal_pdf_cdf(self.current_global_min, mean, standard_deviation)
        if self.task == "min":
            improvement = standard_deviation * (u * cdf + pdf)
        else:
            improvement = -(standard_deviation * (u * cdf + pdf))

        return improvement

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the Expected Improvement and its derivative.

        :param x: locations where the evaluation with gradients is done.
        """
        # Adding an extra time dimension for ABO
        # Restrict the input space via an additional function
        if self.dynamic and self.causal_prior is False:
            x = np.hstack((x, np.repeat(self.temporal_index, x.shape[0])[:, np.newaxis]))

        mean, variance = self.model.predict(x)

        # Variance is computed with MonteCarlo so we might have some numerical stability
        # This is ensuring that negative values or nan values are not generated
        if np.any(np.isnan(variance)):
            variance[np.isnan(variance)] = 0
        elif np.any(variance < 0):
            variance = variance.clip(0.0001)

        standard_deviation = np.sqrt(variance)

        dmean_dx, dvariance_dx = self.model.get_prediction_gradients(x)
        dstandard_deviation_dx = dvariance_dx / (2 * standard_deviation)

        mean += self.jitter
        u, pdf, cdf = get_standard_normal_pdf_cdf(self.current_global_min, mean, standard_deviation)
        if self.task == "min":
            improvement = standard_deviation * (u * cdf + pdf)
            dimprovement_dx = dstandard_deviation_dx * pdf - cdf * dmean_dx
        else:
            improvement = -(standard_deviation * (u * cdf + pdf))
            dimprovement_dx = -(dstandard_deviation_dx * pdf - cdf * dmean_dx)

        return improvement, dimprovement_dx

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return isinstance(self.model, IDifferentiable)


def get_standard_normal_pdf_cdf(
    x: np.array, mean: np.array, standard_deviation: np.array
) -> Tuple[np.array, np.array, np.array]:
    """
    Returns pdf and cdf of standard normal evaluated at (x - mean)/sigma

    :param x: Non-standardized input
    :param mean: Mean to normalize x with
    :param standard_deviation: Standard deviation to normalize x with
    :return: (normalized version of x, pdf of standard normal, cdf of standard normal)
    """
    u = (x - mean) / standard_deviation
    pdf = scipy.stats.norm.pdf(u)
    cdf = scipy.stats.norm.cdf(u)
    return u, pdf, cdf
