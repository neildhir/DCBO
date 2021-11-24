import numpy as np
from GPy.core import Param
from GPy.kern.src.grid_kerns import GridRBF
from GPy.kern.src.psi_comp import PSICOMP_RBF, PSICOMP_RBF_GPU
from GPy.kern.src.stationary import Stationary
from paramz.transformations import Logexp


class CausalRBF(Stationary):
    """
    Radial Basis Function kernel, aka squared-exponential, exponentiated quadratic or Gaussian kernel:

    .. math::

       k(r) = \sigma^2 \exp \\bigg(- \\frac{1}{2} r^2 \\bigg)

    """

    _support_GPU = True

    def __init__(
        self,
        input_dim,
        variance_adjustment,
        variance=1.0,
        lengthscale=None,
        rescale_variance=1.0,
        ARD=False,
        active_dims=None,
        name="rbf",
        useGPU=False,
        inv_l=False,
    ):
        super(CausalRBF, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name, useGPU=useGPU)
        if self.useGPU:
            self.psicomp = PSICOMP_RBF_GPU()
        else:
            self.psicomp = PSICOMP_RBF()
        self.use_invLengthscale = inv_l
        if inv_l:
            self.unlink_parameter(self.lengthscale)
            self.inv_l = Param("inv_lengthscale", 1.0 / self.lengthscale ** 2, Logexp())
            self.link_parameter(self.inv_l)
        self.variance_adjustment = variance_adjustment
        self.rescale_variance = Param("rescale_variance", rescale_variance, Logexp())

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        input_dict = super(CausalRBF, self)._save_to_input_dict()
        input_dict["class"] = "GPy.kern.RBF"
        input_dict["inv_l"] = self.use_invLengthscale
        if input_dict["inv_l"] == True:
            input_dict["lengthscale"] = np.sqrt(1 / float(self.inv_l))
        return input_dict

    def K(self, X, X2=None):
        """
        Kernel function applied on inputs X and X2.
        In the stationary case there is an inner function depending on the
        distances from X to X2, called r.

        K(X, X2) = K_of_r((X-X2)**2)
        """
        if X2 is None:
            X2 = X
        r = self._scaled_dist(X, X2)
        values = self.variance * np.exp(-0.5 * r ** 2)

        value_diagonal_X = self.variance_adjustment(X)
        value_diagonal_X2 = self.variance_adjustment(X2)

        additional_matrix = np.dot(np.sqrt(value_diagonal_X), np.sqrt(np.transpose(value_diagonal_X2)))

        assert additional_matrix.shape == values.shape, (
            additional_matrix.shape,
            values.shape,
        )
        return values + additional_matrix

    def Kdiag(self, X):
        ret = np.empty(X.shape[0])
        ret[:] = np.repeat(0.1, X.shape[0])

        diagonal_terms = ret

        value = self.variance_adjustment(X)

        if X.shape[0] == 1 and X.shape[1] == 1:
            diagonal_terms = value
        else:
            if np.isscalar(value) == True:
                diagonal_terms = value
            else:
                diagonal_terms = value[:, 0]
        return self.variance + diagonal_terms

    def K_of_r(self, r):
        return self.variance * np.exp(-0.5 * r ** 2)

    def dK_dr(self, r):
        return -r * self.K_of_r(r)

    def dK2_drdr(self, r):
        return (r ** 2 - 1) * self.K_of_r(r)

    def dK2_drdr_diag(self):
        return -self.variance  # as the diagonal of r is always filled with zeros

    def __getstate__(self):
        dc = super(CausalRBF, self).__getstate__()
        if self.useGPU:
            dc["psicomp"] = PSICOMP_RBF()
            dc["useGPU"] = False
        return dc

    def __setstate__(self, state):
        self.use_invLengthscale = False
        return super(CausalRBF, self).__setstate__(state)

    def spectrum(self, omega):
        assert self.input_dim == 1
        return self.variance * np.sqrt(2 * np.pi) * self.lengthscale * np.exp(-self.lengthscale * 2 * omega ** 2 / 2)

    def parameters_changed(self):
        if self.use_invLengthscale:
            self.lengthscale[:] = 1.0 / np.sqrt(self.inv_l + 1e-200)
        super(CausalRBF, self).parameters_changed()

    def get_one_dimensional_kernel(self, dim):
        """
        Specially intended for Grid regression.
        """
        oneDkernel = GridRBF(input_dim=1, variance=self.variance.copy(), originalDimensions=dim)
        return oneDkernel

    # ---------------------------------------#
    #             PSI statistics            #
    # ---------------------------------------#

    def psi0(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior)[0]

    def psi1(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior)[1]

    def psi2(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior, return_psi2_n=False)[2]

    def psi2n(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior, return_psi2_n=True)[2]

    def update_gradients_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        dL_dvar, dL_dlengscale = self.psicomp.psiDerivativecomputations(
            self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior
        )[:2]
        self.variance.gradient = dL_dvar
        self.lengthscale.gradient = dL_dlengscale
        if self.use_invLengthscale:
            self.inv_l.gradient = dL_dlengscale * (self.lengthscale ** 3 / -2.0)

    def gradients_Z_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        return self.psicomp.psiDerivativecomputations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)[2]

    def gradients_qX_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        return self.psicomp.psiDerivativecomputations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)[3:]

    def update_gradients_diag(self, dL_dKdiag, X):
        super(CausalRBF, self).update_gradients_diag(dL_dKdiag, X)
        if self.use_invLengthscale:
            self.inv_l.gradient = self.lengthscale.gradient * (self.lengthscale ** 3 / -2.0)

    def update_gradients_full(self, dL_dK, X, X2=None):
        super(CausalRBF, self).update_gradients_full(dL_dK, X, X2)
        if self.use_invLengthscale:
            self.inv_l.gradient = self.lengthscale.gradient * (self.lengthscale ** 3 / -2.0)
