from typing import Union

import numpy as np
from GPy.core import Param
from GPy.kern import RBF, Kern, Matern52
from GPy.kern.src.grid_kerns import GridRBF
from GPy.kern.src.psi_comp import PSICOMP_RBF, PSICOMP_RBF_GPU
from GPy.kern.src.stationary import Stationary
from paramz.transformations import Logexp


class CausalMixtureViaSumAndProduct(Kern):
    """
    Kernel of the form



    k = (1-mix)*(k1 + k2) + mix*k1*k2 + (1-mix)*(sd_h*sd_h + sd_x*sd_x) + mix*(sd_h*sd_h*sd_x*sd_x)
        -----------------------------   -----------------------------------------------------------
        Standard CoCaBO part            Additional variance bc. interventions

    where sd is the standard deviation.

    Parameters
    ----------
    input_dim
        number of all dims (for k1 and k2 together)
    k1
        First kernel
    k2
        Second kernel
    active_dims
        active dims of this kernel
    mix
        see equation above
    fix_variances
        unlinks the variance parameters if set to True
    fix_mix
        Does not register mix as a parameter that can be learned

    """

    def __init__(
        self,
        input_dim: int,
        variance_adjustment,  # TODO: not clear yet how we combine variance from discrete and continuous vars
        k1: Kern,
        k2: Kern,
        active_dims: Union[list, np.ndarray] = None,
        variance=1.0,
        mix: float = 0.5,
        fix_inner_variances: bool = False,
        fix_mix=True,
        fix_variance=True,
    ):

        super().__init__(input_dim, active_dims, "CausalMixtureViaSumAndProduct")

        self.acceptable_kernels = (RBF, Matern52, CategoryOverlapKernel)

        assert isinstance(k1, self.acceptable_kernels)
        assert isinstance(k2, self.acceptable_kernels)

        self.mix = Param("mix", mix, Logexp())
        self.variance = Param("variance", variance, Logexp())

        self.fix_variance = fix_variance
        if not self.fix_variance:
            self.link_parameter(self.variance)

        # If we are learning the mix, then add it as a visible param
        self.fix_mix = fix_mix
        if not self.fix_mix:
            self.link_parameter(self.mix)

        self.k1 = k1
        self.k2 = k2

        self.fix_inner_variances = fix_inner_variances
        if self.fix_inner_variances:
            self.k1.unlink_parameter(self.k1.variance)
            self.k2.unlink_parameter(self.k2.variance)

        self.link_parameters(self.k1, self.k2)
        self.variance_adjustment = variance_adjustment

    def get_dk_dtheta(self, k: Kern, X, X2=None):
        assert isinstance(k, self.acceptable_kernels)

        if X2 is None:
            X2 = X
        X_sliced, X2_sliced = X[:, k.active_dims], X2[:, k.active_dims]

        if isinstance(k, (RBF, Matern52)):
            dk_dr = k.dK_dr_via_X(X_sliced, X2_sliced)

            # dr/dl
            if k.ARD:
                tmp = k._inv_dist(X_sliced, X2_sliced)
                dr_dl = -np.dstack(
                    [
                        tmp * np.square(X_sliced[:, q : q + 1] - X2_sliced[:, q : q + 1].T) / k.lengthscale[q] ** 3
                        for q in range(k.input_dim)
                    ]
                )
                dk_dl = dk_dr[..., None] * dr_dl
            else:
                r = k._scaled_dist(X_sliced, X2_sliced)
                dr_dl = -r / k.lengthscale
                dk_dl = dk_dr * dr_dl

        elif isinstance(k, CategoryOverlapKernel):
            dk_dl = None

        else:
            raise NotImplementedError

        # Return variance grad as well, if not fixed
        if not self.fix_inner_variances:
            return k.K(X, X2) / k.variance, dk_dl
        else:
            return dk_dl

    def update_gradients_full(self, dL_dK, X, X2=None):

        # This gets the values of dk/dtheta as a NxN matrix (no summations)
        if X2 is None:
            X2 = X
        dk1_dtheta1 = self.get_dk_dtheta(self.k1, X, X2)  # N x N
        dk2_dtheta2 = self.get_dk_dtheta(self.k2, X, X2)  # N x N

        # Separate the variance and lengthscale grads (for ARD purposes)
        if self.fix_inner_variances:
            dk1_dl1 = dk1_dtheta1
            dk2_dl2 = dk2_dtheta2
            dk1_dvar1 = []
            dk2_dvar2 = []
        else:
            dk1_dvar1, dk1_dl1 = dk1_dtheta1
            dk2_dvar2, dk2_dl2 = dk2_dtheta2

        # Evaluate each kernel over its own subspace
        k1_xx = self.k1.K(X, X2)  # N x N
        k2_xx = self.k2.K(X, X2)  # N x N

        # dk/dl for l1 and l2
        # If gradient is None, then vars other than lengthscale don't exist.
        # This is relevant for the CategoryOverlapKernel
        if dk1_dl1 is not None:
            # ARD requires a summation along last axis for each lengthscale
            if hasattr(self.k1, "ARD") and self.k1.ARD:
                dk_dl1 = np.sum(
                    dL_dK[..., None]
                    * (
                        0.5 * dk1_dl1 * (1 - self.mix) * self.variance
                        + self.mix * self.variance * dk1_dl1 * k2_xx[..., None]
                    ),
                    (0, 1),
                )
            else:
                dk_dl1 = np.sum(
                    dL_dK
                    * (0.5 * dk1_dl1 * (1 - self.mix) * self.variance + self.mix * self.variance * dk1_dl1 * k2_xx)
                )
        else:
            dk_dl1 = []

        if dk2_dl2 is not None:
            if hasattr(self.k2, "ARD") and self.k2.ARD:
                dk_dl2 = np.sum(
                    dL_dK[..., None]
                    * (
                        0.5 * dk2_dl2 * (1 - self.mix) * self.variance
                        + self.mix * self.variance * dk2_dl2 * k1_xx[..., None]
                    ),
                    (0, 1),
                )
            else:
                dk_dl2 = np.sum(
                    dL_dK
                    * (0.5 * dk2_dl2 * (1 - self.mix) * self.variance + self.mix * self.variance * dk2_dl2 * k1_xx)
                )
        else:
            dk_dl2 = []

        # dk/dvar for var1 and var 2
        if self.fix_inner_variances:
            dk_dvar1 = []
            dk_dvar2 = []
        else:
            dk_dvar1 = np.sum(
                dL_dK
                * (0.5 * dk1_dvar1 * (1 - self.mix) * self.variance + self.mix * self.variance * dk1_dvar1 * k2_xx)
            )
            dk_dvar2 = np.sum(
                dL_dK
                * (0.5 * dk2_dvar2 * (1 - self.mix) * self.variance + self.mix * self.variance * dk2_dvar2 * k1_xx)
            )

        # Combining the gradients into one vector and updating
        dk_dtheta1 = np.hstack((dk_dvar1, dk_dl1))
        dk_dtheta2 = np.hstack((dk_dvar2, dk_dl2))
        self.k1.gradient = dk_dtheta1
        self.k2.gradient = dk_dtheta2

        # if not self.fix_mix:
        self.mix.gradient = np.sum(dL_dK * (-0.5 * (k1_xx + k2_xx) + (k1_xx * k2_xx))) * self.variance

        # if not self.fix_variance:
        self.variance.gradient = np.sum(self.K(X, X2) * dL_dK) / self.variance

    def K(self, X, X2=None):
        k1_xx = self.k1.K(X, X2)
        k2_xx = self.k2.K(X, X2)
        return self.variance * ((1 - self.mix) * 0.5 * (k1_xx + k2_xx) + self.mix * k1_xx * k2_xx)

    def gradients_X(self, dL_dK, X, X2, which_k=2):
        """
        This function evaluates the gradients w.r.t. the kernel's inputs.
        Default is set to the second kernel, due to this function's
        use in categorical+continuous BO requiring gradients w.r.t.
        the continuous space, which is generally the second kernel.

        which_k = 1  # derivative w.r.t. k1 space
        which_k = 2  # derivative w.r.t. k2 space
        """
        active_kern, other_kern = self.get_active_kernel(which_k)

        # Evaluate the kernel grads in a loop, as the function internally
        # sums up results, which is something we want to avoid until
        # the last step
        active_kern_grads = np.zeros((len(X), len(X2), self.input_dim))
        for ii in range(len(X)):
            for jj in range(len(X2)):
                active_kern_grads[ii, jj, :] = active_kern.gradients_X(
                    np.atleast_2d(dL_dK[ii, jj]), np.atleast_2d(X[ii]), np.atleast_2d(X2[jj])
                )

        other_kern_vals = other_kern.K(X, X2)

        out = np.sum(active_kern_grads * (1 - self.mix + self.mix * other_kern_vals[..., None]), axis=1)
        return out

    def gradients_X_diag(self, dL_dKdiag, X, which_k=2):
        active_kern, _ = self.get_active_kernel(which_k)
        if isinstance(active_kern, Stationary):
            return np.zeros(X.shape)
        else:
            raise NotImplementedError("gradients_X_diag not implemented " "for this type of kernel")

    def get_active_kernel(self, which_k):
        if which_k == 1:
            active_kern = self.k1
            other_kern = self.k2
        elif which_k == 2:
            active_kern = self.k2
            other_kern = self.k1
        else:
            raise NotImplementedError(f"Bad selection of which_k = {which_k}")
        return active_kern, other_kern


class CategoryOverlapKernel(Kern):
    """
    Kernel that counts the number of categories that are the same
    between inputs and returns the normalised similarity score:

    k = variance * 1/N_c * (degree of overlap)
    """

    def __init__(self, input_dim, variance=1.0, active_dims=None, name="catoverlap"):
        super().__init__(input_dim, active_dims=active_dims, name=name)
        self.variance = Param("variance", variance, Logexp())
        self.link_parameter(self.variance)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        # Counting the number of categories that are the same using GPy's
        # broadcasting approach
        diff = X[:, None] - X2[None, :]
        # nonzero location = different cat
        diff[np.where(np.abs(diff))] = 1
        # invert, to now count same cats
        diff1 = np.logical_not(diff)
        # dividing by number of cat variables to keep this term in range [0,1]
        k_cat = self.variance * np.sum(diff1, -1) / self.input_dim
        return k_cat

    def update_gradients_full(self, dL_dK, X, X2=None):
        self.variance.gradient = np.sum(self.K(X, X2) * dL_dK) / self.variance


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
        assert self.input_dim == 1  # TODO: higher dim spectra?
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
