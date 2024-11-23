"""Information Bottleneck optimizer."""

import numpy as np

from collections import namedtuple
from scipy.special import logsumexp, log_softmax

from .ba import BaseRDOptimizer
from .tools import PRECISION, information_cond, mutual_info, kl_divergence


def ib_kl(py_x: np.ndarray, qy_xhat: np.ndarray) -> np.ndarray:
    """Compute the IB distortion matrix, the KL divergence between p(y|x) and q(y|xhat), in nats."""
    # D[p(y|x) || q(y|xhat)],
    # input shape `(x, xhat, y)`, output shape `(x, xhat)`
    return kl_divergence(py_x[:, None, :], qy_xhat[None, :, :], axis=2)


##############################################################################
# Return type of each item in `get_results()`
##############################################################################

IBResult = namedtuple(
    "IBResult",
    [
        "qxhat_x",
        "rate",
        "distortion",
        "accuracy",
        "beta",
    ],
)

##############################################################################
# Update equations
##############################################################################


def next_ln_qxhat(ln_px: np.ndarray, ln_qxhat_x: np.ndarray) -> np.ndarray:
    # q(xhat) = sum_x p(x) q(xhat | x),
    # shape `(xhat)`
    return logsumexp(ln_px[:, None] + ln_qxhat_x, axis=0)


def next_ln_qxhat_x(ln_qxhat: np.ndarray, beta: float, dist_mat: np.ndarray):
    # q(x_hat | x) = q(x_hat) exp(- beta * d(x, x_hat)) / Z(x)
    return log_softmax(ln_qxhat[None, :] - beta * dist_mat, axis=1)


def next_ln_qy_xhat(ln_pxy: np.ndarray, ln_qxhat_x: np.ndarray) -> np.ndarray:
    # p(x),
    # shape `(x)`
    ln_px = logsumexp(ln_pxy, axis=1)

    # p(y|x),
    # shape `(x,y)`
    ln_py_x = ln_pxy - ln_px[:, None]

    ln_qx_xhat = next_ln_qx_xhat(ln_px, ln_qxhat_x)  # `(xhat, x)`

    # p(y|xhat) = sum_x p(y|x) p(x|xhat),
    # shape `(xhat, y)`
    ln_qy_xhat = logsumexp(
        ln_py_x[None, :, :] + ln_qx_xhat[:, :, None],  # `(xhat, x, y)`
        axis=1,
    )

    return ln_qy_xhat


def next_ln_qx_xhat(ln_px: np.ndarray, ln_qxhat_x: np.ndarray) -> np.ndarray:
    # q(xhat),
    # shape `(xhat)`
    ln_qxhat = next_ln_qxhat(ln_px, ln_qxhat_x)

    # q(x,xhat) = p(x) q(xhat|x),
    # shape `(x, xhat)`
    ln_qxxhat = ln_px[:, None] + ln_qxhat_x

    # p(x|xhat) = q(x, xhat) / q(xhat),
    # shape `(xhat, x)`
    ln_qx_xhat = ln_qxxhat.T - ln_qxhat[:, None]

    return ln_qx_xhat


##############################################################################
# IB Optimizer
##############################################################################


class IBOptimizer(BaseRDOptimizer):
    def __init__(
        self,
        pxy: np.ndarray,
        betas: np.ndarray,
        *args,
        **kwargs,
    ) -> None:
        """Estimate the optimal encoder for a given value of `beta` for the Information Bottleneck objective [Tishby et al., 1999]:

        $\min_{q} I[X:\hat{X}] + \\beta \mathbb{E}[D_{KL}[p(y|x) || p(y|\hat{x})]].$

        Args:
            pxy: 2D array of shape `(|X|, |Y|)` representing the joint probability mass function of the source and relevance variables.

            beta: (scalar) the slope of the rate-distoriton function at the point where evaluation is required
        """
        super().__init__(betas, *args, **kwargs)

        # Add small value for working in logspace
        pxy_precision = pxy + PRECISION
        pxy_precision /= pxy_precision.sum()
        self.ln_pxy = np.log(pxy_precision)

        self.ln_px = logsumexp(self.ln_pxy, axis=1)  # `(x)`
        self.px = np.exp(self.ln_px)
        self.ln_py_x = self.ln_pxy - logsumexp(
            self.ln_pxy, axis=1, keepdims=True
        )  # `(x, y)`
        self.results: list[IBResult] = None

    def get_results(self) -> list[IBResult]:
        return super().get_results()

    def next_dist_mat(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Vanilla IB distortion matrix."""
        self.dist_mat = ib_kl(np.exp(self.ln_py_x), np.exp(self.ln_qy_xhat))

    def update_eqs(
        self,
        beta,
        *args,
        **kwargs,
    ) -> None:
        """Iterate the vanilla IB update equations."""
        self.ln_qxhat = next_ln_qxhat(self.ln_px, self.ln_qxhat_x)
        self.ln_qy_xhat = next_ln_qy_xhat(self.ln_pxy, self.ln_qxhat_x)
        self.next_dist_mat(*args, **kwargs)
        self.ln_qxhat_x = next_ln_qxhat_x(self.ln_qxhat, beta, self.dist_mat)

    def compute_distortion(self, *args, **kwargs) -> float:
        # NOTE: we may still need to debug this; watch out for negative values
        # return np.exp(logsumexp(self.ln_px + self.ln_qxhat_x + np.log(self.dist_mat)))
        I_xy = mutual_info(np.exp(self.ln_pxy))
        edkl = I_xy - self.compute_accuracy()
        return edkl

    def compute_accuracy(self, *args, **kwargs) -> float:
        return information_cond(
            np.exp(self.ln_qxhat),
            np.exp(self.ln_qy_xhat),
        )

    def next_result(self, beta, *args, **kwargs) -> IBResult:
        """Get the result of the converged BA iteration for the IB objective.

        Returns:
            an IBResult namedtuple of `(qxhat_x, rate, distortion, accuracy, beta)` values. This is:

                `qxhat_x`, the optimal encoder, such that the

                `rate` (in bits) of compressing X into X_hat, is minimized for the level of

                `distortion` between X, X_hat with respect to Y, i.e. the

                `accuracy` I[X_hat:Y] is maximized, for the specified

                `beta` trade-off parameter
        """
        return IBResult(
            np.exp(self.ln_qxhat_x),
            self.compute_rate(),
            self.compute_distortion(),
            self.compute_accuracy(),
            beta,
        )
