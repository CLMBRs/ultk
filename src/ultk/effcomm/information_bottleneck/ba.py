import numpy as np
from collections import namedtuple
from typing import Any
from tqdm import tqdm
from ultk.effcomm.probability import (
    information_cond,
    random_stochastic_matrix,
    add_noise_to_stochastic_matrix,
)

##############################################################################
# Base Rate Distortion class
##############################################################################


class BaseRDOptimizer:
    def __init__(
        self,
        betas: np.ndarray,
        *args,
        max_it: int = 1000,
        eps: float = 1e-15,
        ignore_converge: bool = False,
        xhat_size=None,
        **kwargs,
    ) -> None:
        """Base initializer for a Blahut-Arimoto-based optimizer of the Rate Distortion function.

        Args:
            betas: 1D array, values of beta to search

            max_it: max number of iterations

            args: propagated to `self.beta_iterate` as *kwargs

            eps: accuracy required by the algorithm: the algorithm stops if there is no change in distortion value of more than `eps` between consecutive iterations

            ignore_converge: whether to run the optimization until `max_it`, ignoring the stopping criterion specified by `eps`.

            xhat_size: the size of the output alphabet. The resulting encoders will be of shape (x, xhat)

            kwargs: propagated to `self.beta_iterate` as **kwargs
        """
        self.betas = betas
        self.max_it = max_it
        self.eps = eps
        self.ln_eps = np.log1p(eps)
        self.ignore_converge = ignore_converge

        self.init_args = args
        self.init_kwargs = kwargs

        self.ln_px = None  # shape `(x)`
        self.ln_qxhat_x = None  # shape `(x, xhat)`
        self.dist_mat = None  # shape `(x, xhat)`

        self.xhat_size = xhat_size
        # if xhat_size is None:
        # self.xhat_size = len(self.ln_px)

        self.result: namedtuple = None
        self.results: list[namedtuple] = []

    def get_results(self) -> list[namedtuple]:
        # Re-initialize results
        self.result = None
        self.results = []

        self.beta_iterate(*self.init_args, **self.init_kwargs)
        return self.results

    def update_eqs(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Main Blahut-Arimoto update steps."""
        raise NotImplementedError

    def next_result(self, beta, *args, **kwargs) -> None:
        """Get the result of the converged BA iteration."""
        raise NotImplementedError

    def beta_iterate(
        self,
        *args,
        num_restarts: int = 0,
        ensure_monotonicity: bool = True,
        disable_tqdm: bool = False,
        reverse_annealing: bool = True,
        **kwargs,
    ) -> None:
        """Iterate the BA algorithm for an array of values of beta, using reverse deterministic annealing.

        Args:
            num_restarts: number of times to restart each beta-optimization

            ensure_monotonicity: whether to remove points that would make a rate distortion curve non-monotonic

            reverse_annealing: whether to use reverse annealing or regular annealing. If self.output_size < len(self.ln_px), then this is set to false.
        """
        if (
            self.xhat_size is not None and self.xhat_size < len(self.ln_px)
        ) or not reverse_annealing:
            reverse = False
        else:
            reverse = True

        results = self.anneal(
            *args,
            reverse=reverse,
            num_restarts=num_restarts,
            ensure_monotonicity=ensure_monotonicity,
            disable_tqdm=disable_tqdm,
            **kwargs,
        )

        # Postprocessing
        if ensure_monotonicity:
            indices = compute_lower_bound(
                [(item.rate, item.distortion) for item in results]
            )
            results = [x if i in indices else None for i, x in enumerate(results)]
            self.results = results

    def anneal(
        self,
        *args,
        reverse: bool = True,
        num_restarts: int = 0,
        disable_tqdm: bool = False,
        **kwargs,
    ) -> list:
        results = []
        betas = np.sort(self.betas)

        if reverse:
            # sort betas in decreasing order
            betas = betas[::-1]
            # start with bijective mapping
            init_q = np.eye(len(self.ln_px))
        else:
            # Random degenerate initialization
            xhat = random_stochastic_matrix(shape=(1, self.xhat_size), gamma=1e-2)
            init_q = np.stack([xhat.squeeze()] * len(self.ln_px))

        pbar = tqdm(
            betas,
            disable=disable_tqdm,
            desc=f"{'reverse ' if reverse else ''}annealing beta",
        )
        for beta in pbar:
            pbar.set_description(f"beta={beta:.5f}")
            candidates = []
            for _ in range(num_restarts + 1):
                self.blahut_arimoto(beta, *args, init_q=init_q, **kwargs)
                cand = self.results[-1]
                init_q = cand.qxhat_x
                candidates.append(cand)
            best = min(candidates, key=lambda x: x.rate + beta * x.distortion)
            results.append(best)

        if reverse:
            results = results[::-1]

        return results

    ############################################################################
    # Blahut Arimoto iteration
    ############################################################################

    def blahut_arimoto(
        self,
        beta,
        *args,
        **kwargs,
    ) -> None:
        """Update the self-consistent equations for a Rate Distortion objective.

        Args:
            beta: (scalar) the slope of the rate-distoriton function at the point where evaluation is required
        """
        len_x = len(self.ln_px)
        if "init_q" in kwargs:
            init_q: np.ndarray = kwargs["init_q"]
            # Add small value for working in logspace
            # init_q_precision = init_q + PRECISION
            init_q_precision = add_noise_to_stochastic_matrix(init_q, weight=1e-2)
            init_q_precision /= init_q_precision.sum(axis=1, keepdims=True)
            self.ln_qxhat_x = np.log(init_q_precision)
        else:
            self.ln_qxhat_x = np.log(random_stochastic_matrix((len_x, len_x)))

        it = 0
        converged = False
        while not converged:
            it += 1
            prev_q = np.copy(self.ln_qxhat_x)

            # Main BA update
            self.update_eqs(beta, *args, **kwargs)

            # convergence check
            # TODO: consider updating the Result tuple to include convergence field. if converged, record the iteration. If not, record False/None
            if self.ignore_converge:
                converged = it >= self.max_it
            else:
                converged = (
                    it == self.max_it
                    or np.sum(np.abs(self.ln_qxhat_x - prev_q)) < self.ln_eps
                )

        self.results.append(self.next_result(beta, *args, **kwargs))

    def compute_distortion(self, *args, **kwargs) -> float:
        """Compute the expected distortion for the current p(x), q(xhat|x) and dist_mat."""
        raise NotImplementedError

    def compute_rate(
        self,
        *args,
        **kwargs,
    ) -> float:
        """Compute the information rate for the current p(x), q(xhat|x)."""
        return information_cond(np.exp(self.ln_px), np.exp(self.ln_qxhat_x))


##############################################################################
# Postprocessing helper
##############################################################################


def compute_lower_bound(rd_points):
    """
    Remove all points in a rate-distortion curve that would make it nonmonotonic and
    return only the resulting monotonic indices.

    This is required to remove the random fluctuations in the result induced by the BA algorithm getting stuck in local minima.

    Acknowledgement: https://github.com/epiasini/embo-github-mirror/blob/master/embo/utils.py#L77.

    Args:
        rd_points: list of pairs of floats, where each pair represents an estimated (rate, distortion) pair,
                   and *ordered by increasing rate*.

    Returns:
        selected_indices: 1D numpy array containing the indices of the points selected to ensure monotonically decreasing values.
    """
    pts = np.array(rd_points, dtype=np.float32)
    selected_indices = [0]

    for idx in range(1, len(pts)):
        # Check that each point increases in rate and does not increase in distortion
        if (
            pts[idx, 0] >= pts[selected_indices[-1], 0]  # Monotonically increasing rate
            and pts[idx, 1]
            <= pts[selected_indices[-1], 1]  # Monotonically decreasing distortion
        ):
            selected_indices.append(idx)

    return np.array(selected_indices, dtype=np.int32)
