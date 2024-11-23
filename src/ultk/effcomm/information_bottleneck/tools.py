import numpy as np
from scipy.special import softmax
from scipy.stats import entropy

##############################################################################
# Numerical precision helpers
##############################################################################

PRECISION = 1e-15

def get_gaussian_noise(shape):
    """Small Gaussian noise."""
    return np.random.normal(loc=0.0, scale=1e-15, size=shape)

def add_noise_to_stochastic_matrix(q, weight=1e-2):
    """
    Given an input stochastic matrix `q`, sample a stochastic matrix `p` and
    mix it with the input with a small weight `weight`, i.e., return q + weight * p.
    """
    # Generate a stochastic matrix `p` using a Dirichlet distribution
    p = np.random.dirichlet(np.ones(q.shape[1]), size=q.shape[0])
    # Mix `q` with `p` using the specified weight
    noisy_matrix = q + weight * p
    # Normalize to ensure the rows sum to 1
    noisy_matrix /= noisy_matrix.sum(axis=1, keepdims=True)
    return noisy_matrix

def random_stochastic_matrix(shape, gamma=1e-10):
    """
    Initialize a stochastic matrix (2D array) that sums to 1 along the rows.

    Args:
        shape: tuple, the desired shape of the stochastic matrix (e.g., `(rows, cols)`).
        gamma: float, scaling factor for the random normal initialization.

    Returns:
        A stochastic matrix with rows summing to 1.
    """
    energies = gamma * np.random.randn(*shape)
    return softmax(energies, axis=1)

##############################################################################
# Probability and Information
##############################################################################

def entropy_bits(p: np.ndarray, axis=None) -> float:
    """Compute entropy of p, $H(X) = - \sum_x p(x) \log p(x)$, in bits."""
    return entropy(p, base=2, axis=axis)


def mutual_info(pxy: np.ndarray) -> float:
    """Compute mutual information, $I(X;Y)$ in bits.

    Args:
        pxy: 2D numpy array of shape `(x, y)`
    """
    return (
        entropy_bits(pxy.sum(axis=0))
        + entropy_bits(pxy.sum(axis=1))
        - entropy_bits(pxy)
    )


def kl_divergence(p: np.ndarray, q: np.ndarray, axis=None, base=np.e) -> float:
    """Compute KL divergence (in nats by defaut) between p and q, $D_{KL}[p \| q]$.

    Args:
        p: np.ndarray, lhs of KL divergence

        q: np.ndarray, rhs of KL divergence
    """
    return entropy(
        p,
        q,
        axis=axis,
        base=base,
    )


# Common pattern for rate-distortion optimizations
def information_cond(pA: np.ndarray, pB_A: np.ndarray) -> float:
    """Compute the mutual information $I(A;B)$ from a joint distribution defind by $P(A)$ and $P(B|A)$

    Args:
        pA: array of shape `|A|` the prior probability of an input symbol (i.e., the source)

        pB_A: array of shape `(|A|, |B|)` the probability of an output symbol given the input
    """
    pab = pB_A * pA[:, None]
    mi = mutual_info(pxy=pab)
    if mi < 0.0 and not np.isclose(mi, 0.0, atol=1e-5):
        raise Exception
    return mi

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
            and pts[idx, 1] <= pts[selected_indices[-1], 1]  # Monotonically decreasing distortion
        ):
            selected_indices.append(idx)

    return np.array(selected_indices, dtype=np.int32)
