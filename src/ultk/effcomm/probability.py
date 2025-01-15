import numpy as np
from scipy.special import softmax
from scipy.stats import entropy

# see the utilities in https://github.com/nogazs/ib-color-naming/blob/master/src/tools.py
PRECISION = 1e-15


def marginal(pXY: np.ndarray, axis: int = 1) -> np.ndarray:
    """Computer marginals of a joint distribution.

    Args:
        pXY: a joint distribution of shape (|X|, |Y|), corresponding to p(x, y)
        axis: the axis along which to compute the marginal

    Returns:
        either pY (axis = 0) or pX (default, axis = 1)
    """
    return pXY.sum(axis)


def joint(pY_X: np.ndarray, pX: np.ndarray) -> np.ndarray:
    """Compute a joint distribution from a conditional and a prior.

    Args:
        pY_X: a conditional distribution of shape (|X|, |Y|), corresponding to p(y|x)
        pX: a prior distribution of shape (|X|,), corresponding to p(x)

    Returns:
        a joint distribution of shape (|X|, |Y|), corresponding to p(x, y)
    """
    return pY_X * pX[:, None]


def marginalize(pY_X: np.ndarray, pX: np.ndarray) -> np.ndarray:
    """Marginalize a conditional distribution (without a detour through the joint).

    Args:
        pY_X: a conditional distribution of shape (|X|, |Y|), corresponding to p(y|x)
        pX: a prior distribution of shape (|X|,), corresponding to p(x)

    Returns:
        a marginal distribution of shape (|Y|,), corresponding to p(y)
    """
    return pY_X.T @ pX


def bayes(pY_X: np.ndarray, pX: np.ndarray) -> np.ndarray:
    """Perform Bayesian inference, computing p(x|y) from p(y|x) and p(x).

    Args:
        pY_X: a conditional distribution of shape (|X|, |Y|), corresponding to p(y|x)
        pX: a prior distribution of shape (|X|,), corresponding to p(x)

    Returns:
        a posterior distribution of shape (|Y|, |X|), corresponding to p(x|y)
    """
    # (|X|, |Y|)
    pXY = joint(pY_X, pX)
    print(pXY.shape)
    # (|Y|,)
    pY = marginalize(pY_X, pX)
    print(pY.shape)
    # (|Y|, |X|)
    return np.where(pY > PRECISION, pXY / pY, 1 / pXY.shape[0]).T


##############################################################################
# Numerical precision helpers
##############################################################################


def get_gaussian_noise(
    shape,
    loc=0.0,
    scale=1e-15,
):
    """Small Gaussian noise."""
    return np.random.normal(loc, scale, size=shape)


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
# Information
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
