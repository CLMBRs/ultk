import numpy as np

# see the utilities in https://github.com/nogazs/ib-color-naming/blob/master/src/tools.py
PRECISION = 1e-12


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
