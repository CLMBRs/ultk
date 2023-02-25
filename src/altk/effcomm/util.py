"""Various helper functions for computing complexity and informativity."""
import numpy as np
from scipy.special import logsumexp
from altk.language.semantics import Universe, Meaning
from typing import Callable

##############################################################################
# Miscellaneous helper functions
##############################################################################


def rows_zero_to_uniform(mat) -> np.ndarray:
    """Ensure that `mat` encodes a probability distribution, i.e. each row (indexed by a meaning) is a distribution over expressions: sums to exactly 1.0.

    This is necessary when exploring mathematically possible languages (including natural languages, like Hausa in the case of modals) which sometimes have that a row of the matrix p(word|meaning) is a vector of 0s.
    """

    threshold = 1e-5

    for row in mat:
        # less than 1.0
        if row.sum() and 1.0 - row.sum() > threshold:
            print("row is nonzero and sums to less than 1.0!")
            print(row, row.sum())
            raise Exception
        # greater than 1.0
        if row.sum() and row.sum() - 1.0 > threshold:
            print("row sums to greater than 1.0!")
            print(row, row.sum())
            raise Exception

    return np.array([row if row.sum() else np.ones(len(row)) / len(row) for row in mat])


def build_utility_matrix(
    universe: Universe, utility: Callable[[Meaning, Meaning], float]
) -> np.ndarray:
    """Construct the square matrix specifying the utility function defined for pairs of meanings, used for computing communicative success."""
    return np.array(
        [
            [utility(meaning, meaning_) for meaning_ in universe.referents]
            for meaning in universe.referents
        ]
    )


##############################################################################
# Helper functions for measuring information-theoretic quantities. Code credit belongs to N. Zaslavsky: https://github.com/nogazs/ib-color-naming/blob/master/src/tools.py
##############################################################################


PRECISION = 1e-16

# === DISTRIBUTIONS ===


def marginal(pXY, axis=1):
    """Compute $p(x) = \sum_x p(x,y)$

    Args:
        pXY: a numpy array of shape `(|X|, |Y|)`

    Returns: 
        pY: (axis = 0) or pX (default, axis = 1)
    """
    return pXY.sum(axis)


def conditional(pXY):
    """Compute $p(y|x) = \\frac{p(x,y)}{p(x)}$

    Args:
        pXY: a numpy array of shape `(|X|, |Y|)`

    Returns:  
        pY_X: a numpy array of shape `(|X|, |Y|)`
    """
    pX = pXY.sum(axis=1, keepdims=True)
    return np.where(pX > PRECISION, pXY / pX, 1 / pXY.shape[1])


def joint(pY_X, pX):
    """Compute $p(x,y) = p(y|x) \cdot p(x) $

    Args:
        pY_X: a numpy array of shape `(|X|, |Y|)`

        pX: a numpy array `|X|`
    Returns:
        pXY: a numpy array of the shape `(|X|, |Y|)`
    """
    # breakpoint()
    return pY_X * pX[:, None]


def marginalize(pY_X, pX):
    """Compute $p(y) = \sum_x p(y|x) \cdot p(x)$

    Args:
        pY_X: a numpy array of shape `(|X|, |Y|)`

        pX: a numpy array of shape `|X|`
    
    Returns:  
        pY: a numpy array of shape `|Y|`
    """
    return pY_X.T @ pX


def bayes(pY_X, pX):
    """Compute $p(x|y) = \\frac{p(y|x) \cdot p(x)}{p(y)}$
    Args:
        pY_X: a numpy array of shape `(|X|, |Y|)`
    """
    pXY = joint(pY_X, pX)
    pY = marginalize(pY_X, pX)
    return np.where(pY > PRECISION, pXY / pY, 1 / pXY.shape[0]).T


# === INFORMATION ===


def xlogx(p):
    """Compute $x \\log p(x)$"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p > PRECISION, p * np.log2(p), 0)


def H(p, axis=None):
    """Compute the entropy of p, $H(X) = - \sum_x x \\log p(x)$"""
    return -xlogx(p).sum(axis=axis)


def MI(pXY):
    """Compute mutual information, $I[X:Y]$"""
    return H(pXY.sum(axis=0)) + H(pXY.sum(axis=1)) - H(pXY)


def DKL(p, q, axis=None):
    """Compute KL divergences, $D_{KL}[p||q]$"""
    return (xlogx(p) - np.where(p > PRECISION, p * np.log2(q + PRECISION), 0)).sum(
        axis=axis
    )
