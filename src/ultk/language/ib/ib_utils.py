import numpy as np

from ultk.language.semantics import Meaning
from ultk.util.frozendict import FrozenDict

IB_EPSILON = 0.00001


# Turns off warnings for functions which are known to take the log of 0
def np_log_ignore(f):
    def wrap(*args, **kwargs):
        np.seterr(divide="ignore")
        res = f(*args, **kwargs)
        np.seterr(divide="warn")
        return res

    return wrap


# Take log of array and set all negative infinities to 0
@np_log_ignore
def safe_log(arr: np.ndarray):
    A = np.log2(arr)
    A[np.isinf(A)] = 0
    A[np.isnan(A)] = 0
    return A


# Calculate the KL Divegence of 2 matricies
@np_log_ignore
def kl_divergence(arr1: np.ndarray, arr2: np.ndarray) -> float:
    if abs(np.sum(arr1) - 1) > IB_EPSILON or abs(np.sum(arr2) - 1) > IB_EPSILON:
        raise ValueError("Arrays are not probability distributions")
    return np.sum(arr1 * safe_log(arr1 / arr2))


def mutual_information(pxy: np.ndarray, px: np.ndarray, py: np.ndarray) -> float:
    if abs(np.sum(px) - 1) > IB_EPSILON or abs(np.sum(py) - 1) > IB_EPSILON:
        raise ValueError("Arrays are not probability distributions")
    return np.sum(safe_log(pxy / px[:, None]) * (pxy * py))


def generate_random_expressions(meanings: int, seed=None) -> np.ndarray:
    if seed is not None:
        np.random.seed = seed
    values = np.random.dirichlet(np.ones(meanings), size=meanings).T
    # Normalize because there is a chance the different is larger than IB_EPSILON
    return values.T / np.sum(values.T, axis=0)
