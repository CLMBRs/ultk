import numpy as np

from ultk.language.semantics import Meaning
from ultk.util.frozendict import FrozenDict

IB_EPSILON = 0.00001

# Turns off warnings for functions which are known to take the log of 0
def np_log_ignore(f):
    def wrap(*args, **kwargs):
        np.seterr(divide="ignore", invalid="ignore")
        res = f(*args, **kwargs)
        np.seterr(divide="warn", invalid="warn")
        return res

    return wrap


# Take log of array and set all negative infinities to 0
@np_log_ignore
def safe_log(arr: np.ndarray):
    A = np.log(arr)
    A[np.isinf(A)] = 0
    A[np.isnan(A)] = 0
    return A


# Calculate the KL Divegence of 2 matricies
@np_log_ignore
def kl_divergence(arr1: np.ndarray, arr2: np.ndarray) -> float:
    if (abs(np.sum(arr1) - 1) > IB_EPSILON or abs(np.sum(arr2) - 1) > IB_EPSILON):
        raise ValueError("Arrays are not probability distributions")
    return np.sum(arr1 * safe_log(arr1 / arr2))


def generate_random_expressions(
    meanings: tuple[Meaning[float], ...], seed=None
) -> tuple[FrozenDict[Meaning[float], float], ...]:
    if seed is not None:
        np.random.seed = seed
    values = np.random.dirichlet(np.ones(len(meanings)), size=len(meanings)).T
    return tuple(
        FrozenDict({m: row[i] for i, m in enumerate(meanings)}) for row in values
    )
