import numpy as np

IB_EPSILON = 0.00001


def np_log_ignore(f):
    """Custom function decorator to avoid numpy warnings when taking the log of 0 (since that should be set to 0 in this implmentation)"""

    def wrap(*args, **kwargs):
        np.seterr(divide="ignore")
        res = f(*args, **kwargs)
        np.seterr(divide="warn")
        return res

    return wrap


@np_log_ignore
def safe_log(arr: np.ndarray):
    """Takes the log base 2 of an array and sets all infs and nans to 0

    Args:
        arr (np.ndarray): Input array
    Returns:
        np.ndarray: Output log base 2 array with all nans and infs set to 0
    """
    A = np.log2(arr)
    A[np.isinf(A)] = 0
    A[np.isnan(A)] = 0
    return A


@np_log_ignore
def kl_divergence(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Calculates the KL Divergence of two probability distrubutions

    Args:
        arr1 (np.ndarray): One of the input matricies (must sum to 1)
        arr2 (np.ndarray): One of the input matricies (must sum to 1)
    Returns:
        float: The Kullback-Leibler divergence (in bits)
    """
    if abs(np.sum(arr1) - 1) > IB_EPSILON or abs(np.sum(arr2) - 1) > IB_EPSILON:
        raise ValueError("Arrays are not probability distributions")
    return np.sum(arr1 * safe_log(arr1 / arr2))


def mutual_information(pxy: np.ndarray, px: np.ndarray, py: np.ndarray) -> float:
    """Calculates the mutual of two probability distrubutions using the conditional probability

    Args:
        pxy (np.ndarray): Conditional probability matrix of p(x|y). Dimensions must be ||px|| x ||py||
        px (np.ndarray): Probability distribution of x
        py (np.ndarray): Probability distribution of y
    Returns:
        float: The mutual information between the two (in bits)
    """
    if abs(np.sum(px) - 1) > IB_EPSILON or abs(np.sum(py) - 1) > IB_EPSILON:
        raise ValueError("Arrays are not probability distributions")
    if (np.abs(np.sum(pxy, axis=0) - 1) > IB_EPSILON).any():
        raise ValueError("All columns of conditional probability matrix must sum to 1")
    if pxy.shape[0] != px.shape[0] or pxy.shape[1] != py.shape[0]:
        raise ValueError("pxy is not ||px|| x ||py||")
    return np.sum(safe_log(pxy / px[:, None]) * (pxy * py))


def generate_random_expressions(meanings: int, seed: int = None) -> np.ndarray:
    """Generates a random qwm matrix for a given amount of meanings

    Args:
        meanings (int): The number of meanings for the expressions being generated
        seed (int, optional): Seed for numpy
    Returns:
        np.ndarray: Random qwm matrix for a language
    """
    if seed is not None:
        np.random.seed = seed
    values = np.random.dirichlet(np.ones(meanings), size=meanings).T
    # Normalize because there is a chance the different is larger than IB_EPSILON
    return values.T / np.sum(values.T, axis=0)
