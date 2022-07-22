"""Various helper functions for computing complexity and informativity."""
import numpy as np
from altk.language.semantics import Universe, Meaning
from typing import Callable


def uniform_prior(universe: Universe) -> np.ndarray:
    """Return a 1-D numpy array of size |universe| reprsenting uniform distribution over the referents in a universe."""
    return np.array(
        [1 / len(universe.referents) for _ in range(len(universe.referents))]
    )


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


def compute_sparsity(mat: np.ndarray) -> float:
    """Number of 0s / number of elements in matrix."""
    total = mat.shape[0] * mat.shape[1]
    zeros = np.count_nonzero(mat == 0)
    return float(zeros / total)
