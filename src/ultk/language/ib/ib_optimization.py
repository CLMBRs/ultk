
from ultk.language.ib.ib_language import IBLanguage
from ultk.language.ib.ib_structure import IBStructure
from ultk.language.ib.ib_utils import kl_divergence

import numpy as np


# The original paper "The information bottleneck method" uses x̃ instead of w, x instead of m, and y instead of u

# Calculate all of the normal function results for the meanings
def normals(language: IBLanguage, beta: float) -> np.ndarray:
    divergences = np.array(
        [
            [kl_divergence(k, r) for k in language.structure.mu.T]
            for r in language.reconstructed_meanings.T
        ]
    )
    return np.sum(np.exp(-beta*divergences)*language.expressions_prior[:, None], axis=1)
    

# Do an interation of the BA Algorithm
def recalculate_language(language: IBLanguage, beta: float) -> IBLanguage:
    # Recalculate qwm distribution
    left = language.expressions_prior/normals(language, beta)
    right = np.exp(-beta*np.array(
        [
            [kl_divergence(k, r) for k in language.structure.mu.T]
            for r in language.reconstructed_meanings.T
        ]
    ))
    # Recalculate q(w|m)
    recalculated_qwm = left[:, None]*right
    # Create new language
    return IBLanguage(language.structure, tuple({k: v for k, v in zip(language.structure.meanings, e)} for e in recalculated_qwm))

def calculate_optimal(structure: IBStructure, beta: float) -> IBLanguage:
    pass
