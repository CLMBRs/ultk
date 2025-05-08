
from ultk.language.ib.ib_language import IBLanguage
from ultk.language.ib.ib_structure import IBStructure
from ultk.language.ib.ib_utils import generate_random_expressions, kl_divergence

import numpy as np

EPSILON = 0.00001


# The original paper "The information bottleneck method" uses x̃ instead of w, x instead of m, and y instead of u

# Calculate the normal function results for the meanings
def normal(language: IBLanguage, beta: float) -> np.ndarray:
    divergences = np.array(
        [
            [kl_divergence(k, r) for k in language.structure.mu.T]
            for r in language.reconstructed_meanings.T
        ]
    )
    return np.sum(np.exp(-beta*divergences)*language.expressions_prior[:, None])
    

# Do an iteration of the BA Algorithm
# TODO: For some reason this always converges to a one-expression language
# Most likely has something to do with the fact normalization is needed
def recalculate_language(language: IBLanguage, beta: float) -> IBLanguage:
    # Recalculate qwm distribution
    left = language.expressions_prior/normal(language, beta)
    right = np.exp(-beta*np.array(
        [
            [kl_divergence(k, r) for k in language.structure.mu.T]
            for r in language.reconstructed_meanings.T
        ]
    ))
    # Recalculate q(w|m)
    recalculated_qwm = (left[:, None]*right)
    # Drop unused dimensions
    recalculated_qwm = recalculated_qwm[~np.all(recalculated_qwm <= EPSILON, axis=1)]
    # Normalize (?????) TODO: This should not be needed, investigate
    recalculated_qwm /= np.sum(recalculated_qwm, axis=0)
    # Create new language
    return IBLanguage(language.structure, tuple({k: v for k, v in zip(language.structure.meanings, e)} for e in recalculated_qwm))

def calculate_optimal(structure: IBStructure, beta: float) -> IBLanguage:
    language = IBLanguage(structure, expressions=generate_random_expressions(structure.meanings))

    converged = False

    while not converged:
        old = language.complexity - beta*language.iwu
        language = recalculate_language(language, beta)
        if abs(language.complexity - beta*language.iwu - old) <= EPSILON:
            converged = True
        old = language.complexity - beta*language.iwu
        
    return language
