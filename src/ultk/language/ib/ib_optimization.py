from ultk.language.ib.ib_language import IBLanguage
from ultk.language.ib.ib_structure import IBStructure
from ultk.language.ib.ib_utils import (
    generate_random_expressions,
    kl_divergence,
    IB_EPSILON,
)

import numpy as np


# Calculate the normal function results for the meanings
def normal(language: IBLanguage, beta: float) -> np.ndarray:
    divergences = np.array(
        [
            [kl_divergence(k, r) for k in language.structure.mu.T]
            for r in language.reconstructed_meanings.T
        ]
    )
    return np.sum(np.exp(-beta * divergences) * language.expressions_prior[:, None])


# Do an iteration of the BA Algorithm
def recalculate_language(language: IBLanguage, beta: float) -> IBLanguage:
    # Recalculate qwm distribution
    left = language.expressions_prior / normal(language, beta)
    right = np.exp(
        -beta
        * np.array(
            [
                [kl_divergence(k, r) for k in language.structure.mu.T]
                for r in language.reconstructed_meanings.T
            ]
        )
    )
    # Recalculate q(w|m)
    recalculated_qwm = left[:, None] * right
    # Drop unused dimensions
    recalculated_qwm = recalculated_qwm[~np.all(recalculated_qwm <= IB_EPSILON, axis=1)]
    # Normalize (?????) TODO: This should not be needed, investigate
    recalculated_qwm /= np.sum(recalculated_qwm, axis=0)
    # Create new language
    return IBLanguage(
        language.structure,
        recalculated_qwm,
    )


def calculate_optimal(structure: IBStructure, beta: float) -> IBLanguage:
    language = IBLanguage(structure, generate_random_expressions(structure.mu.shape[1]))

    converged = False

    while not converged:
        old = language.complexity - beta * language.iwu
        language = recalculate_language(language, beta)
        if abs(language.complexity - beta * language.iwu - old) <= IB_EPSILON:
            converged = True
        old = language.complexity - beta * language.iwu
        print(
            language.complexity, language.iwu, language.complexity - beta * language.iwu
        )

    return language
