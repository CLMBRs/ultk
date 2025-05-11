from ultk.language.ib.ib_language import IBLanguage
from ultk.language.ib.ib_structure import IBStructure
from ultk.language.ib.ib_utils import generate_random_expressions, IB_EPSILON

import numpy as np
import multiprocessing as mp


# Calculate the normal function results for the meanings
def normals(language: IBLanguage, beta: float) -> np.ndarray:
    return np.sum(
        np.exp(-beta * language.divergence_array) * language.expressions_prior[:, None],
        axis=0,
    )


# Do an iteration of the BA Algorithm
def recalculate_language(language: IBLanguage, beta: float) -> IBLanguage:
    # Recalculate q(w|m)
    recalculated_qwm = (
        language.expressions_prior[:, None]
        * np.exp(-beta * language.divergence_array)
        / normals(language, beta)
    )

    # Normalize (This is not in the paper but embo does it)
    # This should not be needed but its a nice sanity check, probably should throw a warning if
    # recalculated_qwm's columns do not sum to 1
    recalculated_qwm /= np.sum(recalculated_qwm, axis=0)

    # Drop unused dimensions
    recalculated_qwm = recalculated_qwm[~np.all(recalculated_qwm <= IB_EPSILON, axis=1)]

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
        # TODO: Remove after debugging
        print(
            language.complexity,
            language.iwu,
            language.complexity - beta * language.iwu,
            sep="\t",
        )

    return language


# Modified from embo/Lindsay Skinner's code
# TODO: Test
def get_optimial_languages(
    structure: IBStructure, start: float, end: float, steps: int, threads: int = 1
) -> tuple[tuple[IBLanguage, float], ...]:
    # Get beta values
    beta_vec = np.linspace(start, end, steps)

    # Parallel computing of compression for desired beta values
    with mp.Pool(processes=threads) as pool:
        results = [
            pool.apply_async(calculate_optimal, args=(structure, b)) for b in beta_vec
        ]
        langs = tuple(p.get() for p in results)
    return tuple(zip(langs, beta_vec))
