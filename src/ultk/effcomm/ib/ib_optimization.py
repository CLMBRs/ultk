from ultk.effcomm.ib.ib_language import IBLanguage
from ultk.effcomm.ib.ib_structure import IBStructure
from ultk.effcomm.ib.ib_utils import generate_random_expressions, IB_EPSILON

import numpy as np
import multiprocessing as mp
import math

LOG_2 = math.log(2)


# Calculate the normal function results for the meanings
def normals(language: IBLanguage, beta: float) -> np.ndarray:
    """Calculates the normals (Z(x, b)) for a given language.
    NOTE: This will break if beta is too high and the language is not near fully complex (i.e. language.qwm is not nearing the identity matrix).
    This is because language.divergence_array will have extremely high values which will cause the exponentiation to go down to a full 0 array.

    Args:
        language (IBLanguage): Language to calculate the normals of
        beta (float): Beta to calculate the normal at

    Returns:
        np.ndarray: Array of normals for each meaning in the structure .
    """
    return np.sum(
        np.exp(-beta * language.divergence_array * LOG_2)
        * language.expressions_prior[:, None],
        axis=0,
    )


def recalculate_language(language: IBLanguage, beta: float) -> IBLanguage:
    """Run an iteration of the BA algorithm to get the language closer to the optimal language for the given beta
    NOTE: As with normals this function will break if beta is too high and the language is not nearly fully complex (i.e. language.qwm is not nearing the identity matrix).
    This is because normals will return a matrix filled with 0s, which will invalidate the math inside the function.

    Args:
        language (IBLanguage): Language to run the BA algorithm on
        beta (float): Beta to calculate the normal at

    Returns:
        IBLanguage: An updated language which is closer to the optimal language.
    """
    # Recalculate q(w|m)
    recalculated_qwm = (
        language.expressions_prior[:, None]
        * np.exp(-beta * language.divergence_array * LOG_2)
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


def calculate_optimal(
    structure: IBStructure, beta: float, start: IBLanguage = None
) -> IBLanguage:
    """Find the optimal language for a given beta given a structure
    NOTE: As with recalculate_language this function will break if beta is too high and the seed language is not nearly fully complex (i.e. language.qwm is not nearing the identity matrix).
    This is because recalculate_language will break under these conditions (see notes for it).
    NOTE: This is not guaranteed to find the exact optimal language, as it can get stuck near the optimal frontier. It may be based to use run_deterministic_annealing

    Args:
        structure (IBStructure): The structure for which the language will be optimized
        beta (float): Beta to calculate the normal at
        start (IBLanguage, optional): A starting point for the optimizer. If not passed in a random langauge will be generated and used

    Returns:
        IBLanguage: An updated language which is closer to the optimal language.
    """
    language = (
        start
        if start is not None
        else IBLanguage(structure, generate_random_expressions(structure.pum.shape[1]))
    )

    converged = False
    close_attempts = 0

    while not converged:
        old = language.complexity - beta * language.iwu
        language = recalculate_language(language, beta)
        if abs(language.complexity - beta * language.iwu - old) <= IB_EPSILON:
            close_attempts += 1
            if close_attempts > 2:
                converged = True
        else:
            close_attempts = 0
        old = language.complexity - beta * language.iwu

    return language


# Modified from embo/Lindsay Skinner's code
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


def run_deterministic_annealing(
    structure: IBStructure,
    betas: tuple[float, ...],
    verbose: bool = False,
    seed: IBLanguage = None,
) -> tuple[tuple[IBLanguage, float], ...]:
    """Run forward or reverse deterministic annealing approach. This means that it will start either with a very high value of beta and work down (reverse)
    or a very low value of beta and work up (forward). When going to a new value of beta it will seed the optimizer with the output of the previous beta iteration
    in an attempt to avoid getting stuck on local maxima.

    For reverse deterministic annealing betas can start at around 2^13, and for forward they can start at 0. Betas should not be evenly spaced.

    NOTE: If doing reverse deterministic annealing you should seed the function with a near fully complex language (i.e. one where the qwm is near the idenity matrix).
    For more information on why see the notes for calculate_optimial

    Args:
        structure (IBStructure): The structure for which the languages will be optimized
        betas (float): The betas to optimize for, this will be iterated through in order
        verbose (bool): Output to console when languages converge
        start (IBLanguage, optional): A starting point for the annealing run. If not passed in a random langauge will be generated and used

    Returns:
        tuple[tuple[IBLanguage, float], ...]: Languages and their respective beta values.
    """
    languages = [calculate_optimal(structure, betas[0], seed)]

    if verbose:
        print(f"Beta {betas[0]} converged (1/{len(betas)})")

    for i in range(1, len(betas)):
        languages.append(calculate_optimal(structure, betas[i], languages[i - 1]))
        if verbose:
            print(f"Beta {betas[i]} converged ({i+1}/{len(betas)})")

    return tuple(zip(languages, betas))
