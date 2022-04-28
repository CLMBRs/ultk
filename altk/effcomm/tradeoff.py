"""Functions for constructing an efficient communication analysis by measuring the simplicity/informativeness trade-off languages and formatting results as a dataframe or a plot."""

import numpy as np

from altk.language.language import Language
from altk.effcomm.complexity import ComplexityMeasure
from altk.effcomm.informativity import InformativityMeasure
from pygmo import non_dominated_front_2d
from typing import Callable
from tqdm import tqdm

from scipy import interpolate
from scipy.spatial.distance import cdist

##############################################################################
# Helper measurement functions
##############################################################################


def pareto_optimal_languages(languages: list[Language]) -> list[Language]:
    """Use pygmo.non_dominated_front_2d to compute the Pareto languages."""
    # TODO: refactor
    dominating_indices = non_dominated_front_2d(
        list(
            zip(
                [1 - lang.informativity for lang in languages],
                [lang.complexity for lang in languages],
            )
        )
    )
    dominating_languages = [languages[i] for i in dominating_indices]
    return list(set(dominating_languages))


def pareto_min_distances(languages: list[Language], pareto_points: list):
    """Measure the Pareto optimality of each language by measuring its Euclidean closeness to the frontier."""
    comm_cost = []
    comp = []
    print("Measuring min distance to frontier ...")
    for lang in tqdm(languages):
        comm_cost.append(1 - lang.informativity)
        comp.append(lang.complexity)
    points = np.array(list(zip(comm_cost, comp)))

    # Measure closeness of each language to any frontier point
    distances = cdist(points, pareto_points)
    min_distances = np.min(distances, axis=1)
    min_distances = min_distances / np.sqrt(2)  # max distance is sqrt(1 + 1)
    return min_distances


def interpolate_data(dominating_languages: list[Language]) -> np.ndarray:
    """Interpolate the points yielded by the pareto optimal languages into a continuous (though not necessarily smooth) curve.

    Args:
        dominating_languages: the list of Language objects representing the Pareto frontier.
    """
    dom_cc = []
    dom_comp = []
    for lang in dominating_languages:
        dom_cc.append(1 - lang.informativity)
        dom_comp.append(lang.complexity)

    values = list(set(zip(dom_cc, dom_comp)))
    pareto_x, pareto_y = list(zip(*values))

    interpolated = interpolate.interp1d(pareto_x, pareto_y, fill_value="extrapolate")
    pareto_costs = np.linspace(0, 1.0, num=5000)
    pareto_complexities = interpolated(pareto_costs)
    pareto_points = np.array(list(zip(pareto_costs, pareto_complexities)))
    return pareto_points


##############################################################################
# Main tradeoff function
##############################################################################


def tradeoff(
    languages: list[Language],
    comp_measure: ComplexityMeasure,
    inf_measure: InformativityMeasure,
    degree_naturalness: Callable,
):
    """Builds a final efficient communication analysis of languages.

    A set of languages, measures of informativity and simplicity (complexity) fully define the efficient communication results, which is the relative (near) Pareto optimality of each language. Measure degrees of natualness, or a categorical analogue of naturalness, as e.g. satisfaction with a semantic universal.

    This function does the following:
    Measure a list of languages, update their internal data, and return a pair of (all languages, dominant_languages).

    Args:
        languages: A list representing the pool of all languages to be measured for an efficient communication analysis.

        comp_measure: the complexity measure to use for the trade-off.

        inf_measure: the informativity measure to use for the trade-off.

        degree_naturalness: the function to measure the degree of (quasi) naturalness for any languages.

    Returns:
        languages: the same list of languages, with their internal efficient communication data updated.

        dominating_languages: a list of the Pareto optimal languages in the simplicity/informativeness tradeoff.
    """
    # measure simplicity, informativity, and semantic universals
    print("Measuring languages for simplicity and informativeness...")
    for lang in tqdm(languages):
        lang.complexity = comp_measure.language_complexity(lang)
        lang.informativity = inf_measure.language_informativity(lang)
        lang.naturalness = degree_naturalness(lang)

    dominating_languages = pareto_optimal_languages(languages)
    min_distances = pareto_min_distances(
        languages, interpolate_data(dominating_languages)
    )

    # TODO: is optimality ever not min_distance?
    print("Setting optimality ...")
    for i, lang in enumerate(tqdm(languages)):
        # warning: yaml that saves lang must use float, not numpy.float64 !
        lang.optimality = 1 - float(min_distances[i])

    return languages, dominating_languages
