import pandas as pd
from learn_quant.quantifier import QuantifierModel, QuantifierUniverse
from itertools import product, combinations_with_replacement, permutations
from ultk.language.sampling import powerset
import random
import argparse


def create_universe(m_size: int, x_size: int) -> QuantifierUniverse:
    """
    Create a quantifier universe based on the given parameters.
    All references are quantifier models, which are data classes that represent a relation between sets A, B, and M.

    Args:
        m_size (int): The size of the m set.
        x_size (int): The size of the x set.

    Returns:
        QuantifierUniverse: The created quantifier universe.
    """

    possible_quantifiers = []

    for combination in combinations_with_replacement([0, 1, 2, 3], r=m_size):
        combo = list(combination) + [4] * (x_size - m_size)
        permutations_for_combo = set(permutations(combo, r=len(combo)))
        possible_quantifiers.extend(permutations_for_combo)
    possible_quantifiers_name = set(
        ["".join([str(j) for j in i]) for i in possible_quantifiers]
    )

    quantifier_models = set()
    for name in possible_quantifiers_name:
        quantifier_models.add(QuantifierModel(name=name))
    return QuantifierUniverse(tuple(quantifier_models), m_size, x_size)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate expressions")
    parser.add_argument(
        "--m_size", type=int, default=8, help="maximum size of the universe"
    )
    parser.add_argument(
        "--x_size",
        type=int,
        default=8,
        help="number of unique referents from which M may be comprised",
    )

    args = parser.parse_args()

    quantifier_universe = create_universe(args.m_size, args.x_size)
    print("The size of the universe is {}".format(len(quantifier_universe)))
