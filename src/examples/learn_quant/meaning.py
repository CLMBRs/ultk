import pandas as pd
from altk.language.semantics import Universe
from learn_quant.quantifier import QuantifierModel
from itertools import product, combinations, permutations
from altk.language.sampling import powerset
import random

"""
0 : A
1 : B
2 : A | B
3 : M - (A | B)
4 : X - (M | A | B)
"""

def create_universe(M_SIZE, X_SIZE):
    quantifiers_list = []
    for m_size in range(M_SIZE):
        quantifiers_at_msize = []
        possible_M = [x for x in combinations(range(X_SIZE), M_SIZE)]
        for M in possible_M:
            possible_AorB = [set(x) for x in powerset(M)]
            combs_A_B_for_M = [z for z in product(possible_AorB, possible_AorB)]
            for (A, B) in combs_A_B_for_M:
                quantifiers_at_msize.append([set(M), A, B])
        quantifiers_list.extend(quantifiers_at_msize)
    quantifier_universe = Universe([QuantifierModel(M=m, A=a, B=b) for (m, a, b) in quantifiers_list])
    return quantifier_universe

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate expressions')
    parser.add_argument('--m_size', type=int, default=8, help='maximum size of the universe')
    parser.add_argument('--x_size', type=int, default=8, help='number of unique referents from which M may be comprised')
    args = parser.parse_args()

    quantifier_universe = create_universe(M_SIZE, X_SIZE)
    print("The size of the universe is {}".format(len(quantifier_universe)))