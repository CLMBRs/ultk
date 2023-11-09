import pandas as pd
from altk.language.semantics import Universe
from learn_quant.quantifier import QuantifierModel
from itertools import product, combinations, permutations
from altk.language.sampling import powerset
import random

M_SIZE = 4
X_SIZE = 6


referent_pertinence = {"value": ["A", "B", "both", "neither"]}

def create_universe(referent_pertinence, M_SIZE, X_SIZE):
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
    quantifier_models_list = [QuantifierModel(M=m, A=a, B=b, X=X_SIZE) for (m, a, b) in quantifiers_list]
    return quantifier_models_list

quantifier_universe = create_universe(referent_pertinence, M_SIZE, X_SIZE)
print("The size of the universe is {}".format(len(quantifier_universe)))