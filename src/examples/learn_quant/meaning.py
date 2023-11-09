import pandas as pd
from altk.language.semantics import Universe
from learn_quant.quantifier import QuantifierModel
from itertools import product
import random

M_SIZE = 1
referent_pertinence = pd.read_csv("learn_quant/index.csv").to_dict()
#referent_pertinence = {"value": ["A", "B", "both", "neither"]}

def create_universe(referent_pertinence, M_SIZE):

    quantifier_names = []
    for size in range(M_SIZE+1):
        index = [i for i in range(len(referent_pertinence["value"]))]

        x_universe = product("".join([str(x) for x in index]), repeat=size)

        quantifier_names.extend(["".join(array) for array in x_universe])

        quantifier_list = [QuantifierModel(name) for name in quantifier_names if name != '']

    return Universe(quantifier_list)

quantifiers_universe = create_universe(referent_pertinence, M_SIZE)

"""

print(len(quantifiers_universe))

set([x for x in quantifiers_universe.referents])

set([sorted({2,3,6,4,7})[3]])
set(6)
set(sorted({2,3,6,4,7})[3])
"""