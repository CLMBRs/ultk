import pandas as pd
from altk.language.semantics import Universe
from learn_quant.quantifier import QuantifierModel
from itertools import product

M_SIZE = 6
referent_pertinence = pd.read_csv("learn_quant/index.csv").to_dict()

def create_universe(referent_pertinence, M_SIZE):

    quantifier_names = []
    for size in range(M_SIZE+1):
        index = [i for i in range(len(referent_pertinence["value"]))]
        x_universe = product("".join([str(x) for x in index]), repeat=size)
        quantifier_names.extend(["".join(array) for array in x_universe])
        
    return Universe([QuantifierModel(name) for name in quantifier_names if name != ''])

quantifiers_universe = create_universe(referent_pertinence, M_SIZE)

print(len(quantifiers_universe))