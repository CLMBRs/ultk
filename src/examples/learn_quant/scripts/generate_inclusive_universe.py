from yaml import dump
from itertools import product
import random
import pandas as pd
from time import time
import csv
import argparse
from copy import deepcopy
import pickle

try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper

from ultk.language.semantics import Universe
from learn_quant.quantifier import QuantifierModel
from learn_quant.scripts.generate_expressions import enumerate_quantifiers, save_generation_run

from ..meaning import create_universe
from ..grammar import quantifiers_grammar

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate expressions')
    parser.add_argument('--m_size', type=int, default=6, help='maximum size of the universe')
    parser.add_argument('--x_size', type=int, default=8, help='number of unique referents from which M may be comprised')
    parser.add_argument('--depth', type=int, default=4, help='depth of generation of expressions')
    parser.add_argument('--weight', type=float, default=2.0, help='weight of the index primitives')
    args = parser.parse_args()

    import os
    
    # Flush times as they are output to time log
    master_universe = None
    depth = args.depth

    # Don't consider m_size=0, as this is the trivial meaning
    for m_size in range(1, args.m_size+1):
        
        print("Generating a universe where x_size={} and m_size={}.".format(args.x_size, m_size))

        # Ensure that primitives are added to the grammar up to `m_size`
        quantifiers_grammar_at_depth = deepcopy(quantifiers_grammar)
        quantifiers_grammar_at_depth.add_indices_as_primitives(args.m_size, args.weight)

        # Create the universe
        creation_start = time()
        quantifier_universe = create_universe(m_size, args.x_size)
        creation_elapsed = time() - creation_start
        print("The size of the universe is {}".format(len(quantifier_universe)))

        if not master_universe:
            master_universe = quantifier_universe
        else:
            master_universe += quantifier_universe

    enumeration_start = time()
    print("maximum_msize: ", m_size)
    print("depth: ", depth)

    expressions_by_meaning = enumerate_quantifiers(depth, master_universe, quantifiers_grammar_at_depth)

    enumerate_elapsed = time() - enumeration_start
    print(enumerate_elapsed)
    print("")

    save_generation_run(expressions_by_meaning, master_universe, args.m_size, args.x_size, args.depth)
