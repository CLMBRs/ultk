from yaml import dump
from itertools import product
import random
import pandas as pd
from time import time
import csv
import argparse
from copy import deepcopy

try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper

from ultk.language.semantics import Universe
from learn_quant.quantifier import QuantifierModel
from learn_quant.scripts.generate_expressions import enumerate_quantifiers, save_quantifiers

from ..meaning import create_universe
from ..grammar import quantifiers_grammar

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate expressions')
    parser.add_argument('--m_size', type=int, default=8, help='maximum size of the universe')
    parser.add_argument('--x_size', type=int, default=8, help='number of unique referents from which M may be comprised')
    parser.add_argument('--depth', type=int, default=4, help='maximum depth of the expressions')
    parser.add_argument('--weight', type=float, default=2.0, help='weight of the index primitives')
    args = parser.parse_args()

    import os
    time_log = "learn_quant/outputs/generated_expressions_time_log.csv"

    # Remove old time log
    if os.path.exists(time_log):
        os.remove(time_log)

    # Flush times as they are output to time log
    with open("learn_quant/outputs/generated_expressions_time_log.csv", "a+", newline='') as f:

        writer = csv.writer(f)
        writer.writerow(["m_size", "x_size", "depth", "universe_size", "elapsed_time_enumerate", "elapsed_time_creation"])

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

            # Don't consider depth=1, as depth is not deep enough to generate and expression 
            for depth in range(2, args.depth+1):

                enumeration_start = time()
                print("msize: ", m_size)
                print("depth: ", depth)

                expressions_by_meaning = enumerate_quantifiers(depth, quantifier_universe, quantifiers_grammar_at_depth)

                enumerate_elapsed = time() - enumeration_start
                print(enumerate_elapsed)
                print("")

                from pathlib import Path
                outpath = Path("learn_quant/outputs") / Path("M"+str(m_size)) / Path("X"+str(args.x_size)) / Path(str("d"+str(depth))) / Path("generated_expressions.yml")
                Path(outpath).parent.mkdir(parents=True, exist_ok=True)

                print("Saving generated expressions...")
                save_quantifiers(expressions_by_meaning, outpath)

                writer.writerow([m_size, args.x_size, depth, len(quantifier_universe), enumerate_elapsed, creation_elapsed])
