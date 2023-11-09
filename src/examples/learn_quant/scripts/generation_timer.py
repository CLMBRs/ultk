from yaml import dump
from itertools import product
import random
import pandas as pd
from time import time
import csv
import argparse

try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper

from altk.language.semantics import Universe
from learn_quant.quantifier import QuantifierModel

from ..grammar import quantifiers_grammar
from ..meaning import create_universe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate expressions')
    parser.add_argument('--m_size', type=int, default=8, help='maximum size of the universe')
    parser.add_argument('--depth', type=int, default=4, help='maximum depth of the expressions')
    args = parser.parse_args()

    referent_pertinence = pd.read_csv("learn_quant/index.csv").to_dict()

    for m_size in range(1, args.m_size+1):
        
        quantifiers_universe = create_universe(referent_pertinence, m_size)

        for depth in range(1, args.depth+1):

            start_time = time()
            print("msize: ", m_size)
            print("depth: ", depth)

            expressions_by_meaning = quantifiers_grammar.get_unique_expressions(
                depth,
                max_size=2 ** len(quantifiers_universe),
                unique_key=lambda expr: expr.evaluate(quantifiers_universe),
                compare_func=lambda e1, e2: len(e1) < len(e2),
            )
        
            # filter out the trivial meaning, results in NaNs
            # iterate over keys, since we need to change the dict itself
            for meaning in list(expressions_by_meaning.keys()):
                if len(meaning.referents) == 0:
                    del expressions_by_meaning[meaning]

            elapsed_time = time() - start_time
            print(elapsed_time)
            print("")

            with open("learn_quant/outputs/generated_expressions" + str(m_size) + "_" + str(depth) + ".yml", "w+") as outfile:
                dump(
                    [
                        expressions_by_meaning[meaning].to_dict()
                        for meaning in expressions_by_meaning
                    ],
                    outfile,
                    Dumper=Dumper,
                )
            
            with open("learn_quant/outputs/generated_expressions_times.csv", "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([m_size, depth, elapsed_time])
