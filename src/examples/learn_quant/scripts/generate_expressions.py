from yaml import dump
from time import time
import argparse
from copy import deepcopy
import pickle
import os
import csv

try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper

from ..quantifier import QuantifierUniverse
from ..grammar import QuantifierGrammar
from ..meaning import create_universe

def enumerate_quantifiers(depth, quantifiers_universe: QuantifierUniverse, quantifiers_grammar: QuantifierGrammar):

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
    
    return expressions_by_meaning

def save_quantifiers(expressions_by_meaning, 
                     out_path="learn_quant/outputs/generated_expressions.yml"):
    
    with open(out_path, "w+") as outfile:
        dump(
            [
                expressions_by_meaning[meaning].to_dict()
                for meaning in expressions_by_meaning
            ],
            outfile,
            Dumper=Dumper,
        )

def save_generation_run(expressions_by_meaning, master_universe, m_size, x_size, depth):
    from pathlib import Path

    outpath = Path("learn_quant/outputs/inclusive/" + "M"+str(m_size) + "_" + "X"+str(x_size) + "_" + str("d"+str(depth))) / Path("generated_expressions.yml")
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)

    print("Saving generated expressions...")
    save_quantifiers(expressions_by_meaning, outpath)

    # Create a new path for the pickle file
    pickle_outpath = Path(outpath).parent / "master_universe.pkl"

    # Open the file in write binary mode and dump the object
    with open(pickle_outpath, 'wb') as f:
        pickle.dump(master_universe, f)

    print("Master universe has been pickled and saved to", pickle_outpath)

def generate_expressions(quantifiers_grammar, args, save=True):
    quantifiers_grammar.add_indices_as_primitives(args.m_size, args.weight)
    quantifiers_universe = create_universe(args.m_size, args.x_size)
    expressions_by_meaning = enumerate_quantifiers(args.depth, quantifiers_universe, quantifiers_grammar)
    if save:
        save_quantifiers(expressions_by_meaning)

def generation_time_trial(quantifiers_grammar, args):
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


def generate_inclusive_expressions(quantifiers_grammar, args, save=True):
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
        if save:
            save_generation_run(expressions_by_meaning, master_universe, args.m_size, args.x_size, args.depth)


if __name__ == "__main__":

    """
    Example usage: 
    python -m learn_quant.scripts.generate_expressions --m_size 2 --x_size 3 --depth 2 --inclusive_universes
    """

    parser = argparse.ArgumentParser(description='Generate expressions')
    parser.add_argument('--m_size', type=int, default=6, help='maximum size of the universe')
    parser.add_argument('--x_size', type=int, default=6, help='number of unique referents from which M may be comprised')
    parser.add_argument('--depth', type=int, default=3, help='maximum depth of the expressions')
    parser.add_argument('--weight', type=float, default=2.0, help='weight of the index primitives')
    parser.add_argument('--time_trial', action=argparse.BooleanOptionalAction, help='time the generation of generating expressions of various sizes and depths')
    parser.add_argument('--inclusive_universes', action=argparse.BooleanOptionalAction, help='generate inclusive universes up to size M or not')
    args = parser.parse_args()

    from ..grammar import quantifiers_grammar

    if args.inclusive_universes:
        # Generate expressions for universes up to size M
        generate_inclusive_expressions(quantifiers_grammar, args)
    elif args.time_trial:
        # Time the generation of expressions for universes up to size M and depths up to D at fixed X
        generation_time_trial(quantifiers_grammar, args)
    else:
        # Generate expressions for a single universe at size M
        generate_expressions(quantifiers_grammar, args)
