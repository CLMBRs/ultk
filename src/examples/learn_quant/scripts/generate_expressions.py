from yaml import dump
import argparse
try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper

from ..grammar import quantifiers_grammar
from ..meaning import create_universe

def enumerate_quantifiers(depth, quantifiers_universe):

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

if __name__ == "__main__":

    from ..grammar import quantifiers_grammar

    parser = argparse.ArgumentParser(description='Generate expressions')
    parser.add_argument('--m_size', type=int, default=8, help='maximum size of the universe')
    parser.add_argument('--x_size', type=int, default=8, help='number of unique referents from which M may be comprised')
    parser.add_argument('--depth', type=int, default=4, help='maximum depth of the expressions')
    args = parser.parse_args()

    quantifiers_universe = create_universe(args.m_size, args.x_size)
    expressions_by_meaning = enumerate_quantifiers(args.depth, quantifiers_universe)
    save_quantifiers(expressions_by_meaning)