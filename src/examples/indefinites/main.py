import pandas as pd
from altk.language.sampling import all_expressions, all_meanings

from grammar import indefinites_grammar
from meaning import universe as indefinites_universe


if __name__ == "__main__":

    print(indefinites_universe)

    """
    for exp in indefinites_grammar.enumerate(3):
        print(exp)
        print(exp.to_meaning(indefinites_universe))
    """

    expressions = all_expressions(all_meanings(indefinites_universe))
    for exp in expressions:
        print(exp)
