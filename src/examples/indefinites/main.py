import pandas as pd
from altk.language.semantics import Universe

from grammar import indefinites_grammar
from meaning import universe

if __name__ == "__main__":
   
    print(universe)

    for exp in indefinites_grammar.enumerate(3):
        print(exp)
        print(exp.to_meaning(universe))
