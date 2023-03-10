import pandas as pd
from altk.language.language import Language
from altk.language.sampling import all_expressions, all_meanings, generate_languages

from grammar import indefinites_grammar
from meaning import universe as indefinites_universe


if __name__ == "__main__":

    print(indefinites_universe)

    """
    for exp in indefinites_grammar.enumerate(3):
        print(exp)
        print(exp.to_meaning(indefinites_universe))
    """

    expressions = list(all_expressions(all_meanings(indefinites_universe)))
    for exp in expressions:
        print(exp)

    languages = generate_languages(Language, expressions, 8, 1000)["languages"]
    print(len(languages))
    print(languages[0])
    print(languages[123])