import pandas as pd

from altk.effcomm.informativity import informativity
from altk.language.language import aggregate_expression_complexity
from altk.language.sampling import all_languages, random_languages


from ..grammar import indefinites_grammar
from ..meaning import universe as indefinites_universe
from ..util import read_natural_languages

if __name__ == "__main__":

    # NB: in a larger-scale study, you would probably want to do this once and save
    # the resulting dictionary as a file.  We are not doing that in the present case
    # because it is faster to just generate from scratch than to read from a file and
    # re-parse the inputs.
    meanings_by_expressions = indefinites_grammar.get_unique_expressions(
        3,
        max_size=2 ** len(indefinites_universe),
        unique_key=lambda expr: expr.evaluate(indefinites_universe),
        compare_func=lambda e1, e2: len(e1) < len(e2),
    )

    expressions = list(meanings_by_expressions.values())

    seed_languages = list(random_languages(expressions, 1000, max_size=8))

    language = languages[56]
    print(language)
    print(informativity(language, language.universe.prior_numpy()))

    def complexity(language):
        return aggregate_expression_complexity(
            # TODO: change this measure to closer match the paper?
            language,
            lambda expression: len(meanings_by_expressions[expression.meaning]),
        )
