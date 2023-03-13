from altk.effcomm.informativity import informativity
from altk.language.language import Language, aggregate_expression_complexity
from altk.language.sampling import (
    all_expressions,
    all_languages,
    all_meanings,
    generate_languages,
    random_languages,
)

import timeit

from ..grammar import indefinites_grammar
from ..meaning import universe as indefinites_universe

from tqdm import tqdm

if __name__ == "__main__":

    meanings_by_expressions = indefinites_grammar.get_unique_expressions(
        3,
        max_size=2 ** len(indefinites_universe),
        unique_key=lambda expr: expr.evaluate(indefinites_universe),
        compare_func=lambda e1, e2: len(e1) < len(e2),
    )

    expressions = list(meanings_by_expressions.values())


    languages = list(all_languages(expressions, max_size=3))
    print(len(languages))

    languages = list(random_languages(expressions, 1000, max_size=8))
    print(len(languages))
    print([len(language) for language in languages])

    language = languages[56]
    print(language)
    print(informativity(language, language.universe.prior_numpy()))

    def complexity(language):
        return aggregate_expression_complexity(
            language, lambda gram_expr: len(gram_expr)
        )

    print(complexity(language))

