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

from grammar import indefinites_grammar
from meaning import universe as indefinites_universe

from tqdm import tqdm

if __name__ == "__main__":

    print(indefinites_universe)

    start = timeit.default_timer()
    unique_exprs = indefinites_grammar.get_unique_expressions(
        3,
        max_size=2 ** len(indefinites_universe),
        unique_key=lambda expr: expr.evaluate(indefinites_universe),
        compare_func=lambda e1, e2: len(e1) < len(e2),
    )
    print(len(unique_exprs))
    end = timeit.default_timer()
    print(f"Took {end - start} seconds.")
    [print(f"Meaning: {meaning}\nExpression: {unique_exprs[meaning]}\n\n") for meaning in unique_exprs]
    for meaning in unique_exprs:
        if len(meaning.referents) == 0:
            print(unique_exprs[meaning])
        if len(meaning.referents) == 6:
            print(unique_exprs[meaning])
    print(any(len(meaning.referents) == 0 for meaning in unique_exprs))
    print(any(len(meaning.referents) == 6 for meaning in unique_exprs))

    all_exprs = set()
    for expr in tqdm(
        indefinites_grammar.enumerate(
            3,
            unique_dict=unique_exprs,
            unique_key=lambda expr: expr.evaluate(indefinites_universe),
            compare_func=lambda e1, e2: len(e1) < len(e2),
        )
    ):
        all_exprs.add(expr)
        pass
    print(len(unique_exprs))
    """
    for exp in all_exprs:
        print(exp)
        exp.evaluate(indefinites_universe)
        print(exp.meaning)

    expressions = list(all_expressions(all_meanings(indefinites_universe)))
    for exp in expressions:
        print(exp)

    languages = generate_languages(Language, expressions, 10, 1000)["languages"]
    print(len(languages))
    print([len(language) for language in languages])

    languages = list(all_languages(expressions, max_size=3))
    print(len(languages))

    languages = list(random_languages(expressions, 1000, max_size=7))
    print(len(languages))
    print([len(language) for language in languages])

    language = languages[56]
    print(language)
    print(informativity(language, language.universe.prior_numpy()))

    def complexity(language):
        return aggregate_expression_complexity(
            language, lambda gram_expr: len(gram_expr)
        )
    """

