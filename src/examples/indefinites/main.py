from altk.effcomm.informativity import informativity
from altk.language.language import Language, aggregate_expression_complexity
from altk.language.sampling import (
    all_expressions,
    all_languages,
    all_meanings,
    generate_languages,
    random_languages,
)

from grammar import indefinites_grammar
from meaning import universe as indefinites_universe


if __name__ == "__main__":

    print(indefinites_universe)

    all_exprs, unique_exprs = indefinites_grammar.enumerate(
        4,
        #unique_key=lambda expr: expr.evaluate(indefinites_universe),
        #compare_func=lambda e1, e2: len(e1) < len(e2),
    )
    print(len(all_exprs))
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

