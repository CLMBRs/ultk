import pandas as pd

from altk.effcomm.informativity import informativity
from altk.effcomm.optimization import EvolutionaryOptimizer
from altk.language.language import aggregate_expression_complexity
from altk.language.sampling import all_languages, random_languages


from ..grammar import indefinites_grammar
from ..meaning import universe as indefinites_universe
from ..util import read_expressions, read_natural_languages

if __name__ == "__main__":

    # NB: in a larger-scale study, you would probably want to do this once and save
    # the resulting dictionary as a file.  We are not doing that in the present case
    # because it is faster to just generate from scratch than to read from a file and
    # re-parse the inputs.
    """
    meanings_by_expressions = indefinites_grammar.get_unique_expressions(
        3,
        max_size=2 ** len(indefinites_universe),
        unique_key=lambda expr: expr.evaluate(indefinites_universe),
        compare_func=lambda e1, e2: len(e1) < len(e2),
    )
    expressions = list(meanings_by_expressions.values())
    """

    expressions = read_expressions(
        "indefinites/outputs/generated_expressions.yml", universe=indefinites_universe
    )
    expressions_by_meaning = {expression.meaning: expression for expression in expressions}
    print(len(expressions_by_meaning))

    seed_languages = list(random_languages(expressions, 1000, max_size=8))
    print(all(expr.meaning in expressions_by_meaning for seed_language in seed_languages for expr in seed_language.expressions))

    def complexity(language):
        return aggregate_expression_complexity(
            # TODO: change this measure to closer match the paper?
            language,
            lambda expr: len(expressions_by_meaning[expr.meaning])
            # lambda expression: len(expressions_by_meaning[expression.meaning]),
        )

    prior = indefinites_universe.prior_numpy()
    def comm_cost(language):
        return 1 - informativity(language, prior)

    optimizer = EvolutionaryOptimizer([complexity, comm_cost], expressions, 500, 3, 50, 10)
    result = optimizer.fit(seed_languages)
    # TODO: uniqueness issue here?!?!
    for lang in result["dominating_languages"]:
        print()
        [print(expr) for expr in lang.expressions]
        print(comm_cost(lang))
        print(complexity(lang))

    
