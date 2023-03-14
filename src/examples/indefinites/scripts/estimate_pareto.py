from altk.effcomm.informativity import informativity
from altk.effcomm.optimization import EvolutionaryOptimizer
from altk.language.language import aggregate_expression_complexity
from altk.language.sampling import random_languages


from ..meaning import universe as indefinites_universe
from ..util import read_expressions, write_languages

if __name__ == "__main__":

    expressions, expressions_by_meaning = read_expressions(
        "indefinites/outputs/generated_expressions.yml",
        universe=indefinites_universe,
        return_by_meaning=True,
    )

    seed_languages = random_languages(expressions, 1000, max_size=10)

    def complexity(language):
        return aggregate_expression_complexity(
            language, lambda expr: len(expressions_by_meaning[expr.meaning])
        )

    prior = indefinites_universe.prior_numpy()

    def comm_cost(language):
        return 1 - informativity(language, prior)

    optimizer = EvolutionaryOptimizer(
        [complexity, comm_cost], expressions, 1000, 3, 50, 10
    )
    result = optimizer.fit(seed_languages)

    write_languages(
        result["dominating_languages"],
        "indefinites/outputs/dominating_languages.yml",
        {
            "name": lambda idx, _: f"dominating-{idx}",
            "type": lambda _1, _2: "artificial",
        },
    )
    write_languages(
        result["explored_languages"],
        "indefinites/outputs/explored_languages.yml",
        {
            "name": lambda idx, _: f"explored-{idx}",
            "type": lambda _1, _2: "artificial",
        },
    )
