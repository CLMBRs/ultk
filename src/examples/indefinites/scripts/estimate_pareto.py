from altk.effcomm.optimization import EvolutionaryOptimizer
from altk.language.sampling import random_languages


from ..meaning import universe as indefinites_universe
from ..measures import comm_cost, complexity
from ..util import read_expressions, write_languages

if __name__ == "__main__":
    expressions, expressions_by_meaning = read_expressions(
        "indefinites/outputs/generated_expressions.yml",
        universe=indefinites_universe,
        return_by_meaning=True,
    )

    seed_languages = random_languages(expressions, sampling_strategy="stratified", sample_size=1000, max_size=10)

    def lang_complexity(language):
        return complexity(language, expressions_by_meaning)

    optimizer = EvolutionaryOptimizer(
        [lang_complexity, comm_cost], expressions, 1000, 3, 50, 10
    )
    result = optimizer.fit(seed_languages)

    write_languages(
        result["dominating_languages"],
        "indefinites/outputs/dominating_languages.yml",
        {
            "name": lambda idx, _: f"dominating-{idx}",
            "type": lambda _1, _2: "dominant",
            "complexity": lambda _, lang: lang_complexity(lang),
            "comm_cost": lambda _, lang: comm_cost(lang),
        },
    )
    write_languages(
        result["explored_languages"],
        "indefinites/outputs/explored_languages.yml",
        {
            "name": lambda idx, _: f"explored-{idx}",
            "type": lambda _1, _2: "explored",
            "complexity": lambda _, lang: lang_complexity(lang),
            "comm_cost": lambda _, lang: comm_cost(lang),
        },
    )