from yaml import dump

try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper

from altk.effcomm.informativity import informativity
from altk.effcomm.optimization import EvolutionaryOptimizer
from altk.language.language import Language, aggregate_expression_complexity
from altk.language.sampling import random_languages


from ..meaning import universe as indefinites_universe
from ..util import read_expressions

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

    def write_languages(
        languages: list[Language], filename: str, name_prefix: str = "", **kwargs
    ) -> None:
        lang_dicts = [
            languages[idx].to_dict(
                name=f"{name_prefix}-{idx}",
                complexity=complexity(languages[idx]),
                comm_cost=comm_cost(languages[idx]),
                **kwargs,
            )
            for idx in range(len(languages))
        ]
        with open(filename, "w+") as f:
            dump(lang_dicts, f, Dumper=Dumper)

    write_languages(
        result["dominating_languages"],
        "indefinites/outputs/dominating_languages.yml",
        name_prefix="dominating",
        type="artificial",
    )
    write_languages(
        result["explored_languages"],
        "indefinites/outputs/explored_languages.yml",
        name_prefix="explored",
        type="artificial",
    )
