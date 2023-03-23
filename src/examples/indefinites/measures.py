from altk.effcomm.informativity import informativity
from altk.language.grammar import GrammaticalExpression
from altk.language.language import Language, aggregate_expression_complexity
from altk.language.semantics import Meaning

from .meaning import universe as indefinites_universe


def complexity(
    language: Language, expressions_by_meaning: dict[Meaning, GrammaticalExpression]
) -> float:
    """Get complexity of a language via minimal expression length in LoT.

    Args:
        language: the Language to measure
        expressions_by_meaning: a dictionary with keys as `Meaning`s, that returns the shortest GrammaticalExpression which expresses that Meaning

    Returns:
        sum of the length of the shortest LoT expression for each meaning in the language
    """
    return aggregate_expression_complexity(
        language, lambda expr: len(expressions_by_meaning[expr.meaning])
    )


prior = indefinites_universe.prior_numpy()


def comm_cost(language: Language) -> float:
    """Get C(L) := 1 - informativity(L).
    Passes in the prior from `indefinites_universe` to altk's informativity calculator.
    """
    return 1 - informativity(language, prior)
