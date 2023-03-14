from altk.effcomm.informativity import informativity
from altk.language.language import aggregate_expression_complexity

from .meaning import universe as indefinites_universe

def complexity(language, expressions_by_meaning):
    return aggregate_expression_complexity(
        language, lambda expr: len(expressions_by_meaning[expr.meaning])
    )

prior = indefinites_universe.prior_numpy()
def comm_cost(language):
    return 1 - informativity(language, prior)