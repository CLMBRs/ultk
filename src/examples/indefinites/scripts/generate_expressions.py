from ultk.util.io import write_expressions

from ultk.language.semantics import Meaning
from ultk.language.grammar.grammar import GrammaticalExpression
from ..grammar import indefinites_grammar
from ..meaning import universe as indefinites_universe

if __name__ == "__main__":
    # a shortest expression for every meaning; with this grammar, every meaning is expressible
    # well within this size, and enumeration stops on its own once no new meanings can be found
    expressions_by_meaning: dict[Meaning, GrammaticalExpression] = (
        indefinites_grammar.get_minimal_expressions(indefinites_universe, max_cost=100)
    )

    # filter out the trivial meaning, results in NaNs
    # iterate over keys, since we need to change the dict itself
    for meaning in list(expressions_by_meaning.keys()):
        if meaning.is_uniformly_false:
            del expressions_by_meaning[meaning]

    print(f"Generated {len(expressions_by_meaning)} unique expressions.")
    write_expressions(
        expressions_by_meaning.values(), "indefinites/outputs/generated_expressions.yml"
    )
