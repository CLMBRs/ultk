from ultk.language.language import Expression
from ultk.language.semantics import Meaning, Universe
from ultk.language.grammar import Grammar, GrammaticalExpression
from typing import Iterable
from yaml import dump, Dumper, load, Loader


def write_expressions(expressions: Iterable[Expression], filename: str) -> None:
    """Write expressions to a YAML file.

    This is particularly useful for writing GrammaticalExpressions, which have a `func` field that can't be serialized. This function uses `to_dict` to determine which properties of the Expression to write.

    Args:
        expressions: the expressions to write
        filename: the file to write to
    """
    with open(filename, "w") as f:
        dump([expr.to_dict() for expr in expressions], f, Dumper=Dumper)


def read_grammatical_expressions(
    filename: str,
    grammar: Grammar,
    re_parse: bool = False,
    universe: Universe | None = None,
    return_by_meaning=True,
) -> tuple[list[GrammaticalExpression], dict[Meaning, GrammaticalExpression]]:
    """Read grammatical expressions from a YAML file.

    Optionally re-parse and (re-)evaluate the expressions using the provided grammar and universe.

    Args:
        filename: the file to read
        grammar: the grammar to use for parsing (and for re-populating the `.func` attribute of each GrammaticalExpression)
        re_parse: whether to re-parse the expressions
        universe: the universe to use for evaluation
        return_by_meaning: whether to return a dictionary mapping meanings to expressions or not

    Returns:
        a list of GrammaticalExpressions and a dictionary mapping meanings to expressions (empty if `return_by_meaning` is False)
    """
    if re_parse and (grammar is None or universe is None):
        raise ValueError("Must provide grammar and universe if re-parsing expressions.")

    with open(filename, "r") as f:
        expression_list = load(f, Loader=Loader)

    if re_parse:
        final_exprs = [
            (
                grammar.parse(expr_dict["term_expression"])
                if re_parse
                else GrammaticalExpression(**expr_dict)
            )
            for expr_dict in expression_list
        ]
    else:
        final_exprs = [
            GrammaticalExpression.from_dict(expr_dict, grammar)
            for expr_dict in expression_list
        ]
    if universe is not None:
        [expr.evaluate(universe) for expr in final_exprs]
    by_meaning = {}
    if return_by_meaning:
        by_meaning = {expr.meaning: expr for expr in final_exprs}
    return final_exprs, by_meaning
