from typing import TypeVar, Iterable
from ultk.language.grammar.grammar import GrammaticalExpression
from ultk.language.semantics import Referent

T = TypeVar("T")
Dataset = Iterable[tuple[Referent, T]]


def all_or_nothing(data: Dataset, tree: GrammaticalExpression) -> float:
    """Basic all or nothing likelihood, return 1 if all data are correctly predicted, 0 otherwise

    Args:
        data (Dataset): data for likelihood calculation
        tree (GrammaticalExpression): GrammaticalExpression for likelihood calculation

    Returns:
        float: likelihood
    """
    return float(all(tree(datum[0]) == datum[1] for datum in data))
