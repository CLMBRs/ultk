from typing import TypeVar, Iterable
from ultk.language.grammar.grammar import GrammaticalExpression

T = TypeVar("T")

Dataset = Iterable[tuple["Referent", T]]


def all_or_nothing(data: Dataset, tree: "GrammaticalExpression"):
    return float(all(tree(datum[0]) == datum[1] for datum in data))
