"""Classes for modeling languages as form-meaning mappings, most important among them the Language and Expression classes.

Example usage:

    >>> from ultk.language.language import Expression, Language
    >>> # assuming the meaning `a_few_meaning` has already been constructed
    >>> # define the expression
    >>> a_few = NumeralExpression(form="a few", meaning=a_few_meaning)
    >>> # define a very small language
    >>> lang_1 = Language([a_few])
    >>> # or a slightly larger one with synonymy
    >>> lang_2 = Language([a_few] * 3)
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Generic, Iterable, TypeVar
from ultk.language.semantics import Meaning, Referent
from ultk.util import FrozenDict

# TODO: require Python 3.12 and use type parameter syntax instead? https://docs.python.org/3/reference/compound_stmts.html#type-params
T = TypeVar("T")


@dataclass(eq=True, unsafe_hash=True)
class Expression(Generic[T]):
    """Minimally contains a form and a meaning."""

    # gneric/dummy form and meaning if not specified
    # useful for hashing in certain cases
    # (e.g. a GrammaticalExpression which has not yet been evaluate()'d and so does not yet have a Meaning)
    form: str = ""
    meaning: Meaning[T] = Meaning(FrozenDict(), FrozenDict())

    def can_express(self, referent: Referent) -> bool:
        """Return True if the expression can express the input single meaning point and false otherwise."""
        return referent in self.meaning.referents

    def to_dict(self) -> dict:
        """Return a dictionary representation of the expression."""
        return {"form": self.form, "meaning": self.meaning.__dict__}

    def __str__(self) -> str:
        return self.form

    def __lt__(self, other: object) -> bool:
        return isinstance(other, Expression) and (self.form, other.meaning) < (
            other.form,
            other.meaning,
        )

    def __bool__(self) -> bool:
        return bool(self.form and self.meaning)


class Language:
    """Minimally contains Expression objects."""

    def __init__(self, expressions: tuple[Expression, ...], **kwargs):
        if not expressions:
            raise ValueError(f"Language cannot be empty.")

        self.expressions = tuple(sorted(expressions))
        self.__dict__.update(**kwargs)

    @property
    def expressions(self) -> tuple[Expression, ...]:
        return self._expressions

    @expressions.setter
    def expressions(self, val: tuple[Expression, ...]) -> None:
        if not val:
            raise ValueError("list of Expressions must not be empty.")
        self._expressions = val

    def add_expression(self, e: Expression):
        """Add an expression to the list of expressions in a language."""
        self.expressions = tuple(sorted(tuple(self.expressions) + (e,)))

    def pop(self, index: int) -> Expression:
        """Removes an expression at the specified index of the list of expressions, and returns it."""
        if not len(self):
            raise Exception("Cannot pop expressions from an empty language.")
        popped = self.expressions[index]
        self.expressions = self.expressions[:index] + self.expressions[index + 1 :]
        return popped

    def is_natural(self) -> bool:
        """Whether a language represents a human natural language."""
        raise NotImplementedError

    def degree_property(self, property: Callable[[Expression], bool]) -> float:
        """Count what percentage of expressions in a language have a given property."""
        return sum([property(item) for item in self.expressions]) / len(self)

    def binary_matrix(self) -> np.ndarray:
        """Get a binary matrix of shape `(num_meanings, num_expressions)`
        specifying which expressions can express which meanings."""
        return np.array(
            [
                [float(e.can_express(m)) for e in self.expressions]
                for m in self.universe.referents
            ]
        )

    def as_dict_with_properties(self, **kwargs) -> dict:
        """Return a dictionary representation of the language, including additional properties as keyword arguments.

        This is used in some examples to serialize the language to outputs."""
        the_dict = {"expressions": [str(expr) for expr in self.expressions]}
        the_dict.update(kwargs)
        return the_dict

    def __contains__(self, expression) -> bool:
        """Whether the language has the expression"""
        return expression in self.expressions

    def __hash__(self) -> int:
        return hash(self.expressions)

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Language) and self.expressions == __o.expressions

    def __len__(self) -> int:
        return len(self.expressions)

    def __lt__(self, other) -> bool:
        return self.expressions < other.expressions

    def __str__(self) -> str:
        return (
            "---------\nExpressions:\n"
            + "\n-----\n".join(str(expression) for expression in self.expressions)
            + "\n---------"
        )


# TODO: move this to effcomm?
def aggregate_expression_complexity(
    language: Language,
    expression_complexity_func: Callable[[Expression], float],
    aggregator: Callable[[Iterable[float]], float] = sum,
) -> float:
    """Aggregate complexities for individual `Expression`s into a complexity for a `Language`.

    Args:
        language: the Language to measure
        expression_complexity_func: the function that returns the complexity of an individual expression
        aggregator: (optional, default = sum) the function that aggregates individual complexities

    Returns:
        a float, the complexity of a language
    """
    return aggregator(
        expression_complexity_func(expression) for expression in language.expressions
    )
