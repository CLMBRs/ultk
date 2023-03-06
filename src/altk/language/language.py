"""Classes for modeling languages as form-meaning mappings, most important among them the Language and Expression classes.

Example usage:

    >>> from altk.language.language import Expression, Language
    >>> # assuming the meaning `a_few_meaning` has already been constructed
    >>> # define the expression
    >>> a_few = NumeralExpression(form="a few", meaning=a_few_meaning)
    >>> # define a very small language
    >>> lang_1 = Language([a_few])
    >>> # or a slightly larger one with synonymy
    >>> lang_2 = Language([a_few] * 3)
"""

import numpy as np
from abc import abstractmethod
from altk.language.semantics import Meaning, Referent, Universe
from typing import Callable


class Expression:

    """Minimally contains a form and a meaning."""

    def __init__(self, form: str = None, meaning: Meaning = None):
        self.form = form
        self.meaning = meaning

    def can_express(self, ref: Referent) -> bool:
        """Return True if the expression can express the input single meaning point and false otherwise."""
        return ref in self.meaning.referents

    @abstractmethod
    def yaml_rep(self):
        raise NotImplementedError()

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def __eq__(self, __o: object) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError()


class Language:

    """Minimally contains Expression objects."""

    def __init__(self, expressions: list[Expression], **kwargs):
        # Check that all expressions have the same universe
        if len(set([e.meaning.universe for e in expressions])) != 1:
            raise ValueError(
                f"All expressions must have the same meaning universe. Received universes: {[e.meaning.universe for e in expressions]}"
            )

        self.expressions = expressions
        self.universe = expressions[0].meaning.universe

        if "data" in kwargs:
            self.data = kwargs["data"]

    @property
    def expressions(self) -> list[Expression]:
        return self._expressions

    @expressions.setter
    def expressions(self, val: list[Expression]) -> None:
        if not val:
            raise ValueError("list of Expressions must not be empty.")
        self._expressions = val

    def add_expression(self, e: Expression):
        """Add an expression to the list of expressions in a language."""
        self.expressions = self.expressions + [e]

    def pop(self, index: int) -> Expression:
        """Removes an expression at the specified index of the list of expressions, and returns it."""
        if not len(self):
            raise Exception("Cannot pop expressions from an empty language.")
        expressions = self.expressions
        popped = expressions.pop(index)
        self.expressions = expressions
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

    @property
    def universe(self) -> Universe:
        return self._universe

    @universe.setter
    def universe(self, val) -> None:
        self._universe = val

    def __contains__(self, expression) -> bool:
        """Whether the language has the expression"""
        return expression in self.expressions

    @abstractmethod
    def __hash__(self) -> int:
        return hash(tuple(sorted(self.expressions)))

    def __eq__(self, __o: object) -> bool:
        return self.expressions == __o.expressions

    def __len__(self) -> int:
        return len(self.expressions)
