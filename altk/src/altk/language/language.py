"""Classes for modeling languages as form-meaning mappings, most important among them the Language and Expression classes.

The base object of altk is a Language. This is intended to model a language scientifically (especially parts of its semantics) and to enable various use cases. Most notably, experiments such as analyses of efficient communication, learnability, automatic corpus generation for ML probing, etc.
"""

from abc import abstractmethod

from altk.language.semantics import Universe
from altk.language.semantics import Meaning


class Expression:

    """Minimally contains a form and a meaning."""

    def __init__(self, form=None, meaning=None):
        self.form = form
        self.meaning = meaning

    def can_express(self, m: Meaning) -> bool:
        """Return True if the expression can express the input single meaning point and false otherwise."""
        return m in self.meaning.objects

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
        self.expressions = expressions
        self.universe = expressions[0].meaning.universe

    @property
    def expressions(self) -> list[Expression]:
        return self._expressions
    
    @expressions.setter
    def expressions(self, val: list[Expression]) -> None:
        if not val:
            raise ValueError("list of Expressions must not be empty.")
        self._expressions = val

    def has_expression(self, expression: Expression) -> bool:
        """Whether the language has the expression"""
        return expression in self.expressions

    def add_expression(self, e: Expression):
        """Add an expression to the list of expressions in a language."""
        self.expressions = self.expressions + [e]

    def size(self) -> int:
        """Returns the length of the list of expressions in a language."""
        return len(self.expressions)

    def pop(self, index: int) -> Expression:
        """Removes an expression at the specified index of the list of expressions, and returns it."""
        if not self.size():
            raise Exception("Cannot pop expressions from an empty language.")
        expressions = self.expressions
        popped = expressions.pop(index)
        self.expressions = expressions
        return popped

    def is_natural(self) -> bool:
        """Whether a language represents a human natural language."""
        raise NotImplementedError

    @property
    def universe(self) -> Universe:
        return self._universe

    @universe.setter
    def universe(self, val) -> None:
        self._universe = val

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def __eq__(self, __o: object) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def __hash__(self) -> int:
        return hash(tuple(self.expressions))

    def __eq__(self, __o: object) -> bool:
        return self.expressions == __o.expressions
