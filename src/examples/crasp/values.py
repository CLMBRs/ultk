"""Values of C-RASP operations, and the operations on them that the rules in `grammar.py` are written with.

A C-RASP operation assigns a value (a Boolean or a count) to every position of a string.  The rules in `grammar.py` do
not depend on how those values are stored: they only use the operations of `PositionValues`.  This lets the same rules
compute the values of an operation for one string at a time (`StringValues`, used by
`GrammaticalExpression.evaluate`), or for every string in a universe at once (see `representations/`).

Similarly, terminal rules are given something with a `positions` attribute: either a `StringReferent` (see `meaning.py`),
whose `positions` are those of one string, or a stand-in for all strings in a universe at once.  Either way, `positions`
can build the values of the initial operations `Q_a(i)` and of constants.
"""

from typing import Any, Self

import numpy as np


class HashableArray:
    """A numpy array, made hashable by its contents, so that it can be used as a key in a dictionary.

    Two `HashableArray`s are equal iff they have the same class, dtype and contents.

    Attributes:
        array: the values
    """

    __slots__ = ("array", "_hash")

    def __init__(self, array: np.ndarray):
        self.array = array
        self._hash = hash((array.dtype.str, array.tobytes()))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        return (
            type(self) is type(other)
            and self._hash == other._hash
            and self.array.dtype == other.array.dtype
            and np.array_equal(self.array, other.array)
        )


class PositionValues(HashableArray):
    """The values of a C-RASP operation at a sequence of positions: the interface that the rules in `grammar.py` use.

    Each kind of values (e.g. for one string, or for every prefix in a universe) implements these operations.  Boolean
    and count values are the same class; it is up to the grammar to only apply each operation to the right kind.

    `==` is not an operation on values: it compares two `PositionValues` as a whole (see `HashableArray`).
    """

    __slots__ = ()

    def __invert__(self) -> Self:
        """Negation of Boolean values: `not P(i)`."""
        raise NotImplementedError

    def __and__(self, other: Self) -> Self:
        """Conjunction of Boolean values: `P1(i) and P2(i)`."""
        raise NotImplementedError

    def __le__(self, other: Self) -> Self:
        """Comparison of counts: `C1(i) <= C2(i)`, a Boolean value."""
        raise NotImplementedError

    def __add__(self, other: Self) -> Self:
        """Addition of counts: `C1(i) + C2(i)`."""
        raise NotImplementedError

    def __sub__(self, other: Self) -> Self:
        """Subtraction of counts: `C1(i) - C2(i)`."""
        raise NotImplementedError

    def minimum(self, other: Self) -> Self:
        """Minimum of counts: `min(C1(i), C2(i))`."""
        raise NotImplementedError

    def maximum(self, other: Self) -> Self:
        """Maximum of counts: `max(C1(i), C2(i))`."""
        raise NotImplementedError

    def where(self, if_true: Self, if_false: Self) -> Self:
        """Conditional on Boolean values: `P(i) ? C1(i) : C2(i)`."""
        raise NotImplementedError

    def count_so_far(self) -> Self:
        """Counting of Boolean values: `#[j <= i] P(j)`, the number of positions up to `i` at which `P` is true."""
        raise NotImplementedError

    def at_end(self) -> Any:
        """Whether Boolean values are true at the last position, i.e. whether a program with these values accepts.

        Returns:
            for one string, a `bool`; for many strings at once, a hashable value with one `bool` per string
        """
        raise NotImplementedError


class StringValues(PositionValues):
    """The values of a C-RASP operation at each position of one string.

    Attributes:
        array: one value per position, as `bool` or `int64`
    """

    __slots__ = ()

    def __init__(self, array: np.ndarray):
        if array.dtype != bool:
            array = array.astype(np.int64, copy=False)
        super().__init__(array)

    def __invert__(self) -> "StringValues":
        return StringValues(np.logical_not(self.array))

    def __and__(self, other: "StringValues") -> "StringValues":
        return StringValues(np.logical_and(self.array, other.array))

    def __le__(self, other: "StringValues") -> "StringValues":
        return StringValues(self.array <= other.array)

    def __add__(self, other: "StringValues") -> "StringValues":
        return StringValues(self.array + other.array)

    def __sub__(self, other: "StringValues") -> "StringValues":
        return StringValues(self.array - other.array)

    def minimum(self, other: "StringValues") -> "StringValues":
        return StringValues(np.minimum(self.array, other.array))

    def maximum(self, other: "StringValues") -> "StringValues":
        return StringValues(np.maximum(self.array, other.array))

    def where(
        self, if_true: "StringValues", if_false: "StringValues"
    ) -> "StringValues":
        return StringValues(np.where(self.array, if_true.array, if_false.array))

    def count_so_far(self) -> "StringValues":
        return StringValues(np.cumsum(self.array, dtype=np.int64))

    def at_end(self) -> bool:
        return bool(self.array[-1])

    def __repr__(self) -> str:
        return f"StringValues({self.array.tolist()})"


class StringPositions:
    """The positions of one string, which build the values of terminal rules (see `StringReferent` in `meaning.py`).

    Attributes:
        string: the string
    """

    def __init__(self, string: str):
        self.string = string

    def symbol_is(self, symbol: str) -> StringValues:
        """The values of the initial operation `Q_symbol(i)`: whether `symbol` is at position `i`."""
        return StringValues(
            np.array([character == symbol for character in self.string])
        )

    def constant(self, value: bool | int) -> StringValues:
        """The values of a constant operation, the same at every position."""
        dtype = bool if isinstance(value, bool) else np.int64
        return StringValues(np.full(len(self.string), value, dtype=dtype))
