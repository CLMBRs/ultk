"""Values of C-RASP operations, and the interfaces that the rules in `grammar.py` are written with.

A C-RASP operation assigns a value to every position of a string: a Boolean (for Boolean-valued operations) or an
integer (for count-valued operations).  The rules in `grammar.py` do not depend on how those values are stored, or on
whether they are computed for one string or for many strings at once.  They only use these interfaces, one for each
step of computing the values of a program:

| Interface             | What it is                                   | One string                       | All strings at once (`representations/`) |
|-----------------------|----------------------------------------------|----------------------------------|------------------------------------------|
| `HasPositions`        | what terminal rules are applied to           | `StringReferent`                 | `AllStrings`                             |
| `Positions`           | its `positions`, which build terminal values | `StringPositions`                | `PrefixPositions`                        |
| `Booleans`, `Counts`  | values, which non-terminal rules combine     | `StringBooleans`, `StringCounts` | e.g. `PackedPrefixBooleans`, `PackedPrefixCounts` |

`Booleans` and `Counts` are also the non-terminals of the grammar: each rule's argument and return types (read from its
function's annotations by `Rule.from_callable`) say which kinds of values it combines, and each kind only has the
operations that apply to it.

For one string, these are what `GrammaticalExpression.evaluate` uses (`StringReferent` is in `meaning.py`); for all
strings at once, they are what the representations in `representations/` use.
"""

from typing import Any, Protocol, Self

import numpy as np


class HashableArray:
    """A numpy array, made hashable by its contents, so that it can be used as a key in a dictionary.

    Two `HashableArray`s are equal iff they have the same class, dtype and contents.

    The hash is computed once, from the contents at construction, so the contents must never change afterwards.  A
    `HashableArray` therefore takes ownership of its array and makes it read-only: writing to it, through this object or
    any other reference to the same array, raises an error instead of silently invalidating the hash.  (Values are
    always built from freshly computed arrays, so this does not need a copy.)

    Attributes:
        array: the values, read-only
    """

    __slots__ = ("array", "_hash")

    def __init__(self, array: np.ndarray):
        array.flags.writeable = False
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


class Booleans(HashableArray):
    """The values of a Boolean-valued C-RASP operation (`P` in the paper) at a sequence of positions.

    This is the interface that the rules in `grammar.py` use, and the grammar's non-terminal for Boolean-valued
    operations.  Each kind of values (for one string, or for every prefix in a universe) implements it, together with
    `Counts`.

    `==` is not an operation on values: it compares two values as a whole (see `HashableArray`).
    """

    __slots__ = ()

    def __invert__(self) -> Self:
        """Negation: `not P(i)`."""
        raise NotImplementedError

    def __and__(self, other: Self) -> Self:
        """Conjunction: `P1(i) and P2(i)`."""
        raise NotImplementedError

    def where(self, if_true: "Counts", if_false: "Counts") -> "Counts":
        """Conditional: `P(i) ? C1(i) : C2(i)`."""
        raise NotImplementedError

    def count_so_far(self) -> "Counts":
        """Counting: `#[j <= i] P(j)`, the number of positions up to `i` at which `P` is true."""
        raise NotImplementedError

    def at_end(self) -> Any:
        """Whether the values are true at the last position, i.e. whether a program with these values accepts.

        Returns:
            for one string, a `bool`; for many strings at once, a hashable value with one `bool` per string
        """
        raise NotImplementedError


class Counts(HashableArray):
    """The values of a count-valued C-RASP operation (`C` in the paper) at a sequence of positions.

    This is the interface that the rules in `grammar.py` use, and the grammar's non-terminal for count-valued operations.
    Each kind of values (for one string, or for every prefix in a universe) implements it, together with `Booleans`.

    `==` is not an operation on values: it compares two values as a whole (see `HashableArray`).
    """

    __slots__ = ()

    def __le__(self, other: "Counts") -> Booleans:
        """Comparison: `C1(i) <= C2(i)`."""
        raise NotImplementedError

    def __add__(self, other: Self) -> Self:
        """Addition: `C1(i) + C2(i)`."""
        raise NotImplementedError

    def __sub__(self, other: Self) -> Self:
        """Subtraction: `C1(i) - C2(i)`."""
        raise NotImplementedError

    def minimum(self, other: Self) -> Self:
        """Minimum: `min(C1(i), C2(i))`."""
        raise NotImplementedError

    def maximum(self, other: Self) -> Self:
        """Maximum: `max(C1(i), C2(i))`."""
        raise NotImplementedError


class Positions:
    """The positions that values are assigned to, which build the values of terminal rules.

    Terminal rules get a `Positions` from what they are applied to (see `HasPositions`), and use it to build the values of
    the initial operations `Q_a(i)` and of constants.  Subclasses determine which positions these are (those of one
    string, or the prefixes of all strings in a universe), and so which kind of `Booleans` and `Counts` they build.
    """

    def symbol_is(self, symbol: str) -> Booleans:
        """The values of the initial operation `Q_symbol(i)`: whether `symbol` is at position `i`."""
        raise NotImplementedError

    def constant_booleans(self, value: bool) -> Booleans:
        """The values of a constant Boolean-valued operation, the same at every position."""
        raise NotImplementedError

    def constant_counts(self, value: int) -> Counts:
        """The values of a constant count-valued operation, the same at every position."""
        raise NotImplementedError


class HasPositions(Protocol):
    """What terminal rules are applied to: anything with `positions`.

    In ULTK, the function of a terminal rule is applied to a referent.  For C-RASP, that is a `StringReferent` (see
    `meaning.py`), whose `positions` are those of one string.  To compute values for all strings at once, the
    representations in `representations/` instead apply terminal rules to an `AllStrings`, whose `positions` are the
    prefixes of all strings in the universe.  So that the same terminal functions work for both, they only use
    `positions`.
    """

    @property
    def positions(self) -> Positions: ...


class StringBooleans(Booleans):
    """The values of a Boolean-valued C-RASP operation at each position of one string.

    Attributes:
        array: one `bool` per position
    """

    __slots__ = ()

    def __invert__(self) -> "StringBooleans":
        return StringBooleans(np.logical_not(self.array))

    def __and__(self, other: Booleans) -> "StringBooleans":
        return StringBooleans(np.logical_and(self.array, other.array))

    def where(self, if_true: Counts, if_false: Counts) -> "StringCounts":
        return StringCounts(np.where(self.array, if_true.array, if_false.array))

    def count_so_far(self) -> "StringCounts":
        return StringCounts(np.cumsum(self.array))

    def at_end(self) -> bool:
        return bool(self.array[-1])

    def __repr__(self) -> str:
        return f"StringBooleans({self.array.tolist()})"


class StringCounts(Counts):
    """The values of a count-valued C-RASP operation at each position of one string.

    Attributes:
        array: one count per position, as `int64`
    """

    __slots__ = ()

    def __init__(self, array: np.ndarray):
        super().__init__(array.astype(np.int64, copy=False))

    def __le__(self, other: Counts) -> StringBooleans:
        return StringBooleans(self.array <= other.array)

    def __add__(self, other: Counts) -> "StringCounts":
        return StringCounts(self.array + other.array)

    def __sub__(self, other: Counts) -> "StringCounts":
        return StringCounts(self.array - other.array)

    def minimum(self, other: Counts) -> "StringCounts":
        return StringCounts(np.minimum(self.array, other.array))

    def maximum(self, other: Counts) -> "StringCounts":
        return StringCounts(np.maximum(self.array, other.array))

    def __repr__(self) -> str:
        return f"StringCounts({self.array.tolist()})"


class StringPositions(Positions):
    """The positions of one string, which build the values of terminal rules (see `StringReferent` in `meaning.py`).

    Attributes:
        string: the string
    """

    def __init__(self, string: str):
        self.string = string

    def symbol_is(self, symbol: str) -> StringBooleans:
        return StringBooleans(
            np.array([character == symbol for character in self.string], dtype=bool)
        )

    def constant_booleans(self, value: bool) -> StringBooleans:
        return StringBooleans(np.full(len(self.string), value, dtype=bool))

    def constant_counts(self, value: int) -> StringCounts:
        return StringCounts(np.full(len(self.string), value, dtype=np.int64))
