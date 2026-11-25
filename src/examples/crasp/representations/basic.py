"""The basic `PrefixRepresentation` for C-RASP: values at every prefix as plain numpy arrays (`bool` or `int64`)."""

import numpy as np

from ultk.language.semantics import Universe

from ..values import Booleans, Counts, HashableArray, StringBooleans, StringCounts
from .prefix import PrefixPositions, PrefixRepresentation, PrefixTree


class BasicAcceptance(HashableArray):
    """Whether a program accepts each string in a universe.

    Attributes:
        array: one `bool` per string
    """

    __slots__ = ()

    def per_string(self) -> tuple[bool, ...]:
        return tuple(self.array.tolist())


class BasicPrefixBooleans(Booleans):
    """The values of a Boolean-valued C-RASP operation at every prefix of the strings in a universe.

    Attributes:
        array: one `bool` per prefix (in the order of `tree.prefixes`)
        tree: the prefixes
    """

    __slots__ = ("tree",)

    def __init__(self, array: np.ndarray, tree: PrefixTree):
        super().__init__(array)
        self.tree = tree

    def __invert__(self) -> "BasicPrefixBooleans":
        return BasicPrefixBooleans(np.logical_not(self.array), self.tree)

    def __and__(self, other: Booleans) -> "BasicPrefixBooleans":
        return BasicPrefixBooleans(np.logical_and(self.array, other.array), self.tree)

    def where(self, if_true: Counts, if_false: Counts) -> "BasicPrefixCounts":
        return BasicPrefixCounts(
            np.where(self.array, if_true.array, if_false.array), self.tree
        )

    def count_so_far(self) -> "BasicPrefixCounts":
        return BasicPrefixCounts(self.tree.count(self.array), self.tree)

    def at_end(self) -> BasicAcceptance:
        return BasicAcceptance(self.array[self.tree.string_indices])

    def per_string(self) -> tuple[StringBooleans, ...]:
        """The values at each position of each string in the universe."""
        return tuple(
            StringBooleans(self.array[indices])
            for indices in self.tree.position_indices
        )


class BasicPrefixCounts(Counts):
    """The values of a count-valued C-RASP operation at every prefix of the strings in a universe.

    Attributes:
        array: one `int64` per prefix (in the order of `tree.prefixes`)
        tree: the prefixes
    """

    __slots__ = ("tree",)

    def __init__(self, array: np.ndarray, tree: PrefixTree):
        super().__init__(array.astype(np.int64, copy=False))
        self.tree = tree

    def __le__(self, other: Counts) -> BasicPrefixBooleans:
        return BasicPrefixBooleans(self.array <= other.array, self.tree)

    def __add__(self, other: Counts) -> "BasicPrefixCounts":
        return BasicPrefixCounts(self.array + other.array, self.tree)

    def __sub__(self, other: Counts) -> "BasicPrefixCounts":
        return BasicPrefixCounts(self.array - other.array, self.tree)

    def minimum(self, other: Counts) -> "BasicPrefixCounts":
        return BasicPrefixCounts(np.minimum(self.array, other.array), self.tree)

    def maximum(self, other: Counts) -> "BasicPrefixCounts":
        return BasicPrefixCounts(np.maximum(self.array, other.array), self.tree)

    def per_string(self) -> tuple[StringCounts, ...]:
        """The values at each position of each string in the universe."""
        return tuple(
            StringCounts(self.array[indices]) for indices in self.tree.position_indices
        )


class BasicPrefixPositions(PrefixPositions):
    """Builds the values of terminal rules as `BasicPrefixBooleans` and `BasicPrefixCounts`."""

    def symbol_is(self, symbol: str) -> BasicPrefixBooleans:
        return BasicPrefixBooleans(self.tree.last_symbols == symbol, self.tree)

    def constant_booleans(self, value: bool) -> BasicPrefixBooleans:
        return BasicPrefixBooleans(
            np.full(len(self.tree), value, dtype=bool), self.tree
        )

    def constant_counts(self, value: int) -> BasicPrefixCounts:
        return BasicPrefixCounts(
            np.full(len(self.tree), value, dtype=np.int64), self.tree
        )


class BasicCRASPRepresentation(PrefixRepresentation):
    """Represents the meanings of C-RASP operations by their values at every prefix, as plain numpy arrays."""

    def __init__(self, universe: Universe):
        """Create a BasicCRASPRepresentation.

        Args:
            universe: a universe whose referents are named by non-empty strings (see `meaning.py`)
        """
        super().__init__(universe, BasicPrefixPositions(PrefixTree(universe)))
