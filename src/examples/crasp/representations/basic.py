"""The basic `PrefixRepresentation` for C-RASP: values at every prefix as plain numpy arrays (`bool` or `int64`)."""

import numpy as np

from ultk.language.semantics import Universe

from ..values import HashableArray, PositionValues, StringValues
from .prefix import PrefixPositions, PrefixRepresentation, PrefixTree


class BasicAcceptance(HashableArray):
    """Whether a program accepts each string in a universe.

    Attributes:
        array: one `bool` per string
    """

    __slots__ = ()

    def per_string(self) -> tuple[bool, ...]:
        return tuple(self.array.tolist())


class BasicPrefixValues(PositionValues):
    """The values of a C-RASP operation at every prefix of the strings in a universe.

    Attributes:
        array: one value per prefix (in the order of `tree.prefixes`), as `bool` or `int64`
        tree: the prefixes
    """

    __slots__ = ("tree",)

    def __init__(self, array: np.ndarray, tree: PrefixTree):
        super().__init__(array)
        self.tree = tree

    def _new(self, array: np.ndarray) -> "BasicPrefixValues":
        if array.dtype != bool:
            array = array.astype(np.int64, copy=False)
        return BasicPrefixValues(array, self.tree)

    def __invert__(self) -> "BasicPrefixValues":
        return self._new(np.logical_not(self.array))

    def __and__(self, other: "BasicPrefixValues") -> "BasicPrefixValues":
        return self._new(np.logical_and(self.array, other.array))

    def __le__(self, other: "BasicPrefixValues") -> "BasicPrefixValues":
        return self._new(self.array <= other.array)

    def __add__(self, other: "BasicPrefixValues") -> "BasicPrefixValues":
        return self._new(self.array + other.array)

    def __sub__(self, other: "BasicPrefixValues") -> "BasicPrefixValues":
        return self._new(self.array - other.array)

    def minimum(self, other: "BasicPrefixValues") -> "BasicPrefixValues":
        return self._new(np.minimum(self.array, other.array))

    def maximum(self, other: "BasicPrefixValues") -> "BasicPrefixValues":
        return self._new(np.maximum(self.array, other.array))

    def where(
        self, if_true: "BasicPrefixValues", if_false: "BasicPrefixValues"
    ) -> "BasicPrefixValues":
        return self._new(np.where(self.array, if_true.array, if_false.array))

    def count_so_far(self) -> "BasicPrefixValues":
        return self._new(self.tree.count(self.array))

    def at_end(self) -> BasicAcceptance:
        return BasicAcceptance(self.array[self.tree.string_indices])

    def per_string(self) -> tuple[StringValues, ...]:
        """The values at each position of each string in the universe."""
        return tuple(
            StringValues(self.array[indices]) for indices in self.tree.position_indices
        )


class BasicPrefixPositions(PrefixPositions):
    """Builds the values of terminal rules as `BasicPrefixValues`."""

    def symbol_is(self, symbol: str) -> BasicPrefixValues:
        return BasicPrefixValues(self.tree.last_symbols == symbol, self.tree)

    def constant(self, value: bool | int) -> BasicPrefixValues:
        dtype = bool if isinstance(value, bool) else np.int64
        return BasicPrefixValues(np.full(len(self.tree), value, dtype=dtype), self.tree)


class BasicCRASPRepresentation(PrefixRepresentation):
    """Represents the meanings of C-RASP operations by their values at every prefix, as plain numpy arrays."""

    def __init__(self, universe: Universe):
        """Create a BasicCRASPRepresentation.

        Args:
            universe: a universe whose referents are named by non-empty strings (see `meaning.py`)
        """
        super().__init__(universe, BasicPrefixPositions(PrefixTree(universe)))
