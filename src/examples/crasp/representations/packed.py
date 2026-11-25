"""A compact `PrefixRepresentation` for C-RASP, for enumerating over large universes of strings.

Like `basic.py`, this stores the values of an operation at every prefix as a numpy array, but uses less memory:
- Boolean values are bit-packed: one bit per prefix (or per string) instead of one byte.
- Counts use the smallest integer type that holds them (`int8`, `int16` or `int32`).  The type is chosen from the values
    themselves, so the same meaning is always stored with the same bytes.  Arithmetic is done in `int32` before
    converting, so it cannot overflow.

Memory is what limits enumeration over large universes (see `BENCHMARKS.md`), and almost all of it is used by the
values of count-valued operations, so this makes much larger universes or programs feasible.
"""

import numpy as np

from ultk.language.semantics import Universe

from ..values import HashableArray, PositionValues, StringValues
from .prefix import PrefixPositions, PrefixRepresentation, PrefixTree

_INT_TYPES = (np.int8, np.int16, np.int32)


def _smallest_int(values: np.ndarray) -> np.ndarray:
    """Convert integers to the smallest integer type that holds all of them.

    Args:
        values: an integer array

    Returns:
        the same values, as `int8`, `int16` or `int32`
    """
    low, high = int(values.min()), int(values.max())
    for dtype in _INT_TYPES:
        info = np.iinfo(dtype)
        if info.min <= low and high <= info.max:
            return values.astype(dtype, copy=False)
    raise OverflowError(f"C-RASP count values out of int32 range: [{low}, {high}]")


class PackedAcceptance(HashableArray):
    """Whether a program accepts each string in a universe, bit-packed.

    Attributes:
        array: one bit per string, packed into `uint8`s
        num_strings: the number of strings
    """

    __slots__ = ("num_strings",)

    def __init__(self, array: np.ndarray, num_strings: int):
        super().__init__(array)
        self.num_strings = num_strings

    def per_string(self) -> tuple[bool, ...]:
        return tuple(
            np.unpackbits(self.array, count=self.num_strings).astype(bool).tolist()
        )


class PackedPrefixValues(PositionValues):
    """The values of a C-RASP operation at every prefix of the strings in a universe, stored compactly.

    Boolean values are bit-packed into `uint8`s, and counts are stored as `int8`, `int16` or `int32`, so the two can
    be told apart by their dtype.

    Attributes:
        array: the values at every prefix (in the order of `tree.prefixes`)
        tree: the prefixes
    """

    __slots__ = ("tree",)

    def __init__(self, array: np.ndarray, tree: PrefixTree):
        super().__init__(array)
        self.tree = tree

    @property
    def is_boolean(self) -> bool:
        return self.array.dtype == np.uint8

    def _booleans(self) -> np.ndarray:
        """The Boolean values, one per prefix, unpacked."""
        return np.unpackbits(self.array, count=len(self.tree)).astype(bool)

    def _from_booleans(self, values: np.ndarray) -> "PackedPrefixValues":
        return PackedPrefixValues(np.packbits(values), self.tree)

    def _from_counts(self, values: np.ndarray) -> "PackedPrefixValues":
        return PackedPrefixValues(_smallest_int(values), self.tree)

    def _arithmetic(
        self, operation, other: "PackedPrefixValues"
    ) -> "PackedPrefixValues":
        # in int32, so that the result cannot overflow before its type is chosen
        return self._from_counts(
            operation(self.array.astype(np.int32), other.array.astype(np.int32))
        )

    def __invert__(self) -> "PackedPrefixValues":
        # negate the unpacked values, so that the padding bits in the last byte stay 0
        return self._from_booleans(np.logical_not(self._booleans()))

    def __and__(self, other: "PackedPrefixValues") -> "PackedPrefixValues":
        return PackedPrefixValues(np.bitwise_and(self.array, other.array), self.tree)

    def __le__(self, other: "PackedPrefixValues") -> "PackedPrefixValues":
        return self._from_booleans(self.array <= other.array)

    def __add__(self, other: "PackedPrefixValues") -> "PackedPrefixValues":
        return self._arithmetic(np.add, other)

    def __sub__(self, other: "PackedPrefixValues") -> "PackedPrefixValues":
        return self._arithmetic(np.subtract, other)

    def minimum(self, other: "PackedPrefixValues") -> "PackedPrefixValues":
        return self._arithmetic(np.minimum, other)

    def maximum(self, other: "PackedPrefixValues") -> "PackedPrefixValues":
        return self._arithmetic(np.maximum, other)

    def where(
        self, if_true: "PackedPrefixValues", if_false: "PackedPrefixValues"
    ) -> "PackedPrefixValues":
        return self._from_counts(
            np.where(
                self._booleans(),
                if_true.array.astype(np.int32),
                if_false.array.astype(np.int32),
            )
        )

    def count_so_far(self) -> "PackedPrefixValues":
        return self._from_counts(self.tree.count(self._booleans()))

    def at_end(self) -> PackedAcceptance:
        accepted = self._booleans()[self.tree.string_indices]
        return PackedAcceptance(np.packbits(accepted), len(accepted))

    def per_string(self) -> tuple[StringValues, ...]:
        """The values at each position of each string in the universe."""
        values = self._booleans() if self.is_boolean else self.array
        return tuple(
            StringValues(values[indices]) for indices in self.tree.position_indices
        )


class PackedPrefixPositions(PrefixPositions):
    """Builds the values of terminal rules as `PackedPrefixValues`."""

    def symbol_is(self, symbol: str) -> PackedPrefixValues:
        return PackedPrefixValues(
            np.packbits(self.tree.last_symbols == symbol), self.tree
        )

    def constant(self, value: bool | int) -> PackedPrefixValues:
        if isinstance(value, bool):
            return PackedPrefixValues(
                np.packbits(np.full(len(self.tree), value)), self.tree
            )
        return PackedPrefixValues(
            _smallest_int(np.full(len(self.tree), value, dtype=np.int32)), self.tree
        )


class PackedCRASPRepresentation(PrefixRepresentation):
    """Represents the meanings of C-RASP operations by their values at every prefix, with bit-packed Booleans and
    counts in the smallest integer type that holds them."""

    def __init__(self, universe: Universe):
        """Create a PackedCRASPRepresentation.

        Args:
            universe: a universe whose referents are named by non-empty strings (see `meaning.py`)
        """
        super().__init__(universe, PackedPrefixPositions(PrefixTree(universe)))
