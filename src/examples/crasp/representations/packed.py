"""A compact `PrefixRepresentation` for C-RASP, for enumerating over large universes of strings.

Like `basic.py`, this stores the values of an operation at every prefix as a numpy array, but uses less memory:
- Boolean values are bit-packed: one bit per prefix (or per string) instead of one byte.
- Counts use the smallest integer type that holds them (`int8`, `int16` or `int32`).  The type is chosen from the values
    themselves, so the same meaning is always stored with the same bytes.  Arithmetic is done in a type wide enough
    that it cannot overflow before converting, so a result too large for `int32` raises an error instead of silently
    overflowing.

Memory is what limits enumeration over large universes (see `BENCHMARKS.md`), and almost all of it is used by the
values of count-valued operations, so this makes much larger universes or programs feasible.
"""

from collections.abc import Callable

import numpy as np

from ultk.language.semantics import Universe

from ..values import Booleans, Counts, HashableArray, StringBooleans, StringCounts
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


class PackedPrefixBooleans(Booleans):
    """The values of a Boolean-valued C-RASP operation at every prefix of the strings in a universe, bit-packed.

    Attributes:
        array: one bit per prefix (in the order of `tree.prefixes`), packed into `uint8`s
        tree: the prefixes
    """

    __slots__ = ("tree",)

    def __init__(self, array: np.ndarray, tree: PrefixTree):
        super().__init__(array)
        self.tree = tree

    @classmethod
    def pack(cls, values: np.ndarray, tree: PrefixTree) -> "PackedPrefixBooleans":
        """Build from one unpacked `bool` per prefix."""
        return cls(np.packbits(values), tree)

    def unpack(self) -> np.ndarray:
        """One `bool` per prefix."""
        return np.unpackbits(self.array, count=len(self.tree)).astype(bool)

    def __invert__(self) -> "PackedPrefixBooleans":
        # negate the unpacked values, so that the padding bits in the last byte stay 0
        return PackedPrefixBooleans.pack(np.logical_not(self.unpack()), self.tree)

    def __and__(self, other: Booleans) -> "PackedPrefixBooleans":
        return PackedPrefixBooleans(np.bitwise_and(self.array, other.array), self.tree)

    def where(self, if_true: Counts, if_false: Counts) -> "PackedPrefixCounts":
        return PackedPrefixCounts.from_integers(
            np.where(
                self.unpack(),
                if_true.array.astype(np.int64),
                if_false.array.astype(np.int64),
            ),
            self.tree,
        )

    def count_so_far(self) -> "PackedPrefixCounts":
        return PackedPrefixCounts.from_integers(
            self.tree.count(self.unpack()), self.tree
        )

    def at_end(self) -> PackedAcceptance:
        accepted = self.unpack()[self.tree.string_indices]
        return PackedAcceptance(np.packbits(accepted), len(accepted))

    def per_string(self) -> tuple[StringBooleans, ...]:
        """The values at each position of each string in the universe."""
        values = self.unpack()
        return tuple(
            StringBooleans(values[indices]) for indices in self.tree.position_indices
        )


class PackedPrefixCounts(Counts):
    """The values of a count-valued C-RASP operation at every prefix of the strings in a universe, in the smallest
    integer type that holds them.

    Attributes:
        array: one count per prefix (in the order of `tree.prefixes`), as `int8`, `int16` or `int32`
        tree: the prefixes
    """

    __slots__ = ("tree",)

    def __init__(self, array: np.ndarray, tree: PrefixTree):
        super().__init__(array)
        self.tree = tree

    @classmethod
    def from_integers(
        cls, values: np.ndarray, tree: PrefixTree
    ) -> "PackedPrefixCounts":
        """Build from counts in any integer type, converting them to the smallest type that holds them."""
        return cls(_smallest_int(values), tree)

    def _arithmetic(
        self, operation: Callable[[np.ndarray, np.ndarray], np.ndarray], other: Counts
    ) -> "PackedPrefixCounts":
        # Compute in a type wide enough that the result cannot overflow, so that a result out of the int32 range is
        # caught by `_smallest_int`.  Values stored as int8 or int16 can be added or subtracted in int32 (the result has
        # magnitude at most 2 * 32,768), which is faster; only an int32 operand needs int64.
        wide = (
            np.int64 if np.int32 in (self.array.dtype, other.array.dtype) else np.int32
        )
        return PackedPrefixCounts.from_integers(
            operation(self.array.astype(wide), other.array.astype(wide)), self.tree
        )

    def __le__(self, other: Counts) -> PackedPrefixBooleans:
        return PackedPrefixBooleans.pack(self.array <= other.array, self.tree)

    def __add__(self, other: Counts) -> "PackedPrefixCounts":
        return self._arithmetic(np.add, other)

    def __sub__(self, other: Counts) -> "PackedPrefixCounts":
        return self._arithmetic(np.subtract, other)

    def minimum(self, other: Counts) -> "PackedPrefixCounts":
        return self._arithmetic(np.minimum, other)

    def maximum(self, other: Counts) -> "PackedPrefixCounts":
        return self._arithmetic(np.maximum, other)

    def per_string(self) -> tuple[StringCounts, ...]:
        """The values at each position of each string in the universe."""
        return tuple(
            StringCounts(self.array[indices]) for indices in self.tree.position_indices
        )


class PackedPrefixPositions(PrefixPositions):
    """Builds the values of terminal rules as `PackedPrefixBooleans` and `PackedPrefixCounts`."""

    def symbol_is(self, symbol: str) -> PackedPrefixBooleans:
        return PackedPrefixBooleans.pack(self.tree.last_symbols == symbol, self.tree)

    def constant_booleans(self, value: bool) -> PackedPrefixBooleans:
        return PackedPrefixBooleans.pack(
            np.full(len(self.tree), value, dtype=bool), self.tree
        )

    def constant_counts(self, value: int) -> PackedPrefixCounts:
        return PackedPrefixCounts.from_integers(
            np.full(len(self.tree), value, dtype=np.int32), self.tree
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
