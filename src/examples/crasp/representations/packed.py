"""A compact `MeaningRepresentation` for C-RASP, for enumerating over large universes of strings.

This refines `BasicCRASPRepresentation` (one value per distinct prefix, as a numpy array) to use less memory:
- Boolean values (of type `"P"`, and whole programs) are bit-packed: one bit per prefix instead of one byte.
- Count values (of type `"C"`) use the smallest integer type that holds them (`int8`, `int16` or `int32`).  The type
    is chosen from the values themselves, so the same meaning is always stored with the same bytes.  Arithmetic is
    done in `int32` before converting, so it cannot overflow.
- When the universe contains *every* string up to some length, counting `#[j <= i] P(j)` adds each prefix's count to
    its children's in one slice per length, instead of indexing arbitrary parents.

Memory is what limits enumeration over large universes (see `BENCHMARKS.md`), and almost all of it is used by the
values of count-valued operations, so this makes much larger universes or programs feasible.
"""

from typing import Any, Sequence

import numpy as np

from ultk.language.grammar.grammar import Rule
from ultk.language.semantics import Meaning, Universe

from .basic import BasicCRASPRepresentation, PrefixValue

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


class PackedCRASPRepresentation(BasicCRASPRepresentation):
    """Computes the values of C-RASP operations at every prefix, like `BasicCRASPRepresentation`, but stores them
    more compactly: bit-packed Booleans, and counts in the smallest integer type that holds them.
    """

    def __init__(self, universe: Universe):
        """Create a PackedCRASPRepresentation.

        Args:
            universe: a universe whose referents are named by non-empty strings (see `meaning.py`)
        """
        super().__init__(universe)
        self._num_prefixes = len(self.prefixes)
        self._num_strings = len(self._string_indices)
        # bits beyond the last prefix in the last packed byte must stay 0, so that equal meanings have equal bytes
        num_padding_bits = 8 * ((self._num_prefixes + 7) // 8) - self._num_prefixes
        self._padding_mask = np.uint8((0xFF << num_padding_bits) & 0xFF)
        # if the universe has every string up to some length, prefixes of each length are in a contiguous block,
        # and the children of the i-th prefix of one length are the next `alphabet_size` prefixes of the next length
        symbols = sorted({symbol for prefix in self.prefixes for symbol in prefix})
        max_length = len(self.prefixes[-1])
        self._alphabet_size = len(symbols)
        self._is_complete = self._num_prefixes == sum(
            self._alphabet_size**length for length in range(1, max_length + 1)
        )
        self._length_starts = [
            sum(self._alphabet_size**shorter for shorter in range(1, length))
            for length in range(1, max_length + 2)
        ]

    def _unpack_prefixes(self, bits: np.ndarray) -> np.ndarray:
        return np.unpackbits(bits, count=self._num_prefixes).astype(bool)

    def _count(self, p: np.ndarray) -> np.ndarray:
        """At each prefix, count the positions of the prefix at which `p` is true."""
        if not self._is_complete:
            return super()._count(p)
        counts = p.astype(np.int32)
        for start, next_start, end in zip(
            self._length_starts, self._length_starts[1:], self._length_starts[2:]
        ):
            counts[next_start:end] += np.repeat(
                counts[start:next_start], self._alphabet_size
            )
        return counts

    def evaluate_terminal(self, rule: Rule) -> PrefixValue:
        value = super().evaluate_terminal(rule).array
        if value.dtype == bool:
            return PrefixValue(np.packbits(value))
        return PrefixValue(_smallest_int(value))

    def apply_rule(self, rule: Rule, children: Sequence[PrefixValue]) -> PrefixValue:
        arrays = [child.array for child in children]
        if rule.name == "accept":
            return PrefixValue(
                np.packbits(self._unpack_prefixes(arrays[0])[self._string_indices])
            )
        if rule.name == "not":
            negated = np.bitwise_not(arrays[0])
            negated[-1] &= self._padding_mask
            return PrefixValue(negated)
        if rule.name == "and":
            return PrefixValue(np.bitwise_and(arrays[0], arrays[1]))
        if rule.name == "<=":
            return PrefixValue(np.packbits(arrays[0] <= arrays[1]))
        if rule.name == "#":
            return PrefixValue(
                _smallest_int(self._count(self._unpack_prefixes(arrays[0])))
            )
        if rule.name == "?":
            condition = self._unpack_prefixes(arrays[0])
            return PrefixValue(
                _smallest_int(
                    np.where(condition, arrays[1], arrays[2]).astype(np.int32)
                )
            )
        # arithmetic on counts, in int32 so that it cannot overflow
        operation = {
            "+": np.add,
            "-": np.subtract,
            "min": np.minimum,
            "max": np.maximum,
        }[rule.name]
        return PrefixValue(
            _smallest_int(
                operation(arrays[0].astype(np.int32), arrays[1].astype(np.int32))
            )
        )

    def to_meaning(self, value: PrefixValue, lhs: Any) -> Meaning:
        if lhs is bool:
            array = np.unpackbits(value.array, count=self._num_strings).astype(bool)
        elif lhs == "P":
            array = self._unpack_prefixes(value.array)
        else:
            array = value.array.astype(np.int64)
        return super().to_meaning(PrefixValue(array), lhs)
