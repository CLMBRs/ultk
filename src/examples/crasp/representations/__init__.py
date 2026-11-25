"""Representations of the meanings of C-RASP operations, for faster enumeration with `SemanticEnumerator`.

- `basic.BasicCRASPRepresentation`: one value per distinct prefix of the strings in the universe, as a numpy array.
- `packed.PackedCRASPRepresentation`: the same, but with bit-packed Booleans and the smallest integer types for counts,
    which uses much less memory and is faster on large universes.
"""

from .basic import BasicCRASPRepresentation, PrefixValue
from .packed import PackedCRASPRepresentation

REPRESENTATIONS = {
    "basic": BasicCRASPRepresentation,
    "packed": PackedCRASPRepresentation,
}
