"""Representations of the meanings of C-RASP operations, for faster enumeration with `SemanticEnumerator`.

Both represent the meaning of an operation by its values at every distinct prefix of the strings in a universe (see
`prefix.py`), computed by the functions of the grammar's rules themselves.  They differ only in how values are stored:

- `basic.BasicCRASPRepresentation`: plain numpy arrays (`bool` or `int64`).
- `packed.PackedCRASPRepresentation`: bit-packed Booleans, and counts in the smallest integer type that holds them,
    which uses much less memory and is faster on large universes.
"""

from .basic import BasicCRASPRepresentation
from .packed import PackedCRASPRepresentation
from .prefix import PrefixRepresentation, PrefixTree

REPRESENTATIONS = {
    "basic": BasicCRASPRepresentation,
    "packed": PackedCRASPRepresentation,
}
