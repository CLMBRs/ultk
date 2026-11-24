from collections.abc import Sequence
from itertools import product

from ultk.language.semantics import Referent, Universe

from .grammar import ALPHABET

MAX_LENGTH = 6


def build_string_universe(
    alphabet: Sequence[str] = ALPHABET, max_length: int = MAX_LENGTH
) -> Universe:
    """Build a universe whose referents are all non-empty strings up to a maximum length.

    The empty string is excluded because C-RASP programs accept or reject a string based on the value at
    its last position.  Each referent's name is the string itself.

    Args:
        alphabet: the symbols of the alphabet
        max_length: the maximum length of a string

    Returns:
        a `Universe` of strings, ordered by length and then lexicographically
    """
    return Universe(
        tuple(
            Referent("".join(symbols))
            for length in range(1, max_length + 1)
            for symbols in product(alphabet, repeat=length)
        )
    )


universe = build_string_universe()
