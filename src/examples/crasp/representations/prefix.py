"""Representing the meanings of C-RASP operations by their values at every prefix of the strings in a universe.

Because C-RASP operations only look at the current and earlier positions, the value of an operation at position `i` of
a string `w` is its value at the last position of the prefix `w[:i]`.  So instead of storing a value for every position
of every string, we store one value for every distinct prefix of a string in the universe.  The prefixes form a tree,
and counting `#[j <= i] P(j)` at a prefix is the value of `P` there plus the count at its parent.

The rules of the grammar (see `grammar.py`) compute these values directly: values at every prefix implement the same
operations (`values.PositionValues`) as the values for one string, and terminal rules are given `AllStrings`, whose
`positions` build values at every prefix.  So `PrefixRepresentation` needs no knowledge of the individual rules.
Subclasses only choose how values are stored: see `basic.py` and `packed.py`.
"""

from typing import Any, Sequence

import numpy as np

from ultk.language.grammar.enumeration import MeaningRepresentation
from ultk.language.grammar.grammar import Rule
from ultk.language.semantics import Meaning, Universe

from ..values import HashableArray, Positions


class PrefixTree:
    """The distinct prefixes of the strings in a universe, and how they relate.

    Attributes:
        prefixes: the distinct non-empty prefixes of the strings in the universe, ordered by length and then
            lexicographically
        last_symbols: the last symbol of each prefix
        string_indices: for each string in the universe, the index of the whole string among the prefixes
        position_indices: for each string in the universe, the index of the prefix ending at each of its positions
    """

    def __init__(self, universe: Universe):
        """Create a PrefixTree.

        Args:
            universe: a universe whose referents are named by non-empty strings (see `meaning.py`)
        """
        strings = [referent.name for referent in universe.referents]
        if not all(strings):
            raise ValueError("C-RASP programs are only defined on non-empty strings.")
        self.prefixes: tuple[str, ...] = tuple(
            sorted(
                {
                    string[:end]
                    for string in strings
                    for end in range(1, len(string) + 1)
                },
                key=lambda prefix: (len(prefix), prefix),
            )
        )
        index = {prefix: idx for idx, prefix in enumerate(self.prefixes)}
        self.last_symbols = np.array([prefix[-1] for prefix in self.prefixes])
        self.string_indices = np.array([index[string] for string in strings])
        self.position_indices: list[list[int]] = [
            [index[string[:end]] for end in range(1, len(string) + 1)]
            for string in strings
        ]
        # prefixes are ordered by length, so those of each length form a block, starting at these indices
        max_length = len(self.prefixes[-1])
        lengths = np.array([len(prefix) for prefix in self.prefixes])
        self._length_starts = np.searchsorted(lengths, np.arange(1, max_length + 2))
        self._parents = np.array(
            [index.get(prefix[:-1], -1) for prefix in self.prefixes]
        )
        # if the universe has every string up to its maximum length, the children of the i-th prefix of one length are
        # the next `alphabet_size` prefixes of the next length, which makes counting much faster
        self._alphabet_size = len(set(self.last_symbols.tolist()))
        self._is_complete = len(self.prefixes) == sum(
            self._alphabet_size**length for length in range(1, max_length + 1)
        )

    def __len__(self) -> int:
        return len(self.prefixes)

    def count(self, values: np.ndarray) -> np.ndarray:
        """At each prefix, count the positions of the prefix at which a Boolean value is true.

        Args:
            values: one Boolean per prefix

        Returns:
            one `int32` count per prefix
        """
        counts = values.astype(np.int32)
        # prefixes are processed in order of length, so each parent's count is final before it is used
        for start, next_start, end in zip(
            self._length_starts[:-1], self._length_starts[1:], self._length_starts[2:]
        ):
            if self._is_complete:
                counts[next_start:end] += np.repeat(
                    counts[start:next_start], self._alphabet_size
                )
            else:
                counts[next_start:end] += counts[self._parents[next_start:end]]
        return counts


class PrefixPositions(Positions):
    """The prefixes of the strings in a universe, as `Positions`: they build the values of terminal rules at every
    prefix.

    Subclasses implement `symbol_is` and `constant`, which determine how values are stored (see `basic.py` and
    `packed.py`).

    Attributes:
        tree: the prefixes
    """

    def __init__(self, tree: PrefixTree):
        self.tree = tree


class AllStrings:
    """A stand-in for all strings in a universe at once, which terminal rules are applied to.

    Like a `meaning.StringReferent`, it is a `values.HasPositions`: its `positions` build the values of terminal rules,
    here at every prefix of every string.

    Attributes:
        positions: the prefixes of all the strings
    """

    def __init__(self, positions: PrefixPositions):
        self.positions = positions


class PrefixRepresentation(MeaningRepresentation[HashableArray]):
    """Represents the meanings of C-RASP operations by their values at every prefix of the strings in a universe.

    Values are computed by the functions of the grammar's rules themselves: terminal rules are applied to `AllStrings`,
    and other rules to the values of their children.  The value of an expression of type `"P"` or `"C"` has one entry
    per prefix, and the value of an expression of type `bool` (a whole program) has one entry per string.

    Values must implement `per_string`, which converts them back to one value per string in the universe.
    """

    def __init__(self, universe: Universe, positions: PrefixPositions):
        """Create a PrefixRepresentation.

        Args:
            universe: a universe whose referents are named by non-empty strings (see `meaning.py`)
            positions: the prefixes of the strings in `universe`, which determine how values are stored
        """
        self.universe = universe
        self.all_strings = AllStrings(positions)

    @property
    def tree(self) -> PrefixTree:
        return self.all_strings.positions.tree

    def evaluate_terminal(self, rule: Rule) -> HashableArray:
        return rule.func(self.all_strings)

    def apply_rule(
        self, rule: Rule, children: Sequence[HashableArray]
    ) -> HashableArray:
        return rule.func(*children)

    def to_meaning(self, value: Any, lhs: Any) -> Meaning:
        return Meaning(value.per_string(), self.universe)
