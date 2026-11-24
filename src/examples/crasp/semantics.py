"""A vectorized `Semantics` for C-RASP, for fast enumeration.

The default `PointwiseSemantics` represents the value of a C-RASP operation as a tuple (over strings) of
tuples (over positions), and applies each rule's Python function string by string.  Here, instead, the value of
an operation is a single numpy array, and each rule is one numpy operation.

Because C-RASP operations only look at positions up to the current one, the value of an operation at position `i`
of a string `w` is its value at the last position of the prefix `w[:i]`.  So instead of storing a value for every
position of every string, we store one value for every distinct prefix of a string in the universe.  The prefixes form
a tree (a trie), and counting `#[j <= i] P(j)` at a prefix is the value of `P` there plus the count at its parent.
For a universe of all strings up to some length, this stores roughly `max_length` times fewer numbers than storing
every position of every string, without changing which operations are distinguished.
"""

from typing import Any, Sequence

import numpy as np

from ultk.language.grammar.enumeration import Semantics
from ultk.language.grammar.grammar import Rule
from ultk.language.semantics import Meaning, Universe


class PrefixValue:
    """The value of a C-RASP operation at every prefix: a numpy array, made hashable so it can be its own key.

    Attributes:
        array: one value per prefix, in the order of `CRASPSemantics.prefixes`
    """

    __slots__ = ("array", "_hash")

    def __init__(self, array: np.ndarray):
        self.array = array
        self._hash = hash(array.tobytes())

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, PrefixValue)
            and self._hash == other._hash
            and np.array_equal(self.array, other.array)
        )


class CRASPSemantics(Semantics[PrefixValue]):
    """Computes the values of C-RASP operations at every prefix of every string in a universe.

    The value of an expression of type `"P"` is a boolean array, and of type `"C"` an integer array, each with
    one entry per prefix.  The value of an expression of type `bool` (a whole program) has one entry per string
    in the universe, saying whether the program accepts it.

    Attributes:
        prefixes: the distinct non-empty prefixes of the strings in the universe, ordered by length
    """

    def __init__(self, universe: Universe):
        """Create a CRASPSemantics.

        Args:
            universe: a universe whose referents are named by non-empty strings (see `meaning.py`)
        """
        self.universe = universe
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
        self._last_symbols = np.array([prefix[-1] for prefix in self.prefixes])
        # for each length > 1, the prefixes of that length and their parents (the prefix one shorter)
        self._children_and_parents: list[tuple[np.ndarray, np.ndarray]] = []
        for length in range(2, max(len(string) for string in strings) + 1):
            children = [
                idx for idx, prefix in enumerate(self.prefixes) if len(prefix) == length
            ]
            parents = [index[self.prefixes[idx][:-1]] for idx in children]
            self._children_and_parents.append((np.array(children), np.array(parents)))
        # the prefix corresponding to each string in the universe (i.e. the whole string)
        self._string_indices = np.array([index[string] for string in strings])
        # for each string, the prefix at each of its positions
        self._position_indices: list[list[int]] = [
            [index[string[:end]] for end in range(1, len(string) + 1)]
            for string in strings
        ]
        self._operations = {
            "accept": lambda p: p[self._string_indices],
            "not": np.logical_not,
            "and": np.logical_and,
            "<=": np.less_equal,
            "#": self._count,
            "?": np.where,
            "+": np.add,
            "-": np.subtract,
            "min": np.minimum,
            "max": np.maximum,
        }

    def _count(self, p: np.ndarray) -> np.ndarray:
        """At each prefix, count the positions of the prefix at which `p` is true."""
        counts = p.astype(np.int64)
        # prefixes are processed in order of length, so each parent's count is final before it is used
        for children, parents in self._children_and_parents:
            counts[children] += counts[parents]
        return counts

    def evaluate_terminal(self, rule: Rule) -> PrefixValue:
        if rule.name.startswith("Q_"):
            return PrefixValue(self._last_symbols == rule.name[2:])
        if rule.name == "true":
            return PrefixValue(np.ones(len(self.prefixes), dtype=bool))
        if rule.name == "1":
            return PrefixValue(np.ones(len(self.prefixes), dtype=np.int64))
        raise ValueError(f"Unknown C-RASP terminal rule: {rule.name}")

    def apply_rule(self, rule: Rule, children: Sequence[PrefixValue]) -> PrefixValue:
        return PrefixValue(
            self._operations[rule.name](*(child.array for child in children))
        )

    def to_meaning(self, value: PrefixValue, lhs: Any) -> Meaning:
        if lhs is bool:
            return Meaning(tuple(value.array.tolist()), self.universe)
        # for each string, the tuple of values at its positions, as computed by the rules' functions in `grammar.py`
        values = value.array.tolist()
        return Meaning(
            tuple(
                tuple(values[idx] for idx in indices)
                for indices in self._position_indices
            ),
            self.universe,
        )
