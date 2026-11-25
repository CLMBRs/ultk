import sys
from pathlib import Path

import numpy as np

# the C-RASP example lives in src/examples, which is not an installed package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from crasp.grammar import build_crasp_grammar  # noqa: E402
from crasp.meaning import StringReferent, build_string_universe  # noqa: E402
from crasp.representations import (  # noqa: E402
    BasicCRASPRepresentation,
    PackedCRASPRepresentation,
    PrefixRepresentation,
)
from crasp.representations.basic import (  # noqa: E402
    BasicAcceptance,
    BasicPrefixValues,
)
from crasp.representations.packed import (  # noqa: E402
    PackedAcceptance,
    PackedPrefixValues,
    _smallest_int,
)
from crasp.values import StringValues  # noqa: E402

from ultk.language.grammar.enumeration import SemanticEnumerator  # noqa: E402
from ultk.language.semantics import Universe  # noqa: E402


def is_dyck(string: str) -> bool:
    return all(
        string[:end].count("a") >= string[:end].count("b")
        for end in range(1, len(string) + 1)
    ) and string.count("a") == string.count("b")


def prefix_values(representation: PrefixRepresentation, array: np.ndarray):
    """Build the values at every prefix for a representation, from one value per prefix."""
    tree = representation.tree
    if isinstance(representation, PackedCRASPRepresentation):
        stored = np.packbits(array) if array.dtype == bool else _smallest_int(array)
        return PackedPrefixValues(stored, tree)
    return BasicPrefixValues(array, tree)


def canonical(representation: PrefixRepresentation, value):
    """Rebuild a value from what it represents, so that it is stored canonically: equal meanings must be stored with
    equal bytes, or the enumerator would treat them as different meanings."""
    if isinstance(value, BasicAcceptance):
        return BasicAcceptance(np.array(value.per_string()))
    if isinstance(value, PackedAcceptance):
        accepted = np.array(value.per_string())
        return PackedAcceptance(np.packbits(accepted), len(accepted))
    if isinstance(value, PackedPrefixValues):
        array = value._booleans() if value.is_boolean else value.array.astype(np.int64)
        return prefix_values(representation, array)
    return prefix_values(representation, value.array)


class TestCRASPExample:
    grammar = build_crasp_grammar(("a", "b"))
    universe = build_string_universe(("a", "b"), 4)
    # a universe which is missing many strings and prefixes
    incomplete_universe = Universe(
        tuple(
            StringReferent(name) for name in ("b", "ab", "bba", "aab", "abab", "bbbb")
        )
    )

    # Example 3.2 of Yang & Chiang (2024), with a = "(" and b = ")",
    # writing C(i) = 0 as not(1 <= C(i)) and C1(i) < C2(i) as not(C2(i) <= C1(i))
    dyck_program = (
        "accept(and(not(<=(1, #(not(<=(#(Q_b), #(Q_a)))))), "
        "and(<=(#(Q_a), #(Q_b)), <=(#(Q_b), #(Q_a)))))"
    )

    def test_program_semantics(self):
        meaning = self.grammar.parse(self.dyck_program).evaluate(self.universe)
        assert meaning.mapping == tuple(
            is_dyck(referent.name) for referent in self.universe.referents
        )

    def test_rules_agree_across_values(self):
        """Every rule gives the same values for each string, whether applied to the values for one string at a time
        (as in `GrammaticalExpression.evaluate`), or to the values at every prefix of each representation, and each
        representation stores its results canonically.
        """
        rng = np.random.default_rng(0)
        for universe in (self.universe, self.incomplete_universe):
            for representation_class in (
                BasicCRASPRepresentation,
                PackedCRASPRepresentation,
            ):
                representation = representation_class(universe)
                tree = representation.tree
                for rule in self.grammar.get_all_rules():
                    if rule.rhs is None:
                        expected = tuple(
                            rule.func(referent) for referent in universe.referents
                        )
                        value = representation.evaluate_terminal(rule)
                        assert value.per_string() == expected, rule.name
                        assert value == canonical(representation, value), rule.name
                        continue
                    for _ in range(20):
                        # random values at every prefix: Booleans, or counts that can be negative and large
                        arrays = [
                            (
                                rng.random(len(tree)) < 0.5
                                if child_lhs == "P"
                                else rng.integers(-300, 300, len(tree))
                            )
                            for child_lhs in rule.rhs
                        ]
                        per_string_children = [
                            [
                                StringValues(array[indices])
                                for indices in tree.position_indices
                            ]
                            for array in arrays
                        ]
                        expected = tuple(
                            rule.func(*children)
                            for children in zip(*per_string_children)
                        )
                        value = rule.func(
                            *(prefix_values(representation, array) for array in arrays)
                        )
                        assert value.per_string() == expected, rule.name
                        assert value == canonical(representation, value), rule.name

    def test_representations_agree_with_default(self):
        for universe in (self.universe, self.incomplete_universe):
            default = SemanticEnumerator(self.grammar, universe)
            default.run(9)
            default_programs = {
                meaning: str(expression)
                for meaning, expression in default.get_expressions().items()
            }
            for representation_class in (
                BasicCRASPRepresentation,
                PackedCRASPRepresentation,
            ):
                vectorized = SemanticEnumerator(
                    self.grammar,
                    universe,
                    representation=representation_class(universe),
                )
                vectorized.run(9)
                for lhs in (bool, "P", "C"):
                    assert default.get_costs(lhs) == vectorized.get_costs(lhs)
                assert {
                    meaning: str(expression)
                    for meaning, expression in vectorized.get_expressions().items()
                } == default_programs

    def test_minimal_program(self):
        # a*b*, from Appendix A.2 of Yang & Chiang (2024), writing C(i) = 0 as not(1 <= C(i))
        paper_program = self.grammar.parse(
            "accept(not(<=(1, #(and(Q_a, <=(1, #(Q_b)))))))"
        )
        meaning = paper_program.evaluate(self.universe)
        assert meaning.mapping == tuple(
            "ba" not in referent.name for referent in self.universe.referents
        )
        expressions = self.grammar.get_minimal_expressions(
            self.universe,
            max_cost=len(paper_program),
            representation=PackedCRASPRepresentation(self.universe),
        )
        minimal = expressions[meaning]
        assert len(minimal) <= len(paper_program)
        # the minimal program is a real program with the right meaning
        assert self.grammar.parse(str(minimal)).evaluate(self.universe) == meaning
