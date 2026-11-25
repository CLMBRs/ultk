import sys
from pathlib import Path

# the C-RASP example lives in src/examples, which is not an installed package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from crasp.grammar import build_crasp_grammar  # noqa: E402
from crasp.meaning import build_string_universe  # noqa: E402
from crasp.representations import (  # noqa: E402
    BasicCRASPRepresentation,
    PackedCRASPRepresentation,
)

from ultk.language.grammar.enumeration import SemanticEnumerator  # noqa: E402
from ultk.language.semantics import Referent, Universe  # noqa: E402


def is_dyck(string: str) -> bool:
    return all(
        string[:end].count("a") >= string[:end].count("b")
        for end in range(1, len(string) + 1)
    ) and string.count("a") == string.count("b")


class TestCRASPExample:
    grammar = build_crasp_grammar(("a", "b"))
    universe = build_string_universe(("a", "b"), 4)

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

    def test_representation_agrees_with_default(self):
        default = SemanticEnumerator(self.grammar, self.universe)
        vectorized = SemanticEnumerator(
            self.grammar,
            self.universe,
            representation=BasicCRASPRepresentation(self.universe),
        )
        default.run(9)
        vectorized.run(9)
        for lhs in (bool, "P", "C"):
            assert default.get_costs(lhs) == vectorized.get_costs(lhs)

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
            representation=BasicCRASPRepresentation(self.universe),
        )
        minimal = expressions[meaning]
        assert len(minimal) <= len(paper_program)
        # the minimal program is a real program with the right meaning
        assert self.grammar.parse(str(minimal)).evaluate(self.universe) == meaning

    def test_packed_representation_agrees_with_basic(self):
        # a universe with every string up to length 4, and one which is missing many strings and prefixes
        incomplete = Universe(
            tuple(Referent(name) for name in ("b", "ab", "bba", "aab", "abab", "bbbb"))
        )
        for universe in (self.universe, incomplete):
            basic = SemanticEnumerator(
                self.grammar,
                universe,
                representation=BasicCRASPRepresentation(universe),
            )
            packed = SemanticEnumerator(
                self.grammar,
                universe,
                representation=PackedCRASPRepresentation(universe),
            )
            basic.run(9)
            packed.run(9)
            for lhs in (bool, "P", "C"):
                assert basic.get_costs(lhs) == packed.get_costs(lhs)
            assert {
                meaning: str(expression)
                for meaning, expression in basic.get_expressions().items()
            } == {
                meaning: str(expression)
                for meaning, expression in packed.get_expressions().items()
            }
