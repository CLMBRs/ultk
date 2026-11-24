import sys
from pathlib import Path

# the C-RASP example lives in src/examples, which is not an installed package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from crasp.grammar import build_crasp_grammar  # noqa: E402
from crasp.meaning import build_string_universe  # noqa: E402
from crasp.semantics import CRASPSemantics  # noqa: E402

from ultk.language.grammar.enumeration import (  # noqa: E402
    PointwiseSemantics,
    SemanticEnumerator,
)


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

    def test_vectorized_semantics_agree(self):
        pointwise = SemanticEnumerator(self.grammar, PointwiseSemantics(self.universe))
        vectorized = SemanticEnumerator(self.grammar, CRASPSemantics(self.universe))
        pointwise.run(9)
        vectorized.run(9)
        for lhs in (bool, "P", "C"):
            assert pointwise.get_costs(lhs) == vectorized.get_costs(lhs)

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
            semantics=CRASPSemantics(self.universe),
        )
        minimal = expressions[meaning]
        assert len(minimal) <= len(paper_program)
        # the minimal program is a real program with the right meaning
        assert self.grammar.parse(str(minimal)).evaluate(self.universe) == meaning
