from typing import Protocol

from ultk.language.grammar.grammar import Grammar, GrammaticalExpression, Rule
from ultk.language.semantics import Meaning, Referent, Universe


class TestGrammar:
    referents = [Referent(str(num), {"num": num}) for num in range(4)]
    universe = Universe(referents)

    grammar = Grammar(bool)
    grammar.add_rule(Rule(">", bool, (int, int), lambda x, y: x > y))
    grammar.add_rule(Rule("+", int, (int, int), lambda x, y: x + y))
    grammar.add_rule(Rule("n", int, None, lambda model: model.num))
    grammar.add_rule(Rule(f"0", int, None, lambda *args: 0))
    grammar.add_rule(Rule(f"1", int, None, lambda *args: 1))
    grammar.add_rule(Rule(f"2", int, None, lambda *args: 2))
    grammar.add_rule(Rule(f"3", int, None, lambda *args: 3))

    geq2_expr_str = ">(n, +(1, 1))"

    def test_parse(self):
        parsed_expression = TestGrammar.grammar.parse(TestGrammar.geq2_expr_str)
        assert str(parsed_expression) == TestGrammar.geq2_expr_str

    def test_meaning(self):
        parsed_expression = TestGrammar.grammar.parse(TestGrammar.geq2_expr_str)
        expr_meaning = parsed_expression.evaluate(TestGrammar.universe)
        goal_meaning = Meaning(
            tuple(referent.num > 2 for referent in TestGrammar.referents),
            TestGrammar.universe,
        )
        assert expr_meaning == goal_meaning

    def test_length(self):
        parsed_expression = TestGrammar.grammar.parse(TestGrammar.geq2_expr_str)
        assert len(parsed_expression) == 5

    def test_atom_count(self):
        parsed_expression = TestGrammar.grammar.parse(TestGrammar.geq2_expr_str)
        assert parsed_expression.count_atoms() == 3

    def test_yield(self):
        parsed_expression = TestGrammar.grammar.parse(TestGrammar.geq2_expr_str)
        assert parsed_expression.yield_string() == "n11"

    def test_enumerate(self):
        enumed_grammar = TestGrammar.grammar.get_unique_expressions(
            depth=1,
            lhs=(int, int),
            unique_key=lambda expr: expr.evaluate(self.universe),
            compare_func=lambda e1, e2: len(e1) < len(e2),
        )
        print("Enumed Grammar Rules with len")
        print(enumed_grammar)
        for rule in enumed_grammar:
            print(rule)

    def test_evaluate_new_universe(self):
        parsed_expression = TestGrammar.grammar.parse(TestGrammar.geq2_expr_str)
        parsed_expression.evaluate(TestGrammar.universe)
        smaller_universe = Universe(tuple(TestGrammar.referents[2:]))
        meaning = parsed_expression.evaluate(smaller_universe)
        assert meaning.universe is smaller_universe
        assert meaning.mapping == (False, True)

    def test_from_callable_terminals(self):
        def num(referent: Referent) -> int:
            return referent.num

        def double(x: int) -> int:
            return 2 * x

        class HasNum(Protocol):
            num: int

        def num_protocol(thing: HasNum) -> int:
            return thing.num

        assert Rule.from_callable(num).is_terminal()
        assert not Rule.from_callable(double).is_terminal()
        # by default, only functions of a Referent are terminals
        assert not Rule.from_callable(num_protocol).is_terminal()
        assert Rule.from_callable(num_protocol, terminal_types=(HasNum,)).is_terminal()

    def test_constant_rules(self):
        # a function of no arguments is a constant: a terminal which ignores the referent
        def num(referent: Referent) -> int:
            return referent.num

        def one() -> int:
            return 1

        def plus(x: int, y: int) -> int:
            return x + y

        grammar = Grammar(int)
        for func in (num, one, plus):
            grammar.add_rule(Rule.from_callable(func))
        assert grammar._rules_by_name["one"].is_terminal()

        expected = tuple(referent.num + 1 for referent in TestGrammar.referents)
        assert (
            grammar.parse("plus(num, one)").evaluate(TestGrammar.universe).mapping
            == expected
        )
        minimal = grammar.get_minimal_expressions(TestGrammar.universe, max_cost=3)
        assert {str(expression) for expression in minimal.values()} >= {
            "one",
            "plus(num, one)",
        }
        unique = grammar.get_unique_expressions(
            2,
            unique_key=lambda expression: expression.evaluate(TestGrammar.universe),
            compare_func=lambda e1, e2: len(e1) < len(e2),
        )
        assert {str(expression) for expression in unique.values()} >= {
            "one",
            "plus(num, one)",
        }
