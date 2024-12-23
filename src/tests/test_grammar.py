from ultk.language.grammar import Grammar, GrammaticalExpression, Rule
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
            {referent: referent.num > 2 for referent in TestGrammar.referents},
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
