from collections import defaultdict
from itertools import product
from typing import Any, Callable

import pytest

from ultk.language.grammar.enumeration import (
    PointwiseSemantics,
    SemanticEnumerator,
    unit_cost,
)
from ultk.language.grammar.grammar import Grammar, Rule
from ultk.language.semantics import Referent, Universe


def brute_force_costs(
    grammar: Grammar,
    universe: Universe,
    max_cost: int,
    cost: Callable[[Rule], int] = unit_cost,
) -> dict[Any, dict[tuple, int]]:
    """Get the minimal cost of every meaning of every LHS, by enumerating _all_ expressions up to `max_cost`,
    without any semantic deduplication."""
    # lhs -> cost -> list of values of all expressions of that cost
    values: dict[Any, dict[int, list[tuple]]] = defaultdict(lambda: defaultdict(list))
    rules = grammar.get_all_rules()
    for total in range(1, max_cost + 1):
        for rule in rules:
            budget = total - cost(rule)
            if rule.rhs is None:
                if budget == 0:
                    values[rule.lhs][total].append(
                        tuple(rule.func(referent) for referent in universe.referents)
                    )
                continue
            for child_costs in product(range(1, budget + 1), repeat=len(rule.rhs)):
                if sum(child_costs) != budget:
                    continue
                pools = [
                    values[child_lhs][child_cost]
                    for child_lhs, child_cost in zip(rule.rhs, child_costs)
                ]
                for children in product(*pools):
                    values[rule.lhs][total].append(tuple(map(rule.func, *children)))
    min_costs: dict[Any, dict[tuple, int]] = defaultdict(dict)
    for lhs in values:
        for total in sorted(values[lhs]):
            for value in values[lhs][total]:
                min_costs[lhs].setdefault(value, total)
    return min_costs


def enumerated_costs(enumerator: SemanticEnumerator, lhs: Any) -> dict[tuple, int]:
    return {state.value: state.cost for state in enumerator.get_states(lhs).values()}


class TestEnumeration:
    # Boolean functions of three variables
    boolean_universe = Universe(
        tuple(
            Referent("".join(str(int(bit)) for bit in bits), {"bits": bits})
            for bits in product((False, True), repeat=3)
        )
    )
    boolean_grammar = Grammar(bool)
    boolean_grammar.add_rule(Rule("and", bool, (bool, bool), lambda p, q: p and q))
    boolean_grammar.add_rule(Rule("or", bool, (bool, bool), lambda p, q: p or q))
    boolean_grammar.add_rule(Rule("not", bool, (bool,), lambda p: not p))
    for idx in range(3):
        boolean_grammar.add_rule(
            Rule(f"x{idx}", bool, None, lambda referent, idx=idx: referent.bits[idx])
        )

    # a grammar with two types, as in test_grammar.py
    numbers_universe = Universe(
        tuple(Referent(str(num), {"num": num}) for num in range(4))
    )
    numbers_grammar = Grammar(bool)
    numbers_grammar.add_rule(Rule(">", bool, (int, int), lambda x, y: x > y))
    numbers_grammar.add_rule(Rule("+", int, (int, int), lambda x, y: x + y))
    numbers_grammar.add_rule(Rule("n", int, None, lambda referent: referent.num))
    for num in range(4):
        numbers_grammar.add_rule(Rule(f"{num}", int, None, lambda *args, num=num: num))

    def test_minimal_costs_boolean(self):
        enumerator = SemanticEnumerator(
            self.boolean_grammar, PointwiseSemantics(self.boolean_universe)
        )
        enumerator.run(7)
        expected = brute_force_costs(self.boolean_grammar, self.boolean_universe, 7)
        assert enumerated_costs(enumerator, bool) == expected[bool]

    def test_minimal_costs_multiple_types(self):
        enumerator = SemanticEnumerator(
            self.numbers_grammar, PointwiseSemantics(self.numbers_universe)
        )
        enumerator.run(7)
        expected = brute_force_costs(self.numbers_grammar, self.numbers_universe, 7)
        for lhs in (bool, int):
            assert enumerated_costs(enumerator, lhs) == expected[lhs]

    def test_minimal_costs_weighted(self):
        def cost(rule: Rule) -> int:
            return {"and": 3, "or": 2, "not": 2}.get(rule.name, 1)

        enumerator = SemanticEnumerator(
            self.boolean_grammar, PointwiseSemantics(self.boolean_universe), cost=cost
        )
        enumerator.run(9)
        expected = brute_force_costs(
            self.boolean_grammar, self.boolean_universe, 9, cost=cost
        )
        assert enumerated_costs(enumerator, bool) == expected[bool]

    def test_expressions_match_meanings(self):
        expressions = self.numbers_grammar.get_minimal_expressions(
            self.numbers_universe, max_cost=7
        )
        for meaning, expression in expressions.items():
            assert len(expression) <= 7
            # re-parse, so that the meaning is computed from scratch
            parsed = self.numbers_grammar.parse(str(expression))
            assert parsed.evaluate(self.numbers_universe) == meaning

    def test_agrees_with_get_unique_expressions(self):
        legacy = self.numbers_grammar.get_unique_expressions(
            3,
            unique_key=lambda expr: expr.evaluate(self.numbers_universe),
            compare_func=lambda e1, e2: len(e1) < len(e2),
        )
        max_length = max(len(expression) for expression in legacy.values())
        minimal = self.numbers_grammar.get_minimal_expressions(
            self.numbers_universe, max_cost=max_length
        )
        for meaning, expression in legacy.items():
            assert len(minimal[meaning]) <= len(expression)

    def test_saturation(self):
        enumerator = SemanticEnumerator(
            self.boolean_grammar, PointwiseSemantics(self.boolean_universe)
        )
        enumerator.run(1000)
        assert enumerator.is_saturated
        assert enumerator.completed_cost < 1000
        # every Boolean function of three variables is expressible
        assert len(enumerator.get_states()) == 2**8

    def test_resume(self):
        semantics = PointwiseSemantics(self.numbers_universe)
        resumed = SemanticEnumerator(self.numbers_grammar, semantics)
        resumed.run(4)
        resumed.run(7)
        fresh = SemanticEnumerator(self.numbers_grammar, semantics)
        fresh.run(7)
        for lhs in (bool, int):
            assert enumerated_costs(resumed, lhs) == enumerated_costs(fresh, lhs)

    def test_max_meanings(self):
        semantics = PointwiseSemantics(self.boolean_universe)
        enumerator = SemanticEnumerator(self.boolean_grammar, semantics)
        enumerator.run(1000, max_meanings=50)
        assert len(enumerator.get_states()) == 50
        # the meanings found so far still have minimal costs, and resuming finds the rest
        expected = brute_force_costs(self.boolean_grammar, self.boolean_universe, 7)
        for value, cost in enumerated_costs(enumerator, bool).items():
            assert expected[bool].get(value, cost) == cost
        enumerator.run(7)
        assert enumerated_costs(enumerator, bool) == expected[bool]

    def test_ignore_errors(self):
        grammar = Grammar(int)
        grammar.add_rule(Rule("/", int, (int, int), lambda x, y: x // y))
        grammar.add_rule(Rule("n", int, None, lambda referent: referent.num))
        grammar.add_rule(Rule("1", int, None, lambda *args: 1))
        with pytest.raises(ZeroDivisionError):
            grammar.get_minimal_expressions(self.numbers_universe, max_cost=3)
        expressions = grammar.get_minimal_expressions(
            self.numbers_universe, max_cost=3, ignore_errors=True
        )
        assert {str(expression) for expression in expressions.values()} == {
            "n",
            "1",
        }

    def test_invalid_cost(self):
        with pytest.raises(ValueError):
            SemanticEnumerator(
                self.boolean_grammar,
                PointwiseSemantics(self.boolean_universe),
                cost=lambda rule: 0.5,
            )
