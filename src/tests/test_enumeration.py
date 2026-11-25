from collections import defaultdict
from itertools import product
from typing import Any, Callable

import math

import pytest

from ultk.language.grammar.enumeration import (
    CostFunction,
    KnuthEnumerator,
    SemanticEnumerator,
    additive_cost,
    depth_cost,
    negative_log_probability,
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


def brute_force_costs_by_size(
    grammar: Grammar, universe: Universe, max_size: int, combine: CostFunction
) -> dict[Any, dict[tuple, float]]:
    """Get the minimal cost, under any cost function, of every meaning of every LHS among _all_ expressions with at most
    `max_size` nodes, without any semantic deduplication."""
    # lhs -> size -> list of (value, cost) of all expressions with that many nodes
    expressions: dict[Any, dict[int, list[tuple[tuple, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for size in range(1, max_size + 1):
        for rule in grammar.get_all_rules():
            if rule.rhs is None:
                if size == 1:
                    value = tuple(
                        rule.func(referent) for referent in universe.referents
                    )
                    expressions[rule.lhs][1].append((value, combine(rule, ())))
                continue
            for child_sizes in product(range(1, size), repeat=len(rule.rhs)):
                if sum(child_sizes) != size - 1:
                    continue
                pools = [
                    expressions[child_lhs][child_size]
                    for child_lhs, child_size in zip(rule.rhs, child_sizes)
                ]
                for children in product(*pools):
                    value = tuple(map(rule.func, *(child[0] for child in children)))
                    cost = combine(rule, [child[1] for child in children])
                    expressions[rule.lhs][size].append((value, cost))
    min_costs: dict[Any, dict[tuple, float]] = defaultdict(dict)
    for lhs, by_size in expressions.items():
        for found in by_size.values():
            for value, cost in found:
                if cost < min_costs[lhs].get(value, math.inf):
                    min_costs[lhs][value] = cost
    return min_costs


def enumerated_costs(
    enumerator: SemanticEnumerator | KnuthEnumerator, lhs: Any
) -> dict[tuple, float]:
    return {
        meaning.mapping: cost for meaning, cost in enumerator.get_costs(lhs).items()
    }


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
        enumerator = SemanticEnumerator(self.boolean_grammar, self.boolean_universe)
        enumerator.run(7)
        expected = brute_force_costs(self.boolean_grammar, self.boolean_universe, 7)
        assert enumerated_costs(enumerator, bool) == expected[bool]

    def test_minimal_costs_multiple_types(self):
        enumerator = SemanticEnumerator(self.numbers_grammar, self.numbers_universe)
        enumerator.run(7)
        expected = brute_force_costs(self.numbers_grammar, self.numbers_universe, 7)
        for lhs in (bool, int):
            assert enumerated_costs(enumerator, lhs) == expected[lhs]

    def test_minimal_costs_weighted(self):
        def cost(rule: Rule) -> int:
            return {"and": 3, "or": 2, "not": 2}.get(rule.name, 1)

        enumerator = SemanticEnumerator(
            self.boolean_grammar, self.boolean_universe, cost=cost
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
        enumerator = SemanticEnumerator(self.boolean_grammar, self.boolean_universe)
        enumerator.run(1000)
        assert enumerator.is_saturated
        assert enumerator.completed_cost < 1000
        # every Boolean function of three variables is expressible
        assert len(enumerator.get_states()) == 2**8

    def test_resume(self):
        resumed = SemanticEnumerator(self.numbers_grammar, self.numbers_universe)
        resumed.run(4)
        resumed.run(7)
        fresh = SemanticEnumerator(self.numbers_grammar, self.numbers_universe)
        fresh.run(7)
        for lhs in (bool, int):
            assert enumerated_costs(resumed, lhs) == enumerated_costs(fresh, lhs)

    def test_max_meanings(self):
        enumerator = SemanticEnumerator(self.boolean_grammar, self.boolean_universe)
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
                self.boolean_universe,
                cost=lambda rule: 0.5,
            )


class TestKnuthEnumeration:
    boolean_grammar = TestEnumeration.boolean_grammar
    boolean_universe = TestEnumeration.boolean_universe
    numbers_grammar = TestEnumeration.numbers_grammar
    numbers_universe = TestEnumeration.numbers_universe

    @staticmethod
    def assert_costs_match(
        enumerated: dict[tuple, float], expected: dict[tuple, float], max_cost: float
    ):
        """The enumerated meanings are exactly the expected ones with cost at most `max_cost`, with the same costs."""
        expected = {
            value: cost for value, cost in expected.items() if cost <= max_cost + 1e-9
        }
        assert enumerated.keys() == expected.keys()
        for value, cost in enumerated.items():
            assert cost == pytest.approx(expected[value])

    def test_unit_costs_agree_with_semantic_enumerator(self):
        for grammar, universe in (
            (self.boolean_grammar, self.boolean_universe),
            (self.numbers_grammar, self.numbers_universe),
        ):
            levels = SemanticEnumerator(grammar, universe)
            levels.run(8)
            knuth = KnuthEnumerator(grammar, universe)
            knuth.run(8)
            for lhs in {rule.lhs for rule in grammar.get_all_rules()}:
                assert enumerated_costs(knuth, lhs) == enumerated_costs(levels, lhs)

    def test_real_valued_costs(self):
        rule_costs = {
            "and": 1.5,
            "or": 0.75,
            "not": 0.5,
            "x0": 1.0,
            "x1": 0.6,
            "x2": 1.3,
        }
        combine = additive_cost(lambda rule: rule_costs[rule.name])
        enumerator = KnuthEnumerator(
            self.boolean_grammar, self.boolean_universe, combine=combine
        )
        enumerator.run(4.0)
        # every rule costs at least 0.5, so every expression of cost at most 4 has at most 8 nodes
        expected = brute_force_costs_by_size(
            self.boolean_grammar, self.boolean_universe, 8, combine
        )
        self.assert_costs_match(enumerated_costs(enumerator, bool), expected[bool], 4.0)

    def test_depth(self):
        enumerator = KnuthEnumerator(
            self.boolean_grammar, self.boolean_universe, combine=depth_cost
        )
        enumerator.run(3)
        # every expression of depth at most 3 has at most 7 nodes
        expected = brute_force_costs_by_size(
            self.boolean_grammar, self.boolean_universe, 7, depth_cost
        )
        self.assert_costs_match(enumerated_costs(enumerator, bool), expected[bool], 3)

    def test_negative_log_probability(self):
        weights = {"and": 2, "or": 1, "not": 3, "x0": 4, "x1": 1, "x2": 1}
        grammar = Grammar(bool)
        for rule in self.boolean_grammar.get_all_rules():
            grammar.add_rule(
                Rule(rule.name, rule.lhs, rule.rhs, rule.func, weights[rule.name])
            )
        combine = additive_cost(negative_log_probability(grammar))
        enumerator = KnuthEnumerator(grammar, self.boolean_universe, combine=combine)
        enumerator.run(7.0)
        # every rule costs at least -log(4/12) > 1, so every expression of cost at most 7 has at most 6 nodes
        expected = brute_force_costs_by_size(grammar, self.boolean_universe, 6, combine)
        self.assert_costs_match(enumerated_costs(enumerator, bool), expected[bool], 7.0)
        # the minimal expression for each meaning is a most probable one
        for meaning, expression in enumerator.get_expressions().items():
            assert grammar.log_prior(expression) == pytest.approx(
                -enumerator.get_costs()[meaning]
            )

    def test_zero_costs(self):
        combine = additive_cost(lambda rule: 0 if rule.name == "not" else 1)
        enumerator = KnuthEnumerator(
            self.boolean_grammar, self.boolean_universe, combine=combine
        )
        enumerator.run(3)
        costs = {
            str(expression): enumerator.get_costs()[meaning]
            for meaning, expression in enumerator.get_expressions().items()
        }
        assert costs["not(x0)"] == 1

    def test_resume(self):
        combine = additive_cost(lambda rule: 0.75 if rule.name == "or" else 1.0)
        resumed = KnuthEnumerator(
            self.boolean_grammar, self.boolean_universe, combine=combine
        )
        for max_cost in (1.0, 2.5, 3.0, 4.75, 6.0):
            resumed.run(max_cost)
        fresh = KnuthEnumerator(
            self.boolean_grammar, self.boolean_universe, combine=combine
        )
        fresh.run(6.0)
        assert enumerated_costs(resumed, bool) == enumerated_costs(fresh, bool)

    def test_saturation(self):
        enumerator = KnuthEnumerator(self.boolean_grammar, self.boolean_universe)
        enumerator.run(5)
        assert not enumerator.is_saturated
        enumerator.run(math.inf)
        assert enumerator.is_saturated
        # every Boolean function of three variables is expressible
        assert len(enumerator.get_states()) == 2**8

    def test_max_meanings(self):
        enumerator = KnuthEnumerator(self.boolean_grammar, self.boolean_universe)
        enumerator.run(math.inf, max_meanings=50)
        assert len(enumerator.get_states()) == 50
        enumerator.run(7)
        expected = brute_force_costs(self.boolean_grammar, self.boolean_universe, 7)
        assert enumerated_costs(enumerator, bool) == {
            value: cost for value, cost in expected[bool].items() if cost <= 7
        }

    def test_not_superior(self):
        # `not` is cheaper than its child, so this is not a superior function
        def combine(rule, child_costs):
            return 1 if rule.name == "not" else 1 + sum(child_costs)

        enumerator = KnuthEnumerator(
            self.boolean_grammar, self.boolean_universe, combine=combine
        )
        with pytest.raises(ValueError):
            enumerator.run(5)
