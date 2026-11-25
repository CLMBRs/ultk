"""A grammar for C-RASP programs (Yang & Chiang 2024, "Counting Like Transformers").

C-RASP has two types of operations, which we represent with two non-terminals:
- `"P"`: Boolean-valued operations, which assign a Boolean to every position of a string
- `"C"`: count-valued operations, which assign an integer to every position of a string

The referents of the universe are strings (see `meaning.py`), so the value of an operation at a referent assigns a
value to every position of the string.  Because every operation only looks at positions up to the current one, and in
the same string, the semantics is compositional in the sense required by `ultk.language.grammar.enumeration`.

A program accepts a string iff its last Boolean-valued operation is true at the last position of the string
(Definition 3.1).  This is the job of the start rule `accept : bool -> P`.  Because programs are built as
expressions (i.e. trees), an operation that is used several times in a program is counted every time it is used.

The functions of the rules are written with the operations of `values.PositionValues`, so they do not depend on how
values are stored.  The same functions compute the values of an operation for one string (as `values.StringValues`,
in `GrammaticalExpression.evaluate`), or for every string in a universe at once (see `representations/`).  Likewise,
terminal rules are given something with a `positions` attribute: a `meaning.StringReferent`, or a stand-in for all
strings at once.

The rules follow Definition 3.1 exactly:

| C-RASP operation                | rule                         |
|---------------------------------|------------------------------|
| P(i) := Q_a(i)                  | `Q_a : P`                    |
| P(i) := not P1(i)               | `not : P -> P`               |
| P(i) := P1(i) and P2(i)         | `and : P -> P, P`            |
| P(i) := C1(i) <= C2(i)          | `<= : P -> C, C`             |
| P(i) := 1                       | `true : P`                   |
| C(i) := #[j <= i] P(j)          | `# : C -> P`                 |
| C(i) := P(i) ? C1(i) : C2(i)    | `? : C -> P, C, C`           |
| C(i) := C1(i) + C2(i)           | `+ : C -> C, C`              |
| C(i) := C1(i) - C2(i)           | `- : C -> C, C`              |
| C(i) := min(C1(i), C2(i))       | `min : C -> C, C`            |
| C(i) := max(C1(i), C2(i))       | `max : C -> C, C`            |
| C(i) := 1                       | `1 : C`                      |
"""

from collections.abc import Callable, Sequence
from typing import Any

from ultk.language.grammar.grammar import Grammar, Rule

from .values import PositionValues

ALPHABET = ("a", "b")


def accept(p: PositionValues) -> Any:
    return p.at_end()


def negation(p: PositionValues) -> PositionValues:
    return ~p


def conjunction(p1: PositionValues, p2: PositionValues) -> PositionValues:
    return p1 & p2


def less_equal(c1: PositionValues, c2: PositionValues) -> PositionValues:
    return c1 <= c2


def true(strings: Any) -> PositionValues:
    return strings.positions.constant(True)


def count(p: PositionValues) -> PositionValues:
    return p.count_so_far()


def conditional(
    p: PositionValues, c1: PositionValues, c2: PositionValues
) -> PositionValues:
    return p.where(c1, c2)


def addition(c1: PositionValues, c2: PositionValues) -> PositionValues:
    return c1 + c2


def subtraction(c1: PositionValues, c2: PositionValues) -> PositionValues:
    return c1 - c2


def minimum(c1: PositionValues, c2: PositionValues) -> PositionValues:
    return c1.minimum(c2)


def maximum(c1: PositionValues, c2: PositionValues) -> PositionValues:
    return c1.maximum(c2)


def one(strings: Any) -> PositionValues:
    return strings.positions.constant(1)


def symbol_predicate(symbol: str) -> Callable[[Any], PositionValues]:
    """Get the function for the initial operation Q_symbol, which is true at the positions where `symbol` occurs."""

    def predicate(strings: Any) -> PositionValues:
        return strings.positions.symbol_is(symbol)

    return predicate


def build_crasp_grammar(alphabet: Sequence[str] = ALPHABET) -> Grammar:
    """Build a grammar for C-RASP programs over an alphabet.

    Args:
        alphabet: the symbols of the alphabet; each must be a single character

    Returns:
        a `Grammar` with start symbol `bool`, whose expressions are C-RASP programs
    """
    grammar = Grammar(bool)
    grammar.add_rule(Rule("accept", bool, ("P",), accept))
    for symbol in alphabet:
        grammar.add_rule(Rule(f"Q_{symbol}", "P", None, symbol_predicate(symbol)))
    grammar.add_rule(Rule("not", "P", ("P",), negation))
    grammar.add_rule(Rule("and", "P", ("P", "P"), conjunction))
    grammar.add_rule(Rule("<=", "P", ("C", "C"), less_equal))
    grammar.add_rule(Rule("true", "P", None, true))
    grammar.add_rule(Rule("#", "C", ("P",), count))
    grammar.add_rule(Rule("?", "C", ("P", "C", "C"), conditional))
    grammar.add_rule(Rule("+", "C", ("C", "C"), addition))
    grammar.add_rule(Rule("-", "C", ("C", "C"), subtraction))
    grammar.add_rule(Rule("min", "C", ("C", "C"), minimum))
    grammar.add_rule(Rule("max", "C", ("C", "C"), maximum))
    grammar.add_rule(Rule("1", "C", None, one))
    return grammar


crasp_grammar = build_crasp_grammar()
