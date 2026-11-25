"""A grammar for C-RASP programs (Yang & Chiang 2024, "Counting Like Transformers").

C-RASP has two types of operations, which are the two non-terminals of the grammar (see `values.py`):
- `Booleans`: Boolean-valued operations, which assign a Boolean to every position of a string
- `Counts`: count-valued operations, which assign an integer to every position of a string

The referents of the universe are strings (see `meaning.py`), so the value of an operation at a referent assigns a
value to every position of the string.  Because every operation only looks at positions up to the current one, and in
the same string, the semantics is compositional in the sense required by `ultk.language.grammar.enumeration`.

A program accepts a string iff its last Boolean-valued operation is true at the last position of the string
(Definition 3.1).  This is the job of the start rule `accept`, whose type is `bool`.  Because programs are built as
expressions (i.e. trees), an operation that is used several times in a program is counted every time it is used.

Each rule is defined by one function, whose type annotations give the rule's argument and return types (read by
`Rule.from_callable`) and whose body is the only definition of the operation.  The bodies only use the operations of
`values.Booleans` and `values.Counts`, so the same functions compute the values of an operation for one string (in
`GrammaticalExpression.evaluate`), or for every string in a universe at once (see `representations/`).  Terminal rules
are applied to a `values.HasPositions`: a `meaning.StringReferent`, or a stand-in for all strings at once.  See
`values.py` for how these interfaces fit together.

The rules follow Definition 3.1 exactly:

| C-RASP operation                | rule                                   |
|---------------------------------|----------------------------------------|
| P(i) := Q_a(i)                  | `Q_a : Booleans`                       |
| P(i) := not P1(i)               | `not : Booleans -> Booleans`           |
| P(i) := P1(i) and P2(i)         | `and : Booleans -> Booleans, Booleans` |
| P(i) := C1(i) <= C2(i)          | `<= : Booleans -> Counts, Counts`      |
| P(i) := 1                       | `true : Booleans`                      |
| C(i) := #[j <= i] P(j)          | `# : Counts -> Booleans`               |
| C(i) := P(i) ? C1(i) : C2(i)    | `? : Counts -> Booleans, Counts, Counts` |
| C(i) := C1(i) + C2(i)           | `+ : Counts -> Counts, Counts`         |
| C(i) := C1(i) - C2(i)           | `- : Counts -> Counts, Counts`         |
| C(i) := min(C1(i), C2(i))       | `min : Counts -> Counts, Counts`       |
| C(i) := max(C1(i), C2(i))       | `max : Counts -> Counts, Counts`       |
| C(i) := 1                       | `1 : Counts`                           |
"""

from collections.abc import Callable, Sequence

from ultk.language.grammar.grammar import Grammar, Rule

from .values import Booleans, Counts, HasPositions

ALPHABET = ("a", "b")


def accept(p: Booleans) -> bool:
    # for one string, this is a bool; for all strings at once, one bool per string (see `Booleans.at_end`)
    return p.at_end()


def negation(p: Booleans, name="not") -> Booleans:
    return ~p


def conjunction(p1: Booleans, p2: Booleans, name="and") -> Booleans:
    return p1 & p2


def less_equal(c1: Counts, c2: Counts, name="<=") -> Booleans:
    return c1 <= c2


def true(strings: HasPositions) -> Booleans:
    return strings.positions.constant_booleans(True)


def count(p: Booleans, name="#") -> Counts:
    return p.count_so_far()


def conditional(p: Booleans, c1: Counts, c2: Counts, name="?") -> Counts:
    return p.where(c1, c2)


def addition(c1: Counts, c2: Counts, name="+") -> Counts:
    return c1 + c2


def subtraction(c1: Counts, c2: Counts, name="-") -> Counts:
    return c1 - c2


def minimum(c1: Counts, c2: Counts, name="min") -> Counts:
    return c1.minimum(c2)


def maximum(c1: Counts, c2: Counts, name="max") -> Counts:
    return c1.maximum(c2)


def one(strings: HasPositions, name="1") -> Counts:
    return strings.positions.constant_counts(1)


def symbol_predicate(symbol: str) -> Callable[..., Booleans]:
    """Get the function for the initial operation Q_symbol, which is true at the positions where `symbol` occurs."""

    def predicate(strings: HasPositions, name=f"Q_{symbol}") -> Booleans:
        return strings.positions.symbol_is(symbol)

    return predicate


# the rules other than the initial operations Q_a, which depend on the alphabet
OPERATIONS = (
    negation,
    conjunction,
    less_equal,
    true,
    count,
    conditional,
    addition,
    subtraction,
    minimum,
    maximum,
    one,
)


def build_crasp_grammar(alphabet: Sequence[str] = ALPHABET) -> Grammar:
    """Build a grammar for C-RASP programs over an alphabet.

    Each rule is built by `Rule.from_callable` from one of the functions above.  Their terminals take a `HasPositions`
    rather than a `Referent`, so that the representations in `representations/` can apply them to all strings at once.

    Args:
        alphabet: the symbols of the alphabet; each must be a single character

    Returns:
        a `Grammar` with start symbol `bool`, whose expressions are C-RASP programs
    """

    def from_callable(func: Callable) -> Rule:
        return Rule.from_callable(func, terminal_types=(HasPositions,))

    grammar = Grammar(bool)
    grammar.add_rule(from_callable(accept))
    for symbol in alphabet:
        grammar.add_rule(from_callable(symbol_predicate(symbol)))
    for operation in OPERATIONS:
        grammar.add_rule(from_callable(operation))
    return grammar


crasp_grammar = build_crasp_grammar()
