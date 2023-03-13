from typing import Callable
from altk.language.grammar import Grammar, Rule
from altk.language.semantics import Referent


indefinites_grammar = Grammar(bool)
# basic propositional logic
indefinites_grammar.add_rule(Rule("and", bool, (bool, bool), lambda p1, p2: p1 and p2))
indefinites_grammar.add_rule(Rule("or", bool, (bool, bool), lambda p1, p2: p1 or p2))
indefinites_grammar.add_rule(Rule("not", bool, (bool,), lambda p1: not p1))

"""
from meaning import universe

Feature = Callable[[Referent], bool]

indefinites_grammar.add_rule(Rule("apply", bool, (Feature, Referent), lambda feat, ref: feat(ref)))
indefinites_grammar.add_rule(Rule("K+", Feature, None, lambda ref: ref.name == "specific-known"))
indefinites_grammar.add_rule(Rule("fc", Referent, None, universe["freechoice"]))
"""

# primitive features
# We include "positive" and "negative" features as primitives (instead of definining the latter via negation) for two reasons.
# (1) Conceptually, it's not clear that the positive ones are any more basic than the negative ones.  But defining them in
# terms of negation would make them more complex according to our measure.
# (2) Computationally, this greatly shrinks the space of grammatical expressions that need to be explored before finding one
# that expresses each meaning.
indefinites_grammar.add_rule(
    Rule("K+", bool, None, lambda point: point.name == "specific-known")
)
indefinites_grammar.add_rule(
    Rule("K-", bool, None, lambda point: point.name != "specific-known")
)
indefinites_grammar.add_rule(
    Rule(
        "S+",
        bool,
        None,
        lambda point: point.name in ("specific-known", "specific-unknown"),
    )
)
indefinites_grammar.add_rule(
    Rule(
        "S-",
        bool,
        None,
        lambda point: point.name not in ("specific-known", "specific-unknown"),
    )
)
indefinites_grammar.add_rule(
    Rule(
        "SE+",
        bool,
        None,
        lambda point: point.name in ("npi", "freechoice", "negative-indefinite"),
    )
)
indefinites_grammar.add_rule(
    Rule(
        "SE-",
        bool,
        None,
        lambda point: point.name not in ("npi", "freechoice", "negative-indefinite"),
    )
)
indefinites_grammar.add_rule(
    Rule("N+", bool, None, lambda point: point.name == "negative-indefinite")
)
indefinites_grammar.add_rule(
    Rule("N-", bool, None, lambda point: point.name != "negative-indefinite")
)
indefinites_grammar.add_rule(
    Rule("R+", bool, None, lambda point: point.name in ("negative-indefinite", "npi"))
)
indefinites_grammar.add_rule(
    Rule("R-", bool, None, lambda point: point.name == "freechoice")
)
