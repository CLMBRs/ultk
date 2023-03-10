from altk.language.grammar import Grammar, Rule

indefinites_grammar = Grammar(bool)
# basic propositional logic
indefinites_grammar.add_rule(
    Rule("and", bool, (bool, bool), lambda p1, p2: p1 and p2)
)
indefinites_grammar.add_rule(
    Rule("or", bool, (bool, bool), lambda p1, p2: p1 or p2)
)
indefinites_grammar.add_rule(Rule("not", bool, (bool,), lambda p1: not p1))
# primitive features
indefinites_grammar.add_rule(
    Rule("K+", bool, (), lambda point: point.name == "specific-known")
)
indefinites_grammar.add_rule(
    Rule(
        "S+",
        bool,
        (),
        lambda point: point.name in ("specific-known", "specific-unknown"),
    )
)
indefinites_grammar.add_rule(
    Rule(
        "SE+",
        bool,
        (),
        lambda point: point.name in ("npi", "freechoice", "negative-indefinite"),
    )
)
indefinites_grammar.add_rule(
    Rule("N+", bool, (), lambda point: point.name == "negative-indefinite")
)
indefinites_grammar.add_rule(
    Rule("R+", bool, (), lambda point: point.name in ("negative-indefinite", "npi"))
)
indefinites_grammar.add_rule(
    Rule("R-", bool, (), lambda point: point.name == "freechoice")
)