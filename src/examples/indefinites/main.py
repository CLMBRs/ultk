import pandas as pd
from altk.language.grammar import Grammar, Rule
from altk.language.semantics import Universe


if __name__ == "__main__":
    referents = pd.read_csv("referents.csv")
    prior = pd.read_csv("data/Beekhuizen_priors.csv")
    assert (referents["name"] == prior["name"]).all()
    referents["probability"] = prior["probability"]
    universe = Universe.from_dataframe(referents)
    print(universe)

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
    for exp in indefinites_grammar.enumerate(3):
        print(exp)
        print(exp.to_meaning(universe))
