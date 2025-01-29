from ultk.language.grammar.grammar import Grammar, Rule
import pandas as pd
from ultk.language.semantics import Universe
import ultk.language.grammar.inference


referents = pd.read_csv("../indefinites/referents.csv")
prior = pd.read_csv("../indefinites/data/Beekhuizen_priors.csv")
assert (referents["name"] == prior["name"]).all()
referents["probability"] = prior["probability"]
universe = Universe.from_dataframe(referents)

indefinites_grammar = Grammar.from_yaml("../indefinites/grammar.yml")
# indefinites_grammar = Grammar.from_module("indefinites.grammar_functions")
print(
    ultk.language.grammar.inference.mh_sample(
        indefinites_grammar.parse("and(not(K+), or(N-, not(SE-)))"),
        indefinites_grammar,
        [(i, False) for i in universe.referents],
    )
)
