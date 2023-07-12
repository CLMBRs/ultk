import pandas as pd
from ultk.language.semantics import Universe

referents = pd.read_csv("indefinites/referents.csv")
prior = pd.read_csv("indefinites/data/Beekhuizen_priors.csv")
assert (referents["name"] == prior["name"]).all()
referents["probability"] = prior["probability"]
universe = Universe.from_dataframe(referents)
