import pandas as pd
from altk.language.semantics import Universe

referents = pd.read_csv("referents.csv")
prior = pd.read_csv("data/Beekhuizen_priors.csv")
assert (referents["name"] == prior["name"]).all()
referents["probability"] = prior["probability"]
universe = Universe.from_dataframe(referents)