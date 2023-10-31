import pandas as pd
from altk.language.semantics import Universe

referent_index = pd.read_csv("../index.csv")
referent_pertinence = pd.read_csv("../pertinence.csv")

referents = referent_index.merge(referent_pertinence, how='cross')
referents.columns = ['name', 'pertinence']

universe = Universe.from_dataframe(referents)

