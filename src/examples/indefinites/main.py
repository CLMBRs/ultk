import pandas as pd
from altk.language.semantics import Universe


if __name__ == '__main__':
    referents = pd.read_csv("referents.csv")
    universe = Universe.from_dataframe(referents)
    print(universe)