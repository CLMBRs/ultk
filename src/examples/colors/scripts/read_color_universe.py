import pandas as pd
import pickle

from ultk.language.semantics import Universe


if __name__ == "__main__":
    referents = pd.read_csv("colors/data/cnum-vhcm-lab-new.txt", delimiter="\t")
    referents.sort_values(by="#cnum", inplace=True)
    # add a name column, as required by ULTK
    referents["name"] = referents["#cnum"]
    color_universe = Universe.from_dataframe(referents)

    with open("colors/outputs/color_universe.pkl", "wb") as f:
        pickle.dump(color_universe, f)
