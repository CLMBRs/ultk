import pandas as pd

from ultk.language.semantics import Universe


def munsell_cielab_universe(filename: str) -> Universe:
    """
    Load a Universe from a Munsell CIELAB file.
    """
    referents = pd.read_csv(filename, delimiter="\t")
    print(referents)
    referents["name"] = referents["#cnum"]
    return Universe.from_dataframe(referents)


color_universe = munsell_cielab_universe("data/cnum-vhcm-lab-new.txt")
