import pandas as pd

from ultk.language.semantics import Universe


def munsell_cielab_universe(filename: str) -> Universe:
    """
    Load a Universe from a Munsell CIELAB file.

    Note: the tuple Universe.referents will be ordered by the Munsell Chip Number, as used in the WCS data.
    """
    referents = pd.read_csv(filename, delimiter="\t")
    referents.sort_values(by="#cnum", inplace=True)
    # add a name column, as required by ULTK
    referents["name"] = referents["#cnum"]
    return Universe.from_dataframe(referents)


color_universe = munsell_cielab_universe("data/cnum-vhcm-lab-new.txt")
