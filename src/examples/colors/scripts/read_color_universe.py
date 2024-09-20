import pandas as pd


if __name__ == "__main__":
    referents = pd.read_csv("colors/data/cnum-vhcm-lab-new.txt", delimiter="\t")
    referents.sort_values(by="#cnum", inplace=True)
    # add a name column, as required by ULTK
    referents["name"] = referents["#cnum"]
    # rename columns for access as python properties
    referents.rename(columns={"L*": "L", "a*": "a", "b*": "b"}, inplace=True)
    referents.to_csv("colors/outputs/color_universe.csv", index=False)
