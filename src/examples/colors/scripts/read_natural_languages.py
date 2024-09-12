import pandas as pd

from ultk.language.language import Expression, Language
from ..meaning import color_universe

if __name__ == "__main__":

    term_table = pd.read_csv(
        "data/term.txt", delimiter="\t", names=("lang", "spkr", "cnum", "term")
    )
    print(term_table)

    lang_term_chip_counts = term_table.groupby(["lang", "term", "cnum"]).count()
    print(lang_term_chip_counts)
    print(lang_term_chip_counts.index)

    for lang, lang_df in lang_term_chip_counts.groupby("lang"):
        print(lang)
        for term, term_df in lang_df.groupby("term"):
            print(term)
            print(term_df)
