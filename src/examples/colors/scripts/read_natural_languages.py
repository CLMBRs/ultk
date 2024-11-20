import pickle
from typing import Any

import pandas as pd
from yaml import dump, Dumper

from ultk.language.language import Expression, Language
from ultk.language.semantics import Meaning
from ultk.util.frozendict import FrozenDict

from ..meaning import color_universe

if __name__ == "__main__":

    term_table = pd.read_csv(
        "colors/data/term.txt", delimiter="\t", names=("lang", "spkr", "cnum", "term")
    )

    lang_term_chip_counts = term_table.groupby(["lang", "term", "cnum"]).count()

    num_referents: int = len(color_universe.referents)

    def safe_loc(df: pd.DataFrame, key: Any, default: Any = 0.0) -> Any:
        try:
            return df.loc[key]
        except KeyError:
            return default

    languages: list[Language] = []
    # iterate through each language
    for lang, lang_df in lang_term_chip_counts.groupby("lang"):
        expressions: list[Expression] = []
        # iterate through each term in each language
        for term, term_df in lang_df.groupby("term"):
            # convert counts to probabilities
            total_count = term_df["spkr"].sum()
            chip_probabilities = term_df["spkr"] / total_count
            # Referent -> probability (a float) mapping, for all Referents = Munsell chips
            referent_dict = {
                color_universe.referents[chip_num]: safe_loc(
                    chip_probabilities, (lang, term, chip_num)
                )
                for chip_num in range(num_referents)
            }
            # turn into Meaning
            expression_meaning = Meaning[float](
                FrozenDict(referent_dict), color_universe, FrozenDict(referent_dict)
            )
            # attach form to make Expression
            expression = Expression(form=f"{lang}-{term}", meaning=expression_meaning)
            expressions.append(expression)
        # add Language with its Expressions
        languages.append(
            Language(expressions=tuple(expressions), name=lang, natural=True)
        )

    # write languages to pickle
    # TODO: serialize in better format (e.g. YAML)
    # with open("colors/outputs/natural_languages.pkl", "wb") as f:
    # pickle.dump(languages, f)
    with open("colors/outputs/natural_languages.yaml", "w") as f:
        dump(languages, f, Dumper=Dumper)
