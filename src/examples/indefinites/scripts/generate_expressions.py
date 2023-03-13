import pandas as pd

from ..grammar import indefinites_grammar
from ..meaning import universe as indefinites_universe


if __name__ == "__main__":

    expressions_by_meaning = indefinites_grammar.get_unique_expressions(
        3,
        max_size=2 ** len(indefinites_universe),
        unique_key=lambda expr: expr.evaluate(indefinites_universe),
        compare_func=lambda e1, e2: len(e1) < len(e2),
    )

    expressions_dicts = [
        {
            "grammatical_form": str(expressions_by_meaning[meaning]),
            "flavors": [referent.name for referent in meaning.referents],
            "complexity": len(expressions_by_meaning[meaning]),
        }
        for meaning in expressions_by_meaning
    ]

    pd.DataFrame.from_records(expressions_dicts).to_csv("indefinites/outputs/generated_expressions.csv", index=False)

