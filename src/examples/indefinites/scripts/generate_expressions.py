from yaml import dump

try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper

from ..grammar import indefinites_grammar
from ..meaning import universe as indefinites_universe


if __name__ == "__main__":

    expressions_by_meaning = indefinites_grammar.get_unique_expressions(
        3,
        max_size=2 ** len(indefinites_universe),
        unique_key=lambda expr: expr.evaluate(indefinites_universe),
        compare_func=lambda e1, e2: len(e1) < len(e2),
    )

    with open("indefinites/outputs/generated_expressions.yml", "w") as outfile:
        dump(
            [
                expressions_by_meaning[meaning].to_dict()
                for meaning in expressions_by_meaning
            ],
            outfile,
            Dumper=Dumper,
        )
