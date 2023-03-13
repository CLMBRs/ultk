import pandas as pd

from altk.effcomm.informativity import informativity
from altk.language.language import Expression, Language, aggregate_expression_complexity
from altk.language.sampling import all_languages, random_languages
from altk.language.semantics import Meaning


from ..grammar import indefinites_grammar
from ..meaning import universe as indefinites_universe


def read_natural_languages(filename: str) -> list[Language]:
    lang_data = pd.read_csv(filename)
    lang_data["flavors"] = lang_data.apply(
        lambda row: row[row == True].index.tolist(), axis=1
    )
    language_frame = lang_data.groupby("language")
    languages = set()
    for lang, items in language_frame:
        cur_expressions = []
        for item in items.itertuples():
            cur_meaning = Meaning(
                [indefinites_universe[flavor] for flavor in item.flavors],
                indefinites_universe,
            )
            cur_expressions.append(Expression(item.expression, cur_meaning))
        languages.add(Language(cur_expressions, name=lang, natural=True))
    return languages


if __name__ == "__main__":

    # NB: in a larger-scale study, you would probably want to do this once and save
    # the resulting dictionary as a file.  We are not doing that in the present case
    # because it is faster to just generate from scratch than to read from a file and
    # re-parse the inputs.
    meanings_by_expressions = indefinites_grammar.get_unique_expressions(
        3,
        max_size=2 ** len(indefinites_universe),
        unique_key=lambda expr: expr.evaluate(indefinites_universe),
        compare_func=lambda e1, e2: len(e1) < len(e2),
    )

    expressions = list(meanings_by_expressions.values())

    languages = list(all_languages(expressions, max_size=3))
    print(len(languages))

    languages = list(random_languages(expressions, 1000, max_size=8))
    print(len(languages))
    print([len(language) for language in languages])

    language = languages[56]
    print(language)
    print(informativity(language, language.universe.prior_numpy()))

    def complexity(language):
        return aggregate_expression_complexity(
            # TODO: change this measure to closer match the paper?
            language,
            lambda gram_expr: len(gram_expr),
        )

    print(complexity(language))

    for language in read_natural_languages("indefinites/data/natural_language_indefinites.csv"):
        print(language)
        print(language.name)
        print(language.natural)