from altk.language.language import Expression, Language
from altk.language.semantics import Meaning

from .meaning import universe as indefinites_universe


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
