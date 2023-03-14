from yaml import load

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from altk.language.grammar import GrammaticalExpression
from altk.language.language import Expression, Language
from altk.language.semantics import Meaning, Universe

from .grammar import indefinites_grammar
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


def read_expressions(
    filename: str, universe: Universe = None, return_by_meaning = True
) -> tuple[list[GrammaticalExpression], dict[Meaning, Expression]]:
    with open(filename, "r") as f:
        expression_list = load(f, Loader=Loader)
    parsed_exprs = [
        indefinites_grammar.parse(expr_dict["grammatical_expression"])
        for expr_dict in expression_list
    ]
    if universe is not None:
        [expr.evaluate(universe) for expr in parsed_exprs]
    by_meaning = {}
    if return_by_meaning:
        by_meaning = {expr.meaning: expr for expr in parsed_exprs}
    return parsed_exprs, by_meaning
