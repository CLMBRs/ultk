from typing import Callable, Any
import pandas as pd

from yaml import load, load_all, dump, Loader, Dumper

from ultk.language.grammar import GrammaticalExpression
from ultk.language.language import Expression, Language
from ultk.language.semantics import Meaning, Universe

from .grammar import indefinites_grammar
from .meaning import universe as indefinites_universe


def read_natural_languages(filename: str) -> set[Language]:
    """Read the natural languages from a CSV file.
    Assumes that each row is one expression, with unique strings in "language" column identifying
    which expressions belong to which languages.
    Assumes that there is a boolean-valued column for each Referent in the indefinites_universe,
    identified by its name.

    Args:
        filename: the file to read

    Returns:
        a list of Languages
    """
    lang_data = pd.read_csv(filename)
    lang_data["flavors"] = lang_data.apply(
        lambda row: row[row == True].index.tolist(), axis=1
    )
    # group data frame by language
    language_frame = lang_data.groupby("language")
    languages = set()
    # iterate through each language
    for lang, items in language_frame:
        cur_expressions = []
        for item in items.itertuples():
            # generate Meaning from list of flavors
            cur_meaning = Meaning(
                tuple(indefinites_universe[flavor] for flavor in item.flavors),
                indefinites_universe,
            )
            # add Expression with form and Meaning
            cur_expressions.append(Expression(item.expression, cur_meaning))
        # add Language with its Expressions
        languages.add(Language(tuple(cur_expressions), name=lang, natural=True))
    return languages


def read_expressions(
    filename: str, universe: Universe | None = None, return_by_meaning=True
) -> tuple[list[GrammaticalExpression], dict[Meaning, Expression]]:
    """Read expressions from a YAML file.
    Assumes that the file is a list, and that each item in the list has a field
    "grammatical_expression" with an expression that can be parsed by the
    indefinites_grammar.
    """
    with open(filename, "r") as f:
        expression_list = load(f, Loader=Loader)
    parsed_exprs = [
        indefinites_grammar.parse(expr_dict["term_expression"])
        for expr_dict in expression_list
    ]
    if universe is not None:
        [expr.evaluate(universe) for expr in parsed_exprs]
    by_meaning = {}
    if return_by_meaning:
        by_meaning = {expr.meaning: expr for expr in parsed_exprs}
    return parsed_exprs, by_meaning


def write_languages(
    languages: list[Language],
    filename: str,
    properties_to_add: dict[str, Callable[[int, Language], Any]] = {},
) -> None:
    lang_dicts = [
        language.as_dict_with_properties(
            **{key: properties_to_add[key](idx, language) for key in properties_to_add}
        )
        for idx, language in enumerate(languages)
    ]
    with open(filename, "w+") as f:
        dump(lang_dicts, f, Dumper=Dumper)
