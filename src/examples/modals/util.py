import os
from numbers import Integral
from typing import Any, Callable

import pandas as pd
import yaml
from yaml import dump, Dumper

from ultk.language.language import Expression, Language, Meaning
from ultk.language.semantics import Universe

from .meaning import universe as modals_universe

# Natural language data from the Database of Modal Typology (https://github.com/CLMBRs/modal-typology),
# as used in https://github.com/nathimel/modals-effcomm, with one directory per language
NATURAL_LANGUAGES_DIR = "modals/data/natural_languages"
METADATA_FN = "metadata.yml"
MODALS_FN = "modals.csv"

ALLOWED_REFERENCE_TYPES = ["paper-journal", "elicitation"]
REFERENCE_TYPE_KEY = "Reference-type"


def load_natural_languages(universe: Universe = modals_universe) -> list[Language]:
    """Load the natural language modal systems as Languages.

    Args:
        universe: the universe of force-flavor pairs

    Returns:
        a list of natural Languages
    """
    return dataframes_to_languages(get_modals_data(), universe)


def get_modals_data(data_dir: str = NATURAL_LANGUAGES_DIR) -> dict[str, pd.DataFrame]:
    """Read the modals data for each language whose data comes from a paper or from elicitation.

    Args:
        data_dir: a directory containing one directory per language, each with a metadata and modals file

    Returns:
        a dictionary from language names to dataframes of (expression, force, flavor, can_express) observations
    """
    dataframes = {}
    for language_name in sorted(os.listdir(data_dir)):
        language_dir = os.path.join(data_dir, language_name)
        if not os.path.isdir(language_dir):
            continue
        with open(os.path.join(language_dir, METADATA_FN), "r") as f:
            metadata = yaml.safe_load(f)
        reference_type = metadata[REFERENCE_TYPE_KEY]
        # data from reference grammars is often incomplete, so we skip it
        if reference_type in ALLOWED_REFERENCE_TYPES:
            dataframes[language_name] = pd.read_csv(
                os.path.join(language_dir, MODALS_FN)
            )
        else:
            print(f"Data for {language_name} is of type {reference_type}; skipping.")
    return dataframes


def process_can_express(
    val: Any, can_express: dict[bool, list] = {True: [1], False: ["?", 0]}
) -> bool:
    """For an observation of whether a modal can_express a force-flavor pair, interpret ? as 1.

    Note that the existence of ? in the csv confuses pandas, and causes the type of can_express to be str.

    Args:
        val: the value of the can_express column, possibly an int or str
        can_express: which values to interpret as True and as False

    Returns:
        boolean representing 'yes' if the value should be interpreted as True, False otherwise
    """
    if isinstance(val, Integral):
        return bool(val)
    if val.isnumeric():
        return bool(int(val))

    # different results depending on interpretation of '?'
    return val in can_express[True]


def dataframes_to_languages(
    dataframes: dict[str, pd.DataFrame], universe: Universe
) -> list[Language]:
    """Convert a dictionary of dataframes to a list of ultk Languages."""
    languages: list[Language] = []
    for language_name, df in dataframes.items():
        language = dataframe_to_language(df, language_name, universe)
        if language is not None:
            languages.append(language)
    return languages


def dataframe_to_language(
    df: pd.DataFrame,
    language_name: str,
    universe: Universe,
) -> Language | None:
    """Construct a ultk Language from a dataframe of (expression, meaning) observations by iterating over each row and indicating whether the expression can express the meaning.

    Returns:
        the Language, or None if none of its modals can express any of the referents in `universe`
    """
    forces = set(referent.force for referent in universe.referents)
    flavors = set(referent.flavor for referent in universe.referents)

    vocabulary: dict[str, set[str]] = {}

    # only look at positive polarity modals
    if "polarity" in df:
        df_positive = df[df["polarity"] == "positive"]
    else:
        df_positive = df

    # add each observation
    for _, row in df_positive.iterrows():
        modal = row["expression"]
        # initialize an expression's set of meanings
        if modal not in vocabulary:
            vocabulary[modal] = set()

        # Add only the flavors specified as possible for the experiment
        if row["flavor"] in flavors and row["force"] in forces:
            if process_can_express(row["can_express"]):
                vocabulary[modal].add(f"{row['force']}+{row['flavor']}")

    # Convert vocabulary into list of Expressions
    experiment_vocabulary = []
    for modal, referent_names in vocabulary.items():
        meaning = Meaning(
            tuple(referent.name in referent_names for referent in universe.referents),
            universe,
        )
        # often there will be no usable referents due to can_express being False, above
        if meaning.is_uniformly_false:
            continue
        experiment_vocabulary.append(Expression(modal, meaning))

    if experiment_vocabulary:
        return Language(tuple(experiment_vocabulary), name=language_name, natural=True)
    return None


def write_languages(
    languages: list[Language],
    filename: str,
    properties_to_add: dict[str, Callable[[int, Language], Any]] = {},
) -> None:
    """Write languages to a YAML file, together with some of their properties.

    Args:
        languages: the languages to write
        filename: the file to write to
        properties_to_add: a dictionary from property names to functions of (index, language) computing that property
    """
    lang_dicts = [
        language.as_dict_with_properties(
            **{key: properties_to_add[key](idx, language) for key in properties_to_add}
        )
        for idx, language in enumerate(languages)
    ]
    with open(filename, "w+") as f:
        dump(lang_dicts, f, Dumper=Dumper)
