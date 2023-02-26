"""Functions for sampling expressions into languages.
"""
import copy
import random
import numpy as np
from altk.language.language import Language, Expression
from altk.effcomm.agent import Speaker, LiteralSpeaker
from typing import Callable, Type, Any
from math import comb
from itertools import combinations
from tqdm import tqdm


##############################################################################
# Methods for generating languages from expressions, or permuting the stochastic mapping from words to meanings of an existing language
##############################################################################


def get_hypothetical_variants(
    languages: list[Language] = None,
    speakers: list[Speaker] = None,
    total: int = 0,
) -> list[Any]:
    """For each system (parameterized by a language or else a speaker), generate `num` hypothetical variants by permuting the signals that the system assigns to states.

    Args:
        languages: a list of languages to permute, by constructing LiteralSpeakers and permuting their weights.

        speakers: a list of speakers of a language, whose weights can be directly permuted. Should be used instead of `languages` if possible, because it can be more finegrained (every language can be associated with multiple speakers).

        total: the total number of hypothetical variants to obtain

    Returns:
        hypothetical_variants: a list of type either Language or np.ndarray depending on whether `languages` or `speakers` was passed, representing hypothetical variants of the systems passed. If `speakers` was passed, a list of speakers is returned.
    """

    if (languages is not None and speakers is not None) or (
        languages is None and speakers is None
    ):
        raise Exception(
            "You must pass exactly one of the following: `languages`, `speakers`."
        )

    if languages is not None:
        num_systems = len(languages)
    else:
        num_systems = len(speakers)

    variants_per_system = int(total / num_systems)

    hypothetical_variants = []
    for i in range(num_systems):
        if languages is not None:
            speaker = LiteralSpeaker(languages[i])
        else:
            speaker = speakers[i]

        seen = set()
        while len(seen) < variants_per_system:
            # permute columns of speaker weights
            permuted = np.random.permutation(speaker.weights.T).T
            seen.add(tuple(permuted.flatten()))

        for permuted_weights in seen:
            permuted_speaker = copy.deepcopy(speaker)
            permuted_speaker.weights = np.reshape(
                permuted_weights, (speaker.weights.shape)
            )

            variant = permuted_speaker
            if languages is not None:
                variant = permuted_speaker.to_language()
            hypothetical_variants.append(variant)

    return hypothetical_variants


def generate_languages(
    language_class: Type[Language],
    expressions: list[Expression],
    lang_size: int,
    sample_size: int,
    criterion: Callable[[Expression], bool] = lambda *_: True,
    fixed_wordcount=False,
    dummy_name="sampled_lang_",
    id_start: int = 0,
    exact_sample=False,
    verbose=False,
) -> dict[str, Any]:
    """Generate languages by randomly sampling vocabularies as bags of expressions.

    A predicate (binary-valued property) of expressions may be supplied, which can be used to adjust the composition of vocabularies (e.g., by the percent of expressions satisfying the predicate).

    If sample size <= nCr, then take a random sample_size set of combinations. Otherwise, to prevent repeat languages, treat nCr as the sample size.

    Args:
        expressions: a list of the possible expressions to sample from.

        lang_size: the maximum (or exact) number of expressions in each language.

        sample_size: the number of languages to generate.

        criterion: the predicate, (e.g. semantic universal) by which to split the expressions into those satisfying and those not, and then sample languages with degrees of naturalness based on the percentage from those satisfying. Must apply at the expression level. By default is a trivial criterion, so that all expressions are 'quasi-natural'.

        fixed_wordcount: whether to vary the language size from 1 to lang_size.

        verbose: How detailed the progress of sampling should be, printed to stdout.

        dummy_name: the default name to give to each sampled language, e.g. `sampled_lang_42`. These should not collide with any actual natural language names if the efficient communication experiment does use natural language data.

        id_start: an integer representing the number of languages already generated in an experiment. Languages sampled will be named according to this number. For example, if id_start is 0, the first language sampled will be named `sampled_lang_0`. Note that the largest id does not necessarily track the actual size of languages saved for the experiment, but it does track how many languages have been generated.

        exact_sample: a boolean representing whether to sample until the exact sample size is filled. If True, the resulting pool of languages may not be unique.

        verbose: a boolean representing how verbose output should be during sampling.

    Returns:
        a dict representing the generated pool of languages and the updated id_start, of the form

            {
                "languages": (list of updated languages)
                "id_start": (updated length of languages)
            }

    Examples:

        >>> # Turn the knob on a universal property for modals
        >>> expressions = load_expressions(expressions_file)
        >>> universal_property = iff
        >>> result = generate_languages(
        ...    ModalLanguage,
        ...    expressions,
        ...    lang_size,
        ...    sample_size,
        ...    universal_property,
        ...)
        >>> languages = result["languages"]
        >>> id_start = result["id_start"]

    """
    # split the expressions
    natural_terms = []
    unnatural_terms = []
    for x in expressions:
        natural_terms.append(x) if criterion(x) else unnatural_terms.append(x)

    word_amounts = [lang_size] if fixed_wordcount else range(1, lang_size + 1)
    total_word_amount = len(expressions)
    word_amt_sample_size = int(sample_size / lang_size)

    expressions_indices = list(range(total_word_amount))
    languages = set()

    # For each language size
    for word_amount in word_amounts:
        # If sample size > all possible languages (of any degree), just generate the latter.
        if word_amt_sample_size > comb(total_word_amount, word_amount):
            if verbose:
                print(
                    f"Enumerating {word_amt_sample_size} languages of size {word_amount}"
                )
            result = enumerate_all_languages(
                language_class,
                id_start,
                expressions,
                expressions_indices,
                word_amount,
                dummy_name=dummy_name,
            )
            enumerated_langs = result["languages"]
            languages = languages.union(enumerated_langs)
            id_start = result["id_start"]

        # Otherwise, take random sample
        else:
            if verbose:
                print(
                    f"Generating {word_amt_sample_size} languages of size {word_amount}"
                )
            result = sample_quasi_natural(
                language_class=language_class,
                natural_terms=natural_terms,
                unnatural_terms=unnatural_terms,
                lang_size=word_amount,
                sample_size=word_amt_sample_size,
                id_start=id_start,
                dummy_name=dummy_name,
                verbose=verbose,
            )

            rlangs = result["languages"]
            id_start = result["id_start"]
            languages = languages.union(rlangs)

    if exact_sample:
        # Randomly choose a lang size and continue sampling until sample_size achieveds
        additional_sample = sample_size - len(languages)
        if verbose:
            print(f"Sampled {len(languages)} out of {sample_size} languages.")
        while additional_sample > 0:
            word_amount = random.choice(word_amounts)
            if verbose:
                print(
                    f"Filling remaining languages by sampling {additional_sample} languages of size {word_amount}"
                )

            rlangs = sample_quasi_natural(
                language_class,
                natural_terms,
                unnatural_terms,
                word_amount,
                additional_sample,
                id_start,
                dummy_name=dummy_name,
                verbose=verbose,
            )
            languages = languages.union(rlangs)
            additional_sample = sample_size - len(languages)
            print(additional_sample)

    return {
        "languages": list(languages)[:sample_size],
        "id_start": id_start,
    }


##############################################################################
# Different sampling methods
##############################################################################


def sample_lang_size(
    language_class: Type[Language],
    expressions: list[Expression],
    lang_size: int,
    sample_size: int,
    id_start: int = 0,
    verbose=False,
    dummy_name="sampled_lang_id",
) -> list[Language]:
    """Get a sample of languages each of exactly lang_size.

    Args:
        language_class: a subclass of altk.Language

        expressions: a list of Expressions to sample from

        lang_size: int representing the maximum language size to sample

        sample_size: int representing the number of total languages to return

        id_start: an int representing the number of languages already generated in an experiment.

    Returns:
        a dict containing the randomly sampled languages and the updated id_start, of the form

            {
                "languages": (list of updated languages)
                "id_start": (updated length of languages)
            }
    """
    result = sample_quasi_natural(
        language_class=language_class,
        natural_terms=expressions,
        unnatural_terms=[],
        lang_size=lang_size,
        sample_size=sample_size,
        id_start=id_start,
        dummy_name=dummy_name,
        verbose=verbose,
    )
    return {
        "languages": list(result["languages"]),
        "id_start": result["id_start"],
    }


def sample_quasi_natural(
    language_class: Type[Language],
    natural_terms: list[Expression],
    unnatural_terms: list[Expression],
    lang_size: int,
    sample_size: int,
    id_start: int,
    dummy_name="sampled_lang_id",
    verbose=False,
) -> dict[str, Any]:
    """Turn the knob on degree quasi-naturalness for a sample of languages, either by enumerating or randomly sampling unique subsets of all possible combinations.

    Args:
        natural_terms: expressions satisfying some criteria of quasi-naturalness, e.g, a semantic universal.

        unnatural_terms: expressions not satisfying the criteria.

        lang_size: the exact number of expressions a language must have.

        sample_size: how many languages to sample.

    Returns:
        a dict containing the randomly sampled quasi-natural languages and the updated id_start, of the form

            {
                "languages": (list of updated languages)
                "id_start": (updated length of languages)
            }
    """
    languages = set()

    natural_indices = list(range(len(natural_terms)))
    unnatural_indices = list(range(len(unnatural_terms)))

    # by default, expresions:= natural_terms, i.e. all degree-1.0
    degrees = [lang_size]
    if unnatural_terms:
        degrees = list(range(lang_size + 1))
    degree_sample_size = int(np.ceil(sample_size / len(degrees)))

    # For each fraction of the lang size
    for num_natural in tqdm(degrees):
        num_unnatural = 0
        if unnatural_terms:
            num_unnatural = lang_size - num_natural

        # If sample size > possible languages, just generate the latter.
        possible_langs = comb(len(natural_terms), num_natural) * comb(
            len(unnatural_terms), num_unnatural
        )
        if not possible_langs:
            raise ValueError(
                f"combinations is 0: check comb({len(natural_terms)}, {num_natural}) * comb({len(unnatural_terms)}, {num_unnatural})"
            )

        if degree_sample_size > possible_langs:
            if verbose:
                print(
                    f"Enumerating {possible_langs} for degree {num_natural/lang_size}"
                )
            result = enumerate_all_languages(
                language_class,
                id_start,
                natural_terms,
                natural_indices,
                num_natural,
                unnatural_terms,
                unnatural_indices,
                num_unnatural,
            )
            enumerated_langs = result["languages"]
            languages = languages.union(enumerated_langs)
            id_start = result["id_start"]

        # Otherwise, take a random sample
        else:
            if verbose:
                print(
                    f"Sampling {degree_sample_size} languages of size {lang_size} with degree {num_natural/lang_size}"
                )

            # Sample unique languages
            seen = set()
            for _ in range(degree_sample_size):
                vocabulary = random_combination_vocabulary(
                    seen,
                    num_natural,
                    natural_terms,
                    num_unnatural,
                    unnatural_terms,
                )
                id_start += 1
                language = language_class(
                    vocabulary, name=rename_id(dummy_name, id_start)
                )
                languages.add(language)

    assert len(languages) == len(set(languages))
    return {
        "languages": languages,
        "id_start": id_start,
    }


##############################################################################
# Helper functions for generating languages
##############################################################################


def rename_id(name: str, id: int) -> str:
    """Updates a string of form `sampled_lang_X` with a new id for X."""
    return "".join([c for c in name if not c.isdigit()] + [str(id)])


def enumerate_all_languages(
    language_class: Type[Language],
    id_start: int,
    natural_terms: list[Expression],
    natural_indices: list[int],
    num_natural: int = 0,
    unnatural_terms: list[Expression] = [],
    unnatural_indices: list[int] = [],
    num_unnatural: int = 0,
    dummy_name="sampled_lang_id",
    verbose=False,
) -> dict[str, Any]:
    """When the sample size requested is greater than the size of all possible languages, just enumerate all the possible languages.

    Args:
        language_class: the kind of Language to construct

        id_start: a number to start counting from for assigning names with numerical ids to languages.

        natural_indices: the indices of quasi-natural languages already seen

        num_natural: the number of quasi-natural languages to sample

        natural_terms: the list of quasi-natural terms to sample from

        unnatural_indices: the indices of non-quasi-natural languages already seen

        num_unnatural: the number of non-quasi-natural languages to sample; 0 by default

        unnatural_terms: the list of non-quasi-natural terms to sample from; empty by default.

        dummy_name: the format of the string to name each language constructed.

    Returns:
        a dict containing a set of languages and the updated id_start,  of the form

            {
                "languages": (list of updated languages)
                "id_start": (updated length of languages)
            }
    """
    # combinations is invariant to order
    natural_subsets = list(combinations(natural_indices, num_natural))
    unnatural_subsets = list(combinations(unnatural_indices, num_unnatural))

    languages = set()
    # Construct the languages
    for natural_subset in natural_subsets:
        for unnatural_subset in unnatural_subsets:
            vocabulary = [natural_terms[idx] for idx in natural_subset] + [
                unnatural_terms[idx] for idx in unnatural_subset
            ]
            id_start += 1
            language = language_class(vocabulary, name=rename_id(dummy_name, id_start))
            languages.add(language)
    return {
        "languages": languages,
        "id_start": id_start,
    }


def random_combination_vocabulary(
    seen: set,
    num_natural: int,
    natural_terms: list[Expression],
    num_unnatural: int = 0,
    unnatural_terms: list[Expression] = [],
) -> list[Language]:
    """Get a single vocabulary for a specific language size by choosing a random combination of natural and unnatural terms.

    Args:
        seen: the list of language indices already seen

        num_natural: int

        natural_terms: list[Expression]

        num_unnatural: int=0

        unnatural_terms: list[Expression]=[]

    Returns:
        languages: the extended list of input languages.
    """
    while True:
        nat_sample_indices = tuple(
            sorted(random.sample(range(len(natural_terms)), num_natural))
        )
        unnat_sample_indices = ()
        if unnatural_terms:
            unnat_sample_indices = tuple(
                sorted(random.sample(range(len(unnatural_terms)), num_unnatural))
            )
        sample_indices = (nat_sample_indices, unnat_sample_indices)
        if sample_indices not in seen:
            # keep track of languages chosen
            seen.add(sample_indices)

        # Add language
        vocabulary = [natural_terms[idx] for idx in nat_sample_indices] + [
            unnatural_terms[idx] for idx in unnat_sample_indices
        ]
        break
    return vocabulary
