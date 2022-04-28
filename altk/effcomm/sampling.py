"""Functions for sampling expressions into languages.
"""

import random
import numpy as np
from altk.language.language import Language, Expression
from typing import Callable, Type
from math import comb
from itertools import combinations
from tqdm import tqdm


def generate_languages(
    language_class: Type,
    expressions: list[Expression],
    lang_size: int,
    sample_size: int,
    criterion: Callable = lambda *_: True,
    fixed_wordcount=False,
    dummy_name="sampled_lang_id",
    exact_sample=True,
    verbose=False,
) -> list[Language]:
    """Generate languages by randomly sampling bags of expressions.

    If sample size <= nCr, then take a random sample_size set of combinations. Otherwise, to prevent repeat languages, treat nCr as the sample size.

    Args:
        expressions: a list of the possible modal expressions to sample from.

        lang_size: the maximum (or exact) number of expressions in each language.

        sample_size: the number of languages to generate.

        criterion: the predicate, (e.g. semantic universal) by which to split the expressions into those satisfying and those not, and then sample languages with degrees of naturalness based on the percentage from those satisfying. Must apply at the expression level.

        fixed_wordcount: whether to vary the language size from 1 to lang_size.

        verbose: How detailed the progress of sampling should be, printed to stdout.

        dummy_name: the default name to give to each sampled language, e.g. `sampled_lang_42'. These should not collide with any actual natural language names if the efficient communication experiment does use natural language data.
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
    languages = []

    # For each language size
    for word_amount in word_amounts:
        # If sample size > all possible languages (of any degree), just generate the latter.
        if word_amt_sample_size > comb(total_word_amount, word_amount):
            languages = extend_languages_by_enumeration(
                language_class,
                languages,
                expressions,
                expressions_indices,
                word_amount,
                dummy_name=dummy_name,
            )

        # Otherwise, take random sample
        else:
            if verbose:
                print(
                    f"Generating {word_amt_sample_size} languages of size {word_amount}"
                )
            rlangs = sample_quasi_natural(
                language_class,
                natural_terms,
                unnatural_terms,
                word_amount,
                word_amt_sample_size,
                dummy_name=dummy_name,
                verbose=verbose,
            )
            languages.extend(rlangs)

    if exact_sample:
        # Randomly choose a lang size and continue sampling until sample_size achieveds
        additional_sample = sample_size - len(languages)
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
                dummy_name=dummy_name,
                verbose=verbose,
            )
            languages.extend(rlangs)
            additional_sample = sample_size - len(languages)
        
        languages = languages[:sample_size]

    return languages


##############################################################################
# Different sampling methods
##############################################################################


def sample_lang_size(
    language_class: Type,
    expressions: list[Expression],
    lang_size,
    sample_size,
    verbose=False,
    dummy_name="sampled_lang_id",
) -> list[Language]:
    """Get a sample of languages each of exactly lang_size.

    Args:
        language_class:

        expressions:

        lang_size:

        sample_size:
    """
    return sample_quasi_natural(
        language_class=language_class,
        natural_terms=expressions,
        unnatural_terms=[],
        lang_size=lang_size,
        sample_size=sample_size,
        dummy_name=dummy_name,
        verbose=verbose,
    )


def sample_quasi_natural(
    language_class: Type,
    natural_terms: list[Expression],
    unnatural_terms: list[Expression],
    lang_size,
    sample_size,
    dummy_name="sampled_lang_id",
    verbose=False,
) -> list[Language]:
    """Turn the knob on degree quasi-naturalness for a sample of languages, either by enumerating or randomly sampling unique subsets of all possible combinations.

    Args:
        natural_terms: expressions satisfying some criteria of quasi-naturalness, e.g, a semantic universal.

        unnatural_terms: expressions not satisfying the criteria.

        lang_size: the exact number of expressions a language must have.

        sample_size: how many languages to sample.
    """
    languages = []

    natural_indices = list(range(len(natural_terms)))
    unnatural_indices = list(range(len(unnatural_terms)))

    # by default, expresions:= natural_terms
    degrees = [1 for _ in range(lang_size + 1)]
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
            languages = extend_languages_by_enumeration(
                language_class,
                languages,
                natural_terms,
                natural_indices,
                num_natural,
                unnatural_terms,
                unnatural_indices,
                num_unnatural,
            )

        # Otherwise, take a random sample
        else:
            if verbose:
                print(
                    f"Sampling {degree_sample_size} languages of size {lang_size} with degree {num_natural/lang_size}"
                )

            # Sample unique languages
            seen = []
            for _ in range(degree_sample_size):
                vocabulary = random_combination_vocabulary(
                    seen,
                    num_natural,
                    natural_terms,
                    num_unnatural,
                    unnatural_terms,
                )
                language = language_class(
                    vocabulary, name=f"{dummy_name.replace('id', '')}{len(languages)}"
                )
                languages.append(language)

    assert len(languages) == len(set(languages))
    return languages


##############################################################################
# Helper functions for generating languages
##############################################################################


def extend_languages_by_enumeration(
    language_class: Type,
    languages: list[Language],
    natural_terms: list[Expression],
    natural_indices: list[int],
    num_natural: int,
    unnatural_terms: list[Expression] = [],
    unnatural_indices: list[int] = [],
    num_unnatural: int = 0,
    dummy_name="sampled_lang_id",
    verbose=False,
) -> list[Language]:
    """When the sample size requested is greater than the size of all possible languages, just enumerate all the possible languages and extend the input list of languages with result.

    Args:
        language_class:

        languages: list[Language]

        natural_indices: list[int]

        num_natural: int

        natural_terms: list[Expression]

        unnatural_indices: list[int]=[]

        num_unnatural: int=0

        unnatural_terms: list[Expression]=[]

    Returns:
        languages: the extended list of input languages.
    """
    natural_subsets = list(combinations(natural_indices, num_natural))
    unnatural_subsets = list(combinations(unnatural_indices, num_unnatural))

    # Construct the languages
    for natural_subset in natural_subsets:
        for unnatural_subset in unnatural_subsets:

            vocabulary = [natural_terms[idx] for idx in natural_subset] + [
                unnatural_terms[idx] for idx in unnatural_subset
            ]

            language = language_class(
                vocabulary, name=f"{dummy_name.replace('id', '')}{len(languages)}"
            )
            languages.append(language)
    return languages


def random_combination_vocabulary(
    seen: list,
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
        nat_sample_indices = sorted(
            random.sample(range(len(natural_terms)), num_natural)
        )
        unnat_sample_indices = []
        if unnatural_terms:
            unnat_sample_indices = sorted(
                random.sample(range(len(unnatural_terms)), num_unnatural)
            )
        sample_indices = (nat_sample_indices, unnat_sample_indices)
        if sample_indices not in seen:
            # keep track of languages chosen
            seen.append(sample_indices)

        # Add language
        vocabulary = [natural_terms[idx] for idx in nat_sample_indices] + [
            unnatural_terms[idx] for idx in unnat_sample_indices
        ]
        break
    return vocabulary
