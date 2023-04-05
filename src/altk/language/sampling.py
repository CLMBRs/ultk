import random
from math import comb
import numpy as np
from altk.language.language import Language, Expression
from altk.language.semantics import Meaning, Universe
from typing import Callable, Generator, Iterable, Type, Any
from itertools import chain, combinations
from tqdm import tqdm


def powerset(iterable: Iterable, max_size: int = None) -> Iterable:
    """Enumerate all _non-empty_ subsets of an iterable up to a given maximum size, e.g.:
    powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    lightly adapted from itertools Recipes at
    https://docs.python.org/3/library/itertools.html#itertools-recipes

    Args:
        iterable: elements from which to form subsets
        max_size: largest subsets (inclusive) to return

    Returns:
        iterator over all subsets from size 1 to `max_size` of elements from `iterable`
    """
    s = list(iterable)
    if max_size is None:
        max_size = len(s)
    return chain.from_iterable(combinations(s, r) for r in range(1, max_size + 1))


def all_meanings(universe: Universe) -> Generator[Meaning, None, None]:
    """Generate all Meanings (sets of Referents) from a given Universe."""
    referents = universe.referents
    for refset in powerset(referents):
        yield Meaning(refset, universe)


def all_expressions(meanings: Iterable[Meaning]) -> Generator[Expression, None, None]:
    """Generate Expressions from an iterable of Meanings."""
    # TODO: allow different subclasses of Expression, and kwargs?
    for idx, meaning in enumerate(meanings):
        yield Expression(f"expr-{idx}", meaning)


def all_languages(
    expressions: Iterable[Expression],
    language_class: Type[Language] = Language,
    max_size: int = None,
) -> Generator[Language, None, None]:
    """Generate all Languages (sets of Expressions) from a given set of Expressions.

    Args:
        expressions: iterable of all possible expressions
        language_class: the type of language to generate
        max_size: largest size for a language; if None, all subsets of expressions will be used

    Yields:
        Languages with subsets of Expressions from `expressions`
    """
    for exprset in powerset(expressions, max_size):
        yield language_class(tuple(exprset))


def upto_comb(num: int, max_k: int) -> int:
    """Return the number of ways of choosing _up to max_k_ items from
    n items without repetition.  Just an iterator of math.comb for n from
    1 to max_k."""
    return sum(comb(num, k) for k in range(1, max_k + 1))


def random_languages(
    expressions: Iterable[Expression],
    sampling_strategy: str = "uniform",
    sample_size: int = None,
    language_class: Type[Language] = Language,
    max_size: int = None,
) -> list[Language]:
    """Generate unique Languages by randomly sampling subsets of Expressions, either in a uniform or stratified way.
    If there are fewer than `sample_size` possible Languages up to size `max_size`,
    this method will just return all languages up to that size (and so the sample may
    be smaller than `sample_size`).

    Some use cases:
        - With `sample_size=None`, get all languages.
            >>> random_languages(expressions)
        - With `sample_size` and uniform sampling, get random languages:
            >>> random_languages(expressions, sample_size=1000)
        - Stratified sampling, with and without a `max_size`:
            >>> random_languages(expressions, sample_size=1000, sampling_strategy="stratified")
            >>> random_languages(expressions, sample_size=1000, sampling_strategy="stratified", max_size=10)

    Args:
        expressions: all possible expressions
        sampling_strategy: how to sample subsets of expressions
            uniform: for every expression, choose whether or not to include it in a given language
            stratified: first sample a size for a Language, then choose that many random Expressions
                (i) this has the effect of "upsampling" from smaller Language sizes
                (ii) this can be used with `max_size` to only generate Languages up to a given number of expressions
        sample_size: how many languages to return
            if None, will return all languages up to `max_size`
        language_class: type of Language
        max_size: largest possible Language to generate
            if None, will be the length of `expressions`
            NB: this argument has no effect when `sampling_strategy` is "uniform"

    Returns:
        a list of randomly sampled Languages
    """
    # TODO: update docstring
    if sampling_strategy not in ("uniform", "stratified"):
        raise ValueError("Only 'uniform' and 'stratified' sampling are supported.")
    expressions = list(expressions)
    num_expr = len(expressions)
    if max_size is None:
        max_size = num_expr
    num_subsets = upto_comb(num_expr, max_size)
    if sample_size is None or num_subsets < sample_size:
        print(f"Due to argument combination, returning all languages.")
        return list(
            all_languages(expressions, language_class=language_class, max_size=max_size)
        )
    languages = []
    subsets = set()
    while len(languages) < sample_size:
        if sampling_strategy == "stratified":
            lang_size = random.randint(1, max_size)
            expr_indices = tuple(sorted(random.sample(range(num_expr), lang_size)))
        elif sampling_strategy == "uniform":
            expr_indices = tuple(
                [idx for idx in range(num_expr) if random.choice((True, False))]
            )
        if expr_indices not in subsets:
            subsets.add(expr_indices)
            languages.append(language_class([expressions[idx] for idx in expr_indices]))
    return languages


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

    return {"languages": list(languages)[:sample_size], "id_start": id_start}


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
    return {"languages": list(result["languages"]), "id_start": result["id_start"]}


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
                    seen, num_natural, natural_terms, num_unnatural, unnatural_terms
                )
                id_start += 1
                language = language_class(
                    vocabulary, name=rename_id(dummy_name, id_start)
                )
                languages.add(language)

    assert len(languages) == len(set(languages))
    return {"languages": languages, "id_start": id_start}


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
    return {"languages": languages, "id_start": id_start}


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
