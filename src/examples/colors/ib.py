import numpy as np

from ultk.language.language import Language

# TODO: rename and refactor this file!


def language_to_encoder(language: Language) -> np.ndarray:
    """Convert a Language to an IB encoder, q(w|m).

    Args:
        language: the lexicon from which to infer a speaker (encoder).
        It is assumed that each `Expression` in the `Language` has a `Meaning` that maps to floats.

    Returns:
        a numpy array of shape `(|referents|, |words|)`
        where element (r, w) is the probability of word w given referent r
    """
    universe = language.universe
    # (|referents|, |words|): p(r | w)
    prob_chip_given_expression = np.array(
        [
            [expression.meaning[referent] for expression in language.expressions]
            for referent in universe.referents
        ]
    )
    # (|referents|, 1): p(r)
    # prior = np.array(universe.prior)[:, None]
    # TODO: get encoder from prob_chip_given_expression (i.e. prob_expression_given_chip)
    encoder = prob_chip_given_expression
    print(encoder)
    return encoder
