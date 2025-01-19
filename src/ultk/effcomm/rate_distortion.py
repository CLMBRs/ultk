"""Helper functions for Rate-Distortion based (including Information Bottleneck) efficient communication analyses."""

import numpy as np
from ultk.language.language import Language
from ultk.effcomm.agent import LiteralSpeaker, Listener
from rdot.optimizers import IBOptimizer, IBResult
from rdot.probability import joint, bayes
from rdot.information import information_cond, mutual_info
from rdot.distortions import ib_kl

##############################################################################
# Measuring Languages
##############################################################################


def language_to_ib_point(
    language: Language,
    prior: np.ndarray,
    meaning_dists: np.ndarray,
) -> tuple[float]:
    """Compute the complexity, informativity, and communicative cost under the IB framework (Zaslavsky et al., 2018, i.a.).

    Args:
        language: the Language to measure. This function will infer an IB encoder, which will be represented as a 2D array of shape `(|words|, |meanings|)`

        meaning_dists: array of shape `(|meanings|, |meanings|)` representing the distribution over world states given meanings.

        prior: array of shape `|meanings|` representing the communicative need distribution

    Returns:
        a tuple of floats `(complexity, accuracy, distortion)`, s.t.

            `complexity`: the complexity of the lexicon I[M:W], in bits

            `accuracy`: the accuracy of the lexicon I[W:U], in bits

            `distortion`: the distortion E[DKL[ M || M_hat ]] = I[M:U] - I[W:U], in bits
    """
    result = language_to_ib_encoder_decoder(
        language=language,
        prior=prior,
        meaning_dists=meaning_dists,
    )
    return ib_encoder_to_point(
        prior=prior,
        meaning_dists=meaning_dists,
        encoder=result["encoder"].normalized_weights(),
        decoder=result["decoder"].normalized_weights(),
    )


def language_to_ib_encoder_decoder(
    language: Language,
    prior: np.ndarray,
    meaning_dists: np.ndarray,
) -> dict[str, np.ndarray]:
    """Convert a Language, a mapping of words to meanings, to IB encoder, q(w|m) and IB decoder q(m|w).

    A Bayesian decoder chooses an interpretation according to p(meaning | word), where

    $P(m | w) = \\frac{P(M | W) \cdot P(M)} { P(W) }$

    Furthermore, we will require that each word w is deterministically interpreted as meaning $\hat{m}$ as follows:

    $\hat{m}_{w}(u) = \sum_m p(m|w) \cdot m(u)$

    See https://github.com/nogazs/ib-color-naming/blob/master/src/ib_naming_model.py#L40.

    Args:
        language: the lexicon from which to infer a speaker (encoder).

        prior: array of shape `|meanings|` representing the communicative need distribution

        meaning_dists: array of shape `(|meanings|, |meanings|)` representing the distribution over world states given meanings.

    Returns:
        a dict of the form
        {
            "encoder": np.ndarray of shape `(|meanings|, |words|)`,
            "decoder": np.ndarray of shape `(|words|, |meanings|)`,
        }
    """
    # In the IB framework, the encoder _can_ be a literal speaker and the decoder is a bayes optimal listener.
    speaker = LiteralSpeaker(language)
    speaker.weights = rows_zero_to_uniform(speaker.normalized_weights())
    listener = Listener(language)
    listener.weights = ib_optimal_decoder(speaker.weights, prior, meaning_dists)
    return {
        "encoder": speaker,
        "decoder": listener,
    }


def ib_encoder_to_point(
    prior: np.ndarray,
    meaning_dists: np.ndarray,
    encoder: np.ndarray,
    decoder: np.ndarray = None,
) -> tuple[float]:
    """Get (complexity, accuracy, comm_cost) IB coordinates.

    Args:
        prior: array of shape `|meanings|` representing the communicative need distribution

        meaning_dists: array of shape `(|meanings|, |meanings|)` representing the distribution over world states given meanings.

        encoder: array of shape `(|meanings|, |words|)` representing P(W | M)

        decoder: array of shape `(|words|, |meanings|)` representing P(M | W).  By default is None, and the Bayesian optimal decoder will be inferred.

    Returns:
        a tuple of floats corresponding to `(complexity, accuracy, comm_cost)`.
    """

    if decoder is None:
        decoder = ib_optimal_decoder(encoder, prior, meaning_dists)

    # Unclear that the below is correct
    # encoder = rows_zero_to_uniform(encoder)
    # decoder = rows_zero_to_uniform(decoder)

    # IB complexity = info rate of encoder = I(meanings; words)
    complexity = information_cond(prior, encoder)

    # IB accuracy/informativity = I(words; world states)
    pMW = encoder * prior
    pWU = pMW.T @ meaning_dists
    accuracy = mutual_info(pWU)

    # expected distortion
    I_mu = information_cond(prior, meaning_dists)
    distortion = I_mu - accuracy

    # TODO: the above is only IB optimal; should we look at the emergent listener accuracy? To do that we'll need to compute kl divergence
    # and then do I(M;U) - distortion to get the accuracy.

    # pu_w = decoder @ meaning_dists
    # dist_mat = ib_kl(meaning_dists, pu_w,) # getting infs; I confirmed that this because there exists an x s.t. p(x) > 0 but q(x) = 0. Ask Noga what to do here. Add a little epsilon?
    # distortion = np.sum( prior * ( encoder @ decoder ) * dist_mat )

    decoder_smoothed = decoder + 1e-20
    decoder_smoothed /= decoder_smoothed.sum(axis=1, keepdims=True)
    pu_w = decoder_smoothed @ meaning_dists
    dist_mat = ib_kl(
        meaning_dists,
        pu_w,
    )
    distortion = np.sum(prior * (encoder @ decoder) * dist_mat)
    # but this measure of distortion is almost an order magnitude higher than bayesian decoder
    # breakpoint()

    return (complexity, accuracy, distortion)


def ib_optimal_decoder(
    encoder: np.ndarray,
    prior: np.ndarray,
    meaning_dists: np.ndarray,
) -> np.ndarray:
    """Compute the bayesian optimal decoder. See https://github.com/nogazs/ib-color-naming/blob/master/src/ib_naming_model.py#L40

    Args:
        encoder: array of shape `(|words|, |meanings|)`

        prior: array of shape `(|meanings|,)`

        meaning_dists: array of shape `(|meanings|, |meanings|)`

    Returns:
        array of shape `(|words|, |meanings|)` representing the 'optimal' deterministic decoder
    """
    return bayes(encoder, prior) @ meaning_dists


##############################################################################
# Estimating bounds for a domain
##############################################################################


def get_ib_bound(
    prior: np.ndarray,
    meaning_dists: np.ndarray,
    *args,
    betas: np.ndarray = np.logspace(-2, 5, 30),
    **kwargs,
) -> list[IBResult]:
    """Estimate the IB theoretical bound for a domain, specified by a prior over meanings and (perceptually uncertain) meaning distributions.

    Args:
        meaning_dists: array of shape `(|meanings|, |meanings|)` representing the distribution over world states given meanings.

        prior: array of shape `|meanings|` representing the communicative need distribution

        args: propagated to `IBOptimizer`

        kwargs: propagated to `IBOptimizer`

    Returns:
        a list of `rdot.ba.IBResult` namedtuples.
    """
    return IBOptimizer(
        joint(meaning_dists, prior),
        betas,
        *args,
        **kwargs,
    ).get_results()


##############################################################################
# Helper functions for working with stochastic matrices
##############################################################################


def rows_zero_to_uniform(mat: np.ndarray) -> np.ndarray:
    """Ensure that `mat` encodes a probability distribution, i.e. each row (indexed by a meaning) is a distribution over expressions: sums to exactly 1.0.

    This is necessary when exploring mathematically possible languages (including natural languages, like Hausa in the case of modals) which sometimes have that a row of the matrix p(word|meaning) is a vector of 0s.

    Args:
        mat: a 2D numpy array that should be normalized so that each row is a probability distribution.
    """
    mat = np.array(mat)

    threshold = 1e-5

    # Ensure if p(.|meaning) sums to > 0 at all, it must sum to 1.
    for row in mat:
        # less than 1.0
        if row.sum() and 1.0 - row.sum() > threshold:
            print("row is nonzero and sums to less than 1.0!")
            print(row, row.sum())
            raise Exception
        # greater than 1.0
        if row.sum() and row.sum() - 1.0 > threshold:
            print("row sums to greater than 1.0!")
            print(row, row.sum())
            raise Exception

    return np.array([row if row.sum() else np.ones(len(row)) / len(row) for row in mat])
