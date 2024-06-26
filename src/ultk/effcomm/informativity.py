"""Functions for measuring informativity in efficient communication analyses of languages."""

import numpy as np
from cmath import isclose
from typing import Callable
from ultk.language.language import Language
from ultk.language.semantics import Referent, Universe
from ultk.effcomm.agent import (
    Speaker,
    Listener,
    LiteralListener,
    LiteralSpeaker,
    PragmaticSpeaker,
    PragmaticListener,
)


def build_utility_matrix(
    universe: Universe, utility: Callable[[Referent, Referent], float]
) -> np.ndarray:
    """Construct the square matrix specifying the utility function defined for pairs of meanings, used for computing communicative success."""
    return np.array(
        [
            [utility(ref, ref_) for ref_ in universe.referents]
            for ref in universe.referents
        ]
    )


def indicator_utility(ref1: Referent, ref2: Referent) -> float:
    """Indicator utility function, i.e. delta.  Returns 1.0 iff ref1 equals ref2."""
    return float(ref1 == ref2)


def informativity(
    language: Language,
    prior: np.ndarray,
    utility: Callable[[Referent, Referent], float] = indicator_utility,
    agent_type: str = "literal",
) -> float:
    """The informativity of a language is identified with the successful communication between a speaker and a listener.

    This function is a wrapper for `communicative_success`.

    Args:
        language: the language to compute informativity of.

        prior: a probability distribution representing communicative need (frequency) for Referents.

        utility: a function representing the usefulness of listener guesses about speaker Referents, e.g. Referent similarity. To reward only exact recovery of meanings, use the indicator function (default).

        kind: {"literal, pragmatic"} Whether to measure informativity using literal or pragmatic agents, as canonically described in the Rational Speech Act framework. The default is "literal".

    *Concepts*:
        The speaker can be thought of as a conditional distribution over expressions given meanings. The listener is likewise a conditional distribution over meanings given expressions. The communicative need, or cognitive source, is a prior probability over meanings representing how frequently agents need to use certain meanings in communication. The utility function represents the similarity, or appropriateness, of the listener's guess m' about the speaker's intended meaning m.

    *Formula*:
        The informativity of a language $L$ with meaning space $M$ is defined:

    $I(L) := \sum_{m \in M} p(m) \sum_{i \in L} p(i|m) \sum_{\hat{m} \in i} p(\hat{m}|i) \cdot u(m, \hat{m})$

    *Bounds*:
        A perfectly informative (=1.0) language can be constructed with a exactly one expression for each meaning.

        For u() = indicator(), every language has nonzero informativity because a language must contain at least one expression, and an expression must contain at least one meaning.
    """
    if not language.expressions:
        raise ValueError(f"language empty: {language}")

    speaker = LiteralSpeaker(language)
    listener = LiteralListener(language)

    if agent_type == "literal":
        pass
    elif agent_type == "pragmatic":
        speaker = PragmaticSpeaker(language, listener)
        listener = PragmaticListener(language, speaker, np.diag(prior))
    else:
        raise ValueError(
            f"agent_type must be either 'literal' or 'pragmatic'. Received: {agent_type}."
        )

    inf = communicative_success(speaker, listener, prior, utility)
    # print(inf)

    # Check informativity > 0
    utility_matrix = build_utility_matrix(speaker.language.universe, utility)
    m, _ = utility_matrix.shape  # square matrix
    if np.array_equal(utility, np.eye(m)):
        if isclose(inf, 0.0):
            raise ValueError(
                f"Informativity must be nonzero for indicator utility reward function, but was: {inf}"
            )

    return inf


def communicative_success(
    speaker: Speaker,
    listener: Listener,
    prior: np.ndarray,
    utility: Callable[[Referent, Referent], float],
) -> float:
    """Helper function to compute the literal informativity of a language.


    $I(L) = \sum_{m, \hat{m}} P(m, \hat{m}) \cdot u(m, \hat{m})$

    $ = \sum_{m \in M} p(m) \sum_{i \in L} p(i|m) \sum_{\hat{m} \in i} p(\hat{m} |i) \cdot u(m, m')$

    $ = \sum \\text{diag}(p)SR \odot U $

    For more details, see [docs/vectorized_informativity](https://github.com/CLMBRs/altk/blob/main/docs/vectorized_informativity.pdf).

    Args:
        speaker: a literal or pragmatic speaker, containing a matrix S for P(e | m)

        listener: a literal or pragmatic listener, containing a matrix R for P(m | e)

        prior: p(m), distribution over meanings representing communicative need

        utility: a function u(m, m') representing similarity of meanings, or pair-wise usefulness of listener guesses about speaker meanings.
    """
    S = speaker.normalized_weights()
    R = listener.normalized_weights()
    U = build_utility_matrix(speaker.language.universe, utility)
    return float(np.sum(np.diag(prior) @ S @ R * U))
