"""Functions for measuring informativity in efficient communication analyses of languages."""

import numpy as np
from cmath import isclose
from typing import Callable
from altk.language.language import Language
from altk.language.semantics import Referent
from altk.effcomm.agent import Speaker, Listener, LiteralListener, LiteralSpeaker, PragmaticSpeaker, PragmaticListener
from altk.effcomm.util import build_utility_matrix

def informativity(
    language: Language,
    prior: np.ndarray,
    utility: Callable[[Referent, Referent], float],
    agent_type: str = "literal",
) -> float:
    """The informativity of a language is identified with the successful communication between a Sender and a Receiver.

    This function is a wrapper for `communicative_success`.

    Args:
        language: the language to compute informativity of.

        prior: a probability distribution representing communicative need (frequency) for meanings.

        utility: a function representing the usefulness of listener guesses about speaker meanings, e.g. meaning similarity. To reward only exact recovery of meanings, pass the an indicator function.

        kind: {"literal, pragmatic"} Whether to measure informativity using literal or pragmatic agents, as canonically described in the Rational Speech Act framework. The default is "literal".

    _Concepts_
        The Sender can be thought of as a conditional distribution over expressions given meanings. The Receiver is likewise a conditional distribution over meanings given expressions. The communicative need, or cognitive source, is a prior probability over meanings representing how frequently agents need to use certain meanings in communication. The utility function represents the similarity, or appropriateness, of the Receiver's guess m' about the Sender's intended meaning m.

    _Formula_
        The informativity of a language $L$ with meaning space $M$ is defined:

        $I(L) := \sum_{m \in M} p(m) \sum_{i \in L} p(i|m) \sum_{m' \in i} p(m'|i) * u(m, m')$

    _Bounds_
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

        I(L) = P(m, m') * u(m, m')

             = \sum_{m \in M} p(m) \sum_{i \in L} p(i|m) \sum_{m' \in i} p(m'|i) * u(m, m')

             = sum(diag(p)SR * u)

        For more details, see the docs/vectorized_informativity.pdf.

    Args:
        - speaker: a literal or pragmatic speaker, containing a matrix S for P(e | m)

        - listener: a literal or pragmatic listener, containing a matrix R for P(m | e)

        - prior: p(m), distribution over meanings representing communicative need

        - utility: a function u(m, m') representing similarity of meanings, or pair-wise usefulness of listener guesses about speaker meanings.
    """
    S = speaker.normalized_weights()
    R = listener.normalized_weights()
    U = build_utility_matrix(speaker.language.universe, utility)
    return float(np.sum(np.diag(prior) @ S @ R * U))
