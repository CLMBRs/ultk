"""Functions for measuring informativity in efficient communication analyses of languages."""

import numpy as np
from abc import abstractmethod
from cmath import isclose
from typing import Callable, Iterable
from altk.language.language import Language
from altk.language.semantics import Meaning, Universe
from altk.effcomm.agent import *


##############################################################################
# Informativity Classes
##############################################################################


class InformativityMeasure:
    def __init__(self):
        raise NotImplementedError

    def batch_communicative_cost(self, langs: list[Language]) -> list[float]:
        """Measure the (1 - informativity) of a list of languages."""
        return [1 - self.language_informativity(lang) for lang in langs]

    def batch_informativity(self, langs: list[Language]) -> list[float]:
        """Measure the informativity of a list of languages."""
        return [self.language_informativity(lang) for lang in langs]

    @abstractmethod
    def language_informativity(self, language: Language) -> float:
        """Measure the informativity of a single language."""
        raise NotImplementedError


class SST_Informativity_Measure(InformativityMeasure):

    """Class for computing the measure of informativeness for a language based on the communicative success of speakers and listeners as described in Steinert-Threlkeld (2021)."""

    def __init__(self, prior: Iterable, utility: np.ndarray, agent_type="literal"):
        """Initialize the informativity measure with a utility function and prior.

        Args:
            prior: a probability distribution representing communicative need (frequency) for meanings.

            utility: a 2d numpy array of size |meanings| by |meanings|, containing the function representing the usefulness of listener guesses about speaker meanings, e.g. meaning similarity. For exact recovery, specify an identity matrix.

            kind: Whether to measure informativity using literal or pragmatic agents, as canonically described in the Rational Speech Act framework.
        """
        self.prior = prior
        self.utility = utility
        self.agent_type = agent_type

    def language_informativity(self, language: Language) -> float:
        """The informativity of a language is based on the successful communication between a Sender and a Receiver.

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

        if self.agent_type == "literal":
            pass
        elif self.agent_type == "pragmatic":
            speaker = PragmaticSpeaker(language, listener)
            listener = PragmaticListener(language, speaker, np.diag(self.prior))
        else:
            raise ValueError(
                f"agent_type must be either 'literal' or 'pragmatic'. Received: {self.agent_type}."
            )

        inf = communicative_success(speaker, listener, self.prior, self.utility)

        # Check informativity > 0
        m, _ = self.utility.shape  # square matrix
        if np.array_equal(self.utility, np.eye(m)):
            if isclose(inf, 0.0):
                raise ValueError(
                    f"Informativity must be nonzero for indicator utility reward function, but was: {inf}"
                )

        return inf


##############################################################################
# Main and utility functions for informativity calculation
##############################################################################


def uniform_prior(space: Universe) -> np.ndarray:
    """Return a 1-D numpy array of size |space| reprsenting uniform distribution."""
    return np.array([1 / len(space.objects) for _ in range(len(space.objects))])


def build_utility_matrix(
    space: Universe, utility: Callable[[Meaning, Meaning], float]
) -> np.ndarray:
    """Construct the square matrix specifying the utility function defined for pairs of meanings."""
    return np.array(
        [
            [utility(meaning, meaning_) for meaning_ in space.objects]
            for meaning in space.objects
        ]
    )


def compute_sparsity(mat: np.ndarray) -> float:
    """Number of 0s / number of elements in matrix."""
    total = mat.shape[0] * mat.shape[1]
    zeros = np.count_nonzero(mat == 0)
    return float(zeros / total)


def communicative_success(
    speaker: Speaker,
    listener: Listener,
    prior: np.ndarray,
    utility: np.ndarray,
) -> float:
    """Helper function to compute the literal informativity of a language.

        I(L) = P(m, m') * u(m, m')

             = \sum_{m \in M} p(m) \sum_{i \in L} p(i|m) \sum_{m' \in i} p(m'|i) * u(m, m')

             = sum(diag(p)SR * u)

    Args:
        - speaker: a literal or pragmatic speaker, containing a matrix S for P(e | m)

        - listener: a literal or pragmatic listener, containing a matrix R for P(m | e)

        - prior: p(m), distribution over meanings representing communicative need

        - utility: a function u(m, m') representing similarity of meanings, or pair-wise usefulness of listener guesses about speaker meanings.
    """
    return float(np.sum(np.diag(prior) @ speaker.S @ listener.R * utility))
