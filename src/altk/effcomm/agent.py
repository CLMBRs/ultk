"""Classes for representing communicative agents, such as Senders and Receivers figuring in Lewis-Skyrms signaling games, literal and pragmatic agents in the Rational Speech Act framework, etc."""

import numpy as np
from scipy.special import softmax
from altk.language.language import Language

##############################################################################
# Agent Classes
##############################################################################


class CommunicativeAgent:
    def __init__(self, language: Language):
        """Takes a language to construct a agent to define the relation between meanings and expressions, which can be used to initialize the agent matrices (e.g. `S` or `R`).
        """
        self.language = language
        self._parameters = None


class StaticAgent(CommunicativeAgent):
    """As opposed to a dynamic agent whose learned parameters can evolve over, a static agent is always with the same langauge."""
    def __init__(self, language: Language):
        super().__init__(language)

    def normalize_weights(self) -> None:
        raise NotImplementedError

    @property
    def weights(self) -> np.ndarray:
        return self._parameters
    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        self._parameters = weights

    @classmethod
    def from_weights(cls, weights: np.ndarray):
        """Construct a StaticAgent from a weight matrix."""
        agent = cls(language=None)
        agent.weights = weights
        agent.normalize_weights()
        return agent

class Speaker(StaticAgent):
    def __init__(self, language: Language):
        super().__init__(language)

    @property
    def S(self) -> np.ndarray:
        return self.weights
    @S.setter
    def S(self, mat: np.ndarray) -> None:
        self.weights = mat

    def normalize_weights(self):
        # The sum of p(e | intended m) must be exactly 0 or 1.
        # We check for nans because sometimes a language cannot express a particular meaning at all, resulting in a row sum of 0.
        np.seterr(divide='ignore', invalid='ignore')
        self.S = np.nan_to_num(self.S/self.S.sum(axis=1, keepdims=True))


class Listener(StaticAgent):
    def __init__(self, language: Language):
        super().__init__(language)

    @property
    def R(self) -> np.ndarray:
        return self.weights
    @R.setter
    def R(self, mat: np.ndarray) -> None:
        self.weights = mat

    def normalize_weights(self):
        # The sum of p(m | heard e) must be 1. We can safely divide each row by its sum because every expression has at least one meaning.
        self.R = self.R/self.R.sum(axis=1, keepdims=True)


"""In the RSA framework, communicative agents reason recursively about each other's literal and pragmatic interpretations of utterances. Concretely, each agent is modeled by a conditional distribution. The speaker is represented by the probability of choosing to use an utterance (expression) given an intended meaning, P(e|m). The listener is a mirror of the speaker; it is represented by the probability of guessing a meaning given that they heard an utterance (expression), P(m|e)."""


class LiteralSpeaker(Speaker):
    """A literal speaker chooses utterances without any reasoning about other agents. The literal speaker's conditional probability distribution P(e|m) is uniform over all expressions that can be used to communicate a particular meaning. This is in contrast to a pragmatic speaker, whose conditional distribution is not uniform in this way, but instead biased towards choosing expressions that are less likely to be misinterpreted by some listener."""

    def __init__(self, language: Language):
        super().__init__(language)
        self.S = self.language.binary_matrix()
        self.normalize_weights()


class LiteralListener(Listener):
    """A naive literal listener interprets utterances without any reasoning about other agents. Its conditional probability distribution P(m|e) for guessing meanings is uniform over all meanings that can be denoted by the particular expression heard. This is in contrast to a pragmatic listener, whose conditional distribution is biased to guess meanings that a pragmatic speaker most likely intended."""

    def __init__(self, language: Language):
        super().__init__(language)
        self.R = self.language.binary_matrix().T
        self.normalize_weights()

class PragmaticSpeaker(Speaker):
    """A pragmatic speaker chooses utterances based on how a listener would interpret them. A pragmatic speaker may be initialized with any kind of listener, e.g. literal or pragmatic -- meaning the recursive reasoning can be modeled up to arbitrary depth."""

    def __init__(
        self, language: Language, listener: Listener, temperature=1.0
    ):
        """Initialize the |M|-by-|E| matrix, S, corresponding to the pragmatic speaker's conditional probability distribution over expressions given meanings.
        
        The pragmatic speaker chooses expressions to communicate their intended meaning according to:

            P(e | m) \propto exp(temperature * Utility(e,m))

        where

            Utility(e , m) := log(P_Listener(m | e))

        Args:
            language: the language with |M| meanings and |E| expressions defining the size of S.

            listener: a communicative agent storing a matrix R representing the conditional distribution over expressions given meanings.

            temperature: a float \in [0,1], representing how `optimally rational' the pragmatic speaker is; 1.0 is chosen when no particular assumptions about rationality are made.
        """
        super().__init__(language)

        # Row vector \propto column vector of literal R
        self.S = softmax(np.nan_to_num(np.log(listener.R.T)) * temperature, axis=1)


class PragmaticListener(Listener):
    """A pragmatic listener interprets utterances based on their expectations about a pragmatic speaker's decisions. A pragmatic listener may be initialized with any kind of speaker, e.g. literal or pragmatic -- meaning the recursive reasoning can be modeled up to arbitrary depth."""

    def __init__(
        self, language: Language, speaker: Speaker, prior: np.ndarray
    ):
        """Initialize the |E|-by-|M| matrix, R, corresponding to the pragmatic listener's conditional probability distribution over meanings given expressions.

        The pragmatic listener chooses meanings as their best guesses of the expression they heard according to:

            P(m | e) \propto P_PragmaticSpeaker(e | m)

        Args:
            language: the language with |M| meanings and |E| expressions defining the size of R.

            speaker: a communicative agent storing a matrix S representing the  conditional distribution over expressions given meanings.

            prior: a diagonal matrix of size |M|-by-|M| representing the communicative need probabilities for meanings.
        """
        super().__init__(language)
        # Row vector \propto column vector of pragmatic S

        self.R = np.zeros_like(speaker.S.T)
        for i in range(len(self.R)):
            col = speaker.S[:, i]
            self.R[i] = col @ prior / np.sum(col @ prior)

