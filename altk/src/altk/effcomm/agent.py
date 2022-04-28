"""Classes for representing communicative agents, such as Senders and Receivers figuring in Lewis-Skyrms signaling games, or literal and pragmatic agents in the Rational Speech Act framework."""

import numpy as np
from altk.language.language import Language

##############################################################################
# Agent Classes
##############################################################################


class Communicative_Agent:
    def __init__(self, language: Language):
        """Takes a language to construct a agent to define the relation between meanings and expressions.

        By default initialize to uniform communicative need distribution.
        """
        self.language = language


class Speaker(Communicative_Agent):
    def __init__(self, language: Language):
        super().__init__(language)


class Listener(Communicative_Agent):
    def __init__(self, language: Language):
        super().__init__(language)


"""In the RSA framework, communicative agents reason recursively about each other's literal and pragmatic interpretations of utterances."""


class LiteralSpeaker(Speaker):
    """A literal speaker chooses utterances without any reasoning about other agents."""

    def __init__(self, language: Language):
        super().__init__(language)
        self.S = naive_matrix(self)


class LiteralListener(Listener):
    """A naive literal listener interprets utterances without any reasoning about other agents."""

    def __init__(self, language: Language):
        super().__init__(language)
        self.R = naive_matrix(self)


class PragmaticSpeaker(Speaker):
    """A pragmatic speaker chooses utterances based on how a naive, literal listener would interpret them."""

    def __init__(
        self, language: Language, literal_listener: LiteralListener, temperature=1.0
    ):
        """Initialize the |M|-by-|E| matrix, S, corresponding to the pragmatic speaker's conditional probability distribution over expressions given meanings.
        
        The pragmatic speaker chooses expressions to communicate their intended meaning according to:

            P(e | m) \propto exp(temperature * Utility(e,m))

        where

            Utility(e , m) := log(P_LiteralListener(m | e))

        Args:
            language: the language with |M| meanings and |E| expressions defining the size of S.

            literal_listener: a communicative agent storing a matrix R representing the literal (naive) conditional distribution over expressions given meanings.

            temperature: a float \in [0,1], representing how `optimally rational' the pragmatic speaker is; 1.0 is chosen when no particular assumptions about rationality are made.
        """
        super().__init__(language)
        self.S = np.zeros_like(literal_listener.R.T)
        # Row vector \propto column vector of literal R
        for i in range(len(self.S)):
            col = literal_listener.R[:, i]
            self.S[i] = softmax_temp_log(col, temperature)

class PragmaticListener(Listener):
    """A pragmatic listener interprets utterances based on their expectations about a pragmatic speaker's decisions."""

    def __init__(
        self, language: Language, pragmatic_speaker: PragmaticSpeaker, prior: np.ndarray
    ):
        """Initialize the |E|-by-|M| matrix, R, corresponding to the pragmatic listener's conditional probability distribution over meanings given expressions.

        The pragmatic listener chooses meanings as their best guesses of the expression they heard according to:

            P(m | e) \propto P_PragmaticSpeaker(e | m)

        Args:
            language: the language with |M| meanings and |E| expressions defining the size of R.

            pragmatic_speaker: a communicative agent storing a matrix S representing the pragmatic conditional distribution over expressions given meanings.

            prior: a diagonal matrix of size |M|-by-|M| representing the communicative need probabilities for meanings.
        """
        super().__init__(language)
        self.R = np.zeros_like(pragmatic_speaker.S.T)
        # Row vector \propto column vector of pragmatic S
        for i in range(len(self.R)):
            col = pragmatic_speaker.S[:, i]
            self.R[i] = col @ prior / np.sum(col @ prior)

##############################################################################
# Helper functions
##############################################################################


def naive_matrix(agent: Communicative_Agent) -> np.ndarray:
    """Create and return the matrix representing the conditional distribution relevant to the agent.

    _Sender_
        The distribution P(e | m) represents the probability that a sender (speaker) chooses expression e to communicate her intended meaning m. The row vector S_i represents the distribution over expressions for meaning i.

    _Receiver_
        The distribution P(m | e) represents the probability that a receiver (listener) guesses that the spekaer meant to communicate m using e. The row vector R_i represents the distribution over meanings for expression i.

    Assume that for a particular meaning, every expression that can denote it is equiprobable.

    Args:
        language: an Language from which to define the distributions

        agent: a string, either 'speaker' or 'listener'

    Returns:
        mat: the matrix representing the conditional distribution.
    """
    expressions = tuple(agent.language.expressions)
    meanings = tuple(agent.language.universe.objects)

    len_e = len(expressions)
    len_m = len(meanings)

    mat = np.zeros((len_m, len_e))
    for i, m in enumerate(meanings):
        for j, e in enumerate(expressions):
            mat[i, j] = float(e.can_express(m))

    # The sum of p(e | intended m) must be exactly 0 or 1
    if isinstance(agent, LiteralSpeaker):
        for i in range(len_m):
            # Sometimes a language cannot express a particular meaning at all, resulting in a row sum of 0.
            if mat[i].sum():
                mat[i] = mat[i] / mat[i].sum()

    # The sum of p(m | heard e) must be 1
    elif isinstance(agent, LiteralListener):
        mat = mat.T
        for i in range(len_e):
            # Every expression must have at least one meaning.
            mat[i] = mat[i] / mat[i].sum()

    else:
        raise ValueError(
            f"Communicative agent must be a LiteralSpeaker or LiteralListener in order to build a naive conditional probability matrix. Received type: {type(agent)}"
        )

    return mat


def softmax_temp_log(arr: np.ndarray, temperature: float) -> np.ndarray:
    """Function defining the proportional relationship between literal listener probabilities and speaker probabilies.

    Compute softmax(temperature * log(arr)) but handle 0 probability values.

    Args:
        arr: a vector of real values; in this context it will be a vector of log probabilities scaled by the temperature parameter.

        temperature: a float \in [0,1] representing rational optimality
    Returns:

        an array representing the resulting probability distribution.
    """
    # set dummy values for 0
    arr[arr == 0.0] = 10**-10
    denominator = np.sum(np.exp(temperature * np.log(arr)))
    numerator = np.exp(temperature * np.log(arr))
    return numerator / denominator
