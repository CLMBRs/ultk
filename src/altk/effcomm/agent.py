"""Classes for representing communicative agents, such as Senders and Receivers figuring in Lewis-Skyrms signaling games, literal and pragmatic agents in the Rational Speech Act framework, etc."""

import numpy as np
from typing import Any
from scipy.special import softmax
from altk.language.language import Language
from altk.language.semantics import Universe

##############################################################################
# Agent Classes
##############################################################################


# TODO: leaving this for now to sit and ask Shane about, but one idea: 
# think harder about how to elegantly do single-inheritance, and just say
# that certain agents (e.g. speakers of a Language, in contrast to mere Senders of forms given states) have their parameters _frozen_: they cannot learn / have their probability matrices / behavior change or evolve. This separation from learning parameters from evaluation-time behavior might help keep things speedy, and hopefully remain conceptually neat.

class Agent:
    def __init__(self, universe: Universe, forms=list[str], weights: np.ndarray = None, **kwargs):
        """Takes a language to construct an agent to define the relation between meanings and expressions, which can be used to initialize the agent's conditional probability matrix (e.g. `S` or `R`).

        Args:
            universe: 

            forms:

            weights: 
        """
        if weights is not None:
            self._weights = weights
        else:
            self._weights = np.ones((len(universe), len(forms)))

        self.universe = universe
        self.forms = forms

        # Data structures for matrix lookups
        self.forms_to_indices = {
            e: idx for e, idx in enumerate(sorted(forms))
            }
        self.indices_to_forms = tuple(sorted(forms))

        self.objects_to_indices = {
            m: idx for m, idx in enumerate(sorted(universe.objects))
        }
        self.indices_to_objects = tuple(sorted(universe.objects))

        # Intialize learning parameters
        self.accumulated_rewards = self._weights

    @classmethod
    def from_language(cls, language: Language):
        forms = [e.form for e in language.expressions]
        universe = language.universe
        weights = language.binary_matrix()
        return cls(universe, forms, weights)

    def interact(self, input: Any) -> Any:
        """Defines the (inter)actions that agents can perform; e.g. a Speaker choosing an expression or a Listener guessing an object. """
        raise NotImplementedError
    
    def reward(self, state: str, signal: str, weight: float) -> None:
        """Update the accumulated rewards for an agent."""
        raise NotImplementedError

    def form_to_index(self, form: str) -> int:
        """Return the index of an expression."""
        return self.forms_to_indices[form]

    def index_to_form(self, index: int) -> str:
        """Return the form corresponding to the index supplied."""
        return self.indices_to_forms[index]

    def object_to_index(self, object: str) -> int:
        """Return the index of a meaning."""
        return self.objects_to_indices[object]

    def index_to_object(self, index: int) -> str:
        """Return the form corresponding to the index supplied."""
        return self.indices_to_objects[index]

class Speaker(Agent):
    def __init__(
        self, 
        universe: Universe, 
        forms=list[str], 
        weights: np.ndarray = None
        ):
        super().__init__(universe, forms, weights)
        self.S = self.normalize_rows(self.S)

    @property
    def S(self) -> np.ndarray:
        return self._weights
    @S.setter
    def S(self, mat: np.ndarray) -> None:
        self._weights = mat

    def normalize_rows(matrix: np.ndarray) -> np.ndarray:
        """Normalize rows of a matrix, e.g. accumulated_rewards or weights."""
        # The sum of p(e | intended m) must be exactly 0 or 1.
        # We check for nans because sometimes a language cannot express a particular meaning at all, resulting in a row sum of 0.
        np.seterr(divide='ignore', invalid='ignore')
        return np.nan_to_num(matrix / matrix.sum(axis=1, keepdims=True))

    def interact(self, object: str) -> str:
        """A speaker agent interacts by observing the environment and choosing an appropriate signal (form) to send to a listener."""
        # sample from forms according to the weight matrix
        probabilities = self.S[self.object_to_index(object)]
        form = np.random.choice(a=self.forms, p=probabilities)
        return form
    
    def reward(self, state: str, signal: str, weight: float) -> None:
        """Update the accumulated rewards for a speaker's chosen signal given the state of nature observed.

        Args: 
            state: the observed state of nature

            signal: the chosen signal
        
            weight: the amount to increment the speaker's accumulated rewards by for the `(state, signal)` pair.
        """
        self.accumulated_rewards[
            self.object_to_index[state], self.form_to_index[signal]
            ] += weight
        # adjust _weights according to new accumulated_rewards
        self.S = self.normalize_rows(self.accumulated_rewards)

class Listener(Agent):
    def __init__(self, universe: Universe, forms=list[str], weights: np.ndarray = None):
        super().__init__(universe, forms, weights)

    @property
    def R(self) -> np.ndarray:
        return self._weights

    def normalize_rows(matrix: np.ndarray) -> np.ndarray:
        """Normalize rows of a matrix, e.g. accumulated_rewards or weights."""
        # The sum of p(m | heard e) must be 1. We can safely divide each row by its sum because every expression has at least one meaning.
        return matrix / matrix.sum(axis=1, keepdims=True)

    @R.setter
    def R(self, mat: np.ndarray) -> None:
        self._weights = mat

    def interact(self, form: str) -> str:
        """A speaker agent interacts by observing the environment and choosing an appropriate signal (form) to send to a listener."""
        # sample from forms according to the weight matrix
        probabilities = self.S[self.form_to_index(form)]
        guess = np.random.choice(a=self.universe.objects, p=probabilities)
        return guess

    def reward(self, state: str, signal: str, weight: float) -> None:
        """Update the accumulated rewards for a speaker's chosen signal given the state of nature observed.

        Args: 
            state: the observed state of nature

            signal: the chosen signal
        
            weight: the amount to increment the speaker's accumulated rewards by for the `(state, signal)` pair.
        """
        self.accumulated_rewards[
            self.form_to_index[signal], self.object_to_index[state]
            ] += weight
        # adjust _weights according to new accumulated_rewards
        self.R = self.normalize_rows(self.R)



"""In the RSA framework, communicative agents reason recursively about each other's literal and pragmatic interpretations of utterances. Concretely, each agent is modeled by a conditional distribution. The speaker is represented by the probability of choosing to use an utterance (expression) given an intended meaning, P(e|m). The listener is a mirror of the speaker; it is represented by the probability of guessing a meaning given that they heard an utterance (expression), P(m|e)."""


class LiteralSpeaker(Speaker):
    """A literal speaker chooses utterances without any reasoning about other agents. The literal speaker's conditional probability distribution P(e|m) is uniform over all expressions that can be used to communicate a particular meaning. This is in contrast to a pragmatic speaker, whose conditional distribution is not uniform in this way, but instead biased towards choosing expressions that are less likely to be misinterpreted by some listener."""

    def __init__(self, universe: Universe, forms=list[str], weights: np.ndarray = None):
        super().__init__(universe, forms, weights)


class LiteralListener(Listener):
    """A naive literal listener interprets utterances without any reasoning about other agents. Its conditional probability distribution P(m|e) for guessing meanings is uniform over all meanings that can be denoted by the particular expression heard. This is in contrast to a pragmatic listener, whose conditional distribution is biased to guess meanings that a pragmatic speaker most likely intended."""

    def __init__(self, universe: Universe, forms=list[str], weights: np.ndarray = None):
        super().__init__(universe, forms, weights)
        self.R = self.R.T

class PragmaticSpeaker(Speaker):
    """A pragmatic speaker chooses utterances based on how a listener would interpret them. A pragmatic speaker may be initialized with any kind of listener, e.g. literal or pragmatic -- meaning the recursive reasoning can be modeled up to arbitrary depth."""

    def __init__(self, listener: Listener, temperature=1.0
    ):
        """Initialize the |M|-by-|E| matrix, S, corresponding to the pragmatic speaker's conditional probability distribution over expressions given meanings.
        
        The pragmatic speaker chooses expressions to communicate their intended meaning according to:

            P(e | m) \propto exp(temperature * Utility(e,m))

        where

            Utility(e , m) := log(P_Listener(m | e))

        Args:
            listener: a communicative agent storing a matrix R representing the conditional distribution over expressions given meanings.

            temperature: a float \in [0,1], representing how `optimally rational' the pragmatic speaker is; 1.0 is chosen when no particular assumptions about rationality are made.
        """
        super().__init__(listener.universe, listener.forms)

        # Row vector \propto column vector of literal R
        self.S = softmax(
            np.nan_to_num(np.log(listener.R.T)) * temperature, axis=1
            )

        # self.S = np.zeros_like(listener.R.T)
        # for i in range(len(self.S)):
            # col = listener.R[:, i]
            # self.S[i] = softmax_temp_log(col, temperature)


class PragmaticListener(Listener):
    """A pragmatic listener interprets utterances based on their expectations about a pragmatic speaker's decisions. A pragmatic listener may be initialized with any kind of speaker, e.g. literal or pragmatic -- meaning the recursive reasoning can be modeled up to arbitrary depth."""

    def __init__(self, speaker: Speaker, prior: np.ndarray
    ):
        """Initialize the |E|-by-|M| matrix, R, corresponding to the pragmatic listener's conditional probability distribution over meanings given expressions.

        The pragmatic listener chooses meanings as their best guesses of the expression they heard according to:

            P(m | e) \propto P_PragmaticSpeaker(e | m)

        Args:
            speaker: a communicative agent storing a matrix S representing the  conditional distribution over expressions given meanings.

            prior: a diagonal matrix of size |M|-by-|M| representing the communicative need probabilities for meanings.
        """
        super().__init__(speaker.universe, speaker.forms)
        # Row vector \propto column vector of pragmatic S

        self.R = np.zeros_like(speaker.S.T)
        for i in range(len(self.R)):
            col = speaker.S[:, i]
            self.R[i] = col @ prior / np.sum(col @ prior)

##############################################################################
# Helper functions
##############################################################################


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
