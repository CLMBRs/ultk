import numpy as np
from altk.effcomm.agent import CommunicativeAgent, StaticAgent, Speaker, Listener
from altk.language.semantics import Meaning
from languages import State, Signal, SignalMeaning, SignalingLanguage
from typing import Any

class SignalingAgent(CommunicativeAgent):
    """The agent type used in an Atomi-n signaling game."""
    def __init__(self, language: SignalingLanguage, name: str = None):
        super().__init__(language)
        self.signals = [e for e in language.expressions]
        self.states = list(language.universe.referents)
        
        # matrix indexing lookups
        self._state_to_index = {state: i for i, state in enumerate(self.states)}
        self._index_to_state = tuple(self.states)
        self._signal_to_index = {signal: i for i, signal in enumerate(self.signals)}
        self._index_to_signal = tuple(self.signals)  

        self.shape = None
        self.weights = None
        self.name = name

    def initialize_weights(self, weights: np.ndarray = None, initial='ones') -> None:
        """Initialize the agent's weight matrix.
        
        Args:
            weights: an np.ndarray representing the weights to initialize the agent with. By default None, and the agent's weights will be initialized uniformly.

            initial: {'ones', 'random'} a str reprsenting the initialization method to use. If 'ones' (default), initialize the weight matrix with `np.ones`. If 'random', initalize the weight matrix from `np.random.uniform`.
        """
        if weights is not None:
            if weights.shape != self.shape:
                raise ValueError(f"Inapropriate Sender weight matrix shape for language. Sender is of shape {self.shape} but received {weights.shape}.")
            self.weights = weights
        else:
            # initialize uniformly. Other options could include random initialization.
            if initial == 'ones':
                self.weights = np.ones(self.shape)
            elif initial == 'random':
                self.weights = np.random.uniform(
                    low=0.0, 
                    high=1.0,
                    size=self.shape,
                )
            else:
                raise ValueError(f"Inappropriate value received for argument `initial`. Possible values are {'ones', 'random'}, but received: {initial}. ")
    
    def state_to_index(self, state: State) -> int:
        return self._state_to_index[state]

    def index_to_state(self, index: int) -> State:
        return self._index_to_state[index]
    
    def signal_to_index(self, signal: Signal) -> int:
        return self._signal_to_index[signal]

    def index_to_signal(self, index: int) -> Signal:
        return self._index_to_signal[index]

    def policy_to_indices(self, policy: dict[str, Any]) -> tuple[int]:
        """Given a signal and state, access the corresponding weight."""
        raise NotImplementedError

    def decision_function(self, index: int) -> int:
        """Chooses an entry of the agent's weight matrix by sampling from the row vector specified by the index.
        
        Args:
            index: the integer index representing a row of the weight matrix.

        Returns:
            the integer index of the agent's choice
        """
        choices = self.weights[index]
        choices_normalized = choices / choices.sum()
        return np.random.choice(
            a=range(len(choices)),
            p=choices_normalized,
        )

    def reward(self, policy: dict[str, Any], amount: float) -> None:
        """Reward an agent for a particular state-signal behavior.
        
        In an Atomic-n signaling game, the communicative success of Sender and Receiver language protocols evolve under simple reinforcement learning dynamics. The reward function increments an agent's weight matrix at the specified location by the specified amount.

        Args:
            policy: a dict of the form {"state": State, "signal": Signal}

            amount: a positive number reprsenting how much to reward the behavior
        """
        if set(policy.keys()) != {"state", "signal"}:
            raise ValueError(f"The argument `policy` must take a dict with keys 'state' and 'signal'. Received: {policy.keys()}'")
        if amount < 0:
            raise ValueError(f"Amount to reinforce weight must be a positive number.")
        self.weights[self.policy_to_indices(policy)] += amount

    def to_static_agent(self) -> StaticAgent:
        """Get a static RSA speaker agent from this Sender agent.
        
        The conceptual difference between a static RSA agent and the signaling game Receiver is that an RSA agent is a static 'snapshot' of what linguistic behavior the Sender has learned.
        """
        raise NotImplementedError

    def to_language(
        self, 
        data: dict = {
                "complexity": None, 
                "accuracy": None, 
            },
        threshold: float = 0.1
        ) -> SignalingLanguage:
        """Get a language from the agent.
        
        A language is an object used by a Speaker or Listener representing a static _snapshot_ of the behavior that a Sender or Receiver has learned.

        Use the agent's weight matrix, the set of signals, and the set of states from the agent's initialization language to generate a new language accurately reflecting the new signal meanings, e.g. how the agent interprets signals as meaning zero or more states. 
        
        Args:
            threshold: a float in [0,1] representing the cutoff for determining if a meaning (state) can be communicated by a signal. Because weights are not initialized to 0, it is a good idea to set nonzero values as the threshold.
        """
        agent = self.to_static_agent()

        signals = []
        # loop over agent's initalization vocabulary
        for outdated_signal in self.signals:
            # get all meanings that the signal can communicate
            states = [
                state for state in self.states 
                    if agent.weights[self.policy_to_indices(
                        policy={"state": state, "signal": outdated_signal}
                    )] > threshold # if probability of state is high enough
            ]

            meaning = SignalMeaning(states, self.language.universe)
            # construct the updated signal as a new form-meaning mapping
            signal = Signal(
                form=outdated_signal.form,
                meaning=meaning,
            )
            signals.append(signal)

        if "name" not in data:
            data["name"] = self.name

        return SignalingLanguage(signals=signals, data=data)


class Sender(SignalingAgent):
    """A Sender agent in an Atomic-n signaling game chooses a signal given an observed state of nature, according to P(signal | state). """

    def __init__(
        self, 
        language: SignalingLanguage, 
        weights = None, 
        name: str = None,
        ):
        super().__init__(language, name)
        self.shape = (len(self.states), len(self.signals))
        self.initialize_weights(weights)

    def encode(self, state: Meaning) -> Signal:
        """Choose a signal given the state of nature observed, e.g. encode a discrete input as a discrete symbol."""
        index = self.decision_function(
            index=self.state_to_index(state)
        )
        return self.index_to_signal(index)

    def policy_to_indices(self, policy: dict[str, Any]) -> tuple[int]:
        """Map a policy to a `(state, signal)` index."""
        return (
            self.state_to_index(policy["state"]), 
            self.signal_to_index(policy["signal"]),
            )

    def to_static_agent(self) -> Speaker:
        """Get a static RSA speaker agent from this Sender agent.
        
        The conceptual difference between an RSA agent and the signaling game Receiver is that an RSA agent is a static 'snapshot' of what linguistic behavior the Sender has learned.
        """
        return Speaker.from_weights(self.weights)


class Receiver(SignalingAgent):
    """A Receiver agent in an Atomic-n signaling game chooses an action=state given a signal they received, according to P(state | signal). """

    def __init__(
        self, 
        language: SignalingLanguage, 
        weights = None, 
        name: str = None
        ):
        super().__init__(language, name)
        self.shape = (len(self.signals), len(self.states))
        self.initialize_weights(weights=weights)
            
    def decode(self, signal: Signal) -> Meaning:
        """Choose an action given the signal received, e.g. decode a target state given its discrete encoding. """
        index = self.decision_function(
            index=self.signal_to_index(signal)
        )
        return self.index_to_state(index)

    def policy_to_indices(self, policy: dict[str, Any]) -> tuple[int]:
        """Map a policy to a `(signal, state)` index."""
        return (
            self.signal_to_index(policy["signal"]),
            self.state_to_index(policy["state"]),             
            )

    def to_static_agent(self) -> Listener:
        """Get a static RSA Listener agent from this Sender agent.
        
        The conceptual difference between an RSA agent and the signaling game Receiver is that an RSA agent is a static 'snapshot' of what linguistic behavior the Sender has learned.
        """
        return Listener.from_weights(self.weights)