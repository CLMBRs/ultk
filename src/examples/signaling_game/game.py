import numpy as np
from languages import State, Signal
from agents import Sender, Receiver
from altk.effcomm.agent import CommunicativeAgent
from altk.effcomm.informativity import communicative_success
from altk.effcomm.complexity import encoder_complexity
from typing import Any, Callable
from tqdm import tqdm


class SignalingGame:
    """A signaling game is a tuple $(S, M, A, \sigma, \rho, u, P)$ of states, messages, acts, a sender, a receiver, a utility function, and a distribution over states. The sender and receiver have a common payoff, given by a communicative success function.

    In this signaling game, we identify the acts with the states. For more detail on the communicative success function, see altk.effcomm.informativity.communicative_success.
    """

    def __init__(
        self,
        states: list[State],
        signals: list[Signal],
        sender: Sender,
        receiver: Receiver,
        utility: Callable[[State, State], int],
        prior: np.ndarray,
    ) -> None:
        """Initialize a signaling game.

        Args:
            states: the list of states of 'nature' that function as both input to the sender, and output of the receiver

            signals: the objects (which inherit from Expression) produced by the sender, and are input to receiver

            sender: a distribution over signals, given states

            receiver: a distribution over states, given signals

            utility: a measure of the pairwise utility of sender inputs and receiver outputs, typically the indicator function.

            prior: an array specifying the probability distribution over states, which can represent the objective frequency of certain states in nature, or the prior belief about them.
        """
        # Main game parameters
        self.states = states
        self.signals = signals
        self.sender = sender
        self.receiver = receiver
        self.utility = utility
        self.prior = prior

        # Abbreviate measures of communicative success and complexity
        self.comm_success = lambda s, r: communicative_success(
            speaker=s,
            listener=r,
            utility=self.utility,
            prior=self.prior,
        )
        self.complexity = lambda s: encoder_complexity(s, self.prior)

        # measurements to track throughout game
        self.data = {"accuracy": [], "complexity": []}

    def play(self, num_rounds: int) -> None:
        """Simulate the signaling game.

        Args:
            num_rounds: the number of rounds to pay the signaling game.

            reward_amount: the amount to scale the utility function by, before rewarding agents with the result.
        """
        for _ in tqdm(range(num_rounds)):
            # get input to sender
            target = np.random.choice(a=self.states, p=self.prior)

            # record interaction
            signal = self.sender.encode(target)
            output = self.receiver.decode(signal)
            amount = self.utility(target, output)

            # update agents
            reward(
                agent=self.sender,
                policy={"referent": target, "expression": signal},
                amount=amount,
            )
            reward(
                agent=self.receiver,
                policy={"referent": output, "expression": signal},
                amount=amount,
            )

            # track accuracy and complexity
            self.data["accuracy"].append(self.comm_success(self.sender, self.receiver))
            self.data["complexity"].append(self.complexity(self.sender))


##########################################################################
# Helper functions
##########################################################################


def reward(agent: CommunicativeAgent, policy: dict[str, Any], amount: float) -> None:
    """Reward an agent for a particular referent-expression behavior.

    In a signaling game, the communicative success of Sender and Receiver language protocols evolve under simple reinforcement learning dynamics. The reward function increments an agent's weight matrix at the specified location by the specified amount.

    Args:
        policy: a dict of the form {"referent": referent, "expression": Expression}

        amount: a positive number reprsenting how much to reward the behavior
    """
    if set(policy.keys()) != {"referent", "expression"}:
        raise ValueError(
            f"The argument `policy` must take a dict with keys 'referent' and 'expression'. Received: {policy.keys()}'"
        )
    if amount < 0:
        raise ValueError(f"Amount to reinforce weight must be a positive number.")
    agent.weights[agent.policy_to_indices(policy)] += amount


def indicator(input: State, output: State) -> int:
    return input == output


def distribution_over_states(
    num_states: int, type: str = "deterministic", alpha: np.ndarray = None
):
    """Generate a prior probability distribution over states.

    Varying the entropy of the prior over states models the relative communicative 'need' of states. A natural interpretation is also that these needs reflect objective environmental or cultural pressures that cause certain objects to become more frequent-- and so more useful-- for communication.

    Args:
        num_states: the size of the distribution

        type: {'deterministic', 'random'} a str representing whether to generate a uniform prior or randomly sample one from a Dirichlet distribution.

        alpha: parameter of the Dirichlet distribution to sample from, of shape `(num_states)`. Each element must be greater than or equal to 0. By default set to all ones. Varying this parameter varies the entropy of the prior over states.

    Returns:
        sample: np.ndarray of shape `(num_states)`
    """
    if type == "deterministic":
        sample = (
            np.ones(num_states) / num_states
        )  # TODO: find how to generate uniform with alpha param in dirichlet
    elif type == "random":
        if alpha is None:
            alpha = np.ones(num_states)
        sample = np.random.default_rng().dirichlet(alpha=alpha)
    else:
        raise ValueError(
            f"The argument `prior_type` can take values {{'uniform', 'random'}}, but received {type}."
        )
    return sample
