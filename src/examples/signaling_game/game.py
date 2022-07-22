import numpy as np
from languages import State, Signal
from agents import Sender, Receiver
from altk.effcomm.agent import CommunicativeAgent
from altk.effcomm.informativity import communicative_success, build_utility_matrix
from measures import encoder_complexity
from typing import Any, Callable
from tqdm import tqdm

def reward(agent: CommunicativeAgent, policy: dict[str, Any], amount: float) -> None:
    """Reward an agent for a particular referent-expression behavior.
    
    In a signaling game, the communicative success of Sender and Receiver language protocols evolve under simple reinforcement learning dynamics. The reward function increments an agent's weight matrix at the specified location by the specified amount.

    Args:
        policy: a dict of the form {"referent": referent, "expression": Expression}

        amount: a positive number reprsenting how much to reward the behavior
    """
    if set(policy.keys()) != {"referent", "expression"}:
        raise ValueError(f"The argument `policy` must take a dict with keys 'referent' and 'expression'. Received: {policy.keys()}'")
    if amount < 0:
        raise ValueError(f"Amount to reinforce weight must be a positive number.")
    agent.weights[agent.policy_to_indices(policy)] += amount


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
        # Main game paramters
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
                amount=amount
                )
            reward(
                agent=self.receiver,
                policy={"referent": output, "expression": signal}, 
                amount=amount
                )

            # track accuracy and complexity
            self.data["accuracy"].append(self.comm_success(self.sender, self.receiver))
            self.data["complexity"].append(self.complexity(self.sender))