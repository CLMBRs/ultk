"""Simple Roth-Erev reinforcement learning dynamic for agents of a signaling game."""

import numpy as np
from ultk.effcomm.agent import CommunicativeAgent
from ultk.effcomm.informativity import communicative_success
from ultk.effcomm.information import information_rate
from game import SignalingGame
from tqdm import tqdm
from typing import Any


def simulate_learning(g: SignalingGame, num_rounds: int, learning_rate=1.0) -> None:
    """Simulate Roth-Erev reinforcement learning in the signaling game.

    Args:
        num_rounds: the number of rounds to pay the signaling game.

        reward_amount: the amount to scale the utility function by, before rewarding agents with the result.
    """

    for _ in tqdm(range(num_rounds)):
        # get input to sender
        target = np.random.choice(a=g.states, p=g.prior)

        # record interaction
        signal = g.sender.encode(target)
        output = g.receiver.decode(signal)
        amount = g.utility(target, output) * learning_rate

        # update agents
        reward(
            agent=g.sender,
            strategy={"referent": target, "expression": signal},
            amount=amount,
        )
        reward(
            agent=g.receiver,
            strategy={"referent": output, "expression": signal},
            amount=amount,
        )

        # track accuracy and complexity
        g.data["accuracy"].append(
            communicative_success(
                speaker=g.sender, listener=g.receiver, utility=g.utility, prior=g.prior
            )
        )
        g.data["complexity"].append(
            information_rate(g.prior, g.sender.normalized_weights())
        )

    return g


def reward(agent: CommunicativeAgent, strategy: dict[str, Any], amount: float) -> None:
    """Reward an agent for a particular referent-expression behavior.

    In a signaling game, the communicative success of Sender and Receiver language protocols evolve under simple reinforcement learning dynamics. The reward function increments an agent's weight matrix at the specified location by the specified amount.

    Args:
        strategy: a dict of the form {"referent": referent, "expression": Expression}

        amount: a positive number reprsenting how much to reward the behavior
    """
    if set(strategy.keys()) != {"referent", "expression"}:
        raise ValueError(
            f"The argument `strategy` must take a dict with keys 'referent' and 'expression'. Received: {strategy.keys()}'"
        )
    if amount < 0:
        raise ValueError(f"Amount to reinforce weight must be a positive number.")
    agent.weights[agent.strategy_to_indices(strategy)] += amount
