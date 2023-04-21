import argparse
import yaml
import numpy as np
import random
import plotnine as pn
from languages import SignalingLanguage


def set_seed(seed: int) -> None:
    """Sets various random seeds."""
    random.seed(seed)
    np.random.seed(seed)


def save_weights(fn, sender, receiver) -> None:
    """Save weights to a txt file."""
    sender_n = sender.weights / sender.weights.sum(axis=1, keepdims=True)
    receiver_n = receiver.weights / receiver.weights.sum(axis=0, keepdims=True)
    np.set_printoptions(precision=2)
    weights_string = f"""Sender
    \n------------------
    \nweights:
    \n{sender.weights}
    \ndistribution:
    \n{sender_n}
    \n

    \nReceiver
    \n------------------
    \nweights:
    \n{receiver.weights}
    \ndistribution:
    \n{receiver_n}
    """
    with open(fn, "w") as f:
        f.write(weights_string)


def save_languages(fn: str, languages: list[SignalingLanguage]) -> None:
    """Save a list of languages to a YAML file."""
    data = {"languages": list(lang.yaml_rep() for lang in languages)}
    with open(fn, "w") as outfile:
        yaml.safe_dump(data, outfile)


def save_plot(fn: str, plot: pn.ggplot, width=10, height=10, dpi=300) -> None:
    """Save a plot with some default settings."""
    plot.save(fn, width=10, height=10, dpi=300)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Specify the random seed.")
    parser.add_argument(
        "--num_states",
        type=int,
        default=2,
        help="Number of states for sender and receiver.",
    )
    parser.add_argument(
        "--num_signals",
        type=int,
        default=2,
        help="Number of signals for sender and receiver.",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=100,
        help="How many rounds of the signaling game to run.",
    )
    parser.add_argument(
        "--distribution_over_states",
        type=str,
        choices=["deterministic", "random"],
        default="deterministic",
        help="How to generate distribution over states, either deterministically uniform or randomly sampled.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1.0,
        help="Learning rate for Bush-Mosteller reinforcement.",
    )
    parser.add_argument(
        "--save_languages",
        type=str,
        default="outputs/default/languages.yml",
        help="Languages of agents will be saved to this file as YAML.",
    )
    parser.add_argument(
        "--save_weights",
        type=str,
        default="outputs/default/weights.txt",
        help="Weights of agents will be saved to this file as plain text.",
    )
    parser.add_argument(
        "--save_accuracy_plot",
        type=str,
        default="outputs/default/accuracy.png",
        help="Plot of signaling accuracy will be saved to this file as png.",
    )
    parser.add_argument(
        "--save_complexity_plot",
        type=str,
        default="outputs/default/complexity.png",
        help="Plot of complexity accuracy will be saved to this file as png.",
    )
    parser.add_argument(
        "--save_tradeoff_plot",
        type=str,
        default="outputs/default/tradeoff.png",
        help="Plot of accuracy / complexity tradeoff will be saved to this file as png.",
    )
    parser.add_argument(
        "--save_distribution",
        type=str,
        default="outputs/default/distribution_over_states.png",
        help="Plot of distribution over states tradeoff will be saved to this file as png.",
    )
    args = parser.parse_args()
    return args
