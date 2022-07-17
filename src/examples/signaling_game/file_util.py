import yaml
import numpy as np
import random
import plotnine as pn
from languages import SignalingLanguage

def load_configs(fn: str) -> dict:
    """Load the configs .yml file as a dict."""
    with open(fn, "r") as stream:
        configs = yaml.safe_load(stream)
    return configs

def set_seed(seed: int) -> None:
    """Sets various random seeds."""
    random.seed(seed)
    np.random.seed(seed)

def save_weights(fn, sender, receiver) -> None:
    """Save weights to a txt file."""
    sender_n = sender.weights/sender.weights.sum(axis=1, keepdims=True)
    receiver_n = receiver.weights/receiver.weights.sum(axis=0, keepdims=True)
    np.set_printoptions(precision=2)
    weights_string = (
    f"""Sender
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
    """)
    with open(fn, 'w') as f:
        f.write(weights_string)

def save_languages(fn: str, languages: list[SignalingLanguage]) -> None:
    """Save a list of languages to a YAML file."""
    data = {
        "languages": list(lang.yaml_rep() for lang in languages)
    }
    with open(fn, "w") as outfile:
        yaml.safe_dump(data, outfile)    

def save_plot(fn: str, plot: pn.ggplot, width=10, height=10, dpi=300) -> None:
    """Save a plot with some default settings."""
    plot.save(fn, width=10, height=10, dpi=300)