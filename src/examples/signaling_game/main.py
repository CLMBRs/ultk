import sys
import file_util
import vis
import numpy as np
import measures
from tqdm import tqdm
from agents import Receiver, Sender
from game import SignalingGame
from languages import (
    State, 
    StateSpace, 
    Signal, 
    SignalMeaning, 
    SignalingLanguage
    )

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 src/main.py path_to_config_file")
        raise TypeError(f"Expected {2} arguments but received {len(sys.argv)}.")

    # load game settings
    config_fn = sys.argv[1]
    configs = file_util.load_configs(config_fn)
    
    num_signals = configs['num_signals']
    num_states = configs['num_states']
    reward_amount = configs['reward_amount']
    num_rounds = configs['num_rounds']
    prior_type = configs['prior_over_states']
    seed = configs['random_seed']

    # output files
    paths = configs['paths']
    file_util.set_seed(seed)

    ##########################################################################
    # Define game parameters
    ##########################################################################

    # dummy names for signals, states
    state_names = [f"state_{i+1}" for i in range(num_states)]
    signal_names = [f"signal_{i+1}" for i in range(num_signals)]

    # Construct the universe of states, and language defined over it
    universe = StateSpace([State(name=name) for name in state_names])

    # All meanings are dummy placeholders at this stage, but they can be substantive once agents are given a weight matrix.
    dummy_meaning = SignalMeaning(states=[], universe=universe)
    signals = [Signal(form=name, meaning=dummy_meaning) for name in signal_names]

    # Create a seed language to initialize agents.
    seed_language = SignalingLanguage(signals=signals)
    sender = Sender(seed_language, name="sender")
    receiver = Receiver(seed_language, name="receiver")

    # Construct a prior probability distribution over states
    prior_over_states = measures.distribution_over_states(num_states, type=prior_type)

    ##########################################################################
    # Main simulation
    ##########################################################################

    game = SignalingGame(
        states=universe.referents,
        signals=signals,
        sender=sender,
        receiver=receiver,
        utility=lambda x, y: measures.indicator(x,y) * reward_amount,
        prior=prior_over_states,
    )
    game.play(num_rounds)
    
    ##########################################################################
    # Analysis
    ##########################################################################

    accuracies = game.data["accuracy"]
    complexities = game.data["complexity"]
    languages = [
        agent.to_language(
            # optionally add analysis data
            data={"accuracy": accuracies[-1]},
            threshold=0.2,
            ) for agent in [sender, receiver]
        ]

    file_util.save_weights(paths['weights'], sender, receiver)
    file_util.save_languages(paths['languages'], languages)

    vis.plot_distribution(paths['prior_plot'], prior_over_states)
    vis.plot_accuracy(paths['accuracy_plot'], accuracies)
    vis.plot_complexity(paths['complexity_plot'], complexities)
    vis.plot_tradeoff(paths['tradeoff_plot'], complexities, accuracies)

    print("Done.")

if __name__ == "__main__":
    main()