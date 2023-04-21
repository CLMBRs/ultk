# from . import util
import util
import vis
from agents import Receiver, Sender
from game import distribution_over_states, SignalingGame, indicator
from languages import State, StateSpace, Signal, SignalMeaning, SignalingLanguage
from learning import simulate_learning


def main(args):
    # load game settings
    num_signals = args.num_signals
    num_states = args.num_states
    num_rounds = args.num_rounds
    learning_rate = args.learning_rate
    prior_type = args.distribution_over_states
    seed = args.seed

    util.set_seed(seed)

    ##########################################################################
    # Define game parameters
    ##########################################################################

    # dummy names for signals, states
    state_names = [f"state_{i+1}" for i in range(num_states)]
    signal_names = [f"signal_{i+1}" for i in range(num_signals)]

    # Construct the universe of states, and language defined over it
    universe = StateSpace([State(name=name) for name in state_names])

    # All meanings are dummy placeholders at this stage, but they can be substantive once agents are given a weight matrix.
    dummy_meaning = SignalMeaning(states=universe.referents, universe=universe)
    signals = [Signal(form=name, meaning=dummy_meaning) for name in signal_names]

    # Create a seed language to initialize agents.
    seed_language = SignalingLanguage(signals=signals)
    sender = Sender(seed_language, name="sender")
    receiver = Receiver(seed_language, name="receiver")

    # Construct a prior probability distribution over states
    prior_over_states = distribution_over_states(num_states, type=prior_type)

    ##########################################################################
    # Main simulation
    ##########################################################################

    signaling_game = SignalingGame(
        states=universe.referents,
        signals=signals,
        sender=sender,
        receiver=receiver,
        utility=lambda x, y: indicator(x, y),
        prior=prior_over_states,
    )
    signaling_game = simulate_learning(signaling_game, num_rounds, learning_rate)

    ##########################################################################
    # Analysis
    ##########################################################################

    accuracies = signaling_game.data["accuracy"]
    complexities = signaling_game.data["complexity"]
    languages = [
        agent.to_language(
            # optionally add analysis data
            data={"accuracy": accuracies[-1]},
            threshold=0.2,
        )
        for agent in [sender, receiver]
    ]

    util.save_weights(args.save_weights, sender, receiver)
    util.save_languages(args.save_languages, languages)

    vis.plot_distribution(args.save_distribution, prior_over_states)
    vis.plot_accuracy(args.save_accuracy_plot, accuracies)
    vis.plot_complexity(args.save_complexity_plot, complexities)
    vis.plot_tradeoff(args.save_tradeoff_plot, complexities, accuracies)

    print("Done.")


if __name__ == "__main__":
    args = util.get_args()

    main(args)
