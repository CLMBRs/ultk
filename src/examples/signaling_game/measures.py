import numpy as np
from agents import Receiver, Sender
from altk.effcomm.informativity import communicative_success, build_utility_matrix, uniform_prior
from languages import State
from typing import Callable

def signaling_accuracy(
    sender: Sender, 
    receiver: Receiver,
    utility: Callable[[State, State],float],
    prior: np.ndarray = None,
    ) -> float:
    """Compute the communicative success of sender and receiver as the informativity of RSA speakers and listeners.

    Args:
        language: the seed_language representing the possible signals and states of the game. 

        sender: the agent whose weights determine those of a LiteralSpeaker.

        receiver: the agent wose weights determine those of a LiteralListener

        prior: the prior probability distribution over states, Default is None and will be assigned uniform. This distribution represents 'communicative need', and aims to track how often speakers of a language need to refer to a particular state.

    Wrapper function for ALTK `communicative_success` with some default parameters.
    """
    language = sender.language
    return communicative_success(
        speaker=sender.to_static_agent(),
        listener=receiver.to_static_agent(),
        prior=uniform_prior(language.universe) if prior is None else prior, 
        utility=build_utility_matrix(language.universe, utility),
        )
    
    
def encoder_complexity(sender: Sender, prior_over_states: np.ndarray = None) -> float:
    """Compute the cognitive complexity for the encoder q, given by the information rate of compressing meanings M into words W:

        $I(M;W) = \sum_m p(m) \sum_w q(w|m) log [q(w|m) / q(w)]$
    
    where 
        $q(w) = \sum_m p(m) q(w|m) $

    Args: 
        sender: an Atomic Sender to compute the complexity of

        prior: the prior probability distribution over states. Default is None and will assume a uniform prior.
    """
    if prior_over_states is None:
        prior_over_states = uniform_prior(sender.language.universe)
    total = 0
    # p(m)
    for i, prob_state in enumerate(prior_over_states):
        state_sum = []
        for j, signal_weights in enumerate(sender.weights.T):
            signal_distribution = signal_weights / signal_weights.sum()
            # q(w)
            prob_signal = np.dot(prior_over_states, signal_distribution)
            # q(w|m)
            prob_signal_given_state = signal_distribution[i]
            # log [encoder/prior]
            log_term = np.log(prob_signal_given_state / prob_signal)

            # The pointwise mutual information is I(m;w)
            pmi = prob_signal_given_state * log_term
            state_sum.append(pmi)
        total += prob_state * sum(state_sum)
    
    return total

##########################################################################
# Helper functions
##########################################################################

def indicator(input: State, output: State) -> int:
    return input == output


def distribution_over_states(num_states: int, type: str = 'uniform', alpha: np.ndarray = None):
    """Generate a prior over states by drawing a sample from the Dirichlet distribution.

    Varying the entropy of the prior over states models the relative communicative 'need' of states. A natural interpretation is also that these needs reflect objective environmental or cultural pressures that cause certain objects to become more frequent-- and so more useful-- for communication.
    
    Args:
        num_states: the size of the distribution

        type: {'uniform', 'random'} a str representing whether to generate a uniform prior or randomly sample one from a Dirichlet distribution. 

        alpha: parameter of the distribution, of shape `(num_states)`. Each element must be greater than or equal to 0. By default set to all ones. Varying this parameter varies the entropy of the prior over states.

    Returns: 
        sample: np.ndarray of shape `(num_states)`
    """
    if type == 'uniform':
        sample = np.ones(num_states) / num_states # TODO: find how to generate uniform with alpha param in dirichlet
    elif type == 'random':
        if alpha is None:
            alpha = np.ones(num_states)
        sample = np.random.default_rng().dirichlet(alpha=alpha)
    else:
        raise ValueError(f"The argument `prior_type` can take values {{'uniform', 'random'}}, but received {type}.")
    return sample