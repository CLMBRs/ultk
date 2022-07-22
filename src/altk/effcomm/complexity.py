import numpy as np
from agent import Speaker

def encoder_complexity(encoder: Speaker, prior: np.ndarray) -> float:
    """Compute the cognitive complexity for the encoder q, given by the information rate of compressing meanings M into words W:

        $I(M;W) = \sum_m p(m) \sum_w q(w|m) log [q(w|m) / q(w)]$
    
    where 
        $q(w) = \sum_m p(m) q(w|m) $

    Args: 
        encoder: a Speaker to compute the complexity (information rate) of

        prior: the probability distribution over referents.
    """
    total = 0
    # p(m)
    for i, prob_referent in enumerate(prior):
        referent_sum = []
        for j, expression_weights in enumerate(encoder.weights.T):
            expression_distribution = expression_weights / expression_weights.sum()
            # q(w)
            prob_expression = np.dot(prior, expression_distribution)
            # q(w|m)
            prob_expression_given_referent = expression_distribution[i]
            # log [encoder/prior]
            log_term = np.log(prob_expression_given_referent / prob_expression)

            referent_sum.append(prob_expression_given_referent * log_term)
        total += prob_referent * sum(referent_sum)
    
    return total
