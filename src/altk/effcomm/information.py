"""Helper functions for Rate-Distortion based (including Information Bottleneck) efficient communication analyses."""

import numpy as np
from altk.language.language import Language
from altk.language.semantics import Universe, Referent
from altk.effcomm.agent import LiteralSpeaker, BayesianListener
from altk.effcomm import util
from embo import InformationBottleneck
from typing import Callable


def information_rate(source: np.ndarray, encoder: np.ndarray) -> float:
    """Compute the information rate / complexity of the encoder q(w|m) as $I[W:M]$."""
    pXY = util.joint(pY_X=encoder, pX=source)
    return util.MI(pXY=pXY)


##############################################################################
# Rate-Distortion Theory
##############################################################################


def get_rd_curve(
    prior: np.ndarray,
    dist_mat: np.ndarray,
    betas: np.ndarray = None,
) -> list[tuple[float]]:
    """Use the Blahut Arimoto algorithm to obtain a list of (rate, distortion) points."""
    if betas is None:
        # B-A gets a bit sparse in low-rate regions for just one np.linspace
        # betas = np.linspace(start=0, stop=2**7, num=50)
        betas = np.concatenate([
            np.linspace(start=0, stop=0.29, num=333),
            np.linspace(start=0.3, stop=0.9, num=333),
            np.linspace(start=1.0, stop=2**7, num=334),
        ])
    # TODO: unify or something
    prior = np.array(prior)
    dist_mat = np.array(dist_mat)

    rd = lambda beta: blahut_arimoto(dist_mat, p_x=prior, beta=beta)["final"]
    pareto_points = [rd(beta) for beta in betas]
    return pareto_points


def expected_distortion(
    p_x: np.ndarray, p_xhat_x: np.ndarray, dist_mat: np.ndarray
) -> float:
    """$D[X, \hat{X}] = \sum_x p(x) \sum_{\hat{x}} p(\hat{x}|x) \cdot d(x, \hat{x})$"""
    return np.sum(p_x @ (p_xhat_x * dist_mat))


def compute_rate_distortion(
    p_x,
    p_xhat_x,
    dist_mat,
) -> tuple[np.ndarray]:
    """Compute the information rate $I(X;\hat{X})$ and total distortion $D[X, \hat{X}]$ of a joint distribution defind by $P(X)$ and $P(\hat{X}|X)$.

    Args:
        p_x: array of shape `|X|` the prior probability of an input symbol (i.e., the source)

        p_xhat_x: array of shape `(|X|, |X_hat|)` the probability of an output symbol given the input

        dist_mat: array of shape `(|X|, |X_hat|)` representing the distoriton matrix between the input alphabet and the reconstruction alphabet.

    Returns:
        a (rate, distortion) tuple containing the information rate (in bits) of compressing X into X_hat and the expected distortion between X, X_hat
    """
    return (
        information_rate(p_x, p_xhat_x),
        expected_distortion(p_x, p_xhat_x, dist_mat),
    )


def blahut_arimoto(
    dist_mat: np.ndarray,
    p_x: np.ndarray,
    beta: float,
    max_it: int = 200,
    eps: float = 1e-5,
    ignore_converge: bool = False,
) -> tuple[float]:
    """Compute the rate-distortion function of an i.i.d distribution

    Args:
        dist_mat: array of shape `(|X|, |X_hat|)` representing the distortion matrix between the input alphabet and the reconstruction alphabet. dist_mat[i,j] = dist(x[i],x_hat[j]). In this context, X is a random variable representing the a speaker's meaning (target referent), and X_hat is a random variable representing a listener's meaning (guessed  referent).

        p_x: (1D array of shape `|X|`) representing the probability mass function of the source. In this context, the prior over states of nature.

        beta: (scalar) the slope of the rate-distoriton function at the point where evaluation is required

        max_it: max number of iterations

        eps: accuracy required by the algorithm: the algorithm stops if there is no change in distoriton value of more than 'eps' between consequtive iterations

        ignore_converge: whether to run the optimization until `max_it`, ignoring the stopping criterion specified by `eps`.

    Returns:
        a dict of the form

            {
                'final': a tuple of (rate, distortion) values. This is the rate (in bits) of compressing X into X_hat, and distortion between X, X_hat

                'trajectory': a list of the (rate, distortion) points discovered during optimization
            }
    """
    # start with iid conditional distribution, as p(x) may not be uniform
    p_xhat_x = np.tile(p_x, (dist_mat.shape[1], 1)).T

    # normalize
    p_x /= np.sum(p_x)
    p_xhat_x /= np.sum(p_xhat_x, 1, keepdims=True)

    it = 0
    traj = []
    distortion = 2 * eps
    converged = False
    while not converged:
        it += 1
        distortion_prev = distortion

        # p(x_hat) = sum p(x) p(x_hat | x)
        p_xhat = p_x @ p_xhat_x

        # p(x_hat | x) = p(x_hat) exp(- beta * d(x_hat, x)) / Z
        p_xhat_x = np.exp(-beta * dist_mat) * p_xhat
        p_xhat_x /= np.expand_dims(np.sum(p_xhat_x, 1), 1)

        # update for convergence check
        rate, distortion = compute_rate_distortion(p_x, p_xhat_x, dist_mat)

        # collect point
        traj.append((rate, distortion))

        # convergence check
        if ignore_converge:
            converged = it == max_it
        else:
            converged = it == max_it or np.abs(distortion - distortion_prev) < eps

    return {
        "final": (rate, distortion),
        "trajectory": traj,
    }


##############################################################################
# Information Bottleneck
##############################################################################

# === Main IB methods ===

def get_ib_curve(
    prior: np.ndarray,
    meaning_dists: np.ndarray,
    maxbeta: float,
    minbeta: float,
    numbeta: float,
    processes: int = 1, 
    curve_type: str = "informativity",        
) -> tuple[float]:
    """Get a list of (complexity, accuracy) or (complexity, distortion) points. A minimal wrapper of `get_bottleneck.`

    Args:
        prior: array of shape `|meanings|`

        meaning_dists: array of shape `(|meanings|, |meanings|)` representing the distribution over world states given meanings.

        curve_type: {'informativity', 'comm_cost'} specifies whether to return the (classic) IB axes of informativity vs. complexity, or the more Rate-Distortion Theory aligned axes of comm_cost vs. complexity. The latter can be obtained easily from the former by subtracting each informativity value from I[M:U], which is a constant for all languages in the same domain.

        maxbeta: the maximum value of beta to use to compute the curve.

        minbeta: the minimum value of beta to use.

        numbeta: the number of (equally-spaced) beta values to consider to compute the curve.

        processes: number of cpu threads to run in parallel (default = 1)    

    Returns:
        an array of shape `(num_points, 2)` representing the list of (accuracy/comm_cost, complexity) points on the information plane.
    """

    complexity, accuracy, distortion = get_bottleneck(prior, meaning_dists, maxbeta, minbeta, numbeta, processes)
    if curve_type == "comm_cost":
        return np.array(list(zip(
            distortion,
            complexity,
        ))) # expected kl divergence, complexity

    else:
        points = np.array(list(zip(
            accuracy,
            complexity,
        ))) # informativity, complexity
    return points
    

def get_bottleneck(
    prior: np.ndarray,
    meaning_dists: np.ndarray,
    maxbeta: float,
    minbeta: float,
    numbeta: float,
    processes: int = 1,
) -> np.ndarray:
    """Compute the IB curve bound (I[M:W] vs. I[W:U]). We use the embo package, which does not allow one to specify the number of betas, which means some interpolation might be necessary later.

    Args:
        prior: array of shape `|meanings|`

        meaning_dists: array of shape `(|meanings|, |meanings|)` representing the distribution over world states given meanings.

        curve_type: {'informativity', 'comm_cost'} specifies whether to return the (classic) IB axes of informativity vs. complexity, or the more Rate-Distortion Theory aligned axes of comm_cost vs. complexity. The latter can be obtained easily from the former by subtracting each informativity value from I[M:U], which is a constant for all languages in the same domain.

        maxbeta: the maximum value of beta to use to compute the curve.

        minbeta: the minimum value of beta to use.

        numbeta: the number of (equally-spaced) beta values to consider to compute the curve.

        processes: number of cpu threads to run in parallel (default = 1)

    Returns:
        a tuple of arrays (I[M:W], I[W:U], E[D_KL[M || M']]),
        where each array is of shape `numbeta`.
    """
    # N.B.: embo only uses numpy and scipy
    prior = np.array(prior)
    meaning_dists = np.array(meaning_dists)

    joint_pmu = util.joint(meaning_dists, prior)  # P(u) = P(m)
    I_mu = util.MI(joint_pmu)

    # I[M:W], I[W:U], H[W], beta
    I_mw, I_wu, _, _ = InformationBottleneck(
        pxy=joint_pmu,
        maxbeta=maxbeta,
        minbeta=minbeta,
        numbeta=numbeta,
        processes=processes,
        ).get_bottleneck()

    return I_mw, I_wu, I_mu - I_wu


def ib_complexity(
    language: Language,
    prior: np.ndarray,
) -> float:
    """Compute the IB encoder complexity of a language $I[M:W]$."""
    return float(
        information_rate(
            source=prior,
            encoder=language_to_ib_encoder_decoder(
                language,
                prior,
            )["encoder"],
        )
    )


def ib_informativity(
    language: Language,
    prior: np.ndarray,
    meaning_dists: np.ndarray,
) -> float:
    """Compute the expected informativity (accuracy) $I[W:U]$ of a lexicon.

    Args:
        language: the Language to measure for informativity

        prior: communicative need distribution

        meaning_dists: array of shape `(|meanings|, |meanings|)` representing the distribution over world states given meanings.

    Returns:
        the informativity of the language I[W:U] in bits.
    """
    return float(
        util.MI(
            language_to_joint_distributions(language, prior, meaning_dists)["joint_pwu"]
        )
    )


def ib_comm_cost(
    language: Language,
    prior: np.ndarray,
    meaning_dists: np.ndarray,
) -> float:
    """Compute the IB communicative cost, i.e. expected KL-divergence betweeen speaker and listener meanings, for a language.

    Args:
        language: the Language to measure for communicative cost

        prior: communicative need distribution

        meaning_dists: array of shape `(|meanings|, |meanings|)` representing the distribution over world states given meanings.

    Returns:
        the communicative cost, $\mathbb{E}[D_{KL}[M || \hat{M}]] = I[M:U] - I[W:U]$ in bits.
    """
    dists = language_to_joint_distributions(language, prior, meaning_dists)
    return float(util.MI(dists["joint_pmu"]) - util.MI(dists["joint_pwu"]))


def language_to_joint_distributions(
    language: Language,
    prior: np.ndarray,
    meaning_dists: np.ndarray,
) -> float:
    """Given a Language, get P(M,U) the joint distribution over meanings and referents, and P(W,U) the joint distribution over words and referents.

    Args:
        language: the Language to convert to distributions

        prior: communicative need distribution

        meaning_dists: array of shape `(|meanings|, |meanings|)` representing the distribution over world states given meanings.
        
    Returns:
        a dict of the form

            {
            "joint_pmu": an array of shape `(|U|, |M|)` representing P(U, M)
            "joint_pwu": an array of shape `(|W|, |U|)` representing P(W, U)
            }

    """
    system = language_to_ib_encoder_decoder(language, prior)
    encoder = system["encoder"]
    decoder = system["decoder"]

    return encoder_decoder_to_joint_distributions(encoder, decoder, meaning_dists, prior)

def encoder_decoder_to_joint_distributions(encoder: np.ndarray, decoder: np.ndarray, meaning_dists: np.ndarray, prior: np.ndarray) -> dict[str, np.ndarray]:
    """
    Args:
        encoder: array of shape `(|M|, |W|)` representing P(W | M)

        decoder: array of shape `(|W|, |M|)` representing P(M | W)

        conditional_pum: array of shape `(|M|, |U|)` representing P(U | M)

        prior: array of shape `|M|` representing P(M)

    Returns:
        a dict of the form

            {
            "joint_pmu": an array of shape `(|U|, |M|)` representing P(U, M)
            "joint_pwu": an array of shape `(|W|, |U|)` representing P(W, U)
            }
    """
    conditional_puw = deterministic_decoder(decoder, meaning_dists)
    joint_pmu = util.joint(meaning_dists, prior)
    p_w = util.marginalize(encoder, prior)
    joint_pwu = util.joint(conditional_puw, p_w)
    return {
        "joint_pmu": joint_pmu,
        "joint_pwu": joint_pwu,
    }

    
def ib_encoder_to_point(
    meaning_dists: np.ndarray,
    prior: np.ndarray,
    encoder: np.ndarray,
    decoder: np.ndarray = None,    
) -> tuple[float]:
    """Return (complexity, accuracy, comm_cost) IB coordinates.
    
    Args:
        meaning_dists: array of shape `(|meanings|, |meanings|)` representing the distribution over world states given meanings.        

        prior: array of shape `|M|` representing the cognitive source

        encoder: array of shape `(|M|, |W|)` representing P(W | M)

        decoder: array of shape `(|W|, |M|)` representing P(M | W).  By default is None, and the Bayesian optimal decoder will be inferred.
    """
    # TODO: be consistent about tensors vs arrays 
    encoder = np.array(encoder)
    meaning_dists = np.array(meaning_dists)
    prior = np.array(prior)
    if decoder is not None:
        decoder = np.array(decoder)
    else:
        decoder = util.bayes(encoder, prior)

    dists = encoder_decoder_to_joint_distributions(encoder, decoder,meaning_dists, prior)

    complexity = information_rate(prior, encoder)
    accuracy = util.MI(dists["joint_pwu"])
    distortion = util.MI(dists["joint_pmu"]) - accuracy

    return (complexity, accuracy, distortion)

# === IB Helpers ===


def language_to_ib_encoder_decoder(
    language: Language,
    prior: np.ndarray,
) -> dict[str, np.ndarray]:
    """Convert a Language, a mapping of words to meanings, to IB encoder, q(w|m) and IB decoder q(m|w).

    Args:
        language: the lexicon from which to infer a speaker (encoder).

        prior: communicative need distribution

    Returns:
        a dict of the form
        {
            "encoder": np.ndarray of shape `(|meanings|, |words|)`,
            "decoder": np.ndarray of shape `(|words|, |meanings|)`,
        }
    """
    # In the IB framework, the encoder is _typically_ a literal speaker and the decoder is a bayes optimal listener.
    speaker = LiteralSpeaker(language)
    speaker.weights = util.rows_zero_to_uniform(speaker.normalized_weights())
    listener = BayesianListener(speaker, prior)
    return {
        "encoder": speaker.normalized_weights(),
        "decoder": listener.normalized_weights(),
    }

def deterministic_decoder(
    decoder: np.ndarray, meaning_distributions: np.ndarray
) -> np.ndarray:
    """Compute $\hat{m}_{w}(u) = \sum_m  p(m|w) \cdot m(u) $

    Args:
        decoder: array of shape `(|words|, |meanings|)`

        meaning_distributions: array of shape `(|meanings|, |meanings|)`

    Returns:
        array of shape `(|words|, |meanings|)` representing the 'optimal' deterministic decoder
    """
    return decoder @ meaning_distributions
