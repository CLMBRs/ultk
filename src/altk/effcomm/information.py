"""Helper functions for Rate-Distortion based (including Information Bottleneck) efficient communication analyses."""

import numpy as np
from altk.language.language import Language
from altk.language.semantics import Universe, Referent
from altk.effcomm.agent import LiteralSpeaker, BayesianListener
from altk.effcomm import util
from embo import InformationBottleneck
from typing import Callable


def information_rate(source: np.ndarray, encoder: np.ndarray) -> float:
    """Compute the information rate / complexity of the encoder q(w|m) as I[W:M]."""
    pXY = util.joint(pY_X=encoder, pX=source)
    return util.MI(pXY=pXY)


##############################################################################
# Rate-Distortion Theory
##############################################################################


def get_rd_curve(
    prior: np.ndarray,
    dist_mat: np.ndarray,
    betas: np.ndarray = np.linspace(start=0, stop=2**7, num=1500),
) -> list[tuple[float]]:
    """Use the Blahut Arimoto algorithm to obtain a list of points."""
    rd = lambda beta: blahut_arimoto(dist_mat, p_x=prior, beta=beta)["final"]
    pareto_points = [rd(beta) for beta in betas]
    return pareto_points


def expected_distortion(
    p_x: np.ndarray, p_xhat_x: np.ndarray, dist_mat: np.ndarray
) -> float:
    """D[X, Xhat] = sum_x p(x) sum_xhat p(xhat|x) * d(x, xhat)"""
    return np.sum(p_x @ (p_xhat_x * dist_mat))


def compute_rate_distortion(
    p_x,
    p_xhat_x,
    dist_mat,
) -> tuple[np.ndarray]:
    """Compute the information rate I(X;Xhat) and total distortion D[X, Xhat] of a joint distribution defind by P(X) and P(Xhat|X).

    Args:
        p_x: (1D array of shape `|X|`) the prior probability of an input symbol (i.e., the source)

        p_xhat_x: (2D array of shape `(|X|, |Xhat|)`) the probability of an output symbol given the input

        dist_mat: (2D array of shape `(|X|, |X_hat|)`) representing the distoriton matrix between the input alphabet and the reconstruction alphabet.

    Returns:
        a tuple containing
        rate: rate (in bits) of compressing X into X_hat
        distortion: expected distortion between X, X_hat
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
        dist_mat: array of shape `(|X|, |X_hat|)` representing the distoriton matrix between the input alphabet and the reconstruction alphabet. dist_mat[i,j] = dist(x[i],x_hat[j]). In this context, X is a random variable representing the state of Nature, and X_hat is a random variable representing actions appropriate.

        p_x: (1D array of shape `|X|`) representing the probability mass function of the source. In this context, the prior over states of nature.

        beta: (scalar) the slope of the rate-distoriton function at the point where evaluation is required

        max_it: max number of iterations

        eps: accuracy required by the algorithm: the algorithm stops if there is no change in distoriton value of more than 'eps' between consequtive iterations

        ignore_converge: whether to run the optimization until `max_it`, ignoring the stopping criterion specified by `eps`.

    Returns:
        a dict containing

            'final': a tuple of (rate, distortion) values. This is the rate (in bits) of compressing X into X_hat, and distortion between X, X_hat

            'trajectory': a list of the (rate, distortion) points discovered during optimization
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
    space: Universe,
    decay: float,
    utility: str,
    curve_type: str = "informativity",
) -> np.ndarray:
    """Compute the IB curve bound (I[M:W] vs. I[W:U]) for a given semantic space. We use the embo package, which does not allow one to specify the number of betas, which means some interpolation might be necessary later.

    Args:
        prior: array of shape `|meanings|`

        space: the ModalMeaningSpace on which meanings are defined

        decay: parameter for meaning distribution p(u|m) generation

        utility: parameter for meaning distribution p(u|m) generation

        curve_type: {'informativity', 'comm_cost'} specifies whether to return the (classic) IB axes of informativity vs. complexity, or the more Rate-Distortion Theory aligned axes of comm_cost vs. complexity. The latter can be obtained easily from the former by subtracting each informativity value from I[M:U], which is a constant for all languages in the same domain.

    Returns:
        an array of shape `(num_points, 2)` representing the list of (accuracy/comm_cost, complexity) points on the information plane.
    """
    conditional_pum = generate_meaning_distributions(space, decay, utility)
    joint_pmu = util.joint(conditional_pum, prior)  # P(u) = P(m)
    I_mu = util.MI(joint_pmu)

    # I[M:W], I[W:U], H[W], beta
    I_mw, I_wu, _, _ = InformationBottleneck(pxy=joint_pmu).get_bottleneck()

    if curve_type == "comm_cost":
        points = np.array(
            list(zip(I_mu - I_wu, I_mw))
        )  # expected kl divergence, complexity
    else:
        points = np.array(list(zip(I_wu, I_mw)))  # informativity, complexity
    return points


def ib_complexity(
    language: Language,
    prior: np.ndarray,
) -> float:
    """Compute the IB encoder complexity of a language."""
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
    decay: float,
    utility: str,
) -> float:
    """Compute the expected informativity (accuracy) I[W:U] of a lexicon.

    Args:
        language: the Language to measure for informativity

        prior: communicative need distribution

        decay: parameter for meaning distribution p(u|m) generation

        utility: parameter for meaning distribution p(u|m) generation

    Returns:
        the informativity of the language I[W:U] in bits.
    """
    return float(
        util.MI(
            language_to_joint_distributions(language, prior, decay, utility)[
                "joint_pwu"
            ]
        )
    )


def ib_comm_cost(
    language: Language,
    prior: np.ndarray,
    decay: float,
    utility: str,
) -> float:
    """Compute the IB communicative cost, i.e. expected KL-divergence betweeen speaker and listener meanings, for a language.

    Args:
        language: the Language to measure for communicative cost

        prior: communicative need distribution

        decay: parameter for meaning distribution p(u|m) generation

        utility: parameter for meaning distribution p(u|m) generation

    Returns:
        the communicative cost, E[D[M || \hat{M}]] = I[M:U] - I[W:U] in bits.
    """
    dists = language_to_joint_distributions(language, prior, decay, utility)
    return float(util.MI(dists["joint_pmu"]) - util.MI(dists["joint_pwu"]))


def language_to_joint_distributions(
    language: Language,
    prior: np.ndarray,
    decay: float,
    utility: str,
) -> float:
    """Given a Language, get P(M,U) the joint distribution over meanings and referents, and P(W,U) the joint distribution over words and referents.

    Args:
        language: the Language to convert to distributions

        prior: communicative need distribution

        decay: parameter for meaning distribution p(u|m) generation

        utility: parameter for meaning distribution p(u|m) generation
    """
    system = language_to_ib_encoder_decoder(language, prior)
    encoder = system["encoder"]
    decoder = system["decoder"]
    space = language.universe

    conditional_pum = generate_meaning_distributions(space, decay, utility)
    conditional_puw = deterministic_decoder(decoder, conditional_pum)
    joint_pmu = util.joint(conditional_pum, prior)
    p_w = util.marginalize(encoder, prior)
    joint_pwu = util.joint(conditional_puw, p_w)

    return {
        "joint_pmu": joint_pmu,
        "joint_pwu": joint_pwu,
    }


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
    # In the IB framework, the encoder is _typically_ a literal speaker and the decoder is a bayes optimal listener. TODO: There are obviously other possible choices here.
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
    """Compute \hat{m}_{w}(u) = sum_m [ p(m|w) * m(u) ]

    Args:
        decoder: array of shape `(|words|, |meanings|)`

        meaning_distributions: array of shape `(|meanings|, |meanings|)`

    Returns:
        array of shape `(|words|, |meanings|)` representing the 'optimal' deterministic decoder
    """
    return decoder @ meaning_distributions


def generate_meaning_distributions(
    space: Universe,
    decay: float,
    cost: Callable[[Referent, Referent], float],
) -> np.ndarray:
    """Generate a conditional distribution over world states given meanings, p(u|m), for each meaning.

    Args:
        space: the ModalMeaningSpace on which meanings are defined

        decay: a float in [0,1]. controls informativity, by decaying how much probability mass is assigned to perfect recoveries. As decay approaches 0, only perfect recovery is rewarded (which overrides any partial credit structure built into the utility/cost function). As decay approaches 1, the worst guesses become most likely.

        cost: a cost function defining the pairwise communicative cost for confusing one Referent in the Universe with another. If you have a (scaled) communicative utility matrix, a natural choice for cost might be `lambda x, y: 1 - utility(x, y)`.

    Returns:
        p_u_m: an array of shape `(|space.referents|, |space.referents|)`
    """

    # construct p(u|m) for each meaning
    meaning_distributions = np.array(
        [[decay ** cost(m, u) for u in space.referents] for m in space.referents]
    )
    # each row sums to 1.0
    np.seterr(divide="ignore", invalid="ignore")
    meaning_distributions = np.nan_to_num(
        meaning_distributions / meaning_distributions.sum(axis=1, keepdims=True)
    )

    return meaning_distributions
