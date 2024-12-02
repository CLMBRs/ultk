"""Re-implementation of the IBNamingModel at https://github.com/nogazs/ib-color-naming/blob/master/src/ib_naming_model.py."""

import numpy as np
import warnings
from ultk.util.io import read_pickle, write_pickle
from ultk.language.language import Language, Expression, Meaning, FrozenDict, Universe
from .tools import mutual_info, information_cond
from .ib import IBOptimizer, IBResult
from ..probability import joint

##############################################################################
# Base IBNamingModel class
##############################################################################


class IBNamingModel:
    """A model for simulating Information Bottleneck (IB) naming systems."""

    def __init__(
        self,
        pM: np.ndarray,
        pU_M: np.ndarray,
        betas: np.ndarray,
        IB_curve: tuple[np.ndarray, np.ndarray],
        qW_M: np.ndarray,
    ):
        """
        Initializes the IBNamingModel with given parameters.

        Args:
            pM (np.ndarray): Prior distribution over meanings. Expected shape is (n, 1).
            pU_M (np.ndarray): Conditional distribution of observations given meanings.
            betas (np.ndarray): Array of beta values used for the IB curve.
            IB_curve (tuple[np.ndarray, np.ndarray]): tuple containing complexity and accuracy values along the IB curve.
            qW_M (np.ndarray): Initial encoder (naming system) matrix.
        """
        self.pM = pM if len(pM.shape) == 2 else pM[:, None]
        self.pU_M = pU_M
        self.I_MU = mutual_info(pU_M * self.pM)
        self.betas = betas
        self.IB_curve = np.array(IB_curve)
        self.qW_M = qW_M
        self.qW_M_orig = None
        self.F = IB_curve[0] - betas * IB_curve[1]

    def m_hat(self, qW_M: np.ndarray) -> np.ndarray:
        """
        Computes the optimal Bayesian listener corresponding to the encoder.

        Args:
            qW_M (np.ndarray): Encoder (naming system) matrix.

        Returns:
            np.ndarray: Optimal decoder that corresponds to the given encoder.
        """
        pMW = qW_M * self.pM
        pM_W = pMW.T / pMW.sum(axis=0)[:, None]
        return pM_W.dot(self.pU_M)

    def complexity(self, pW_M: np.ndarray) -> float:
        """
        Calculates the mutual information I(M;W) for a given encoder.

        Args:
            pW_M (np.ndarray): Encoder (naming system) matrix.

        Returns:
            float: Mutual information I(M;W).
        """
        return mutual_info(pW_M * self.pM)

    def accuracy(self, pW_M: np.ndarray) -> float:
        """
        Calculates the mutual information I(W;U) for a given encoder.

        Args:
            pW_M (np.ndarray): Encoder (naming system) matrix.

        Returns:
            float: Mutual information I(W;U).
        """
        pMW = pW_M * self.pM
        pWU = pMW.T @ self.pU_M
        return mutual_info(pWU)

    def d_IB(self, pW_M: np.ndarray) -> float:
        """
        Calculates the IB distortion for a given encoder, i.e. the KL divergence between speaker and listener meanings, $
        \mathbb{E}\left[D[m||\hat{m}]\right] = I(M;U) - I(W;U)$.

        Args:
            pW_M (np.ndarray): Encoder (naming system) matrix.

        Returns:
            float: Deviation from the optimal IB solution.
        """
        return self.I_MU - self.accuracy(pW_M)

    def fit(self, pW_M: np.ndarray) -> tuple[float, float, float, np.ndarray]:
        """
        Fits the naming system to the IB curve.

        Args:
            pW_M (np.ndarray): Encoder (naming system) matrix.

        Returns:
            tuple containing:
                - epsilon (float): Deviation from optimality of pW_M.
                - gnid (float): Generalized normalized information distance (gNID) between qW_M and qW_M_fit.
                - bl (float): Fitted value of beta.
                - qW_M_fit (np.ndarray): Optimal IB system at bl.
        """
        Fl = self.complexity(pW_M) - self.betas * self.accuracy(pW_M)
        dFl = Fl - self.F
        bl_ind = dFl.argmin()
        bl = self.betas[bl_ind]
        epsilon = dFl.min() / bl
        qW_M_fit = self.qW_M[bl_ind]
        gnid = gNID(pW_M, qW_M_fit, self.pM)
        return epsilon, gnid, bl, qW_M_fit

    def save(self, fn: str = "ib_naming_model.pkl") -> None:
        """Save as pickle binary."""
        write_pickle(fn, self)

    @classmethod
    def from_pickle(cls, fn: str):
        breakpoint()
        return read_pickle(fn)


# Helper
def gNID(pW_X: np.ndarray, pV_X: np.ndarray, pX: np.ndarray):
    """Compute Generalized Normalized Informational Distance (gNID, in Zaslavsky et al. 2018, SI, Section 3.2) between two encoders. Code credit: https://github.com/nogazs/ib-color-naming/blob/master/src/tools.py#L94

    Args:
        pW_X: first encoder of shape `(|meanings|, |words|)`

        pV_X: second encoder of shape `(|meanings|, |words|)`

        pX: prior over source variables of shape `(|meanings|,)`
    """
    if len(pX.shape) == 1:
        pX = pX[:, None]
    elif pX.shape[0] == 1 and pX.shape[1] > 1:
        pX = pX.T
    pXW = pW_X * pX
    pWV = pXW.T @ (pV_X)
    pWW = pXW.T @ (pW_X)
    pVV = (pV_X * pX).T @ (pV_X)
    score = 1 - mutual_info(pWV) / (np.max([mutual_info(pWW), mutual_info(pVV)]))
    if score < 0:
        # N.B.: gNID is not necessarily non-negative (See SI, Section 3.2, paragraph 2.)
        warnings.warn(f"Negative gNID: {score}.")
    return score


##############################################################################
# IB Bound computation
##############################################################################


def compute_bound(
    pU_M: np.ndarray,
    pM: np.ndarray,
    betas: np.ndarray = np.logspace(0, 10, 100),
    **kwargs,
) -> list[IBResult]:
    """
    Computes the IB bound based on input distributions.

    Args:
        pU_M (np.ndarray): Conditional distribution of observations given meanings.
        pM (np.ndarray, optional): Prior distribution over meanings. Defaults to None.
        betas (np.ndarray, optional): Range of beta values for the IB curve. Defaults to logspace(0, 10, 100).
        **kwargs: Additional parameters for the IB optimizer.

    Returns:
        list[IBResult]: List of results from the IB optimizer.
    """
    pxy = joint(pU_M, pM)
    optim = IBOptimizer(
        pxy,
        betas,
        **kwargs,
    )
    results = optim.get_results()
    return results


def get_ib_naming_model(
    pU_M: np.ndarray,
    pM: np.ndarray = None,
    **bound_kwargs,
) -> IBNamingModel:
    """
    Constructs an IBNamingModel by constructing the IB bound for the domain distribution P(M,U).

    Args:
        pU_M (np.ndarray): Conditional distribution of observations given meanings.
        pM (np.ndarray, optional): Prior distribution over meanings. Defaults to None.
        gammas (np.ndarray, optional): Range of gamma values for similarity selection. Defaults to logspace(-2, 2, 1000).
        **bound_kwargs: Additional parameters for IB bound computation. See `compute_bound` kwargs.

    Returns:
        IBNamingModel: An IBNamingModel instance configured with the computed IB bound.

    """
    results = compute_bound(pU_M, pM, **bound_kwargs)

    qW_M, complexity, accuracy, beta = zip(
        *[
            (res.qxhat_x, res.rate, res.accuracy, res.beta)
            for res in results
            if res is not None
        ]
    )

    IB_curve = (np.array(complexity), np.array(accuracy))

    naming_model = IBNamingModel(
        pM[:, None],
        pU_M,
        beta,
        IB_curve,
        qW_M,
    )

    return naming_model


##############################################################################
# Integration with ULTK Language
##############################################################################


def encoder_to_language(
    qW_M: np.ndarray,
    naming_model: IBNamingModel,
    universe: Universe,
    words: list[str] = None,
    name: str = None,
    natural: bool = False,
) -> Language:
    """Convert a stochastic encoder to a ULTK Language using an IBNamingModel bayesian decoder.

    Args:
        qW_M (np.ndarray): A stochastic matrix where rows correspond to meanings
            and columns correspond to words, defining the encoder.
        naming_model (IBNamingModel): An instance of the IBNamingModel used to
            decode the encoder into a language.
        universe (Universe): The universe containing referents and the structure
            in which the meanings are defined.
        words (list[str], optional): A list of word forms to use. If None, default
            numeric indices are used. Defaults to None.
        name (str, optional): The name of the resulting Language. Defaults to None.
        natural (bool, optional): Whether the resulting Language is a natural
            language. Defaults to False.

    Returns:
        Language: The constructed Language object, where each expression maps a
        word form to its corresponding meaning.
    """

    if words is None:
        words = range(qW_M.shape[1])

    return Language(
        expressions=tuple(
            [
                Expression(
                    form=str(words[i]),
                    meaning=Meaning[float](
                        FrozenDict(
                            {
                                # define each mapping from referent -> probability
                                universe.referents[chip_num]: qm[chip_num]
                                for chip_num in range(qW_M.shape[0])
                            }
                        ),
                        universe,
                    ),
                )
                for i, qm in enumerate(naming_model.m_hat(qW_M))
            ]
        ),
        name=name,
        natural=natural,
    )


def pU_M_from_similarity(gamma: float, sim_mat: np.ndarray) -> np.ndarray:
    """
    Computes the conditional distribution p(U|M) based on similarity.

    Args:
        gamma (float): Scaling factor for the similarity matrix.
        sim_mat (np.ndarray): Similarity matrix representing similarity between meanings (M) and referents (U).

    Returns:
        np.ndarray: Conditional distribution p(U|M).
    """
    pU_M = np.exp(gamma * sim_mat)
    pU_M /= pU_M.sum(axis=1, keepdims=True)
    return pU_M


def get_imu(gamma: float, sim_mat: np.ndarray, pM: np.ndarray = None) -> np.ndarray:
    """
    Calculates the mutual information I(M;U) for a distribution p(U|M) ∝ exp(gamma * sim(u, m)).

    Args:
        gamma (float): Scaling factor for the similarity matrix.
        sim_mat (np.ndarray): Similarity matrix representing similarity between meanings (M) and referents (U).
        pM (np.ndarray, optional): Prior distribution over meanings (M). Defaults to a uniform distribution.

    Returns:
        np.ndarray: Mutual information I(M;U).
    """
    return information_cond(
        pB_A=pU_M_from_similarity(gamma, sim_mat),
        pA=pM if pM is not None else np.full(sim_mat.shape[0], 1 / sim_mat.shape[0]),
    )


def select_gamma(
    similarity_matrix: np.ndarray,
    pM: np.ndarray = None,
    gammas: np.ndarray = np.logspace(-2, 2, 1000),
) -> tuple[float, float, int, np.ndarray, np.ndarray]:
    """
    Selects the gamma value that corresponds to the midpoint of I(M;U) for a distribution p(U|M) ∝ exp(gamma * sim(u, m)).

    Args:
        similarity_matrix (np.ndarray): Matrix encoding pairwise similarities between meanings (M) and referents (U).
        pM (np.ndarray, optional): Communicative need distribution over meanings (M). Defaults to None.
        gammas (np.ndarray, optional): Range of gamma values to sample. Defaults to logspace(-2, 2, 1000).

    Returns:
        tuple: A tuple containing:
            - float: Gamma value corresponding to the midpoint of I(M;U).
            - float: Midpoint of I(M;U).
            - int: Index of the midpoint in the gamma array.
            - np.ndarray: Array of gamma values used.
            - np.ndarray: Array of computed I(M;U) values.
    """
    imus = np.array([get_imu(g, similarity_matrix, pM) for g in gammas])
    mid = (np.max(imus) - np.min(imus)) / 2
    mid_ind = np.argmin((imus - mid) ** 2)
    return gammas[mid_ind], mid, mid_ind, gammas, imus
