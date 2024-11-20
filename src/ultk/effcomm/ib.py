"""Re-implementation of the IBNamingModel at https://github.com/nogazs/ib-color-naming/blob/master/src/ib_naming_model.py."""

import numpy as np
from ultk.util.io import read_pickle, write_pickle
from rdot.information import mutual_info, information_cond, gNID
from rdot.optimizers.ib import IBOptimizer, IBResult
from rdot.probability import joint

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
        Calculates the deviation from the IB curve, E[D[m||m_hat]] = I(M;U) - I(W;U).

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
        return read_pickle(fn)


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

    qW_M, complexity, accuracy, beta = zip(*[(res.qxhat_x, res.rate, res.accuracy, res.beta) for res in results if res is not None])

    IB_curve = (np.array(complexity), np.array(accuracy))
    
    naming_model = IBNamingModel(
        pM[:, None],
        pU_M,
        beta,
        IB_curve,
        qW_M,
    )

    return naming_model    