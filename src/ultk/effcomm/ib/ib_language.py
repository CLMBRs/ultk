import numpy as np

from functools import cached_property
from ultk.effcomm.ib.ib_structure import IBStructure
from ultk.effcomm.ib.ib_utils import IB_EPSILON, mutual_information, kl_divergence
from ultk.language.semantics import Meaning
from ultk.util.frozendict import FrozenDict


class IBLanguage:
    """A language has expressions which are mapped to from meanings and which can map to expressions.

    Properties:
        structure: This is the structure in which the language exists.

        qwm: This is a conditional probaiblity matrix which maps a meaning distribution to expressions. Dimensions are ||W|| x ||M||

        qmw: Reconstructed conditional probability matrix which maps an expression distrubution to meanings. Created using Bayes' rule.
        Dimensions are ||M|| x ||W||.

        complexity: Mutual information between expressions and meanings. Formally I(W; M).

        expressions_prior: Probability distribution for expressions. Constructed from the structure's meaning priors and qwm.

        reconstructed_meanings: Conditional probability matrix which maps an expression distrubition to referents. Created using qmw and structure.pum.
        Dimensions are ||R|| x ||W||.

        divergence_array: Matrix which stores the different KL Divergences between the referent probability distrubutions per meaning and per expression.
        Dimensions are ||W|| x ||M||. (It is important to note that the KL Divergence function uses base 2 logarithms)

        expected_divergence: This is the expected KL Divergence between the language's reconstructed meanings and the structure's meanings.
        expected divergence = I(U; M) - I(W; U)

        iwu: The mutual information between the expressions of a langauge and the referents. Also referred to as accuracy.
    """

    structure: IBStructure
    qwm: np.ndarray

    def __init__(
        self,
        structure: IBStructure,
        qwm: np.ndarray,
    ):
        if len(qwm.shape) != 2:
            raise ValueError("Must be a 2d matrix")
        if qwm.shape[1] != structure.pum.shape[1]:
            raise ValueError(
                f"Input matrix is for {qwm.shape[1]} meanings, not {structure.pum.shape[1]}"
            )
        if (np.abs(np.sum(qwm, axis=0) - 1) > IB_EPSILON).any():
            raise ValueError(
                "All columns of conditional probability matrix must sum to 1"
            )
        self.structure = structure
        self.qwm = qwm

    @cached_property
    def qmw(self) -> np.ndarray:
        # Apply Bayes' rule
        return (
            self.qwm.T * self.structure.meanings_prior[:, None] / self.expressions_prior
        )

    @cached_property
    def complexity(self) -> float:
        return mutual_information(
            self.qwm, self.expressions_prior, self.structure.meanings_prior
        )

    @cached_property
    def expressions_prior(self) -> np.ndarray:
        # Normalization does become important at really small values
        intermediate = self.qwm @ self.structure.meanings_prior
        return intermediate / np.sum(intermediate)

    @cached_property
    def reconstructed_meanings(self) -> np.ndarray:
        # Normalization does become important at really small values
        intermediate = self.structure.pum @ self.qmw
        return intermediate / np.sum(intermediate, axis=0)

    @cached_property
    def divergence_array(self) -> np.ndarray:
        return np.array(
            [
                [kl_divergence(k, r) for k in self.structure.pum.T]
                for r in self.reconstructed_meanings.T
            ]
        )

    @cached_property
    def expected_divergence(self) -> float:
        left = self.qwm * self.structure.meanings_prior
        return np.sum(left * self.divergence_array)

    @cached_property
    def iwu(self) -> float:
        return mutual_information(
            self.reconstructed_meanings,
            self.structure.referents_prior,
            self.expressions_prior,
        )


def language_from_meaning_dict(
    expressions: tuple[FrozenDict[Meaning[float], float], ...],
    meanings: tuple[Meaning[float], ...],
    structure: IBStructure,
) -> IBLanguage:
    """Converts a tuple of expressions which map meanings to floats, meanings, and a structure into an IBLanguage.

    Args:
        expressions (tuple[FrozenDict[Meaning[float], float], ...]): The list of expressions. Each meaning is mapped to a float which is the conditional
        probability of the expression given the meeaning. (Effectively the value in there is q(w|m))
        meanings (tuple[Meaning[float], ...]): The meanings which the expressions reference
        structure (IBStructure): The structure the language will be based off of

    Returns:
        IBLanguage: Language which has the same expressions as what was passed in
    """
    return IBLanguage(
        structure,
        np.array(
            [
                [expression[meaning] for meaning in meanings]
                for expression in expressions
            ]
        ),
    )
