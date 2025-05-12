import numpy as np

from functools import cached_property
from ultk.ib.ib_structure import IBStructure
from ultk.ib.ib_utils import mutual_information, kl_divergence
from ultk.language.semantics import Meaning
from ultk.util.frozendict import FrozenDict


# TODO: May or may not make this frozen, who knows
class IBLanguage:
    structure: IBStructure
    # qwm: Matrix to go from meaning probability distribution to word probability distribtion
    qwm: np.ndarray

    def __init__(
        self,
        structure: IBStructure,
        qwm: np.ndarray,
    ):
        if len(qwm.shape) != 2:
            raise ValueError("Must be a 2d matrix")
        if qwm.shape[1] != structure.mu.shape[1]:
            raise ValueError(
                f"Input matrix is for {qwm.shape[1]} meanings, not {len(structure.meanings)}"
            )
        self.structure = structure
        self.qwm = qwm

    # qmw: Matrix to go from word probability distribution to meaning probability distribution
    @cached_property
    def qmw(self) -> np.ndarray:
        # Apply Bayes' rule
        return (
            self.qwm.T * self.structure.meanings_prior[:, None] / self.expressions_prior
        )

    # This is consistent with IB Color naming data
    @cached_property
    def complexity(self) -> float:
        return mutual_information(
            self.qwm, self.expressions_prior, self.structure.meanings_prior
        )

    @cached_property
    def expressions_prior(self) -> np.ndarray:
        return self.qwm @ self.structure.meanings_prior

    # The reconstructed meanings from the decoder
    @cached_property
    def reconstructed_meanings(self) -> np.ndarray:
        return self.structure.mu @ self.qmw

    # This is used in multiple places
    # TODO: Check this, seems wrong in optimizer
    @cached_property
    def divergence_array(self) -> np.ndarray:
        return np.array(
            [
                [kl_divergence(k, r) for k in self.structure.mu.T]
                for r in self.reconstructed_meanings.T
            ]
        )

    # Expected KL Divergence for the language
    @cached_property
    def expected_divergence(self) -> float:
        left = self.qwm * self.structure.meanings_prior
        return np.sum(left * self.divergence_array)

    # I(W; U): Accuracy of the lexicon
    # Note: self.structure.mutual_information - self.iwu == self.expected_divergence
    @cached_property
    def iwu(self) -> float:
        return mutual_information(
            self.reconstructed_meanings,
            self.structure.referents_prior,
            self.expressions_prior,
        )


def language_from_meaning_dict(
    expressions: tuple[FrozenDict[Meaning[float], float], ...], structure: IBStructure
) -> IBLanguage:
    return IBLanguage(
        structure,
        np.array(
            [
                [expression[meaning] for meaning in structure.meanings]
                for expression in expressions
            ]
        ),
    )
