import numpy as np

from functools import cached_property
from ultk.language.ib.ib_structure import IBStructure
from ultk.language.ib.ib_utils import safe_log, kl_divergence
from ultk.language.semantics import Meaning
from ultk.util.frozendict import FrozenDict


# TODO: May or may not make this frozen, who knows
class IBLanguage:
    structure: IBStructure
    expressions: tuple[FrozenDict[Meaning, float], ...]
    expressions_prior: np.ndarray

    def __init__(
        self,
        structure: IBStructure,
        expressions: tuple[FrozenDict[Meaning, float], ...],
    ):
        if len(expressions) == 0:
            raise ValueError("Cannot have no expressions")
        for expression in expressions:
            for meaning in structure.meanings:
                if meaning not in expression:
                    raise ValueError("Expression list is invalid for given structure")
        self.structure = structure
        self.expressions = expressions

    # qwm: Matrix to go from meaning probability distribution to word probability distribution
    @cached_property
    def qwm(self) -> np.ndarray:
        return np.array(
            [
                [expression[meaning] for meaning in self.structure.meanings]
                for expression in self.expressions
            ]
        )

    # qmw: Matrix to go from word probability distribution to meaning probability distribution
    @cached_property
    def qmw(self) -> np.ndarray:
        # Apply Bayes' rule
        return (
            self.qwm.transpose()
            * self.structure.meanings_prior[:, None]
            / self.expressions_prior
        )

    # TODO: CHECK THIS MORE EXTENSIVELY
    @cached_property
    def complexity(self) -> float:
        return np.sum(
            safe_log(self.qwm / self.expressions_prior[:, None])
            * (self.qwm * self.structure.meanings_prior)
        )

    @cached_property
    def expressions_prior(self) -> np.ndarray:
        return self.qwm @ self.structure.meanings_prior

    # The reconstructed meanings from the decoder
    # TODO: Check this
    @cached_property
    def reconstructed_meanings(self) -> np.ndarray:
        return self.structure.mu @ self.qmw

    # Expected KL Divergence for the language
    # TODO: Check this
    @cached_property
    def expected_divergence(self) -> float:
        left = self.qwm * self.structure.meanings_prior
        # This should be able to be done better
        right = np.array(
            [
                [kl_divergence(k, r) for k in self.structure.mu.T]
                for r in self.reconstructed_meanings.T
            ]
        )
        return np.sum(left * right)

    # I(W; U): Accuracy of the lexicon
    # Note: self.structure.mutual_information - self.iwu == self.expected_divergence
    # TODO: Check this extensively
    @cached_property
    def iwu(self) -> float:
        A = safe_log(
            self.reconstructed_meanings / self.structure.referents_prior[:, None]
        )
        return np.sum(A * (self.reconstructed_meanings * self.expressions_prior))
