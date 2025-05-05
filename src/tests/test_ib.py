import pytest
from ultk.language.ib.ib_language import IBLanguage
from ultk.language.ib.ib_structure import IBStructure
from ultk.language.semantics import Meaning, Referent
from ultk.util.frozendict import FrozenDict

import numpy as np


def generate_number_meanings(referents):
    return tuple(
        Meaning(FrozenDict({r: 1 if r.amount == i + 1 else 0 for r in referents}), None)
        for i in range(10)
    )


EPSILON = 0.0001


class TestIB:
    referents = tuple(Referent(str(i), amount=i) for i in range(1, 11))

    meanings = generate_number_meanings(referents)

    structure = IBStructure(
        referents, meanings, tuple(2 / (3**i) for i in range(1, 11))
    )

    expr_all = FrozenDict({meaning: 1 for meaning in meanings})
    expr_one = FrozenDict(
        {meaning: meaning[Referent("1", amount=1)] for meaning in meanings}
    )
    expr_many = FrozenDict(
        {meaning: int(meaning[Referent("1", amount=1)] == 0) for meaning in meanings}
    )

    language_dual = IBLanguage(structure, expressions=(expr_one, expr_many))
    language_all = IBLanguage(structure, expressions=(expr_all,))

    def test_expression_priors(self):
        assert abs(np.sum(self.language_dual.expressions_prior) - 1) < EPSILON
        assert abs(np.sum(self.language_all.expressions_prior) - 1) < EPSILON
        assert abs(self.language_dual.expressions_prior[0] - 2 / 3) < EPSILON
        assert abs(self.language_dual.expressions_prior[1] - 1 / 3) < EPSILON
        assert abs(self.language_all.expressions_prior[0] - 1) < EPSILON

    # TODO: Test specific values
    def test_referent_priors(self):
        assert abs(np.sum(self.structure.referents_prior) - 1) < EPSILON
        assert all(
            self.structure.referents_prior[i] == self.structure.meanings_prior[i]
            for i in range(len(self.meanings))
        )

    # Tests that the expected KL Divergence is equal to I(M; U) - I(W; U)
    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in divide:RuntimeWarning"
    )
    def test_divergence_iwu_equality(self):
        assert (
            abs(
                self.structure.mutual_information
                - self.language_all.iwu
                - self.language_all.expected_divergence
            )
            < EPSILON
        )
        assert (
            abs(
                self.structure.mutual_information
                - self.language_dual.iwu
                - self.language_dual.expected_divergence
            )
            < EPSILON
        )

    # TODO: Get a specific value for language_dual
    def test_complexity(self):
        assert abs(self.language_all.complexity) < EPSILON

    # TODO: Test specific values
    def test_reconstructed_meanings(self):
        assert all(
            np.sum(r) - 1 < EPSILON for r in self.language_all.reconstructed_meanings.T
        )
        assert all(
            np.sum(r) - 1 < EPSILON for r in self.language_dual.reconstructed_meanings.T
        )

    # TODO: Write more tests
