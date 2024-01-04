import itertools
import pandas as pd
import pytest
from copy import deepcopy

from ultk.language.semantics import Universe
from ultk.language.semantics import Meaning
from ultk.language.semantics import Referent

forces = ("weak", "strong")
flavors = ("epistemic", "deontic")


class TestSemantics:
    points = [
        {"name": f"{force}+{flavor}", "force": force, "flavor": flavor}
        for (force, flavor) in itertools.product(forces, flavors)
    ]
    dataframe = pd.DataFrame(points)
    universe = Universe.from_dataframe(dataframe)

    ref1 = Referent(
        name="weak+epistemic", properties={"force": "weak", "flavor": "epistemic"}
    )

    def test_universe_from_df(self):
        universe_referents = []
        for referent in TestSemantics.universe.referents:
            copied = deepcopy(referent.__dict__)
            copied.pop("_frozen", None)
            universe_referents.append(copied)

        assert TestSemantics.points == [referent for referent in universe_referents]

    def test_referent_match(self):
        ref2 = Referent(
            name="weak+epistemic", properties={"force": "weak", "flavor": "epistemic"}
        )
        assert TestSemantics.ref1 == ref2

    def test_referent_mismatch(self):
        ref3 = Referent(
            name="neutral+epistemic",
            properties={"force": "neutral", "flavor": "epistemic"},
        )
        assert TestSemantics.ref1 != ref3

    def test_universe_match(self):
        second_dataframe = pd.DataFrame(TestSemantics.points)
        assert Universe.from_dataframe(second_dataframe) == TestSemantics.universe

    def test_universe_mismatch(self):
        ref_list = [TestSemantics.ref1]
        assert Universe(ref_list) != TestSemantics.universe

    def test_meaning_subset(self):
        ref_list = [TestSemantics.ref1]
        meaning = Meaning(
            ref_list, TestSemantics.universe
        )  # This meaning should exist within the set of semantics
        assert TestSemantics.ref1 in meaning.referents

        with pytest.raises(ValueError):
            ref_list.append(
                Referent(
                    name="neutral+epistemic",
                    properties={"force": "neutral", "flavor": "epistemic"},
                )
            )
            meaning = Meaning(
                ref_list, TestSemantics.universe
            )  # This meaning should not
