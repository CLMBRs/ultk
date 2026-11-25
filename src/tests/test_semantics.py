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
        refs = (TestSemantics.ref1,)
        assert Universe(refs) != TestSemantics.universe

    def test_meaning_from_function(self):
        meaning = Meaning.from_function(
            lambda referent: referent.force == "weak", TestSemantics.universe
        )
        assert meaning.mapping == tuple(
            referent.force == "weak" for referent in TestSemantics.universe.referents
        )

    def test_meaning_apply(self):
        weak = Meaning.from_function(
            lambda referent: referent.force == "weak", TestSemantics.universe
        )
        epistemic = Meaning.from_function(
            lambda referent: referent.flavor == "epistemic", TestSemantics.universe
        )
        both = Meaning.apply(lambda p, q: p and q, weak, epistemic)
        assert both.universe is TestSemantics.universe
        assert both.mapping == tuple(
            referent.name == "weak+epistemic"
            for referent in TestSemantics.universe.referents
        )
        with pytest.raises(ValueError):
            Meaning.apply(lambda p: p)
        other_universe = Universe((TestSemantics.ref1,))
        with pytest.raises(ValueError):
            Meaning.apply(lambda p, q: p, weak, Meaning((True,), other_universe))

    def test_meaning_equality_and_hash(self):
        mapping = (True, False, True, False)
        meaning = Meaning(mapping, TestSemantics.universe)
        # an equal, but distinct, universe
        equal_universe = Universe.from_dataframe(pd.DataFrame(TestSemantics.points))
        assert meaning == Meaning(mapping, equal_universe)
        assert hash(meaning) == hash(Meaning(mapping, equal_universe))
        assert meaning != Meaning((True, True, True, False), TestSemantics.universe)
        assert meaning != Meaning(mapping, TestSemantics.universe, dist=(1, 0, 0, 0))
        assert meaning != Meaning(
            (True,) * len(TestSemantics.ref1.__dict__),
            Universe((TestSemantics.ref1,)),
        )
        assert len({meaning, Meaning(mapping, TestSemantics.universe)}) == 1
