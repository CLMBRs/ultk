import itertools
import pandas as pd
import pytest

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
    

    def test_universe_from_df(self):
        assert TestSemantics.points == [referent.__dict__ for referent in TestSemantics.universe.referents]

    def check_universe_match(self):
        second_dataframe = pd.DataFrame(TestSemantics.points)
        assert Universe.from_dataframe(second_dataframe) == TestSemantics.universe
        
    def test_meaning_match(self):
        ref_list = (Referent(name="weak+epistemic", properties={"force":"weak", "flavor":"epistemic"}))
        meaning = Meaning(ref_list, TestSemantics.universe) #This meaning should exist within the set of semantics
        with pytest.raises(ValueError):
            ref_list.append(Referent(name="neutral+epistemic", properties={"force":"neutral", "flavor":"epistemic"}))
            meaning = Meaning(ref_list, TestSemantics.universe) #This meaning should not
    
