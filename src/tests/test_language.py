import itertools
import pandas as pd
import pytest

from ultk.language.language import Expression, Language
from ultk.language.semantics import Referent, Universe, Meaning

class TestLanguage:
    pairs = {"the":"det", "dog":"noun", "jumps":"verb", "cat":"noun"}
    pairs2 = {"the":"det", "dog":"noun", "jumps":"verb", "bird":"noun"}

    uni_refs = [Referent(key, {"pos": val}) for (key,val) in pairs.items()]
    uni = Universe(uni_refs)
    uni2 = Universe([Referent(key, {"pos": val}) for (key,val) in pairs2.items()])

    meaning = Meaning(referents=uni_refs, universe=uni)
    exp = Expression(form="dog", meaning=Meaning(referents=[Referent("dog", {"pos":"noun"})], universe=uni))
    exp2 = Expression(form="cat", meaning=Meaning(referents=[Referent("cat", {"pos":"noun"})], universe=uni))
    exp3 = Expression(form="jumps", meaning=Meaning(referents=[Referent("jumps", {"pos":"verb"})], universe=uni))

    lang = Language(expressions=[exp, exp2])

    def test_exp_subset(self):
        assert TestLanguage.exp.can_express(Referent("dog", {"pos":"noun"}))
    def test_exp_subset(self):
        assert not TestLanguage.exp.can_express(Referent("cat", {"pos":"noun"}))
    def test_language_has_expressions(self):
        with pytest.raises(ValueError):
            lang2 = Language([])
    def test_language_universe_check(self):
        with pytest.raises(ValueError):
            lang2 = Language([TestLanguage.exp, 
                              Expression(form="dog", meaning=Meaning(referents=[Referent("dog", {"pos":"noun"})], universe=TestLanguage.uni2))])
 