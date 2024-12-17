import itertools
import pandas as pd
import pytest

from ultk.language.language import Expression, Language
from ultk.language.semantics import Referent, Universe, Meaning
from ultk.util.frozendict import FrozenDict


class TestLanguage:
    pairs = {
        "shroom": "fungus",
        "dog": "animal",
        "tree": "plant",
        "cat": "animal",
        "bird": "animal",
    }
    pairs2 = {"shroom": "fungus", "dog": "animal", "tree": "plant", "bird": "bird"}

    uni_refs = tuple(Referent(key, {"phylum": val}) for (key, val) in pairs.items())
    uni = Universe(uni_refs)

    uni2 = Universe(
        tuple(Referent(key, {"phylum": val}) for (key, val) in pairs2.items())
    )

    dog = Expression(
        form="dog",
        meaning=Meaning(
            mapping=FrozenDict({ref: ref.name == "dog" for ref in uni_refs}),
            universe=uni,
        ),
    )
    cat = Expression(
        form="cat",
        meaning=Meaning(
            mapping=FrozenDict({ref: ref.name == "cat" for ref in uni_refs}),
            universe=uni,
        ),
    )
    tree = Expression(
        form="tree",
        meaning=Meaning(
            mapping=FrozenDict({ref: ref.name == "tree" for ref in uni_refs}),
            universe=uni,
        ),
    )
    shroom = Expression(
        form="shroom",
        meaning=Meaning(
            mapping=FrozenDict({ref: ref.name == "shroom" for ref in uni_refs}),
            universe=uni,
        ),
    )
    bird = Expression(
        form="bird",
        meaning=Meaning(
            mapping=FrozenDict({ref: ref.name == "bird" for ref in uni_refs}),
            universe=uni,
        ),
    )

    lang = Language(expressions=tuple([dog, cat, tree, shroom]))
    lang_one_different_expr = Language(expressions=tuple([dog, cat, tree, bird]))
    lang_subset_expr = Language(expressions=tuple([dog, cat, tree]))
    lang_of_different_order = Language(expressions=tuple([dog, cat, shroom, tree]))

    def test_exp_subset(self):
        assert TestLanguage.dog.can_express(Referent("dog", {"phylum": "animal"}))

    def test_exp_subset(self):
        assert not TestLanguage.dog.can_express(Referent("cat", {"phylum": "animal"}))

    def test_language_has_expressions(self):
        with pytest.raises(ValueError):
            lang = Language(list())

    def test_language_universe_check(self):
        with pytest.raises(ValueError):
            lang_one_different_expr = Language(
                [
                    TestLanguage.dog,
                    Expression(
                        form="dog",
                        meaning=Meaning(
                            mapping=FrozenDict(
                                {
                                    ref: ref.name == "dog"
                                    for ref in TestLanguage.uni.referents
                                }
                            ),
                            universe=TestLanguage.uni2,
                        ),
                    ),
                ]
            )

    def test_language_degree(self):
        def isAnimal(exp: Expression) -> bool:
            print("checking phylum of " + str(exp))
            for k, v in exp.meaning.mapping.items():
                if v and k.phylum != "animal":
                    return False
            return True

        assert TestLanguage.lang.degree_property(isAnimal) == 0.5

    def test_language_len(self):
        assert len(TestLanguage.lang) == 4

    def test_language_equality(self):
        assert TestLanguage.lang != TestLanguage.lang_one_different_expr
        assert TestLanguage.lang != TestLanguage.lang_subset_expr

        # Test that order doesn't matter
        assert TestLanguage.lang == TestLanguage.lang_of_different_order
