import itertools
import pandas as pd

from altk.language.semantics import Universe


forces = ("weak", "strong")
flavors = ("epistemic", "deontic")


def test_universe_from_df():
    points = [
        {"name": f"{force}+{flavor}", "force": force, "flavor": flavor}
        for (force, flavor) in itertools.product(forces, flavors)
    ]
    dataframe = pd.DataFrame(points)
    universe = Universe.from_dataframe(dataframe)
    assert points == [referent.__dict__ for referent in universe.referents]
