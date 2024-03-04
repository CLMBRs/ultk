"""Classes for modeling the meanings of a language.

    Meanings are modeled as things which map linguistic forms to objects of reference. The linguistic forms and objects of reference can in principle be very detailed, and future work may elaborate the meaning classes and implement a Form class.

    In efficient communication analyses, simplicity and informativeness can be measured as properties of semantic aspects of a language. E.g., a meaning is simple if it is easy to represent, or to compress into some code; a meaning is informative if it is easy for a listener to recover a speaker's intended literal meaning.

    Examples:

        >>> from ultk.language.semantics import Referent, Meaning, Universe
        >>> from ultk.language.language import Expression
        >>> # construct the meaning space for numerals
        >>> numerals_universe = NumeralUniverse(referents=[NumeralReferent(str(i)) for i in range(1, 100)])
        >>> # construct a list of referents for the expression 'a few'
        >>> a_few_refs = [NumeralReferent(name=str(i)) for i in range(2, 6)]
        >>> a_few_meaning = NumeralMeaning(referents=a_few_refs, universe=numerals_universe)
        >>> # define the expression
        >>> a_few = NumeralExpression(form="a few", meaning=a_few_meaning)
"""

from typing import Any, Iterable, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from functools import cached_property


class Referent:
    """A referent is some object in the universe for a language.

    Conceptually, a Referent can be any kind of object.  This functions like a generic python object that is _immutable_ after initialization.
    At initialization, properties can be specified either by passing a dictionary or by keyword arguments.
    """

    def __init__(self, name: str, properties: dict[str, Any] = {}, **kwargs) -> None:
        """Initialize a referent.

        Args:
            name: a string representing the name of the referent
        """
        self.name = name
        self.__dict__.update(properties, **kwargs)
        self._frozen = True

    def __setattr__(self, __name: str, __value: Any) -> None:
        if hasattr(self, "_frozen") and self._frozen:
            raise AttributeError("Referents are immutable.")
        else:
            object.__setattr__(self, __name, __value)

    def to_dict(self) -> dict:
        return self.__dict__

    def __str__(self) -> str:
        return str(self.__dict__)

    def __lt__(self, other):
        return self.name < other.name

    def __eq__(self, other) -> bool:
        return self.name == other.name and self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.__dict__)))


@dataclass(frozen=True)
class Universe:

    """The universe is the set of possible referent objects for a meaning."""

    referents: tuple[Referent]
    prior: tuple[float] = None

    @cached_property
    def _referents_by_name(self):
        return {referent.name: referent for referent in self.referents}

    @cached_property
    def size(self):
        return len(self.referents)

    @cached_property
    def _prior(self):
        return (
            self.prior if self.prior is not None else tuple([1 / self.size] * self.size)
        )

    def prior_numpy(self) -> np.ndarray:
        return np.array(self._prior)

    def __getitem__(self, key: Union[str, int]) -> Referent:
        if type(key) is str:
            return self._referents_by_name[key]
        elif type(key) is int:
            return self.referents[key]
        else:
            raise KeyError("Key must either be an int or str.")

    def __str__(self):
        referents_str = ",\n\t".join([str(point) for point in self.referents])
        return f"Points:\n\t{referents_str}\nDistribution:\n\t{self._prior}"

    def __len__(self) -> int:
        return len(self.referents)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        """Build a Universe from a DataFrame.
        It's assumed that each row specifies one Referent, and each column will be a property
        of that Referent.  We assume that `name` is one of the columns of the DataFrame.

        Args:
            a DataFrame representing the meaning space of interest, assumed to have a column `name`
        """
        prior = None
        if "probability" in df.columns:
            prior = tuple(df["probability"])
        records = df.to_dict("records")
        referents = tuple(Referent(record["name"], record) for record in records)
        return cls(referents, prior)

    @classmethod
    def from_csv(cls, filename: str):
        """Build a Universe from a CSV file.  This is a small wrapper around
        `Universe.from_dataframe`, so see that documentation for more information.
        """
        df = pd.read_csv(filename)
        return cls.from_dataframe(df)


@dataclass(frozen=True)
class Meaning:
    referents: tuple[Referent]
    universe: Universe
    _dist: dict[str, float] = None
    """A meaning picks out a set of objects from the universe.

    Following one tradition (from formal semantics), we might model an underspecified meaning as a subset of the universe.
    Sometimes these different referents are not equally likely,
    in which it can be helpful to define a meaning explicitly as a distribution over the universe.

    Args:
        referents: a list of Referent objects, which must be a subset of the referents in `universe`.

        universe: a Universe object that defines the probability space for a meaning.

        dist: a dict of with Referent names as keys and weights or probabilities as values, representing the distribution over referents to associate with the meaning. By default is None, and the distribution will be uniform over the passed referents, and any remaining referents are assigned 0 probability.
    """

    def __post_init__(self):
        if not isinstance(self.referents, tuple):
            raise TypeError(f"The `referents` field of Meaning must be a tuple.")

        if not set(self.referents).issubset(set(self.universe.referents)):
            print("referents:")
            print(tuple(str(r) for r in self.referents))
            print("universe:")
            print(tuple(str(r) for r in self.universe.referents))
            raise ValueError(
                f"The set of referents for a meaning must be a subset of the universe of discourse."
            )

    @property
    def dist(self) -> np.ndarray:
        zeros = {
            ref.name: 0.0 for ref in set(self.universe.referents) - set(self.referents)
        }
        if self._dist is not None:
            # normalize weights to distribution
            total_weight = sum(self._dist.values())
            _dist = {
                ref.name: self._dist[ref.name] / total_weight for ref in self.referents
            } | zeros
            return np.array(_dist.values())
        else:
            _dist = {
                ref.name: 1 / len(self.referents) for ref in self.referents
            } | zeros
            return np.array(_dist.values())

    def to_dict(self) -> dict:
        return {"referents": [referent.to_dict() for referent in self.referents]}

    def __bool__(self):
        return bool(self.referents) and bool(self.universe)

    def __str__(self):
        return f"Referents:\n\t{','.join(str(referent) for referent in self.referents)}\
            \nDistribution:\n\t{self.dist}\n"
