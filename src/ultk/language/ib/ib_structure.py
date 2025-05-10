from dataclasses import dataclass
from functools import cached_property
from ultk.language.ib.ib_utils import mutual_information
from ultk.language.semantics import Meaning, Referent

import numpy as np


@dataclass(frozen=True)
class IBStructure:
    referents: tuple[Referent, ...]
    meanings_prior: np.ndarray
    mu: np.ndarray

    def __init__(
        self,
        mu: np.ndarray,
        prior: tuple[float, ...] = None,
    ):
        if len(mu.shape) != 2:
            raise ValueError("Must be a 2d matrix")

        if prior is not None and mu.shape[1] != len(prior):
            raise ValueError(
                f"Input matrix is for {mu.shape[1]} meanings, but {len(prior)} priors are given"
            )

        # When only meanings are passed in, make the priors a unifrom distribution
        object.__setattr__(
            self,
            "meanings_prior",
            (
                prior
                if prior is not None
                else np.array([1 / len(mu.shape[1]) for _ in range(mu.shape[1])])
            ),
        )
        object.__setattr__(self, "mu", mu)

    @cached_property
    def referents_prior(self) -> np.ndarray:
        return self.mu @ self.meanings_prior

    # TODO: CHECK THIS MORE EXTENSIVELY
    @cached_property
    def mutual_information(self) -> float:
        return mutual_information(self.mu, self.referents_prior, self.meanings_prior)


def structure_from_meanings(
    meanings: tuple[Meaning[float], ...],
    meanings_prior: tuple[float, ...],
    referents: tuple[Referent, ...],
) -> IBStructure:
    return IBStructure(
        # I do not want to assume everything is in a nice order
        np.array(
            [
                [meaning.mapping[referent] for meaning in meanings]
                for referent in referents
            ]
        ),
        np.array(meanings_prior),
    )
