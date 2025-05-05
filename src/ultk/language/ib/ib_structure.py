from dataclasses import dataclass
from functools import cached_property
from ultk.language.ib.ib_utils import safe_log
from ultk.language.semantics import Meaning, Referent

import numpy as np


@dataclass(frozen=True)
class IBStructure:
    referents: tuple[Referent, ...]
    meanings_prior: np.ndarray
    meanings: tuple[Meaning[float], ...]

    def __init__(
        self,
        referents: tuple[Referent, ...],
        meanings: tuple[Meaning[float], ...],
        prior: tuple[float, ...] = None,
    ):
        # Ensure correct that if there is a dimension each referent has a location
        if prior is not None and len(meanings) != len(prior):
            raise ValueError("Meanings list and priors list are not of same size.")

        for meaning in meanings:
            for referent in referents:
                if referent not in meaning:
                    raise ValueError(
                        "Meanings provided are not compatible with referents"
                    )

        # use of __setattr__ is to work around the issues with @dataclass(frozen=True)
        object.__setattr__(self, "referents", referents)
        # When only meanings are passed in, make the priors a unifrom distribution
        object.__setattr__(
            self,
            "meanings_prior",
            np.array(prior or [1 / len(meanings) for _ in meanings]),
        )
        object.__setattr__(self, "meanings", np.array(meanings))

    # Referent probability distribution given meaning probabilty distribution
    @cached_property
    def mu(self) -> np.ndarray:
        # I do not want to assume everything is in a nice order
        return np.array(
            [
                [meaning.mapping[referent] for referent in self.referents]
                for meaning in self.meanings
            ]
        )

    @cached_property
    def referents_prior(self) -> np.ndarray:
        return self.mu @ self.meanings_prior

    # TODO: CHECK THIS MORE EXTENSIVELY
    @cached_property
    def mutual_information(self) -> float:
        A = safe_log(self.mu / self.referents_prior[:, None])
        return np.sum(A * (self.mu * self.meanings_prior))
