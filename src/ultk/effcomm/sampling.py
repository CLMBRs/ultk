"""Functions for sampling expressions into languages."""

import copy
from typing import Any
import numpy as np
from ultk.language.language import Language
from ultk.effcomm.agent import Speaker, LiteralSpeaker
from math import comb
from tqdm import tqdm


##############################################################################
# Methods for generating languages from expressions, or permuting the stochastic mapping from words to meanings of an existing language
##############################################################################


def get_hypothetical_variants(
    languages: list[Language] = None, speakers: list[Speaker] = None, total: int = 0
) -> list[Any]:
    """For each system (parameterized by a language or else a speaker), generate `num` hypothetical variants by permuting the signals that the system assigns to states.

    Args:
        languages: a list of languages to permute, by constructing LiteralSpeakers and permuting their weights.

        speakers: a list of speakers of a language, whose weights can be directly permuted. Should be used instead of `languages` if possible, because it can be more finegrained (every language can be associated with multiple speakers).

        total: the total number of hypothetical variants to obtain. Should be greater than the number of languages.

    Returns:
        hypothetical_variants: a list of type either Language or np.ndarray depending on whether `languages` or `speakers` was passed, representing hypothetical variants of the systems passed. If `speakers` was passed, a list of speakers is returned.
    """

    if (languages is not None and speakers is not None) or (
        languages is None and speakers is None
    ):
        raise Exception(
            "You must pass exactly one of the following: `languages`, `speakers`."
        )

    if languages is not None:
        num_systems = len(languages)
    else:
        num_systems = len(speakers)

    variants_per_system = int(total / num_systems)

    if variants_per_system == 0:
        raise Exception(
            "Number of languages exceeds the number of languages to be generated. "
        )

    hypothetical_variants = []
    for i in range(num_systems):
        if languages is not None:
            speaker = LiteralSpeaker(languages[i])
        else:
            speaker = speakers[i]

        seen = set()
        while len(seen) < variants_per_system:
            # permute columns of speaker weights
            permuted = np.random.permutation(speaker.weights.T).T
            seen.add(tuple(permuted.flatten()))

        for permuted_weights in seen:
            permuted_speaker = copy.deepcopy(speaker)
            permuted_speaker.weights = np.reshape(
                permuted_weights, (speaker.weights.shape)
            )

            variant = permuted_speaker
            if languages is not None:
                variant = permuted_speaker.to_language()
            hypothetical_variants.append(variant)

    return hypothetical_variants
