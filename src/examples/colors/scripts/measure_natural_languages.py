import pandas as pd
import numpy as np
from yaml import load, Loader

from ultk.effcomm.rate_distortion import ib_encoder_to_point

from ..ib import language_to_encoder
from ..meaning import color_universe, meaning_distributions

if __name__ == "__main__":

    with open("colors/outputs/natural_languages.yaml", "r") as f:
        languages = load(f, Loader=Loader)

    """

    with open("colors/outputs/language0.yaml", "r") as f:
        language = load(f, Loader=Loader)

    print(language)
    encoder = language_to_encoder(language)
    print(encoder)
    print(encoder.sum(axis=0))

    print(
        ib_encoder_to_point(
            np.array(color_universe.prior), meaning_distributions, encoder
        )
    )
    """

    prior = np.array(color_universe.prior)
    information_plane = pd.DataFrame.from_records(
        [
            (lang.name,)
            + ib_encoder_to_point(
                prior, meaning_distributions, language_to_encoder(lang)
            )
            for lang in languages
        ],
        columns=("language", "complexity", "accuracy", "distortion"),
    )
    information_plane.to_csv(
        "colors/outputs/natural_language_information_plane.csv", index=False
    )
