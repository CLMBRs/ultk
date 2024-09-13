import pickle
from yaml import load, Loader

from ..ib import language_to_encoder

if __name__ == "__main__":

    with open("colors/outputs/natural_languages.yaml", "r") as f:
        languages = load(f, Loader=Loader)

    language = languages[0]
    print(language)
    encoder = language_to_encoder(language)
    print(encoder)
    print(encoder.sum(axis=0))
