import pickle
from yaml import load, Loader, dump, Dumper

from ..ib import language_to_encoder

if __name__ == "__main__":

    # with open("colors/outputs/natural_languages.yaml", "r") as f:
    # languages = load(f, Loader=Loader)

    with open("colors/outputs/language0.yaml", "r") as f:
        language = load(f, Loader=Loader)

    print(language)
    encoder = language_to_encoder(language)
    print(encoder)
    print(encoder.sum(axis=0))
