import csv
import math
import pandas as pd
from ultk.language.semantics import Meaning, Universe, Referent
from ultk.language.language import Language, Expression
import ultk.effcomm.informativity as informativity
import ultk.effcomm.rate_distortion as rd
import os
import numpy as np

wcs_dialect = csv.Dialect
wcs_dialect.delimiter = "\t"

language_codes = dict()
language_terms = dict()

# Generate all WCS color codes
"""
hues = range(0,41)
lightnesses = ["A","B","C","D","E","F","G","H","I","J"]
color_codes = []
for lightness in lightnesses:
    color_codes.extend([ {"name":(lightness + str(hue))} for hue in hues] )

"""
current_dir = os.path.dirname(os.path.realpath(__file__))
munsell_to_cielab = {}
with open(
    f"{current_dir}/data/cnum-vhcm-lab-new.txt", newline="", encoding="utf-8"
) as csvfile:
    color_reader = csv.DictReader(csvfile, delimiter="\t")
    for row in color_reader:
        munsell_to_cielab[row["V"] + row["H"]] = [
            float(row["L*"]),
            float(row["a*"]),
            float(row["b*"]),
        ]
color_codes = [{"name": key} for key in munsell_to_cielab.keys()]

# Generate referents for all color codes
referents = pd.DataFrame(color_codes)
color_universe = Universe.from_dataframe(referents)

with open(f"{current_dir}/data/lang.txt", newline="", encoding="utf-8") as csvfile:
    lang_reader = csv.DictReader(csvfile, delimiter="\t")
    for row in lang_reader:
        language_codes[row["LNUM"]] = row["LNAME"]

# print(language_codes)

with open(f"{current_dir}/data/dict.txt", newline="", encoding="utf-8") as csvfile:
    term_reader = csv.DictReader(csvfile, delimiter="\t")
    for row in term_reader:
        lang_num = row["LNUM"]
        lang_term_num = row["TNUM"]
        lang_term_transcription = row["TRAN"]
        lang_term_abbrev = row["WCSC"]
        if lang_num not in language_terms:
            language_terms[lang_num] = []
        language_terms[lang_num].append(lang_term_abbrev)

# print(language_terms)
expressions_by_language = {}
speakers_by_language = {}
with open(f"{current_dir}/data/foci-exp.txt", newline="", encoding="utf-8") as csvfile:
    color_reader = csv.DictReader(csvfile, delimiter="\t")
    for row in color_reader:
        lang_num = row["LNUM"]
        speaker_num = row["SNUM"]
        transcription = row["WCSC"]
        color = row["COLOR"]

        # Filter AX to A0 and JX to J0 - both of these represent pure white/black respectively
        if color.startswith("A"):
            color = "A0"
        elif color.startswith("J"):
            color = "J0"

        # Update speaker records
        if lang_num not in speakers_by_language:
            speakers_by_language[lang_num] = set()
        speakers_by_language[lang_num].add(speaker_num)

        # Assemble list of languages by speaker
        if lang_num not in expressions_by_language:
            expressions_by_language[lang_num] = {}
        if speaker_num not in expressions_by_language[lang_num]:
            expressions_by_language[lang_num][speaker_num] = {}
        expressions_by_language[lang_num][speaker_num]

        if transcription not in expressions_by_language[lang_num][speaker_num]:
            expressions_by_language[lang_num][speaker_num][transcription] = []
        (expressions_by_language[lang_num][speaker_num])[transcription].append(color)
# print(expressions_by_language)

# For now, assume that if any speaker refers to a color by a given term, that color can be referred to by that term
# for language in expressions_by_language:
#    for expression in expressions_by_language[language]:
#        expressions_by_language[language][expression] = set(expressions_by_language[language][expression])

languages = []

for language in expressions_by_language:
    for speaker in expressions_by_language[language]:
        expressions = []
        for expression in expressions_by_language[language][speaker]:
            # print(f"Language:{language} | Expression:{expression} | Colors:{expressions_by_language[language][speaker][expression]}")
            expressions.append(
                Expression(
                    form=expression,
                    meaning=Meaning(
                        tuple(
                            [
                                Referent(name=color)
                                for color in expressions_by_language[language][speaker][
                                    expression
                                ]
                            ]
                        ),
                        universe=color_universe,
                    ),
                )
            )
        languages.append(Language(expressions))

# print(languages)

# Generate the uniform probability prior as a baseline
# uniform_prior = color_universe.prior_numpy()
uniform_prior = np.array(color_universe._prior)

"""
example_informativity = informativity.informativity(languages[0], uniform_prior)
example_informativity_2 = informativity.informativity(languages[100], uniform_prior)

print(f"Informativity of language 1: {example_informativity}")
print(f"Informativity of language 2: {example_informativity_2}")
"""


sigma_squared_scalar = 64


# Calculate the meaning space as an isotropic Gaussian centered at the first chip C, for all other points
def meaning(center, point):
    return math.exp((-1 / (2 * sigma_squared_scalar) * np.linalg.norm(center - point)))


munsell_to_cielab = np.array(list(munsell_to_cielab.values()))
print("Munsell to CIELAB hues:")
print(munsell_to_cielab)

# Generate the meaning space
meaning_space_indices = np.zeros(shape=(len(munsell_to_cielab), len(munsell_to_cielab)))
print(meaning_space_indices)
for center_index, center in enumerate(munsell_to_cielab):
    for point_index, point in enumerate(munsell_to_cielab):
        meaning_space_indices[center_index][point_index] = meaning(center, point)

print("Language:")
print(languages[0])

print("Prior:")
print(uniform_prior)

print("Meaning space indices:")
print(meaning_space_indices)

# result = meaning(munsell_to_cielab[meaning_space_indices[0]], munsell_to_cielab[meaning_space_indices[1]])

# breakpoint()
meaning_dists = meaning_space_indices / meaning_space_indices.sum(axis=1, keepdims=True)

print(
    rd.language_to_ib_point(
        language=languages[0], prior=uniform_prior, meaning_dists=meaning_dists
    )
)
print(
    rd.language_to_ib_point(
        language=languages[100], prior=uniform_prior, meaning_dists=meaning_dists
    )
)
print(
    rd.language_to_ib_point(
        language=languages[200], prior=uniform_prior, meaning_dists=meaning_dists
    )
)

np.save("prior.npy", uniform_prior)
np.save("meaning_dists.npy", meaning_dists)

# bound = rd.get_ib_bound(prior=uniform_prior, meaning_dists=meaning_dists)
# plot_data = pd.DataFrame(
#     [(x.rate, x.accuracy) for x in bound if x is not None], 
#     columns=[
#         "rate", 
#         "accuracy",
#     ]
# )
# import plotnine as pn
# plot = pn.ggplot(
#     plot_data,
#     pn.aes(x="rate", y="accuracy"),
# ) + pn.geom_point() + pn.geom_line()
# plot.save("plot.png")