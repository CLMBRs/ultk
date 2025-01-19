import csv
import math
import pandas as pd
from ultk.language.semantics import Meaning, Universe, Referent
from ultk.language.language import Language, Expression
import ultk.effcomm.informativity as informativity
import ultk.effcomm.rate_distortion as rd
from collections import Counter
import os
import numpy as np
from urllib.request import urlretrieve
from rdot.ba import IBResult

import pickle
import matplotlib.pyplot as plt

import pandas as pdcombined
import plotnine as pn

from zipfile import ZipFile

wcs_dialect = csv.Dialect
wcs_dialect.delimiter = "\t"

language_codes = dict()
language_terms = dict()


# Generate all WCS color codes
current_dir = os.path.dirname(os.path.realpath(__file__))

#####----TESTING - Grab the Noga model -- remove later
DEFAULT_MODEL_URL = (
    "https://www.dropbox.com/s/70w953orv27kz1o/IB_color_naming_model.zip?dl=1"
)


def load_model(filename=None, model_dir="./model/"):
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    if filename is None:
        filename = model_dir + "IB_color_naming_model/model.pkl"
    if not os.path.isfile(filename):
        print("downloading default model from %s  ..." % DEFAULT_MODEL_URL)
        urlretrieve(DEFAULT_MODEL_URL, model_dir + "temp.zip")
        print("extracting model files ...")
        with ZipFile(model_dir + "temp.zip", "r") as zf:
            zf.extractall(model_dir)
            # os.remove(model_dir + 'temp.zip')
            os.rename(model_dir + "IB_color_naming_model/IB_color_naming.pkl", filename)
    with open(filename, "rb") as f:
        print("loading model from file: %s" % filename)
        model_data = pickle.load(f)
        return model_data


model_data = load_model(
    filename=f"{current_dir}/model/model.pkl", model_dir=f"{current_dir}/model/"
)


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

# Collect language information, such as term names, abbreviations, etc
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


# Generate the uniform probability prior as a baseline
uniform_prior = color_universe.prior_numpy()
# uniform_prior = np.array(color_universe._prior)

SIGMA_SQUARED_SCALAR = 64


# Calculate the meaning space as an isotropic Gaussian centered at the first chip C, for all other points
def meaning(center, point):
    return math.exp((-1 / (2 * SIGMA_SQUARED_SCALAR) * np.linalg.norm(center - point)))


munsell_to_cielab = np.array(list(munsell_to_cielab.values()))
print(f"Munsell to CIELAB hues:{munsell_to_cielab}")

# Generate the meaning space
meaning_space_indices = np.zeros(shape=(len(munsell_to_cielab), len(munsell_to_cielab)))
print(meaning_space_indices)
for center_index, center in enumerate(munsell_to_cielab):
    for point_index, point in enumerate(munsell_to_cielab):
        meaning_space_indices[center_index][point_index] = meaning(center, point)


meaning_dists = meaning_space_indices / meaning_space_indices.sum(axis=1, keepdims=True)

# print(language_terms)
expressions_by_speaker = {}
average_language_by_meaning = {}
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
        if lang_num not in expressions_by_speaker:
            expressions_by_speaker[lang_num] = {}
            average_language_by_meaning[lang_num] = {}
        if speaker_num not in expressions_by_speaker[lang_num]:
            expressions_by_speaker[lang_num][speaker_num] = {}
        if transcription not in expressions_by_speaker[lang_num][speaker_num]:
            expressions_by_speaker[lang_num][speaker_num][transcription] = []
        if color not in average_language_by_meaning[lang_num]:
            average_language_by_meaning[lang_num][color] = Counter()

        (expressions_by_speaker[lang_num][speaker_num])[transcription].append(color)

        average_language_by_meaning[lang_num][color][transcription] += 1


# print(expressions_by_language)

# For now, assume that if any speaker refers to a color by a given term, that color can be referred to by that term
# for language in expressions_by_language:
#    for expression in expressions_by_language[language]:
#        expressions_by_language[language][expression] = set(expressions_by_language[language][expression])

languages = {}

# Average out the language, in case of disagreements. For each color, find the most common term associated with that color
for language_code in average_language_by_meaning:
    expressions = {}
    for color in average_language_by_meaning[language_code]:
        # Find the most common term associated with each color
        most_frequent_expression = average_language_by_meaning[language_code][
            color
        ].most_common(1)[0][0]
        # print(f"Most frequent expr:{most_frequent_expression}")
        if most_frequent_expression not in expressions:
            expressions[most_frequent_expression] = []
        expressions[most_frequent_expression].append(color)
    language_expressions = []
    for expression_form in expressions:
        # language_expressions.append(Expression(form=expression_form, meaning=Meaning(tuple([Referent(name=color) for color in expressions[expression_form]]), universe=color_universe)))
        language_expressions.append(
            Expression(
                form=expression_form,
                meaning=Meaning(
                    tuple(
                        [Referent(name=color) for color in expressions[expression_form]]
                    ),
                    universe=color_universe,
                ),
            )
        )

    languages[language_code] = Language(language_expressions, lang_code=language_code)

# Generate the imshow heatmap for the meaning
plt.imshow(meaning_dists, cmap="hot")
plt.savefig(f"{current_dir}/outputs/old_meaning_dists.jpg")

# Temporarily use Zaslavsky data to verify information
meaning_dists = model_data["pU_M"]
noga_prior = np.array([row[0] for row in model_data["pM"]])
noga_bound = model_data["IB_curve"]

# Generate the heatmap for the Zaslavasky meaning function
plt.imshow(meaning_dists, cmap="hot")
plt.savefig(f"{current_dir}/outputs/noga_meaning_dists.jpg")

# result = meaning(munsell_to_cielab[meaning_space_indices[0]], munsell_to_cielab[meaning_space_indices[1]])
# Generate the meaning/accuracy/complexity for all languages based on the prior, meaning and Language
language_data = []
for language_code in languages:
    language = languages[language_code]
    # Dereference the lang code to get the actual language associated with it
    language_name = language_codes[language_code]
    language_data.append(
        (language_name, "natural")
        + rd.language_to_ib_point(
            language=language, prior=noga_prior, meaning_dists=(meaning_dists)
        )
    )
combined_data = pd.DataFrame(
    language_data, columns=["name", "type", "complexity", "informativity", "comm_cost"]
)

"""
#Generate languages per speaker
for language in expressions_by_speaker:
    for speaker in expressions_by_speaker[language]:
        expressions = []
        for expression in expressions_by_speaker[language][speaker]:
            #print(f"Language:{language} | Expression:{expression} | Colors:{expressions_by_language[language][speaker][expression]}")
            expressions.append(Expression(form=expression, meaning=Meaning(tuple([Referent(name=color) for color in expressions_by_speaker[language][speaker][expression]]), universe=color_universe)))
        languages[(language, speaker)] = (Language(expressions, lang_code=language, speaker=speaker))

        
#result = meaning(munsell_to_cielab[meaning_space_indices[0]], munsell_to_cielab[meaning_space_indices[1]])
language_data = []
for language_info in languages:
    language = languages[language_info]
    language_code = language_info[0]
    speaker_id = language_info[1]
    #Dereference the lang code to get the actual language associated with it
    language_name = language_codes[language_code] 
    language_data.append((language_name, "natural", speaker_id) + rd.language_to_ib_point(language=language, prior=uniform_prior, meaning_dists=meaning_dists))
combined_data = pd.DataFrame(language_data, columns =['name','type','speaker_id','complexity', 'informativity', 'comm_cost'])

"""

# Get the IB bound for the specified parameters
ib_boundary = rd.get_ib_bound(
    prior=uniform_prior, meaning_dists=meaning_dists, betas=np.logspace(-2, 2, 10)
)
ib_boundary_points = pd.DataFrame(
    [
        ("ib_bound", "ib_bound", ib_point.rate, ib_point.accuracy, ib_point.distortion)
        for ib_point in ib_boundary
        if ib_point is not None
    ],
    columns=["name", "type", "complexity", "informativity", "comm_cost"],
)

combined_data = pd.concat([ib_boundary_points, combined_data])

# Generate and save plots
plot = (
    pn.ggplot(pn.aes(x="complexity", y="comm_cost"))
    + pn.geom_point(combined_data, pn.aes(color="type"))
    + pn.geom_text(
        combined_data[combined_data["type"] == "natural"],
        pn.aes(label="name"),
        ha="left",
        size=5,
        nudge_x=0.1,
    )
)


plot.save(f"{current_dir}/outputs/complexity-commcost.png", width=8, height=6, dpi=300)

plot = (
    pn.ggplot(pn.aes(x="complexity", y="informativity"))
    + pn.geom_point(combined_data, pn.aes(color="type"))
    + pn.geom_text(
        combined_data[combined_data["type"] == "natural"],
        pn.aes(label="name"),
        ha="left",
        size=5,
        nudge_x=0.1,
    )
)

plot.save(
    f"{current_dir}/outputs/complexity-informativity.png", width=8, height=6, dpi=300
)

plot = pn.ggplot(pn.aes(x="informativity", y="comm_cost")) + pn.geom_point(
    combined_data, pn.aes(color="type")
)

plot.save(
    f"{current_dir}/outputs/informativity-commcost.png", width=8, height=6, dpi=300
)
