import csv
import math
import os
import numpy as np
from urllib.request import urlretrieve
from rdot.ba import IBResult
import pickle
import matplotlib.pyplot as plt
import dill
import pandas as pd
from ultk.language.semantics import Meaning, Universe, Referent
from color_grammar import ColorLanguage, HashableMeaning
from ultk.language.language import Expression, Language
import ultk.language.sampling as sampling
import ultk.effcomm.rate_distortion as rd
from collections import Counter

import plotnine as pn

from zipfile import ZipFile

"""Generate ULTK language structures based off of the WCS data. 
"""


def load_noga_model(filename=None, model_dir="./model/"):
    """Loads the Zaslavasky model and associated emanings.

    Args:
        filename (_type_, optional): _description_. Defaults to None.
        model_dir (str, optional): _description_. Defaults to './model/'.

    Returns:
        tuple: container for the model data.
    """
    DEFAULT_MODEL_URL = (
        "https://www.dropbox.com/s/70w953orv27kz1o/IB_color_naming_model.zip?dl=1"
    )
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


def check_noga_model(language: Language):
    model = load_noga_model()
    print(model.keys())
    model_info = rd.language_to_ib_point(language, model["pM"], model["pU_M"])
    print(model_info)


def generate_color_languages(num_languages=-1, color_chip_threshold=5):
    """Generates a set of natural languages based on the WCS data, and generates a set of artificial languages based on the real languages.

    Args:
        num_languages (bool, optional): Number of languages to analyze. Defaults to -1, meaning that we analyze the whole dataset.
        color_chip_threshold (int, optional): Number of color chips to be considered a major term. Defaults to 5.

    Returns:
        (list[Language], list[Language], nparray): Tuple of list of natural languages, list of artificial languages, and the meaning space.
    """
    wcs_dialect = csv.Dialect
    wcs_dialect.delimiter = "\t"

    language_name_from_code = dict()
    language_terms = dict()

    # Get current dir for relative paths
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Get the expression from a language based on the color name
    def get_expression_from_language(color_name: str, language: ColorLanguage):
        for expression in language.expressions:
            for referent in expression.meaning.referents:
                if referent.name == color_name:
                    return expression
        return None

    # if USE_NOGA_ARRAYS:
    #     model_data = load_noga_model(filename=f"{current_dir}/model/model.pkl", model_dir=f"{current_dir}/model/")

    # Generate all WCS color codes
    # Convert the Munsell hues of the WCS data to CIELab data
    munsell_to_cielab = {}
    referents_by_color_code = {}
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
            referents_by_color_code[row["V"] + row["H"]] = Referent(
                name=row["V"] + row["H"],
                properties={
                    "V": row["V"],
                    "H": row["H"],
                    "L": float(row["L*"]),
                    "a": float(row["a*"]),
                    "b": float(row["b*"]),
                },
            )

    # print(f"Color codes:{referents_by_color_code.keys()}")

    # Generate referents for all color codes
    color_universe = Universe(referents=referents_by_color_code.values())
    color_referent_tuple = tuple(referents_by_color_code.values())

    with open(f"{current_dir}/data/lang.txt", newline="", encoding="utf-8") as csvfile:
        lang_reader = csv.DictReader(csvfile, delimiter="\t")
        for row in lang_reader:
            language_name_from_code[row["LNUM"]] = row["LNAME"]

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
    # uniform_prior = np.array(color_universe._prior)

    # Munsell to Cielab hues
    munsell_to_cielab = np.array(list(munsell_to_cielab.values()))
    # print(f"Munsell to CIELAB hues:{munsell_to_cielab}")

    SIGMA_SQUARED_SCALAR = 64

    # Calculate the meaning space as an isotropic Gaussian centered at the first chip C, for all other points
    def meaning_distance(center, point):
        return math.exp(
            (-1 / (2 * SIGMA_SQUARED_SCALAR) * np.linalg.norm(center - point))
        )

    # Generate the meaning space
    meaning_dists = np.zeros(
        shape=(len(color_universe.referents), len(color_universe.referents))
    )
    for c1_index, c1 in enumerate(color_universe.referents):
        for c2_index, c2 in enumerate(color_universe.referents):
            meaning_dists[c1_index][c2_index] = meaning_distance(
                np.array((c1.L, c1.a, c1.b)), np.array((c2.L, c2.a, c2.b))
            )

    # meaning_dists = meaning_dists / meaning_dists.sum(axis=1, keepdims=True)

    # print(language_terms)
    expressions_by_speaker = {}
    language_colors_to_expressions = {}
    speakers_by_language = {}
    with open(
        f"{current_dir}/data/foci-exp.txt", newline="", encoding="utf-8"
    ) as csvfile:
        color_reader = csv.DictReader(csvfile, delimiter="\t")
        for row in color_reader:
            lang_num = row["LNUM"]
            speaker_num = row["SNUM"]
            transcription = row["WCSC"]
            color = row["COLOR"]

            # Filter AX(A1, A2, A3....) to A0 and JX to J0 - both of these represent pure white/black respectively
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
                language_colors_to_expressions[lang_num] = {}
            if speaker_num not in expressions_by_speaker[lang_num]:
                expressions_by_speaker[lang_num][speaker_num] = {}
            if transcription not in expressions_by_speaker[lang_num][speaker_num]:
                expressions_by_speaker[lang_num][speaker_num][transcription] = []
            if color not in language_colors_to_expressions[lang_num]:
                language_colors_to_expressions[lang_num][color] = Counter()

            (expressions_by_speaker[lang_num][speaker_num])[transcription].append(color)

            language_colors_to_expressions[lang_num][color][transcription] += 1

    languages = []

    # Create a list of Expressions to pull from later when randomly sampling for artificial languages
    expression_list = []

    # Average out the language, in case of disagreements. For each color, find the most common term associated with that color
    for language_code, language_colors in language_colors_to_expressions.items():
        color_names = {}
        for color in language_colors:
            # Find the most common term associated with each color chip
            most_frequent_color_term = language_colors_to_expressions[language_code][
                color
            ].most_common(1)[0][0]
            if most_frequent_color_term not in color_names:
                color_names[most_frequent_color_term] = []

            color_names[most_frequent_color_term].append(color)

        # Filter out colors that have fewer than COLOR_CHIP_THRESHOLD terms associated with them
        temp_color_terms = {}
        major_color_terms = []
        filtered_color_terms = []
        major_colors = []
        for color_name, color_name_colors in color_names.items():
            if len(color_name_colors) >= color_chip_threshold:
                temp_color_terms[color_name] = color_name_colors
                major_color_terms.append(color_name)
                major_colors += color_name_colors
            else:
                filtered_color_terms.append(color_name)

        color_names = temp_color_terms

        # Create a zeroed np array of the number of colors x the number of major color terms within the language
        expression_meanings = np.array(
            [
                [0.0 for _ in range(len(color_universe.referents))]
                for _ in range(len(major_color_terms))
            ]
        )

        # Print the expression array dimensions
        # print(f"Expression array dimensions: {expression_meanings.shape}")

        # For each major color chip in the language, add the Gaussian meaning row corresponding to that chip to the row for that color term.
        for color in language_colors:
            color_meaning_dist = meaning_dists[
                list(color_universe.referents).index(referents_by_color_code[color])
            ]
            for major_term, major_term_count in language_colors_to_expressions[
                language_code
            ][color].items():
                if major_term in major_color_terms:
                    # Option: Take the maximum
                    expression_meanings[major_color_terms.index(major_term)] = (
                        np.maximum(
                            expression_meanings[major_color_terms.index(major_term)],
                            color_meaning_dist,
                        )
                    )
                    # Option: Add just one value, for the color represented
                    # expression_meanings[major_color_terms.index(major_term)][list(color_universe.referents).index(referents_by_color_code[color])] = major_term_count

        # Normalize the expression matrix so that each the column of each color chip sums up to 1
        # expression_meanings_normalized = expression_meanings / expression_meanings.sum(axis=0, keepdims=True)

        # For each column in the expression matrix, set the most probable term to 1 and the rest to 0
        # for column_index in range(expression_meanings.shape[1]):
        #     max_index = np.argmax(expression_meanings[:, column_index])
        #     expression_meanings[max_index, column_index] = 1
        #     for row_index in range(expression_meanings.shape[0]):
        #         if row_index != max_index:
        #             expression_meanings[row_index, column_index] = 0

        most_probable_colors = {}
        discrete_language_expressions = []
        prob_language_expressions = []

        # Find the colors for which the major term is the most probable
        for color_ref_index, color_ref in enumerate(color_referent_tuple):
            most_probable_major_term = major_color_terms[
                np.argmax(expression_meanings[:, color_ref_index])
            ]
            most_probable_colors.setdefault(most_probable_major_term, [])
            most_probable_colors[most_probable_major_term].append(color_ref)

        for major_term, major_term_colors in most_probable_colors.items():
            major_term_meaning = HashableMeaning(
                referents=tuple(major_term_colors), universe=color_universe
            )
            major_term_expression = Expression(
                form=major_term, meaning=major_term_meaning
            )
            expression_list.append(major_term_expression)
            discrete_language_expressions.append(major_term_expression)

        for major_term_index, major_term in enumerate(major_color_terms):
            # Create a meaning from the expression row
            major_term_meaning = HashableMeaning(
                referents=color_referent_tuple,
                universe=color_universe,
                _dist=dict(
                    zip(
                        [referent.name for referent in color_referent_tuple],
                        expression_meanings[major_term_index],
                    )
                ),
            )

            # Create an expression for that major term
            # discrete_major_expression = Expression(form=major_term, meaning=Meaning(tuple([referents_by_color_code[color] for color in color_names[major_term]]), universe=color_universe))
            # graph_expression(discrete_major_expression, f"discrete_expr_{language_name_from_code[language_code]}_{major_term}")

            major_term_expression = Expression(
                form=major_term, meaning=major_term_meaning
            )
            # graph_expression(major_term_expression, f"expr_{language_name_from_code[language_code]}_{major_term}")

            # expression_list.append(major_term_expression)
            prob_language_expressions.append(major_term_expression)

        # Fill in additional color chips
        # if(GENERATE_ADDITIONAL_COLOR_CHIPS):
        #     for additional_color in color_universe.referents:
        #         if additional_color.name not in major_color_terms and additional_color.name not in filtered_color_terms:
        #             #Get the closest color to the current color
        #             closest_color_with_term = min([color_term_ref for color_term_ref in color_universe.referents if color_term_ref.name in major_colors],
        #                                             key=lambda x: np.linalg.norm(np.array((x.L, x.a, x.b)) - np.array((additional_color.L, additional_color.a, additional_color.b))))
        #             color_names[average_language_by_meaning[language_code][closest_color_with_term.name].most_common(1)[0][0]].append(additional_color.name)

        # Create list of expressions to add to the Language
        # for color_name in color_names:
        #     #language_expressions.append(Expression(form=expression_form, meaning=Meaning(tuple([Referent(name=color) for color in expressions[expression_form]]), universe=color_universe)))
        #     major_term = Expression(form=color_name, meaning=Meaning(tuple([referents_by_color_code[color] for color in color_names[color_name]]), universe=color_universe))
        #     expression_set.add(major_term)
        #     language_expressions.append(major_term)

        languages.append(
            ColorLanguage(
                discrete_language_expressions,
                lang_code=language_code,
                name=language_name_from_code[language_code] + " (D)",
                natural=True,
            )
        )
        languages.append(
            ColorLanguage(
                prob_language_expressions,
                lang_code=language_code,
                name=language_name_from_code[language_code],
                natural=True,
            )
        )

    # result = meaning(munsell_to_cielab[meaning_space_indices[0]], munsell_to_cielab[meaning_space_indices[1]])
    if num_languages > 0:
        languages = languages[:num_languages]
        print(f"Using {num_languages} language(s) for analysis")

    # Temporarily use Zaslavsky data to verify information
    # if USE_NOGA_ARRAYS:
    #     noga_meaning_dists = model_data['pU_M']
    #     noga_prior = np.array([row[0] for row in model_data['pM']])
    #     noga_bound = model_data['IB_curve']

    # Generate the heatmap for the Zaslavsky meaning function
    #     plt.imshow(noga_meaning_dists, cmap="hot")
    #     plt.savefig(f"{current_dir}/outputs/noga_meaning_dists.jpg")

    # Generate some fake languages using the real languages as a baseline via permutation
    artificial_languages = [
        ColorLanguage(language.expressions, is_natural=False)
        for language in sampling.random_languages(
            expressions=expression_list,
            sampling_strategy="stratified",
            sample_size=20,
            max_size=15,
        )
    ]
    # Give enumerated names to each of the artificial languages
    for index, artificial_language in enumerate(artificial_languages):
        artificial_language.name = f"artificial_lang_{index}"

    # dill.detect.baditems(languages)

    # Write out the artificial and natural languages and expressions to file
    with open(f"{current_dir}/outputs/natural-languages.txt", "w") as f:
        for language in languages:
            f.write(f"Name: {language.name} \nLang Code: {language.lang_code}\n")
            for expression in language.expressions:
                f.write(f"\tExpression: {expression.form} \n")
                for referent in expression.meaning.referents:
                    f.write(f"\t\tReferent: {referent.name} \n")
                f.write(f"\tDist : {expression.meaning._dist} \n")

    # pickle.dump(languages, open(f"{current_dir}/outputs/natural-languages.pkl", "wb"))
    # pickle.dump(artificial_languages, open(f"{current_dir}/outputs/artificial-languages.pkl", "wb"))
    # pickle.dump(expression_list, open(f"{current_dir}/outputs/expressions.pkl", "wb"))

    prior = color_universe.prior_numpy()
    pickle.dump(meaning_dists, open(f"{current_dir}/outputs/meaning_dists.pkl", "wb"))
    pickle.dump(prior, open(f"{current_dir}/outputs/prior.pkl", "wb"))

    return languages, artificial_languages, meaning_dists, prior
