import csv
import math
import pandas as pd
from ultk.language.semantics import Meaning, Universe, Referent
from ultk.language.language import Language, Expression
import ultk.effcomm.informativity as informativity
import ultk.effcomm.rate_distortion as rd
#import ultk.effcomm.sampling as sampling
import ultk.language.sampling as sampling
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

language_name_from_code = dict()
language_terms = dict()

GENERATE_IB_BOUND=True #True to generate the IB bound for the specified parameters
USE_RKK = False #Whether to use the RKK metric for complexity
USE_NOGA_ARRAYS = False #True to use the Zaslavsky data for the IB bound, false otherwise
GENERATE_LANG_COLOR_INFO=False #True to generate color information for each language
GENERATE_ADDITIONAL_COLOR_CHIPS = True #If true, will expand color terms based off color distance for both natural and artificial languages
USE_ONE_LANGUAGE = False #If true, just uses the first language in the list of languages

#Get current dir for relative paths
current_dir = os.path.dirname(os.path.realpath(__file__))

#Get the expression from a language based on the color name
def get_expression_from_language(color_name:str, language:Language):
    for expression in language.expressions:
        for referent in expression.meaning.referents:
            if referent.name == color_name:
                return expression
    return None

if(USE_NOGA_ARRAYS):

    #####----TESTING - Grab the Noga model -- remove later
    DEFAULT_MODEL_URL = 'https://www.dropbox.com/s/70w953orv27kz1o/IB_color_naming_model.zip?dl=1'
    def load_model(filename=None, model_dir='./model/'):
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        if filename is None:
            filename = model_dir + 'IB_color_naming_model/model.pkl'
        if not os.path.isfile(filename):
            print('downloading default model from %s  ...' % DEFAULT_MODEL_URL)
            urlretrieve(DEFAULT_MODEL_URL, model_dir + 'temp.zip')
            print('extracting model files ...')
            with ZipFile(model_dir + 'temp.zip', 'r') as zf:
                zf.extractall(model_dir)
                #os.remove(model_dir + 'temp.zip')
                os.rename(model_dir + 'IB_color_naming_model/IB_color_naming.pkl', filename)
        with open(filename, 'rb') as f:
            print('loading model from file: %s' % filename)
            model_data = pickle.load(f)
            return model_data

    model_data = load_model(filename=f"{current_dir}/model/model.pkl", model_dir=f"{current_dir}/model/")

#Generate all WCS color codes
#Convert the Munsell hues of the WCS data to CIELab data 
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
        referents_by_color_code[row["V"] + row["H"]] = Referent(name=row["V"] + row["H"], properties={"L": float(row["L*"]), "a": float(row["a*"]), "b": float(row["b*"])})

#Generate a list of Meanings
#color_codes = [{"name": key, "L":munsell_to_cielab[key][0], "a":munsell_to_cielab[key][1], "b":munsell_to_cielab[key][2]} for key in munsell_to_cielab.keys()]

print(f"Color codes:{referents_by_color_code}")

# Generate referents for all color codes
color_universe = Universe(referents=referents_by_color_code.values())

with open(f"{current_dir}/data/lang.txt", newline="", encoding="utf-8") as csvfile:
    lang_reader = csv.DictReader(csvfile, delimiter="\t")
    for row in lang_reader:
        language_name_from_code[row["LNUM"]] = row["LNAME"]

#Collect language information, such as term names, abbreviations, etc
with open(f'{current_dir}/data/dict.txt', newline='', encoding="utf-8" ) as csvfile:
    term_reader = csv.DictReader(csvfile, delimiter='\t')
    for row in term_reader:
        lang_num = row["LNUM"]
        lang_term_num = row["TNUM"]
        lang_term_transcription = row["TRAN"]
        lang_term_abbrev = row["WCSC"]
        if lang_num not in language_terms:
            language_terms[lang_num] = []
        language_terms[lang_num].append(lang_term_abbrev)


#Generate the uniform probability prior as a baseline
uniform_prior = color_universe.prior_numpy()
#uniform_prior = np.array(color_universe._prior)

#Munsell to Cielab hues
munsell_to_cielab = np.array(list(munsell_to_cielab.values()))
#print(f"Munsell to CIELAB hues:{munsell_to_cielab}")

SIGMA_SQUARED_SCALAR = 64
#Calculate the meaning space as an isotropic Gaussian centered at the first chip C, for all other points
def meaning(center, point):
    return math.exp((-1/(2*SIGMA_SQUARED_SCALAR) * np.linalg.norm(center-point)))

#Generate the meaning space
meaning_space_indices = np.zeros(shape=(len(color_universe.referents), len(color_universe.referents)))
print(meaning_space_indices)
for c1_index, c1 in enumerate(color_universe.referents):
    for c2_index, c2 in enumerate(color_universe.referents):
        meaning_space_indices[c1_index][c2_index] = meaning(np.array((c1.L, c1.a, c1.b)), np.array((c2.L, c2.a, c2.b)))



meaning_dists = meaning_space_indices / meaning_space_indices.sum(axis=1, keepdims=True)

#print(language_terms)
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

        # Filter AX(A1, A2, A3....) to A0 and JX to J0 - both of these represent pure white/black respectively
        if color.startswith("A"):
            color = "A0"
        elif color.startswith("J"):
            color = "J0"

        # Update speaker records
        if lang_num not in speakers_by_language:
            speakers_by_language[lang_num] = set()
        speakers_by_language[lang_num].add(speaker_num)
        
        #Assemble list of languages by speaker
        if lang_num not in expressions_by_speaker:
            expressions_by_speaker[lang_num] = {}
            average_language_by_meaning[lang_num] = {}
        if speaker_num not in expressions_by_speaker[lang_num]:
            expressions_by_speaker[lang_num][speaker_num] = {}
        if transcription not in expressions_by_speaker[lang_num][speaker_num]:
            expressions_by_speaker[lang_num][speaker_num][transcription] = []
        if color not in average_language_by_meaning[lang_num]:
            average_language_by_meaning[lang_num][color]= Counter()
            
        (expressions_by_speaker[lang_num][speaker_num])[transcription].append(color)

        average_language_by_meaning[lang_num][color][transcription] += 1
        

#print(expressions_by_language)

# For now, assume that if any speaker refers to a color by a given term, that color can be referred to by that term
# for language in expressions_by_language:
#    for expression in expressions_by_language[language]:
#        expressions_by_language[language][expression] = set(expressions_by_language[language][expression])

languages = {}

#Create a Set of Expressions to pull from later when rrandomly sampling for artificial languages
expression_set = set()

#Average out the language, in case of disagreements. For each color, find the most common term associated with that color
for language_code in average_language_by_meaning:
    color_names = {}
    for color in average_language_by_meaning[language_code]:
        #Find the most common term associated with each color chip
        most_frequent_color_term = average_language_by_meaning[language_code][color].most_common(1)[0][0]
        if most_frequent_color_term not in color_names:
            color_names[most_frequent_color_term] = []
        color_names[most_frequent_color_term].append(color)

    #Fill in additional color chips
    for additional_color in color_universe.referents:
        if additional_color not in average_language_by_meaning[language_code]:
            #Get the closest color to the current color
            closest_color_with_term = min([color_term_ref for color_term_ref in color_universe.referents if color_term_ref.name in average_language_by_meaning[language_code].keys()],
                                            key=lambda x: np.linalg.norm(np.array((x.L, x.a, x.b)) - np.array((additional_color.L, additional_color.a, additional_color.b))))
            color_names[average_language_by_meaning[language_code][closest_color_with_term.name].most_common(1)[0][0]].append(additional_color.name)

    #Create list of expressions to add to the Language
    language_expressions = []
    for color_name in color_names:
        #language_expressions.append(Expression(form=expression_form, meaning=Meaning(tuple([Referent(name=color) for color in expressions[expression_form]]), universe=color_universe)))
        expression = Expression(form=color_name, meaning=Meaning(tuple([referents_by_color_code[color] for color in color_names[color_name]]), universe=color_universe))
        expression_set.add(expression)
        language_expressions.append(expression)

    languages[language_code] = Language(language_expressions, lang_code=language_code, name=language_name_from_code[language_code])

#Generate the imshow heatmap for the meaning
plt.imshow(meaning_dists,  cmap="hot")
plt.savefig(f"{current_dir}/outputs/old_meaning_dists.jpg")

#Temporarily use Zaslavsky data to verify information
if(USE_NOGA_ARRAYS):
    noga_meaning_dists = model_data['pU_M']
    noga_prior = np.array([row[0] for row in model_data['pM']])
    noga_bound = model_data['IB_curve']

#Generate the heatmap for the Zaslavasky meaning function
if(USE_NOGA_ARRAYS):
    plt.imshow(noga_meaning_dists, cmap="hot")
    plt.savefig(f"{current_dir}/outputs/noga_meaning_dists.jpg")

#result = meaning(munsell_to_cielab[meaning_space_indices[0]], munsell_to_cielab[meaning_space_indices[1]])
if(USE_ONE_LANGUAGE): 
    first_key = list(languages.keys())[0]
    languages = {first_key: languages[first_key]}
    print(f"Using one language: {languages}")

#Generate the meaning/accuracy/complexity for all languages based on the prior, meaning and Language
language_data = []
for language_code in languages:
    language = languages[language_code]
    #Dereference the lang code to get the actual language associated with it
    language_name = language_name_from_code[language_code] 

    #Exclude the languages Amuzgo, Camsa, Candoshi, Chayahuita, Chiquitano, Cree, Garífuna (Black Carib), Ifugao, Micmac, Nahuatl, Papago (O’odham), Slave, Tacana, Tarahumara (Central), Tarahumara (Western).
    #These languages have fewer than 5 definitions for major terms
    excluded_language_codes = [7, 19, 20, 25, 27, 31, 38, 48, 70, 78, 80, 88, 91, 92, 93]

    if int(language_code) not in excluded_language_codes:
        #RKK - complexity is the number of color terms in the language
        if USE_NOGA_ARRAYS:
            ib_point =  rd.language_to_ib_point(language=language, prior=noga_prior, meaning_dists=(noga_meaning_dists))
        else:
            ib_point =  rd.language_to_ib_point(language=language, prior=uniform_prior, meaning_dists=(meaning_dists))

        if(USE_RKK):
            language_data.append((language_name, "natural", math.log(len(language.expressions)), ib_point[1], ib_point[2]))
        else:
            #Use just the information bound metric
            language_data.append((language_name, "natural", ib_point[0], ib_point[1], ib_point[2]))



#Generate some fake languages using the real languages as a baseline via permutation
#artificial_languages = sampling.get_hypothetical_variants(languages=list(languages.values()), total=400)
artificial_languages = sampling.random_languages(expressions=expression_set, sampling_strategy="stratified", sample_size=20, max_size=25)
#Give enumerated names to each of the artificial languages
for index, artificial_language in enumerate(artificial_languages):
    artificial_language.name = f"artificial_lang_{index}"

#Analyze each of the artificial languages
artificial_lang_count = 0
for artificial_language in artificial_languages:
    artificial_lang_count +=1
    if USE_NOGA_ARRAYS:
        artificial_ib_point =  rd.language_to_ib_point(language=artificial_language, prior=noga_prior, meaning_dists=(noga_meaning_dists))
    else:
        artificial_ib_point =  rd.language_to_ib_point(language=artificial_language, prior=uniform_prior, meaning_dists=(meaning_dists))
    if(USE_RKK):
        language_data.append((f"artificial lang {artificial_lang_count}", "artificial", math.log(len(artificial_language.expressions)), artificial_ib_point[1], artificial_ib_point[2]))
    else:
        language_data.append((f"artificial lang {artificial_lang_count}", "artificial", artificial_ib_point[0],  artificial_ib_point[1], artificial_ib_point[2]))
    
    #Fill in additional color chips
    referents_in_language = []
    for artificial_expression in artificial_language.expressions:
        referents_in_language += artificial_expression.meaning.referents
    colors_in_language = [referent.name for referent in referents_in_language]
    print(f"Artificial language color set size: {len(set(colors_in_language))} colors:{set(colors_in_language)}")
    if(GENERATE_ADDITIONAL_COLOR_CHIPS):
        for additional_color in color_universe.referents:
            if additional_color.name not in colors_in_language:
                #Get the closest color to the current color
                closest_color_with_term = min([color_term_ref for color_term_ref in color_universe.referents if color_term_ref.name in colors_in_language],
                                                key=lambda x: np.linalg.norm(np.array((x.L, x.a, x.b)) - np.array((additional_color.L, additional_color.a, additional_color.b))))
                closest_expression = get_expression_from_language(closest_color_with_term.name, artificial_language)
                new_refs = list(closest_expression.meaning.referents)+[additional_color]
                # Assign the color to the closest expression
                closest_expression.meaning= Meaning(tuple(new_refs), universe=color_universe)



#print(f"Artificial languages{artificial_languages}")

#Convert the real and artificial languages to DataFrames
combined_data = pd.DataFrame(language_data, columns =['name','type','complexity', 'informativity', 'comm_cost'])


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

#Get the IB bound for the specified parameters
#ib_boundary = rd.get_ib_bound(prior=uniform_prior, meaning_dists=meaning_dists, betas=np.logspace(-2, 2, 10))
if(GENERATE_IB_BOUND):
    IB_START = -2
    IB_STOP = 5
    IB_STEP = 30
    betas = np.logspace(IB_START, IB_STOP, IB_STEP)

    #If a cached version of the IB bound already exists, load it
    ib_bound_filename = f"{current_dir}/outputs/ib_bound_{IB_START}_{IB_STOP}_{IB_STEP}.pkl"
    if os.path.exists(ib_bound_filename):
        with open(ib_bound_filename, "rb") as f:
            print(f"Loading information bound from file:{ib_bound_filename}")
            ib_boundary = pickle.load(f)
    else:
        if(USE_NOGA_ARRAYS):
            ib_boundary = rd.get_ib_bound(prior=noga_prior, meaning_dists=noga_meaning_dists, betas=betas)
        else:
            ib_boundary = rd.get_ib_bound(prior=uniform_prior, meaning_dists=meaning_dists, betas=betas)
        #Save the IB bound to a file
        with open(ib_bound_filename, "wb") as f:
            pickle.dump(ib_boundary, f)
            
    ib_boundary_points = pd.DataFrame([("ib_bound", "ib_bound", ib_point.rate, ib_point.accuracy, ib_point.distortion)
                    for ib_point in ib_boundary if ib_point is not None], columns =['name','type','complexity', 'informativity', 'comm_cost'])
    combined_data = pd.concat([ib_boundary_points, combined_data])

#Combine artificial and natural languages for processing
languages = languages | {artificial_language.name:artificial_language for artificial_language in artificial_languages}

#Generate plot for color data across languages
if GENERATE_LANG_COLOR_INFO:
    for language_code in languages:
        language:Language = languages[language_code]
        language_name = language.name
        
        language_color_data = []
        for expression in language.expressions:
            form = expression.form
            for referent in expression.meaning.referents:
                language_color_data.append((form, ord(referent.name[0])-96, int(referent.name[1:])))
        language_color_data = pd.DataFrame(language_color_data, columns =['form','V','H'])
        plot = (
            pn.ggplot(pn.aes(x="H", y="V"))
            + pn.geom_point(language_color_data, pn.aes(color="form"))
        )
        plot.save(f"{current_dir}/outputs/lang-color/color-terms-{language_name}.png", width=8, height=6, dpi=300)
        


#Generate and save plots
plot = (
    pn.ggplot(pn.aes(x="complexity", y="comm_cost"))
    + pn.geom_point(combined_data, pn.aes(color="type"))
    + pn.geom_text(
        combined_data[combined_data["type"] == "natural"],
        pn.aes(label="name"),
        ha="left",
        size=5,
        nudge_x=0.1,)
)


plot.save(f"{current_dir}/outputs/complexity-commcost.png", width=8, height=6, dpi=300)

plot = (
    pn.ggplot(pn.aes(x="complexity", y="informativity"))
    + pn.geom_point(combined_data, pn.aes(color="type"))

    )
"""
    + pn.geom_text(
        combined_data[combined_data["type"] == "natural"],
        pn.aes(label="name"),
        ha="left",
        size=5,
        nudge_x=0.1,)
    """
    
plot.save(f"{current_dir}/outputs/complexity-informativity.png", width=8, height=6, dpi=300)

plot = (
    pn.ggplot(pn.aes(x="informativity", y="comm_cost"))
    + pn.geom_point(combined_data, pn.aes(color="type"))

    )

plot.save(f"{current_dir}/outputs/informativity-commcost.png", width=8, height=6, dpi=300)
