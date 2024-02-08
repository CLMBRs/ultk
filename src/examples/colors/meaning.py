import csv
import pandas as pd
from ultk.language.semantics import Meaning, Universe, Referent
from ultk.language.language import Language, Expression

wcs_dialect = csv.Dialect
wcs_dialect.delimiter = "\t"

language_codes = dict()
language_terms = dict()

#Generate all WCS color codes
hues = range(0,41)
lightnesses = ["A","B","C","D","E","F","G","H","I","J"]
color_codes = []
for lightness in lightnesses:
    color_codes.extend([ {"name":(lightness + str(hue))} for hue in hues] )

#Generate referents for all color codes
referents = pd.DataFrame(color_codes)
color_universe = Universe.from_dataframe(referents)

with open('data/lang.txt',newline='', encoding="utf-8" ) as csvfile:
    lang_reader = csv.DictReader(csvfile, delimiter='\t')
    for row in lang_reader:
        language_codes[row["LNUM"]] = row["LNAME"]

#print(language_codes)

with open('data/dict.txt', newline='', encoding="utf-8" ) as csvfile:
    term_reader = csv.DictReader(csvfile, delimiter='\t')
    for row in term_reader:
        lang_num = row["LNUM"]
        lang_term_num = row["TNUM"]
        lang_term_transcription = row["TRAN"]
        lang_term_abbrev = row["WCSC"]
        if lang_num not in language_terms:
            language_terms[lang_num] = []
        language_terms[lang_num].append(lang_term_abbrev)

#print(language_terms)
expressions_by_language = {}
with open('data/foci-exp.txt', newline='', encoding="utf-8" ) as csvfile:
    color_reader = csv.DictReader(csvfile, delimiter='\t')
    for row in color_reader:
        lang_num = row["LNUM"]
        transcription = row["WCSC"]
        color = row["COLOR"]
        if lang_num not in expressions_by_language:
            expressions_by_language[lang_num] = {}
        if transcription not in expressions_by_language[lang_num]:
            expressions_by_language[lang_num][transcription] = []
        (expressions_by_language[lang_num])[transcription].append(color)
print(expressions_by_language)

#For now, assume that if any speaker refers to a color by a given term, that color can be referred to by that term
for language in expressions_by_language:
    for expression in expressions_by_language[language]:
        expressions_by_language[language][expression] = set(expressions_by_language[language][expression])

languages = []

for language in expressions_by_language:
    expressions = []
    for expression in expressions_by_language[language]:
        print(f"Language:{language} | Expression:{expression} | Colors:{expressions_by_language[language][expression]}")
        expressions.append(Expression(form=expression, meaning=Meaning(tuple([Referent(name=color) for color in expressions_by_language[language][expression]]), universe=color_universe)))
    languages.append(Language(expressions))

print(languages)
