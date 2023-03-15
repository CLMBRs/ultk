# Indefinite Pronouns Optimize the Simplicity/Informativeness Trade-Off

See [the paper](https://doi.org/10.1111/cogs.13142) and [the corresponding original repo](https://github.com/milicaden/indefinite-pronouns-simplicity-informativeness).

This README will continue to be updated!

## Contents

- `data`: various pieces of data to be used in the analysis
    - `raw/languages_real_40_updated.csv`: this contains Haspelmath's 40 language sample of indefinite systems, as compiled by Bill Croft.  Each row corresponds to an expression, with columns for (i) ontological category (e.g. person) and then (ii) the nine functions from Haspelmath 1997.  Annotations for `PERSON`, `DETER`, and `neg.frag` were added by Milica Denic (see appendices in the above repo). Expressions/items are named with a number for language and a letter indicating the expression.  For example, `03s` corresponds to "some-": `03` is for English and `s` means the "some"-series.
    - `Beekhuizen_prior.csv`: the prior distribution over semantic flavors, estimated from the annotated corpus of Beekhuizen et al 2017
    - `natural_language_indefinites.csv`: a cleaned version of the raw data above (see item 1 in the Usage section below)
- `scripts`: a set of scripts for executing various components of the efficient communication analysis.  These are explained in more detail in the Usage section below.
- `outputs`: outputs from the analysis scripts
- `referents.csv`: this file defines the set of points of communication (which will become `Referent`s in ALTK terms).

    This is a very simple example, with six discrete points which have only on property (a name).  In general, ALTK expects each row to correspond to a unique `Referent` and each column to a feature of that point, with no constraint on the number or type of features.  A `name` column is expected.  See `altk.language.semantics.Universe.from_dataframe` for more details.
- `meaning.py`: this file defines the meaning space (a `Universe` in ALTK terms) of the six flavors defined in `referents.csv` together with their prior
- `grammar.py`: defines the Language of Thought grammar (an ALTK `Grammar`) for this domain, using basic propositional logic and the five semantic features identified in Haspelmath 1997.
- `measures.py`: the measures of simplicity and informativity.  These are basic wrappers around tools from ALTK, linking them to the indefinites grammar and universe.
- `util.py`: utilities for reading and writing ALTK `Expression`s and `Language`s, as well as the natural language data

## Usage

From the `src/examples` directory:
1. `python -m indefinites.scripts.convert_haspelmath`: this script consumes `data/raw/languages_real_40_updated.csv` and generates `data/natural_language_indefinites.csv`.  It primarily transforms Haslpemath's functions into our six semantic flavors, as outlined in the paper linked above (along with a few other cleaning things).
2. `python -m indefinites.scripts.generate_expressions`: generates `outputs/generated_expressions.yml`
3. `python -m indefinites.scripts.esimate_pareto`: generates `outputs/dominating_languages.yml` and `outputs/explored_languages.yml`
4. `python -m indefinites.scripts.measure_natural_languages`: generates `outputs/natural_languages.yml`
5. `python -m indefinites.scripts.combine_data`: generates `outputs/combined_data.csv`
6. `python -m indefinites.scripts.analyze`: generates `outputs/plot.png`
