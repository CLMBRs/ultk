# Modals Optimize the Simplicity/Informativeness Trade-Off

This example is a ULTK version of the analysis in Imel & Steinert-Threlkeld (2022), "Modals in natural language optimize the simplicity/informativeness trade-off" (SALT 32), whose original code is at [nathimel/modals-effcomm](https://github.com/nathimel/modals-effcomm).  It follows that project's main setup: a meaning space of 2 forces and 3 flavors, communicative need estimated from a corpus, literal agents with a half-credit utility, and the Independence of Force and Flavor (IFF) universal.  Unlike the original, complexity is the length of the shortest expression in a Boolean language of thought (see `data/grammar.yaml`), found exactly with `Grammar.get_minimal_expressions`.

## Contents

- `data`:
    - `universe.csv`: the six (force, flavor) pairs, with the prior (communicative need) estimated in the original project
    - `grammar.yaml`: the language of thought: Boolean connectives over force and flavor features
    - `natural_languages`: one directory per language, with modals data (`modals.csv`) and metadata (`metadata.yml`), from the [Database of Modal Typology](https://github.com/CLMBRs/modal-typology), copied from `data/natural_languages` of [nathimel/modals-effcomm](https://github.com/nathimel/modals-effcomm) (commit `58f9925`).  Only languages whose data comes from a journal paper or from elicitation are used.
- `meaning.py`, `grammar.py`: the ULTK `Universe` and `Grammar`
- `measures.py`: complexity, communicative cost, and the IFF universal
- `util.py`: reading the natural language data, and writing languages
- `scripts`: the steps of the analysis (see below)
- `outputs`: outputs of the scripts

## Usage

From the `src/examples` directory:

1. `python -m modals.scripts.generate_expressions`: generates `outputs/generated_expressions.yml`, a shortest expression for every meaning
2. `python -m modals.scripts.estimate_pareto`: estimates the Pareto frontier with an evolutionary algorithm, generating `outputs/dominating_languages.yml` and `outputs/explored_languages.yml`
3. `python -m modals.scripts.measure_natural_languages`: generates `outputs/natural_languages.yml`
4. `python -m modals.scripts.combine_data`: generates `outputs/combined_data.csv`
5. `python -m modals.scripts.analyze`: generates `outputs/plot.png`, and prints some statistics
