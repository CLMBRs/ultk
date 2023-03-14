# Indefinite Pronouns Optimize the Simplicity/Informativeness Trade-Off

See [the paper](https://doi.org/10.1111/cogs.13142) and [the corresponding repo](https://github.com/milicaden/indefinite-pronouns-simplicity-informativeness).

This README will be updated in due time!

## Usage

From the `src/examples` directory:
0. `python -m indefinites.scripts.convert_haspelmath`: generates `data/natural_language_indefinites.csv`
1. `python -m indefinites.scripts.generate_expressions`: generates `outputs/generated_expressions.yml`
2. `python -m indefinites.scripts.esimate_pareto`: generates `outputs/dominating_languages.yml` and `outputs/explored_languages.yml`
3. `python -m indefinites.scripts.measure_natural_languages`: generates `outputs/natural_languages.yml`