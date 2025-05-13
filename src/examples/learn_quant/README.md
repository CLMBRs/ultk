# Introduction
This module provides code used in the publication `Quantifiers of Greater Monotonicity are Easier to Learn` presented at SALT35 sponsored by the Linguistic Society of America. 

This code provides an example of utilizing the `ultk` package for generating abstract data models of ordered referents in a universe and defining a grammar to generate unique quantifier expressions, as well as enumerating quantifier expressions and evaluating their meaning with respect to a universe of referents. 
The example also includes code for training neural models to correctly verify a given quantifier expression, in addition to functions that compute a quantifier's degree of monotonicity, as described in the published manuscript.

For an introduction to the data structures and research question, please refer to the publication and refer to the (tutorial)[src/examples/learn_quant/notebooks/tutorial.ipynb].

It is highly recommended that the user review the docs of the (`hydra` package)[www.hydra.cc].

# Usage

## Generation
From the `src/examples` directory:
`python -m learn_quant.scripts.generate_expressions`: generates `generated_expressions.yml` files that catalog licensed `QuantifierModel`s given a `Grammar` and `QuantifierUniverse` given the config at `conf/expressions.yaml`.

Using `hydra`, you may refer to the recipe files at `conf/recipes/`:
`python -m learn_quant.scripts.generate_expressions recipe=3_3_3_xi.yaml`
This would generate unique expressions evaluated over a universe and depth specified in the selected recipe config.

You may also override specific parameters:
`python -m learn_quant.scripts.generate_expressions recipe=3_3_3_xi.yaml ++universe.m_size=4`

## Learning

### Sampling
At large universe sizes and genereation depths, the number of generated expressions can be too numerous for completing learning experiments for given compute resources.

After generating a list of expressions, you may sample them using `notebooks/randomize_expression_index.ipynb`. This generates a `.csv` file that simply draws the desired number of expressions and maps them to their ordering in the original `generated_expressions.yaml` file.

### Training with `slurm`
On your `slurm` configured node:

Uncomment the following lines in `conf/learn.yaml`:
```
#  - override hydra/launcher: swarm 
#  - override hydra/sweeper: sweep
```

Run:
`HYDRA_FULL_ERROR=1 python -m learn_quant.scripts.learn_quantifiers --multirun training.lightning=true training.strategy=multirun training.device=cpu model=mvlstm grammar.indices=false`. 

This command will read the config at `conf/learn`, prepare training data based on the chosen quantifier expressions, and run 1 training job per expression using the `hydra` `submitit` plugin **in parallel**. To specify specific `slurm` parameters, you may modify `conf/hydra/launcher/swarm.yaml`.

### Without Slurm
Run:
`HYDRA_FULL_ERROR=1 python -m learn_quant.scripts.learn_quantifiers training.lightning=true training.strategy=multirun training.device=cpu model=mvlstm grammar.indices=false`. 

This command will read the config at `conf/learn.yaml`, prepare training data based on the chosen quantifier expressions, and sequentially run 1 training job for all expressions on your local machine.

### Tracking

If you would like to track experimental runs to MLFlow, you may run an `mlflow` server at the endpoint specified at `tracking.mlflow.host` and have `learn_quant.scripts.learn_quantifiers` track metrics to the server.

You may turn off tracking with MLFlow by setting the config value `tracking.mlflow.active` to `false`.

## Calculation of monotonicity
The `measures.py` script calculates monotonicity for specified quantifier expressions and at given universe sizes. This references the config `conf/learn.yaml`. For expressions, it references the generated expressions at the folder associated with the parameter values at the `expressions` keyspace. If universe parameters are defined at the `measures.monotonicity.universe` keyspace, they will define the size of the universe at which the monotonicity value will be calculated for each expression. `measures.expressions` specifies which expressions will be calculated.

Run `python -m learn_quant.measures` to generate a `.csv` file of the specified monotonicity measurements.

# Content Descriptions

- `scripts`: a set of scripts for generating `QuantifierModels` and measuring various properties of individual models and sets of models.
    - `generate_expressions.py` - This script will reference the configuration file at `conf/expressions.yaml` to generate a `Universe` of the specified dimensions and generate all expressions from a defined `Grammar`. Outputs will be saved in the `outputs` folder. The script will the _shortest_ expression (ULTK `GrammaticalExpression`s) for each possible `Meaning` (set of `Referent`s) verified by licit permutations of composed functions defined in `grammar.yml`. In particular, ULTK provides methods for enumerating all grammatical expressions up to a given depth, with user-provided keys for uniqueness and for comparison in the case of a clash.  By setting the former to get the `Meaning` from an expression and the latter to compare along length of the expression, the enumeration method returns a mapping from meanings to shortest expressions which express them.
    - `learn_quantifiers.py` - This script will reference the configuration file at `conf/learn.yaml`. It loads expressions that are saved to the `output` folder after running the `generate_expressions.py` script. It transforms the data into a format that allows the training of a neural network the relationship between quantifier models and the truth values verified by a particular expression. The script then iterates through loaded expressions and uses Pytorch Lightning to train a neural model to verify randomly sampled models of particular sizes (determined by `M` and `X` parameters). Logs of parameters, metrics, and other artifacts are saved to an `mlruns` folder in directories specified by the configuration of the running `mlflow` server.
- `grammar.yml`: defines the "language of thought" grammar (a ULTK `Grammar` is created from this file in one line in `grammar.py`) for this domain, using the functions in [van de Pol 2023](https://pubmed.ncbi.nlm.nih.gov/36563568/).
- `measures.py`: functions to measure degrees of monotonicity of quantifier expressions according to Section 5 of [Steinert-Threlkeld, 2021](https://doi.org/10.3390/e23101335)
- `outputs`: outputs from the generation routines for creating `QuantifierModel`s and `QuantifierUniverse`s
- `quantifier.py`: Subclasses `ultk`'s `Referent` and `Universe` classes that add additional properties and functionality for quantifier learning with `ultk`
- `sampling.py` - Functions for sampling quantifier models as training data
- `set_primitives.py` - Optional module-defined functions for primitives of the basic grammar. Not used unless specified by the `grammar.typed_rules` key
- `training.py`: Base `torch` classes and helper functions. Referenced only when `training.lightning=false`. Not maintained.
- `training_lightning.py`: Primary training classes and functions. Uses `lightning`. 
- `util.py`: utility functions, I/O, etc.

# TODO:
- Fully implement `hydra`'s `Structured Config` (example begun with `conf/expressions.py`)
- Show example of adding custom primitives with custom-implemented classes (`quantifiers_grammar_xprimitives`)