# Analyzing the Relationship between Complexity and Informativity across the World's Languages

Based off [Zaslavsky, Kemp et al's paper on color complexity ](https://www.pnas.org/doi/full/10.1073/pnas.1800521115) and [the corresponding original repo](https://github.com/nogazs/ib-color-naming).

This example creates a "conceptual" / miniature replication of the above paper using the tools provided by the ULTK library.  Right now, the final analysis produces the following plot:
![a plot showing communicative cost and complexity of natural, explored, and dominant languages](https://github.com/CLMBRs/altk/blob/main/src/examples/colors/outputs/plot.png?raw=true)

This README first explains the contents of this example directory, focusing on what the user has to provide that's specific to the color case study, before then explaining the concrete steps taken to produce the above plot.  After that, there is some discussion of what's missing from the above paper and other next steps for this example.

## Contents
`data` consists of language and color data provided by the [World Color Survey](https://linguistics.berkeley.edu/wcs/data.html). Certain files have been slightly edited in order for simplicity of parsing, such as providing a header row.

`outputs` contains outputs of various scripts, as outlined below. 


    `lang_colors` consists of per-language color distributions. Major color terms are graphed per language.

`analyze_data.py` contains functions for graphing the distribution of color terms across language expressions and languages themselves.

`color_grammar.py` contains class definitions for the ColorLanguage and other utility structures.

`generate_wcs_languages.py` contains the function for reading and converting the WCS data to ULTK language structures. It also generates 

`complexity.py` calculates the complexity and informativity of the various color WCS color languages, passed in as a pandas DataFrame.

`graph_colors.py`  contains functions for graphing the distribution of color terms across language expressions and languages themselves.

`util.py` contains utility functions, including the argument parser for running this tool from shell. 

## Usage

From `ultk/examples` base directory:
1. Run `python -m colors.scripts.read_color_universe`: this generates the color universe (the 330 Munsell chips) to be re-used throughout.
    - Consumes: `data/cnum-vhcm-lab-new.txt`
    - Produces: `outputs/color_universe.pkl`
2. Run `python -m colors.scripts.read_natural_languages`: this reads the natural language WCS data and produces ULTK `Language` objects.  (NOTE: still a work-in-progress)
    - Consumes: `data/data/term.txt`, `outputs/color_universe.yaml`
    - Produces: `outputs/natural_languages.yaml`

NOTE: below this is 
Run `python analyze_data.py` from the `colors` folder. This calls `generate_wcs_languages` to generate the language data, then `complexity.py` to generate the complexity, then  Several options are available as command-line settings.:


## Remaining Tasks

At the moment, the density of the probability function per major color term is not factored into the final graphs generated. 

Additionally, the mutual information when probability is taken into account using an assigned probability to the weight matrix gives a large negative value, which should be impossible given the prior is entirely uniform. 

At the moment 


