# The Unnatural Language ToolKit (ULTK)

<img alt="Four examples of many recent results showing that natural languages are optimized for efficient communication." src="https://raw.githubusercontent.com/CLMBRs/ultk/main/images/plots.jpeg" width="100%" />

## Introduction

ULTK is a software library that aims to support [efficient communication analyses](https://github.com/CLMBRs/ultk/blob/main/images/mit-altk.pdf) of natural language. This is a line of research that aims to explain why natural languages have the structure that they do in terms competing pressures to minimize cognitive complexity and maximize communicative accuracy.

Key features:

- Primitives for constructing semantic spaces, expressions, and languages
- Tools for measuring informativity of languages, communicative success of RSA speakers and listeners
- Language population sampling and optimization w.r.t Pareto fronts


ULTK is a long term project and it is currently in its early stages. It is intended to help lower the barrier to entry for certain research in computational semantics, and to unify methodologies. If you find something confusing, please open an issue. If you have a phenomena of interest in linguistic semantics that you want to run an efficient communication analysis on, please contact the contributors.

Read the [documentation](https://clmbr.shane.st/ultk).

## Installing ULTK

First, set up a virtual environment (e.g. via [miniconda](https://docs.conda.io/en/latest/miniconda.html), `conda create -n ultk python=3.11`, and `conda activate ultk`).

1. Download or clone this repository and navigate to the root folder.

2. Install ULTK (We recommend doing this inside a virtual environment)

    `pip install -e .`

3. In addition, this project requires [rdot](), a python library of rate-distortion optimization tools. When a stable version is available, we will add this to the ULTK `setup.py` file; for now, install via git:

    `python3 -m pip install git+https://github.com/nathimel/rdot.git`

## Getting started

- Check out the [examples](https://github.com/CLMBRs/ultk/tree/main/src/examples), starting with a basic signaling game.  The examples folder also contains a simiple efficient communication analysis of [indefinites](https://github.com/CLMBRs/ultk/tree/main/src/examples/indefinites).
- To see more scaled up usage examples, visit the codebase for an efficient communication analysis of [modals](https://github.com/nathimel/modals-effcomm) or [sim-max games](https://github.com/nathimel/rdsg).
- For an introduction to efficient communication research, here is a [survey paper](https://www.annualreviews.org/doi/abs/10.1146/annurev-linguistics-011817-045406) of the field.
- For an introduction to the RSA framework, see [this online textbook](http://www.problang.org/).

## Modules

There are two modules. The first is [ultk.effcomm](https://clmbr.shane.st/ultk/ultk/effcomm.html), which includes methods for measuring informativity of languages and/or communicative success of Rational Speech Act agents, and for language population sampling and optimization w.r.t Pareto fronts.

The second module is [ultk.language](https://clmbr.shane.st/ultk/ultk/language.html), which contains primitives for constructing semantic spaces, expressions, and languages.  It also has a `grammar` module which can be used for building expressions in a Language of Thought and measuring complexity in terms of minimum description length, as well as for natural language syntax.

The source code is available on github [here](https://github.com/CLMBRs/ultk).

## Testing

Unit tests are written in [pytest](https://docs.pytest.org/en/7.3.x/) and executed via running `pytest` in the `src/tests` folder. 

## References

<details>
<summary>Figures:</summary>

> Kinship Categories Across Languages Reflect General Communicative Principles | Science. (n.d.). Retrieved February 27, 2023, from https://www.science.org/doi/10.1126/science.1218811

> Zaslavsky, N., Kemp, C., Regier, T., & Tishby, N. (2018). Efficient compression in color naming and its evolution. Proceedings of the National Academy of Sciences, 115(31), 7937–7942. https://doi.org/10.1073/pnas.1800521115

> Denić, M., Steinert-Threlkeld, S., & Szymanik, J. (2022). Indefinite Pronouns Optimize the Simplicity/Informativeness Trade-Off. Cognitive Science, 46(5), e13142. https://doi.org/10.1111/cogs.13142

> Steinert-Threlkeld, S. (2021). Quantifiers in Natural Language: Efficient Communication and Degrees of Semantic Universals. Entropy, 23(10), Article 10. https://doi.org/10.3390/e23101335

</details>

<details>
<summary>Links:</summary>

> Imel, N. (2023). The evolution of efficient compression in signaling games. PsyArXiv. https://doi.org/10.31234/osf.io/b62de

> Imel, N., & Steinert-Threlkeld, S. (2022). Modal semantic universals optimize the simplicity/informativeness trade-off. Semantics and Linguistic Theory, 1(0), Article 0. https://doi.org/10.3765/salt.v1i0.5346

> Kemp, C., Xu, Y., & Regier, T. (2018). Semantic Typology and Efficient Communication. Annual Review of Linguistics, 4(1), 109–128. https://doi.org/10.1146/annurev-linguistics-011817-045406

</details>
