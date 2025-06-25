# The Unnatural Language ToolKit (ULTK)


## Introduction

ULTK is an open-source Python library for computational semantic typology research. ULTK's key features include unifying data structures, algorithms for generating artificial languages, and data analysis tools for related computational experiments.

Read the [documentation](https://clmbr.shane.st/ultk).

## Installing ULTK

First, set up a virtual environment (e.g. via [miniconda](https://docs.conda.io/en/latest/miniconda.html), `conda create -n ultk python=3.11`, and `conda activate ultk`).

1. Download or clone this repository and navigate to the root folder.

2. Install ULTK (We recommend doing this inside a virtual environment)

    `pip install -e .`

## Getting started

- Check out the [examples](https://github.com/CLMBRs/ultk/tree/main/src/examples), starting with a simple efficient communication analysis of [indefinites](https://github.com/CLMBRs/ultk/tree/main/src/examples/indefinites) and [modals](https://github.com/CLMBRs/ultk/tree/main/src/examples/modals).
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

> Kemp, C. & Regier, T. (2012). Kinship Categories Across Languages Reflect General Communicative Principles. Science. https://www.science.org/doi/10.1126/science.1218811

> Zaslavsky, N., Kemp, C., Regier, T., & Tishby, N. (2018). Efficient compression in color naming and its evolution. Proceedings of the National Academy of Sciences, 115(31), 7937–7942. https://doi.org/10.1073/pnas.1800521115

> Denić, M., Steinert-Threlkeld, S., & Szymanik, J. (2022). Indefinite Pronouns Optimize the Simplicity/Informativeness Trade-Off. Cognitive Science, 46(5), e13142. https://doi.org/10.1111/cogs.13142

> Steinert-Threlkeld, S. (2021). Quantifiers in Natural Language: Efficient Communication and Degrees of Semantic Universals. Entropy, 23(10), Article 10. https://doi.org/10.3390/e23101335

</details>

<details>
<summary>Links:</summary>


> Imel, N., & Steinert-Threlkeld, S. (2022). Modal semantic universals optimize the simplicity/informativeness trade-off. Semantics and Linguistic Theory, 1(0), Article 0. https://doi.org/10.3765/salt.v1i0.5346

> Kemp, C., Xu, Y., & Regier, T. (2018). Semantic Typology and Efficient Communication. Annual Review of Linguistics, 4(1), 109–128. https://doi.org/10.1146/annurev-linguistics-011817-045406

</details>

## Citation

```
@article{imel2025ultk,
  author    = {Imel, Nathaniel and Haberland, Claire and Steinert-Threlkeld, Shane},
  title     = {The Unnatural Language ToolKit (ULTK)},
  journal   = {Proceedings of the Society for Computation in Linguistics},
  volume    = {8},
  number    = {1},
  pages     = {46},
  year      = {2025},
  doi       = {10.7275/scil.3144},
  url       = {https://doi.org/10.7275/scil.3144}
}
```
