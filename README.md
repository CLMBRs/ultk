# The Artificial Language ToolKit (ALTK)

## Introduction

ALTK is a toolikt that aims to allow researchers to do "Unnatural Language Semantics" -- a style of research that tries to describe and explain natural languages by situating them among the logically possible ones. A particular focus is on _efficient communication_: determining whether particular (especially semantic) domains are optimized for a trade-off between cognitive complexity and communicative precision.

Key features:

- Tools for constructing arbitrary semantic spaces, expressions, and languages
- Simplified logic for measuring informativity of languages / communicative success of signaling agents
- Methods for language population sampling and optimization w.r.t Pareto fronts

## List of implemented analyses

List of reproduced experiments from published work:

- modals
- quantifiers
- color terms

## Installing ALTK

1. Create a fresh conda environment with Python 3.6 or newer

2. Install via pip

3. Run an experiment, e.g. an efficient communication analysis of natural language modals

## ALTK structure

The repo is organized as follows:

```md
src
└── altk
    ├── effcomm # efficient communication module
    └── language # general artificial language module
```

## How-to-start

- Some resources for an introduction to efficient communication.
- A jupyter/colab tutorial for ALTK.
- Original code for experiments reproduced with ALTK.

Do you have a phenomena of interest in linguistic semantics that you want to run an efficient communication analysis on? ALTK is designed to lower the barrier to entry for such research and unify methodologies, so if you find something confusing or would like to collaborate, please contact us, open an issue or start contributing!

## Contributing

Link to a Contributing.md.

### TODO

For now, ALTK is focused on providing a unified library for efficient communication analyses of natural language semantic domains. Future work may extend the library to:

- providing other causal analyses of linguistic domains, e.g. in terms of ease of learnability
- building blocks to support closer integration of emergent communication, language modeling, and evolutionary analyses
- generating artificial data for NLP experiments
- constructing languages for psycholinguistics research
