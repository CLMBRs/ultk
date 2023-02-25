"""ALTK is a software library that aims to support research in [Unnatural Language Semantics](https://gu-clasp.github.io/static/951dfcd6d280ce7416e79e206c291358/clasp.pdf) -- a program in linguistics and cognitive science that tries to describe and explain the properties of natural languages by comparing them to the much larger set of mathematically possible languages.

A current focus is on *efficient communication*: determining whether linguistic meanings are optimized for a trade-off between cognitive complexity and communicative precision.

There are two modules. The first is `altk.effcomm`, which includes methods for measuring informativity of languages / communicative success of Rational Speech Act agents, and for language population sampling and optimization w.r.t Pareto fronts. The `altk.effcomm.information` submodule includes tools for running Information Bottleneck style analyses of languages.

The second module is `altk.language`, which contains primitives for constructing semantic spaces, expressions, and languages.

See `examples` for a demo and the [README](https://github.com/CLMBRs/altk#readme) for links and references.
"""

__docformat__ = "google"
