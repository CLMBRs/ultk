"""Tools for measuring languages for communicative efficiency.

Submodules divide the labor of a computational experiment performing an efficiency analysis of a language into several parts: generating and sampling the space of possible languages, measuring their properties, and determining which languages optimize efficient trade-offs w.r.t these properties.

The `ultk.effcomm.sampling` submodule implements several methods for generating hypothetically possible languages of a given type, by sampling from a set of possible expressions, or permuting the expression-meaning mapping of an existing language.

The `ultk.effcomm.optimization` submodule contains a general implementation of an evolutionary algorithm, which can be used to estimate a Pareto frontier of optimal solutions to an efficiency trade-off. It can also be used as a technique for randomly exploring the space of possible languages.

The `ultk.effcomm.tradeoff` submodule contains tools for measuring a pool of languages for various properties, finding which languages are Pareto dominant with respect to two properties, and setting attributes of the language objects for further analysis.

The `ultk.effcomm.analysis` submodule contains tools for performing numerical analyses and producing paradigmatic plots of languages in 2D trade-off space.

The `ultk.effcomm.information_bottleneck` submodule contains tools for Information Bottleneck based analyses of the communicative efficiency of languages. Specificially, it includes an implementation of the IB-BA algorithm and code for evaluating the efficiency of semantic category systems.

The `ultk.effcomm.agent` submodule implements classes for constructing various speakers and listeners of a language. These are unified abstractions from the Rational Speech Act framework.

The `ultk.effcomm.informativity` submodule implements tools for computing the literal or pragmatic informativity of a language, based on speaker/listener  abstractions described above.
"""
