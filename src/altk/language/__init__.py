"""Classes for modeling (natural or hypothetical) languagese.

At the current stage of development, ALTK focuses on supporting abstractions to model the mapping between expressions and meanings of a language. So far, we leave almost everything besides this basic mapping (morphosyntax, phonology, phonetic inventories, among other features of human languages) to future work.

The `altk.language.language` submodule contains classes for constructing a language, which can contain one or more expressions. 

The `altk.language.semantics` submodule contains classes for defining a universe (meaning space) of referents (denotations) and meanings (categories).
"""