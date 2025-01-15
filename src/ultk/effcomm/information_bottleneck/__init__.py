"""Submodule for Information Bottleneck based analyses of the efficiency of semantic category systems. 

The `ultk.effcomm.information_bottleneck.modeling` includes a friendly API for obtaining IB theoretical bounds and naming models given a specification of the statistics of the semantic domain. This is likely the only submodule you need to import.

The `ultk.effcomm.information_bottlneck.ib` implements the IB update equations, and includes an optimizer object that inherits from the base object in `ba`.

The `ultk.effcomm.information_bottleneck.ba` submodule implements the Blahut-Arimoto algorithm for computing the theoretical bounds of efficient compression. It includes code for simulated annealing (reverse or not) of $\\beta$. 

The `ultk.effcomm.information_bottleneck.tools` submodule includes helper methods for computing informational quantities and dealing with numerical instability.
"""
