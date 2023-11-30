"""Example demonstrating how to extend the objects and tools in the `ultk.language` and `ultk.effcomm` modules to a specific use case.

This example implements a dynamic Lewis-Skyrms 'atomic' signaling game. Two players, Sender and Receiver learn to coordinate on a shared language via rudimentary reinforcement learning. The evolution of meaning conventions is interesting from the efficient communication perspective because we can track the trajectory of the informativity and the cognitive cost of the players' languages.

The `languages` file implements an extension of the fundamental ULTK language, expression, and meaning abstractions into the signaling game use case.

The `agents` file shows how to extend the basic `ultk.effcomm.agent` objects to a dynamic learning agents.

The driver script `main` invokes these implementations and runs the simulation of the learning dynamics on the signaling game. These details are implemented in the `learning` and `game` files, respectively.

See the [README](https://github.com/CLMBRs/altk/tree/main/src/examples/signaling_game#readme) for more information.
"""
