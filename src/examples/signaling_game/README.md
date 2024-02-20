# Signaling game

## Introduction

A signaling game is a an evolutionary game theoretic model, wherein a Sender and Receiver aim to maximize a joint payoff by coordinating on signals. When strategies evolve towards such coordination over repeated plays of the game, this can model the emergence of communication.

This example repo exists to demo simulations of the evolution of meaning using very simple reinforcement learning and ULTK tools. We use various ultk modules for the game objects and for analysis (esp. [ultk.effcomm.agent](../../ultk/effcomm/agent.py)).

See a more scaled-up [example](https://github.com/nathimel/rdsg) of signaling game simulations using ultk.

## Running a simulation

To run a basic signaling game simulation, use the following command:

`python3 main.py`

or

`./run_example.sh`

This will produce results in the folders `outputs/default` or `outputs/example`, respectively. They include plots, weights and the resulting 'languages'.

## Experimenting

<details>
<summary> Adjusting parameters and using ULTK more generally to do signaling games.
</summary>

### Existing game parameters

Run different games by supplying command line arguments, described in `python3 main.py -h`. One can vary, for example:

- the number of states
- the number of signals
- the number of rounds in a game
- the learning rate

### Using ULTK for signaling games

The most general aspects of the communicative agents, measures of communicative success and the language primitives are implemented in ULTK. Some signaling-specific concepts and wrappers implemented in the `.py` files.

This example is limited for simplicity, but is also intended to be an recylable outline for additional analyses, such as:

- extending to more than two agents
- defining more powerful learning agents
- defining different or multiple objectives besides perfect recovery of atomic states
- exploring different evolutionary trajectories of languages in the 2D trade-off space.

</details>

## Links

Here are a few links to more information about signaling games.
<details>
<summary>
Links
</summary>
<br>

- The idea of a signaling game was introduced by David Lewis in his book, [Convention](https://www.wiley.com/en-us/Convention:+A+Philosophical+Study-p-9780631232568).
- A gentle but profound introduction to signaling games research is Brian Skyrms' book, [Signals](https://oxford.universitypressscholarship.com/view/10.1093/acprof:oso/9780199580828.001.0001/acprof-9780199580828).
- [EGG](https://github.com/facebookresearch/EGG) is a software library for emergent communication and includes a neural agent signaling game [example](https://github.com/facebookresearch/EGG/tree/main/egg/zoo/signal_game).

References

> Kharitonov, Eugene, Roberto Dessì, Rahma Chaabouni, Diane Bouchacourt, and Marco Baroni. 2021. “EGG: A Toolkit for Research on Emergence of LanGuage in Games.” <https://github.com/facebookresearch/EGG>.

> Lazaridou, Angeliki, Alexander Peysakhovich, and Marco Baroni. 2017. “Multi-Agent Cooperation and the Emergence of (Natural) Language,” April. <https://openreview.net/forum?id=Hk8N3Sclg>.

> Lewis, David K. (David Kellogg). 1969. “Convention: A Philosophical Study.” Cambridge: Harvard University Press.

> Skyrms, Brian. 2010. Signals: Evolution, Learning, and Information. Oxford: Oxford University Press. <https://doi.org/10.1093/acprof:oso/9780199580828.001.0001>.

</details>
