from typing import Iterable
from altk.language.language import Language, Expression
from altk.language.semantics import Universe, Meaning, Referent


class State(Referent):
    """In a simple Lewis-Skyrms signaling game, a state represents the observed input to a Sender and the chosen action of a Receiver; in both cases the state can be naturally interpreted as a meaning. Then the signaling game is about Receiver guessing Sender's intended meanings."""

    def __init__(self, name: str, weight: float = None) -> None:
        """
        Args:
            name: a str representing a single state in the universe.

            weight: a float that can represent the importance of the state. This quantity can be identified with the prior probability of a state. It can also be used (normalized) to determine the meaning of a signal as a probability distribution over states.
        """
        super().__init__(name, weight)

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, __o: object) -> bool:
        return self.name == __o.name


class StateSpace(Universe):
    """The StateSpace represents the agents' environment."""

    def __init__(self, states: Iterable[State]):
        """Construct a universe shared by all agents in a signaling game.

        Args:
            states: the list of _all_ states that can be referred to.
        """
        super().__init__(states)

    def __hash__(self) -> int:
        return hash(tuple(sorted(state.name for state in self.referents)))


class SignalMeaning(Meaning):
    def __init__(self, states: list[State], universe: StateSpace) -> None:
        """Construct the meaning of a signal as the set of states it can refer to.

        In altk, Meanings can be generalized from a set of referents to distributions over those referents. by default, we let be a peaked distribution over a single point, where one state has probability 1.0, and all others 0.0.

        Args:

            states: the list of atomic states that a signal can be used to communicate.

            universe: the semantic space that the signal meaning is a subset of.
        """
        super().__init__(states, universe)

    def yaml_rep(self) -> dict:
        """Convert to a dictionary representation of the meaning for compact saving to .yml files."""
        return [str(state) for state in self.referents]


class Signal(Expression):
    """In the simple atomic signaling game, a signal is a single discrete symbol encoding one or more states.

    One of the senses in which the signaling game is 'atomic' is that the signals are atomic -- they do not encode any special structure (e.g. features such as size, color, etc.). The only information they can encode about the states of nature are their identity.
    """

    def __init__(self, form: str, meaning: SignalMeaning = None):
        """A signal is characterized by its form and meaning.

        We treat signals as equal up to form, even they might communicate different meanings. Note the `__eq__` and `__hash__` implementations encode this choice.

        Args:

            form: a str representing the identity of the signal, e.g. the sound a Sender produces and a Receiver hears.

            meaning: a SignalMeaning representing the set of states the signal can be used to refer to. Default is None to reflect the idea that in signaling games, signals do not have a standing meaning until convention is achieved.
        """
        super().__init__(form, meaning)

    def yaml_rep(self):
        """Convert to a dictionary representation of the expression for compact saving to .yml files."""
        return {
            "form": self.form,
            "meaning": self.meaning.yaml_rep(),
        }

    def __str__(self) -> str:
        return self.form

    def __eq__(self, __o: object) -> bool:
        return self.form == __o.form

    def __hash__(self) -> int:
        return hash(self.form)


class SignalingLanguage(Language):
    """In the simple atomic signaling game, a language is a list of signals and their associated states."""

    def __init__(
        self,
        signals: list[Signal],
        data: dict = {
            "complexity": None,
            "accuracy": None,
            "name": None,
        },
    ):
        super().__init__(signals, data=data)

    def yaml_rep(self) -> dict[str, dict]:
        """Get a data structure for safe compact saving in a .yml file.

        A dict of the language name and its data. This data is itself a dict of a list of the expressions, and other data.
        """
        data = {
            self.data["name"]: {
                "expressions": [e.yaml_rep() for e in self.expressions],
                "data": self.data,  # e.g., complexity and informativity
            },
        }
        return data
