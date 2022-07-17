from typing import Iterable
from altk.language.language import Language, Expression
from altk.language.semantics import Universe, Meaning, Referent

class State(Referent):
    """In an Atomic-n signaling game, a state represents the observed input to a Sender and the chosen action of a Receiver; in both cases the state can be naturally interpreted as a meaning. Then the signaling game is about Receiver guessing Sender's intended meanings.
    """
    def __init__(self, name: str, weight: float = None) -> None:
        """
        Args:
            name: a str representing the object in the universe.

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
    def __init__(self, objects: Iterable[State]):
        super().__init__(objects)

    def __hash__(self) -> int:
        return hash(tuple(sorted(state.name for state in self.objects)))

class SignalMeaning(Meaning):
    def __init__(self, states: list[State], universe: StateSpace) -> None:
        """Construct the meaning of a signal as a distribution of objects over the state space.
        
        Since altk Meanings are generalized to distributions over objects in a universe, let an atomic state be a peaked distribution over a single point, where one state has probability 1.0, and all others 0.0.

        Args:
            dist: a probability distribution over states
        """
        super().__init__(states, universe)

    def yaml_rep(self) -> dict:
        """Convert to a dictionary representation of the meaning for compact saving to .yml files."""
        return [str(state) for state in self.objects]


class Signal(Expression):
    """In an Atomic-n signaling game, a signal is a single discrete symbol encoding one or more states. 
    
    One of the sense in which the signaling game is 'atomic' is that the signals are atomic -- they do not encode any special structure. The only information they can encode about the states of nature are their identity.
    """
    def __init__(self, form: str, meaning: SignalMeaning = None):
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
    """In an Atomic-N signaling game, a language is a list of signals and their associated states."""

    def __init__(
        self, 
        signals: list[Signal], 
        data: dict = {
                "complexity": None, 
                "accuracy": None, 
                "name": None,
            },
        ):
        self.data = data
        super().__init__(signals)

    def yaml_rep(self) -> dict[str, dict]:
        """Get a data structure for safe compact saving in a .yml file.

        A dict of the language name and its data. This data is itself a dict of a list of the expressions, and other data.
        """
        data = {
            self.data["name"]:
            {
                "expressions": [e.yaml_rep() for e in self.expressions],
                "data": self.data, # e.g., complexity and informativity
            },
        }
        return data