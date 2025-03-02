from typing import Generic, TypeVar
from yaml import YAMLObject
from copy import deepcopy

K = TypeVar("K")
V = TypeVar("V")


class FrozenDict(dict[K, V], Generic[K, V], YAMLObject):
    yaml_tag = "!frozendict"

    def __hash__(self):
        return hash(frozenset(self.items()))

    def __deepcopy__(self, memo):
        output = FrozenDict(
            {deepcopy(k, memo): deepcopy(v, memo) for k, v in self.items()}
        )
        memo[id(self)] = output
        return output

    def __setitem__(self, key, value):
        raise TypeError("FrozenDict is immutable")

    def __delitem__(self, key):
        raise TypeError("FrozenDict is immutable")

    def clear(self):
        raise TypeError("FrozenDict is immutable")

    def pop(self, key, default=None):
        raise TypeError("FrozenDict is immutable")

    def popitem(self):
        raise TypeError("FrozenDict is immutable")

    def setdefault(self, key, default=None):
        raise TypeError("FrozenDict is immutable")

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_mapping(cls.yaml_tag, dict(data))

    @classmethod
    def from_yaml(cls, loader, node):
        return FrozenDict(loader.construct_mapping(node, deep=True))

    def update(self, *args, **kwargs):
        raise TypeError("FrozenDict is immutable")

    def __repr__(self):
        return f"FrozenDict({super().__repr__()})"
