class FrozenDict(dict):
    def __hash__(self):
        return hash(frozenset(self.items()))

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

    def update(self, *args, **kwargs):
        raise TypeError("FrozenDict is immutable")

    def __repr__(self):
        return f"FrozenDict({super().__repr__()})"
