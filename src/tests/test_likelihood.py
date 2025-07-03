from ultk.language.grammar.likelihood import (
    all_or_nothing,
    percent_match_unique,
    percent_match,
    noise_match,
)
from math import log


# The expression can simply act as a function for this purpose
def expression(_):
    return True


def even(i):
    return i % 2 == 0


class TestLikelihood:
    all_true = [(i, True) for i in range(10)]
    all_false = [(i, False) for i in range(10)]
    half = [(i, i % 2 == 0) for i in range(10)]

    def test_all_or_nothing(self):
        assert all_or_nothing(self.all_true, expression) == 1
        assert all_or_nothing(self.all_false, expression) == 0
        assert all_or_nothing(self.half, expression) == 0

    def test_percent_match(self):
        assert percent_match(self.all_true, expression) == 1
        assert percent_match(self.all_false, expression) == 0
        assert percent_match(self.half, expression) == 0.5

    def test_percent_match_unique(self):
        assert percent_match_unique(self.all_true, expression) == 0
        assert percent_match_unique(self.all_false, expression) == 0
        assert percent_match_unique(self.half, expression) == 0
        assert percent_match_unique(self.all_true, even) == 0.5
        assert percent_match_unique(self.all_false, even) == 0.5
        assert percent_match_unique(self.half, even) == 1

    def test_noise_match(self):
        noise_match_func = noise_match(2)
        assert (
            abs(noise_match_func(self.all_true, expression) - log(0.995) * 10) < 0.00001
        )
        assert (
            abs(noise_match_func(self.all_false, expression) - log(0.005) * 10)
            < 0.00001
        )
        assert (
            abs(
                noise_match_func(self.half, expression)
                - (log(0.995) * 5 + log(0.005) * 5)
            )
            < 0.00001
        )
