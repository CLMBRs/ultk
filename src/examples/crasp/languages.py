"""Named formal languages over the alphabet {a, b}, as predicates on strings, for finding their minimal C-RASP programs.

Several are examples from Yang & Chiang (2024); the rest are simple counting and positional properties.
"""

import re
from typing import Callable


def min_prefix_balance(string: str) -> int:
    """The minimum, over prefixes of `string` (including the empty one), of (number of a's - number of b's)."""
    balance = lowest = 0
    for symbol in string:
        balance += 1 if symbol == "a" else -1
        lowest = min(lowest, balance)
    return lowest


NAMED_LANGUAGES: dict[str, Callable[[str], bool]] = {
    "ends in b": lambda s: s[-1] == "b",
    "starts with a": lambda s: s[0] == "a",
    "contains a b": lambda s: "b" in s,
    "only a's (a*)": lambda s: "b" not in s,
    "exactly one b": lambda s: s.count("b") == 1,
    "at most one b": lambda s: s.count("b") <= 1,
    "length at least 2": lambda s: len(s) >= 2,
    "length exactly 1": lambda s: len(s) == 1,
    "more a's than b's": lambda s: s.count("a") > s.count("b"),
    "at least as many a's as b's": lambda s: s.count("a") >= s.count("b"),
    "equal numbers of a's and b's": lambda s: s.count("a") == s.count("b"),
    "at least twice as many a's as b's": lambda s: s.count("a") >= 2 * s.count("b"),
    "a*b*": lambda s: re.fullmatch("a*b*", s) is not None,
    "b*a*": lambda s: re.fullmatch("b*a*", s) is not None,
    "some a before some b": lambda s: re.search("a.*b", s) is not None,
    "last symbol is a strict majority": lambda s: s.count(s[-1]) > len(s) / 2,
    "every prefix has at least as many a's as b's": lambda s: min_prefix_balance(s)
    >= 0,
    "first symbol equals last symbol": lambda s: s[0] == s[-1],
    "a*b*a*": lambda s: re.fullmatch("a*b*a*", s) is not None,
    "a^n b^n": lambda s: re.fullmatch("a*b*", s) is not None
    and s.count("a") == s.count("b"),
    "Dyck-1 (a = open, b = close)": lambda s: min_prefix_balance(s) >= 0
    and s.count("a") == s.count("b"),
    "even number of a's (parity)": lambda s: s.count("a") % 2 == 0,
}
