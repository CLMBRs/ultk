from ultk.language.language import Expression
from typing import Iterable
from yaml import dump, Dumper


def write_expressions(expressions: Iterable[Expression], filename: str) -> None:
    """Write expressions to a YAML file."""
    with open(filename, "w") as f:
        dump([expr.to_dict() for expr in expressions], f, Dumper=Dumper)
