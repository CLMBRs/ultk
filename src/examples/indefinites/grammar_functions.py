from ultk.language.semantics import Referent


def _and(a: bool, b: bool) -> bool:
    return a and b


def _or(a: bool, b: bool) -> bool:
    return a or b


def _not(a: bool) -> bool:
    return not a


def Kplus(point: Referent) -> bool:
    return point.name == "specific-known"


def Kminus(point: Referent) -> bool:
    return point.name != "specific-known"
