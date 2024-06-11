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


def Splus(point: Referent) -> bool:
    return point.name in ("specific-known", "specific-unknown")


def Sminus(point: Referent) -> bool:
    return point.name not in ("specific-known", "specific-unknown")


def SEplus(point: Referent) -> bool:
    return point.name in ("npi", "freechoice", "negative-indefinite")


def SEminus(point: Referent) -> bool:
    return point.name not in ("npi", "freechoice", "negative-indefinite")


def Nplus(point: Referent) -> bool:
    return point.name == "negative-indefinite"


def Nminus(point: Referent) -> bool:
    return point.name != "negative-indefinite"


# NB: the grammar should be modified in such a way that R+ and R- can only occur with SE+
# easiest would be to just split SE+ into two features
# more elegant: extra grammar rule (will preserve the impact on complexity)


def Rplus(point: Referent) -> bool:
    return point.name in ("negative-indefinite", "npi")


def Rminus(point: Referent) -> bool:
    return point.name == "freechoice"
