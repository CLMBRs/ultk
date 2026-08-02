"""Explicit variants of the entropy-based degree of monotonicity.

The original paper defines upward degree from the least upward-monotone
*majorant* of a Boolean function:

    C_up(Q)(x) = max {Q(y) : y <= x}.

There are four natural monotone approximations of Q on a partial order:

    C_up(Q)    least upward-monotone majorant   (true predecessor exists)
    I_up(Q)    greatest upward-monotone minorant (all successors are true)
    C_down(Q)  least downward-monotone majorant (true successor exists)
    I_down(Q)  greatest downward-monotone minorant (all predecessors are true)

The manuscript-era implementation used C_up for upward and C_down for
downward. This is order-dual: reversing the order swaps the two computations.
It is not generally complement-invariant because complementation swaps closure
and interior:

    C_up(not Q) = not I_down(Q).

The complement-dual implementation uses C_up for upward and I_down for
downward, equivalently ``down(Q) = up(not Q)``. It is complement-invariant but
not order-dual as a one-sided approximation.

A two-sided score combines closure and interior for each direction:

    up_two_sided(Q)   = combine(score(C_up(Q)), score(I_up(Q)))
    down_two_sided(Q) = combine(score(C_down(Q)), score(I_down(Q))).

Any symmetric combination such as the arithmetic mean, minimum, or maximum
preserves both order duality and complement invariance. They encode different
constructs: the mean averages both approximations, the maximum accepts the
better one, and the minimum requires both to support the directional score.
The minimum is therefore the conservative choice when boundary agreement from
an interior approximation should not by itself raise monotonicity.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np


DirectionScores = np.ndarray
EntropyScorer = Callable[..., float]
Combine = Literal["mean", "min", "max"]


def entropy_variant_scores(
    all_models: np.ndarray,
    reference_a: np.ndarray,
    reference_b: np.ndarray,
    quantifier: np.ndarray,
    scorer: EntropyScorer,
    cfg,
    *,
    variant: Literal[
        "majorant",
        "complement_dual",
        "two_sided_mean",
        "two_sided_min",
        "two_sided_max",
    ],
) -> DirectionScores:
    """Return [right-up, left-up, right-down, left-down] for one variant.

    ``scorer`` is ``upward_monotonicity_entropy`` from ``measures.py``.
    Its ``flip=False`` path scores an upward closure (C_up); ``flip=True``
    scores a downward closure (C_down). Interior scores are obtained using
    De Morgan duality and mutual-information invariance under Boolean
    complementation.
    """

    q = np.asarray(quantifier, dtype=int).reshape(-1)
    not_q = 1 - q
    references = (reference_a, reference_b)

    up_closure = np.array(
        [scorer(all_models, reference, q, cfg, False) for reference in references]
    )
    down_closure = np.array(
        [scorer(all_models, reference, q, cfg, True) for reference in references]
    )
    # score(Q, I_up(Q)) = score(not Q, C_down(not Q))
    up_interior = np.array(
        [scorer(all_models, reference, not_q, cfg, True) for reference in references]
    )
    # score(Q, I_down(Q)) = score(not Q, C_up(not Q))
    down_interior = np.array(
        [scorer(all_models, reference, not_q, cfg, False) for reference in references]
    )

    if variant == "majorant":
        return np.concatenate([up_closure, down_closure])
    if variant == "complement_dual":
        return np.concatenate([up_closure, down_interior])

    combine_name = variant.removeprefix("two_sided_")
    combine = {"mean": np.mean, "min": np.min, "max": np.max}[combine_name]
    upward = combine(np.stack([up_closure, up_interior]), axis=0)
    downward = combine(np.stack([down_closure, down_interior]), axis=0)
    return np.concatenate([upward, downward])


def comparable_pair_relation(
    all_models: np.ndarray, reference_models: np.ndarray
) -> np.ndarray:
    """Return relation[x, y] == True exactly when x <= y in one argument."""

    model_ints = all_models.dot(1 << np.arange(all_models.shape[-1]))
    reference_ints = reference_models.dot(1 << np.arange(reference_models.shape[-1]))
    subset = (model_ints[:, None] & model_ints[None, :]) == model_ints[:, None]
    same_base = reference_ints[:, None] == reference_ints[None, :]
    return subset & same_base


def violation_rate_scores(
    all_models: np.ndarray,
    reference_a: np.ndarray,
    reference_b: np.ndarray,
    quantifier: np.ndarray,
) -> DirectionScores:
    """Score directions as 1 minus violations / all proper comparable pairs.

    This variant has transparent numerator semantics and satisfies both order
    duality and complement invariance. Its limitation is scale: large lattices
    contain many nonviolating pairs, so even visibly nonmonotone functions can
    receive high scores.
    """

    q = np.asarray(quantifier, dtype=bool).reshape(-1)
    relations = [
        comparable_pair_relation(all_models, reference_a),
        comparable_pair_relation(all_models, reference_b),
    ]
    scores = []
    for relation in relations:
        proper = relation & ~np.eye(len(q), dtype=bool)
        upward_violations = proper & q[:, None] & ~q[None, :]
        scores.append(1 - upward_violations.sum() / proper.sum())
    for relation in relations:
        proper = relation & ~np.eye(len(q), dtype=bool)
        downward_violations = proper & ~q[:, None] & q[None, :]
        scores.append(1 - downward_violations.sum() / proper.sum())
    return np.asarray(scores, dtype=float)
