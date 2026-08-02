"""Alternative graded monotonicity metrics for finite set-theoretic universes."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import permutations
from math import comb, log2

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import maximum_flow


Predicate = Callable[[frozenset[int], frozenset[int]], bool]
DIRECTIONS = ("RU", "LU", "RD", "LD")


@dataclass(frozen=True)
class FiniteSetUniverse:
    """All ordered pairs of subsets of an ``n``-element domain."""

    n: int
    a_bits: np.ndarray
    b_bits: np.ndarray

    @classmethod
    def create(cls, n: int) -> "FiniteSetUniverse":
        size = 1 << n
        a_bits = np.repeat(np.arange(size, dtype=np.int64), size)
        b_bits = np.tile(np.arange(size, dtype=np.int64), size)
        return cls(n=n, a_bits=a_bits, b_bits=b_bits)

    @property
    def size(self) -> int:
        return len(self.a_bits)

    def truth_values(self, predicate: Predicate) -> np.ndarray:
        sets = [
            frozenset(index for index in range(self.n) if bits & (1 << index))
            for bits in range(1 << self.n)
        ]
        return np.fromiter(
            (
                predicate(sets[int(a_bits)], sets[int(b_bits)])
                for a_bits, b_bits in zip(self.a_bits, self.b_bits)
            ),
            dtype=np.int8,
            count=self.size,
        )

    def relation(self, direction: str, *, immediate: bool = False) -> np.ndarray:
        """Return ``relation[x, y]`` for movement in ``direction``."""

        if direction not in DIRECTIONS:
            raise ValueError(f"Unknown direction: {direction}")
        is_right = direction[0] == "R"
        fixed = self.a_bits if is_right else self.b_bits
        varied = self.b_bits if is_right else self.a_bits
        subset = (varied[:, None] & varied[None, :]) == varied[:, None]
        relation = subset & (fixed[:, None] == fixed[None, :])
        if direction[1] == "D":
            relation = relation.T
        if immediate:
            counts = np.fromiter(
                (int(value).bit_count() for value in varied),
                dtype=np.int8,
                count=self.size,
            )
            relation &= np.abs(counts[None, :] - counts[:, None]) == 1
        return relation

    def strict_relation(self, direction: str) -> np.ndarray:
        relation = self.relation(direction)
        return relation & ~np.eye(self.size, dtype=bool)

    def context_ids(self, direction: str) -> np.ndarray:
        return self.a_bits if direction[0] == "R" else self.b_bits

    def movement_distance(self, direction: str) -> np.ndarray:
        varied = self.b_bits if direction[0] == "R" else self.a_bits
        return np.abs(
            np.fromiter(
                (int(value).bit_count() for value in varied),
                dtype=np.int8,
                count=self.size,
            )[:, None]
            - np.fromiter(
                (int(value).bit_count() for value in varied),
                dtype=np.int8,
                count=self.size,
            )[None, :]
        )

    def maximal_chains(self, direction: str) -> np.ndarray:
        """Return uniformly enumerated maximal chains for one argument."""

        is_right = direction[0] == "R"
        chains = []
        for fixed in range(1 << self.n):
            for ordering in permutations(range(self.n)):
                varied = 0
                chain = []
                for bit in (None, *ordering):
                    if bit is not None:
                        varied |= 1 << bit
                    a_bits, b_bits = (fixed, varied) if is_right else (varied, fixed)
                    chain.append(a_bits * (1 << self.n) + b_bits)
                if direction[1] == "D":
                    chain.reverse()
                chains.append(chain)
        return np.asarray(chains, dtype=np.int32)


def closure(values: np.ndarray, relation: np.ndarray) -> np.ndarray:
    q = np.asarray(values, dtype=bool)
    return np.any(relation & q[:, None], axis=0)


def interior(values: np.ndarray, relation: np.ndarray) -> np.ndarray:
    q = np.asarray(values, dtype=bool)
    return ~np.any(relation & ~q[None, :], axis=1)


def binary_entropy(values: np.ndarray) -> float:
    probability = np.asarray(values, dtype=float).mean()
    if probability in (0.0, 1.0):
        return 0.0
    return float(
        -probability * np.log2(probability)
        - (1 - probability) * np.log2(1 - probability)
    )


def entropy_score(target: np.ndarray, feature: np.ndarray) -> float:
    q = np.asarray(target, dtype=np.int8)
    predictor = np.asarray(feature, dtype=np.int8)
    target_entropy = binary_entropy(q)
    if target_entropy == 0:
        return 1.0
    conditional_entropy = 0.0
    for value in (0, 1):
        selected = q[predictor == value]
        if len(selected):
            conditional_entropy += len(selected) / len(q) * binary_entropy(selected)
    return float(1 - conditional_entropy / target_entropy)


def majorant_entropy_score(values: np.ndarray, relation: np.ndarray) -> float:
    return entropy_score(values, closure(values, relation))


def two_sided_min_score(values: np.ndarray, relation: np.ndarray) -> float:
    return min(
        entropy_score(values, closure(values, relation)),
        entropy_score(values, interior(values, relation)),
    )


def exact_monotonicity(values: np.ndarray, strict_relation: np.ndarray) -> bool:
    q = np.asarray(values, dtype=bool)
    return not np.any(strict_relation & q[:, None] & ~q[None, :])


def pairwise_preservation_score(
    values: np.ndarray, strict_relation: np.ndarray
) -> float:
    """Truth preservation conditional on a true source."""

    q = np.asarray(values, dtype=bool)
    eligible = strict_relation & q[:, None]
    if not eligible.any():
        return 1.0
    return float(1 - (eligible & ~q[None, :]).sum() / eligible.sum())


def unconditional_pair_score(values: np.ndarray, strict_relation: np.ndarray) -> float:
    q = np.asarray(values, dtype=bool)
    if not strict_relation.any():
        return 1.0
    violations = strict_relation & q[:, None] & ~q[None, :]
    return float(1 - violations.sum() / strict_relation.sum())


def closure_inflation_score(values: np.ndarray, relation: np.ndarray) -> float:
    q = np.asarray(values, dtype=bool)
    false_count = (~q).sum()
    if false_count == 0:
        return 1.0
    additions = (closure(q, relation) & ~q).sum()
    return float(1 - additions / false_count)


def closure_precision_score(values: np.ndarray, relation: np.ndarray) -> float:
    q = np.asarray(values, dtype=bool)
    completed = closure(q, relation)
    if not completed.any():
        return 1.0
    return float(q.sum() / completed.sum())


def nearest_monotone_edit(
    values: np.ndarray, immediate_relation: np.ndarray
) -> tuple[int, float]:
    """Return minimum flips and a minority-normalized similarity score.

    The minimum is solved as an s-t cut. A source-side node is assigned true;
    infinite-capacity order edges enforce upward closure.
    """

    q = np.asarray(values, dtype=bool)
    point_count = len(q)
    source = point_count
    sink = point_count + 1
    capacity = lil_matrix((point_count + 2, point_count + 2), dtype=np.int64)
    for index, is_true in enumerate(q):
        if is_true:
            capacity[source, index] = 1
        else:
            capacity[index, sink] = 1
    infinite = point_count + 1
    origins, targets = np.where(immediate_relation)
    capacity[origins, targets] = infinite
    edit_count = int(maximum_flow(capacity.tocsr(), source, sink).flow_value)
    minority_count = int(min(q.sum(), (~q).sum()))
    score = 1.0 if minority_count == 0 else 1 - edit_count / minority_count
    return edit_count, float(score)


def switch_simplicity_score(values: np.ndarray, chains: np.ndarray) -> float:
    """Score low switch count, without imposing a switch direction."""

    chain_values = np.asarray(values, dtype=np.int8)[chains]
    switches = np.count_nonzero(np.diff(chain_values, axis=1), axis=1)
    denominator = max(chains.shape[1] - 2, 1)
    excess = np.maximum(switches - 1, 0) / denominator
    return float(1 - excess.mean())


def chain_inversion_score(values: np.ndarray, chains: np.ndarray) -> float:
    chain_values = np.asarray(values, dtype=np.int8)[chains]
    inversions = np.zeros(len(chains), dtype=float)
    for left in range(chains.shape[1]):
        inversions += (
            chain_values[:, left, None] * (1 - chain_values[:, left + 1 :])
        ).sum(axis=1)
    maximum = (chains.shape[1] ** 2) // 4
    return float(1 - (inversions / maximum).mean())


def distance_robustness_curve(
    values: np.ndarray,
    strict_relation: np.ndarray,
    distances: np.ndarray,
    maximum_distance: int,
) -> dict[int, float]:
    q = np.asarray(values, dtype=bool)
    result = {}
    for distance in range(1, maximum_distance + 1):
        relation = strict_relation & (distances == distance)
        eligible = relation & q[:, None]
        result[distance] = (
            float("nan")
            if not eligible.any()
            else float(1 - (eligible & ~q[None, :]).sum() / eligible.sum())
        )
    return result


def equal_distance_robustness_score(
    values: np.ndarray,
    strict_relation: np.ndarray,
    distances: np.ndarray,
    maximum_distance: int,
) -> float:
    curve = distance_robustness_curve(
        values, strict_relation, distances, maximum_distance
    )
    values_at_distance = np.asarray(list(curve.values()), dtype=float)
    return float(np.nanmean(values_at_distance))


def derivative_sign_score(values: np.ndarray, edge_relation: np.ndarray) -> float:
    q = np.asarray(values, dtype=bool)
    favorable = edge_relation & ~q[:, None] & q[None, :]
    unfavorable = edge_relation & q[:, None] & ~q[None, :]
    changing = favorable.sum() + unfavorable.sum()
    if changing == 0:
        return 1.0
    return float(favorable.sum() / changing)


def context_preservation_profile(
    values: np.ndarray,
    strict_relation: np.ndarray,
    context_ids: np.ndarray,
) -> tuple[float, float, float]:
    scores = []
    covered = 0
    for context in np.unique(context_ids):
        mask = context_ids == context
        local_relation = strict_relation & mask[:, None] & mask[None, :]
        q = np.asarray(values, dtype=bool)
        eligible = local_relation & q[:, None]
        if eligible.any():
            covered += 1
            scores.append(float(1 - (eligible & ~q[None, :]).sum() / eligible.sum()))
    if not scores:
        return 1.0, 0.0, 0.0
    return (
        float(np.mean(scores)),
        float(np.std(scores)),
        covered / len(np.unique(context_ids)),
    )


def best_simple_threshold_accuracy(
    values: np.ndarray,
    universe: FiniteSetUniverse,
    direction: str,
) -> tuple[float, str]:
    q = np.asarray(values, dtype=bool)
    overlap = np.fromiter(
        (
            (int(a_bits) & int(b_bits)).bit_count()
            for a_bits, b_bits in zip(universe.a_bits, universe.b_bits)
        ),
        dtype=np.int8,
        count=universe.size,
    )
    a_size = np.fromiter(
        (int(value).bit_count() for value in universe.a_bits),
        dtype=np.int8,
        count=universe.size,
    )
    b_size = np.fromiter(
        (int(value).bit_count() for value in universe.b_bits),
        dtype=np.int8,
        count=universe.size,
    )
    a_minus_b = a_size - overlap
    b_minus_a = b_size - overlap
    if direction[0] == "R":
        features = {
            "overlap": overlap,
            "B size": b_size,
            "-(A-B) size": -a_minus_b,
            "(B-A) size": b_minus_a,
            "overlap - (A-B)": overlap - a_minus_b,
        }
    else:
        features = {
            "overlap": overlap,
            "A size": a_size,
            "-(B-A) size": -b_minus_a,
            "(A-B) size": a_minus_b,
            "overlap - (B-A)": overlap - b_minus_a,
        }
    if direction[1] == "D":
        features = {f"-({name})": -feature for name, feature in features.items()}

    best_accuracy = -1.0
    best_description = ""
    for name, feature in features.items():
        for threshold in range(int(feature.min()), int(feature.max()) + 2):
            accuracy = float(((feature >= threshold) == q).mean())
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_description = f"{name} >= {threshold}"
    return best_accuracy, best_description


def exception_code_score(edit_count: int, point_count: int) -> float:
    """Enumerative exception-code proxy, not a complete LoT/MDL measure."""

    if edit_count == 0:
        return 1.0
    possibilities = sum(comb(point_count, size) for size in range(edit_count + 1))
    return float(1 - log2(possibilities) / point_count)


def directional_profile(
    scores: Iterable[float],
) -> tuple[float, float, float]:
    values = np.asarray(tuple(scores), dtype=float)
    return (
        float(values.max()),
        float(values.mean()),
        float(values.max() - values.mean()),
    )


def cardinality_lattice_scores(
    truth_by_size: Iterable[bool],
) -> dict[str, float]:
    """Score a cardinality-only predicate on an exact Boolean subset lattice.

    ``truth_by_size[k]`` is the truth value for every set of cardinality ``k``.
    Combinatorial weights account for every subset without constructing the
    full pairwise relation matrix.
    """

    q = np.asarray(tuple(truth_by_size), dtype=bool)
    n = len(q) - 1
    weights = np.asarray([comb(n, size) for size in range(n + 1)], dtype=float)
    point_count = 2**n

    def weighted_entropy_score(feature: np.ndarray) -> float:
        target_probability = float(weights[q].sum() / point_count)
        if target_probability in (0.0, 1.0):
            return 1.0
        target_entropy = -target_probability * log2(target_probability) - (
            1 - target_probability
        ) * log2(1 - target_probability)
        conditional_entropy = 0.0
        for feature_value in (False, True):
            selected_weight = weights[feature == feature_value].sum()
            if selected_weight == 0:
                continue
            true_weight = weights[(feature == feature_value) & q].sum()
            probability = float(true_weight / selected_weight)
            entropy = (
                0.0
                if probability in (0.0, 1.0)
                else -probability * log2(probability)
                - (1 - probability) * log2(1 - probability)
            )
            conditional_entropy += selected_weight / point_count * entropy
        return float(1 - conditional_entropy / target_entropy)

    upward_closure = np.maximum.accumulate(q)
    upward_interior = np.minimum.accumulate(q[::-1])[::-1]
    majorant_entropy = weighted_entropy_score(upward_closure)
    two_sided_min = min(majorant_entropy, weighted_entropy_score(upward_interior))

    eligible_pairs = 0
    violating_pairs = 0
    eligible_edges = 0
    violating_edges = 0
    favorable_edges = 0
    for source_size in range(n + 1):
        source_count = comb(n, source_size)
        if q[source_size]:
            eligible_pairs += source_count * (2 ** (n - source_size) - 1)
            if source_size < n:
                edge_count = source_count * (n - source_size)
                eligible_edges += edge_count
                if not q[source_size + 1]:
                    violating_edges += edge_count
            for target_size in range(source_size + 1, n + 1):
                if not q[target_size]:
                    violating_pairs += source_count * comb(
                        n - source_size, target_size - source_size
                    )
        elif source_size < n and q[source_size + 1]:
            favorable_edges += source_count * (n - source_size)

    pairwise_preservation = (
        1.0 if eligible_pairs == 0 else 1 - violating_pairs / eligible_pairs
    )
    edge_preservation = (
        1.0 if eligible_edges == 0 else 1 - violating_edges / eligible_edges
    )
    all_strict_pairs = 3**n - 2**n
    unconditional_pair = (
        1.0 if all_strict_pairs == 0 else 1 - violating_pairs / all_strict_pairs
    )

    false_count = weights[~q].sum()
    additions = weights[upward_closure & ~q].sum()
    closure_inflation = 1.0 if false_count == 0 else 1 - additions / false_count
    closure_count = weights[upward_closure].sum()
    closure_precision = 1.0 if closure_count == 0 else weights[q].sum() / closure_count

    switches = int(np.count_nonzero(np.diff(q.astype(np.int8))))
    switch_simplicity = 1 - max(switches - 1, 0) / max(n - 1, 1)
    inversions = sum(
        q[left] and not q[right]
        for left in range(n + 1)
        for right in range(left + 1, n + 1)
    )
    max_inversions = max(((n + 1) ** 2) // 4, 1)
    chain_inversion = 1 - inversions / max_inversions

    distance_scores = []
    for distance in range(1, n + 1):
        eligible = 0
        violations = 0
        for source_size in range(n - distance + 1):
            if not q[source_size]:
                continue
            pair_count = comb(n, source_size) * comb(n - source_size, distance)
            eligible += pair_count
            if not q[source_size + distance]:
                violations += pair_count
        if eligible:
            distance_scores.append(1 - violations / eligible)
    equal_distance_robustness = (
        1.0 if not distance_scores else float(np.mean(distance_scores))
    )

    changing_edges = favorable_edges + violating_edges
    derivative_sign = 1.0 if changing_edges == 0 else favorable_edges / changing_edges

    threshold_repair_count = min(
        sum(comb(n, size) for size in range(n + 1) if q[size] != (size >= threshold))
        for threshold in range(n + 2)
    )
    minority_count = min(weights[q].sum(), weights[~q].sum())
    threshold_repair_score = (
        1.0 if minority_count == 0 else 1 - threshold_repair_count / minority_count
    )

    return {
        "majorant_entropy": majorant_entropy,
        "two_sided_min": two_sided_min,
        "pairwise_preservation": float(pairwise_preservation),
        "edge_preservation": float(edge_preservation),
        "unconditional_pair": float(unconditional_pair),
        "best_cardinality_threshold_repair": float(threshold_repair_score),
        "closure_inflation": float(closure_inflation),
        "closure_precision": float(closure_precision),
        "chain_switch": float(switch_simplicity),
        "chain_inversion": float(chain_inversion),
        "equal_distance_robustness": equal_distance_robustness,
        "derivative_sign": float(derivative_sign),
    }
