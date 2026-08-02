import itertools

import numpy as np

from examples.learn_quant import alternative_monotonicity_metrics as metrics


def test_nearest_monotone_edit_matches_brute_force():
    universe = metrics.FiniteSetUniverse.create(1)
    for direction in metrics.DIRECTIONS:
        strict = universe.strict_relation(direction)
        edge = universe.relation(direction, immediate=True)
        monotone_candidates = []
        for candidate in itertools.product((0, 1), repeat=universe.size):
            values = np.asarray(candidate, dtype=np.int8)
            if metrics.exact_monotonicity(values, strict):
                monotone_candidates.append(values)

        for target in itertools.product((0, 1), repeat=universe.size):
            values = np.asarray(target, dtype=np.int8)
            expected = min(
                np.count_nonzero(values != candidate)
                for candidate in monotone_candidates
            )
            actual, _ = metrics.nearest_monotone_edit(values, edge)
            assert actual == expected


def test_exact_threshold_scores_one_in_its_direction():
    universe = metrics.FiniteSetUniverse.create(3)
    values = universe.truth_values(lambda _a, b: len(b) >= 2)
    relation = universe.relation("RU")
    strict = universe.strict_relation("RU")
    edge = universe.relation("RU", immediate=True)

    assert metrics.exact_monotonicity(values, strict)
    assert metrics.majorant_entropy_score(values, relation) == 1
    assert metrics.pairwise_preservation_score(values, strict) == 1
    assert metrics.nearest_monotone_edit(values, edge) == (0, 1)


def test_distance_curve_marks_unsupported_distances_undefined():
    universe = metrics.FiniteSetUniverse.create(3)
    values = universe.truth_values(lambda _a, b: len(b) >= 2)
    curve = metrics.distance_robustness_curve(
        values,
        universe.strict_relation("RU"),
        universe.movement_distance("RU"),
        universe.n,
    )

    assert curve[1] == 1
    assert np.isnan(curve[2])
    assert np.isnan(curve[3])


def test_switch_simplicity_is_not_directional():
    universe = metrics.FiniteSetUniverse.create(3)
    downward_values = universe.truth_values(lambda _a, b: len(b) <= 1)

    assert not metrics.exact_monotonicity(
        downward_values, universe.strict_relation("RU")
    )
    assert (
        metrics.switch_simplicity_score(downward_values, universe.maximal_chains("RU"))
        == 1
    )
    assert (
        metrics.chain_inversion_score(downward_values, universe.maximal_chains("RU"))
        == 0
    )


def test_simple_threshold_fits_familiar_monotone_features():
    universe = metrics.FiniteSetUniverse.create(3)
    examples = [
        (lambda a, b: a <= b, "RU"),
        (lambda a, b: bool(a & b), "RU"),
        (lambda a, b: len(a & b) > len(a - b), "RU"),
        (lambda _a, b: len(b) >= 2, "RU"),
    ]

    for predicate, direction in examples:
        accuracy, _ = metrics.best_simple_threshold_accuracy(
            universe.truth_values(predicate), universe, direction
        )
        assert accuracy == 1


def test_cardinality_compression_matches_explicit_parity_lattice():
    n = 4
    subset_bits = np.arange(1 << n, dtype=np.int64)
    cardinalities = np.asarray([int(value).bit_count() for value in subset_bits])
    values = cardinalities % 2 == 0
    relation = (subset_bits[:, None] & subset_bits[None, :]) == subset_bits[:, None]
    strict = relation & ~np.eye(len(values), dtype=bool)
    edge = relation & (cardinalities[None, :] == cardinalities[:, None] + 1)
    distance = cardinalities[None, :] - cardinalities[:, None]
    chain = np.asarray([[(1 << size) - 1 for size in range(n + 1)]])
    compressed = metrics.cardinality_lattice_scores(
        [size % 2 == 0 for size in range(n + 1)]
    )

    expected = {
        "majorant_entropy": metrics.majorant_entropy_score(values, relation),
        "two_sided_min": metrics.two_sided_min_score(values, relation),
        "pairwise_preservation": metrics.pairwise_preservation_score(values, strict),
        "edge_preservation": metrics.pairwise_preservation_score(values, edge),
        "unconditional_pair": metrics.unconditional_pair_score(values, strict),
        "closure_inflation": metrics.closure_inflation_score(values, relation),
        "closure_precision": metrics.closure_precision_score(values, relation),
        "chain_switch": metrics.switch_simplicity_score(values, chain),
        "chain_inversion": metrics.chain_inversion_score(values, chain),
        "equal_distance_robustness": metrics.equal_distance_robustness_score(
            values, strict, distance, n
        ),
        "derivative_sign": metrics.derivative_sign_score(values, edge),
    }
    for name, score in expected.items():
        assert compressed[name] == score


def test_parity_threshold_repair_matches_exact_edit_through_n6():
    for n in range(2, 7):
        subset_bits = np.arange(1 << n, dtype=np.int64)
        cardinalities = np.asarray([int(value).bit_count() for value in subset_bits])
        values = cardinalities % 2 == 0
        relation = (subset_bits[:, None] & subset_bits[None, :]) == subset_bits[:, None]
        edge = relation & (cardinalities[None, :] == cardinalities[:, None] + 1)
        _, exact_score = metrics.nearest_monotone_edit(values, edge)
        compressed = metrics.cardinality_lattice_scores(
            [size % 2 == 0 for size in range(n + 1)]
        )
        assert compressed["best_cardinality_threshold_repair"] == exact_score
