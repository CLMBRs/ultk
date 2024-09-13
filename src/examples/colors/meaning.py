import numpy as np

from ultk.language.semantics import Universe

color_universe = Universe.from_csv("colors/outputs/color_universe.csv")

# TODO: do this once and serialize?
SIGMA_SQUARED = 64
# shape (330, 3): L*a*b* values for each Munsell chip
cielab_points = np.array([(ref.L, ref.a, ref.b) for ref in color_universe.referents])


def meaning_distance(
    center: np.ndarray, other_point: np.ndarray, sigma_squared: float = 64.0
) -> float:
    """Calculate the distance between two points in CIELAB space.

    Args:
        center: the first point (e.g. (L, a, b) for a Munsell chip)
        other_point: the second point
        sigma_squared: the variance of the Gaussian kernel

    Returns:
        exp(-||center - other_point||^2 / (2 * sigma_squared))
    """
    return np.exp(np.linalg.norm(center - other_point) ** 2 / (2 * sigma_squared))


# shape: (330, 330)
# meaning_distributions[i, j] = meaning_distance(cielab_points[i], cielab_points[j])
# this is p(u | m), or m_c(u) in the paper
meaning_distributions = np.array(
    [
        [meaning_distance(center, other_point) for other_point in cielab_points]
        for center in cielab_points
    ]
)
# normalize each row into a distribution
meaning_distributions /= meaning_distributions.sum(axis=1, keepdims=True)
