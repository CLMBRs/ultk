import numpy as np

from game import SignalingGame
from rdot.distortions import hamming
from rdot.ba import RateDistortionOptimizer


# Generate a hamming bound
def generate_hamming_bound(sg: SignalingGame) -> list[tuple[float]]:
    """Given an atomic signaling game, return a list of (rate, distortion) pairs corresponding to the rate distortion bound on the game."""
    state_arr = np.arange(len(sg.states))
    dist_mat = hamming(x=state_arr, y=state_arr)

    optimizer = RateDistortionOptimizer(
        px=sg.prior, dist_mat=dist_mat, betas=np.logspace(-2, 10, 100)
    )
    return [
        (result.rate, result.distortion)
        for result in optimizer.get_results()
        if result is not None
    ]
