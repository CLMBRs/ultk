import numpy as np
import pickle

from ultk.effcomm import ib
from ultk.language.semantics import Universe
from examples.colors.meaning import meaning_distributions, cielab_points

if __name__ == "__main__":

    # shape: `(330, 330)`
    pU_M = meaning_distributions

    # pM = np.full(pU_M.shape[0], 1/pU_M.shape[0])
    pM = np.load("src/examples/colors/data/zkrt18_prior.npy").squeeze()

    model = ib.get_ib_naming_model(
        pU_M,
        pM,
        # add custom beta values here
        betas=np.logspace(
            0,
            5,
            1600,
        ),
    )

    # write model
    fn = "src/examples/colors/outputs/naming_model.pkl"
    with open(fn, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote a pickle binary to {fn}.")
