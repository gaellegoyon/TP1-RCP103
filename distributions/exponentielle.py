"""Distribution exponentielle."""

import numpy as np
from config.parametres import GRAINE, EXPONENTIELLE_MOYENNE

NOM    = "Exponentielle"
LABEL  = f"Exponentielle (moyenne={EXPONENTIELLE_MOYENNE})"
EST_DISCRETE = False


def generer(n: int) -> np.ndarray:
    rng = np.random.default_rng(GRAINE)
    return rng.exponential(
        scale = EXPONENTIELLE_MOYENNE,
        size  = n,
    )