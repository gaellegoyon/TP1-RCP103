"""Distribution uniforme discrète."""

import numpy as np
from config.parametres import GRAINE, UNIF_ENTIERE_MIN, UNIF_ENTIERE_MAX

NOM    = "Uniforme entière"
LABEL  = f"Unif. entière [{UNIF_ENTIERE_MIN}, {UNIF_ENTIERE_MAX}]"
EST_DISCRETE = True


def generer(n: int) -> np.ndarray:
    rng = np.random.default_rng(GRAINE)
    return rng.integers(
        low  = UNIF_ENTIERE_MIN,
        high = UNIF_ENTIERE_MAX + 1,
        size = n,
    )