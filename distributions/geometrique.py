"""Distribution géométrique."""

import numpy as np
from config.parametres import GRAINE, GEOMETRIQUE_P

NOM    = "Géométrique"
LABEL  = f"Géométrique (p={GEOMETRIQUE_P})"
EST_DISCRETE = True


def generer(n: int) -> np.ndarray:
    rng = np.random.default_rng(GRAINE)
    return rng.geometric(
        p    = GEOMETRIQUE_P,
        size = n,
    )