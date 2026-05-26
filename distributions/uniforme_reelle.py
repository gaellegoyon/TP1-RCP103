"""Distribution uniforme continue."""

import numpy as np
from config.parametres import GRAINE, UNIF_REELLE_MIN, UNIF_REELLE_MAX

NOM    = "Uniforme réelle"
LABEL  = f"Unif. réelle [{UNIF_REELLE_MIN}, {UNIF_REELLE_MAX}]"
EST_DISCRETE = False


def generer(n: int) -> np.ndarray:
    rng = np.random.default_rng(GRAINE)
    return rng.uniform(
        low  = UNIF_REELLE_MIN,
        high = UNIF_REELLE_MAX,
        size = n,
    )