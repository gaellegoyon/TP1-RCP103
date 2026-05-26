"""Distribution normale."""

import numpy as np
from config.parametres import GRAINE, NORMALE_MOYENNE, NORMALE_ECART_TYPE

NOM    = "Normale"
LABEL  = f"Normale (µ={NORMALE_MOYENNE}, σ={NORMALE_ECART_TYPE})"
EST_DISCRETE = False


def generer(n: int) -> np.ndarray:
    rng = np.random.default_rng(GRAINE)
    return rng.normal(
        loc   = NORMALE_MOYENNE,
        scale = NORMALE_ECART_TYPE,
        size  = n,
    )