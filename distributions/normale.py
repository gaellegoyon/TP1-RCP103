"""
Distribution normale (gaussienne).
Génère des valeurs aléatoires selon une loi N(µ, σ).
"""

import numpy as np
from config.parametres import GRAINE, NORMALE_MOYENNE, NORMALE_ECART_TYPE

NOM    = "Normale"
LABEL  = f"Normale (µ={NORMALE_MOYENNE}, σ={NORMALE_ECART_TYPE})"
EST_DISCRETE = False


def generer(n: int) -> np.ndarray:
    """Génère n valeurs suivant une loi normale N(µ, σ)."""
    generateur = np.random.default_rng(GRAINE)
    return generateur.normal(
        loc   = NORMALE_MOYENNE,
        scale = NORMALE_ECART_TYPE,
        size  = n,
    )
