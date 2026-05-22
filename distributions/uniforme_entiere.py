"""
Distribution uniforme discrète (entière).
Génère des entiers aléatoires entre un minimum et un maximum inclus.
"""

import numpy as np
from config.parametres import GRAINE, UNIF_ENTIERE_MIN, UNIF_ENTIERE_MAX

NOM    = "Uniforme entière"
LABEL  = f"Unif. entière [{UNIF_ENTIERE_MIN}, {UNIF_ENTIERE_MAX}]"
EST_DISCRETE = True


def generer(n: int) -> np.ndarray:
    """Génère n entiers suivant une loi uniforme discrète sur [MIN, MAX]."""
    generateur = np.random.default_rng(GRAINE)
    return generateur.integers(
        low  = UNIF_ENTIERE_MIN,
        high = UNIF_ENTIERE_MAX + 1,   # +1 car la borne haute est exclue
        size = n,
    )
