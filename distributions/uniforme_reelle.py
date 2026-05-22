"""
Distribution uniforme continue (réelle).
Génère des réels aléatoires entre un minimum et un maximum.
"""

import numpy as np
from config.parametres import GRAINE, UNIF_REELLE_MIN, UNIF_REELLE_MAX

NOM    = "Uniforme réelle"
LABEL  = f"Unif. réelle [{UNIF_REELLE_MIN}, {UNIF_REELLE_MAX}]"
EST_DISCRETE = False


def generer(n: int) -> np.ndarray:
    """Génère n réels suivant une loi uniforme continue sur [MIN, MAX)."""
    generateur = np.random.default_rng(GRAINE)
    return generateur.uniform(
        low  = UNIF_REELLE_MIN,
        high = UNIF_REELLE_MAX,
        size = n,
    )
