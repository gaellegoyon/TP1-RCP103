"""
Distribution exponentielle.
Génère des valeurs aléatoires selon une loi exponentielle de moyenne e = 1/λ.
"""

import numpy as np
from config.parametres import GRAINE, EXPONENTIELLE_MOYENNE

NOM    = "Exponentielle"
LABEL  = f"Exponentielle (moyenne={EXPONENTIELLE_MOYENNE})"
EST_DISCRETE = False


def generer(n: int) -> np.ndarray:
    """Génère n valeurs suivant une loi exponentielle de moyenne e (= 1/λ)."""
    generateur = np.random.default_rng(GRAINE)
    return generateur.exponential(
        scale = EXPONENTIELLE_MOYENNE,   # numpy attend scale = 1/λ = moyenne
        size  = n,
    )
