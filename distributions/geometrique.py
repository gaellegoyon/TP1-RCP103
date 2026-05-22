"""
Distribution géométrique.
Génère des valeurs aléatoires représentant le nombre d'essais
jusqu'au premier succès, avec une probabilité de succès p.
"""

import numpy as np
from config.parametres import GRAINE, GEOMETRIQUE_P

NOM    = "Géométrique"
LABEL  = f"Géométrique (p={GEOMETRIQUE_P})"
EST_DISCRETE = True


def generer(n: int) -> np.ndarray:
    """Génère n valeurs suivant une loi géométrique de paramètre p.

    Les valeurs représentent le nombre d'essais jusqu'au premier succès
    (support : 1, 2, 3, …).
    """
    generateur = np.random.default_rng(GRAINE)
    return generateur.geometric(
        p    = GEOMETRIQUE_P,
        size = n,
    )
