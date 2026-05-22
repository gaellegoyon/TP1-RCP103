"""
Affichage des valeurs générées dans le terminal.
"""

import numpy as np
from config.parametres import VALEURS_N


def afficher_valeurs(nom: str, label: str, generateur_fn) -> None:
    """Affiche dans le terminal les valeurs générées pour chaque n."""

    print(f"\n{'═' * 60}")
    print(f"  {label}")
    print(f"{'═' * 60}")

    for n in VALEURS_N:
        valeurs = generateur_fn(n)
        print(f"\n  n = {n:>6,}  →  {valeurs}")
