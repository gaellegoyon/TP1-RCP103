"""Affichage des valeurs générées dans le terminal."""

import numpy as np


def afficher_valeurs(nom: str, label: str, generateur_fn, valeurs_n: list) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {label}")
    print(f"{'═' * 60}")

    for n in valeurs_n:
        valeurs = generateur_fn(n)
        print(f"\n  n = {n:>7,}  →  {valeurs}")