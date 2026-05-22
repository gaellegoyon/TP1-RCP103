"""
distributions.py
================

Generation de variables aleatoires a partir de l'uniforme U ~ U([0, 1)).

Toutes les fonctions de ce module prennent un nombre `n` d'echantillons
et un parametre `rng` (un objet numpy.random.Generator). Cela permet de
fixer la graine une seule fois (seed = 2 pour le groupe 2) et d'avoir
des resultats reproductibles.

Les algorithmes sont implementes "a la main" (transformee inverse,
Box-Muller, etc.) plutot que d'appeler numpy.random.exponential ou
numpy.random.normal directement, afin de bien montrer la mecanique.
La seule fonction de numpy utilisee en interne est `rng.random()` qui
fournit l'uniforme de base.

Auteur : Groupe 2 - RCP103 - CNAM
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


# =============================================================================
# 1. Lois discretes
# =============================================================================

def uniforme_discrete(n: int, a: int, b: int, rng: np.random.Generator) -> np.ndarray:
    """Loi uniforme discrete sur {a, a+1, ..., b}.

    Algorithme : on tire U ~ U([0, 1)) puis X = a + floor((b - a + 1) * U).
    P(X = k) = 1 / (b - a + 1) pour k = a, ..., b.
    """
    u = rng.random(n)
    return a + np.floor((b - a + 1) * u).astype(int)


def bernoulli(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Loi de Bernoulli : X = 1 avec probabilite p, 0 sinon.

    Algorithme : X = 1 si U < p, sinon 0.
    """
    u = rng.random(n)
    return (u < p).astype(int)


def geometrique(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Loi geometrique de parametre p (support {1, 2, 3, ...}).

    Modelise le nombre d'essais Bernoulli(p) jusqu'a obtenir le premier succes.
    P(X = k) = (1 - p)^(k-1) * p, E[X] = 1/p.

    Algorithme par transformee inverse :
        Si U ~ U(0, 1), alors X = ceil(ln(1 - U) / ln(1 - p)) suit la geometrique.
        On peut remplacer (1 - U) par U car les deux sont uniformes.
    """
    u = rng.random(n)
    # ceil pour eviter X = 0 et obtenir le support {1, 2, ...}.
    # log(1 - p) < 0 donc on divise un nombre <= 0 par un nombre < 0 : resultat >= 0.
    return np.ceil(np.log(1.0 - u) / np.log(1.0 - p)).astype(int)


def poisson(n: int, lam: float, rng: np.random.Generator) -> np.ndarray:
    """Loi de Poisson de parametre lambda (algorithme de Knuth).

    On tire des U_i iid jusqu'a ce que leur produit passe sous e^{-lambda} ;
    X = nombre de U_i tires moins 1.
    """
    L = math.exp(-lam)
    out = np.empty(n, dtype=int)
    for i in range(n):
        k = 0
        prod = 1.0
        while True:
            prod *= rng.random()
            if prod <= L:
                break
            k += 1
        out[i] = k
    return out


# =============================================================================
# 2. Lois continues
# =============================================================================

def uniforme_continue(n: int, a: float, b: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Loi uniforme continue sur [a, b]. Algorithme : X = a + (b - a) U."""
    u = rng.random(n)
    return a + (b - a) * u


def exponentielle(n: int, mean: float, rng: np.random.Generator) -> np.ndarray:
    """Loi exponentielle de moyenne `mean` (taux lambda = 1 / mean).

    Algorithme par transformee inverse :
        F(x) = 1 - exp(-lambda * x)  =>  X = -ln(1 - U) / lambda
        On utilise X = -mean * ln(U) (equivalent car U et 1 - U sont uniformes).
    """
    u = rng.random(n)
    # On evite log(0) en bornant U tres legerement au dessus de 0.
    u = np.clip(u, 1e-12, 1.0)
    return -mean * np.log(u)


def normale(n: int, mu: float, sigma: float,
            rng: np.random.Generator) -> np.ndarray:
    """Loi normale N(mu, sigma) generee par Box-Muller (forme cartesienne).

    Si U1, U2 ~ U(0, 1) iid alors
        Z1 = sqrt(-2 ln U1) cos(2 pi U2)
        Z2 = sqrt(-2 ln U1) sin(2 pi U2)
    sont N(0, 1) independants. On retourne ensuite mu + sigma * Z.
    """
    # On tire des paires pour utiliser les deux sorties Z1 et Z2.
    n_pairs = (n + 1) // 2
    u1 = rng.random(n_pairs)
    u2 = rng.random(n_pairs)
    u1 = np.clip(u1, 1e-12, 1.0)

    r = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2
    z1 = r * np.cos(theta)
    z2 = r * np.sin(theta)

    z = np.concatenate([z1, z2])[:n]
    return mu + sigma * z


# =============================================================================
# 3. Statistiques theoriques (pour comparaison)
# =============================================================================

def stats_theoriques(distribution: str, **params) -> Tuple[float, float]:
    """Retourne (moyenne, ecart-type) theoriques pour controle.

    distribution :
      - 'uniforme_discrete' : params a, b
      - 'uniforme_continue' : params a, b
      - 'exponentielle'      : params mean
      - 'normale'            : params mu, sigma
      - 'geometrique'        : params p
      - 'bernoulli'          : params p
    """
    d = distribution
    if d == "uniforme_discrete":
        a, b = params["a"], params["b"]
        n = b - a + 1
        mean = (a + b) / 2
        var = (n ** 2 - 1) / 12
        return mean, math.sqrt(var)
    if d == "uniforme_continue":
        a, b = params["a"], params["b"]
        mean = (a + b) / 2
        var = (b - a) ** 2 / 12
        return mean, math.sqrt(var)
    if d == "exponentielle":
        mean = params["mean"]
        return mean, mean  # sigma = mean pour l'exponentielle
    if d == "normale":
        return params["mu"], params["sigma"]
    if d == "geometrique":
        p = params["p"]
        return 1.0 / p, math.sqrt((1 - p) / p ** 2)
    if d == "bernoulli":
        p = params["p"]
        return p, math.sqrt(p * (1 - p))
    raise ValueError(f"distribution inconnue : {d}")
