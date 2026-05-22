"""
distributions.py
================

Generation des differentes lois a partir de l'uniforme U sur [0, 1).

On utilise la classe numpy.random.Generator pour obtenir U,
mais on code "a la main" la transformation U -> loi voulue
(plutot que d'appeler numpy.random.exponential par exemple).
L'idee est de montrer comment ca marche.

Toutes les fonctions prennent :
  - n   : le nombre de valeurs a generer
  - les parametres de la loi
  - rng : un numpy.random.Generator (avec seed=2 pour notre groupe)

TP1 RCP103 - Groupe 2 (seed = 2)
"""

import math
import numpy as np


# =============================================================================
# Lois discretes
# =============================================================================

def uniforme_discrete(n, a, b, rng):
    """Loi uniforme discrete sur {a, a+1, ..., b}.

    On tire U sur [0,1) puis on calcule a + floor((b - a + 1) * U).
    Chaque entier de a a b a la meme probabilite 1/(b-a+1).
    """
    u = rng.random(n)
    return a + np.floor((b - a + 1) * u).astype(int)


def bernoulli(n, p, rng):
    """Loi de Bernoulli : X = 1 si U < p, sinon X = 0."""
    u = rng.random(n)
    return (u < p).astype(int)


def geometrique(n, p, rng):
    """Loi geometrique de parametre p (support 1, 2, 3, ...).

    On utilise la transformee inverse :
        X = ceil( ln(1 - U) / ln(1 - p) )

    Cette loi compte le nombre d'essais avant le premier succes
    dans une suite de Bernoulli(p).
    E[X] = 1/p.
    """
    u = rng.random(n)
    # log(1 - p) est negatif et log(1 - U) aussi -> resultat positif.
    return np.ceil(np.log(1.0 - u) / np.log(1.0 - p)).astype(int)


def poisson(n, lam, rng):
    """Loi de Poisson de parametre lambda (algorithme de Knuth).

    On tire des U_i un par un et on multiplie. Des que le produit
    passe sous exp(-lambda), on s'arrete et X est le nombre de tirages
    qu'on a fait (moins 1).
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
# Lois continues
# =============================================================================

def uniforme_continue(n, a, b, rng):
    """Loi uniforme continue sur [a, b]. Formule : X = a + (b - a) * U."""
    u = rng.random(n)
    return a + (b - a) * u


def exponentielle(n, mean, rng):
    """Loi exponentielle de moyenne `mean` (donc lambda = 1/mean).

    Transformee inverse :
        F(x) = 1 - exp(-x/mean)  =>  X = -mean * ln(1 - U)
    On peut remplacer 1 - U par U car les deux sont uniformes,
    d'ou X = -mean * ln(U).
    """
    u = rng.random(n)
    # On evite log(0) si jamais U vaut 0 (tres improbable, mais bon).
    u = np.clip(u, 1e-12, 1.0)
    return -mean * np.log(u)


def normale(n, mu, sigma, rng):
    """Loi normale N(mu, sigma) generee avec Box-Muller.

    A partir de deux uniformes U1, U2 independantes :
        Z1 = sqrt(-2 * ln U1) * cos(2 * pi * U2)
        Z2 = sqrt(-2 * ln U1) * sin(2 * pi * U2)
    Z1 et Z2 sont N(0, 1) independants.
    Ensuite on retourne mu + sigma * Z.
    """
    # On tire des paires pour utiliser Z1 et Z2 en meme temps.
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
# Statistiques theoriques (utilisees pour comparer avec la simulation)
# =============================================================================

def stats_theoriques(distribution, **params):
    """Renvoie (moyenne, ecart-type) theoriques pour controler nos resultats."""
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
        # Particularite : pour l'exponentielle, ecart-type = moyenne.
        return mean, mean
    if d == "normale":
        return params["mu"], params["sigma"]
    if d == "geometrique":
        p = params["p"]
        return 1.0 / p, math.sqrt((1 - p) / p ** 2)
    if d == "bernoulli":
        p = params["p"]
        return p, math.sqrt(p * (1 - p))
    raise ValueError(f"distribution inconnue : {d}")
