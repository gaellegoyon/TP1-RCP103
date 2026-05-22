"""
generators.py
=============

Generateurs de nombres pseudo-aleatoires pour le TP RCP103 (Groupe 2).

Ce module implemente deux familles de generateurs congruentiels :

1. Le generateur congruentiel multiplicatif (Lehmer) :
       X_{n+1} = (a * X_n) mod m

2. Le generateur congruentiel lineaire (LCG) :
       X_{n+1} = (a * X_n + c) mod m

Les sorties U_n = X_n / m sont des reels approximativement uniformes sur [0, 1).

Pour le groupe 2 la seed est fixee a 2 (X_0 = 2).

Auteur : Groupe 2 - RCP103 - CNAM
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple


# -----------------------------------------------------------------------------
# Classes de generateurs
# -----------------------------------------------------------------------------

@dataclass
class LCG:
    """Generateur congruentiel lineaire : X_{n+1} = (a*X_n + c) mod m.

    Si c = 0 on retombe sur le generateur multiplicatif (Lehmer).
    L'attribut state contient la valeur courante X_n.
    """

    a: int          # multiplicateur
    c: int          # increment
    m: int          # module
    state: int      # graine X_0 puis etat courant
    name: str = "LCG"

    def next_int(self) -> int:
        """Avance d'un pas et retourne le nouvel entier X_{n+1}."""
        self.state = (self.a * self.state + self.c) % self.m
        return self.state

    def next_uniform(self) -> float:
        """Retourne U = X / m, valeur reelle sur [0, 1)."""
        return self.next_int() / self.m

    def sample_ints(self, n: int) -> List[int]:
        """Genere n entiers consecutifs."""
        return [self.next_int() for _ in range(n)]

    def sample_uniforms(self, n: int) -> List[float]:
        """Genere n reels uniformes consecutifs sur [0, 1)."""
        return [self.next_uniform() for _ in range(n)]

    def reset(self, seed: int) -> None:
        """Reinitialise l'etat avec une nouvelle graine."""
        self.state = seed


# -----------------------------------------------------------------------------
# Generateurs preconfigures
# -----------------------------------------------------------------------------

def make_minstd(seed: int = 2) -> LCG:
    """Generateur multiplicatif Park-Miller (MINSTD).

    Parametres : a = 16807, c = 0, m = 2^31 - 1.
    C'est le generateur de reference pour tester les LCG.
    Periode maximale theorique : m - 1 = 2 147 483 646.
    """
    return LCG(a=16807, c=0, m=2**31 - 1, state=seed, name="MINSTD (Park-Miller)")


def make_numerical_recipes(seed: int = 2) -> LCG:
    """Generateur lineaire de Numerical Recipes (Press et al.).

    Parametres : a = 1664525, c = 1013904223, m = 2^32.
    Periode maximale = m = 2^32 ; periode complete grace au theoreme de Hull-Dobell.
    """
    return LCG(a=1664525, c=1013904223, m=2**32, state=seed,
               name="Numerical Recipes")


def make_randu(seed: int = 2) -> LCG:
    """Generateur RANDU d'IBM (1960).

    Parametres : a = 65539, c = 0, m = 2^31. Tristement celebre :
    les triplets (U_n, U_{n+1}, U_{n+2}) tombent dans 15 hyperplans
    de R^3, ce qui le rend impropre aux simulations.
    On l'inclut pour illustrer un mauvais generateur.
    """
    # RANDU exige une graine impaire pour fonctionner ; on force la parite si besoin.
    s = seed if seed % 2 == 1 else seed + 1
    return LCG(a=65539, c=0, m=2**31, state=s, name="RANDU (IBM)")


def make_bad_short_period(seed: int = 2) -> LCG:
    """Mauvais LCG a tres courte periode (parametres pedagogiques).

    Parametres : a = 5, c = 3, m = 16. Periode maximale = 16. Sert a
    visualiser concretement le phenomene de cycle.
    """
    return LCG(a=5, c=3, m=16, state=seed, name="LCG court (a=5, c=3, m=16)")


def make_bad_multiplicative(seed: int = 2) -> LCG:
    """Mauvais generateur multiplicatif a courte periode.

    Parametres : a = 6, c = 0, m = 13. Sert a illustrer une situation ou
    le choix de a et m donne une periode inferieure a m-1.
    """
    # X_0 = 0 produirait une suite nulle ; on protege.
    s = seed if seed != 0 else 1
    return LCG(a=6, c=0, m=13, state=s, name="Multiplicatif court (a=6, m=13)")


# -----------------------------------------------------------------------------
# Outils d'analyse
# -----------------------------------------------------------------------------

def detect_period(gen: LCG, max_steps: int = 100_000) -> Tuple[int, List[int]]:
    """Detecte la periode d'un LCG en cherchant le premier retour a l'etat initial.

    Retourne (periode, sequence_visitee). Si aucun cycle n'est detecte avant
    max_steps, retourne (-1, sequence).
    """
    initial = gen.state
    seen = {initial: 0}
    sequence: List[int] = [initial]
    for i in range(1, max_steps + 1):
        x = gen.next_int()
        if x in seen:
            # On a recroise un etat deja rencontre : periode du cycle final
            period = i - seen[x]
            sequence.append(x)
            return period, sequence
        seen[x] = i
        sequence.append(x)
    return -1, sequence


def pairs(values: List[float]) -> Tuple[List[float], List[float]]:
    """Retourne (U_n, U_{n+1}) pour test visuel de Marsaglia."""
    return values[:-1], values[1:]


# -----------------------------------------------------------------------------
# Programme principal de demonstration (lance directement avec python generators.py)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Demonstration rapide des generateurs (seed = 2) ===\n")

    for builder in (make_minstd, make_numerical_recipes,
                    make_bad_short_period, make_bad_multiplicative):
        g = builder(seed=2)
        ints = g.sample_ints(10)
        print(f"{g.name:35s} -> 10 premiers entiers : {ints}")

    print("\n--- Detection de periode (sur petits generateurs) ---")
    for builder in (make_bad_short_period, make_bad_multiplicative):
        g = builder(seed=2)
        period, seq = detect_period(g, max_steps=200)
        print(f"{g.name:35s} -> periode = {period}, debut sequence = {seq[:20]}")
