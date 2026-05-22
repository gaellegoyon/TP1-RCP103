"""
generators.py
=============

Generateurs pseudo-aleatoires "maison" pour la partie 1 du TP.

On a implemente deux familles :

1. Le generateur congruentiel lineaire (LCG) :
       X_{n+1} = (a * X_n + c) mod m

2. Le generateur multiplicatif (cas particulier c = 0, dit "de Lehmer") :
       X_{n+1} = a * X_n mod m

A partir des entiers X_n on fait U_n = X_n / m pour avoir des reels
entre 0 et 1.

Pour notre groupe (le groupe 2), on utilise toujours seed = 2.

TP1 RCP103 - Groupe 2
"""

from dataclasses import dataclass


@dataclass
class LCG:
    """Generateur congruentiel : X_{n+1} = (a * X_n + c) mod m.

    Si c = 0, c'est un generateur multiplicatif (Lehmer).
    """

    a: int          # multiplicateur
    c: int          # increment
    m: int          # module
    state: int      # valeur courante (au debut : la graine)
    name: str = "LCG"

    def next_int(self):
        """Avance d'un pas et renvoie X_{n+1}."""
        self.state = (self.a * self.state + self.c) % self.m
        return self.state

    def next_uniform(self):
        """Renvoie U_n = X_n / m, donc un reel sur [0, 1)."""
        return self.next_int() / self.m

    def sample_ints(self, n):
        """Genere n entiers a la suite."""
        return [self.next_int() for _ in range(n)]

    def sample_uniforms(self, n):
        """Genere n reels uniformes a la suite."""
        return [self.next_uniform() for _ in range(n)]

    def reset(self, seed):
        """Remet le generateur dans son etat initial avec une nouvelle graine."""
        self.state = seed


# -----------------------------------------------------------------------------
# Generateurs preconfigures - bons exemples
# -----------------------------------------------------------------------------

def make_minstd(seed=2):
    """MINSTD (Park-Miller) : a = 16807, c = 0, m = 2^31 - 1.

    C'est un generateur classique souvent utilise comme reference.
    """
    return LCG(a=16807, c=0, m=2**31 - 1, state=seed,
               name="MINSTD (Park-Miller)")


def make_numerical_recipes(seed=2):
    """Generateur de Numerical Recipes (Press et al.) :
    a = 1664525, c = 1013904223, m = 2^32.
    Periode maximale (= m).
    """
    return LCG(a=1664525, c=1013904223, m=2**32, state=seed,
               name="Numerical Recipes")


# -----------------------------------------------------------------------------
# Generateurs preconfigures - mauvais exemples (pour illustrer les defauts)
# -----------------------------------------------------------------------------

def make_randu(seed=2):
    """RANDU : a = 65539, c = 0, m = 2^31. Le defaut historique d'IBM.

    Il a l'air OK en 1D et 2D, mais en 3D les triplets tombent sur
    seulement 15 plans. On l'utilise pour illustrer le probleme.
    """
    # RANDU doit demarrer avec une graine impaire
    s = seed if seed % 2 == 1 else seed + 1
    return LCG(a=65539, c=0, m=2**31, state=s, name="RANDU (IBM)")


def make_bad_short_period(seed=2):
    """Petit LCG pedagogique : a = 5, c = 3, m = 16.

    Periode maximale de 16. C'est minuscule, mais ca permet de
    visualiser le cyclage.
    """
    return LCG(a=5, c=3, m=16, state=seed, name="LCG court (a=5, c=3, m=16)")


def make_bad_multiplicative(seed=2):
    """Petit multiplicatif : a = 6, c = 0, m = 13. Periode 12 au mieux."""
    # Si seed = 0 on aurait une suite nulle, on protege.
    s = seed if seed != 0 else 1
    return LCG(a=6, c=0, m=13, state=s, name="Multiplicatif court (a=6, m=13)")


# -----------------------------------------------------------------------------
# Detection de la periode (utile pour les petits generateurs)
# -----------------------------------------------------------------------------

def detect_period(gen, max_steps=100_000):
    """Cherche la periode du generateur en attendant la 1re repetition.

    Renvoie (periode, sequence visitee). Si on ne trouve pas avant
    max_steps, renvoie (-1, sequence).
    """
    initial = gen.state
    seen = {initial: 0}
    sequence = [initial]
    for i in range(1, max_steps + 1):
        x = gen.next_int()
        if x in seen:
            # Une valeur deja vue : on a boucle.
            period = i - seen[x]
            sequence.append(x)
            return period, sequence
        seen[x] = i
        sequence.append(x)
    return -1, sequence


# -----------------------------------------------------------------------------
# Petit test rapide quand on lance directement le fichier
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Test rapide des generateurs avec seed = 2\n")
    for builder in (make_minstd, make_numerical_recipes,
                    make_bad_short_period, make_bad_multiplicative):
        g = builder(seed=2)
        print(f"{g.name:35s} -> 10 premiers : {g.sample_ints(10)}")
