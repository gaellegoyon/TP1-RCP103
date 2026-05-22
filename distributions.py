"""
RCP103 - TP Génération de variables aléatoires
Groupe 2 - Paramètres:
  - Uniforme entière  : (10, 30)
  - Uniforme réelle   : (1.0, 3.0)
  - Normale           : (µ=0, σ=0.75)
  - Exponentielle     : moyenne e = 2  (λ = 0.5)
  - Géométrique       : p = 0.3
  - Seed              : 2
"""

import numpy as np


SEED = 2

# ── Groupe 2 parameters ────────────────────────────────────────────────────────
UNIF_INT_MIN,  UNIF_INT_MAX  = 10, 30
UNIF_REAL_MIN, UNIF_REAL_MAX = 1.0, 3.0
NORMAL_MU,     NORMAL_SIGMA  = 0, 0.75
EXP_MEAN                     = 2          # e = 1/λ  →  λ = 0.5
GEOM_P                       = 0.3
# ──────────────────────────────────────────────────────────────────────────────


def make_rng() -> np.random.Generator:
    """Return a fresh Generator seeded with the group seed."""
    return np.random.default_rng(SEED)


# ── Individual generators ──────────────────────────────────────────────────────

def generate_uniform_int(n: int) -> np.ndarray:
    """Uniform discrete integer in [UNIF_INT_MIN, UNIF_INT_MAX] (inclusive)."""
    rng = make_rng()
    return rng.integers(low=UNIF_INT_MIN, high=UNIF_INT_MAX + 1, size=n)


def generate_uniform_real(n: int) -> np.ndarray:
    """Uniform continuous real in [UNIF_REAL_MIN, UNIF_REAL_MAX)."""
    rng = make_rng()
    return rng.uniform(low=UNIF_REAL_MIN, high=UNIF_REAL_MAX, size=n)


def generate_normal(n: int) -> np.ndarray:
    """Normal distribution N(NORMAL_MU, NORMAL_SIGMA)."""
    rng = make_rng()
    return rng.normal(loc=NORMAL_MU, scale=NORMAL_SIGMA, size=n)


def generate_exponential(n: int) -> np.ndarray:
    """Exponential distribution with mean EXP_MEAN (scale = 1/λ = mean)."""
    rng = make_rng()
    return rng.exponential(scale=EXP_MEAN, size=n)


def generate_geometric(n: int) -> np.ndarray:
    """Geometric distribution with success probability GEOM_P.

    numpy's geometric gives the number of trials until the first success
    (support: 1, 2, 3, …).
    """
    rng = make_rng()
    return rng.geometric(p=GEOM_P, size=n)


# ── Registry ──────────────────────────────────────────────────────────────────

DISTRIBUTIONS: dict[str, dict] = {
    "Uniforme entière": {
        "generator": generate_uniform_int,
        "label": f"Unif. entière [{UNIF_INT_MIN}, {UNIF_INT_MAX}]",
        "discrete": True,
    },
    "Uniforme réelle": {
        "generator": generate_uniform_real,
        "label": f"Unif. réelle [{UNIF_REAL_MIN}, {UNIF_REAL_MAX}]",
        "discrete": False,
    },
    "Normale": {
        "generator": generate_normal,
        "label": f"Normale (µ={NORMAL_MU}, σ={NORMAL_SIGMA})",
        "discrete": False,
    },
    "Exponentielle": {
        "generator": generate_exponential,
        "label": f"Exponentielle (moyenne={EXP_MEAN})",
        "discrete": False,
    },
    "Géométrique": {
        "generator": generate_geometric,
        "label": f"Géométrique (p={GEOM_P})",
        "discrete": True,
    },
}


# ── Affichage des valeurs ──────────────────────────────────────────────────────

if __name__ == "__main__":
    N_VALUES = [10, 100, 1_000, 10_000]

    for dist_name, dist_info in DISTRIBUTIONS.items():
        print(f"\n{'═' * 60}")
        print(f"  {dist_info['label']}")
        print(f"{'═' * 60}")

        for n in N_VALUES:
            data = dist_info["generator"](n)
            print(f"\n  n = {n:>6,}  →  {data}")

