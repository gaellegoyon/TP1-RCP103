"""
main.py
=======

Programme principal du TP - RCP103 Groupe 2 (seed = 2).

Lance :

  python3 main.py

Produit :
  - data/*.txt           : suites de valeurs generees (n = 10, 100, 1000, 10000)
  - rapport/figures/*    : figures pour le rapport
  - console              : suites courtes + tableaux de statistiques

Parametres du groupe 2 :
  - Uniforme discrete  : (10, 30)
  - Uniforme continue  : (1.0, 3.0)
  - Normale            : N(mu = 0, sigma = 0.75)
  - Exponentielle      : moyenne e = 2 (lambda = 1/2)
  - Loi additionnelle  : geometrique avec p = 0.3
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np

import distributions as dist
import generators as gen
import plots


# =============================================================================
# Configuration du groupe 2
# =============================================================================

GROUP = 2
SEED = 2                # consigne : seed = numero de groupe

UNIF_INT = (10, 30)     # uniforme discrete
UNIF_REAL = (1.0, 3.0)  # uniforme continue
NORMAL_PARAMS = (0.0, 0.75)  # (mu, sigma) ; sigma = ecart-type
EXP_MEAN = 2.0          # exponentielle de moyenne 2 (lambda = 0.5)
GEOM_P = 0.3            # geometrique avec p = 0.3

SAMPLE_SIZES = (10, 100, 1000, 10000)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def banner(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n  {title}\n{bar}")


def save_samples(name: str, samples_per_n: dict) -> None:
    """Ecrit chaque echantillon dans un fichier texte pour archive et reproduction."""
    for n, samples in samples_per_n.items():
        path = DATA_DIR / f"{name}_n{n}.txt"
        # Formatage : 8 colonnes lisibles
        with open(path, "w") as f:
            f.write(f"# Distribution : {name}\n")
            f.write(f"# n = {n} valeurs\n")
            f.write(f"# Seed = {SEED} (groupe {GROUP})\n")
            for i, v in enumerate(samples):
                f.write(f"{v}\n")


def print_short(name: str, samples_per_n: dict, max_show: int = 10) -> None:
    """Affiche un apercu des valeurs sur la console (max_show premieres pour chaque n)."""
    print(f"\n  --- Apercu : {name} ---")
    for n, samples in samples_per_n.items():
        head = samples[:max_show]
        if isinstance(head[0], (np.integer, int)):
            fmt = "[" + ", ".join(str(int(x)) for x in head) + "]"
        else:
            fmt = "[" + ", ".join(f"{x:.4f}" for x in head) + "]"
        suffix = " ..." if n > max_show else ""
        print(f"    n = {n:>5d} (premieres {min(max_show, n)} valeurs) : {fmt}{suffix}")


def stats(samples: np.ndarray) -> dict:
    """Retourne un dict {moyenne, ecart-type, min, max} pour une suite."""
    return {
        "moyenne": float(np.mean(samples)),
        "ecart_type": float(np.std(samples, ddof=0)),
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
    }


def print_stats_table(name: str, samples_per_n: dict,
                       theo_mean: float, theo_std: float) -> None:
    """Affiche un tableau comparatif theorie / simulation."""
    print(f"\n  --- Statistiques : {name} ---")
    print(f"    Theorique : moyenne = {theo_mean:.4f}, "
          f"ecart-type = {theo_std:.4f}")
    print(f"    {'n':>6s} | {'moyenne':>10s} | {'ecart-type':>10s} "
          f"| {'min':>8s} | {'max':>8s}")
    print(f"    {'-'*6} + {'-'*10} + {'-'*10} + {'-'*8} + {'-'*8}")
    for n, samples in samples_per_n.items():
        s = stats(samples)
        print(f"    {n:>6d} | {s['moyenne']:>10.4f} | {s['ecart_type']:>10.4f} "
              f"| {s['min']:>8.2f} | {s['max']:>8.2f}")


# =============================================================================
# PARTIE 1 -- Generateurs pseudo-aleatoires
# =============================================================================

def part1_prng() -> None:
    banner("PARTIE 1 : Generateurs pseudo-aleatoires (seed = 2)")

    # ---- 1.a : courtes sequences pour bien voir le mecanisme
    print("\n[1.a] Premieres valeurs entieres X_n des generateurs etudies")
    for builder in (gen.make_minstd, gen.make_numerical_recipes,
                    gen.make_bad_short_period, gen.make_bad_multiplicative,
                    gen.make_randu):
        g = builder(seed=SEED)
        ints = g.sample_ints(10)
        print(f"  {g.name:38s} : {ints}")

    # ---- 1.b : detection de periode (utile pour les petits modules)
    print("\n[1.b] Detection de la periode")
    for builder in (gen.make_bad_short_period, gen.make_bad_multiplicative):
        g = builder(seed=SEED)
        period, sequence = gen.detect_period(g, max_steps=200)
        print(f"  {g.name:38s} : periode = {period}")
        # Petite trace pour voir le cycle
        plots.plot_lcg_sequence(g.name, sequence, period)
        # Sauvegarde du cycle complet en data/
        with open(DATA_DIR / f"prng_seq_{builder.__name__}.txt", "w") as f:
            f.write(f"# {g.name}\n# periode detectee = {period}\n")
            for v in sequence:
                f.write(f"{v}\n")

    # ---- 1.c : histogrammes des U_n pour un bon generateur (MINSTD)
    print("\n[1.c] Histogramme des U_n (MINSTD / Park-Miller)")
    g = gen.make_minstd(seed=SEED)
    U = np.array(g.sample_uniforms(10_000))
    plots.plot_lcg_uniform_hist(g.name, U, bins=40)

    # ---- 1.d : test visuel des paires
    print("\n[1.d] Diagramme des paires (test visuel)")
    # MINSTD : bon comportement attendu
    g = gen.make_minstd(seed=SEED)
    U = np.array(g.sample_uniforms(3000))
    plots.plot_pairs_scatter(g.name, U)

    # RANDU : structures geometriques caracteristiques
    g = gen.make_randu(seed=SEED)
    U = np.array(g.sample_uniforms(3000))
    plots.plot_pairs_scatter(g.name, U)
    # En 3D le defaut historique apparait : on prend un angle bien choisi
    # pour voir les 15 plans paralleles (relation 9*Un - 6*Un+1 + Un+2 ≡ 0).
    g = gen.make_randu(seed=SEED)
    U3 = np.array(g.sample_uniforms(10_000))
    plots.plot_triples_scatter(g.name, U3, elev=30, azim=-130)

    # Le mauvais multiplicatif (a=6, m=13) : tres peu de valeurs distinctes
    g = gen.make_bad_multiplicative(seed=SEED)
    U = np.array(g.sample_uniforms(500))
    plots.plot_pairs_scatter(g.name, U)


# =============================================================================
# PARTIE 2 -- Lois discretes
# =============================================================================

def part2_discrete(rng_factory) -> None:
    banner("PARTIE 2 : Variables aleatoires discretes (seed = 2)")

    # --- Uniforme discrete sur [10, 30]
    a, b = UNIF_INT
    samples_per_n = {n: dist.uniforme_discrete(n, a, b, rng_factory()) for n in SAMPLE_SIZES}
    print_short("Uniforme discrete (10, 30)", samples_per_n)
    save_samples("uniforme_discrete", samples_per_n)
    theo_mean, theo_std = dist.stats_theoriques("uniforme_discrete", a=a, b=b)
    print_stats_table("Uniforme discrete", samples_per_n, theo_mean, theo_std)
    pmf = lambda k: 1.0 / (b - a + 1)
    plots.plot_histograms_discrete(
        "uniforme_discrete", samples_per_n, theoretical_pmf=pmf,
        support=range(a, b + 1), xlabel="k",
        title_prefix=f"Loi uniforme discrete sur [{a},{b}] - seed = {SEED}")

    # --- Bernoulli(p=0.3) - utile en pre-requis de la geometrique
    p_bern = 0.3
    samples_per_n = {n: dist.bernoulli(n, p_bern, rng_factory()) for n in SAMPLE_SIZES}
    print_short("Bernoulli(0.3)", samples_per_n)
    save_samples("bernoulli", samples_per_n)
    theo_mean, theo_std = dist.stats_theoriques("bernoulli", p=p_bern)
    print_stats_table("Bernoulli(0.3)", samples_per_n, theo_mean, theo_std)
    pmf_b = lambda k: p_bern if k == 1 else (1 - p_bern)
    plots.plot_histograms_discrete(
        "bernoulli", samples_per_n, theoretical_pmf=pmf_b,
        support=(0, 1), xlabel="k (0 = echec, 1 = succes)",
        title_prefix=f"Loi de Bernoulli(p = {p_bern}) - seed = {SEED}")

    # --- Geometrique (loi additionnelle du groupe 2)
    p = GEOM_P
    samples_per_n = {n: dist.geometrique(n, p, rng_factory()) for n in SAMPLE_SIZES}
    print_short("Geometrique(0.3)", samples_per_n)
    save_samples("geometrique", samples_per_n)
    theo_mean, theo_std = dist.stats_theoriques("geometrique", p=p)
    print_stats_table("Geometrique(0.3)", samples_per_n, theo_mean, theo_std)
    pmf_g = lambda k: (1 - p) ** (k - 1) * p
    # On affiche jusqu'au 99eme percentile pour ne pas perdre la lecture
    kmax = max(1, int(math.ceil(math.log(0.01) / math.log(1 - p))))
    plots.plot_histograms_discrete(
        "geometrique", samples_per_n, theoretical_pmf=pmf_g,
        support=range(1, kmax + 1), xlabel="k (essai du premier succes)",
        title_prefix=f"Loi geometrique(p = {p}) - seed = {SEED}")

    # --- Poisson (lambda = 2, equivalent du e = 2 pour une intuition de la cadence)
    lam = 2.0
    samples_per_n = {n: dist.poisson(n, lam, rng_factory()) for n in SAMPLE_SIZES}
    print_short(f"Poisson(lambda = {lam})", samples_per_n)
    save_samples("poisson", samples_per_n)
    # E[X] = lambda, Var[X] = lambda
    theo_mean, theo_std = lam, math.sqrt(lam)
    print_stats_table(f"Poisson(lambda = {lam})", samples_per_n, theo_mean, theo_std)
    # PMF : exp(-lambda) * lambda^k / k!
    pmf_p = lambda k: math.exp(-lam) * (lam ** k) / math.factorial(k)
    plots.plot_histograms_discrete(
        "poisson", samples_per_n, theoretical_pmf=pmf_p,
        support=range(0, 12), xlabel="k",
        title_prefix=f"Loi de Poisson(lambda = {lam}) - seed = {SEED}")


# =============================================================================
# PARTIE 3 -- Lois continues
# =============================================================================

def part3_continuous(rng_factory) -> None:
    banner("PARTIE 3 : Variables aleatoires continues (seed = 2)")

    # --- Uniforme continue sur [1, 3]
    a, b = UNIF_REAL
    samples_per_n = {n: dist.uniforme_continue(n, a, b, rng_factory()) for n in SAMPLE_SIZES}
    print_short(f"Uniforme continue ({a}, {b})", samples_per_n)
    save_samples("uniforme_continue", samples_per_n)
    theo_mean, theo_std = dist.stats_theoriques("uniforme_continue", a=a, b=b)
    print_stats_table(f"Uniforme continue ({a}, {b})", samples_per_n, theo_mean, theo_std)
    pdf = lambda x: np.where((x >= a) & (x <= b), 1.0 / (b - a), 0.0)
    plots.plot_histograms_continuous(
        "uniforme_continue", samples_per_n, theoretical_pdf=pdf,
        xlim=(a - 0.2, b + 0.2), xlabel="x",
        title_prefix=f"Loi uniforme continue sur [{a},{b}] - seed = {SEED}",
        bins=20)

    # --- Exponentielle (moyenne = 2)
    mean = EXP_MEAN
    lam = 1.0 / mean
    samples_per_n = {n: dist.exponentielle(n, mean, rng_factory()) for n in SAMPLE_SIZES}
    print_short(f"Exponentielle (moyenne = {mean})", samples_per_n)
    save_samples("exponentielle", samples_per_n)
    theo_mean, theo_std = dist.stats_theoriques("exponentielle", mean=mean)
    print_stats_table(f"Exponentielle (moyenne = {mean})", samples_per_n, theo_mean, theo_std)
    pdf = lambda x: np.where(x >= 0, lam * np.exp(-lam * x), 0.0)
    # On limite l'axe x pour la lecture (P(X > 10) ~ 7e-3)
    plots.plot_histograms_continuous(
        "exponentielle", samples_per_n, theoretical_pdf=pdf,
        xlim=(0, 12), xlabel="x",
        title_prefix=f"Loi exponentielle (moyenne = {mean}) - seed = {SEED}",
        bins=40)

    # --- Normale N(0, 0.75)
    mu, sigma = NORMAL_PARAMS
    samples_per_n = {n: dist.normale(n, mu, sigma, rng_factory()) for n in SAMPLE_SIZES}
    print_short(f"Normale N({mu}, sigma = {sigma})", samples_per_n)
    save_samples("normale", samples_per_n)
    theo_mean, theo_std = dist.stats_theoriques("normale", mu=mu, sigma=sigma)
    print_stats_table(f"Normale N({mu}, sigma = {sigma})", samples_per_n, theo_mean, theo_std)
    pdf = lambda x: (1.0 / (sigma * math.sqrt(2 * math.pi))) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2)
    plots.plot_histograms_continuous(
        "normale", samples_per_n, theoretical_pdf=pdf,
        xlim=(mu - 4 * sigma, mu + 4 * sigma), xlabel="x",
        title_prefix=f"Loi normale N({mu}, {sigma}) - seed = {SEED}",
        bins=40)


# =============================================================================
# PARTIE 4 -- Analyse globale : convergence (LLN)
# =============================================================================

def part4_analysis(rng_factory) -> None:
    banner("PARTIE 4 : Analyse - loi des grands nombres")

    # On reprend chaque loi avec n = 10000 et on trace la moyenne cumulee
    cases = [
        ("uniforme_continue",
         lambda r: dist.uniforme_continue(10_000, *UNIF_REAL, r),
         (UNIF_REAL[0] + UNIF_REAL[1]) / 2),
        ("exponentielle",
         lambda r: dist.exponentielle(10_000, EXP_MEAN, r), EXP_MEAN),
        ("normale",
         lambda r: dist.normale(10_000, *NORMAL_PARAMS, r), NORMAL_PARAMS[0]),
        ("geometrique",
         lambda r: dist.geometrique(10_000, GEOM_P, r), 1.0 / GEOM_P),
    ]
    for name, sampler, mean in cases:
        samples = sampler(rng_factory())
        plots.plot_lln(name, samples, mean)
        print(f"  LLN {name:18s} : moyenne finale = "
              f"{np.mean(samples):.4f}  (theorique = {mean:.4f})")


# =============================================================================
# Point d'entree
# =============================================================================

def main() -> None:
    print(f">> Reprise : seed = {SEED}, groupe = {GROUP}")
    # On utilise la classe numpy.random.Generator pour la qualite statistique
    # mais on remet la seed a chaque appel pour avoir des suites independantes
    # entre lois (et donc des resultats reproductibles meme si l'ordre change).
    def rng_factory() -> np.random.Generator:
        return np.random.default_rng(SEED)

    part1_prng()
    part2_discrete(rng_factory)
    part3_continuous(rng_factory)
    part4_analysis(rng_factory)

    print("\n>> Tout est genere. Voir :")
    print("   - data/      (suites de valeurs)")
    print("   - rapport/figures/ (figures PDF + PNG)")


if __name__ == "__main__":
    main()
