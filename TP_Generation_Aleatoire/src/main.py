"""
main.py
=======

Programme principal du TP - Groupe 2 (seed = 2).

Lance :
    python3 main.py

Ce script :
  - genere les valeurs pour les 4 tailles d'echantillons (n = 10, 100, 1000, 10000)
  - sauve les suites dans data/
  - trace les histogrammes dans rapport/figures/
  - affiche un apercu et des statistiques dans la console

Parametres du groupe 2 (d'apres le tableau du sujet) :
  - Uniforme discrete : (10, 30)
  - Uniforme continue : (1.0, 3.0)
  - Normale           : N(mu = 0, sigma = 0.75)
  - Exponentielle     : moyenne = 2
  - Loi en plus       : geometrique avec p = 0.3
"""

import math
from pathlib import Path

import numpy as np

import distributions as dist
import generators as gen
import plots


# =============================================================================
# Parametres du groupe 2
# =============================================================================

GROUP = 2
SEED = 2  # consigne : la seed est le numero du groupe

UNIF_INT = (10, 30)
UNIF_REAL = (1.0, 3.0)
NORMAL_PARAMS = (0.0, 0.75)   # (mu, sigma)
EXP_MEAN = 2.0
GEOM_P = 0.3

SAMPLE_SIZES = (10, 100, 1000, 10000)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Utilitaires d'affichage et de sauvegarde
# =============================================================================

def banner(title):
    bar = "=" * 78
    print(f"\n{bar}\n  {title}\n{bar}")


def save_samples(name, samples_per_n):
    """Ecrit chaque echantillon dans un fichier texte (un fichier par n)."""
    for n, samples in samples_per_n.items():
        path = DATA_DIR / f"{name}_n{n}.txt"
        with open(path, "w") as f:
            f.write(f"# Loi : {name}\n")
            f.write(f"# n = {n} valeurs\n")
            f.write(f"# Seed = {SEED} (groupe {GROUP})\n")
            for v in samples:
                f.write(f"{v}\n")


def print_short(name, samples_per_n, max_show=10):
    """Affiche les premieres valeurs pour chaque taille d'echantillon."""
    print(f"\n  --- Apercu : {name} ---")
    for n, samples in samples_per_n.items():
        head = samples[:max_show]
        if isinstance(head[0], (np.integer, int)):
            txt = "[" + ", ".join(str(int(x)) for x in head) + "]"
        else:
            txt = "[" + ", ".join(f"{x:.4f}" for x in head) + "]"
        suffix = " ..." if n > max_show else ""
        print(f"    n = {n:>5d} : {txt}{suffix}")


def stats(samples):
    return {
        "moyenne": float(np.mean(samples)),
        "ecart_type": float(np.std(samples, ddof=0)),
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
    }


def print_stats_table(name, samples_per_n, theo_mean, theo_std):
    """Petit tableau theorie / simulation."""
    print(f"\n  --- Stats : {name} ---")
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

def part1_prng():
    banner("PARTIE 1 : Generateurs pseudo-aleatoires (seed = 2)")

    # Premieres valeurs pour chaque generateur
    print("\n[1.a] Premieres valeurs entieres X_n")
    for builder in (gen.make_minstd, gen.make_numerical_recipes,
                    gen.make_bad_short_period, gen.make_bad_multiplicative):
        g = builder(seed=SEED)
        print(f"  {g.name:38s} : {g.sample_ints(10)}")

    # Detection de la periode pour les petits generateurs
    print("\n[1.b] Detection de la periode (sur les petits generateurs)")
    for builder in (gen.make_bad_short_period, gen.make_bad_multiplicative):
        g = builder(seed=SEED)
        period, sequence = gen.detect_period(g, max_steps=200)
        print(f"  {g.name:38s} : periode = {period}")
        # Sauvegarde du cycle complet pour archive
        with open(DATA_DIR / f"prng_seq_{builder.__name__}.txt", "w") as f:
            f.write(f"# {g.name}\n# periode detectee = {period}\n")
            for v in sequence:
                f.write(f"{v}\n")

    # Histogramme du bon generateur (notre seule figure de la partie 1)
    print("\n[1.c] Histogramme de MINSTD (10000 valeurs)")
    g = gen.make_minstd(seed=SEED)
    U = np.array(g.sample_uniforms(10_000))
    plots.plot_lcg_uniform_hist(g.name, U, bins=40)


# =============================================================================
# PARTIE 2 -- Lois discretes
# =============================================================================

def part2_discrete(rng_factory):
    banner("PARTIE 2 : Lois discretes (seed = 2)")

    # Uniforme discrete sur [10, 30]
    a, b = UNIF_INT
    samples_per_n = {n: dist.uniforme_discrete(n, a, b, rng_factory())
                     for n in SAMPLE_SIZES}
    print_short("Uniforme discrete (10, 30)", samples_per_n)
    save_samples("uniforme_discrete", samples_per_n)
    theo_mean, theo_std = dist.stats_theoriques("uniforme_discrete", a=a, b=b)
    print_stats_table("Uniforme discrete", samples_per_n, theo_mean, theo_std)
    pmf = lambda k: 1.0 / (b - a + 1)
    plots.plot_histograms_discrete(
        "uniforme_discrete", samples_per_n, theoretical_pmf=pmf,
        support=range(a, b + 1), xlabel="k",
        title_prefix=f"Loi uniforme discrete sur [{a},{b}] - seed = {SEED}")

    # Bernoulli(0.3) - utile pour comprendre la geometrique
    p_bern = 0.3
    samples_per_n = {n: dist.bernoulli(n, p_bern, rng_factory())
                     for n in SAMPLE_SIZES}
    print_short("Bernoulli(0.3)", samples_per_n)
    save_samples("bernoulli", samples_per_n)
    theo_mean, theo_std = dist.stats_theoriques("bernoulli", p=p_bern)
    print_stats_table("Bernoulli(0.3)", samples_per_n, theo_mean, theo_std)
    pmf_b = lambda k: p_bern if k == 1 else (1 - p_bern)
    plots.plot_histograms_discrete(
        "bernoulli", samples_per_n, theoretical_pmf=pmf_b,
        support=(0, 1), xlabel="k (0 = echec, 1 = succes)",
        title_prefix=f"Loi de Bernoulli(p = {p_bern}) - seed = {SEED}")

    # Geometrique(0.3) - la loi supplementaire du groupe 2
    p = GEOM_P
    samples_per_n = {n: dist.geometrique(n, p, rng_factory())
                     for n in SAMPLE_SIZES}
    print_short("Geometrique(0.3)", samples_per_n)
    save_samples("geometrique", samples_per_n)
    theo_mean, theo_std = dist.stats_theoriques("geometrique", p=p)
    print_stats_table("Geometrique(0.3)", samples_per_n, theo_mean, theo_std)
    pmf_g = lambda k: (1 - p) ** (k - 1) * p
    # On limite l'affichage a un k raisonnable (jusqu'au 99e percentile environ)
    kmax = max(1, int(math.ceil(math.log(0.01) / math.log(1 - p))))
    plots.plot_histograms_discrete(
        "geometrique", samples_per_n, theoretical_pmf=pmf_g,
        support=range(1, kmax + 1), xlabel="k (essai du premier succes)",
        title_prefix=f"Loi geometrique(p = {p}) - seed = {SEED}")

    # Poisson(2) - en lien avec l'exponentielle de moyenne 2
    lam = 2.0
    samples_per_n = {n: dist.poisson(n, lam, rng_factory())
                     for n in SAMPLE_SIZES}
    print_short(f"Poisson(lambda = {lam})", samples_per_n)
    save_samples("poisson", samples_per_n)
    theo_mean, theo_std = lam, math.sqrt(lam)
    print_stats_table(f"Poisson(lambda = {lam})", samples_per_n, theo_mean, theo_std)
    pmf_p = lambda k: math.exp(-lam) * (lam ** k) / math.factorial(k)
    plots.plot_histograms_discrete(
        "poisson", samples_per_n, theoretical_pmf=pmf_p,
        support=range(0, 12), xlabel="k",
        title_prefix=f"Loi de Poisson(lambda = {lam}) - seed = {SEED}")


# =============================================================================
# PARTIE 3 -- Lois continues
# =============================================================================

def part3_continuous(rng_factory):
    banner("PARTIE 3 : Lois continues (seed = 2)")

    # Uniforme continue sur [1.0, 3.0]
    a, b = UNIF_REAL
    samples_per_n = {n: dist.uniforme_continue(n, a, b, rng_factory())
                     for n in SAMPLE_SIZES}
    print_short(f"Uniforme continue ({a}, {b})", samples_per_n)
    save_samples("uniforme_continue", samples_per_n)
    theo_mean, theo_std = dist.stats_theoriques("uniforme_continue", a=a, b=b)
    print_stats_table(f"Uniforme continue ({a}, {b})", samples_per_n,
                      theo_mean, theo_std)
    pdf = lambda x: np.where((x >= a) & (x <= b), 1.0 / (b - a), 0.0)
    plots.plot_histograms_continuous(
        "uniforme_continue", samples_per_n, theoretical_pdf=pdf,
        xlim=(a - 0.2, b + 0.2), xlabel="x",
        title_prefix=f"Loi uniforme continue sur [{a},{b}] - seed = {SEED}",
        bins=20)

    # Exponentielle de moyenne 2
    mean = EXP_MEAN
    lam = 1.0 / mean
    samples_per_n = {n: dist.exponentielle(n, mean, rng_factory())
                     for n in SAMPLE_SIZES}
    print_short(f"Exponentielle (moyenne = {mean})", samples_per_n)
    save_samples("exponentielle", samples_per_n)
    theo_mean, theo_std = dist.stats_theoriques("exponentielle", mean=mean)
    print_stats_table(f"Exponentielle (moyenne = {mean})", samples_per_n,
                      theo_mean, theo_std)
    pdf = lambda x: np.where(x >= 0, lam * np.exp(-lam * x), 0.0)
    plots.plot_histograms_continuous(
        "exponentielle", samples_per_n, theoretical_pdf=pdf,
        xlim=(0, 12), xlabel="x",
        title_prefix=f"Loi exponentielle (moyenne = {mean}) - seed = {SEED}",
        bins=40)

    # Normale N(0, 0.75)
    mu, sigma = NORMAL_PARAMS
    samples_per_n = {n: dist.normale(n, mu, sigma, rng_factory())
                     for n in SAMPLE_SIZES}
    print_short(f"Normale N({mu}, sigma = {sigma})", samples_per_n)
    save_samples("normale", samples_per_n)
    theo_mean, theo_std = dist.stats_theoriques("normale", mu=mu, sigma=sigma)
    print_stats_table(f"Normale N({mu}, sigma = {sigma})", samples_per_n,
                      theo_mean, theo_std)
    pdf = lambda x: (1.0 / (sigma * math.sqrt(2 * math.pi))) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2)
    plots.plot_histograms_continuous(
        "normale", samples_per_n, theoretical_pdf=pdf,
        xlim=(mu - 4 * sigma, mu + 4 * sigma), xlabel="x",
        title_prefix=f"Loi normale N({mu}, {sigma}) - seed = {SEED}",
        bins=40)


# =============================================================================
# Point d'entree
# =============================================================================

def main():
    print(f">> Lancement du TP (groupe = {GROUP}, seed = {SEED})")

    # On reinitialise la seed a chaque appel pour avoir des suites
    # independantes entre lois mais toujours reproductibles.
    def rng_factory():
        return np.random.default_rng(SEED)

    part1_prng()
    part2_discrete(rng_factory)
    part3_continuous(rng_factory)

    print("\n>> Termine.")
    print("   - data/             : suites de valeurs generees")
    print("   - rapport/figures/  : histogrammes")


if __name__ == "__main__":
    main()
