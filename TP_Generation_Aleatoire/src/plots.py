"""
plots.py
========

Fonctions de trace pour le TP : uniquement des histogrammes.

Toutes les figures sont sauvees en PDF (pour le rapport) et en PNG
(pour visualiser rapidement).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sans serveur graphique
import matplotlib.pyplot as plt
import numpy as np


# Dossier ou on ecrit les figures
FIG_DIR = Path(__file__).resolve().parent.parent / "rapport" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _savefig(fig, basename):
    """Sauve la figure en PDF + PNG."""
    pdf_path = FIG_DIR / f"{basename}.pdf"
    png_path = FIG_DIR / f"{basename}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  -> figure sauvegardee : {pdf_path.name}")


# -----------------------------------------------------------------------------
# Histogrammes pour les lois continues
# -----------------------------------------------------------------------------

def plot_histograms_continuous(name, samples_per_n, theoretical_pdf=None,
                               xlim=None, xlabel="Valeur",
                               title_prefix=None, bins=30):
    """Trace 4 histogrammes (un par n) sur une meme figure.

    On peut superposer la densite theorique en rouge pour comparer.
    """
    title_prefix = title_prefix or name
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for i, (n, samples) in enumerate(samples_per_n.items()):
        ax = axes[i]
        ax.hist(samples, bins=bins, density=True, color="steelblue",
                edgecolor="white", alpha=0.85,
                label="Histogramme empirique")
        if theoretical_pdf is not None:
            if xlim is not None:
                xs = np.linspace(xlim[0], xlim[1], 400)
            else:
                xs = np.linspace(samples.min(), samples.max(), 400)
            ax.plot(xs, theoretical_pdf(xs), color="crimson", linewidth=2.0,
                    label="Densite theorique")
        ax.set_title(f"n = {n}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Densite")
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle(f"{title_prefix}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _savefig(fig, f"hist_{name}")


# -----------------------------------------------------------------------------
# Histogrammes pour les lois discretes
# -----------------------------------------------------------------------------

def plot_histograms_discrete(name, samples_per_n, theoretical_pmf=None,
                             support=None, xlabel="Valeur",
                             title_prefix=None):
    """Trace 4 histogrammes (un par n) pour une loi discrete.

    Barres alignees sur les entiers ; on superpose la PMF theorique
    (croix rouges) si elle est donnee.
    """
    title_prefix = title_prefix or name
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for i, (n, samples) in enumerate(samples_per_n.items()):
        ax = axes[i]
        if support is None:
            vmin, vmax = int(samples.min()), int(samples.max())
            sup = np.arange(vmin, vmax + 1)
        else:
            sup = np.asarray(support)

        counts = np.array([(samples == k).sum() for k in sup]) / len(samples)
        ax.bar(sup, counts, width=0.8, color="steelblue", alpha=0.85,
               edgecolor="white", label="Frequence empirique")

        if theoretical_pmf is not None:
            pmf_vals = np.array([theoretical_pmf(k) for k in sup])
            ax.plot(sup, pmf_vals, "rx", markersize=8, markeredgewidth=2,
                    label="PMF theorique")

        ax.set_title(f"n = {n}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequence / Probabilite")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle(f"{title_prefix}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _savefig(fig, f"hist_{name}")


# -----------------------------------------------------------------------------
# Histogramme du generateur uniforme (partie 1)
# -----------------------------------------------------------------------------

def plot_lcg_uniform_hist(name, samples, bins=30):
    """Histogramme des U_n produits par un PRNG.

    Sert a verifier que le generateur produit bien une loi uniforme
    sur [0, 1].
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(samples, bins=bins, density=True, color="steelblue",
            edgecolor="white", alpha=0.85)
    ax.axhline(1.0, color="crimson", linestyle="--",
               label="Densite theorique U(0,1) = 1")
    ax.set_title(f"Distribution des U_n produits par : {name}\n"
                 f"(n = {len(samples)} valeurs)")
    ax.set_xlabel("U_n = X_n / m")
    ax.set_ylabel("Densite")
    ax.set_xlim(0, 1)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    safe = name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '')
    _savefig(fig, f"prng_unif_{safe}")
