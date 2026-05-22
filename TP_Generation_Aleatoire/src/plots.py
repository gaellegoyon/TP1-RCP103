"""
plots.py
========

Outils de trace pour le TP. Produit les histogrammes et les figures
specifiques au generateur (paires, suites). Toutes les figures sont
sauvegardees au format PDF et PNG dans rapport/figures/.

Auteur : Groupe 2 - RCP103 - CNAM
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # backend non interactif (utile sans serveur X)
import matplotlib.pyplot as plt
import numpy as np


# Repertoire ou seront ecrites les figures (rapport/figures/)
FIG_DIR = Path(__file__).resolve().parent.parent / "rapport" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# Tailles d'echantillons demandees par la consigne
SAMPLE_SIZES: Sequence[int] = (10, 100, 1000, 10000)


# -----------------------------------------------------------------------------
# Utilitaires generaux
# -----------------------------------------------------------------------------

def _savefig(fig: plt.Figure, basename: str) -> None:
    """Sauvegarde une figure au format PDF (pour le rapport) et PNG (pour visu)."""
    pdf_path = FIG_DIR / f"{basename}.pdf"
    png_path = FIG_DIR / f"{basename}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  -> figure sauvegardee : {pdf_path.name} ({pdf_path})")


# -----------------------------------------------------------------------------
# Histogrammes des lois etudiees
# -----------------------------------------------------------------------------

def plot_histograms_continuous(name: str, samples_per_n: dict,
                                theoretical_pdf=None,
                                xlim: Optional[tuple] = None,
                                xlabel: str = "Valeur",
                                title_prefix: Optional[str] = None,
                                bins: int = 30) -> None:
    """Trace les histogrammes pour chaque n et superpose la densite theorique.

    samples_per_n : {n : numpy.ndarray}
    theoretical_pdf : fonction x -> f(x) (densite theorique)
    """
    title_prefix = title_prefix or name
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for i, (n, samples) in enumerate(samples_per_n.items()):
        ax = axes[i]
        ax.hist(samples, bins=bins, density=True, color="steelblue",
                edgecolor="white", alpha=0.85, label=f"Histogramme empirique")
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


def plot_histograms_discrete(name: str, samples_per_n: dict,
                              theoretical_pmf=None,
                              support: Optional[Sequence[int]] = None,
                              xlabel: str = "Valeur",
                              title_prefix: Optional[str] = None) -> None:
    """Trace les histogrammes des lois discretes (barres alignees sur les entiers).

    theoretical_pmf : fonction k -> P(X = k)
    support : liste des valeurs possibles a afficher (sinon deduite des donnees).
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

        # Frequence empirique
        counts = np.array([(samples == k).sum() for k in sup]) / len(samples)
        ax.bar(sup, counts, width=0.8, color="steelblue", alpha=0.85,
               edgecolor="white", label="Frequence empirique")

        # PMF theorique (croix rouges)
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
# Figures specifiques aux generateurs pseudo-aleatoires
# -----------------------------------------------------------------------------

def plot_lcg_sequence(name: str, sequence: List[int], period: int) -> None:
    """Trace la suite generee par un LCG pour montrer le cycle."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(len(sequence)), sequence, "o-", markersize=4, color="steelblue")
    if period > 0:
        ax.axvline(period, color="crimson", linestyle="--",
                   label=f"Periode = {period}")
        # On colore une zone correspondant a la periode
        ax.axvspan(0, period, color="crimson", alpha=0.08)
    ax.set_title(f"Suite produite par : {name}")
    ax.set_xlabel("n (rang dans la suite)")
    ax.set_ylabel("X_n")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, f"prng_seq_{name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '')}")


def plot_lcg_uniform_hist(name: str, samples: np.ndarray, bins: int = 30) -> None:
    """Histogramme de la distribution uniforme produite par un LCG."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(samples, bins=bins, density=True, color="steelblue",
            edgecolor="white", alpha=0.85)
    ax.axhline(1.0, color="crimson", linestyle="--",
               label="Densite theorique U(0,1) = 1")
    ax.set_title(f"Distribution des U_n produits par : {name}\n"
                 f"(n = {len(samples)} echantillons)")
    ax.set_xlabel("U_n = X_n / m")
    ax.set_ylabel("Densite")
    ax.set_xlim(0, 1)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, f"prng_unif_{name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '')}")


def plot_triples_scatter(name: str, samples: np.ndarray,
                         elev: float = 25, azim: float = 30) -> None:
    """Diagramme 3D des triplets (U_n, U_{n+1}, U_{n+2}).

    Pour RANDU on doit voir 15 plans paralleles.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    u1 = samples[:-2]
    u2 = samples[1:-1]
    u3 = samples[2:]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(u1, u2, u3, s=2.5, alpha=0.45, color="steelblue")
    ax.set_xlabel("U_n")
    ax.set_ylabel("U_{n+1}")
    ax.set_zlabel("U_{n+2}")
    ax.set_title(f"Triplets (U_n, U_{{n+1}}, U_{{n+2}}) - {name}\n"
                 f"{len(u1)} points (orientation angle {azim} degrees)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    _savefig(fig, f"prng_triples_{name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '')}")


def plot_pairs_scatter(name: str, samples: np.ndarray) -> None:
    """Diagramme des paires (U_n, U_{n+1}) - test visuel de Marsaglia.

    Pour un bon generateur on doit obtenir un nuage uniforme dans [0,1]^2.
    Un mauvais generateur (RANDU, etc.) produit des structures geometriques.
    """
    u1 = samples[:-1]
    u2 = samples[1:]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(u1, u2, s=4, alpha=0.45, color="steelblue")
    ax.set_xlabel("U_n")
    ax.set_ylabel("U_{n+1}")
    ax.set_title(f"Paires (U_n, U_{{n+1}}) - {name}\n"
                 f"{len(u1)} points")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _savefig(fig, f"prng_pairs_{name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '')}")


# -----------------------------------------------------------------------------
# Figure de convergence (loi des grands nombres)
# -----------------------------------------------------------------------------

def plot_lln(name: str, samples: np.ndarray, true_mean: float) -> None:
    """Trace la moyenne empirique cumulee en fonction de n.

    Illustration de la loi des grands nombres : on doit voir cette moyenne
    converger vers la moyenne theorique au fur et a mesure que n augmente.
    """
    running = np.cumsum(samples) / np.arange(1, len(samples) + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(1, len(samples) + 1), running, color="steelblue",
            label="Moyenne empirique cumulee")
    ax.axhline(true_mean, color="crimson", linestyle="--",
               label=f"Moyenne theorique = {true_mean:.4f}")
    ax.set_xscale("log")
    ax.set_title(f"Convergence de la moyenne empirique - {name}")
    ax.set_xlabel("n (echelle log)")
    ax.set_ylabel("Moyenne")
    ax.legend(loc="best")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    _savefig(fig, f"lln_{name}")
