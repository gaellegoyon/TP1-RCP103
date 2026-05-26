"""Génération et sauvegarde des histogrammes."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

from config.parametres import (
    UNIF_ENTIERE_MIN, UNIF_ENTIERE_MAX,
    GRAINE,
    UNIF_REELLE_MIN, UNIF_REELLE_MAX,
    NORMALE_MOYENNE, NORMALE_ECART_TYPE,
    EXPONENTIELLE_MOYENNE,
    GEOMETRIQUE_P,
)

DOSSIER_FIGURES = "figures"
DPI             = 150
TAILLE_FIGURE   = (6, 4)

COULEURS = {
    10:      "#4C72B0",
    100:     "#DD8452",
    1_000:   "#55A868",
    10_000:  "#C44E52",
    100_000: "#8172B2",
}


def _calculer_bins(valeurs: np.ndarray, est_discrete: bool, n: int):
    if est_discrete:
        minimum = int(valeurs.min())
        maximum = int(valeurs.max())
        return np.arange(minimum, maximum + 2) - 0.5
    nb_bins = int(np.sqrt(n))
    return max(20, min(120, nb_bins))


def _courbe_theorique(ax, nom_distribution: str, valeurs: np.ndarray) -> None:
    x_min, x_max = valeurs.min(), valeurs.max()
    x = np.linspace(x_min, x_max, 500)

    if nom_distribution == "Uniforme réelle":
        y = stats.uniform.pdf(x, loc=UNIF_REELLE_MIN,
                              scale=UNIF_REELLE_MAX - UNIF_REELLE_MIN)

    elif nom_distribution == "Normale":
        y = stats.norm.pdf(x, loc=NORMALE_MOYENNE, scale=NORMALE_ECART_TYPE)

    elif nom_distribution == "Exponentielle":
        y = stats.expon.pdf(x, scale=EXPONENTIELLE_MOYENNE)

    elif nom_distribution == "Uniforme entière":
        nb_valeurs = UNIF_ENTIERE_MAX - UNIF_ENTIERE_MIN + 1
        p_theorique = 1 / nb_valeurs
        ax.hlines(p_theorique, UNIF_ENTIERE_MIN - 0.5, UNIF_ENTIERE_MAX + 0.5,
                  colors="red", linewidth=1.8, label="Théorique", zorder=5)
        ax.legend(fontsize=7)
        return

    elif nom_distribution == "Géométrique":
        k = np.arange(1, int(valeurs.max()) + 1)
        p_theorique = GEOMETRIQUE_P * (1 - GEOMETRIQUE_P) ** (k - 1)
        k_cont = np.linspace(1, int(valeurs.max()), 500)
        p_cont = GEOMETRIQUE_P * (1 - GEOMETRIQUE_P) ** (k_cont - 1)
        ax.plot(k_cont, p_cont, color="red", linewidth=1.8,
                label="Théorique", zorder=5)
        ax.legend(fontsize=7)
        return

    else:
        return

    ax.plot(x, y, color="red", linewidth=1.8, label="Théorique", zorder=5)
    ax.legend(fontsize=7)


def _boite_statistiques(valeurs: np.ndarray, n: int) -> str:
    return (
        f"n = {n:,}\n"
        f"µ̂ = {valeurs.mean():.4f}\n"
        f"σ̂ = {valeurs.std():.4f}\n"
        f"min = {valeurs.min():.4f}\n"
        f"max = {valeurs.max():.4f}"
    )


def sauvegarder_histogramme(
    valeurs:          np.ndarray,
    titre:            str,
    label_x:          str,
    nom_distribution: str,
    n:                int,
    est_discrete:     bool,
    chemin:           str,
) -> None:
    fig, ax = plt.subplots(figsize=TAILLE_FIGURE)

    bins    = _calculer_bins(valeurs, est_discrete, n)
    couleur = COULEURS[n]

    ax.hist(valeurs, bins=bins, color=couleur, edgecolor="white",
            linewidth=0.3, density=True, alpha=0.75)

    _courbe_theorique(ax, nom_distribution, valeurs)

    ax.set_title(f"{titre}  –  n = {n:,}", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(label_x, fontsize=9)
    ax.set_ylabel("Densité de probabilité", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    ax.text(
        0.98, 0.97,
        _boite_statistiques(valeurs, n),
        transform           = ax.transAxes,
        fontsize            = 7.5,
        verticalalignment   = "top",
        horizontalalignment = "right",
        bbox                = dict(boxstyle="round,pad=0.4", facecolor="white",
                                   edgecolor="#cccccc", alpha=0.85),
    )

    fig.tight_layout()
    fig.savefig(chemin, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  Sauvegardé : {chemin}")


def sauvegarder_overlay(
    nom:              str,
    label:            str,
    generateur_fn,
    valeurs_n:        list,
    est_discrete:     bool,
    chemin:           str,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11, 13))
    fig.suptitle(f"{label}  –  Graine = {GRAINE}",
                 fontsize=13, fontweight="bold")

    for ax, n in zip(axes.flat, valeurs_n):
        valeurs = generateur_fn(n)
        bins    = _calculer_bins(valeurs, est_discrete, n)
        couleur = COULEURS[n]

        ax.hist(valeurs, bins=bins, color=couleur, edgecolor="white",
                linewidth=0.3, density=True, alpha=0.75)

        _courbe_theorique(ax, nom, valeurs)

        stats_str = f"µ̂={valeurs.mean():.3f}  σ̂={valeurs.std():.3f}"
        ax.set_title(f"n = {n:,}   ({stats_str})", fontsize=9)
        ax.set_xlabel(nom, fontsize=8)
        ax.set_ylabel("Densité", fontsize=8)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7)

    axes.flat[len(valeurs_n)].set_visible(False)

    fig.tight_layout()
    fig.savefig(chemin, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  Overlay sauvegardé : {chemin}")

def generer_tous_les_histogrammes(distributions: list[dict]) -> None:
    os.makedirs(DOSSIER_FIGURES, exist_ok=True)

    for dist in distributions:
        nom          = dist["nom"]
        label        = dist["label"]
        generateur   = dist["generateur"]
        est_discrete = dist["est_discrete"]
        nom_fichier  = dist["nom_fichier"]
        valeurs_n    = dist["valeurs_n"]

        print(f"\n── {nom} ──")

        for n in valeurs_n:
            valeurs = generateur(n)
            chemin  = os.path.join(DOSSIER_FIGURES, f"{nom_fichier}_n{n}.png")
            sauvegarder_histogramme(
                valeurs          = valeurs,
                titre            = label,
                label_x          = nom,
                nom_distribution = nom,
                n                = n,
                est_discrete     = est_discrete,
                chemin           = chemin,
            )

        chemin_overlay = os.path.join(DOSSIER_FIGURES, f"{nom_fichier}_overlay.png")
        sauvegarder_overlay(
            nom           = nom,
            label         = label,
            generateur_fn = generateur,
            valeurs_n     = valeurs_n,
            est_discrete  = est_discrete,
            chemin        = chemin_overlay,
        )