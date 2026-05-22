"""
Génération et sauvegarde des histogrammes.
Un histogramme individuel par (distribution, n) + une figure overlay 2x2.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from config.parametres import VALEURS_N, GRAINE

DOSSIER_FIGURES = "figures"
DPI             = 150
TAILLE_FIGURE   = (6, 4)

# Une couleur par valeur de n
COULEURS = {
    10:      "#4C72B0",
    100:     "#DD8452",
    1_000:   "#55A868",
    10_000:  "#C44E52",
}


def _calculer_bins(valeurs: np.ndarray, est_discrete: bool):
    """Calcule les bins adaptés au type de distribution."""
    if est_discrete:
        # Une barre par valeur entière présente
        minimum = int(valeurs.min())
        maximum = int(valeurs.max())
        return np.arange(minimum, maximum + 2) - 0.5
    # Règle de Sturges pour les distributions continues
    return max(10, min(60, int(np.ceil(np.log2(len(valeurs)) + 1))))


def _boite_statistiques(valeurs: np.ndarray, n: int) -> str:
    """Formate les statistiques descriptives à afficher sur la figure."""
    return (
        f"n = {n:,}\n"
        f"µ̂ = {valeurs.mean():.4f}\n"
        f"σ̂ = {valeurs.std():.4f}\n"
        f"min = {valeurs.min():.4f}\n"
        f"max = {valeurs.max():.4f}"
    )


def sauvegarder_histogramme(
    valeurs:      np.ndarray,
    titre:        str,
    label_x:      str,
    n:            int,
    est_discrete: bool,
    chemin:       str,
) -> None:
    """Génère et sauvegarde un histogramme pour une distribution et un n donné."""
    fig, ax = plt.subplots(figsize=TAILLE_FIGURE)

    bins   = _calculer_bins(valeurs, est_discrete)
    couleur = COULEURS[n]

    ax.hist(valeurs, bins=bins, color=couleur, edgecolor="white",
            linewidth=0.4, density=True, alpha=0.85)

    ax.set_title(f"{titre}  –  n = {n:,}", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(label_x, fontsize=9)
    ax.set_ylabel("Densité de probabilité", fontsize=9)

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    ax.text(
        0.98, 0.97,
        _boite_statistiques(valeurs, n),
        transform          = ax.transAxes,
        fontsize           = 7.5,
        verticalalignment  = "top",
        horizontalalignment= "right",
        bbox               = dict(boxstyle="round,pad=0.4", facecolor="white",
                                  edgecolor="#cccccc", alpha=0.85),
    )

    fig.tight_layout()
    fig.savefig(chemin, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  Sauvegardé : {chemin}")


def sauvegarder_overlay(
    nom:          str,
    label:        str,
    generateur_fn,
    est_discrete: bool,
    chemin:       str,
) -> None:
    """Génère une figure 2×2 avec un subplot par valeur de n."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(
        f"{label}  –  Graine = {GRAINE}",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for ax, n in zip(axes.flat, VALEURS_N):
        valeurs = generateur_fn(n)
        bins    = _calculer_bins(valeurs, est_discrete)
        couleur = COULEURS[n]

        ax.hist(valeurs, bins=bins, color=couleur, edgecolor="white",
                linewidth=0.4, density=True, alpha=0.85)

        stats = f"µ̂={valeurs.mean():.3f}  σ̂={valeurs.std():.3f}"
        ax.set_title(f"n = {n:,}   ({stats})", fontsize=9)
        ax.set_xlabel(nom, fontsize=8)
        ax.set_ylabel("Densité", fontsize=8)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(chemin, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  Overlay sauvegardé : {chemin}")


def generer_tous_les_histogrammes(distributions: list[dict]) -> None:
    """Génère tous les histogrammes pour toutes les distributions."""
    os.makedirs(DOSSIER_FIGURES, exist_ok=True)

    for dist in distributions:
        nom          = dist["nom"]
        label        = dist["label"]
        generateur   = dist["generateur"]
        est_discrete = dist["est_discrete"]
        nom_fichier  = dist["nom_fichier"]

        print(f"\n── {nom} ──")

        # Histogrammes individuels
        for n in VALEURS_N:
            valeurs = generateur(n)
            chemin  = os.path.join(DOSSIER_FIGURES, f"{nom_fichier}_n{n}.png")
            sauvegarder_histogramme(
                valeurs      = valeurs,
                titre        = label,
                label_x      = nom,
                n            = n,
                est_discrete = est_discrete,
                chemin       = chemin,
            )

        # Figure overlay 2×2
        chemin_overlay = os.path.join(DOSSIER_FIGURES, f"{nom_fichier}_overlay.png")
        sauvegarder_overlay(
            nom          = nom,
            label        = label,
            generateur_fn= generateur,
            est_discrete = est_discrete,
            chemin       = chemin_overlay,
        )
