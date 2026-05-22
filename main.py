"""
RCP103 – TP Génération de variables aléatoires
Groupe 2

Point d'entrée principal du projet.
Lance l'affichage terminal et la génération des histogrammes
pour toutes les distributions du groupe 2.
"""

import distributions.uniforme_entiere  as uniforme_entiere
import distributions.uniforme_reelle   as uniforme_reelle
import distributions.normale           as normale
import distributions.exponentielle     as exponentielle
import distributions.geometrique       as geometrique

from affichage.terminal      import afficher_valeurs
from affichage.histogrammes  import generer_tous_les_histogrammes


# ── Liste des distributions du groupe 2 ───────────────────────────────────────
# Chaque entrée regroupe toutes les infos nécessaires à l'affichage
DISTRIBUTIONS = [
    {
        "nom":         uniforme_entiere.NOM,
        "label":       uniforme_entiere.LABEL,
        "generateur":  uniforme_entiere.generer,
        "est_discrete":uniforme_entiere.EST_DISCRETE,
        "nom_fichier": "uniforme_entiere",
    },
    {
        "nom":         uniforme_reelle.NOM,
        "label":       uniforme_reelle.LABEL,
        "generateur":  uniforme_reelle.generer,
        "est_discrete":uniforme_reelle.EST_DISCRETE,
        "nom_fichier": "uniforme_reelle",
    },
    {
        "nom":         normale.NOM,
        "label":       normale.LABEL,
        "generateur":  normale.generer,
        "est_discrete":normale.EST_DISCRETE,
        "nom_fichier": "normale",
    },
    {
        "nom":         exponentielle.NOM,
        "label":       exponentielle.LABEL,
        "generateur":  exponentielle.generer,
        "est_discrete":exponentielle.EST_DISCRETE,
        "nom_fichier": "exponentielle",
    },
    {
        "nom":         geometrique.NOM,
        "label":       geometrique.LABEL,
        "generateur":  geometrique.generer,
        "est_discrete":geometrique.EST_DISCRETE,
        "nom_fichier": "geometrique",
    },
]
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    # 1. Affichage des valeurs dans le terminal
    print("\n📋  AFFICHAGE DES VALEURS GÉNÉRÉES")
    for dist in DISTRIBUTIONS:
        afficher_valeurs(
            nom          = dist["nom"],
            label        = dist["label"],
            generateur_fn= dist["generateur"],
        )

    # 2. Génération des histogrammes
    print("\n\n📊  GÉNÉRATION DES HISTOGRAMMES")
    generer_tous_les_histogrammes(DISTRIBUTIONS)

    print("\n✅  Terminé. Les figures sont dans le dossier « figures/ ».")


if __name__ == "__main__":
    main()
