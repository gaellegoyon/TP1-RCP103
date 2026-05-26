"""Point d'entrée du TP"""

import distributions.uniforme_entiere as uniforme_entiere
import distributions.uniforme_reelle  as uniforme_reelle
import distributions.normale          as normale
import distributions.exponentielle    as exponentielle
import distributions.geometrique      as geometrique

from config.parametres import VALEURS_N
from affichage.terminal import afficher_valeurs
from affichage.histogrammes import generer_tous_les_histogrammes


DISTRIBUTIONS = [
    {
        "nom":          uniforme_entiere.NOM,
        "label":        uniforme_entiere.LABEL,
        "generateur":   uniforme_entiere.generer,
        "est_discrete": uniforme_entiere.EST_DISCRETE,
        "nom_fichier":  "uniforme_entiere",
        "valeurs_n":    VALEURS_N,
    },
    {
        "nom":          uniforme_reelle.NOM,
        "label":        uniforme_reelle.LABEL,
        "generateur":   uniforme_reelle.generer,
        "est_discrete": uniforme_reelle.EST_DISCRETE,
        "nom_fichier":  "uniforme_reelle",
        "valeurs_n":    VALEURS_N,
    },
    {
        "nom":          normale.NOM,
        "label":        normale.LABEL,
        "generateur":   normale.generer,
        "est_discrete": normale.EST_DISCRETE,
        "nom_fichier":  "normale",
        "valeurs_n":    VALEURS_N,
    },
    {
        "nom":          exponentielle.NOM,
        "label":        exponentielle.LABEL,
        "generateur":   exponentielle.generer,
        "est_discrete": exponentielle.EST_DISCRETE,
        "nom_fichier":  "exponentielle",
        "valeurs_n":    VALEURS_N,
    },
    {
        "nom":          geometrique.NOM,
        "label":        geometrique.LABEL,
        "generateur":   geometrique.generer,
        "est_discrete": geometrique.EST_DISCRETE,
        "nom_fichier":  "geometrique",
        "valeurs_n":    VALEURS_N,
    },
]
def main() -> None:
    print("\nAFFICHAGE DES VALEURS GÉNÉRÉES")
    for dist in DISTRIBUTIONS:
        afficher_valeurs(
            nom           = dist["nom"],
            label         = dist["label"],
            generateur_fn = dist["generateur"],
            valeurs_n     = dist["valeurs_n"],
        )

    print("\n\nGÉNÉRATION DES HISTOGRAMMES")
    generer_tous_les_histogrammes(DISTRIBUTIONS)

    print("\nTerminé. Les figures sont dans le dossier « figures/ ».")


if __name__ == "__main__":
    main()