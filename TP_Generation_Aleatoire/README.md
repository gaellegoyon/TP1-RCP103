# TP1 — Génération de variables aléatoires (RCP103)

Évaluation des performances des systèmes informatiques — CNAM Paris, NCA USEEJ7.

**Groupe : 2**
**Seed : 2 (numéro du groupe)**

---

## Paramètres du groupe 2

D'après le tableau du sujet (page 5) :

| Distribution             | Paramètres                                  |
| ------------------------ | ------------------------------------------- |
| Uniforme discrète        | `(min=10, max=30)`                          |
| Uniforme continue        | `(min=1.0, max=3.0)`                        |
| Normale                  | `μ = 0`, `σ = 0.75`                         |
| Exponentielle            | moyenne `e = 2` (donc `λ = 1/2`)            |
| Loi supplémentaire       | Géométrique avec `p = 0.3`                  |

Toutes les simulations utilisent `seed = 2`, ce qui permet de reproduire les mêmes nombres à chaque exécution.

---

## Organisation du projet

```
TP_Generation_Aleatoire/
├── README.md
├── rapport/
│   ├── rapport.tex          ← Source LaTeX du rapport
│   ├── rapport.pdf          ← À générer (voir plus bas)
│   └── figures/             ← Toutes les figures (PDF + PNG)
├── src/
│   ├── generators.py        ← Générateurs congruentiels (LCG, multiplicatif)
│   ├── distributions.py     ← Génération des 5 lois (+ Bernoulli, Poisson)
│   ├── plots.py             ← Fonctions de tracé
│   └── main.py              ← Script principal (seed=2 fixé ici)
└── data/                    ← Suites générées (n = 10, 100, 1000, 10000)
```

---

## 1. Lancer la simulation Python

Pré-requis : Python 3 + NumPy + Matplotlib.

```bash
cd src/
python3 main.py
```

Le script :
1. Affiche dans la console un aperçu des valeurs et les statistiques pour chaque loi.
2. Sauve toutes les suites dans `data/<loi>_n<n>.txt`.
3. Trace toutes les figures dans `rapport/figures/` (PDF + PNG).

Comme la seed est fixée à 2, les résultats sont strictement reproductibles.

---

## 2. Générer le PDF du rapport

Le rapport est en LaTeX. Le sujet recommande **Overleaf** :

1. Aller sur [https://www.overleaf.com](https://www.overleaf.com).
2. *New Project* → *Upload Project*.
3. Téléverser le dossier `rapport/` (avec `rapport.tex` et `figures/`).
4. Overleaf devrait détecter `rapport.tex` automatiquement.
5. *Recompile* → le PDF s'affiche.

Si on a TeX Live installé en local, on peut aussi compiler avec :
```bash
cd rapport/
pdflatex rapport.tex
pdflatex rapport.tex   # 2 passes pour la table des matières
```

---

## 3. Plan du rapport

1. Introduction
2. Objectifs
3. Un peu de théorie (LCG, transformée inverse, Box-Muller, Knuth)
4. Notre implémentation
5. Partie 1 — Générateurs pseudo-aléatoires (séquences, périodes, RANDU en 3D)
6. Partie 2 — Lois discrètes (uniforme, Bernoulli, géométrique, Poisson)
7. Partie 3 — Lois continues (uniforme, exponentielle, normale)
8. Partie 4 — Analyse (loi des grands nombres, lien avec les systèmes)
9. Conclusion

Chaque figure est légendée et commentée.

---

## 4. Fichiers de données

Le dossier `data/` contient un fichier `.txt` par loi et par valeur de `n`, soit 4 fichiers par loi. Ces fichiers permettent de vérifier les résultats sans relancer le code.

---

*RCP103 — TP1 — Groupe 2.*
