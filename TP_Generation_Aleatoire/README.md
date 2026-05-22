# TP — Génération de variables aléatoires (RCP103)

Évaluation des performances des systèmes informatiques — CNAM Paris, NCA USEEJ7.

**Groupe : 2**
**Seed : 2 (numéro du groupe)**

---

## Paramètres officiels du groupe 2

D'après le tableau du cours (page 5 du sujet) :

| Distribution             | Paramètres                                    |
| ------------------------ | --------------------------------------------- |
| Uniforme discrète        | `(min=10, max=30)`                            |
| Uniforme continue        | `(min=1.0, max=3.0)`                          |
| Normale                  | `μ = 0`, `σ = 0.75`  (couple `(0, 0.75)`)     |
| Exponentielle            | moyenne `e = 2` (donc `λ = 1/2`)              |
| Distribution additionnelle | **Géométrique** avec `p = 0.3`              |

Toutes les simulations utilisent `seed = 2` afin que les résultats soient strictement reproductibles.

---

## Structure du projet

```
TP_Generation_Aleatoire/
├── README.md
├── rapport/
│   ├── rapport.tex          ← Source LaTeX du rapport
│   ├── rapport.pdf          ← À générer (cf. ci-dessous)
│   └── figures/             ← Toutes les figures (PDF + PNG)
├── src/
│   ├── generators.py        ← Générateurs congruentiels (LCG, multiplicatif)
│   ├── distributions.py     ← Génération des lois (5 + Bernoulli + Poisson)
│   ├── plots.py             ← Traceurs (histogrammes, paires, LLN)
│   └── main.py              ← Point d'entrée — fixe seed=2 et lance tout
└── data/                    ← Suites de valeurs générées (n=10/100/1000/10000)
```

---

## 1. Reproduire les simulations Python

Pré-requis : Python 3, NumPy, Matplotlib (SciPy facultatif).

```bash
cd src/
python3 main.py
```

Le script :
1. Affiche dans la console les 10 premières valeurs pour chaque loi et chaque `n`.
2. Sauve les suites complètes dans `data/<loi>_n<n>.txt`.
3. Régénère toutes les figures dans `rapport/figures/` (PDF + PNG).

Toutes les exécutions produisent **exactement** les mêmes nombres grâce à `seed=2`.

---

## 2. Générer le PDF du rapport

Le rapport est rédigé en LaTeX. Comme indiqué dans la consigne du professeur, le rapport est destiné à être compilé sur **Overleaf**.

### Méthode A — Overleaf (recommandée par la consigne)

1. Aller sur [https://www.overleaf.com](https://www.overleaf.com).
2. *New Project* → *Upload Project*.
3. Téléverser une archive ZIP contenant le dossier `rapport/`
   (`rapport.tex` + dossier `figures/`).
4. Overleaf détecte automatiquement `rapport.tex` comme fichier principal.
5. Cliquer sur *Recompile* → le PDF apparaît à droite.

> Astuce : si Overleaf ne détecte pas le bon fichier principal, ouvrir
> *Menu → Main document* et choisir `rapport.tex`.

### Méthode B — Compilation locale (si vous avez TeX Live installé)

```bash
cd rapport/
pdflatex rapport.tex
pdflatex rapport.tex   # 2e passe pour la table des matières
```

Le PDF `rapport.pdf` est généré dans `rapport/`.

---

## 3. Contenu du rapport

Le rapport couvre :

1. **Introduction** — pourquoi générer des variables aléatoires en évaluation de performances.
2. **Objectifs** — rappel des paramètres du groupe 2.
3. **Théorie & méthodes** — LCG, multiplicatif, Hull-Dobell, transformée inverse, Box-Muller, Knuth.
4. **Implémentation** — organisation du code.
5. **Partie I — PRNG** — sequences, périodes, histogrammes, test des paires (Marsaglia / RANDU).
6. **Partie II — Lois discrètes** — uniforme discrète `(10,30)`, Bernoulli(0.3), géométrique(0.3), Poisson(2).
7. **Partie III — Lois continues** — uniforme `(1.0, 3.0)`, exponentielle (moy. = 2), normale $\mathcal{N}(0, 0.75)$.
8. **Partie IV — Analyse** — convergence (loi des grands nombres), impact du nombre d'échantillons, lien avec les systèmes informatiques.
9. **Conclusion**.

Chaque figure est légendée et accompagnée d'une lecture détaillée.

---

## 4. Fichiers de données générés

Le dossier `data/` contient, pour chaque distribution, **quatre** fichiers texte
(`*_n10.txt`, `*_n100.txt`, `*_n1000.txt`, `*_n10000.txt`) avec, dans chaque
fichier, exactement `n` valeurs. Ces fichiers permettent une vérification
indépendante du code.

---

## 5. Bibliothèques utilisées

| Bibliothèque | Rôle                              |
| ------------ | --------------------------------- |
| `numpy`      | génération du `U ~ U(0,1)`, vecteurs |
| `matplotlib` | tracés (histogrammes, scatter, LLN)|

Aucune fonction « toute prête » comme `numpy.random.exponential` n'est utilisée
pour générer les lois : la *mécanique* de chaque génération (transformée inverse,
Box-Muller, Knuth) est implémentée explicitement, conformément à l'objectif
pédagogique du TP.

---

*Pedro Braconnot Velloso — RCP103, CNAM Paris — Groupe 2.*
