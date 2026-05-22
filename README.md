# RCP103 – TP Génération de variables aléatoires
**Groupe 2** 

## Prérequis

```bash
pip install numpy matplotlib
```

## Lancer le projet

```bash
python main.py
```

Ça fait les deux choses d'un coup :
1. **Affiche dans le terminal** les valeurs générées pour n = 10, 100, 1 000, 10 000
2. **Génère les histogrammes** dans le dossier `figures/`

## Structure

```
rcp103_groupe2/
├── main.py                   ← point d'entrée
├── config/
│   └── parametres.py         ← toutes les constantes du groupe 2
├── distributions/
│   ├── uniforme_entiere.py
│   ├── uniforme_reelle.py
│   ├── normale.py
│   ├── exponentielle.py
│   └── geometrique.py
├── affichage/
│   ├── terminal.py
│   └── histogrammes.py
└── figures/                  ← histogrammes générés (PNG)
```

## Paramètres groupe 2

| Distribution      | Paramètres                  |
|-------------------|-----------------------------|
| Uniforme entière  | [10, 30]                    |
| Uniforme réelle   | [1.0, 3.0]                  |
| Normale           | µ = 0, σ = 0.75             |
| Exponentielle     | moyenne = 2  (λ = 0.5)      |
| Géométrique       | p = 0.3                     |
| **Seed**          | **2**                       |
