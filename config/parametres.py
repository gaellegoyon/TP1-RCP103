"""
Paramètres du groupe 2 pour le TP RCP103.
Toutes les constantes sont centralisées ici.
"""

# ── Graine aléatoire ───────────────────────────────────────────────────────────
GRAINE = 2

# ── Valeurs de n à tester ──────────────────────────────────────────────────────
VALEURS_N = [10, 100, 1_000, 10_000]

# ── Uniforme entière ───────────────────────────────────────────────────────────
UNIF_ENTIERE_MIN = 10
UNIF_ENTIERE_MAX = 30

# ── Uniforme réelle ────────────────────────────────────────────────────────────
UNIF_REELLE_MIN = 1.0
UNIF_REELLE_MAX = 3.0

# ── Normale ────────────────────────────────────────────────────────────────────
NORMALE_MOYENNE    = 0
NORMALE_ECART_TYPE = 0.75

# ── Exponentielle ──────────────────────────────────────────────────────────────
# e = moyenne = 1/λ  →  λ = 0.5
EXPONENTIELLE_MOYENNE = 2

# ── Géométrique ────────────────────────────────────────────────────────────────
# Probabilité de succès à chaque essai
GEOMETRIQUE_P = 0.3
