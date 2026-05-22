import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PARAMÈTRES – À ADAPTER SELON TON GROUPE
# ============================================================
SEED = 1          # Remplace par ton numéro de groupe
MIN_VAL = 0       # Remplace par le min de ton groupe
MAX_VAL = 20      # Remplace par le max de ton groupe
# ============================================================

np.random.seed(SEED)

valeurs_n = [10, 100, 1000, 10000]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(
    f"Distribution Uniforme Discrète – Entiers entre {MIN_VAL} et {MAX_VAL}\n(Seed = {SEED})",
    fontsize=14, fontweight='bold'
)

axes = axes.flatten()

for i, n in enumerate(valeurs_n):
    data = np.random.randint(MIN_VAL, MAX_VAL + 1, size=n)

    bins = np.arange(MIN_VAL - 0.5, MAX_VAL + 1.5, 1)
    axes[i].hist(data, bins=bins, color='steelblue', edgecolor='black', density=True)
    axes[i].set_title(f"n = {n}")
    axes[i].set_xlabel("Valeur")
    axes[i].set_ylabel("Fréquence relative")
    axes[i].set_xlim(MIN_VAL - 1, MAX_VAL + 1)

    # Ligne théorique
    prob_theorique = 1 / (MAX_VAL - MIN_VAL + 1)
    axes[i].axhline(
        y=prob_theorique,
        color='red', linestyle='--', linewidth=1.5,
        label=f"Théorique = {prob_theorique:.4f}"
    )
    axes[i].legend(fontsize=8)

    print(f"\n--- n = {n} ---")
    print(data)

plt.tight_layout()
plt.savefig("uniforme_discrete.png", dpi=150)
plt.show()
print("\nFigure sauvegardée : uniforme_discrete.png")
