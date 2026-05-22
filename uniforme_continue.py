import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

# ============================================================
# PARAMÈTRES – À ADAPTER SELON TON GROUPE
# ============================================================
SEED = 1          # Remplace par ton numéro de groupe
MIN_VAL = 0.0     # Remplace par le min de ton groupe
MAX_VAL = 1.0     # Remplace par le max de ton groupe
# ============================================================

np.random.seed(SEED)

valeurs_n = [10, 100, 1000, 10000]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(
    f"Distribution Uniforme Continue – Réels entre {MIN_VAL} et {MAX_VAL}\n(Seed = {SEED})",
    fontsize=14, fontweight='bold'
)

axes = axes.flatten()

for i, n in enumerate(valeurs_n):
    data = np.random.uniform(MIN_VAL, MAX_VAL, size=n)

    axes[i].hist(data, bins=30, color='mediumseagreen', edgecolor='black', density=True)
    axes[i].set_title(f"n = {n}")
    axes[i].set_xlabel("Valeur")
    axes[i].set_ylabel("Densité de probabilité")

    # Courbe théorique
    x = np.linspace(MIN_VAL - 0.1, MAX_VAL + 0.1, 300)
    axes[i].plot(
        x,
        uniform.pdf(x, loc=MIN_VAL, scale=MAX_VAL - MIN_VAL),
        'r--', linewidth=2, label="Densité théorique"
    )
    axes[i].legend(fontsize=8)

    print(f"\n--- n = {n} ---")
    print(data)

plt.tight_layout()
plt.savefig("uniforme_continue.png", dpi=150)
plt.show()
print("\nFigure sauvegardée : uniforme_continue.png")