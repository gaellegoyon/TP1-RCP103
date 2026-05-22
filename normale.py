import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ============================================================
# PARAMÈTRES – À ADAPTER SELON TON GROUPE
# ============================================================
SEED = 1          # Remplace par ton numéro de groupe
MU = 0            # Moyenne µ de ton groupe
SIGMA = 1         # Écart-type σ de ton groupe
# ============================================================

np.random.seed(SEED)

valeurs_n = [10, 100, 1000, 10000]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(
    f"Distribution Normale – µ = {MU}, σ = {SIGMA}\n(Seed = {SEED})",
    fontsize=14, fontweight='bold'
)

axes = axes.flatten()

for i, n in enumerate(valeurs_n):
    data = np.random.normal(loc=MU, scale=SIGMA, size=n)

    axes[i].hist(data, bins=30, color='coral', edgecolor='black', density=True)
    axes[i].set_title(f"n = {n}")
    axes[i].set_xlabel("Valeur")
    axes[i].set_ylabel("Densité de probabilité")

    # Courbe théorique
    x = np.linspace(MU - 4 * SIGMA, MU + 4 * SIGMA, 300)
    axes[i].plot(
        x,
        norm.pdf(x, loc=MU, scale=SIGMA),
        'b-', linewidth=2, label=f"N({MU}, {SIGMA}²)"
    )
    axes[i].legend(fontsize=8)

    print(f"\n--- n = {n} ---")
    print(data)

plt.tight_layout()
plt.savefig("normale.png", dpi=150)
plt.show()
print("\nFigure sauvegardée : normale.png")