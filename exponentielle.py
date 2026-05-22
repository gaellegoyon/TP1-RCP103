import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

# ============================================================
# PARAMÈTRES – À ADAPTER SELON TON GROUPE
# ============================================================
SEED = 1          # Remplace par ton numéro de groupe
MOYENNE = 1       # Moyenne e = 1/λ de ton groupe
# ============================================================

np.random.seed(SEED)

valeurs_n = [10, 100, 1000, 10000]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(
    f"Distribution Exponentielle – Moyenne e = {MOYENNE} (λ = {1/MOYENNE:.4f})\n(Seed = {SEED})",
    fontsize=14, fontweight='bold'
)

axes = axes.flatten()

for i, n in enumerate(valeurs_n):
    # scale = moyenne = 1/λ
    data = np.random.exponential(scale=MOYENNE, size=n)

    axes[i].hist(data, bins=30, color='mediumpurple', edgecolor='black', density=True)
    axes[i].set_title(f"n = {n}")
    axes[i].set_xlabel("Valeur")
    axes[i].set_ylabel("Densité de probabilité")

    # Courbe théorique
    x = np.linspace(0, data.max() * 1.1, 300)
    axes[i].plot(
        x,
        expon.pdf(x, scale=MOYENNE),
        'r--', linewidth=2, label=f"Exp(λ={1/MOYENNE:.4f})"
    )
    axes[i].legend(fontsize=8)

    print(f"\n--- n = {n} ---")
    print(data)

plt.tight_layout()
plt.savefig("exponentielle.png", dpi=150)
plt.show()
print("\nFigure sauvegardée : exponentielle.png")
