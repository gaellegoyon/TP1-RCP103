"""
RCP103 - TP Génération de variables aléatoires
Groupe 2 – Génération des histogrammes

Génère, pour chaque distribution et chaque valeur de n,
un histogramme sauvegardé en PNG dans le dossier ./figures/.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from distributions import DISTRIBUTIONS, SEED

# ── Configuration ──────────────────────────────────────────────────────────────
N_VALUES    = [10, 100, 1_000, 10_000]
OUTPUT_DIR  = "figures"
DPI         = 150
FIGSIZE     = (6, 4)

# Colour palette (one colour per n value)
PALETTE = {
    10:      "#4C72B0",
    100:     "#DD8452",
    1_000:   "#55A868",
    10_000:  "#C44E52",
}
# ──────────────────────────────────────────────────────────────────────────────


def _n_bins(data: np.ndarray, discrete: bool) -> int | np.ndarray:
    """Choose sensible bin edges/count depending on the distribution type."""
    if discrete:
        # One bar per integer value present in the data
        lo, hi = int(data.min()), int(data.max())
        return np.arange(lo, hi + 2) - 0.5   # centred on integers
    # Sturges' rule, capped between 10 and 60
    return max(10, min(60, int(np.ceil(np.log2(len(data)) + 1))))


def plot_histogram(
    data: np.ndarray,
    title: str,
    xlabel: str,
    n: int,
    discrete: bool,
    filepath: str,
) -> None:
    """Plot and save a single histogram."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    bins  = _n_bins(data, discrete)
    color = PALETTE[n]

    ax.hist(data, bins=bins, color=color, edgecolor="white",
            linewidth=0.4, density=True, alpha=0.85)

    # Axes labels & title
    ax.set_title(f"{title}  –  n = {n:,}", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Densité de probabilité", fontsize=9)

    # Light grid on y-axis only
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    # Descriptive stats as a text box
    stats = (
        f"n = {n:,}\n"
        f"μ̂ = {data.mean():.4f}\n"
        f"σ̂ = {data.std():.4f}\n"
        f"min = {data.min():.4f}\n"
        f"max = {data.max():.4f}"
    )
    ax.text(
        0.98, 0.97, stats,
        transform=ax.transAxes,
        fontsize=7.5, verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.85),
    )

    fig.tight_layout()
    fig.savefig(filepath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  Saved: {filepath}")


def plot_all_n_overlay(
    name: str,
    dist_info: dict,
    filepath: str,
) -> None:
    """One figure with four subplots (one per n) for a given distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(
        f"{dist_info['label']}  –  Seed = {SEED}",
        fontsize=13, fontweight="bold", y=1.01,
    )

    generator = dist_info["generator"]
    discrete  = dist_info["discrete"]

    for ax, n in zip(axes.flat, N_VALUES):
        data  = generator(n)
        bins  = _n_bins(data, discrete)
        color = PALETTE[n]

        ax.hist(data, bins=bins, color=color, edgecolor="white",
                linewidth=0.4, density=True, alpha=0.85)
        ax.set_title(f"n = {n:,}", fontsize=10, fontweight="semibold")
        ax.set_xlabel(dist_info["label"].split("(")[0].strip(), fontsize=8)
        ax.set_ylabel("Densité", fontsize=8)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7)

        stats = f"μ̂={data.mean():.3f}  σ̂={data.std():.3f}"
        ax.set_title(f"n = {n:,}   ({stats})", fontsize=9)

    fig.tight_layout()
    fig.savefig(filepath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  Saved overlay: {filepath}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for dist_name, dist_info in DISTRIBUTIONS.items():
        print(f"\n── {dist_name} ──")

        generator = dist_info["generator"]
        discrete  = dist_info["discrete"]
        safe_name = dist_name.lower().replace(" ", "_").replace("é", "e").replace("è", "e")

        # Individual histogram for each n
        for n in N_VALUES:
            data     = generator(n)
            filename = f"{safe_name}_n{n}.png"
            filepath = os.path.join(OUTPUT_DIR, filename)
            plot_histogram(
                data      = data,
                title     = dist_info["label"],
                xlabel    = dist_name,
                n         = n,
                discrete  = discrete,
                filepath  = filepath,
            )

        # Overlay (2×2) figure
        overlay_path = os.path.join(OUTPUT_DIR, f"{safe_name}_overlay.png")
        plot_all_n_overlay(dist_name, dist_info, overlay_path)

    print("\n✅  Tous les histogrammes ont été générés dans le dossier « figures/ ».")


if __name__ == "__main__":
    main()
