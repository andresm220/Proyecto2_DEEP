"""Figuras comparativas entre runs (mismo ruido fijo): rejillas lado a lado y curvas superpuestas.

    python -m src.compare --runs base exp_hinge --out outputs/figs/cmp_base_vs_hinge
    -> outputs/figs/cmp_base_vs_hinge_grids.png y outputs/figs/cmp_base_vs_hinge_curves.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .utils import moving_average, read_losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--out", required=True, help="prefijo de salida (sin extensión)")
    ap.add_argument("--window", type=int, default=50)
    args = ap.parse_args()
    runs = Path(args.out_root) / "runs"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Como todos los runs usan el mismo fixed_noise.pt, la celda (i, j) de cada rejilla viene del mismo z:
    # las diferencias entre paneles se deben solo al cambio de configuración.
    fig, axes = plt.subplots(1, len(args.runs), figsize=(4 * len(args.runs), 4.3), dpi=150)
    for ax, name in zip(np.atleast_1d(axes), args.runs):
        ax.imshow(Image.open(runs / name / "fixed_grid.png"), interpolation="nearest")
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle("Mismo ruido fijo (fixed_noise.pt), última época")
    fig.tight_layout()
    fig.savefig(f"{args.out}_grids.png")
    plt.close(fig)

    # Solo medias móviles: los valores crudos por iteración de varios runs superpuestos son ilegibles.
    keys = [("loss_D", "Pérdida D"), ("loss_G", "Pérdida G"), ("D_x", "D(x)"), ("D_G_z", "D(G(z))")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8), dpi=150)
    for name in args.runs:
        d = read_losses(runs / name / "losses.csv")
        for ax, (k, title) in zip(axes, keys):
            ma = moving_average(d[k], args.window)
            ax.plot(np.arange(len(d[k]) - len(ma), len(d[k])), ma, label=name)
            ax.set_title(title)
            ax.set_xlabel("iteración")
            ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(f"{args.out}_curves.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
