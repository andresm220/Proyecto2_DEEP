"""Tabla capa por capa de G y D (forma de salida y parámetros) para la diapositiva de arquitectura.

    python scripts/architecture_table.py   ->  outputs/figs/arquitectura.md
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import build_models  # noqa: E402
from src.utils import load_config  # noqa: E402


def layer_table(model, x):
    rows, hooks = [], []
    for layer in model.net:
        hooks.append(layer.register_forward_hook(
            lambda m, i, o: rows.append((type(m).__name__, tuple(o.shape[1:]),
                                         sum(p.numel() for p in m.parameters())))))
    with torch.no_grad():
        model.eval()(x)
    for h in hooks:
        h.remove()
    lines = ["| Capa | Salida (C×H×W) | Parámetros |", "|---|---|---|"]
    lines += [f"| {n} | {'×'.join(map(str, s))} | {p:,} |" for n, s, p in rows]
    lines.append(f"| **Total** | | **{sum(r[2] for r in rows):,}** |")
    return "\n".join(lines)


def main():
    cfg = load_config("configs/base.yaml")
    G, D = build_models(cfg)
    out = ["# Arquitectura DCGAN (configuración base)", "",
           "## Generador", layer_table(G, torch.randn(1, cfg["nz"], 1, 1)), "",
           "## Discriminador", layer_table(D, torch.randn(1, 3, 64, 64)), "",
           "## Configuración base (control)", "", "| Clave | Valor |", "|---|---|"]
    out += [f"| `{k}` | `{v}` |" for k, v in cfg.items()]
    Path("outputs/figs").mkdir(parents=True, exist_ok=True)
    Path("outputs/figs/arquitectura.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
