"""Utilidades compartidas: semillas, configs, rejillas de imágenes y gráficas de pérdida."""
import csv
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.utils import make_grid


# ---------------------------------------------------------------- reproducibilidad
def set_seed(seed: int) -> None:
    """Fija todas las fuentes de aleatoriedad (Python, NumPy, PyTorch CPU y GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN determinista: algo más lento, pero dos corridas con la misma semilla
    # quedan mucho más parecidas. Aun así, ConvTranspose en GPU no es 100 %
    # determinista; por eso la galería se regenera desde PESOS guardados, no re-entrenando.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- configs
def load_config(path) -> dict:
    """Carga un YAML. Si tiene `inherit: otro.yaml`, parte de ese archivo y sobrescribe
    solo las claves que define. Así exp_hinge.yaml contiene literalmente el único cambio."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    parent = cfg.pop("inherit", None)
    if parent is not None:
        base = load_config(path.parent / parent)
        base.update(cfg)
        cfg = base
    return cfg


def diff_configs(cfg: dict, base: dict) -> dict:
    """Claves en las que `cfg` difiere de `base` (sin contar `name`). Sirve para comprobar
    que un experimento cambia una sola cosa."""
    return {k: (base.get(k), v) for k, v in cfg.items() if k != "name" and base.get(k) != v}


# ---------------------------------------------------------------- ruido fijo
def get_fixed_noise(path, n: int = 16, nz: int = 100, seed: int = 1234) -> torch.Tensor:
    """Carga `fixed_noise.pt` si existe; si no, lo crea y lo guarda.
    Usa un generador local con la semilla del contrato: da exactamente los mismos valores que
    `torch.manual_seed(1234); torch.randn(16, 100, 1, 1)` pero sin tocar el RNG global."""
    path = Path(path)
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, nz, 1, 1, generator=g)
    if path.exists():
        # Verificar contra el contrato: un archivo hecho de otra forma (p. ej. con randn en CUDA) tendría la
        # misma forma pero otros valores, y las rejillas de distintos runs dejarían de ser comparables.
        loaded = torch.load(path)
        if loaded.shape != z.shape or not torch.equal(loaded, z):
            raise ValueError(f"{path} no coincide con torch.manual_seed({seed}); torch.randn({n}, {nz}, 1, 1). "
                             f"Bórralo para regenerarlo o copia el de Persona A.")
        return z
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(z, path)
    return z


# ---------------------------------------------------------------- imágenes
def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 255] uint8."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def grid_image(images: torch.Tensor, nrow: int = 4, scale: int = 4, padding: int = 2) -> Image.Image:
    """Rejilla de imágenes en [-1, 1] como PIL, ampliada con NEAREST para no emborronar el pixel art."""
    grid = make_grid(images.detach().cpu(), nrow=nrow, padding=padding, normalize=False, pad_value=1.0)
    arr = to_uint8(grid).permute(1, 2, 0).numpy()
    img = Image.fromarray(arr)
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    return img


def save_grid(images: torch.Tensor, path, nrow: int = 4, scale: int = 4) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid_image(images, nrow=nrow, scale=scale).save(path)


# ---------------------------------------------------------------- gráficas
def read_losses(csv_path) -> dict:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0].keys()}


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_losses(csv_path, out_path, window: int = 50, title: str | None = None) -> None:
    """Dos paneles: pérdidas de D y G, y D(x) vs D(G(z)). Línea tenue = valor crudo por iteración;
    línea fuerte = media móvil (las pérdidas de GAN son muy ruidosas por iteración)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = read_losses(csv_path)
    it = np.arange(len(d["loss_D"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
    for ax, keys, ylabel in [
        (axes[0], [("loss_D", "Pérdida D"), ("loss_G", "Pérdida G")], "pérdida"),
        (axes[1], [("D_x", "D(x) reales"), ("D_G_z", "D(G(z)) falsas")], "sigmoide(logit de D)"),
    ]:
        for key, label in keys:
            line, = ax.plot(it, d[key], alpha=0.2, linewidth=0.7)
            ma = moving_average(d[key], window)
            ax.plot(it[len(it) - len(ma):], ma, color=line.get_color(), linewidth=1.8, label=label)
        ax.set_xlabel("iteración")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def progress_figure(samples_dir, out_path, max_cols: int = 6) -> None:
    """Evolución de la rejilla de ruido fijo a lo largo del entrenamiento (diapositiva 6):
    toma snapshots equiespaciados de samples/epoch_XXX.png y los pone lado a lado."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(Path(samples_dir).glob("epoch_*.png"))
    if not files:
        return
    idx = np.unique(np.linspace(0, len(files) - 1, min(max_cols, len(files))).round().astype(int))
    files = [files[i] for i in idx]
    fig, axes = plt.subplots(1, len(files), figsize=(3 * len(files), 3.3), dpi=150)
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, files):
        ax.imshow(Image.open(f), interpolation="nearest")
        ax.set_title(f"época {int(f.stem.split('_')[1])}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
