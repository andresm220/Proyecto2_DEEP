"""Dataset TinyHero (sprites LPC 64x64, 4 vistas).

Fuente: https://github.com/AgaMiko/pixel_character_generator (archivo data.zip).
Licencia: CC-BY-SA 3.0 y GNU GPL 3.0 (arte del Liberated Pixel Cup).

Uso como script (descarga + estadísticas + rejilla de muestras):
    python -m src.data --root data --out outputs
"""
import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import save_grid

DATA_URL = "https://github.com/AgaMiko/pixel_character_generator/raw/master/data.zip"

# Las carpetas 0..3 del zip son las 4 vistas, en el orden de las filas de la hoja LPC
# (arriba, izquierda, abajo, derecha). Verificado en outputs/figs/muestras_por_vista.png.
# Ojo: el archivo k.png es el MISMO personaje en las 4 carpetas -> 912 personajes únicos x 4 vistas.
VIEW_NAMES = {0: "espalda", 1: "izquierda", 2: "frente", 3: "derecha"}

# Fondo sobre el que se componen los PNG transparentes (la GAN genera RGB, no aprende un canal alfa).
# Por defecto blanco, pero el control usa NEGRO (`background: black` en configs/base.yaml): con fondo blanco
# G colapsaba a un solo personaje en todas las configuraciones probadas (ver README, "Problemas encontrados").
BACKGROUND = (255, 255, 255)
BACKGROUNDS = {"white": (255, 255, 255), "black": (0, 0, 0)}


def download_tinyhero(root="data") -> Path:
    """Descarga y descomprime TinyHero en `root/tinyhero/`. Devuelve la carpeta con las vistas 0..3."""
    root = Path(root)
    img_dir = root / "tinyhero" / "data"
    if img_dir.exists() and len(list(img_dir.glob("*/*.png"))) == 3648:
        return img_dir
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "data.zip"
    if not zip_path.exists():
        print(f"Descargando {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(root / "tinyhero")
    return img_dir


def load_rgb(path, size: int = 64, background=BACKGROUND) -> np.ndarray:
    """PNG RGBA -> composición sobre fondo fijo -> RGB uint8 (H, W, 3)."""
    im = Image.open(path).convert("RGBA")
    if im.size != (size, size):
        # NEAREST y no bilineal: el bilineal mezcla píxeles vecinos y emborrona el pixel art.
        im = im.resize((size, size), Image.NEAREST)
    bg = Image.new("RGBA", im.size, tuple(background) + (255,))
    return np.asarray(Image.alpha_composite(bg, im).convert("RGB"))


class TinyHero(Dataset):
    """Todo el dataset cabe en memoria (3648 x 64 x 64 x 3 uint8 ≈ 45 MB), así que se carga una sola vez
    y cada __getitem__ solo normaliza y, opcionalmente, voltea. Devuelve (imagen en [-1, 1], vista)."""

    def __init__(self, root="data", image_size: int = 64, hflip: bool = True, download: bool = True,
                 background=BACKGROUND):
        img_dir = download_tinyhero(root) if download else Path(root) / "tinyhero" / "data"
        files = sorted(img_dir.glob("*/*.png"), key=lambda p: (int(p.parent.name), int(p.stem)))
        if not files:
            raise FileNotFoundError(f"No hay PNG en {img_dir}")
        self.files = files
        self.views = torch.tensor([int(p.parent.name) for p in files])
        arr = np.stack([load_rgb(p, image_size, background) for p in files])        # (N, H, W, 3) uint8
        self.images = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # (N, 3, H, W)
        # Flip horizontal p=0.5. Válido en el modelo INCONDICIONAL: un sprite mirando a la izquierda
        # volteado es un sprite plausible mirando a la derecha. Si se condicionara por vista,
        # el flip mezclaría etiquetas izquierda/derecha y habría que desactivarlo.
        self.hflip = hflip
        self.background = tuple(background)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        # [0, 255] -> [-1, 1], porque el generador termina en tanh (rango [-1, 1]).
        x = self.images[i].float() / 127.5 - 1.0
        if self.hflip and torch.rand(1).item() < 0.5:
            x = x.flip(-1)
        return x, self.views[i]

    def stats(self) -> dict:
        counts = torch.bincount(self.views, minlength=4).tolist()
        return {
            "n_images": len(self),
            "n_per_view": {VIEW_NAMES[v]: counts[v] for v in range(4)},
            "resolution": list(self.images.shape[-2:]),
            "channels": "RGB (PNG RGBA compuesto sobre fondo "
                        f"{'negro' if self.background == (0, 0, 0) else 'blanco' if self.background == (255, 255, 255) else self.background})",
            "normalization": "[-1, 1]  (x / 127.5 - 1), coincide con la salida tanh de G",
            "augmentations": ["RandomHorizontalFlip(p=0.5)"] if self.hflip else [],
            "resize": "ninguno necesario (todas son 64x64); si hiciera falta: NEAREST",
            "source": DATA_URL,
            "license": "CC-BY-SA 3.0 y GNU GPL 3.0 (Liberated Pixel Cup)",
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--background", default="black", choices=list(BACKGROUNDS), help="el mismo que configs/base.yaml")
    args = ap.parse_args()

    ds = TinyHero(args.root, hflip=False, background=BACKGROUNDS[args.background])
    out = Path(args.out)
    (out / "figs").mkdir(parents=True, exist_ok=True)

    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(ds), generator=g)[:64]
    save_grid(torch.stack([ds[i][0] for i in idx]), out / "figs" / "muestras_dataset.png", nrow=8, scale=3)

    # Una fila por vista: sirve para verificar el mapeo carpeta -> nombre de vista.
    per_view = [ds[int(i)][0] for v in range(4) for i in torch.nonzero(ds.views == v).flatten()[:8]]
    save_grid(torch.stack(per_view), out / "figs" / "muestras_por_vista.png", nrow=8, scale=3)

    # Las estadísticas se calculan sin flip, pero en entrenamiento sí se usa (config augment_hflip).
    stats = {**ds.stats(), "augmentations": ["RandomHorizontalFlip(p=0.5) (solo en entrenamiento)"],
             "n_unique_characters": len(ds) // 4}
    with open(out / "data_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
