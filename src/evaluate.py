"""Evaluación de un generador: diversidad, novedad y vecinos más cercanos (sección 4.3 de la spec).

    python -m src.evaluate --run outputs/runs/base                  # -> metrics.json + vecinos_muestra.png
    python -m src.evaluate --table base exp_hinge exp_sn            # -> outputs/comparacion.csv
    python -m src.evaluate --gallery galeria/seeds.json --weights W # -> galeria/vecinos*.png + vecinos.json

Todas las medidas usan features de ResNet-18 preentrenada en ImageNet (salida del avgpool, 512-d).
No es un "juez" perfecto para pixel art, pero da una distancia perceptual razonable y es la misma para todos los runs.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import BACKGROUNDS, VIEW_NAMES, TinyHero
from .models import Generator

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ------------------------------------------------------------------ generación
def load_generator(weights_path, device="cpu"):
    """Carga G desde un checkpoint del contrato (usa su `config` para la arquitectura). Devuelve G en eval."""
    ck = torch.load(weights_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    G = Generator(cfg["nz"], cfg["ngf"])
    G.load_state_dict(ck["G"])
    return G.to(device).eval(), cfg


def z_from_seed(seed: int, nz: int = 100) -> torch.Tensor:
    """Contrato de la galería: la muestra i usa torch.Generator().manual_seed(i). Así cada personaje es un entero."""
    return torch.randn(1, nz, 1, 1, generator=torch.Generator().manual_seed(int(seed)))


@torch.no_grad()
def generate(G, seeds, nz: int = 100, batch_size: int = 1):
    """Genera una imagen por seed con G.eval(). Devuelve (imágenes [-1,1] en CPU, z).
    batch_size=1 por defecto: en GPU, cuDNN puede elegir otro algoritmo según el tamaño del batch y cambiar
    algún píxel al redondear; con batch 1 (y en CPU) la galería se regenera exacta."""
    G.eval()
    device = next(G.parameters()).device
    z = torch.cat([z_from_seed(s, nz) for s in seeds])
    out = [G(z[i:i + batch_size].to(device)).cpu() for i in range(0, len(z), batch_size)]
    return torch.cat(out), z


# ------------------------------------------------------------------ features
class FeatureExtractor:
    def __init__(self, device="cpu"):
        from torchvision.models import ResNet18_Weights, resnet18
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()  # quitar la capa de clasificación: queda la salida del avgpool (512-d)
        self.model = model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
        feats = []
        for i in range(0, len(images), batch_size):
            x = (images[i:i + batch_size].to(self.device) + 1) / 2          # [-1,1] -> [0,1]
            # 64 -> 224 con NEAREST: cada píxel se vuelve un bloque, sin inventar colores intermedios.
            x = F.interpolate(x, size=224, mode="nearest")
            x = (x - IMAGENET_MEAN.to(self.device)) / IMAGENET_STD.to(self.device)
            feats.append(self.model(x).cpu())
        return torch.cat(feats)


def train_dataset_for(cfg: dict, data_root: str) -> TinyHero:
    """El train con el MISMO fondo con el que se entrenó el modelo (config `background`). Si no coincidiera,
    un personaje sobre negro comparado con reales sobre blanco parecería 'novedoso' solo por el fondo."""
    return TinyHero(data_root, hflip=False, background=BACKGROUNDS[cfg.get("background", "white")])


def train_features(ds: TinyHero, extractor: FeatureExtractor, cache_path=None) -> torch.Tensor:
    """Features de las 3648 imágenes reales (sin flip). Se cachean: tardan ~1 min en CPU y no cambian."""
    if cache_path is not None and Path(cache_path).exists():
        feats = torch.load(cache_path)
        if len(feats) == len(ds):
            return feats
    imgs = ds.images.float() / 127.5 - 1
    feats = extractor(imgs)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(feats, cache_path)
    return feats


# ------------------------------------------------------------------ métricas
def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.normalize(a, dim=1) @ F.normalize(b, dim=1).T


def diversity(feats: torch.Tensor) -> float:
    """Distancia coseno media entre todos los pares distintos. Más alta = más diverso.
    Si G colapsa a pocos modos, muchas muestras se parecen y este número baja."""
    d = 1 - cosine_similarity_matrix(feats, feats)
    n = len(feats)
    return ((d.sum() - d.diagonal().sum()) / (n * (n - 1))).item()


def nearest_neighbors(query: torch.Tensor, ref: torch.Tensor, k: int = 3):
    """Top-k vecinos por similitud coseno. Devuelve (índices [Q,k], similitudes [Q,k])."""
    sim, idx = cosine_similarity_matrix(query, ref).topk(k, dim=1)
    return idx, sim


def novelty(query: torch.Tensor, ref: torch.Tensor) -> float:
    """Distancia coseno media al vecino más cercano del train. Muy baja = posible memorización."""
    _, sim = nearest_neighbors(query, ref, k=1)
    return (1 - sim[:, 0]).mean().item()


def pixel_neighbors(query: torch.Tensor, ref: torch.Tensor, k: int = 3):
    """Respaldo sin red neuronal: distancia L2 en píxeles. Detecta copias casi exactas que las features
    de ImageNet podrían no distinguir. Devuelve (índices [Q,k], distancias [Q,k])."""
    # Sin el atajo por multiplicación de matrices de cdist: pierde precisión y una copia exacta no daría 0.
    d = torch.cdist(query.flatten(1).float(), ref.flatten(1).float(),
                    compute_mode="donot_use_mm_for_euclid_dist")
    dist, idx = d.topk(k, dim=1, largest=False)
    return idx, dist


def describe_train_index(ds: TinyHero, i: int) -> str:
    """Índice del dataset -> 'personaje k (vista)'. k.png es el mismo personaje en las 4 vistas, así que un
    vecino 'casi idéntico' puede ser otra vista del mismo personaje: conviene verlo en la figura."""
    p = ds.files[int(i)]
    return f"personaje {int(p.stem)} ({VIEW_NAMES[int(p.parent.name)]})"


# ------------------------------------------------------------------ figura de vecinos
def neighbors_figure(gen, train_imgs, idx, scores, labels, path, title, score_name="sim", row_names=None):
    """Estilo Fig. 2 de Goodfellow et al. (2014): fila = muestra generada | sus top-k vecinos del train."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    to_img = lambda x: ((x.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()
    n, k = idx.shape
    fig, axes = plt.subplots(n, k + 1, figsize=(2.1 * (k + 1), 2.3 * n), dpi=150, squeeze=False)
    for r in range(n):
        axes[r, 0].imshow(to_img(gen[r]), interpolation="nearest")
        axes[r, 0].set_title(row_names[r] if row_names else "generado", fontsize=8, fontweight="bold")
        for c in range(k):
            j = int(idx[r, c])
            axes[r, c + 1].imshow(to_img(train_imgs[j]), interpolation="nearest")
            axes[r, c + 1].set_title(f"{labels[j]}\n{score_name}={float(scores[r, c]):.3f}", fontsize=7)
        for ax in axes[r]:
            ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ modos de la CLI
def evaluate_run(run_dir: Path, ckpt: str, n: int, data_root: str, device: str) -> dict:
    out_root = run_dir.parent.parent
    G, cfg = load_generator(run_dir / "checkpoints" / ckpt, device)
    ds = train_dataset_for(cfg, data_root)
    ext = FeatureExtractor(device)
    ref = train_features(ds, ext, out_root / f"train_feats_{cfg.get('background', 'white')}.pt")

    # Semillas 0..n-1 fijas: los mismos z para todos los runs. Aquí se puede usar batch grande (solo métricas).
    imgs, _ = generate(G, range(n), cfg["nz"], batch_size=100)
    feats = ext(imgs)
    _, sim = nearest_neighbors(feats, ref, k=1)
    nov = 1 - sim[:, 0]
    # Referencia: diversidad de n imágenes reales al azar (semilla 0). Sirve para leer el número de G.
    real_idx = torch.randperm(len(ref), generator=torch.Generator().manual_seed(0))[:n]
    m = {
        "run": cfg["name"], "checkpoint": ckpt, "epoch": torch.load(run_dir / "checkpoints" / ckpt,
                                                                  map_location="cpu", weights_only=False)["epoch"],
        "n_samples": n, "seeds": f"0-{n - 1}",
        "diversity": diversity(feats), "train_diversity": diversity(ref[real_idx]),
        "novelty": nov.mean().item(), "novelty_p05": nov.quantile(0.05).item(), "novelty_min": nov.min().item(),
        "features": "ResNet-18 ImageNet avgpool 512-d, 64->224 NEAREST", "distance": "coseno",
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)

    labels = [describe_train_index(ds, i) for i in range(len(ds))]
    train_imgs = ds.images.float() / 127.5 - 1
    idx, s = nearest_neighbors(feats[:8], ref, k=3)
    neighbors_figure(imgs[:8], train_imgs, idx, s, labels, run_dir / "vecinos_muestra.png",
                     f"{cfg['name']}: seeds 0-7 y sus 3 vecinos más cercanos (coseno, ResNet-18)",
                     row_names=[f"seed {i}" for i in range(8)])
    return m


def comparison_table(out_root: Path, runs) -> Path:
    rows = []
    for name in runs:
        rd = out_root / "runs" / name
        m = json.loads((rd / "metrics.json").read_text(encoding="utf-8"))
        cfg = torch.load(rd / "checkpoints" / m["checkpoint"], map_location="cpu", weights_only=False)["config"]
        with open(rd / "losses.csv", newline="") as f:
            losses = list(csv.DictReader(f))
        last_epoch = max(int(r["epoch"]) for r in losses)
        last = [r for r in losses if int(r["epoch"]) == last_epoch]
        mean = lambda k: float(np.mean([float(r[k]) for r in last]))
        rows.append({"run": name, "loss": cfg["loss"], "spectral_norm_d": cfg["spectral_norm_d"],
                     "epochs": m["epoch"], "diversity": round(m["diversity"], 4),
                     "train_diversity": round(m["train_diversity"], 4), "novelty": round(m["novelty"], 4),
                     "novelty_p05": round(m["novelty_p05"], 4),
                     "loss_D_ult_epoca": round(mean("loss_D"), 4), "loss_G_ult_epoca": round(mean("loss_G"), 4),
                     "D_x_ult_epoca": round(mean("D_x"), 4), "D_G_z_ult_epoca": round(mean("D_G_z"), 4)})
    path = out_root / "comparacion.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def gallery_neighbors(seeds_json: Path, weights: str, data_root: str, device: str, out_root: Path,
                      k: int = 3) -> dict:
    meta = json.loads(Path(seeds_json).read_text(encoding="utf-8"))
    chars = meta["characters"]
    G, cfg = load_generator(weights, "cpu")
    imgs, _ = generate(G, [c["seed"] for c in chars], cfg["nz"])  # CPU, batch 1: las mismas que la galería
    ds = train_dataset_for(cfg, data_root)
    ext = FeatureExtractor(device)
    ref = train_features(ds, ext, out_root / f"train_feats_{cfg.get('background', 'white')}.pt")
    train_imgs = ds.images.float() / 127.5 - 1
    labels = [describe_train_index(ds, i) for i in range(len(ds))]
    names = [f"{c['file'].split('.')[0]} (seed {c['seed']})" for c in chars]
    out_dir = Path(seeds_json).parent

    idx, sim = nearest_neighbors(ext(imgs), ref, k)
    neighbors_figure(imgs, train_imgs, idx, sim, labels, out_dir / "vecinos.png",
                     "Galería: top-3 vecinos del train por similitud coseno (ResNet-18)", row_names=names)
    pidx, pdist = pixel_neighbors(imgs, train_imgs, k)
    # Distancia L2 normalizada a RMSE por píxel en escala [0,1], más fácil de leer que la suma cruda.
    rmse = pdist / (2 * np.sqrt(imgs[0].numel()))
    neighbors_figure(imgs, train_imgs, pidx, rmse, labels, out_dir / "vecinos_pixeles.png",
                     "Galería: top-3 vecinos del train por distancia en píxeles (RMSE)", score_name="rmse",
                     row_names=names)
    report = {c["file"]: {"seed": c["seed"],
                          "coseno": [{"vecino": labels[int(j)], "sim": round(float(s), 4)} for j, s in zip(idx[r], sim[r])],
                          "pixeles": [{"vecino": labels[int(j)], "rmse": round(float(s), 4)} for j, s in zip(pidx[r], rmse[r])]}
              for r, c in enumerate(chars)}
    with open(out_dir / "vecinos.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="carpeta de un run: calcula metrics.json")
    ap.add_argument("--ckpt", default="last.pt")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--table", nargs="+", help="runs a juntar en comparacion.csv")
    ap.add_argument("--gallery", help="galeria/seeds.json: vecinos más cercanos de los personajes")
    ap.add_argument("--weights", help="pesos del modelo final (para --gallery)")
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.run:
        print(json.dumps(evaluate_run(Path(args.run), args.ckpt, args.n, args.data_root, device), indent=2))
    if args.table:
        print("Escrito", comparison_table(Path(args.out_root), args.table))
    if args.gallery:
        if not args.weights:
            raise SystemExit("--gallery necesita --weights")
        print(json.dumps(gallery_neighbors(Path(args.gallery), args.weights, args.data_root, device,
                                           Path(args.out_root)),
                         indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
