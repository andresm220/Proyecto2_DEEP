"""Galería: 200 candidatos -> k-means (k=10) -> revisión visual -> 10 personajes, y regeneración exacta.

    1) python -m src.gallery --candidates --weights W      -> galeria/candidatos.json, galeria/candidatos/cluster_XX.png,
                                                             galeria/picks.json (por defecto: el más cercano al centroide)
    2) Revisión visual: mirar las hojas cluster_XX.png y, si hace falta, cambiar en picks.json la seed elegida de
       cada cluster por la más legible según la vara del universo (figura completa, atuendo coherente, distinta).
    3) python -m src.gallery --select --weights W          -> galeria/personaje_XX.png (+ _256), galeria.png,
                                                             seeds.json, z_vectors.pt
    4) python -m src.gallery --regenerate --weights W --seeds galeria/seeds.json   (verificación píxel a píxel)

Todo se genera en CPU, con G.eval() y una muestra a la vez: así la regeneración es exacta (ver README).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .evaluate import FeatureExtractor, generate, load_generator
from .utils import to_uint8

K = 10
CRITERIO = ("k-means (k=10, random_state=42) sobre features ResNet-18 (normalizadas) de los candidatos; de cada "
            "cluster se eligió por revisión visual la muestra más legible según la vara del universo (figura "
            "humanoide completa, atuendo coherente con una clase del reino, distinta de las otras 9). Por defecto, "
            "la más cercana al centroide.")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_png_array(img: torch.Tensor) -> np.ndarray:
    """[-1,1] (3,H,W) -> uint8 (H,W,3), exactamente lo que se guarda en el PNG."""
    return to_uint8(img).permute(1, 2, 0).numpy()


def contact_sheet(imgs, seeds, path, title, cols=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = int(np.ceil(len(imgs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.2 * rows), dpi=120, squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, img, s in zip(axes.flat, imgs, seeds):
        ax.imshow(to_png_array(img), interpolation="nearest")
        ax.set_title(f"seed {s}", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ 1) candidatos
def make_candidates(weights, n: int, out: Path) -> dict:
    from sklearn.cluster import KMeans
    if n < K:
        raise SystemExit(f"--n {n}: hacen falta al menos k={K} candidatos")
    G, cfg = load_generator(weights, "cpu")
    seeds = list(range(n))
    imgs, _ = generate(G, seeds, cfg["nz"])
    feats = FeatureExtractor("cpu")(imgs)
    feats = torch.nn.functional.normalize(feats, dim=1).numpy()  # normalizadas: k-means ~ distancia coseno
    km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(feats)
    dist = np.linalg.norm(feats - km.cluster_centers_[km.labels_], axis=1)

    clusters = {}
    (out / "candidatos").mkdir(parents=True, exist_ok=True)
    for c in range(K):
        members = sorted(np.flatnonzero(km.labels_ == c), key=lambda i: dist[i])
        clusters[str(c)] = [int(seeds[i]) for i in members]
        shown = members[:18]
        contact_sheet(imgs[shown], [seeds[i] for i in shown], out / "candidatos" / f"cluster_{c:02d}.png",
                      f"Cluster {c} ({len(members)} muestras), ordenadas por cercanía al centroide")
    meta = {"weights_file": Path(weights).name, "weights_sha256": sha256(weights), "n_candidates": n,
            "seeds": f"0-{n - 1}", "k": K, "clusters": clusters}
    (out / "candidatos.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    picks = out / "picks.json"
    if not picks.exists():
        picks.write_text(json.dumps({c: m[0] for c, m in clusters.items()}, indent=2), encoding="utf-8")
    return meta


# ------------------------------------------------------------------ 2) selección
def select(weights, out: Path) -> dict:
    cand = json.loads((out / "candidatos.json").read_text(encoding="utf-8"))
    if sha256(weights) != cand["weights_sha256"]:
        raise SystemExit("Los candidatos se generaron con otros pesos: vuelve a correr --candidates.")
    picks = json.loads((out / "picks.json").read_text(encoding="utf-8"))
    for c, s in picks.items():
        if s not in cand["clusters"][c]:
            raise SystemExit(f"picks.json: la seed {s} no pertenece al cluster {c}")

    old = {}
    if (out / "seeds.json").exists():  # conservar nombres/clases escritos a mano
        old = {ch["seed"]: ch for ch in json.loads((out / "seeds.json").read_text(encoding="utf-8"))["characters"]}

    G, cfg = load_generator(weights, "cpu")
    order = sorted(picks, key=int)
    seeds = [int(picks[c]) for c in order]
    imgs, z = generate(G, seeds, cfg["nz"])
    chars = []
    for i, (c, s, img) in enumerate(zip(order, seeds, imgs)):
        f64, f256 = f"personaje_{i:02d}.png", f"personaje_{i:02d}_256.png"
        Image.fromarray(to_png_array(img)).save(out / f64)
        Image.fromarray(to_png_array(img)).resize((256, 256), Image.NEAREST).save(out / f256)
        prev = old.get(s, {})
        chars.append({"file": f64, "file_256": f256, "seed": s, "cluster": int(c),
                      "nombre": prev.get("nombre", ""), "clase": prev.get("clase", "")})
    torch.save(z, out / "z_vectors.pt")
    meta = {"weights_file": cand["weights_file"], "weights_sha256": cand["weights_sha256"],
            "generacion": "G.eval(), CPU, batch 1, z_i = torch.randn(1, nz, 1, 1, generator=torch.Generator().manual_seed(seed))",
            "n_candidates": cand["n_candidates"], "seeds_candidatos": cand["seeds"],
            "tasa_seleccion": f"{K} de {cand['n_candidates']}", "criterio": CRITERIO, "characters": chars}
    (out / "seeds.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    gallery_figure(imgs, chars, out / "galeria.png")
    return meta


def gallery_figure(imgs, chars, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.8), dpi=150)
    for ax, img, ch in zip(axes.flat, imgs, chars):
        ax.imshow(to_png_array(img), interpolation="nearest")
        label = ch["nombre"] or ch["file"].split(".")[0]
        ax.set_title(f"{label}\n{ch['clase']}  (seed {ch['seed']})" if ch["clase"] else f"{label}\n(seed {ch['seed']})",
                     fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ 3) regeneración
def regenerate(weights_path, seeds_json) -> dict:
    """Reconstruye la galería desde los pesos + seeds y la compara píxel a píxel con los PNG guardados."""
    seeds_json = Path(seeds_json)
    out = seeds_json.parent
    meta = json.loads(seeds_json.read_text(encoding="utf-8"))
    G, cfg = load_generator(weights_path, "cpu")
    chars = meta["characters"]
    imgs, z = generate(G, [c["seed"] for c in chars], cfg["nz"])
    mismatches = {}
    for c, img in zip(chars, imgs):
        saved = np.array(Image.open(out / c["file"]).convert("RGB"))
        mismatches[c["file"]] = int((saved != to_png_array(img)).any(axis=2).sum())
    z_ok = (out / "z_vectors.pt").exists() and torch.equal(torch.load(out / "z_vectors.pt"), z)
    weights_ok = sha256(weights_path) == meta["weights_sha256"]
    return {"ok": weights_ok and z_ok and all(v == 0 for v in mismatches.values()),
            "weights_ok": weights_ok, "z_ok": z_ok, "mismatches": mismatches}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", action="store_true")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="galeria")
    ap.add_argument("--seeds", default=None, help="seeds.json para --regenerate (por defecto <out>/seeds.json)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.candidates:
        meta = make_candidates(args.weights, args.n, out)
        print(f"{args.n} candidatos en {K} clusters -> {out / 'candidatos'}; editar {out / 'picks.json'} y correr --select")
        print({c: len(m) for c, m in meta["clusters"].items()})
    if args.select:
        meta = select(args.weights, out)
        print(f"Galería: {meta['tasa_seleccion']} -> {out}")
    if args.regenerate:
        r = regenerate(args.weights, args.seeds or out / "seeds.json")
        print(json.dumps(r, indent=2))
        print("OK: la galería se regenera exacta" if r["ok"] else "FALLO: la galería NO coincide")
        raise SystemExit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
