"""Loop de entrenamiento genérico de la DCGAN, controlado por un YAML de configs/.

    python -m src.train --config configs/base.yaml
    python -m src.train --config configs/base.yaml --resume          # continúa desde last.pt
    python -m src.train --config configs/base.yaml --out-root /content/drive/MyDrive/proyecto2-gan/outputs

Salidas en <out-root>/runs/<name>/:
    checkpoints/epoch_XXX.pt, checkpoints/last.pt   (formato del contrato, sección 5)
    samples/epoch_XXX.png                           (rejilla de ruido fijo cada `snapshot_every`)
    losses.csv                                      (epoch, iter, loss_D, loss_G, D_x, D_G_z)
    fixed_grid.png, losses.png, progress.png, config.yaml, train_log.json
"""
import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .data import TinyHero
from .losses import get_loss
from .models import build_models
from .utils import (diff_configs, get_device, get_fixed_noise, get_rng_state, load_config,
                    plot_losses, progress_figure, save_grid, set_rng_state, set_seed)

CSV_FIELDS = ["epoch", "iter", "loss_D", "loss_G", "D_x", "D_G_z"]


def instance_noise_std(epoch: int, epochs: int, start: float) -> float:
    """Desviación del instance noise en una época (1-indexada): baja linealmente de `start` a 0.
    Se apaga al final para que G termine aprendiendo a producir imágenes nítidas, no ruidosas."""
    if start <= 0:
        return 0.0
    return max(0.0, start * (1 - (epoch - 1) / epochs))


def add_instance_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    """Instance noise (Sønderby et al. 2016; Arjovsky y Bottou 2017): ruido gaussiano a las entradas de D,
    reales y falsas por igual. El pixel art vive en una paleta discreta y G produce valores continuos: sin ruido,
    las dos distribuciones casi no se solapan y D gana trivialmente. Con ruido se solapan y el gradiente de G
    vuelve a ser informativo."""
    if std <= 0:
        return x
    return x + std * torch.randn_like(x)


def save_checkpoint(path: Path, **state) -> None:
    # Escribir a un temporal y renombrar: si Colab se cae a mitad de escritura,
    # last.pt anterior queda intacto en vez de corrupto.
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, device) -> dict:
    # Siempre a CPU: los estados RNG (ByteTensor) deben quedar en CPU o set_rng_state falla en GPU.
    # load_state_dict copia después los pesos (y el estado de Adam) al dispositivo de cada parámetro.
    return torch.load(path, map_location="cpu", weights_only=False)


@torch.no_grad()
def snapshot(G, fixed_noise, path):
    # Siempre G.eval(): BatchNorm usa sus estadísticas acumuladas, así la rejilla no depende del batch.
    G.eval()
    save_grid(G(fixed_noise), path, nrow=4, scale=4)
    G.train()


def truncate_csv(csv_path: Path, last_epoch: int) -> None:
    """Al reanudar, borra filas de épocas posteriores al checkpoint (se van a re-entrenar)."""
    if not csv_path.exists():
        return
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r["epoch"]) <= last_epoch]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true", help="continuar desde checkpoints/last.pt")
    ap.add_argument("--overwrite", action="store_true", help="borrar el run existente y empezar de cero")
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--data-root", default="data")
    # Solo para pruebas / medir tiempo por época. Para los experimentos reales NO usar:
    # base, hinge y SN deben correr con las épocas del YAML.
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-iters", type=int, default=None, help="cortar cada época tras N iteraciones")
    ap.add_argument("--name", default=None, help="sobrescribir el nombre del run")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.name is not None:
        cfg["name"] = args.name
    base_path = Path(args.config).parent / "base.yaml"
    if base_path.exists():
        print("Diferencias con base:", diff_configs(cfg, load_config(base_path)) or "ninguna")

    out_root = Path(args.out_root)
    run_dir = out_root / "runs" / cfg["name"]
    ckpt_dir, samples_dir = run_dir / "checkpoints", run_dir / "samples"
    # Validar antes de crear o escribir nada: un error aquí no debe dejar el run a medias.
    if args.resume and not (ckpt_dir / "last.pt").exists():
        raise SystemExit(f"--resume: no existe {ckpt_dir / 'last.pt'} (¿Drive montado? ¿out-root correcto?)")
    if not args.resume and (ckpt_dir / "last.pt").exists():
        if not args.overwrite:
            raise SystemExit(f"{run_dir} ya tiene checkpoints/last.pt. Usa --resume para continuar "
                             f"o --overwrite para borrarlo y empezar de cero.")
        shutil.rmtree(run_dir)  # sin restos de la corrida anterior (samples/epoch_XXX viejos, checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    set_seed(cfg["seed"])

    # ------------------------------------------------------------ datos
    ds = TinyHero(args.data_root, image_size=cfg["image_size"], hflip=cfg["augment_hflip"])
    loader_gen = torch.Generator().manual_seed(cfg["seed"])  # orden de barajado reproducible
    loader = DataLoader(
        ds, batch_size=cfg["batch_size"], shuffle=True, generator=loader_gen,
        # drop_last: 3648 = 28*128 + 64. Un último batch más chico cambia las estadísticas de
        # BatchNorm y el peso de ese paso; tirarlo cuesta 64 imágenes por época (otras salen en la siguiente).
        drop_last=True, num_workers=0, pin_memory=device.type == "cuda",
    )

    # ------------------------------------------------------------ modelos
    G, D = build_models(cfg)
    G, D = G.to(device), D.to(device)
    optG = torch.optim.Adam(G.parameters(), lr=cfg["lr_g"], betas=tuple(cfg["betas"]))
    optD = torch.optim.Adam(D.parameters(), lr=cfg["lr_d"], betas=tuple(cfg["betas"]))
    d_loss_fn, g_loss_fn = get_loss(cfg["loss"])

    # El MISMO ruido fijo para todos los runs: vive en <out-root>/fixed_noise.pt, no dentro del run.
    fixed_noise = get_fixed_noise(out_root / "fixed_noise.pt", cfg["n_fixed"], cfg["nz"],
                                  cfg["fixed_noise_seed"]).to(device)

    start_epoch, global_iter, epoch_times = 0, 0, []
    csv_path = run_dir / "losses.csv"
    if args.resume:
        ck = load_checkpoint(ckpt_dir / "last.pt", device)
        # Solo se permite cambiar `epochs` al reanudar; cualquier otro cambio mezclaría dos runs distintos.
        changed = {k: v for k, v in diff_configs(cfg, ck["config"]).items() if k != "epochs"}
        if changed:
            raise SystemExit(f"--resume: la config cambió respecto al checkpoint: {changed}")
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"])
        optG.load_state_dict(ck["optG"]); optD.load_state_dict(ck["optD"])
        start_epoch, global_iter = ck["epoch"], ck["global_iter"]
        set_rng_state(ck["rng"])
        loader_gen.set_state(ck["loader_rng"])
        epoch_times = ck.get("epoch_times", [])
        truncate_csv(csv_path, start_epoch)
        print(f"Reanudando desde la época {start_epoch}")
    else:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_FIELDS)
        snapshot(G, fixed_noise, samples_dir / "epoch_000.png")  # punto de partida: ruido puro

    # Se escribe después de validar --resume: un intento rechazado no debe pisar el config del run.
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    n_params = lambda m: sum(p.numel() for p in m.parameters())
    print(f"Dispositivo: {device} | G: {n_params(G):,} parámetros | D: {n_params(D):,} | "
          f"{len(loader)} iteraciones/época | pérdida: {cfg['loss']} | SN en D: {cfg['spectral_norm_d']}")

    for epoch in range(start_epoch + 1, cfg["epochs"] + 1):
        t0 = time.time()
        rows = []
        sigma = instance_noise_std(epoch, cfg["epochs"], cfg.get("instance_noise", 0.0))
        for i, (real, _) in enumerate(loader):
            if args.max_iters is not None and i >= args.max_iters:
                break
            real = real.to(device, non_blocking=True)
            z = torch.randn(real.size(0), cfg["nz"], 1, 1, device=device)
            fake = G(z)

            # ---- paso de D: separar reales de falsas. detach() para no propagar gradiente a G aquí.
            optD.zero_grad(set_to_none=True)
            real_logits = D(add_instance_noise(real, sigma))
            fake_logits = D(add_instance_noise(fake.detach(), sigma))
            loss_D = d_loss_fn(real_logits, fake_logits)
            loss_D.backward()
            optD.step()

            # ---- paso de G: engañar al D recién actualizado (por eso se vuelve a evaluar D(fake)).
            optG.zero_grad(set_to_none=True)
            loss_G = g_loss_fn(D(add_instance_noise(fake, sigma)))
            loss_G.backward()
            optG.step()

            # D(x) y D(G(z)) como sigmoide del logit, también con hinge, para poder comparar runs
            # en la misma escala [0, 1]. Con hinge es solo un indicador, no una probabilidad calibrada.
            global_iter += 1
            rows.append([epoch, global_iter, loss_D.item(), loss_G.item(),
                         torch.sigmoid(real_logits).mean().item(), torch.sigmoid(fake_logits).mean().item()])

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerows(rows)

        is_last = epoch == cfg["epochs"]
        if epoch % cfg["snapshot_every"] == 0 or is_last:
            snapshot(G, fixed_noise, samples_dir / f"epoch_{epoch:03d}.png")

        epoch_times.append(time.time() - t0)
        # last.pt solo cada `ckpt_every` épocas: cada checkpoint pesa ~75 MB y escribirlo en Drive en cada
        # época cuesta tanto como entrenarla (y puede llenar la cuota). Si Colab se cae se pierden como mucho
        # ckpt_every-1 épocas; al reanudar, truncate_csv borra las filas de esas épocas.
        if epoch % cfg["ckpt_every"] == 0 or is_last:
            state = dict(G=G.state_dict(), D=D.state_dict(), optG=optG.state_dict(), optD=optD.state_dict(),
                         epoch=epoch, global_iter=global_iter, config=cfg, rng=get_rng_state(),
                         loader_rng=loader_gen.get_state(), epoch_times=epoch_times)
            save_checkpoint(ckpt_dir / "last.pt", **state)
            save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pt", **state)
        epoch_times[-1] = time.time() - t0  # el tiempo por época incluye guardar en Drive
        last = rows[-1]
        print(f"[{epoch:3d}/{cfg['epochs']}] loss_D={last[2]:.3f} loss_G={last[3]:.3f} "
              f"D(x)={last[4]:.2f} D(G(z))={last[5]:.2f} | {epoch_times[-1]:.1f}s")

    # ------------------------------------------------------------ resúmenes del run
    snapshot(G, fixed_noise, run_dir / "fixed_grid.png")
    plot_losses(csv_path, run_dir / "losses.png", title=f"Run: {cfg['name']}")
    progress_figure(samples_dir, run_dir / "progress.png")
    with open(run_dir / "train_log.json", "w") as f:
        json.dump({"epochs": cfg["epochs"], "iters_per_epoch": len(loader), "device": str(device),
                   "sec_per_epoch_mean": sum(epoch_times) / max(len(epoch_times), 1),
                   "total_minutes": sum(epoch_times) / 60, "epoch_times": epoch_times}, f, indent=2)
    print(f"Listo. Resultados en {run_dir}")


if __name__ == "__main__":
    main()
